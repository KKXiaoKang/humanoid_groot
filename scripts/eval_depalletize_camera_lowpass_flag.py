import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

# Initialize GUI windows if requested
def init_gui_windows(enable_gui=False, camera_config=None):
    """
    Initialize GUI windows if enabled
    
    Args:
        enable_gui: Whether to enable GUI windows
        camera_config: Dictionary of camera names from topic_info (e.g., {'image': ..., 'chest_image': ...})
    """
    if not enable_gui:
        print(" ======================  GUI windows disabled ====================== ")
        return
    
    print(" ======================  Initializing GUI windows ====================== ")
    
    # 根据相机配置动态创建窗口
    if camera_config is None:
        # 如果没有提供配置，使用默认3相机配置（向后兼容）
        from configs.config import topic_info
        camera_config = {name: info for name, info in topic_info.items() if 'image' in name}
    
    # 相机名称到窗口名称的映射
    camera_window_map = {
        'image': 'head Camera',
        'chest_image': 'chest Camera',
        'left_shoulder_image': 'left_shoulder Camera',
        'right_shoulder_image': 'right_shoulder Camera'
    }
    
    # 创建相机窗口
    for camera_name in camera_config.keys():
        if camera_name in camera_window_map:
            window_name = camera_window_map[camera_name]
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)
            print(f"   Created window: {window_name}")
    
    print(f" ======================  GUI windows ready ({len(camera_config)} cameras) ====================== ")

# GUI窗口将在解析命令行参数后初始化

from collections import deque
from typing import Optional
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "robot_envs")))
from robot_envs.kuavo_depalletize_env import GrabBoxMpcEnv
from configs.config import topic_info, TASK_DATA_MODE, get_camera_observation_key, ACTION_COMPONENTS

# 使用GrootPolicy模型
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# import torchvision
# import matplotlib.pyplot as plt
from pathlib import Path
import torch
import time
import argparse
import rospy
from std_msgs.msg import Float64MultiArray
from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.robot_sdk import RobotSDK
from kuavo_humanoid_sdk.msg.kuavo_msgs.srv import (changeArmCtrlMode, changeArmCtrlModeRequest)

MODEL_ACTION_DT = 0.1  # seconds between predicted actions during training
MODEL_ACTION_FREQUENCY = 1.0 / MODEL_ACTION_DT
TARGET_CONTROL_FREQUENCY = 100.0
TARGET_CONTROL_DT = 1.0 / TARGET_CONTROL_FREQUENCY
CHUNK_TRANSITION_DURATION_S = 0.2  # seconds of low-pass smoothing at chunk boundary
LOWPASS_ALPHA = 0.85  # closer to 1 => smoother (slower) transitions
ENABLE_CHUNK_TRANSITION_LOWPASS = False  # Enable/disable low-pass filtering at chunk boundaries (default: False, only linear interpolation within chunks)


def resample_action_chunk(action_chunk: np.ndarray,
                          source_dt: float = MODEL_ACTION_DT,
                          target_dt: float = TARGET_CONTROL_DT) -> np.ndarray:
    """
    Resample an action chunk predicted at a lower frequency to a higher control frequency.

    Args:
        action_chunk: Array of shape (N, action_dim) predicted at intervals of source_dt.
        source_dt: Time interval between successive actions in the chunk.
        target_dt: Desired time interval for control commands.

    Returns:
        Array of shape (M, action_dim) where M approximates (N-1)*source_dt/target_dt + 1,
        interpolated with linear interpolation along time.
    """
    action_chunk = np.asarray(action_chunk)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk.reshape(1, -1)

    if action_chunk.shape[0] <= 1 or np.isclose(source_dt, target_dt):
        # Nothing to resample, either single action or already at target frequency
        return action_chunk

    total_duration = source_dt * (action_chunk.shape[0] - 1)
    if total_duration <= 0:
        repeat_factor = max(int(round(source_dt / target_dt)), 1)
        return np.repeat(action_chunk, repeats=repeat_factor, axis=0)

    num_target_steps = int(round(total_duration / target_dt)) + 1
    source_times = np.linspace(0.0, total_duration, num=action_chunk.shape[0])
    target_times = np.linspace(0.0, total_duration, num=num_target_steps)

    interpolated = np.empty((num_target_steps, action_chunk.shape[1]), dtype=action_chunk.dtype)
    for dim in range(action_chunk.shape[1]):
        interpolated[:, dim] = np.interp(target_times, source_times, action_chunk[:, dim])

    return interpolated


def apply_lowpass_transition(actions: np.ndarray,
                             previous_action: Optional[np.ndarray],
                             alpha: float = LOWPASS_ALPHA,
                             transition_steps: Optional[int] = None,
                             smooth_slice: slice | tuple | np.ndarray = slice(None)) -> np.ndarray:
    """
    Smooth the beginning of a resampled chunk with an exponential low-pass filter
    to reduce discontinuities at chunk boundaries.

    Args:
        actions: Resampled action chunk at control frequency, shape (N, action_dim).
        previous_action: Last action that was executed on the robot. If None, no smoothing applied.
        alpha: Low-pass smoothing coefficient (0 < alpha < 1). Larger alpha = smoother/slower response.
        transition_steps: Number of control steps over which to apply smoothing. If None, smooth entire chunk.
        smooth_slice: Indices/slice specifying which action dimensions to smooth (e.g., only arm joints).

    Returns:
        Smoothed action chunk (same shape as input).
    """
    if previous_action is None:
        return actions

    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)

    smoothed = actions.copy()
    prev = np.asarray(previous_action, dtype=smoothed.dtype)
    if prev.ndim == 1:
        prev = prev.reshape(1, -1)
    prev = prev[0]

    num_steps = smoothed.shape[0]
    if transition_steps is None or transition_steps > num_steps:
        transition_steps = num_steps
    transition_steps = max(1, transition_steps)

    if isinstance(smooth_slice, slice) or isinstance(smooth_slice, tuple):
        smooth_indices = smooth_slice
    else:
        smooth_indices = smooth_slice

    for idx in range(transition_steps):
        prev_slice = prev[smooth_indices]
        smoothed_slice = smoothed[idx][smooth_indices]
        filtered = alpha * prev_slice + (1.0 - alpha) * smoothed_slice
        prev[smooth_indices] = filtered
        smoothed[idx][smooth_indices] = filtered

    return smoothed


def resample_chunk_with_claw_hold(action_chunk: np.ndarray,
                                  previous_action: Optional[np.ndarray],
                                  control_frequency: float,
                                  source_dt: float = MODEL_ACTION_DT,
                                  arm_dims: slice = slice(0, 14),
                                  claw_dims: slice = slice(14, 16)) -> np.ndarray:
    """
    Resample an action chunk so that arm joints are interpolated to the control frequency
    while claw positions are held at the original (low) frequency.
    """
    action_chunk = np.asarray(action_chunk)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk.reshape(1, -1)

    if previous_action is not None:
        chunk_with_bridge = np.vstack([previous_action, action_chunk])
        resampled = resample_action_chunk(
            chunk_with_bridge,
            source_dt=source_dt,
            target_dt=1.0 / control_frequency
        )[1:]
        source_array = chunk_with_bridge
    else:
        resampled = resample_action_chunk(
            action_chunk,
            source_dt=source_dt,
            target_dt=1.0 / control_frequency
        )
        source_array = action_chunk

    # Zero-order hold for claw dimensions (keep 10Hz updates)
    if source_array.shape[0] > 0 and resampled.shape[0] > 0:
        total_duration = source_dt * max(source_array.shape[0] - 1, 1)
        if total_duration <= 0:
            hold_indices = np.zeros(resampled.shape[0], dtype=int)
        else:
            target_times = np.linspace(0.0, total_duration, num=resampled.shape[0], endpoint=True)
            source_times = np.linspace(0.0, total_duration, num=source_array.shape[0], endpoint=True)
            hold_indices = np.searchsorted(source_times, target_times, side="right") - 1
            hold_indices = np.clip(hold_indices, 0, source_array.shape[0] - 1)
        resampled[:, claw_dims] = source_array[hold_indices][:, claw_dims]

    return resampled


def direct_to_wbc(control_mode):
    """
        切换手臂到wbc轨迹控制模式
        Args:
            control_mode: 控制模式
                0: 禁用wbc控制轨迹模式
                1: wbc轨迹控制模式
    """
    rospy.wait_for_service('/enable_wbc_arm_trajectory_control', timeout=5)
    try:
        change_mode = rospy.ServiceProxy('/enable_wbc_arm_trajectory_control', changeArmCtrlMode)
        req = changeArmCtrlModeRequest()
        req.control_mode = control_mode
        res = change_mode(req)
        if res.result:
            rospy.loginfo("wbc轨迹控制模式已更改为 %d", control_mode)
        else:
            rospy.logerr("无法将wbc轨迹控制模式更改为 %d", control_mode)
    except rospy.ServiceException as e:
        rospy.logerr("服务调用失败: %s", e)


def replay(lerobot_dataset_path, episode, control_arm=True, control_claw=True):
    """
    直接replay数据集里的轨迹（depalletize任务）
    """
    repo_id = 0

    dataset = LeRobotDataset(repo_id=repo_id, root=lerobot_dataset_path, episodes=[episode])
    actions = dataset.hf_dataset.select_columns("action")
    env = GrabBoxMpcEnv()
    env.obs_buffer.wait_buffer_ready()
    time.sleep(1)

    for idx in range(dataset.num_frames):
        action = actions[idx]["action"]
        action = np.expand_dims(action, axis=0)

        env.exec_actions(actions=action,
                         control_arm=control_arm,
                         control_claw=control_claw,
                         )

def publish_joint_positions(action_chunk,
                            joint_pub,
                            source_frequency_hz: float,
                            target_frequency_hz: Optional[float] = None):
    """
    从动作块中提取左右手关节位置和夹爪位置并合并发布到ROS话题，
    可选地将动作插值到更高的控制频率后再发布。
    
    Args:
        action_chunk: shape为(N, action_dim)的动作块
                     支持格式:
                     - 16维: [14个手臂关节, 2个夹爪位置]
                     - 18维: [7个左手臂关节, 7个右手臂关节, 2个夹爪位置, 2个cmd_pose维度]
        joint_pub: 关节位置发布器
        source_frequency_hz: 原始动作块的频率（Hz）
        target_frequency_hz: 如果提供，则将动作块插值到该频率后再发布
    """
    try:
        action_chunk = np.asarray(action_chunk)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk.reshape(1, -1)

        if target_frequency_hz is not None and target_frequency_hz > source_frequency_hz:
            action_chunk = resample_action_chunk(
                action_chunk,
                source_dt=1.0 / source_frequency_hz,
                target_dt=1.0 / target_frequency_hz
            )

        action_dim = action_chunk.shape[1]
        
        # 支持16维和18维动作格式
        if action_dim == 16:
            # 16维格式: [14个手臂关节, 2个夹爪位置]
            rospy.logdebug(f"Using depalletize 16-dim action format")
            left_joints_all_steps = action_chunk[:, :7]    # shape: (action_chunk_size, 7)
            right_joints_all_steps = action_chunk[:, 7:14] # shape: (action_chunk_size, 7)
            claw_positions = action_chunk[:, 14:16]  # shape: (action_chunk_size, 2)
            # 合并左右手关节位置和夹爪位置：先左手，后右手，最后是夹爪位置
            combined_joints = np.concatenate([left_joints_all_steps, right_joints_all_steps, claw_positions], axis=1)  # shape: (action_chunk_size, 16)
            
        elif action_dim == 18:
            # 18维格式: [7个左手臂关节, 7个右手臂关节, 2个夹爪位置, 2个cmd_pose维度]
            # 发送完整的18维数据
            rospy.logdebug(f"Using depalletize 18-dim action format (including cmd_pose dimensions)")
            left_joints_all_steps = action_chunk[:, :7]    # shape: (action_chunk_size, 7)
            right_joints_all_steps = action_chunk[:, 7:14] # shape: (action_chunk_size, 7)
            claw_positions = action_chunk[:, 14:16]  # shape: (action_chunk_size, 2)
            cmd_pose = action_chunk[:, 16:18]  # shape: (action_chunk_size, 2)
            # 合并所有组件：先左手，后右手，然后是夹爪，最后是cmd_pose
            combined_joints = np.concatenate([left_joints_all_steps, right_joints_all_steps, claw_positions, cmd_pose], axis=1)  # shape: (action_chunk_size, 18)
            
        else:
            rospy.logwarn(f"Action chunk dimension {action_dim} not supported (expected 16 or 18 for depalletize task)")
            return

        # 发布合并后的关节位置（完整的action_dim维度）
        joint_msg = Float64MultiArray()
        joint_msg.data = combined_joints.flatten().tolist()  # 展平为一维数组
        joint_pub.publish(joint_msg)
        
        rospy.logdebug(f"Published combined joint positions: {combined_joints.shape} (action_dim={action_dim})")
        
    except Exception as e:
        rospy.logerr(f"Error publishing joint positions: {str(e)}")


def eval(ckpt_path, model_type, control_arm=True, control_claw=True, action_chunk_size=50, lerobot_dataset_path=None, enable_gui=False):
    """
    在这里和实机/仿真交互，做网络推理（depalletize任务）
    
    Args:
        ckpt_path: 模型checkpoint路径
        model_type: 模型类型（已废弃，保留用于兼容性，现在只使用GrootPolicy）
        control_arm: 是否控制手臂
        control_claw: 是否控制夹爪
        action_chunk_size: 动作块大小
        lerobot_dataset_path: 数据集路径（用于加载统计信息，可选）
        enable_gui: 是否启用GUI窗口显示相机图像
    """

    # ---------- 1. load GrootPolicy from checkpoint ---------------
    device = "cuda:0"
    print(" =================== Loading GrootPolicy =================== ")
    policy = GrootPolicy.from_pretrained(Path(ckpt_path), strict=False)
    policy.config.device = device
    policy.config.n_action_steps = action_chunk_size
    
    # Load dataset statistics for normalization
    print(f"\n📂 Loading dataset for statistics...")
    dataset_stats = None
    if lerobot_dataset_path:
        try:
            dataset_for_stats = LeRobotDataset(repo_id=0, root=lerobot_dataset_path)
            dataset_stats = dataset_for_stats.meta.stats if hasattr(dataset_for_stats.meta, 'stats') else None
            print(f"✅ Dataset statistics loaded: {list(dataset_stats.keys()) if dataset_stats else 'None'}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load dataset statistics: {e}")
            print("   This may cause normalization issues during inference")
    else:
        print("⚠️ Warning: No dataset path provided. Using default dataset for statistics.")
        try:
            dataset_for_stats = LeRobotDataset(repo_id=0, root='/home/lab/lerobot_groot/lerobot_data/new_demo/1118_sim_depalletize')
            dataset_stats = dataset_for_stats.meta.stats if hasattr(dataset_for_stats.meta, 'stats') else None
            print(f"✅ Dataset statistics loaded from default path: {list(dataset_stats.keys()) if dataset_stats else 'None'}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load default dataset statistics: {e}")
    
    # Create preprocessor and postprocessor
    print(f"\n🔧 Creating preprocessor and postprocessor...")
    preprocessor, postprocessor = make_groot_pre_post_processors(
        config=policy.config,
        dataset_stats=dataset_stats,
    )
    print("✅ Preprocessor and postprocessor created")
    
    # Debug: Print model configuration
    print(f"🔍 Model configuration input_features keys: {list(policy.config.input_features.keys()) if hasattr(policy.config, 'input_features') else 'N/A'}")
    print(f"🔍 Model configuration output_features keys: {list(policy.config.output_features.keys()) if hasattr(policy.config, 'output_features') else 'N/A'}")
    
    # Print action mode configuration
    print("\n" + "="*80)
    print("🎯 DEPALLETIZE TASK CONFIGURATION (GrootPolicy)")
    print("="*80)
    print(f"🤖 Control arm: {control_arm}")
    print(f"🦾 Control claw: {control_claw}")
    print(f"📊 ACTION_COMPONENTS: {ACTION_COMPONENTS}")
    print(f"📊 Action dimension: Will be determined by model output (expected: {len(ACTION_COMPONENTS) * 7 if 'Left_arm' in ACTION_COMPONENTS and 'Right_arm' in ACTION_COMPONENTS else 'varies'}D based on config)")
    print(f"📦 Action chunk size: {action_chunk_size}")
    # 根据ACTION_COMPONENTS判断是否包含cmd_pose
    has_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
    print(f"🎯 Cmd_pose control: {'Enabled' if has_cmd_pose else 'Disabled'} (based on ACTION_COMPONENTS)")
    print("="*80 + "\n")
    
    policy.eval()
    policy.to(device)
    policy.reset()
    
    step_counter = 0

    # Initialize ROS publishers for action visualization
    # rospy.init_node('act_eval_visualizer', anonymous=True)
    joint_pub = rospy.Publisher('/policy/action/eef_pose_marker_all', Float64MultiArray, queue_size=10)
    
    rospy.loginfo(f"Initialized ROS publishers for action visualization with chunk size: {action_chunk_size}")

    # Initialize real-time environment
    env = GrabBoxMpcEnv()
    print(f"🤖 Environment initialized for depalletize task")
    print(" ======================  Waiting for buffer ready ====================== ")
    env.obs_buffer.wait_buffer_ready()
    print(" ======================  Buffer ready ====================== ")
    time.sleep(1)
    
    obs_data, camera_obs, camera_obs_ts, robot_obs, robot_obs_ts = env.get_obs()

    # TODO: 检查时间同步情况
    # TODO: 显示图像
    rospy.loginfo(f"Initialized action visualization with chunk size: {action_chunk_size}")
    
    # ---------- 2. 模型推理（实时模式） ----------------------
    # Real-time environment evaluation loop
    direct_to_wbc(1)
    input(f"direct_to_wbc 结束, 按回车继续 ==== 切换手臂到wbc轨迹控制模式成功 ==== \n")
    time.sleep(1.0)
    resampled_action_queue: deque[np.ndarray] = deque()
    last_executed_action: Optional[np.ndarray] = None

    while True:
        try:
            state = torch.from_numpy(obs_data["state"]).float()
            # print(f" ==== state ==== {state.shape} ==== ")
            
            # 根据topic_info动态处理所有相机图像
            # 填充网络的obs
            observation = {}
            
            # 动态处理所有相机观测 - 使用新的key格式
            for camera_name in topic_info.keys():
                if 'image' in camera_name and camera_name in obs_data:
                    camera_images = torch.from_numpy(np.moveaxis(obs_data[camera_name], 3, 1)).float() / 255
                    # 使用新的key格式: observation.images.cam_*
                    obs_key = get_camera_observation_key(camera_name, use_image_features=False)
                    observation[obs_key] = camera_images.to('cuda:0')

            # observation['observation.environment_state'] = environment_state
            observation['observation.state'] = state.to('cuda:0')

            if not resampled_action_queue:
                # 使用GrootPolicy的predict_action_chunk
                # 首先使用预处理器处理输入
                processed_observation = preprocessor(observation)
                
                # 模型推理
                with torch.inference_mode():
                    pred_actions = policy.predict_action_chunk(processed_observation)
                
                # pred_actions shape: (batch_size, chunk_size, action_dim)
                # 注意：pred_actions是归一化后的值，范围在[-1, 1]
                # 需要手动反归一化到真实单位
                
                # 手动反归一化整个chunk
                if dataset_stats and 'action' in dataset_stats:
                    action_stats = dataset_stats['action']
                    if 'min' in action_stats and 'max' in action_stats:
                        action_min = torch.as_tensor(action_stats['min'], dtype=torch.float32, device=pred_actions.device)
                        action_max = torch.as_tensor(action_stats['max'], dtype=torch.float32, device=pred_actions.device)
                        
                        # 确保维度匹配
                        action_dim = pred_actions.shape[-1]
                        if action_min.numel() < action_dim:
                            action_min = torch.nn.functional.pad(action_min.flatten()[:action_dim], (0, max(0, action_dim - action_min.numel())))
                        if action_max.numel() < action_dim:
                            action_max = torch.nn.functional.pad(action_max.flatten()[:action_dim], (0, max(0, action_dim - action_max.numel())))
                        
                        action_min = action_min[:action_dim]
                        action_max = action_max[:action_dim]
                        
                        # 反归一化公式：x = (y + 1) / 2 * (max - min) + min
                        denom = action_max - action_min
                        mask = denom != 0
                        safe_denom = torch.where(mask, denom, torch.ones_like(denom))
                        
                        # 反归一化整个chunk
                        pred_actions_unnorm = (pred_actions + 1.0) * 0.5 * safe_denom + action_min
                        pred_actions_unnorm = torch.where(mask, pred_actions_unnorm, action_min)
                        
                        # 转换为numpy
                        action_chunk = pred_actions_unnorm[0].cpu().numpy()  # (chunk_size, action_dim)
                    else:
                        # 如果没有统计信息，使用原始值（可能已经是反归一化的）
                        action_chunk = pred_actions[0].cpu().numpy()  # (chunk_size, action_dim)
                        rospy.logwarn("⚠️ Warning: No action min/max stats found. Using raw predictions (may be normalized).")
                else:
                    # 如果没有统计信息，使用原始值
                    action_chunk = pred_actions[0].cpu().numpy()  # (chunk_size, action_dim)
                    rospy.logwarn("⚠️ Warning: No dataset stats found. Using raw predictions (may be normalized).")

                # 根据动作维度动态确定claw维度
                action_dim = action_chunk.shape[1]
                if action_dim == 16:
                    arm_dims = slice(0, 14)
                    claw_dims = slice(14, 16)
                elif action_dim == 18:
                    # 18维格式: 前14维是手臂关节，14-16是夹爪，16-18是cmd_pose（保留完整18维）
                    arm_dims = slice(0, 14)
                    claw_dims = slice(14, 16)
                    # 注意：cmd_pose维度(16-18)会在resample时一起处理，不需要单独处理
                else:
                    # 默认使用前14维作为手臂，14-16作为夹爪
                    arm_dims = slice(0, 14)
                    claw_dims = slice(14, min(16, action_dim))
                    rospy.logwarn(f"Unknown action dimension {action_dim}, using default arm/claw split")
                
                resampled_chunk = resample_chunk_with_claw_hold(
                    action_chunk,
                    previous_action=last_executed_action,
                    control_frequency=env.control_frequency,
                    source_dt=MODEL_ACTION_DT,
                    arm_dims=arm_dims,
                    claw_dims=claw_dims
                )

                # Apply low-pass filtering at chunk boundaries only if enabled
                if ENABLE_CHUNK_TRANSITION_LOWPASS:
                    if last_executed_action is not None:
                        transition_steps = max(
                            1,
                            int(round(env.control_frequency * CHUNK_TRANSITION_DURATION_S))
                        )
                    else:
                        transition_steps = None
                    
                    resampled_chunk = apply_lowpass_transition(
                        resampled_chunk,
                        previous_action=last_executed_action,
                        alpha=LOWPASS_ALPHA,
                        transition_steps=transition_steps,
                        smooth_slice=arm_dims  # 只对手臂关节进行低通滤波
                    )

                publish_joint_positions(
                    resampled_chunk,
                    joint_pub,
                    source_frequency_hz=env.control_frequency,
                    target_frequency_hz=None
                )
                rospy.loginfo(f"Prepared resampled chunk of size {resampled_chunk.shape[0]} for execution")

                resampled_action_queue = deque(np.array(step, copy=True) for step in resampled_chunk)

            current_action = resampled_action_queue.popleft()
            # 根据ACTION_COMPONENTS配置决定是否控制cmd_pose
            # 如果ACTION_COMPONENTS包含Cmd_pose_z或Cmd_pose_pitch，则启用cmd_pose控制
            control_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
            
            env.exec_actions(actions=current_action,
                             control_arm=control_arm,
                             control_claw=control_claw,
                             control_cmd_pose=control_cmd_pose)
            step_counter += 1
            last_executed_action = current_action.copy()

            obs_data, camera_obs, camera_obs_ts, robot_obs, robot_obs_ts = env.get_obs()

            if enable_gui:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC to quit
                    print("\n[GUI] Exiting by user request (q or ESC pressed)")
                    break

        except KeyboardInterrupt:
            print("\n[Interrupted] Exiting by user Ctrl+C.")
            break
    
    # Cleanup GUI windows
    if enable_gui:
        cv2.destroyAllWindows()




if __name__ == '__main__':
    # 机器人低头
    robot_sdk = RobotSDK()
    robot_sdk.control.control_head(0, np.deg2rad(10))
    robot_sdk.control.set_external_control_arm_mode()  # 切换手臂到外部控制模式
    print(" ==== 机器人头部俯仰调节角度: 10 成功 ==== ")
    print(" ==== 切换手臂到外部控制模式成功 ==== ")
    
    # python 参数解析器
    parser = argparse.ArgumentParser(
        description='Depalletize Task Evaluation Script',
        epilog='This script evaluates models for the depalletize task (16-dim actions: 14 arm joints + 2 claw positions).'
    )
    parser.add_argument('--ckpt-path', type=str, default='/home/lab/kuavo-manip/outputs/train/box_only_vel_obs/checkpoints/080000/pretrained_model',
                        help='Path to the checkpoint directory')
    parser.add_argument('--model-type', type=str, default='groot', choices=['groot', 'act', 'dp'],
                        help='Type of model to use (now only groot is supported, act/dp are deprecated)')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model in real-time environment')
    parser.add_argument('--replay', action='store_true', help='Replay the model')
    parser.add_argument('--action_chunk_size', type=int, default=20, help='Number of action steps')
    parser.add_argument('--lerobot_dataset_path', type=str, default=None, help='Path to the LeRobot dataset for loading statistics (optional)')
    parser.add_argument('--enable_gui', action='store_true',
                        help='Enable GUI windows for camera display (default: disabled)')
    
    args = parser.parse_args()
    
    # 根据命令行参数和相机配置初始化GUI窗口
    camera_config = {name: info for name, info in topic_info.items() if 'image' in name}
    init_gui_windows(enable_gui=args.enable_gui, camera_config=camera_config)
    
    # 打印相机配置信息
    print(f"\n📷 Camera Configuration (TASK_DATA_MODE: {TASK_DATA_MODE}):")
    print(f"   Detected {len(camera_config)} cameras: {list(camera_config.keys())}")
    
    print("\n" + "="*80)
    print("🎯 Depalletize Task Evaluation (GrootPolicy)")
    print("="*80)
    print(f"📂 Checkpoint: {args.ckpt_path}")
    print(f"🤖 Model type: {args.model_type} (using GrootPolicy)")
    if args.model_type != 'groot':
        print(f"⚠️  Warning: model-type '{args.model_type}' is deprecated. Using GrootPolicy instead.")
    print(f"📊 Action chunk size: {args.action_chunk_size}")
    print(f"📦 Action dimension: Supports 16 or 18 (14 arm joints + 2 claw positions [+ 2 cmd_pose])")
    print(f"🖼️  Enable GUI: {args.enable_gui}")
    if args.lerobot_dataset_path:
        print(f"📁 Dataset path (for stats): {args.lerobot_dataset_path}")
    print("="*80 + "\n")

    if args.eval:
        print("🚀 Starting real-time evaluation...")
        eval(args.ckpt_path, model_type=args.model_type, control_arm=True, control_claw=True, 
             action_chunk_size=args.action_chunk_size, 
             lerobot_dataset_path=args.lerobot_dataset_path,
             enable_gui=args.enable_gui)
    elif args.replay:
        print("Replaying the model")
        lerobot_dataset_path = '/home/lab/kuavo-manip/lerobot_data/vel_wrend_box_613'
        replay(lerobot_dataset_path, episode=0, control_arm=True, control_claw=True)
    else:
        print("Please specify either --eval or --replay")
        exit(1)

    # --------------------------------------- #

