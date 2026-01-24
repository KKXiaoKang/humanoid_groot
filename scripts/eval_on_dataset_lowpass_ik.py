#!/usr/bin/env python3
"""
Evaluate GrootPolicy Model on Dataset with Lowpass Visualization

This script evaluates a GrootPolicy model on a LeRobot dataset and computes error metrics.
It visualizes chunk interpolation and lowpass filtering between chunks.
It supports optional MuJoCo visualization.

Usage:
    python scripts/eval_on_dataset_losspass.py \
        --ckpt-path <checkpoint_path> \
        --dataset-root <dataset_path> \
        --episode <episode_number> \
        [--image-zero]  # Optional: set all images to zero to test model dependency on images
        [--state-zero]  # Optional: set all state inputs to zero to test model dependency on state
        [--cam-head-zero]  # Optional: set cam_head (image) to zero to test model dependency on cam_head
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from pathlib import Path
import argparse
import time
from collections import OrderedDict
from tqdm import tqdm

# 使用GrootPolicy模型
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.converters import policy_action_to_transition

# 导入配置模块（如果存在）
try:
    from configs.config import topic_info, TASK_DATA_MODE, get_camera_observation_key, get_camera_names, CAMERA_COMPONENTS, action_names, CAMERA_KEY_MAPPING
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: configs.config not available. Using defaults.")
    CONFIG_AVAILABLE = False
    topic_info = {}
    TASK_DATA_MODE = "unknown"
    CAMERA_COMPONENTS = []
    action_names = []
    CAMERA_KEY_MAPPING = {}
    def get_camera_observation_key(camera_name, use_image_features=False):
        return f"observation.images.{camera_name}" if use_image_features else f"observation.images.{camera_name}"
    def get_camera_names(camera_components=None):
        return []

# 插值与低通滤波常量
MODEL_ACTION_DT = 0.1
MODEL_ACTION_FREQUENCY = 1.0 / MODEL_ACTION_DT
TARGET_CONTROL_FREQUENCY = 100.0
TARGET_CONTROL_DT = 1.0 / TARGET_CONTROL_FREQUENCY
CHUNK_TRANSITION_DURATION_S = 0.2
LOWPASS_ALPHA = 0.85
COLOR_INTERP = [0, 128, 255]
COLOR_LOWPASS = [255, 140, 0]

# 可选的可视化工具（如果不存在则禁用）
try:
    from visualization_tools.visualizers import RerunVisualizer, KeyboardManager
    RERUN_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: RerunVisualizer not available. Visualization will be disabled.")
    RERUN_AVAILABLE = False
    RerunVisualizer = None
    KeyboardManager = None

# ROS 和机器人 SDK（仅在需要时导入，用于 MuJoCo 可视化）
try:
    import rospy
    from sensor_msgs.msg import JointState
    # from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.robot_sdk import RobotSDK
    # from kuavo_humanoid_sdk.msg.kuavo_msgs.srv import (changeArmCtrlMode, changeArmCtrlModeRequest)
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    JointState = None
    print("⚠️  Warning: ROS dependencies not available. MuJoCo visualization with robot control will be disabled.")

def direct_to_wbc(control_mode):
    """
    切换手臂到wbc轨迹控制模式
    Args:
        control_mode: 控制模式
            0: 禁用wbc控制轨迹模式
            1: wbc轨迹控制模式
    """
    if not ROS_AVAILABLE:
        print("⚠️  Warning: ROS not available, cannot call direct_to_wbc")
        return
    
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


def reconstruct_rotation_matrix_6d(rotation_6d: np.ndarray) -> np.ndarray:
    """
    从6D表示重构3x3旋转矩阵
    使用Gram-Schmidt正交化确保是有效的旋转矩阵
    
    Args:
        rotation_6d: 6维向量 [R11, R21, R31, R12, R22, R32]
        
    Returns:
        3x3旋转矩阵
    """
    # 提取前两列
    col1 = rotation_6d[:3]  # [R11, R21, R31]
    col2 = rotation_6d[3:6]  # [R12, R22, R32]
    
    # Gram-Schmidt正交化
    # 第一列归一化
    col1_norm = np.linalg.norm(col1)
    if col1_norm < 1e-8:
        col1_normalized = np.array([1.0, 0.0, 0.0])
    else:
        col1_normalized = col1 / col1_norm
    
    # 第二列正交化并归一化
    col2_projected = col2 - np.dot(col2, col1_normalized) * col1_normalized
    col2_norm = np.linalg.norm(col2_projected)
    if col2_norm < 1e-8:
        col2_normalized = np.array([0.0, 1.0, 0.0])
    else:
        col2_normalized = col2_projected / col2_norm
    
    # 第三列通过叉积得到
    col3_normalized = np.cross(col1_normalized, col2_normalized)
    
    # 组合成旋转矩阵
    rotation_matrix = np.column_stack([col1_normalized, col2_normalized, col3_normalized])
    
    return rotation_matrix


def rot6d_to_quaternion_xyzw(rot_6d: np.ndarray) -> np.ndarray:
    """
    将6D旋转表示转换为四元数 (x, y, z, w)
    
    Args:
        rot_6d: 6维向量 [R11, R21, R31, R12, R22, R32]
        
    Returns:
        四元数 [x, y, z, w]
    """
    from scipy.spatial.transform import Rotation as R
    # 重构旋转矩阵
    R_mat = reconstruct_rotation_matrix_6d(rot_6d)
    # 转换为四元数
    quat = R.from_matrix(R_mat).as_quat()  # scipy返回的是 (x, y, z, w)
    return quat


def convert_eef_action_to_joint_action(eef_action: np.ndarray, model_type: str = '60') -> np.ndarray:
    """
    将20D EEF action转换为16D joint action（用于MuJoCo执行）
    
    Args:
        eef_action: 20D EEF action [left_eef(9D) + right_eef(9D) + claw(2D)]
        model_type: 机器人型号 ('45', '46', '60')
        
    Returns:
        16D joint action [left_arm(7D) + right_arm(7D) + claw(2D)]
    """
    try:
        from kuavo_ik.ik_library import IKAnalytical
    except ImportError:
        print("⚠️  Warning: IKAnalytical not available. Cannot convert EEF to joint space.")
        # 返回零向量作为fallback
        return np.zeros(16)
    
    # 提取左右臂的eef pose
    left_eef = eef_action[0:9]  # [x, y, z, R11, R21, R31, R12, R22, R32]
    right_eef = eef_action[9:18]
    claw = eef_action[18:20]  # [left_gripper, right_gripper]
    
    # 提取位置和旋转
    left_pos = left_eef[:3]
    left_rot6d = left_eef[3:9]
    right_pos = right_eef[:3]
    right_rot6d = right_eef[3:9]
    
    # 转换为四元数用于IK
    left_quat = rot6d_to_quaternion_xyzw(left_rot6d)
    right_quat = rot6d_to_quaternion_xyzw(right_rot6d)
    
    # 解IK
    try:
        left_joints = IKAnalytical.compute(
            eef_pos=left_pos,
            eef_quat_xyzw=left_quat,
            eef_frame='zarm_l7_link',
            model_type=model_type,
            limit=True
        )
    except Exception as e:
        print(f"⚠️  Left arm IK failed: {e}")
        left_joints = np.zeros(7)
    
    try:
        right_joints = IKAnalytical.compute(
            eef_pos=right_pos,
            eef_quat_xyzw=right_quat,
            eef_frame='zarm_r7_link',
            model_type=model_type,
            limit=True
        )
    except Exception as e:
        print(f"⚠️  Right arm IK failed: {e}")
        right_joints = np.zeros(7)
    
    # 组合成16D joint action
    joint_action = np.concatenate([left_joints, right_joints, claw])
    
    return joint_action


def publish_arm_joints_to_ros(arm_joints: np.ndarray, publisher: rospy.Publisher):
    """
    发布手臂关节角到ROS话题 /kuavo_arm_traj
    
    Args:
        arm_joints: 手臂关节角度 (14维) [left_arm(7D) + right_arm(7D)]，单位：弧度
        publisher: ROS Publisher对象，用于发布JointState消息
    """
    if not ROS_AVAILABLE:
        print("⚠️  Warning: ROS not available, cannot publish arm joints")
        return
    
    if publisher is None:
        print("⚠️  Warning: Publisher is None, cannot publish arm joints")
        return
    
    # 创建JointState消息
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = [
        "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", 
        "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
        # 左手七个关节
        "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", 
        "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
        # 右手七个关节
    ]
    
    # 将弧度转换为角度（ROS消息需要角度）
    msg.position = np.rad2deg(arm_joints.tolist())
    
    # 发布消息
    publisher.publish(msg)


def resample_action_chunk(action_chunk: np.ndarray,
                          source_dt: float = MODEL_ACTION_DT,
                          target_dt: float = TARGET_CONTROL_DT) -> np.ndarray:
    action_chunk = np.asarray(action_chunk)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk.reshape(1, -1)

    if action_chunk.shape[0] <= 1 or np.isclose(source_dt, target_dt):
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
                             previous_action: np.ndarray | None,
                             alpha: float = LOWPASS_ALPHA,
                             transition_steps: int | None = None,
                             smooth_slice: slice | tuple | np.ndarray = slice(None)) -> np.ndarray:
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

    indices = smooth_slice
    for idx in range(transition_steps):
        prev_slice = prev[indices]
        smoothed_slice = smoothed[idx][indices]
        filtered = alpha * prev_slice + (1.0 - alpha) * smoothed_slice
        prev[indices] = filtered
        smoothed[idx][indices] = filtered

    return smoothed


def resample_chunk_with_claw_hold(action_chunk: np.ndarray,
                                  previous_action: np.ndarray | None,
                                  control_frequency: float,
                                  source_dt: float = MODEL_ACTION_DT,
                                  arm_dims: slice = slice(0, 14),
                                  claw_dims: slice = slice(14, 16)) -> np.ndarray:
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


def eval_on_dataset(ckpt_path,
                    lerobot_dataset_path,
                    episode,
                    visualize_in_mujoco=False,
                    n_actions=16,
                    show_progress=True,
                    image_zero=False,
                    state_zero=False,
                    cam_head_zero=False,
                    infer_per_frame: int = 1,
                    task_description: str | None = None,
                    publish_arm_commands: bool = False,
                    ik_model_type: str = '60'):
    """
    在数据集上评估模型
    
    Args:
        ckpt_path: 模型checkpoint路径
        lerobot_dataset_path: 数据集根目录
        episode: episode编号
        visualize_in_mujoco: 是否在MuJoCo中可视化执行
        n_actions: action chunk大小
        show_progress: 是否显示进度条
        image_zero: 是否将图像输入置零（用于验证模型对图像的依赖性）
        state_zero: 是否将状态输入置零（用于验证模型对状态的依赖性）
        infer_per_frame: 每隔多少个frame重新推理一次（>=1）。
        task_description: 任务描述字符串（language instruction），如果提供则覆盖数据集中的task，否则使用数据集原本的task。
        publish_arm_commands: 是否将模型输出的EEF动作转换为关节角并发布到ROS话题 /kuavo_arm_traj
        ik_model_type: IK求解使用的机器人型号 ('45', '46', '60')
    """
    # ----------- 一些参数 ----------------
    mse_per_action_dim = OrderedDict() # 记录每个动作维度的MSE
    mae_per_action_dim = OrderedDict() # 记录每个动作维度的MAE
    infer_per_frame = max(1, infer_per_frame)  # 至少每帧推理一次
    
    # 初始化IK可用性标志（用于20D eef space的转换）
    IK_AVAILABLE = False

    # ------------- 初始化visualizer (可选) -------------
    if RERUN_AVAILABLE:
        vizer = RerunVisualizer()
        kb = KeyboardManager()
        print("✅ RerunVisualizer initialized")
    else:
        vizer = None
        kb = None
        print("⚠️  Running without RerunVisualizer")

    # ------------- 初始化数据集和模型 -------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*80}")
    print(f"🔧 Device: {device}")
    
    # ✅ 使用GrootPolicy加载模型
    print(f"📂 Loading GrootPolicy model from {ckpt_path}...")
    policy = GrootPolicy.from_pretrained(Path(ckpt_path), strict=False)
    policy.config.device = device
    policy.config.n_action_steps = n_actions
    
    print(f"📊 Action chunk size: {n_actions}")
    print(f"🔄 Inference frequency: Every {infer_per_frame} frame(s) (infer_per_frame={infer_per_frame})")
    if task_description is not None:
        print(f"📝 Task description (overridden): '{task_description}'")
    else:
        print(f"📝 Task description: Will use task from dataset")
    if image_zero:
        print(f"⚠️  IMAGE ZERO MODE: All image inputs will be set to zero (for dependency testing)")
    if state_zero:
        print(f"⚠️  STATE ZERO MODE: All state inputs will be set to zero (for dependency testing)")
    if cam_head_zero:
        print(f"⚠️  CAM_HEAD ZERO MODE: cam_head (image) will be set to zero (for dependency testing)")
    
    policy.eval().to(device)
    
    # 从 checkpoint 加载 preprocessor 和 postprocessor（必须包含 dataset_stats）
    print(f"\n🔧 Loading preprocessor and postprocessor from checkpoint...")
    try:
        # 从 checkpoint 加载，不提供 dataset_stats，让它从 checkpoint 中加载
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=ckpt_path,
        )
        print("✅ Preprocessor and postprocessor loaded from checkpoint")
        
        # 检查 postprocessor 中是否有 stats
        # 从 postprocessor 的步骤中提取 stats（如果存在）
        dataset_stats = None
        for step in postprocessor.steps:
            if hasattr(step, 'stats') and step.stats is not None:
                dataset_stats = step.stats
                print(f"✅ Found dataset_stats in checkpoint postprocessor")
                break
        
        if dataset_stats is None:
            raise ValueError(
                "❌ ERROR: No dataset_stats found in checkpoint postprocessor. "
                "The checkpoint must contain dataset_stats for normalization. "
                "Please ensure the checkpoint was saved with proper statistics."
            )
        
        print(f"✅ Using dataset_stats from checkpoint: {list(dataset_stats.keys()) if dataset_stats else 'None'}")
        
        # 检查 postprocessor 中的部分归一化配置
        for step in postprocessor.steps:
            if hasattr(step, 'action_space_type') and hasattr(step, 'action_component_indices'):
                action_space_type = getattr(step, 'action_space_type', None)
                action_component_indices = getattr(step, 'action_component_indices', None)
                if action_space_type and action_component_indices:
                    print(f"✅ Partial normalization configuration found:")
                    print(f"   - action_space_type: {action_space_type}")
                    print(f"   - action_component_indices: {list(action_component_indices.keys())}")
                    if action_space_type in ["Delta eef", "Absolute eef"]:
                        print(f"   ⚠️  6D rotation components (left_eef_rot6d, right_eef_rot6d) will NOT be unnormalized")
                    break
                elif action_space_type:
                    print(f"⚠️  Warning: action_space_type={action_space_type} but action_component_indices is None")
                    print(f"   This may cause incorrect unnormalization for 6D rotation components!")
                break
                
    except ValueError as e:
        # 如果是我们抛出的 ValueError（stats 缺失），直接抛出
        raise
    except Exception as e:
        raise RuntimeError(
            f"❌ ERROR: Failed to load processors from checkpoint: {e}\n"
            f"   Please ensure the checkpoint path is correct and contains preprocessor/postprocessor files."
        ) from e
    
    # Debug: Print model configuration
    print(f"🔍 Model configuration input_features keys: {list(policy.config.input_features.keys()) if hasattr(policy.config, 'input_features') else 'N/A'}")
    print(f"🔍 Model configuration output_features keys: {list(policy.config.output_features.keys()) if hasattr(policy.config, 'output_features') else 'N/A'}")
    
    policy.reset()
    print("✅ Model loaded and ready")
    
    # 初始化ROS publisher（如果启用发布手臂命令功能）
    arm_joint_publisher = None
    if publish_arm_commands:
        if not ROS_AVAILABLE:
            print("⚠️  Warning: ROS not available. Cannot publish arm commands. Disabling publish_arm_commands.")
            publish_arm_commands = False
        else:
            try:
                # 初始化ROS节点（如果尚未初始化）
                try:
                    rospy.init_node('eval_on_dataset_arm_publisher', anonymous=True)
                except rospy.exceptions.ROSException:
                    # ROS节点已经初始化，继续执行
                    pass
                
                # 创建publisher
                arm_joint_publisher = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=10)
                # 等待publisher连接（给一些时间让subscriber连接）
                rospy.sleep(0.5)
                print(f"✅ ROS arm joint publisher initialized: /kuavo_arm_traj")
                print(f"   IK model type: {ik_model_type}")
                print(f"   ⚠️  Note: This feature only works with 20D EEF action space")
            except Exception as e:
                print(f"⚠️  Warning: Failed to initialize ROS publisher: {e}")
                print(f"   Disabling publish_arm_commands.")
                publish_arm_commands = False
                arm_joint_publisher = None
    
    previous_resampled_action = None
    last_inferred_chunk: np.ndarray | None = None
    last_resampled_chunk: np.ndarray | None = None
    last_lowpass_chunk: np.ndarray | None = None
    last_inference_step = -1
    infer_per_frame = max(1, infer_per_frame)

    control_step_cursor = 0
    
    # ✅ 使用标准的LeRobotDataset加载数据
    print(f"\n📂 Loading dataset from {lerobot_dataset_path}")
    print(f"📹 Episode: {episode}")
    
    # 对于本地数据集，repo_id应该是一个字符串标识符（不包含"/"）
    # 使用数据集路径的最后一部分作为标识符，或者使用"local"
    dataset_name = Path(lerobot_dataset_path).name if lerobot_dataset_path else "local"
    
    # 注意：LeRobotDataset的episodes参数主要用于下载时选择文件
    # 但在加载后需要手动过滤数据，因为多个episodes可能存储在同一个parquet文件中
    dataset = LeRobotDataset(repo_id=dataset_name, root=lerobot_dataset_path, episodes=[episode])
    
    # 使用episode的索引范围直接切片，比filter快得多
    # 这是必要的，因为v3.0格式中多个episodes可能存储在同一个文件中
    print(f"🔍 Filtering dataset to episode {episode}...")
    if episode >= len(dataset.meta.episodes):
        raise ValueError(f"Episode {episode} out of range. Available episodes: 0-{len(dataset.meta.episodes)-1}")
    
    ep_meta = dataset.meta.episodes[episode]
    ep_start = ep_meta["dataset_from_index"]
    ep_end = ep_meta["dataset_to_index"]
    
    # 使用切片而不是filter，这样快得多
    dataset.hf_dataset = dataset.hf_dataset.select(range(ep_start, ep_end))
    print(f"✅ Filtered dataset. Total frames in episode {episode}: {len(dataset.hf_dataset)} (indices {ep_start}-{ep_end-1})")
    
    # 打印相机配置信息
    if CONFIG_AVAILABLE:
        # 使用CAMERA_COMPONENTS来获取相机名称
        camera_names = get_camera_names(CAMERA_COMPONENTS)
        camera_config = {name: info for name, info in topic_info.items() if 'image' in name}
        print(f"\n📷 Camera Configuration (TASK_DATA_MODE: {TASK_DATA_MODE}):")
        print(f"   CAMERA_COMPONENTS: {CAMERA_COMPONENTS}")
        print(f"   Camera names: {camera_names}")
        print(f"   Detected {len(camera_config)} cameras in topic_info: {list(camera_config.keys())}")
    else:
        # 从数据集元数据中检测相机
        sample = dataset[0]
        image_keys = [k for k in sample.keys() if 'image' in k.lower()]
        print(f"\n📷 Camera Configuration:")
        print(f"   Detected {len(image_keys)} image keys: {image_keys}")
    
    # 创建dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        pin_memory=(device == "cuda:0"),
        drop_last=False,
    )
    
    print(f"✅ Dataset loaded. Total frames: {dataset.num_frames}")

    # 获取action维度
    first_batch = next(iter(dataloader))
    action_dim = first_batch['action'].shape[1]
    obs_dim = first_batch['observation.state'].shape[1]
    print(f"📊 Action dimension: {action_dim}")
    print(f"📊 Observation dimension: {obs_dim}")
    
    # 根据action_dim动态设置arm和claw的维度切片
    # 16D joint space: arm(0-13, 14D) + claw(14-15, 2D)
    # 20D eef space: left_arm(0-8, 9D) + right_arm(9-17, 9D) + claw(18-19, 2D)
    if action_dim == 20:
        # EEF action space (20D)
        arm_dims = slice(0, 18)  # left_arm + right_arm (9+9)
        claw_dims = slice(18, 20)  # claw (2)
        smooth_slice = slice(0, 18)  # Apply lowpass to arm parts only
        print(f"✅ Detected EEF action space (20D):")
        print(f"   - Arm dimensions: {arm_dims} (left_arm 0-8 + right_arm 9-17)")
        print(f"   - Claw dimensions: {claw_dims}")
        
        # 检查publish_arm_commands是否启用
        if publish_arm_commands:
            print(f"   ✅ publish_arm_commands enabled: Will convert EEF to joint space and publish to ROS")
    elif publish_arm_commands:
        # 如果启用了publish_arm_commands但action_dim不是20，给出警告
        print(f"⚠️  Warning: publish_arm_commands is enabled but action_dim={action_dim} (not 20D EEF space)")
        print(f"   publish_arm_commands only works with 20D EEF action space. Disabling publish_arm_commands.")
        publish_arm_commands = False
        if arm_joint_publisher is not None:
            arm_joint_publisher = None
    elif action_dim == 16:
        # Joint action space (16D)
        arm_dims = slice(0, 14)  # arm joints (14)
        claw_dims = slice(14, 16)  # claw (2)
        smooth_slice = slice(0, 14)  # Apply lowpass to arm joints only
        print(f"✅ Detected Joint action space (16D):")
        print(f"   - Arm dimensions: {arm_dims} (14 joints)")
        print(f"   - Claw dimensions: {claw_dims}")
    else:
        # Fallback: assume last 2 dims are claw
        arm_dims = slice(0, action_dim - 2)
        claw_dims = slice(action_dim - 2, action_dim)
        smooth_slice = slice(0, action_dim - 2)
        print(f"⚠️  Unknown action space ({action_dim}D), using fallback:")
        print(f"   - Arm dimensions: {arm_dims}")
        print(f"   - Claw dimensions: {claw_dims}")
    
    # 重新创建dataloader（因为已经消耗了第一个batch）
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        pin_memory=(device == "cuda:0"),
        drop_last=False,
    )
    
    # # 初始化环境（如果需要在mujoco中可视化）
    # if visualize_in_mujoco:
    #     # 首先初始化机器人控制（参考 eval_depalletize_camera.py）
    #     if ROS_AVAILABLE:
    #         print(f"\n🤖 Initializing robot control...")
    #         try:
    #             # 初始化 ROS 节点（如果尚未初始化）
    #             try:
    #                 rospy.init_node('eval_on_dataset_robot_control', anonymous=True)
    #             except rospy.exceptions.ROSException:
    #                 # ROS 节点已经初始化，继续执行
    #                 pass
                
    #             # 初始化机器人 SDK 并设置头部和控制模式
    #             robot_sdk = RobotSDK()
    #             robot_sdk.control.control_head(0, np.deg2rad(10))
    #             robot_sdk.control.set_external_control_arm_mode()  # 切换手臂到外部控制模式
    #             print(f"✅ Robot SDK initialized")
    #             print(f"   - 机器人头部俯仰调节角度: 10 成功")
    #             print(f"   - 切换手臂到外部控制模式成功")
                
    #             # 切换到 WBC 轨迹控制模式
    #             direct_to_wbc(1)
    #             input(f"direct_to_wbc 结束, 按回车继续 ==== 切换手臂到wbc轨迹控制模式成功 ==== \n")
    #             time.sleep(1.0)
    #         except Exception as e:
    #             print(f"⚠️  Warning: Failed to initialize robot control: {e}")
    #             print(f"   Continuing with MuJoCo environment initialization...")
    #     else:
    #         print(f"⚠️  Warning: ROS not available, skipping robot control initialization")
        
    print(f"\n🤖 Initializing MuJoCo environment...")
    # 根据action维度判断使用哪个环境
    # 16维动作 = depalletize任务（joint space），使用kuavo_depalletize_env
    # 20维动作 = depalletize任务（eef space），使用kuavo_depalletize_env（需要IK转换）
    # 其他维度 = com控制任务，使用kuavo_com_env
    if action_dim == 16:
        from robot_envs.kuavo_depalletize_env import GrabBoxMpcEnv
        mujoco_env = GrabBoxMpcEnv()
        print(f"✅ MuJoCo environment initialized (depalletize task - joint space)")
        print(f"   - Action dimension: 16 (14 arm joints + 2 claw positions)")
    elif action_dim == 20:
        from robot_envs.kuavo_depalletize_env import GrabBoxMpcEnv
        mujoco_env = GrabBoxMpcEnv()
        print(f"✅ MuJoCo environment initialized (depalletize task - eef space)")
        print(f"   - Action dimension: 20 (left_eef 9D + right_eef 9D + claw 2D)")
        print(f"   ⚠️  Note: EEF actions will need IK conversion to joint space for execution")
        # Import IK for eef->joint conversion
        try:
            from kuavo_ik.ik_library import IKAnalytical
            IK_AVAILABLE = True
        except ImportError:
            print(f"   ⚠️  Warning: IKAnalytical not available. Cannot convert EEF to joint space.")
            IK_AVAILABLE = False
    else:
        try:
            from robot_envs.kuavo_com_env import GrabBoxMpcEnv
            # GrootPolicy uses absolute actions by default
            mujoco_env = GrabBoxMpcEnv(use_action_history_reference=False)
            print(f"✅ MuJoCo environment initialized (com control task)")
            print(f"   - use_action_history_reference: False (absolute actions)")
        except ImportError:
            print("⚠️  Warning: robot_envs.kuavo_com_env not available. MuJoCo visualization disabled.")
            visualize_in_mujoco = False
            mujoco_env = None
    
    # ========= 可视化数据集里的groundtruth (如果启用RerunVisualizer) =========
    if vizer is not None:
        print(f"\n📊 Visualizing ground truth data...")
        # 加载所有ground truth actions用于可视化
        all_gt_actions = []
        all_gt_states = []
        
        temp_dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=0, batch_size=1, shuffle=False, drop_last=False
        )
        
        for batch in temp_dataloader:
            all_gt_actions.append(batch['action'][0].cpu().numpy())
            all_gt_states.append(batch['observation.state'][0].cpu().numpy())
        
        all_gt_actions = np.array(all_gt_actions)
        all_gt_states = np.array(all_gt_states)
        
        # 可视化ground truth actions
        for dim in range(action_dim):
            vizer.visualize_chunk(
                name=f"chunk/action_dim_{dim}/gt",
                chunk_data=all_gt_actions[:, dim],
                step_id=0,
                width=3.0
            )
        
        # 可视化observations
        for dim in range(obs_dim):
            vizer.visualize_chunk(
                name=f"obs/obs_{dim}",
                chunk_data=all_gt_states[:, dim],
                step_id=0,
                width=3.0
            )
        
        print(f"✅ Ground truth visualization ready")

    # ========= 开始模型推理 =========
    print("\n" + "="*80)
    print("🚀 Starting inference...")
    print("="*80 + "\n")
    
    last_data_step = 0
    predictions = []
    ground_truths = []
    inference_times = []  # 记录每次推理的时间
    
    # 使用tqdm显示进度（如果启用）
    iterator = tqdm(enumerate(dataloader), total=dataset.num_frames, desc="Processing") if show_progress else enumerate(dataloader)
    
    for data_step, batch in iterator:
        # 暂停控制（如果启用）
        if vizer is not None and kb is not None:
            time.sleep(0.05)
            if kb.paused:
                print(f'===== 暂停中，按下空格开始 =====')
            while kb.paused:
                time.sleep(0.1)
        
        # ✅ 准备observation - 使用Groot预处理器
        # 首先构建原始observation字典
        observation = {
            'observation.state': batch['observation.state'],
        }
        
        # 添加图像观测（根据CAMERA_COMPONENTS配置）
        if CONFIG_AVAILABLE:
            # 使用CAMERA_COMPONENTS来明确指定需要哪些相机
            camera_names = get_camera_names(CAMERA_COMPONENTS)
            for camera_name in camera_names:
                # 使用get_camera_observation_key获取正确的观测键名
                obs_key = get_camera_observation_key(camera_name, use_image_features=False)
                if obs_key in batch:
                    observation[obs_key] = batch[obs_key]
                else:
                    # 如果找不到，尝试直接使用相机名称作为键（向后兼容）
                    fallback_key = f"observation.images.{camera_name}"
                    if fallback_key in batch:
                        observation[fallback_key] = batch[fallback_key]
                    elif data_step == 0:
                        print(f"⚠️  Warning: Camera observation key '{obs_key}' not found in batch. Available keys: {[k for k in batch.keys() if 'image' in k.lower()]}")
        else:
            # 如果没有config，回退到原来的方法：添加所有图像观测
            for key in batch.keys():
                if 'image' in key.lower() and key.startswith('observation'):
                    observation[key] = batch[key]
        
        # 如果启用state_zero模式，将状态输入置零（用于验证模型对状态的依赖性）
        if state_zero:
            # 保持相同的形状和设备，但将所有状态值设为0
            observation['observation.state'] = torch.zeros_like(observation['observation.state'])
        
        # 如果启用image_zero模式，将所有图像输入置零（用于验证模型对图像的依赖性）
        if image_zero:
            for key in observation.keys():
                if 'image' in key:
                    # 保持相同的形状和设备，但将所有像素值设为0
                    observation[key] = torch.zeros_like(observation[key])
        
        # 如果启用cam_head_zero模式，将cam_head（image）的图像输入置零（用于验证模型对cam_head的依赖性）
        if cam_head_zero:
            # cam_head对应的相机名称是"image"，观测键是"observation.images.cam_head"
            cam_head_obs_key = "observation.images.cam_head"
            if cam_head_obs_key in observation:
                observation[cam_head_obs_key] = torch.zeros_like(observation[cam_head_obs_key])
            else:
                # 向后兼容：尝试使用"image"作为键名
                fallback_key = "observation.images.image"
                if fallback_key in observation:
                    observation[fallback_key] = torch.zeros_like(observation[fallback_key])
                elif data_step == 0:
                    print(f"⚠️  Warning: cam_head observation key not found. Available keys: {[k for k in observation.keys() if 'image' in k.lower()]}")
        
        # 添加 task 字段（language instruction）
        # 如果提供了 task_description，则使用它覆盖数据集中的 task；否则使用数据集原本的 task
        if task_description is not None:
            observation['task'] = task_description
        elif 'task' in batch:
            # 从 batch 中获取 task（LeRobotDataset 会在 __getitem__ 中添加 task 字段）
            batch_task = batch['task']
            # 处理 batch_task 可能是列表或字符串的情况
            if isinstance(batch_task, (list, tuple)) and len(batch_task) > 0:
                observation['task'] = batch_task[0]
            elif isinstance(batch_task, str):
                observation['task'] = batch_task
            else:
                # 如果类型不匹配，尝试转换为字符串
                observation['task'] = str(batch_task) if batch_task is not None else ""
        else:
            # 如果 batch 中没有 task，尝试从数据集元数据中获取（使用第一个任务作为默认值）
            if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'tasks') and len(dataset.meta.tasks) > 0:
                observation['task'] = dataset.meta.tasks.index[0]
            else:
                # 如果都没有，使用空字符串
                observation['task'] = ""
        
        # 获取ground truth action
        gt_action = batch['action'][0].cpu().numpy()  # (action_dim,)
        
        # 判断是否需要执行推理（根据infer_per_frame参数）
        should_infer = (data_step % infer_per_frame == 0)
        
        # 模型推理
        tic = time.time()
        if should_infer:
            # 需要推理：执行完整的推理流程
            # 使用预处理器处理输入
            processed_observation = preprocessor(observation)
            
            # 精确测量 predict_action_chunk 的推理时间
            # 使用 CUDA 同步确保准确测量 GPU 推理时间
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            inference_start = time.perf_counter()
            
            # 模型推理
            with torch.inference_mode():
                pred_actions = policy.predict_action_chunk(processed_observation)
            
            # 确保 GPU 操作完成后再记录结束时间
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            inference_end = time.perf_counter()
            inference_time = inference_end - inference_start
            inference_times.append(inference_time)
            
            # 打印action维度
            print(f"pred_actions shape: {pred_actions.shape}")
            
            # 调试：检查归一化后的6D旋转值范围（应该在[-1, 1]范围内，因为模型输出是归一化的）
            if action_dim == 20:
                # 检查第一个action的6D旋转值
                first_action = pred_actions[0, 0, :].cpu().numpy()
                left_rot6d = first_action[3:9]
                right_rot6d = first_action[12:18]
                print(f"🔍 Before postprocessor - 6D rotation value ranges:")
                print(f"   Left rot6d: min={left_rot6d.min():.4f}, max={left_rot6d.max():.4f}, mean={left_rot6d.mean():.4f}")
                print(f"   Right rot6d: min={right_rot6d.min():.4f}, max={right_rot6d.max():.4f}, mean={right_rot6d.mean():.4f}")
            
            # pred_actions shape: (batch_size, chunk_size, action_dim)
            # 使用 postprocessor 进行反归一化
            # postprocessor 期望输入是 (B, action_dim)，所以需要处理整个 chunk
            _, chunk_size, _ = pred_actions.shape
            processed_actions = []
            for i in range(chunk_size):
                # 提取单个 action: (B, action_dim)
                single_action = pred_actions[:, i, :]
                # 使用 postprocessor 进行反归一化
                processed_action = postprocessor(single_action)
                processed_actions.append(processed_action)
            
            # 调试：检查反归一化后的6D旋转值范围（应该保持原样，不被反归一化）
            if action_dim == 20 and len(processed_actions) > 0:
                first_processed = processed_actions[0][0].cpu().numpy()
                left_rot6d_processed = first_processed[3:9]
                right_rot6d_processed = first_processed[12:18]
                print(f"🔍 After postprocessor - 6D rotation value ranges:")
                print(f"   Left rot6d: min={left_rot6d_processed.min():.4f}, max={left_rot6d_processed.max():.4f}, mean={left_rot6d_processed.mean():.4f}")
                print(f"   Right rot6d: min={right_rot6d_processed.min():.4f}, max={right_rot6d_processed.max():.4f}, mean={right_rot6d_processed.mean():.4f}")
                print(f"   ⚠️  If 6D rotation values are still in [-1, 1] range, they should NOT be unnormalized!")
                print(f"   ⚠️  If values changed significantly, partial normalization may not be working correctly!")
            
            # 堆叠回 (B, chunk_size, action_dim)，然后转换为 numpy
            pred_actions_unnorm = torch.stack(processed_actions, dim=1)  # (B, chunk_size, action_dim)
            pred_chunk = pred_actions_unnorm[0].cpu().numpy()  # (chunk_size, action_dim)
            pred_action_single = pred_chunk[0]  # (action_dim,) - 取第一个 action
            
            # 保存预测结果供后续帧使用
            last_inferred_chunk = pred_chunk.copy()
            last_inference_step = data_step
        else:
            # 不需要推理：复用上一次的预测结果
            if last_inferred_chunk is not None:
                pred_chunk = last_inferred_chunk.copy()
                pred_action_single = pred_chunk[0]  # 取第一个action
            else:
                # 如果这是第一帧且infer_per_frame > 1，需要先推理一次
                if data_step == 0:
                    print(f"⚠️  Warning: First frame but no previous prediction. Performing inference anyway.")
                    # 执行推理
                    processed_observation = preprocessor(observation)
                    
                    # 精确测量 predict_action_chunk 的推理时间
                    # 使用 CUDA 同步确保准确测量 GPU 推理时间
                    if device.startswith('cuda'):
                        torch.cuda.synchronize()
                    inference_start = time.perf_counter()
                    
                    with torch.inference_mode():
                        pred_actions = policy.predict_action_chunk(processed_observation)
                    
                    # 确保 GPU 操作完成后再记录结束时间
                    if device.startswith('cuda'):
                        torch.cuda.synchronize()
                    inference_end = time.perf_counter()
                    inference_time = inference_end - inference_start
                    inference_times.append(inference_time)
                    
                    # 使用 postprocessor 进行反归一化
                    _, chunk_size, _ = pred_actions.shape
                    processed_actions = []
                    for i in range(chunk_size):
                        single_action = pred_actions[:, i, :]
                        processed_action = postprocessor(single_action)
                        processed_actions.append(processed_action)
                    
                    pred_actions_unnorm = torch.stack(processed_actions, dim=1)
                    pred_chunk = pred_actions_unnorm[0].cpu().numpy()
                    pred_action_single = pred_chunk[0]
                    
                    last_inferred_chunk = pred_chunk.copy()
                    last_inference_step = data_step
                else:
                    # 如果还没有预测结果，使用零向量（不应该发生）
                    print(f"⚠️  Warning: No previous prediction available at frame {data_step}. Using zeros.")
                    pred_action_single = np.zeros(action_dim)
                    pred_chunk = np.zeros((n_actions, action_dim))
    
        transition_steps = None
        if previous_resampled_action is not None:
            transition_steps = max(1, int(round(TARGET_CONTROL_FREQUENCY * CHUNK_TRANSITION_DURATION_S)))
        if should_infer or last_resampled_chunk is None:
            resampled_chunk = resample_chunk_with_claw_hold(
                pred_chunk,
                previous_action=previous_resampled_action,
                control_frequency=TARGET_CONTROL_FREQUENCY,
                source_dt=MODEL_ACTION_DT,
                arm_dims=arm_dims,
                claw_dims=claw_dims
            )
            lowpass_chunk = apply_lowpass_transition(
                resampled_chunk,
                previous_action=previous_resampled_action,
                alpha=LOWPASS_ALPHA,
                transition_steps=transition_steps,
                smooth_slice=smooth_slice
            )
            last_resampled_chunk = resampled_chunk
            last_lowpass_chunk = lowpass_chunk
        else:
            resampled_chunk = last_resampled_chunk
            lowpass_chunk = last_lowpass_chunk

        if lowpass_chunk is not None and lowpass_chunk.size > 0:
            previous_resampled_action = lowpass_chunk[-1].copy()
    
        inference_time = time.time() - tic
        
        # 如果启用发布手臂命令功能，且是20D EEF space，进行IK转换并发布
        if publish_arm_commands and action_dim == 20 and arm_joint_publisher is not None:
            try:
                # 将EEF action转换为joint action
                joint_action = convert_eef_action_to_joint_action(pred_action_single, model_type=ik_model_type)
                # 提取手臂关节角（前14维）
                arm_joints = joint_action[:14]  # (14,) - left_arm(7D) + right_arm(7D)
                # 发布到ROS
                publish_arm_joints_to_ros(arm_joints, arm_joint_publisher)
            except Exception as e:
                if data_step % 10 == 0:  # 每10帧打印一次错误，避免刷屏
                    print(f"⚠️  Warning: Failed to publish arm joints at frame {data_step}: {e}")
        
        # 保存预测和真实值
        predictions.append(pred_action_single)
        ground_truths.append(gt_action)
        
        # 计算每个维度的MSE和MAE
        for dim in range(action_dim):
            error = pred_action_single[dim] - gt_action[dim]
            mse = error ** 2
            mae = abs(error)
            
            if dim not in mse_per_action_dim:
                mse_per_action_dim[dim] = []
                mae_per_action_dim[dim] = []
            
            mse_per_action_dim[dim].append(mse)
            mae_per_action_dim[dim].append(mae)
        
        # 可视化（如果启用）
        if vizer is not None:
            # 显示图像 - 动态查找可用的相机图像
            if CONFIG_AVAILABLE:
                camera_names = get_camera_names(CAMERA_COMPONENTS)
                for camera_name in camera_names:
                    obs_key = get_camera_observation_key(camera_name, use_image_features=False)
                    fallback_key = f"observation.images.{camera_name}"
                    
                    # 优先使用新格式，如果不存在则使用旧格式
                    key_to_use = obs_key if obs_key in batch else fallback_key
                    if key_to_use in batch:
                        img = batch[key_to_use][0]  # (C, H, W)
                        # 从obs_key中提取组件名（如 observation.images.cam_head -> cam_head）
                        # 或者从camera_name映射到组件名
                        if obs_key in batch:
                            # 使用新格式：从 observation.images.cam_head 提取 cam_head
                            display_name = obs_key.replace('observation.images.', '')
                        else:
                            # 使用旧格式：从 camera_name 映射到组件名
                            display_name = CAMERA_KEY_MAPPING.get(camera_name, camera_name)
                        vizer.show_img(
                            name=f"images.{display_name}",
                            image_data=img.to("cpu"),
                            step_id=data_step
                        )
            else:
                # 如果没有config，回退到原来的方法
                for key in batch.keys():
                    if 'image' in key.lower() and key.startswith('observation'):
                        img = batch[key][0]  # (C, H, W)
                        camera_name = key.replace('observation.', '').replace('observation.images.', '')
                        vizer.show_img(
                            name=camera_name,
                            image_data=img.to("cpu"),
                            step_id=data_step
                        )
            
            # 可视化预测的chunk
            for dim in range(action_dim):
                # 可视化MSE
                vizer.visualize_chunk(
                    name=f"mse/action_dim_{dim}",
                    chunk_data=mse_per_action_dim[dim][-1],
                    step_id=data_step,
                    width=3.0,
                )
                
                if should_infer:
                    vizer.visualize_chunk(
                        name=f"chunk/action_dim_{dim}/pred_seg_{data_step}",
                        chunk_data=pred_chunk[:, dim],
                        step_id=data_step,
                        width=2
                    )

                    if last_data_step != data_step and last_data_step > 0:
                        vizer.del_chunk(
                            name=f"chunk/action_dim_{dim}/pred_seg_{last_data_step}",
                            chunk_data=pred_chunk[:, dim],
                            step_id=last_data_step,
                            width=0.5
                        )
            
            if should_infer and resampled_chunk is not None and resampled_chunk.size > 0:
                resampled_steps_axis = np.arange(control_step_cursor, control_step_cursor + resampled_chunk.shape[0], dtype=int)
                start_step = resampled_steps_axis[0]
                end_step = resampled_steps_axis[-1]
                start_point = resampled_chunk[0]
                end_point = resampled_chunk[-1]
                raw_start_point = pred_chunk[0]
                for dim in range(action_dim):
                    if hasattr(vizer, "clear_path"):
                        vizer.clear_path(f"chunk_interp/action_dim_{dim}/start_point")
                        vizer.clear_path(f"chunk_interp/action_dim_{dim}/start_point_raw")
                        vizer.clear_path(f"chunk_interp/action_dim_{dim}/end_point")
                        vizer.clear_path(f"chunk_lowpass/action_dim_{dim}/start_point")
                        vizer.clear_path(f"chunk_lowpass/action_dim_{dim}/start_point_raw")
                        vizer.clear_path(f"chunk_lowpass/action_dim_{dim}/end_point")
                    vizer.visualize_chunk(
                        name=f"chunk_interp/action_dim_{dim}/pred_seg_{data_step}",
                        chunk_data=resampled_chunk[:, dim],
                        step_id=0,
                        x_axis=resampled_steps_axis,
                        width=1.5,
                        color=COLOR_INTERP
                    )
                    vizer.visualize_points(
                        name=f"chunk_interp/action_dim_{dim}/start_point",
                        xs=np.array([start_step]),
                        ys=np.array([start_point[dim]]),
                        colors=np.array([[0, 255, 0]])
                    )
                    vizer.visualize_points(
                        name=f"chunk_interp/action_dim_{dim}/start_point_raw",
                        xs=np.array([start_step]),
                        ys=np.array([raw_start_point[dim]]),
                        colors=np.array([[0, 128, 0]])
                    )
                    vizer.visualize_points(
                        name=f"chunk_interp/action_dim_{dim}/end_point",
                        xs=np.array([end_step]),
                        ys=np.array([end_point[dim]]),
                        colors=np.array([[255, 0, 0]])
                    )
                    vizer.visualize_chunk(
                        name=f"chunk_lowpass/action_dim_{dim}/pred_seg_{data_step}",
                        chunk_data=lowpass_chunk[:, dim],
                        step_id=0,
                        x_axis=resampled_steps_axis,
                        width=1.5,
                        color=COLOR_LOWPASS
                    )
                    vizer.visualize_points(
                        name=f"chunk_lowpass/action_dim_{dim}/start_point",
                        xs=np.array([start_step]),
                        ys=np.array([lowpass_chunk[0, dim]]),
                        colors=np.array([[0, 255, 0]])
                    )
                    vizer.visualize_points(
                        name=f"chunk_lowpass/action_dim_{dim}/start_point_raw",
                        xs=np.array([start_step]),
                        ys=np.array([raw_start_point[dim]]),
                        colors=np.array([[0, 128, 0]])
                    )
                    vizer.visualize_points(
                        name=f"chunk_lowpass/action_dim_{dim}/end_point",
                        xs=np.array([end_step]),
                        ys=np.array([lowpass_chunk[-1, dim]]),
                        colors=np.array([[255, 0, 0]])
                    )
                control_step_cursor += resampled_chunk.shape[0]
        
        last_data_step = data_step
        
        # ========== 在mujoco里执行动作 (如果启用) =========
        if visualize_in_mujoco:
            action_np = pred_action_single[np.newaxis, :]  # (1, action_dim)
            
            # 根据action维度选择执行方法
            if action_dim == 16:
                # depalletize任务：16维动作 (14 arm joints + 2 claw positions)
                mujoco_env.exec_actions(
                    actions=action_np,
                    control_arm=True,
                    control_claw=True
                )
            elif action_dim == 20:
                # depalletize任务：20维动作 (left_eef 9D + right_eef 9D + claw 2D)
                # 需要将EEF pose转换为joint space
                if 'IK_AVAILABLE' in locals() and IK_AVAILABLE:
                    # 转换为joint space
                    joint_action = convert_eef_action_to_joint_action(pred_action_single)
                    joint_action_np = joint_action[np.newaxis, :]  # (1, 16)
                    mujoco_env.exec_actions(
                        actions=joint_action_np,
                        control_arm=True,
                        control_claw=True
                    )
                else:
                    print(f"⚠️  Warning: Cannot convert EEF to joint space. Skipping MuJoCo execution.")
            else:
                # com控制任务：使用绝对动作执行（GrootPolicy默认使用绝对动作）
                if mujoco_env is not None:
                    mujoco_env.exec_absolute_actions(
                        actions=action_np,
                        control_arm=True,
                        control_base=True,
                        control_wrench=False
                    )

    # ========= 打印最终统计结果 =========
    print("\n" + "="*80)
    print("📊 Final Statistics")
    print("="*80)
    
    # Action名称定义 - 根据action维度自动选择
    # 优先使用config中的action_names（如果可用且维度匹配）
    if CONFIG_AVAILABLE and action_names and len(action_names) == action_dim:
        eval_action_names = action_names
    elif action_dim == 16:
        # depalletize任务：16维动作 (14 arm joints + 2 claw positions)
        eval_action_names = [f"Arm_joint_{i+1}" for i in range(14)] + ["Left_claw", "Right_claw"]
    elif action_dim == 20:
        # depalletize任务：20维动作 (left_eef 9D + right_eef 9D + claw 2D)
        eval_action_names = (
            [f"Left_eef_pos_{i}" for i in range(3)] +  # Left eef position: x, y, z
            [f"Left_eef_rot6d_{i}" for i in range(6)] +  # Left eef 6D rotation: R11, R21, R31, R12, R22, R32
            [f"Right_eef_pos_{i}" for i in range(3)] +  # Right eef position: x, y, z
            [f"Right_eef_rot6d_{i}" for i in range(6)] +  # Right eef 6D rotation: R11, R21, R31, R12, R22, R32
            ["Left_gripper", "Right_gripper"]  # Claw positions
        )
    elif action_dim == 18:
        # 18维动作：Left_arm(7) + Right_arm(7) + Left_claw(1) + Right_claw(1) + Cmd_pose_z(1) + Cmd_pose_pitch(1)
        eval_action_names = (
            [f"arm_joint_{i+1}" for i in range(7)] +  # Left_arm: arm_joint_1-7
            [f"arm_joint_{i+8}" for i in range(7)] +  # Right_arm: arm_joint_8-14
            ["left_claw_position", "right_claw_position", "cmd_pose_z", "cmd_pose_pitch"]
        )
    elif action_dim == 24:
        # com控制任务：24维 = 9 COM + 14 Arm + 1 Gait
        eval_action_names = (
            ["COM_dx", "COM_dy", "COM_dz", "COM_dR11", "COM_dR21", "COM_dR31", "COM_dR12", "COM_dR22", "COM_dR32"] +
            [f"Arm_joint_{i+1}" for i in range(14)] +
            ["Gait_mode"]
        )
    else:
        # 其他维度：使用通用命名
        eval_action_names = [f"Action_dim_{i}" for i in range(action_dim)]
    
    print(f"\n{'Dimension':<20} {'MSE':<15} {'MAE':<15}")
    print("-" * 80)
    
    for dim in range(action_dim):
        mse_mean = np.mean(mse_per_action_dim[dim])
        mae_mean = np.mean(mae_per_action_dim[dim])
        dim_name = eval_action_names[dim] if dim < len(eval_action_names) else f"Dim_{dim}"
        print(f'{dim_name:<20} {mse_mean:<15.8f} {mae_mean:<15.8f}')
    
    overall_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(action_dim)])
    overall_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(action_dim)])
    
    print("-" * 80)
    print(f'{"Overall":<20} {overall_mse:<15.8f} {overall_mae:<15.8f}')
    
    # ========= 推理时间统计 =========
    if len(inference_times) > 0:
        print("\n" + "="*80)
        print("⏱️  Inference Time Statistics")
        print("="*80)
        avg_inference_time = np.mean(inference_times)
        min_inference_time = np.min(inference_times)
        max_inference_time = np.max(inference_times)
        median_inference_time = np.median(inference_times)
        std_inference_time = np.std(inference_times)
        
        print(f"Total inference calls: {len(inference_times)}")
        print(f"Average inference time: {avg_inference_time*1000:.2f} ms")
        print(f"Median inference time: {median_inference_time*1000:.2f} ms")
        print(f"Min inference time: {min_inference_time*1000:.2f} ms")
        print(f"Max inference time: {max_inference_time*1000:.2f} ms")
        print(f"Std inference time: {std_inference_time*1000:.2f} ms")
        
        # 计算理论最大推理频率（基于平均推理时间）
        # 这表示如果连续推理，理论上可以达到的最大频率
        if avg_inference_time > 0:
            max_frequency = 1.0 / avg_inference_time
            print(f"Max theoretical inference frequency: {max_frequency:.2f} Hz")
            print(f"  (Based on average inference time: {avg_inference_time*1000:.2f} ms)")
        
        print("="*80)
    else:
        print("\n⚠️  Warning: No inference time statistics available (no inference was performed)")
    
    # 分组统计 - 根据action维度选择统计方式
    print("\n📊 Grouped Statistics:")
    print("-" * 80)
    
    if action_dim == 16:
        # depalletize任务：16维 = 14 arm joints + 2 claw positions
        arm_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(14)])
        arm_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(14)])
        print(f'{"Arm (avg)":<20} {arm_mse:<15.8f} {arm_mae:<15.8f}')
        
        claw_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(14, 16)])
        claw_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(14, 16)])
        print(f'{"Claw (avg)":<20} {claw_mse:<15.8f} {claw_mae:<15.8f}')
    elif action_dim == 20:
        # depalletize任务：20维 = left_eef(9D) + right_eef(9D) + claw(2D)
        left_eef_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(9)])
        left_eef_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(9)])
        print(f'{"Left EEF (avg)":<20} {left_eef_mse:<15.8f} {left_eef_mae:<15.8f}')
        
        right_eef_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(9, 18)])
        right_eef_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(9, 18)])
        print(f'{"Right EEF (avg)":<20} {right_eef_mse:<15.8f} {right_eef_mae:<15.8f}')
        
        claw_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(18, 20)])
        claw_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(18, 20)])
        print(f'{"Claw (avg)":<20} {claw_mse:<15.8f} {claw_mae:<15.8f}')
        
        # 进一步细分：位置和旋转
        left_pos_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(3)])
        left_pos_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(3)])
        print(f'{"  Left EEF Pos":<20} {left_pos_mse:<15.8f} {left_pos_mae:<15.8f}')
        
        left_rot_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(3, 9)])
        left_rot_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(3, 9)])
        print(f'{"  Left EEF Rot6d":<20} {left_rot_mse:<15.8f} {left_rot_mae:<15.8f}')
        
        right_pos_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(9, 12)])
        right_pos_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(9, 12)])
        print(f'{"  Right EEF Pos":<20} {right_pos_mse:<15.8f} {right_pos_mae:<15.8f}')
        
        right_rot_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(12, 18)])
        right_rot_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(12, 18)])
        print(f'{"  Right EEF Rot6d":<20} {right_rot_mse:<15.8f} {right_rot_mae:<15.8f}')
    else:
        # com控制任务：标准分组统计
        com_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(9)])
        com_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(9)])
        print(f'{"COM (avg)":<20} {com_mse:<15.8f} {com_mae:<15.8f}')
        
        arm_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(9, 23)])
        arm_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(9, 23)])
        print(f'{"Arm (avg)":<20} {arm_mse:<15.8f} {arm_mae:<15.8f}')
        
        if action_dim > 23:
            gait_mse = np.mean(mse_per_action_dim[23])
            gait_mae = np.mean(mae_per_action_dim[23])
            print(f'{"Gait":<20} {gait_mse:<15.8f} {gait_mae:<15.8f}')
    
    print("="*80)

    if vizer is not None:
        print("\n[Offline Eval] Visualization active. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n✅ Exiting...")
    else:
        print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate GrootPolicy Model on Dataset with Lowpass Visualization',
        epilog='Evaluates a trained GrootPolicy model on a LeRobot dataset with chunk interpolation and lowpass filtering visualization.'
    )
    parser.add_argument('--ckpt-path', type=str, required=True,
                       help='Path to the model checkpoint directory')
    parser.add_argument('--dataset-root', '--dataset_root', type=str, required=True,
                       dest='dataset_root',
                       help='Path to the LeRobot dataset root directory')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode number to evaluate (default: 0)')
    parser.add_argument('--action-chunk-size', type=int, default=50,
                       help='Action chunk size (default: 50, should match training config)')
    parser.add_argument('--with-mujoco', action='store_true',
                       help='Visualize and execute in MuJoCo environment')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    parser.add_argument('--image-zero', action='store_true',
                       help='Set all image inputs to zero (for testing model dependency on images)')
    parser.add_argument('--state-zero', action='store_true',
                       help='Set all state inputs to zero (for testing model dependency on state)')
    parser.add_argument('--cam-head-zero', action='store_true',
                       help='Set cam_head (image) input to zero (for testing model dependency on cam_head)')
    parser.add_argument('--infer-per-frame', type=int, default=1,
                       help='Run policy inference every N frames (default: 1 = every frame)')
    parser.add_argument('--task-description', type=str, default=None,
                       help='Task description (language instruction) to override the task from dataset. If not provided, will use the task from dataset.')
    parser.add_argument('--publish-arm-commands', action='store_true',
                       help='Publish arm joint commands to ROS topic /kuavo_arm_traj. Requires 20D EEF action space and ROS available.')
    parser.add_argument('--ik-model-type', type=str, default='60',
                       choices=['45', '46', '60'],
                       help='Robot model type for IK solving (default: 60). Used when --publish-arm-commands is enabled.')

    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎯 GrootPolicy Dataset Evaluation with Lowpass Visualization")
    print("="*80)
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Dataset: {args.dataset_root}")
    print(f"Episode: {args.episode}")
    print(f"Action Chunk Size: {args.action_chunk_size}")
    print(f"MuJoCo Visualization: {args.with_mujoco}")
    print(f"Image Zero Mode: {args.image_zero}")
    print(f"State Zero Mode: {args.state_zero}")
    print(f"Cam Head Zero Mode: {args.cam_head_zero}")
    print(f"Infer Every N Frames: {args.infer_per_frame}")
    if args.task_description:
        print(f"Task Description (overridden): '{args.task_description}'")
    else:
        print(f"Task Description: Will use task from dataset")
    print(f"Publish Arm Commands: {args.publish_arm_commands}")
    if args.publish_arm_commands:
        print(f"IK Model Type: {args.ik_model_type}")
    print("="*80)
    
    eval_on_dataset(
        ckpt_path=args.ckpt_path,
        lerobot_dataset_path=args.dataset_root,
        episode=args.episode,
        n_actions=args.action_chunk_size,
        visualize_in_mujoco=args.with_mujoco,
        show_progress=not args.no_progress,
        image_zero=args.image_zero,
        state_zero=args.state_zero,
        cam_head_zero=args.cam_head_zero,
        infer_per_frame=args.infer_per_frame,
        task_description=args.task_description,
        publish_arm_commands=args.publish_arm_commands,
        ik_model_type=args.ik_model_type
    )
