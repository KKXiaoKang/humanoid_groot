import sys, os

from pandas.core.missing import F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import rosbag
from sensor_msgs.msg import JointState
import json
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest

def resample_actions_with_speed_limit(actions: np.ndarray, dt: float, v_max, arm_dims: slice = slice(None), constant_velocity: bool = False):
    '''
        resample actions (joint positions) which satisfy joint velocity limits
        Only applies speed limit to arm dimensions, other dimensions are interpolated normally
        
        Args:
            actions: Array of shape (T, D) where T is number of timesteps, D is action dimension
            dt: Time interval between actions (seconds)
            v_max: Maximum velocity (rad/s). Can be scalar or array of shape (arm_dim,)
            arm_dims: Slice or indices specifying which dimensions are arm joints (to apply speed limit)
            constant_velocity: If True, ensure each segment executes at constant velocity (within v_max limit)
        
        Returns:
            Array of shape (M, D) where M >= T, with speed-limited resampling
    '''
    T, D = actions.shape
    actions = np.asarray(actions)
    
    # Convert v_max to array format
    v_max = np.asarray(v_max)
    if v_max.ndim == 0:
        # If scalar, apply to all arm dimensions
        if isinstance(arm_dims, slice):
            arm_dim_size = len(range(*arm_dims.indices(D)))
        else:
            arm_dim_size = len(arm_dims)
        v_max = np.full(arm_dim_size, v_max)
    
    new_actions = [actions[0]]

    for t in range(T-1):
        a0 = actions[t]
        a1 = actions[t+1]

        # Extract arm dimensions
        if isinstance(arm_dims, slice):
            arm_a0 = a0[arm_dims]
            arm_a1 = a1[arm_dims]
        else:
            arm_a0 = a0[arm_dims]
            arm_a1 = a1[arm_dims]

        delta = arm_a1 - arm_a0
        v_required = np.abs(delta) / dt

        if constant_velocity:
            # 匀速模式：确保所有关节以恒定速度执行，且不超过v_max限制
            # 策略：找到所有关节中速度限制最严格的（即需要最长时间完成的），
            # 然后所有关节都以相同的总时间完成，每个关节以恒定速度（受v_max限制）执行
            # 但每个关节的最终位置必须等于目标位置（arm_a1）
            
            # 计算每个关节的所需速度
            v_per_joint = np.abs(delta) / dt
            
            # 计算每个关节在v_max限制下所需的时间
            # 如果某个关节的所需速度超过v_max，则需要更多时间
            time_per_joint = np.where(
                v_per_joint > v_max,
                np.abs(delta) / v_max,  # 受速度限制，需要更长时间
                dt  # 不受限制，使用原始时间
            )
            
            # 找到最长的执行时间（瓶颈关节）
            total_time = np.max(time_per_joint) if len(time_per_joint) > 0 and np.any(time_per_joint > 0) else dt
            total_time = max(total_time, dt)  # 至少需要dt时间
            
            # 计算子步数
            num_sub = max(1, int(np.ceil(total_time / dt)))
            
            # 计算每个关节在total_time内的实际速度
            # 关键：每个关节必须完成原始位移delta，所以实际速度 = |delta| / total_time
            # 由于total_time已经考虑了所有关节的速度限制，所以 |delta|/total_time <= v_max 应该总是成立
            # 但为了安全起见，仍然应用v_max限制
            actual_v_per_joint = np.minimum(np.abs(delta) / total_time, v_max)
            
            # 计算每个关节在total_time内的实际位移
            # 关键：必须确保最终位置等于目标位置arm_a1
            # 由于total_time已经足够长，使得所有关节都能在total_time内以不超过v_max的速度完成位移
            # 理论上：actual_v_per_joint * total_time = |delta|，所以actual_delta = delta
            # 但为了确保速度恒定，我们使用actual_v_per_joint * total_time来计算
            actual_delta = np.sign(delta) * actual_v_per_joint * total_time
            
            # 验证：确保最终位置正确（理论上应该相等，允许小的数值误差）
            # 由于total_time的计算方式，actual_delta应该等于delta（在数值精度范围内）
            # 如果误差太大，说明逻辑有问题，使用delta作为后备方案
            max_error = np.max(np.abs(actual_delta - delta))
            if max_error > 1e-6:
                # 这种情况理论上不应该发生，但为了安全，使用delta确保到达目标位置
                # 注意：这里不使用rospy，因为这是通用函数，可能不在ROS环境中
                print(f"⚠️  Constant velocity mode: actual_delta differs from delta by {max_error:.6f}, using delta to ensure target position")
                actual_delta = delta
            
            # 生成匀速插值（所有关节同步完成，每个关节以恒定速度执行）
            for s in range(1, num_sub + 1):
                alpha = s / num_sub
                # 对于手臂关节，使用匀速插值（基于实际位移）
                # 这样每个关节以恒定速度执行：速度 = actual_delta / total_time = actual_v_per_joint
                new_arm = arm_a0 + actual_delta * alpha
                # 对于其他维度，使用线性插值
                new_a = a0.copy()
                if isinstance(arm_dims, slice):
                    new_a[arm_dims] = new_arm
                else:
                    new_a[arm_dims] = new_arm
                # 其他维度线性插值
                other_dims = np.ones(D, dtype=bool)
                if isinstance(arm_dims, slice):
                    other_dims[arm_dims] = False
                else:
                    other_dims[arm_dims] = False
                new_a[other_dims] = a0[other_dims] * (1 - alpha) + a1[other_dims] * alpha
                new_actions.append(new_a)
        else:
            # 原有模式：只限制最大速度，不保证匀速
            # Calculate scale factor based on arm velocity limits
            scale = np.max(v_required / v_max) if len(v_max) > 0 and np.any(v_max > 0) else 1.0
            scale = max(scale, 1.0)

            # number of sub_steps
            num_sub = int(np.ceil(scale))

            # interpolate all dimensions
            for s in range(1, num_sub + 1):
                alpha = s / num_sub
                new_a = a0 * (1 - alpha) + a1 * alpha
                new_actions.append(new_a)

    return np.stack(new_actions, axis=0)

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
from configs.config import topic_info, TASK_DATA_MODE, get_camera_observation_key, get_camera_names, CAMERA_COMPONENTS, ACTION_COMPONENTS
from configs.config import ROBOT_VERSION
# 使用GrootPolicy模型
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# MergeVLA 适配层支持
from lerobot.policies.groot.weight_merge_groot import DistributionAdapter
from lerobot.policies.groot.groot_n1 import BACKBONE_FEATURE_KEY
import glob
import logging

# import torchvision
# import matplotlib.pyplot as plt
from pathlib import Path
import torch
import time
import argparse
import rospy

# MergeVLA 日志
logging.basicConfig(level=logging.INFO)
mergevla_logger = logging.getLogger("MergeVLA")
from std_msgs.msg import Float64MultiArray
from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.robot_sdk import RobotSDK
from kuavo_humanoid_sdk.msg.kuavo_msgs.srv import (changeArmCtrlMode, changeArmCtrlModeRequest)

# Default MODEL_ACTION_DT - can be overridden by command line argument
# This represents the time interval between predicted actions during training
# Smaller values = higher inference frequency (e.g., 0.1 = 10 Hz, 0.05 = 20 Hz, 0.033 = 30 Hz)
DEFAULT_MODEL_ACTION_DT = 0.1
MODEL_ACTION_DT = DEFAULT_MODEL_ACTION_DT  # Will be updated by command line argument if provided
MODEL_ACTION_FREQUENCY = 1.0 / MODEL_ACTION_DT
TARGET_CONTROL_FREQUENCY = 100.0
TARGET_CONTROL_DT = 1.0 / TARGET_CONTROL_FREQUENCY
CHUNK_TRANSITION_DURATION_S = 0.2  # seconds of low-pass smoothing at chunk boundary
LOWPASS_ALPHA = 0.85  # closer to 1 => smoother (slower) transitions
ENABLE_CHUNK_TRANSITION_LOWPASS = True  # Enable/disable low-pass filtering at chunk boundaries (default: False, only linear interpolation within chunks)
FIRST_MODEL_INFERENCE = True

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

def change_arm_ctrl_mode(control_mode):
    rospy.wait_for_service('/humanoid_change_arm_ctrl_mode')
    try:
        change_mode = rospy.ServiceProxy('/humanoid_change_arm_ctrl_mode', changeArmCtrlMode)
        req = changeArmCtrlModeRequest()
        req.control_mode = control_mode
        res = change_mode(req)
        if res.result:
            rospy.loginfo("手臂控制模式已更改为 %d", control_mode)
        else:
            rospy.logerr("无法将手臂控制模式更改为 %d", control_mode)
    except rospy.ServiceException as e:
        rospy.logerr("服务调用失败: %s", e)

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

def load_and_replay_init_trajectory(bag_path: str, env, control_arm: bool = True, control_claw: bool = True):
    """
    从rosbag文件中加载初始轨迹并回放
    
    Args:
        bag_path: rosbag文件路径
        env: GrabBoxMpcEnv环境实例
        control_arm: 是否控制手臂
        control_claw: 是否控制夹爪
    """
    if not os.path.exists(bag_path):
        rospy.logerr(f"Bag file not found: {bag_path}")
        return False
    
    rospy.loginfo(f"Loading initial trajectory from bag: {bag_path}")
    
    # 期望的关节名称顺序（与publish_target_arm_claw中的顺序一致）
    expected_joint_names = [
        "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", 
        "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
        "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", 
        "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
    ]
    
    # 读取bag文件中的JointState消息
    joint_states = []
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            topic_name = '/mm_kuavo_arm_traj'
            
            # 检查话题是否存在
            bag_info = bag.get_type_and_topic_info()
            if topic_name not in bag_info[1]:
                rospy.logwarn(f"Topic {topic_name} not found in bag file. Available topics: {list(bag_info[1].keys())}")
                return False
            
            # 读取所有JointState消息
            # 注意：由于已经通过topic_name过滤，所有消息都应该是JointState类型
            # 但isinstance检查可能不工作（rosbag可能返回包装类型），所以直接使用消息
            message_count = 0
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                message_count += 1
                # 直接使用消息，不进行类型检查（因为已经通过topic过滤）
                joint_states.append({
                    'timestamp': t.to_sec(),
                    'msg': msg
                })
            
            rospy.loginfo(f"Read {message_count} messages from topic {topic_name}")
            
            # 按时间戳排序
            joint_states.sort(key=lambda x: x['timestamp'])
            
            if len(joint_states) == 0:
                rospy.logwarn(f"No JointState messages found in topic {topic_name}")
                return False
            
            rospy.loginfo(f"Loaded {len(joint_states)} joint states from bag file")
            
    except Exception as e:
        rospy.logerr(f"Error loading bag file: {e}")
        return False
    
    # 获取当前夹爪状态（用于填充16维action）
    current_claw_state = np.array([0.0, 0.0])  # 默认值
    try:
        obs_data, _, _, robot_obs, _ = env.get_obs()
        if 'claw_state' in robot_obs and len(robot_obs['claw_state']) > 0:
            # 获取最新的夹爪状态
            claw_data = robot_obs['claw_state']
            if claw_data.ndim == 2:
                # 如果是2D数组，取最后一行
                current_claw_state = np.array(claw_data[-1], dtype=np.float32)
            elif claw_data.ndim == 1:
                # 如果是1D数组，直接使用
                current_claw_state = np.array(claw_data, dtype=np.float32)
            
            # 确保是2维
            if current_claw_state.shape[0] != 2:
                rospy.logwarn(f"Claw state has unexpected shape: {current_claw_state.shape}, using default")
                current_claw_state = np.array([0.0, 0.0])
            else:
                rospy.loginfo(f"Current claw state: {current_claw_state}")
    except Exception as e:
        rospy.logwarn(f"Could not get current claw state: {e}, using default [0.0, 0.0]")
        current_claw_state = np.array([0.0, 0.0])
    
    # 回放轨迹（按照rosbag中的时间戳间隔）
    rospy.loginfo("Starting trajectory replay...")
    replay_start_time = time.time()
    bag_start_timestamp = joint_states[0]['timestamp']  # bag中的第一个时间戳
    
    for i, joint_data in enumerate(joint_states):
        msg = joint_data['msg']
        bag_timestamp = joint_data['timestamp']
        
        # 提取关节位置
        # JointState的position是角度（度），需要转换为弧度
        if len(msg.position) < 14:
            rospy.logwarn(f"JointState message {i} has insufficient positions: {len(msg.position)} < 14")
            continue
        
        # 直接使用position数组的前14个元素（跳过名称检查）
        # bag文件中的关节顺序是: arm_joint_1 ~ arm_joint_14
        # 对应: 左手7个关节 + 右手7个关节
        # 直接使用前14个位置，假设顺序正确
        arm_action = np.deg2rad(np.array(msg.position[:14]))
        
        # 组合成16维action: [14个手臂关节, 2个夹爪位置]
        action = np.concatenate([arm_action, current_claw_state])
        
        # 计算应该等待的时间（按照bag中的时间戳间隔）
        if i == 0:
            # 第一个动作立即执行
            expected_elapsed = 0.0
        else:
            # 计算从bag开始到当前消息应该经过的时间
            bag_elapsed = bag_timestamp - bag_start_timestamp
            # 计算实际经过的时间
            actual_elapsed = time.time() - replay_start_time
            # 需要等待的时间
            expected_elapsed = bag_elapsed - actual_elapsed
        
        # 如果时间还没到，等待
        if expected_elapsed > 0:
            time.sleep(expected_elapsed)
        
        # 执行动作（不使用env.exec_actions，因为它会按照100Hz频率控制，我们直接发布）
        # 直接使用env的target_publisher发布，不经过env.exec_actions的频率控制
        env.target_publisher.publish_target_arm_claw(
            arm_action=arm_action,
            claw_action=current_claw_state,
            control_arm=control_arm,
            control_claw=control_claw
        )
        
        # 打印进度
        if (i + 1) % 10 == 0 or i == len(joint_states) - 1:
            elapsed = time.time() - replay_start_time
            bag_total_time = joint_states[-1]['timestamp'] - bag_start_timestamp
            rospy.loginfo(f"Replayed {i + 1}/{len(joint_states)} steps (elapsed: {elapsed:.2f}s, bag time: {bag_total_time:.2f}s)")
    
    total_time = time.time() - replay_start_time
    bag_total_time = joint_states[-1]['timestamp'] - bag_start_timestamp
    rospy.loginfo(f"Trajectory replay completed! Real time: {total_time:.2f}s, Bag time: {bag_total_time:.2f}s, {len(joint_states)} steps")
    
    return True

def reset_inference_state(policy, env):
    """
    重置推理状态，为下一次推理做准备
    
    Args:
        policy: GrootPolicy模型实例
        env: GrabBoxMpcEnv环境实例
    """
    rospy.loginfo("🔄 Resetting inference state...")
    
    # 重置policy状态
    policy.reset()
    rospy.loginfo("   ✅ Policy reset")
    
    # 重置夹爪锁定状态
    env.reset_claw_lock()
    rospy.loginfo("   ✅ Claw lock reset")
    
    # 等待buffer重新ready（buffer会自动保持最新数据，但确保数据充足）
    rospy.loginfo("   ⏳ Waiting for buffer to be ready...")
    env.obs_buffer.wait_buffer_ready()
    rospy.loginfo("   ✅ Buffer ready")
    
    rospy.loginfo("✅ Inference state reset complete")


# ============================================================
# MergeVLA 适配层加载函数
# ============================================================

def load_merge_config(model_path: str) -> dict | None:
    """加载融合配置"""
    merge_config_path = Path(model_path) / "merge_config.json"
    if merge_config_path.exists():
        with open(merge_config_path, 'r') as f:
            return json.load(f)
    return None


def load_adapter(policy, model_path: str, merge_config: dict, device: str = "cuda:0"):
    """加载 MergeVLA 适配层"""
    from safetensors.torch import load_file
    
    adapter_type = merge_config.get('adapter_type', 'sparse_lora')
    lora_rank = merge_config.get('lora_rank', 16)
    num_tasks = merge_config.get('num_tasks', 2)
    sparsity = merge_config.get('sparsity', 0.5)
    
    # 获取 hidden_size
    groot_model = policy._groot_model
    hidden_size = 2048
    if hasattr(groot_model, 'action_head') and hasattr(groot_model.action_head, 'config'):
        hidden_size = getattr(groot_model.action_head.config, 'backbone_embedding_dim', 2048)
    
    mergevla_logger.info(f"   📐 hidden_size: {hidden_size}, adapter_type: {adapter_type}, lora_rank: {lora_rank}")
    
    # 创建适配层
    adapter = DistributionAdapter(
        hidden_size=hidden_size,
        adapter_type=adapter_type,
        lora_rank=lora_rank,
        num_tasks=num_tasks,
        sparsity=sparsity,
    )
    
    # 加载适配层权重
    model_path = Path(model_path)
    safetensors_files = glob.glob(str(model_path / "model*.safetensors"))
    
    state_dict = {}
    for f in sorted(safetensors_files):
        state_dict.update(load_file(f))
    
    # 提取适配层权重
    adapter_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_groot_model.distribution_adapter.'):
            new_key = key[len('_groot_model.distribution_adapter.'):]
            adapter_state_dict[new_key] = value
    
    if not adapter_state_dict:
        raise RuntimeError("未找到 distribution_adapter 权重")
    
    adapter.load_state_dict(adapter_state_dict, strict=True)
    adapter.eval()
    adapter.to(device)
    
    # ⚠️ 关键修复：限制 residual_scale 以避免 chunk 变平
    # 实验发现：当 residual_scale > 0.10 时，Flow Matching 迭代去噪会崩溃
    # 导致所有时间步收敛到相似的值（chunk 变平）
    MAX_SAFE_RESIDUAL_SCALE = 0.10
    with torch.no_grad():
        if hasattr(adapter, 'adapter') and hasattr(adapter.adapter, 'residual_scale'):
            original_scale = adapter.adapter.residual_scale.item()
            if original_scale > MAX_SAFE_RESIDUAL_SCALE:
                adapter.adapter.residual_scale.fill_(MAX_SAFE_RESIDUAL_SCALE)
                mergevla_logger.warning(f"   ⚠️ 修复 residual_scale: {original_scale:.4f} → {MAX_SAFE_RESIDUAL_SCALE:.4f}")
                mergevla_logger.warning(f"      原因：residual_scale > 0.10 会导致 Flow Matching 崩溃，chunk 变平")
            else:
                mergevla_logger.info(f"   ✅ residual_scale={original_scale:.4f} 在安全范围内")
    
    mergevla_logger.info(f"   ✅ 适配层加载成功，参数量: {sum(p.numel() for p in adapter.parameters()):,}")
    return adapter


def wrap_policy_with_adapter(
    policy, 
    adapter,
    task_type: str = None,
    use_smart_routing: bool = False,
    swap_task_mapping: bool = False,  # ⚠️ 是否交换任务映射
):
    """
    包装 GrootPolicy，使其在推理时使用适配层
    
    Args:
        policy: GrootPolicy 实例
        adapter: DistributionAdapter 实例
        task_type: 任务类型 ("narrower", "wider", None)
                   - "narrower": 使用 task_id=0 的适配器参数（或 task_id=1 如果 swap）
                   - "wider": 使用 task_id=1 的适配器参数（或 task_id=0 如果 swap）
                   - None: 如果 use_smart_routing=True，使用智能任务路由；
                          否则使用所有任务的平均
        use_smart_routing: ⭐ 是否使用 MergeVLA 风格的智能任务路由
                          当 task_type=None 时，根据输入特征自动推断任务类型
        swap_task_mapping: ⚠️ 是否交换任务映射（narrower↔wider）
                          如果 Smart Routing 结果相反，使用此选项
    """
    import torch
    
    # ⚠️ 关键：根据任务类型设置 task_id
    if task_type == "narrower":
        # 如果 swap，narrower 使用 task_id=1
        actual_task_id = 1 if swap_task_mapping else 0
        fixed_task_id = torch.tensor([actual_task_id], device=next(adapter.parameters()).device)
        if swap_task_mapping:
            mergevla_logger.info(f"   ⚠️ 使用任务路由: task_type=narrower → task_id=1 (已交换映射)")
        else:
            mergevla_logger.info(f"   ⚠️ 使用任务路由: task_type=narrower (task_id=0)")
    elif task_type == "wider":
        # 如果 swap，wider 使用 task_id=0
        actual_task_id = 0 if swap_task_mapping else 1
        fixed_task_id = torch.tensor([actual_task_id], device=next(adapter.parameters()).device)
        if swap_task_mapping:
            mergevla_logger.info(f"   ⚠️ 使用任务路由: task_type=wider → task_id=0 (已交换映射)")
        else:
            mergevla_logger.info(f"   ⚠️ 使用任务路由: task_type=wider (task_id=1)")
    else:
        fixed_task_id = None
        if use_smart_routing:
            mergevla_logger.info(f"   ⭐ 使用 MergeVLA 智能任务路由 (Test-Time Task Routing)")
            mergevla_logger.info(f"      根据输入特征自动推断任务相关性，无需手动指定任务类型")
        else:
            mergevla_logger.warning(f"   ⚠️ 未指定任务类型，使用所有任务的平均（可能导致动作混乱！）")
            mergevla_logger.warning(f"      建议：使用 --smart-routing 启用智能任务路由")
            mergevla_logger.warning(f"      或者：使用 --task-type narrower/wider 指定任务类型")
    
    def get_action_with_adapter(inputs: dict, **kwargs):
        backbone_inputs, action_inputs = policy._groot_model.prepare_input(inputs)
        backbone_outputs = policy._groot_model.backbone(backbone_inputs)
        
        # 通过适配层（支持智能任务路由）
        backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]
        adapted_features = adapter(
            backbone_features, 
            task_id=fixed_task_id,
            use_smart_routing=use_smart_routing
        )
        backbone_outputs[BACKBONE_FEATURE_KEY] = adapted_features
        
        # 通过 action_head
        rtc_enabled = kwargs.pop('rtc_enabled', policy._groot_model._rtc_enabled())
        return policy._groot_model.action_head.get_action(
            backbone_outputs, action_inputs, rtc_enabled=rtc_enabled, **kwargs
        )
    
    policy._groot_model.get_action = get_action_with_adapter
    routing_mode = "智能任务路由" if use_smart_routing else ("固定任务" if fixed_task_id is not None else "简单平均")
    mergevla_logger.info(f"   ✅ Policy 已包装适配层 (路由模式: {routing_mode})")


def load_model_and_env(ckpt_path, model_type, action_chunk_size=50, enable_gui=False, rotate_head_camera=False, state_zero=False, task_description=None, claw_lock_threshold=50.0, claw_lock_count_threshold=5, claw_locked_value=90.0, disable_adapter=False, task_type=None, use_smart_routing=False, swap_task_mapping=False):
    """
    加载模型和环境（只执行一次，避免重复加载）
    
    支持 MergeVLA 融合模型：自动检测并加载适配层
    
    Args:
        ckpt_path: 模型checkpoint路径
        model_type: 模型类型（已废弃，保留用于兼容性）
        action_chunk_size: 动作块大小
        enable_gui: 是否启用GUI
        rotate_head_camera: 是否旋转头部相机
        state_zero: 是否将状态置零
        task_description: 任务描述字符串，如果为None则使用默认值
        disable_adapter: 是否禁用 MergeVLA 适配层（调试用）
        task_type: 任务类型 ("narrower", "wider", None)
                   ⚠️ 关键：对于 sparse_lora 适配器，如果不使用智能路由，
                   必须指定任务类型！否则会使用所有任务的平均。
        use_smart_routing: ⭐ 是否使用 MergeVLA 风格的智能任务路由
                          当任务身份未知时，根据模型内部参数子空间
                          自动推断任务相关性（参考 MergeVLA 论文 Section 3.3）
    
    Returns:
        tuple: (policy, preprocessor, postprocessor, env, task_description, device, adapter)
    """
    # ---------- 1. load GrootPolicy from checkpoint ---------------
    device = "cuda:0"
    print(" =================== Loading GrootPolicy =================== ")
    policy = GrootPolicy.from_pretrained(Path(ckpt_path), strict=False)
    policy.config.device = device
    policy.config.n_action_steps = action_chunk_size
    
    # 确定要使用的任务描述
    if task_description is None:
        # 使用通用默认值
        task_description = "Depalletize the box"
        print(f"📝 Using default task description: '{task_description}'")
    else:
        print(f"📝 Using provided task description: '{task_description}'")
    
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
    
    policy.eval()
    policy.to(device)
    policy.reset()
    
    # ============================================================
    # MergeVLA 适配层加载
    # ============================================================
    adapter = None
    merge_config = load_merge_config(ckpt_path)
    
    if merge_config is not None:
        print("\n" + "=" * 60)
        print("🔗 检测到 MergeVLA 融合模型")
        print("=" * 60)
        print(f"   📋 融合方法: {merge_config.get('merge_method', 'N/A')}")
        print(f"   📋 适配器类型: {merge_config.get('adapter_type', 'sparse_lora')}")
        print(f"   📋 LoRA Rank: {merge_config.get('lora_rank', 16)}")
        print(f"   📋 任务数: {merge_config.get('num_tasks', 2)}")
        print(f"   📋 稀疏度: {merge_config.get('sparsity', 0.5)}")
        
        # ⭐ MergeVLA Section 4.1: 参数级稀疏掩码融合配置
        use_sparse_merge = merge_config.get('use_sparse_merge', None)
        sparse_merge_lambda = merge_config.get('sparse_merge_lambda', None)
        if use_sparse_merge is not None:
            print(f"\n   🔥 MergeVLA Section 4.1 参数级稀疏掩码融合:")
            print(f"      - 稀疏掩码融合: {'✅ 启用' if use_sparse_merge else '❌ 禁用'}")
            if use_sparse_merge and sparse_merge_lambda is not None:
                print(f"      - Lambda (容忍度系数): {sparse_merge_lambda}")
                print(f"      - 公式: S_m = I[|τ_m| > λ|τ_merge - τ_m|]")
        
        if disable_adapter:
            print("   ⚠️ --disable-adapter 已设置，跳过适配层加载")
        else:
            try:
                adapter = load_adapter(policy, ckpt_path, merge_config, device)
                wrap_policy_with_adapter(
                    policy, 
                    adapter,
                    task_type=task_type,
                    use_smart_routing=use_smart_routing,
                    swap_task_mapping=swap_task_mapping,  # ⚠️ 交换任务映射
                )
                print("=" * 60)
            except Exception as e:
                print(f"   ❌ 适配层加载失败: {e}")
                print("   ⚠️ 将使用不带适配层的模型继续运行")
                print("=" * 60)
    else:
        print("\n📦 标准 GROOT 模型（非 MergeVLA 融合模型）")
    
    # Initialize real-time environment
    env = GrabBoxMpcEnv(
        claw_lock_threshold=claw_lock_threshold,
        claw_lock_count_threshold=claw_lock_count_threshold,
        claw_locked_value=claw_locked_value
    )
    print(f"🤖 Environment initialized for depalletize task")
    print(" ======================  Waiting for buffer ready ====================== ")
    env.obs_buffer.wait_buffer_ready()
    print(" ======================  Buffer ready ====================== ")
    time.sleep(1)
    
    return policy, preprocessor, postprocessor, env, task_description, device, adapter

def set_arm_quick_mode(enable: bool) -> bool:
    """开关手臂快速模式"""
    rospy.loginfo(f"call set_arm_quick_mode:{enable}")
    try:
        rospy.wait_for_service('/enable_lb_arm_quick_mode', timeout=5.0)
        cli = rospy.ServiceProxy('/enable_lb_arm_quick_mode', SetBool)
        resp = cli(enable)
        if resp.success:
            rospy.loginfo(f"Successfully {'enabled' if enable else 'disabled'} arm quick mode")
            return True
        else:
            rospy.logwarn(f"Failed to {'enable' if enable else 'disable'} arm quick mode")
            return False
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return False

def run_inference_loop(policy, preprocessor, postprocessor, env, task_description, device, 
                       control_arm=True, control_claw=True, action_chunk_size=50, 
                       enable_gui=False, rotate_head_camera=False, state_zero=False,
                       is_first_inference=True, chunk_start=None, chunk_end=None, model_action_dt=None,
                       sync_mode=False, max_joint_velocity=None, constant_velocity=False, action_stride=1):
    """
    运行推理循环（可以多次调用，每次调用开始新的推理会话）
    
    Args:
        policy: 已加载的GrootPolicy模型
        preprocessor: 预处理器
        postprocessor: 后处理器（用于反归一化）
        env: 已初始化的GrabBoxMpcEnv环境
        task_description: 任务描述
        device: 设备
        control_arm: 是否控制手臂
        control_claw: 是否控制夹爪
        action_chunk_size: 动作块大小
        enable_gui: 是否启用GUI
        rotate_head_camera: 是否旋转头部相机
        state_zero: 是否将状态置零
        is_first_inference: 是否是第一次推理（第一次会加载bag文件，后续使用json文件重置）
        chunk_start: 要执行的chunk起始索引（从0开始，包含）。如果为None，从第一个action开始
        chunk_end: 要执行的chunk结束索引（从0开始，包含）。如果为None，执行到最后一个action
        model_action_dt: 模型动作时间间隔（秒），控制推理频率。如果为None，使用全局MODEL_ACTION_DT
        sync_mode: 是否使用同步推理模式。如果True，推理一个chunk -> 执行完整个chunk -> get_obs -> 再推理下一个chunk
        max_joint_velocity: 最大关节速度限制（rad/s）。如果提供，将对arm关节应用速度限制
        constant_velocity: 是否启用匀速模式。如果True，确保动作执行时速度恒定（在max_joint_velocity限制内）
        action_stride: 动作采样间隔，用于加速执行。例如：action_stride=2表示每隔2个action执行一次，跳过中间的action。
                       设置为1表示不跳过任何action（正常速度）。设置为N表示执行速度约为原来的N倍。
                       注意：这不会改变速度限制，只是减少执行的action数量。
    
    Returns:
        bool: True表示正常退出（按q），False表示被中断（Ctrl+C）
    """
    global FIRST_MODEL_INFERENCE
    # 使用传入的model_action_dt或全局MODEL_ACTION_DT
    if model_action_dt is None:
        model_action_dt = MODEL_ACTION_DT
    model_action_frequency = 1.0 / model_action_dt
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
    if rotate_head_camera:
        print(f"🔄 Head camera rotation enabled: images from 'image' camera will be rotated 180 degrees")
    if state_zero:
        print(f"⚠️  STATE ZERO MODE: All state inputs will be set to zero (for dependency testing)")
    if chunk_start is not None or chunk_end is not None:
        start_idx = chunk_start if chunk_start is not None else 0
        end_idx = chunk_end if chunk_end is not None else action_chunk_size - 1
        print(f"⏭️  Chunk selection: will execute actions from index {start_idx} to {end_idx} (inclusive)")
    if sync_mode:
        print(f"🔄 Sync mode: Enabled (inference -> execute chunk -> get_obs -> repeat)")
    else:
        print(f"⚡ Model action DT: {model_action_dt:.3f}s (inference frequency: {model_action_frequency:.1f} Hz)")
    if max_joint_velocity is not None:
        print(f"🚦 Max joint velocity limit: {max_joint_velocity:.2f} rad/s")
    if constant_velocity and max_joint_velocity is not None:
        print(f"⚙️  Constant velocity mode: Enabled (actions will execute at constant velocity within speed limit)")
    if action_stride > 1:
        print(f"⚡ Action stride: {action_stride} (executing every {action_stride}-th action, ~{action_stride}x speedup)")
    print(f"📝 Task description: '{task_description}'")
    print("="*80 + "\n")
    
    # 重置policy状态
    policy.reset()
    
    step_counter = 0

    # Initialize ROS publishers for action visualization
    joint_pub = rospy.Publisher('/policy/action/eef_pose_marker_all', Float64MultiArray, queue_size=10)
    
    rospy.loginfo(f"Initialized ROS publishers for action visualization with chunk size: {action_chunk_size}")
    
    # 获取初始观测
    obs_data, camera_obs, camera_obs_ts, robot_obs, robot_obs_ts = env.get_obs()

    # TODO: 检查时间同步情况
    # TODO: 显示图像
    rospy.loginfo(f"Initialized action visualization with chunk size: {action_chunk_size}")
    
    # ---------- 2. 模型推理（实时模式） ----------------------
    # Real-time environment evaluation loop
    robot_sdk.control.set_external_control_arm_mode()
    time.sleep(1)
    resampled_action_queue: deque[np.ndarray] = deque()
    last_executed_action: Optional[np.ndarray] = None
    
    # 加载并回放初始轨迹（只在第一次推理时加载bag文件）
    if is_first_inference:
        init_traj_bag_path = '/home/lab/kuavo-manip/robot_depalletize_init_traj.bag'
        # if os.path.exists(init_traj_bag_path):
        #     rospy.loginfo("Loading and replaying initial trajectory from bag file (first inference only)...")
        #     # FIXME:第一帧的位置4pro和5wheel不一样，需要处理
        if ROBOT_VERSION == "4_pro":
            load_and_replay_init_trajectory(
                bag_path=init_traj_bag_path,
                env=env,
                control_arm=control_arm,
                control_claw=control_claw
            )
            rospy.logwarn(f"Initial trajectory bag file not found: {init_traj_bag_path}")
            rospy.loginfo("4_pro robot Initial trajectory replay completed. Starting model inference...")
            time.sleep(1.0)
        elif ROBOT_VERSION == "5_wheel":
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            final_reset_arm(
                json_path=os.path.join(cur_dir, 'utils/start_arm_traj.json'), 
                env=env,
                control_arm=control_arm,
                control_claw=control_claw
            )
        
        input(f"轨迹回放 结束, 按回车继续 ==== 轨迹回放成功 ==== \n")
        time.sleep(1.0)
        
        # 重要：在bag回放完成后，重新获取最新的观测数据
        # 这样才能获取到bag回放后的真实手臂位置
        rospy.loginfo("🔄 Updating observation data after bag replay...")
        obs_data, camera_obs, camera_obs_ts, robot_obs, robot_obs_ts = env.get_obs()
        rospy.loginfo("✅ Observation data updated with post-bag-replay robot state")
    else:
        rospy.loginfo("Skipping bag file replay (not first inference). Using JSON reset instead.")

    # 根据机器人版本切换手臂控制模式
    if ROBOT_VERSION == "4_pro":
        direct_to_wbc(1)
        function_key = "direct_to_wbc"
    elif ROBOT_VERSION == "5_wheel":
        set_arm_quick_mode(True)
        function_key = "set_arm_quick_mode"  
    time.sleep(1)
    input(f"当前机器人模式为: {ROBOT_VERSION} | 控制模式 {function_key} 结束, 按回车继续 ==== 切换手臂到wbc轨迹控制模式成功 ==== \n")
    
    print("\n" + "="*80)
    print("🚀 Starting inference loop...")
    print("💡 Press 'q' + Enter to stop current inference and prepare for next run")
    print("💡 Press Ctrl+C to exit the program completely")
    print("="*80 + "\n")
    
    # 同步模式：执行完整个chunk后再推理下一个
    if sync_mode:
        while True:
            try:
                # 准备观测
                state = torch.from_numpy(obs_data["state"]).float()
                observation = {}
                
                # 根据CAMERA_COMPONENTS动态处理相机图像
                camera_names = get_camera_names(CAMERA_COMPONENTS)
                for camera_name in camera_names:
                    if camera_name in obs_data:
                        camera_img_np = obs_data[camera_name]
                        if camera_img_np.ndim != 4:
                            rospy.logwarn(f"⚠️  Unexpected camera image shape: {camera_img_np.shape}, expected (T, H, W, C)")
                            continue
                        if rotate_head_camera and camera_name == "image":
                            camera_img_np = np.rot90(camera_img_np, k=2, axes=(1, 2)).copy()
                        camera_images = torch.from_numpy(np.moveaxis(camera_img_np, 3, 1).copy()).float() / 255
                        obs_key = get_camera_observation_key(camera_name, use_image_features=False)
                        observation[obs_key] = camera_images.to('cuda:0')
                    elif step_counter == 0:
                        rospy.logwarn(f"⚠️  Camera '{camera_name}' from CAMERA_COMPONENTS not found in obs_data.")
                
                if state_zero:
                    observation['observation.state'] = torch.zeros_like(state).to('cuda:0')
                else:
                    observation['observation.state'] = state.to('cuda:0')
                observation['task'] = task_description
                
                # 推理
                processed_observation = preprocessor(observation)
                with torch.inference_mode():
                    pred_actions = policy.predict_action_chunk(processed_observation)
                
                # 使用 postprocessor 进行反归一化
                # pred_actions shape: (batch_size, chunk_size, action_dim)
                # postprocessor 期望输入是 (B, action_dim)，所以需要处理整个 chunk
                _, chunk_size, _ = pred_actions.shape
                processed_actions = []
                for i in range(chunk_size):
                    # 提取单个 action: (B, action_dim)
                    single_action = pred_actions[:, i, :]
                    # 使用 postprocessor 进行反归一化
                    processed_action = postprocessor(single_action)
                    processed_actions.append(processed_action)
                
                # 堆叠回 (B, chunk_size, action_dim)，然后转换为 numpy
                pred_actions_unnorm = torch.stack(processed_actions, dim=1)  # (B, chunk_size, action_dim)
                action_chunk = pred_actions_unnorm[0].cpu().numpy()  # (chunk_size, action_dim)
                
                # 根据chunk_start和chunk_end选择要执行的action范围
                chunk_size = action_chunk.shape[0]
                start_idx = chunk_start if chunk_start is not None else 0
                end_idx = chunk_end if chunk_end is not None else chunk_size - 1
                
                # 验证索引范围
                if start_idx < 0:
                    rospy.logwarn(f"⚠️ Warning: chunk_start {start_idx} is negative, using 0 instead")
                    start_idx = 0
                if end_idx >= chunk_size:
                    rospy.logwarn(f"⚠️ Warning: chunk_end {end_idx} is >= chunk_size {chunk_size}, using {chunk_size - 1} instead")
                    end_idx = chunk_size - 1
                if start_idx > end_idx:
                    rospy.logwarn(f"⚠️ Warning: chunk_start {start_idx} > chunk_end {end_idx}, using last action only")
                    action_chunk = action_chunk[-1:].copy()
                else:
                    # 使用切片选择范围（Python切片是左闭右开，所以end_idx+1）
                    action_chunk = action_chunk[start_idx:end_idx+1].copy()
                    rospy.loginfo(f"⏭️  Selected actions from index {start_idx} to {end_idx} (inclusive): {action_chunk.shape[0]} actions")

                # 确定arm和claw维度（需要在FIRST_MODEL_INFERENCE检查之前确定，以便提取手臂状态）
                action_dim = action_chunk.shape[1]
                if action_dim == 16:
                    arm_dims = slice(0, 14)
                    claw_dims = slice(14, 16)
                elif action_dim == 18:
                    arm_dims = slice(0, 14)
                    claw_dims = slice(14, 16)
                else:
                    arm_dims = slice(0, 14)
                    claw_dims = slice(14, min(16, action_dim))
                
                # 如果是第一次模型推理，需要在当前状态和第一个action之间进行插值
                # 将插值动作序列和chunk合并，统一进行resample和速度限制处理
                transition_chunk = None
                if FIRST_MODEL_INFERENCE:
                    rospy.loginfo("🔄 First model inference: generating smooth transition from current robot state to first action")
                    
                    # 获取当前机器人的手臂状态
                    current_arm_state = obs_data["state"][0][arm_dims]  # 当前手臂关节位置（14维）
                    
                    # 获取当前夹爪状态
                    current_claw_state = np.array([0.0, 0.0])  # 默认值
                    try:
                        if 'claw_state' in robot_obs and len(robot_obs['claw_state']) > 0:
                            claw_data = robot_obs['claw_state']
                            if claw_data.ndim == 2:
                                current_claw_state = np.array(claw_data[-1], dtype=np.float32)
                            elif claw_data.ndim == 1:
                                current_claw_state = np.array(claw_data, dtype=np.float32)
                            if current_claw_state.shape[0] != 2:
                                current_claw_state = np.array([0.0, 0.0])
                    except Exception as e:
                        rospy.logwarn(f"Could not get current claw state: {e}, using default [0.0, 0.0]")
                        current_claw_state = np.array([0.0, 0.0])
                    
                    # 获取第一个chunk的第一个action（已经根据chunk_start/chunk_end选择之后）
                    if action_chunk.shape[0] > 0:
                        first_action = action_chunk[0].copy()
                        target_arm_state = first_action[arm_dims]  # 目标手臂关节位置
                        target_claw_state = first_action[claw_dims]  # 目标夹爪位置
                        
                        # 检查是否需要cmd_pose
                        has_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
                        if has_cmd_pose and action_dim >= 18:
                            target_cmd_pose = first_action[16:18]
                            current_cmd_pose = np.array([0.0, 0.0])  # 默认cmd_pose
                        else:
                            target_cmd_pose = None
                            current_cmd_pose = None
                        
                        # 计算插值参数
                        transition_duration = 0.2  # 过渡时间（秒），第一次推理时快速过渡到第一个action
                        num_interp_steps = int(round(transition_duration / env.control_dt))
                        num_interp_steps = max(1, num_interp_steps)  # 至少1步
                        
                        rospy.loginfo(f"   Current arm state: {current_arm_state}... (showing first 3 joints)")
                        rospy.loginfo(f"   Target arm state: {target_arm_state}... (showing first 3 joints)")
                        rospy.loginfo(f"   Generating {num_interp_steps} interpolation steps over {transition_duration:.2f}s")
                        
                        # 生成插值动作序列（作为过渡chunk，不立即执行）
                        interp_actions = []
                        for i in range(num_interp_steps):
                            alpha = (i + 1) / num_interp_steps  # 从1/num_steps到1.0
                            
                            # 线性插值手臂关节
                            interp_arm = current_arm_state + (target_arm_state - current_arm_state) * alpha
                            
                            # 线性插值夹爪
                            interp_claw = current_claw_state + (target_claw_state - current_claw_state) * alpha
                            
                            # 构建完整的action
                            if has_cmd_pose and target_cmd_pose is not None:
                                # 18维格式：插值cmd_pose
                                interp_cmd_pose = current_cmd_pose + (target_cmd_pose - current_cmd_pose) * alpha
                                interp_action = np.concatenate([interp_arm, interp_claw, interp_cmd_pose])
                            else:
                                # 16维格式
                                interp_action = np.concatenate([interp_arm, interp_claw])
                            
                            interp_actions.append(interp_action)
                        
                        # 将插值动作序列转换为numpy数组（作为过渡chunk）
                        transition_chunk = np.array(interp_actions)  # shape: (num_interp_steps, action_dim)
                        rospy.loginfo(f"   Generated transition chunk of size {transition_chunk.shape[0]}")
                        
                        # 将过渡chunk和原始chunk合并
                        # 注意：transition_chunk的最后一个action应该等于first_action（或非常接近）
                        # 但为了确保连续性，我们将transition_chunk和action_chunk合并
                        action_chunk = np.vstack([transition_chunk, action_chunk])
                        rospy.loginfo(f"   Combined transition + chunk: {transition_chunk.shape[0]} + {action_chunk.shape[0] - transition_chunk.shape[0]} = {action_chunk.shape[0]} steps")
                    else:
                        rospy.logwarn("⚠️  Warning: action_chunk is empty after chunk selection, cannot generate transition")
                    
                    FIRST_MODEL_INFERENCE = False
                
                # 如果需要连接上一个chunk，添加桥接
                if last_executed_action is not None:
                    # 在chunk前添加上一个action，确保chunk间平滑连接
                    action_chunk_with_bridge = np.vstack([last_executed_action, action_chunk])
                else:
                    action_chunk_with_bridge = action_chunk
                
                # 应用速度限制（如果提供）
                if max_joint_velocity is not None:
                    # 保存原始chunk的夹爪值（在速度限制前）
                    original_chunk_for_claw = action_chunk.copy()
                    
                    # 使用控制频率的dt
                    control_dt = env.control_dt
                    
                    # 如果有transition_chunk，需要分开处理：transition部分不应用速度限制（快速过渡），chunk部分应用速度限制
                    if transition_chunk is not None:
                        transition_size = transition_chunk.shape[0]
                        transition_part = action_chunk[:transition_size]  # transition部分（不应用速度限制）
                        chunk_part = action_chunk[transition_size:]  # chunk部分（应用速度限制）
                        
                        # 对chunk部分进行resample到control_dt频率
                        if chunk_part.shape[0] > 0:
                            resampled_chunk_part = resample_action_chunk(
                                chunk_part,
                                source_dt=model_action_dt if model_action_dt is not None else DEFAULT_MODEL_ACTION_DT,
                                target_dt=control_dt
                            )
                            
                            # 对chunk部分应用速度限制（不包括transition部分）
                            # 需要连接transition的最后一个action和chunk部分
                            if transition_size > 0:
                                chunk_with_transition_end = np.vstack([transition_part[-1:], resampled_chunk_part])
                                resampled_chunk_part = resample_actions_with_speed_limit(
                                    chunk_with_transition_end,
                                    dt=control_dt,
                                    v_max=max_joint_velocity,
                                    arm_dims=arm_dims,
                                    constant_velocity=constant_velocity
                                )[1:]  # 移除transition的最后一个action
                            
                            # 合并transition（不应用速度限制）和resampled chunk（已应用速度限制）
                            action_chunk = np.vstack([transition_part, resampled_chunk_part])
                        else:
                            # 如果chunk部分为空，只保留transition部分
                            action_chunk = transition_part
                    else:
                        # 没有transition_chunk，正常处理整个chunk
                        # 如果需要连接上一个chunk，添加桥接
                        if last_executed_action is not None:
                            action_chunk_with_bridge = np.vstack([last_executed_action, action_chunk])
                        else:
                            action_chunk_with_bridge = action_chunk
                        
                        # 只对手臂关节应用速度限制
                        action_chunk_with_bridge = resample_actions_with_speed_limit(
                            action_chunk_with_bridge,
                            dt=control_dt,
                            v_max=max_joint_velocity,
                            arm_dims=arm_dims,
                            constant_velocity=constant_velocity
                        )
                        # 移除桥接的action（如果添加了）
                        if last_executed_action is not None:
                            action_chunk = action_chunk_with_bridge[1:]
                        else:
                            action_chunk = action_chunk_with_bridge
                    
                    # 对夹爪应用zero-order hold（从原始chunk中提取）
                    if action_chunk.shape[0] > 0 and original_chunk_for_claw.shape[0] > 0:
                        # 将夹爪值插值到resampled chunk的时间点
                        if original_chunk_for_claw.shape[0] > 1:
                            # 对于合并后的chunk（包含transition），需要特殊处理
                            # transition部分使用control_dt，chunk部分使用model_action_dt
                            if transition_chunk is not None:
                                # transition部分：使用control_dt
                                transition_duration = transition_chunk.shape[0] * control_dt
                                # chunk部分：使用model_action_dt
                                source_dt_used = model_action_dt if model_action_dt is not None else DEFAULT_MODEL_ACTION_DT
                                chunk_duration = (original_chunk_for_claw.shape[0] - transition_chunk.shape[0]) * source_dt_used
                                
                                # 构建源时间轴（transition部分 + chunk部分）
                                transition_times = np.linspace(0.0, transition_duration, num=transition_chunk.shape[0], endpoint=False)
                                chunk_start_time = transition_duration
                                chunk_times = np.linspace(chunk_start_time, chunk_start_time + chunk_duration, 
                                                        num=original_chunk_for_claw.shape[0] - transition_chunk.shape[0])
                                source_times = np.concatenate([transition_times, chunk_times])
                            else:
                                source_dt_used = model_action_dt if model_action_dt is not None else DEFAULT_MODEL_ACTION_DT
                                source_times = np.linspace(0.0, source_dt_used * (original_chunk_for_claw.shape[0] - 1), num=original_chunk_for_claw.shape[0])
                            
                            target_times = np.linspace(0.0, control_dt * (action_chunk.shape[0] - 1), num=action_chunk.shape[0])
                            hold_indices = np.searchsorted(source_times, target_times, side="right") - 1
                            hold_indices = np.clip(hold_indices, 0, original_chunk_for_claw.shape[0] - 1)
                            action_chunk[:, claw_dims] = original_chunk_for_claw[hold_indices][:, claw_dims]
                        else:
                            action_chunk[:, claw_dims] = original_chunk_for_claw[0, claw_dims]
                else:
                    # 如果没有速度限制，使用resample_chunk_with_claw_hold来保持夹爪的zero-order hold
                    # 但如果有transition_chunk，需要特殊处理
                    if transition_chunk is not None:
                        # 对于包含transition的情况，需要分别处理transition和chunk部分
                        # transition部分已经是在control_dt频率下，不需要resample
                        # chunk部分需要resample
                        transition_size = transition_chunk.shape[0]
                        chunk_part = action_chunk[transition_size:]
                        if chunk_part.shape[0] > 0:
                            resampled_chunk_part = resample_chunk_with_claw_hold(
                                chunk_part,
                                previous_action=action_chunk[transition_size - 1] if transition_size > 0 else last_executed_action,
                                control_frequency=env.control_frequency,
                                source_dt=model_action_dt if model_action_dt is not None else DEFAULT_MODEL_ACTION_DT,
                                arm_dims=arm_dims,
                                claw_dims=claw_dims
                            )
                            action_chunk = np.vstack([action_chunk[:transition_size], resampled_chunk_part])
                    else:
                        action_chunk = resample_chunk_with_claw_hold(
                            action_chunk,
                            previous_action=last_executed_action,
                            control_frequency=env.control_frequency,
                            source_dt=model_action_dt if model_action_dt is not None else DEFAULT_MODEL_ACTION_DT,
                            arm_dims=arm_dims,
                            claw_dims=claw_dims
                        )
                
                # 应用action_stride：先选择每隔action_stride个action
                if action_stride > 1:
                    strided_chunk = action_chunk[::action_stride]
                    rospy.loginfo(f"Applied action stride {action_stride}: {action_chunk.shape[0]} -> {strided_chunk.shape[0]} actions")
                    
                    # 重要：在应用stride后，需要重新应用速度限制
                    # 因为跳过的action之间的时间间隔变大了（control_dt * stride）
                    # 如果不重新限制，可能会违反速度限制
                    if max_joint_velocity is not None:
                        # 确保control_dt已定义
                        control_dt = env.control_dt
                        
                        # 保存原始chunk的夹爪值
                        original_strided_chunk_for_claw = strided_chunk.copy()
                        
                        # 重新应用速度限制，使用新的时间间隔（control_dt * stride）
                        if last_executed_action is not None:
                            strided_chunk_with_bridge = np.vstack([last_executed_action, strided_chunk])
                        else:
                            strided_chunk_with_bridge = strided_chunk
                        
                        strided_chunk_with_bridge = resample_actions_with_speed_limit(
                            strided_chunk_with_bridge,
                            dt=control_dt * action_stride,  # 使用新的时间间隔
                            v_max=max_joint_velocity,
                            arm_dims=arm_dims,
                            constant_velocity=constant_velocity
                        )
                        
                        # 移除桥接的action（如果添加了）
                        if last_executed_action is not None:
                            strided_chunk = strided_chunk_with_bridge[1:]
                        else:
                            strided_chunk = strided_chunk_with_bridge
                        
                        # 对夹爪应用zero-order hold（从原始strided chunk中提取）
                        if strided_chunk.shape[0] > 0 and original_strided_chunk_for_claw.shape[0] > 0:
                            if original_strided_chunk_for_claw.shape[0] > 1:
                                source_times = np.linspace(0.0, (control_dt * action_stride) * (original_strided_chunk_for_claw.shape[0] - 1), 
                                                          num=original_strided_chunk_for_claw.shape[0])
                                target_times = np.linspace(0.0, control_dt * (strided_chunk.shape[0] - 1), num=strided_chunk.shape[0])
                                hold_indices = np.searchsorted(source_times, target_times, side="right") - 1
                                hold_indices = np.clip(hold_indices, 0, original_strided_chunk_for_claw.shape[0] - 1)
                                strided_chunk[:, claw_dims] = original_strided_chunk_for_claw[hold_indices][:, claw_dims]
                            else:
                                strided_chunk[:, claw_dims] = original_strided_chunk_for_claw[0, claw_dims]
                        
                        rospy.loginfo(f"Re-applied velocity limit after stride: {strided_chunk.shape[0]} actions (dt={control_dt * action_stride:.4f}s)")
                    
                    action_chunk = strided_chunk
                
                # 执行整个chunk
                rospy.loginfo(f"Executing chunk of size {action_chunk.shape[0]} in sync mode")
                control_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
                
                for action_step in action_chunk:
                    env.exec_actions(actions=action_step,
                                     control_arm=control_arm,
                                     control_claw=control_claw,
                                     control_cmd_pose=control_cmd_pose)
                    step_counter += 1
                    last_executed_action = action_step.copy()
                    
                    # 键盘监听
                    key = 0
                    if enable_gui:
                        key = cv2.waitKey(1) & 0xFF
                    else:
                        try:
                            import select
                            if select.select([sys.stdin], [], [], 0)[0]:
                                import termios
                                import tty
                                old_settings = termios.tcgetattr(sys.stdin)
                                try:
                                    tty.setraw(sys.stdin.fileno())
                                    ch = sys.stdin.read(1)
                                    if ch:
                                        key = ord(ch)
                                finally:
                                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        except (ImportError, OSError, AttributeError):
                            pass
                    
                    if key == ord('q') or key == 27:
                        print("\n[Keyboard] Stopping current inference by user request")
                        FIRST_MODEL_INFERENCE = True
                        return True
                
                # 执行完chunk后，获取新的观测
                obs_data, camera_obs, camera_obs_ts, robot_obs, robot_obs_ts = env.get_obs()
                
            except KeyboardInterrupt:
                print("\n[Interrupted] Exiting by user Ctrl+C.")
                FIRST_MODEL_INFERENCE = True
                return False
    
    # 异步模式：原有的实现
    while True:
        try:
            state = torch.from_numpy(obs_data["state"]).float()
            # print(f" ==== state ==== {state.shape} ==== ")
            
            # 根据CAMERA_COMPONENTS动态处理相机图像
            # 填充网络的obs
            observation = {}
            
            # 根据CAMERA_COMPONENTS明确指定需要处理的相机
            camera_names = get_camera_names(CAMERA_COMPONENTS)
            for camera_name in camera_names:
                # 检查相机数据是否在obs_data中
                if camera_name in obs_data:
                    # 获取相机图像，obs_data中的图像格式是 (T, H, W, C)，其中T是时间步数
                    camera_img_np = obs_data[camera_name]
                    
                    # 检查图像维度，应该是 (T, H, W, C) 格式
                    if camera_img_np.ndim != 4:
                        rospy.logwarn(f"⚠️  Unexpected camera image shape: {camera_img_np.shape}, expected (T, H, W, C)")
                        continue
                    
                    # 如果启用头部相机旋转且当前是头部相机（image），则对每一帧旋转180度
                    if rotate_head_camera and camera_name == "image":
                        # 旋转180度：使用np.rot90，k=2表示旋转180度，axes=(1,2)表示在H和W维度上旋转
                        # camera_img_np shape: (T, H, W, C)
                        # 对每一帧进行旋转，axes=(1,2)表示在H和W维度上旋转（保持T和C维度不变）
                        # 注意：np.rot90可能产生负步长的视图，需要copy()来创建连续数组，以便PyTorch可以处理
                        camera_img_np = np.rot90(camera_img_np, k=2, axes=(1, 2)).copy()
                    
                    # 转换为 (T, C, H, W) 格式并归一化
                    # 使用np.moveaxis将 (T, H, W, C) 转换为 (T, C, H, W)
                    # 注意：np.moveaxis也可能产生负步长，使用copy()确保数组连续
                    camera_images = torch.from_numpy(np.moveaxis(camera_img_np, 3, 1).copy()).float() / 255
                    # 使用新的key格式: observation.images.cam_*
                    obs_key = get_camera_observation_key(camera_name, use_image_features=False)
                    observation[obs_key] = camera_images.to('cuda:0')
                else:
                    # 只在第一次出现时打印警告
                    if step_counter == 0:
                        rospy.logwarn(f"⚠️  Camera '{camera_name}' from CAMERA_COMPONENTS not found in obs_data. Available cameras: {[k for k in obs_data.keys() if 'image' in k.lower()]}")

            # observation['observation.environment_state'] = environment_state
            # 如果启用state_zero模式，将状态输入置零（用于验证模型对状态的依赖性）
            if state_zero:
                # 保持相同的形状和设备，但将所有状态值设为0
                observation['observation.state'] = torch.zeros_like(state).to('cuda:0')
            else:
                observation['observation.state'] = state.to('cuda:0')
            
            # 添加 task 字段（language instruction）
            # processor 会从 complementary_data 中的 "task" 字段读取并转换为 language
            observation['task'] = task_description

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
                
                # 使用 postprocessor 进行反归一化
                # pred_actions shape: (batch_size, chunk_size, action_dim)
                # postprocessor 期望输入是 (B, action_dim)，所以需要处理整个 chunk
                _, chunk_size, _ = pred_actions.shape
                processed_actions = []
                for i in range(chunk_size):
                    # 提取单个 action: (B, action_dim)
                    single_action = pred_actions[:, i, :]
                    # 使用 postprocessor 进行反归一化
                    processed_action = postprocessor(single_action)
                    processed_actions.append(processed_action)
                
                # 堆叠回 (B, chunk_size, action_dim)，然后转换为 numpy
                pred_actions_unnorm = torch.stack(processed_actions, dim=1)  # (B, chunk_size, action_dim)
                action_chunk = pred_actions_unnorm[0].cpu().numpy()  # (chunk_size, action_dim)

                # 根据chunk_start和chunk_end选择要执行的action范围
                chunk_size = action_chunk.shape[0]
                start_idx = chunk_start if chunk_start is not None else 0
                end_idx = chunk_end if chunk_end is not None else chunk_size - 1
                
                # 验证索引范围
                if start_idx < 0:
                    rospy.logwarn(f"⚠️ Warning: chunk_start {start_idx} is negative, using 0 instead")
                    start_idx = 0
                if end_idx >= chunk_size:
                    rospy.logwarn(f"⚠️ Warning: chunk_end {end_idx} is >= chunk_size {chunk_size}, using {chunk_size - 1} instead")
                    end_idx = chunk_size - 1
                if start_idx > end_idx:
                    rospy.logwarn(f"⚠️ Warning: chunk_start {start_idx} > chunk_end {end_idx}, using last action only")
                    action_chunk = action_chunk[-1:].copy()
                else:
                    # 使用切片选择范围（Python切片是左闭右开，所以end_idx+1）
                    action_chunk = action_chunk[start_idx:end_idx+1].copy()
                    rospy.loginfo(f"⏭️  Selected actions from index {start_idx} to {end_idx} (inclusive): {action_chunk.shape[0]} actions")

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
                
                # 如果是第一次模型推理，需要在当前状态和第一个action之间进行插值
                # 将插值动作序列和chunk合并，统一进行resample和速度限制处理
                transition_chunk = None
                if FIRST_MODEL_INFERENCE and action_chunk.shape[0] > 0:
                    rospy.loginfo("🔄 First model inference (async mode): generating smooth transition from current robot state to first action")
                    
                    # 在异步模式下，需要先获取最新的obs_data和robot_obs（确保获取到bag回放后的真实状态）
                    obs_data, camera_obs, camera_obs_ts, robot_obs_for_transition, robot_obs_ts = env.get_obs()
                    
                    # 获取当前机器人的手臂状态
                    current_arm_state = obs_data["state"][0][arm_dims]  # 当前手臂关节位置（14维）
                    
                    # 获取当前夹爪状态
                    current_claw_state = np.array([0.0, 0.0])  # 默认值
                    try:
                        if 'claw_state' in robot_obs_for_transition and len(robot_obs_for_transition['claw_state']) > 0:
                            claw_data = robot_obs_for_transition['claw_state']
                            if claw_data.ndim == 2:
                                current_claw_state = np.array(claw_data[-1], dtype=np.float32)
                            elif claw_data.ndim == 1:
                                current_claw_state = np.array(claw_data, dtype=np.float32)
                            if current_claw_state.shape[0] != 2:
                                current_claw_state = np.array([0.0, 0.0])
                    except Exception as e:
                        rospy.logwarn(f"Could not get current claw state: {e}, using default [0.0, 0.0]")
                        current_claw_state = np.array([0.0, 0.0])
                    
                    # 获取第一个chunk的第一个action（已经根据chunk_start/chunk_end选择之后）
                    first_action = action_chunk[0].copy()
                    target_arm_state = first_action[arm_dims]  # 目标手臂关节位置
                    target_claw_state = first_action[claw_dims]  # 目标夹爪位置
                    
                    # 检查是否需要cmd_pose
                    has_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
                    if has_cmd_pose and action_dim >= 18:
                        target_cmd_pose = first_action[16:18]
                        current_cmd_pose = np.array([0.0, 0.0])  # 默认cmd_pose
                    else:
                        target_cmd_pose = None
                        current_cmd_pose = None
                    
                    # 计算插值参数
                    transition_duration = 0.2  # 过渡时间（秒），第一次推理时快速过渡到第一个action
                    num_interp_steps = int(round(transition_duration / env.control_dt))
                    num_interp_steps = max(1, num_interp_steps)  # 至少1步
                    
                    rospy.loginfo(f"   Current arm state: {current_arm_state[:3]}... (showing first 3 joints)")
                    rospy.loginfo(f"   Target arm state: {target_arm_state[:3]}... (showing first 3 joints)")
                    rospy.loginfo(f"   Generating {num_interp_steps} interpolation steps over {transition_duration:.2f}s")
                    
                    # 生成插值动作序列（作为过渡chunk，不立即执行）
                    interp_actions = []
                    for i in range(num_interp_steps):
                        alpha = (i + 1) / num_interp_steps  # 从1/num_steps到1.0
                        
                        # 线性插值手臂关节
                        interp_arm = current_arm_state + (target_arm_state - current_arm_state) * alpha
                        
                        # 线性插值夹爪
                        interp_claw = current_claw_state + (target_claw_state - current_claw_state) * alpha
                        
                        # 构建完整的action
                        if has_cmd_pose and target_cmd_pose is not None:
                            # 18维格式：插值cmd_pose
                            interp_cmd_pose = current_cmd_pose + (target_cmd_pose - current_cmd_pose) * alpha
                            interp_action = np.concatenate([interp_arm, interp_claw, interp_cmd_pose])
                        else:
                            # 16维格式
                            interp_action = np.concatenate([interp_arm, interp_claw])
                        
                        interp_actions.append(interp_action)
                    
                    # 将插值动作序列转换为numpy数组（作为过渡chunk）
                    transition_chunk = np.array(interp_actions)  # shape: (num_interp_steps, action_dim)
                    rospy.loginfo(f"   Generated transition chunk of size {transition_chunk.shape[0]}")
                    
                    # 将过渡chunk和原始chunk合并
                    action_chunk = np.vstack([transition_chunk, action_chunk])
                    rospy.loginfo(f"   Combined transition + chunk: {transition_chunk.shape[0]} + {action_chunk.shape[0] - transition_chunk.shape[0]} = {action_chunk.shape[0]} steps")
                    
                    FIRST_MODEL_INFERENCE = False
                
                # 如果需要连接上一个chunk，添加桥接
                if last_executed_action is not None:
                    action_chunk_with_bridge = np.vstack([last_executed_action, action_chunk])
                else:
                    action_chunk_with_bridge = action_chunk
                
                # 应用速度限制（如果提供）
                if max_joint_velocity is not None:
                    # 保存原始chunk的夹爪值（在速度限制前）
                    original_chunk_for_claw = action_chunk.copy()
                    
                    # 如果有transition_chunk，需要分开处理：transition部分不应用速度限制（快速过渡），chunk部分应用速度限制
                    if transition_chunk is not None:
                        transition_size = transition_chunk.shape[0]
                        transition_part = action_chunk[:transition_size]  # transition部分（不应用速度限制）
                        chunk_part = action_chunk[transition_size:]  # chunk部分（应用速度限制）
                        
                        # 对chunk部分进行resample到control_dt频率
                        if chunk_part.shape[0] > 0:
                            resampled_chunk_part = resample_action_chunk(
                                chunk_part,
                                source_dt=model_action_dt,
                                target_dt=env.control_dt
                            )
                            
                            # 对chunk部分应用速度限制（不包括transition部分）
                            # 需要连接transition的最后一个action和chunk部分
                            if transition_size > 0:
                                chunk_with_transition_end = np.vstack([transition_part[-1:], resampled_chunk_part])
                                resampled_chunk_part = resample_actions_with_speed_limit(
                                    chunk_with_transition_end,
                                    dt=env.control_dt,
                                    v_max=max_joint_velocity,
                                    arm_dims=arm_dims,
                                    constant_velocity=constant_velocity
                                )[1:]  # 移除transition的最后一个action
                            
                            # 合并transition（不应用速度限制）和resampled chunk（已应用速度限制）
                            resampled_chunk = np.vstack([transition_part, resampled_chunk_part])
                        else:
                            # 如果chunk部分为空，只保留transition部分
                            resampled_chunk = transition_part
                    else:
                        # 没有transition_chunk，正常处理整个chunk
                        # 先resample到控制频率
                        if last_executed_action is not None:
                            resampled_bridge = resample_action_chunk(
                                action_chunk_with_bridge,
                                source_dt=model_action_dt,
                                target_dt=env.control_dt
                            )
                            resampled_chunk = resampled_bridge[1:]  # 移除桥接的action
                        else:
                            resampled_chunk = resample_action_chunk(
                                action_chunk,
                                source_dt=model_action_dt,
                                target_dt=env.control_dt
                            )
                        
                        # 应用速度限制到arm关节
                        if last_executed_action is not None:
                            # 将上一个action和resampled chunk连接，应用速度限制
                            chunk_with_prev = np.vstack([last_executed_action, resampled_chunk])
                            resampled_chunk = resample_actions_with_speed_limit(
                                chunk_with_prev,
                                dt=env.control_dt,
                                v_max=max_joint_velocity,
                                arm_dims=arm_dims,
                                constant_velocity=constant_velocity
                            )[1:]  # 移除桥接的action
                        else:
                            resampled_chunk = resample_actions_with_speed_limit(
                                resampled_chunk,
                                dt=env.control_dt,
                                v_max=max_joint_velocity,
                                arm_dims=arm_dims,
                                constant_velocity=constant_velocity
                            )
                    
                    # 对夹爪应用zero-order hold（从原始chunk中提取）
                    if resampled_chunk.shape[0] > 0 and original_chunk_for_claw.shape[0] > 0:
                        # 将夹爪值插值到resampled chunk的时间点
                        if original_chunk_for_claw.shape[0] > 1:
                            # 对于合并后的chunk（包含transition），需要特殊处理
                            if transition_chunk is not None:
                                # transition部分：使用control_dt
                                transition_duration = transition_chunk.shape[0] * env.control_dt
                                # chunk部分：使用model_action_dt
                                chunk_duration = (original_chunk_for_claw.shape[0] - transition_chunk.shape[0]) * model_action_dt
                                
                                # 构建源时间轴（transition部分 + chunk部分）
                                transition_times = np.linspace(0.0, transition_duration, num=transition_chunk.shape[0], endpoint=False)
                                chunk_start_time = transition_duration
                                chunk_times = np.linspace(chunk_start_time, chunk_start_time + chunk_duration, 
                                                        num=original_chunk_for_claw.shape[0] - transition_chunk.shape[0])
                                source_times = np.concatenate([transition_times, chunk_times])
                            else:
                                source_times = np.linspace(0.0, model_action_dt * (original_chunk_for_claw.shape[0] - 1), num=original_chunk_for_claw.shape[0])
                            
                            target_times = np.linspace(0.0, env.control_dt * (resampled_chunk.shape[0] - 1), num=resampled_chunk.shape[0])
                            hold_indices = np.searchsorted(source_times, target_times, side="right") - 1
                            hold_indices = np.clip(hold_indices, 0, original_chunk_for_claw.shape[0] - 1)
                            resampled_chunk[:, claw_dims] = original_chunk_for_claw[hold_indices][:, claw_dims]
                        else:
                            resampled_chunk[:, claw_dims] = original_chunk_for_claw[0, claw_dims]
                else:
                    # 没有速度限制，使用原有的resample方法
                    # 但如果有transition_chunk，需要特殊处理
                    if transition_chunk is not None:
                        # 对于包含transition的情况，需要分别处理transition和chunk部分
                        # transition部分已经是在control_dt频率下，不需要resample
                        # chunk部分需要resample
                        transition_size = transition_chunk.shape[0]
                        chunk_part = action_chunk[transition_size:]
                        if chunk_part.shape[0] > 0:
                            resampled_chunk_part = resample_chunk_with_claw_hold(
                                chunk_part,
                                previous_action=action_chunk[transition_size - 1] if transition_size > 0 else last_executed_action,
                                control_frequency=env.control_frequency,
                                source_dt=model_action_dt,
                                arm_dims=arm_dims,
                                claw_dims=claw_dims
                            )
                            resampled_chunk = np.vstack([action_chunk[:transition_size], resampled_chunk_part])
                        else:
                            resampled_chunk = action_chunk[:transition_size]
                    else:
                        resampled_chunk = resample_chunk_with_claw_hold(
                            action_chunk,
                            previous_action=last_executed_action,
                            control_frequency=env.control_frequency,
                            source_dt=model_action_dt,
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
                
                # 应用action_stride：只保留每隔action_stride个action
                if action_stride > 1:
                    # 使用切片选择每隔stride个action：[::stride]
                    strided_chunk = resampled_chunk[::action_stride]
                    rospy.loginfo(f"Applied action stride {action_stride}: {resampled_chunk.shape[0]} -> {strided_chunk.shape[0]} actions")
                    
                    # 重要：在应用stride后，需要重新应用速度限制
                    # 因为跳过的action之间的时间间隔变大了（control_dt * stride）
                    # 如果不重新限制，可能会违反速度限制
                    if max_joint_velocity is not None:
                        # 保存原始chunk的夹爪值
                        original_strided_chunk_for_claw = strided_chunk.copy()
                        
                        # 重新应用速度限制，使用新的时间间隔（control_dt * stride）
                        if last_executed_action is not None:
                            strided_chunk_with_bridge = np.vstack([last_executed_action, strided_chunk])
                        else:
                            strided_chunk_with_bridge = strided_chunk
                        
                        strided_chunk_with_bridge = resample_actions_with_speed_limit(
                            strided_chunk_with_bridge,
                            dt=env.control_dt * action_stride,  # 使用新的时间间隔
                            v_max=max_joint_velocity,
                            arm_dims=arm_dims,
                            constant_velocity=constant_velocity
                        )
                        
                        # 移除桥接的action（如果添加了）
                        if last_executed_action is not None:
                            strided_chunk = strided_chunk_with_bridge[1:]
                        else:
                            strided_chunk = strided_chunk_with_bridge
                        
                        # 对夹爪应用zero-order hold（从原始strided chunk中提取）
                        if strided_chunk.shape[0] > 0 and original_strided_chunk_for_claw.shape[0] > 0:
                            if original_strided_chunk_for_claw.shape[0] > 1:
                                source_times = np.linspace(0.0, (env.control_dt * action_stride) * (original_strided_chunk_for_claw.shape[0] - 1), 
                                                          num=original_strided_chunk_for_claw.shape[0])
                                target_times = np.linspace(0.0, env.control_dt * (strided_chunk.shape[0] - 1), num=strided_chunk.shape[0])
                                hold_indices = np.searchsorted(source_times, target_times, side="right") - 1
                                hold_indices = np.clip(hold_indices, 0, original_strided_chunk_for_claw.shape[0] - 1)
                                strided_chunk[:, claw_dims] = original_strided_chunk_for_claw[hold_indices][:, claw_dims]
                            else:
                                strided_chunk[:, claw_dims] = original_strided_chunk_for_claw[0, claw_dims]
                        
                        rospy.loginfo(f"Re-applied velocity limit after stride: {strided_chunk.shape[0]} actions (dt={env.control_dt * action_stride:.4f}s)")
                    
                    resampled_chunk = strided_chunk

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

            # 键盘监听（无论是否启用GUI都监听，参考eval_depalletize_camera_dagger.py的实现方式）
            key = 0
            if enable_gui:
                key = cv2.waitKey(1) & 0xFF
            else:
                # 非GUI模式下使用非阻塞键盘监听
                try:
                    import select
                    if select.select([sys.stdin], [], [], 0)[0]:
                        import termios
                        import tty
                        old_settings = termios.tcgetattr(sys.stdin)
                        try:
                            tty.setraw(sys.stdin.fileno())
                            ch = sys.stdin.read(1)
                            if ch:
                                key = ord(ch)
                        finally:
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except (ImportError, OSError, AttributeError):
                    # 如果select或termios不可用，跳过键盘监听
                    pass
            
            if key == ord('q') or key == 27:  # 'q' or ESC to quit current inference
                print("\n[Keyboard] Stopping current inference by user request (q or ESC pressed)")
                FIRST_MODEL_INFERENCE = True
                return True  # 返回True表示正常退出当前推理
            
        except KeyboardInterrupt:
            print("\n[Interrupted] Exiting by user Ctrl+C.")
            FIRST_MODEL_INFERENCE = True
            return False  # 返回False表示被中断
    
    return True  # 正常情况下不会到达这里

def final_reset_arm(json_path, env, control_arm=True, control_claw=True):
    """
    使用JSON文件中的手臂轨迹重置手臂位置
    
    Args:
        json_path: JSON文件路径，包含初始手臂轨迹
        env: GrabBoxMpcEnv环境实例
        control_arm: 是否控制手臂
        control_claw: 是否控制夹爪
    """
    # 先重置夹爪锁定状态，确保可以正常打开夹爪
    rospy.loginfo("Resetting claw lock state before opening claws...")
    env.reset_claw_lock()
    rospy.loginfo("✅ Claw lock reset complete")
    
    # 先打开夹爪
    rospy.loginfo("Opening claws before reset...")
    # 获取当前状态
    obs_data, camera_obs, camera_obs_ts, robot_obs, robot_obs_ts = env.get_obs()
    current_arm_state = obs_data["state"][0][:14]  # 当前手臂位置
    current_claw_state = np.array([0.0, 0.0])  # 默认值
    try:
        if 'claw_state' in robot_obs and len(robot_obs['claw_state']) > 0:
            claw_data = robot_obs['claw_state']
            if claw_data.ndim == 2:
                current_claw_state = np.array(claw_data[-1], dtype=np.float32)
            elif claw_data.ndim == 1:
                current_claw_state = np.array(claw_data, dtype=np.float32)
            if current_claw_state.shape[0] != 2:
                current_claw_state = np.array([0.0, 0.0])
    except Exception as e:
        rospy.logwarn(f"Could not get current claw state: {e}, using default [0.0, 0.0]")
        current_claw_state = np.array([0.0, 0.0])
    
    # 打开夹爪（设置为0），保持手臂位置不变
    # 注意：夹爪的0值表示打开状态
    open_claw_value = np.zeros([2])  # [0.0, 0.0] 表示打开夹爪
    has_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
    if has_cmd_pose:
        # 18维格式
        open_claw_action = np.concatenate([current_arm_state, open_claw_value, np.array([0.0, 0.0])])
    else:
        # 16维格式
        open_claw_action = np.concatenate([current_arm_state, open_claw_value])
    env.exec_actions(actions=open_claw_action, control_arm=False, control_claw=control_claw)
    time.sleep(1)
    
    # 更新current_claw_state为打开后的状态（0），用于后续的手臂重置过程
    current_claw_state = open_claw_value.copy()

    # 加载JSON文件中的手臂轨迹
    rospy.loginfo(f"Loading initial arm trajectory from JSON: {json_path}")
    with open(json_path, 'r') as f:
        init_traj = json.load(f)
        arm_actions = init_traj['arm_action']  # List of arm actions
        dt = init_traj.get('dt', 0.1)  # 获取时间间隔，默认0.1秒

    obs_data, camera_obs, camera_obs_ts, robot_obs, robot_obs_ts = env.get_obs()
    init_joints = np.array(arm_actions[-1])  # 目标关节位置（14维）
    current_joints = obs_data["state"][0][:14]  # 当前关节位置（14维）

    # 确保init_joints是14维
    if len(init_joints) != 14:
        rospy.logwarn(f"Expected 14 joint positions, got {len(init_joints)}. Using first 14 elements.")
        init_joints = np.array(init_joints[:14])

    total_time = 5.0  # 总时间（秒）
    num_points = int(total_time / dt)
    
    rospy.loginfo(f"Resetting arm from current position to initial position over {total_time}s ({num_points} steps)...")
    
    # 从current_joints到init_joints插值
    for i in range(1, num_points + 1):
        # 线性插值
        alpha = i / num_points
        interp_joints = current_joints + (init_joints - current_joints) * alpha
        
        # 构建完整的动作数组（根据ACTION_COMPONENTS格式）
        # 根据ACTION_COMPONENTS动态确定动作维度
        has_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
        
        if has_cmd_pose:
            # 18维格式: [14个手臂关节, 2个夹爪位置, 2个cmd_pose]
            # 保持cmd_pose不变（使用当前值或0）
            cmd_pose = np.array([0.0, 0.0])  # 默认cmd_pose值
            action = np.concatenate([interp_joints, current_claw_state, cmd_pose])
        else:
            # 16维格式: [14个手臂关节, 2个夹爪位置]
            action = np.concatenate([interp_joints, current_claw_state])
        
        # 使用exec_actions执行动作
        env.exec_actions(actions=action, control_arm=control_arm, control_claw=control_claw)
        time.sleep(dt)
    
    rospy.loginfo("Arm reset completed!")


def eval(ckpt_path, model_type, control_arm=True, control_claw=True, action_chunk_size=50, enable_gui=False, rotate_head_camera=False, state_zero=False, task_description=None, chunk_start=None, chunk_end=None, model_action_dt=None, sync_mode=False, max_joint_velocity=None, constant_velocity=False, action_stride=1, claw_lock_threshold=50.0, claw_lock_count_threshold=5, claw_locked_value=90.0, disable_adapter=False, task_type=None, use_smart_routing=False, swap_task_mapping=False):
    """
    在这里和实机/仿真交互，做网络推理（depalletize任务）
    支持多次推理：按'q'退出当前推理，可以快速重新开始下一次推理而无需重新加载模型
    
    支持 MergeVLA 融合模型：自动检测并加载适配层
    
    Args:
        ckpt_path: 模型checkpoint路径
        model_type: 模型类型（已废弃，保留用于兼容性，现在只使用GrootPolicy）
        control_arm: 是否控制手臂
        control_claw: 是否控制夹爪
        action_chunk_size: 动作块大小
        enable_gui: 是否启用GUI窗口显示相机图像
        rotate_head_camera: 是否旋转头部相机图像180度
        state_zero: 是否将状态输入置零（用于验证模型对状态的依赖性）
        task_description: 任务描述字符串（language instruction），如果为None则使用默认值
        chunk_start: 要执行的chunk起始索引（从0开始，包含）。如果为None，从第一个action开始
        chunk_end: 要执行的chunk结束索引（从0开始，包含）。如果为None，执行到最后一个action
        model_action_dt: 模型动作时间间隔（秒），控制推理频率。例如：0.1 = 10 Hz, 0.05 = 20 Hz, 0.033 = 30 Hz
                        如果为None，使用默认值 0.1 秒（10 Hz）。在sync_mode下不使用此参数
        sync_mode: 是否使用同步推理模式。如果True，推理一个chunk -> 执行完整个chunk -> get_obs -> 再推理下一个chunk
        max_joint_velocity: 最大关节速度限制（rad/s）。如果提供，将对arm关节应用速度限制
        constant_velocity: 是否启用匀速模式。如果True，确保动作执行时速度恒定（在max_joint_velocity限制内）
        action_stride: 动作采样间隔，用于加速执行。例如：action_stride=2表示每隔2个action执行一次，跳过中间的action。
                       设置为1表示不跳过任何action（正常速度）。设置为N表示执行速度约为原来的N倍。
                       注意：这不会改变速度限制，只是减少执行的action数量。建议值：1-5。
        disable_adapter: 是否禁用 MergeVLA 适配层（调试用）
        task_type: 任务类型 ("narrower", "wider", None)
                   ⚠️ 关键：对于 sparse_lora 适配器，如果不使用智能路由，
                   必须指定任务类型！否则会使用所有任务的平均。
        use_smart_routing: ⭐ 是否使用 MergeVLA 风格的智能任务路由
                          当任务身份未知时，根据模型内部参数子空间
                          自动推断任务相关性（参考 MergeVLA 论文 Section 3.3）
    """
    
    # 加载模型和环境（只执行一次）
    policy, preprocessor, postprocessor, env, final_task_description, device, adapter = load_model_and_env(
        ckpt_path=ckpt_path,
        model_type=model_type,
        action_chunk_size=action_chunk_size,
        enable_gui=enable_gui,
        rotate_head_camera=rotate_head_camera,
        state_zero=state_zero,
        task_description=task_description,
        claw_lock_threshold=claw_lock_threshold,
        claw_lock_count_threshold=claw_lock_count_threshold,
        claw_locked_value=claw_locked_value,
        disable_adapter=disable_adapter,
        task_type=task_type,
        use_smart_routing=use_smart_routing,
        swap_task_mapping=swap_task_mapping,  # ⚠️ 交换任务映射
    )
    
    # 主循环：支持多次推理
    inference_count = 0
    while True:
        try:
            inference_count += 1
            is_first_inference = (inference_count == 1)
            
            print(f"\n{'='*80}")
            print(f"🔄 Starting inference session #{inference_count}")
            if is_first_inference:
                print(f"📦 First inference: will load bag file for initial trajectory")
            else:
                print(f"📦 Subsequent inference: will use JSON file for arm reset")
            print(f"{'='*80}\n")
            
            # 重置推理状态
            reset_inference_state(
                policy=policy,
                env=env
            )
            
            # 运行推理循环
            normal_exit = run_inference_loop(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                env=env,
                task_description=final_task_description,
                device=device,
                control_arm=control_arm,
                control_claw=control_claw,
                action_chunk_size=action_chunk_size,
                enable_gui=enable_gui,
                rotate_head_camera=rotate_head_camera,
                state_zero=state_zero,
                is_first_inference=is_first_inference,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                model_action_dt=model_action_dt,
                sync_mode=sync_mode,
                max_joint_velocity=max_joint_velocity,
                constant_velocity=constant_velocity,
                action_stride=action_stride
            )
            
            if normal_exit:
                # 正常退出（按q），准备下一次推理
                print(f"\n{'='*80}")
                print(f"✅ Inference session #{inference_count} stopped by user (q pressed)")
                
                # ⭐ 打印智能任务路由统计（如果使用了 smart_routing）
                if use_smart_routing and adapter is not None:
                    routing_stats = adapter.get_routing_stats()
                    if routing_stats:
                        print(f"\n{'='*60}")
                        print(f"⭐ MergeVLA Smart Routing 统计")
                        print(f"{'='*60}")
                        print(f"   总调用次数: {routing_stats.get('total_calls', 0)}")
                        print(f"   倾向 Task 0 (narrower): {routing_stats.get('task_0', 0)} 次 "
                              f"({routing_stats.get('task_0_ratio', 0)*100:.1f}%)")
                        print(f"   倾向 Task 1 (wider): {routing_stats.get('task_1', 0)} 次 "
                              f"({routing_stats.get('task_1_ratio', 0)*100:.1f}%)")
                        print(f"   混合路由: {routing_stats.get('mixed', 0)} 次 "
                              f"({routing_stats.get('mixed_ratio', 0)*100:.1f}%)")
                        print(f"{'='*60}")
                        
                        # 重置路由统计（为下一次推理准备）
                        adapter.reset_routing_stats()
                
                # 立即重置policy状态，确保策略状态干净
                rospy.loginfo("Resetting policy state after inference stop...")
                policy.reset()
                rospy.loginfo("   ✅ Policy reset complete")
                
                cur_dir = os.path.dirname(os.path.abspath(__file__))
                # 每次退出时都使用JSON文件重置手臂位置
                # 第一次推理开始时使用bag文件，后续推理开始时跳过bag文件（在run_inference_loop中处理）
                rospy.loginfo("Resetting arm position using JSON file...")
                final_reset_arm(
                    json_path=os.path.join(cur_dir, 'utils/initial_arm_traj.json'), 
                    env=env,
                    control_arm=control_arm,
                    control_claw=control_claw
                )
                print(f"💡 Ready for next inference session. Press Enter to start, or Ctrl+C to exit.")
                print(f"{'='*80}\n")
                
                # 等待用户输入以开始下一次推理
                try:
                    user_input = input("Press Enter to start next inference, or 'q'+Enter to exit: ").strip().lower()
                    if user_input == 'q':
                        print("\n👋 Exiting program. Goodbye!")
                        break
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Exiting program. Goodbye!")
                    break
            else:
                # 被Ctrl+C中断，退出程序
                print("\n👋 Exiting program due to Ctrl+C. Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n👋 Exiting program due to Ctrl+C. Goodbye!")
            break
        except Exception as e:
            rospy.logerr(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            print("\n❌ Error occurred. Exiting program.")
            break
    
    # Cleanup GUI windows
    if enable_gui:
        cv2.destroyAllWindows()




if __name__ == '__main__':
    # 机器人低头
    robot_sdk = RobotSDK()
    robot_sdk.control.control_head(0, np.deg2rad(20))
    robot_sdk.control.set_external_control_arm_mode()  # 切换手臂到外部控制模式
    print(" ==== 机器人头部俯仰调节角度: 20 成功 ==== ")
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
    parser.add_argument('--enable_gui', action='store_true',
                        help='Enable GUI windows for camera display (default: disabled)')
    parser.add_argument('--rotate-head-camera', action='store_true',
                        help='If set, rotate head camera images (image) by 180 degrees.')
    parser.add_argument('--state-zero', action='store_true',
                        help='If set, set all state inputs to zero (for testing model dependency on state)')
    parser.add_argument('--task-description', type=str, default=None,
                        help='Task description (language instruction) for the model. If not provided, will use the first task from dataset or a default value.')
    parser.add_argument('--chunk-start', type=int, default=None,
                        help='Start index (0-based, inclusive) of the chunk to execute. If not provided, starts from the first action.')
    parser.add_argument('--chunk-end', type=int, default=None,
                        help='End index (0-based, inclusive) of the chunk to execute. If not provided, executes to the last action.')
    parser.add_argument('--model-action-dt', type=float, default=None,
                        help='Time interval between predicted actions in seconds (controls inference frequency). '
                             'Smaller values = higher frequency. Examples: 0.1 = 10 Hz, 0.05 = 20 Hz, 0.033 = 30 Hz. '
                             'Default: 0.1 (10 Hz). Note: Model was trained with 0.1s interval. Ignored in sync mode.')
    parser.add_argument('--sync-mode', action='store_true',
                        help='Enable synchronous inference mode: inference -> execute chunk -> get_obs -> repeat. '
                             'In this mode, model_action_dt is ignored.')
    parser.add_argument('--max-joint-velocity', type=float, default=None,
                        help='Maximum joint velocity limit in rad/s. If provided, will apply speed limiting to arm joints. '
                             'Example: 2.0 means max 2.0 rad/s per joint.')
    parser.add_argument('--constant-velocity', action='store_true',
                        help='Enable constant velocity mode. If set, ensures actions execute at constant velocity '
                             '(within max_joint_velocity limit). Requires --max-joint-velocity to be set.')
    parser.add_argument('--action-stride', type=int, default=1,
                        help='Action stride for speedup. If set to N, executes every N-th action, skipping intermediate ones. '
                             'Example: --action-stride=2 means ~2x speedup. Default: 1 (no skipping). '
                             'Note: This reduces the number of executed actions but does not change velocity limits.')
    parser.add_argument('--claw-lock-threshold', type=float, default=50.0,
                        help='Claw value threshold to trigger lock mechanism. If claw command exceeds this value '
                             'continuously, it will trigger locking. Default: 50.0')
    parser.add_argument('--claw-lock-count-threshold', type=int, default=1,
                        help='Number of consecutive high claw values (>= threshold) required to lock the claw. '
                             'Default: 1')
    parser.add_argument('--claw-locked-value', type=float, default=80.0,
                        help='Claw value to use when locked (fully closed). Default: 80.0')
    parser.add_argument('--disable-adapter', action='store_true',
                        help='Disable MergeVLA distribution adapter (for debugging). '
                             'If set, the model will run without the adapter even if merge_config.json exists.')
    parser.add_argument('--task-type', type=str, default=None,
                        choices=['narrower', 'wider'],
                        dest='task_type',
                        help='Specify task type for fixed routing. '
                             '"narrower" for narrow box task (task_id=0), '
                             '"wider" for wide box task (task_id=1). '
                             'If not specified, uses --smart-routing or average.')
    parser.add_argument('--smart-routing', action='store_true',
                        dest='smart_routing',
                        help='⭐ Enable MergeVLA-style Test-Time Task Routing. '
                             'When task identity is unknown, automatically infer task relevance '
                             'based on model internal parameter subspaces (value projection). '
                             'This is the recommended mode for mixed-task evaluation! '
                             '(Reference: MergeVLA paper Section 3.3)')
    parser.add_argument('--swap-task-mapping', action='store_true',
                        dest='swap_task_mapping',
                        help='⚠️ Swap task mapping: narrower↔wider. '
                             'Use this if Smart Routing gives opposite results. '
                             'When enabled: --task-type narrower uses task_id=1, '
                             '--task-type wider uses task_id=0.')
    
    args = parser.parse_args()
    
    # 验证chunk_start和chunk_end
    if args.chunk_start is not None and args.chunk_start < 0:
        parser.error(f"--chunk-start must be >= 0, got {args.chunk_start}")
    if args.chunk_end is not None and args.chunk_end < 0:
        parser.error(f"--chunk-end must be >= 0, got {args.chunk_end}")
    if args.chunk_start is not None and args.chunk_end is not None and args.chunk_start > args.chunk_end:
        parser.error(f"--chunk-start ({args.chunk_start}) must be <= --chunk-end ({args.chunk_end})")
    
    # 验证action_stride
    if args.action_stride < 1:
        parser.error(f"--action-stride must be >= 1, got {args.action_stride}")
    
    # 验证model_action_dt
    if args.model_action_dt is not None:
        if args.model_action_dt <= 0.0:
            parser.error(f"--model-action-dt must be positive, got {args.model_action_dt}")
        if args.model_action_dt > 1.0:
            parser.error(f"--model-action-dt seems too large (> 1.0s), got {args.model_action_dt}")
        print(f"⚡ Using custom MODEL_ACTION_DT: {args.model_action_dt:.3f}s (inference frequency: {1.0/args.model_action_dt:.1f} Hz)")
    else:
        print(f"⚡ Using default MODEL_ACTION_DT: {DEFAULT_MODEL_ACTION_DT:.3f}s (inference frequency: {1.0/DEFAULT_MODEL_ACTION_DT:.1f} Hz)")
    
    # 根据命令行参数和相机配置初始化GUI窗口
    camera_config = {name: info for name, info in topic_info.items() if 'image' in name}
    init_gui_windows(enable_gui=args.enable_gui, camera_config=camera_config)
    
    # 打印相机配置信息
    camera_names = get_camera_names(CAMERA_COMPONENTS)
    print(f"\n📷 Camera Configuration (TASK_DATA_MODE: {TASK_DATA_MODE}):")
    print(f"   CAMERA_COMPONENTS: {CAMERA_COMPONENTS}")
    print(f"   Camera names: {camera_names}")
    print(f"   Detected {len(camera_config)} cameras in topic_info: {list(camera_config.keys())}")
    
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
    if args.rotate_head_camera:
        print(f"🔄 Rotate head camera: Enabled (images from 'image' camera will be rotated 180 degrees)")
    if args.state_zero:
        print(f"⚠️  State zero mode: Enabled (all state inputs will be set to zero)")
    if args.task_description:
        print(f"📝 Task description: '{args.task_description}'")
    if args.chunk_start is not None or args.chunk_end is not None:
        start_idx = args.chunk_start if args.chunk_start is not None else 0
        end_idx = args.chunk_end if args.chunk_end is not None else args.action_chunk_size - 1
        print(f"⏭️  Chunk selection: will execute actions from index {start_idx} to {end_idx} (inclusive)")
    if args.sync_mode:
        print(f"🔄 Sync mode: Enabled")
    elif args.model_action_dt is not None:
        print(f"⚡ Model action DT: {args.model_action_dt:.3f}s (inference frequency: {1.0/args.model_action_dt:.1f} Hz)")
    if args.max_joint_velocity is not None:
        print(f"🚦 Max joint velocity limit: {args.max_joint_velocity:.2f} rad/s")
    if args.constant_velocity:
        if args.max_joint_velocity is None:
            print("⚠️  Warning: --constant-velocity requires --max-joint-velocity to be set. Ignoring constant-velocity mode.")
        else:
            print(f"⚙️  Constant velocity mode: Enabled (actions will execute at constant velocity within speed limit)")
    if args.action_stride > 1:
        print(f"⚡ Action stride: {args.action_stride} (executing every {args.action_stride}-th action, ~{args.action_stride}x speedup)")
    print(f"🔒 Claw lock mechanism: threshold={args.claw_lock_threshold}, count_threshold={args.claw_lock_count_threshold}, locked_value={args.claw_locked_value}")
    
    # 打印任务路由模式
    if args.task_type:
        print(f"🎯 Task Routing: Fixed task_type={args.task_type}")
    elif args.smart_routing:
        print(f"⭐ Task Routing: MergeVLA Smart Routing (自动推断任务类型)")
    else:
        print(f"⚠️ Task Routing: Simple Average (可能导致动作混乱！)")
        print(f"   建议：使用 --smart-routing 或 --task-type narrower/wider")
    print("="*80 + "\n")

    if args.eval:
        print("🚀 Starting real-time evaluation...")
        if args.disable_adapter:
            print("⚠️ MergeVLA adapter is DISABLED")
        if args.smart_routing:
            print("⭐ MergeVLA Smart Routing is ENABLED")
        eval(args.ckpt_path, model_type=args.model_type, control_arm=True, control_claw=True, 
             action_chunk_size=args.action_chunk_size, 
             enable_gui=args.enable_gui,
             rotate_head_camera=args.rotate_head_camera,
             state_zero=args.state_zero,
             task_description=args.task_description,
             chunk_start=args.chunk_start,
             chunk_end=args.chunk_end,
             model_action_dt=args.model_action_dt,
             sync_mode=args.sync_mode,
             max_joint_velocity=args.max_joint_velocity,
             constant_velocity=args.constant_velocity if args.max_joint_velocity is not None else False,
             action_stride=args.action_stride,
             claw_lock_threshold=args.claw_lock_threshold,
             claw_lock_count_threshold=args.claw_lock_count_threshold,
             claw_locked_value=args.claw_locked_value,
             disable_adapter=args.disable_adapter,
             task_type=args.task_type,
             use_smart_routing=args.smart_routing,
             swap_task_mapping=args.swap_task_mapping,  # ⚠️ 交换任务映射
        )
    elif args.replay:
        print("Replaying the model")
        lerobot_dataset_path = '/home/lab/kuavo-manip/lerobot_data/vel_wrend_box_613'
        replay(lerobot_dataset_path, episode=0, control_arm=True, control_claw=True)
    else:
        print("Please specify either --eval or --replay")
        exit(1)

    # --------------------------------------- #

