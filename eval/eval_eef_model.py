#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Demo script showing how to use Real-Time Chunking (RTC) with action chunking policies on real robots.
Supports EEF action space with IK conversion to joint space.

This script demonstrates:
1. Creating a robot and policy (Groot,SmolVLA, Pi0, etc.) with RTC
2. Consuming actions from the policy while the robot executes
3. Periodically requesting new action chunks in the background using threads
4. Managing action buffers and timing for real-time operation
5. IK conversion for EEF action space (20D -> 16D joint space)

Usage:
    python eval/eval_multi_model.py \
        --model-path=/path/to/checkpoint \
        --rtc.enabled=true \
        --rtc.execution_horizon=10 \
        --task="Depalletize the box" \
        --duration=30 \
        --ik-model-type=60
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import math

import time
import traceback
from dataclasses import dataclass, field
from threading import Event, Thread
from pathlib import Path

import torch

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from configs.config import get_camera_observation_key, get_camera_names, CAMERA_COMPONENTS, ACTION_COMPONENTS

from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from robot_envs.kuavo_depalletize_env import GrabBoxMpcEnv
from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.robot_sdk import RobotSDK

import rospy
import numpy as np

from typing import Optional
from dataclasses import dataclass

from eval_online import final_reset_arm

# 导入 Pinocchio 用于 forward kinematics
try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    pin = None
    logger.warning("⚠️  Pinocchio not available. Forward kinematics will be disabled.")


class PinocchioFK:
    """使用 Pinocchio 进行正向运动学计算（用于将 joint positions 转换为 EEF pose）"""
    def __init__(self, 
                 urdf_path: str, 
                 arm_start_idx: int = 0,
                 arm_end_idx: int = 14,
                 left_eef_frame: str = 'zarm_l7_link', 
                 right_eef_frame: str = 'zarm_r7_link',
                 shared_reference_frame: str = 'base_link'):
        """
        初始化 Pinocchio FK 计算器
        
        Args:
            urdf_path: URDF 文件路径
            arm_start_idx: 手臂关节在全关节向量中的起始索引（默认0，因为输入已经是14维关节角数组）
            arm_end_idx: 手臂关节在全关节向量中的结束索引（默认14，因为输入已经是14维关节角数组）
            left_eef_frame: 左臂末端执行器 frame 名称
            right_eef_frame: 右臂末端执行器 frame 名称
            shared_reference_frame: 共享参考 frame 名称（用于计算相对位姿）
        """
        if not PINOCCHIO_AVAILABLE:
            raise ImportError("Pinocchio is required for forward kinematics. Please install pinocchio.")
        
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()
        
        # 检查URDF的实际关节数量
        actual_nq = self.pin_model.nq
        
        # 如果URDF只有手臂关节（14个），则使用 [0:14]
        # 如果URDF包含完整机器人（更多关节），则使用传入的索引
        if actual_nq == 14:
            # URDF只包含手臂，直接使用 [0:14]
            self.arm_start_idx = 0
            self.arm_end_idx = 14
        else:
            # URDF包含完整机器人，使用传入的索引
            self.arm_start_idx = arm_start_idx
            self.arm_end_idx = arm_end_idx
            # 验证索引是否有效
            if self.arm_end_idx > actual_nq:
                raise ValueError(f"arm_end_idx ({arm_end_idx}) exceeds model nq ({actual_nq})")
            if self.arm_end_idx - self.arm_start_idx != 14:
                raise ValueError(f"Expected 14 arm joints, but indices [{self.arm_start_idx}:{self.arm_end_idx}] gives {self.arm_end_idx - self.arm_start_idx} joints")
        
        # 获取 frame IDs
        self.left_eef_frame_id = self.pin_model.getFrameId(left_eef_frame)
        self.right_eef_frame_id = self.pin_model.getFrameId(right_eef_frame)
        self.shared_reference_frame_id = self.pin_model.getFrameId(shared_reference_frame)
        
        # 初始化全关节位置向量
        self.wb_joint_pos = np.zeros(self.pin_model.nq)
        
        print(f"[PinocchioFK] ✅ Pinocchio FK initialized:")
        print(f"[PinocchioFK]   URDF: {urdf_path}")
        print(f"[PinocchioFK]   Model nq: {actual_nq}")
        print(f"[PinocchioFK]   Arm joint indices: {self.arm_start_idx} to {self.arm_end_idx-1} (total: {self.arm_end_idx - self.arm_start_idx} joints)")
        print(f"[PinocchioFK]   Left EEF frame: {left_eef_frame} (ID: {self.left_eef_frame_id})")
        print(f"[PinocchioFK]   Right EEF frame: {right_eef_frame} (ID: {self.right_eef_frame_id})")
        print(f"[PinocchioFK]   Reference frame: {shared_reference_frame} (ID: {self.shared_reference_frame_id})")
    
    def __call__(self, arm_joint_pos: np.ndarray):
        """
        计算左右臂的末端执行器位姿
        
        Args:
            arm_joint_pos: numpy array of shape (14,) - 左臂7个关节 + 右臂7个关节
        
        Returns:
            tuple: (left_eef_pose, right_eef_pose)
                - left_eef_pose: numpy array of shape (9,) - [x, y, z, R11, R21, R31, R12, R22, R32]
                - right_eef_pose: numpy array of shape (9,) - [x, y, z, R11, R21, R31, R12, R22, R32]
        """
        # 确保输入是 numpy array 且长度为14
        arm_joint_pos = np.array(arm_joint_pos, dtype=np.float64)
        if len(arm_joint_pos) != 14:
            raise ValueError(f"Expected 14 joint angles, got {len(arm_joint_pos)}")
        
        # 将手臂关节角度设置到全关节向量中
        self.wb_joint_pos[self.arm_start_idx:self.arm_end_idx] = arm_joint_pos
        
        # 执行正向运动学
        pin.forwardKinematics(self.pin_model, self.pin_data, self.wb_joint_pos)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        # 获取参考 frame 和末端执行器 frame 的位姿
        shared_reference_pose = self.pin_data.oMf[self.shared_reference_frame_id]
        left_eef_frame_pose = self.pin_data.oMf[self.left_eef_frame_id]
        right_eef_frame_pose = self.pin_data.oMf[self.right_eef_frame_id]
        
        # 计算相对于参考 frame 的位姿
        left_eef_pose = shared_reference_pose.inverse() * left_eef_frame_pose
        right_eef_pose = shared_reference_pose.inverse() * right_eef_frame_pose
        
        # 转换为 6D 旋转表示
        return self._se3_to_xyz_rot6d(left_eef_pose), self._se3_to_xyz_rot6d(right_eef_pose)
    
    def _se3_to_xyz_rot6d(self, se3) -> np.ndarray:
        """
        将 Pinocchio SE3 转换为 9维数组: [x, y, z, R11, R21, R31, R12, R22, R32]
        
        Args:
            se3: Pinocchio SE3 对象
        
        Returns:
            numpy array of shape (9,)
        """
        trans = se3.translation
        rot = se3.rotation
        return np.array([
            trans[0], trans[1], trans[2],  # x, y, z
            rot[0, 0], rot[1, 0], rot[2, 0],  # R11, R21, R31 (第一列)
            rot[0, 1], rot[1, 1], rot[2, 1],  # R12, R22, R32 (第二列)
        ], dtype=np.float32)


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
        logger.error("IKAnalytical not available. Cannot convert EEF to joint space.")
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
        logger.warning(f"Left arm IK failed: {e}")
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
        logger.warning(f"Right arm IK failed: {e}")
        right_joints = np.zeros(7)
    
    # 组合成16D joint action
    joint_action = np.concatenate([left_joints, right_joints, claw])
    
    return joint_action


@dataclass
class EpisodeState:
    first_inference: bool = True
    last_executed_action: Optional[torch.Tensor] = None
    action_dim: int = 20  # 默认EEF space (20D)
    current_reference_pose: Optional[np.ndarray] = None  # 当前reference pose（用于relative action mode）


@dataclass
class ModelBundle:
    """封装一个模型的所有组件：policy、preprocessor、postprocessor"""
    name: str
    policy: object
    preprocessor: object
    postprocessor: object
    pretrained_path: str
    action_dim: int = 20  # 模型输出的action维度
    action_space_type: Optional[str] = None  # Action space类型 ("Delta eef", "Absolute eef", "Absolute joint")
    is_relative_action_mode: bool = False  # 是否为relative action mode (Delta eef)
    postprocessor_step: Optional[object] = None  # postprocessor step引用（用于relative action转换）
    fk_getter: Optional[PinocchioFK] = None  # Forward kinematics计算器（用于将joint positions转换为EEF pose state和reference pose）
    
    def reset(self):
        """重置模型状态用于新的推理会话"""
        self.policy.reset()
        self.policy.init_rtc_processor()


@dataclass
class RTCDemoConfig:
    """Configuration for RTC demo with single model support and IK conversion."""

    # Model path
    model_path: str = field(
        # default="/home/lab/humanoid_groot/outputs/train/0124_multi_dataset_h100x4_absolute_eef_4322_2X3_groot_cross-attention_ignore_rotation/checkpoints/020000/pretrained_model",
        default="/home/lab/humanoid_groot/outputs/train/0126_multi_dataset_h100x4_relative_eef_action_4322/checkpoints/014000/pretrained_model",
        metadata={"help": "Path to the model checkpoint"}
    )

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=16,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    # Demo parameters
    duration: float = 30.0  # Duration to run the demo (seconds)
    fps: float = 10.0  # Action execution frequency (Hz)

    # Compute device
    device: str = "cuda:0"  # Device to run on

    # Get new actions horizon. The amount of executed steps after which will be requested new actions.
    # It should be higher than inference delay + execution horizon.
    action_queue_size_to_get_new_actions: int = 90
    
    # Task to execute
    task: str = field(default="Depalletize the box", metadata={"help": "Task to execute"})

    # IK configuration
    ik_model_type: str = field(
        default="60",
        metadata={"help": "Robot model type for IK solving ('45', '46', '60')"}
    )
    
    # URDF path for forward kinematics (used to convert joint positions to EEF pose for state)
    urdf_path: str = field(
        default="/home/lab/kuavo-manip/lerobot_datasets/utils/biped_s60_only_arm.urdf",
        metadata={"help": "URDF file path for forward kinematics (used to convert joint positions to EEF pose for state)"}
    )

    # Torch compile configuration
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (PyTorch 2.0+)"},
    )

    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "Backend for torch.compile (inductor, aot_eager, cudagraphs)"},
    )

    torch_compile_mode: str = field(
        default="default",
        metadata={"help": "Compilation mode (default, reduce-overhead, max-autotune)"},
    )

    torch_compile_disable_cudagraphs: bool = field(
        default=True,
        metadata={
            "help": "Disable CUDA graphs in torch.compile. Required due to in-place tensor "
            "operations in denoising loop (x_t += dt * v_t) which cause tensor aliasing issues."
        },
    )


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def resample_action_chunk(action_chunk: np.ndarray,
                          source_dt: float = 0.1,
                          target_dt: float = 0.01) -> np.ndarray:
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


def resample_chunk_with_claw_hold(action_chunk: np.ndarray,
                                  previous_action: Optional[np.ndarray],
                                  control_frequency: float,
                                  source_dt: float = 0.1,
                                  arm_dims: slice = slice(0, 14),
                                  claw_dims: slice = slice(14, 16),
                                  device: torch.device = torch.device('cuda:0')) -> torch.Tensor:
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

    return torch.from_numpy(resampled).to(device)


def apply_first_chunk_smooth(action_chunk: torch.Tensor, obs_data: dict, env: GrabBoxMpcEnv, action_dim: int, ik_model_type: str = '60'):
    """
    Apply smooth to the first chunk of actions.
    For EEF space (20D), we need to convert to joint space first for smooth interpolation.
    """
    if action_dim == 20:
        # EEF space: convert to joint space for smooth interpolation
        # Extract first action and convert to joint space
        first_action_eef = action_chunk[0].cpu().numpy().copy()
        first_action_joint = convert_eef_action_to_joint_action(first_action_eef, model_type=ik_model_type)
        
        # Get current arm state from observation
        current_arm_state = obs_data["state"][0][:14]  # First 14 dims are arm joints
        current_claw_state = np.array([0.0, 0.0])
        
        target_arm_state = first_action_joint[:14]
        target_claw_state = first_action_joint[14:16]
        
        # Calculate interpolation parameters
        transition_duration = 0.2  # Transition time (seconds)
        num_interp_steps = int(round(transition_duration / env.control_dt))
        num_interp_steps = max(1, num_interp_steps)
        
        rospy.loginfo(f"   Current arm state: {current_arm_state[:3]}... (showing first 3 joints)")
        rospy.loginfo(f"   Target arm state: {target_arm_state[:3]}... (showing first 3 joints)")
        rospy.loginfo(f"   Generating {num_interp_steps} interpolation steps over {transition_duration:.2f}s")
        
        # Generate interpolated action sequence (in joint space)
        interp_actions = []
        for i in range(num_interp_steps):
            alpha = (i + 1) / num_interp_steps
            
            # Linear interpolation for arm joints
            interp_arm = current_arm_state + (target_arm_state - current_arm_state) * alpha
            
            # Linear interpolation for claw
            interp_claw = current_claw_state + (target_claw_state - current_claw_state) * alpha
            
            # Build complete action (16D joint space)
            interp_action = np.concatenate([interp_arm, interp_claw])
            interp_actions.append(interp_action)
        
        # Convert remaining EEF actions to joint space
        remaining_eef_actions = action_chunk.cpu().numpy()
        remaining_joint_actions = []
        for eef_action in remaining_eef_actions:
            joint_action = convert_eef_action_to_joint_action(eef_action, model_type=ik_model_type)
            remaining_joint_actions.append(joint_action)
        
        # Combine transition chunk with remaining actions (all in joint space now)
        transition_chunk = np.array(interp_actions)  # (num_interp_steps, 16)
        remaining_chunk = np.array(remaining_joint_actions)  # (N, 16)
        action_chunk_smoothed = np.vstack([transition_chunk, remaining_chunk])
        rospy.loginfo(f"   Generated smoothed action chunk of size {action_chunk_smoothed.shape[0]} (converted to joint space)")
        
        return torch.from_numpy(action_chunk_smoothed).to(action_chunk.device)
    
    else:
        # Joint space (16D or 18D): original logic
        if action_dim == 16:
            arm_dims = slice(0, 14)
            claw_dims = slice(14, 16)
        elif action_dim == 18:
            arm_dims = slice(0, 14)
            claw_dims = slice(14, 16)
        else:
            arm_dims = slice(0, 14)
            claw_dims = slice(14, min(16, action_dim))

        current_arm_state = obs_data["state"][0][arm_dims]
        current_claw_state = np.array([0.0, 0.0])

        if action_chunk.shape[0] > 0:
            first_action = action_chunk[0].cpu().numpy().copy()
            target_arm_state = first_action[arm_dims]
            target_claw_state = first_action[claw_dims]
            
            # Check if cmd_pose is needed
            has_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
            if has_cmd_pose and action_dim >= 18:
                target_cmd_pose = first_action[16:18]
                current_cmd_pose = np.array([0.0, 0.0])
            else:
                target_cmd_pose = None
                current_cmd_pose = None
            
            # Calculate interpolation parameters
            transition_duration = 0.2
            num_interp_steps = int(round(transition_duration / env.control_dt))
            num_interp_steps = max(1, num_interp_steps)
            
            rospy.loginfo(f"   Current arm state: {current_arm_state[:3]}... (showing first 3 joints)")
            rospy.loginfo(f"   Target arm state: {target_arm_state[:3]}... (showing first 3 joints)")
            rospy.loginfo(f"   Generating {num_interp_steps} interpolation steps over {transition_duration:.2f}s")
            
            # Generate interpolated action sequence
            interp_actions = []
            for i in range(num_interp_steps):
                alpha = (i + 1) / num_interp_steps
                
                interp_arm = current_arm_state + (target_arm_state - current_arm_state) * alpha
                interp_claw = current_claw_state + (target_claw_state - current_claw_state) * alpha
                
                if has_cmd_pose and target_cmd_pose is not None:
                    interp_cmd_pose = current_cmd_pose + (target_cmd_pose - current_cmd_pose) * alpha
                    interp_action = np.concatenate([interp_arm, interp_claw, interp_cmd_pose])
                else:
                    interp_action = np.concatenate([interp_arm, interp_claw])
                
                interp_actions.append(interp_action)
            
            transition_chunk = np.array(interp_actions)
            action_chunk_smoothed = np.vstack([transition_chunk, action_chunk.cpu().numpy()])
            rospy.loginfo(f"   Generated smoothed action chunk of size {action_chunk_smoothed.shape[0]}")

        return torch.from_numpy(action_chunk_smoothed).to(action_chunk.device)


def get_actions(
    model_bundle: ModelBundle,
    env: GrabBoxMpcEnv,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
    episode_state: EpisodeState,
):
    """Thread function to request action chunks from the policy.

    Args:
        model_bundle: 包含policy、preprocessor、postprocessor的模型包
        env: 机器人环境
        action_queue: Queue to put new action chunks
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
        episode_state: 当前episode的状态
    """

    try:
        logger.info(f"[GET_ACTIONS] Starting get actions thread with model: {model_bundle.name}")

        latency_tracker = LatencyTracker()
        fps = cfg.fps
        time_per_chunk = 1.0 / fps

        preprocessor = model_bundle.preprocessor
        postprocessor = model_bundle.postprocessor
        policy = model_bundle.policy
        action_dim = model_bundle.action_dim

        logger.info(f"[GET_ACTIONS] Using preprocessor/postprocessor from model: {model_bundle.name}")
        logger.info(f"[GET_ACTIONS] Action dimension: {action_dim}D")
        if model_bundle.action_space_type:
            logger.info(f"[GET_ACTIONS] Action space type: {model_bundle.action_space_type}")
        if model_bundle.fk_getter is not None:
            logger.info(f"[GET_ACTIONS] Forward kinematics available: Will convert joint positions to EEF pose for state")

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions

        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        while not shutdown_event.is_set():
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk)

                obs_data, *_ = env.get_obs()

                # 对于 EEF action space (Delta eef 或 Absolute eef)，需要将 joint positions 转换为 EEF pose
                # 因为训练时 state 是 20 维的 EEF pose，而推理时只有 16 维的 joint positions
                is_eef_mode = (model_bundle.action_space_type in ["Delta eef", "Absolute eef"] and action_dim == 20)
                if is_eef_mode and model_bundle.fk_getter is not None:
                    # 从 joint positions 计算 EEF pose
                    current_state_np = obs_data["state"][0]  # (state_dim,) - 取第一个时间步
                    
                    # 从 state 中提取 arm joint positions 和 gripper state
                    # 根据 STATE_COMPONENTS 配置，默认是 ["J_q", "Claw_pos"]，即前14维是关节，接下来2维是gripper
                    state_dim = len(current_state_np)
                    
                    # 尝试从 config 获取 STATE_COMPONENTS（如果可用）
                    try:
                        from configs.config import STATE_COMPONENTS
                        # 计算各个组件的索引
                        joint_start_idx = 0
                        joint_end_idx = 0
                        gripper_start_idx = 0
                        gripper_end_idx = 0
                        
                        current_idx = 0
                        for component in STATE_COMPONENTS:
                            if component == "J_q":
                                joint_start_idx = current_idx
                                joint_end_idx = current_idx + 14
                                current_idx += 14
                            elif component == "Claw_pos":
                                gripper_start_idx = current_idx
                                gripper_end_idx = current_idx + 2
                                current_idx += 2
                            elif component == "IMU":
                                current_idx += 6  # 跳过 IMU
                            elif component == "Com_z_pitch":
                                current_idx += 2  # 跳过 Com_z_pitch
                    except ImportError:
                        # 如果无法导入 config，使用默认配置：前14维是关节，接下来2维是gripper
                        joint_start_idx = 0
                        joint_end_idx = 14
                        gripper_start_idx = 14
                        gripper_end_idx = 16
                    
                    # 提取 arm joint positions (14维)
                    if joint_end_idx <= state_dim:
                        arm_joint_pos = current_state_np[joint_start_idx:joint_end_idx]
                    else:
                        logger.warning(f"[GET_ACTIONS] State dimension ({state_dim}) is too small to extract arm joints. Using zeros.")
                        arm_joint_pos = np.zeros(14, dtype=np.float32)
                    
                    # 提取 gripper state (2维)
                    if gripper_end_idx <= state_dim:
                        gripper_state = current_state_np[gripper_start_idx:gripper_end_idx]
                    else:
                        logger.warning(f"[GET_ACTIONS] State dimension ({state_dim}) is too small to extract gripper state. Using zeros.")
                        gripper_state = np.zeros(2, dtype=np.float32)
                    
                    # 使用 forward kinematics 计算 EEF pose
                    try:
                        left_eef_pose, right_eef_pose = model_bundle.fk_getter(arm_joint_pos)
                        # 组合成 20D EEF pose: left_eef(9) + right_eef(9) + gripper(2)
                        eef_state = np.concatenate([
                            left_eef_pose,   # (9,)
                            right_eef_pose,  # (9,)
                            gripper_state    # (2,)
                        ]).astype(np.float32)
                        
                        # 转换为 torch tensor，并添加时间维度以匹配训练格式
                        state = torch.from_numpy(eef_state).float().unsqueeze(0)  # (1, 20)
                        if episode_state.first_inference:
                            print(f"[GET_ACTIONS] ✅ Converted joint positions (16D) to EEF pose state (20D) using FK")
                            print(f"[GET_ACTIONS]   Original joint state shape: {obs_data['state'].shape}")
                            print(f"[GET_ACTIONS]   Converted EEF state shape: {state.shape}")
                            print(f"[GET_ACTIONS]   EEF state components: left_eef(9) + right_eef(9) + gripper(2) = 20D")
                            print(f"[GET_ACTIONS]   Left EEF pose (first 3 pos + first 2 rot6d): {eef_state[0:5]}")
                            print(f"[GET_ACTIONS]   Right EEF pose (first 3 pos + first 2 rot6d): {eef_state[9:14]}")
                            print(f"[GET_ACTIONS]   Gripper state: {eef_state[18:20]}")
                        else:
                            logger.debug(f"[GET_ACTIONS] ✅ Converted joint positions to EEF pose state (20D) using FK")
                    except Exception as e:
                        logger.error(f"[GET_ACTIONS] ❌ Failed to compute FK: {e}")
                        # Fallback: 使用原始 state（16D），但模型可能无法正常工作
                        logger.warning(f"[GET_ACTIONS] Using original joint state (16D) as fallback - model may not work correctly!")
                        state = torch.from_numpy(obs_data["state"]).float()
                else:
                    # Joint space 或 FK 不可用：使用原始 state
                    state = torch.from_numpy(obs_data["state"]).float()
                    if episode_state.first_inference:
                        if is_eef_mode and model_bundle.fk_getter is None:
                            print(f"[GET_ACTIONS] ⚠️  EEF mode detected but FK not available!")
                            print(f"[GET_ACTIONS]   action_space_type={model_bundle.action_space_type}, action_dim={action_dim}")
                            print(f"[GET_ACTIONS]   fk_getter={model_bundle.fk_getter}")
                            print(f"[GET_ACTIONS]   Using original joint state (16D) - model will NOT work correctly!")
                            print(f"[GET_ACTIONS]   State shape: {state.shape}, Expected: (1, 20)")
                        elif is_eef_mode:
                            print(f"[GET_ACTIONS] ⚠️  EEF mode detected but FK conversion was skipped!")
                            print(f"[GET_ACTIONS]   is_eef_mode={is_eef_mode}, fk_getter={model_bundle.fk_getter}")
                            print(f"[GET_ACTIONS]   State shape: {state.shape}, Expected: (1, 20)")
                        else:
                            print(f"[GET_ACTIONS] Using original state (not EEF mode): shape={state.shape}")
                
                observation = {}
                
                # 根据CAMERA_COMPONENTS动态处理相机图像
                camera_names = get_camera_names(CAMERA_COMPONENTS)
                for camera_name in camera_names:
                    if camera_name in obs_data:
                        camera_img_np = obs_data[camera_name]
                        if camera_img_np.ndim != 4:
                            rospy.logwarn(f"Unexpected camera image shape: {camera_img_np.shape}, expected (T, H, W, C)")
                            continue
                        camera_images = torch.from_numpy(np.moveaxis(camera_img_np, 3, 1).copy()).float() / 255
                        obs_key = get_camera_observation_key(camera_name, use_image_features=False)
                        observation[obs_key] = camera_images.to('cuda:0')
                
                observation['observation.state'] = state.to('cuda:0')
                observation['task'] = cfg.task
                
                # 验证 state 维度（仅在第一次推理时）
                if episode_state.first_inference:
                    state_dim = state.shape[1]
                    expected_state_dim = 20 if is_eef_mode else 16
                    if state_dim != expected_state_dim:
                        print(f"[GET_ACTIONS] ❌ State dimension mismatch!")
                        print(f"[GET_ACTIONS]   Expected: {expected_state_dim}D, Got: {state_dim}D")
                        print(f"[GET_ACTIONS]   is_eef_mode={is_eef_mode}, action_space_type={model_bundle.action_space_type}")
                    else:
                        print(f"[GET_ACTIONS] ✅ State dimension verified: {state_dim}D (expected: {expected_state_dim}D)")
                
                processed_observation = preprocessor(observation)

                # Generate actions WITH RTC
                actions = policy.predict_action_chunk(
                    processed_observation,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

                original_actions = actions.squeeze(0).clone()

                _, chunk_size, _ = actions.shape
                processed_actions = []
                for i in range(chunk_size):
                    single_action = actions[:, i, :]
                    processed_action = postprocessor(single_action)
                    processed_actions.append(processed_action)
                
                pred_actions_unnorm = torch.stack(processed_actions, dim=1)
                postprocessed_actions = pred_actions_unnorm[0].clone()
                
                # 对于 relative action mode (Delta eef)，需要将 relative action 转换为 absolute pose
                if model_bundle.is_relative_action_mode and model_bundle.postprocessor_step is not None:
                    # 每次 RTC 触发新推理时，从当前 robot joint state 通过 FK 计算 reference pose
                    # 这是正确的做法：使用当前机器人的实际状态作为 reference
                    current_state_np = obs_data["state"][0]  # (state_dim,) - 取第一个时间步
                    
                    # 从 state 中提取 arm joint positions 和 gripper state
                    # 根据 STATE_COMPONENTS 配置，默认是 ["J_q", "Claw_pos"]，即前14维是关节，接下来2维是gripper
                    state_dim = len(current_state_np)
                    
                    # 尝试从 config 获取 STATE_COMPONENTS（如果可用）
                    try:
                        from configs.config import STATE_COMPONENTS
                        # 计算各个组件的索引
                        joint_start_idx = 0
                        joint_end_idx = 0
                        gripper_start_idx = 0
                        gripper_end_idx = 0
                        
                        current_idx = 0
                        for component in STATE_COMPONENTS:
                            if component == "J_q":
                                joint_start_idx = current_idx
                                joint_end_idx = current_idx + 14
                                current_idx += 14
                            elif component == "Claw_pos":
                                gripper_start_idx = current_idx
                                gripper_end_idx = current_idx + 2
                                current_idx += 2
                            elif component == "IMU":
                                current_idx += 6  # 跳过 IMU
                            elif component == "Com_z_pitch":
                                current_idx += 2  # 跳过 Com_z_pitch
                    except ImportError:
                        # 如果无法导入 config，使用默认配置：前14维是关节，接下来2维是gripper
                        joint_start_idx = 0
                        joint_end_idx = 14
                        gripper_start_idx = 14
                        gripper_end_idx = 16
                    
                    # 提取 arm joint positions (14维)
                    if joint_end_idx <= state_dim:
                        arm_joint_pos = current_state_np[joint_start_idx:joint_end_idx]
                    else:
                        logger.warning(f"[GET_ACTIONS] State dimension ({state_dim}) is too small to extract arm joints. Using zeros.")
                        arm_joint_pos = np.zeros(14, dtype=np.float32)
                    
                    # 提取 gripper state (2维)
                    if gripper_end_idx <= state_dim:
                        gripper_state = current_state_np[gripper_start_idx:gripper_end_idx]
                    else:
                        logger.warning(f"[GET_ACTIONS] State dimension ({state_dim}) is too small to extract gripper state. Using zeros.")
                        gripper_state = np.zeros(2, dtype=np.float32)
                    
                    # 使用 forward kinematics 计算 EEF pose（每次推理都重新计算）
                    if model_bundle.fk_getter is not None:
                        try:
                            left_eef_pose, right_eef_pose = model_bundle.fk_getter(arm_joint_pos)
                            # 组合成 20D EEF pose: left_eef(9) + right_eef(9) + gripper(2)
                            current_reference_pose = np.concatenate([
                                left_eef_pose,   # (9,)
                                right_eef_pose,  # (9,)
                                gripper_state    # (2,)
                            ]).astype(np.float32)
                            
                            episode_state.current_reference_pose = current_reference_pose
                            logger.debug(f"[GET_ACTIONS] ✅ Computed reference pose from current joint state using FK (relative action mode)")
                            logger.debug(f"[GET_ACTIONS]   Arm joints: {arm_joint_pos[:3]}... (showing first 3)")
                            logger.debug(f"[GET_ACTIONS]   Gripper: {gripper_state}")
                        except Exception as e:
                            logger.error(f"[GET_ACTIONS] ❌ Failed to compute FK: {e}")
                            # Fallback: 如果 FK 失败，使用上一次的 reference pose 或零向量
                            if episode_state.current_reference_pose is None:
                                logger.warning(f"[GET_ACTIONS] Using zero vector as fallback reference pose")
                                episode_state.current_reference_pose = np.zeros(action_dim, dtype=np.float32)
                    else:
                        # FK 不可用，使用上一次的 reference pose 或零向量
                        if episode_state.current_reference_pose is None:
                            logger.warning(f"[GET_ACTIONS] ⚠️  FK not available. Using zero vector as initial reference pose")
                            episode_state.current_reference_pose = np.zeros(action_dim, dtype=np.float32)
                    
                    # 将 numpy 转换为 torch tensor
                    pred_chunk_tensor = postprocessed_actions.unsqueeze(0)  # (1, chunk_size, action_dim)
                    reference_pose_tensor = torch.from_numpy(episode_state.current_reference_pose).to(
                        pred_chunk_tensor.device
                    ).unsqueeze(0)  # (1, action_dim)
                    
                    # 调用 postprocessor step 的转换方法
                    pred_chunk_absolute = model_bundle.postprocessor_step._convert_relative_to_absolute_eef_action(
                        pred_chunk_tensor, reference_pose_tensor
                    )
                    
                    # 更新 postprocessed_actions
                    postprocessed_actions = pred_chunk_absolute[0].clone()  # (chunk_size, action_dim)
                    
                    # 注意：reference pose 已经在每次推理时从当前 joint state 通过 FK 计算
                    # 不需要在这里更新，因为下一次推理时会重新从最新的 joint state 计算
                    logger.debug(f"[GET_ACTIONS] Converted relative actions to absolute poses (relative action mode)")

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] cfg.action_queue_size_to_get_new_actions Too small, It should be higher than inference delay + execution horizon."
                    )

                # Convert EEF actions to joint actions if needed
                if action_dim == 20:
                    # EEF space: convert to joint space
                    if episode_state.first_inference:
                        # For first inference, apply smooth transition (converts to joint space internally)
                        postprocessed_actions = apply_first_chunk_smooth(postprocessed_actions, obs_data, env, action_dim, ik_model_type=cfg.ik_model_type)
                        episode_state.first_inference = False
                    else:
                        # For subsequent inferences, just convert EEF to joint space
                        postprocessed_actions_np = postprocessed_actions.cpu().numpy()
                        joint_actions_list = []
                        for i in range(postprocessed_actions_np.shape[0]):
                            eef_action = postprocessed_actions_np[i]
                            joint_action = convert_eef_action_to_joint_action(eef_action, model_type=cfg.ik_model_type)
                            joint_actions_list.append(joint_action)
                        postprocessed_actions = torch.from_numpy(np.array(joint_actions_list)).to(postprocessed_actions.device)
                    # Update arm_dims and claw_dims for joint space (16D)
                    arm_dims = slice(0, 14)
                    claw_dims = slice(14, 16)
                else:
                    # Joint space: use original dimensions
                    if episode_state.first_inference:
                        postprocessed_actions = apply_first_chunk_smooth(postprocessed_actions, obs_data, env, action_dim)
                        episode_state.first_inference = False
                    arm_dims = slice(0, 14)
                    claw_dims = slice(14, 16)

                postprocessed_resampled = resample_chunk_with_claw_hold(
                    postprocessed_actions.cpu().numpy(),
                    previous_action=(episode_state.last_executed_action.cpu().numpy() if episode_state.last_executed_action is not None else None),
                    control_frequency=100.0,
                    source_dt=0.1,
                    arm_dims=arm_dims,
                    claw_dims=claw_dims,
                    device=postprocessed_actions.device,
                )

                episode_state.last_executed_action = postprocessed_resampled[-get_actions_threshold].clone()
                action_queue.merge(
                    original_actions, postprocessed_resampled, new_delay, action_index_before_inference
                )
            else:
                time.sleep(0.01)

        logger.info("[GET_ACTIONS] get actions thread shutting down")
    except Exception as e:
        print("\n" + "="*80, file=sys.stderr, flush=True)
        print("[GET_ACTIONS] FATAL EXCEPTION", file=sys.stderr, flush=True)
        traceback.print_exc()
        print("="*80 + "\n", file=sys.stderr, flush=True)
        shutdown_event.set()
        return


def actor_control(
    env: GrabBoxMpcEnv,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    """Thread function to execute actions on the robot.

    Args:
        env: The robot environment
        action_queue: Queue to get actions from
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_count = 0
        action_interval = 1.0 / 100

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            if action_queue.qsize() > 0:
                action = action_queue.get()
            else:
                action = None

            if action is not None:
                action = action.cpu()
                control_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
                env.exec_actions(
                    actions=action,
                    control_arm=True,
                    control_claw=True,
                    control_cmd_pose=control_cmd_pose
                )
                action_count += 1

            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, (action_interval - dt_s) - 0.001))

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        print("\n" + "="*80, file=sys.stderr, flush=True)
        print("[ACTOR] FATAL EXCEPTION", file=sys.stderr, flush=True)
        traceback.print_exc()
        print("="*80 + "\n", file=sys.stderr, flush=True)
        shutdown_event.set()
        return


def _apply_torch_compile(policy, cfg: RTCDemoConfig):
    """Apply torch.compile to the policy's predict_action_chunk method."""
    if policy.type == "groot":
        return policy

    try:
        if not hasattr(torch, "compile"):
            logger.warning(
                f"torch.compile is not available. Requires PyTorch 2.0+. "
                f"Current version: {torch.__version__}. Skipping compilation."
            )
            return policy

        logger.info("Applying torch.compile to predict_action_chunk...")
        logger.info(f"  Backend: {cfg.torch_compile_backend}")
        logger.info(f"  Mode: {cfg.torch_compile_mode}")
        logger.info(f"  Disable CUDA graphs: {cfg.torch_compile_disable_cudagraphs}")

        compile_kwargs = {
            "backend": cfg.torch_compile_backend,
            "mode": cfg.torch_compile_mode,
        }

        if cfg.torch_compile_disable_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}

        original_method = policy.predict_action_chunk
        compiled_method = torch.compile(original_method, **compile_kwargs)
        policy.predict_action_chunk = compiled_method
        logger.info("✓ Successfully compiled predict_action_chunk")

    except Exception as e:
        logger.error(f"Failed to apply torch.compile: {e}")
        logger.warning("Continuing without torch.compile")

    return policy


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


def load_model_bundle(
    model_path: str,
    cfg: RTCDemoConfig,
    device: str = "cuda:0"
) -> ModelBundle:
    """
    加载单个模型并创建ModelBundle
    
    Args:
        model_path: 模型路径
        cfg: 配置
        device: 设备
    
    Returns:
        ModelBundle 包含policy、preprocessor、postprocessor
    """
    logger.info(f"[LOAD] Loading model from {model_path}")
    
    # 加载配置
    config = PreTrainedConfig.from_pretrained(model_path)
    
    if config.type in ["pi05", "pi0", "groot"]:
        config.compile_model = cfg.use_torch_compile
    
    # 加载policy
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(model_path, config=config)
    policy.config.rtc_config = cfg.rtc
    policy.init_rtc_processor()
    policy = policy.to(device)
    policy.eval()
    
    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)
    
    # 加载preprocessor和postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=model_path,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )
    
    # 检测action维度
    # 从config中获取action_dim
    if hasattr(config, 'action_dim') and config.action_dim is not None:
        action_dim = config.action_dim
    elif hasattr(policy, 'actual_action_dim'):
        action_dim = policy.actual_action_dim
    else:
        # 默认值
        action_dim = 20
        logger.warning(f"Could not determine action_dim from config, using default: {action_dim}")
    
    # 检测 action_space_type
    action_space_type = getattr(config, 'action_space_type', None)
    print(f"[LOAD] Detected action_space_type: {action_space_type}")
    print(f"[LOAD] Detected action_dim: {action_dim}")
    print(f"[LOAD] PINOCCHIO_AVAILABLE: {PINOCCHIO_AVAILABLE}")
    
    is_relative_action_mode = (action_space_type == "Delta eef" and action_dim == 20)
    postprocessor_step = None
    fk_getter = None
    
    # 对于 EEF action space (Delta eef 或 Absolute eef)，初始化 forward kinematics 计算器
    # 因为训练时 state 是 20 维的 EEF pose，而推理时只有 16 维的 joint positions
    is_eef_mode = (action_space_type in ["Delta eef", "Absolute eef"] and action_dim == 20)
    print(f"[LOAD] EEF mode check: action_space_type={action_space_type}, action_dim={action_dim}, is_eef_mode={is_eef_mode}")
    
    if is_eef_mode:
        print(f"[LOAD] Entering EEF mode initialization block")
        if PINOCCHIO_AVAILABLE:
            print(f"[LOAD] Pinocchio is available, attempting FK initialization...")
            print(f"[LOAD] URDF path: {cfg.urdf_path}")
            try:
                print(f"[LOAD] Creating PinocchioFK instance...")
                fk_getter = PinocchioFK(
                    urdf_path=cfg.urdf_path,
                    arm_start_idx=0,
                    arm_end_idx=14,
                    left_eef_frame='zarm_l7_link',
                    right_eef_frame='zarm_r7_link',
                    shared_reference_frame='base_link'
                )
                print(f"[LOAD] ✅ Forward kinematics initialized for EEF mode")
                print(f"[LOAD]   Will convert joint positions (16D) to EEF pose (20D) for state")
                if is_relative_action_mode:
                    print(f"[LOAD]   Will also use FK to compute reference pose for relative action conversion")
            except Exception as e:
                print(f"[LOAD] ❌ Failed to initialize forward kinematics: {e}")
                print(f"[LOAD]   Exception type: {type(e).__name__}")
                import traceback
                print(f"[LOAD]   Traceback:")
                traceback.print_exc()
                print(f"[LOAD]   Model may not work correctly without FK for state conversion!")
                fk_getter = None
        else:
            print(f"[LOAD] ⚠️  Pinocchio not available. Cannot convert joint positions to EEF pose for state.")
            print(f"[LOAD]   Model may not work correctly without FK for state conversion!")
    else:
        print(f"[LOAD] Not in EEF mode, skipping FK initialization")
    
    # 如果是 relative action mode，查找 postprocessor step
    if is_relative_action_mode:
        # 查找postprocessor step以获取转换方法
        for step in postprocessor.steps:
            if hasattr(step, 'action_space_type') and hasattr(step, 'action_component_indices'):
                if step.action_space_type == "Delta eef" and step.action_component_indices is not None:
                    postprocessor_step = step
                    logger.info(f"[LOAD] ✅ Relative action mode detected (Delta eef)")
                    logger.info(f"[LOAD]   - Will convert relative actions to absolute poses during inference")
                    logger.info(f"[LOAD]   - Reference pose will be computed from current robot joint state using FK")
                    break
    
    print(f"[LOAD] Model loaded successfully")
    print(f"[LOAD] Action dimension: {action_dim}D")
    if action_space_type:
        print(f"[LOAD] Action space type: {action_space_type}")
    if fk_getter is not None:
        print(f"[LOAD] ✅ FK getter is ready: {fk_getter is not None}")
    else:
        print(f"[LOAD] ⚠️  FK getter is None - state conversion will fail!")
    
    return ModelBundle(
        name="Single Model",
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        pretrained_path=model_path,
        action_dim=action_dim,
        action_space_type=action_space_type,
        is_relative_action_mode=is_relative_action_mode,
        postprocessor_step=postprocessor_step,
        fk_getter=fk_getter,
    )


@parser.wrap()
def demo_cli(cfg: RTCDemoConfig):
    """Main entry point for RTC demo with draccus configuration."""

    init_logging()
    logger.info(f"[MAIN] Using device: {cfg.device}")

    # 设置信号处理器
    from lerobot.rl.process import ProcessSignalHandler
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    env = GrabBoxMpcEnv()
    robot_sdk = RobotSDK()

    # 初始化手臂位置
    robot_sdk.control.set_external_control_arm_mode()
    robot_sdk.control.control_head(0, np.deg2rad(20))
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    final_reset_arm(
        json_path=os.path.join(cur_dir, 'utils/initial_arm_traj.json'), 
        env=env,
        control_arm=True,
        control_claw=True
    )
    set_arm_quick_mode(True)
    
    # ========== 加载单模型 ==========
    logger.info(f"[MAIN] Loading model from {cfg.model_path}")
    
    model_bundle = load_model_bundle(
        model_path=cfg.model_path,
        cfg=cfg,
        device=cfg.device
    )
    
    logger.info(f"[MAIN] Model loaded successfully!")
    logger.info(f"[MAIN] Action dimension: {model_bundle.action_dim}D")
    if model_bundle.action_space_type:
        logger.info(f"[MAIN] Action space type: {model_bundle.action_space_type}")
    if model_bundle.action_dim == 20:
        logger.info(f"[MAIN] EEF action space detected - IK conversion will be applied")
        logger.info(f"[MAIN] IK model type: {cfg.ik_model_type}")
        if model_bundle.is_relative_action_mode:
            logger.info(f"[MAIN] Relative action mode (Delta eef) - will convert relative actions to absolute poses")
    
    # 等待观测buffer准备好
    env.obs_buffer.wait_buffer_ready()

    logger.info(f"[MAIN] Task: {cfg.task}")
    logger.info(f"[MAIN] Duration: {cfg.duration}s")
    
    while True:
        input("\n按回车开始推理...")
        
        # 重置模型状态
        model_bundle.reset()
        shutdown_event.clear()
        episode_state = EpisodeState(action_dim=model_bundle.action_dim)
        action_queue = ActionQueue(cfg.rtc)
        logger.info(f"[MAIN] ActionQueue created")
        
        # 启动线程
        get_actions_thread = Thread(
            target=get_actions, 
            args=(model_bundle, env, action_queue, shutdown_event, cfg, episode_state), 
            daemon=True, 
            name="GetActions"
        )
        actor_thread = Thread(
            target=actor_control, 
            args=(env, action_queue, shutdown_event, cfg), 
            daemon=True, 
            name="Actor"
        )
        get_actions_thread.start()
        actor_thread.start()
        
        print(f"💡 推理中... {cfg.duration}秒后自动停止，或输入 'q' 手动停止")
        
        # 使用 duration 自动停止
        start_time = time.perf_counter()
        import select
        import sys as _sys
        
        try:
            while not shutdown_event.is_set():
                elapsed = time.perf_counter() - start_time
                if elapsed >= cfg.duration:
                    logger.info(f"⏱️ 达到时间限制 ({cfg.duration}秒)，自动停止")
                    break
                
                # 非阻塞检查用户输入
                if select.select([_sys.stdin], [], [], 0.1)[0]:
                    user_input = _sys.stdin.readline().strip().lower()
                    if user_input == 'q':
                        logger.info("👆 用户手动停止")
                        break
        except KeyboardInterrupt:
            logger.info("[MAIN] Keyboard interrupt received")
            shutdown_event.set()
        
        shutdown_event.set()
        env.reset_claw_lock()
        final_reset_arm(
            json_path=os.path.join(cur_dir, 'utils/initial_arm_traj.json'), 
            env=env,
            control_arm=True,
            control_claw=True
        )
        
        get_actions_thread.join()
        actor_thread.join()
        logger.info("[MAIN] 本轮推理完成，可以继续下一轮或输入 'q' 退出程序")
        
        # 询问是否继续
        try:
            user_input = input("按回车继续下一轮推理，或输入 'q' 退出程序: ").strip().lower()
            if user_input == 'q':
                logger.info("[MAIN] 用户选择退出程序")
                break
        except KeyboardInterrupt:
            logger.info("[MAIN] Keyboard interrupt received, exiting")
            break
    
    logger.info("[MAIN] Demo finished.")


if __name__ == "__main__":
    demo_cli()
    logging.info("RTC demo finished")
