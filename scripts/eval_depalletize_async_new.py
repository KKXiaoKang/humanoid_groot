#!/usr/bin/env python
"""
异步推理客户端：将 GROOT policy 与 GrabBoxMpcEnv 结合进行异步推理

这个脚本实现了异步推理架构，将推理任务分离到服务器端，客户端专注于
观测获取和动作执行，从而提升整体性能。

架构：
- 服务器端 (policy_server.py): 运行模型推理，返回动作块
- 客户端 (本脚本): 发送观测，接收动作，执行控制

使用方法：
1. 启动服务器：
   python -m lerobot.async_inference.policy_server \
       --host=127.0.0.1 \
       --port=8080 \
       --fps=30 \
       --inference_latency=0.033

2. 运行客户端（本脚本）：
   python scripts/eval_depalletize_async.py \
       --server_address=127.0.0.1:8080 \
       --ckpt_path=/path/to/checkpoint \
       --task_description="Depalletize the box"
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import threading
import pickle
from queue import Queue, Empty
from collections import deque
from typing import Optional
import argparse

import numpy as np
import torch
import grpc
import rospy

from robot_envs.kuavo_depalletize_env import GrabBoxMpcEnv
from configs.config import topic_info, \
    get_camera_observation_key, get_camera_names, \
    TASK_DATA_MODE, CAMERA_COMPONENTS, ACTION_COMPONENTS, ROBOT_VERSION, STATE_COMPONENTS
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import PolicyFeature
from pathlib import Path

from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks, receive_bytes_in_chunks
from lerobot.async_inference.helpers import (
    TimedObservation,
    TimedAction,
    RawObservation,
    RemotePolicyConfig,
    get_logger,
)

# 导入必要的工具函数
from scripts.eval_depalletize_camera_model_reload import (
    resample_chunk_with_claw_hold,
    apply_lowpass_transition,
    publish_joint_positions,
    load_and_replay_init_trajectory,
    final_reset_arm,
    set_arm_quick_mode,
    direct_to_wbc,
    change_arm_ctrl_mode
)

MODEL_ACTION_DT = 0.1  # seconds between predicted actions during training
MODEL_ACTION_FREQUENCY = 1.0 / MODEL_ACTION_DT
TARGET_CONTROL_FREQUENCY = 100.0
TARGET_CONTROL_DT = 1.0 / TARGET_CONTROL_FREQUENCY
CHUNK_TRANSITION_DURATION_S = 0.2  # seconds of low-pass smoothing at chunk boundary
LOWPASS_ALPHA = 0.85  # closer to 1 => smoother (slower) transitions
ENABLE_CHUNK_TRANSITION_LOWPASS = True

logger = get_logger("groot_async_client")


class GrootAsyncClient:
    """异步推理客户端，适配 GrabBoxMpcEnv 和 GROOT policy"""
    
    def __init__(
        self,
        server_address: str,
        ckpt_path: str,
        action_chunk_size: int = 20,
        lerobot_dataset_path: Optional[str] = None,
        task_description: str = "Depalletize the box",
        control_arm: bool = True,
        control_claw: bool = True,
        fps: float = 30.0,
        chunk_size_threshold: float = 0.5,
        rotate_head_camera: bool = False,
    ):
        """
        Args:
            server_address: gRPC 服务器地址 (格式: "host:port")
            ckpt_path: GROOT 模型 checkpoint 路径
            action_chunk_size: 动作块大小
            lerobot_dataset_path: 数据集路径（用于加载统计信息）
            task_description: 任务描述
            control_arm: 是否控制手臂
            control_claw: 是否控制夹爪
            fps: 控制频率
            chunk_size_threshold: 动作队列阈值（当队列大小/动作块大小 < threshold 时发送新观测）
            rotate_head_camera: 是否旋转头部相机图像180度
        """
        self.server_address = server_address
        self.ckpt_path = ckpt_path
        self.action_chunk_size = action_chunk_size
        self.task_description = task_description
        self.control_arm = control_arm
        self.control_claw = control_claw
        self.fps = fps
        self.environment_dt = 1.0 / fps
        self.chunk_size_threshold = chunk_size_threshold
        self.rotate_head_camera = rotate_head_camera
        
        # 初始化环境
        rospy.loginfo("Initializing GrabBoxMpcEnv...")
        self.env = GrabBoxMpcEnv()
        self.env.obs_buffer.wait_buffer_ready()
        time.sleep(1)
        rospy.loginfo("Environment ready")
        
        # 加载数据集统计信息（用于特征映射）
        self.dataset_stats = None
        self.lerobot_features = self._load_dataset_and_get_features(lerobot_dataset_path)
        
        # 连接 gRPC 服务器
        rospy.loginfo(f"Connecting to policy server at {server_address}...")
        self.channel = grpc.insecure_channel(
            server_address, 
            grpc_channel_options(initial_backoff=f"{self.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        
        # 发送 Ready 信号
        try:
            self.stub.Ready(services_pb2.Empty())
            rospy.loginfo("Connected to policy server")
        except grpc.RpcError as e:
            logger.error(f"Failed to connect to server: {e}")
            raise
        
        # 发送策略配置
        policy_config = RemotePolicyConfig(
            policy_type="groot",
            pretrained_name_or_path=ckpt_path,
            lerobot_features=self.lerobot_features,
            actions_per_chunk=action_chunk_size,
            device="cuda:0",
            dataset_stats=self.dataset_stats,
        )
        policy_config_bytes = pickle.dumps(policy_config)
        policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)
        self.stub.SendPolicyInstructions(policy_setup)
        rospy.loginfo("Policy configuration sent to server")
        
        # 动作队列和同步
        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.latest_action = -1
        self.latest_action_lock = threading.Lock()
        self.action_chunk_size_received = action_chunk_size
        
        # 线程控制
        self.shutdown_event = threading.Event()
        self.must_go = threading.Event()
        self.must_go.set()  # 初始设置为可发送观测
        
        # ROS publishers
        from std_msgs.msg import Float64MultiArray
        self.joint_pub = rospy.Publisher(
            '/policy/action/eef_pose_marker_all', 
            Float64MultiArray,
            queue_size=10
        )
        
        rospy.loginfo("GrootAsyncClient initialized")
    
    def _load_dataset_and_get_features(self, lerobot_dataset_path: Optional[str]) -> dict[str, dict]:
        """加载数据集并获取特征映射（用于服务器端转换）
        
        返回格式: dict[str, dict] - LeRobot 数据集特征格式（带 "observation." 前缀）
        这个格式会被服务器端用于将 RawObservation 转换为 LeRobot 格式
        
        注意：build_dataset_frame 期望：
        - ds_features: 数据集特征格式（如 "observation.state", "observation.images.image"）
        - values: 硬件特征值（如 {"state": tensor, "image": tensor}）
        - 通过 ds_features 中的 "names" 字段将 values 中的键映射到数据集特征
        """
        if not lerobot_dataset_path:
            # 使用默认路径
            lerobot_dataset_path = '/home/lab/lerobot_groot/lerobot_data/new_demo/1118_sim_depalletize'
        
        try:
            # 加载数据集以获取特征定义
            dataset = LeRobotDataset(repo_id=0, root=lerobot_dataset_path)
            self.dataset_stats = dataset.meta.stats if hasattr(dataset.meta, 'stats') else None
            
            # 从数据集获取特征定义（LeRobot 格式，带 "observation." 前缀）
            # dataset.meta.features 包含完整的数据集特征定义
            if hasattr(dataset.meta, 'features') and dataset.meta.features:
                # 只保留 observation 相关的特征
                lerobot_features = {
                    k: v for k, v in dataset.meta.features.items() 
                    if k.startswith("observation.")
                }
                
                # 重要：如果状态特征的 names 字段存在，我们需要确保它们使用标准状态名称
                # 数据集可能使用了旧的命名格式（如 {"motors": [...]}），我们需要将其转换为列表
                from lerobot.utils.constants import OBS_STR
                state_key = f"{OBS_STR}.state"
                if state_key in lerobot_features and "names" in lerobot_features[state_key]:
                    # 获取数据集中的状态名称
                    dataset_state_names_raw = lerobot_features[state_key]["names"]
                    
                    # 处理 names 字段：可能是列表或字典（旧格式）
                    if isinstance(dataset_state_names_raw, dict):
                        # 如果是字典格式（如 {"motors": [...]}），提取所有值并扁平化
                        dataset_state_names = []
                        for key, value_list in dataset_state_names_raw.items():
                            if isinstance(value_list, list):
                                dataset_state_names.extend(value_list)
                            else:
                                dataset_state_names.append(value_list)
                        rospy.loginfo(f"Converted dictionary format names to list: {len(dataset_state_names)} names")
                    elif isinstance(dataset_state_names_raw, list):
                        # 如果已经是列表，直接使用
                        dataset_state_names = dataset_state_names_raw
                    else:
                        logger.warning(f"Unexpected names format: {type(dataset_state_names_raw)}. Using standard names.")
                        dataset_state_names = None
                    
                    # 获取标准状态名称
                    from configs.config import get_states_names, STATE_COMPONENTS
                    state_components = STATE_COMPONENTS if STATE_COMPONENTS else ["J_q"]
                    standard_state_names = get_states_names(state_components=state_components)
                    
                    # 如果数据集的状态名称与标准名称不匹配，使用标准名称替换
                    # 这样可以确保客户端发送的键名与服务器端期望的一致
                    if dataset_state_names is None or dataset_state_names != standard_state_names:
                        if dataset_state_names is not None:
                            logger.warning(
                                f"Dataset state names ({len(dataset_state_names)} names) don't match "
                                f"standard names ({len(standard_state_names)} names). "
                                f"Using standard names for compatibility."
                            )
                            logger.debug(f"Dataset names (first 5): {dataset_state_names[:5] if len(dataset_state_names) >= 5 else dataset_state_names}")
                            logger.debug(f"Standard names (first 5): {standard_state_names[:5] if len(standard_state_names) >= 5 else standard_state_names}")
                        else:
                            rospy.loginfo(f"Using standard state names: {len(standard_state_names)} names")
                        
                        # 更新特征定义使用标准名称
                        lerobot_features[state_key]["names"] = standard_state_names
                        lerobot_features[state_key]["shape"] = (len(standard_state_names),)
                    else:
                        # 即使匹配，也要确保 names 是列表格式（不是字典）
                        lerobot_features[state_key]["names"] = dataset_state_names
                
                rospy.loginfo(f"Loaded lerobot_features from dataset: {list(lerobot_features.keys())}")
                return lerobot_features
            
            # 如果数据集没有特征定义，从 policy 配置构建
            logger.warning("Dataset does not have features, building from policy config...")
            policy = GrootPolicy.from_pretrained(Path(self.ckpt_path), strict=False)
            
            # 构建硬件特征映射（用于 hw_to_dataset_features）
            from lerobot.datasets.utils import hw_to_dataset_features
            from lerobot.utils.constants import OBS_STR
            
            hw_features = {}
            
            # 添加状态特征（硬件格式）
            # 使用配置文件中的标准状态名称函数
            from configs.config import get_states_names
            
            state_components = STATE_COMPONENTS if STATE_COMPONENTS else ["J_q"]
            # 使用 get_states_names 获取标准状态名称列表
            state_names = get_states_names(state_components=state_components)
            state_dim = len(state_names)
            
            # 对于 hw_to_dataset_features，我们需要将状态作为多个单独的 float 特征
            # 但实际上，build_dataset_frame 期望 state 特征有 names 字段
            # 所以我们需要手动构建特征定义
            camera_names = get_camera_names(CAMERA_COMPONENTS)
            
            # 手动构建 LeRobot 特征格式
            lerobot_features = {}
            
            # 添加状态特征
            lerobot_features[f"{OBS_STR}.state"] = {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": state_names,
            }
            
            # 导入相机键名映射
            from configs.config import CAMERA_KEY_MAPPING
            
            # 添加相机特征
            for camera_name in camera_names:
                # 将相机名称映射为基础名称（cam_head, cam_chest等）
                # 数据集特征键名使用基础名称：observation.images.cam_head
                cam_base_name = CAMERA_KEY_MAPPING.get(camera_name, f"cam_{camera_name}")
                
                # 从 policy config 获取图像形状
                # 注意：policy.config.image_features 中的键可能是 observation.images.cam_head 格式
                # 我们需要检查两种可能的键名格式
                img_feat = None
                if hasattr(policy.config, 'image_features'):
                    # 先尝试直接使用 camera_name（向后兼容）
                    if camera_name in policy.config.image_features:
                        img_feat = policy.config.image_features[camera_name]
                    # 再尝试使用 observation.images.{base_name} 格式
                    elif f"observation.images.{cam_base_name}" in policy.config.image_features:
                        img_feat = policy.config.image_features[f"observation.images.{cam_base_name}"]
                
                if img_feat is not None:
                    # PolicyFeature.shape 是 (C, H, W)
                    h, w, c = img_feat.shape[1], img_feat.shape[2], img_feat.shape[0]
                else:
                    h, w, c = 224, 224, 3
                
                lerobot_features[f"{OBS_STR}.images.{cam_base_name}"] = {
                    "dtype": "image",
                    "shape": (h, w, c),
                    "names": ["height", "width", "channels"],
                }
            
            rospy.loginfo(f"Built lerobot_features from policy: {list(lerobot_features.keys())}")
            return lerobot_features
            
        except Exception as e:
            logger.warning(f"Could not load dataset: {e}. Using default features.")
            import traceback
            traceback.print_exc()
            # 返回默认特征映射（LeRobot 格式）
            from lerobot.utils.constants import OBS_STR
            return {
                f"{OBS_STR}.state": {
                    "dtype": "float32",
                    "shape": (16,),
                    "names": [f"state_{i}" for i in range(16)],
                },
                f"{OBS_STR}.images.image": {
                    "dtype": "image",
                    "shape": (224, 224, 3),
                    "names": ["height", "width", "channels"],
                },
            }
    
    def _convert_obs_to_lerobot_format(self, obs_data: dict) -> RawObservation:
        """将 GrabBoxMpcEnv 的观测转换为 LeRobot 格式
        
        注意：
        1. build_dataset_frame 期望状态被分解为单独的组件（根据 lerobot_features 中的 names）
        2. 对于图像，build_dataset_frame 会从键名去掉 "observation.images." 前缀来查找 values 中的键
           例如：特征键是 "observation.images.cam_head"，会从 values["cam_head"] 中获取值
           所以我们需要将相机名称映射为基础名称（cam_head, cam_chest等）
        3. 状态键名必须与 lerobot_features 中 "observation.state" 的 "names" 字段完全匹配
        """
        raw_obs = {}
        
        # 导入相机键名映射
        from configs.config import CAMERA_KEY_MAPPING
        
        # 转换相机图像
        camera_names = get_camera_names(CAMERA_COMPONENTS)
        for camera_name in camera_names:
            if camera_name in obs_data:
                # obs_data 中的图像格式是 (T, H, W, C)
                # 转换为 numpy 并取最后一帧
                img_np = obs_data[camera_name]
                if img_np.ndim == 4:
                    # 取最后一帧: (T, H, W, C) -> (H, W, C)
                    img_np = img_np[-1]
                # 保持为 numpy array，格式: (H, W, C)
                
                # 如果启用头部相机旋转且当前是头部相机（image），则旋转180度
                if self.rotate_head_camera and camera_name == "image":
                    # 旋转180度：使用np.rot90，k=2表示旋转180度，axes=(0,1)表示在H和W维度上旋转
                    # img_np shape: (H, W, C)
                    # 注意：np.rot90可能产生负步长的视图，需要copy()来创建连续数组
                    img_np = np.rot90(img_np, k=2, axes=(0, 1)).copy()
                
                # 将相机名称映射为基础名称（cam_head, cam_chest等）
                # build_dataset_frame 会从 "observation.images.{base_name}" 去掉前缀，查找 values[base_name]
                cam_base_name = CAMERA_KEY_MAPPING.get(camera_name, f"cam_{camera_name}")
                raw_obs[cam_base_name] = img_np.astype(np.float32) if isinstance(img_np, np.ndarray) else np.array(img_np, dtype=np.float32)
        
        # 转换状态：需要分解为单独组件
        # 重要：使用与 lerobot_features 中完全相同的状态名称列表
        if "state" in obs_data:
            state_np = obs_data["state"]
            if state_np.ndim == 2:
                # 取最后一帧: (T, D) -> (D,)
                state_np = state_np[-1]
            
            # 从 lerobot_features 获取状态名称列表（与服务器端使用的完全一致）
            from lerobot.utils.constants import OBS_STR
            state_key = f"{OBS_STR}.state"
            
            if state_key in self.lerobot_features and "names" in self.lerobot_features[state_key]:
                # 使用特征定义中的状态名称（这是服务器端期望的）
                state_names_raw = self.lerobot_features[state_key]["names"]
                
                # 处理 names 字段：可能是列表或字典（旧格式）
                if isinstance(state_names_raw, dict):
                    # 如果是字典格式（如 {"motors": [...]}），提取所有值并扁平化
                    state_names = []
                    for key, value_list in state_names_raw.items():
                        if isinstance(value_list, list):
                            state_names.extend(value_list)
                        else:
                            state_names.append(value_list)
                    logger.debug(f"Converted dictionary format names to list: {len(state_names)} names")
                elif isinstance(state_names_raw, list):
                    # 如果已经是列表，直接使用
                    state_names = state_names_raw
                else:
                    # 如果不是列表也不是字典，使用标准名称
                    logger.warning(f"Unexpected names format: {type(state_names_raw)}. Using fallback.")
                    from configs.config import get_states_names
                    state_components = STATE_COMPONENTS if STATE_COMPONENTS else ["J_q"]
                    state_names = get_states_names(state_components=state_components)
            else:
                # 如果没有特征定义，使用配置文件中的标准状态名称
                from configs.config import get_states_names
                state_components = STATE_COMPONENTS if STATE_COMPONENTS else ["J_q"]
                state_names = get_states_names(state_components=state_components)
                logger.warning("Using fallback state names from config")
            
            # 将状态 tensor 分解为单独组件，使用特征定义中的状态名称
            for idx, state_name in enumerate(state_names):
                if idx < len(state_np):
                    raw_obs[state_name] = float(state_np[idx])
                else:
                    # 如果状态维度不足，使用 0.0 填充
                    logger.warning(f"State dimension mismatch: expected {len(state_names)}, got {len(state_np)}. Padding with 0.0")
                    raw_obs[state_name] = 0.0
            
            # 如果还有剩余的状态值，添加为通用状态组件（这种情况不应该发生，但作为保护）
            if len(state_np) > len(state_names):
                logger.warning(
                    f"State has more dimensions ({len(state_np)}) than expected ({len(state_names)}). "
                    f"Extra dimensions will be ignored."
                )
        
        # 添加任务描述
        raw_obs["task"] = self.task_description
        
        return raw_obs
    
    def _convert_action_to_robot_format(self, action_tensor: torch.Tensor) -> np.ndarray:
        """将模型输出的动作转换为机器人格式
        
        注意：服务器端的 postprocessor 已经进行了反归一化，
        所以这里只需要转换为 numpy array 即可，不需要再次反归一化
        """
        # action_tensor 是 (action_dim,) 的 tensor
        # 服务器端已经完成了反归一化，直接转换为 numpy
        action_np = action_tensor.cpu().numpy()
        
        return action_np
    
    def receive_actions(self):
        """接收动作线程"""
        rospy.loginfo("Action receiving thread started")
        
        while not self.shutdown_event.is_set():
            try:
                # 从服务器获取动作块
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    time.sleep(0.01)  # 避免 busy wait
                    continue
                
                # 反序列化动作
                timed_actions = pickle.loads(actions_chunk.data)
                
                if len(timed_actions) == 0:
                    continue
                
                # 更新动作块大小
                self.action_chunk_size_received = max(
                    self.action_chunk_size_received, 
                    len(timed_actions)
                )
                
                # 将动作添加到队列
                added_count = 0
                skipped_count = 0
                action_timesteps_in_chunk = [a.get_timestep() for a in timed_actions]
                
                # 获取当前 latest_action（用于过期检查）
                with self.latest_action_lock:
                    current_latest = self.latest_action
                
                # 检查动作块的时间步范围
                if len(timed_actions) > 0:
                    chunk_first_ts = action_timesteps_in_chunk[0]
                    chunk_last_ts = action_timesteps_in_chunk[-1]
                    
                    # 如果整个动作块都已过期（所有时间步 <= latest_action），跳过整个块
                    if chunk_last_ts <= current_latest:
                        rospy.logwarn(
                            f"[CLIENT] ⏭️  Skipping entire action chunk (expired) | "
                            f"Chunk timesteps: {chunk_first_ts}..{chunk_last_ts} | "
                            f"Latest action: {current_latest}"
                        )
                        # 但仍然设置 must_go，以便发送新观测
                        self.must_go.set()
                        continue
                    
                    # 检查动作块是否部分过期（第一个动作 <= latest_action，但最后一个 > latest_action）
                    # 这种情况下，我们应该跳过整个块，因为动作块必须连续
                    if chunk_first_ts <= current_latest:
                        rospy.logwarn(
                            f"[CLIENT] ⏭️  Skipping partially expired action chunk | "
                            f"Chunk timesteps: {chunk_first_ts}..{chunk_last_ts} | "
                            f"Latest action: {current_latest} | "
                            f"This would break continuity!"
                        )
                        # 跳过整个块，但仍然设置 must_go
                        self.must_go.set()
                        continue
                
                # 所有动作都未过期，添加到队列
                with self.action_queue_lock:
                    for action in timed_actions:
                        action_ts = action.get_timestep()
                        # 双重检查：确保动作未过期（虽然上面已经检查过了，但这是安全措施）
                        if action_ts <= current_latest:
                            skipped_count += 1
                            rospy.logwarn(
                                f"[CLIENT] ⚠️  Action timestep {action_ts} <= latest_action {current_latest}, "
                                f"but chunk passed initial check. This should not happen!"
                            )
                            continue
                        
                        # 转换动作格式
                        action_np = self._convert_action_to_robot_format(action.get_action())
                        self.action_queue.put((action_ts, action_np))
                        added_count += 1
                
                # 设置 must_go 标志，表示可以发送新观测
                self.must_go.set()
                
                if len(timed_actions) > 0:
                    chunk_first_ts = action_timesteps_in_chunk[0]
                    chunk_last_ts = action_timesteps_in_chunk[-1]
                    rospy.loginfo(
                        f"[CLIENT] 📥 Actions received from server | "
                        f"Total: {len(timed_actions)} | "
                        f"Added: {added_count} | "
                        f"Skipped: {skipped_count} | "
                        f"Timesteps: {chunk_first_ts}..{chunk_last_ts} | "
                        f"Queue size: {self.action_queue.qsize()}"
                    )
                
            except grpc.RpcError as e:
                logger.error(f"Error receiving actions: {e}")
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Unexpected error in receive_actions: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def _ready_to_send_observation(self) -> bool:
        """检查是否应该发送新观测
        
        参考标准 RobotClient 的实现，只要队列大小小于阈值就可以发送观测。
        不需要等待队列完全为空，这样可以保持观测的及时性。
        """
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            threshold = self.action_chunk_size_received * self.chunk_size_threshold
            return queue_size < threshold
    
    def send_observation(self, obs_data: dict) -> tuple[bool, int]:
        """发送观测到服务器
        
        Returns:
            (success: bool, timestep: int): 是否成功发送，以及发送的观测时间步
        """
        if not self._ready_to_send_observation():
            return (False, -1)
        
        # 检查 must_go 标志（确保之前发送的观测已被处理）
        # 注意：不需要队列完全为空，只要队列大小小于阈值即可发送新观测
        if not self.must_go.is_set():
            return (False, -1)
        
        try:
            # 转换观测格式
            raw_obs = self._convert_obs_to_lerobot_format(obs_data)
            
            # 创建 TimedObservation
            # 重要：timestep 应该是下一个预期的时间步（latest_action + 1）
            # 服务器会基于这个观测生成从 timestep 开始的连续动作块
            with self.latest_action_lock:
                # 使用 latest_action + 1 作为新观测的时间步
                # 如果 latest_action 是 -1（初始状态），则使用 0
                timestep = max(self.latest_action + 1, 0)
            
            timed_obs = TimedObservation(
                timestamp=time.time(),
                observation=raw_obs,
                timestep=timestep,
            )
            
            # 设置 must_go 标志
            timed_obs.must_go = True
            
            # 序列化并发送
            obs_bytes = pickle.dumps(timed_obs)
            observation_iterator = send_bytes_in_chunks(
                obs_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            self.stub.SendObservations(observation_iterator)
            
            # 清除 must_go 标志（将在收到动作后重新设置）
            self.must_go.clear()
            
            logger.debug(f"Sent observation #{timestep}")
            return (True, timestep)
            
        except grpc.RpcError as e:
            logger.error(f"Error sending observation: {e}")
            return (False, -1)
    
    def get_action_from_queue(self) -> Optional[np.ndarray]:
        """从队列获取动作（不更新 latest_action）
        
        注意：此方法只从队列取出动作，不更新 latest_action。
        latest_action 应该在动作块构建完成后统一更新。
        """
        try:
            with self.action_queue_lock:
                if self.action_queue.empty():
                    return None
                
                timestep, action = self.action_queue.get_nowait()
                # 返回动作和时间步的元组，让调用者决定如何处理时间步
                return (timestep, action)
        except Empty:
            return None
    
    def update_latest_action_timestep(self, timestep: int):
        """更新 latest_action 时间步"""
        with self.latest_action_lock:
            self.latest_action = max(self.latest_action, timestep)
    
    def run(self):
        """运行异步推理循环"""
        rospy.loginfo("Starting async inference loop...")
        
        # 启动动作接收线程
        action_receiver_thread = threading.Thread(target=self.receive_actions, daemon=True)
        action_receiver_thread.start()
        
        # 等待线程启动
        time.sleep(1)
        
        # 初始化机器人控制
        if ROBOT_VERSION == "4_pro":
            change_arm_ctrl_mode(2)  # 启用外部控制
            direct_to_wbc(1)
            function_key = "direct_to_wbc"
        elif ROBOT_VERSION == "5_wheel":
            change_arm_ctrl_mode(2)  # 启用外部控制
            set_arm_quick_mode(True)
            function_key = "set_arm_quick_mode"
        
        input(f"当前机器人模式为: {ROBOT_VERSION} | 控制模式 {function_key} 结束, 按回车继续 ==== \n")
        time.sleep(1.0)
        
        # 回放初始轨迹
        init_traj_bag_path = '/home/lab/kuavo-manip/robot_depalletize_init_traj.bag'
        if os.path.exists(init_traj_bag_path):
            rospy.loginfo("Loading and replaying initial trajectory...")
            load_and_replay_init_trajectory(
                bag_path=init_traj_bag_path,
                env=self.env,
                control_arm=self.control_arm,
                control_claw=self.control_claw
            )
            time.sleep(1.0)
        
        input(f"轨迹回放结束, 按回车继续 ==== \n")
        rospy.loginfo("=" * 80)
        rospy.loginfo("🚀 Starting async inference control loop...")
        rospy.loginfo(f"   Server address: {self.server_address}")
        rospy.loginfo(f"   Control frequency: {self.fps} Hz (dt={self.environment_dt:.4f}s)")
        rospy.loginfo(f"   Action chunk size: {self.action_chunk_size}")
        rospy.loginfo(f"   Task: {self.task_description}")
        rospy.loginfo(f"   Control arm: {self.control_arm}, Control claw: {self.control_claw}")
        rospy.loginfo("=" * 80)
        time.sleep(1.0)
        
        # 用于动作重采样和低通滤波
        resampled_action_queue: deque = deque()
        last_executed_action: Optional[np.ndarray] = None
        
        step_counter = 0
        last_status_log_time = time.time()
        status_log_interval = 2.0  # 每2秒打印一次状态信息
        last_obs_sent_time = 0
        last_chunk_received_time = 0
        
        # 检查动作接收线程状态
        with self.action_queue_lock:
            initial_queue_size = self.action_queue.qsize()
        rospy.loginfo(f"[CLIENT] Action receiver thread status: Running | Initial queue size: {initial_queue_size}")
        
        try:
            rospy.loginfo("[CLIENT] Entering main control loop...")
            loop_iteration = 0
            while not self.shutdown_event.is_set():
                loop_start = time.perf_counter()
                loop_iteration += 1
                
                # 在第一次迭代时打印详细信息
                if loop_iteration == 1:
                    rospy.loginfo("[CLIENT] 🔄 First loop iteration started")
                    with self.action_queue_lock:
                        queue_size = self.action_queue.qsize()
                    with self.latest_action_lock:
                        latest_timestep = self.latest_action
                    rospy.loginfo(
                        f"[CLIENT] Initial state | Queue size: {queue_size} | "
                        f"Latest timestep: {latest_timestep} | "
                        f"Must go: {self.must_go.is_set()}"
                    )
                
                # 1. 获取观测
                obs_data, camera_obs, camera_obs_ts, robot_obs, robot_obs_ts = self.env.get_obs()
                
                # 在第一次成功获取观测时打印
                if loop_iteration == 1:
                    state_shape_str = 'N/A'
                    if 'state' in obs_data:
                        state_data = obs_data['state']
                        if hasattr(state_data, 'shape'):
                            state_shape_str = str(state_data.shape)
                        else:
                            state_shape_str = str(type(state_data))
                    rospy.loginfo(
                        f"[CLIENT] ✅ First observation obtained | "
                        f"Cameras: {list(camera_obs.keys()) if camera_obs else 'None'} | "
                        f"State shape: {state_shape_str}"
                    )
                
                # 2. 发送观测（如果需要）
                obs_sent, sent_timestep = self.send_observation(obs_data)
                if obs_sent:
                    last_obs_sent_time = time.time()
                    with self.latest_action_lock:
                        latest_action_ts = self.latest_action
                    rospy.loginfo(
                        f"[CLIENT] ✅ Observation sent | Timestep: {sent_timestep} | "
                        f"Queue size: {self.action_queue.qsize()} | "
                        f"Resampled queue: {len(resampled_action_queue)} | "
                        f"Latest action timestep: {latest_action_ts}"
                    )
                
                # 3. 获取动作（从队列或重采样队列）
                current_action = None
                
                if len(resampled_action_queue) == 0:
                    # 从服务器动作队列获取新动作块
                    # 重要：需要按时间步顺序取出动作，确保连续性
                    action_chunk = []
                    chunk_timesteps = []
                    
                    with self.action_queue_lock:
                        server_queue_size = self.action_queue.qsize()
                    
                    # 从队列中按顺序取出动作，构建动作块
                    # 注意：动作队列应该是按时间步顺序的（FIFO）
                    with self.latest_action_lock:
                        expected_first_timestep = self.latest_action + 1
                    
                    while len(action_chunk) < self.action_chunk_size:
                        result = self.get_action_from_queue()
                        if result is None:
                            break
                        timestep, action = result
                        action_chunk.append(action)
                        chunk_timesteps.append(timestep)
                    
                    if len(action_chunk) > 0:
                        chunk_first_timestep = chunk_timesteps[0]
                        chunk_last_timestep = chunk_timesteps[-1]
                        
                        # 检查时间步连续性
                        is_continuous = True
                        gap_info = None
                        if len(chunk_timesteps) > 1:
                            for i in range(1, len(chunk_timesteps)):
                                if chunk_timesteps[i] != chunk_timesteps[i-1] + 1:
                                    is_continuous = False
                                    gap_info = f"Gap at index {i}: {chunk_timesteps[i-1]} -> {chunk_timesteps[i]} (expected: {chunk_timesteps[i-1] + 1})"
                                    rospy.logwarn(
                                        f"[CLIENT] ⚠️  NON-CONTINUOUS timesteps detected! | {gap_info}"
                                    )
                                    break
                        
                        # 检查是否与预期时间步对齐
                        timestep_aligned = (chunk_first_timestep == expected_first_timestep) if expected_first_timestep >= 0 else True
                        if not timestep_aligned:
                            rospy.logwarn(
                                f"[CLIENT] ⚠️  Timestep misalignment! | "
                                f"Expected first: {expected_first_timestep} | "
                                f"Actual first: {chunk_first_timestep} | "
                                f"Gap: {chunk_first_timestep - expected_first_timestep}"
                            )
                        
                        # 更新 latest_action 为动作块中最后一个动作的时间步
                        # 重要：只有在确认动作块连续且对齐后才更新
                        if is_continuous and timestep_aligned:
                            self.update_latest_action_timestep(chunk_last_timestep)
                        else:
                            # 即使不连续，也要更新，避免卡死
                            # 但记录警告
                            rospy.logwarn(
                                f"[CLIENT] ⚠️  Updating latest_action despite discontinuity: {chunk_last_timestep}"
                            )
                            self.update_latest_action_timestep(chunk_last_timestep)
                        
                        last_chunk_received_time = time.time()
                        status_icon = "✅" if (is_continuous and timestep_aligned) else "⚠️"
                        rospy.loginfo(
                            f"[CLIENT] 📦 Action chunk received {status_icon} | "
                            f"Size: {len(action_chunk)} | "
                            f"Timesteps: {chunk_first_timestep}..{chunk_last_timestep} | "
                            f"Continuous: {is_continuous} | "
                            f"Aligned: {timestep_aligned} | "
                            f"Expected first: {expected_first_timestep} | "
                            f"Server queue (before): {server_queue_size} | "
                            f"Server queue (after): {self.action_queue.qsize()}"
                        )
                        
                        # 如果时间步不连续或不对齐，打印详细信息用于调试
                        if not is_continuous or not timestep_aligned:
                            rospy.logwarn(
                                f"[CLIENT] ⚠️  Timestep sequence (first 20): {chunk_timesteps[:20]} | "
                                f"Full sequence length: {len(chunk_timesteps)}"
                            )
                            # 检查队列中剩余动作的时间步（用于调试）
                            with self.action_queue_lock:
                                remaining_timesteps = []
                                temp_queue = []
                                while not self.action_queue.empty():
                                    ts, act = self.action_queue.get_nowait()
                                    remaining_timesteps.append(ts)
                                    temp_queue.append((ts, act))
                                # 重新放回队列
                                for ts, act in temp_queue:
                                    self.action_queue.put((ts, act))
                                if remaining_timesteps:
                                    rospy.logwarn(
                                        f"[CLIENT] ⚠️  Remaining queue timesteps (first 10): {remaining_timesteps[:10]}"
                                    )
                        # 转换为 numpy array
                        action_chunk = np.array(action_chunk)  # (chunk_size, action_dim)
                        
                        # 重采样到控制频率
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
                        
                        resampled_chunk = resample_chunk_with_claw_hold(
                            action_chunk,
                            previous_action=last_executed_action,
                            control_frequency=self.env.control_frequency,
                            source_dt=MODEL_ACTION_DT,
                            arm_dims=arm_dims,
                            claw_dims=claw_dims
                        )
                        
                        # 应用低通滤波
                        if ENABLE_CHUNK_TRANSITION_LOWPASS and last_executed_action is not None:
                            transition_steps = max(
                                1,
                                int(round(self.env.control_frequency * CHUNK_TRANSITION_DURATION_S))
                            )
                            resampled_chunk = apply_lowpass_transition(
                                resampled_chunk,
                                previous_action=last_executed_action,
                                alpha=LOWPASS_ALPHA,
                                transition_steps=transition_steps,
                                smooth_slice=arm_dims
                            )
                        
                        # 发布关节位置（用于可视化）
                        publish_joint_positions(
                            resampled_chunk,
                            self.joint_pub,
                            source_frequency_hz=self.env.control_frequency,
                            target_frequency_hz=None
                        )
                        
                        # 添加到重采样队列
                        resampled_action_queue.extend(resampled_chunk)
                        rospy.loginfo(
                            f"[CLIENT] 🔄 Action chunk resampled | "
                            f"Original: {len(action_chunk)} | "
                            f"Resampled: {len(resampled_chunk)} | "
                            f"Resampled queue size: {len(resampled_action_queue)}"
                        )
                
                # 4. 从重采样队列获取当前动作
                if len(resampled_action_queue) > 0:
                    current_action = resampled_action_queue.popleft()
                
                # 5. 执行动作
                if current_action is not None:
                    control_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
                    self.env.exec_actions(
                        actions=current_action,
                        control_arm=self.control_arm,
                        control_claw=self.control_claw,
                        control_cmd_pose=control_cmd_pose
                    )
                    last_executed_action = current_action.copy()
                    step_counter += 1
                else:
                    # 如果没有动作可执行，记录警告
                    if step_counter % 100 == 0:  # 每100步打印一次，避免日志过多
                        logger.warning(
                            f"[CLIENT] ⚠️  No action available | "
                            f"Server queue: {self.action_queue.qsize()} | "
                            f"Resampled queue: {len(resampled_action_queue)}"
                        )
                
                # 6. 定期打印状态信息
                current_time = time.time()
                if current_time - last_status_log_time >= status_log_interval:
                    with self.action_queue_lock:
                        server_queue_size = self.action_queue.qsize()
                    with self.latest_action_lock:
                        latest_timestep = self.latest_action
                    
                    time_since_last_obs = current_time - last_obs_sent_time if last_obs_sent_time > 0 else 0
                    time_since_last_chunk = current_time - last_chunk_received_time if last_chunk_received_time > 0 else 0
                    
                    rospy.loginfo(
                        f"[CLIENT] 📊 Status Summary | "
                        f"Steps: {step_counter} | "
                        f"Latest timestep: {latest_timestep} | "
                        f"Server queue: {server_queue_size}/{self.action_chunk_size_received} | "
                        f"Resampled queue: {len(resampled_action_queue)} | "
                        f"Time since last obs: {time_since_last_obs:.2f}s | "
                        f"Time since last chunk: {time_since_last_chunk:.2f}s"
                    )
                    last_status_log_time = current_time
                
                # 7. 控制循环频率
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0, self.environment_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # 记录循环时间过长的情况
                if elapsed > self.environment_dt * 1.5:  # 超过目标时间的1.5倍
                    logger.warning(
                        f"[CLIENT] ⚠️  Loop time exceeded | "
                        f"Elapsed: {elapsed*1000:.2f}ms | "
                        f"Target: {self.environment_dt*1000:.2f}ms"
                    )
                
        except KeyboardInterrupt:
            rospy.loginfo("Interrupted by user")
        finally:
            self.shutdown_event.set()
            self.channel.close()
            rospy.loginfo("Async client stopped")


def main():
    parser = argparse.ArgumentParser(description='GROOT Async Inference Client for Depalletize Task')
    parser.add_argument('--server_address', type=str, default='127.0.0.1:8080',
                        help='gRPC server address (host:port)')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to GROOT checkpoint')
    parser.add_argument('--action_chunk_size', type=int, default=20,
                        help='Action chunk size')
    parser.add_argument('--lerobot_dataset_path', type=str, default=None,
                        help='Path to LeRobot dataset (for statistics)')
    parser.add_argument('--task_description', type=str, default='Depalletize the box',
                        help='Task description')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Control frequency (FPS)')
    parser.add_argument('--chunk_size_threshold', type=float, default=0.5,
                        help='Threshold for sending new observations (queue_size/chunk_size < threshold)')
    parser.add_argument('--rotate-head-camera', action='store_true',
                        help='If set, rotate head camera images (image) by 180 degrees.')
    
    args = parser.parse_args()
    
    # 初始化 ROS 节点
    if not rospy.get_node_uri():
        rospy.init_node('groot_async_client', anonymous=True)
    
    # 创建客户端并运行
    client = GrootAsyncClient(
        server_address=args.server_address,
        ckpt_path=args.ckpt_path,
        action_chunk_size=args.action_chunk_size,
        lerobot_dataset_path=args.lerobot_dataset_path,
        task_description=args.task_description,
        fps=args.fps,
        chunk_size_threshold=args.chunk_size_threshold,
        rotate_head_camera=args.rotate_head_camera,
    )
    
    client.run()


if __name__ == '__main__':
    main()

