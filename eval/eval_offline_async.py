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
       --fps=10 \
       --inference_latency=0.033

2. 运行客户端（本脚本）：
    python eval/eval_offline_async.py \
        --server_address=127.0.0.1:8080 \
        --ckpt_path=/path/to/checkpoint \
        --action_chunk_size=16 \
        --lerobot_dataset_path=/path/to/dataset \
        --task_description="Depalletize the box" 
        --fps=10 \
        --chunk_size_threshold=0.5
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from collections.abc import Callable
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

from kuavo_env.kuavo_depalletize_env import GrabBoxMpcEnv
# from viz.visualizer import RerunVisualizer
from configs.config import topic_info, \
    get_camera_observation_key, get_camera_names, \
    TASK_DATA_MODE, CAMERA_COMPONENTS, ACTION_COMPONENTS, ROBOT_VERSION, STATE_COMPONENTS
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
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

from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.robot_sdk import RobotSDK

# 导入必要的工具函数
from eval import (
    resample_chunk_with_claw_hold,
    apply_lowpass_transition,
    publish_joint_positions,
    final_reset_arm,
)

from tqdm import tqdm

MODEL_ACTION_DT = 0.1  # seconds between predicted actions during training
MODEL_ACTION_FREQUENCY = 1.0 / MODEL_ACTION_DT
TARGET_CONTROL_FREQUENCY = 100.0
TARGET_CONTROL_DT = 1.0 / TARGET_CONTROL_FREQUENCY
CHUNK_TRANSITION_DURATION_S = 0.2  # seconds of low-pass smoothing at chunk boundary
LOWPASS_ALPHA = 0.85  # closer to 1 => smoother (slower) transitions
ENABLE_CHUNK_TRANSITION_LOWPASS = True

logger = get_logger("groot_async_client")

AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]

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
        self.robot_sdk = RobotSDK()
        self.env = GrabBoxMpcEnv()
        # self.env.obs_buffer.wait_buffer_ready()
        time.sleep(1)
        rospy.loginfo("Environment ready")
        
        # 加载数据集统计信息（用于特征映射）
        self.dataset_stats = None
        self.lerobot_dataset_path = lerobot_dataset_path
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
        self.aggregate_fn = get_aggregate_function("weighted_average")
        
        # 线程控制
        self.shutdown_event = threading.Event()
        self.must_go = threading.Event()
        self.must_go.set()  # 初始设置为可发送观测
        
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
            lerobot_dataset_path = '/home/wuzr/il/kuavo-manip/benchmark/5w_four_box_random'
        
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

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            # timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
            timestamps = sorted([action[0] for action in self.action_queue.queue])
        rospy.loginfo(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps
    
    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        # current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}
        current_action_queue = {action[0]: action[1] for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                action_np = self._convert_action_to_robot_format(new_action.get_action())
                future_action_queue.put((new_action.get_timestep(), action_np))
                continue

            aggregated_action = aggregate_fn(
                current_action_queue[new_action.get_timestep()], new_action.get_action().cpu().numpy()
            )
            
            future_action_queue.put((new_action.get_timestep(), aggregated_action))

        with self.action_queue_lock:
            self.action_queue = future_action_queue
    
    def receive_actions(self):
        """接收动作线程"""
        rospy.loginfo("Action receiving thread started")
        
        while not self.shutdown_event.is_set():
            try:
                # 从服务器获取动作块
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                receive_time = time.time()
                
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)
                deserialize_time = time.perf_counter() - deserialize_start

                self.action_chunk_size_received = max(
                    self.action_chunk_size_received, 
                    len(timed_actions)
                )

                if len(timed_actions) > 0:
                    with self.latest_action_lock:
                        latest_action = self.latest_action
                
                    rospy.loginfo(f"Current latest action: {latest_action}")


                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]

                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    rospy.loginfo(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()

                # Get queue state after changes
                new_size, new_timesteps = self._inspect_action_queue()

                with self.latest_action_lock:
                    latest_action = self.latest_action

                rospy.loginfo(
                    f"Latest action: {latest_action} | "
                    f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                    f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                    # f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    f"Updated action steps: {new_timesteps}"
                )
                rospy.loginfo(
                    f"Queue update complete ({queue_update_time:.6f}s) | "
                    f"Before: {old_size} items | "
                    f"After: {new_size} items | "
                )

            except grpc.RpcError as e:
                rospy.logerr(f"Error receiving actions: {e}")

    def _ready_to_send_observation(self, resampled_action_queue: deque) -> bool:
        """检查是否应该发送新观测"""
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            executed_size = len(resampled_action_queue)
            # threshold = self.action_chunk_size_received * 10 * self.chunk_size_threshold
            threshold = 100
            return (queue_size < self.action_chunk_size_received) and (executed_size == threshold or executed_size == 0)
    
    def send_observation(self, obs_buffer: deque, resampled_action_queue: deque) -> bool:
        """发送观测到服务器"""
        if not self._ready_to_send_observation(resampled_action_queue):
            return False
        
        # 检查 must_go 标志
        with self.action_queue_lock:
            queue_empty = self.action_queue.empty() and len(obs_buffer) == 0

        if not self.must_go.is_set():
            return False
        
        try:
            if len(obs_buffer) != 0:
                raw_obs = self._convert_obs_to_lerobot_format(obs_buffer.popleft())
            else:
                raw_obs = {}
            for _ in range(8):
                if len(obs_buffer) != 0:
                    obs_buffer.popleft()
                else:
                    break
            
            # 创建 TimedObservation
            with self.latest_action_lock:
                timestep = max(self.latest_action, 0)
            
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
            return True
            
        except grpc.RpcError as e:
            logger.error(f"Error sending observation: {e}")
            return False
    
    def get_action_from_queue(self) -> Optional[np.ndarray]:
        """从队列获取动作"""
        try:
            with self.action_queue_lock:
                if self.action_queue.empty():
                    return None
                
                timestep, action = self.action_queue.get_nowait()
                
                with self.latest_action_lock:
                    self.latest_action = timestep
                
                return action
        except Empty:
            return None
    
    def _prepare_obs_buffer(self, dataloader, dataset):
        obs_buffer = deque(maxlen=99)
        iterator = tqdm(enumerate(dataloader), total=dataset.num_frames, desc="Processing")

        for data_step, batch in iterator:
            obs_data = {
                'state': batch['observation.state'],
            }

            camera_names = get_camera_names(CAMERA_COMPONENTS)
            for camera_name in camera_names:
                obs_key = get_camera_observation_key(camera_name, use_image_features=False)
                if obs_key in batch:
                    obs_data[camera_name] = (batch[obs_key][0].cpu().numpy().astype(np.float32) / 255.0).transpose(1, 2, 0)
                else:
                    fallback_key = f"observation.images.{camera_name}"
                    if fallback_key in batch:
                        obs_data[fallback_key] = batch[fallback_key]
                    elif data_step == 0:
                        print(f"[WARN]: Camera observation key '{obs_key}' not found in batch. Available keys: {[k for k in batch.keys() if 'image' in k.lower()]}")

            obs_buffer.append(obs_data)

        return obs_buffer
        
    def run(self):
        """运行异步推理循环"""
        rospy.loginfo("Starting async inference loop...")
        
        # 启动动作接收线程
        action_receiver_thread = threading.Thread(target=self.receive_actions, daemon=True)
        action_receiver_thread.start()
        
        # 等待线程启动
        time.sleep(1)
        self.robot_sdk.control.set_external_control_arm_mode()
        
        # 初始化机器人控制
        if ROBOT_VERSION == "4_pro":
            self.robot_sdk.control.set_direct_to_wbc()
            function_key = "direct_to_wbc"
        elif ROBOT_VERSION == "5_wheel":
            self.robot_sdk.control.set_arm_quick_mode(True)
            function_key = "set_arm_quick_mode"
        
        input(f"当前机器人模式为: {ROBOT_VERSION} | 控制模式 {function_key} 结束, 按回车继续 ==== \n")
        time.sleep(1.0)
        
        # 初始化手臂位置
        self.robot_sdk.control.control_arm_joint_positions([
            -0.13108279410748547, 0.8129094913924698, -0.24350018506614143, -2.417029590409915, 1.3373353902132268, 0.40744714341534344, 0.9697406158372316,
            -0.13080062822719687, -0.8167237074915805, 0.2473180179648636, -2.410544954816824, -1.3350419244819163,-0.40477685656131995, 0.9716470019773075
        ])

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
        
        # ---------------------------------------------------------------

        dataset = LeRobotDataset(repo_id=0, root=self.lerobot_dataset_path, episodes=[0])
        ep_meta = dataset.meta.episodes[0]
        ep_start = ep_meta["dataset_from_index"]
        ep_end = ep_meta["dataset_to_index"]
        dataset.hf_dataset = dataset.hf_dataset.select(range(ep_start, ep_end))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=0,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        first_batch = next(iter(dataloader))
        action_dim = first_batch['action'].shape[1]
        obs_dim = first_batch['observation.state'].shape[1]
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=0,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        obs_buffer = self._prepare_obs_buffer(dataloader, dataset)
        # ---------------------------------------------------------------

        # 检查动作接收线程状态
        with self.action_queue_lock:
            initial_queue_size = self.action_queue.qsize()
        rospy.loginfo(f"[CLIENT] Action receiver thread status: Running | Initial queue size: {initial_queue_size}")
        
        try:
            rospy.loginfo("[CLIENT] Entering main control loop...")
            loop_iteration = 0
            while not self.shutdown_event.is_set():
                loop_iteration += 1
                
                # 在第一次迭代时打印详细信息
                if loop_iteration == 1:
                    rospy.loginfo("[CLIENT] First loop iteration started")
                    with self.action_queue_lock:
                        queue_size = self.action_queue.qsize()
                    with self.latest_action_lock:
                        latest_timestep = self.latest_action
                    rospy.loginfo(
                        f"[CLIENT] Initial state | Queue size: {queue_size} | "
                        f"Latest timestep: {latest_timestep} | "
                        f"Must go: {self.must_go.is_set()}"
                    )
                
                # # 1. 获取观测(online mode)
                # obs_data, camera_obs, camera_obs_ts, robot_obs, robot_obs_ts = self.env.get_obs()

                # 2. 发送观测
                obs_sent = self.send_observation(obs_buffer, resampled_action_queue)
                if obs_sent:
                    last_obs_sent_time = time.time()
                    with self.latest_action_lock:
                        rospy.loginfo(
                            f"[CLIENT] Observation sent | Timestep: {self.latest_action} | "
                            f"Queue size: {self.action_queue.qsize()} | "
                            f"Resampled queue: {len(resampled_action_queue)}"
                        )
                
                # 3. 获取动作（从队列或重采样队列）
                current_action = None
            
                # if len(resampled_action_queue) == 0:
                if (len(resampled_action_queue) < 75):
                    # 从服务器动作队列获取新动作块
                    action_chunk = []
                    with self.action_queue_lock:
                        server_queue_size = self.action_queue.qsize()
                    
                    while len(action_chunk) < self.action_chunk_size / 2:
                        action = self.get_action_from_queue()
                        if action is None:
                            break
                        action_chunk.append(action)
                    
                    if len(action_chunk) > 0:
                        last_chunk_received_time = time.time()
                        rospy.loginfo(
                            f"[CLIENT] Action chunk received | Size: {len(action_chunk)} | "
                            f"Server queue (before): {server_queue_size} | "
                            f"Server queue (after): {self.action_queue.qsize()}"
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
                        
                        # 添加到重采样队列
                        if len(resampled_action_queue) != 0:
                            resampled_action_queue.clear()
                        resampled_action_queue.extend(resampled_chunk)
                        rospy.loginfo(
                            f"[CLIENT] Action chunk resampled | "
                            f"Original: {len(action_chunk)} | "
                            f"Resampled: {len(resampled_chunk)} | "
                            f"Resampled queue size: {len(resampled_action_queue)}"
                        )
            
                # 4. 从重采样队列获取当前动作
                # if len(resampled_action_queue) > 0:
                if len(resampled_action_queue) > 0:
                    current_action = resampled_action_queue.popleft()
                else:
                    continue
            
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
                            f"[CLIENT] No action available | "
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
                        f"[CLIENT] Status Summary | "
                        f"Steps: {step_counter} | "
                        f"Latest timestep: {latest_timestep} | "
                        f"Server queue: {server_queue_size}/{self.action_chunk_size_received} | "
                        f"Resampled queue: {len(resampled_action_queue)} | "
                        f"Time since last obs: {time_since_last_obs:.2f}s | "
                        f"Time since last chunk: {time_since_last_chunk:.2f}s"
                    )
                    last_status_log_time = current_time

            self.shutdown_event.set()
                
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