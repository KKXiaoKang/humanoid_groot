#!/usr/bin/env python3
"""
异步推理数据集评估脚本

这个脚本使用异步推理架构（client-server模式）在LeRobot数据集上评估模型性能。
与eval_on_dataset_lowpass.py不同，这个脚本：
- 使用异步推理架构（policy_server + client）
- 不进行chunk内部线性插值
- 不进行chunk间低通滤波
- 直接使用模型输出的action进行误差计算

使用方法：
1. 启动服务器：
   python -m lerobot.async_inference.policy_server \
       --host=127.0.0.1 \
       --port=8080 \
       --fps=30 \
       --inference_latency=0.033

2. 运行客户端（本脚本）：
   python scripts/eval_async_dataset_base.py \
       --server_address=127.0.0.1:8080 \
       --ckpt_path=/path/to/checkpoint \
       --dataset-root=/path/to/dataset \
       --episode=0 \
       --action-chunk-size=16
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 确保localhost连接绕过代理（gRPC连接不应该经过HTTP代理）
# 设置NO_PROXY环境变量，让localhost和127.0.0.1绕过代理
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")

import time
import threading
import pickle
import argparse
from queue import Queue
from collections import OrderedDict
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import grpc
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.async_inference.helpers import (
    TimedObservation,
    TimedAction,
    RawObservation,
    RemotePolicyConfig,
    get_logger,
)

# 导入配置模块（如果存在）
try:
    from configs.config import (
        topic_info, TASK_DATA_MODE, get_camera_observation_key, 
        get_camera_names, CAMERA_COMPONENTS, action_names, CAMERA_KEY_MAPPING
    )
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
        return f"observation.images.{camera_name}"
    def get_camera_names(camera_components=None):
        return []

# 可选的可视化工具（如果不存在则禁用）
try:
    from scripts.visualization_tools.visualizers import RerunVisualizer, KeyboardManager
    RERUN_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: RerunVisualizer not available. Visualization will be disabled.")
    RERUN_AVAILABLE = False
    RerunVisualizer = None
    KeyboardManager = None

logger = get_logger("async_dataset_eval_client")


class DatasetAsyncClient:
    """异步推理客户端，用于在数据集上评估模型"""
    
    def __init__(
        self,
        server_address: str,
        ckpt_path: str,
        dataset: LeRobotDataset,
        action_chunk_size: int = 16,
        task_description: Optional[str] = None,
        fps: float = 30.0,
        chunk_size_threshold: float = 0.5,
    ):
        """
        Args:
            server_address: gRPC 服务器地址 (格式: "host:port")
            ckpt_path: GROOT 模型 checkpoint 路径
            dataset: LeRobotDataset 实例
            action_chunk_size: 动作块大小
            task_description: 任务描述字符串（如果提供则覆盖数据集中的task）
            fps: 控制频率
            chunk_size_threshold: 动作队列阈值
        """
        self.server_address = server_address
        self.ckpt_path = ckpt_path
        self.dataset = dataset
        self.action_chunk_size = action_chunk_size
        self.task_description = task_description
        self.fps = fps
        self.environment_dt = 1.0 / fps
        self.chunk_size_threshold = chunk_size_threshold
        
        # 获取数据集特征映射（用于服务器端转换）
        self.lerobot_features = self._get_lerobot_features_from_dataset()
        
        # 连接 gRPC 服务器
        logger.info(f"Connecting to policy server at {server_address}...")
        self.channel = grpc.insecure_channel(
            server_address,
            grpc_channel_options(initial_backoff=f"{self.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        
        # 发送 Ready 信号
        try:
            self.stub.Ready(services_pb2.Empty())
            logger.info("Connected to policy server")
        except grpc.RpcError as e:
            logger.error(f"Failed to connect to server: {e}")
            raise
        
        # 发送策略配置
        policy_config = RemotePolicyConfig(
            policy_type="groot",
            pretrained_name_or_path=ckpt_path,
            lerobot_features=self.lerobot_features,
            actions_per_chunk=action_chunk_size,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        policy_config_bytes = pickle.dumps(policy_config)
        policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)
        self.stub.SendPolicyInstructions(policy_setup)
        logger.info("Policy configuration sent to server")
        
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
        
        logger.info("DatasetAsyncClient initialized")
    
    def _get_lerobot_features_from_dataset(self) -> dict[str, dict]:
        """从数据集获取LeRobot特征映射
        
        返回格式: dict[str, dict] - LeRobot 数据集特征格式（带 "observation." 前缀）
        """
        # 从数据集meta中获取特征定义（如果可用）
        features = {}
        
        # 尝试从数据集meta中获取特征定义
        if hasattr(self.dataset, 'meta') and hasattr(self.dataset.meta, 'features'):
            # 如果有meta.features，直接使用，但需要处理字典格式的names
            for key, feat_def in self.dataset.meta.features.items():
                if key.startswith("observation"):
                    # 复制特征定义
                    feat_def_copy = feat_def.copy()
                    
                    # 如果names是字典格式，需要转换为列表格式（build_dataset_frame期望列表）
                    if "names" in feat_def_copy and isinstance(feat_def_copy["names"], dict):
                        # 将字典格式的names扁平化为列表
                        # 例如: {"motors": ["arm_joint_1", ...]} -> ["arm_joint_1", ...]
                        state_names = []
                        for dict_key, value_list in feat_def_copy["names"].items():
                            if isinstance(value_list, list):
                                state_names.extend(value_list)
                            else:
                                state_names.append(value_list)
                        feat_def_copy["names"] = state_names
                        logger.debug(f"Converted dictionary format names to list for {key}: {len(state_names)} names")
                    
                    features[key] = feat_def_copy
            logger.info(f"Loaded lerobot features from dataset meta: {list(features.keys())}")
            return features
        
        # 如果没有meta.features，从数据中推断
        # 添加state特征
        if "observation.state" in self.dataset.hf_dataset.column_names:
            sample_state = self.dataset[0]["observation.state"]
            if isinstance(sample_state, torch.Tensor):
                state_dim = sample_state.shape[0] if sample_state.ndim == 1 else sample_state.shape[-1]
            else:
                state_dim = len(sample_state)
            
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": [f"state_{i}" for i in range(state_dim)],
            }
        
        # 添加图像特征
        for key in self.dataset.hf_dataset.column_names:
            if key.startswith("observation.images."):
                sample_img = self.dataset[0][key]
                if isinstance(sample_img, torch.Tensor):
                    if sample_img.ndim == 3:  # (C, H, W)
                        img_shape = sample_img.shape
                    elif sample_img.ndim == 4:  # (B, C, H, W)
                        img_shape = sample_img.shape[1:]
                    else:
                        continue
                else:
                    continue
                
                # 转换为(H, W, C)格式（服务器端期望的格式）
                if len(img_shape) == 3:  # (C, H, W)
                    img_shape_hwc = (img_shape[1], img_shape[2], img_shape[0])  # (H, W, C)
                else:
                    img_shape_hwc = img_shape
                
                features[key] = {
                    "dtype": "image",
                    "shape": img_shape_hwc,  # (H, W, C)
                    "names": ["height", "width", "channels"],
                }
        
        logger.info(f"Extracted lerobot features from data: {list(features.keys())}")
        return features
    
    def _convert_dataset_obs_to_raw_obs(self, batch: dict, frame_idx: int) -> RawObservation:
        """将数据集中的观测转换为RawObservation格式
        
        Args:
            batch: 从DataLoader获取的batch（batch_size=1）
            frame_idx: 当前帧索引
            
        Returns:
            RawObservation: 原始观测字典，格式：
                - 状态组件：键名为lerobot_features中"observation.state"的"names"字段中的值
                - 图像：键名为去掉"observation.images."前缀后的相机名称（如cam_head）
        """
        raw_obs = {}
        
        # 转换state：需要分解为单独组件（根据lerobot_features中的names）
        if "observation.state" in batch:
            state = batch["observation.state"][0]  # (state_dim,)
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            
            # 从lerobot_features获取状态名称列表
            from lerobot.utils.constants import OBS_STR
            state_key = f"{OBS_STR}.state"
            
            if state_key in self.lerobot_features and "names" in self.lerobot_features[state_key]:
                state_names_raw = self.lerobot_features[state_key]["names"]
                
                # 处理names字段：可能是列表或字典（旧格式）
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
                    # 如果不是列表也不是字典，使用通用命名
                    logger.warning(f"Unexpected names format: {type(state_names_raw)}. Using fallback.")
                    state_names = [f"state_{i}" for i in range(len(state))]
                
                # 将状态分解为单独组件
                for idx, state_name in enumerate(state_names):
                    if idx < len(state):
                        raw_obs[state_name] = float(state[idx])
                    else:
                        raw_obs[state_name] = 0.0
            else:
                # 如果没有特征定义，使用通用命名
                for idx in range(len(state)):
                    raw_obs[f"state_{idx}"] = float(state[idx])
        
        # 转换图像（需要去掉"observation.images."前缀）
        # 注意：服务器端的resize_robot_observation_image期望输入是(H, W, C)格式（会在内部转换为(C, H, W)）
        # build_dataset_frame从values中直接取numpy array，所以RawObservation中的图像应该是(H, W, C)格式的numpy array
        # 从eval_depalletize_async.py来看，它使用(H, W, C)格式的numpy array（float32）
        # 为了保持一致性，我们使用(H, W, C)格式的float32 numpy array [0, 1]范围
        for key in batch.keys():
            if key.startswith("observation.images."):
                camera_base_name = key.replace("observation.images.", "")
                img = batch[key][0]  # 数据集中的格式通常是(C, H, W)，值在[0, 1]范围
                
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                
                # 确保转换为(H, W, C)格式
                if img.ndim == 3:
                    if img.shape[0] == 3 or img.shape[0] == 1:  # (C, H, W)
                        img = img.transpose(1, 2, 0)  # (H, W, C)
                    # 如果已经是(H, W, C)格式，保持不变
                
                # 转换为float32格式（与eval_depalletize_async.py保持一致）
                # 如果图像值在[0, 1]范围，直接使用；如果在[0, 255]范围，需要归一化
                if img.dtype == np.uint8 or (img.dtype in [np.float32, np.float64] and img.max() > 1.0):
                    # 如果是uint8或在[0, 255]范围，归一化到[0, 1]
                    img = img.astype(np.float32) / 255.0
                else:
                    # 如果已经是[0, 1]范围的float32/float64，转换为float32
                    img = img.astype(np.float32)
                
                raw_obs[camera_base_name] = img
        
        # 添加task（如果提供或从数据集中获取）
        if self.task_description is not None:
            raw_obs["task"] = self.task_description
        elif "task" in batch:
            task = batch["task"]
            if isinstance(task, (list, tuple)) and len(task) > 0:
                raw_obs["task"] = task[0]
            elif isinstance(task, str):
                raw_obs["task"] = task
            else:
                raw_obs["task"] = str(task) if task is not None else ""
        else:
            raw_obs["task"] = ""
        
        return raw_obs
    
    def receive_actions(self):
        """接收动作的线程函数"""
        logger.info("Action receiving thread starting")
        
        while not self.shutdown_event.is_set():
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # 收到空响应，继续等待
                
                # 反序列化动作
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                
                # 更新队列
                with self.action_queue_lock:
                    for action in timed_actions:
                        self.action_queue.put(action)
                
                self.action_chunk_size_received = max(self.action_chunk_size_received, len(timed_actions))
                self.must_go.set()  # 收到动作后，允许发送新观测
                
                if len(timed_actions) > 0:
                    logger.debug(f"Received {len(timed_actions)} actions, queue size: {self.action_queue.qsize()}")
            
            except grpc.RpcError as e:
                if not self.shutdown_event.is_set():
                    logger.error(f"Error receiving actions: {e}")
    
    def actions_available(self) -> bool:
        """检查是否有动作可用"""
        with self.action_queue_lock:
            return not self.action_queue.empty()
    
    def _ready_to_send_observation(self) -> bool:
        """检查是否准备好发送新观测"""
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            return queue_size / self.action_chunk_size_received <= self.chunk_size_threshold
    
    def send_observation(self, obs: TimedObservation) -> bool:
        """发送观测到服务器"""
        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")
        
        try:
            observation_bytes = pickle.dumps(obs)
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            logger.debug(f"Sent observation #{obs.get_timestep()}")
            return True
        except grpc.RpcError as e:
            logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False
    
    def get_action(self) -> Optional[TimedAction]:
        """从队列获取单个动作"""
        with self.action_queue_lock:
            if self.action_queue.empty():
                return None
            action = self.action_queue.get_nowait()
            with self.latest_action_lock:
                self.latest_action = action.get_timestep()
            return action
    
    def get_action_chunk(self, chunk_size: Optional[int] = None) -> Optional[list[TimedAction]]:
        """从队列获取动作chunk（用于可视化）
        
        Args:
            chunk_size: 要获取的chunk大小，如果为None则使用self.action_chunk_size
            
        Returns:
            动作chunk列表，如果队列中没有足够的动作则返回None
        """
        if chunk_size is None:
            chunk_size = self.action_chunk_size
        
        with self.action_queue_lock:
            if self.action_queue.qsize() < chunk_size:
                return None
            
            action_chunk = []
            for _ in range(chunk_size):
                if self.action_queue.empty():
                    # 如果队列在获取过程中变空，返回已收集的部分
                    return action_chunk if len(action_chunk) > 0 else None
                action = self.action_queue.get_nowait()
                action_chunk.append(action)
            
            # 更新latest_action为chunk中最后一个action的timestep
            if len(action_chunk) > 0:
                with self.latest_action_lock:
                    self.latest_action = action_chunk[-1].get_timestep()
            
            return action_chunk
    
    def stop(self):
        """停止客户端"""
        self.shutdown_event.set()
        self.channel.close()
        logger.info("Client stopped")


def eval_async_on_dataset(
    server_address: str,
    ckpt_path: str,
    dataset_root: str,
    episode: int,
    action_chunk_size: int = 16,
    task_description: Optional[str] = None,
    show_progress: bool = True,
    enable_visualization: bool = True,
):
    """
    使用异步推理在数据集上评估模型
    
    Args:
        server_address: 服务器地址 (格式: "host:port")
        ckpt_path: 模型checkpoint路径
        dataset_root: 数据集根目录
        episode: episode编号
        action_chunk_size: 动作块大小
        task_description: 任务描述（如果提供则覆盖数据集中的task）
        show_progress: 是否显示进度条
    """
    # 加载数据集
    print(f"\n{'='*80}")
    print(f"📂 Loading dataset from {dataset_root}")
    print(f"📹 Episode: {episode}")
    
    # 对于本地数据集，repo_id应该是一个字符串标识符（不包含"/"）
    # 使用数据集路径的最后一部分作为标识符，或者使用"local"
    dataset_name = Path(dataset_root).name if dataset_root else "local"
    dataset = LeRobotDataset(repo_id=dataset_name, root=dataset_root, episodes=[episode])
    
    # 过滤到指定episode
    if episode >= len(dataset.meta.episodes):
        raise ValueError(f"Episode {episode} out of range. Available episodes: 0-{len(dataset.meta.episodes)-1}")
    
    ep_meta = dataset.meta.episodes[episode]
    ep_start = ep_meta["dataset_from_index"]
    ep_end = ep_meta["dataset_to_index"]
    dataset.hf_dataset = dataset.hf_dataset.select(range(ep_start, ep_end))
    print(f"✅ Filtered dataset. Total frames in episode {episode}: {len(dataset.hf_dataset)} (indices {ep_start}-{ep_end-1})")
    
    # 创建DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    
    # 获取action维度
    first_batch = next(iter(dataloader))
    action_dim = first_batch['action'].shape[1]
    print(f"📊 Action dimension: {action_dim}")
    
    # 重新创建dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    
    # 初始化客户端
    print(f"\n🔧 Initializing async client...")
    client = DatasetAsyncClient(
        server_address=server_address,
        ckpt_path=ckpt_path,
        dataset=dataset,
        action_chunk_size=action_chunk_size,
        task_description=task_description,
    )
    
    # 启动接收动作的线程
    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()
    
    # 等待服务器准备
    time.sleep(1.0)
    
    # ------------- 初始化visualizer (可选) -------------
    vizer = None
    kb = None
    if enable_visualization and RERUN_AVAILABLE:
        vizer = RerunVisualizer()
        kb = KeyboardManager()
        print("✅ RerunVisualizer initialized")
    elif enable_visualization:
        print("⚠️  Running without RerunVisualizer (not available)")
    
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
            if 'observation.state' in batch:
                all_gt_states.append(batch['observation.state'][0].cpu().numpy())
        
        all_gt_actions = np.array(all_gt_actions)
        all_gt_states = np.array(all_gt_states) if all_gt_states else None
        
        # 可视化ground truth actions（与eval_on_dataset_lowpass.py保持一致）
        for dim in range(action_dim):
            vizer.visualize_chunk(
                name=f"chunk/action_dim_{dim}/gt",
                chunk_data=all_gt_actions[:, dim],
                step_id=0,
                width=3.0
            )
        
        # 可视化observations (如果可用)
        if all_gt_states is not None and len(all_gt_states) > 0:
            obs_dim = all_gt_states.shape[1]
            for dim in range(obs_dim):
                vizer.visualize_chunk(
                    name=f"obs/obs_{dim}",
                    chunk_data=all_gt_states[:, dim],
                    step_id=0,
                    width=3.0
                )
        
        print(f"✅ Ground truth visualization ready")
    
    # 开始评估
    print("\n" + "="*80)
    print("🚀 Starting evaluation...")
    print("="*80 + "\n")
    
    mse_per_action_dim = OrderedDict()
    mae_per_action_dim = OrderedDict()
    predictions = []
    ground_truths = []
    last_data_step = -1  # 用于跟踪上次可视化的step
    cached_action_chunk = []  # 缓存的action chunk（numpy数组格式）
    cached_chunk_start_frame = -1  # 缓存的chunk起始frame
    
    iterator = tqdm(enumerate(dataloader), total=len(dataset.hf_dataset), desc="Processing") if show_progress else enumerate(dataloader)
    
    for frame_idx, batch in iterator:
        # 暂停控制（如果启用）
        if vizer is not None and kb is not None:
            time.sleep(0.05)
            if kb.paused:
                print(f'===== 暂停中，按下空格开始 =====')
            while kb.paused:
                time.sleep(0.1)
        
        # 准备观测
        raw_obs = client._convert_dataset_obs_to_raw_obs(batch, frame_idx)
        
        # 创建TimedObservation
        with client.latest_action_lock:
            latest_action = client.latest_action
        
        timed_obs = TimedObservation(
            timestamp=time.time(),
            observation=raw_obs,
            timestep=max(latest_action + 1, frame_idx),
        )
        
        # 检查是否应该发送观测
        with client.action_queue_lock:
            timed_obs.must_go = client.must_go.is_set() and client.action_queue.empty()
            current_queue_size = client.action_queue.qsize()
        
        if client._ready_to_send_observation() or timed_obs.must_go:
            client.send_observation(timed_obs)
            if timed_obs.must_go:
                client.must_go.clear()
        
        # 如果缓存的chunk已经用完了，尝试获取新的chunk
        if cached_chunk_start_frame >= 0:
            chunk_idx_in_cache = frame_idx - cached_chunk_start_frame
        else:
            chunk_idx_in_cache = -1  # 没有缓存的chunk
        
        if len(cached_action_chunk) == 0 or chunk_idx_in_cache < 0 or chunk_idx_in_cache >= len(cached_action_chunk):
            # 尝试从队列获取新的chunk
            action_chunk = client.get_action_chunk(action_chunk_size)
            if action_chunk is not None and len(action_chunk) > 0:
                # 将TimedAction列表转换为numpy数组
                pred_chunk = np.array([
                    ta.get_action().cpu().numpy() if isinstance(ta.get_action(), torch.Tensor) 
                    else ta.get_action() 
                    for ta in action_chunk
                ])
                if pred_chunk.ndim == 2:  # (chunk_size, action_dim)
                    cached_action_chunk = pred_chunk
                    cached_chunk_start_frame = frame_idx
                    chunk_idx_in_cache = 0
                else:
                    logger.warning(f"Unexpected chunk shape: {pred_chunk.shape}")
                    cached_action_chunk = []
                    cached_chunk_start_frame = -1
        
        # 从缓存的chunk中获取当前frame对应的action
        pred_action = None
        if len(cached_action_chunk) > 0 and 0 <= chunk_idx_in_cache < len(cached_action_chunk):
            # 使用缓存的chunk中的对应action
            pred_action = cached_action_chunk[chunk_idx_in_cache]
        else:
            # 如果缓存中没有，尝试从队列获取单个action（fallback）
            timed_action = client.get_action()
            if timed_action is not None:
                pred_action = timed_action.get_action()
                if isinstance(pred_action, torch.Tensor):
                    pred_action = pred_action.cpu().numpy()
                if pred_action.ndim > 1:
                    pred_action = pred_action[0]
        
        if pred_action is not None:
            # 获取ground truth
            gt_action = batch['action'][0].cpu().numpy()  # (action_dim,)
            
            # 确保维度匹配
            if pred_action.shape[0] != action_dim:
                logger.warning(f"Action dimension mismatch: pred={pred_action.shape[0]}, gt={action_dim}")
                continue
            
            # 保存预测和真实值
            predictions.append(pred_action)
            ground_truths.append(gt_action)
            
            # 计算每个维度的MSE和MAE
            for dim in range(action_dim):
                error = pred_action[dim] - gt_action[dim]
                mse = error ** 2
                mae = abs(error)
                
                if dim not in mse_per_action_dim:
                    mse_per_action_dim[dim] = []
                    mae_per_action_dim[dim] = []
                
                mse_per_action_dim[dim].append(mse)
                mae_per_action_dim[dim].append(mae)
            
            # 可视化（如果启用）- 与eval_on_dataset_lowpass.py对齐
            if vizer is not None:
                # 显示图像 - 动态查找可用的相机图像（与eval_on_dataset_lowpass.py保持一致）
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
                                step_id=frame_idx
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
                                step_id=frame_idx
                            )
                
                # 可视化预测的action chunk（与eval_on_dataset_lowpass.py对齐）
                for dim in range(action_dim):
                    # 可视化MSE（与eval_on_dataset_lowpass.py保持一致）
                    vizer.visualize_chunk(
                        name=f"mse/action_dim_{dim}",
                        chunk_data=mse_per_action_dim[dim][-1],
                        step_id=frame_idx,
                        width=3.0,
                    )
                
                # 如果有完整的chunk，可视化整个chunk（与eval_on_dataset_lowpass.py对齐）
                # 只在chunk的第一个frame时可视化整个chunk
                should_visualize_chunk = (len(cached_action_chunk) > 0 and 
                                         cached_chunk_start_frame == frame_idx)
                
                if should_visualize_chunk:
                    for dim in range(action_dim):
                        # 删除之前的预测可视化（如果存在）
                        if last_data_step != frame_idx and last_data_step >= 0:
                            vizer.del_chunk(
                                name=f"chunk/action_dim_{dim}/pred_seg_{last_data_step}",
                                chunk_data=np.array([0.0]),  # 占位数据
                                step_id=last_data_step,
                                width=0.5
                            )
                        
                        # 可视化预测的整个chunk（与eval_on_dataset_lowpass.py对齐）
                        vizer.visualize_chunk(
                            name=f"chunk/action_dim_{dim}/pred_seg_{frame_idx}",
                            chunk_data=cached_action_chunk[:, dim],
                            step_id=frame_idx,
                            width=2.0,
                        )
        else:
            # 如果没有收到动作，使用零向量（不应该发生，但处理一下）
            logger.warning(f"No action received for frame {frame_idx}, using zeros")
            gt_action = batch['action'][0].cpu().numpy()
            pred_action = np.zeros(action_dim)
            
            predictions.append(pred_action)
            ground_truths.append(gt_action)
        
        # 更新last_data_step用于可视化
        if pred_action is not None:
            last_data_step = frame_idx
        
        # 控制循环频率
        time.sleep(max(0, client.environment_dt - 0.001))
    
    # 停止客户端
    client.stop()
    action_receiver_thread.join(timeout=5.0)
    
    # 打印统计结果
    print("\n" + "="*80)
    print("📊 Final Statistics")
    print("="*80)
    
    # Action名称定义
    if CONFIG_AVAILABLE and action_names and len(action_names) == action_dim:
        eval_action_names = action_names
    elif action_dim == 16:
        eval_action_names = [f"Arm_joint_{i+1}" for i in range(14)] + ["Left_claw", "Right_claw"]
    elif action_dim == 18:
        eval_action_names = (
            [f"arm_joint_{i+1}" for i in range(7)] +
            [f"arm_joint_{i+8}" for i in range(7)] +
            ["left_claw_position", "right_claw_position", "cmd_pose_z", "cmd_pose_pitch"]
        )
    elif action_dim == 24:
        eval_action_names = (
            ["COM_dx", "COM_dy", "COM_dz", "COM_dR11", "COM_dR21", "COM_dR31", "COM_dR12", "COM_dR22", "COM_dR32"] +
            [f"Arm_joint_{i+1}" for i in range(14)] +
            ["Gait_mode"]
        )
    else:
        eval_action_names = [f"Action_dim_{i}" for i in range(action_dim)]
    
    print(f"\n{'Dimension':<20} {'MSE':<15} {'MAE':<15}")
    print("-" * 80)
    
    for dim in range(action_dim):
        if dim in mse_per_action_dim and len(mse_per_action_dim[dim]) > 0:
            mse_mean = np.mean(mse_per_action_dim[dim])
            mae_mean = np.mean(mae_per_action_dim[dim])
            dim_name = eval_action_names[dim] if dim < len(eval_action_names) else f"Dim_{dim}"
            print(f'{dim_name:<20} {mse_mean:<15.8f} {mae_mean:<15.8f}')
    
    if len(mse_per_action_dim) > 0:
        overall_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in mse_per_action_dim.keys()])
        overall_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in mae_per_action_dim.keys()])
        
        print("-" * 80)
        print(f'{"Overall":<20} {overall_mse:<15.8f} {overall_mae:<15.8f}')
    
    print("="*80)
    print("\n✅ Evaluation completed!")
    
    if vizer is not None:
        print("\n[Offline Eval] Visualization active. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n✅ Exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate GrootPolicy Model on Dataset using Async Inference',
        epilog='Evaluates a trained GrootPolicy model on a LeRobot dataset using async inference architecture.'
    )
    parser.add_argument('--server_address', type=str, required=True,
                       help='Server address (format: host:port)')
    parser.add_argument('--ckpt-path', '--ckpt_path', type=str, required=True, dest='ckpt_path',
                       help='Path to the model checkpoint directory')
    parser.add_argument('--dataset-root', '--dataset_root', type=str, required=True, dest='dataset_root',
                       help='Path to the LeRobot dataset root directory')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode number to evaluate (default: 0)')
    parser.add_argument('--action-chunk-size', '--action_chunk_size', type=int, default=16, dest='action_chunk_size',
                       help='Action chunk size (default: 16, should match training config)')
    parser.add_argument('--task-description', '--task_description', type=str, default=None, dest='task_description',
                       help='Task description (language instruction) to override the task from dataset')
    parser.add_argument('--no-progress', '--no_progress', action='store_true', dest='no_progress',
                       help='Disable progress bar')
    parser.add_argument('--no-visualization', '--no_visualization', action='store_true', dest='no_visualization',
                       help='Disable Rerun visualization')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎯 GrootPolicy Async Inference Dataset Evaluation")
    print("="*80)
    print(f"Server: {args.server_address}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Dataset: {args.dataset_root}")
    print(f"Episode: {args.episode}")
    print(f"Action Chunk Size: {args.action_chunk_size}")
    if args.task_description:
        print(f"Task Description (overridden): '{args.task_description}'")
    else:
        print(f"Task Description: Will use task from dataset")
    print("="*80)
    
    eval_async_on_dataset(
        server_address=args.server_address,
        ckpt_path=args.ckpt_path,
        dataset_root=args.dataset_root,
        episode=args.episode,
        action_chunk_size=args.action_chunk_size,
        task_description=args.task_description,
        show_progress=not args.no_progress,
        enable_visualization=not args.no_visualization,
    )
