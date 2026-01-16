#!/usr/bin/env python
"""
MergeVLA 融合模型实时评估脚本

评估通过 MergeVLA 方法融合的 GROOT 模型，支持稀疏 LoRA 适配层

使用方式：
    python eval/eval_merged_groot.py \
        --model_path ./outputs/merged_groot_mergevla/pretrained_model \
        --rtc.enabled=true \
        --rtc.execution_horizon=10 \
        --task="Depalletize the box" \
        --duration=30
    
    # 禁用适配层（用于调试）
    python eval/eval_merged_groot.py \
        --model_path ./outputs/merged_groot_mergevla/pretrained_model \
        --disable_adapter=true
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import math
import time
import traceback
import json
import glob
from pathlib import Path
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Optional

import torch
import numpy as np
import rospy

from lerobot.configs import parser
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.utils.utils import init_logging
from configs.config import get_camera_observation_key, get_camera_names, CAMERA_COMPONENTS, ACTION_COMPONENTS

from robot_envs.kuavo_depalletize_env import GrabBoxMpcEnv
from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.robot_sdk import RobotSDK
from eval_online import final_reset_arm
from eval_multi_model import (
    resample_chunk_with_claw_hold,
    apply_first_chunk_smooth,
    set_arm_quick_mode,
    EpisodeState,
)

from lerobot.policies.groot.weight_merge_groot import DistributionAdapter
from lerobot.policies.groot.groot_n1 import BACKBONE_FEATURE_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MergeVLAConfig:
    """MergeVLA 评估配置（使用 draccus 解析器）"""
    
    # RTC 配置
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=16,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )
    
    # 模型路径（必需）
    model_path: str = field(
        default="./outputs/merged_groot_mergevla/pretrained_model",
        metadata={"help": "MergeVLA 融合模型路径"}
    )
    
    # 推理参数
    duration: float = field(default=30.0, metadata={"help": "单次推理周期（秒）"})
    fps: float = field(default=10.0, metadata={"help": "推理频率 (Hz)"})
    device: str = field(default="cuda:0", metadata={"help": "运行设备"})
    task: str = field(default="Depalletize the box", metadata={"help": "任务描述"})
    
    # 适配层控制
    disable_adapter: bool = field(default=False, metadata={"help": "禁用适配层（调试用）"})
    
    # 动作队列配置
    action_queue_size_to_get_new_actions: int = field(
        default=90,
        metadata={"help": "触发新推理的动作队列阈值"}
    )


@dataclass
class ModelWrapper:
    """模型包装器"""
    name: str
    policy: object
    preprocessor: object
    postprocessor: object
    adapter: object = None
    
    def reset(self):
        self.policy.reset()
        self.policy.init_rtc_processor()


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
    
    logger.info(f"   📐 hidden_size: {hidden_size}, adapter_type: {adapter_type}, lora_rank: {lora_rank}")
    
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
    
    logger.info(f"   ✅ 适配层加载成功，参数量: {sum(p.numel() for p in adapter.parameters()):,}")
    return adapter


def wrap_policy_with_adapter(policy, adapter):
    """包装 GrootPolicy，使其在推理时使用适配层"""
    
    def get_action_with_adapter(inputs: dict, **kwargs):
        backbone_inputs, action_inputs = policy._groot_model.prepare_input(inputs)
        backbone_outputs = policy._groot_model.backbone(backbone_inputs)
        
        # 通过适配层
        backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]
        adapted_features = adapter(backbone_features, task_id=None)
        backbone_outputs[BACKBONE_FEATURE_KEY] = adapted_features
        
        # 通过 action_head
        rtc_enabled = kwargs.pop('rtc_enabled', policy._groot_model._rtc_enabled())
        return policy._groot_model.action_head.get_action(
            backbone_outputs, action_inputs, rtc_enabled=rtc_enabled, **kwargs
        )
    
    policy._groot_model.get_action = get_action_with_adapter
    logger.info(f"   ✅ Policy 已包装适配层")


def load_mergevla_model(cfg: MergeVLAConfig) -> ModelWrapper:
    """加载 MergeVLA 融合模型"""
    
    logger.info(f"🚀 加载 MergeVLA 模型: {cfg.model_path}")
    
    # 加载融合配置
    merge_config = load_merge_config(cfg.model_path)
    if merge_config is None:
        raise RuntimeError(f"未找到 merge_config.json: {cfg.model_path}")
    
    if merge_config.get('merge_method') != 'mergevla':
        raise RuntimeError(f"不是 MergeVLA 模型: {merge_config.get('merge_method')}")
    
    # 加载 policy
    config = PreTrainedConfig.from_pretrained(cfg.model_path)
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(cfg.model_path, config=config)
    policy.config.rtc_config = cfg.rtc
    policy.init_rtc_processor()
    policy = policy.to(cfg.device)
    policy.eval()
    
    # 加载适配层
    adapter = None
    if not cfg.disable_adapter:
        logger.info(f"🔧 加载 MergeVLA 适配层...")
        adapter = load_adapter(policy, cfg.model_path, merge_config, cfg.device)
        wrap_policy_with_adapter(policy, adapter)
    else:
        logger.warning("⚠️ 适配层已禁用")
    
    # 加载预处理器
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=cfg.model_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )
    
    return ModelWrapper(
        name="mergevla_model",
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        adapter=adapter,
    )


def get_actions(
    model: ModelWrapper,
    env: GrabBoxMpcEnv,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: MergeVLAConfig,
    episode_state: EpisodeState,
):
    """推理线程"""
    try:
        latency_tracker = LatencyTracker()
        time_per_chunk = 1.0 / cfg.fps
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
                
                # 获取观测
                obs_data, *_ = env.get_obs()
                state = torch.from_numpy(obs_data["state"]).float()
                observation = {'observation.state': state.to(cfg.device), 'task': cfg.task}
                
                camera_names = get_camera_names(CAMERA_COMPONENTS)
                for camera_name in camera_names:
                    if camera_name in obs_data:
                        camera_img_np = obs_data[camera_name]
                        if camera_img_np.ndim == 4:
                            camera_images = torch.from_numpy(
                                np.moveaxis(camera_img_np, 3, 1).copy()
                            ).float() / 255
                            obs_key = get_camera_observation_key(camera_name, use_image_features=False)
                            observation[obs_key] = camera_images.to(cfg.device)
                
                # 推理
                processed_observation = model.preprocessor(observation)
                actions = model.policy.predict_action_chunk(
                    processed_observation,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )
                
                original_actions = actions.squeeze(0).clone()
                
                # 后处理
                _, chunk_size, _ = actions.shape
                processed_actions = []
                for i in range(chunk_size):
                    processed_action = model.postprocessor(actions[:, i, :])
                    processed_actions.append(processed_action)
                
                postprocessed_actions = torch.stack(processed_actions, dim=1)[0].clone()
                
                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)
                
                # 第一次推理平滑处理
                if episode_state.first_inference:
                    postprocessed_actions = apply_first_chunk_smooth(postprocessed_actions, obs_data, env)
                    episode_state.first_inference = False
                
                # 重采样
                postprocessed_resampled = resample_chunk_with_claw_hold(
                    postprocessed_actions.cpu().numpy(),
                    previous_action=(
                        episode_state.last_executed_action.cpu().numpy()
                        if episode_state.last_executed_action is not None else None
                    ),
                    control_frequency=100.0,
                    source_dt=0.1,
                    arm_dims=slice(0, 14),
                    claw_dims=slice(14, 16),
                    device=postprocessed_actions.device,
                )
                
                episode_state.last_executed_action = postprocessed_resampled[-get_actions_threshold].clone()
                action_queue.merge(original_actions, postprocessed_resampled, new_delay, action_index_before_inference)
            else:
                time.sleep(0.01)
        
    except Exception as e:
        traceback.print_exc()
        shutdown_event.set()


def actor_control(env: GrabBoxMpcEnv, action_queue: ActionQueue, shutdown_event: Event):
    """执行动作的线程"""
    try:
        action_interval = 1.0 / 100  # 100 Hz
        
        while not shutdown_event.is_set():
            start_time = time.perf_counter()
            
            if action_queue.qsize() > 0:
                action = action_queue.get().cpu()
                control_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
                env.exec_actions(actions=action, control_arm=True, control_claw=True, control_cmd_pose=control_cmd_pose)
            
            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, action_interval - dt_s - 0.001))
        
    except Exception as e:
        traceback.print_exc()
        shutdown_event.set()


@parser.wrap()
def main(cfg: MergeVLAConfig):
    """MergeVLA 融合模型实时推理入口"""
    
    init_logging()
    logger.info(f"[MAIN] Using device: {cfg.device}")
    
    rospy.init_node("eval_mergevla", anonymous=True)
    
    # 初始化环境
    env = GrabBoxMpcEnv()
    robot_sdk = RobotSDK()
    robot_sdk.control.set_external_control_arm_mode()
    robot_sdk.control.control_head(0, np.deg2rad(20))
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    final_reset_arm(
        json_path=os.path.join(cur_dir, 'utils/initial_arm_traj.json'),
        env=env, control_arm=True, control_claw=True
    )
    set_arm_quick_mode(True)
    
    # 加载模型
    model = load_mergevla_model(cfg)
    
    env.obs_buffer.wait_buffer_ready()
    
    from lerobot.rl.process import ProcessSignalHandler
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event
    
    print(f"\n{'='*60}")
    print(f"🤖 MergeVLA 实时推理")
    print(f"   模型: {cfg.model_path}")
    print(f"   适配层: {'✅ 已启用' if model.adapter else '❌ 已禁用'}")
    print(f"   任务: {cfg.task}")
    print(f"   频率: {cfg.fps} Hz")
    print(f"   周期: {cfg.duration} 秒")
    print(f"   RTC: {'✅ 启用' if cfg.rtc.enabled else '❌ 禁用'}")
    print(f"   RTC execution_horizon: {cfg.rtc.execution_horizon}")
    print(f"{'='*60}")
    
    while True:
        input("\n按回车开始推理...")
        
        model.reset()
        shutdown_event.clear()
        episode_state = EpisodeState()
        action_queue = ActionQueue(cfg.rtc)
        
        get_actions_thread = Thread(
            target=get_actions,
            args=(model, env, action_queue, shutdown_event, cfg, episode_state),
            daemon=True
        )
        actor_thread = Thread(
            target=actor_control,
            args=(env, action_queue, shutdown_event),
            daemon=True
        )
        
        get_actions_thread.start()
        actor_thread.start()
        
        print(f"💡 推理中... {cfg.duration}秒后自动停止，或输入 'q' 手动停止")
        
        # 使用 duration 自动停止
        start_time = time.perf_counter()
        import select
        import sys as _sys
        
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
        
        shutdown_event.set()
        env.reset_claw_lock()
        final_reset_arm(
            json_path=os.path.join(cur_dir, 'utils/initial_arm_traj.json'),
            env=env, control_arm=True, control_claw=True
        )
        
        get_actions_thread.join()
        actor_thread.join()
        
        logger.info("✅ 本轮推理完成")


if __name__ == "__main__":
    main()
