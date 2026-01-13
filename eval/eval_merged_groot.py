#!/usr/bin/env python
"""
融合模型评估脚本

评估通过权重融合得到的 GROOT 模型

使用方式：
    # 评估融合模型
    python eval/eval_merged_groot.py --model_path ./outputs/merged_groot/pretrained_model
    
    # 对比评估多个模型
    python eval/eval_merged_groot.py --compare \\
        --model_paths ./outputs/merged_groot/pretrained_model,/path/to/narrower,/path/to/wider
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
import math
import time
import traceback
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Optional

import torch
import numpy as np
import rospy

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.configs.types import RTCAttentionSchedule
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MergedModelConfig:
    """Configuration for merged model evaluation.
    
    与 eval_multi_model.py 的 RTCDemoConfig 对齐
    """
    
    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=16,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )
    
    # Demo parameters
    fps: float = 10.0  # Action execution frequency (Hz)
    
    # Compute device
    device: str = "cuda:0"
    
    # Get new actions horizon
    action_queue_size_to_get_new_actions: int = 90
    
    # Task to execute
    task: str = "Depalletize the box"
    
    # Torch compile configuration
    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    torch_compile_disable_cudagraphs: bool = True


@dataclass
class ModelWrapper:
    """模型包装器 - 与 eval_multi_model.py 的 ModelBundle 对齐"""
    name: str
    policy: object
    preprocessor: object
    postprocessor: object
    path: str
    
    def reset(self):
        """重置模型状态用于新的推理会话"""
        self.policy.reset()
        self.policy.init_rtc_processor()


def _apply_torch_compile(policy, cfg: MergedModelConfig):
    """Apply torch.compile to the policy's predict_action_chunk method.
    
    与 eval_multi_model.py 对齐
    """
    # GROOT models handle their own compilation
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


def load_model(
    model_path: str, 
    name: str = None, 
    cfg: MergedModelConfig = None,
    device: str = "cuda:0"
) -> ModelWrapper:
    """
    加载单个模型 - 与 eval_multi_model.py 的 load_model_bundle 对齐
    
    Args:
        model_path: 模型路径
        name: 模型名称
        cfg: 配置（可选）
        device: 设备
    
    Returns:
        ModelWrapper 实例
    """
    if cfg is None:
        cfg = MergedModelConfig(device=device)
    
    name = name or os.path.basename(model_path)
    logger.info(f"[LOAD] Loading model '{name}' from {model_path}")
    
    # 加载配置
    config = PreTrainedConfig.from_pretrained(model_path)
    
    # 设置 torch.compile
    if config.type in ["pi05", "pi0", "groot"]:
        config.compile_model = cfg.use_torch_compile
    
    # 加载 policy
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(model_path, config=config)
    policy.config.rtc_config = cfg.rtc
    policy.init_rtc_processor()
    policy = policy.to(device)
    policy.eval()
    
    # 应用 torch.compile（如果启用）
    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)
    
    # 加载 preprocessor 和 postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=model_path,
        dataset_stats=None,  # Will load from pretrained processor files
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )
    
    logger.info(f"[LOAD] Model '{name}' loaded successfully")
    
    return ModelWrapper(
        name=name,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        path=model_path,
    )


def get_actions(
    model: ModelWrapper,
    env: GrabBoxMpcEnv,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: MergedModelConfig,
    episode_state: EpisodeState,
):
    """推理线程 - 与 eval_multi_model.py 的 get_actions 对齐
    
    Args:
        model: 模型包装器
        env: 机器人环境
        action_queue: 动作队列
        shutdown_event: 关闭事件
        cfg: 配置
        episode_state: 当前episode的状态
    """
    
    try:
        logger.info(f"[GET_ACTIONS] Starting get actions thread with model: {model.name}")
        
        # 使用 LatencyTracker 精确跟踪延迟（与 eval_multi_model.py 对齐）
        latency_tracker = LatencyTracker()
        fps = cfg.fps
        time_per_chunk = 1.0 / fps
        
        preprocessor = model.preprocessor
        postprocessor = model.postprocessor
        policy = model.policy
        
        logger.info(f"[GET_ACTIONS] Using preprocessor/postprocessor from model: {model.name}")
        
        get_actions_threshold = cfg.action_queue_size_to_get_new_actions
        
        # 检查 RTC 是否启用
        if not cfg.rtc.enabled:
            get_actions_threshold = 0
        
        while not shutdown_event.is_set():
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()
                
                # 使用 LatencyTracker 计算 inference_delay（与 eval_multi_model.py 对齐）
                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk)
                
                # 获取观测
                obs_data, *_ = env.get_obs()
                
                state = torch.from_numpy(obs_data["state"]).float()
                observation = {}
                
                # 处理相机图像（根据 CAMERA_COMPONENTS 动态处理）
                camera_names = get_camera_names(CAMERA_COMPONENTS)
                for camera_name in camera_names:
                    if camera_name in obs_data:
                        camera_img_np = obs_data[camera_name]
                        if camera_img_np.ndim != 4:
                            rospy.logwarn(f"Unexpected camera image shape: {camera_img_np.shape}, expected (T, H, W, C)")
                            continue
                        camera_images = torch.from_numpy(
                            np.moveaxis(camera_img_np, 3, 1).copy()
                        ).float() / 255
                        obs_key = get_camera_observation_key(camera_name, use_image_features=False)
                        observation[obs_key] = camera_images.to(cfg.device)
                
                observation['observation.state'] = state.to(cfg.device)
                observation['task'] = cfg.task
                
                # 预处理
                processed_observation = preprocessor(observation)
                
                # 推理（使用 RTC）
                actions = policy.predict_action_chunk(
                    processed_observation,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )
                
                original_actions = actions.squeeze(0).clone()
                
                # 后处理
                _, chunk_size, _ = actions.shape
                processed_actions = []
                for i in range(chunk_size):
                    single_action = actions[:, i, :]
                    processed_action = postprocessor(single_action)
                    processed_actions.append(processed_action)
                
                pred_actions_unnorm = torch.stack(processed_actions, dim=1)
                postprocessed_actions = pred_actions_unnorm[0].clone()
                
                # 计算新的延迟并更新 LatencyTracker
                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)
                
                # 队列大小检查警告（与 eval_multi_model.py 对齐）
                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] cfg.action_queue_size_to_get_new_actions Too small, "
                        "It should be higher than inference delay + execution horizon."
                    )
                
                # 第一次推理平滑处理
                if episode_state.first_inference:
                    postprocessed_actions = apply_first_chunk_smooth(
                        postprocessed_actions, obs_data, env
                    )
                    episode_state.first_inference = False
                
                # 重采样
                postprocessed_resampled = resample_chunk_with_claw_hold(
                    postprocessed_actions.cpu().numpy(),
                    previous_action=(
                        episode_state.last_executed_action.cpu().numpy()
                        if episode_state.last_executed_action is not None
                        else None
                    ),
                    control_frequency=100.0,
                    source_dt=0.1,
                    arm_dims=slice(0, 14),
                    claw_dims=slice(14, 16),
                    device=postprocessed_actions.device,
                )
                
                episode_state.last_executed_action = postprocessed_resampled[-get_actions_threshold].clone()
                action_queue.merge(
                    original_actions, postprocessed_resampled, new_delay, action_index_before_inference
                )
            else:
                # Small sleep to prevent busy waiting
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
    cfg: MergedModelConfig,
):
    """执行动作的线程 - 与 eval_multi_model.py 的 actor_control 对齐
    
    Args:
        env: 机器人环境
        action_queue: 动作队列
        shutdown_event: 关闭事件
        cfg: 配置
    """
    
    try:
        logger.info("[ACTOR] Starting actor thread")
        
        action_count = 0
        action_interval = 1.0 / 100  # 100 Hz 控制频率
        
        while not shutdown_event.is_set():
            start_time = time.perf_counter()
            
            # Try to get an action from the queue with timeout
            if action_queue.qsize() > 0:
                action = action_queue.get()
            else:
                action = None
            
            if action is not None:
                action = action.cpu()
                control_cmd_pose = (
                    "Cmd_pose_z" in ACTION_COMPONENTS or
                    "Cmd_pose_pitch" in ACTION_COMPONENTS
                )
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Merged GROOT Model")
    parser.add_argument("--model_path", type=str, default="./outputs/merged_groot/pretrained_model",
                       help="Path to merged model")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run on")
    parser.add_argument("--fps", type=float, default=10.0,
                       help="Inference frequency (Hz)")
    parser.add_argument("--task", type=str, default="Depalletize the box",
                       help="Task description for the model")
    parser.add_argument("--use_torch_compile", action="store_true",
                       help="Use torch.compile for faster inference")
    parser.add_argument("--action_queue_size", type=int, default=90,
                       help="Action queue size threshold to get new actions")
    args = parser.parse_args()
    
    # 初始化 ROS
    rospy.init_node("eval_merged_groot", anonymous=True)
    
    # 创建配置（与 eval_multi_model.py 对齐）
    cfg = MergedModelConfig(
        fps=args.fps,
        device=args.device,
        task=args.task,
        use_torch_compile=args.use_torch_compile,
        action_queue_size_to_get_new_actions=args.action_queue_size,
    )
    
    logger.info(f"[MAIN] Using device: {cfg.device}")
    
    # 初始化环境
    env = GrabBoxMpcEnv()
    robot_sdk = RobotSDK()
    
    # 初始化手臂位置（与 eval_multi_model.py 对齐）
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
    
    # 加载模型
    logger.info(f"\n{'='*60}")
    logger.info("🚀 Loading Merged GROOT Model")
    logger.info(f"{'='*60}")
    
    model = load_model(
        model_path=args.model_path, 
        name="merged_model", 
        cfg=cfg,
        device=cfg.device
    )
    
    # 等待观测 buffer 就绪
    env.obs_buffer.wait_buffer_ready()
    logger.info("✅ Observation buffer ready")
    
    # 信号处理（与 eval_multi_model.py 对齐）
    from lerobot.rl.process import ProcessSignalHandler
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event
    
    print(f"\n{'='*60}")
    print(f"🤖 融合模型评估")
    print(f"   模型路径: {args.model_path}")
    print(f"   任务描述: {cfg.task}")
    print(f"   推理频率: {cfg.fps} Hz")
    print(f"   动作队列阈值: {cfg.action_queue_size_to_get_new_actions}")
    print(f"   RTC execution_horizon: {cfg.rtc.execution_horizon}")
    print(f"{'='*60}")
    
    while True:
        input("\n按回车开始推理...")
        
        # 重置模型状态（与 eval_multi_model.py 对齐）
        model.reset()
        shutdown_event.clear()
        episode_state = EpisodeState()
        action_queue = ActionQueue(cfg.rtc)
        
        logger.info(f"[MAIN] ActionQueue created, using model: {model.name}")
        
        # 启动线程（与 eval_multi_model.py 对齐）
        get_actions_thread = Thread(
            target=get_actions,
            args=(model, env, action_queue, shutdown_event, cfg, episode_state),
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
        
        logger.info(f"[MAIN] Threads started with {model.name}. Running demo...")
        print(f"\n💡 提示: 输入 'q' 停止当前推理并返回")
        
        while True:
            user_input = input().strip().lower()
            if user_input == 'q':
                shutdown_event.set()
                env.reset_claw_lock()
                final_reset_arm(
                    json_path=os.path.join(cur_dir, 'utils/initial_arm_traj.json'),
                    env=env,
                    control_arm=True,
                    control_claw=True
                )
                break
        
        get_actions_thread.join()
        actor_thread.join()
        
        logger.info("[MAIN] Demo finished. Returning to start...")


if __name__ == "__main__":
    main()
