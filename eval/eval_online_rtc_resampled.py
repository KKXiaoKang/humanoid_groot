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

This script demonstrates:
1. Creating a robot and policy (Groot,SmolVLA, Pi0, etc.) with RTC
2. Consuming actions from the policy while the robot executes
3. Periodically requesting new action chunks in the background using threads
4. Managing action buffers and timing for real-time operation

For simulation environments, see eval_with_simulation.py

Usage:
    # Run eval online with RTC and resampled actions
    uv run eval/eval_online_rtc_resampled.py \
        --policy.path=/path/to/checkpoint \
        --policy.device=cuda \
        --rtc.enabled=true \
        --rtc.execution_horizon=10 \
        --task="Depalletize the box" \
        --duration=300
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from robot_envs.kuavo_depalletize_env import GrabBoxMpcEnv
from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.robot_sdk import RobotSDK

import rospy
import numpy as np

from typing import Optional
import numpy as np
import time

from eval_online import final_reset_arm

FIRST_MODEL_INFERENCE = True

@dataclass
class RTCDemoConfig(HubMixin):
    """Configuration for RTC demo with action chunking policies and real robots."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

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
    device: str | None = None  # Device to run on (cuda, cpu, auto)

    # Get new actions horizon. The amount of executed steps after which will be requested new actions.
    # It should be higher than inference delay + execution horizon.
    action_queue_size_to_get_new_actions: int = 50

    # Task to execute
    task: str = field(default="Depalletize the box", metadata={"help": "Task to execute"})

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

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


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
                                  device: torch.device = torch.device('cuda:0')) -> np.ndarray:
    """
    Resample an action chunk so that arm joints are interpolated to the control frequency
    while claw positions are held at the original (low) frequency.
    """
    # print(f"[RESAMPLE_CHUNK_WITH_CLAW_HOLD] previous_action: {previous_action.shape}")
    # print(f"[RESAMPLE_CHUNK_WITH_CLAW_HOLD] action_chunk: {action_chunk.shape}")
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

def apply_first_chunk_smooth(action_chunk: torch.Tensor, obs_data: dict, env: GrabBoxMpcEnv):
    """
    Apply smooth to the first chunk of actions.
    """
    action_dim = action_chunk.cpu().numpy().shape[1]
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
        action_chunk_smoothed = np.vstack([transition_chunk, action_chunk])
        rospy.loginfo(f"   Generated smoothed action chunk of size {action_chunk_smoothed.shape[0]}")

    global FIRST_MODEL_INFERENCE
    FIRST_MODEL_INFERENCE = False
    
    return torch.from_numpy(action_chunk_smoothed).to(action_chunk.device)

def get_actions(
    policy,
    env: GrabBoxMpcEnv,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    """Thread function to request action chunks from the policy.

    Args:
        policy: The policy instance (SmolVLA, Pi0, etc.)
        robot: The robot instance for getting observations
        action_queue: Queue to put new action chunks
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """

    global FIRST_MODEL_INFERENCE

    try:
        logger.info("[GET_ACTIONS] Starting get actions thread")

        latency_tracker = LatencyTracker()  # Track latency of action chunks
        fps = cfg.fps
        time_per_chunk = 1.0 / fps

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,  # Will load from pretrained processor files
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
            },
        )

        logger.info("[GET_ACTIONS] Preprocessor/postprocessor loaded successfully with embedded stats")

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions

        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        last_executed_action = None

        while not shutdown_event.is_set():
            # print(f"[GET_ACTIONS] action_queue.qsize(): {action_queue.qsize()}")
            # print(f"[GET_ACTIONS] get_actions_threshold: {get_actions_threshold}")
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk)

                obs_data, *_ = env.get_obs()

                state = torch.from_numpy(obs_data["state"]).float()
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
                observation['task'] = "Depalletize the box"
                processed_observation = preprocessor(observation)

                # Generate actions WITH RTC
                actions = policy.predict_action_chunk(
                    processed_observation,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

                original_actions = actions.squeeze(0).clone()
                # print(f"[GET_ACTIONS] original_actions: {original_actions.shape}")

                _, chunk_size, _ = actions.shape
                processed_actions = []
                for i in range(chunk_size):
                    # 提取单个 action: (B, action_dim)
                    single_action = actions[:, i, :]
                    # 使用 postprocessor 进行反归一化
                    processed_action = postprocessor(single_action)
                    processed_actions.append(processed_action)
                
                # 堆叠回 (B, chunk_size, action_dim)，然后转换为 numpy
                pred_actions_unnorm = torch.stack(processed_actions, dim=1)  # (B, chunk_size, action_dim)
                postprocessed_actions = pred_actions_unnorm[0].clone()

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] cfg.action_queue_size_to_get_new_actions Too small, It should be higher than inference delay + execution horizon."
                    )

                if FIRST_MODEL_INFERENCE:
                    postprocessed_actions = apply_first_chunk_smooth(postprocessed_actions, obs_data, env)

                postprocessed_resampled = resample_chunk_with_claw_hold(
                    postprocessed_actions.cpu().numpy(),
                    previous_action=last_executed_action.cpu().numpy() if last_executed_action is not None else None,
                    control_frequency=100.0,
                    source_dt=0.1,
                    arm_dims=slice(0, 14),
                    claw_dims=slice(14, 16),
                    device=postprocessed_actions.device,
                )

                # print(f"[GET_ACTIONS] postprocessed_resampled: {postprocessed_resampled.shape}")
                last_executed_action = postprocessed_resampled[-get_actions_threshold].clone()
                # print(f"[GET_ACTIONS] last_executed_action: {last_executed_action.shape}")
                action_queue.merge(
                    original_actions, postprocessed_resampled, new_delay, action_index_before_inference
                )
            else:
                # Small sleep to prevent busy waiting
                # print(f"[GET_ACTIONS] action_queue.qsize() > get_actions_threshold, sleep 0.01s")
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
        robot: The robot instance
        action_queue: Queue to get actions from
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_count = 0
        # action_interval = 1.0 / cfg.fps
        action_interval = 1.0 / 100

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            # Try to get an action from the queue with timeout
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
    """Apply torch.compile to the policy's predict_action_chunk method.

    Args:
        policy: Policy instance to compile
        cfg: Configuration containing torch compile settings

    Returns:
        Policy with compiled predict_action_chunk method
    """

    # PI models handle their own compilation
    # if policy.type == "pi05" or policy.type == "pi0":
    if policy.type == "groot":
        return policy

    try:
        # Check if torch.compile is available (PyTorch 2.0+)
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

        # Compile the predict_action_chunk method
        # - CUDA graphs disabled to prevent tensor aliasing from in-place ops (x_t += dt * v_t)
        compile_kwargs = {
            "backend": cfg.torch_compile_backend,
            "mode": cfg.torch_compile_mode,
        }

        # Disable CUDA graphs if requested (prevents tensor aliasing issues)
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

@parser.wrap()
def demo_cli(cfg: RTCDemoConfig):
    """Main entry point for RTC demo with draccus configuration, with verbose printing for debugging."""

    # 初始化日志
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
    robot_sdk.control.set_arm_quick_mode(True)

    # 加载 policy
    policy_class = get_policy_class(cfg.policy.type)
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)

    if cfg.policy.type in ["pi05", "pi0", "groot"]:
        config.compile_model = cfg.use_torch_compile

    logger.info(f"[MAIN] Loading policy {cfg.policy.type} from {cfg.policy.pretrained_path}")
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)
    policy.config.rtc_config = cfg.rtc
    policy.init_rtc_processor()
    policy = policy.to("cuda:0")
    policy.eval()
    logger.info("[MAIN] Policy loaded and set to eval mode")

    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)

    # 初始化 ActionQueue
    action_queue = ActionQueue(cfg.rtc)
    logger.info("[MAIN] ActionQueue created")

    # # 初始化手臂位置
    # robot_sdk.control.set_external_control_arm_mode()
    # robot_sdk.control.control_head(0, np.deg2rad(20))
    # cur_dir = os.path.dirname(os.path.abspath(__file__))
    # final_reset_arm(
    #     json_path=os.path.join(cur_dir, 'utils/initial_arm_traj.json'), 
    #     env=env,
    #     control_arm=True,
    #     control_claw=True
    # )
    # robot_sdk.control.set_arm_quick_mode(True)

    env.obs_buffer.wait_buffer_ready()

    input(f"轨迹回放结束, 按回车继续 ==== \n")

    # 启动线程
    get_actions_thread = Thread(target=get_actions, args=(policy, env, action_queue, shutdown_event, cfg), daemon=True, name="GetActions")
    actor_thread = Thread(target=actor_control, args=(env, action_queue, shutdown_event, cfg), daemon=True, name="Actor")
    get_actions_thread.start()
    actor_thread.start()

    logger.info("[MAIN] Threads started. Running demo...")
    start_time = time.time()
    while (time.time() - start_time) < cfg.duration and not shutdown_event.is_set():
        time.sleep(5)
        logger.info(f"[MAIN] Queue size: {action_queue.qsize()}")

    shutdown_event.set()
    get_actions_thread.join()
    actor_thread.join()
    logger.info("[MAIN] Demo finished.")


if __name__ == "__main__":
    demo_cli()
    logging.info("RTC demo finished")