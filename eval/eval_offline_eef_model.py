#!/usr/bin/env python
"""
Offline inference for EEF model.

This script reuses eval_eef_model logic (model loading, FK state conversion,
relative-action conversion, IK conversion) and runs inference on a LeRobot
dataset stream without connecting to robot hardware.
"""
import os
import sys
import json
import time
import math
import logging
import traceback
from dataclasses import dataclass, field
from typing import Optional, List
from threading import Event, Thread, Lock

import numpy as np
import torch
from torch.utils.data import DataLoader
import rospy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lerobot.configs import parser
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

from configs.config import get_camera_observation_key, get_camera_names, CAMERA_COMPONENTS
from configs.config import ACTION_COMPONENTS
from robot_envs.kuavo_depalletize_env import GrabBoxMpcEnv

from eval_eef_model import (
    load_model_bundle,
    EpisodeState,
    RTCDemoConfig,
    convert_eef_action_to_joint_action,
    resample_chunk_with_claw_hold,
    apply_first_chunk_smooth,
    _load_arm_lock_values,
)

logger = logging.getLogger(__name__)


def play_reset_arm_trajectory_from_json(
    env: GrabBoxMpcEnv,
    json_path: str,
    control_arm: bool = True,
    control_claw: bool = True,
) -> None:
    """
    Replay arm_action from JSON without env.get_obs() (avoids empty camera timestamp buffers).
    Expects keys: arm_action (list of rows), optional dt (default 0.1).
    Rows are typically 14 (joints); appends claw [0,0] and cmd_pose [0,0] if needed.
    """
    path_obj = os.path.abspath(json_path)
    if not os.path.isfile(path_obj):
        logger.error("[OFFLINE] reset arm JSON not found: %s", path_obj)
        return

    with open(path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)

    arm_actions = data.get("arm_action")
    if not isinstance(arm_actions, list) or len(arm_actions) == 0:
        logger.error("[OFFLINE] reset JSON missing non-empty 'arm_action': %s", path_obj)
        return

    dt = float(data.get("dt", 0.1))
    has_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
    logger.info("[OFFLINE] Playing reset trajectory from %s (%d steps, dt=%.3fs)", path_obj, len(arm_actions), dt)

    for i, row in enumerate(arm_actions):
        vec = np.asarray(row, dtype=np.float32).ravel()
        if vec.size < 14:
            logger.warning("[OFFLINE] reset step %d: expected >=14 values, got %d, skip", i, vec.size)
            continue

        arm14 = vec[:14].copy()
        if vec.size >= 18:
            claw = vec[14:16].copy()
            cmd_pose = vec[16:18].copy()
        elif vec.size >= 16:
            claw = vec[14:16].copy()
            cmd_pose = np.array([0.0, 0.0], dtype=np.float32)
        else:
            claw = np.array([0.0, 0.0], dtype=np.float32)
            cmd_pose = np.array([0.0, 0.0], dtype=np.float32)

        if has_cmd_pose:
            action = np.concatenate([arm14, claw, cmd_pose]).astype(np.float32)
        else:
            action = np.concatenate([arm14, claw]).astype(np.float32)

        env.exec_actions(
            actions=action,
            control_arm=control_arm,
            control_claw=control_claw,
            control_cmd_pose=has_cmd_pose,
        )
        time.sleep(dt)

    logger.info("[OFFLINE] Reset trajectory playback finished.")


class FakeObsStream:
    """Dataset-backed observation stream with env.get_obs()-like output."""

    def __init__(
        self,
        dataset_path: str,
        fps: float,
        camera_components,
        shuffle: bool = False,
        loop: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        episodes: Optional[List[int]] = None,
    ):
        if episodes is not None and isinstance(episodes, int):
            episodes = [episodes]
        self.dataset = LeRobotDataset(repo_id="local", root=dataset_path, episodes=episodes)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        self.it = iter(self.dataloader)
        self.loop = loop
        self.dt = 1.0 / float(fps)
        self.last_ts = time.time()
        self.camera_names = get_camera_names(camera_components)

    def _next_batch(self):
        try:
            batch = next(self.it)
        except StopIteration:
            if not self.loop:
                raise
            self.it = iter(self.dataloader)
            batch = next(self.it)
        return batch

    def get_obs(self):
        batch = self._next_batch()
        obs_data = {}

        state = batch["observation.state"][0].detach().cpu().numpy().astype(np.float32)
        obs_data["state"] = state[None, :]

        for cam in self.camera_names:
            obs_key = get_camera_observation_key(cam, use_image_features=False)
            if obs_key in batch:
                img_chw = batch[obs_key][0].detach().cpu().numpy()
            else:
                fallback_key = f"observation.images.{cam}"
                if fallback_key in batch:
                    img_chw = batch[fallback_key][0].detach().cpu().numpy()
                else:
                    continue

            if img_chw.dtype == np.uint8:
                img_uint8 = img_chw
            else:
                img_uint8 = (np.clip(img_chw, 0.0, 1.0) * 255.0).astype(np.uint8)
            img_hwc = np.transpose(img_uint8, (1, 2, 0))
            obs_data[cam] = img_hwc[None, ...]

        now = time.time()
        self.last_ts = now
        return obs_data, None, now, None, now


@dataclass
class OfflineEEFConfig(RTCDemoConfig):
    dataset_path: str = field(
        default="/mnt/ssd/datasets/0415_pick_cube",
        metadata={"help": "LeRobot dataset root path"},
    )
    max_steps: int = field(
        default=1000,
        metadata={"help": "Max offline inference steps"},
    )
    loop_dataset: bool = field(
        default=False,
        metadata={"help": "Loop dataset after end"},
    )
    episodes: Optional[List[int]] = field(
        default_factory=lambda: [0],
        metadata={"help": "Episode ids to run; None means all"},
    )
    non_rtc_replace_queue_on_new_chunk: bool = field(
        default=True,
        metadata={"help": "When rtc.enabled=false, replace queue with new chunk instead of append"},
    )
    non_rtc_trigger_threshold: int = field(
        default=140,
        metadata={"help": "When rtc.enabled=false, trigger inference if queue size <= this threshold"},
    )
    reset_arm_json_path: str = field(
        default="utils/start_arm_traj_0417_s62.json",
        metadata={"help": "Relative path under eval/ for final_reset_arm trajectory json"},
    )


def _build_state_for_model(model_bundle, obs_data: dict, episode_state: EpisodeState) -> torch.Tensor:
    action_dim = model_bundle.action_dim
    is_eef_mode = (model_bundle.action_space_type in ["Delta eef", "Absolute eef"] and action_dim == 20)
    if not is_eef_mode:
        return torch.from_numpy(obs_data["state"]).float()

    current_state_np = obs_data["state"][0]
    # Offline dataset state is already EEF format:
    # left_eef(0:9) + right_eef(9:18) + claw(18:20)
    if len(current_state_np) >= 20:
        left_eef_pose = current_state_np[0:9]
        right_eef_pose = current_state_np[9:18]
        gripper_state = current_state_np[18:20]
    else:
        logger.warning(
            "[OFFLINE] EEF mode expects >=20D state, got %d. Falling back with zero padding.",
            len(current_state_np),
        )
        padded = np.zeros(20, dtype=np.float32)
        padded[: len(current_state_np)] = current_state_np
        left_eef_pose = padded[0:9]
        right_eef_pose = padded[9:18]
        gripper_state = padded[18:20]

    eef_state = np.concatenate([left_eef_pose, right_eef_pose, gripper_state]).astype(np.float32)

    if model_bundle.is_relative_action_mode:
        episode_state.current_reference_pose = eef_state.copy()
    return torch.from_numpy(eef_state).float().unsqueeze(0)


def _convert_relative_actions_if_needed(model_bundle, episode_state: EpisodeState, postprocessed_actions: torch.Tensor):
    if not model_bundle.is_relative_action_mode or model_bundle.postprocessor_step is None:
        return postprocessed_actions
    if episode_state.current_reference_pose is None:
        print("!!!!!! current_reference_pose is None !!!!!!!")
        episode_state.current_reference_pose = np.zeros(model_bundle.action_dim, dtype=np.float32)

    pred_chunk_tensor = postprocessed_actions.unsqueeze(0)
    reference_pose_tensor = torch.from_numpy(episode_state.current_reference_pose).to(
        pred_chunk_tensor.device
    ).unsqueeze(0)
    pred_chunk_absolute = model_bundle.postprocessor_step._convert_relative_to_absolute_eef_action(
        pred_chunk_tensor, reference_pose_tensor
    )
    return pred_chunk_absolute[0].clone()


@dataclass
class OfflineRuntimeState:
    consumed_steps: int = 0
    produced_chunks: int = 0
    stop_reason: str = ""


def get_actions(
    model_bundle,
    env: GrabBoxMpcEnv,
    obs_stream: FakeObsStream,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: OfflineEEFConfig,
    episode_state: EpisodeState,
    runtime_state: OfflineRuntimeState,
    action_lock: Lock,
):
    latency_tracker = LatencyTracker()
    time_per_chunk = 1.0 / cfg.fps
    get_actions_threshold = cfg.action_queue_size_to_get_new_actions
    if not cfg.rtc.enabled:
        get_actions_threshold = cfg.non_rtc_trigger_threshold
    control_frequency = 100.0
    control_dt = 1.0 / control_frequency
    dataset_dt = 1.0 / float(cfg.fps)
    frames_to_skip_before_next_obs = 0

    while not shutdown_event.is_set():
        if runtime_state.consumed_steps >= cfg.max_steps:
            runtime_state.stop_reason = runtime_state.stop_reason or "reach_max_steps"
            shutdown_event.set()
            break

        if action_queue.qsize() > get_actions_threshold:
            time.sleep(0.005)
            continue

        try:
            while frames_to_skip_before_next_obs > 0:
                obs_stream.get_obs()
                frames_to_skip_before_next_obs -= 1
            obs_data, *_ = obs_stream.get_obs()
        except StopIteration:
            runtime_state.stop_reason = runtime_state.stop_reason or "dataset_exhausted"
            shutdown_event.set()
            break

        t0 = time.perf_counter()
        action_index_before_inference = action_queue.get_action_index()
        prev_actions = action_queue.get_left_over()
        inference_delay = math.ceil(latency_tracker.max() / time_per_chunk)

        observation = {}
        for camera_name in get_camera_names(CAMERA_COMPONENTS):
            if camera_name not in obs_data:
                continue
            camera_img_np = obs_data[camera_name]
            if camera_img_np.ndim != 4:
                continue
            camera_images = torch.from_numpy(np.moveaxis(camera_img_np, 3, 1).copy()).float() / 255.0
            obs_key = get_camera_observation_key(camera_name, use_image_features=False)
            observation[obs_key] = camera_images.to(cfg.device)

        state = _build_state_for_model(model_bundle, obs_data, episode_state)
        observation["observation.state"] = state.to(cfg.device)
        observation["task"] = cfg.task
        processed_observation = model_bundle.preprocessor(observation)

        with torch.inference_mode():
            actions = model_bundle.policy.predict_action_chunk(
                processed_observation,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_actions,
            )
        original_actions = actions.squeeze(0).clone()

        processed_actions = []
        _, chunk_size, _ = actions.shape
        for i in range(chunk_size):
            processed_actions.append(model_bundle.postprocessor(actions[:, i, :]))
        postprocessed_actions = torch.stack(processed_actions, dim=1)[0].clone()
        postprocessed_actions = _convert_relative_actions_if_needed(
            model_bundle, episode_state, postprocessed_actions
        )

        if model_bundle.action_dim == 20:
            post_np = postprocessed_actions.detach().cpu().numpy()
            joint_actions = [
                convert_eef_action_to_joint_action(a, model_type=cfg.ik_model_type) for a in post_np
            ]
            postprocessed_actions = torch.from_numpy(np.asarray(joint_actions)).to(cfg.device)
        elif episode_state.first_inference:
            postprocessed_actions = apply_first_chunk_smooth(
                postprocessed_actions, obs_data, env, model_bundle.action_dim
            )
            episode_state.first_inference = False

        with action_lock:
            previous_action_np = (
                episode_state.last_executed_action.detach().cpu().numpy()
                if episode_state.last_executed_action is not None
                else None
            )

        postprocessed_resampled = resample_chunk_with_claw_hold(
            postprocessed_actions.detach().cpu().numpy(),
            previous_action=previous_action_np,
            control_frequency=control_frequency,
            source_dt=0.1,
            arm_dims=slice(0, 14),
            claw_dims=slice(14, 16),
            device=torch.device(cfg.device),
        )
        resampled_chunk_len = int(postprocessed_resampled.shape[0])

        new_latency = time.perf_counter() - t0
        latency_tracker.add(new_latency)
        new_delay = math.ceil(new_latency / time_per_chunk)
        if (not cfg.rtc.enabled) and cfg.non_rtc_replace_queue_on_new_chunk:
            with action_queue.lock:
                action_queue.original_queue = original_actions.clone()
                action_queue.queue = postprocessed_resampled.clone()
                action_queue.last_index = 0
                action_queue.original_last_index = 0
        else:
            action_queue.merge(
                original_actions,
                postprocessed_resampled,
                new_delay,
                action_index_before_inference,
            )
        runtime_state.produced_chunks += 1

        # Align dataset progression with executed control duration. When threshold=0,
        # producer waits until queue is drained, so each inferred chunk corresponds
        # to one executed segment in time and should advance offline observations.
        executed_duration = max(0.0, (resampled_chunk_len - 1) * control_dt)
        frames_to_advance = max(1, int(round(executed_duration / dataset_dt)))
        frames_to_skip_before_next_obs = max(0, frames_to_advance - 1)

        if runtime_state.produced_chunks % 10 == 0:
            logger.info(
                "[GET_ACTIONS] produced_chunks=%d queue=%d latency_max=%.4fs",
                runtime_state.produced_chunks,
                action_queue.qsize(),
                latency_tracker.max(),
            )


def actor_control(
    env: GrabBoxMpcEnv,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: OfflineEEFConfig,
    episode_state: EpisodeState,
    runtime_state: OfflineRuntimeState,
    action_lock: Lock,
    left_arm_lock_values: Optional[np.ndarray] = None,
    right_arm_lock_values: Optional[np.ndarray] = None,
):
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_count = 0
        action_interval = 1.0 / 100
        while not shutdown_event.is_set():
            if runtime_state.consumed_steps >= cfg.max_steps:
                runtime_state.stop_reason = runtime_state.stop_reason or "reach_max_steps"
                shutdown_event.set()
                break

            start_time = time.perf_counter()
            action = action_queue.get() if action_queue.qsize() > 0 else None

            if action is not None:
                action = action.cpu()
                if action.numel() >= 14:
                    action_np = action.numpy().copy()
                    if left_arm_lock_values is not None:
                        action_np[0:7] = left_arm_lock_values
                    if right_arm_lock_values is not None:
                        action_np[7:14] = right_arm_lock_values
                    action = torch.from_numpy(action_np)

                with action_lock:
                    episode_state.last_executed_action = action.clone()

                control_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
                env.exec_actions(
                    actions=action,
                    control_arm=True,
                    control_claw=True,
                    control_cmd_pose=control_cmd_pose,
                )
                action_count += 1
                runtime_state.consumed_steps += 1
                if runtime_state.consumed_steps % 50 == 0:
                    logger.info("[ACTOR] step=%d queue=%d", runtime_state.consumed_steps, action_queue.qsize())

            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, (action_interval - dt_s) - 0.001))

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception:
        print("\n" + "=" * 80, file=sys.stderr, flush=True)
        print("[ACTOR] FATAL EXCEPTION", file=sys.stderr, flush=True)
        traceback.print_exc()
        print("=" * 80 + "\n", file=sys.stderr, flush=True)
        shutdown_event.set()
        return


@parser.wrap()
def eval_offline_eef_model(cfg: OfflineEEFConfig):
    init_logging()
    logger.info("[OFFLINE] device=%s", cfg.device)
    logger.info("[OFFLINE] dataset=%s", cfg.dataset_path)
    if not rospy.get_node_uri():
        rospy.init_node("eval_offline_eef_model", anonymous=True)
        logger.info("[OFFLINE] ROS node initialized")

    claw_lock_threshold = float("inf") if cfg.disable_claw_lock else cfg.claw_lock_threshold
    env = GrabBoxMpcEnv(
        claw_lock_threshold=claw_lock_threshold,
        claw_lock_count_threshold=cfg.claw_lock_count_threshold,
        claw_locked_value=cfg.claw_locked_value,
    )
    if cfg.disable_claw_lock:
        logger.info("[OFFLINE] Claw lock mechanism: DISABLED")
    else:
        logger.info(
            "[OFFLINE] Claw lock mechanism: threshold=%s, count_threshold=%s, locked_value=%s",
            cfg.claw_lock_threshold,
            cfg.claw_lock_count_threshold,
            cfg.claw_locked_value,
        )

    left_arm_lock_values = None
    right_arm_lock_values = None
    if cfg.lock_left_arm:
        left_arm_lock_values = _load_arm_lock_values(cfg.left_arm_lock_json_path, arm_side="left")
        if left_arm_lock_values is None:
            logger.warning("[OFFLINE] Left arm lock requested but disabled due to invalid lock JSON.")
        else:
            logger.info("[OFFLINE] Left arm lock ENABLED.")
    else:
        logger.info("[OFFLINE] Left arm lock DISABLED.")

    if cfg.lock_right_arm:
        right_arm_lock_values = _load_arm_lock_values(cfg.right_arm_lock_json_path, arm_side="right")
        if right_arm_lock_values is None:
            logger.warning("[OFFLINE] Right arm lock requested but disabled due to invalid lock JSON.")
        else:
            logger.info("[OFFLINE] Right arm lock ENABLED.")
    else:
        logger.info("[OFFLINE] Right arm lock DISABLED.")

    model_bundle = load_model_bundle(model_path=cfg.model_path, cfg=cfg, device=cfg.device)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    reset_arm_json_abs = cfg.reset_arm_json_path
    if not os.path.isabs(reset_arm_json_abs):
        reset_arm_json_abs = os.path.join(cur_dir, cfg.reset_arm_json_path)

    while True:
        play_reset_arm_trajectory_from_json(
            env,
            reset_arm_json_abs,
            control_arm=True,
            control_claw=True,
        )
        user_input = input("按回车开始推理，输入 q 退出: ").strip().lower()
        if user_input == "q":
            logger.info("[OFFLINE] User requested exit before run.")
            break

        model_bundle.reset()
        episode_state = EpisodeState(action_dim=model_bundle.action_dim)
        runtime_state = OfflineRuntimeState()
        shutdown_event = Event()
        action_lock = Lock()

        obs_stream = FakeObsStream(
            dataset_path=cfg.dataset_path,
            fps=cfg.fps,
            camera_components=CAMERA_COMPONENTS,
            shuffle=False,
            loop=cfg.loop_dataset,
            episodes=cfg.episodes,
        )

        action_queue = ActionQueue(cfg.rtc)
        get_actions_thread = Thread(
            target=get_actions,
            args=(
                model_bundle,
                env,
                obs_stream,
                action_queue,
                shutdown_event,
                cfg,
                episode_state,
                runtime_state,
                action_lock,
            ),
            daemon=True,
            name="GetActions",
        )
        actor_thread = Thread(
            target=actor_control,
            args=(
                env,
                action_queue,
                shutdown_event,
                cfg,
                episode_state,
                runtime_state,
                action_lock,
                left_arm_lock_values,
                right_arm_lock_values,
            ),
            daemon=True,
            name="Actor",
        )

        get_actions_thread.start()
        actor_thread.start()
        get_actions_thread.join()
        actor_thread.join()

        logger.info(
            "[OFFLINE] finished. consumed_steps=%d produced_chunks=%d reason=%s",
            runtime_state.consumed_steps,
            runtime_state.produced_chunks,
            runtime_state.stop_reason or "completed",
        )

        user_input = input("按回车开始下一轮，输入 q 退出: ").strip().lower()
        if user_input == "q":
            logger.info("[OFFLINE] User requested exit.")
            break


if __name__ == "__main__":
    eval_offline_eef_model()
