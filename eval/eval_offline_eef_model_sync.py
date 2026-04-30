#!/usr/bin/env python
"""
Offline synchronous inference for EEF model.

Sync semantics:
  1) get one observation (dataset frame) for the chunk
  2) predict the whole action chunk
  3) execute the whole chunk at control frequency
  4) advance the dataset stream to the next chunk boundary
  5) repeat
"""

import os
import sys
import time
import math
import logging
import traceback
from typing import Optional

from threading import Event

import numpy as np
import torch
import rospy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lerobot.utils.utils import init_logging
from lerobot.configs import parser

from configs.config import get_camera_observation_key, get_camera_names, CAMERA_COMPONENTS, ACTION_COMPONENTS

from robot_envs.kuavo_depalletize_env import GrabBoxMpcEnv

from eval_eef_model import (
    load_model_bundle,
    EpisodeState,
    convert_eef_action_to_joint_action,
    resample_chunk_with_claw_hold,
    apply_first_chunk_smooth,
    _load_arm_lock_values,
)
from eval_offline_eef_model import (
    OfflineEEFConfig,
    FakeObsStream,
    play_reset_arm_trajectory_from_json,
    _build_state_for_model,
)

logger = logging.getLogger(__name__)


def _convert_relative_actions_if_needed_no_print(
    model_bundle,
    episode_state: EpisodeState,
    postprocessed_actions: torch.Tensor,
) -> torch.Tensor:
    """Convert relative Delta-eef actions to absolute EEF actions (no debug prints)."""
    if not model_bundle.is_relative_action_mode or model_bundle.postprocessor_step is None:
        return postprocessed_actions

    if episode_state.current_reference_pose is None:
        episode_state.current_reference_pose = np.zeros(model_bundle.action_dim, dtype=np.float32)

    pred_chunk_tensor = postprocessed_actions.unsqueeze(0)  # (1, chunk_size, action_dim)
    reference_pose_tensor = torch.from_numpy(episode_state.current_reference_pose).to(
        pred_chunk_tensor.device
    ).unsqueeze(0)  # (1, action_dim)

    pred_chunk_absolute = model_bundle.postprocessor_step._convert_relative_to_absolute_eef_action(
        pred_chunk_tensor, reference_pose_tensor
    )
    return pred_chunk_absolute[0].clone()


def _exec_action_step(
    env: GrabBoxMpcEnv,
    action_step: torch.Tensor,
    *,
    control_arm: bool,
    control_claw: bool,
    control_cmd_pose: bool,
    left_arm_lock_values: Optional[np.ndarray],
    right_arm_lock_values: Optional[np.ndarray],
    episode_state: EpisodeState,
) -> None:
    """Execute one control-step action and record last_executed_action."""
    action_cpu = action_step.detach().cpu()
    action_np = action_cpu.numpy().copy()

    if action_np.size >= 14:
        if left_arm_lock_values is not None:
            action_np[0:7] = left_arm_lock_values
        if right_arm_lock_values is not None:
            action_np[7:14] = right_arm_lock_values

    action_to_exec = torch.from_numpy(action_np)
    episode_state.last_executed_action = action_to_exec.clone()

    env.exec_actions(
        actions=action_to_exec,
        control_arm=control_arm,
        control_claw=control_claw,
        control_cmd_pose=control_cmd_pose,
    )


@parser.wrap()
def eval_offline_eef_model_sync(cfg: OfflineEEFConfig):
    init_logging()
    logger.info("[OFFLINE_SYNC] device=%s", cfg.device)
    logger.info("[OFFLINE_SYNC] dataset=%s", cfg.dataset_path)

    if not rospy.get_node_uri():
        rospy.init_node("eval_offline_eef_model_sync", anonymous=True)
        logger.info("[OFFLINE_SYNC] ROS node initialized")

    claw_lock_threshold = float("inf") if cfg.disable_claw_lock else cfg.claw_lock_threshold
    env = GrabBoxMpcEnv(
        claw_lock_threshold=claw_lock_threshold,
        claw_lock_count_threshold=cfg.claw_lock_count_threshold,
        claw_locked_value=cfg.claw_locked_value,
    )

    if cfg.disable_claw_lock:
        logger.info("[OFFLINE_SYNC] Claw lock mechanism: DISABLED")
    else:
        logger.info(
            "[OFFLINE_SYNC] Claw lock mechanism: threshold=%s, count_threshold=%s, locked_value=%s",
            cfg.claw_lock_threshold,
            cfg.claw_lock_count_threshold,
            cfg.claw_locked_value,
        )

    left_arm_lock_values = None
    right_arm_lock_values = None

    if cfg.lock_left_arm:
        left_arm_lock_values = _load_arm_lock_values(cfg.left_arm_lock_json_path, arm_side="left")
        if left_arm_lock_values is None:
            logger.warning("[OFFLINE_SYNC] Left arm lock requested but disabled due to invalid lock JSON.")
        else:
            logger.info("[OFFLINE_SYNC] Left arm lock ENABLED.")
    else:
        logger.info("[OFFLINE_SYNC] Left arm lock DISABLED.")

    if cfg.lock_right_arm:
        right_arm_lock_values = _load_arm_lock_values(cfg.right_arm_lock_json_path, arm_side="right")
        if right_arm_lock_values is None:
            logger.warning("[OFFLINE_SYNC] Right arm lock requested but disabled due to invalid lock JSON.")
        else:
            logger.info("[OFFLINE_SYNC] Right arm lock ENABLED.")
    else:
        logger.info("[OFFLINE_SYNC] Right arm lock DISABLED.")

    model_bundle = load_model_bundle(model_path=cfg.model_path, cfg=cfg, device=cfg.device)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    reset_arm_json_abs = cfg.reset_arm_json_path
    if not os.path.isabs(reset_arm_json_abs):
        reset_arm_json_abs = os.path.join(cur_dir, cfg.reset_arm_json_path)

    # control timing (match eval_offline_eef_model)
    control_frequency = 100.0
    control_dt = 1.0 / control_frequency
    action_interval = control_dt
    control_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)

    while True:
        play_reset_arm_trajectory_from_json(
            env,
            reset_arm_json_abs,
            control_arm=True,
            control_claw=True,
        )

        user_input = input("按回车开始同步推理，输入 q 退出: ").strip().lower()
        if user_input == "q":
            logger.info("[OFFLINE_SYNC] User requested exit before run.")
            break

        model_bundle.reset()
        episode_state = EpisodeState(action_dim=model_bundle.action_dim)

        # No threads in sync mode: one loop does inference + full chunk execution.
        shutdown_event = Event()

        runtime_consumed_steps = 0
        produced_chunks = 0

        obs_stream = FakeObsStream(
            dataset_path=cfg.dataset_path,
            fps=cfg.fps,
            camera_components=CAMERA_COMPONENTS,
            shuffle=False,
            loop=cfg.loop_dataset,
            episodes=cfg.episodes,
        )

        try:
            obs_data, *_ = obs_stream.get_obs()
        except StopIteration:
            logger.info("[OFFLINE_SYNC] dataset_exhausted before start.")
            break

        try:
            while not shutdown_event.is_set() and runtime_consumed_steps < cfg.max_steps:
                # --------------------------
                # 1) predict one chunk
                # --------------------------
                observation = {}
                for camera_name in get_camera_names(CAMERA_COMPONENTS):
                    if camera_name not in obs_data:
                        continue
                    camera_img_np = obs_data[camera_name]
                    if camera_img_np.ndim != 4:
                        continue
                    camera_images = (
                        torch.from_numpy(np.moveaxis(camera_img_np, 3, 1).copy()).float() / 255.0
                    )
                    obs_key = get_camera_observation_key(camera_name, use_image_features=False)
                    observation[obs_key] = camera_images.to(cfg.device)

                state = _build_state_for_model(model_bundle, obs_data, episode_state)
                observation["observation.state"] = state.to(cfg.device)
                observation["task"] = cfg.task
                processed_observation = model_bundle.preprocessor(observation)

                with torch.inference_mode():
                    actions = model_bundle.policy.predict_action_chunk(
                        processed_observation,
                        inference_delay=0,
                        prev_chunk_left_over=None,
                    )

                _, chunk_size, _ = actions.shape

                processed_actions = []
                for i in range(chunk_size):
                    processed_actions.append(model_bundle.postprocessor(actions[:, i, :]))

                postprocessed_actions = torch.stack(processed_actions, dim=1)[0].clone()

                postprocessed_actions = _convert_relative_actions_if_needed_no_print(
                    model_bundle, episode_state, postprocessed_actions
                )

                # Convert EEF -> joint if needed (same as eval_offline_eef_model.get_actions)
                if model_bundle.action_dim == 20:
                    post_np = postprocessed_actions.detach().cpu().numpy()
                    joint_actions = [
                        convert_eef_action_to_joint_action(a, model_type=cfg.ik_model_type) for a in post_np
                    ]
                    postprocessed_actions = torch.from_numpy(np.asarray(joint_actions)).to(cfg.device)
                elif episode_state.first_inference:
                    postprocessed_actions = apply_first_chunk_smooth(
                        postprocessed_actions, obs_data, env, model_bundle.action_dim, ik_model_type=cfg.ik_model_type
                    )
                    episode_state.first_inference = False

                previous_action_np = (
                    episode_state.last_executed_action.detach().cpu().numpy()
                    if episode_state.last_executed_action is not None
                    else None
                )

                # --------------------------
                # 2) resample to control dt
                # --------------------------
                postprocessed_resampled = resample_chunk_with_claw_hold(
                    postprocessed_actions.detach().cpu().numpy(),
                    previous_action=previous_action_np,
                    control_frequency=control_frequency,
                    source_dt=0.1,
                    arm_dims=slice(0, 14),
                    claw_dims=slice(14, 16),
                    device=torch.device(cfg.device),
                )

                # --------------------------
                # 3) execute the full chunk
                # --------------------------
                chunk_len = int(postprocessed_resampled.shape[0])
                logger.info(
                    "[OFFLINE_SYNC] chunk=%d predict_chunk_actions=%d execute_control_steps=%d queue=0",
                    produced_chunks,
                    int(chunk_size),
                    chunk_len,
                )

                # clamp if max_steps would overshoot
                if runtime_consumed_steps + chunk_len > cfg.max_steps:
                    postprocessed_resampled = postprocessed_resampled[: max(0, cfg.max_steps - runtime_consumed_steps)]
                    chunk_len = int(postprocessed_resampled.shape[0])

                for action_step in postprocessed_resampled:
                    if shutdown_event.is_set():
                        break
                    t0 = time.perf_counter()
                    _exec_action_step(
                        env,
                        action_step,
                        control_arm=True,
                        control_claw=True,
                        control_cmd_pose=control_cmd_pose,
                        left_arm_lock_values=left_arm_lock_values,
                        right_arm_lock_values=right_arm_lock_values,
                        episode_state=episode_state,
                    )
                    dt_s = time.perf_counter() - t0
                    time.sleep(max(0.0, max(0.0, action_interval - dt_s) - 0.001))
                    runtime_consumed_steps += 1

                produced_chunks += 1

                if runtime_consumed_steps >= cfg.max_steps:
                    break

                # --------------------------
                # 4) advance dataset stream to next chunk boundary
                # --------------------------
                # FakeObsStream advances by 1 frame per get_obs() call.
                dataset_dt = 1.0 / float(cfg.fps)
                executed_duration = max(0.0, (chunk_len - 1) * control_dt)
                frames_to_advance = int(round(executed_duration / dataset_dt))

                # always move forward at least 1 frame to avoid potential infinite loops
                frames_to_advance = max(1, frames_to_advance)

                try:
                    # First consume frames_to_advance-1 frames as "discard", then keep the last as next obs.
                    for _ in range(frames_to_advance):
                        obs_data, *_ = obs_stream.get_obs()
                except StopIteration:
                    logger.info("[OFFLINE_SYNC] dataset_exhausted after chunk execution.")
                    break

        except KeyboardInterrupt:
            logger.info("[OFFLINE_SYNC] Interrupted by user.")
            break
        except Exception:
            print("\n" + "=" * 80, file=sys.stderr, flush=True)
            print("[OFFLINE_SYNC] FATAL EXCEPTION", file=sys.stderr, flush=True)
            traceback.print_exc()
            print("=" * 80 + "\n", file=sys.stderr, flush=True)
            break

        logger.info(
            "[OFFLINE_SYNC] finished. consumed_steps=%d produced_chunks=%d",
            runtime_consumed_steps,
            produced_chunks,
        )

        user_input = input("按回车开始下一轮（会重新reset arm），输入 q 退出: ").strip().lower()
        if user_input == "q":
            logger.info("[OFFLINE_SYNC] User requested exit.")
            break


if __name__ == "__main__":
    eval_offline_eef_model_sync()
