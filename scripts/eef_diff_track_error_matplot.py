# -*- coding: utf-8 -*-
"""
1. 读取 LeRobotDataset 指定 episode 的所有帧 state/action。
2. 计算左右手 EEF command(action) 与 state 的跟踪误差。
3. 绘制 position 和 6d rotation 的误差曲线。

单位说明：
- position 相关误差：米（m）
- rotation6d L2 误差：无量纲（dimensionless）
- rotation angle 误差：角度（deg）
"""
import argparse
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE


def resolve_dataset_root(root: Path | None, repo_id: str) -> Path | None:
    """
    兼容两种传参：
    1) root 直接是数据集目录（包含 meta/info.json）
    2) root 是父目录，实际数据在 root/repo_id
    """
    if root is None:
        return None

    root = Path(root)
    if (root / "meta" / "info.json").exists():
        return root

    candidate = root / repo_id
    if (candidate / "meta" / "info.json").exists():
        return candidate

    return root


class EpisodeSampler(torch.utils.data.Sampler):
    """与 lerobot_dataset_viz.py 一致：只采样一个 episode 的 frame id。"""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
        to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

def reconstruct_rotation_matrix_6d(rotation_6d):
    """
    从6D表示重构3x3旋转矩阵
    使用Gram-Schmidt正交化确保是有效的旋转矩阵
    #        | R11  R12  R13 |
    #    R = | R21  R22  R23 |
    #        | R31  R32  R33 |
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

def rotation_matrix_to_6d(rotation_matrix):
    """
    将3x3旋转矩阵转换为6D表示
    
    Args:
        rotation_matrix: 3x3旋转矩阵
    #        | R11  R12  R13 |
    #    R = | R21  R22  R23 |
    #        | R31  R32  R33 |
    Returns:
        6维向量 [R11, R21, R31, R12, R22, R32]
    """
    # 提取前两列
    col1 = rotation_matrix[:, 0]  # [R11, R21, R31]
    col2 = rotation_matrix[:, 1]  # [R12, R22, R32]
    
    # 组合成6D表示
    rotation_6d = np.concatenate([col1, col2])
    
    return rotation_6d


def rotation_angle_error_deg(cmd_rot6d: np.ndarray, state_rot6d: np.ndarray) -> float:
    """通过相对旋转矩阵计算角度误差（degree）。"""
    r_cmd = reconstruct_rotation_matrix_6d(cmd_rot6d)
    r_state = reconstruct_rotation_matrix_6d(state_rot6d)
    r_rel = r_cmd @ r_state.T
    trace = np.trace(r_rel)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    return float(np.degrees(theta_rad))


def compute_eef_tracking_errors(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    """按帧计算你定义的左右手 position 与 6d tracking error（含单位约定）。"""
    sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=sampler,
    )

    frame_index = []
    left_pos_l2 = []
    right_pos_l2 = []
    left_pos_axis_abs = []
    right_pos_axis_abs = []
    left_rot6d_l2 = []
    right_rot6d_l2 = []
    left_rot_angle_deg = []
    right_rot_angle_deg = []

    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="Computing EEF errors"):
        if ACTION not in batch or OBS_STATE not in batch:
            raise KeyError(f"batch must contain `{ACTION}` and `{OBS_STATE}`")

        action_np = batch[ACTION].cpu().numpy()
        state_np = batch[OBS_STATE].cpu().numpy()
        frame_idx_np = batch["frame_index"].cpu().numpy()

        for i in range(action_np.shape[0]):
            action = action_np[i]
            state = state_np[i]

            # 索引定义（按你的说明）
            # 左手 position: [0:3], 右手 position: [9:12]
            # 左手 6d: [3:9], 右手 6d: [12:18]
            l_pos_cmd = action[0:3]
            l_pos_state = state[0:3]
            l_rot_cmd = action[3:9]
            l_rot_state = state[3:9]

            r_pos_cmd = action[9:12]
            r_pos_state = state[9:12]
            r_rot_cmd = action[12:18]
            r_rot_state = state[12:18]

            frame_index.append(int(frame_idx_np[i]))

            left_pos_l2.append(float(np.linalg.norm(l_pos_cmd - l_pos_state)))
            right_pos_l2.append(float(np.linalg.norm(r_pos_cmd - r_pos_state)))
            left_pos_axis_abs.append(np.abs(l_pos_cmd - l_pos_state))
            right_pos_axis_abs.append(np.abs(r_pos_cmd - r_pos_state))
            left_rot6d_l2.append(float(np.linalg.norm(l_rot_cmd - l_rot_state)))
            right_rot6d_l2.append(float(np.linalg.norm(r_rot_cmd - r_rot_state)))
            left_rot_angle_deg.append(rotation_angle_error_deg(l_rot_cmd, l_rot_state))
            right_rot_angle_deg.append(rotation_angle_error_deg(r_rot_cmd, r_rot_state))

    return {
        "frame_index": np.asarray(frame_index),
        "left_pos_l2": np.asarray(left_pos_l2),
        "right_pos_l2": np.asarray(right_pos_l2),
        "left_pos_axis_abs": np.asarray(left_pos_axis_abs),   # shape [N, 3]: x/y/z
        "right_pos_axis_abs": np.asarray(right_pos_axis_abs), # shape [N, 3]: x/y/z
        "left_rot6d_l2": np.asarray(left_rot6d_l2),
        "right_rot6d_l2": np.asarray(right_rot6d_l2),
        "left_rot_angle_deg": np.asarray(left_rot_angle_deg),
        "right_rot_angle_deg": np.asarray(right_rot_angle_deg),
    }


def plot_eef_tracking_errors(
    errors: dict[str, np.ndarray],
    repo_id: str,
    episode_index: int,
    save_path: Path | None,
):
    x = errors["frame_index"]

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(
        f"EEF tracking error | {repo_id} | episode={episode_index} "
        "(pos: m, rot6d_l2: unitless, rot_angle: deg)",
        fontsize=14,
    )

    axes[0].plot(x, errors["left_pos_l2"], label="left_pos_l2")
    axes[0].plot(x, errors["right_pos_l2"], label="right_pos_l2")
    axes[0].set_ylabel("Position L2 (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    left_axis = errors["left_pos_axis_abs"]
    right_axis = errors["right_pos_axis_abs"]
    axes[1].plot(x, left_axis[:, 0], label="left_pos_abs_x")
    axes[1].plot(x, left_axis[:, 1], label="left_pos_abs_y")
    axes[1].plot(x, left_axis[:, 2], label="left_pos_abs_z")
    axes[1].set_ylabel("Left Axis Abs Err (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(x, right_axis[:, 0], label="right_pos_abs_x")
    axes[2].plot(x, right_axis[:, 1], label="right_pos_abs_y")
    axes[2].plot(x, right_axis[:, 2], label="right_pos_abs_z")
    axes[2].set_ylabel("Right Axis Abs Err (m)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(x, errors["left_rot6d_l2"], label="left_rot6d_l2")
    axes[3].plot(x, errors["right_rot6d_l2"], label="right_rot6d_l2")
    axes[3].set_ylabel("Rotation6D L2 (unitless)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    axes[4].plot(x, errors["left_rot_angle_deg"], label="left_rot_angle_deg")
    axes[4].plot(x, errors["right_rot_angle_deg"], label="right_rot_angle_deg")
    axes[4].set_ylabel("Rotation Angle Error (deg)")
    axes[4].set_xlabel("frame_index")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
        print(f"Saved figure to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot EEF tracking errors from LeRobotDataset episode "
            "(position: m, rotation6d_l2: unitless, rotation_angle: deg)."
        )
    )
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset repo id")
    parser.add_argument("--episode-index", type=int, required=True, help="Episode index to analyze")
    parser.add_argument("--root", type=Path, default=None, help="Local dataset root directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Dataloader batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--tolerance-s", type=float, default=1e-4, help="LeRobotDataset tolerance_s")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Output png path, e.g. ./outputs/eef_error_episode10.png",
    )
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.root, args.repo_id)
    dataset = LeRobotDataset(
        args.repo_id,
        episodes=[args.episode_index],
        root=dataset_root,
        tolerance_s=args.tolerance_s,
    )

    errors = compute_eef_tracking_errors(
        dataset=dataset,
        episode_index=args.episode_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    plot_eef_tracking_errors(
        errors=errors,
        repo_id=args.repo_id,
        episode_index=args.episode_index,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()