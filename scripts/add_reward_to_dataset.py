#!/usr/bin/env python3
"""
为 LeRobot 数据集添加 reward 字段的工具脚本。

支持多种奖励方案:
1. sparse_final: 稀疏终末奖励 (最后一帧 r=1, 其余 r=0) — π*0.6 论文标准方式
2. progress_linear: 线性进度奖励 (r_t = t / T) — 适用于全正样本场景
3. progress_exp: 指数递增奖励 (r_t = exp(α * t/T) / exp(α)) — 更强调末尾
4. time_penalty: 时间惩罚奖励 (更快完成的 episode 获得更高奖励)
5. custom_fn: 自定义函数（传入回调函数）

使用示例:
    python scripts/add_reward_to_dataset.py \
        --dataset_path /path/to/dataset \
        --reward_type sparse_final \
        --output_path /path/to/output

    # 或使用进度奖励（推荐用于全正样本场景）
    python scripts/add_reward_to_dataset.py \
        --dataset_path /path/to/dataset \
        --reward_type progress_linear \
        --output_path /path/to/output
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# 奖励函数定义
# ============================================================================


def sparse_final_reward(frame_index: int, episode_length: int, **kwargs) -> float:
    """稀疏终末奖励: 最后一帧 r=1, 其余 r=0。

    这是 π*0.6 论文的标准方式，但在全正样本场景下效果有限。

    Return 分布: R_t = 1 (对所有 t，因为 Σ_{t'=t}^T r_{t'} = r_T = 1)
    """
    if frame_index == episode_length - 1:
        return 1.0
    return 0.0


def progress_linear_reward(frame_index: int, episode_length: int, **kwargs) -> float:
    """线性进度奖励: r_t = (t + 1) / T。

    越接近终点的 frame 奖励越高。
    这样 Return_t = Σ_{t'=t}^T (t'+1)/T 会对不同时间步产生不同的 Return，
    Value Function 可以学到有意义的状态价值梯度。

    推荐用于全正样本的 demonstration 数据集。
    """
    return (frame_index + 1) / episode_length


def progress_exp_reward(frame_index: int, episode_length: int, alpha: float = 3.0, **kwargs) -> float:
    """指数递增进度奖励: r_t = exp(α * (t+1)/T) / exp(α)。

    比线性进度更加强调接近任务完成的时刻。
    α 越大，奖励越集中在 episode 末尾。
    """
    normalized_t = (frame_index + 1) / episode_length
    return float(np.exp(alpha * normalized_t) / np.exp(alpha))


def time_penalty_reward(
    frame_index: int,
    episode_length: int,
    max_episode_length: int = None,
    **kwargs,
) -> float:
    """时间惩罚奖励: 更短的 episode 整体获得更高的奖励。

    r_t = 1 - episode_length / max_episode_length (对最后一帧)
    r_t = 0 (对其他帧)

    或者连续版本: r_t = (1 - episode_length / max_length) / episode_length + sparse_end

    这种奖励让 Value Function 能区分不同效率的 episode。
    """
    if max_episode_length is None:
        max_episode_length = 200  # 默认最大长度

    # 效率得分: episode 越短越好
    efficiency = 1.0 - (episode_length / max_episode_length)
    efficiency = max(0.0, efficiency)  # 不要负奖励

    # 每帧给一个小的效率奖励 + 最后一帧给成功奖励
    per_step_reward = efficiency / episode_length
    final_bonus = 1.0 if frame_index == episode_length - 1 else 0.0

    return per_step_reward + final_bonus


def dense_distance_reward(frame_index: int, episode_length: int, **kwargs) -> float:
    """密集距离递减奖励: r_t = 1 - t/T * (1 - 1/T)。

    第一帧 r≈1/T，最后帧 r=1。
    提供一个持续递增的奖励信号。
    """
    if episode_length <= 1:
        return 1.0
    return frame_index / (episode_length - 1)


REWARD_FUNCTIONS = {
    "sparse_final": sparse_final_reward,
    "progress_linear": progress_linear_reward,
    "progress_exp": progress_exp_reward,
    "time_penalty": time_penalty_reward,
    "dense_distance": dense_distance_reward,
}


# ============================================================================
# Return (目标价值) 计算
# ============================================================================


def compute_episode_returns(rewards: list[float]) -> list[float]:
    """计算单个 episode 每个时间步的 Return (累积奖励)。

    Return_t = Σ_{t'=t}^{T} r_{t'}

    这是 Value Function 的训练目标:
        V^π(o_t, ℓ) ← argmin_V E[(V(o_t, ℓ) - Return_t)²]

    Args:
        rewards: 单个 episode 的奖励序列 [r_0, r_1, ..., r_T]

    Returns:
        returns: 每个时间步的 Return [R_0, R_1, ..., R_T]
    """
    T = len(rewards)
    returns = [0.0] * T
    cumsum = 0.0
    for t in range(T - 1, -1, -1):
        cumsum += rewards[t]
        returns[t] = cumsum
    return returns


# ============================================================================
# 数据集处理
# ============================================================================


def analyze_episodes(dataset_path: Path) -> dict:
    """分析数据集的 episode 结构。"""
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    episode_info = {}
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        ep_idx = df["episode_index"].iloc[0]
        episode_info[ep_idx] = {
            "length": len(df),
            "file_path": pf,
            "frame_start": df["frame_index"].min(),
            "frame_end": df["frame_index"].max(),
        }

    return episode_info


def add_reward_to_dataset(
    dataset_path: Path,
    output_path: Path,
    reward_type: str = "sparse_final",
    reward_kwargs: dict = None,
    inplace: bool = False,
):
    """为数据集添加 reward 字段。

    Args:
        dataset_path: 原始数据集路径
        output_path: 输出数据集路径
        reward_type: 奖励类型
        reward_kwargs: 奖励函数的额外参数
        inplace: 是否就地修改（修改原始数据集）
    """
    if reward_kwargs is None:
        reward_kwargs = {}

    reward_fn = REWARD_FUNCTIONS.get(reward_type)
    if reward_fn is None:
        raise ValueError(f"Unknown reward type: {reward_type}. Available: {list(REWARD_FUNCTIONS.keys())}")

    print(f"📊 分析数据集: {dataset_path}")
    episode_info = analyze_episodes(dataset_path)
    max_episode_length = max(ep["length"] for ep in episode_info.values())
    print(f"   总 episodes: {len(episode_info)}")
    print(f"   最长 episode: {max_episode_length} frames")
    print(f"   最短 episode: {min(ep['length'] for ep in episode_info.values())} frames")
    print(f"   奖励类型: {reward_type}")

    # 创建输出目录
    if inplace:
        output_path = dataset_path
    else:
        if output_path.exists():
            print(f"⚠️  输出目录已存在: {output_path}")
            print(f"   将覆盖现有内容...")
            shutil.rmtree(output_path)
        # 复制整个数据集
        print(f"📁 复制数据集到: {output_path}")
        shutil.copytree(dataset_path, output_path)

    # 处理每个 parquet 文件
    data_dir = output_path / "data"
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    print(f"\n🔄 为 {len(parquet_files)} 个文件添加 reward 和 target_value 字段...")
    reward_stats = {"total_frames": 0, "total_reward": 0.0, "rewards": [], "target_values": []}

    for pf in tqdm(parquet_files, desc="Processing"):
        df = pd.read_parquet(pf)
        ep_idx = df["episode_index"].iloc[0]
        ep_length = episode_info[ep_idx]["length"]

        rewards = []
        for _, row in df.iterrows():
            r = reward_fn(
                frame_index=int(row["frame_index"]),
                episode_length=ep_length,
                max_episode_length=max_episode_length,
                **reward_kwargs,
            )
            rewards.append(r)

        # 计算 target_value (Return): R_t = Σ_{t'=t}^T r_{t'}
        # 这是 Value Function 的训练目标
        target_values = compute_episode_returns(rewards)

        df["reward"] = rewards
        df["target_value"] = target_values
        df.to_parquet(pf, index=False)

        reward_stats["total_frames"] += len(rewards)
        reward_stats["total_reward"] += sum(rewards)
        reward_stats["rewards"].extend(rewards)
        reward_stats["target_values"].extend(target_values)

    # 更新 info.json
    info_path = output_path / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)

    features_updated = False
    if "reward" not in info["features"]:
        info["features"]["reward"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
            "fps": info["fps"],
        }
        features_updated = True

    if "target_value" not in info["features"]:
        info["features"]["target_value"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
            "fps": info["fps"],
        }
        features_updated = True

    if features_updated:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print("✅ info.json 已更新，添加了 reward 和 target_value 特征")

    # 打印统计
    rewards_array = np.array(reward_stats["rewards"])
    target_values_array = np.array(reward_stats["target_values"])
    print(f"\n📈 奖励统计:")
    print(f"   总帧数: {reward_stats['total_frames']}")
    print(f"   平均奖励: {rewards_array.mean():.4f}")
    print(f"   奖励标准差: {rewards_array.std():.4f}")
    print(f"   最小奖励: {rewards_array.min():.4f}")
    print(f"   最大奖励: {rewards_array.max():.4f}")
    print(f"   非零奖励比例: {(rewards_array > 0).mean():.2%}")

    print(f"\n📊 Target Value (Return) 统计:")
    print(f"   平均 target_value: {target_values_array.mean():.4f}")
    print(f"   target_value 标准差: {target_values_array.std():.4f}")
    print(f"   target_value 范围: [{target_values_array.min():.4f}, {target_values_array.max():.4f}]")

    # 显示 Return 分布分析
    print(f"\n📊 Episode Return (t=0 处的 target_value) 分布分析:")
    returns_per_episode = []
    for ep_idx, ep in episode_info.items():
        ep_rewards = rewards_array[
            sum(episode_info[i]["length"] for i in range(ep_idx) if i in episode_info) :
            sum(episode_info[i]["length"] for i in range(ep_idx + 1) if i in episode_info)
        ]
        if len(ep_rewards) > 0:
            # Return at t=0 (sum of all rewards in the episode)
            ep_return = ep_rewards.sum()
            returns_per_episode.append(ep_return)

    returns_arr = np.array(returns_per_episode) if returns_per_episode else np.array([0.0])
    print(f"   Episode Return 均值: {returns_arr.mean():.4f}")
    print(f"   Episode Return 标准差: {returns_arr.std():.4f}")
    print(f"   Episode Return 范围: [{returns_arr.min():.4f}, {returns_arr.max():.4f}]")

    if returns_arr.std() < 1e-6:
        print("\n⚠️  警告: 所有 Episode 的 Return 几乎相同！")
        print("   这意味着 Value Function 将无法学到有意义的区分。")
        print("   建议使用 progress_linear 或 time_penalty 奖励代替 sparse_final。")
    else:
        print(f"\n✅ Episode Return 有足够的方差 (std={returns_arr.std():.4f})，")
        print(f"   Value Function 可以学到有意义的状态价值区分。")
        print(f"   ✅ target_value 已预计算，可以直接用于 Value Function 训练。")

    print(f"\n✅ 完成! 数据集已保存到: {output_path}")
    return output_path


def verify_reward(dataset_path: Path, num_episodes: int = 5):
    """验证数据集中的 reward 和 target_value 字段。"""
    print(f"\n🔍 验证数据集 reward/target_value 字段: {dataset_path}")
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("*/*.parquet"))[:num_episodes]

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        ep_idx = df["episode_index"].iloc[0]

        if "reward" not in df.columns:
            print(f"  ❌ Episode {ep_idx}: 没有 reward 列!")
            continue

        rewards = df["reward"].values
        print(f"  ✅ Episode {ep_idx}: length={len(df)}, "
              f"reward_sum={rewards.sum():.4f}, "
              f"reward_range=[{rewards.min():.4f}, {rewards.max():.4f}], "
              f"first_5={rewards[:5].tolist()}, "
              f"last_5={rewards[-5:].tolist()}")

        if "target_value" in df.columns:
            target_values = df["target_value"].values
            print(f"       target_value: range=[{target_values.min():.4f}, {target_values.max():.4f}], "
                  f"first_3={target_values[:3].tolist()}, "
                  f"last_3={target_values[-3:].tolist()}")
        else:
            print(f"  ⚠️  Episode {ep_idx}: 没有 target_value 列 (需要重新运行 add_reward)")


# ============================================================================
# 主函数
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="为 LeRobot 数据集添加 reward 字段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
奖励类型说明:
  sparse_final     - 稀疏终末奖励: 最后一帧 r=1, 其余 r=0
                     ⚠️ 全正样本时: Return 全为 1, Value Function 无法区分
  
  progress_linear  - 线性进度奖励: r_t = (t+1)/T
                     ✅ 推荐: 不同时间步有不同 Return, 且不同长度 episode 有不同总 Return
  
  progress_exp     - 指数递增奖励: r_t = exp(α*(t+1)/T) / exp(α)
                     更强调末尾阶段，α=3.0 适用于需要精确末端操作的任务
  
  time_penalty     - 时间惩罚奖励: 快速完成的 episode 获得更高奖励
                     适用于效率优先的任务
  
  dense_distance   - 密集距离奖励: r_t = t / (T-1)
                     持续递增的信号，提供平滑的价值梯度
        """,
    )

    parser.add_argument("--dataset_path", type=str, required=True, help="输入数据集路径")
    parser.add_argument("--output_path", type=str, default=None, help="输出数据集路径 (默认: 输入路径_with_reward)")
    parser.add_argument(
        "--reward_type",
        type=str,
        default="sparse_final",
        choices=list(REWARD_FUNCTIONS.keys()),
        help="奖励类型 (默认: sparse_final)",
    )
    parser.add_argument("--inplace", action="store_true", help="就地修改（直接修改原始数据集）")
    parser.add_argument("--verify", action="store_true", help="处理后验证 reward 字段")
    parser.add_argument("--verify_only", action="store_true", help="只验证，不添加 reward")
    parser.add_argument("--alpha", type=float, default=3.0, help="指数奖励的 α 参数")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        sys.exit(1)

    if args.verify_only:
        verify_reward(dataset_path)
        return

    if args.output_path is None:
        args.output_path = str(dataset_path) + f"_with_{args.reward_type}_reward"
    output_path = Path(args.output_path)

    reward_kwargs = {}
    if args.reward_type == "progress_exp":
        reward_kwargs["alpha"] = args.alpha

    result_path = add_reward_to_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        reward_type=args.reward_type,
        reward_kwargs=reward_kwargs,
        inplace=args.inplace,
    )

    if args.verify:
        verify_reward(result_path)


if __name__ == "__main__":
    main()
