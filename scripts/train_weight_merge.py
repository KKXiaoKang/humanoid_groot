#!/usr/bin/env python
"""
Expert Merging 训练脚本

基于论文 "Expert Merging: Model Merging with Unsupervised Expert Alignment"
https://arxiv.org/pdf/2509.25712

⭐ 默认使用 Expert Merging 方法（效果最好）

📚 校准数据说明：
    - 可以使用原始训练数据集作为校准数据！
    - 只需要 5-10 个样本
    - 不需要动作标签，只使用观测数据
    - 建议混合使用两个任务的数据

使用方式：

    # 方式1: Expert Merging（默认，效果最好）⭐ 推荐
    # 自动使用 lerobot_data 中的训练数据集作为校准数据
    python scripts/train_weight_merge.py --output_path ./outputs/merged_groot
    
    # 指定自定义校准数据集
    python scripts/train_weight_merge.py \\
        --data_path /path/to/narrow_data,/path/to/wide_data \\
        --output_path ./outputs/merged_groot
    
    # 方式2: Task Arithmetic（无需训练，快速）
    python scripts/train_weight_merge.py --method task_arithmetic --output_path ./outputs/merged_groot
    
    # 方式3: 直接插值（无需训练，最简单）
    python scripts/train_weight_merge.py --method interpolation --alpha 0.5 --output_path ./outputs/merged_groot

训练后评估：
    python eval/eval_merged_groot.py --model_path ./outputs/merged_groot/pretrained_model
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型路径
MODEL_NARROW_PATH = \
    "/home/kangkk/humanoid_groot_base/outputs/0112_h100x4_groot_cross_attention_narrower_very_conservative/checkpoints/020000/pretrained_model"
MODEL_WIDE_PATH = \
    "/home/kangkk/humanoid_groot_base/outputs/0113_h100x4_groot_cross_attention_wider_very_conservative_mix_dense/checkpoints/014000/pretrained_model"

# ⚠️ 重要：Base 模型必须与专家模型有相同的架构！
# 不能使用 nvidia/GR00T-N1.5-3B，因为原始预训练模型的 action_head 结构不同
# 使用第一个专家模型（narrower）作为 base 模型
# 这样 Task Vector 计算为：τ_wider = wider - narrower，τ_narrower = 0
BASE_MODEL_PATH = MODEL_NARROW_PATH

# 默认校准数据集路径（使用原始训练数据集）
# 混合使用窄箱子和宽箱子的数据，以确保融合模型能处理两种任务
DEFAULT_CALIBRATION_DATASETS = [
    # 窄箱子相关数据集 (narrower)
    "/home/kangkk/humanoid_groot_base/lerobot_data/split_dataset/narrower/four",
    "/home/kangkk/humanoid_groot_base/lerobot_data/split_dataset/narrower/random",
    "/home/kangkk/humanoid_groot_base/lerobot_data/split_dataset/narrower/dense",
    "/home/kangkk/humanoid_groot_base/lerobot_data/split_dataset/narrower/mix",
    # 宽箱子相关数据集 (wider)
    "/home/kangkk/humanoid_groot_base/lerobot_data/split_dataset/wider/four",
    "/home/kangkk/humanoid_groot_base/lerobot_data/split_dataset/wider/random",
    "/home/kangkk/humanoid_groot_base/lerobot_data/split_dataset/wider/dense",
    "/home/kangkk/humanoid_groot_base/lerobot_data/split_dataset/wider/mix",
]


def create_calibration_dataloader(
    data_paths: list[str] = None,
    batch_size: int = 1,
    num_samples: int = 10,
    use_default_datasets: bool = True,
):
    """
    创建校准数据加载器
    
    对于 Expert Merging，只需要少量无标签样本（5-10个）
    
    ⭐ 重要：校准数据应该混合来自两个任务的样本！
    
    数据来源选项：
    1. 指定 data_paths：使用指定的数据集
    2. use_default_datasets=True：使用默认的训练数据集
    3. 都不提供：使用合成数据（仅用于测试）
    
    Args:
        data_paths: 数据路径列表（可以是 LeRobot 数据集路径）
        batch_size: 批次大小
        num_samples: 每个数据集采样数量
        use_default_datasets: 是否使用默认数据集
    
    Returns:
        DataLoader 或 生成器
    """
    # 确定要使用的数据路径
    paths_to_use = []
    
    if data_paths:
        paths_to_use = data_paths
        logger.info(f"Using specified data paths: {data_paths}")
    elif use_default_datasets:
        # 检查默认数据集是否存在
        for path in DEFAULT_CALIBRATION_DATASETS:
            if Path(path).exists():
                paths_to_use.append(path)
            else:
                logger.warning(f"Default dataset not found: {path}")
        
        if paths_to_use:
            logger.info(f"Using {len(paths_to_use)} default calibration datasets")
        else:
            logger.warning("No default datasets found")
    
    if not paths_to_use:
        logger.warning("No data paths provided, using synthetic data for demonstration")
        logger.warning("⚠️  建议使用真实数据以获得最佳效果！")
        return create_synthetic_dataloader(batch_size, num_samples)
    
    # 使用 LeRobot 数据集
    return create_lerobot_dataloader(paths_to_use, batch_size, num_samples)


class TaskLabeledDataset:
    """
    带任务标签的数据集包装器
    
    为每个样本添加 task_source 字段，标识来自哪个专家任务
    """
    def __init__(self, dataset, task_source: int):
        self.dataset = dataset
        self.task_source = task_source  # 0=narrower, 1=wider
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # 添加任务来源标签
        if isinstance(item, dict):
            item = dict(item)  # 复制以避免修改原数据
            item['task_source'] = self.task_source
        return item


def create_lerobot_dataloader(
    data_paths: list[str],
    batch_size: int = 1,
    num_samples: int = 10,
):
    """
    从 LeRobot 数据集创建校准数据加载器
    
    ⚠️ 关键：为每个样本添加 task_source 标签，用于区分来自哪个专家任务
    
    参考 eval_on_dataset_lowpass.py 中的正确加载方式
    
    Args:
        data_paths: LeRobot 数据集路径列表
        batch_size: 批次大小
        num_samples: 每个数据集采样数量
    
    Returns:
        DataLoader
    """
    # 正确的导入路径
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader, ConcatDataset, Subset
    
    datasets = []
    total_samples = 0
    
    for data_path in data_paths:
        try:
            # 正确的加载方式：参考 eval_on_dataset_lowpass.py
            # repo_id 使用数据集路径的最后一部分作为标识符
            dataset_name = Path(data_path).name
            
            logger.info(f"📂 Loading dataset from {data_path}")
            logger.info(f"   repo_id: {dataset_name}")
            
            # ⚠️ 关键：根据路径判断任务来源
            # narrower 数据集路径包含 "narrower"，wider 数据集路径包含 "wider"
            if "narrower" in data_path.lower():
                task_source = 0  # narrower
                task_label = "narrower"
            elif "wider" in data_path.lower():
                task_source = 1  # wider
                task_label = "wider"
            else:
                # 默认：根据位置判断（前半部分是 narrower，后半部分是 wider）
                task_source = 0 if len(datasets) < 4 else 1
                task_label = "narrower" if task_source == 0 else "wider"
            
            logger.info(f"   Task source: {task_label} (id={task_source})")
            
            # 使用正确的参数加载 LeRobotDataset
            dataset = LeRobotDataset(repo_id=dataset_name, root=data_path)
            
            # 获取数据集总帧数
            total_frames = dataset.num_frames
            logger.info(f"   Total frames in dataset: {total_frames}")
            
            # 随机采样
            sample_count = min(num_samples, total_frames)
            indices = np.random.choice(total_frames, sample_count, replace=False)
            subset = Subset(dataset, indices.tolist())
            
            # ⚠️ 关键：包装数据集以添加任务标签
            labeled_subset = TaskLabeledDataset(subset, task_source)
            datasets.append(labeled_subset)
            total_samples += sample_count
            
            logger.info(f"✅ Loaded {sample_count} samples from {data_path} (task={task_label})")
            
        except Exception as e:
            logger.warning(f"❌ Failed to load dataset {data_path}: {e}")
            import traceback
            traceback.print_exc()
            
            # 尝试直接从路径加载（备用方法）
            try:
                dataset = load_dataset_from_path(data_path, num_samples)
                if dataset is not None:
                    # 根据路径判断任务来源
                    task_source = 0 if "narrower" in data_path.lower() else 1
                    labeled_dataset = TaskLabeledDataset(dataset, task_source)
                    datasets.append(labeled_dataset)
                    total_samples += len(dataset)
                    logger.info(f"✅ Loaded {len(dataset)} samples from {data_path} (fallback)")
            except Exception as e2:
                logger.warning(f"❌ Fallback also failed: {e2}")
    
    if not datasets:
        logger.warning("No datasets loaded, using synthetic data")
        return create_synthetic_dataloader(batch_size, num_samples)
    
    logger.info(f"📊 Total calibration samples: {total_samples}")
    
    combined_dataset = ConcatDataset(datasets)
    
    def collate_fn(batch):
        """
        自定义 collate function 处理 LeRobot 数据集格式
        
        LeRobot 数据集的每个样本包含：
        - observation.state: 状态观测
        - observation.images.*: 图像观测
        - action: 动作
        - task: 任务描述
        - task_source: 任务来源 (0=narrower, 1=wider) ⚠️ 新增
        等字段
        """
        result = {}
        for key in batch[0]:
            values = [b[key] for b in batch]
            if isinstance(values[0], torch.Tensor):
                # 确保维度正确
                if values[0].dim() == 0:
                    result[key] = torch.stack(values)
                else:
                    result[key] = torch.stack(values, dim=0)
            elif isinstance(values[0], (int, float)):
                result[key] = torch.tensor(values)
            elif isinstance(values[0], str):
                result[key] = values
            else:
                result[key] = values
        return result
    
    return DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 避免多进程问题
        pin_memory=torch.cuda.is_available(),
    )


def load_dataset_from_path(data_path: str, num_samples: int = 10):
    """
    直接从路径加载数据集（备用方法）
    
    尝试读取 parquet 文件或其他格式
    """
    from pathlib import Path
    import glob
    
    data_path = Path(data_path)
    
    # 检查是否有 parquet 文件
    parquet_files = list(data_path.glob("**/*.parquet"))
    if parquet_files:
        try:
            import pandas as pd
            
            dfs = []
            for pf in parquet_files[:3]:  # 最多读取3个文件
                df = pd.read_parquet(pf)
                dfs.append(df)
            
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                # 随机采样
                if len(combined) > num_samples:
                    combined = combined.sample(n=num_samples)
                
                # 转换为 Dataset 格式
                return ParquetDataset(combined)
        except Exception as e:
            logger.warning(f"Failed to load parquet: {e}")
    
    return None


class ParquetDataset:
    """简单的 Parquet 数据集包装器"""
    
    def __init__(self, df):
        self.df = df
        self.length = len(df)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 尝试提取相关字段
        item = {}
        
        # 状态
        if 'observation.state' in row:
            state = row['observation.state']
            if isinstance(state, np.ndarray):
                item['state'] = torch.from_numpy(state).float().unsqueeze(0)
            else:
                item['state'] = torch.tensor(state).float().unsqueeze(0)
        else:
            item['state'] = torch.randn(1, 16)
        
        item['state_mask'] = torch.ones_like(item['state'])
        item['embodiment_id'] = torch.tensor([0])
        
        return item


def create_synthetic_dataloader(batch_size: int = 1, num_samples: int = 10):
    """
    创建合成数据加载器（用于测试）
    
    注意：这只是用于代码测试，实际使用时应该用真实数据
    """
    logger.warning("Using synthetic data - for testing only!")
    
    class SyntheticDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # 创建符合 GROOT 输入格式的合成数据
            return {
                # 状态
                "state": torch.randn(1, 16),
                "state_mask": torch.ones(1, 16),
                
                # Eagle 输入（视觉-语言）
                "eagle_pixel_values": torch.randn(1, 3, 224, 224),
                "eagle_input_ids": torch.randint(0, 1000, (1, 128)),
                "eagle_attention_mask": torch.ones(1, 128),
                "eagle_image_sizes": torch.tensor([[224, 224]]),
                
                # Embodiment
                "embodiment_id": torch.tensor([0]),
            }
    
    dataset = SyntheticDataset(num_samples)
    
    def collate_fn(batch):
        result = {}
        for key in batch[0]:
            values = [b[key] for b in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.cat(values, dim=0)
            else:
                result[key] = values
        return result
    
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def run_task_arithmetic_merge(args):
    """
    运行 Task Arithmetic 融合
    
    θ_merged = θ_base + α_narrow * (θ_narrow - θ_base) + α_wide * (θ_wide - θ_base)
    
    ⚠️ 重要：如果 skip_action_head=True，只融合 backbone，action_head 使用 narrower 的
    DiT 对参数变化非常敏感，不能直接线性插值！
    """
    from lerobot.policies.groot.weight_merge_groot import simple_task_arithmetic_merge
    
    simple_task_arithmetic_merge(
        narrower_path=args.narrower_path,
        wider_path=args.wider_path,
        output_path=args.output_path,
        base_model_path=args.base_model_path,
        narrower_weight=args.narrower_weight,
        wider_weight=args.wider_weight,
        skip_action_head=args.skip_action_head,
    )


def run_interpolation_merge(args):
    """
    运行直接插值融合
    
    θ_merged = α * θ_narrow + (1-α) * θ_wide
    
    ⚠️ 重要：如果 skip_action_head=True，只融合 backbone，action_head 使用 narrower 的
    DiT 对参数变化非常敏感，不能直接线性插值！
    """
    from lerobot.policies.groot.weight_merge_groot import direct_interpolation_merge
    
    direct_interpolation_merge(
        narrower_path=args.narrower_path,
        wider_path=args.wider_path,
        output_path=args.output_path,
        alpha=args.alpha,
        skip_action_head=args.skip_action_head,
    )


def run_expert_merge(args):
    """
    运行 Expert Merging 训练（推荐方法）⭐
    
    这是效果最好的方法，通过学习最优系数来融合模型。
    
    校准数据来源（按优先级）：
    1. 通过 --data_path 指定的数据集
    2. 默认的训练数据集（混合窄箱子和宽箱子数据）
    3. 合成数据（仅用于测试）
    
    建议使用原始训练数据的子集作为校准数据！
    """
    from lerobot.policies.groot.weight_merge_groot import (
        ExpertMergeConfig,
        ExpertMerger,
    )
    
    print(f"\n{'='*60}")
    print(f"⭐ Expert Merging - 效果最好的融合方法")
    print(f"{'='*60}")
    print(f"\n📚 校准数据说明：")
    print(f"   - 可以使用原始训练数据集作为校准数据")
    print(f"   - 只需要 5-10 个样本")
    print(f"   - 建议混合两个任务的数据")
    print(f"   - 不需要动作标签，只使用观测数据")
    print(f"\n")
    
    # 创建配置
    config = ExpertMergeConfig(
        expert_paths=[args.narrower_path, args.wider_path],
        expert_names=["narrower", "wider"],
        base_model_path=args.base_model_path,
        merge_mode="layer_wise",
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        regularization_weight=args.regularization_weight,
        initial_coefficient=args.initial_coefficient,
        hidden_alignment_weight=args.hidden_weight,
        logit_alignment_weight=args.logit_weight,
        task_weights=[args.narrower_task_weight, args.wider_task_weight],
        merge_backbone_only=args.merge_backbone_only,
        action_head_source=args.action_head_source,
        device=args.device,
    )
    
    # 创建 Merger
    merger = ExpertMerger(config)
    merger.load_models()
    
    # 创建数据加载器
    # 优先使用用户指定的数据路径，否则使用默认数据集
    data_paths = args.data_path.split(",") if args.data_path else None
    
    dataloader = create_calibration_dataloader(
        data_paths=data_paths,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        use_default_datasets=args.use_default_datasets,
    )
    
    # 训练
    merger.train(
        train_dataloader=dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
    )
    
    # 保存
    merger.save_merged_model(args.output_path)


def run_ties_merge(args):
    """
    运行 TIES Merging
    
    TIES: TrIm, Elect Sign, and Scale
    1. Trim: 剪枝小的参数变化
    2. Elect Sign: 解决符号冲突
    3. Scale: 重缩放
    
    ⚠️ 重要：如果 skip_action_head=True，只融合 backbone，action_head 使用 narrower 的
    DiT 对参数变化非常敏感，不能直接线性插值！
    """
    from pathlib import Path
    import glob
    from safetensors.torch import load_file, save_file
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
    import shutil
    
    print(f"\n{'='*60}")
    print(f"🔧 TIES Merging")
    print(f"   Trim ratio: {args.ties_trim_ratio}")
    print(f"   Scale: {args.ties_scale}")
    if args.skip_action_head:
        print(f"   ⚠️ Skip action_head: True (使用 narrower 的 DiT)")
    print(f"{'='*60}\n")
    
    def load_weights(path: str) -> dict:
        path = Path(path)
        safetensors_files = glob.glob(str(path / "model*.safetensors"))
        state_dict = {}
        for f in sorted(safetensors_files):
            state_dict.update(load_file(f))
        return state_dict
    
    # 加载权重
    try:
        base_local_path = snapshot_download(args.base_model_path, repo_type="model")
    except (HFValidationError, RepositoryNotFoundError):
        base_local_path = args.base_model_path
    
    base_state_dict = load_weights(base_local_path)
    narrower_state_dict = load_weights(args.narrower_path)
    wider_state_dict = load_weights(args.wider_path)
    
    # 计算 Task Vectors
    τ_narrow = {k: narrower_state_dict[k] - base_state_dict[k] for k in base_state_dict if k in narrower_state_dict}
    τ_wide = {k: wider_state_dict[k] - base_state_dict[k] for k in base_state_dict if k in wider_state_dict}
    
    def ties_merge_vectors(task_vectors: list[dict], trim_ratio: float = 0.2, scale: float = 1.0):
        """
        TIES Merging implementation
        
        1. Trim: 将小于阈值的参数置为0
        2. Elect Sign: 对于冲突的参数，选择多数符号
        3. Scale: 缩放合并后的向量
        """
        merged = {}
        
        for key in task_vectors[0]:
            # 收集所有任务的参数
            all_deltas = torch.stack([tv[key] for tv in task_vectors if key in tv])  # (num_tasks, ...)
            
            # 1. Trim: 将小的参数变化置为0
            abs_deltas = all_deltas.abs()
            threshold = torch.quantile(abs_deltas.flatten(), trim_ratio)
            trimmed_deltas = torch.where(abs_deltas > threshold, all_deltas, torch.zeros_like(all_deltas))
            
            # 2. Elect Sign: 使用加权投票选择符号
            signs = torch.sign(trimmed_deltas)
            sign_votes = signs.sum(dim=0)  # 正票 - 负票
            elected_sign = torch.sign(sign_votes)  # 多数票
            elected_sign = torch.where(elected_sign == 0, torch.ones_like(elected_sign), elected_sign)
            
            # 3. 只保留与选定符号一致的参数
            mask = (signs == elected_sign.unsqueeze(0))
            consistent_deltas = torch.where(mask, trimmed_deltas, torch.zeros_like(trimmed_deltas))
            
            # 4. 平均合并
            num_contributors = mask.float().sum(dim=0).clamp(min=1)
            merged_delta = consistent_deltas.sum(dim=0) / num_contributors
            
            # 5. Scale
            merged[key] = merged_delta * scale
        
        return merged
    
    # TIES 融合
    merged_task_vector = ties_merge_vectors(
        [τ_narrow, τ_wide],
        trim_ratio=args.ties_trim_ratio,
        scale=args.ties_scale,
    )
    
    # 应用到 base
    # ⚠️ 重要：如果 skip_action_head=True，跳过 action_head 层的融合
    merged_state_dict = {}
    skipped_action_head_layers = 0
    for k in base_state_dict:
        # 检查是否是 action_head 层
        if args.skip_action_head and k.startswith('action_head.'):
            # 使用 narrower 的 action_head
            if k in narrower_state_dict:
                merged_state_dict[k] = narrower_state_dict[k]
                skipped_action_head_layers += 1
            else:
                merged_state_dict[k] = base_state_dict[k]
        elif k in merged_task_vector:
            merged_state_dict[k] = base_state_dict[k] + merged_task_vector[k]
        else:
            merged_state_dict[k] = base_state_dict[k]
    
    if args.skip_action_head:
        print(f"⚠️ Skipped {skipped_action_head_layers} action_head layers (使用 narrower 的 DiT)")
    
    # 保存
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_file(merged_state_dict, str(output_path / "model.safetensors"))
    
    # 复制配置文件
    narrower_path = Path(args.narrower_path)
    for config_file in ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"]:
        src = narrower_path / config_file
        if src.exists():
            shutil.copy(src, output_path / config_file)
    
    for pattern in ["policy_preprocessor*.safetensors", "policy_postprocessor*.safetensors"]:
        for src in narrower_path.glob(pattern):
            shutil.copy(src, output_path / src.name)
    
    merge_config = {
        "merge_method": "ties",
        "narrower_path": str(args.narrower_path),
        "wider_path": str(args.wider_path),
        "base_model_path": args.base_model_path,
        "trim_ratio": args.ties_trim_ratio,
        "scale": args.ties_scale,
        "skip_action_head": args.skip_action_head,
        "action_head_source": "narrower" if args.skip_action_head else "merged",
    }
    with open(output_path / "merge_config.json", "w") as f:
        json.dump(merge_config, f, indent=2)
    
    print(f"\n✅ TIES merged model saved to {output_path}")


def run_dare_merge(args):
    """
    运行 DARE Merging
    
    DARE: Drop And REscale
    随机丢弃部分参数变化，然后重新缩放
    
    ⚠️ 重要：如果 skip_action_head=True，只融合 backbone，action_head 使用 narrower 的
    DiT 对参数变化非常敏感，不能直接线性插值！
    """
    from pathlib import Path
    import glob
    from safetensors.torch import load_file, save_file
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
    import shutil
    
    print(f"\n{'='*60}")
    print(f"🔧 DARE Merging")
    print(f"   Drop rate: {args.dare_drop_rate}")
    if args.skip_action_head:
        print(f"   ⚠️ Skip action_head: True (使用 narrower 的 DiT)")
    print(f"{'='*60}\n")
    
    def load_weights(path: str) -> dict:
        path = Path(path)
        safetensors_files = glob.glob(str(path / "model*.safetensors"))
        state_dict = {}
        for f in sorted(safetensors_files):
            state_dict.update(load_file(f))
        return state_dict
    
    # 加载权重
    try:
        base_local_path = snapshot_download(args.base_model_path, repo_type="model")
    except (HFValidationError, RepositoryNotFoundError):
        base_local_path = args.base_model_path
    
    base_state_dict = load_weights(base_local_path)
    narrower_state_dict = load_weights(args.narrower_path)
    wider_state_dict = load_weights(args.wider_path)
    
    # 计算 Task Vectors
    τ_narrow = {k: narrower_state_dict[k] - base_state_dict[k] for k in base_state_dict if k in narrower_state_dict}
    τ_wide = {k: wider_state_dict[k] - base_state_dict[k] for k in base_state_dict if k in wider_state_dict}
    
    def dare_process(task_vector: dict, drop_rate: float = 0.1):
        """
        DARE: Drop and Rescale
        
        1. 随机丢弃部分参数
        2. 重新缩放保留的参数
        """
        processed = {}
        for key, delta in task_vector.items():
            # 创建 drop mask
            mask = torch.rand_like(delta) > drop_rate
            # Drop and rescale
            processed[key] = delta * mask / (1 - drop_rate)
        return processed
    
    # DARE 处理
    τ_narrow_dare = dare_process(τ_narrow, args.dare_drop_rate)
    τ_wide_dare = dare_process(τ_wide, args.dare_drop_rate)
    
    # 融合
    # ⚠️ 重要：如果 skip_action_head=True，跳过 action_head 层的融合
    merged_state_dict = {}
    skipped_action_head_layers = 0
    for k in base_state_dict:
        # 检查是否是 action_head 层
        if args.skip_action_head and k.startswith('action_head.'):
            # 使用 narrower 的 action_head
            if k in narrower_state_dict:
                merged_state_dict[k] = narrower_state_dict[k]
                skipped_action_head_layers += 1
            else:
                merged_state_dict[k] = base_state_dict[k]
        else:
            merged = base_state_dict[k].clone()
            if k in τ_narrow_dare:
                merged = merged + args.narrower_weight * τ_narrow_dare[k]
            if k in τ_wide_dare:
                merged = merged + args.wider_weight * τ_wide_dare[k]
            merged_state_dict[k] = merged
    
    if args.skip_action_head:
        print(f"⚠️ Skipped {skipped_action_head_layers} action_head layers (使用 narrower 的 DiT)")
    
    # 保存
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_file(merged_state_dict, str(output_path / "model.safetensors"))
    
    # 复制配置文件
    narrower_path = Path(args.narrower_path)
    for config_file in ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"]:
        src = narrower_path / config_file
        if src.exists():
            shutil.copy(src, output_path / config_file)
    
    for pattern in ["policy_preprocessor*.safetensors", "policy_postprocessor*.safetensors"]:
        for src in narrower_path.glob(pattern):
            shutil.copy(src, output_path / src.name)
    
    merge_config = {
        "merge_method": "dare",
        "narrower_path": str(args.narrower_path),
        "wider_path": str(args.wider_path),
        "base_model_path": args.base_model_path,
        "drop_rate": args.dare_drop_rate,
        "narrower_weight": args.narrower_weight,
        "wider_weight": args.wider_weight,
        "skip_action_head": args.skip_action_head,
        "action_head_source": "narrower" if args.skip_action_head else "merged",
    }
    with open(output_path / "merge_config.json", "w") as f:
        json.dump(merge_config, f, indent=2)
    
    print(f"\n✅ DARE merged model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GROOT Model Weight Merging")
    
    # 融合方法 - 默认使用 expert_merge（效果最好）
    parser.add_argument(
        "--method", 
        type=str, 
        default="expert_merge",  # 默认使用 Expert Merging
        choices=["expert_merge", "task_arithmetic", "interpolation", "ties", "dare"],
        help="Merge method: expert_merge (best, default), task_arithmetic, "
             "interpolation (simplest), ties, dare"
    )
    
    # 模型路径
    parser.add_argument("--narrower_path", type=str, default=MODEL_NARROW_PATH,
                       help="Path to narrower model checkpoint")
    parser.add_argument("--wider_path", type=str, default=MODEL_WIDE_PATH,
                       help="Path to wider model checkpoint")
    parser.add_argument("--base_model_path", type=str, default=BASE_MODEL_PATH,
                       help="Path to base model (for task arithmetic)")
    parser.add_argument("--output_path", type=str, default="./outputs/merged_groot/pretrained_model",
                       help="Output path for merged model")
    
    # Task Arithmetic 参数
    parser.add_argument("--narrower_weight", type=float, default=0.5,
                       help="Weight for narrower model task vector")
    parser.add_argument("--wider_weight", type=float, default=0.5,
                       help="Weight for wider model task vector")
    
    # Interpolation 参数
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Interpolation coefficient (0=wider, 1=narrower)")
    
    # Expert Merge 参数
    parser.add_argument("--data_path", type=str, default=None,
                       help="Comma-separated paths to calibration datasets. "
                            "可以使用原始训练数据集！例如: /path/to/narrow_data,/path/to/wide_data")
    parser.add_argument("--use_default_datasets", action="store_true", default=True,
                       help="Use default calibration datasets (from lerobot_data folder)")
    parser.add_argument("--no_default_datasets", action="store_false", dest="use_default_datasets",
                       help="Disable default calibration datasets")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs for expert merge")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate for expert merge")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for training")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of calibration samples per dataset (5-10 recommended)")
    parser.add_argument("--regularization_weight", type=float, default=0.8,
                       help="Regularization weight (γ), controls coefficient drift")
    parser.add_argument("--initial_coefficient", type=float, default=0.5,
                       help="Initial coefficient value for task vectors")
    parser.add_argument("--hidden_weight", type=float, default=1.0,
                       help="Hidden alignment loss weight")
    parser.add_argument("--logit_weight", type=float, default=0.0,
                       help="Logit (action) alignment loss weight (default 0.0, use hidden loss only)")
    parser.add_argument("--narrower_task_weight", type=float, default=1.0,
                       help="Task weight for narrower expert (increase to prioritize)")
    parser.add_argument("--wider_task_weight", type=float, default=1.0,
                       help="Task weight for wider expert (increase to prioritize)")
    
    # 融合范围控制
    parser.add_argument("--merge_backbone_only", action="store_true", default=False,
                       help="Only merge backbone weights, use specified expert's action_head (recommended if action loss explodes)")
    parser.add_argument("--action_head_source", type=str, default="first_expert",
                       choices=["first_expert", "second_expert", "interpolate"],
                       help="Action head source when merge_backbone_only=True: first_expert (narrower), second_expert (wider), or interpolate")
    parser.add_argument("--skip_action_head", action="store_true", default=False,
                       help="⚠️ 重要：跳过 action_head (DiT) 的融合，使用 narrower 模型的 action_head。"
                            "DiT 对参数变化非常敏感，不能直接线性插值！")
    
    # TIES 参数
    parser.add_argument("--ties_trim_ratio", type=float, default=0.2,
                       help="TIES trim ratio (percentage of small changes to drop)")
    parser.add_argument("--ties_scale", type=float, default=1.0,
                       help="TIES scale factor")
    
    # DARE 参数
    parser.add_argument("--dare_drop_rate", type=float, default=0.1,
                       help="DARE drop rate")
    
    # 设备
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for training")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🚀 GROOT Model Weight Merging")
    print(f"   Method: {args.method}")
    print(f"   Narrower: {args.narrower_path}")
    print(f"   Wider: {args.wider_path}")
    print(f"   Output: {args.output_path}")
    print(f"{'='*60}\n")
    
    if args.method == "task_arithmetic":
        run_task_arithmetic_merge(args)
    elif args.method == "interpolation":
        run_interpolation_merge(args)
    elif args.method == "expert_merge":
        run_expert_merge(args)
    elif args.method == "ties":
        run_ties_merge(args)
    elif args.method == "dare":
        run_dare_merge(args)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    print(f"\n✅ Done! Merged model saved to {args.output_path}")
    print(f"\n💡 To evaluate the merged model:")
    print(f"   python eval/eval_online.py --policy.path={args.output_path}")


if __name__ == "__main__":
    main()
