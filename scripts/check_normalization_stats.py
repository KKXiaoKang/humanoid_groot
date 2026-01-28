#!/usr/bin/env python
"""
检查训练和推理时的归一化stats一致性

用于验证 "Delta eef" 模式下，训练时保存的stats和推理时加载的stats是否一致，
以及relative action的归一化是否正确应用。
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from pathlib import Path
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.datasets.compute_stats import aggregate_stats


def load_stats_from_checkpoint(ckpt_path: str):
    """从checkpoint加载stats"""
    print(f"\n{'='*80}")
    print(f"📂 Loading stats from checkpoint: {ckpt_path}")
    print(f"{'='*80}")
    
    # 加载模型配置
    policy = GrootPolicy.from_pretrained(Path(ckpt_path), strict=False)
    action_space_type = getattr(policy.config, 'action_space_type', "Absolute joint")
    print(f"✅ Action space type: {action_space_type}")
    
    # 加载preprocessor和postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt_path,
    )
    
    # 从preprocessor中提取stats
    preprocessor_stats = None
    for step in preprocessor.steps:
        if hasattr(step, 'stats') and step.stats is not None:
            preprocessor_stats = step.stats
            print(f"✅ Found stats in preprocessor step: {type(step).__name__}")
            break
    
    # 从postprocessor中提取stats
    postprocessor_stats = None
    for step in postprocessor.steps:
        if hasattr(step, 'stats') and step.stats is not None:
            postprocessor_stats = step.stats
            print(f"✅ Found stats in postprocessor step: {type(step).__name__}")
            break
    
    # 提取action_space_type和action_component_indices
    action_space_type_from_processor = None
    action_component_indices = None
    for step in preprocessor.steps:
        if hasattr(step, 'action_space_type') and hasattr(step, 'action_component_indices'):
            action_space_type_from_processor = getattr(step, 'action_space_type', None)
            action_component_indices = getattr(step, 'action_component_indices', None)
            if action_space_type_from_processor:
                print(f"✅ Found action_space_type in preprocessor: {action_space_type_from_processor}")
                if action_component_indices:
                    print(f"✅ Found action_component_indices: {list(action_component_indices.keys())}")
                break
    
    return {
        'preprocessor_stats': preprocessor_stats,
        'postprocessor_stats': postprocessor_stats,
        'action_space_type': action_space_type_from_processor or action_space_type,
        'action_component_indices': action_component_indices,
    }


def load_stats_from_dataset(dataset_path: str, repo_ids: list[str] | None = None):
    """从数据集加载stats（支持单个或多个数据集）"""
    print(f"\n{'='*80}")
    print(f"📂 Loading stats from dataset: {dataset_path}")
    if repo_ids:
        print(f"   Multiple datasets: {repo_ids}")
    print(f"{'='*80}")
    
    if repo_ids and len(repo_ids) > 1:
        # 多数据集：创建MultiLeRobotDataset并聚合stats
        print(f"📦 Creating MultiLeRobotDataset with {len(repo_ids)} datasets...")
        dataset = MultiLeRobotDataset(
            repo_ids=repo_ids,
            root=dataset_path,
        )
        # MultiLeRobotDataset会自动聚合stats
        dataset_stats = dataset.stats
        print(f"✅ Aggregated stats from {len(repo_ids)} datasets")
        print(f"   Datasets: {repo_ids}")
    else:
        # 单数据集
        if repo_ids and len(repo_ids) == 1:
            repo_id = repo_ids[0]
            # root指向父目录，repo_id是子目录名
            dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)
        else:
            # 如果dataset_path直接指向数据集目录（包含meta/和data/）
            # 需要从路径中提取repo_id
            dataset_path_obj = Path(dataset_path)
            if (dataset_path_obj / "meta").exists() and (dataset_path_obj / "data").exists():
                # dataset_path就是数据集目录，使用父目录作为root，目录名作为repo_id
                repo_id = dataset_path_obj.name
                root = str(dataset_path_obj.parent)
                dataset = LeRobotDataset(repo_id=repo_id, root=root)
            else:
                # 如果dataset_path是父目录，尝试查找子目录
                # 这种情况下，我们需要知道repo_id，但如果没有提供，无法加载
                raise ValueError(
                    f"Cannot determine dataset repo_id from path: {dataset_path}. "
                    f"Please provide --dataset-repo-ids for single dataset or ensure dataset_path points to a dataset directory."
                )
        
        dataset_stats = dataset.meta.stats if hasattr(dataset.meta, 'stats') else None
    
    if dataset_stats is None:
        print("⚠️  No stats found in dataset")
        return None
    
    print(f"✅ Dataset stats keys: {list(dataset_stats.keys())}")
    return dataset_stats


def load_and_aggregate_multiple_datasets(dataset_root: str, repo_ids: list[str]):
    """手动加载多个数据集并聚合stats（用于对比验证）"""
    print(f"\n{'='*80}")
    print(f"📂 Loading and aggregating stats from multiple datasets")
    print(f"   Root: {dataset_root}")
    print(f"   Datasets: {repo_ids}")
    print(f"{'='*80}")
    
    # 加载每个数据集的stats
    individual_stats = []
    for repo_id in repo_ids:
        print(f"   Loading {repo_id}...")
        # MultiLeRobotDataset内部使用 root/repo_id 作为每个数据集的root
        # 所以我们需要将root设置为dataset_root，repo_id保持为字符串
        try:
            dataset = LeRobotDataset(repo_id=repo_id, root=Path(dataset_root) / repo_id)
            if hasattr(dataset.meta, 'stats') and dataset.meta.stats:
                individual_stats.append(dataset.meta.stats)
                print(f"     ✅ Loaded stats with keys: {list(dataset.meta.stats.keys())}")
            else:
                print(f"     ⚠️  No stats found in {repo_id}")
        except Exception as e:
            print(f"     ❌ Failed to load {repo_id}: {e}")
            # 尝试另一种方式：直接使用repo_id作为root
            try:
                dataset_path = Path(dataset_root) / repo_id
                if (dataset_path / "meta").exists():
                    # 数据集目录存在，尝试直接加载
                    dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))
                    if hasattr(dataset.meta, 'stats') and dataset.meta.stats:
                        individual_stats.append(dataset.meta.stats)
                        print(f"     ✅ Loaded stats (fallback method) with keys: {list(dataset.meta.stats.keys())}")
            except Exception as e2:
                print(f"     ❌ Fallback also failed: {e2}")
    
    if not individual_stats:
        print("⚠️  No stats found in any dataset")
        return None
    
    # 聚合stats
    print(f"\n   Aggregating stats from {len(individual_stats)} datasets...")
    aggregated_stats = aggregate_stats(individual_stats)
    print(f"✅ Aggregated stats keys: {list(aggregated_stats.keys())}")
    
    return aggregated_stats


def compare_stats_values(ckpt_stats: dict, dataset_stats: dict, key: str = 'action') -> bool:
    """对比两个stats字典中的值是否一致"""
    if key not in ckpt_stats or key not in dataset_stats:
        return False
    
    ckpt_key_stats = ckpt_stats[key]
    dataset_key_stats = dataset_stats[key]
    
    # 对比min和max
    for stat_name in ['min', 'max']:
        if stat_name not in ckpt_key_stats or stat_name not in dataset_key_stats:
            continue
        
        ckpt_val = ckpt_key_stats[stat_name]
        dataset_val = dataset_key_stats[stat_name]
        
        # 转换为tensor进行比较
        if isinstance(ckpt_val, torch.Tensor):
            ckpt_tensor = ckpt_val
        else:
            ckpt_tensor = torch.as_tensor(ckpt_val)
        
        if isinstance(dataset_val, torch.Tensor):
            dataset_tensor = dataset_val
        else:
            dataset_tensor = torch.as_tensor(dataset_val)
        
        # 比较是否相等（允许小的数值误差）
        if not torch.allclose(ckpt_tensor, dataset_tensor, atol=1e-6, rtol=1e-6):
            return False
    
    return True


def print_stats_comparison(checkpoint_stats: dict, dataset_stats: dict | None, action_space_type: str):
    """打印stats对比"""
    print(f"\n{'='*80}")
    print(f"📊 STATS COMPARISON")
    print(f"{'='*80}")
    
    # 检查action stats
    ckpt_action_stats = None
    if 'action' in checkpoint_stats.get('preprocessor_stats', {}):
        ckpt_action_stats = checkpoint_stats['preprocessor_stats']['action']
        print(f"\n✅ Checkpoint action stats:")
        if 'min' in ckpt_action_stats:
            min_vals = ckpt_action_stats['min']
            if isinstance(min_vals, torch.Tensor):
                print(f"   min shape: {min_vals.shape}")
                print(f"   min values (first 10): {min_vals[:10] if len(min_vals) >= 10 else min_vals}")
            else:
                print(f"   min: {min_vals}")
        if 'max' in ckpt_action_stats:
            max_vals = ckpt_action_stats['max']
            if isinstance(max_vals, torch.Tensor):
                print(f"   max shape: {max_vals.shape}")
                print(f"   max values (first 10): {max_vals[:10] if len(max_vals) >= 10 else max_vals}")
            else:
                print(f"   max: {max_vals}")
    
    if dataset_stats and 'action' in dataset_stats:
        dataset_action_stats = dataset_stats['action']
        print(f"\n✅ Dataset action stats:")
        if 'min' in dataset_action_stats:
            min_vals = dataset_action_stats['min']
            if isinstance(min_vals, torch.Tensor):
                print(f"   min shape: {min_vals.shape}")
                print(f"   min values (first 10): {min_vals[:10] if len(min_vals) >= 10 else min_vals}")
            else:
                print(f"   min: {min_vals}")
        if 'max' in dataset_action_stats:
            max_vals = dataset_action_stats['max']
            if isinstance(max_vals, torch.Tensor):
                print(f"   max shape: {max_vals.shape}")
                print(f"   max values (first 10): {max_vals[:10] if len(max_vals) >= 10 else max_vals}")
            else:
                print(f"   max: {max_vals}")
        
        # 对比checkpoint和dataset的stats是否一致
        if ckpt_action_stats:
            print(f"\n🔍 Comparing checkpoint stats vs dataset stats...")
            ckpt_stats_dict = {'action': ckpt_action_stats}
            dataset_stats_dict = {'action': dataset_action_stats}
            
            if compare_stats_values(ckpt_stats_dict, dataset_stats_dict, 'action'):
                print(f"   ✅ CHECKPOINT STATS MATCH DATASET STATS!")
                print(f"      The stats in checkpoint are identical to the aggregated dataset stats.")
            else:
                print(f"   ⚠️  CHECKPOINT STATS DO NOT MATCH DATASET STATS!")
                print(f"      There may be a mismatch. Checking differences...")
                
                # 详细对比
                for stat_name in ['min', 'max']:
                    if stat_name in ckpt_action_stats and stat_name in dataset_action_stats:
                        ckpt_val = ckpt_action_stats[stat_name]
                        dataset_val = dataset_action_stats[stat_name]
                        
                        ckpt_tensor = ckpt_val if isinstance(ckpt_val, torch.Tensor) else torch.as_tensor(ckpt_val)
                        dataset_tensor = dataset_val if isinstance(dataset_val, torch.Tensor) else torch.as_tensor(dataset_val)
                        
                        diff = torch.abs(ckpt_tensor - dataset_tensor)
                        max_diff = torch.max(diff)
                        print(f"      {stat_name} max difference: {max_diff.item():.6e}")
                        if max_diff > 1e-6:
                            print(f"      First 10 differences: {diff[:10] if len(diff) >= 10 else diff}")
    
    # 对于 "Delta eef" 模式，分析归一化逻辑
    if action_space_type == "Delta eef":
        print(f"\n{'='*80}")
        print(f"🔄 DELTA EEF MODE NORMALIZATION ANALYSIS")
        print(f"{'='*80}")
        
        action_component_indices = checkpoint_stats.get('action_component_indices')
        if action_component_indices:
            print(f"\n📋 Action component indices:")
            for comp_name, (start_idx, end_idx) in action_component_indices.items():
                print(f"   {comp_name}: indices [{start_idx}:{end_idx}]")
            
            # 分析每个组件的归一化方式
            print(f"\n📊 Normalization strategy:")
            print(f"   - Position components (left_eef_pos, right_eef_pos):")
            print(f"     * Use MIN_MAX normalization")
            print(f"     * For relative actions: adjusted range (1.5x absolute range, centered at 0)")
            print(f"   - 6D rotation components (left_eef_rot6d, right_eef_rot6d):")
            print(f"     * Use IDENTITY normalization (no normalization)")
            print(f"     * Values are kept as-is")
            print(f"   - Gripper components (left_gripper, right_gripper):")
            print(f"     * Use MIN_MAX normalization")
            
            # 检查checkpoint中的stats是否包含这些信息
            if 'action' in checkpoint_stats.get('preprocessor_stats', {}):
                ckpt_action_stats = checkpoint_stats['preprocessor_stats']['action']
                if 'min' in ckpt_action_stats and 'max' in ckpt_action_stats:
                    min_vals = ckpt_action_stats['min']
                    max_vals = ckpt_action_stats['max']
                    
                    print(f"\n🔍 Stats analysis for relative action normalization:")
                    for comp_name, (start_idx, end_idx) in action_component_indices.items():
                        comp_min = min_vals[start_idx:end_idx] if isinstance(min_vals, torch.Tensor) else min_vals[start_idx:end_idx]
                        comp_max = max_vals[start_idx:end_idx] if isinstance(max_vals, torch.Tensor) else max_vals[start_idx:end_idx]
                        
                        print(f"\n   {comp_name} (indices [{start_idx}:{end_idx}]):")
                        if isinstance(comp_min, torch.Tensor):
                            print(f"     min: {comp_min.tolist()}")
                            print(f"     max: {comp_max.tolist()}")
                        else:
                            print(f"     min: {comp_min}")
                            print(f"     max: {comp_max}")
                        
                        if "rot6d" in comp_name:
                            print(f"     ⚠️  NOTE: This component uses IDENTITY normalization (no normalization applied)")
                            print(f"        The min/max values here are from absolute eef pose stats,")
                            print(f"        but they are NOT used for normalization during training/inference")
                        elif "pos" in comp_name:
                            print(f"     ✅ This component uses MIN_MAX normalization")
                            print(f"     ⚠️  For relative actions, the normalization range will be adjusted:")
                            print(f"        - abs_range = max(|min|, |max|)")
                            print(f"        - rel_range = abs_range * 1.5")
                            print(f"        - normalized_range = [-rel_range, rel_range]")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Check normalization stats consistency between training and inference'
    )
    parser.add_argument('--ckpt-path', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='Path to dataset root directory (parent directory containing dataset subdirectories)')
    parser.add_argument('--dataset-repo-ids', type=str, default=None,
                        help='Comma-separated list of dataset repo_ids (e.g., "eef_3x2,eef_mix_color,eef_short_dense")')
    parser.add_argument('--single-dataset-path', type=str, default=None,
                        help='Optional: Path to single dataset directory (for comparison with single dataset)')
    
    args = parser.parse_args()
    
    # 从checkpoint加载stats
    checkpoint_stats = load_stats_from_checkpoint(args.ckpt_path)
    
    # 从数据集加载stats（如果提供）
    dataset_stats = None
    if args.dataset_root and args.dataset_repo_ids:
        # 多数据集模式：加载并聚合stats
        repo_ids = [repo_id.strip() for repo_id in args.dataset_repo_ids.split(',')]
        print(f"\n{'='*80}")
        print(f"🔍 PROOF: Loading and aggregating stats from training datasets")
        print(f"{'='*80}")
        
        # 方法1：使用MultiLeRobotDataset（与训练时相同的方式）
        dataset_stats = load_stats_from_dataset(args.dataset_root, repo_ids)
        
        # 方法2：手动聚合（用于验证）
        manual_aggregated_stats = load_and_aggregate_multiple_datasets(args.dataset_root, repo_ids)
        
        # 验证两种方法的结果是否一致
        if dataset_stats and manual_aggregated_stats:
            if compare_stats_values(dataset_stats, manual_aggregated_stats, 'action'):
                print(f"\n✅ VERIFICATION: MultiLeRobotDataset.stats matches manual aggregate_stats()")
            else:
                print(f"\n⚠️  WARNING: MultiLeRobotDataset.stats differs from manual aggregate_stats()")
        
    elif args.single_dataset_path:
        # 单数据集模式
        dataset_stats = load_stats_from_dataset(args.single_dataset_path)
    
    # 对比stats
    print_stats_comparison(
        checkpoint_stats,
        dataset_stats,
        checkpoint_stats['action_space_type']
    )
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
