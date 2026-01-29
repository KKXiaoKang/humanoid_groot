#!/usr/bin/env python
"""
检查训练好的模型中的relative action统计值，并与训练日志对比。

用法:
    python scripts/check_relative_action_stats.py \
        --ckpt-path outputs/train/your_checkpoint/checkpoints/020000/pretrained_model
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig


def load_relative_action_stats_from_preprocessor(ckpt_path: str) -> dict[str, Any] | None:
    """
    从preprocessor中加载relative_action_stats
    
    注意：由于relative_action_stats不是初始化参数，我们需要：
    1. 先加载preprocessor（排除relative_action_stats）
    2. 然后从safetensors加载state_dict
    """
    print(f"\n📊 从preprocessor加载relative_action_stats...")
    
    try:
        # 加载配置
        config = PreTrainedConfig.from_pretrained(ckpt_path)
        
        # 加载preprocessor，使用overrides排除relative_action_stats
        # relative_action_stats不是__init__参数，应该通过load_state_dict加载
        # 但pipeline会合并配置，所以我们需要在加载后手动处理
        try:
            preprocessor, _ = make_pre_post_processors(
                policy_cfg=config,
                pretrained_path=ckpt_path,
            )
        except ValueError as e:
            # 如果因为relative_action_stats导致加载失败，尝试从JSON和safetensors直接读取
            if "relative_action_stats" in str(e):
                print(f"⚠️  加载preprocessor时遇到relative_action_stats问题，跳过preprocessor加载")
                return None
            raise
        
        # 查找GrootPackInputsStep
        pack_step = None
        for step in preprocessor.steps:
            if step.__class__.__name__ == "GrootPackInputsStep":
                pack_step = step
                break
        
        if pack_step is None:
            print("⚠️  未找到GrootPackInputsStep")
            return None
        
        # 从safetensors文件加载state_dict
        ckpt_dir = Path(ckpt_path)
        safetensors_files = list(ckpt_dir.glob("**/*groot_pack_inputs*.safetensors"))
        
        if safetensors_files:
            state_dict = {}
            for safetensors_file in safetensors_files:
                with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("relative_action."):
                            state_dict[key] = f.get_tensor(key)
            
            if state_dict:
                pack_step.load_state_dict(state_dict)
                print(f"✅ 从safetensors加载state_dict")
        
        # 检查是否有relative_action_stats
        if hasattr(pack_step, 'relative_action_stats') and pack_step.relative_action_stats is not None:
            print(f"✅ 找到relative_action_stats")
            
            # 转换为可序列化格式
            stats_dict = {}
            for comp_name, stats in pack_step.relative_action_stats.items():
                stats_dict[comp_name] = {
                    "min": stats["min"].cpu().tolist() if isinstance(stats["min"], torch.Tensor) else stats["min"],
                    "max": stats["max"].cpu().tolist() if isinstance(stats["max"], torch.Tensor) else stats["max"],
                    "count": int(stats["count"].item() if isinstance(stats["count"], torch.Tensor) else stats["count"]),
                }
            return stats_dict
        
        print("⚠️  preprocessor中未找到relative_action_stats")
        return None
        
    except Exception as e:
        print(f"⚠️  从preprocessor加载失败: {e}")
        return None


def load_relative_action_stats_from_json(ckpt_path: str) -> dict[str, Any] | None:
    """从JSON配置文件中加载relative_action_stats"""
    json_path = Path(ckpt_path) / "policy_preprocessor.json"
    
    if not json_path.exists():
        print(f"⚠️  JSON文件不存在: {json_path}")
        return None
    
    print(f"\n📄 从JSON文件加载relative_action_stats: {json_path}")
    
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # 查找包含relative_action_stats的step
    # relative_action_stats在config字段内
    if "steps" in config:
        for step_config in config["steps"]:
            if "config" in step_config and "relative_action_stats" in step_config["config"]:
                print(f"✅ 在JSON中找到relative_action_stats")
                return step_config["config"]["relative_action_stats"]
    
    print("⚠️  JSON中未找到relative_action_stats")
    return None


def load_relative_action_stats_from_safetensors(ckpt_path: str) -> dict[str, Any] | None:
    """从safetensors文件中加载relative_action_stats"""
    ckpt_dir = Path(ckpt_path)
    
    # 查找所有safetensors文件
    safetensors_files = list(ckpt_dir.glob("**/*groot_pack_inputs*.safetensors"))
    
    if not safetensors_files:
        print(f"⚠️  未找到safetensors文件")
        return None
    
    print(f"\n🔒 从safetensors文件加载relative_action_stats...")
    
    relative_stats = {}
    
    for safetensors_file in safetensors_files:
        print(f"   检查文件: {safetensors_file.name}")
        
        with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
            keys = f.keys()
            
            for key in keys:
                if key.startswith("relative_action."):
                    # 格式: "relative_action.{comp_name}.{stat_name}"
                    parts = key.split(".")
                    if len(parts) == 3:
                        _, comp_name, stat_name = parts
                        
                        if comp_name not in relative_stats:
                            relative_stats[comp_name] = {}
                        
                        tensor = f.get_tensor(key)
                        # 转换为Python原生类型
                        if stat_name == "count":
                            relative_stats[comp_name][stat_name] = int(tensor.item())
                        else:
                            relative_stats[comp_name][stat_name] = tensor.tolist()
    
    if relative_stats:
        print(f"✅ 从safetensors加载到{len(relative_stats)}个组件的统计值")
        return relative_stats
    else:
        print("⚠️  safetensors中未找到relative_action_stats")
        return None


def compare_stats(
    loaded_stats: dict[str, Any],
    expected_stats: dict[str, dict[str, Any]],
    tolerance: float = 1e-6
) -> bool:
    """比较加载的统计值与期望值"""
    print(f"\n🔍 比较统计值 (tolerance={tolerance})...")
    
    all_match = True
    
    for comp_name, expected in expected_stats.items():
        if comp_name not in loaded_stats:
            print(f"❌ {comp_name}: 未找到")
            all_match = False
            continue
        
        loaded = loaded_stats[comp_name]
        
        # 比较count
        expected_count = expected["count"]
        loaded_count = loaded["count"]
        count_match = expected_count == loaded_count
        
        # 比较min
        expected_min = expected["min"]
        loaded_min = loaded["min"]
        min_match = all(
            abs(e - l) < tolerance
            for e, l in zip(expected_min, loaded_min)
        )
        
        # 比较max
        expected_max = expected["max"]
        loaded_max = loaded["max"]
        max_match = all(
            abs(e - l) < tolerance
            for e, l in zip(expected_max, loaded_max)
        )
        
        print(f"\n📋 {comp_name}:")
        print(f"   Count: 期望={expected_count}, 加载={loaded_count}, {'✅' if count_match else '❌'}")
        print(f"   Min:   期望={expected_min}")
        print(f"         加载={loaded_min}, {'✅' if min_match else '❌'}")
        print(f"   Max:   期望={expected_max}")
        print(f"         加载={loaded_max}, {'✅' if max_match else '❌'}")
        
        if not (count_match and min_match and max_match):
            all_match = False
    
    return all_match


def main():
    parser = argparse.ArgumentParser(description="检查relative action统计值")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Checkpoint路径 (例如: outputs/train/xxx/checkpoints/020000/pretrained_model)",
    )
    parser.add_argument(
        "--expected-left-count",
        type=int,
        default=14784,
        help="期望的left_eef_pos count值",
    )
    parser.add_argument(
        "--expected-right-count",
        type=int,
        default=14768,
        help="期望的right_eef_pos count值",
    )
    parser.add_argument(
        "--expected-left-min",
        type=float,
        nargs=3,
        default=[-0.4000306725502014, -0.2921929955482483, -0.27782565355300903],
        help="期望的left_eef_pos min值 (x y z)",
    )
    parser.add_argument(
        "--expected-left-max",
        type=float,
        nargs=3,
        default=[0.40717843174934387, 0.10939988493919373, 0.28066787123680115],
        help="期望的left_eef_pos max值 (x y z)",
    )
    parser.add_argument(
        "--expected-right-min",
        type=float,
        nargs=3,
        default=[-0.4208398759365082, -0.10475221276283264, -0.278958797454834],
        help="期望的right_eef_pos min值 (x y z)",
    )
    parser.add_argument(
        "--expected-right-max",
        type=float,
        nargs=3,
        default=[0.424162358045578, 0.2723754942417145, 0.27899107336997986],
        help="期望的right_eef_pos max值 (x y z)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="数值比较的容差",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("检查Relative Action统计值")
    print("=" * 80)
    print(f"Checkpoint路径: {args.ckpt_path}")
    
    # 期望的统计值（从训练日志）
    expected_stats = {
        "left_eef_pos": {
            "count": args.expected_left_count,
            "min": args.expected_left_min,
            "max": args.expected_left_max,
        },
        "right_eef_pos": {
            "count": args.expected_right_count,
            "min": args.expected_right_min,
            "max": args.expected_right_max,
        },
    }
    
    print(f"\n期望的统计值:")
    for comp_name, stats in expected_stats.items():
        print(f"  {comp_name}:")
        print(f"    count={stats['count']}")
        print(f"    min={stats['min']}")
        print(f"    max={stats['max']}")
    
    # 尝试从不同来源加载统计值
    loaded_stats = None
    
    # 1. 优先从JSON加载（最简单直接）
    loaded_stats = load_relative_action_stats_from_json(args.ckpt_path)
    
    # 2. 如果失败，尝试从safetensors加载
    if loaded_stats is None:
        loaded_stats = load_relative_action_stats_from_safetensors(args.ckpt_path)
    
    # 3. 最后尝试从preprocessor加载（需要处理relative_action_stats不是init参数的问题）
    if loaded_stats is None:
        loaded_stats = load_relative_action_stats_from_preprocessor(args.ckpt_path)
    
    if loaded_stats is None:
        print("\n❌ 无法从任何来源加载relative_action_stats")
        return 1
    
    print(f"\n✅ 成功加载统计值:")
    for comp_name, stats in loaded_stats.items():
        print(f"  {comp_name}:")
        print(f"    count={stats['count']}")
        print(f"    min={stats['min']}")
        print(f"    max={stats['max']}")
    
    # 比较统计值
    matches = compare_stats(loaded_stats, expected_stats, tolerance=args.tolerance)
    
    if matches:
        print("\n" + "=" * 80)
        print("✅ 所有统计值匹配！")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ 统计值不匹配！")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
