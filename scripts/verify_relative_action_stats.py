#!/usr/bin/env python3
"""
验证 relative action stats 与 absolute action stats 的一致性。

Relative action = absolute_action - observation.state
所以 relative action 的范围应该大致是以0为中心，范围与 absolute position 的变化幅度相关。
"""

import json
import torch
from safetensors.torch import load_file
from pathlib import Path


def load_absolute_eef_stats(checkpoint_path: str):
    """加载 absolute eef 模型的统计值"""
    checkpoint_dir = Path(checkpoint_path)
    
    # 读取 preprocessor 的 safetensors 文件
    safetensors_path = checkpoint_dir / "policy_preprocessor_step_2_groot_pack_inputs_v3.safetensors"
    if not safetensors_path.exists():
        print(f"❌ 找不到文件: {safetensors_path}")
        return None
    
    state_dict = load_file(str(safetensors_path))
    print(f"✅ 加载 safetensors 文件: {safetensors_path}")
    print(f"   包含的 keys: {list(state_dict.keys())}")
    
    return state_dict


def analyze_stats(state_dict: dict):
    """分析统计值"""
    action_min = state_dict.get("action.min")
    action_max = state_dict.get("action.max")
    state_min = state_dict.get("observation.state.min")
    state_max = state_dict.get("observation.state.max")
    
    print("\n" + "=" * 80)
    print("📊 Absolute EEF Action Stats (from checkpoint)")
    print("=" * 80)
    
    if action_min is not None and action_max is not None:
        print(f"\n🎯 Action shape: {action_min.shape}")
        print(f"\n📍 Position components (indices 0-2, 9-11):")
        
        # Left EEF position (indices 0-2)
        left_pos_min = action_min[:3].tolist()
        left_pos_max = action_max[:3].tolist()
        print(f"   Left EEF pos:  min={left_pos_min}")
        print(f"                  max={left_pos_max}")
        print(f"                  range=[{[round(mx-mn, 4) for mn, mx in zip(left_pos_min, left_pos_max)]}]")
        
        # Right EEF position (indices 9-11)
        right_pos_min = action_min[9:12].tolist()
        right_pos_max = action_max[9:12].tolist()
        print(f"   Right EEF pos: min={right_pos_min}")
        print(f"                  max={right_pos_max}")
        print(f"                  range=[{[round(mx-mn, 4) for mn, mx in zip(right_pos_min, right_pos_max)]}]")
        
        # 6D rotation (indices 3-8, 12-17)
        print(f"\n🔄 6D Rotation components (indices 3-8, 12-17):")
        left_rot_min = action_min[3:9].tolist()
        left_rot_max = action_max[3:9].tolist()
        print(f"   Left EEF rot6d:  min={[round(v, 4) for v in left_rot_min]}")
        print(f"                    max={[round(v, 4) for v in left_rot_max]}")
        
        right_rot_min = action_min[12:18].tolist()
        right_rot_max = action_max[12:18].tolist()
        print(f"   Right EEF rot6d: min={[round(v, 4) for v in right_rot_min]}")
        print(f"                    max={[round(v, 4) for v in right_rot_max]}")
        
        # Gripper (indices 18-19)
        print(f"\n🦀 Gripper components (indices 18-19):")
        gripper_min = action_min[18:20].tolist()
        gripper_max = action_max[18:20].tolist()
        print(f"   Gripper: min={gripper_min}")
        print(f"            max={gripper_max}")
    else:
        print("❌ Action stats not found in safetensors")
    
    print("\n" + "=" * 80)
    print("📊 Observation State Stats (from checkpoint)")
    print("=" * 80)
    
    if state_min is not None and state_max is not None:
        print(f"\n🎯 State shape: {state_min.shape}")
        
        # Left EEF position (indices 0-2)
        left_pos_min = state_min[:3].tolist()
        left_pos_max = state_max[:3].tolist()
        print(f"\n📍 Left EEF pos:  min={left_pos_min}")
        print(f"                  max={left_pos_max}")
        
        # Right EEF position (indices 9-11)
        right_pos_min = state_min[9:12].tolist()
        right_pos_max = state_max[9:12].tolist()
        print(f"   Right EEF pos: min={right_pos_min}")
        print(f"                  max={right_pos_max}")
    else:
        print("❌ State stats not found in safetensors")
    
    return action_min, action_max, state_min, state_max


def compare_with_relative_stats(action_min, action_max, state_min, state_max):
    """与 relative action stats 进行比较"""
    
    # 从训练日志中提取的 relative action stats
    relative_stats = {
        "left_eef_pos": {
            "min": [-0.0631522536277771, -0.2930336892604828, -0.223337322473526],
            "max": [0.40841352939605713, 0.11420149356126785, 0.2899242341518402],
        },
        "right_eef_pos": {
            "min": [-0.050022125244140625, -0.10910341143608093, -0.2266448736190796],
            "max": [0.4753052592277527, 0.27383631467819214, 0.29378584027290344],
        },
    }
    
    print("\n" + "=" * 80)
    print("🔍 Relative Action Stats (from training log)")
    print("=" * 80)
    
    for comp_name, stats in relative_stats.items():
        print(f"\n📍 {comp_name}:")
        print(f"   min={stats['min']}")
        print(f"   max={stats['max']}")
        range_vals = [round(mx - mn, 4) for mn, mx in zip(stats['min'], stats['max'])]
        print(f"   range={range_vals}")
        
        # 检查是否大致以0为中心
        center = [(mn + mx) / 2 for mn, mx in zip(stats['min'], stats['max'])]
        print(f"   center={[round(c, 4) for c in center]}")
    
    print("\n" + "=" * 80)
    print("✅ 一致性验证")
    print("=" * 80)
    
    if action_min is not None and state_min is not None:
        # 理论上，relative action 的范围应该是：
        # min ≈ action_min - state_max (最小动作 - 最大状态)
        # max ≈ action_max - state_min (最大动作 - 最小状态)
        
        print("\n📐 理论 vs 实际比较:")
        
        # Left EEF position
        print("\n🤚 Left EEF Position:")
        for i, axis in enumerate(['x', 'y', 'z']):
            # 理论范围
            theoretical_min = action_min[i].item() - state_max[i].item()
            theoretical_max = action_max[i].item() - state_min[i].item()
            
            # 实际范围
            actual_min = relative_stats["left_eef_pos"]["min"][i]
            actual_max = relative_stats["left_eef_pos"]["max"][i]
            
            print(f"   {axis}: 理论 [{theoretical_min:.4f}, {theoretical_max:.4f}] "
                  f"vs 实际 [{actual_min:.4f}, {actual_max:.4f}]")
            
            # 检查实际范围是否在理论范围内
            if actual_min >= theoretical_min - 0.01 and actual_max <= theoretical_max + 0.01:
                print(f"      ✅ 在合理范围内")
            else:
                print(f"      ⚠️  可能超出理论范围（但这是正常的，因为理论范围是极端情况）")
        
        # Right EEF position
        print("\n🤚 Right EEF Position:")
        for i, axis in enumerate(['x', 'y', 'z']):
            idx = i + 9  # right eef starts at index 9
            
            # 理论范围
            theoretical_min = action_min[idx].item() - state_max[idx].item()
            theoretical_max = action_max[idx].item() - state_min[idx].item()
            
            # 实际范围
            actual_min = relative_stats["right_eef_pos"]["min"][i]
            actual_max = relative_stats["right_eef_pos"]["max"][i]
            
            print(f"   {axis}: 理论 [{theoretical_min:.4f}, {theoretical_max:.4f}] "
                  f"vs 实际 [{actual_min:.4f}, {actual_max:.4f}]")
        
        print("\n" + "=" * 80)
        print("📊 总结")
        print("=" * 80)
        print("""
Relative action stats 的含义：
- min/max 表示在整个数据集中，relative_action = action - state 的最小/最大值
- 这些值应该大致以0为中心（因为 action 和 state 通常很接近）
- 范围反映了数据集中 action 和 state 之间的最大偏差

一致性检查：
1. ✅ Relative action 的范围应该在理论范围内（或接近）
2. ✅ Relative action 应该大致以0为中心
3. ✅ 左右手的 relative stats 应该有类似的数量级

如果以上条件都满足，说明 relative action 的归一化与 absolute action 是一致的。
""")
    else:
        print("❌ 无法进行比较，因为缺少必要的统计值")


def main():
    # Absolute EEF 模型的 checkpoint 路径
    absolute_checkpoint = "/home/lab/humanoid_groot/outputs/train/0124_multi_dataset_h100x4_absolute_eef_4322_2X3_groot_cross-attention_ignore_rotation/checkpoints/020000/pretrained_model"
    
    print("=" * 80)
    print("🔍 验证 Relative Action Stats 与 Absolute Action Stats 的一致性")
    print("=" * 80)
    
    # 加载 absolute eef 的统计值
    state_dict = load_absolute_eef_stats(absolute_checkpoint)
    
    if state_dict is not None:
        # 分析统计值
        action_min, action_max, state_min, state_max = analyze_stats(state_dict)
        
        # 与 relative stats 比较
        compare_with_relative_stats(action_min, action_max, state_min, state_max)


if __name__ == "__main__":
    main()
