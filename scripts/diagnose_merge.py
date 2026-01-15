#!/usr/bin/env python
"""
诊断模型融合问题的脚本

这个脚本会检查：
1. 原始模型的权重键名和形状
2. 融合后保存的权重键名和形状
3. 加载时是否有权重丢失
4. action_dim 的处理是否正确
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from pathlib import Path
from safetensors.torch import load_file
import json


def load_weights(path: str) -> dict:
    """加载模型权重"""
    path = Path(path)
    import glob
    safetensors_files = glob.glob(str(path / "model*.safetensors"))
    state_dict = {}
    for f in sorted(safetensors_files):
        state_dict.update(load_file(f))
    return state_dict


def compare_weights(name1: str, weights1: dict, name2: str, weights2: dict):
    """比较两个权重字典"""
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())
    
    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    print(f"\n{'='*60}")
    print(f"比较 {name1} vs {name2}")
    print(f"{'='*60}")
    print(f"{name1} 键数: {len(keys1)}")
    print(f"{name2} 键数: {len(keys2)}")
    print(f"共同键数: {len(common_keys)}")
    print(f"仅在 {name1}: {len(only_in_1)}")
    print(f"仅在 {name2}: {len(only_in_2)}")
    
    if only_in_1:
        print(f"\n仅在 {name1} 中的键 (前20个):")
        for k in list(only_in_1)[:20]:
            print(f"  - {k}: {weights1[k].shape}")
    
    if only_in_2:
        print(f"\n仅在 {name2} 中的键 (前20个):")
        for k in list(only_in_2)[:20]:
            print(f"  - {k}: {weights2[k].shape}")
    
    # 检查形状不匹配
    shape_mismatch = []
    for k in common_keys:
        if weights1[k].shape != weights2[k].shape:
            shape_mismatch.append((k, weights1[k].shape, weights2[k].shape))
    
    if shape_mismatch:
        print(f"\n形状不匹配的键 ({len(shape_mismatch)} 个):")
        for k, s1, s2 in shape_mismatch[:20]:
            print(f"  - {k}: {s1} vs {s2}")
    else:
        print(f"\n✅ 所有共同键的形状都匹配")
    
    return common_keys, only_in_1, only_in_2, shape_mismatch


def check_action_head_dims(weights: dict, name: str):
    """检查 action_head 相关的维度"""
    print(f"\n{'='*60}")
    print(f"检查 {name} 的 action_head 维度")
    print(f"{'='*60}")
    
    # 检查关键层的维度
    key_layers = [
        "action_head.action_encoder.W1.W",  # (num_embodiments, action_dim, hidden)
        "action_head.action_decoder.layer1.W",  # 如果有
        "action_head.action_arm_decoder.layer1.W",  # 如果有
        "action_head.action_claw_decoder.layer1.W",  # 如果有
        "action_head.action_left_arm_decoder.layer1.W",  # 如果有
        "action_head.action_right_arm_decoder.layer1.W",  # 如果有
        "action_head.shared_arm_decoder.shared_layer.W",  # 如果有
        "action_head.future_tokens.weight",
    ]
    
    for key in key_layers:
        if key in weights:
            shape = weights[key].shape
            print(f"  {key}: {shape}")
            
            # 特殊检查 action_encoder.W1
            if "action_encoder.W1.W" in key:
                print(f"    -> action_dim (输入维度): {shape[1]}")
            
            # 检查 future_tokens
            if "future_tokens.weight" in key:
                print(f"    -> num_target_vision_tokens: {shape[0]}")
    
    # 检查所有 action_head 相关的键
    action_head_keys = [k for k in weights.keys() if k.startswith("action_head.")]
    print(f"\n  action_head 相关的键总数: {len(action_head_keys)}")
    
    # 检查是否有 decoder 相关的层
    decoder_keys = [k for k in action_head_keys if "decoder" in k]
    print(f"  decoder 相关的键总数: {len(decoder_keys)}")
    
    for k in decoder_keys[:10]:
        print(f"    - {k}: {weights[k].shape}")


def check_config(path: str):
    """检查模型配置"""
    config_path = Path(path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"检查 {path} 的配置")
        print(f"{'='*60}")
        
        # 检查 action 相关配置
        print(f"  action_dim: {config.get('action_dim', 'N/A')}")
        print(f"  action_horizon: {config.get('action_horizon', 'N/A')}")
        
        action_head_cfg = config.get('action_head_cfg', {})
        print(f"\n  action_head_cfg:")
        print(f"    action_dim: {action_head_cfg.get('action_dim', 'N/A')}")
        print(f"    action_horizon: {action_head_cfg.get('action_horizon', 'N/A')}")
        print(f"    pretrained_action_dim: {action_head_cfg.get('pretrained_action_dim', 'N/A')}")
        print(f"    use_multi_action_heads: {action_head_cfg.get('use_multi_action_heads', 'N/A')}")
        print(f"    split_arm_heads: {action_head_cfg.get('split_arm_heads', 'N/A')}")
        print(f"    action_arm_dim: {action_head_cfg.get('action_arm_dim', 'N/A')}")
        print(f"    action_claw_dim: {action_head_cfg.get('action_claw_dim', 'N/A')}")
        print(f"    action_left_arm_dim: {action_head_cfg.get('action_left_arm_dim', 'N/A')}")
        print(f"    action_right_arm_dim: {action_head_cfg.get('action_right_arm_dim', 'N/A')}")
        print(f"    num_target_vision_tokens: {action_head_cfg.get('num_target_vision_tokens', 'N/A')}")
        print(f"    hidden_size: {action_head_cfg.get('hidden_size', 'N/A')}")
        print(f"    backbone_embedding_dim: {action_head_cfg.get('backbone_embedding_dim', 'N/A')}")
        
        return config
    else:
        print(f"\n⚠️ 找不到配置文件: {config_path}")
        return None


def check_merge_config(path: str):
    """检查融合配置"""
    merge_config_path = Path(path) / "merge_config.json"
    if merge_config_path.exists():
        with open(merge_config_path) as f:
            config = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"检查 {path} 的融合配置")
        print(f"{'='*60}")
        
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        return config
    else:
        print(f"\n⚠️ 找不到融合配置文件: {merge_config_path}")
        return None


def simulate_model_load(model_path: str):
    """模拟模型加载，检查权重加载情况"""
    print(f"\n{'='*60}")
    print(f"模拟加载模型: {model_path}")
    print(f"{'='*60}")
    
    # 加载保存的权重
    saved_weights = load_weights(model_path)
    print(f"保存的权重键数: {len(saved_weights)}")
    
    # 分类权重
    backbone_keys = [k for k in saved_weights.keys() if k.startswith("backbone.")]
    action_head_keys = [k for k in saved_weights.keys() if k.startswith("action_head.")]
    adapter_keys = [k for k in saved_weights.keys() if k.startswith("distribution_adapter.")]
    other_keys = [k for k in saved_weights.keys() 
                  if not k.startswith("backbone.") 
                  and not k.startswith("action_head.") 
                  and not k.startswith("distribution_adapter.")]
    
    print(f"  backbone 键数: {len(backbone_keys)}")
    print(f"  action_head 键数: {len(action_head_keys)}")
    print(f"  distribution_adapter 键数: {len(adapter_keys)}")
    print(f"  其他键数: {len(other_keys)}")
    
    if other_keys:
        print(f"\n  其他键 (前10个):")
        for k in other_keys[:10]:
            print(f"    - {k}")
    
    if adapter_keys:
        print(f"\n  适配层键 (前10个):")
        for k in adapter_keys[:10]:
            print(f"    - {k}: {saved_weights[k].shape}")
    
    # 尝试加载模型并检查
    try:
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        
        print(f"\n尝试使用 GrootPolicy.from_pretrained 加载模型...")
        policy = GrootPolicy.from_pretrained(Path(model_path), strict=False)
        
        # 获取模型的 state_dict
        model_state_dict = policy._groot_model.state_dict()
        print(f"\n加载后模型的权重键数: {len(model_state_dict)}")
        
        # 检查哪些权重被加载了
        loaded_keys = set()
        for k in saved_weights.keys():
            if k in model_state_dict:
                # 检查权重是否一致（使用一个简单的检查）
                if saved_weights[k].shape == model_state_dict[k].shape:
                    loaded_keys.add(k)
        
        not_loaded = set(saved_weights.keys()) - loaded_keys - set(adapter_keys)
        
        print(f"成功加载的权重键数: {len(loaded_keys)}")
        print(f"未加载的权重键数 (不含适配层): {len(not_loaded)}")
        
        if not_loaded:
            print(f"\n⚠️ 未加载的权重键 (前20个):")
            for k in list(not_loaded)[:20]:
                print(f"    - {k}: {saved_weights[k].shape}")
        
        # 检查 action_head 配置
        action_head = policy._groot_model.action_head
        print(f"\n模型 action_head 配置:")
        print(f"  action_dim: {action_head.action_dim}")
        print(f"  action_horizon: {action_head.action_horizon}")
        print(f"  encoder_action_dim: {action_head.encoder_action_dim}")
        print(f"  actual_action_dim: {action_head.actual_action_dim}")
        
        # 检查 future_tokens
        future_tokens = action_head.future_tokens.weight
        print(f"  future_tokens.weight.shape: {future_tokens.shape}")
        
        # 检查配置
        config = action_head.config
        print(f"  config.num_target_vision_tokens: {config.num_target_vision_tokens}")
        print(f"  config.use_multi_action_heads: {config.use_multi_action_heads}")
        print(f"  config.split_arm_heads: {getattr(config, 'split_arm_heads', 'N/A')}")
        
    except Exception as e:
        print(f"\n❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="诊断模型融合问题")
    parser.add_argument("--narrower", type=str, 
                       default="/home/kangkk/humanoid_groot_base/outputs/0112_h100x4_groot_cross_attention_narrower_very_conservative/checkpoints/020000/pretrained_model",
                       help="narrower 模型路径")
    parser.add_argument("--wider", type=str,
                       default="/home/kangkk/humanoid_groot_base/outputs/0113_h100x4_groot_cross_attention_wider_very_conservative_mix_dense/checkpoints/014000/pretrained_model",
                       help="wider 模型路径")
    parser.add_argument("--merged", type=str,
                       default="./outputs/merged_groot_mergevla/pretrained_model",
                       help="融合后模型路径")
    
    args = parser.parse_args()
    
    print("="*60)
    print("模型融合诊断")
    print("="*60)
    
    # 1. 检查原始模型
    print("\n\n" + "="*60)
    print("第1部分：检查原始模型")
    print("="*60)
    
    if Path(args.narrower).exists():
        check_config(args.narrower)
        narrower_weights = load_weights(args.narrower)
        check_action_head_dims(narrower_weights, "narrower")
    else:
        print(f"⚠️ narrower 模型不存在: {args.narrower}")
        narrower_weights = None
    
    if Path(args.wider).exists():
        check_config(args.wider)
        wider_weights = load_weights(args.wider)
        check_action_head_dims(wider_weights, "wider")
    else:
        print(f"⚠️ wider 模型不存在: {args.wider}")
        wider_weights = None
    
    # 2. 比较原始模型
    if narrower_weights and wider_weights:
        compare_weights("narrower", narrower_weights, "wider", wider_weights)
    
    # 3. 检查融合后模型
    print("\n\n" + "="*60)
    print("第2部分：检查融合后模型")
    print("="*60)
    
    if Path(args.merged).exists():
        check_config(args.merged)
        check_merge_config(args.merged)
        merged_weights = load_weights(args.merged)
        check_action_head_dims(merged_weights, "merged")
        
        # 比较融合后与原始
        if narrower_weights:
            compare_weights("narrower", narrower_weights, "merged", merged_weights)
        
        # 模拟加载
        simulate_model_load(args.merged)
    else:
        print(f"⚠️ 融合模型不存在: {args.merged}")
    
    print("\n\n" + "="*60)
    print("诊断完成")
    print("="*60)


if __name__ == "__main__":
    main()
