#!/usr/bin/env python
"""
诊断脚本：检查 Groot 训练流程中图像是否被正确使用
"""

import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

def check_dataset_images(dataset_root: str):
    """检查数据集中是否包含图像"""
    print("=" * 60)
    print("1. 检查数据集中的图像")
    print("=" * 60)
    
    dataset = LeRobotDataset(dataset_root)
    
    # 获取一个样本
    sample = dataset[0]
    
    print(f"\n数据集样本的键: {list(sample.keys())}")
    
    # 检查图像键
    image_keys = [k for k in sample.keys() if "image" in k.lower()]
    print(f"\n图像相关的键: {image_keys}")
    
    for key in image_keys:
        if key in sample:
            img = sample[key]
            if isinstance(img, torch.Tensor):
                print(f"  {key}: shape={img.shape}, dtype={img.dtype}, min={img.min():.3f}, max={img.max():.3f}")
            else:
                print(f"  {key}: type={type(img)}")
    
    # 检查 state 和 action
    if "observation.state" in sample:
        state = sample["observation.state"]
        print(f"\nobservation.state: shape={state.shape if hasattr(state, 'shape') else 'N/A'}")
    
    if "action" in sample:
        action = sample["action"]
        print(f"action: shape={action.shape if hasattr(action, 'shape') else 'N/A'}")
    
    return sample

def check_preprocessor(dataset_root: str, config: GrootConfig):
    """检查预处理器是否正确处理图像"""
    print("\n" + "=" * 60)
    print("2. 检查预处理器")
    print("=" * 60)
    
    dataset = LeRobotDataset(dataset_root)
    sample = dataset[0]
    
    # 创建预处理器
    preprocessor, postprocessor = make_groot_pre_post_processors(config, dataset_stats=None)
    
    # 处理样本
    processed = preprocessor(sample)
    
    print(f"\n预处理后的键: {list(processed.keys())}")
    
    # 检查是否有 eagle_* 键
    eagle_keys = [k for k in processed.keys() if k.startswith("eagle_")]
    print(f"\nEagle 编码的键 ({len(eagle_keys)} 个):")
    for key in eagle_keys:
        val = processed[key]
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: type={type(val)}")
    
    # 检查是否有 video 键（应该在 Eagle 编码后被删除）
    if "video" in processed:
        print(f"\n⚠️  警告: 'video' 键仍然存在，可能 Eagle 编码未执行")
    else:
        print(f"\n✓ 'video' 键已被删除（Eagle 编码后）")
    
    # 检查是否有原始图像键
    original_img_keys = [k for k in processed.keys() if k.startswith("observation.images.")]
    if original_img_keys:
        print(f"\n⚠️  警告: 原始图像键仍然存在: {original_img_keys}")
    else:
        print(f"\n✓ 原始图像键已被删除")
    
    return processed

def check_model_inputs(config: GrootConfig, processed_sample: dict):
    """检查模型输入"""
    print("\n" + "=" * 60)
    print("3. 检查模型输入")
    print("=" * 60)
    
    # 创建模型（不加载权重，只检查输入）
    policy = GrootPolicy(config)
    
    # 模拟 batch（添加 batch 维度）
    batch = {}
    for k, v in processed_sample.items():
        if isinstance(v, torch.Tensor):
            if v.dim() == 0:
                batch[k] = v.unsqueeze(0)
            elif v.dim() == 1:
                batch[k] = v.unsqueeze(0)
            else:
                batch[k] = v.unsqueeze(0) if v.dim() < 3 else v
        else:
            batch[k] = v
    
    # 检查 forward 方法会接收什么
    allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
    groot_inputs = {
        k: v
        for k, v in batch.items()
        if (k in allowed_base or k.startswith("eagle_")) and not (k.startswith("next.") or k == "info")
    }
    
    print(f"\n模型 forward 会接收的键 ({len(groot_inputs)} 个):")
    for key in sorted(groot_inputs.keys()):
        val = groot_inputs[key]
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: type={type(val)}")
    
    # 检查是否有 eagle_* 键
    eagle_keys = [k for k in groot_inputs.keys() if k.startswith("eagle_")]
    if eagle_keys:
        print(f"\n✓ 找到 {len(eagle_keys)} 个 eagle_* 键，图像应该被正确使用")
    else:
        print(f"\n⚠️  警告: 没有找到 eagle_* 键！图像可能没有被正确编码")
    
    return groot_inputs

def check_backbone_forward(config: GrootConfig, groot_inputs: dict):
    """检查 backbone 是否正确处理 eagle 输入"""
    print("\n" + "=" * 60)
    print("4. 检查 Backbone 处理")
    print("=" * 60)
    
    policy = GrootPolicy(config)
    backbone = policy._groot_model.backbone
    
    # 检查 backbone 的 forward_eagle 方法
    from transformers.feature_extraction_utils import BatchFeature
    
    # 构建 BatchFeature
    vl_input = BatchFeature(data=groot_inputs)
    
    # 检查 eagle_* 键
    eagle_prefix = "eagle_"
    eagle_input = {
        k.removeprefix(eagle_prefix): v 
        for k, v in vl_input.items() 
        if k.startswith(eagle_prefix)
    }
    
    print(f"\nBackbone 会提取的 eagle 输入键 ({len(eagle_input)} 个):")
    for key in sorted(eagle_input.keys()):
        val = eagle_input[key]
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: type={type(val)}")
    
    if eagle_input:
        print(f"\n✓ Backbone 应该能正确提取 eagle 输入")
    else:
        print(f"\n⚠️  警告: Backbone 无法提取 eagle 输入！")

def main():
    dataset_root = "/root/lerobot/lerobot_data/1118_sim_depalletize"
    
    # 创建配置
    config = GrootConfig(
        base_model_path="nvidia/GR00T-N1.5-3B",
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        max_state_dim=64,
        max_action_dim=32,
        chunk_size=50,
        n_action_steps=50,
    )
    
    # 设置输入特征（模拟数据集配置）
    from lerobot.configs.types import FeatureType, PolicyFeature
    
    config.input_features = {
        "observation.images.cam_head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.cam_chest": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.cam_left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.cam_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(18,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(18,)),
    }
    
    try:
        # 1. 检查数据集
        sample = check_dataset_images(dataset_root)
        
        # 2. 检查预处理器
        processed = check_preprocessor(dataset_root, config)
        
        # 3. 检查模型输入
        groot_inputs = check_model_inputs(config, processed)
        
        # 4. 检查 backbone
        check_backbone_forward(config, groot_inputs)
        
        print("\n" + "=" * 60)
        print("诊断完成！")
        print("=" * 60)
        print("\n建议:")
        print("1. 如果看到 eagle_* 键，说明图像被正确编码")
        print("2. 如果 loss 很低，可能是:")
        print("   - 数据集太小或太简单")
        print("   - 模型已经过拟合")
        print("   - 需要检查验证集上的表现")
        print("3. 可以通过打印 batch 中的键来确认训练时是否有 eagle_* 键")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

