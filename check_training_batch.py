#!/usr/bin/env python
"""检查训练时实际使用的batch，确认图像是否被正确使用"""

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.configs.types import FeatureType, PolicyFeature

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
    
    # 设置输入特征
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
    
    # 创建数据集和预处理器
    dataset = LeRobotDataset(dataset_root)
    preprocessor, _ = make_groot_pre_post_processors(config, dataset_stats=None)
    
    print("=" * 60)
    print("检查训练时实际使用的batch")
    print("=" * 60)
    
    # 模拟DataLoader的batch（跳过样本0，因为它的图像全为0）
    batch_indices = [100, 200, 300, 400]
    samples = [dataset[idx] for idx in batch_indices]
    
    print(f"\n检查 {len(samples)} 个样本的图像数据:")
    for i, (idx, sample) in enumerate(zip(batch_indices, samples)):
        print(f"\n样本 {idx}:")
        for key in ["observation.images.cam_head", "observation.images.cam_chest"]:
            if key in sample:
                img = sample[key]
                if isinstance(img, torch.Tensor):
                    print(f"  {key}: min={img.min():.3f}, max={img.max():.3f}, mean={img.mean():.3f}")
    
    # 模拟DataLoader的collate（PyTorch默认行为）
    print("\n" + "=" * 60)
    print("模拟DataLoader collate后的batch")
    print("=" * 60)
    
    # PyTorch DataLoader的默认collate会stack相同键的tensor
    batched = {}
    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                if key not in batched:
                    batched[key] = []
                batched[key].append(value)
            else:
                # 非tensor值，保持为列表
                if key not in batched:
                    batched[key] = []
                batched[key].append(value)
    
    # Stack tensors
    for key in list(batched.keys()):
        if isinstance(batched[key][0], torch.Tensor):
            batched[key] = torch.stack(batched[key])
    
    print(f"\nBatch键: {list(batched.keys())}")
    print(f"Batch大小: {batched['observation.images.cam_head'].shape[0] if 'observation.images.cam_head' in batched else 'N/A'}")
    
    # 检查batch中的图像
    for key in ["observation.images.cam_head", "observation.images.cam_chest"]:
        if key in batched:
            img_batch = batched[key]
            print(f"\n{key} (batch):")
            print(f"  shape: {img_batch.shape}")
            print(f"  min: {img_batch.min():.3f}, max: {img_batch.max():.3f}")
            print(f"  mean: {img_batch.mean():.3f}, std: {img_batch.std():.3f}")
    
    # 通过预处理器处理batch
    print("\n" + "=" * 60)
    print("通过预处理器处理batch")
    print("=" * 60)
    
    # 注意：预处理器期望单个transition，所以我们需要逐个处理
    processed_samples = []
    for sample in samples:
        processed = preprocessor(sample)
        processed_samples.append(processed)
    
    # 检查处理后的eagle_*键
    print(f"\n处理后的样本数量: {len(processed_samples)}")
    if processed_samples:
        first_processed = processed_samples[0]
        eagle_keys = [k for k in first_processed.keys() if k.startswith("eagle_")]
        print(f"\nEagle键 ({len(eagle_keys)} 个): {eagle_keys}")
        
        for key in eagle_keys[:3]:  # 只显示前3个
            val = first_processed[key]
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                if "pixel_values" in key:
                    print(f"    min={val.min():.3f}, max={val.max():.3f}, mean={val.mean():.3f}")
    
    # 检查所有样本的eagle_pixel_values
    print("\n" + "=" * 60)
    print("检查所有样本的eagle_pixel_values")
    print("=" * 60)
    
    for i, processed in enumerate(processed_samples):
        if "eagle_pixel_values" in processed:
            pv = processed["eagle_pixel_values"]
            print(f"\n样本 {batch_indices[i]} 的 eagle_pixel_values:")
            print(f"  shape: {pv.shape}")
            print(f"  min: {pv.min():.3f}, max: {pv.max():.3f}")
            print(f"  mean: {pv.mean():.3f}, std: {pv.std():.3f}")
            if pv.min() == 0 and pv.max() == 0:
                print(f"  ⚠️  警告: 图像全为0！")
            else:
                print(f"  ✓ 图像数据正常")

if __name__ == "__main__":
    main()

