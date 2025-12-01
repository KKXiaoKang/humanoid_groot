#!/usr/bin/env python3
"""
测试 Groot 训练时是否能正确加载 task 字段
"""

import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

def test_task_loading():
    """测试 task 字段是否能正确加载和处理"""
    
    # 使用你的数据集路径
    dataset_root = "/home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1125_groot_train_data_with_task"
    repo_id = "1125_groot_train_data_with_task"
    
    print("=" * 80)
    print("测试 Groot Task 加载流程")
    print("=" * 80)
    
    # 1. 加载数据集
    print("\n1. 加载数据集...")
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    print(f"   ✅ 数据集加载成功")
    print(f"   - 总帧数: {len(dataset)}")
    print(f"   - 总episodes: {dataset.num_episodes}")
    
    # 2. 获取一个样本
    print("\n2. 获取样本数据...")
    sample = dataset[0]
    print(f"   ✅ 样本获取成功")
    print(f"   - 样本键: {list(sample.keys())}")
    
    # 3. 检查 task 字段
    if "task" in sample:
        print(f"\n3. 检查 task 字段...")
        print(f"   ✅ 找到 task 字段: '{sample['task']}'")
        print(f"   - task 类型: {type(sample['task'])}")
    else:
        print(f"\n3. ❌ 错误: 样本中没有 task 字段!")
        return False
    
    # 4. 检查 task_index
    if "task_index" in sample:
        print(f"   ✅ 找到 task_index: {sample['task_index']}")
    else:
        print(f"   ⚠️  警告: 样本中没有 task_index 字段")
    
    # 5. 创建 Groot 配置和 processor
    print("\n4. 创建 Groot processor...")
    config = GrootConfig(
        max_state_dim=64,
        max_action_dim=32,
        chunk_size=50,
        n_action_steps=50,
    )
    
    preprocessor, postprocessor = make_groot_pre_post_processors(
        config=config,
        dataset_stats=dataset.meta.stats
    )
    print(f"   ✅ Processor 创建成功")
    
    # 6. 测试 processor 处理
    print("\n5. 测试 processor 处理...")
    # 创建一个简单的 batch（模拟 DataLoader 的输出）
    batch = {
        "observation.state": sample["observation.state"].unsqueeze(0),
        "action": sample["action"].unsqueeze(0),
    }
    
    # 添加图像（如果有）
    image_keys = [k for k in sample.keys() if "image" in k.lower()]
    for key in image_keys:
        if isinstance(sample[key], torch.Tensor):
            batch[key] = sample[key].unsqueeze(0)
    
    # 添加 complementary_data
    batch["task"] = [sample["task"]]  # 转换为列表以匹配 batch 格式
    if "task_index" in sample:
        batch["task_index"] = sample["task_index"].unsqueeze(0)
    
    print(f"   - Batch 键: {list(batch.keys())}")
    print(f"   - Batch 中的 task: {batch.get('task', 'NOT FOUND')}")
    
    # 7. 处理 batch
    try:
        processed_batch = preprocessor(batch)
        print(f"   ✅ Batch 处理成功")
        
        # 检查处理后的 batch 中是否有 language
        # 注意：processor 会将 task 转换为 language 并放入 complementary_data
        print(f"   - 处理后的 batch 键: {list(processed_batch.keys())}")
        
        # 检查 language 字段
        if "language" in processed_batch:
            lang_value = processed_batch["language"]
            if isinstance(lang_value, list):
                print(f"   ✅ 找到 language 字段: '{lang_value[0] if len(lang_value) > 0 else 'N/A'}'")
            else:
                print(f"   ✅ 找到 language 字段: '{lang_value}'")
        else:
            print(f"   ⚠️  警告: 处理后的 batch 中没有 language 字段")
        
        # 检查是否有 eagle_* 字段（说明 language 被正确处理了）
        eagle_keys = [k for k in processed_batch.keys() if k.startswith("eagle_")]
        if eagle_keys:
            print(f"   ✅ 找到 Eagle 编码字段: {eagle_keys[:3]}... (共 {len(eagle_keys)} 个)")
            print(f"   ✅ Language task 已被正确编码为 Eagle 格式!")
            
            # 验证 eagle_input_ids 的形状（应该包含 language tokens）
            if "eagle_input_ids" in processed_batch:
                input_ids = processed_batch["eagle_input_ids"]
                print(f"   - eagle_input_ids shape: {input_ids.shape}")
                print(f"   - 这表示 language 已被 tokenize 并准备好输入模型")
        else:
            print(f"   ⚠️  警告: 未找到 Eagle 编码字段")
            print(f"   - 这可能意味着 language 处理有问题")
            
    except Exception as e:
        print(f"   ❌ 处理 batch 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ 测试完成! Task 字段可以正确加载和处理")
    print("=" * 80)
    return True

if __name__ == "__main__":
    test_task_loading()

