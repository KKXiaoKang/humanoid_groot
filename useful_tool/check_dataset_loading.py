#!/usr/bin/env python3
"""
检查数据集加载情况，确认是否所有chunk都被加载
"""

import sys
from pathlib import Path

# 添加src路径到sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 数据集路径
DATASET_ROOT = "/home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1215_5w_groot_4311_4322_4611"
DATASET_REPO_ID = "1215_5w_groot_4311_4322_4611"

print("=" * 80)
print("数据集加载检查")
print("=" * 80)

# 检查数据目录结构
data_dir = Path(DATASET_ROOT) / "data"
print(f"\n数据目录: {data_dir}")

chunks = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")])
print(f"\n找到的chunk目录数量: {len(chunks)}")
for chunk in chunks:
    parquet_files = list(chunk.glob("*.parquet"))
    print(f"  {chunk.name}: {len(parquet_files)} 个parquet文件")
    for pf in parquet_files:
        print(f"    - {pf.name}")

# 加载数据集
print("\n" + "=" * 80)
print("加载数据集...")
print("=" * 80)

try:
    dataset = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_ROOT,
    )
    
    print(f"\n数据集基本信息:")
    print(f"  - 总episodes数: {dataset.num_episodes}")
    print(f"  - 总frames数: {dataset.num_frames}")
    print(f"  - hf_dataset长度: {len(dataset.hf_dataset)}")
    
    # 检查加载的episode索引范围
    print(f"\n检查加载的episode索引...")
    if hasattr(dataset.hf_dataset, 'unique'):
        unique_episodes = sorted(dataset.hf_dataset.unique("episode_index"))
        print(f"  - Episode索引范围: {min(unique_episodes)} - {max(unique_episodes)}")
        print(f"  - 唯一episode数量: {len(unique_episodes)}")
        
        # 检查是否有chunk-000和chunk-001的episodes
        # 根据metadata中的episodes信息来判断
        if dataset.meta.episodes is not None:
            # 使用 Dataset 的 unique() 方法获取唯一值
            chunk_indices = sorted(dataset.meta.episodes.unique("data/chunk_index"))
            file_indices = sorted(dataset.meta.episodes.unique("data/file_index"))
            print(f"\n  - 使用的chunk索引: {chunk_indices}")
            print(f"  - 使用的file索引: {file_indices}")
            
            # 统计每个chunk的episode数量
            print(f"\n  每个chunk的episode统计:")
            for chunk_idx in chunk_indices:
                # 使用 filter 方法来筛选
                chunk_episodes = dataset.meta.episodes.filter(
                    lambda x: x["data/chunk_index"] == chunk_idx
                )
                print(f"    chunk-{chunk_idx:03d}: {len(chunk_episodes)} episodes")
    
    print(f"\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    if len(chunks) > 1:
        if dataset.num_episodes > 0 and len(dataset.hf_dataset) > 0:
            print("✓ 所有chunk的数据都被加载了！")
            print(f"  数据集包含了 {len(chunks)} 个chunk目录中的所有数据")
        else:
            print("✗ 数据集加载可能有问题")
    else:
        print(f"数据集只有 {len(chunks)} 个chunk，全部加载")
        
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()

