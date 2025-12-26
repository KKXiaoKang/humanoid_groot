#!/usr/bin/env python3
"""
统计数据集中4311、4611、4322三类箱体的数据条目数
直接读取parquet文件，只读取task_index列，不加载图片数据
支持对"Pick up"任务进行降采样
"""

import sys
import argparse
import random
from pathlib import Path
import re
from collections import defaultdict

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    print("需要安装 pyarrow: pip install pyarrow")
    sys.exit(1)

# 添加src路径到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes



# ============================================================================
# 统计三个数据集中的任务类型
# ============================================================================

# 三个数据集路径
DATASET_PATHS = [
    Path("/home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1215_5w_groot_4311_4322_4611_4633_downsample"),
    Path("/home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1221_5w_random_height_4322_4611_downsample"),
    Path("/home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1223_5w_dense_stacking_downsample"),
]

def find_pickup_episodes(dataset_paths):
    """
    查找所有包含"Pick up"任务的episode
    
    Returns:
        dict: {dataset_path: [episode_indices]}
    """
    pattern_pickup = re.compile(r'Pick up', re.IGNORECASE)
    pickup_episodes = {}
    
    for dataset_path in dataset_paths:
        dataset_name = dataset_path.name
        print(f"\n查找数据集 {dataset_name} 中的'Pick up'任务...")
        
        pickup_indices = []
        episodes_dir = dataset_path / "meta" / "episodes"
        if not episodes_dir.exists():
            print(f"  警告: {episodes_dir} 不存在，跳过")
            pickup_episodes[dataset_path] = []
            continue
        
        episodes_files = sorted(episodes_dir.glob("**/*.parquet"))
        print(f"  找到 {len(episodes_files)} 个episodes文件")
        
        for episodes_file in episodes_files:
            try:
                table = pq.read_table(episodes_file)
                df = table.to_pandas()
                
                for _, row in df.iterrows():
                    episode_index = row.get('episode_index', None)
                    if episode_index is None:
                        continue
                    
                    tasks = row.get('tasks', [])
                    
                    # 处理不同的存储格式
                    task_list = []
                    if isinstance(tasks, (list, tuple)):
                        task_list = list(tasks)
                    elif hasattr(tasks, '__iter__') and not isinstance(tasks, str):
                        try:
                            task_list = list(tasks)
                        except:
                            task_list = [str(tasks)]
                    elif isinstance(tasks, str):
                        try:
                            import ast
                            parsed = ast.literal_eval(tasks)
                            if isinstance(parsed, (list, tuple)):
                                task_list = list(parsed)
                            else:
                                task_list = [parsed]
                        except:
                            task_list = [tasks]
                    else:
                        task_list = [str(tasks)]
                    
                    # 检查是否包含"Pick up"任务
                    has_pickup = False
                    for task in task_list:
                        if isinstance(task, str) and pattern_pickup.search(task):
                            has_pickup = True
                            break
                    
                    if has_pickup:
                        pickup_indices.append(int(episode_index))
            
            except Exception as e:
                print(f"  读取文件 {episodes_file} 时出错: {e}")
                continue
        
        pickup_episodes[dataset_path] = sorted(set(pickup_indices))
        print(f"  找到 {len(pickup_episodes[dataset_path])} 个包含'Pick up'任务的episode")
    
    return pickup_episodes

def downsample_pickup_episodes(dataset_paths, target_count=850, random_seed=42):
    """
    对"Pick up"任务进行降采样
    
    Args:
        dataset_paths: 数据集路径列表
        target_count: 目标保留的episode数量
        random_seed: 随机种子
    
    Returns:
        dict: {dataset_path: [episodes_to_delete]}
    """
    print("\n" + "=" * 80)
    print("开始降采样'Pick up'任务")
    print("=" * 80)
    
    # 查找所有包含"Pick up"的episode
    pickup_episodes = find_pickup_episodes(dataset_paths)
    
    # 收集所有episode（带数据集信息）
    all_pickup_episodes = []
    for dataset_path, episode_indices in pickup_episodes.items():
        for ep_idx in episode_indices:
            all_pickup_episodes.append((dataset_path, ep_idx))
    
    total_pickup = len(all_pickup_episodes)
    print(f"\n总共找到 {total_pickup} 个包含'Pick up'任务的episode")
    
    if total_pickup <= target_count:
        print(f"⚠️  警告: 总数量 ({total_pickup}) 小于等于目标数量 ({target_count})，无需降采样")
        return {path: [] for path in dataset_paths}
    
    # 随机选择要保留的episode
    random.seed(random_seed)
    episodes_to_keep = random.sample(all_pickup_episodes, target_count)
    episodes_to_keep_set = set(episodes_to_keep)
    
    # 计算每个数据集需要删除的episode
    episodes_to_delete = {}
    for dataset_path in dataset_paths:
        episodes_to_delete[dataset_path] = []
        for ep_idx in pickup_episodes[dataset_path]:
            if (dataset_path, ep_idx) not in episodes_to_keep_set:
                episodes_to_delete[dataset_path].append(ep_idx)
    
    # 打印统计信息
    print(f"\n降采样计划:")
    print(f"  总episode数: {total_pickup}")
    print(f"  保留episode数: {target_count}")
    print(f"  删除episode数: {total_pickup - target_count}")
    print(f"\n各数据集删除计划:")
    for dataset_path in dataset_paths:
        delete_count = len(episodes_to_delete[dataset_path])
        keep_count = len(pickup_episodes[dataset_path]) - delete_count
        print(f"  {dataset_path.name}: 保留 {keep_count}, 删除 {delete_count}")
    
    return episodes_to_delete

def main():
    parser = argparse.ArgumentParser(description='统计数据集任务类型并支持降采样')
    parser.add_argument('--downsample-pickup', type=int, default=None,
                        help='对"Pick up"任务进行降采样，保留指定数量的episode（例如：850）')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='随机种子（默认：42）')
    parser.add_argument('--dry-run', action='store_true',
                        help='只显示计划，不实际执行删除操作')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（如果为None，则在原数据集目录旁边创建_downsamp版本，保留原数据集不变）')
    
    args = parser.parse_args()
    
    # ============================================================================
    # 统计任务类型
    # ============================================================================
    
    print("\n\n" + "=" * 80)
    print("任务类型统计（1215_four_box, 1221_random, 1223_dense）")
    print("=" * 80)
    
    # 任务类型统计
    task_type_counts = {
        "Depalletize on the left": 0,
        "Depalletize on the right": 0,
        "Pick up": 0,
    }
    
    # 正则表达式模式
    pattern_left = re.compile(r'Depalletize.*on the left', re.IGNORECASE)
    pattern_right = re.compile(r'Depalletize.*on the right', re.IGNORECASE)
    pattern_pickup = re.compile(r'Pick up', re.IGNORECASE)
    
    total_episodes = 0
    dataset_stats = {}  # 每个数据集的统计信息
    
    for dataset_path in DATASET_PATHS:
        dataset_name = dataset_path.name
        print(f"\n处理数据集: {dataset_name}")
        
        # 初始化该数据集的统计
        dataset_task_counts = {
            "Depalletize on the left": 0,
            "Depalletize on the right": 0,
            "Pick up": 0,
        }
        dataset_episodes = 0
        
        # 查找所有episodes文件
        episodes_dir = dataset_path / "meta" / "episodes"
        if not episodes_dir.exists():
            print(f"  警告: {episodes_dir} 不存在，跳过")
            continue
        
        episodes_files = sorted(episodes_dir.glob("**/file-000.parquet"))
        print(f"  找到 {len(episodes_files)} 个episodes文件")
        
        for episodes_file in episodes_files:
            try:
                # 读取episodes文件
                table = pq.read_table(episodes_file)
                df = table.to_pandas()
                
                # 遍历每个episode
                for _, row in df.iterrows():
                    total_episodes += 1
                    dataset_episodes += 1
                    tasks = row.get('tasks', [])
                    
                    # 处理不同的存储格式
                    task_list = []
                    if isinstance(tasks, (list, tuple)):
                        task_list = list(tasks)
                    elif hasattr(tasks, '__iter__') and not isinstance(tasks, str):
                        try:
                            task_list = list(tasks)
                        except:
                            task_list = [str(tasks)]
                    elif isinstance(tasks, str):
                        # 可能是字符串形式的列表，尝试解析
                        try:
                            import ast
                            parsed = ast.literal_eval(tasks)
                            if isinstance(parsed, (list, tuple)):
                                task_list = list(parsed)
                            else:
                                task_list = [parsed]
                        except:
                            task_list = [tasks]
                    else:
                        task_list = [str(tasks)]
                    
                    # 统计任务类型
                    for task in task_list:
                        if isinstance(task, str):
                            if pattern_left.search(task):
                                task_type_counts["Depalletize on the left"] += 1
                                dataset_task_counts["Depalletize on the left"] += 1
                            elif pattern_right.search(task):
                                task_type_counts["Depalletize on the right"] += 1
                                dataset_task_counts["Depalletize on the right"] += 1
                            elif pattern_pickup.search(task):
                                task_type_counts["Pick up"] += 1
                                dataset_task_counts["Pick up"] += 1
            
            except Exception as e:
                print(f"  读取文件 {episodes_file} 时出错: {e}")
                continue
    
        # 保存该数据集的统计信息
        dataset_stats[dataset_name] = {
            "episodes": dataset_episodes,
            "tasks": dataset_task_counts.copy()
        }
        print(f"  完成: {dataset_episodes} 个episodes")
    
    # 输出统计结果
    print(f"\n{'=' * 80}")
    print("任务类型统计结果:")
    print(f"{'=' * 80}")
    
    # 按数据集输出
    print(f"\n各数据集的统计:")
    for dataset_name, stats in dataset_stats.items():
        print(f"\n  {dataset_name}:")
        print(f"    Episodes数: {stats['episodes']:,}")
        dataset_total = sum(stats['tasks'].values())
        for task_type, count in stats['tasks'].items():
            percentage = (count / dataset_total * 100) if dataset_total > 0 else 0
            print(f"      {task_type}: {count:,} 条 ({percentage:.2f}%)")
    
    # 总体统计
    print(f"\n{'=' * 80}")
    print("总体统计:")
    print(f"{'=' * 80}")
    print(f"\n总episodes数: {total_episodes:,}")
    print(f"\n各任务类型的总数量:")
    total_tasks = sum(task_type_counts.values())
    for task_type, count in task_type_counts.items():
        percentage = (count / total_tasks * 100) if total_tasks > 0 else 0
        print(f"  {task_type}: {count:,} 条 ({percentage:.2f}%)")
    
    # ============================================================================
    # 执行降采样（如果指定）
    # ============================================================================
    
    if args.downsample_pickup is not None:
        print(f"\n{'=' * 80}")
        print("开始执行降采样")
        print(f"{'=' * 80}")
        
        # 计算需要删除的episode
        episodes_to_delete = downsample_pickup_episodes(
            DATASET_PATHS,
            target_count=args.downsample_pickup,
            random_seed=args.random_seed
        )
        
        if args.dry_run:
            print("\n⚠️  DRY RUN模式：只显示计划，不实际执行删除")
            print("要实际执行删除，请移除 --dry-run 参数")
        else:
            # 确认操作
            total_to_delete = sum(len(episodes) for episodes in episodes_to_delete.values())
            if total_to_delete > 0:
                print(f"\n⚠️  警告: 即将删除 {total_to_delete} 个episode")
                response = input("确认继续？(yes/no): ").strip().lower()
                if response != 'yes':
                    print("操作已取消")
                    return
            
            # 对每个数据集执行删除
            for dataset_path, delete_indices in episodes_to_delete.items():
                if not delete_indices:
                    print(f"\n跳过数据集 {dataset_path.name}（无需删除）")
                    continue
                
                print(f"\n处理数据集: {dataset_path.name}")
                print(f"  准备删除 {len(delete_indices)} 个episode...")
                
                try:
                    # 加载数据集
                    # root应该指向数据集根目录（包含meta/和data/的目录）
                    # 设置force_cache_sync=False以避免从Hub下载
                    dataset = LeRobotDataset(
                        repo_id=dataset_path.name,
                        root=dataset_path,  # 直接使用数据集路径作为root
                        force_cache_sync=False
                    )
                    
                    # 验证要删除的episode索引是否有效
                    total_episodes = dataset.meta.total_episodes
                    episodes_to_keep = total_episodes - len(delete_indices)
                    print(f"  📊 数据集统计:")
                    print(f"     总episode数: {total_episodes}")
                    print(f"     删除episode数: {len(delete_indices)}")
                    print(f"     保留episode数: {episodes_to_keep}")
                    print(f"  💡 注意: 'Processing data files' 显示的是数据文件数量，不是episode数量")
                    print(f"     一个数据文件可能包含多个episode的数据")
                    invalid_indices = [idx for idx in delete_indices if idx < 0 or idx >= total_episodes]
                    if invalid_indices:
                        print(f"  ⚠️  警告: 发现无效的episode索引: {invalid_indices[:10]}{'...' if len(invalid_indices) > 10 else ''}")
                        print(f"  ⚠️  数据集总episode数: {total_episodes}")
                        # 过滤掉无效的索引
                        delete_indices = [idx for idx in delete_indices if idx not in invalid_indices]
                        if not delete_indices:
                            print(f"  ⚠️  所有要删除的episode索引都无效，跳过此数据集")
                            continue
                        print(f"  ⚠️  过滤后，实际删除 {len(delete_indices)} 个episode")
                    
                    # 确定输出目录和repo_id
                    # 默认在原数据集目录旁边创建新数据集（保留原数据集不变）
                    if args.output_dir:
                        output_dir = Path(args.output_dir) / f"{dataset_path.name}_downsample"
                    else:
                        # 在原数据集目录的父目录下创建新数据集
                        output_dir = dataset_path.parent / f"{dataset_path.name}_downsample"
                    
                    # 如果输出目录已存在，删除它
                    if output_dir.exists():
                        print(f"  ⚠️  输出目录已存在: {output_dir}")
                        print(f"  ⚠️  正在删除旧目录...")
                        import shutil
                        shutil.rmtree(output_dir)
                        print(f"  ✅ 旧目录已删除")
                    
                    # 新数据集的repo_id
                    new_repo_id = f"{dataset_path.name}_downsample"
                    
                    # 执行删除
                    new_dataset = delete_episodes(
                        dataset=dataset,
                        episode_indices=delete_indices,
                        output_dir=output_dir,
                        repo_id=new_repo_id
                    )
                    
                    print(f"  ✅ 完成！新数据集保存在: {new_dataset.root}")
                    print(f"  ✅ 新数据集包含 {new_dataset.meta.total_episodes} 个episode")
                
                except Exception as e:
                    print(f"  ❌ 处理数据集 {dataset_path.name} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"\n{'=' * 80}")
            print("降采样完成！")
            print(f"{'=' * 80}")

if __name__ == '__main__':
    main()

