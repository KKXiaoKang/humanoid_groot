#!/usr/bin/env python3
"""
Evaluate Merged GROOT Model on Dataset

评估融合后的 GROOT 模型在训练数据集上的表现，支持 Rerun 可视化

使用方式：
    # 评估单个数据集
    python scripts/eval_merged_groot_on_dataset.py \
        --model-path /home/lab/humanoid_groot/outputs/merged_groot/pretrained_model \
        --dataset-root /path/to/dataset \
        --episode 0
    
    # 评估多个数据集（使用默认的训练数据集）
    python scripts/eval_merged_groot_on_dataset.py \
        --model-path /home/lab/humanoid_groot/outputs/merged_groot/pretrained_model \
        --use-default-datasets
    
    # 启用 Rerun 可视化
    python scripts/eval_merged_groot_on_dataset.py \
        --model-path /home/lab/humanoid_groot/outputs/merged_groot/pretrained_model \
        --dataset-root /path/to/dataset \
        --episode 0 \
        --visualize
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from pathlib import Path
import argparse
import json
import time
from collections import OrderedDict
from tqdm import tqdm

# 使用GrootPolicy模型
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 可选的可视化工具（如果不存在则禁用）
try:
    from visualization_tools.visualizers import RerunVisualizer, KeyboardManager
    RERUN_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: RerunVisualizer not available. Visualization will be disabled.")
    RERUN_AVAILABLE = False
    RerunVisualizer = None
    KeyboardManager = None

# 默认训练数据集路径
DEFAULT_NARROWER_DATASETS = [
    "/home/kangkk/humanoid_groot/lerobot_data/split_dataset/narrower/four",
    "/home/kangkk/humanoid_groot/lerobot_data/split_dataset/narrower/random",
    "/home/kangkk/humanoid_groot/lerobot_data/split_dataset/narrower/dense",
    "/home/kangkk/humanoid_groot/lerobot_data/split_dataset/narrower/mix",
]

DEFAULT_WIDER_DATASETS = [
    "/home/kangkk/humanoid_groot/lerobot_data/split_dataset/wider/four",
    "/home/kangkk/humanoid_groot/lerobot_data/split_dataset/wider/random",
    "/home/kangkk/humanoid_groot/lerobot_data/split_dataset/wider/dense",
    "/home/kangkk/humanoid_groot/lerobot_data/split_dataset/wider/mix",
]


def load_merge_config(model_path: str) -> dict | None:
    """加载融合配置"""
    merge_config_path = Path(model_path) / "merge_config.json"
    if merge_config_path.exists():
        with open(merge_config_path, 'r') as f:
            return json.load(f)
    return None


def eval_on_dataset(
    model_path: str,
    lerobot_dataset_path: str | None = None,
    episode: int = 0,
    n_actions: int = 16,
    show_progress: bool = True,
    use_default_datasets: bool = False,
    visualize: bool = False,
):
    """
    在数据集上评估融合模型
    
    Args:
        model_path: 融合模型路径
        lerobot_dataset_path: 数据集根目录（如果 use_default_datasets=False）
        episode: episode编号
        n_actions: action chunk大小
        show_progress: 是否显示进度条
        use_default_datasets: 是否使用默认的训练数据集（评估所有数据集）
        visualize: 是否启用 Rerun 可视化
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # ------------- 初始化 visualizer (可选) -------------
    vizer = None
    kb = None
    if visualize:
        if RERUN_AVAILABLE:
            vizer = RerunVisualizer()
            kb = KeyboardManager()
            print("✅ RerunVisualizer initialized")
        else:
            print("⚠️  Visualization requested but RerunVisualizer not available")
            visualize = False
    
    print(f"\n{'='*80}")
    print(f"🚀 评估融合 GROOT 模型")
    print(f"{'='*80}")
    print(f"📂 模型路径: {model_path}")
    print(f"🔧 设备: {device}")
    print(f"📊 Action chunk size: {n_actions}")
    
    # 加载融合配置
    merge_config = load_merge_config(model_path)
    if merge_config:
        print(f"\n📋 融合配置:")
        print(f"   融合模式: {merge_config.get('merge_mode', 'N/A')}")
        print(f"   专家模型: {merge_config.get('expert_names', [])}")
        if 'coefficient_stats' in merge_config:
            stats = merge_config['coefficient_stats']
            per_expert_mean = stats.get('per_expert_mean', [])
            if len(per_expert_mean) >= 2:
                print(f"   融合系数统计:")
                print(f"     - 平均系数: {stats.get('mean', 'N/A'):.4f}")
                print(f"     - 标准差: {stats.get('std', 'N/A'):.4f}")
                print(f"     - Per-expert 平均: {per_expert_mean}")
        
        # 读取融合范围配置
        # 如果 merge_config.json 中没有这些字段，使用默认值（基于 merge_groot_models.sh 的默认配置）
        merge_backbone_only = merge_config.get('merge_backbone_only')
        action_head_source = merge_config.get('action_head_source')
        
        # 如果配置中没有这些字段，使用默认值（基于 merge_groot_models.sh）
        if merge_backbone_only is None:
            merge_backbone_only = True  # 默认值：只融合 backbone
            print(f"\n   ⚠️  注意: merge_config.json 中没有 merge_backbone_only 字段")
            print(f"     使用默认值（基于 merge_groot_models.sh）: merge_backbone_only=True")
        if action_head_source is None:
            action_head_source = "interpolate"  # 默认值：插值
            print(f"   ⚠️  注意: merge_config.json 中没有 action_head_source 字段")
            print(f"     使用默认值（基于 merge_groot_models.sh）: action_head_source=interpolate")
        
        print(f"\n   🔧 融合范围配置:")
        if merge_backbone_only:
            print(f"     - ✅ 只融合 backbone 权重")
            if action_head_source == "first_expert":
                print(f"     - Action Head: 使用第一个专家模型（narrower）的权重")
            elif action_head_source == "second_expert":
                print(f"     - Action Head: 使用第二个专家模型（wider）的权重")
            elif action_head_source == "interpolate":
                print(f"     - Action Head: 使用插值（interpolate）: 0.5 * narrower + 0.5 * wider")
        else:
            print(f"     - ✅ 融合所有权重（包括 backbone 和 action_head）")
    else:
        print(f"\n⚠️  警告: 未找到 merge_config.json，无法显示融合配置")
    
    # 加载模型
    print(f"\n📦 加载模型...")
    policy = GrootPolicy.from_pretrained(Path(model_path), strict=False)
    policy.config.device = device
    policy.config.n_action_steps = n_actions
    policy.eval().to(device)
    
    # 加载 preprocessor 和 postprocessor
    print(f"🔧 加载 preprocessor 和 postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
    )
    print("✅ 模型加载完成")
    
    policy.reset()
    
    # 确定要评估的数据集
    if use_default_datasets:
        # 使用默认的训练数据集
        datasets_to_eval = []
        
        # 检查并添加 narrower 数据集
        for dataset_path in DEFAULT_NARROWER_DATASETS:
            if Path(dataset_path).exists():
                datasets_to_eval.append(("narrower", dataset_path))
        
        # 检查并添加 wider 数据集
        for dataset_path in DEFAULT_WIDER_DATASETS:
            if Path(dataset_path).exists():
                datasets_to_eval.append(("wider", dataset_path))
        
        if not datasets_to_eval:
            print(f"\n❌ 错误: 未找到默认数据集！")
            print(f"   请检查数据集路径是否存在，或使用 --dataset-root 指定数据集")
            return
        
        print(f"\n📚 将评估 {len(datasets_to_eval)} 个数据集:")
        for task_type, dataset_path in datasets_to_eval:
            print(f"   - {task_type}: {dataset_path}")
        
        # 评估每个数据集
        all_results = {}
        for task_type, dataset_path in datasets_to_eval:
            print(f"\n{'='*80}")
            print(f"📊 评估数据集: {task_type} - {dataset_path}")
            print(f"{'='*80}")
            
            # 加载数据集
            dataset_name = Path(dataset_path).name
            dataset = LeRobotDataset(repo_id=dataset_name, root=dataset_path)
            
            # 评估所有 episodes
            num_episodes = len(dataset.meta.episodes) if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'episodes') else 1
            print(f"📹 数据集包含 {num_episodes} 个 episodes")
            
            dataset_results = []
            for ep in range(num_episodes):
                try:
                    result = eval_single_episode(
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        dataset_path=dataset_path,
                        episode=ep,
                        device=device,
                        n_actions=n_actions,
                        show_progress=show_progress,
                        vizer=vizer,
                        kb=kb,
                    )
                    dataset_results.append(result)
                except Exception as e:
                    print(f"⚠️  Episode {ep} 评估失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 汇总数据集结果
            if dataset_results:
                all_results[task_type] = aggregate_results(dataset_results)
            
            print(f"\n✅ {task_type} 数据集评估完成")
        
        # 打印总体结果
        print_results_summary(all_results)
        
    else:
        # 评估单个数据集
        if not lerobot_dataset_path:
            print(f"\n❌ 错误: 必须指定 --dataset-root 或使用 --use-default-datasets")
            return
        
        print(f"\n📊 评估数据集: {lerobot_dataset_path}")
        print(f"📹 Episode: {episode}")
        
        result = eval_single_episode(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_path=lerobot_dataset_path,
            episode=episode,
            device=device,
            n_actions=n_actions,
            show_progress=show_progress,
            vizer=vizer,
            kb=kb,
        )
        
        print_single_result(result)
    
    # 如果启用了可视化，等待用户退出
    if vizer is not None:
        print("\n[Offline Eval] Visualization active. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n✅ Exiting...")


def eval_single_episode(
    policy: GrootPolicy,
    preprocessor,
    postprocessor,
    dataset_path: str,
    episode: int,
    device: str,
    n_actions: int,
    show_progress: bool = True,
    vizer = None,
    kb = None,
) -> dict:
    """评估单个 episode"""
    # 加载数据集
    dataset_name = Path(dataset_path).name
    dataset = LeRobotDataset(repo_id=dataset_name, root=dataset_path, episodes=[episode])
    
    # 过滤到指定 episode
    if episode >= len(dataset.meta.episodes):
        raise ValueError(f"Episode {episode} out of range. Available episodes: 0-{len(dataset.meta.episodes)-1}")
    
    ep_meta = dataset.meta.episodes[episode]
    ep_start = ep_meta["dataset_from_index"]
    ep_end = ep_meta["dataset_to_index"]
    dataset.hf_dataset = dataset.hf_dataset.select(range(ep_start, ep_end))
    
    # 创建 dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        pin_memory=(device.startswith('cuda')),
        drop_last=False,
    )
    
    # 获取 action 维度和 observation 维度
    first_batch = next(iter(dataloader))
    action_dim = first_batch['action'].shape[1]
    obs_dim = first_batch['observation.state'].shape[1]
    
    # 重新创建 dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        pin_memory=(device.startswith('cuda')),
        drop_last=False,
    )
    
    # ========= 可视化 ground truth (如果启用 RerunVisualizer) =========
    if vizer is not None:
        print(f"\n📊 Visualizing ground truth data for episode {episode}...")
        # 加载所有 ground truth actions 用于可视化
        all_gt_actions = []
        all_gt_states = []
        
        temp_dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=0, batch_size=1, shuffle=False, drop_last=False
        )
        
        for batch in temp_dataloader:
            all_gt_actions.append(batch['action'][0].cpu().numpy())
            all_gt_states.append(batch['observation.state'][0].cpu().numpy())
        
        all_gt_actions = np.array(all_gt_actions)
        all_gt_states = np.array(all_gt_states)
        
        # 可视化 ground truth actions
        for dim in range(action_dim):
            vizer.visualize_chunk(
                name=f"chunk/action_dim_{dim}/gt",
                chunk_data=all_gt_actions[:, dim],
                step_id=0,
                width=3.0
            )
        
        # 可视化 observations
        for dim in range(obs_dim):
            vizer.visualize_chunk(
                name=f"obs/obs_{dim}",
                chunk_data=all_gt_states[:, dim],
                step_id=0,
                width=3.0
            )
        
        print(f"✅ Ground truth visualization ready")
        
        # 重新创建 dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=0,
            batch_size=1,
            shuffle=False,
            pin_memory=(device.startswith('cuda')),
            drop_last=False,
        )
    
    # 评估
    mse_per_action_dim = OrderedDict()
    mae_per_action_dim = OrderedDict()
    inference_times = []
    last_data_step = 0
    
    iterator = tqdm(enumerate(dataloader), total=len(dataset.hf_dataset), desc=f"Episode {episode}") if show_progress else enumerate(dataloader)
    
    for data_step, batch in iterator:
        # 暂停控制（如果启用）
        if vizer is not None and kb is not None:
            time.sleep(0.05)
            if kb.paused:
                print(f'===== 暂停中，按下空格开始 =====')
            while kb.paused:
                time.sleep(0.1)
        
        # 准备 observation
        observation = {
            'observation.state': batch['observation.state'],
        }
        
        # 添加图像观测
        for key in batch.keys():
            if 'image' in key.lower() and key.startswith('observation'):
                observation[key] = batch[key]
        
        # 添加 task 字段
        if 'task' in batch:
            batch_task = batch['task']
            if isinstance(batch_task, (list, tuple)) and len(batch_task) > 0:
                observation['task'] = batch_task[0]
            elif isinstance(batch_task, str):
                observation['task'] = batch_task
            else:
                observation['task'] = str(batch_task) if batch_task is not None else ""
        else:
            observation['task'] = ""
        
        # 获取 ground truth
        gt_action = batch['action'][0].cpu().numpy()
        
        # 模型推理（精确测量推理时间）
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        inference_start = time.perf_counter()
        
        processed_observation = preprocessor(observation)
        with torch.inference_mode():
            pred_actions = policy.predict_action_chunk(processed_observation)
        
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start
        inference_times.append(inference_time)
        
        # 使用 postprocessor 进行反归一化
        _, chunk_size, _ = pred_actions.shape
        processed_actions = []
        for i in range(chunk_size):
            single_action = pred_actions[:, i, :]
            processed_action = postprocessor(single_action)
            processed_actions.append(processed_action)
        
        pred_actions_unnorm = torch.stack(processed_actions, dim=1)
        pred_chunk = pred_actions_unnorm[0].cpu().numpy()  # (chunk_size, action_dim)
        pred_action_single = pred_chunk[0]  # 取第一个 action
        
        # 计算误差
        for dim in range(action_dim):
            error = pred_action_single[dim] - gt_action[dim]
            mse = error ** 2
            mae = abs(error)
            
            if dim not in mse_per_action_dim:
                mse_per_action_dim[dim] = []
                mae_per_action_dim[dim] = []
            
            mse_per_action_dim[dim].append(mse)
            mae_per_action_dim[dim].append(mae)
        
        # ========= 可视化（如果启用）=========
        if vizer is not None:
            # 显示图像
            for key in batch.keys():
                if 'image' in key.lower() and key.startswith('observation'):
                    img = batch[key][0]  # (C, H, W)
                    camera_name = key.replace('observation.images.', '').replace('observation.', '')
                    vizer.show_img(
                        name=f"images.{camera_name}",
                        image_data=img.to("cpu"),
                        step_id=data_step
                    )
            
            # 可视化预测的 chunk 和 MSE
            for dim in range(action_dim):
                # 可视化 MSE
                vizer.visualize_chunk(
                    name=f"mse/action_dim_{dim}",
                    chunk_data=mse_per_action_dim[dim][-1],
                    step_id=data_step,
                    width=3.0,
                )
                
                # 可视化预测的 action chunk
                vizer.visualize_chunk(
                    name=f"chunk/action_dim_{dim}/pred_seg_{data_step}",
                    chunk_data=pred_chunk[:, dim],
                    step_id=data_step,
                    width=2
                )
                
                # 删除旧的可视化
                if last_data_step != data_step and last_data_step > 0:
                    vizer.del_chunk(
                        name=f"chunk/action_dim_{dim}/pred_seg_{last_data_step}",
                        chunk_data=pred_chunk[:, dim],
                        step_id=last_data_step,
                        width=0.5
                    )
        
        last_data_step = data_step
    
    # 计算平均误差
    result = {
        'episode': episode,
        'dataset_path': dataset_path,
        'action_dim': action_dim,
        'num_frames': len(dataset.hf_dataset),
        'mse_per_dim': {dim: np.mean(mse_per_action_dim[dim]) for dim in range(action_dim)},
        'mae_per_dim': {dim: np.mean(mae_per_action_dim[dim]) for dim in range(action_dim)},
        'overall_mse': np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(action_dim)]),
        'overall_mae': np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(action_dim)]),
    }
    
    # 打印推理时间统计
    if len(inference_times) > 0:
        avg_time = np.mean(inference_times)
        print(f"   ⏱️  Average inference time: {avg_time*1000:.2f} ms")
        if avg_time > 0:
            print(f"   📊 Max theoretical frequency: {1.0/avg_time:.2f} Hz")
    
    return result


def aggregate_results(results: list[dict]) -> dict:
    """汇总多个 episode 的结果"""
    if not results:
        return {}
    
    action_dim = results[0]['action_dim']
    total_frames = sum(r['num_frames'] for r in results)
    
    # 加权平均（按帧数加权）
    aggregated = {
        'num_episodes': len(results),
        'total_frames': total_frames,
        'action_dim': action_dim,
        'mse_per_dim': {},
        'mae_per_dim': {},
        'overall_mse': 0.0,
        'overall_mae': 0.0,
    }
    
    for dim in range(action_dim):
        weighted_mse = sum(r['mse_per_dim'][dim] * r['num_frames'] for r in results) / total_frames
        weighted_mae = sum(r['mae_per_dim'][dim] * r['num_frames'] for r in results) / total_frames
        aggregated['mse_per_dim'][dim] = weighted_mse
        aggregated['mae_per_dim'][dim] = weighted_mae
    
    aggregated['overall_mse'] = np.mean([aggregated['mse_per_dim'][dim] for dim in range(action_dim)])
    aggregated['overall_mae'] = np.mean([aggregated['mae_per_dim'][dim] for dim in range(action_dim)])
    
    return aggregated


def print_single_result(result: dict):
    """打印单个结果"""
    print(f"\n{'='*80}")
    print(f"📊 评估结果")
    print(f"{'='*80}")
    print(f"Episode: {result['episode']}")
    print(f"数据集: {result['dataset_path']}")
    print(f"总帧数: {result['num_frames']}")
    print(f"Action 维度: {result['action_dim']}")
    print(f"\n{'维度':<20} {'MSE':<15} {'MAE':<15}")
    print("-" * 80)
    
    for dim in range(result['action_dim']):
        mse = result['mse_per_dim'][dim]
        mae = result['mae_per_dim'][dim]
        print(f'{f"Dim_{dim}":<20} {mse:<15.8f} {mae:<15.8f}')
    
    print("-" * 80)
    print(f'{"Overall":<20} {result["overall_mse"]:<15.8f} {result["overall_mae"]:<15.8f}')
    print("=" * 80)


def print_results_summary(all_results: dict):
    """打印汇总结果"""
    print(f"\n{'='*80}")
    print(f"📊 总体评估结果汇总")
    print(f"{'='*80}")
    
    for task_type, result in all_results.items():
        print(f"\n📋 {task_type.upper()} 任务:")
        print(f"   Episodes: {result['num_episodes']}")
        print(f"   总帧数: {result['total_frames']}")
        print(f"   Overall MSE: {result['overall_mse']:.8f}")
        print(f"   Overall MAE: {result['overall_mae']:.8f}")
    
    # 计算总体平均
    if len(all_results) > 1:
        overall_mse = np.mean([r['overall_mse'] for r in all_results.values()])
        overall_mae = np.mean([r['overall_mae'] for r in all_results.values()])
        print(f"\n📊 所有任务平均:")
        print(f"   Overall MSE: {overall_mse:.8f}")
        print(f"   Overall MAE: {overall_mae:.8f}")
    
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate Merged GROOT Model on Dataset',
        epilog='评估融合后的 GROOT 模型在训练数据集上的表现'
    )
    parser.add_argument('--model-path', '--model_path', type=str, 
                       default="/home/lab/humanoid_groot/outputs/merged_groot/pretrained_model",
                       dest='model_path',
                       help='Path to the merged model checkpoint directory')
    parser.add_argument('--dataset-root', '--dataset_root', type=str, default=None,
                       dest='dataset_root',
                       help='Path to the LeRobot dataset root directory (if not using --use-default-datasets)')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode number to evaluate (default: 0, only used if --dataset-root is specified)')
    parser.add_argument('--action-chunk-size', type=int, default=50,
                       dest='action_chunk_size',
                       help='Action chunk size (default: 50, should match training config)')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    parser.add_argument('--use-default-datasets', action='store_true',
                       help='Evaluate on default training datasets (narrower and wider)')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable Rerun visualization')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎯 Merged GROOT Model Dataset Evaluation")
    print("="*80)
    print(f"Model: {args.model_path}")
    if args.dataset_root:
        print(f"Dataset: {args.dataset_root}")
        print(f"Episode: {args.episode}")
    if args.use_default_datasets:
        print(f"Using default training datasets")
    print(f"Action Chunk Size: {args.action_chunk_size}")
    print(f"Visualization: {args.visualize}")
    print("="*80)
    
    eval_on_dataset(
        model_path=args.model_path,
        lerobot_dataset_path=args.dataset_root,
        episode=args.episode,
        n_actions=args.action_chunk_size,
        show_progress=not args.no_progress,
        use_default_datasets=args.use_default_datasets,
        visualize=args.visualize,
    )
