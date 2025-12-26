#!/usr/bin/env python3
"""
可视化三个动作类别的 3D t-SNE 分布

用于检查 "Depalletize on the left"、"Depalletize on the right" 和 "Pick up" 
这三个动作类别是否形成流形以及是否已经分开。
"""

import sys
import os
from pathlib import Path
import argparse
import re
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

# 尝试导入 plotly 用于交互式可视化
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly 未安装，将使用 matplotlib 静态可视化。安装 plotly 以获得交互式可视化: pip install plotly")

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors


def extract_task_from_episode(dataset, episode_idx):
    """从 episode 中提取任务名称"""
    try:
        ep_meta = dataset.meta.episodes[episode_idx]
        
        # 方法1: 直接从episode的tasks字段获取（优先）
        if "tasks" in ep_meta:
            tasks = ep_meta["tasks"]
            if isinstance(tasks, (list, tuple)) and len(tasks) > 0:
                return tasks[0]  # 返回第一个任务
            elif isinstance(tasks, str):
                return tasks
        
        # 方法2: 通过task_index从tasks表中获取
        task_idx = ep_meta.get("task_index", None)
        if task_idx is not None and hasattr(dataset.meta, 'tasks') and dataset.meta.tasks is not None:
            if isinstance(task_idx, (list, tuple)) and len(task_idx) > 0:
                task_idx = task_idx[0]
            if isinstance(task_idx, (int, np.integer)) and task_idx < len(dataset.meta.tasks):
                task_name = dataset.meta.tasks.iloc[task_idx].name
                return task_name
        
        # 方法3: 从数据集中直接读取一帧来获取task
        try:
            ep_start = ep_meta.get("dataset_from_index", 0)
            if ep_start < len(dataset):
                sample = dataset[ep_start]
                if "task" in sample:
                    return sample["task"]
        except:
            pass
            
    except Exception as e:
        print(f"Warning: Failed to extract task from episode {episode_idx}: {e}")
    
    return None


def classify_task(task_name):
    """将任务名称分类到三个类别之一"""
    if task_name is None:
        return None
    
    task_lower = str(task_name).lower()
    
    # 更灵活的正则表达式匹配
    if re.search(r'depalletize.*left|left.*depalletize', task_lower):
        return "Depalletize on the left"
    elif re.search(r'depalletize.*right|right.*depalletize', task_lower):
        return "Depalletize on the right"
    elif re.search(r'pick\s*up|pickup', task_lower):
        return "Pick up"
    
    return None


def extract_backbone_features(policy, preprocessor, sample, device):
    """从模型中提取 backbone 特征"""
    policy.eval()
    
    with torch.no_grad():
        # 准备输入
        observation = {}
        
        # 处理图像
        for key in sample.keys():
            if key.startswith("observation.images."):
                img = sample[key]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                if img.ndim == 3:
                    img = img.unsqueeze(0)  # 添加 batch 维度
                observation[key] = img.to(device)
            elif key == "observation.state":
                state = sample[key]
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state).float()
                if state.ndim == 1:
                    state = state.unsqueeze(0)  # 添加 batch 维度
                observation[key] = state.to(device)
            elif key == "task":
                observation["task"] = sample[key]
        
        # 预处理
        processed_obs = preprocessor(observation)
        
        # 提取 backbone 特征
        # 通过模型的 backbone 获取特征
        groot_inputs = {
            k: v
            for k, v in processed_obs.items()
            if (k in {"state", "state_mask", "embodiment_id"} or k.startswith("eagle_"))
            and not (k.startswith("next.") or k == "info")
        }
        
        # 获取 backbone 输出
        backbone = policy._groot_model.backbone
        from transformers.feature_extraction_utils import BatchFeature
        
        vl_input = BatchFeature(data=groot_inputs)
        backbone_output = backbone.forward(vl_input)
        
        # 提取 backbone_features
        # backbone_output 是 BatchFeature，需要访问 data 属性
        if hasattr(backbone_output, 'data'):
            features = backbone_output.data.get("backbone_features")
        elif isinstance(backbone_output, dict):
            features = backbone_output.get("backbone_features")
        else:
            # 尝试直接访问
            features = getattr(backbone_output, "backbone_features", None)
        
        # 如果 features 是 (B, T, D) 形状，取平均池化或最后一个时间步
        if features is not None:
            if features.ndim == 3:
                # 对时间维度进行平均池化
                features = features.mean(dim=1)  # (B, D)
            elif features.ndim == 2:
                pass  # 已经是 (B, D)
            else:
                # 展平
                features = features.flatten(start_dim=1)
            
            # 转换为 float32 以避免 bfloat16 不支持的问题
            features = features.float()
            
            # 返回第一个样本的特征（去除 batch 维度）
            return features[0].cpu().numpy()
    
    return None


def collect_features(
    dataset_path,
    ckpt_path,
    num_samples_per_task=0,
    max_episodes=None,
    device="cuda:0"
):
    """收集所有三个任务类别的特征"""
    
    print("=" * 80)
    print("加载数据集和模型")
    print("=" * 80)
    
    # 加载数据集
    # 对于本地数据集，repo_id 应该是字符串（使用数据集目录名）
    dataset_name = Path(dataset_path).name
    dataset = LeRobotDataset(
        repo_id=dataset_name,
        root=dataset_path,
        force_cache_sync=False
    )
    print(f"✅ 数据集加载完成: {dataset.meta.total_episodes} 个 episodes")
    
    # 加载模型
    policy = GrootPolicy.from_pretrained(Path(ckpt_path), strict=False)
    policy.config.device = device
    policy.eval().to(device)
    print(f"✅ 模型加载完成")
    
    # 加载预处理器
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt_path,
    )
    print(f"✅ 预处理器加载完成")
    
    # 统计每个任务类别的 episode
    task_episodes = defaultdict(list)
    
    print("\n" + "=" * 80)
    print("统计任务类别")
    print("=" * 80)
    
    total_episodes = dataset.meta.total_episodes
    if max_episodes is not None:
        total_episodes = min(total_episodes, max_episodes)
    
    # 收集所有任务名称用于调试
    all_task_names = set()
    unclassified_tasks = {}
    
    for ep_idx in range(total_episodes):
        task_name = extract_task_from_episode(dataset, ep_idx)
        if task_name:
            all_task_names.add(task_name)
        
        task_category = classify_task(task_name)
        
        if task_category is not None:
            task_episodes[task_category].append(ep_idx)
        else:
            if task_name:
                unclassified_tasks[task_name] = unclassified_tasks.get(task_name, 0) + 1
    
    print(f"\n任务类别统计:")
    for task_cat, episodes in task_episodes.items():
        print(f"  {task_cat}: {len(episodes)} 个 episodes")
    
    # 显示所有找到的任务名称（用于调试）
    if all_task_names:
        print(f"\n所有找到的任务名称 ({len(all_task_names)} 种):")
        for task_name in sorted(all_task_names):
            count = sum(1 for ep_idx in range(total_episodes) 
                       if extract_task_from_episode(dataset, ep_idx) == task_name)
            print(f"  - '{task_name}': {count} 个 episodes")
    
    # 显示未分类的任务
    if unclassified_tasks:
        print(f"\n⚠️  未分类的任务 ({len(unclassified_tasks)} 种):")
        for task_name, count in sorted(unclassified_tasks.items(), key=lambda x: x[1], reverse=True):
            print(f"  - '{task_name}': {count} 个 episodes")
    
    # 收集特征
    print("\n" + "=" * 80)
    print("提取特征")
    print("=" * 80)
    
    all_features = []
    all_labels = []
    all_episode_indices = []
    
    for task_category, episode_indices in task_episodes.items():
        print(f"\n处理任务类别: {task_category}")
        print(f"  总共有 {len(episode_indices)} 个 episodes，将全部处理")
        
        # 如果指定了 num_samples_per_task，则限制采样数量；否则处理所有episodes
        if num_samples_per_task > 0:
            selected_episodes = episode_indices[:num_samples_per_task]
            print(f"  限制采样数量为: {len(selected_episodes)} 个 episodes")
        else:
            selected_episodes = episode_indices
            print(f"  处理所有 {len(selected_episodes)} 个 episodes")
        
        for ep_idx in tqdm(selected_episodes, desc=f"  {task_category}"):
            try:
                # 获取 episode 的帧范围
                ep_meta = dataset.meta.episodes[ep_idx]
                ep_start = ep_meta["dataset_from_index"]
                ep_end = ep_meta["dataset_to_index"]
                ep_length = ep_end - ep_start
                
                # 从 episode 开头开始的第3帧采样（索引从0开始，所以是+2）
                if ep_length < 3:
                    # 如果episode长度小于3帧，使用第一帧
                    frame_idx = ep_start
                else:
                    frame_idx = ep_start + 2  # 第3帧（索引0, 1, 2）
                
                # 获取样本
                sample = dataset[frame_idx]
                
                # 提取特征
                features = extract_backbone_features(policy, preprocessor, sample, device)
                
                if features is not None:
                    all_features.append(features)
                    all_labels.append(task_category)
                    all_episode_indices.append(ep_idx)
            
            except Exception as e:
                print(f"  ⚠️  处理 episode {ep_idx} 时出错: {e}")
                continue
    
    print(f"\n✅ 特征提取完成: 共 {len(all_features)} 个样本")
    
    return np.array(all_features), np.array(all_labels), np.array(all_episode_indices)


def visualize_tsne_3d(features, labels, episode_indices=None, output_path=None, perplexity=30, max_iter=1000, interactive=True):
    """使用 t-SNE 降维到 3D 并可视化
    
    Args:
        features: 特征数组
        labels: 标签数组
        episode_indices: episode索引数组，用于在hover信息中显示（可选）
        output_path: 输出路径
        perplexity: t-SNE perplexity参数
        max_iter: t-SNE最大迭代次数
        interactive: 是否使用交互式可视化
    """
    
    print("\n" + "=" * 80)
    print("t-SNE 降维到 3D")
    print("=" * 80)
    
    # 执行 t-SNE
    print(f"执行 t-SNE (perplexity={perplexity}, max_iter={max_iter})...")
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=42,
        verbose=1
    )
    
    features_3d = tsne.fit_transform(features)
    
    print(f"✅ t-SNE 完成: {features.shape} -> {features_3d.shape}")
    
    # 可视化
    print("\n" + "=" * 80)
    print("生成 3D 可视化")
    print("=" * 80)
    
    # 定义颜色映射
    color_map = {
        "Depalletize on the left": "#FF0000",  # 红色
        "Depalletize on the right": "#0000FF",  # 蓝色
        "Pick up": "#00FF00"  # 绿色
    }
    
    # 使用 plotly 进行交互式可视化（如果可用）
    if interactive and PLOTLY_AVAILABLE:
        print("使用 Plotly 生成交互式可视化...")
        
        # 创建 DataFrame 用于 plotly
        import pandas as pd
        df_data = {
            'x': features_3d[:, 0],
            'y': features_3d[:, 1],
            'z': features_3d[:, 2],
            'label': labels
        }
        
        # 如果提供了episode_indices，添加到DataFrame中
        if episode_indices is not None:
            df_data['episode_index'] = episode_indices
        
        df = pd.DataFrame(df_data)
        
        # 创建交互式 3D 散点图
        fig = go.Figure()
        
        for task_category in sorted(set(labels)):
            mask = df['label'] == task_category
            color = color_map.get(task_category, "#808080")
            
            # 构建hover信息
            if episode_indices is not None and 'episode_index' in df.columns:
                # 使用customdata传递episode索引，然后在hovertemplate中引用
                masked_df = df[mask]
                fig.add_trace(go.Scatter3d(
                    x=masked_df['x'],
                    y=masked_df['y'],
                    z=masked_df['z'],
                    mode='markers',
                    name=task_category,
                    customdata=masked_df['episode_index'].values,
                    hovertemplate=(
                        f'<b>{task_category}</b><br>' +
                        'Episode: %{customdata}<br>' +
                        'X: %{x:.2f}<br>' +
                        'Y: %{y:.2f}<br>' +
                        'Z: %{z:.2f}<extra></extra>'
                    ),
                    marker=dict(
                        size=5,
                        color=color,
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    )
                ))
            else:
                # 如果没有episode索引，使用简单的hover信息
                fig.add_trace(go.Scatter3d(
                    x=df[mask]['x'],
                    y=df[mask]['y'],
                    z=df[mask]['z'],
                    mode='markers',
                    name=task_category,
                    marker=dict(
                        size=5,
                        color=color,
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    hovertemplate=f'<b>{task_category}</b><br>' +
                                'X: %{x:.2f}<br>' +
                                'Y: %{y:.2f}<br>' +
                                'Z: %{z:.2f}<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(
                text='3D t-SNE 可视化：动作类别的特征分布',
                font=dict(size=16, color='black')
            ),
            scene=dict(
                xaxis_title='t-SNE 维度 1',
                yaxis_title='t-SNE 维度 2',
                zaxis_title='t-SNE 维度 3',
                bgcolor='white',
                xaxis=dict(backgroundcolor='white', gridcolor='lightgray'),
                yaxis=dict(backgroundcolor='white', gridcolor='lightgray'),
                zaxis=dict(backgroundcolor='white', gridcolor='lightgray')
            ),
            width=1000,
            height=800,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        # 显示交互式图表
        fig.show()
        
        # 保存为 HTML 文件（可交互）
        if output_path:
            html_path = output_path.replace('.png', '.html')
            fig.write_html(html_path)
            print(f"✅ 交互式 HTML 已保存到: {html_path}")
        
        # 同时保存静态图片（如果 kaleido 可用）
        if output_path:
            try:
                fig.write_image(output_path, width=1200, height=1000, scale=2)
                print(f"✅ 静态图像已保存到: {output_path}")
            except Exception as e:
                print(f"⚠️  无法使用 kaleido 保存静态图像: {e}")
                print(f"   尝试使用 matplotlib 保存静态图片...")
                # 使用 matplotlib 作为备选方案保存静态图片
                try:
                    fig_static = plt.figure(figsize=(12, 10))
                    ax_static = fig_static.add_subplot(111, projection='3d')
                    
                    for task_category in sorted(set(labels)):
                        mask = labels == task_category
                        if np.any(mask):
                            color = color_map.get(task_category, "gray")
                            ax_static.scatter(
                                features_3d[mask, 0],
                                features_3d[mask, 1],
                                features_3d[mask, 2],
                                c=color,
                                label=task_category,
                                alpha=0.6,
                                s=50
                            )
                    
                    ax_static.set_xlabel('t-SNE 维度 1', fontsize=12)
                    ax_static.set_ylabel('t-SNE 维度 2', fontsize=12)
                    ax_static.set_zlabel('t-SNE 维度 3', fontsize=12)
                    ax_static.set_title('3D t-SNE 可视化：动作类别的特征分布', fontsize=14, fontweight='bold')
                    ax_static.legend(fontsize=10)
                    ax_static.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close(fig_static)
                    print(f"✅ 使用 matplotlib 保存静态图像到: {output_path}")
                except Exception as e2:
                    print(f"⚠️  使用 matplotlib 保存也失败: {e2}")
                    print(f"   但交互式 HTML 文件已成功保存: {html_path}")
    
    else:
        # 使用 matplotlib 进行静态可视化
        if not PLOTLY_AVAILABLE and interactive:
            print("⚠️  Plotly 不可用，使用 matplotlib 静态可视化")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 为每个类别绘制点
        for task_category in sorted(set(labels)):
            mask = labels == task_category
            if np.any(mask):
                color = color_map.get(task_category, "gray")
                ax.scatter(
                    features_3d[mask, 0],
                    features_3d[mask, 1],
                    features_3d[mask, 2],
                    c=color,
                    label=task_category,
                    alpha=0.6,
                    s=50
                )
        
        ax.set_xlabel('t-SNE 维度 1', fontsize=12)
        ax.set_ylabel('t-SNE 维度 2', fontsize=12)
        ax.set_zlabel('t-SNE 维度 3', fontsize=12)
        ax.set_title('3D t-SNE 可视化：动作类别的特征分布', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ 图像已保存到: {output_path}")
        else:
            plt.show()
    
    return features_3d


def main():
    parser = argparse.ArgumentParser(
        description='可视化三个动作类别的 3D t-SNE 分布'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='数据集路径'
    )
    parser.add_argument(
        '--ckpt-path',
        type=str,
        required=True,
        help='模型 checkpoint 路径'
    )
    parser.add_argument(
        '--num-samples-per-task',
        type=int,
        default=0,
        help='每个任务类别采样的样本数。设置为0表示处理所有episodes（默认：0，处理所有）'
    )
    parser.add_argument(
        '--max-episodes',
        type=int,
        default=None,
        help='最大处理的 episodes 数（默认：全部）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='tsne_3d_visualization.png',
        help='输出图像路径（默认：tsne_3d_visualization.png）'
    )
    parser.add_argument(
        '--perplexity',
        type=int,
        default=30,
        help='t-SNE perplexity 参数（默认：30）'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=1000,
        help='t-SNE 最大迭代次数（默认：1000）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='计算设备（默认：cuda:0）'
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='禁用交互式可视化，只使用 matplotlib 静态图'
    )
    
    args = parser.parse_args()
    
    # 收集特征
    features, labels, episode_indices = collect_features(
        dataset_path=args.dataset_path,
        ckpt_path=args.ckpt_path,
        num_samples_per_task=args.num_samples_per_task,
        max_episodes=args.max_episodes,
        device=args.device
    )
    
    if len(features) == 0:
        print("❌ 错误: 没有提取到任何特征")
        return
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("样本统计")
    print("=" * 80)
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} 个样本")
    
    # 可视化
    features_3d = visualize_tsne_3d(
        features=features,
        labels=labels,
        episode_indices=episode_indices,
        output_path=args.output,
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        interactive=not args.no_interactive
    )
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"特征维度: {features.shape[1]} -> 3D")
    print(f"可视化已保存到: {args.output}")


if __name__ == '__main__':
    main()

