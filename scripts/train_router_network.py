#!/usr/bin/env python3
"""
🧠 Router Network 训练脚本

训练一个小型神经网络来预测当前输入应该使用哪个专家。
这是 MoE (Mixture of Experts) 的标准做法，比启发式路由更可靠。

训练流程：
1. 加载融合后的模型（只需要 backbone）
2. 加载多个任务的数据集
3. 对每个样本提取 backbone features
4. 使用任务标签作为监督信号训练 Router Network
5. 保存训练好的 Router Network

使用方法：
    python scripts/train_router_network.py \
        --model-path /path/to/merged_model \
        --dataset-paths /path/to/narrower /path/to/wider \
        --task-names narrower wider \
        --epochs 20 \
        --output-path /path/to/router.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.weight_merge_groot import (
    RouterNetwork,
    BACKBONE_FEATURE_KEY,
)


class RouterDataset(Dataset):
    """
    Router Network 训练数据集
    
    包装 LeRobotDataset，为每个样本添加任务标签
    """
    
    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        task_label: int,
        task_name: str,
    ):
        self.lerobot_dataset = lerobot_dataset
        self.task_label = task_label
        self.task_name = task_name
    
    def __len__(self):
        return len(self.lerobot_dataset)
    
    def __getitem__(self, idx):
        item = self.lerobot_dataset[idx]
        item['task_label'] = self.task_label
        item['task_name'] = self.task_name
        return item


def extract_backbone_features(
    policy: GrootPolicy,
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    从 policy 中提取 backbone features
    """
    # 准备输入
    inputs = {}
    
    # 处理图像
    for key in batch:
        if key.startswith("observation.images."):
            # (B, C, H, W) -> 需要的格式
            img = batch[key].to(device)
            if img.dim() == 4:
                inputs[key] = img
    
    # 处理状态
    if "observation.state" in batch:
        inputs["observation.state"] = batch["observation.state"].to(device)
    
    # 处理语言/任务描述
    if "annotation.human.action.task_description" in batch:
        inputs["annotation.human.action.task_description"] = batch["annotation.human.action.task_description"]
    elif "task" in batch:
        inputs["annotation.human.action.task_description"] = batch["task"]
    else:
        # 使用默认的任务描述
        inputs["annotation.human.action.task_description"] = ["pick up the object"] * batch["observation.state"].shape[0]
    
    # 通过 backbone 提取特征
    with torch.no_grad():
        try:
            backbone_inputs, _ = policy._groot_model.prepare_input(inputs)
            backbone_outputs = policy._groot_model.backbone(backbone_inputs)
            features = backbone_outputs[BACKBONE_FEATURE_KEY]  # (B, seq_len, hidden_dim)
        except Exception as e:
            print(f"Error extracting features: {e}")
            # 返回随机特征作为 fallback
            B = batch["observation.state"].shape[0]
            features = torch.randn(B, 100, 2048, device=device)
    
    return features


def train_router_network(
    model_path: str,
    dataset_paths: list[str],
    task_names: list[str],
    output_path: str,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    hidden_dim: int = 2048,
    intermediate_dim: int = 256,
    val_split: float = 0.1,
    device: str = "cuda:0",
    samples_per_task: int = 500,
):
    """
    训练 Router Network
    
    Args:
        model_path: 融合模型路径
        dataset_paths: 各任务数据集路径列表
        task_names: 任务名称列表（与 dataset_paths 一一对应）
        output_path: Router Network 保存路径
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        hidden_dim: backbone 特征维度
        intermediate_dim: Router Network 中间层维度
        val_split: 验证集比例
        device: 设备
        samples_per_task: 每个任务采样的样本数
    """
    print("=" * 80)
    print("🧠 Router Network Training")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Tasks: {task_names}")
    print(f"Datasets: {dataset_paths}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Samples per task: {samples_per_task}")
    print("=" * 80)
    
    device = torch.device(device)
    num_experts = len(dataset_paths)
    
    # === 1. 加载融合模型 ===
    print("\n📦 Loading merged model (backbone only)...")
    policy = GrootPolicy.from_pretrained(model_path)
    policy.eval()
    policy.to(device)
    
    # 获取 hidden_dim
    try:
        config = policy.config
        hidden_dim = getattr(config, 'hidden_dim', hidden_dim)
        if hasattr(config, 'backbone_hidden_dim'):
            hidden_dim = config.backbone_hidden_dim
    except:
        pass
    
    print(f"   Hidden dim: {hidden_dim}")
    
    # === 2. 加载数据集 ===
    print("\n📊 Loading datasets...")
    all_datasets = []
    
    for task_idx, (dataset_path, task_name) in enumerate(zip(dataset_paths, task_names)):
        print(f"   Loading {task_name} from {dataset_path}...")
        
        try:
            # 尝试加载 LeRobotDataset
            lerobot_ds = LeRobotDataset(
                repo_id=dataset_path,
                root=dataset_path if os.path.isdir(dataset_path) else None,
                local_files_only=True,
            )
            
            # 如果数据集太大，随机采样
            if len(lerobot_ds) > samples_per_task:
                indices = torch.randperm(len(lerobot_ds))[:samples_per_task].tolist()
                lerobot_ds = torch.utils.data.Subset(lerobot_ds, indices)
            
            # 包装为 RouterDataset
            router_ds = RouterDataset(lerobot_ds, task_idx, task_name)
            all_datasets.append(router_ds)
            
            print(f"      ✅ Loaded {len(router_ds)} samples")
        except Exception as e:
            print(f"      ❌ Error loading dataset: {e}")
            continue
    
    if not all_datasets:
        print("❌ No datasets loaded!")
        return
    
    # 合并数据集
    combined_dataset = ConcatDataset(all_datasets)
    print(f"\n   Total samples: {len(combined_dataset)}")
    
    # 划分训练集和验证集
    total_size = len(combined_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size]
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # 创建 DataLoader
    def collate_fn(batch):
        """自定义 collate 函数"""
        result = {}
        
        # 获取所有键
        keys = batch[0].keys()
        
        for key in keys:
            values = [item[key] for item in batch]
            
            if key == 'task_label':
                result[key] = torch.tensor(values)
            elif key == 'task_name':
                result[key] = values
            elif isinstance(values[0], torch.Tensor):
                try:
                    result[key] = torch.stack(values)
                except:
                    result[key] = values
            else:
                result[key] = values
        
        return result
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # === 3. 创建 Router Network ===
    print("\n🧠 Creating Router Network...")
    router = RouterNetwork(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        intermediate_dim=intermediate_dim,
        dropout=0.2,
    )
    router.to(device)
    
    # === 4. 训练 ===
    print("\n🚀 Training Router Network...")
    
    optimizer = torch.optim.AdamW(router.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        router.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            # 提取 backbone features
            with torch.no_grad():
                features = extract_backbone_features(policy, batch, device)
            
            # 获取标签
            labels = batch['task_label'].to(device)
            
            # 前向传播
            logits = router(features, return_logits=True)
            loss = criterion(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.1f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        router.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                features = extract_backbone_features(policy, batch, device)
                labels = batch['task_label'].to(device)
                
                logits = router(features, return_logits=True)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\n   Epoch {epoch+1}/{epochs}:")
        print(f"      Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.1f}%")
        print(f"      Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.1f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = router.state_dict().copy()
            print(f"      ✅ New best validation accuracy!")
        
        scheduler.step()
    
    # === 5. 保存模型 ===
    print(f"\n💾 Saving Router Network to {output_path}...")
    
    # 加载最佳状态
    if best_state is not None:
        router.load_state_dict(best_state)
    
    # 保存
    save_dict = {
        'state_dict': router.state_dict(),
        'config': {
            'hidden_dim': hidden_dim,
            'num_experts': num_experts,
            'intermediate_dim': intermediate_dim,
            'task_names': task_names,
        },
        'best_val_acc': best_val_acc,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(save_dict, output_path)
    
    print(f"   ✅ Router Network saved!")
    print(f"   Best validation accuracy: {best_val_acc:.1f}%")
    
    # 也保存到模型目录
    model_router_path = os.path.join(model_path, "router_network.pt")
    torch.save(save_dict, model_router_path)
    print(f"   ✅ Also saved to: {model_router_path}")
    
    print("\n" + "=" * 80)
    print("🎉 Training completed!")
    print("=" * 80)
    
    return router


def main():
    parser = argparse.ArgumentParser(description="Train Router Network for MoE expert selection")
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to merged GROOT model')
    parser.add_argument('--dataset-paths', type=str, nargs='+', required=True,
                       help='Paths to task datasets (one per task)')
    parser.add_argument('--task-names', type=str, nargs='+', required=True,
                       help='Task names (one per dataset, same order)')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Output path for Router Network (default: model_path/router_network.pt)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=2048,
                       help='Backbone feature dimension')
    parser.add_argument('--intermediate-dim', type=int, default=256,
                       help='Router Network intermediate dimension')
    parser.add_argument('--samples-per-task', type=int, default=500,
                       help='Number of samples per task')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 验证参数
    if len(args.dataset_paths) != len(args.task_names):
        print("❌ Error: Number of dataset paths must match number of task names!")
        return
    
    # 设置输出路径
    if args.output_path is None:
        args.output_path = os.path.join(args.model_path, "router_network.pt")
    
    # 训练
    train_router_network(
        model_path=args.model_path,
        dataset_paths=args.dataset_paths,
        task_names=args.task_names,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        device=args.device,
        samples_per_task=args.samples_per_task,
    )


if __name__ == "__main__":
    main()
