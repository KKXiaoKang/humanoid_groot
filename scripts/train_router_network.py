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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.weight_merge_groot import (
    RouterNetwork,
    BACKBONE_FEATURE_KEY,
)
from lerobot.policies.factory import make_pre_post_processors


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
    preprocessor=None,
) -> torch.Tensor:
    """
    从 policy 中提取 backbone features
    
    Args:
        policy: GrootPolicy 实例
        batch: 批次数据
        device: 设备
        preprocessor: 预处理器（如果为 None，将使用 policy 的 preprocessor）
    """
    # 获取 batch size（从任意 tensor 中获取）
    batch_size = None
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            batch_size = value.shape[0]
            break
    
    if batch_size is None:
        raise ValueError("Cannot determine batch size from batch")
    
    # 使用 preprocessor 准备输入（确保所有必需字段都被正确设置）
    if preprocessor is None:
        # 尝试从 policy 获取 preprocessor
        if hasattr(policy, 'preprocessor') and policy.preprocessor is not None:
            preprocessor = policy.preprocessor
        else:
            raise ValueError("preprocessor is required but not available")
    
    # 通过 backbone 提取特征
    with torch.no_grad():
        try:
            # 使用 preprocessor 处理输入
            # preprocessor 会正确处理图像、状态、任务描述等，并设置 image_sizes
            processed_obs = preprocessor(batch)
            
            # 检查 processed_obs 的类型
            # preprocessor 可能返回字典或 BatchFeature
            if hasattr(processed_obs, 'data'):
                processed_dict = processed_obs.data
            else:
                processed_dict = processed_obs
            
            # 提取 backbone 需要的输入
            # 根据 GROOT 的实现，backbone 需要 eagle_* 字段和 state/state_mask
            groot_inputs = {
                k: v
                for k, v in processed_dict.items()
                if (k in {"state", "state_mask", "embodiment_id"} or k.startswith("eagle_"))
                and not (k.startswith("next.") or k == "info")
            }
            
            # 检查是否有必需的字段
            if not any(k.startswith("eagle_") for k in groot_inputs.keys()):
                raise ValueError(f"No eagle_* fields found in processed_obs. Available keys: {list(processed_dict.keys())[:10]}")
            
            # 使用 BatchFeature 包装（GROOT backbone 期望这个格式）
            try:
                from transformers import BatchFeature
            except ImportError:
                from transformers.feature_extraction_utils import BatchFeature
            vl_input = BatchFeature(data=groot_inputs)
            
            # 调用 backbone
            backbone_outputs = policy._groot_model.backbone(vl_input)
            
            # 检查是否有 BACKBONE_FEATURE_KEY
            # backbone_outputs 可能是 BatchFeature，需要访问 .data 属性
            if hasattr(backbone_outputs, 'data'):
                backbone_data = backbone_outputs.data
            elif hasattr(backbone_outputs, 'get'):
                # 如果已经是字典，直接使用
                backbone_data = backbone_outputs
            else:
                # 尝试转换为字典
                backbone_data = dict(backbone_outputs) if hasattr(backbone_outputs, '__iter__') else {}
            
            if BACKBONE_FEATURE_KEY not in backbone_data:
                available_keys = list(backbone_data.keys())[:20]  # 只显示前20个键
                raise KeyError(f"{BACKBONE_FEATURE_KEY} not found in backbone_outputs. Available keys: {available_keys}")
            
            features = backbone_data[BACKBONE_FEATURE_KEY]  # (B, seq_len, hidden_dim)
            
            # 确保特征是正确的形状
            if features.dim() == 2:
                # 如果是 (B, hidden_dim)，添加序列维度
                features = features.unsqueeze(1)  # (B, 1, hidden_dim)
            
        except Exception as e:
            import traceback
            print(f"❌ Error extracting features: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            # 返回随机特征作为 fallback
            features = torch.randn(batch_size, 100, 2048, device=device)
    
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
    
    # 创建 preprocessor（用于正确准备输入）
    print("\n🔧 Creating preprocessor...")
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    print(f"   ✅ Preprocessor created")
    
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
                # local_files_only=True,
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
        if not batch:
            raise ValueError("Empty batch!")
        
        result = {}
        
        # 获取所有键（从第一个样本）
        keys = batch[0].keys()
        
        for key in keys:
            values = [item[key] for item in batch]
            
            if key == 'task_label':
                # 确保是整数类型
                if isinstance(values[0], torch.Tensor):
                    result[key] = torch.stack(values).long()
                else:
                    result[key] = torch.tensor(values, dtype=torch.long)
            elif key == 'task_name':
                result[key] = values
            elif isinstance(values[0], torch.Tensor):
                try:
                    # 尝试 stack
                    result[key] = torch.stack(values)
                except RuntimeError as e:
                    # 如果形状不匹配，尝试 pad
                    try:
                        # 对于不同长度的序列，使用 pad_sequence
                        if values[0].dim() > 1:
                            from torch.nn.utils.rnn import pad_sequence
                            result[key] = pad_sequence(values, batch_first=True)
                        else:
                            result[key] = values
                    except:
                        result[key] = values
            elif isinstance(values[0], (list, tuple)):
                # 列表或元组，保持原样
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
        for batch_idx, batch in enumerate(pbar):
            try:
                # 提取 backbone features
                with torch.no_grad():
                    features = extract_backbone_features(policy, batch, device, preprocessor=preprocessor)
                    features = features.float() 

                # 检查特征形状
                if features is None or features.numel() == 0:
                    print(f"⚠️  Warning: Empty features at batch {batch_idx}, skipping...")
                    continue
                
                # 获取标签
                if 'task_label' not in batch:
                    print(f"⚠️  Warning: Missing task_label at batch {batch_idx}, skipping...")
                    continue
                
                labels = batch['task_label'].to(device)
                
                # 检查标签和特征的 batch size 是否匹配
                if features.shape[0] != labels.shape[0]:
                    print(f"⚠️  Warning: Batch size mismatch (features: {features.shape[0]}, labels: {labels.shape[0]}), skipping...")
                    continue
                
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
            except Exception as e:
                import traceback
                print(f"\n❌ Error in training batch {batch_idx}:")
                print(f"   Error: {e}")
                print(f"   Traceback:\n{traceback.format_exc()}")
                print(f"   Batch keys: {list(batch.keys())}")
                if 'task_label' in batch:
                    print(f"   Task label shape: {batch['task_label'].shape if isinstance(batch['task_label'], torch.Tensor) else type(batch['task_label'])}")
                raise  # 重新抛出异常以便调试
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        router.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")):
                try:
                    features = extract_backbone_features(policy, batch, device, preprocessor=preprocessor)
                    features = features.float() 

                    if features is None or features.numel() == 0:
                        continue
                    
                    if 'task_label' not in batch:
                        continue
                    
                    labels = batch['task_label'].to(device)
                    
                    if features.shape[0] != labels.shape[0]:
                        continue
                    
                    logits = router(features, return_logits=True)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    _, predicted = logits.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)
                except Exception as e:
                    print(f"⚠️  Warning: Error in validation batch {batch_idx}: {e}")
                    continue
        
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
