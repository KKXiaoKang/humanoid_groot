# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Expert Merging for GROOT Models
# 
# 基于论文 "Expert Merging: Model Merging with Unsupervised Expert Alignment"
# https://arxiv.org/pdf/2509.25712
#
# 核心思想：
# 1. 计算 Task Vectors: τ_k = θ_k - θ_base
# 2. 学习 Layer-wise Coefficients: θ_merged = θ_base + Σ_k α_k^ℓ * τ_k^ℓ
# 3. 对齐损失：Hidden States + Logits (Action Predictions)
# 4. 正则化：防止系数过度偏移
#
# 适用场景：
# - 两个从相同base模型全量微调的GROOT模型
# - 希望合并成一个能处理多任务的模型

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import copy
import json
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from lerobot.policies.groot.groot_n1 import (
    GR00TN15,
    GR00TN15Config,
    BACKBONE_FEATURE_KEY,
    ACTION_KEY,
)


@dataclass
class ExpertMergeConfig:
    """Expert Merging 配置"""
    
    # 专家模型路径
    expert_paths: list[str] = field(default_factory=list)
    expert_names: list[str] = field(default_factory=list)
    
    # Base 模型路径（用于计算 Task Vectors）
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    
    # 融合模式
    merge_mode: Literal["layer_wise", "global", "chunk_wise"] = "layer_wise"
    
    # 融合范围控制
    merge_backbone_only: bool = False  # 如果为 True，只融合 backbone，action_head 使用第一个专家的
    action_head_source: Literal["first_expert", "second_expert", "interpolate"] = "first_expert"  # action_head 的来源
    
    # 训练配置
    learning_rate: float = 1e-3
    num_epochs: int = 10
    regularization_weight: float = 0.8  # γ in the paper
    temperature: float = 2.0  # KL divergence temperature
    
    # 初始系数（用于 Task Arithmetic 初始化）
    initial_coefficient: float = 0.5
    
    # 对齐配置
    hidden_alignment_layers: list[int] = field(default_factory=lambda: [-1])  # 对齐哪些层的hidden states
    hidden_alignment_weight: float = 1.0
    logit_alignment_weight: float = 1.0
    
    # 任务权重（控制不同专家的优先级）
    task_weights: list[float] = field(default_factory=lambda: [1.0, 1.0])
    
    # 计算配置
    compute_dtype: str = "bfloat16"
    device: str = "cuda:0"


class TaskVector:
    """
    Task Vector: τ = θ_expert - θ_base
    
    表示从 base 模型到 expert 模型的参数偏移
    """
    
    def __init__(
        self,
        base_state_dict: dict[str, torch.Tensor],
        expert_state_dict: dict[str, torch.Tensor],
        device: str = "cpu",
    ):
        """
        计算 Task Vector
        
        Args:
            base_state_dict: Base 模型的 state_dict
            expert_state_dict: Expert 模型的 state_dict
            device: 存储设备
        """
        self.task_vector = {}
        self.device = device
        
        # 计算差值
        for key in expert_state_dict:
            if key in base_state_dict:
                # τ = θ_expert - θ_base
                self.task_vector[key] = (
                    expert_state_dict[key].to(device) - base_state_dict[key].to(device)
                )
            else:
                # 如果 base 没有这个参数，直接使用 expert 的参数
                self.task_vector[key] = expert_state_dict[key].to(device)
        
        # 计算统计信息
        self._compute_stats()
    
    def _compute_stats(self):
        """计算 Task Vector 的统计信息"""
        self.layer_stats = {}
        for key, value in self.task_vector.items():
            if value.numel() > 0:
                self.layer_stats[key] = {
                    "mean_abs": value.abs().mean().item(),
                    "std": value.std().item(),
                    "numel": value.numel(),
                }
    
    def get_layer_importance(self, key: str) -> float:
        """获取某层的重要性分数（基于参数变化量）"""
        if key in self.layer_stats:
            stats = self.layer_stats[key]
            return stats["mean_abs"] * stats["numel"]
        return 0.0
    
    def __getitem__(self, key: str) -> torch.Tensor:
        return self.task_vector[key]
    
    def keys(self):
        return self.task_vector.keys()
    
    def items(self):
        return self.task_vector.items()


class LayerWiseCoefficients(nn.Module):
    """
    Layer-wise 可学习系数
    
    θ_merged^ℓ = θ_base^ℓ + Σ_k α_k^ℓ * τ_k^ℓ
    
    每层每个专家有一个独立的系数
    """
    
    def __init__(
        self,
        num_experts: int,
        layer_names: list[str],
        initial_value: float = 0.5,
        merge_backbone_only: bool = False,
        action_head_source: str = "first_expert",
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.layer_names = layer_names
        self.num_layers = len(layer_names)
        self.merge_backbone_only = merge_backbone_only
        self.action_head_source = action_head_source
        
        # 创建系数参数: (num_experts, num_layers)
        # 使用 ParameterDict 以便按层名访问
        self.coefficients = nn.ParameterDict()
        for layer_name in layer_names:
            # 如果只融合 backbone，跳过 action_head 层的系数
            if merge_backbone_only and layer_name.startswith('action_head.'):
                continue  # 不创建 action_head 层的系数
            
            # 每层的系数: (num_experts,)
            safe_name = layer_name.replace(".", "_")
            self.coefficients[safe_name] = nn.Parameter(
                torch.full((num_experts,), initial_value)
            )
        
        # 保存初始值用于正则化
        self.initial_value = initial_value
        self.layer_name_mapping = {
            name: name.replace(".", "_") for name in layer_names
        }
    
    def get_coefficient(self, layer_name: str, expert_idx: int) -> torch.Tensor:
        """获取某层某专家的系数"""
        safe_name = self.layer_name_mapping.get(layer_name)
        if safe_name and safe_name in self.coefficients:
            return self.coefficients[safe_name][expert_idx]
        return torch.tensor(self.initial_value)
    
    def get_all_coefficients(self, layer_name: str) -> torch.Tensor:
        """获取某层所有专家的系数"""
        safe_name = self.layer_name_mapping.get(layer_name)
        if safe_name and safe_name in self.coefficients:
            return self.coefficients[safe_name]
        return torch.full((self.num_experts,), self.initial_value)
    
    def regularization_loss(self) -> torch.Tensor:
        """
        系数正则化损失
        L_reg = (1/KL) * Σ_k Σ_ℓ |α_k^ℓ - α_init|
        """
        total_loss = 0.0
        count = 0
        for safe_name, coef in self.coefficients.items():
            total_loss = total_loss + (coef - self.initial_value).abs().sum()
            count += coef.numel()
        return total_loss / max(count, 1)
    
    def get_stats(self) -> dict:
        """获取系数统计信息"""
        all_coefs = []
        for safe_name, coef in self.coefficients.items():
            all_coefs.append(coef.detach())
        
        if all_coefs:
            all_coefs = torch.stack(all_coefs, dim=0)  # (num_layers, num_experts)
            return {
                "mean": all_coefs.mean().item(),
                "std": all_coefs.std().item(),
                "min": all_coefs.min().item(),
                "max": all_coefs.max().item(),
                "per_expert_mean": all_coefs.mean(dim=0).tolist(),
            }
        return {}


class MergedGR00TModel(nn.Module):
    """
    融合后的 GROOT 模型
    
    使用 Task Vectors 和 Layer-wise Coefficients 动态计算融合权重
    """
    
    def __init__(
        self,
        base_model: GR00TN15,
        task_vectors: list[TaskVector],
        coefficients: LayerWiseCoefficients,
        expert_names: list[str],
    ):
        super().__init__()
        
        self.base_model = base_model
        self.task_vectors = task_vectors
        self.coefficients = coefficients
        self.expert_names = expert_names
        self.num_experts = len(task_vectors)
        self.expert_models = []  # 将在 ExpertMerger 中设置
        
        # 注册 base model 的参数（不可训练）
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 缓存融合后的参数
        self._merged_state_dict = None
        self._cached = False
    
    def _is_backbone_layer(self, key: str) -> bool:
        """
        判断一个层是否属于 backbone
        
        Backbone 层通常以 'backbone.' 开头
        Action head 层通常以 'action_head.' 开头
        """
        return key.startswith('backbone.')
    
    def _is_action_head_layer(self, key: str) -> bool:
        """
        判断一个层是否属于 action_head
        """
        return key.startswith('action_head.')
    
    def _apply_merge(self):
        """
        应用权重融合
        
        θ_merged = θ_base + Σ_k α_k * τ_k
        
        如果 merge_backbone_only=True，只融合 backbone，action_head 使用指定专家的
        
        ⚠️ 重要：返回的 merged_state_dict 中的 tensor 需要保留梯度连接！
        不能使用 .clone()，而要使用直接计算，这样梯度才能反向传播到系数。
        """
        base_state_dict = self.base_model.state_dict()
        merged_state_dict = {}
        
        # 获取第一个和第二个专家的 state_dict（用于 action_head 选择）
        first_expert_state_dict = None
        second_expert_state_dict = None
        if len(self.expert_models) > 0:
            first_expert_state_dict = self.expert_models[0].state_dict()
        if len(self.expert_models) > 1:
            second_expert_state_dict = self.expert_models[1].state_dict()
        
        for key in base_state_dict:
            # 如果只融合 backbone，action_head 使用指定专家的
            if hasattr(self.coefficients, 'merge_backbone_only') and self.coefficients.merge_backbone_only:
                if self._is_action_head_layer(key):
                    # Action head 层：使用指定专家的权重（detach，不需要梯度）
                    if self.coefficients.action_head_source == "first_expert" and first_expert_state_dict and key in first_expert_state_dict:
                        merged_state_dict[key] = first_expert_state_dict[key].clone().detach()
                        continue
                    elif self.coefficients.action_head_source == "second_expert" and second_expert_state_dict and key in second_expert_state_dict:
                        merged_state_dict[key] = second_expert_state_dict[key].clone().detach()
                        continue
                    elif self.coefficients.action_head_source == "interpolate":
                        # 插值：使用固定系数 0.5 进行插值（简单插值）
                        if first_expert_state_dict and second_expert_state_dict and key in first_expert_state_dict and key in second_expert_state_dict:
                            merged_state_dict[key] = (0.5 * first_expert_state_dict[key] + 0.5 * second_expert_state_dict[key]).detach()
                            continue
                    # 如果找不到，使用 base 的
                    merged_state_dict[key] = base_state_dict[key].clone().detach()
                    continue
            
            # Backbone 层或全量融合：正常融合（保留梯度连接！）
            # ⚠️ 关键修复：不能使用 .clone()，要直接计算，这样梯度才能传播到系数
            base_param = base_state_dict[key]
            # ⚠️ 关键修复：确保 merged_param 的计算图正确，保留梯度连接
            # 先 detach base_param（base 不需要梯度），然后通过 coef * task_vec 添加梯度连接
            merged_param = base_param.detach().clone()  # detach base，因为 base 不需要梯度
            
            # 添加所有专家的 Task Vector 贡献（保留梯度连接）
            for expert_idx, task_vector in enumerate(self.task_vectors):
                if key in task_vector.task_vector:
                    # 检查是否有该层的系数（如果只融合 backbone，action_head 层可能没有系数）
                    try:
                        coef = self.coefficients.get_coefficient(key, expert_idx)
                        # ⚠️ 关键：直接计算，保留梯度连接
                        # coef 是可训练参数，task_vec 是 detach 的（不需要梯度）
                        # 所以 merged_param 的梯度来自 coef
                        task_vec = task_vector[key].to(merged_param.device).detach()  # task_vec 不需要梯度
                        # ⚠️ 关键：这里 coef 有梯度，所以 merged_param 也会有梯度！
                        merged_param = merged_param + coef * task_vec
                    except:
                        # 如果没有系数（比如 action_head 层在只融合 backbone 模式下），跳过
                        pass
            
            # ⚠️ 关键：确保 merged_param 有梯度（如果 coef 有梯度）
            # 但当我们使用 param.data = merged_param 时，梯度会断开
            # 所以我们需要确保在前向传播时，loss 的计算图连接到 merged_param
            
            merged_state_dict[key] = merged_param
        
        return merged_state_dict
    
    def get_merged_model(self) -> GR00TN15:
        """
        获取融合后的模型实例
        
        用于最终保存和推理
        """
        merged_state_dict = self._apply_merge()
        
        # 创建新模型并加载融合权重
        merged_model = copy.deepcopy(self.base_model)
        merged_model.load_state_dict(merged_state_dict, strict=False)
        
        return merged_model
    
    def forward_with_merged_weights(self, inputs: dict, compute_actions: bool = True) -> dict:
        """
        使用融合权重进行前向传播
        
        这是训练时使用的方法，会动态计算融合权重
        
        Args:
            inputs: 输入数据
            compute_actions: 是否计算 action outputs（如果只需要 hidden states，设为 False）
        
        ⚠️ 关键修复：使用 functional_call 保留梯度连接
        """
        from torch.func import functional_call
        
        # 应用融合权重（保留梯度连接）
        merged_state_dict = self._apply_merge()
        
        # 获取设备类型
        device = next(self.base_model.parameters()).device
        use_bf16 = getattr(self.base_model, "compute_dtype", None) == "bfloat16"
        
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            # 先准备输入
            backbone_inputs, action_inputs = self.base_model.prepare_input(inputs)
            
            # ⚠️ 关键：使用 functional_call 调用 backbone，保留梯度连接
            # 提取 backbone 的参数，去掉 backbone. 前缀
            backbone_params = {}
            for key, value in merged_state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key[len('backbone.'):]  # 去掉 backbone. 前缀
                    backbone_params[new_key] = value
            
            # 使用 functional_call 调用 backbone
            backbone_outputs = functional_call(
                self.base_model.backbone,
                backbone_params,
                (backbone_inputs,),
                tie_weights=False,
                strict=False,
            )
            
            # 如果不需要计算 actions（比如 action loss 权重为 0），只返回 backbone outputs
            if not compute_actions:
                return backbone_outputs
            
            # 提取 action_head 的参数，去掉 action_head. 前缀
            action_head_params = {}
            for key, value in merged_state_dict.items():
                if key.startswith('action_head.'):
                    new_key = key[len('action_head.'):]  # 去掉 action_head. 前缀
                    action_head_params[new_key] = value
            
            # 使用 functional_call 调用 action_head
            # action_head.forward 接受两个参数：backbone_output 和 action_input
            action_outputs = functional_call(
                self.base_model.action_head,
                action_head_params,
                (backbone_outputs, action_inputs),  # 两个位置参数
                tie_weights=False,
                strict=False,
            )
        
        return action_outputs
    
    def get_hidden_states(self, inputs: dict) -> torch.Tensor:
        """
        获取融合模型的隐藏状态
        
        用于 Hidden Alignment Loss
        
        ⚠️ 关键修复：使用 functional_call 保留梯度连接
        """
        from torch.func import functional_call
        
        # 应用融合权重（保留梯度连接）
        merged_state_dict = self._apply_merge()
        
        # 获取设备类型
        device = next(self.base_model.parameters()).device
        use_bf16 = getattr(self.base_model, "compute_dtype", None) == "bfloat16"
        
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            # 准备输入
            backbone_inputs, action_inputs = self.base_model.prepare_input(inputs)
            
            # ⚠️ 关键：使用 functional_call 调用 backbone，保留梯度连接
            # 提取 backbone 的参数，去掉 backbone. 前缀
            backbone_params = {}
            for key, value in merged_state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key[len('backbone.'):]  # 去掉 backbone. 前缀
                    backbone_params[new_key] = value
            
            # 使用 functional_call 调用 backbone
            backbone_outputs = functional_call(
                self.base_model.backbone,
                backbone_params,
                (backbone_inputs,),
                tie_weights=False,
                strict=False,
            )
            hidden_states = backbone_outputs[BACKBONE_FEATURE_KEY]
        
        return hidden_states


class ExpertMerger:
    """
    Expert Merging 主类
    
    实现论文中的完整训练流程
    """
    
    def __init__(self, config: ExpertMergeConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 加载模型
        print(f"\n{'='*60}")
        print(f"🚀 Expert Merging for GROOT")
        print(f"   Base model: {config.base_model_path}")
        print(f"   Experts: {config.expert_names}")
        print(f"   Mode: {config.merge_mode}")
        print(f"{'='*60}\n")
        
        self.base_model = None
        self.expert_models = []
        self.task_vectors = []
        self.coefficients = None
        self.merged_model = None
        
        # 预处理器和后处理器（从第一个专家模型加载）
        self.preprocessor = None
        self.postprocessor = None
    
    def load_models(self):
        """
        加载 Base 模型和 Expert 模型
        
        ⚠️ 重要：Base 模型必须与专家模型有相同的架构！
        
        如果 base_model_path 与第一个专家模型路径相同：
        - 只加载一次模型，复用作为 base 和 expert
        - narrower 的 Task Vector 为 0
        - wider 的 Task Vector = wider - narrower
        - 融合结果 = narrower + α * (wider - narrower)
        """
        
        # 检查 base 模型路径是否与第一个专家模型相同
        base_is_first_expert = (
            len(self.config.expert_paths) > 0 and
            self._normalize_path(self.config.base_model_path) == 
            self._normalize_path(self.config.expert_paths[0])
        )
        
        if base_is_first_expert:
            print(f"\n⚠️  Base 模型与第一个专家模型相同，使用优化加载策略")
            print(f"   Base/Expert[0]: {self.config.base_model_path}")
        
        # 1. 加载 Base 模型
        print(f"\n📦 Loading base model from {self.config.base_model_path}...")
        self.base_model = self._load_model(self.config.base_model_path)
        base_state_dict = self.base_model.state_dict()
        print(f"   ✅ Base model loaded: {sum(p.numel() for p in self.base_model.parameters()) / 1e6:.2f}M params")
        
        # 2. 加载 Expert 模型并计算 Task Vectors
        for i, expert_path in enumerate(self.config.expert_paths):
            name = self.config.expert_names[i] if i < len(self.config.expert_names) else f"expert_{i}"
            
            # 如果是第一个专家且与 base 相同，复用 base 模型
            if i == 0 and base_is_first_expert:
                print(f"\n📦 Expert '{name}' = Base model (复用)")
                expert_model = self.base_model
                expert_state_dict = base_state_dict
                self.expert_models.append(expert_model)
                
                # Task Vector 为 0
                print(f"   📐 Task vector = 0 (same as base)")
                task_vector = TaskVector(
                    base_state_dict=base_state_dict,
                    expert_state_dict=base_state_dict,  # 相同，所以 τ = 0
                    device="cpu",
                )
                self.task_vectors.append(task_vector)
                
                # 加载预处理器
                self._load_processors(expert_path)
                continue
            
            print(f"\n📦 Loading expert '{name}' from {expert_path}...")
            expert_model = self._load_model(expert_path)
            expert_state_dict = expert_model.state_dict()
            self.expert_models.append(expert_model)
            
            # 计算 Task Vector
            print(f"   📐 Computing task vector...")
            task_vector = TaskVector(
                base_state_dict=base_state_dict,
                expert_state_dict=expert_state_dict,
                device="cpu",  # 存储在 CPU 以节省 GPU 内存
            )
            self.task_vectors.append(task_vector)
            
            # 打印 Task Vector 统计
            total_change = sum(
                stats["mean_abs"] * stats["numel"] 
                for stats in task_vector.layer_stats.values()
            )
            print(f"   ✅ Task vector computed: total change = {total_change:.4f}")
            
            # 如果 base 不是第一个专家，从第一个专家模型加载预处理器
            if i == 0:
                self._load_processors(expert_path)
        
        # 3. 创建 Layer-wise Coefficients
        layer_names = list(base_state_dict.keys())
        self.coefficients = LayerWiseCoefficients(
            num_experts=len(self.task_vectors),
            layer_names=layer_names,
            initial_value=self.config.initial_coefficient,
            merge_backbone_only=self.config.merge_backbone_only,
            action_head_source=self.config.action_head_source,
        ).to(self.device)
        
        # 如果只融合 backbone，打印信息
        if self.config.merge_backbone_only:
            backbone_layers = [k for k in layer_names if k.startswith('backbone.')]
            action_head_layers = [k for k in layer_names if k.startswith('action_head.')]
            print(f"\n⚠️  只融合 Backbone 模式已启用")
            print(f"   Backbone 层数: {len(backbone_layers)}")
            print(f"   Action Head 层数: {len(action_head_layers)}")
            print(f"   Action Head 来源: {self.config.action_head_source}")
            if self.config.action_head_source == "first_expert":
                print(f"   → Action Head 将使用第一个专家模型（narrower）的权重")
            elif self.config.action_head_source == "second_expert":
                print(f"   → Action Head 将使用第二个专家模型（wider）的权重")
            elif self.config.action_head_source == "interpolate":
                print(f"   → Action Head 将使用两个专家模型的插值")
        
        print(f"\n📊 Coefficient stats:")
        print(f"   Layers: {len(layer_names)}")
        print(f"   Experts: {len(self.task_vectors)}")
        print(f"   Total trainable coefficients: {sum(p.numel() for p in self.coefficients.parameters())}")
        
        if base_is_first_expert:
            print(f"\n💡 融合策略说明:")
            print(f"   θ_merged = θ_narrower + α_wider * (θ_wider - θ_narrower)")
            print(f"   当 α_wider = 0 时，结果为 narrower 模型")
            print(f"   当 α_wider = 1 时，结果为 wider 模型")
            print(f"   α_wider ∈ (0, 1) 时，为两者的插值融合")
        
        # 4. 创建融合模型
        self.merged_model = MergedGR00TModel(
            base_model=self.base_model.to(self.device),
            task_vectors=self.task_vectors,
            coefficients=self.coefficients,
            expert_names=self.config.expert_names,
        )
        
        print(f"\n✅ All models loaded and ready for training")
    
    def _normalize_path(self, path: str) -> str:
        """标准化路径用于比较"""
        from pathlib import Path
        return str(Path(path).resolve())
    
    def _load_processors(self, model_path: str):
        """
        从模型 checkpoint 加载预处理器和后处理器
        
        参考 eval_on_dataset_lowpass.py 中的加载方式
        """
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig
        
        print(f"\n🔧 Loading preprocessor and postprocessor from {model_path}...")
        
        try:
            # 加载配置
            config = PreTrainedConfig.from_pretrained(model_path)
            
            # 加载预处理器和后处理器
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=config,
                pretrained_path=model_path,
                preprocessor_overrides={
                    "device_processor": {"device": str(self.device)},
                },
            )
            print(f"   ✅ Preprocessor and postprocessor loaded successfully")
            
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to load processors: {e}")
            print(f"   Will use raw inputs (may not work correctly)")
            self.preprocessor = None
            self.postprocessor = None
    
    def _load_model(self, model_path: str) -> GR00TN15:
        """
        加载单个模型
        
        参考 eval_depalletize_camera_model_reload_limit_vel_select.py 中的正确加载方式：
        使用 GrootPolicy.from_pretrained 加载，然后获取内部的 _groot_model
        """
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from pathlib import Path
        
        print(f"   Loading model from {model_path}...")
        
        # 使用正确的 GrootPolicy.from_pretrained 方式加载模型
        # 参考 eval_depalletize_camera_model_reload_limit_vel_select.py 第671行
        policy = GrootPolicy.from_pretrained(Path(model_path), strict=False)
        
        # 获取内部的 GR00TN15 模型实例
        model = policy._groot_model
        
        return model
    
    def compute_hidden_alignment_loss(
        self,
        merged_hidden: torch.Tensor,
        expert_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hidden Alignment Loss
        
        L_hid = ||h_merged - h_expert||_2^2
        """
        return F.mse_loss(merged_hidden, expert_hidden)
    
    def compute_logit_alignment_loss(
        self,
        merged_actions: torch.Tensor,
        expert_actions: torch.Tensor,
        temperature: float = 2.0,
    ) -> torch.Tensor:
        """
        Logit (Action) Alignment Loss
        
        对于 GROOT，我们对齐 action predictions 而不是 logits
        使用 MSE loss 因为 actions 是连续值
        
        L_logit = T^2 * KL(softmax(z_expert/T) || softmax(z_merged/T))
        
        但对于连续动作，我们使用 MSE:
        L_action = ||a_merged - a_expert||_2^2
        """
        return F.mse_loss(merged_actions, expert_actions)
    
    def train_step(
        self,
        batch: dict,
        optimizer: torch.optim.Optimizer,
        batch_idx: int = 0,
    ) -> dict:
        """
        单步训练
        
        Args:
            batch: 包含观测数据的字典（LeRobot 数据集格式）
            optimizer: 优化器
        
        Returns:
            损失字典
        """
        optimizer.zero_grad()
        
        # 准备输入：先转换为观测格式，然后使用预处理器
        observation = self._prepare_observation(batch)
        
        # 使用预处理器处理输入（如果可用）
        if self.preprocessor is not None:
            try:
                inputs = self.preprocessor(observation)
                # 确保预处理器返回的输入数据类型正确（转换为 bfloat16 如果需要）
                inputs = self._ensure_correct_dtype(inputs)
            except Exception as e:
                print(f"   ⚠️  Preprocessor failed: {e}, using raw observation")
                inputs = self._prepare_inputs(observation)
        else:
            inputs = self._prepare_inputs(observation)
        
        total_loss = 0.0
        loss_dict = {}
        
        # ⚠️ 关键修复：根据样本的任务来源，只对齐对应的专家！
        # 这样就不会有"左右脑互博"的问题
        
        # 1. 获取样本的任务来源
        # task_source: 0=narrower, 1=wider
        if 'task_source' in batch:
            task_source = batch['task_source']
            if isinstance(task_source, torch.Tensor):
                task_source = task_source[0].item()  # 取第一个样本的任务来源
            elif isinstance(task_source, list):
                task_source = task_source[0]
        else:
            # 如果没有 task_source，默认交替使用两个专家
            task_source = batch_idx % len(self.expert_models)
        
        # 2. 选择对应的专家模型
        expert_idx = int(task_source)
        if expert_idx >= len(self.expert_models):
            expert_idx = 0
        
        expert_model = self.expert_models[expert_idx]
        expert_name = self.config.expert_names[expert_idx]
        
        if batch_idx == 0:
            print(f"\n   🎯 Aligning with expert: {expert_name} (task_source={task_source})")
        
        # 3. 获取目标专家的 hidden states
        with torch.no_grad():
            expert_model.eval()
            expert_model = expert_model.to(self.device)
            
            device = next(expert_model.parameters()).device
            use_bf16 = getattr(expert_model, "compute_dtype", None) == "bfloat16"
            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                expert_backbone_inputs, _ = expert_model.prepare_input(inputs)
                expert_backbone_outputs = expert_model.backbone(expert_backbone_inputs)
                target_hidden = expert_backbone_outputs[BACKBONE_FEATURE_KEY]
        
        # 4. 获取融合模型的 hidden states
        merged_hidden = self.merged_model.get_hidden_states(inputs)
        
        # 5. 计算与目标专家的对齐损失（不是加权平均，而是直接对齐目标专家！）
        hidden_loss = self.compute_hidden_alignment_loss(merged_hidden, target_hidden)
        
        # 记录损失
        loss_dict[f"{expert_name}_hidden_loss"] = hidden_loss.item()
        loss_dict[f"{expert_name}_action_loss"] = 0.0
        loss_dict[f"{expert_name}_total_loss"] = hidden_loss.item()
        
        # 记录其他专家的 hidden loss（仅用于诊断，不影响优化）
        with torch.no_grad():
            for other_idx, other_model in enumerate(self.expert_models):
                if other_idx != expert_idx:
                    other_name = self.config.expert_names[other_idx]
                    other_model.eval()
                    other_model = other_model.to(self.device)
                    other_device = next(other_model.parameters()).device
                    other_bf16 = getattr(other_model, "compute_dtype", None) == "bfloat16"
                    with torch.autocast(device_type=other_device.type, dtype=torch.bfloat16, enabled=other_bf16):
                        other_backbone_inputs, _ = other_model.prepare_input(inputs)
                        other_backbone_outputs = other_model.backbone(other_backbone_inputs)
                        other_hidden = other_backbone_outputs[BACKBONE_FEATURE_KEY]
                    other_loss = F.mse_loss(merged_hidden.detach(), other_hidden)
                    loss_dict[f"{other_name}_hidden_loss"] = other_loss.item()
                    loss_dict[f"{other_name}_action_loss"] = 0.0
                    loss_dict[f"{other_name}_total_loss"] = other_loss.item()
            
        # 6. 计算总损失 = hidden_alignment_loss + regularization_loss
        total_loss = self.config.hidden_alignment_weight * hidden_loss
        loss_dict["hidden_loss"] = hidden_loss.item()
        
        # 检查损失是否正常
        if not torch.isfinite(hidden_loss):
            print(f"   ⚠️  Warning: Non-finite hidden_loss: {hidden_loss.item()}")
            optimizer.zero_grad()
            return loss_dict
        
        # 6. 添加正则化损失
        reg_loss = self.coefficients.regularization_loss()
        total_loss = total_loss + self.config.regularization_weight * reg_loss
        
        loss_dict["regularization_loss"] = reg_loss.item()
        loss_dict["total_loss"] = total_loss.item()
        
        # 检查 NaN/Inf
        if not torch.isfinite(total_loss):
            print(f"   ⚠️  Warning: Non-finite loss detected: {total_loss.item()}")
            print(f"   Hidden losses: {[loss_dict.get(f'{name}_hidden_loss', 'N/A') for name in self.config.expert_names]}")
            print(f"   Action losses: {[loss_dict.get(f'{name}_action_loss', 'N/A') for name in self.config.expert_names]}")
            # 跳过这个 batch
            optimizer.zero_grad()
            return loss_dict
        
        # 6. 反向传播
        total_loss.backward()
        
        # 检查系数是否有梯度（调试）
        if batch_idx == 0:
            has_grad = False
            for name, param in self.coefficients.named_parameters():
                if param.grad is not None:
                    has_grad = True
                    print(f"   ✅ Coefficient {name} has gradient: {param.grad.norm().item():.6f}")
                    break
            if not has_grad:
                print(f"   ⚠️  Warning: No gradients found in coefficients!")
                print(f"   This means the computation graph is broken.")
                print(f"   Checking merged_state_dict...")
                # 检查 merged_state_dict 中的值是否有梯度
                merged_state_dict = self.merged_model._apply_merge()
                has_grad_in_merged = False
                for key, value in list(merged_state_dict.items())[:5]:  # 只检查前5个
                    if hasattr(value, 'requires_grad') and value.requires_grad:
                        has_grad_in_merged = True
                        print(f"   ✅ merged_state_dict['{key}'] has gradient: {value.requires_grad}")
                        break
                if not has_grad_in_merged:
                    print(f"   ⚠️  merged_state_dict values don't have gradients!")
        
        # 7. 梯度裁剪（防止梯度爆炸）
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.coefficients.parameters(),
            max_norm=1.0,  # 梯度裁剪阈值
            error_if_nonfinite=False
        )
        loss_dict["grad_norm"] = grad_norm.item()
        
        # 检查梯度是否正常
        if not torch.isfinite(torch.tensor(grad_norm)) or grad_norm > 100.0:
            print(f"   ⚠️  Warning: Large gradient norm detected: {grad_norm:.4f}")
            # 清零梯度，跳过这个 step
            optimizer.zero_grad()
            return loss_dict
        
        optimizer.step()
        
        return loss_dict
    
    def _prepare_observation(self, batch: dict) -> dict:
        """
        将 LeRobot 数据集批次转换为模型观测格式
        
        参考 eval_on_dataset_lowpass.py 中的处理方式
        
        LeRobot batch 格式:
        - observation.state: (B, state_dim)
        - observation.images.cam_head: (B, C, H, W)
        - action: (B, action_dim)
        - task: str 或 list[str]
        
        输出格式（用于预处理器）:
        - observation.state: (B, state_dim)
        - observation.images.*: (B, C, H, W)
        - task: str
        """
        observation = {}
        
        # 复制所有观测数据
        for key, value in batch.items():
            if key.startswith('observation'):
                if isinstance(value, torch.Tensor):
                    observation[key] = value.to(self.device)
                else:
                    observation[key] = value
        
        # 处理 task 字段
        if 'task' in batch:
            task_value = batch['task']
            if isinstance(task_value, (list, tuple)) and len(task_value) > 0:
                observation['task'] = task_value[0]
            elif isinstance(task_value, str):
                observation['task'] = task_value
            else:
                observation['task'] = str(task_value) if task_value is not None else "Depalletize the box"
        else:
            observation['task'] = "Depalletize the box"
        
        return observation
    
    def _prepare_inputs(self, batch: dict) -> dict:
        """
        准备模型输入
        
        将 LeRobot 数据集格式转换为 GROOT 模型需要的格式
        
        LeRobot 数据集格式：
        - observation.state: (B, state_dim) 状态观测
        - observation.images.cam_head: (B, C, H, W) 图像观测
        - action: (B, action_dim) 动作
        - task: str 或 list[str] 任务描述
        
        GROOT 模型输入格式：
        - observation.state: (B, state_dim)
        - observation.images.*: (B, C, H, W)
        - task: str
        """
        def to_device(x):
            if isinstance(x, torch.Tensor):
                if torch.is_floating_point(x):
                    if self.config.compute_dtype == "bfloat16":
                        return x.to(self.device, dtype=torch.bfloat16)
                    return x.to(self.device)
                return x.to(self.device)
            return x
        
        # 转换批次数据
        inputs = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = to_device(value)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # 对于 task 等字段，可能是列表，取第一个
                if isinstance(value[0], str):
                    inputs[key] = value[0]  # 取第一个字符串
                else:
                    inputs[key] = value
            else:
                inputs[key] = value
        
        # 确保有 task 字段（如果没有则使用默认值）
        if 'task' not in inputs:
            inputs['task'] = "Depalletize the box"
        
        return inputs
    
    def _ensure_correct_dtype(self, inputs: dict) -> dict:
        """
        确保输入数据类型正确（转换为 bfloat16 如果需要）
        
        递归处理嵌套字典和列表中的张量
        """
        # 导入 tree 模块（与 groot_n1.py 相同的方式）
        try:
            import tree
        except ImportError:
            try:
                import dm_tree as tree
            except ImportError:
                # Fallback: 简单的递归实现
                def _map_structure(func, structure):
                    if isinstance(structure, dict):
                        return {k: _map_structure(func, v) for k, v in structure.items()}
                    elif isinstance(structure, (list, tuple)):
                        return type(structure)(_map_structure(func, item) for item in structure)
                    else:
                        return func(structure)
                tree = type('Tree', (), {'map_structure': staticmethod(_map_structure)})()
        
        def convert_tensor(x):
            if isinstance(x, torch.Tensor):
                if torch.is_floating_point(x):
                    if self.config.compute_dtype == "bfloat16":
                        return x.to(self.device, dtype=torch.bfloat16)
                    return x.to(self.device)
                return x.to(self.device)
            return x
        
        # 使用 tree.map_structure 递归处理嵌套结构
        return tree.map_structure(convert_tensor, inputs)
    
    def train(
        self,
        train_dataloader,
        num_epochs: int = None,
        learning_rate: float = None,
    ):
        """
        训练融合系数
        
        Args:
            train_dataloader: 训练数据加载器（包含少量校准样本）
            num_epochs: 训练轮数
            learning_rate: 学习率
        """
        num_epochs = num_epochs or self.config.num_epochs
        learning_rate = learning_rate or self.config.learning_rate
        
        # 只优化系数
        # 使用较小的学习率和梯度裁剪来防止训练不稳定
        optimizer = torch.optim.AdamW(
            self.coefficients.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),  # 默认 beta 值
            eps=1e-8,
        )
        
        print(f"\n{'='*60}")
        print(f"🏋️ Training Expert Merge Coefficients")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Regularization weight: {self.config.regularization_weight}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_dataloader):
                loss_dict = self.train_step(batch, optimizer, batch_idx)
                epoch_losses.append(loss_dict["total_loss"])
                
                if batch_idx % 10 == 0:
                    grad_norm_str = f", grad_norm = {loss_dict.get('grad_norm', 'N/A'):.4f}" if 'grad_norm' in loss_dict else ""
                    print(f"   Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}: loss = {loss_dict['total_loss']:.4f}{grad_norm_str}")
                    
                    # 打印详细的损失信息（前几个 batch）
                    if batch_idx < 3:
                        for key, value in loss_dict.items():
                            if key != 'total_loss' and key != 'grad_norm':
                                print(f"      {key}: {value:.4f}")
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            coef_stats = self.coefficients.get_stats()
            
            print(f"\n📊 Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"   Average loss: {avg_loss:.4f}")
            print(f"   Coefficient stats: mean={coef_stats.get('mean', 0):.4f}, "
                  f"std={coef_stats.get('std', 0):.4f}, "
                  f"range=[{coef_stats.get('min', 0):.4f}, {coef_stats.get('max', 0):.4f}]")
            print(f"   Per-expert mean: {coef_stats.get('per_expert_mean', [])}")
        
        print(f"\n✅ Training completed!")
        
        # 打印融合质量评估和建议
        self._print_merge_quality_report(coef_stats)
    
    def _print_merge_quality_report(self, coef_stats: dict):
        """
        打印融合质量评估报告
        
        ⚠️ 重要：Hidden loss 不是融合质量的好指标！
        真正的质量评估需要在实际任务上测试。
        """
        print(f"\n{'='*60}")
        print(f"📊 融合质量评估报告")
        print(f"{'='*60}")
        
        # 1. 系数分析
        per_expert_mean = coef_stats.get('per_expert_mean', [0.5, 0.5])
        print(f"\n1️⃣ 系数分析:")
        print(f"   Per-expert mean: {per_expert_mean}")
        
        if len(per_expert_mean) >= 2:
            alpha_narrower = per_expert_mean[0]
            alpha_wider = per_expert_mean[1]
            
            # 由于 base = narrower，τ_narrower = 0
            # 所以实际融合权重是 α_wider
            print(f"\n   💡 融合策略解读:")
            print(f"   θ_merged = θ_narrower + α_wider × (θ_wider - θ_narrower)")
            print(f"   实际融合比例: α_wider = {alpha_wider:.4f}")
            print(f"   → 约 {(1-alpha_wider)*100:.1f}% narrower + {alpha_wider*100:.1f}% wider")
        
        # 2. 融合质量警告
        print(f"\n2️⃣ 重要说明:")
        print(f"   ⚠️  Hidden alignment loss 不能直接反映融合质量！")
        print(f"   ⚠️  Loss 不下降是正常的（这是多目标优化问题）")
        
        # 3. 建议
        print(f"\n3️⃣ 下一步建议:")
        print(f"   1. 在实际任务上测试融合模型:")
        print(f"      python eval/eval_merged_groot.py --model_path ./outputs/merged_groot/pretrained_model")
        print(f"")
        print(f"   2. 分别测试 narrower 和 wider 任务的成功率")
        print(f"   3. 与原始专家模型对比性能")
        print(f"")
        print(f"   4. 如果效果不好，尝试其他融合方法:")
        print(f"      ./merge_groot_models.sh task_arithmetic 6  # 无需训练")
        print(f"      ./merge_groot_models.sh interpolation 6    # 直接插值")
        print(f"{'='*60}\n")
    
    def get_merged_model(self) -> GR00TN15:
        """获取训练后的融合模型"""
        return self.merged_model.get_merged_model()
    
    def save_merged_model(self, output_path: str):
        """
        保存融合后的模型
        
        Args:
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取融合模型
        merged_model = self.get_merged_model()
        
        # 保存权重
        state_dict = merged_model.state_dict()
        
        # ⚠️ 修复：处理共享权重（tied weights）问题
        # 语言模型中 lm_head.weight 和 embed_tokens.weight 共享内存
        # safetensors 不允许保存共享内存的张量，需要先克隆
        state_dict_cloned = {}
        for key, value in state_dict.items():
            # 克隆所有张量以断开共享内存
            state_dict_cloned[key] = value.clone().contiguous()
        
        save_file(state_dict_cloned, str(output_path / "model.safetensors"))
        
        # 保存配置（使用第一个专家的配置作为模板）
        first_expert_path = Path(self.config.expert_paths[0])
        
        # 复制配置文件
        import shutil
        for config_file in ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"]:
            src = first_expert_path / config_file
            if src.exists():
                shutil.copy(src, output_path / config_file)
        
        # 复制 preprocessor/postprocessor safetensors
        for pattern in ["policy_preprocessor*.safetensors", "policy_postprocessor*.safetensors"]:
            for src in first_expert_path.glob(pattern):
                shutil.copy(src, output_path / src.name)
        
        # 保存融合配置
        merge_config = {
            "merge_mode": self.config.merge_mode,
            "expert_paths": self.config.expert_paths,
            "expert_names": self.config.expert_names,
            "base_model_path": self.config.base_model_path,
            "coefficient_stats": self.coefficients.get_stats(),
            "merge_backbone_only": self.config.merge_backbone_only,
            "action_head_source": self.config.action_head_source,
        }
        with open(output_path / "merge_config.json", "w") as f:
            json.dump(merge_config, f, indent=2)
        
        print(f"\n✅ Merged model saved to {output_path}")
        print(f"   - model.safetensors")
        print(f"   - config.json")
        print(f"   - merge_config.json")


# ============================================================
# MergeVLA 风格的融合方法
# ============================================================
# 
# 基于论文 "MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent"
# https://arxiv.org/pdf/2511.18810
# 
# 核心思想：
# 1. 稀疏激活的 LoRA 适配器（通过任务掩码）
# 2. Cross-attention-only action head（GROOT 已满足）
# 3. 测试时任务路由
# 
# 对于全量微调的模型：
# - 计算 Task Vectors: τ = θ_expert - θ_base
# - 使用稀疏 LoRA 适配器对齐 backbone 分布
# - Action head 使用 cross-attention，可以尝试直接融合或任务特定头

# ============================================================
# 分布适配层 (Distribution Adapter)
# ============================================================
# 
# 基于 MergeVLA 的思想：使用稀疏激活的 LoRA 适配器
# 
# 问题：融合 backbone 后，输出分布发生漂移，导致 action_head 崩溃
# 解决：在 backbone 和 action_head 之间插入一个轻量级适配层
# 
# 架构：
# backbone_merged → [DistributionAdapter] → action_head_narrower → action
#                          ↑
#                   只训练这一层！

class SparseLoRAAdapter(nn.Module):
    """
    稀疏激活的 LoRA 适配层（MergeVLA 风格）
    
    使用任务掩码来稀疏激活不同的 LoRA 参数子集
    这样可以减少不同任务之间的冲突
    
    参考 MergeVLA 论文：Sparsely activated LoRA adapters via task masks
    
    ⭐ MergeVLA 测试时任务路由（Test-Time Task Routing）：
    当任务身份未知时，根据模型内部参数子空间（value projection）
    直接推断任务相关性，无需训练。
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        rank: int = 16,
        alpha: float = 16.0,
        num_tasks: int = 2,  # 任务数量（narrower, wider）
        sparsity: float = 0.5,  # 稀疏度：每个任务激活的参数比例
        dropout: float = 0.0,
        routing_temperature: float = 0.1,  # 路由分数的温度参数
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        self.num_tasks = num_tasks
        self.sparsity = sparsity
        self.routing_temperature = routing_temperature
        
        # 为每个任务创建独立的 LoRA 参数
        # A: (num_tasks, hidden_size, rank)
        # B: (num_tasks, rank, hidden_size)
        self.lora_A = nn.Parameter(torch.randn(num_tasks, hidden_size, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(num_tasks, rank, hidden_size))
        
        # 任务掩码：每个任务激活哪些参数
        # mask: (num_tasks, hidden_size) - 二进制掩码
        # 使用可学习的掩码（通过 sigmoid 实现软掩码）
        # ⚠️ 关键修复：初始化掩码以实现稀疏度
        # 如果 sparsity=0.5，我们希望 sigmoid(mask) 后约 50% 的参数被激活
        # 使用伯努利分布初始化：约 sparsity 比例的参数初始化为较大的值
        mask_init = torch.zeros(num_tasks, hidden_size)
        num_active = int(hidden_size * sparsity)
        for t in range(num_tasks):
            # 随机选择 sparsity 比例的参数初始化为较大的值（经过 sigmoid 后接近 1）
            # 其余参数保持为较小的值（经过 sigmoid 后接近 0）
            indices = torch.randperm(hidden_size)[:num_active]
            mask_init[t, indices] = 5.0  # sigmoid(5.0) ≈ 0.993，表示激活
            # 其余位置初始化为 -5.0（sigmoid(-5.0) ≈ 0.007，表示不激活）
            inactive_mask = torch.ones(hidden_size, dtype=torch.bool)
            inactive_mask[indices] = False
            mask_init[t, inactive_mask] = -5.0
        self.task_masks = nn.Parameter(mask_init)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual_scale = nn.Parameter(torch.ones(1))
        
        # ⭐ 用于测试时任务路由的诊断计数器
        self._routing_call_count = 0
        self._routing_stats = {'task_0': 0, 'task_1': 0, 'mixed': 0}
        
        print(f"   Sparse LoRA Adapter (MergeVLA style):")
        print(f"      rank={rank}, alpha={alpha}, num_tasks={num_tasks}")
        print(f"      sparsity={sparsity}, params={self._count_params():,}")
        print(f"      ⭐ Test-Time Task Routing enabled (temperature={routing_temperature})")
    
    def _count_params(self):
        return self.lora_A.numel() + self.lora_B.numel() + self.task_masks.numel() + 1
    
    def compute_task_routing_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        ⭐ MergeVLA 测试时任务路由（Test-Time Task Routing）
        
        当任务身份未知时，根据输入特征与各任务 LoRA 参数子空间的相似度
        来推断任务相关性。
        
        核心思想（参考 MergeVLA 论文 Section 3.3）：
        - 使用 LoRA A 参数作为各任务的"任务表示"（value projection 的代理）
        - 将输入投影到各任务的 LoRA 子空间
        - 投影后的范数越大，表示输入与该任务越相关
        
        Args:
            x: (B, T, hidden_size) 输入特征
        
        Returns:
            (B, num_tasks) 任务路由权重（softmax 归一化后的概率分布）
        """
        B, T, H = x.shape
        
        # 使用 LoRA A 参数作为任务表示
        # lora_A: (num_tasks, hidden_size, rank)
        lora_A = self.lora_A.to(dtype=x.dtype)
        task_masks = torch.sigmoid(self.task_masks).to(dtype=x.dtype)  # (num_tasks, hidden_size)
        
        # 计算各任务的"任务表示"：使用掩码后的 LoRA A
        # masked_A[t]: (hidden_size, rank) - 第 t 个任务的特征投影矩阵
        
        scores = []
        for t in range(self.num_tasks):
            # 应用任务掩码
            A_t = lora_A[t] * task_masks[t].unsqueeze(-1)  # (hidden_size, rank)
            
            # 将输入投影到任务子空间
            # x: (B, T, hidden_size), A_t: (hidden_size, rank)
            # projection: (B, T, rank)
            projection = x @ A_t
            
            # 计算投影的范数作为相似度分数
            # 范数越大，表示输入在该任务子空间的投影越强，即越相关
            # 使用 L2 范数，对时间维度取平均
            score = projection.norm(dim=-1).mean(dim=-1)  # (B,)
            scores.append(score)
        
        # 堆叠并归一化为概率分布
        scores = torch.stack(scores, dim=-1)  # (B, num_tasks)
        
        # 使用 softmax 归一化，temperature 控制分布的锐度
        # temperature 越小，分布越尖锐（更倾向于选择单一任务）
        # temperature 越大，分布越平坦（更倾向于混合）
        routing_weights = F.softmax(scores / self.routing_temperature, dim=-1)
        
        return routing_weights
    
    def forward(
        self, 
        x: torch.Tensor, 
        task_id: torch.Tensor = None,
        use_smart_routing: bool = True,  # ⭐ 是否使用智能任务路由
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, hidden_size)
            task_id: (B,) - 任务ID，0=narrower, 1=wider。
                     如果为 None 且 use_smart_routing=True，使用智能任务路由
                     如果为 None 且 use_smart_routing=False，使用简单平均
            use_smart_routing: 当 task_id=None 时，是否使用 MergeVLA 风格的智能任务路由
        """
        lora_A = self.lora_A.to(dtype=x.dtype)
        lora_B = self.lora_B.to(dtype=x.dtype)
        task_masks = torch.sigmoid(self.task_masks).to(dtype=x.dtype)  # 软掩码
        residual_scale = self.residual_scale.to(dtype=x.dtype)
        
        B, T, H = x.shape
        
        if task_id is not None:
            # 使用指定任务的 LoRA 参数
            # task_id: (B,)
            task_id = task_id.long()
            
            # 选择每个样本对应的任务参数
            # A_selected: (B, H, rank)
            A_selected = lora_A[task_id]  # (B, H, rank)
            B_selected = lora_B[task_id]  # (B, rank, H)
            mask_selected = task_masks[task_id]  # (B, H)
            
            # 应用任务掩码到 LoRA A
            # mask_selected: (B, H) -> (B, H, 1) for broadcasting
            A_masked = A_selected * mask_selected.unsqueeze(-1)  # (B, H, rank)
            
            # LoRA 变换: x @ A @ B
            # x: (B, T, H), A_masked: (B, H, rank), B_selected: (B, rank, H)
            lora_output = torch.bmm(torch.bmm(self.dropout(x), A_masked), B_selected)  # (B, T, H)
        
        elif use_smart_routing:
            # ⭐ MergeVLA 风格：测试时智能任务路由
            # 根据输入特征推断任务相关性，动态加权各任务输出
            
            # 计算任务路由权重
            routing_weights = self.compute_task_routing_scores(x)  # (B, num_tasks)
            
            # 诊断：记录路由统计
            self._routing_call_count += 1
            with torch.no_grad():
                avg_weights = routing_weights.mean(dim=0)  # (num_tasks,)
                dominant_task = avg_weights.argmax().item()
                weight_diff = abs(avg_weights[0] - avg_weights[1]).item()
                
                if weight_diff > 0.3:  # 明显偏向某个任务
                    if dominant_task == 0:
                        self._routing_stats['task_0'] += 1
                    else:
                        self._routing_stats['task_1'] += 1
                else:
                    self._routing_stats['mixed'] += 1
                
                # 前几次调用打印诊断信息
                if self._routing_call_count <= 5:
                    print(f"\n   🎯 Smart Task Routing (call #{self._routing_call_count}):")
                    print(f"      Routing weights: task_0={avg_weights[0].item():.4f}, "
                          f"task_1={avg_weights[1].item():.4f}")
                    if dominant_task == 0:
                        print(f"      → Leaning towards task 0 (narrower)")
                    else:
                        print(f"      → Leaning towards task 1 (wider)")
            
            # 计算各任务的输出
            outputs = []
            for t in range(self.num_tasks):
                A_t = lora_A[t] * task_masks[t].unsqueeze(-1)  # (H, rank)
                B_t = lora_B[t]  # (rank, H)
                output_t = self.dropout(x) @ A_t @ B_t  # (B, T, H)
                outputs.append(output_t)
            
            outputs = torch.stack(outputs, dim=0)  # (num_tasks, B, T, H)
            
            # 使用路由权重加权各任务的输出
            # routing_weights: (B, num_tasks) -> (num_tasks, B, 1, 1) for broadcasting
            weights = routing_weights.permute(1, 0).unsqueeze(-1).unsqueeze(-1)  # (num_tasks, B, 1, 1)
            lora_output = (outputs * weights).sum(dim=0)  # (B, T, H)
        
        else:
            # 简单平均（旧方法，不推荐）
            outputs = []
            for t in range(self.num_tasks):
                A_t = lora_A[t] * task_masks[t].unsqueeze(-1)  # (H, rank)
                B_t = lora_B[t]  # (rank, H)
                output_t = self.dropout(x) @ A_t @ B_t  # (B, T, H)
                outputs.append(output_t)
            
            # 平均所有任务的输出
            lora_output = torch.stack(outputs, dim=0).mean(dim=0)  # (B, T, H)
        
        # 缩放并添加到输入
        return x + residual_scale * (self.alpha / self.rank) * lora_output
    
    def get_routing_stats(self) -> dict:
        """获取任务路由统计信息"""
        total = sum(self._routing_stats.values())
        if total == 0:
            return self._routing_stats
        
        return {
            'task_0_ratio': self._routing_stats['task_0'] / total,
            'task_1_ratio': self._routing_stats['task_1'] / total,
            'mixed_ratio': self._routing_stats['mixed'] / total,
            'total_calls': total,
            **self._routing_stats,
        }
    
    def reset_routing_stats(self):
        """重置路由统计"""
        self._routing_call_count = 0
        self._routing_stats = {'task_0': 0, 'task_1': 0, 'mixed': 0}


class LoRAAdapter(nn.Module):
    """
    LoRA 适配层（标准版本，用于向后兼容）
    
    使用低秩分解：W = W_base + A @ B
    其中 A: (hidden_size, rank), B: (rank, hidden_size)
    rank << hidden_size，参数量小但表达能力足够
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        rank: int = 16,  # LoRA rank，通常 8-32
        alpha: float = 16.0,  # LoRA scaling factor
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        
        # LoRA 参数：A 和 B
        self.lora_A = nn.Parameter(torch.randn(hidden_size, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, hidden_size))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 残差连接的系数
        self.residual_scale = nn.Parameter(torch.ones(1))
        
        print(f"   LoRA Adapter: rank={rank}, alpha={alpha}, params={self._count_params():,}")
    
    def _count_params(self):
        return self.lora_A.numel() + self.lora_B.numel() + 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoRA forward: x + residual_scale * (alpha/rank) * dropout(x) @ A @ B
        """
        # ⚠️ 关键修复：确保 LoRA 参数与输入 x 的 dtype 匹配
        # 因为 x 可能是 bfloat16（在 autocast 中），而 LoRA 参数默认是 float32
        lora_A = self.lora_A.to(dtype=x.dtype)
        lora_B = self.lora_B.to(dtype=x.dtype)
        residual_scale = self.residual_scale.to(dtype=x.dtype)
        
        # LoRA 变换
        lora_output = self.dropout(x) @ lora_A @ lora_B
        # 缩放并添加到输入（使用 residual_scale 控制强度）
        return x + residual_scale * (self.alpha / self.rank) * lora_output


class DistributionAdapter(nn.Module):
    """
    分布适配层：将融合后的 backbone 输出分布映射回 action_head 期望的分布
    
    基于 MergeVLA 的思想：使用稀疏激活的 LoRA 适配器
    
    支持多种适配层类型：
    - "sparse_lora": 稀疏激活的 LoRA（MergeVLA 风格，推荐）⭐
    - "lora": 标准 LoRA 适配层
    - "linear": 简单线性层
    - "mlp": 两层 MLP
    - "layernorm_only": 只用 LayerNorm
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        adapter_type: str = "sparse_lora",  # "sparse_lora", "lora", "linear", "mlp", "layernorm_only"
        dropout: float = 0.0,
        lora_rank: int = 16,  # LoRA rank（仅用于 lora/sparse_lora 类型）
        num_tasks: int = 2,  # 任务数量（仅用于 sparse_lora）
        sparsity: float = 0.5,  # 稀疏度（仅用于 sparse_lora）
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_type = adapter_type
        
        if adapter_type == "sparse_lora":
            # ⭐ MergeVLA 风格：稀疏激活的 LoRA 适配器
            self.adapter = SparseLoRAAdapter(
                hidden_size=hidden_size,
                rank=lora_rank,
                alpha=16.0,
                num_tasks=num_tasks,
                sparsity=sparsity,
                dropout=dropout,
            )
            self.residual_scale = None
        elif adapter_type == "lora":
            # 标准 LoRA 适配层
            self.adapter = LoRAAdapter(
                hidden_size=hidden_size,
                rank=lora_rank,
                alpha=16.0,
                dropout=dropout,
            )
            # LoRA 不需要额外的 residual_scale（已经在 LoRAAdapter 中）
            self.residual_scale = None
        
        elif adapter_type == "linear":
            # ⚠️ 警告：Linear adapter 表达能力不够，可能导致训练失败
            # 建议使用 MLP adapter
            print(f"   ⚠️ Warning: Linear adapter may not have enough capacity!")
            print(f"      Consider using 'mlp' adapter_type for better results.")
            # 简单线性变换 + LayerNorm（最轻量）
            self.adapter = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.LayerNorm(hidden_size),
            )
            # 初始化为接近恒等映射
            nn.init.eye_(self.adapter[0].weight)
            nn.init.zeros_(self.adapter[0].bias)
            
        elif adapter_type == "mlp":
            # 两层 MLP（更强表达能力）+ 残差连接
            self.adapter = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            # ⚠️ 关键改进：更好的初始化策略
            # 第一层：Xavier 初始化，小 gain（接近恒等映射但允许变化）
            nn.init.xavier_uniform_(self.adapter[0].weight, gain=0.1)
            nn.init.zeros_(self.adapter[0].bias)
            # 最后一层：初始化为接近零（残差风格，更容易学习）
            nn.init.zeros_(self.adapter[-2].weight)
            nn.init.zeros_(self.adapter[-2].bias)
            
        elif adapter_type == "layernorm_only":
            # 只用 LayerNorm（最简单）
            self.adapter = nn.LayerNorm(hidden_size)
            
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        # 残差连接的系数（可学习，仅用于非 LoRA 类型）
        if adapter_type != "lora":
            self.residual_scale = nn.Parameter(torch.ones(1))
        else:
            self.residual_scale = None
        
    def forward(
        self, 
        x: torch.Tensor, 
        task_id: torch.Tensor = None,
        use_smart_routing: bool = True,  # ⭐ 是否使用智能任务路由
    ) -> torch.Tensor:
        """
        Args:
            x: backbone 输出 [B, T, hidden_size]
            task_id: (B,) - 任务ID，0=narrower, 1=wider（仅用于 sparse_lora）
                     如果为 None 且 use_smart_routing=True，使用智能任务路由
            use_smart_routing: 当 task_id=None 时，是否使用 MergeVLA 风格的智能任务路由
        Returns:
            适配后的特征 [B, T, hidden_size]
        """
        if self.adapter_type == "sparse_lora":
            # MergeVLA 风格：稀疏激活的 LoRA
            # 支持智能任务路由（当 task_id=None 时）
            return self.adapter(x, task_id=task_id, use_smart_routing=use_smart_routing)
        elif self.adapter_type == "lora":
            # LoRA 适配层（内置残差连接）
            return self.adapter(x)
        elif self.adapter_type == "mlp":
            # ⚠️ 关键：残差连接让适配层更容易学习
            # 初始时 residual_scale=1.0，adapter 输出接近 0，所以输出 ≈ x
            # 训练时 adapter 学习到有用的变换，residual_scale 可以调整强度
            return x + self.residual_scale * self.adapter(x)
        else:
            return self.adapter(x)
    
    def get_routing_stats(self) -> dict | None:
        """获取任务路由统计信息（仅 sparse_lora 适配器支持）"""
        if self.adapter_type == "sparse_lora" and hasattr(self.adapter, 'get_routing_stats'):
            return self.adapter.get_routing_stats()
        return None
    
    def reset_routing_stats(self):
        """重置路由统计（仅 sparse_lora 适配器支持）"""
        if self.adapter_type == "sparse_lora" and hasattr(self.adapter, 'reset_routing_stats'):
            self.adapter.reset_routing_stats()


class MergedModelWithAdapter(nn.Module):
    """
    带适配层的融合模型
    
    结构：
    - backbone: 融合后的 backbone（冻结）
    - adapter: 分布适配层（可训练）
    - action_head: narrower 的 action_head（冻结）
    """
    
    def __init__(
        self,
        merged_backbone_state_dict: dict,
        narrower_action_head_state_dict: dict,
        base_model: GR00TN15,
        adapter_type: str = "linear",
        hidden_size: int = 1024,
        lora_rank: int = 16,
        num_tasks: int = 2,  # 任务数量（用于 sparse_lora）
        sparsity: float = 0.5,  # 稀疏度（用于 sparse_lora）
    ):
        super().__init__()
        
        # 1. 创建模型结构（使用 base_model 的架构）
        self.model = copy.deepcopy(base_model)
        
        # 2. 加载融合后的 backbone 权重
        backbone_keys = [k for k in merged_backbone_state_dict.keys() if k.startswith('backbone.')]
        backbone_state = {k: merged_backbone_state_dict[k] for k in backbone_keys}
        self.model.load_state_dict(backbone_state, strict=False)
        
        # 3. 加载 narrower 的 action_head 权重
        action_head_keys = [k for k in narrower_action_head_state_dict.keys() if k.startswith('action_head.')]
        action_head_state = {k: narrower_action_head_state_dict[k] for k in action_head_keys}
        self.model.load_state_dict(action_head_state, strict=False)
        
        # 4. 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 5. 创建适配层（可训练）
        self.adapter = DistributionAdapter(
            hidden_size=hidden_size,
            adapter_type=adapter_type,
            lora_rank=lora_rank,
            num_tasks=num_tasks,
            sparsity=sparsity,
        )
        
        print(f"✅ MergedModelWithAdapter initialized")
        print(f"   Adapter type: {adapter_type}")
        print(f"   Adapter params: {sum(p.numel() for p in self.adapter.parameters()):,}")
    
    def forward(self, inputs: dict, task_id: torch.Tensor = None) -> dict:
        """
        前向传播（用于训练适配层）
        
        ⚠️ 关键：确保 inputs 包含 GROOT action_head 所需的所有字段：
        - action: (B, action_horizon, action_dim) - ground truth action
        - action_mask: (B, action_horizon, action_dim) - 有效维度掩码
        - state: (B, 1, state_dim) - 状态观测
        - embodiment_id: (B,) - embodiment ID
        - video 或图像数据 - 视觉输入
        
        Args:
            inputs: 模型输入
            task_id: (B,) - 任务ID，0=narrower, 1=wider（用于 sparse_lora）
        """
        # 获取计算设备和数据类型
        device = next(self.model.parameters()).device
        use_bf16 = getattr(self.model, "compute_dtype", None) == "bfloat16"
        
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            # 1. 准备输入
            # model.prepare_input 会调用 backbone.prepare_input 和 action_head.prepare_input
            # action_head.prepare_input 只是返回 BatchFeature(data=inputs)
            backbone_inputs, action_inputs = self.model.prepare_input(inputs)
            
            # 2. 通过 backbone
            backbone_outputs = self.model.backbone(backbone_inputs)
            
            # 3. 通过适配层（传递 task_id 用于 sparse_lora）
            backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]
            adapted_features = self.adapter(backbone_features, task_id=task_id)
            backbone_outputs[BACKBONE_FEATURE_KEY] = adapted_features
            
            # 4. 通过 action_head（计算 Flow Matching loss）
            # action_head.forward 会：
            # - 从 action_inputs 获取 action, action_mask, state, embodiment_id
            # - 计算 Flow Matching loss: MSE(pred_velocity, action - noise)
            action_outputs = self.model.action_head(backbone_outputs, action_inputs)
        
        return action_outputs
    
    def get_action(self, inputs: dict, task_id: torch.Tensor = None, **kwargs) -> dict:
        """
        推理时获取动作
        
        Args:
            inputs: 模型输入
            task_id: (B,) - 任务ID，0=narrower, 1=wider（用于 sparse_lora，如果为 None 则使用平均）
            **kwargs: 其他参数（如 rtc_enabled）
        """
        # 1. 通过 backbone
        backbone_inputs, action_inputs = self.model.prepare_input(inputs)
        backbone_outputs = self.model.backbone(backbone_inputs)
        
        # 2. 通过适配层（传递 task_id，如果为 None 则使用所有任务的平均）
        backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]
        adapted_features = self.adapter(backbone_features, task_id=task_id)
        backbone_outputs[BACKBONE_FEATURE_KEY] = adapted_features
        
        # 3. 通过 action_head (推理模式)
        # ⚠️ 修复：get_action 需要 rtc_enabled 参数
        rtc_enabled = kwargs.pop('rtc_enabled', False)
        action_outputs = self.model.action_head.get_action(backbone_outputs, action_inputs, rtc_enabled=rtc_enabled, **kwargs)
        
        return action_outputs
    
    def forward_with_direct_loss(self, inputs: dict, gt_action: torch.Tensor) -> dict:
        """
        使用直接 action MSE loss 进行训练（推荐！）
        
        ⚠️ 关键改进：
        - 直接比较预测 action 和 ground truth action
        - 梯度直接流向适配层，比 Flow Matching loss 强得多
        
        Args:
            inputs: 模型输入（不需要 action 字段）
            gt_action: ground truth action (B, action_dim) 或 (B, T, action_dim)
        
        Returns:
            dict with:
            - 'loss': action MSE loss
            - 'action_pred': predicted action
            - 'left_arm_loss', 'right_arm_loss', 'claw_loss': per-component losses
        """
        device = next(self.model.parameters()).device
        use_bf16 = getattr(self.model, "compute_dtype", None) == "bfloat16"
        
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            # 1. 通过 backbone
            backbone_inputs, action_inputs = self.model.prepare_input(inputs)
            backbone_outputs = self.model.backbone(backbone_inputs)
            
            # 2. 通过适配层（保留梯度！）
            backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]
            adapted_features = self.adapter(backbone_features)
            backbone_outputs[BACKBONE_FEATURE_KEY] = adapted_features
            
            # 3. 通过 action_head 获取预测 action
            # ⚠️ 注意：get_action 方法有 @torch.no_grad() 装饰器
            # 所以这个方法的 loss 无法用于反向传播！推荐使用 Flow Matching loss
            action_outputs = self.model.action_head.get_action(backbone_outputs, action_inputs, rtc_enabled=False)
            action_pred = action_outputs.get('action_pred')  # (B, T, action_dim)
            # ⚠️ action_pred 没有梯度，此方法仅用于评估，不能用于训练！
        
        # 4. 计算直接 action MSE loss
        # 确保维度匹配
        if gt_action.dim() == 2:
            # gt_action: (B, action_dim) -> 只取预测的第一个时间步
            pred_first = action_pred[:, 0, :]  # (B, action_dim)
            total_loss = F.mse_loss(pred_first, gt_action)
            
            # 分组计算 loss（假设 action 结构为 [left_arm:7, right_arm:7, left_claw:1, right_claw:1, ...]）
            # 根据 GROOT 的默认配置
            if gt_action.shape[-1] >= 16:
                left_arm_loss = F.mse_loss(pred_first[:, :7], gt_action[:, :7])
                right_arm_loss = F.mse_loss(pred_first[:, 7:14], gt_action[:, 7:14])
                claw_loss = F.mse_loss(pred_first[:, 14:16], gt_action[:, 14:16])
            else:
                left_arm_loss = right_arm_loss = claw_loss = total_loss
        else:
            # gt_action: (B, T, action_dim)
            total_loss = F.mse_loss(action_pred, gt_action)
            
            if gt_action.shape[-1] >= 16:
                left_arm_loss = F.mse_loss(action_pred[:, :, :7], gt_action[:, :, :7])
                right_arm_loss = F.mse_loss(action_pred[:, :, 7:14], gt_action[:, :, 7:14])
                claw_loss = F.mse_loss(action_pred[:, :, 14:16], gt_action[:, :, 14:16])
            else:
                left_arm_loss = right_arm_loss = claw_loss = total_loss
        
        # 5. 计算双臂协调 loss（左右臂差异的一致性）
        if gt_action.dim() == 2 and gt_action.shape[-1] >= 14:
            gt_diff = gt_action[:, :7] - gt_action[:, 7:14]
            pred_diff = pred_first[:, :7] - pred_first[:, 7:14]
            arm_coordination_loss = F.mse_loss(pred_diff, gt_diff)
        else:
            arm_coordination_loss = torch.tensor(0.0, device=device)
        
        return {
            'loss': total_loss,
            'action_pred': action_pred,
            'left_arm_loss': left_arm_loss.item() if isinstance(left_arm_loss, torch.Tensor) else left_arm_loss,
            'right_arm_loss': right_arm_loss.item() if isinstance(right_arm_loss, torch.Tensor) else right_arm_loss,
            'claw_loss': claw_loss.item() if isinstance(claw_loss, torch.Tensor) else claw_loss,
            'arm_coordination_loss': arm_coordination_loss.item() if isinstance(arm_coordination_loss, torch.Tensor) else arm_coordination_loss,
        }


class MergeVLAMerger:
    """
    MergeVLA 风格的融合器 ⭐
    
    基于论文 "MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent"
    https://arxiv.org/pdf/2511.18810
    
    核心思想：
    1. 计算 Task Vectors: τ = θ_expert - θ_base
    2. 使用稀疏激活的 LoRA 适配器对齐 backbone 分布
    3. Action head 使用 cross-attention（GROOT 已满足），可以尝试直接融合
    
    对于全量微调的模型：
    - 如果 base_model_path == narrower_path，则 τ_narrower = 0，τ_wider = wider - narrower
    - 融合公式：θ_merged = θ_base + α_narrower * τ_narrower + α_wider * τ_wider
    - 使用稀疏 LoRA 适配器处理分布漂移
    """
    
    def __init__(
        self,
        narrower_path: str,
        wider_path: str,
        base_model_path: str = None,
        narrower_weight: float = 0.5,
        wider_weight: float = 0.5,
        adapter_type: str = "sparse_lora",  # "sparse_lora" (MergeVLA), "lora", "mlp"
        device: str = "cuda:0",
        lora_rank: int = 16,
        sparsity: float = 0.5,  # 稀疏度（仅用于 sparse_lora）
        merge_action_head: bool = False,  # 是否融合 action_head（GROOT 使用 cross-attention，可以尝试）
        use_sparse_merge: bool = True,  # ⭐ 是否使用 Section 4.1 的参数级稀疏掩码融合
        sparse_merge_lambda: float = 1.0,  # ⭐ 容忍度系数 λ（论文默认 1.0）
    ):
        self.narrower_path = narrower_path
        self.wider_path = wider_path
        self.base_model_path = base_model_path or narrower_path
        self.narrower_weight = narrower_weight
        self.wider_weight = wider_weight
        self.adapter_type = adapter_type
        self.lora_rank = lora_rank
        self.sparsity = sparsity
        self.merge_action_head = merge_action_head
        self.use_sparse_merge = use_sparse_merge
        self.sparse_merge_lambda = sparse_merge_lambda
        self.device = torch.device(device)
        
        self.merged_model = None
        self.preprocessor = None
        self.postprocessor = None
    
    def load_and_merge(self):
        """
        MergeVLA 风格的融合：
        1. 计算 Task Vectors
        2. 融合权重（backbone + 可选的 action_head）
        3. 创建稀疏 LoRA 适配器
        """
        print(f"\n{'='*60}")
        print(f"🚀 MergeVLA-Style Merging")
        print(f"   基于论文: https://arxiv.org/pdf/2511.18810")
        print(f"   Narrower weight: {self.narrower_weight}")
        print(f"   Wider weight: {self.wider_weight}")
        print(f"   Adapter type: {self.adapter_type}")
        print(f"{'='*60}\n")
        
        # 加载权重
        def load_weights(path: str) -> dict:
            path = Path(path)
            safetensors_files = glob.glob(str(path / "model*.safetensors"))
            state_dict = {}
            for f in sorted(safetensors_files):
                state_dict.update(load_file(f))
            return state_dict
        
        base_state_dict = load_weights(self.base_model_path)
        narrower_state_dict = load_weights(self.narrower_path)
        wider_state_dict = load_weights(self.wider_path)
        
        print(f"✅ Loaded weights")
        print(f"   Base: {len(base_state_dict)} tensors")
        print(f"   Narrower: {len(narrower_state_dict)} tensors")
        print(f"   Wider: {len(wider_state_dict)} tensors")
        
        # 计算 Task Vectors
        print(f"\n📐 Computing Task Vectors...")
        τ_narrower = {}
        τ_wider = {}
        
        for k in base_state_dict:
            if k in narrower_state_dict:
                τ_narrower[k] = narrower_state_dict[k] - base_state_dict[k]
            if k in wider_state_dict:
                τ_wider[k] = wider_state_dict[k] - base_state_dict[k]
        
        # ⭐ MergeVLA Section 4.1: 参数级稀疏掩码融合
        # 公式: S_m = I[|τ_m| > λ|τ_merge - τ_m|]
        # 融合: Θ^(m)_merge = Θ_0 + S_m ⊙ τ_merge
        if self.use_sparse_merge:
            print(f"\n🔥 Applying MergeVLA Section 4.1 Sparse Mask Merging")
            print(f"   Formula: S_m = I[|τ_m| > λ|τ_merge - τ_m|]")
            print(f"   Lambda (tolerance): {self.sparse_merge_lambda}")
            
            # 首先计算 τ_merge（加权和）
            τ_merge = {}
            for k in base_state_dict:
                if k.startswith('backbone.') or (self.merge_action_head and k.startswith('action_head.')):
                    τ_merge_k = torch.zeros_like(base_state_dict[k])
                    if k in τ_narrower:
                        τ_merge_k = τ_merge_k + self.narrower_weight * τ_narrower[k]
                    if k in τ_wider:
                        τ_merge_k = τ_merge_k + self.wider_weight * τ_wider[k]
                    τ_merge[k] = τ_merge_k
            
            # 计算 S_narrower 和 S_wider（参数级稀疏掩码）
            S_narrower = {}
            S_wider = {}
            
            total_params = 0
            narrower_active_params = 0
            wider_active_params = 0
            shared_params = 0  # 被两个任务共同保留的参数
            selfish_params = 0  # 只被一个任务保留的参数（论文称为 "selfish"）
            
            for k in τ_merge:
                if k in τ_narrower:
                    # S_narrower = I[|τ_narrower| > λ|τ_merge - τ_narrower|]
                    abs_τ_narrower = torch.abs(τ_narrower[k])
                    abs_diff_narrower = torch.abs(τ_merge[k] - τ_narrower[k])
                    S_narrower[k] = (abs_τ_narrower > self.sparse_merge_lambda * abs_diff_narrower).float()
                else:
                    S_narrower[k] = torch.zeros_like(base_state_dict[k])
                
                if k in τ_wider:
                    # S_wider = I[|τ_wider| > λ|τ_merge - τ_wider|]
                    abs_τ_wider = torch.abs(τ_wider[k])
                    abs_diff_wider = torch.abs(τ_merge[k] - τ_wider[k])
                    S_wider[k] = (abs_τ_wider > self.sparse_merge_lambda * abs_diff_wider).float()
                else:
                    S_wider[k] = torch.zeros_like(base_state_dict[k])
                
                # 统计参数激活情况
                param_count = S_narrower[k].numel()
                total_params += param_count
                narrower_active = (S_narrower[k] > 0).sum().item()
                wider_active = (S_wider[k] > 0).sum().item()
                narrower_active_params += narrower_active
                wider_active_params += wider_active
                
                # 共同保留 vs 自私参数
                both_active = ((S_narrower[k] > 0) & (S_wider[k] > 0)).sum().item()
                only_one_active = ((S_narrower[k] > 0) ^ (S_wider[k] > 0)).sum().item()
                shared_params += both_active
                selfish_params += only_one_active
            
            print(f"\n📊 Sparse Mask Statistics:")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Narrower active: {narrower_active_params:,} ({100*narrower_active_params/total_params:.1f}%)")
            print(f"   Wider active: {wider_active_params:,} ({100*wider_active_params/total_params:.1f}%)")
            print(f"   Shared (both tasks keep): {shared_params:,} ({100*shared_params/total_params:.1f}%)")
            print(f"   Selfish (only one task keeps): {selfish_params:,} ({100*selfish_params/total_params:.1f}%)")
            print(f"   (MergeVLA论文报告约75%参数为'selfish'，表明任务掩码有效)")
        
        # 融合权重
        merged_state_dict = {}
        backbone_count = 0
        action_head_count = 0
        
        for k in base_state_dict:
            if k.startswith('backbone.'):
                # 融合 backbone
                merged = base_state_dict[k].clone()
                
                if self.use_sparse_merge and k in τ_merge:
                    # ⭐ MergeVLA Section 4.1: 使用稀疏掩码融合
                    # 为简化，我们使用两个掩码的并集（保留任一任务认为重要的参数）
                    # 也可以考虑使用加权平均的方式
                    unified_mask = torch.max(S_narrower.get(k, torch.zeros_like(merged)), 
                                             S_wider.get(k, torch.zeros_like(merged)))
                    merged = merged + unified_mask * τ_merge[k]
                else:
                    # 简单线性插值（备选方案）
                    if k in τ_narrower:
                        merged = merged + self.narrower_weight * τ_narrower[k]
                    if k in τ_wider:
                        merged = merged + self.wider_weight * τ_wider[k]
                
                merged_state_dict[k] = merged
                backbone_count += 1
            elif k.startswith('action_head.'):
                # Action head 处理
                if self.merge_action_head:
                    # 融合 action_head（GROOT 使用 cross-attention，可以尝试）
                    merged = base_state_dict[k].clone()
                    
                    if self.use_sparse_merge and k in τ_merge:
                        # ⭐ MergeVLA Section 4.1: 使用稀疏掩码融合
                        unified_mask = torch.max(S_narrower.get(k, torch.zeros_like(merged)), 
                                                 S_wider.get(k, torch.zeros_like(merged)))
                        merged = merged + unified_mask * τ_merge[k]
                    else:
                        if k in τ_narrower:
                            merged = merged + self.narrower_weight * τ_narrower[k]
                        if k in τ_wider:
                            merged = merged + self.wider_weight * τ_wider[k]
                    
                    merged_state_dict[k] = merged
                else:
                    # 使用 narrower 的 action_head（更安全）
                    merged_state_dict[k] = narrower_state_dict.get(k, base_state_dict[k])
                action_head_count += 1
            else:
                merged_state_dict[k] = base_state_dict[k]
        
        merge_method = "Sparse Mask (Section 4.1)" if self.use_sparse_merge else "Linear Interpolation"
        print(f"\n📊 Merging completed ({merge_method}):")
        print(f"   Backbone layers: {backbone_count} (merged)")
        print(f"   Action head layers: {action_head_count} ({'merged' if self.merge_action_head else 'using narrower'})")
        
        # 加载 base 模型结构
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        policy = GrootPolicy.from_pretrained(Path(self.narrower_path), strict=False)
        base_model = policy._groot_model
        
        # 获取 hidden_size
        hidden_size = None
        if hasattr(base_model, 'action_head') and hasattr(base_model.action_head, 'config'):
            hidden_size = getattr(base_model.action_head.config, 'backbone_embedding_dim', None)
        if hidden_size is None and hasattr(base_model.config, 'action_head_cfg'):
            hidden_size = base_model.config.action_head_cfg.get('backbone_embedding_dim', None)
        if hidden_size is None:
            try:
                if hasattr(base_model.backbone, 'eagle_linear'):
                    if isinstance(base_model.backbone.eagle_linear, torch.nn.Identity):
                        hidden_size = 2048
                    elif isinstance(base_model.backbone.eagle_linear, torch.nn.Linear):
                        hidden_size = base_model.backbone.eagle_linear.out_features
                    else:
                        hidden_size = 2048
                else:
                    hidden_size = 2048
            except Exception as e:
                print(f"   ⚠️ Warning: Failed to infer hidden_size: {e}")
                hidden_size = 2048
        
        print(f"   📐 Detected backbone hidden_size: {hidden_size}")
        
        # 创建带适配层的模型
        self.merged_model = MergedModelWithAdapter(
            merged_backbone_state_dict=merged_state_dict,
            narrower_action_head_state_dict=narrower_state_dict,
            base_model=base_model,
            adapter_type=self.adapter_type,
            hidden_size=hidden_size,
            lora_rank=self.lora_rank,
            num_tasks=2,  # narrower, wider
            sparsity=self.sparsity,
        ).to(self.device)
        
        # 加载预处理器
        self._load_processors(self.narrower_path)
        
        print(f"\n✅ MergeVLA merging completed, ready for adapter training")
    
    def _load_processors(self, model_path: str):
        """加载预处理器和后处理器"""
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig
        
        try:
            config = PreTrainedConfig.from_pretrained(model_path)
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=config,
                pretrained_path=model_path,
                preprocessor_overrides={
                    "device_processor": {"device": str(self.device)},
                },
            )
            print(f"   ✅ Preprocessor and postprocessor loaded")
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to load processors: {e}")
    
    def train_adapter(
        self,
        train_dataloader,
        num_epochs: int = 20,
        learning_rate: float = 1e-3,  # MergeVLA 使用较大的学习率
    ):
        """
        训练稀疏 LoRA 适配器（MergeVLA 风格）
        
        使用 action loss 来优化适配层
        """
        print(f"\n{'='*60}")
        print(f"🏋️ Training MergeVLA Adapter")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Adapter type: {self.adapter_type}")
        print(f"{'='*60}\n")
        
        optimizer = torch.optim.AdamW(
            self.merged_model.adapter.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )
        
        # MergeVLA 使用固定学习率
        scheduler = None
        
        self.merged_model.train()
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                # 准备输入
                observation = self._prepare_observation(batch)
                
                # 获取任务ID（如果可用）
                task_id = None
                if 'task_source' in batch:
                    task_source = batch['task_source']
                    if isinstance(task_source, torch.Tensor):
                        task_id = task_source.to(self.device)
                    elif isinstance(task_source, (list, tuple)):
                        task_id = torch.tensor(task_source[0], device=self.device).unsqueeze(0)
                
                # 使用预处理器处理输入
                if self.preprocessor is not None:
                    try:
                        inputs = self.preprocessor(observation)
                        inputs = self._to_device(inputs)
                    except Exception as e:
                        if batch_idx == 0:
                            print(f"   ⚠️ Preprocessor failed: {e}")
                        continue
                else:
                    inputs = self._to_device(observation)
                
                try:
                    # 前向传播（使用 Flow Matching loss）
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        outputs = self.merged_model(inputs, task_id=task_id)
                    
                    if hasattr(outputs, 'data'):
                        outputs_dict = outputs.data
                    elif isinstance(outputs, dict):
                        outputs_dict = outputs
                    else:
                        continue
                    
                    if 'loss' not in outputs_dict:
                        continue
                    
                    loss = outputs_dict['loss']
                    
                except Exception as e:
                    if batch_idx == 0:
                        print(f"   ⚠️ Forward failed: {e}")
                        import traceback
                        traceback.print_exc()
                    continue
                
                # 检查 NaN
                if not torch.isfinite(loss):
                    if batch_idx < 3:
                        print(f"   ⚠️ Non-finite loss: {loss.item()}")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.merged_model.adapter.parameters(),
                    max_norm=1.0,
                )
                
                optimizer.step()
                epoch_losses.append(loss.item())
                
                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"   Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}: "
                          f"loss = {loss.item():.4f}, grad_norm = {grad_norm:.4f}, lr = {current_lr:.2e}")
            
            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                print(f"\n📊 Epoch {epoch+1}/{num_epochs}: avg_loss = {avg_loss:.4f}")
        
        print(f"\n✅ Adapter training completed")
    
    def _prepare_observation(self, batch: dict) -> dict:
        """将 LeRobot batch 转换为预处理器期望的格式"""
        result = {}
        
        for key, value in batch.items():
            if key.startswith('observation.'):
                if isinstance(value, torch.Tensor):
                    result[key] = value.to(self.device)
                else:
                    result[key] = value
        
        if 'action' in batch:
            action = batch['action']
            if isinstance(action, torch.Tensor):
                result['action'] = action.to(self.device)
        
        if 'task' in batch:
            task_value = batch['task']
            if isinstance(task_value, (list, tuple)) and len(task_value) > 0:
                result['task'] = task_value[0]
            elif isinstance(task_value, str):
                result['task'] = task_value
            else:
                result['task'] = "Depalletize the box"
        else:
            result['task'] = "Depalletize the box"
        
        return result
    
    def _to_device(self, inputs: dict) -> dict:
        """将输入移动到设备"""
        result = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = self._to_device(value)
            else:
                result[key] = value
        return result
    
    def save(self, output_path: str):
        """
        保存融合后的模型（包含适配层）
        
        ⚠️ 关键修复：
        1. 权重键名必须添加 `_groot_model.` 前缀（GrootPolicy.from_pretrained 期望的格式）
        2. config.json 中的 `base_model_path` 必须指向本地路径，而不是 HuggingFace
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取完整的 state_dict
        model_state_dict = self.merged_model.model.state_dict()
        
        # ⚠️ 关键修复：添加 `_groot_model.` 前缀
        # GrootPolicy.from_pretrained 期望的键名格式是 `_groot_model.xxx`
        # 而 GR00TN15.state_dict() 返回的是 `backbone.xxx` 和 `action_head.xxx`
        state_dict = {}
        for k, v in model_state_dict.items():
            # 添加 _groot_model. 前缀
            new_key = f"_groot_model.{k}"
            state_dict[new_key] = v
        
        # 添加适配层权重（也需要前缀）
        adapter_state = self.merged_model.adapter.state_dict()
        for k, v in adapter_state.items():
            state_dict[f"_groot_model.distribution_adapter.{k}"] = v
        
        # 克隆以处理共享内存
        state_dict_cloned = {k: v.clone().contiguous() for k, v in state_dict.items()}
        
        print(f"\n📊 保存的权重统计:")
        print(f"   总键数: {len(state_dict_cloned)}")
        print(f"   前5个键: {list(state_dict_cloned.keys())[:5]}")
        
        save_file(state_dict_cloned, str(output_path / "model.safetensors"))
        
        # ⚠️ 关键修复：修改 config.json 中的 base_model_path
        # 让它指向 narrower 模型的本地路径，而不是 HuggingFace
        import shutil
        narrower_path = Path(self.narrower_path)
        
        # 复制并修改 config.json
        config_src = narrower_path / "config.json"
        if config_src.exists():
            with open(config_src, 'r') as f:
                config = json.load(f)
            
            # ⚠️ 关键：修改 base_model_path 指向本地 narrower 路径
            # 这样加载时会使用 narrower 的模型结构（包括正确的 num_target_vision_tokens）
            config['base_model_path'] = str(narrower_path.resolve())
            
            print(f"\n⚠️ 修改 config.json:")
            print(f"   base_model_path: {config['base_model_path']}")
            
            with open(output_path / "config.json", 'w') as f:
                json.dump(config, f, indent=4)
        
        # 复制其他配置文件
        for config_file in ["policy_preprocessor.json", "policy_postprocessor.json"]:
            src = narrower_path / config_file
            if src.exists():
                shutil.copy(src, output_path / config_file)
        
        for pattern in ["policy_preprocessor*.safetensors", "policy_postprocessor*.safetensors"]:
            for src in narrower_path.glob(pattern):
                shutil.copy(src, output_path / src.name)
        
        # 保存融合配置
        merge_config = {
            "merge_method": "mergevla",
            "narrower_path": str(self.narrower_path),
            "wider_path": str(self.wider_path),
            "base_model_path": str(self.base_model_path),
            "narrower_weight": self.narrower_weight,
            "wider_weight": self.wider_weight,
            "adapter_type": self.adapter_type,
            "lora_rank": self.lora_rank,
            "sparsity": self.sparsity,
            "num_tasks": 2,
            "merge_action_head": self.merge_action_head,
            "action_head_source": "merged" if self.merge_action_head else "narrower",
            # ⭐ MergeVLA Section 4.1: 参数级稀疏掩码融合配置
            "use_sparse_merge": self.use_sparse_merge,
            "sparse_merge_lambda": self.sparse_merge_lambda,
        }
        with open(output_path / "merge_config.json", "w") as f:
            json.dump(merge_config, f, indent=2)
        
        print(f"\n✅ MergeVLA model saved to {output_path}")
        print(f"   - model.safetensors (包含 _groot_model.* 和 distribution_adapter 权重)")
        print(f"   - config.json (base_model_path 已修改为本地路径)")
        print(f"   - merge_config.json")


class TwoStageExpertMerger:
    """
    两阶段融合器（推荐方法）⭐
    
    基于 kai0 Model Arithmetic 的思想：
    https://mmlab.hk/research/kai0
    
    阶段 1: 融合 backbone（使用 Expert Merging 或 Task Arithmetic）
    阶段 2: 训练分布适配层（使用 action loss）
    
    这种方法解决了：
    1. action_head (DiT) 对参数敏感不能直接融合的问题
    2. 只融合 backbone 导致输入分布漂移的问题
    """
    
    def __init__(
        self,
        narrower_path: str,
        wider_path: str,
        base_model_path: str = None,
        merge_method: str = "interpolation",  # "interpolation", "task_arithmetic", "expert_merge"
        alpha: float = 0.5,  # 融合系数（仅用于 interpolation）
        adapter_type: str = "lora",  # "linear", "mlp", "lora", "layernorm_only"
        device: str = "cuda:0",
        lora_rank: int = 16,  # LoRA rank（仅用于 lora 类型）
    ):
        self.narrower_path = narrower_path
        self.wider_path = wider_path
        self.base_model_path = base_model_path or narrower_path
        self.merge_method = merge_method
        self.alpha = alpha
        self.adapter_type = adapter_type
        self.lora_rank = lora_rank
        self.device = torch.device(device)
        
        self.merged_model = None
        self.preprocessor = None
        self.postprocessor = None
    
    def load_and_merge(self):
        """
        阶段 1：加载模型并融合 backbone
        """
        print(f"\n{'='*60}")
        print(f"🚀 Two-Stage Expert Merging (kai0 style)")
        print(f"   Stage 1: Merge backbone ({self.merge_method})")
        print(f"   Stage 2: Train distribution adapter")
        print(f"{'='*60}\n")
        
        # 加载权重
        def load_weights(path: str) -> dict:
            path = Path(path)
            safetensors_files = glob.glob(str(path / "model*.safetensors"))
            state_dict = {}
            for f in sorted(safetensors_files):
                state_dict.update(load_file(f))
            return state_dict
        
        narrower_state_dict = load_weights(self.narrower_path)
        wider_state_dict = load_weights(self.wider_path)
        
        print(f"✅ Loaded weights")
        print(f"   Narrower: {len(narrower_state_dict)} tensors")
        print(f"   Wider: {len(wider_state_dict)} tensors")
        
        # 融合 backbone
        merged_state_dict = {}
        backbone_count = 0
        action_head_count = 0
        
        for k in narrower_state_dict:
            if k.startswith('backbone.'):
                # 融合 backbone
                if k in wider_state_dict:
                    merged_state_dict[k] = (
                        self.alpha * narrower_state_dict[k] + 
                        (1 - self.alpha) * wider_state_dict[k]
                    )
                else:
                    merged_state_dict[k] = narrower_state_dict[k]
                backbone_count += 1
            elif k.startswith('action_head.'):
                # 保留 narrower 的 action_head
                merged_state_dict[k] = narrower_state_dict[k]
                action_head_count += 1
            else:
                merged_state_dict[k] = narrower_state_dict[k]
        
        print(f"\n📊 Stage 1 完成: Backbone 融合")
        print(f"   Backbone 层: {backbone_count} (融合比例 {self.alpha:.1%} narrower + {1-self.alpha:.1%} wider)")
        print(f"   Action head 层: {action_head_count} (使用 narrower)")
        print(f"\n   ⚠️ 警告：简单插值融合 backbone 可能导致分布漂移！")
        print(f"      如果推理时动作震荡，说明融合后的 backbone 输出分布与")
        print(f"      narrower 的 action_head 期望的分布不匹配。")
        print(f"      建议：使用 Expert Merging 方法（更可靠）")
        
        # 加载 base 模型结构
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        policy = GrootPolicy.from_pretrained(Path(self.narrower_path), strict=False)
        base_model = policy._groot_model
        
        # 获取 hidden_size（backbone 输出维度）
        hidden_size = None
        
        # 方法1: 从 action_head 配置中获取 backbone_embedding_dim
        if hasattr(base_model, 'action_head') and hasattr(base_model.action_head, 'config'):
            hidden_size = getattr(base_model.action_head.config, 'backbone_embedding_dim', None)
        
        # 方法2: 如果方法1失败，从 action_head_cfg 中获取
        if hidden_size is None and hasattr(base_model.config, 'action_head_cfg'):
            hidden_size = base_model.config.action_head_cfg.get('backbone_embedding_dim', None)
        
        # 方法3: 如果还是 None，尝试从实际的 backbone 输出推断
        if hidden_size is None:
            # 创建一个虚拟输入来推断维度
            try:
                # 检查 backbone 是否有 project_to_dim 或输出维度信息
                if hasattr(base_model.backbone, 'eagle_linear'):
                    if isinstance(base_model.backbone.eagle_linear, torch.nn.Identity):
                        # Identity 表示不投影，输出是 2048
                        hidden_size = 2048
                    elif isinstance(base_model.backbone.eagle_linear, torch.nn.Linear):
                        # Linear 投影层，输出维度是 out_features
                        hidden_size = base_model.backbone.eagle_linear.out_features
                    else:
                        hidden_size = 2048  # 默认值
                else:
                    hidden_size = 2048  # GROOT N1.5 默认 backbone 输出维度
            except Exception as e:
                print(f"   ⚠️ Warning: Failed to infer hidden_size: {e}")
                hidden_size = 2048  # 默认值
        
        print(f"   📐 Detected backbone hidden_size: {hidden_size}")
        
        # 创建带适配层的模型
        self.merged_model = MergedModelWithAdapter(
            merged_backbone_state_dict=merged_state_dict,
            narrower_action_head_state_dict=narrower_state_dict,
            base_model=base_model,
            adapter_type=self.adapter_type,
            hidden_size=hidden_size,
            lora_rank=self.lora_rank,
            num_tasks=getattr(self, 'num_tasks', 2),  # 默认2个任务
            sparsity=getattr(self, 'sparsity', 0.5),  # 默认稀疏度
        ).to(self.device)
        
        # 加载预处理器
        self._load_processors(self.narrower_path)
        
        print(f"\n✅ Stage 1 完成，准备进行 Stage 2: 训练适配层")
    
    def _load_processors(self, model_path: str):
        """加载预处理器和后处理器"""
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig
        
        try:
            config = PreTrainedConfig.from_pretrained(model_path)
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=config,
                pretrained_path=model_path,
                preprocessor_overrides={
                    "device_processor": {"device": str(self.device)},
                },
            )
            print(f"   ✅ Preprocessor and postprocessor loaded")
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to load processors: {e}")
    
    def train_adapter(
        self,
        train_dataloader,
        num_epochs: int = 20,
        learning_rate: float = 1e-4,
        use_direct_loss: bool = False,  # ⚠️ 修改：默认使用 Flow Matching loss
    ):
        """
        阶段 2：训练分布适配层
        
        ⚠️ 重要说明：
        - 默认使用 Flow Matching loss（use_direct_loss=False）
        - 因为 get_action 方法有 @torch.no_grad() 装饰器，直接 action loss 无法反向传播
        - Flow Matching loss 是训练 action head 的标准方式，有完整的梯度支持
        
        基于 kai0 Model Arithmetic 的思想：
        使用 action loss 来优化适配层
        """
        loss_type = "Direct Action MSE" if use_direct_loss else "Flow Matching"
        
        # ⚠️ 警告：Direct Action MSE 由于 @torch.no_grad() 装饰器无法工作
        if use_direct_loss:
            print(f"\n⚠️ 警告：Direct Action MSE loss 由于 get_action 方法的 @torch.no_grad() 装饰器无法反向传播！")
            print(f"   自动切换到 Flow Matching loss...")
            use_direct_loss = False
            loss_type = "Flow Matching"
        
        print(f"\n{'='*60}")
        print(f"🏋️ Stage 2: Training Distribution Adapter")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Loss type: {loss_type}")
        print(f"   Optimizer: AdamW")
        print(f"{'='*60}\n")
        
        # ⚠️ 关键改进：使用更大的学习率（Flow Matching loss 梯度弱）
        # 默认学习率 1e-4 太小，建议使用 1e-3 或更大
        effective_lr = learning_rate
        if learning_rate <= 1e-4:
            print(f"   ⚠️ Warning: Learning rate {learning_rate} may be too small!")
            print(f"      Recommended: 1e-3 or larger for adapter training")
            print(f"      Flow Matching loss has weak gradients, need larger LR")
        
        optimizer = torch.optim.AdamW(
            self.merged_model.adapter.parameters(),
            lr=effective_lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )
        
        # ⚠️ 关键修复：使用固定学习率或非常缓慢的衰减
        # 问题：CosineAnnealing 会让学习率降得太低（最后降到 1e-6），导致适配层无法学习
        # 解决方案：使用固定学习率，或者非常缓慢的线性衰减
        use_constant_lr = True  # 改为 True 使用固定学习率
        
        if use_constant_lr:
            # 使用固定学习率（推荐！）
            print(f"   ⚠️ Using CONSTANT learning rate (no decay)")
            print(f"      This is critical for adapter training!")
            print(f"      CosineAnnealing was causing LR to drop too low (1e-6)")
            print(f"      which prevented the adapter from learning!")
            scheduler = None  # 不使用 scheduler
        else:
            # 使用非常缓慢的线性衰减（如果必须使用衰减）
            warmup_epochs = max(1, num_epochs // 10)  # 10% warmup
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs  # Linear warmup
                else:
                    # 非常缓慢的线性衰减：从 1.0 降到 0.5（而不是 0）
                    progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                    return 1.0 - 0.5 * progress  # 线性衰减到 0.5
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        self.merged_model.train()
        
        # 统计
        total_batches = 0
        successful_batches = 0
        
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_component_losses = {'left_arm': [], 'right_arm': [], 'claw': [], 'coord': []}
            
            for batch_idx, batch in enumerate(train_dataloader):
                total_batches += 1
                optimizer.zero_grad()
                
                # 准备输入
                observation = self._prepare_observation(batch)
                
                # 获取 ground truth action（在预处理之前）
                gt_action = batch.get('action')
                if gt_action is not None and isinstance(gt_action, torch.Tensor):
                    gt_action = gt_action.to(self.device)
                
                # 使用预处理器处理输入
                if self.preprocessor is not None:
                    try:
                        inputs = self.preprocessor(observation)
                        inputs = self._to_device(inputs)
                    except Exception as e:
                        if batch_idx == 0:
                            print(f"   ⚠️ Preprocessor failed: {e}")
                            import traceback
                            traceback.print_exc()
                        continue
                else:
                    inputs = self._to_device(observation)
                
                # 检查输入是否包含必要的字段
                if batch_idx == 0 and epoch == 0:
                    print(f"\n   📋 Input keys: {list(inputs.keys())}")
                    if 'action' in inputs:
                        action_shape = inputs['action'].shape if hasattr(inputs['action'], 'shape') else type(inputs['action'])
                        print(f"   📋 Action shape: {action_shape}")
                    if gt_action is not None:
                        print(f"   📋 GT action shape: {gt_action.shape}")
                    if 'action_mask' in inputs:
                        mask_shape = inputs['action_mask'].shape if hasattr(inputs['action_mask'], 'shape') else type(inputs['action_mask'])
                        print(f"   📋 Action mask shape: {mask_shape}")
                
                try:
                    if use_direct_loss and gt_action is not None:
                        # ⭐ 使用直接 action loss（推荐）
                        outputs_dict = self.merged_model.forward_with_direct_loss(inputs, gt_action)
                        loss = outputs_dict['loss']
                        
                        # 记录组件 loss
                        epoch_component_losses['left_arm'].append(outputs_dict.get('left_arm_loss', 0))
                        epoch_component_losses['right_arm'].append(outputs_dict.get('right_arm_loss', 0))
                        epoch_component_losses['claw'].append(outputs_dict.get('claw_loss', 0))
                        epoch_component_losses['coord'].append(outputs_dict.get('arm_coordination_loss', 0))
                    else:
                        # 使用 Flow Matching loss（旧方法）
                        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                            outputs = self.merged_model(inputs)
                        
                        if hasattr(outputs, 'data'):
                            outputs_dict = outputs.data
                        elif isinstance(outputs, dict):
                            outputs_dict = outputs
                        else:
                            continue
                        
                        if 'loss' in outputs_dict:
                            loss = outputs_dict['loss']
                        else:
                            continue
                    
                except Exception as e:
                    if batch_idx == 0:
                        print(f"   ⚠️ Forward failed: {e}")
                        import traceback
                        traceback.print_exc()
                    continue
                
                # 首次打印详细 loss
                if batch_idx == 0 and epoch == 0:
                    print(f"\n   📊 Loss components:")
                    for k, v in outputs_dict.items():
                        if 'loss' in k.lower() and k != 'loss':
                            if isinstance(v, (int, float)):
                                print(f"      {k}: {v:.4f}")
                
                # 检查 NaN
                if not torch.isfinite(loss):
                    if batch_idx < 3:
                        print(f"   ⚠️ Non-finite loss: {loss.item()}")
                    continue
                
                # ⚠️ 关键：在反向传播之前，确保适配层参数与 loss 的 dtype 匹配
                # 如果 loss 是 bfloat16，适配层参数也应该是 bfloat16（在 autocast 中）
                # 但适配层参数默认是 float32，这可能导致问题
                # 实际上，PyTorch 的自动混合精度会处理这个问题，但我们需要确保梯度能正确传播
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.merged_model.adapter.parameters(),
                    max_norm=1.0,
                )
                
                # ⚠️ 关键诊断：详细检查梯度传播
                if batch_idx < 3 and epoch == 0:
                    print(f"\n   🔍 详细梯度诊断 (Batch {batch_idx}):")
                    
                    # 1. 检查 loss 是否有梯度
                    if hasattr(loss, 'grad_fn') and loss.grad_fn is not None:
                        print(f"      ✅ Loss has grad_fn: {type(loss.grad_fn).__name__}")
                    else:
                        print(f"      ⚠️ CRITICAL: Loss has NO grad_fn! Cannot backpropagate!")
                    
                    # 2. 检查适配层输入是否有梯度
                    # 注意：backbone 是冻结的，所以 backbone_features 不需要梯度（这是正常的）
                    # 但我们需要检查适配层是否能正常工作
                    with torch.no_grad():
                        backbone_inputs, _ = self.merged_model.model.prepare_input(inputs)
                        backbone_outputs = self.merged_model.model.backbone(backbone_inputs)
                        backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]
                    
                    print(f"      Backbone features: shape={backbone_features.shape}, dtype={backbone_features.dtype}")
                    print(f"      Backbone features require grad: {backbone_features.requires_grad} (expected: False, backbone is frozen)")
                    
                    # 3. 检查适配层输出是否有梯度
                    # 适配层是可训练的，所以输出应该有梯度
                    adapted_features = self.merged_model.adapter(backbone_features)
                    if adapted_features.requires_grad:
                        print(f"      ✅ Adapted features require grad (adapter is trainable)")
                    else:
                        print(f"      ⚠️ CRITICAL: Adapted features do NOT require grad!")
                        print(f"         This means the adapter computation graph is broken!")
                    
                    # 4. 检查适配层参数的梯度
                    grad_sum = 0
                    grad_max = 0
                    param_count = 0
                    has_any_grad = False
                    
                    for name, p in self.merged_model.adapter.named_parameters():
                        if p.grad is not None:
                            has_any_grad = True
                            grad_sum += p.grad.abs().sum().item()
                            grad_max = max(grad_max, p.grad.abs().max().item())
                            param_count += p.numel()
                            print(f"      ✅ {name}: grad_norm={p.grad.norm().item():.6f}, "
                                  f"grad_mean={p.grad.abs().mean().item():.6f}")
                        else:
                            print(f"      ⚠️ {name}: NO gradient!")
                    
                    if has_any_grad:
                        grad_mean = grad_sum / param_count if param_count > 0 else 0
                        print(f"      📊 Overall: grad_mean={grad_mean:.6f}, grad_max={grad_max:.6f}")
                        if grad_mean < 1e-6:
                            print(f"      ⚠️ CRITICAL: Gradients are extremely small!")
                            print(f"         This means the adapter cannot learn effectively!")
                            print(f"         Possible causes:")
                            print(f"         1. Flow Matching loss gradients are too weak")
                            print(f"         2. Computation graph is broken")
                            print(f"         3. Need to use Expert Merging instead")
                    else:
                        print(f"      ⚠️ CRITICAL: Adapter has NO gradients at all!")
                        print(f"         The computation graph is broken!")
                        print(f"         Recommendation: Use Expert Merging method instead")
                    
                    # 5. 检查 action_head 输出是否有梯度
                    if hasattr(outputs_dict, 'get'):
                        loss_tensor = outputs_dict.get('loss')
                        if loss_tensor is not None and hasattr(loss_tensor, 'requires_grad'):
                            if loss_tensor.requires_grad:
                                print(f"      ✅ Loss tensor requires grad")
                            else:
                                print(f"      ⚠️ Loss tensor does NOT require grad!")
                    
                    print()  # 空行
                
                optimizer.step()
                epoch_losses.append(loss.item())
                successful_batches += 1
                
                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"   Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}: "
                          f"loss = {loss.item():.4f}, grad_norm = {grad_norm:.4f}, lr = {current_lr:.2e}")
            
            if scheduler is not None:
                scheduler.step()
            
            # ⚠️ 关键诊断：每个 epoch 检查适配层权重是否在学习
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                with torch.no_grad():
                    print(f"\n   🔍 Adapter learning check (Epoch {epoch+1}):")
                    
                    # 检查适配层类型
                    if self.adapter_type == "lora":
                        # LoRA 适配层
                        if hasattr(self.merged_model.adapter, 'adapter') and hasattr(self.merged_model.adapter.adapter, 'lora_A'):
                            lora_A = self.merged_model.adapter.adapter.lora_A
                            lora_B = self.merged_model.adapter.adapter.lora_B
                            lora_A_abs_mean = lora_A.abs().mean().item()
                            lora_B_abs_mean = lora_B.abs().mean().item()
                            
                            print(f"      LoRA A weight abs mean: {lora_A_abs_mean:.6f}")
                            print(f"      LoRA B weight abs mean: {lora_B_abs_mean:.6f}")
                            
                            if lora_A_abs_mean < 0.001 or lora_B_abs_mean < 0.001:
                                print(f"      ⚠️ CRITICAL: LoRA adapter is NOT learning!")
                                print(f"         Current LR: {optimizer.param_groups[0]['lr']:.2e}")
                                print(f"         Recommendation: Try Expert Merging method instead!")
                            elif lora_A_abs_mean < 0.01 or lora_B_abs_mean < 0.01:
                                print(f"      ⚠️ Warning: LoRA adapter learning is weak")
                            else:
                                print(f"      ✅ LoRA adapter is learning (weights are changing)")
                    else:
                        # MLP 或其他适配层
                        if hasattr(self.merged_model.adapter, 'adapter') and len(self.merged_model.adapter.adapter) > 0:
                            adapter_0_weight = self.merged_model.adapter.adapter[0].weight
                            adapter_0_abs_mean = adapter_0_weight.abs().mean().item()
                            
                            if len(self.merged_model.adapter.adapter) > 3:
                                adapter_3_weight = self.merged_model.adapter.adapter[3].weight
                                adapter_3_abs_mean = adapter_3_weight.abs().mean().item()
                                print(f"      Layer 0 weight abs mean: {adapter_0_abs_mean:.6f}")
                                print(f"      Layer 3 weight abs mean: {adapter_3_abs_mean:.6f}")
                            else:
                                print(f"      Layer 0 weight abs mean: {adapter_0_abs_mean:.6f}")
                            
                            if adapter_0_abs_mean < 0.01:
                                print(f"      ⚠️ CRITICAL: Adapter is NOT learning! Weights are almost zero!")
                                print(f"         Current LR: {optimizer.param_groups[0]['lr']:.2e}")
                                print(f"         Recommendation: Try LoRA adapter or Expert Merging!")
                            elif adapter_0_abs_mean < 0.05:
                                print(f"      ⚠️ Warning: Adapter learning is weak")
                            else:
                                print(f"      ✅ Adapter is learning (weights are changing)")
            
            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                print(f"\n📊 Epoch {epoch+1}/{num_epochs}: avg_loss = {avg_loss:.4f}")
                
                # 打印组件 loss 平均值
                if use_direct_loss and epoch_component_losses['left_arm']:
                    print(f"   Component losses:")
                    print(f"      left_arm:  {sum(epoch_component_losses['left_arm'])/len(epoch_component_losses['left_arm']):.4f}")
                    print(f"      right_arm: {sum(epoch_component_losses['right_arm'])/len(epoch_component_losses['right_arm']):.4f}")
                    print(f"      claw:      {sum(epoch_component_losses['claw'])/len(epoch_component_losses['claw']):.4f}")
                    print(f"      coord:     {sum(epoch_component_losses['coord'])/len(epoch_component_losses['coord']):.4f}")
            else:
                print(f"\n⚠️ Epoch {epoch+1}/{num_epochs}: No successful batches!")
        
        print(f"\n✅ Stage 2 完成: 适配层训练完成")
        print(f"   成功批次: {successful_batches}/{total_batches}")
        
        # 打印适配层参数变化
        self._print_adapter_stats(optimizer)
        
        if successful_batches == 0:
            print(f"\n⚠️ 警告：没有成功训练的批次！")
            print(f"   可能的原因：")
            print(f"   1. 数据格式不正确")
            print(f"   2. 预处理器配置问题")
            print(f"   3. 模型配置不匹配")
    
    def _print_adapter_stats(self, optimizer=None):
        """打印适配层参数统计"""
        import numpy as np
        print(f"\n📊 Adapter parameter stats after training:")
        
        if optimizer is not None:
            final_lr = optimizer.param_groups[0]['lr']
            print(f"   Final learning rate: {final_lr:.2e}")
        
        adapter_trained = False
        
        for name, param in self.merged_model.adapter.named_parameters():
            p_np = param.detach().cpu().numpy()
            print(f"   {name}:")
            print(f"      Mean: {p_np.mean():.6f}, Std: {p_np.std():.6f}")
            
            # 检查 LoRA 参数（如果使用 LoRA 适配层）
            if 'lora_A' in name or 'lora_B' in name:
                weight_abs_mean = np.abs(p_np).mean()
                weight_std = p_np.std()
                print(f"      Weight abs mean: {weight_abs_mean:.6f}, Std: {weight_std:.6f}")
                
                if weight_abs_mean < 0.001:
                    print(f"      ⚠️ CRITICAL: LoRA weights are almost zero!")
                    print(f"         This means the adapter is NOT learning!")
                elif weight_abs_mean < 0.01:
                    print(f"      ⚠️ Warning: LoRA weights are very small!")
                else:
                    print(f"      ✅ LoRA adapter has learned meaningful weights")
                    adapter_trained = True
            
            # 检查 Linear weight 是否远离初始值（对于 MLP，第一层是 hidden_size -> hidden_size*2）
            if 'adapter.0.weight' in name and len(p_np.shape) == 2:
                # 对于 MLP adapter，第一层是 (hidden_size*2, hidden_size)
                # 检查权重是否远离零（初始化为 Xavier，gain=0.1，所以初始值应该很小但非零）
                weight_abs_mean = np.abs(p_np).mean()
                weight_std = p_np.std()
                print(f"      Weight abs mean: {weight_abs_mean:.6f}, Std: {weight_std:.6f}")
                
                # 如果权重几乎为零，说明没有学习
                if weight_abs_mean < 0.01:
                    print(f"      ⚠️ CRITICAL: Adapter layer 0 weights are almost zero!")
                    print(f"         This means the adapter is NOT learning!")
                    print(f"         The model will collapse during inference!")
                    print(f"      Possible reasons:")
                    print(f"        1. Learning rate too small (need 1e-2 or larger)")
                    print(f"        2. Flow Matching loss gradients too weak")
                    print(f"        3. Learning rate decay too aggressive (use constant LR)")
                    print(f"        4. Not enough training epochs or data")
                elif weight_abs_mean < 0.05:
                    print(f"      ⚠️ Warning: Adapter layer 0 weights are very small!")
                    print(f"         Adapter may not be learning effectively")
                else:
                    print(f"      ✅ Adapter layer 0 has learned meaningful weights")
                    adapter_trained = True
            
            # 检查中间层（adapter.3，第二层 Linear）
            if 'adapter.3.weight' in name and len(p_np.shape) == 2:
                weight_abs_mean = np.abs(p_np).mean()
                weight_std = p_np.std()
                print(f"      Weight abs mean: {weight_abs_mean:.6f}, Std: {weight_std:.6f}")
                
                if weight_abs_mean < 0.01:
                    print(f"      ⚠️ CRITICAL: Adapter layer 3 weights are almost zero!")
                    print(f"         This means the adapter is NOT learning!")
                elif weight_abs_mean < 0.05:
                    print(f"      ⚠️ Warning: Adapter layer 3 weights are very small!")
                else:
                    print(f"      ✅ Adapter layer 3 has learned meaningful weights")
                    adapter_trained = True
            
            # 检查 residual_scale
            if 'residual_scale' in name:
                scale_val = p_np.item()
                print(f"      Residual scale: {scale_val:.6f}")
                if abs(scale_val - 1.0) < 0.01:
                    print(f"      ⚠️ Residual scale unchanged! Adapter may not be learning")
                else:
                    print(f"      ✅ Residual scale adjusted")
                    adapter_trained = True
        
        if not adapter_trained:
            print(f"\n⚠️ 警告：适配层可能没有被训练！")
            print(f"   建议：")
            print(f"   1. 使用 LoRA adapter: --adapter_type lora --lora_rank 32")
            print(f"   2. 增大学习率: --adapter_lr 1e-2")
            print(f"   3. 增加训练轮数: --adapter_epochs 100")
            print(f"   4. 增加样本数: --num_samples 500")
            print(f"   5. ⭐ 推荐：使用 Expert Merging 方法（不依赖适配层）:")
            print(f"      ./merge_groot_models.sh expert_merge")
            print(f"   6. 或者使用 Task Arithmetic（无需训练）:")
            print(f"      ./merge_groot_models.sh task_arithmetic")
    
    def _prepare_observation(self, batch: dict) -> dict:
        """
        将 LeRobot batch 转换为预处理器期望的格式
        
        ⚠️ 关键理解：
        预处理器 pipeline 使用 batch_to_transition 将输入转换为 EnvTransition 格式，
        然后使用 transition_to_batch 将处理后的数据转换回 batch 格式。
        
        所以我们应该提供扁平的 batch 格式：
        {
            'observation.state': (B, state_dim),
            'observation.images.cam_head': (B, C, H, W),
            'action': (B, action_dim),  # 预处理器会扩展为 (B, T, max_action_dim)
            'task': str,  # 会被放入 complementary_data
        }
        
        预处理器会：
        1. 将图像转换为 video 格式并编码为 Eagle 特征
        2. 将 state 归一化并填充
        3. 将单帧 action 扩展为 action_horizon 长度并创建 action_mask
        4. 添加 embodiment_id
        """
        result = {}
        
        # 1. 复制所有观测数据（保持 observation.* 格式）
        for key, value in batch.items():
            if key.startswith('observation.'):
                if isinstance(value, torch.Tensor):
                    result[key] = value.to(self.device)
                else:
                    result[key] = value
        
        # 2. 处理 action
        if 'action' in batch:
            action = batch['action']
            if isinstance(action, torch.Tensor):
                result['action'] = action.to(self.device)
        
        # 3. 处理 task
        if 'task' in batch:
            task_value = batch['task']
            if isinstance(task_value, (list, tuple)) and len(task_value) > 0:
                result['task'] = task_value[0]
            elif isinstance(task_value, str):
                result['task'] = task_value
            else:
                result['task'] = "Depalletize the box"
        else:
            result['task'] = "Depalletize the box"
        
        return result
    
    def _to_device(self, inputs: dict) -> dict:
        """将输入移动到设备"""
        result = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = self._to_device(value)
            else:
                result[key] = value
        return result
    
    def save(self, output_path: str):
        """
        保存融合后的模型（包含适配层）
        
        ⚠️ 关键修复：
        1. 权重键名必须添加 `_groot_model.` 前缀（GrootPolicy.from_pretrained 期望的格式）
        2. config.json 中的 `base_model_path` 必须指向本地路径，而不是 HuggingFace
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取完整的 state_dict
        model_state_dict = self.merged_model.model.state_dict()
        
        # ⚠️ 关键修复：添加 `_groot_model.` 前缀
        state_dict = {}
        for k, v in model_state_dict.items():
            new_key = f"_groot_model.{k}"
            state_dict[new_key] = v
        
        # 添加适配层权重（也需要前缀）
        adapter_state = self.merged_model.adapter.state_dict()
        for k, v in adapter_state.items():
            state_dict[f"_groot_model.distribution_adapter.{k}"] = v
        
        # 克隆以处理共享内存
        state_dict_cloned = {k: v.clone().contiguous() for k, v in state_dict.items()}
        
        print(f"\n📊 保存的权重统计:")
        print(f"   总键数: {len(state_dict_cloned)}")
        print(f"   前5个键: {list(state_dict_cloned.keys())[:5]}")
        
        save_file(state_dict_cloned, str(output_path / "model.safetensors"))
        
        # ⚠️ 关键修复：修改 config.json 中的 base_model_path
        import shutil
        narrower_path = Path(self.narrower_path)
        
        # 复制并修改 config.json
        config_src = narrower_path / "config.json"
        if config_src.exists():
            with open(config_src, 'r') as f:
                config = json.load(f)
            
            # 修改 base_model_path 指向本地 narrower 路径
            config['base_model_path'] = str(narrower_path.resolve())
            
            print(f"\n⚠️ 修改 config.json:")
            print(f"   base_model_path: {config['base_model_path']}")
            
            with open(output_path / "config.json", 'w') as f:
                json.dump(config, f, indent=4)
        
        # 复制其他配置文件
        for config_file in ["policy_preprocessor.json", "policy_postprocessor.json"]:
            src = narrower_path / config_file
            if src.exists():
                shutil.copy(src, output_path / config_file)
        
        for pattern in ["policy_preprocessor*.safetensors", "policy_postprocessor*.safetensors"]:
            for src in narrower_path.glob(pattern):
                shutil.copy(src, output_path / src.name)
        
        # 保存融合配置
        merge_config = {
            "merge_method": "two_stage_adapter",
            "backbone_merge_method": self.merge_method,
            "backbone_alpha": self.alpha,
            "adapter_type": self.adapter_type,
            "lora_rank": self.lora_rank,
            "narrower_path": str(self.narrower_path),
            "wider_path": str(self.wider_path),
            "action_head_source": "narrower",
        }
        with open(output_path / "merge_config.json", "w") as f:
            json.dump(merge_config, f, indent=2)
        
        print(f"\n✅ Model saved to {output_path}")
        print(f"   - model.safetensors (包含 _groot_model.* 和 distribution_adapter 权重)")
        print(f"   - config.json (base_model_path 已修改为本地路径)")
        print(f"   - merge_config.json")


# ============================================================
# 便捷函数
# ============================================================

def create_expert_merger(
    narrower_path: str,
    wider_path: str,
    base_model_path: str = "nvidia/GR00T-N1.5-3B",
    initial_coefficient: float = 0.5,
    regularization_weight: float = 0.8,
    device: str = "cuda:0",
) -> ExpertMerger:
    """
    创建 Expert Merger
    
    Args:
        narrower_path: 窄箱子模型路径
        wider_path: 宽箱子模型路径
        base_model_path: Base 模型路径
        initial_coefficient: 初始系数值
        regularization_weight: 正则化权重
        device: 计算设备
    
    Returns:
        ExpertMerger 实例
    """
    config = ExpertMergeConfig(
        expert_paths=[narrower_path, wider_path],
        expert_names=["narrower", "wider"],
        base_model_path=base_model_path,
        initial_coefficient=initial_coefficient,
        regularization_weight=regularization_weight,
        device=device,
    )
    
    merger = ExpertMerger(config)
    merger.load_models()
    
    return merger


def simple_task_arithmetic_merge(
    narrower_path: str,
    wider_path: str,
    output_path: str,
    base_model_path: str = "nvidia/GR00T-N1.5-3B",
    narrower_weight: float = 0.5,
    wider_weight: float = 0.5,
    skip_action_head: bool = False,
) -> None:
    """
    简单的 Task Arithmetic 融合（无需训练）
    
    θ_merged = θ_base + α_narrow * τ_narrow + α_wide * τ_wide
    
    ⚠️ 重要：如果 skip_action_head=True，只融合 backbone，action_head 使用 narrower 的
    DiT 对参数变化非常敏感，不能直接线性插值！
    
    Args:
        narrower_path: 窄箱子模型路径
        wider_path: 宽箱子模型路径
        output_path: 输出路径
        base_model_path: Base 模型路径
        narrower_weight: 窄箱子模型权重
        wider_weight: 宽箱子模型权重
        skip_action_head: 是否跳过 action_head (DiT) 的融合
    """
    print(f"\n{'='*60}")
    print(f"🔧 Simple Task Arithmetic Merge")
    print(f"   narrower weight: {narrower_weight}")
    print(f"   wider weight: {wider_weight}")
    if skip_action_head:
        print(f"   ⚠️ Skip action_head: True (使用 narrower 的 DiT)")
    print(f"{'='*60}\n")
    
    # 加载权重
    def load_weights(path: str) -> dict:
        path = Path(path)
        safetensors_files = glob.glob(str(path / "model*.safetensors"))
        state_dict = {}
        for f in sorted(safetensors_files):
            state_dict.update(load_file(f))
        return state_dict
    
    # 加载 base 模型配置和权重
    try:
        base_local_path = snapshot_download(base_model_path, repo_type="model")
    except (HFValidationError, RepositoryNotFoundError):
        base_local_path = base_model_path
    
    base_state_dict = load_weights(base_local_path)
    narrower_state_dict = load_weights(narrower_path)
    wider_state_dict = load_weights(wider_path)
    
    print(f"✅ Loaded weights")
    print(f"   Base: {len(base_state_dict)} tensors")
    print(f"   Narrower: {len(narrower_state_dict)} tensors")
    print(f"   Wider: {len(wider_state_dict)} tensors")
    
    # 计算 Task Vectors
    τ_narrow = {k: narrower_state_dict[k] - base_state_dict[k] for k in base_state_dict if k in narrower_state_dict}
    τ_wide = {k: wider_state_dict[k] - base_state_dict[k] for k in base_state_dict if k in wider_state_dict}
    
    # 融合
    # ⚠️ 重要：如果 skip_action_head=True，跳过 action_head 层的融合
    merged_state_dict = {}
    skipped_action_head_layers = 0
    for k in base_state_dict:
        # 检查是否是 action_head 层
        if skip_action_head and k.startswith('action_head.'):
            # 使用 narrower 的 action_head
            if k in narrower_state_dict:
                merged_state_dict[k] = narrower_state_dict[k]
                skipped_action_head_layers += 1
            else:
                merged_state_dict[k] = base_state_dict[k]
        else:
            merged = base_state_dict[k].clone()
            if k in τ_narrow:
                merged = merged + narrower_weight * τ_narrow[k]
            if k in τ_wide:
                merged = merged + wider_weight * τ_wide[k]
            merged_state_dict[k] = merged
    
    if skip_action_head:
        print(f"⚠️ Skipped {skipped_action_head_layers} action_head layers (使用 narrower 的 DiT)")
    
    # 保存
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_file(merged_state_dict, str(output_path / "model.safetensors"))
    
    # 复制配置文件
    import shutil
    narrower_path = Path(narrower_path)
    for config_file in ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"]:
        src = narrower_path / config_file
        if src.exists():
            shutil.copy(src, output_path / config_file)
    
    for pattern in ["policy_preprocessor*.safetensors", "policy_postprocessor*.safetensors"]:
        for src in narrower_path.glob(pattern):
            shutil.copy(src, output_path / src.name)
    
    # 保存融合配置
    merge_config = {
        "merge_method": "task_arithmetic",
        "narrower_path": str(narrower_path),
        "wider_path": str(wider_path),
        "base_model_path": base_model_path,
        "narrower_weight": narrower_weight,
        "wider_weight": wider_weight,
        "skip_action_head": skip_action_head,
        "action_head_source": "narrower" if skip_action_head else "merged",
    }
    with open(output_path / "merge_config.json", "w") as f:
        json.dump(merge_config, f, indent=2)
    
    print(f"\n✅ Merged model saved to {output_path}")


def direct_interpolation_merge(
    narrower_path: str,
    wider_path: str,
    output_path: str,
    alpha: float = 0.5,
    skip_action_head: bool = False,
) -> None:
    """
    直接权重插值（最简单的方法）
    
    θ_merged = α * θ_narrow + (1-α) * θ_wide
    
    ⚠️ 重要：如果 skip_action_head=True，只融合 backbone，action_head 使用 narrower 的
    DiT 对参数变化非常敏感，不能直接线性插值！
    
    Args:
        narrower_path: 窄箱子模型路径
        wider_path: 宽箱子模型路径
        output_path: 输出路径
        alpha: 插值系数 (0=全用wider, 1=全用narrower)
        skip_action_head: 是否跳过 action_head (DiT) 的融合
    """
    print(f"\n{'='*60}")
    print(f"🔧 Direct Interpolation Merge")
    print(f"   alpha (narrower): {alpha}")
    print(f"   1-alpha (wider): {1-alpha}")
    if skip_action_head:
        print(f"   ⚠️ Skip action_head: True (使用 narrower 的 DiT)")
    print(f"{'='*60}\n")
    
    def load_weights(path: str) -> dict:
        path = Path(path)
        safetensors_files = glob.glob(str(path / "model*.safetensors"))
        state_dict = {}
        for f in sorted(safetensors_files):
            state_dict.update(load_file(f))
        return state_dict
    
    narrower_state_dict = load_weights(narrower_path)
    wider_state_dict = load_weights(wider_path)
    
    # 直接插值
    # ⚠️ 重要：如果 skip_action_head=True，跳过 action_head 层的融合
    merged_state_dict = {}
    skipped_action_head_layers = 0
    for k in narrower_state_dict:
        # 检查是否是 action_head 层
        if skip_action_head and k.startswith('action_head.'):
            # 使用 narrower 的 action_head（不进行插值！）
            merged_state_dict[k] = narrower_state_dict[k]
            skipped_action_head_layers += 1
        elif k in wider_state_dict:
            merged_state_dict[k] = alpha * narrower_state_dict[k] + (1 - alpha) * wider_state_dict[k]
        else:
            merged_state_dict[k] = narrower_state_dict[k]
    
    if skip_action_head:
        print(f"⚠️ Skipped {skipped_action_head_layers} action_head layers (使用 narrower 的 DiT)")
    
    # 保存
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_file(merged_state_dict, str(output_path / "model.safetensors"))
    
    # 复制配置文件
    import shutil
    narrower_path = Path(narrower_path)
    for config_file in ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"]:
        src = narrower_path / config_file
        if src.exists():
            shutil.copy(src, output_path / config_file)
    
    for pattern in ["policy_preprocessor*.safetensors", "policy_postprocessor*.safetensors"]:
        for src in narrower_path.glob(pattern):
            shutil.copy(src, output_path / src.name)
    
    merge_config = {
        "merge_method": "direct_interpolation",
        "narrower_path": str(narrower_path),
        "wider_path": str(wider_path),
        "alpha": alpha,
        "skip_action_head": skip_action_head,
        "action_head_source": "narrower" if skip_action_head else "interpolated",
    }
    with open(output_path / "merge_config.json", "w") as f:
        json.dump(merge_config, f, indent=2)
    
    print(f"\n✅ Merged model saved to {output_path}")
