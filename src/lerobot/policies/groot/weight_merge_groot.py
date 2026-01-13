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
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.layer_names = layer_names
        self.num_layers = len(layer_names)
        
        # 创建系数参数: (num_experts, num_layers)
        # 使用 ParameterDict 以便按层名访问
        self.coefficients = nn.ParameterDict()
        for layer_name in layer_names:
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
        
        # 注册 base model 的参数（不可训练）
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 缓存融合后的参数
        self._merged_state_dict = None
        self._cached = False
    
    def _apply_merge(self):
        """
        应用权重融合
        
        θ_merged = θ_base + Σ_k α_k * τ_k
        """
        base_state_dict = self.base_model.state_dict()
        merged_state_dict = {}
        
        for key in base_state_dict:
            merged_param = base_state_dict[key].clone()
            
            # 添加所有专家的 Task Vector 贡献
            for expert_idx, task_vector in enumerate(self.task_vectors):
                if key in task_vector.task_vector:
                    coef = self.coefficients.get_coefficient(key, expert_idx)
                    merged_param = merged_param + coef * task_vector[key].to(merged_param.device)
            
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
    
    def forward_with_merged_weights(self, inputs: dict) -> dict:
        """
        使用融合权重进行前向传播
        
        这是训练时使用的方法，会动态计算融合权重
        """
        # 应用融合权重
        merged_state_dict = self._apply_merge()
        
        # 临时加载融合权重
        original_state = self.base_model.state_dict()
        self.base_model.load_state_dict(merged_state_dict, strict=False)
        
        try:
            # 获取设备类型
            device = next(self.base_model.parameters()).device
            # 使用 autocast 确保数据类型正确（参考 modeling_groot.py）
            use_bf16 = getattr(self.base_model, "compute_dtype", None) == "bfloat16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                # 前向传播
                outputs = self.base_model.get_action(inputs)
        finally:
            # 恢复原始权重
            self.base_model.load_state_dict(original_state, strict=False)
        
        return outputs
    
    def get_hidden_states(self, inputs: dict) -> torch.Tensor:
        """
        获取融合模型的隐藏状态
        
        用于 Hidden Alignment Loss
        """
        merged_state_dict = self._apply_merge()
        original_state = self.base_model.state_dict()
        self.base_model.load_state_dict(merged_state_dict, strict=False)
        
        try:
            # 获取设备类型
            device = next(self.base_model.parameters()).device
            # 使用 autocast 确保数据类型正确（参考 modeling_groot.py）
            use_bf16 = getattr(self.base_model, "compute_dtype", None) == "bfloat16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                backbone_inputs, action_inputs = self.base_model.prepare_input(inputs)
                backbone_outputs = self.base_model.backbone(backbone_inputs)
                hidden_states = backbone_outputs[BACKBONE_FEATURE_KEY]
        finally:
            self.base_model.load_state_dict(original_state, strict=False)
        
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
        ).to(self.device)
        
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
        
        # 对每个专家计算对齐损失
        for expert_idx, expert_model in enumerate(self.expert_models):
            expert_name = self.config.expert_names[expert_idx]
            task_weight = self.config.task_weights[expert_idx] if expert_idx < len(self.config.task_weights) else 1.0
            
            # 1. 获取专家模型的输出（作为目标）
            with torch.no_grad():
                expert_model.eval()
                expert_model = expert_model.to(self.device)
                
                # 获取设备类型和是否使用 bf16
                device = next(expert_model.parameters()).device
                use_bf16 = getattr(expert_model, "compute_dtype", None) == "bfloat16"
                
                # 使用 autocast 确保数据类型正确（参考 modeling_groot.py）
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    # 获取 hidden states
                    expert_backbone_inputs, expert_action_inputs = expert_model.prepare_input(inputs)
                    expert_backbone_outputs = expert_model.backbone(expert_backbone_inputs)
                    expert_hidden = expert_backbone_outputs[BACKBONE_FEATURE_KEY]
                    
                    # 获取 actions
                    expert_outputs = expert_model.get_action(inputs)
                    expert_actions = expert_outputs[ACTION_KEY]
            
            # 2. 获取融合模型的输出
            # 获取 hidden states
            merged_hidden = self.merged_model.get_hidden_states(inputs)
            
            # 获取 actions
            merged_outputs = self.merged_model.forward_with_merged_weights(inputs)
            merged_actions = merged_outputs[ACTION_KEY]
            
            # 3. 计算对齐损失
            hidden_loss = self.compute_hidden_alignment_loss(merged_hidden, expert_hidden)
            action_loss = self.compute_logit_alignment_loss(merged_actions, expert_actions)
            
            # 4. 加权组合
            expert_loss = (
                self.config.hidden_alignment_weight * hidden_loss +
                self.config.logit_alignment_weight * action_loss
            ) * task_weight
            
            total_loss = total_loss + expert_loss
            
            loss_dict[f"{expert_name}_hidden_loss"] = hidden_loss.item()
            loss_dict[f"{expert_name}_action_loss"] = action_loss.item()
            loss_dict[f"{expert_name}_total_loss"] = expert_loss.item()
        
        # 5. 添加正则化损失
        reg_loss = self.coefficients.regularization_loss()
        total_loss = total_loss + self.config.regularization_weight * reg_loss
        
        loss_dict["regularization_loss"] = reg_loss.item()
        loss_dict["total_loss"] = total_loss.item()
        
        # 6. 反向传播
        total_loss.backward()
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
        optimizer = torch.optim.AdamW(
            self.coefficients.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
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
                loss_dict = self.train_step(batch, optimizer)
                epoch_losses.append(loss_dict["total_loss"])
                
                if batch_idx % 10 == 0:
                    print(f"   Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}: loss = {loss_dict['total_loss']:.4f}")
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            coef_stats = self.coefficients.get_stats()
            
            print(f"\n📊 Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"   Average loss: {avg_loss:.4f}")
            print(f"   Coefficient stats: mean={coef_stats.get('mean', 0):.4f}, "
                  f"std={coef_stats.get('std', 0):.4f}, "
                  f"range=[{coef_stats.get('min', 0):.4f}, {coef_stats.get('max', 0):.4f}]")
            print(f"   Per-expert mean: {coef_stats.get('per_expert_mean', [])}")
        
        print(f"\n✅ Training completed!")
    
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
        save_file(state_dict, str(output_path / "model.safetensors"))
        
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
        }
        with open(output_path / "merge_config.json", "w") as f:
            json.dump(merge_config, f, indent=2)
        
        print(f"\n✅ Merged model saved to {output_path}")
        print(f"   - model.safetensors")
        print(f"   - config.json")
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
) -> None:
    """
    简单的 Task Arithmetic 融合（无需训练）
    
    θ_merged = θ_base + α_narrow * τ_narrow + α_wide * τ_wide
    
    Args:
        narrower_path: 窄箱子模型路径
        wider_path: 宽箱子模型路径
        output_path: 输出路径
        base_model_path: Base 模型路径
        narrower_weight: 窄箱子模型权重
        wider_weight: 宽箱子模型权重
    """
    print(f"\n{'='*60}")
    print(f"🔧 Simple Task Arithmetic Merge")
    print(f"   narrower weight: {narrower_weight}")
    print(f"   wider weight: {wider_weight}")
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
    merged_state_dict = {}
    for k in base_state_dict:
        merged = base_state_dict[k].clone()
        if k in τ_narrow:
            merged = merged + narrower_weight * τ_narrow[k]
        if k in τ_wide:
            merged = merged + wider_weight * τ_wide[k]
        merged_state_dict[k] = merged
    
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
    }
    with open(output_path / "merge_config.json", "w") as f:
        json.dump(merge_config, f, indent=2)
    
    print(f"\n✅ Merged model saved to {output_path}")


def direct_interpolation_merge(
    narrower_path: str,
    wider_path: str,
    output_path: str,
    alpha: float = 0.5,
) -> None:
    """
    直接权重插值（最简单的方法）
    
    θ_merged = α * θ_narrow + (1-α) * θ_wide
    
    Args:
        narrower_path: 窄箱子模型路径
        wider_path: 宽箱子模型路径
        output_path: 输出路径
        alpha: 插值系数 (0=全用wider, 1=全用narrower)
    """
    print(f"\n{'='*60}")
    print(f"🔧 Direct Interpolation Merge")
    print(f"   alpha (narrower): {alpha}")
    print(f"   1-alpha (wider): {1-alpha}")
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
    merged_state_dict = {}
    for k in narrower_state_dict:
        if k in wider_state_dict:
            merged_state_dict[k] = alpha * narrower_state_dict[k] + (1 - alpha) * wider_state_dict[k]
        else:
            merged_state_dict[k] = narrower_state_dict[k]
    
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
    }
    with open(output_path / "merge_config.json", "w") as f:
        json.dump(merge_config, f, indent=2)
    
    print(f"\n✅ Merged model saved to {output_path}")
