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

# 导入适配层相关类（用于 two_stage_adapter 方法）
try:
    from lerobot.policies.groot.weight_merge_groot import DistributionAdapter
    from lerobot.policies.groot.groot_n1 import BACKBONE_FEATURE_KEY
    ADAPTER_AVAILABLE = True
except ImportError:
    ADAPTER_AVAILABLE = False
    DistributionAdapter = None
    BACKBONE_FEATURE_KEY = "backbone_features"

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


def load_adapter_if_needed(policy: GrootPolicy, model_path: str, merge_config: dict | None) -> DistributionAdapter | None:
    """
    如果需要，加载适配层权重
    
    Args:
        policy: GrootPolicy 实例
        model_path: 模型路径
        merge_config: 融合配置（如果为 None 会尝试加载）
    
    Returns:
        适配层实例，如果不需要适配层则返回 None
    """
    if not ADAPTER_AVAILABLE:
        return None
    
    # 检查是否为 two_stage_adapter 方法
    if merge_config is None:
        merge_config = load_merge_config(model_path)
    
    if merge_config is None:
        return None
    
    merge_method = merge_config.get('merge_method', '')
    if merge_method not in ['two_stage_adapter', 'mergevla', 'mergevla_moe']:
        return None
    
    merge_method_name = "MergeVLA" if merge_method == 'mergevla' else "Two-Stage Adapter"
    print(f"\n🔧 检测到 {merge_method_name} 融合方法，正在加载适配层...")
    
    # 加载适配层配置
    adapter_type = merge_config.get('adapter_type', 'sparse_lora' if merge_method == 'mergevla' else 'linear')
    lora_rank = merge_config.get('lora_rank', 16)  # ⚠️ 关键：从 merge_config 读取 lora_rank
    num_tasks = merge_config.get('num_tasks', 2)  # 任务数量（用于 sparse_lora）
    sparsity = merge_config.get('sparsity', 0.5)  # 稀疏度（用于 sparse_lora）
    
    # 获取 hidden_size
    groot_model = policy._groot_model
    hidden_size = None
    
    # 方法1: 从 action_head 配置中获取 backbone_embedding_dim
    if hasattr(groot_model, 'action_head') and hasattr(groot_model.action_head, 'config'):
        hidden_size = getattr(groot_model.action_head.config, 'backbone_embedding_dim', None)
    
    # 方法2: 如果方法1失败，从 action_head_cfg 中获取
    if hidden_size is None and hasattr(groot_model.config, 'action_head_cfg'):
        hidden_size = groot_model.config.action_head_cfg.get('backbone_embedding_dim', None)
    
    # 方法3: 如果还是 None，尝试从实际的 backbone 输出推断
    if hidden_size is None:
        try:
            if hasattr(groot_model.backbone, 'eagle_linear'):
                if isinstance(groot_model.backbone.eagle_linear, torch.nn.Identity):
                    hidden_size = 2048
                elif isinstance(groot_model.backbone.eagle_linear, torch.nn.Linear):
                    hidden_size = groot_model.backbone.eagle_linear.out_features
                else:
                    hidden_size = 2048
            else:
                hidden_size = 2048  # GROOT N1.5 默认 backbone 输出维度
        except Exception as e:
            print(f"   ⚠️ Warning: Failed to infer hidden_size: {e}")
            hidden_size = 2048
    
    print(f"   📐 Detected backbone hidden_size: {hidden_size}")
    print(f"   📐 Adapter type: {adapter_type}")
    if adapter_type == 'lora':
        print(f"   📐 LoRA rank: {lora_rank} (from merge_config.json)")
    
    # 创建适配层
    adapter = DistributionAdapter(
        hidden_size=hidden_size,
        adapter_type=adapter_type,
        lora_rank=lora_rank,  # ⚠️ 关键：传递 lora_rank
        num_tasks=num_tasks,  # 任务数量（用于 sparse_lora）
        sparsity=sparsity,  # 稀疏度（用于 sparse_lora）
    )
    
    # 加载适配层权重
    try:
        from safetensors.torch import load_file
        import glob
        
        model_path = Path(model_path)
        safetensors_files = glob.glob(str(model_path / "model*.safetensors"))
        
        if not safetensors_files:
            print(f"   ⚠️ Warning: 未找到 model.safetensors 文件")
            return None
        
        # 加载权重
        state_dict = {}
        for f in sorted(safetensors_files):
            state_dict.update(load_file(f))
        
        # 提取适配层权重
        # ⚠️ 关键修复：支持两种键名格式
        # 1. 新格式：`_groot_model.distribution_adapter.*`
        # 2. 旧格式：`distribution_adapter.*`（向后兼容）
        adapter_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_groot_model.distribution_adapter.'):
                # 新格式：去掉 `_groot_model.distribution_adapter.` 前缀
                new_key = key[len('_groot_model.distribution_adapter.'):]
                adapter_state_dict[new_key] = value
            elif key.startswith('distribution_adapter.'):
                # 旧格式：去掉 `distribution_adapter.` 前缀
                new_key = key[len('distribution_adapter.'):]
                adapter_state_dict[new_key] = value
        
        if not adapter_state_dict:
            print(f"   ⚠️ Warning: 未找到 distribution_adapter.* 权重")
            print(f"   📋 可用的权重键前缀:")
            prefixes = set()
            for k in state_dict.keys():
                prefix = k.split('.')[0]
                prefixes.add(prefix)
            for p in sorted(prefixes):
                print(f"      - {p}")
            return None
        
        # 加载适配层权重
        adapter.load_state_dict(adapter_state_dict, strict=True)
        adapter.eval()  # 设置为评估模式
        adapter.to(next(policy.parameters()).device)
        
        # ⚠️ 关键修复：限制 residual_scale 以避免 chunk 变平
        # 实验发现：当 residual_scale > 0.10 时，Flow Matching 迭代去噪会崩溃
        # 导致所有时间步收敛到相似的值（chunk 变平）
        MAX_SAFE_RESIDUAL_SCALE = 0.10
        with torch.no_grad():
            if hasattr(adapter, 'adapter') and hasattr(adapter.adapter, 'residual_scale'):
                original_scale = adapter.adapter.residual_scale.item()
                if original_scale > MAX_SAFE_RESIDUAL_SCALE:
                    adapter.adapter.residual_scale.fill_(MAX_SAFE_RESIDUAL_SCALE)
                    print(f"   ⚠️ 修复 residual_scale: {original_scale:.4f} → {MAX_SAFE_RESIDUAL_SCALE:.4f}")
                    print(f"      原因：residual_scale > 0.10 会导致 Flow Matching 崩溃，chunk 变平")
                else:
                    print(f"   ✅ residual_scale={original_scale:.4f} 在安全范围内")
        
        print(f"   ✅ 适配层权重加载成功")
        print(f"   📊 适配层参数数量: {sum(p.numel() for p in adapter.parameters()):,}")
        
        # 验证加载的权重键名
        print(f"   📋 加载的适配层权重键: {list(adapter_state_dict.keys())}")
        
        # ⚠️ 关键诊断：检查适配层权重是否正常
        with torch.no_grad():
            for name, param in adapter.named_parameters():
                param_mean = param.mean().item()
                param_std = param.std().item()
                param_abs_mean = param.abs().mean().item()
                print(f"   📊 {name}: mean={param_mean:.6f}, std={param_std:.6f}, abs_mean={param_abs_mean:.6f}")
                
                # 检查权重是否异常
                if 'lora_A' in name or 'lora_B' in name:
                    if param_abs_mean < 0.01:
                        print(f"      ⚠️ WARNING: {name} weights are very small! May not be effective.")
                    elif param_abs_mean > 1.0:
                        print(f"      ⚠️ WARNING: {name} weights are very large! May cause instability.")
                elif 'residual_scale' in name:
                    if param_abs_mean < 0.01:
                        print(f"      ⚠️ WARNING: residual_scale is very small! Adapter contribution is minimal.")
                    elif param_abs_mean > 1.0:
                        print(f"      ⚠️ WARNING: residual_scale is large! May amplify features too much.")
        
        return adapter
        
    except Exception as e:
        print(f"   ⚠️ Warning: 加载适配层权重失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def wrap_policy_with_adapter(
    policy: GrootPolicy, 
    adapter: DistributionAdapter, 
    task_type: str = None,
    use_smart_routing: bool = False,  # ⭐ 是否使用 MergeVLA 风格的智能任务路由
    swap_task_mapping: bool = False,  # ⚠️ 是否交换任务映射
):
    """
    包装 GrootPolicy，使其在推理时使用适配层
    
    Args:
        policy: GrootPolicy 实例
        adapter: DistributionAdapter 实例
        task_type: 任务类型 ("narrower", "wider", None)
                   - "narrower": 使用 task_id=0 的适配器参数（或 task_id=1 如果 swap）
                   - "wider": 使用 task_id=1 的适配器参数（或 task_id=0 如果 swap）
                   - None: 如果 use_smart_routing=True，使用智能任务路由；
                          否则使用所有任务的平均
        use_smart_routing: ⭐ 是否使用 MergeVLA 风格的智能任务路由
                          当 task_type=None 时，根据输入特征自动推断任务类型
        swap_task_mapping: ⚠️ 是否交换任务映射（narrower↔wider）
                          如果 Smart Routing 结果相反，使用此选项
    """
    if adapter is None:
        return
    
    # ⚠️ 关键：根据任务类型设置 task_id
    if task_type == "narrower":
        # 如果 swap，narrower 使用 task_id=1
        actual_task_id = 1 if swap_task_mapping else 0
        fixed_task_id = torch.tensor([actual_task_id], device=next(adapter.parameters()).device)
        if swap_task_mapping:
            print(f"   ⚠️ 使用任务路由: task_type=narrower → task_id=1 (已交换映射)")
        else:
            print(f"   ⚠️ 使用任务路由: task_type=narrower (task_id=0)")
    elif task_type == "wider":
        # 如果 swap，wider 使用 task_id=0
        actual_task_id = 0 if swap_task_mapping else 1
        fixed_task_id = torch.tensor([actual_task_id], device=next(adapter.parameters()).device)
        if swap_task_mapping:
            print(f"   ⚠️ 使用任务路由: task_type=wider → task_id=0 (已交换映射)")
        else:
            print(f"   ⚠️ 使用任务路由: task_type=wider (task_id=1)")
    else:
        fixed_task_id = None
        if use_smart_routing:
            print(f"   ⭐ 使用 MergeVLA 智能任务路由 (Test-Time Task Routing)")
            print(f"      根据输入特征自动推断任务相关性，无需手动指定任务类型")
        else:
            print(f"   ⚠️ 未指定任务类型，使用所有任务的平均（可能导致动作混乱！）")
            print(f"      建议：使用 --smart-routing 启用智能任务路由")
            print(f"      或者：使用 --task-type narrower/wider 指定任务类型")
    
    # 保存原始的 get_action 方法
    original_get_action = policy._groot_model.get_action
    
    # 定义新的 get_action 方法（使用适配层）
    def get_action_with_adapter(inputs: dict, **kwargs):
        # 准备输入
        backbone_inputs, action_inputs = policy._groot_model.prepare_input(inputs)
        
        # 通过 backbone
        backbone_outputs = policy._groot_model.backbone(backbone_inputs)
        
        # 通过适配层
        backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]
        
        # ⚠️ 诊断：检查适配层输入输出的统计信息（前几次调用）
        if not hasattr(get_action_with_adapter, '_call_count'):
            get_action_with_adapter._call_count = 0
        get_action_with_adapter._call_count += 1
        
        if get_action_with_adapter._call_count <= 5:
            print(f"\n   🔍 适配层诊断 (调用 #{get_action_with_adapter._call_count}):")
            print(f"      Backbone features: mean={backbone_features.mean().item():.6f}, "
                  f"std={backbone_features.std().item():.6f}, "
                  f"min={backbone_features.min().item():.6f}, "
                  f"max={backbone_features.max().item():.6f}")
            # 检查是否有异常值
            if torch.isnan(backbone_features).any():
                print(f"      ⚠️ Warning: Backbone features contain NaN!")
            if torch.isinf(backbone_features).any():
                print(f"      ⚠️ Warning: Backbone features contain Inf!")
        
        # ⭐ 关键：使用适配层（支持智能任务路由）
        # - 如果 fixed_task_id 不为 None，使用固定的任务 ID
        # - 如果 fixed_task_id 为 None 且 use_smart_routing=True，使用智能任务路由
        # - 如果 fixed_task_id 为 None 且 use_smart_routing=False，使用简单平均
        adapted_features = adapter(
            backbone_features, 
            task_id=fixed_task_id, 
            use_smart_routing=use_smart_routing
        )
        
        # ⚠️ 关键诊断：检查适配层是否过度改变了特征分布
        # 如果适配层改变了特征的统计特性（mean/std），可能导致迭代去噪不稳定
        # 因为 Flow Matching 的迭代去噪对输入特征分布很敏感
        backbone_mean = backbone_features.mean()
        backbone_std = backbone_features.std()
        adapted_mean = adapted_features.mean()
        adapted_std = adapted_features.std()
        
        if get_action_with_adapter._call_count <= 5:
            print(f"      Backbone features: mean={backbone_mean.item():.6f}, std={backbone_std.item():.6f}")
            print(f"      Adapted features: mean={adapted_mean.item():.6f}, std={adapted_std.item():.6f}")
            
            # 检查分布变化
            mean_diff_ratio = abs(adapted_mean - backbone_mean) / (abs(backbone_mean) + 1e-8)
            std_diff_ratio = abs(adapted_std - backbone_std) / (abs(backbone_std) + 1e-8)
            
            print(f"      Distribution change: mean_diff={mean_diff_ratio.item()*100:.1f}%, std_diff={std_diff_ratio.item()*100:.1f}%")
            
            if mean_diff_ratio > 0.3 or std_diff_ratio > 0.3:
                print(f"      ⚠️ WARNING: Adapter significantly changed feature distribution!")
                print(f"         This may cause instability in iterative denoising!")
                print(f"         Consider: 1) Re-training with feature normalization, 2) Using Expert Merging instead")
        
        # ⚠️ 关键修复：如果适配层过度改变了特征分布，应用温和的归一化
        # 保持适配层的方向性变换，但稳定特征的统计特性
        # 这样可以减少迭代去噪过程中的误差累积
        if not hasattr(get_action_with_adapter, '_use_feature_stabilization'):
            # 只在第一次调用时决定是否使用特征稳定化
            mean_diff_ratio = abs(adapted_mean - backbone_mean) / (abs(backbone_mean) + 1e-8)
            std_diff_ratio = abs(adapted_std - backbone_std) / (abs(backbone_std) + 1e-8)
            
            # 如果分布变化超过阈值，启用特征稳定化
            if mean_diff_ratio > 0.5 or std_diff_ratio > 0.5:
                get_action_with_adapter._use_feature_stabilization = True
                print(f"      🔧 Enabling feature stabilization (distribution change too large)")
            else:
                get_action_with_adapter._use_feature_stabilization = False
        
        if get_action_with_adapter._use_feature_stabilization:
            # 应用温和的归一化：保持适配层的方向性，但稳定统计特性
            # 使用混合策略：adapted_features * (1-alpha) + normalized_features * alpha
            alpha = 0.3  # 混合系数，30% 归一化，70% 保持适配层输出
            normalized_features = (adapted_features - adapted_mean) / (adapted_std + 1e-8) * backbone_std + backbone_mean
            adapted_features = (1 - alpha) * adapted_features + alpha * normalized_features
        
        if get_action_with_adapter._call_count <= 5:
            print(f"      Adapted features (final): mean={adapted_features.mean().item():.6f}, "
                  f"std={adapted_features.std().item():.6f}, "
                  f"min={adapted_features.min().item():.6f}, "
                  f"max={adapted_features.max().item():.6f}")
            # 检查适配层是否改变了分布
            feature_diff = (adapted_features - backbone_features).abs().mean().item()
            print(f"      Feature change (|adapted - backbone|): {feature_diff:.6f}")
            if feature_diff < 1e-6:
                print(f"      ⚠️ Warning: Adapter barely changes features! May not be working.")
            if torch.isnan(adapted_features).any():
                print(f"      ⚠️ Warning: Adapted features contain NaN!")
            if torch.isinf(adapted_features).any():
                print(f"      ⚠️ Warning: Adapted features contain Inf!")
        
        # 直接修改 BatchFeature 中的数据（BatchFeature 支持字典式赋值）
        backbone_outputs[BACKBONE_FEATURE_KEY] = adapted_features
        
        # 通过 action_head
        # ⚠️ 关键：如果 kwargs 中没有 rtc_enabled，使用模型的默认值
        rtc_enabled = kwargs.pop('rtc_enabled', policy._groot_model._rtc_enabled())
        action_head_outputs = policy._groot_model.action_head.get_action(
            backbone_outputs, 
            action_inputs, 
            rtc_enabled=rtc_enabled,
            **kwargs
        )
        
        # ⚠️ 诊断：检查预测动作的统计信息
        if get_action_with_adapter._call_count <= 5:
            action_pred = action_head_outputs.get('action_pred')
            if action_pred is not None:
                print(f"      Predicted action: mean={action_pred.mean().item():.6f}, "
                      f"std={action_pred.std().item():.6f}, "
                      f"min={action_pred.min().item():.6f}, "
                      f"max={action_pred.max().item():.6f}")
                # 检查动作是否异常大
                if action_pred.abs().max().item() > 10.0:
                    print(f"      ⚠️ Warning: Action values are very large! May cause instability.")
        
        return action_head_outputs
    
    # 替换 get_action 方法
    policy._groot_model.get_action = get_action_with_adapter
    
    routing_mode = "智能任务路由" if use_smart_routing else ("固定任务" if fixed_task_id is not None else "简单平均")
    print(f"   ✅ Policy 已包装，适配层将在推理时使用 (路由模式: {routing_mode})")


def load_moe_expert_heads(policy: GrootPolicy, model_path: str, merge_config: dict, device: str = "cuda:0"):
    """
    ⭐ 加载 MoE 模式的多个 Expert Heads (action_head)
    
    基于 MergeVLA 论文 Section 3.2 "Expert Head" 概念：
    - 每个任务保留自己独立的 action_head (DiT)
    - 通过 Smart Routing 选择使用哪个专家
    
    Args:
        policy: GrootPolicy 实例
        model_path: 模型路径
        merge_config: 融合配置
        device: 设备
    
    Returns:
        MoEActionHead 实例
    """
    import copy
    from safetensors.torch import load_file
    import glob
    
    try:
        from lerobot.policies.groot.weight_merge_groot import MoEActionHead
    except ImportError:
        print("   ⚠️ MoEActionHead 不可用，无法加载 MoE 模式")
        return None
    
    expert_names = merge_config.get('expert_names', ['narrower', 'wider'])
    num_experts = len(expert_names)
    
    print(f"\n🎯 加载 MoE Expert Heads...")
    print(f"   专家列表: {expert_names}")
    
    # 加载模型权重
    model_path = Path(model_path)
    safetensors_files = glob.glob(str(model_path / "model*.safetensors"))
    
    state_dict = {}
    for f in sorted(safetensors_files):
        state_dict.update(load_file(f))
    
    # 创建并加载每个专家的 action_head
    expert_heads = torch.nn.ModuleList()
    groot_model = policy._groot_model
    
    for i, expert_name in enumerate(expert_names):
        # 复制 action_head 结构
        expert_head = copy.deepcopy(groot_model.action_head)
        
        # 提取该专家的权重
        prefix = f'_groot_model.expert_heads.{i}.'
        expert_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                expert_state_dict[new_key] = value
        
        if expert_state_dict:
            expert_head.load_state_dict(expert_state_dict, strict=False)
            print(f"   ✅ Expert {expert_name}: 加载 {len(expert_state_dict)} 层权重")
        else:
            print(f"   ⚠️ Expert {expert_name}: 未找到专用权重，使用默认 action_head")
        
        expert_head.eval()
        expert_heads.append(expert_head)
    
    # 创建 MoE 动作头
    moe_head = MoEActionHead(
        expert_heads=expert_heads,
        expert_names=expert_names,
        use_soft_routing=merge_config.get('use_soft_routing', False),
    )
    moe_head.to(device)
    moe_head.eval()
    
    print(f"   ✅ MoE Action Head 创建成功，{num_experts} 个专家")
    return moe_head


def wrap_policy_with_moe(
    policy: GrootPolicy, 
    adapter: DistributionAdapter,
    moe_head,
    task_type: str = None,
    use_smart_routing: bool = False,
    swap_task_mapping: bool = False,
):
    """
    ⭐ 包装 GrootPolicy，使其使用 MoE 动作头
    
    在 MoE 模式下：
    - Backbone: 融合后的 backbone
    - Adapter: Sparse LoRA 适配层
    - MoE Head: 多个独立的 Expert Head (action_head)
    
    Args:
        policy: GrootPolicy 实例
        adapter: DistributionAdapter 实例
        moe_head: MoEActionHead 实例
        task_type: 任务类型 ("narrower", "wider", None)
        use_smart_routing: 是否使用智能任务路由
        swap_task_mapping: 是否交换任务映射
    """
    if adapter is None or moe_head is None:
        return
    
    # ⭐ 设置 action heads 用于 MergeVLA SVD-based 路由
    # 这是论文 Section 3.3 的关键：需要访问值投影矩阵进行 SVD 分解
    if hasattr(adapter, 'adapter') and hasattr(adapter.adapter, 'set_action_heads'):
        if hasattr(moe_head, 'expert_heads'):
            adapter.adapter.set_action_heads(list(moe_head.expert_heads))
            print(f"   ⭐ SVD-based routing: 已连接 {len(moe_head.expert_heads)} 个专家的 action heads")
    
    # 设置 task_id
    if task_type == "narrower":
        actual_task_id = 1 if swap_task_mapping else 0
        fixed_task_id = torch.tensor([actual_task_id], device=next(adapter.parameters()).device)
        print(f"   🎯 MoE 路由: task_type=narrower → task_id={actual_task_id} → 使用 {moe_head.expert_names[actual_task_id]} 专家")
    elif task_type == "wider":
        actual_task_id = 0 if swap_task_mapping else 1
        fixed_task_id = torch.tensor([actual_task_id], device=next(adapter.parameters()).device)
        print(f"   🎯 MoE 路由: task_type=wider → task_id={actual_task_id} → 使用 {moe_head.expert_names[actual_task_id]} 专家")
    else:
        fixed_task_id = None
        if use_smart_routing:
            print(f"   ⭐ MoE 智能任务路由 (根据输入特征自动选择专家)")
        else:
            print(f"   ⚠️ MoE: 未指定任务类型，使用默认专家 ({moe_head.expert_names[0]})")
            fixed_task_id = torch.tensor([0], device=next(adapter.parameters()).device)
    
    # 定义新的 get_action 方法
    def get_action_with_moe(inputs: dict, **kwargs):
        backbone_inputs, action_inputs = policy._groot_model.prepare_input(inputs)
        backbone_outputs = policy._groot_model.backbone(backbone_inputs)
        
        # 通过适配层
        backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]
        
        # 计算路由权重（如果使用智能路由）
        routing_weights = None
        if fixed_task_id is None and use_smart_routing:
            routing_weights = adapter.adapter.compute_task_routing_scores(backbone_features)
        
        # 应用适配
        adapted_features = adapter(
            backbone_features, 
            task_id=fixed_task_id,
            use_smart_routing=use_smart_routing
        )
        backbone_outputs[BACKBONE_FEATURE_KEY] = adapted_features
        
        # 通过 MoE 动作头
        rtc_enabled = kwargs.pop('rtc_enabled', policy._groot_model._rtc_enabled())
        return moe_head.get_action(
            backbone_outputs, 
            action_inputs, 
            task_id=fixed_task_id,
            routing_weights=routing_weights,
            rtc_enabled=rtc_enabled,
            **kwargs
        )
    
    policy._groot_model.get_action = get_action_with_moe
    routing_mode = "智能任务路由" if use_smart_routing else ("固定专家" if fixed_task_id is not None else "默认专家")
    print(f"   ✅ Policy 已包装 MoE 动作头 (路由模式: {routing_mode})")


def eval_on_dataset(
    model_path: str,
    lerobot_dataset_path: str | None = None,
    episode: int = 0,
    n_actions: int = 16,
    show_progress: bool = True,
    use_default_datasets: bool = False,
    visualize: bool = False,
    disable_adapter: bool = False,
    infer_per_frame: int = 1,
    task_type: str = None,
    use_smart_routing: bool = False,  # ⭐ 是否使用 MergeVLA 智能任务路由
    swap_task_mapping: bool = False,  # ⚠️ 是否交换任务映射
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
        disable_adapter: 是否禁用适配层（用于测试）
        infer_per_frame: 每隔多少帧重新推理一次（>=1，默认1=每帧推理）
        task_type: 任务类型 ("narrower", "wider", None)
                   ⚠️ 关键：对于 sparse_lora 适配器，如果不使用智能路由，
                   必须指定任务类型！否则会使用所有任务的平均。
        use_smart_routing: ⭐ 是否使用 MergeVLA 风格的智能任务路由
                          当任务身份未知时，根据模型内部参数子空间
        swap_task_mapping: ⚠️ 是否交换任务映射（如果 Smart Routing 结果相反）
                          自动推断任务相关性（参考 MergeVLA 论文 Section 3.3）
    """
    infer_per_frame = max(1, infer_per_frame)  # 至少每帧推理一次
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
    print(f"🔄 推理频率: 每 {infer_per_frame} 帧推理一次 (infer_per_frame={infer_per_frame})")
    
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
        
        # ⭐ MergeVLA Section 4.1: 参数级稀疏掩码融合配置
        use_sparse_merge = merge_config.get('use_sparse_merge', None)
        sparse_merge_lambda = merge_config.get('sparse_merge_lambda', None)
        if use_sparse_merge is not None:
            print(f"\n   🔥 MergeVLA Section 4.1 参数级稀疏掩码融合:")
            print(f"     - 稀疏掩码融合: {'✅ 启用' if use_sparse_merge else '❌ 禁用'}")
            if use_sparse_merge and sparse_merge_lambda is not None:
                print(f"     - Lambda (容忍度系数): {sparse_merge_lambda}")
                print(f"     - 公式: S_m = I[|τ_m| > λ|τ_merge - τ_m|]")
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
    
    # ⭐ 检测 MoE 模式
    use_moe = merge_config.get('use_moe', False) if merge_config else False
    
    # 先加载默认的 preprocessor 和 postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
    )
    
    # ⭐ 智能路由支持：加载两个专家的 postprocessor
    expert_postprocessors = {}
    use_smart_postprocessor = False
    
    if use_moe:
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        
        # ⭐ 智能路由模式（task_type=None + use_smart_routing=True）
        if use_smart_routing and task_type is None:
            print(f"   ⭐ 智能路由模式: 加载所有专家的 postprocessor")
            expert_names = ["narrower", "wider"]
            
            for expert_name in expert_names:
                expert_post_dir = Path(model_path) / "expert_postprocessors" / expert_name
                if expert_post_dir.exists():
                    try:
                        expert_postprocessors[expert_name] = PolicyProcessorPipeline.from_pretrained(
                            pretrained_model_name_or_path=str(expert_post_dir),
                            config_filename="policy_postprocessor.json",
                            overrides={},
                            to_transition=policy_action_to_transition,
                            to_output=transition_to_policy_action,
                        )
                        print(f"      ✅ 已加载 {expert_name} 专家的 postprocessor")
                    except Exception as e:
                        print(f"      ⚠️ 加载 {expert_name} postprocessor 失败: {e}")
                else:
                    print(f"      ⚠️ {expert_name} postprocessor 目录不存在: {expert_post_dir}")
            
            if len(expert_postprocessors) == 2:
                use_smart_postprocessor = True
                print(f"   ✅ 智能 postprocessor 选择已启用")
                print(f"      根据每次推理的路由决策动态选择 postprocessor")
            else:
                print(f"   ⚠️ 无法加载所有专家 postprocessor，使用默认 postprocessor")
        
        # 固定任务类型模式
        elif task_type is not None:
            expert_post_dir = Path(model_path) / "expert_postprocessors" / task_type
            if expert_post_dir.exists():
                print(f"   ⭐ MoE 模式: 将使用 {task_type} 专家的 postprocessor")
                print(f"      路径: {expert_post_dir}")
                try:
                    postprocessor = PolicyProcessorPipeline.from_pretrained(
                        pretrained_model_name_or_path=str(expert_post_dir),
                        config_filename="policy_postprocessor.json",
                        overrides={},
                        to_transition=policy_action_to_transition,
                        to_output=transition_to_policy_action,
                    )
                    print(f"   ✅ 已加载 {task_type} 专家的 postprocessor")
                except Exception as e:
                    print(f"   ⚠️ 加载专家 postprocessor 失败: {e}")
                    print(f"      使用默认 postprocessor")
            else:
                print(f"   ⚠️ 专家 postprocessor 目录不存在: {expert_post_dir}")
                print(f"      使用默认 postprocessor（可能导致 {task_type} 任务不准确）")
    
    print("✅ 模型加载完成")
    
    if use_moe:
        print(f"\n🎯 检测到 MoE 模式（多专家动作头）")
    
    adapter = load_adapter_if_needed(policy, model_path, merge_config)
    moe_head = None
    
    if adapter is not None:
        if not disable_adapter:
            if use_moe:
                # ⭐ MoE 模式：加载多个专家头
                moe_head = load_moe_expert_heads(policy, model_path, merge_config, device)
                if moe_head is not None:
                    wrap_policy_with_moe(
                        policy, 
                        adapter,
                        moe_head,
                        task_type=task_type,
                        use_smart_routing=use_smart_routing,
                        swap_task_mapping=swap_task_mapping,
                    )
                else:
                    # MoE 头加载失败，回退到单 action_head 模式
                    print(f"   ⚠️ MoE 头加载失败，回退到单 action_head 模式")
                    wrap_policy_with_adapter(
                        policy, 
                        adapter, 
                        task_type=task_type,
                        use_smart_routing=use_smart_routing,
                        swap_task_mapping=swap_task_mapping,
                    )
            else:
                # 原有模式：单 action_head
                wrap_policy_with_adapter(
                    policy, 
                    adapter, 
                    task_type=task_type,
                    use_smart_routing=use_smart_routing,  # ⭐ MergeVLA 智能任务路由
                    swap_task_mapping=swap_task_mapping,  # ⚠️ 交换任务映射
                )
        else:
            print(f"\n⚠️  WARNING: Adapter layer is DISABLED for testing!")
            print(f"   Using raw backbone features (without adapter)")
            print(f"   This helps diagnose if the adapter is causing instability")
    
    # ⚠️ 关键诊断：检查模型配置是否匹配
    print(f"\n🔍 模型配置诊断:")
    print(f"   Action chunk size: {n_actions}")
    print(f"   Policy config chunk_size: {policy.config.chunk_size}")
    print(f"   Policy config n_action_steps: {policy.config.n_action_steps}")
    
    # 检查 future_tokens 配置（从之前的警告信息看，可能存在不匹配）
    if hasattr(policy._groot_model, 'action_head') and hasattr(policy._groot_model.action_head, 'future_tokens'):
        future_tokens_shape = policy._groot_model.action_head.future_tokens.weight.shape
        print(f"   Action head future_tokens shape: {future_tokens_shape}")
        if future_tokens_shape[0] != n_actions:
            print(f"   ⚠️ WARNING: future_tokens shape ({future_tokens_shape[0]}) != action chunk size ({n_actions})!")
            print(f"      This mismatch may cause instability!")
    
    # 如果禁用适配层后仍然震荡，说明问题可能在 backbone 融合
    if disable_adapter and adapter is not None:
        print(f"\n⚠️  DIAGNOSIS: Adapter disabled but actions still oscillatory")
        print(f"   This suggests the problem is NOT in the adapter layer!")
        print(f"   Possible causes:")
        print(f"   1. Backbone fusion itself is problematic (simple interpolation may not work)")
        print(f"   2. Action head configuration mismatch (future_tokens, etc.)")
        print(f"   3. Model weights not properly merged")
        print(f"\n   💡 建议解决方案：")
        print(f"      ⭐ 使用 Expert Merging 方法（不依赖适配层，更可靠）：")
        print(f"         ./merge_groot_models.sh expert_merge")
        print(f"      或者使用 Task Arithmetic（无需训练，快速）：")
        print(f"         ./merge_groot_models.sh task_arithmetic")
    
    # 如果适配层加载失败，也给出诊断
    if adapter is None and merge_config and merge_config.get('merge_method') in ['two_stage_adapter', 'mergevla']:
        print(f"\n⚠️  CRITICAL: Adapter layer failed to load!")
        print(f"   This means the model is using raw fused backbone features")
        print(f"   without distribution adaptation.")
        print(f"   If actions are oscillatory, this confirms the problem:")
        print(f"   Fused backbone output distribution doesn't match action_head expectations.")
        print(f"\n   💡 解决方案：")
        print(f"      1. 重新训练适配层（确保 merge_config.json 包含 lora_rank）")
        print(f"      2. ⭐ 使用 Expert Merging 方法（推荐，不依赖适配层）")
        print(f"         ./merge_groot_models.sh expert_merge")
    
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
                        infer_per_frame=infer_per_frame,
                        # ⭐ 智能 postprocessor 支持
                        adapter=adapter,
                        expert_postprocessors=expert_postprocessors,
                        use_smart_postprocessor=use_smart_postprocessor,
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
        
        # ⭐ 打印智能任务路由统计（如果使用了 smart_routing）
        if use_smart_routing and adapter is not None:
            routing_stats = adapter.get_routing_stats()
            if routing_stats:
                print(f"\n{'='*80}")
                print(f"⭐ MergeVLA Smart Routing 统计")
                print(f"{'='*80}")
                print(f"   总调用次数: {routing_stats.get('total_calls', 0)}")
                print(f"   倾向 Task 0 (narrower): {routing_stats.get('task_0', 0)} 次 "
                      f"({routing_stats.get('task_0_ratio', 0)*100:.1f}%)")
                print(f"   倾向 Task 1 (wider): {routing_stats.get('task_1', 0)} 次 "
                      f"({routing_stats.get('task_1_ratio', 0)*100:.1f}%)")
                print(f"   混合路由: {routing_stats.get('mixed', 0)} 次 "
                      f"({routing_stats.get('mixed_ratio', 0)*100:.1f}%)")
                print(f"{'='*80}")
        
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
            infer_per_frame=infer_per_frame,
            # ⭐ 智能 postprocessor 支持
            adapter=adapter,
            expert_postprocessors=expert_postprocessors,
            use_smart_postprocessor=use_smart_postprocessor,
        )
        
        print_single_result(result)
    
    # ⭐ 打印智能任务路由统计（如果使用了 smart_routing）
    if use_smart_routing and adapter is not None:
        routing_stats = adapter.get_routing_stats()
        if routing_stats:
            print(f"\n{'='*80}")
            print(f"⭐ MergeVLA Smart Routing 统计")
            print(f"{'='*80}")
            print(f"   总调用次数: {routing_stats.get('total_calls', 0)}")
            print(f"   倾向 Task 0 (narrower): {routing_stats.get('task_0', 0)} 次 "
                  f"({routing_stats.get('task_0_ratio', 0)*100:.1f}%)")
            print(f"   倾向 Task 1 (wider): {routing_stats.get('task_1', 0)} 次 "
                  f"({routing_stats.get('task_1_ratio', 0)*100:.1f}%)")
            print(f"   混合路由: {routing_stats.get('mixed', 0)} 次 "
                  f"({routing_stats.get('mixed_ratio', 0)*100:.1f}%)")
            print(f"{'='*80}")
    
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
    infer_per_frame: int = 1,
    # ⭐ 智能 postprocessor 选择支持
    adapter = None,  # DistributionAdapter 实例（用于获取路由决策）
    expert_postprocessors: dict = None,  # {"narrower": postprocessor, "wider": postprocessor}
    use_smart_postprocessor: bool = False,  # 是否使用智能 postprocessor 选择
) -> dict:
    """评估单个 episode
    
    Args:
        infer_per_frame: 每隔多少帧重新推理一次（>=1，默认1=每帧推理）
        adapter: DistributionAdapter 实例（用于智能 postprocessor 选择）
        expert_postprocessors: 专家 postprocessor 字典（用于智能选择）
        use_smart_postprocessor: 是否启用智能 postprocessor 选择
    """
    infer_per_frame = max(1, infer_per_frame)
    last_inferred_chunk: np.ndarray | None = None
    last_inference_step: int = -1
    
    # ⭐ 智能 postprocessor 选择辅助函数
    def get_current_postprocessor():
        """根据最后一次路由决策选择 postprocessor"""
        if not use_smart_postprocessor or expert_postprocessors is None:
            return postprocessor
        
        # 从 adapter 获取最后一次路由决策
        if adapter is not None and hasattr(adapter, 'get_last_task_decision'):
            task_decision = adapter.get_last_task_decision()
            if task_decision and task_decision in expert_postprocessors:
                return expert_postprocessors[task_decision]
        
        return postprocessor  # 默认 postprocessor
    
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
        
        # 判断是否需要执行推理（根据 infer_per_frame 参数）
        should_infer = (data_step % infer_per_frame == 0)
        
        # 模型推理（精确测量推理时间）
        if should_infer:
            # 需要推理：执行完整的推理流程
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
            # ⭐ 智能 postprocessor 选择：推理后根据路由决策选择 postprocessor
            current_postprocessor = get_current_postprocessor()
            
            processed_actions = []
            for i in range(chunk_size):
                single_action = pred_actions[:, i, :]
                processed_action = current_postprocessor(single_action)
                processed_actions.append(processed_action)
            
            pred_actions_unnorm = torch.stack(processed_actions, dim=1)
            pred_chunk = pred_actions_unnorm[0].cpu().numpy()  # (chunk_size, action_dim)
            pred_action_single = pred_chunk[0]  # 取第一个 action
            
            # 保存预测结果供后续帧使用
            last_inferred_chunk = pred_chunk.copy()
            last_inference_step = data_step
        else:
            # 不需要推理：复用上一次的预测结果
            if last_inferred_chunk is not None:
                pred_chunk = last_inferred_chunk.copy()
                # 根据距离上次推理的帧数，选择 chunk 中的对应 action
                # 例如：如果 infer_per_frame=3，data_step=4，last_inference_step=3
                # 则 offset = 4 - 3 = 1，取 pred_chunk[1]
                offset = data_step - last_inference_step
                if offset < pred_chunk.shape[0]:
                    pred_action_single = pred_chunk[offset]
                else:
                    # 如果超出 chunk 范围，使用最后一个 action
                    pred_action_single = pred_chunk[-1]
            else:
                # 如果这是第一帧且 infer_per_frame > 1，需要先推理一次
                print(f"⚠️  Warning: No previous prediction at frame {data_step}. Performing inference anyway.")
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
                
                _, chunk_size, _ = pred_actions.shape
                # ⭐ 智能 postprocessor 选择
                current_postprocessor = get_current_postprocessor()
                
                processed_actions = []
                for i in range(chunk_size):
                    single_action = pred_actions[:, i, :]
                    processed_action = current_postprocessor(single_action)
                    processed_actions.append(processed_action)
                
                pred_actions_unnorm = torch.stack(processed_actions, dim=1)
                pred_chunk = pred_actions_unnorm[0].cpu().numpy()
                pred_action_single = pred_chunk[0]
                
                last_inferred_chunk = pred_chunk.copy()
                last_inference_step = data_step
        
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
                
                # 只在推理时更新预测的 action chunk 可视化
                if should_infer:
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
    parser.add_argument('--action-chunk-size', type=int, default=32,
                       dest='action_chunk_size',
                       help='Action chunk size (default: 32 for GROOT, should match training config)')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    parser.add_argument('--use-default-datasets', action='store_true',
                       help='Evaluate on default training datasets (narrower and wider)')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable Rerun visualization')
    parser.add_argument('--disable-adapter', action='store_true',
                       help='Disable adapter layer for testing (use raw backbone features)')
    parser.add_argument('--infer-per-frame', type=int, default=1,
                       dest='infer_per_frame',
                       help='Run policy inference every N frames (default: 1 = every frame). '
                            'Higher values reduce computation but may decrease accuracy.')
    parser.add_argument('--task-type', type=str, default=None,
                       choices=['narrower', 'wider'],
                       dest='task_type',
                       help='Specify task type for fixed routing. '
                            '"narrower" for narrow box task (task_id=0), '
                            '"wider" for wide box task (task_id=1). '
                            'If not specified, uses --smart-routing or average.')
    parser.add_argument('--smart-routing', action='store_true',
                       dest='smart_routing',
                       help='⭐ Enable MergeVLA-style Test-Time Task Routing. '
                            'When task identity is unknown, automatically infer task relevance '
                            'based on model internal parameter subspaces (value projection). '
                            '🔥 NEW: Also auto-selects the correct postprocessor for each task! '
                            'No need to specify --task-type. (Reference: MergeVLA paper Section 3.3)')
    parser.add_argument('--swap-task-mapping', action='store_true',
                       dest='swap_task_mapping',
                       help='⚠️ Swap task mapping: narrower↔wider. '
                            'Use this if Smart Routing gives opposite results. '
                            'When enabled: --task-type narrower uses task_id=1, '
                            '--task-type wider uses task_id=0.')
    
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
    print(f"Infer Every N Frames: {args.infer_per_frame}")
    
    # 打印任务路由模式
    if args.task_type:
        print(f"🎯 Task Routing: Fixed task_type={args.task_type}")
    elif args.smart_routing:
        print(f"⭐ Task Routing: MergeVLA Smart Routing (自动推断任务类型)")
        print(f"   🔥 自动选择 postprocessor: 根据每次推理的路由决策动态选择")
    else:
        print(f"⚠️ Task Routing: Simple Average (可能导致动作混乱！)")
        print(f"   建议：使用 --smart-routing 自动推断任务类型和 postprocessor")
    print("="*80)
    
    eval_on_dataset(
        model_path=args.model_path,
        lerobot_dataset_path=args.dataset_root,
        episode=args.episode,
        n_actions=args.action_chunk_size,
        show_progress=not args.no_progress,
        use_default_datasets=args.use_default_datasets,
        visualize=args.visualize,
        disable_adapter=args.disable_adapter,
        infer_per_frame=args.infer_per_frame,
        task_type=args.task_type,
        use_smart_routing=args.smart_routing,  # ⭐ MergeVLA 智能任务路由
        swap_task_mapping=args.swap_task_mapping,  # ⚠️ 交换任务映射
    )
