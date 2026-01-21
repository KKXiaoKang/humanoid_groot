#!/usr/bin/env python
"""
MergeVLA 融合模型实时评估脚本

评估通过 MergeVLA 方法融合的 GROOT 模型，支持稀疏 LoRA 适配层

使用方式：
    python eval/eval_merged_groot.py \
        --model_path ./outputs/merged_groot_mergevla/pretrained_model \
        --rtc.enabled=true \
        --rtc.execution_horizon=10 \
        --task="Depalletize the box" \
        --duration=30
    
    # 禁用适配层（用于调试）
    python eval/eval_merged_groot.py \
        --model_path ./outputs/merged_groot_mergevla/pretrained_model \
        --disable_adapter=true
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import math
import time
import traceback
import json
import glob
from pathlib import Path
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import rospy

from lerobot.configs import parser
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.utils.utils import init_logging
from configs.config import get_camera_observation_key, get_camera_names, CAMERA_COMPONENTS, ACTION_COMPONENTS

from robot_envs.kuavo_depalletize_env import GrabBoxMpcEnv
from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.robot_sdk import RobotSDK
from eval_online import final_reset_arm
from eval_multi_model import (
    resample_chunk_with_claw_hold,
    apply_first_chunk_smooth,
    set_arm_quick_mode,
    EpisodeState,
)

from lerobot.policies.groot.weight_merge_groot import DistributionAdapter
from lerobot.policies.groot.groot_n1 import BACKBONE_FEATURE_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MergeVLAConfig:
    """MergeVLA 评估配置（使用 draccus 解析器）"""
    
    # RTC 配置
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=16,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )
    
    # 模型路径（必需）
    model_path: str = field(
        default="./outputs/merged_groot_mergevla/pretrained_model",
        metadata={"help": "MergeVLA 融合模型路径"}
    )
    
    # 推理参数
    duration: float = field(default=30.0, metadata={"help": "单次推理周期（秒）"})
    fps: float = field(default=10.0, metadata={"help": "推理频率 (Hz)"})
    device: str = field(default="cuda:0", metadata={"help": "运行设备"})
    task: str = field(default="Depalletize the box", metadata={"help": "任务描述"})
    
    # 适配层控制
    disable_adapter: bool = field(default=False, metadata={"help": "禁用适配层（调试用）"})
    
    # ⭐ 任务路由控制
    task_type: Optional[str] = field(
        default=None, 
        metadata={"help": "任务类型: narrower 或 wider。不指定则使用 smart_routing 或平均"}
    )
    smart_routing: bool = field(
        default=False, 
        metadata={"help": "启用 MergeVLA 智能任务路由（根据输入特征自动推断任务类型）"}
    )
    swap_task_mapping: bool = field(
        default=False, 
        metadata={"help": "交换任务映射 (narrower↔wider)，当 Smart Routing 结果相反时使用"}
    )
    
    # 动作队列配置
    # ⚠️ 关键参数：控制何时触发新推理
    # 值太大(如90)会导致频繁推理，chunk间不连续导致抖动
    # 值太小可能导致队列耗尽，动作断档
    # 推荐：设置为 execution_horizon * 5 左右（如 10 * 5 = 50）
    action_queue_size_to_get_new_actions: int = field(
        default=50,  # 从90改为50，减少推理频率
        metadata={"help": "触发新推理的动作队列阈值（推荐: execution_horizon * 5）"}
    )
    
    # 动作平滑参数
    action_smoothing: bool = field(
        default=True,
        metadata={"help": "启用 chunk 间动作平滑（减少抖动）"}
    )
    smoothing_alpha: float = field(
        default=0.3,
        metadata={"help": "平滑系数 (0-1)，越大越平滑但延迟越高"}
    )


@dataclass
class ModelWrapper:
    """模型包装器"""
    name: str
    policy: object
    preprocessor: object
    postprocessor: object
    adapter: object = None
    # ⭐ 智能路由支持：存储两个专家的 postprocessor
    expert_postprocessors: dict = None  # {"narrower": postprocessor, "wider": postprocessor}
    use_smart_postprocessor: bool = False  # 是否使用智能 postprocessor 选择
    
    def reset(self):
        self.policy.reset()
        self.policy.init_rtc_processor()
    
    def get_postprocessor_for_task(self, task_decision: str = None):
        """
        根据任务决策获取对应的 postprocessor
        
        Args:
            task_decision: "narrower" 或 "wider"，如果为 None 则使用默认 postprocessor
        
        Returns:
            对应任务的 postprocessor
        """
        if not self.use_smart_postprocessor or self.expert_postprocessors is None:
            return self.postprocessor
        
        if task_decision is None:
            # 尝试从 adapter 获取最后一次路由决策
            if self.adapter is not None and hasattr(self.adapter, 'get_last_task_decision'):
                task_decision = self.adapter.get_last_task_decision()
        
        if task_decision and task_decision in self.expert_postprocessors:
            return self.expert_postprocessors[task_decision]
        
        return self.postprocessor  # 默认 postprocessor


def load_merge_config(model_path: str) -> dict | None:
    """加载融合配置"""
    merge_config_path = Path(model_path) / "merge_config.json"
    if merge_config_path.exists():
        with open(merge_config_path, 'r') as f:
            return json.load(f)
    return None


def load_adapter(policy, model_path: str, merge_config: dict, device: str = "cuda:0"):
    """加载 MergeVLA 适配层"""
    from safetensors.torch import load_file
    
    adapter_type = merge_config.get('adapter_type', 'sparse_lora')
    lora_rank = merge_config.get('lora_rank', 16)
    num_tasks = merge_config.get('num_tasks', 2)
    sparsity = merge_config.get('sparsity', 0.5)
    
    # 获取 hidden_size
    groot_model = policy._groot_model
    hidden_size = 2048
    if hasattr(groot_model, 'action_head') and hasattr(groot_model.action_head, 'config'):
        hidden_size = getattr(groot_model.action_head.config, 'backbone_embedding_dim', 2048)
    
    logger.info(f"   📐 hidden_size: {hidden_size}, adapter_type: {adapter_type}, lora_rank: {lora_rank}")
    
    # 创建适配层
    adapter = DistributionAdapter(
        hidden_size=hidden_size,
        adapter_type=adapter_type,
        lora_rank=lora_rank,
        num_tasks=num_tasks,
        sparsity=sparsity,
    )
    
    # 加载适配层权重
    model_path = Path(model_path)
    safetensors_files = glob.glob(str(model_path / "model*.safetensors"))
    
    state_dict = {}
    for f in sorted(safetensors_files):
        state_dict.update(load_file(f))
    
    # 提取适配层权重
    adapter_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_groot_model.distribution_adapter.'):
            new_key = key[len('_groot_model.distribution_adapter.'):]
            adapter_state_dict[new_key] = value
    
    if not adapter_state_dict:
        raise RuntimeError("未找到 distribution_adapter 权重")
    
    adapter.load_state_dict(adapter_state_dict, strict=True)
    adapter.eval()
    adapter.to(device)
    
    # ⚠️ 关键修复：限制 residual_scale 以避免 chunk 变平
    # 实验发现：当 residual_scale > 0.10 时，Flow Matching 迭代去噪会崩溃
    # 导致所有时间步收敛到相似的值（chunk 变平）
    MAX_SAFE_RESIDUAL_SCALE = 0.10
    with torch.no_grad():
        if hasattr(adapter, 'adapter') and hasattr(adapter.adapter, 'residual_scale'):
            original_scale = adapter.adapter.residual_scale.item()
            if original_scale > MAX_SAFE_RESIDUAL_SCALE:
                adapter.adapter.residual_scale.fill_(MAX_SAFE_RESIDUAL_SCALE)
                logger.warning(f"   ⚠️ 修复 residual_scale: {original_scale:.4f} → {MAX_SAFE_RESIDUAL_SCALE:.4f}")
                logger.warning(f"      原因：residual_scale > 0.10 会导致 Flow Matching 崩溃，chunk 变平")
            else:
                logger.info(f"   ✅ residual_scale={original_scale:.4f} 在安全范围内")
    
    logger.info(f"   ✅ 适配层加载成功，参数量: {sum(p.numel() for p in adapter.parameters()):,}")
    return adapter


def wrap_policy_with_adapter(
    policy, 
    adapter,
    task_type: str = None,
    use_smart_routing: bool = False,
    swap_task_mapping: bool = False,
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
        swap_task_mapping: ⚠️ 是否交换任务映射（narrower↔wider）
    """
    # ⚠️ 关键：根据任务类型设置 task_id
    if task_type == "narrower":
        actual_task_id = 1 if swap_task_mapping else 0
        fixed_task_id = torch.tensor([actual_task_id], device=next(adapter.parameters()).device)
        if swap_task_mapping:
            logger.info(f"   ⚠️ 使用任务路由: task_type=narrower → task_id=1 (已交换映射)")
        else:
            logger.info(f"   ⚠️ 使用任务路由: task_type=narrower (task_id=0)")
    elif task_type == "wider":
        actual_task_id = 0 if swap_task_mapping else 1
        fixed_task_id = torch.tensor([actual_task_id], device=next(adapter.parameters()).device)
        if swap_task_mapping:
            logger.info(f"   ⚠️ 使用任务路由: task_type=wider → task_id=0 (已交换映射)")
        else:
            logger.info(f"   ⚠️ 使用任务路由: task_type=wider (task_id=1)")
    else:
        fixed_task_id = None
        if use_smart_routing:
            logger.info(f"   ⭐ 使用 MergeVLA 智能任务路由 (Test-Time Task Routing)")
        else:
            logger.warning(f"   ⚠️ 未指定任务类型，使用所有任务的平均")
    
    def get_action_with_adapter(inputs: dict, **kwargs):
        backbone_inputs, action_inputs = policy._groot_model.prepare_input(inputs)
        backbone_outputs = policy._groot_model.backbone(backbone_inputs)
        
        # 通过适配层（支持智能任务路由）
        backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]
        adapted_features = adapter(
            backbone_features, 
            task_id=fixed_task_id,
            use_smart_routing=use_smart_routing
        )
        backbone_outputs[BACKBONE_FEATURE_KEY] = adapted_features
        
        # 通过 action_head
        rtc_enabled = kwargs.pop('rtc_enabled', policy._groot_model._rtc_enabled())
        return policy._groot_model.action_head.get_action(
            backbone_outputs, action_inputs, rtc_enabled=rtc_enabled, **kwargs
        )
    
    policy._groot_model.get_action = get_action_with_adapter
    routing_mode = "智能任务路由" if use_smart_routing else ("固定任务" if fixed_task_id is not None else "简单平均")
    logger.info(f"   ✅ Policy 已包装适配层 (路由模式: {routing_mode})")


def load_moe_expert_heads(policy, model_path: str, merge_config: dict, device: str = "cuda:0"):
    """
    ⭐ 加载 MoE 模式的多个 Expert Heads (action_head)
    
    基于 MergeVLA 论文 Section 3.2 "Expert Head" 概念
    """
    import copy
    from safetensors.torch import load_file
    from lerobot.policies.groot.weight_merge_groot import MoEActionHead
    
    expert_names = merge_config.get('expert_names', ['narrower', 'wider'])
    num_experts = len(expert_names)
    
    logger.info(f"   🎯 加载 MoE Expert Heads: {expert_names}")
    
    # 加载模型权重
    model_path = Path(model_path)
    safetensors_files = glob.glob(str(model_path / "model*.safetensors"))
    
    state_dict = {}
    for f in sorted(safetensors_files):
        state_dict.update(load_file(f))
    
    # 创建并加载每个专家的 action_head
    expert_heads = nn.ModuleList()
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
            logger.info(f"   ✅ Expert {expert_name}: 加载 {len(expert_state_dict)} 层权重")
        else:
            logger.warning(f"   ⚠️ Expert {expert_name}: 未找到专用权重，使用默认 action_head")
        
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
    
    logger.info(f"   ✅ MoE Action Head 创建成功，{num_experts} 个专家")
    return moe_head


def wrap_policy_with_moe(
    policy, 
    adapter,
    moe_head,
    task_type: str = None,
    use_smart_routing: bool = False,
    swap_task_mapping: bool = False,
):
    """
    ⭐ 包装 GrootPolicy，使其使用 MoE 动作头
    
    Args:
        policy: GrootPolicy 实例
        adapter: DistributionAdapter 实例
        moe_head: MoEActionHead 实例
        task_type: 任务类型 ("narrower", "wider", None)
        use_smart_routing: 是否使用智能任务路由
        swap_task_mapping: 是否交换任务映射
    """
    # ⭐ 设置 action heads 用于 MergeVLA SVD-based 路由
    # 这是论文 Section 3.3 的关键：需要访问值投影矩阵进行 SVD 分解
    if hasattr(adapter, 'adapter') and hasattr(adapter.adapter, 'set_action_heads'):
        if hasattr(moe_head, 'expert_heads'):
            adapter.adapter.set_action_heads(list(moe_head.expert_heads))
            logger.info(f"   ⭐ SVD-based routing: 已连接 {len(moe_head.expert_heads)} 个专家的 action heads")
    
    # 设置 task_id
    if task_type == "narrower":
        actual_task_id = 1 if swap_task_mapping else 0
        fixed_task_id = torch.tensor([actual_task_id], device=next(adapter.parameters()).device)
        logger.info(f"   ⚠️ MoE 路由: task_type=narrower → task_id={actual_task_id}")
    elif task_type == "wider":
        actual_task_id = 0 if swap_task_mapping else 1
        fixed_task_id = torch.tensor([actual_task_id], device=next(adapter.parameters()).device)
        logger.info(f"   ⚠️ MoE 路由: task_type=wider → task_id={actual_task_id}")
    else:
        fixed_task_id = None
        if use_smart_routing:
            logger.info(f"   ⭐ MoE 智能任务路由 (根据输入选择专家)")
        else:
            logger.warning(f"   ⚠️ MoE: 未指定任务类型，使用默认专家 (narrower)")
            fixed_task_id = torch.tensor([0], device=next(adapter.parameters()).device)
    
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
    logger.info(f"   ✅ Policy 已包装 MoE 动作头")


def load_mergevla_model(cfg: MergeVLAConfig) -> ModelWrapper:
    """加载 MergeVLA 融合模型"""
    
    logger.info(f"🚀 加载 MergeVLA 模型: {cfg.model_path}")
    
    # 加载融合配置
    merge_config = load_merge_config(cfg.model_path)
    if merge_config is None:
        raise RuntimeError(f"未找到 merge_config.json: {cfg.model_path}")
    
    merge_method = merge_config.get('merge_method', 'mergevla')
    if merge_method not in ('mergevla', 'mergevla_moe'):
        raise RuntimeError(f"不是 MergeVLA 模型: {merge_method}")
    
    use_moe = merge_config.get('use_moe', False)
    if use_moe:
        logger.info(f"🎯 检测到 MoE 模式（多专家动作头）")
    
    # 加载 policy
    config = PreTrainedConfig.from_pretrained(cfg.model_path)
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(cfg.model_path, config=config)
    policy.config.rtc_config = cfg.rtc
    policy.init_rtc_processor()
    policy = policy.to(cfg.device)
    policy.eval()
    
    # 加载适配层和专家头
    adapter = None
    moe_head = None
    
    if not cfg.disable_adapter:
        logger.info(f"🔧 加载 MergeVLA 适配层...")
        adapter = load_adapter(policy, cfg.model_path, merge_config, cfg.device)
        
        if use_moe:
            # MoE 模式：加载多个专家头
            logger.info(f"🔧 加载 MoE Expert Heads...")
            moe_head = load_moe_expert_heads(policy, cfg.model_path, merge_config, cfg.device)
            wrap_policy_with_moe(
                policy, 
                adapter,
                moe_head,
                task_type=cfg.task_type,
                use_smart_routing=cfg.smart_routing,
                swap_task_mapping=cfg.swap_task_mapping,
            )
        else:
            # 原有模式：单个 action_head
            wrap_policy_with_adapter(
                policy, 
                adapter,
                task_type=cfg.task_type,
                use_smart_routing=cfg.smart_routing,
                swap_task_mapping=cfg.swap_task_mapping,
            )
    else:
        logger.warning("⚠️ 适配层已禁用")
    
    # 加载预处理器和默认 postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=cfg.model_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )
    
    # ⭐ 智能路由支持：加载两个专家的 postprocessor
    expert_postprocessors = None
    use_smart_postprocessor = False
    
    if use_moe:
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        
        # ⭐ 智能路由模式：加载所有专家的 postprocessor
        if cfg.smart_routing and cfg.task_type is None:
            logger.info(f"   ⭐ 智能路由模式: 加载所有专家的 postprocessor")
            expert_postprocessors = {}
            expert_names = ["narrower", "wider"]
            
            for expert_name in expert_names:
                expert_post_dir = Path(cfg.model_path) / "expert_postprocessors" / expert_name
                if expert_post_dir.exists():
                    try:
                        expert_postprocessors[expert_name] = PolicyProcessorPipeline.from_pretrained(
                            pretrained_model_name_or_path=str(expert_post_dir),
                            config_filename="policy_postprocessor.json",
                            overrides={},
                            to_transition=policy_action_to_transition,
                            to_output=transition_to_policy_action,
                        )
                        logger.info(f"      ✅ 已加载 {expert_name} 专家的 postprocessor")
                    except Exception as e:
                        logger.warning(f"      ⚠️ 加载 {expert_name} postprocessor 失败: {e}")
                else:
                    logger.warning(f"      ⚠️ {expert_name} postprocessor 目录不存在: {expert_post_dir}")
            
            if len(expert_postprocessors) == 2:
                use_smart_postprocessor = True
                logger.info(f"   ✅ 智能 postprocessor 选择已启用")
                logger.info(f"      根据每次推理的路由决策动态选择 postprocessor")
            else:
                logger.warning(f"   ⚠️ 无法加载所有专家 postprocessor，使用默认 postprocessor")
        
        # 固定任务类型模式：只加载指定专家的 postprocessor
        elif cfg.task_type is not None:
            expert_post_dir = Path(cfg.model_path) / "expert_postprocessors" / cfg.task_type
            if expert_post_dir.exists():
                logger.info(f"   ⭐ MoE 模式: 将使用 {cfg.task_type} 专家的 postprocessor")
                logger.info(f"      路径: {expert_post_dir}")
                try:
                    postprocessor = PolicyProcessorPipeline.from_pretrained(
                        pretrained_model_name_or_path=str(expert_post_dir),
                        config_filename="policy_postprocessor.json",
                        overrides={},
                        to_transition=policy_action_to_transition,
                        to_output=transition_to_policy_action,
                    )
                    logger.info(f"   ✅ 已加载 {cfg.task_type} 专家的 postprocessor")
                except Exception as e:
                    logger.warning(f"   ⚠️ 加载专家 postprocessor 失败: {e}")
                    logger.warning(f"      使用默认 postprocessor")
            else:
                logger.warning(f"   ⚠️ 专家 postprocessor 目录不存在: {expert_post_dir}")
                logger.warning(f"      使用默认 postprocessor（可能导致 {cfg.task_type} 任务不准确）")
    
    return ModelWrapper(
        name="mergevla_model",
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        adapter=adapter,
        expert_postprocessors=expert_postprocessors,
        use_smart_postprocessor=use_smart_postprocessor,
    )


def smooth_chunk_transition(
    new_chunk: torch.Tensor,
    prev_last_action: torch.Tensor | None,
    alpha: float = 0.3,
    transition_steps: int = 10,
) -> torch.Tensor:
    """
    平滑 chunk 之间的过渡，减少抖动
    
    Args:
        new_chunk: 新的 action chunk (chunk_size, action_dim)
        prev_last_action: 上一个 chunk 的最后一个 action (action_dim,)
        alpha: 平滑系数，越大越平滑
        transition_steps: 过渡的步数
    
    Returns:
        平滑后的 chunk
    """
    if prev_last_action is None:
        return new_chunk
    
    smoothed = new_chunk.clone()
    chunk_size = new_chunk.shape[0]
    transition_steps = min(transition_steps, chunk_size)
    
    for i in range(transition_steps):
        # 权重从 alpha 逐渐降到 0
        weight = alpha * (1 - i / transition_steps)
        smoothed[i] = (1 - weight) * new_chunk[i] + weight * prev_last_action
    
    return smoothed


def get_actions(
    model: ModelWrapper,
    env: GrabBoxMpcEnv,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: MergeVLAConfig,
    episode_state: EpisodeState,
):
    """推理线程"""
    try:
        latency_tracker = LatencyTracker()
        time_per_chunk = 1.0 / cfg.fps
        get_actions_threshold = cfg.action_queue_size_to_get_new_actions
        
        if not cfg.rtc.enabled:
            get_actions_threshold = 0
        
        # 用于 chunk 间平滑的状态
        prev_chunk_last_action = None
        inference_count = 0
        
        while not shutdown_event.is_set():
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()
                
                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk)
                
                # 获取观测
                obs_data, *_ = env.get_obs()
                state = torch.from_numpy(obs_data["state"]).float()
                observation = {'observation.state': state.to(cfg.device), 'task': cfg.task}
                
                camera_names = get_camera_names(CAMERA_COMPONENTS)
                for camera_name in camera_names:
                    if camera_name in obs_data:
                        camera_img_np = obs_data[camera_name]
                        if camera_img_np.ndim == 4:
                            camera_images = torch.from_numpy(
                                np.moveaxis(camera_img_np, 3, 1).copy()
                            ).float() / 255
                            obs_key = get_camera_observation_key(camera_name, use_image_features=False)
                            observation[obs_key] = camera_images.to(cfg.device)
                
                # 推理
                processed_observation = model.preprocessor(observation)
                actions = model.policy.predict_action_chunk(
                    processed_observation,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )
                
                original_actions = actions.squeeze(0).clone()
                
                # ⭐ 智能 postprocessor 选择：根据路由决策选择对应的 postprocessor
                # 推理后，adapter 中会记录最后一次路由决策
                postprocessor = model.get_postprocessor_for_task()
                
                # 后处理
                _, chunk_size, _ = actions.shape
                processed_actions = []
                for i in range(chunk_size):
                    processed_action = postprocessor(actions[:, i, :])
                    processed_actions.append(processed_action)
                
                postprocessed_actions = torch.stack(processed_actions, dim=1)[0].clone()
                
                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)
                
                # 诊断日志（前几次推理）
                inference_count += 1
                if inference_count <= 3:
                    logger.info(f"[INF #{inference_count}] latency={new_latency*1000:.1f}ms, "
                               f"delay={new_delay}, queue={action_queue.qsize()}, "
                               f"threshold={get_actions_threshold}")
                
                # 第一次推理平滑处理
                if episode_state.first_inference:
                    postprocessed_actions = apply_first_chunk_smooth(postprocessed_actions, obs_data, env)
                    episode_state.first_inference = False
                
                # ⚠️ 关键：chunk 间平滑（减少 MergeVLA 适配层导致的抖动）
                if cfg.action_smoothing and prev_chunk_last_action is not None:
                    postprocessed_actions = smooth_chunk_transition(
                        postprocessed_actions,
                        prev_chunk_last_action,
                        alpha=cfg.smoothing_alpha,
                        transition_steps=int(cfg.fps),  # 约 1 秒的过渡
                    )
                
                # 保存当前 chunk 的最后一个 action 用于下次平滑
                prev_chunk_last_action = postprocessed_actions[-1].clone()
                
                # 重采样
                postprocessed_resampled = resample_chunk_with_claw_hold(
                    postprocessed_actions.cpu().numpy(),
                    previous_action=(
                        episode_state.last_executed_action.cpu().numpy()
                        if episode_state.last_executed_action is not None else None
                    ),
                    control_frequency=100.0,
                    source_dt=0.1,
                    arm_dims=slice(0, 14),
                    claw_dims=slice(14, 16),
                    device=postprocessed_actions.device,
                )
                
                episode_state.last_executed_action = postprocessed_resampled[-get_actions_threshold].clone()
                action_queue.merge(original_actions, postprocessed_resampled, new_delay, action_index_before_inference)
            else:
                time.sleep(0.01)
        
    except Exception as e:
        traceback.print_exc()
        shutdown_event.set()


def actor_control(env: GrabBoxMpcEnv, action_queue: ActionQueue, shutdown_event: Event):
    """执行动作的线程"""
    try:
        action_interval = 1.0 / 100  # 100 Hz
        
        while not shutdown_event.is_set():
            start_time = time.perf_counter()
            
            if action_queue.qsize() > 0:
                action = action_queue.get().cpu()
                control_cmd_pose = ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS)
                env.exec_actions(actions=action, control_arm=True, control_claw=True, control_cmd_pose=control_cmd_pose)
            
            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, action_interval - dt_s - 0.001))
        
    except Exception as e:
        traceback.print_exc()
        shutdown_event.set()


@parser.wrap()
def main(cfg: MergeVLAConfig):
    """MergeVLA 融合模型实时推理入口"""
    
    init_logging()
    logger.info(f"[MAIN] Using device: {cfg.device}")
    
    rospy.init_node("eval_mergevla", anonymous=True)
    
    # 初始化环境
    env = GrabBoxMpcEnv()
    robot_sdk = RobotSDK()
    robot_sdk.control.set_external_control_arm_mode()
    robot_sdk.control.control_head(0, np.deg2rad(20))
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    final_reset_arm(
        json_path=os.path.join(cur_dir, 'utils/initial_arm_traj.json'),
        env=env, control_arm=True, control_claw=True
    )
    set_arm_quick_mode(True)
    
    # 加载模型
    model = load_mergevla_model(cfg)
    
    env.obs_buffer.wait_buffer_ready()
    
    from lerobot.rl.process import ProcessSignalHandler
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event
    
    print(f"\n{'='*60}")
    print(f"🤖 MergeVLA 实时推理")
    print(f"{'='*60}")
    print(f"   模型: {cfg.model_path}")
    print(f"   适配层: {'✅ 已启用' if model.adapter else '❌ 已禁用'}")
    print(f"   任务描述: {cfg.task}")
    print(f"{'='*60}")
    # ⭐ 任务路由配置
    if cfg.task_type:
        swap_info = " (映射已交换)" if cfg.swap_task_mapping else ""
        print(f"   🎯 任务路由: 固定 task_type={cfg.task_type}{swap_info}")
    elif cfg.smart_routing:
        print(f"   ⭐ 任务路由: MergeVLA 智能路由")
        if model.use_smart_postprocessor:
            print(f"   🔥 智能 postprocessor: 根据路由决策自动选择")
    else:
        print(f"   ⚠️ 任务路由: 简单平均（可能导致动作混乱！）")
        print(f"   💡 建议: 使用 --smart_routing=true 自动推断任务")
    print(f"{'='*60}")
    print(f"   推理频率: {cfg.fps} Hz")
    print(f"   推理周期: {cfg.duration} 秒")
    print(f"   RTC: {'✅ 启用' if cfg.rtc.enabled else '❌ 禁用'}")
    print(f"   RTC execution_horizon: {cfg.rtc.execution_horizon}")
    print(f"{'='*60}")
    print(f"   动作队列阈值: {cfg.action_queue_size_to_get_new_actions}")
    print(f"   chunk平滑: {'✅ 启用' if cfg.action_smoothing else '❌ 禁用'}")
    if cfg.action_smoothing:
        print(f"   平滑系数: {cfg.smoothing_alpha}")
    print(f"{'='*60}")
    print(f"💡 如果动作抖动，尝试调整以下参数：")
    print(f"   --action_queue_size_to_get_new_actions=30  (减小阈值)")
    print(f"   --smoothing_alpha=0.5                      (增大平滑)")
    print(f"{'='*60}")
    
    while True:
        input("\n按回车开始推理...")
        
        model.reset()
        shutdown_event.clear()
        episode_state = EpisodeState()
        action_queue = ActionQueue(cfg.rtc)
        
        get_actions_thread = Thread(
            target=get_actions,
            args=(model, env, action_queue, shutdown_event, cfg, episode_state),
            daemon=True
        )
        actor_thread = Thread(
            target=actor_control,
            args=(env, action_queue, shutdown_event),
            daemon=True
        )
        
        get_actions_thread.start()
        actor_thread.start()
        
        print(f"💡 推理中... {cfg.duration}秒后自动停止，或输入 'q' 手动停止")
        
        # 使用 duration 自动停止
        start_time = time.perf_counter()
        import select
        import sys as _sys
        
        while not shutdown_event.is_set():
            elapsed = time.perf_counter() - start_time
            if elapsed >= cfg.duration:
                logger.info(f"⏱️ 达到时间限制 ({cfg.duration}秒)，自动停止")
                break
            
            # 非阻塞检查用户输入
            if select.select([_sys.stdin], [], [], 0.1)[0]:
                user_input = _sys.stdin.readline().strip().lower()
                if user_input == 'q':
                    logger.info("👆 用户手动停止")
                    break
        
        shutdown_event.set()
        env.reset_claw_lock()
        final_reset_arm(
            json_path=os.path.join(cur_dir, 'utils/initial_arm_traj.json'),
            env=env, control_arm=True, control_claw=True
        )
        
        get_actions_thread.join()
        actor_thread.join()
        
        # ⭐ 打印智能任务路由统计（如果使用了 smart_routing）
        if cfg.smart_routing and model.adapter is not None:
            try:
                routing_stats = model.adapter.get_routing_stats()
                if routing_stats:
                    print(f"\n{'='*60}")
                    print(f"⭐ MergeVLA Smart Routing 统计")
                    print(f"{'='*60}")
                    print(f"   总调用次数: {routing_stats.get('total_calls', 0)}")
                    print(f"   倾向 Task 0 (narrower): {routing_stats.get('task_0', 0)} 次 "
                          f"({routing_stats.get('task_0_ratio', 0)*100:.1f}%)")
                    print(f"   倾向 Task 1 (wider): {routing_stats.get('task_1', 0)} 次 "
                          f"({routing_stats.get('task_1_ratio', 0)*100:.1f}%)")
                    print(f"   混合路由: {routing_stats.get('mixed', 0)} 次 "
                          f"({routing_stats.get('mixed_ratio', 0)*100:.1f}%)")
                    print(f"{'='*60}")
                    
                    # 重置路由统计（为下一次推理准备）
                    model.adapter.reset_routing_stats()
            except AttributeError:
                # 适配层可能没有路由统计方法
                pass
        
        logger.info("✅ 本轮推理完成")


if __name__ == "__main__":
    main()
