#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
π*0.6 (RECAP) Value Function 实现

基于论文: π*₀.₆: a VLA That Learns From Experience

Value Function 架构:
- 视觉编码器: SigLIP (400M) - 可与策略网络共享权重
- 语言模型: Gemma (270M/300M) - 比策略网络小得多（策略网络是 2B/4B）
- 价值头 (Value Head): MLP，输出单个标量值 V(s)

训练方式: 监督学习 (MSE 损失)
目标值: 累积奖励 (Return)
"""

import logging
import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    PaliGemmaForConditionalGeneration = None

from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE


def _get_gemma_config(variant: str):
    """获取 Gemma 配置（避免循环导入）。"""
    from lerobot.policies.pi05.modeling_pi05 import get_gemma_config
    return get_gemma_config(variant)


def _make_att_2d_masks(pad_masks, att_masks):
    """构造 2D 注意力掩码（避免循环导入）。"""
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    return make_att_2d_masks(pad_masks, att_masks)


# ============================================================================
# Value Function 模型
# ============================================================================


class ValueHeadMLP(nn.Module):
    """MLP Value Head，将深度表示映射为单个标量值 V(s)。

    架构: hidden_dim -> hidden_dim//2 -> hidden_dim//4 -> 1
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, hidden_dim) 深度表示
        Returns:
            (batch_size,) 标量值
        """
        return self.net(x).squeeze(-1)


class ValueFunctionPytorch(nn.Module):
    """π*0.6 Value Function 核心模型。

    使用较小的 VLA 模型预测状态价值 V(s, ℓ)。

    架构:
    - SigLIP 视觉编码器 (400M): 编码图像输入
    - Gemma 语言模型 (270M/300M): 处理多模态融合特征
    - MLP Value Head: 输出标量价值

    输入: 观察 (图像 + 语言指令)
    输出: 标量价值 V(s) ∈ R (训练后通常在 [0, 1] 范围内)
    """

    def __init__(
        self,
        vlm_variant: str = "gemma_300m",
        image_resolution: tuple[int, int] = (224, 224),
        precision: Literal["bfloat16", "float32"] = "float32",
        value_head_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        share_vision_encoder: bool = False,
        policy_paligemma: PaliGemmaForConditionalGeneration | None = None,
    ):
        """
        Args:
            vlm_variant: Gemma 变体名称 ("gemma_300m" 用于 Value Function)
            image_resolution: 图像分辨率
            precision: 精度 ("bfloat16" 或 "float32")
            value_head_dropout: Value Head MLP 的 dropout 率
            gradient_checkpointing: 是否启用梯度检查点
            share_vision_encoder: 是否与策略网络共享视觉编码器
            policy_paligemma: 策略网络的 PaliGemma 模型（用于共享视觉编码器）
        """
        super().__init__()
        self.vlm_variant = vlm_variant
        self.image_resolution = image_resolution
        self.gradient_checkpointing_enabled = gradient_checkpointing
        self.share_vision_encoder = share_vision_encoder

        # 获取 Gemma 配置（较小的 270M/300M 模型）
        vlm_config = _get_gemma_config(vlm_variant)
        self.hidden_dim = vlm_config.width

        # 创建 PaliGemma 模型（包含 SigLIP + 小型 Gemma）
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = False
        vlm_config_hf.text_config.adarms_cond_dim = None
        vlm_config_hf.vision_config.intermediate_size = 4304
        # projection_dim 必须与 Gemma text model 的 hidden_size 一致，
        # 否则 SigLIP 输出 (2048) 和 Gemma embed_tokens 输出 (1024) 无法 concat
        vlm_config_hf.vision_config.projection_dim = vlm_config.width
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)

        # 如果共享视觉编码器，替换为策略网络的视觉编码器
        if share_vision_encoder and policy_paligemma is not None:
            logging.info("Value Function: 共享策略网络的视觉编码器 (SigLIP)")
            self.paligemma.model.vision_tower = policy_paligemma.model.vision_tower
            # 冻结共享的视觉编码器（可选）
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False

        # Value Head MLP
        self.value_head = ValueHeadMLP(
            hidden_dim=self.hidden_dim,
            dropout=value_head_dropout,
        )

        # 设置精度
        self._apply_precision(precision)

    def _apply_precision(self, precision: str):
        """设置模型精度。"""
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
            # 保持某些参数为 float32
            params_to_keep_float32 = [
                "vision_tower.vision_model.embeddings.patch_embedding.weight",
                "vision_tower.vision_model.embeddings.patch_embedding.bias",
                "vision_tower.vision_model.embeddings.position_embedding.weight",
                "input_layernorm",
                "post_attention_layernorm",
                "model.norm",
            ]
            for name, param in self.named_parameters():
                if any(selector in name for selector in params_to_keep_float32):
                    param.data = param.data.to(dtype=torch.float32)
            # Value Head 始终使用 float32
            self.value_head.to(dtype=torch.float32)
        elif precision == "float32":
            self.to(dtype=torch.float32)

    def gradient_checkpointing_enable(self):
        """启用梯度检查点以优化内存。"""
        self.gradient_checkpointing_enabled = True
        self.paligemma.language_model.gradient_checkpointing = True
        self.paligemma.vision_tower.gradient_checkpointing = True
        logging.info("Value Function: 已启用梯度检查点")

    def gradient_checkpointing_disable(self):
        """禁用梯度检查点。"""
        self.gradient_checkpointing_enabled = False
        self.paligemma.language_model.gradient_checkpointing = False
        self.paligemma.vision_tower.gradient_checkpointing = False
        logging.info("Value Function: 已禁用梯度检查点")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """辅助方法：应用梯度检查点。"""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像特征。"""
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """编码语言 token。"""
        return self.paligemma.language_model.embed_tokens(tokens)

    def embed_prefix(
        self, images: list[Tensor], img_masks: list[Tensor], tokens: Tensor, masks: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """编码图像和语言 token（与 PI05Pytorch.embed_prefix 类似）。

        Args:
            images: 图像列表 [(B, C, H, W), ...]
            img_masks: 图像掩码列表 [(B,), ...]
            tokens: 语言 token (B, seq_len)
            masks: 语言掩码 (B, seq_len)

        Returns:
            embs: 拼接后的嵌入 (B, total_len, hidden_dim)
            pad_masks: 填充掩码 (B, total_len)
            att_masks: 注意力掩码 (B, total_len)
        """
        embs = []
        pad_masks = []
        att_masks = []

        # 处理图像
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # 处理语言 token
        def lang_embed_func(tokens):
            lang_emb = self.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
    ) -> Tensor:
        """前向传播：预测状态价值 V(s, ℓ)。

        Args:
            images: 图像列表 [(B, C, H, W), ...]
            img_masks: 图像掩码列表 [(B,), ...]
            tokens: 语言 token (B, seq_len)
            masks: 语言掩码 (B, seq_len)

        Returns:
            values: (B,) 预测的状态价值
        """
        # 1. 编码图像和语言 token
        embs, pad_masks, att_masks = self.embed_prefix(images, img_masks, tokens, masks)

        # 2. 构造注意力掩码
        att_2d_masks = _make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        att_2d_masks_4d = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

        # 确保嵌入与模型权重精度匹配
        if self.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            embs = embs.to(dtype=torch.bfloat16)

        # 3. 通过语言模型处理
        def lm_forward_func(embs, att_2d_masks_4d, position_ids):
            output = self.paligemma.language_model.forward(
                inputs_embeds=embs,
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                adarms_cond=None,
            )
            return output.last_hidden_state

        hidden_states = self._apply_checkpoint(
            lm_forward_func, embs, att_2d_masks_4d, position_ids
        )

        # 4. 池化操作: 对有效 token 做 mean pooling
        # hidden_states: (B, seq_len, hidden_dim)
        pad_masks_expanded = pad_masks.unsqueeze(-1).float()  # (B, seq_len, 1)
        hidden_states_float = hidden_states.to(dtype=torch.float32)
        pooled = (hidden_states_float * pad_masks_expanded).sum(dim=1) / (
            pad_masks_expanded.sum(dim=1).clamp(min=1.0)
        )  # (B, hidden_dim)

        # 5. Value Head: 映射为标量值
        values = self.value_head(pooled)  # (B,)

        return values

    def compute_value_loss(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        target_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """计算 Value Function 的 MSE 损失。

        训练公式:
            V^π(o_t, ℓ) ← argmin_V E[(V(o_t, ℓ) - Σ_{t'=t}^T r_{t'})²]

        Args:
            images: 图像列表
            img_masks: 图像掩码列表
            tokens: 语言 token
            masks: 语言掩码
            target_values: 目标价值 (B,) - 累积奖励 Return

        Returns:
            loss: MSE 损失标量
            predicted_values: (B,) 预测的价值
        """
        predicted_values = self.forward(images, img_masks, tokens, masks)
        loss = F.mse_loss(predicted_values, target_values.float())
        return loss, predicted_values


# ============================================================================
# Advantage 计算器
# ============================================================================


class AdvantageComputer:
    """Advantage 计算和二值化工具。

    支持两种计算模式:
    1. 预训练模式 (N=T): 使用整个 episode 的累积奖励
    2. 微调模式 (N=50): 使用固定 N-step lookahead

    公式:
    - 预训练: A(o_t, a_t, ℓ) = Σ_{t'=t}^T r_{t'} - V^π(o_t, ℓ)
    - 微调:   A(o_t, a_t, ℓ) = Σ_{t'=t}^{t+N-1} r_{t'} + V^π(o_{t+N}, ℓ) - V^π(o_t, ℓ)
    """

    def __init__(
        self,
        n_steps: int | None = None,
        positive_advantage_ratio: float = 0.3,
        advantage_dropout: float = 0.3,
    ):
        """
        Args:
            n_steps: lookahead 步数。
                - None: 预训练模式，使用整个 episode (N=T)
                - int: 微调模式，使用固定 N-step lookahead (如 N=50)
            positive_advantage_ratio: 正优势数据的目标比例
                - 预训练: 约 0.3 (30%)
                - 微调: 约 0.4 (40%)
            advantage_dropout: Advantage conditioning 的 dropout 比例 (默认 0.3)
        """
        self.n_steps = n_steps
        self.positive_advantage_ratio = positive_advantage_ratio
        self.advantage_dropout = advantage_dropout
        self._threshold = None

    @staticmethod
    def compute_episode_returns(
        rewards: Tensor,
        n_steps: int | None = None,
        values: Tensor | None = None,
    ) -> Tensor:
        """计算每个时间步的 Return (累积奖励)。

        Args:
            rewards: (T,) 或 (B, T) 每个时间步的奖励
            n_steps: lookahead 步数，None 表示使用整个 episode
            values: (T,) 或 (B, T) Value Function 的预测值（微调模式需要）

        Returns:
            returns: 与 rewards 同形状的 Return 张量
        """
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, T = rewards.shape
        returns = torch.zeros_like(rewards)

        if n_steps is None:
            # 预训练模式: N=T，从后向前累加
            cumsum = torch.zeros(B, device=rewards.device)
            for t in range(T - 1, -1, -1):
                cumsum = cumsum + rewards[:, t]
                returns[:, t] = cumsum
        else:
            # 微调模式: 固定 N-step lookahead
            for t in range(T):
                end_idx = min(t + n_steps, T)
                immediate_return = rewards[:, t:end_idx].sum(dim=1)

                if t + n_steps < T and values is not None:
                    # 加上 N 步后的状态价值
                    future_value = values[:, t + n_steps]
                    returns[:, t] = immediate_return + future_value
                else:
                    returns[:, t] = immediate_return

        if squeeze:
            returns = returns.squeeze(0)

        return returns

    def compute_advantages(
        self,
        returns: Tensor,
        predicted_values: Tensor,
    ) -> Tensor:
        """计算 Advantage。

        A(o_t, a_t, ℓ) = Return_t - V^π(o_t, ℓ)

        Args:
            returns: (B, T) 或 (B,) 每个时间步的 Return
            predicted_values: (B, T) 或 (B,) Value Function 的预测值

        Returns:
            advantages: 连续值 Advantage
        """
        return returns - predicted_values

    def compute_threshold(self, advantages: Tensor) -> float:
        """动态计算 Advantage 阈值。

        使用百分位数方法:
        - 预训练: 约 30% 正优势 → 70th percentile
        - 微调: 约 40% 正优势 → 60th percentile

        Args:
            advantages: 所有数据的连续 Advantage 值

        Returns:
            threshold: Advantage 阈值 ε_ℓ
        """
        target_percentile = (1.0 - self.positive_advantage_ratio) * 100.0
        advantages_flat = advantages.flatten().detach().cpu().numpy()
        self._threshold = float(np.percentile(advantages_flat, target_percentile))
        return self._threshold

    def binarize_advantages(
        self,
        advantages: Tensor,
        threshold: float | None = None,
    ) -> Tensor:
        """将连续 Advantage 二值化。

        I_t = 1 if A(o_t, a_t, ℓ) > ε_ℓ, 0 otherwise

        Args:
            advantages: 连续 Advantage 值
            threshold: 阈值 ε_ℓ，None 则使用最近计算的阈值

        Returns:
            indicators: 二值化的 Advantage 指标 I_t ∈ {0, 1}
        """
        if threshold is None:
            if self._threshold is None:
                threshold = self.compute_threshold(advantages)
            else:
                threshold = self._threshold

        return (advantages > threshold).long()

    def apply_dropout(
        self,
        advantage_indicators: Tensor,
        training: bool = True,
    ) -> Tensor:
        """对 Advantage Indicator 应用 Dropout（用于 CFG 训练）。

        训练时随机将 30% 的 advantage conditioning 设为 -1（表示无条件）。
        这样在推理时可以使用 Classifier-Free Guidance (CFG)。

        Args:
            advantage_indicators: (B,) 或 (B, T) 二值化的 Advantage 指标
            training: 是否处于训练模式

        Returns:
            indicators: 应用 dropout 后的指标（-1 表示无条件）
        """
        if not training:
            return advantage_indicators

        indicators = advantage_indicators.clone()
        dropout_mask = torch.rand_like(indicators.float()) < self.advantage_dropout
        indicators[dropout_mask] = -1  # -1 表示无条件（dropout）
        return indicators


# ============================================================================
# Advantage Conditioning 嵌入
# ============================================================================


class AdvantageConditioningEmbedding(nn.Module):
    """Advantage Conditioning 嵌入模块。

    将二值化的 Advantage Indicator I_t ∈ {0, 1, -1} 转换为嵌入向量，
    作为策略网络的额外条件输入。

    嵌入表:
    - I_t = 0: 低优势嵌入（差的行为）
    - I_t = 1: 高优势嵌入（好的行为）
    - I_t = -1: 无条件嵌入（dropout，用于 CFG）
    """

    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: 嵌入维度（应与策略网络的 hidden_dim 匹配）
        """
        super().__init__()
        # 3 个嵌入: 低优势(0), 高优势(1), 无条件(-1)
        self.embedding = nn.Embedding(3, embed_dim)
        # 投影层（可选，用于将嵌入投影到策略网络的维度）
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, advantage_indicators: Tensor) -> Tensor:
        """
        Args:
            advantage_indicators: (B,) Advantage 指标，值为 {-1, 0, 1}
                - 0: 低优势
                - 1: 高优势
                - -1: 无条件（dropout）

        Returns:
            embeddings: (B, embed_dim) Advantage 条件嵌入
        """
        # 将 {-1, 0, 1} 映射到 {0, 1, 2} 用于 nn.Embedding 索引
        indices = (advantage_indicators + 1).long().clamp(0, 2)
        emb = self.embedding(indices)
        emb = self.proj(emb)
        return emb


# ============================================================================
# 辅助函数
# ============================================================================


def compute_n_step_returns_for_episode(
    rewards: list[float] | np.ndarray | Tensor,
    value_predictions: list[float] | np.ndarray | Tensor | None = None,
    n_steps: int | None = None,
) -> Tensor:
    """计算单个 Episode 的 N-step Return。

    预训练 (n_steps=None, N=T):
        R_t = Σ_{t'=t}^T r_{t'}

    微调 (n_steps=50):
        R_t = Σ_{t'=t}^{t+N-1} r_{t'} + V(o_{t+N})

    Args:
        rewards: 单个 episode 的奖励序列
        value_predictions: Value Function 对每个状态的预测值（微调时需要）
        n_steps: lookahead 步数，None=使用整个 episode

    Returns:
        returns: 每个时间步的 Return
    """
    if isinstance(rewards, (list, np.ndarray)):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    if isinstance(value_predictions, (list, np.ndarray)):
        value_predictions = torch.tensor(value_predictions, dtype=torch.float32)

    T = len(rewards)
    returns = torch.zeros(T, dtype=torch.float32)

    if n_steps is None:
        # 预训练模式: 从后向前累加
        cumsum = 0.0
        for t in range(T - 1, -1, -1):
            cumsum += rewards[t].item()
            returns[t] = cumsum
    else:
        for t in range(T):
            end_idx = min(t + n_steps, T)
            immediate_return = rewards[t:end_idx].sum().item()

            if t + n_steps < T and value_predictions is not None:
                future_value = value_predictions[t + n_steps].item()
                returns[t] = immediate_return + future_value
            else:
                returns[t] = immediate_return

    return returns


def prepare_value_training_data(
    episodes: list[dict],
    n_steps: int | None = None,
    value_function: ValueFunctionPytorch | None = None,
) -> list[dict]:
    """准备 Value Function 的训练数据。

    对于每个 episode，计算每个时间步的目标值 (Return)。

    Args:
        episodes: episode 列表，每个包含:
            - "observations": 观察序列
            - "rewards": 奖励序列 [0, 0, ..., 0, 1] (sparse)
            - "task_label": 任务标签
        n_steps: lookahead 步数
        value_function: 微调时需要提供 Value Function（用于计算 N-step return）

    Returns:
        training_data: 训练数据列表，每个包含:
            - "observation": 单个时间步的观察
            - "task_label": 任务标签
            - "target_value": 目标值 (Return)
    """
    training_data = []

    for episode in episodes:
        rewards = episode["rewards"]
        T = len(rewards)

        if isinstance(rewards, (list, np.ndarray)):
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        else:
            rewards_tensor = rewards.float()

        # 如果需要 Value Function 预测（微调模式）
        value_predictions = None
        if n_steps is not None and value_function is not None:
            # 这里需要对每个时间步的观察做 Value Function 推理
            # 实际使用时应该批量处理
            value_predictions = episode.get("value_predictions", None)

        # 计算 Return
        returns = compute_n_step_returns_for_episode(
            rewards_tensor, value_predictions, n_steps
        )

        for t in range(T):
            training_data.append({
                "observation": episode["observations"][t] if "observations" in episode else None,
                "task_label": episode.get("task_label", ""),
                "target_value": returns[t].item(),
                "episode_return": rewards_tensor.sum().item(),
            })

    return training_data
