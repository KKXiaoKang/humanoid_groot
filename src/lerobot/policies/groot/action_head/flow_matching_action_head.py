# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.distributions import Beta

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import PretrainedConfig
    from transformers.feature_extraction_utils import BatchFeature
else:
    PretrainedConfig = object
    BatchFeature = None

from lerobot.policies.groot.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer
from typing_extensions import Unpack
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


class ReasoningHead(nn.Module):
    """
    Chain of Causation (CoC) Reasoning Head
    
    实现真正的Chain of Causation推理链：
    1. 从backbone_features生成reasoning trace（思维链）
    2. 基于reasoning trace生成action decision（动作决策）
    3. 使用reasoning conditioning指导动作生成
    
    这是真正的因果关系链：backbone → reasoning trace → action decision → action
    
    支持6种action decision类型：
    1. left_search_grasp_pull: 机器人移动左手寻找箱子左侧边缘，夹爪抓取后并拉开，右手保持不动
    2. left_hold_right_search_grasp: 机器人左手抓住箱子边缘保持不动，右手找到箱子的边缘并且抓住
    3. right_search_grasp_pull: 机器人移动右手寻找箱子右侧边缘，夹爪抓取后并拉开，左手保持不动
    4. right_hold_left_search_grasp: 机器人右手抓住箱子边缘保持不动，左手找到箱子的边缘并且抓住
    5. both_search_grasp: 机器人左右手同时找到箱子的左右边缘，并且抓取
    6. both_hold_lift: 机器人左手右手已经抓住箱子边缘，同时上抬提起箱子
    
    关键设计：
    - 训练时：使用ground truth reasoning labels，基于reasoning trace生成action decision
    - 推理时：自回归生成reasoning trace，然后基于生成的reasoning trace生成action decision
    - 这确保了reasoning trace和action decision之间的因果关系，符合Chain of Causation的设计理念
    """
    def __init__(
        self,
        backbone_embedding_dim: int,
        reasoning_hidden_dim: int,
        reasoning_vocab_size: int,
        reasoning_max_length: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.reasoning_hidden_dim = reasoning_hidden_dim
        self.reasoning_vocab_size = reasoning_vocab_size
        self.reasoning_max_length = reasoning_max_length
        
        # 将backbone特征投影到reasoning空间
        self.backbone_proj = nn.Linear(backbone_embedding_dim, reasoning_hidden_dim)
        
        # 小型Transformer用于生成reasoning tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=reasoning_hidden_dim,
            nhead=8,
            dim_feedforward=reasoning_hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.reasoning_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Token embedding和位置编码
        self.token_embedding = nn.Embedding(reasoning_vocab_size, reasoning_hidden_dim)
        self.position_embedding = nn.Embedding(reasoning_max_length, reasoning_hidden_dim)
        
        # 输出层：生成reasoning tokens的logits
        self.output_proj = nn.Linear(reasoning_hidden_dim, reasoning_vocab_size)
        
        # 条件化embedding：将reasoning tokens编码为条件向量，用于指导动作生成
        self.conditioning_proj = nn.Linear(reasoning_hidden_dim, reasoning_hidden_dim)
        
        # Action decision prediction: 预测action decision类型
        self.action_decision_predictor = nn.Sequential(
            nn.Linear(reasoning_hidden_dim, reasoning_hidden_dim),
            nn.ReLU(),
            nn.Linear(reasoning_hidden_dim, 6),  # 6种决策类型
        )
        
        # Action decision embedding: 将action decision类型编码为条件向量
        # 用于直接指导decoder的动作生成方向
        self.action_decision_embedding = nn.Embedding(6, reasoning_hidden_dim)  # 6种决策类型
        
    def forward(
        self, 
        backbone_features: torch.Tensor, 
        reasoning_labels: torch.Tensor | None = None,
        action_decision_labels: torch.Tensor | None = None,
    ):
        """
        Args:
            backbone_features: (B, T, backbone_embedding_dim) - 来自backbone的特征
            reasoning_labels: (B, L) - 可选的ground truth reasoning token ids，用于训练
            action_decision_labels: (B,) - 可选的ground truth action decision labels，用于训练
        
        Returns:
            reasoning_logits: (B, L, vocab_size) - reasoning tokens的logits
            reasoning_conditioning: (B, reasoning_hidden_dim) - 用于条件化动作生成的向量（融合了action decision信息）
            action_decision_logits: (B, 6) - action decision类型的logits (6种决策类型)
        """
        B, T, _ = backbone_features.shape
        
        # 1. 投影backbone特征
        backbone_proj = self.backbone_proj(backbone_features)  # (B, T, reasoning_hidden_dim)
        
        # 2. 聚合backbone特征（使用平均池化或CLS token）
        # 使用平均池化得到全局表示
        backbone_global = backbone_proj.mean(dim=1)  # (B, reasoning_hidden_dim)
        
        # 3. 生成reasoning tokens
        reasoning_output = None  # 用于后续生成action decision
        if reasoning_labels is not None:
            # 训练模式：使用ground truth labels
            L = reasoning_labels.shape[1]  # reasoning sequence length
            token_embeds = self.token_embedding(reasoning_labels)  # (B, L, reasoning_hidden_dim)
            
            # 添加位置编码
            pos_ids = torch.arange(L, device=reasoning_labels.device).unsqueeze(0).expand(B, -1)
            pos_embeds = self.position_embedding(pos_ids)
            token_embeds = token_embeds + pos_embeds
            
            # 将backbone全局特征作为初始token
            # 拼接: [backbone_global, token_embeds]
            reasoning_input = torch.cat([backbone_global.unsqueeze(1), token_embeds], dim=1)  # (B, 1+L, reasoning_hidden_dim)
            
            # 通过Transformer
            reasoning_output = self.reasoning_transformer(reasoning_input)  # (B, 1+L, reasoning_hidden_dim)
            
            # 只取token部分（不包括backbone_global）
            reasoning_output = reasoning_output[:, 1:]  # (B, L, reasoning_hidden_dim)
            
            # 生成logits
            reasoning_logits = self.output_proj(reasoning_output)  # (B, L, vocab_size)
        else:
            # 推理模式：自回归生成reasoning trace
            # 这是真正的Chain of Causation：从backbone特征生成reasoning trace
            reasoning_logits, reasoning_output = self._generate_reasoning_autoregressive(
                backbone_global, max_length=self.reasoning_max_length
            )
        
        # 4. 生成action decision logits（基于reasoning trace，而不是直接基于backbone）
        # 这是Chain of Causation的关键：action decision应该基于reasoning trace生成
        if reasoning_output is not None:
            # 使用reasoning trace的聚合特征来预测action decision
            reasoning_aggregated = reasoning_output.mean(dim=1)  # (B, reasoning_hidden_dim)
            action_decision_logits = self._predict_action_decision(reasoning_aggregated)  # (B, 6)
        else:
            # 如果没有reasoning trace，回退到backbone特征（用于训练初期或兼容性）
            action_decision_logits = self._predict_action_decision(backbone_global)  # (B, 6)
        
        # 5. 生成reasoning conditioning向量（用于条件化动作生成）
        # 关键改进：将action decision的信息融入到conditioning中，使其能够真正引导动作生成
        if reasoning_output is not None:
            # 使用reasoning trace的聚合特征（平均池化）来生成基础conditioning
            reasoning_aggregated = reasoning_output.mean(dim=1)  # (B, reasoning_hidden_dim)
            base_conditioning = self.conditioning_proj(reasoning_aggregated)  # (B, reasoning_hidden_dim)
        else:
            # 如果没有reasoning trace，使用backbone特征
            base_conditioning = self.conditioning_proj(backbone_global)  # (B, reasoning_hidden_dim)
        
        # 将action decision的embedding融入到conditioning中
        # 这是关键：让action decision真正引导动作生成
        # 在训练时，优先使用ground truth action_decision_labels（teacher forcing）
        # 在推理时，使用预测的action_decision_logits
        if action_decision_labels is not None:
            # 训练时：使用ground truth action_decision_labels（teacher forcing）
            # 这确保了训练时conditioning使用的是正确的action decision
            action_decision_idx = action_decision_labels  # (B,)
            action_decision_emb = self.action_decision_embedding(action_decision_idx)  # (B, reasoning_hidden_dim)
        elif action_decision_logits is not None:
            # 推理时：使用预测的action_decision_logits
            predicted_decision_idx = torch.argmax(action_decision_logits, dim=-1)  # (B,)
            action_decision_emb = self.action_decision_embedding(predicted_decision_idx)  # (B, reasoning_hidden_dim)
        else:
            # 如果没有action decision信息，只使用base conditioning
            action_decision_emb = None
        
        # 将action decision embedding与base conditioning融合
        # 使用残差连接，让action decision的信息直接注入到conditioning中
        # 这样action decision就能真正引导DiT的动作生成方向
        if action_decision_emb is not None:
            reasoning_conditioning = base_conditioning + action_decision_emb  # (B, reasoning_hidden_dim)
        else:
            reasoning_conditioning = base_conditioning
        
        return reasoning_logits, reasoning_conditioning, action_decision_logits
    
    def _predict_action_decision(self, features: torch.Tensor) -> torch.Tensor:
        """预测action decision类型"""
        return self.action_decision_predictor(features)
    
    def _generate_reasoning_autoregressive(
        self, 
        backbone_global: torch.Tensor, 
        max_length: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        自回归生成reasoning trace
        
        Args:
            backbone_global: (B, reasoning_hidden_dim) - backbone的全局特征
            max_length: 最大生成长度
            temperature: 采样温度
        
        Returns:
            reasoning_logits: (B, L, vocab_size) - 最后一个token的logits（用于损失计算，推理时可能为None）
            reasoning_output: (B, L, reasoning_hidden_dim) - 生成的reasoning trace的隐藏状态
        """
        B = backbone_global.shape[0]
        device = backbone_global.device
        
        # 初始化：从backbone_global开始
        current_input = backbone_global.unsqueeze(1)  # (B, 1, reasoning_hidden_dim)
        generated_tokens = []
        generated_embeds = []
        
        # 自回归生成
        for step in range(max_length):
            # 通过Transformer处理当前序列
            reasoning_output_step = self.reasoning_transformer(current_input)  # (B, seq_len, reasoning_hidden_dim)
            
            # 取最后一个token的输出（用于预测下一个token）
            last_token_output = reasoning_output_step[:, -1:]  # (B, 1, reasoning_hidden_dim)
            
            # 生成下一个token的logits
            next_token_logits = self.output_proj(last_token_output)  # (B, 1, vocab_size)
            
            # 采样下一个token（使用greedy decoding或temperature sampling）
            if temperature == 0.0:
                # Greedy decoding
                next_token_id = torch.argmax(next_token_logits, dim=-1)  # (B, 1)
            else:
                # Temperature sampling
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs.squeeze(1), num_samples=1).unsqueeze(1)  # (B, 1)
            
            # 检查是否遇到结束token（这里假设0是结束token，实际应该根据vocab定义）
            # 简化实现：如果生成的token是0，则停止（实际应该使用专门的结束token，如EOS token）
            if (next_token_id == 0).all():
                break
            
            generated_tokens.append(next_token_id)
            
            # 将新生成的token embedding添加到输入中
            next_token_embed = self.token_embedding(next_token_id.squeeze(1))  # (B, reasoning_hidden_dim)
            pos_embed = self.position_embedding(
                torch.full((B,), step + 1, device=device, dtype=torch.long)
            )  # (B, reasoning_hidden_dim)
            next_token_embed = next_token_embed + pos_embed.unsqueeze(1)  # (B, 1, reasoning_hidden_dim)
            
            # 更新输入：拼接新生成的token
            current_input = torch.cat([current_input, next_token_embed], dim=1)  # (B, seq_len+1, reasoning_hidden_dim)
        
        # 重新通过Transformer处理完整序列，获取所有token的隐藏状态
        # 这样可以得到完整的reasoning trace表示，用于后续的action decision预测
        if len(generated_tokens) > 0:
            # 重新处理完整序列以获取所有token的隐藏状态
            reasoning_output = self.reasoning_transformer(current_input)  # (B, 1+L, reasoning_hidden_dim)
            # 只取生成的token部分（不包括初始的backbone_global）
            reasoning_output = reasoning_output[:, 1:]  # (B, L, reasoning_hidden_dim)
        else:
            # 如果没有生成任何token，使用backbone_global
            reasoning_output = backbone_global.unsqueeze(1)  # (B, 1, reasoning_hidden_dim)
        
        # 推理时不需要返回logits（因为已经采样了），但为了接口一致性，返回None
        reasoning_logits = None
        
        return reasoning_logits, reasoning_output
    
    def get_action_decision_embedding(self, decision_type: str) -> torch.Tensor:
        """
        获取action decision类型的embedding
        
        Args:
            decision_type: 6种决策类型之一：
                - "left_search_grasp_pull": 左手搜索抓取拉开，右手不动
                - "left_hold_right_search_grasp": 左手保持，右手搜索抓取
                - "right_search_grasp_pull": 右手搜索抓取拉开，左手不动
                - "right_hold_left_search_grasp": 右手保持，左手搜索抓取
                - "both_search_grasp": 双手同时搜索抓取
                - "both_hold_lift": 双手保持并上抬
        
        Returns:
            embedding: (reasoning_hidden_dim,) - action decision的embedding向量
        """
        decision_map = {
            "left_search_grasp_pull": 0,
            "left_hold_right_search_grasp": 1,
            "right_search_grasp_pull": 2,
            "right_hold_left_search_grasp": 3,
            "both_search_grasp": 4,
            "both_hold_lift": 5,
        }
        if decision_type not in decision_map:
            raise ValueError(
                f"Unknown decision type: {decision_type}. "
                f"Valid types: {list(decision_map.keys())}"
            )
        idx = decision_map[decision_type]
        return self.action_decision_embedding(torch.tensor(idx))


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_w = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_w) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class SharedBottomArmDecoder(nn.Module):
    """
    共享底层特征的左右手decoder，提升协调性
    
    注意：如果 use_cross_attention=False，这个方案在参数上几乎等价于
    "合成一个MLP输出14维然后split"，主要区别是：
    1. 输出层分离，可以分别控制左右手的损失权重
    2. 可以分别学习不同的输出映射
    
    真正的价值在于启用交叉注意力（use_cross_attention=True），
    让左右手特征能够相互关注，这是"合成一个MLP"无法实现的。
    """
    def __init__(self, num_categories, input_dim, hidden_dim, left_output_dim, right_output_dim, use_cross_attention=False):
        super().__init__()
        self.num_categories = num_categories
        self.use_cross_attention = use_cross_attention
        
        # 共享的底层特征提取层
        # 注意：如果只是共享底层，确实和"合成一个MLP然后split"类似
        # 但输出层分离允许分别控制损失权重和学习不同的映射
        self.shared_layer = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        
        # 左右手各自的输出层
        # 这是和"合成一个MLP"的主要区别：输出层分离
        self.left_output_layer = CategorySpecificLinear(num_categories, hidden_dim, left_output_dim)
        self.right_output_layer = CategorySpecificLinear(num_categories, hidden_dim, right_output_dim)
        
        # 交叉注意力机制：这是真正的价值所在
        # 让左右手特征能够相互关注，这是"合成一个MLP"无法实现的
        if use_cross_attention:
            # 简单的交叉注意力：左右手特征相互关注
            self.cross_attn_left = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.cross_attn_right = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.layer_norm_left = nn.LayerNorm(hidden_dim)
            self.layer_norm_right = nn.LayerNorm(hidden_dim)
            print(f"   ✅ Cross-attention enabled: left↔right arm features can attend to each other")
        else:
            print(f"   ⚠️  Cross-attention disabled: This is similar to 'single MLP then split'")
            print(f"      Main difference: separate output layers allow different loss weights")
    
    def forward(self, x, cat_ids):
        """
        x: (B, T, input_dim)
        cat_ids: (B,)
        returns: (left_features, right_features) 或 (left_output, right_output)
        """
        # 共享底层特征提取
        shared_features = F.relu(self.shared_layer(x, cat_ids))  # (B, T, hidden_dim)
        
        if self.use_cross_attention:
            # 交叉注意力：左右手特征相互关注
            # 这是真正的价值：让左右手能够感知对方的状态
            # 这是"合成一个MLP然后split"无法实现的
            # 使用对称的交叉注意力，确保信息交换的一致性
            left_features = self.layer_norm_left(shared_features)
            right_features = self.layer_norm_right(shared_features)
            
            # 对称的交叉注意力：同时计算，避免信息不对称
            # 左手的query关注右手的key/value（使用原始right_features）
            left_attended, _ = self.cross_attn_left(
                left_features, right_features, right_features
            )
            # 右手的query关注左手的key/value（使用原始left_features）
            right_attended, _ = self.cross_attn_right(
                right_features, left_features, left_features
            )
            
            # 残差连接：保持原始特征，只添加注意力信息
            left_features = left_features + left_attended
            right_features = right_features + right_attended
            
            # 输出层
            left_output = self.left_output_layer(left_features, cat_ids)
            right_output = self.right_output_layer(right_features, cat_ids)
        else:
            # 不使用交叉注意力，直接输出
            # 注意：这种情况下，确实和"合成一个MLP然后split"类似
            # 主要区别是输出层分离，可以分别控制损失权重
            left_output = self.left_output_layer(shared_features, cat_ids)
            right_output = self.right_output_layer(shared_features, cat_ids)
        
        return left_output, right_output


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        b, t, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == b:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, t)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maximum Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    # num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})
    num_target_vision_tokens: int = field(default=64, metadata={"help": "Number of target vision tokens."})

    # Multi-head action prediction
    use_multi_action_heads: bool = field(default=True, metadata={"help": "Whether to use multi-head action prediction"})
    action_arm_dim: int = field(default=14, metadata={"help": "Arm joint dimensions (0-13) - absolute actions"})
    action_claw_dim: int = field(default=2, metadata={"help": "Claw position dimensions (14-15) - absolute actions"})
    
    # Split arm into left and right hands
    split_arm_heads: bool = field(default=True, metadata={"help": "Whether to split arm head into left and right arm heads"})
    action_left_arm_dim: int = field(default=7, metadata={"help": "Left arm joint dimensions (0-6) - absolute actions"})
    action_right_arm_dim: int = field(default=7, metadata={"help": "Right arm joint dimensions (7-13) - absolute actions"})
    
    # Coordination mechanisms for split arms
    # 最优方案：共享底层特征 + 交叉注意力 + 协调性损失
    # 这样可以平衡左右手的独立性和协调性
    use_shared_arm_features: bool = field(default=True, metadata={"help": "Whether to share bottom layer features between left and right arms for better coordination"})
    use_cross_attention_arms: bool = field(default=True, metadata={"help": "Whether to use cross-attention between left and right arm features. Recommended: True for bimanual tasks"})
    arm_coordination_loss_weight: float = field(default=0.2, metadata={"help": "Weight for arm coordination loss (encourages synchronized movements). Recommended: 0.1-0.3"})
    
    # Loss weights for different action heads
    arm_loss_weight: float = field(default=1.0, metadata={"help": "Arm absolute position loss weight"})
    left_arm_loss_weight: float = field(default=1.0, metadata={"help": "Left arm absolute position loss weight"})
    right_arm_loss_weight: float = field(default=1.0, metadata={"help": "Right arm absolute position loss weight"})
    claw_loss_weight: float = field(default=1.0, metadata={"help": "Claw position loss weight"})
    
    # Learnable uncertainty weights (参考 https://arxiv.org/pdf/1705.07115)
    use_learnable_loss_weights: bool = field(default=True, metadata={"help": "Enable learnable loss weights based on uncertainty"})
    
    # Pretrained action dimension (for compatibility with pretrained models)
    pretrained_action_dim: int = field(default=None, metadata={"help": "Action dimension of pretrained model (for compatibility)"})
    
    # Chain of Causation (CoC) reasoning configuration
    use_coc_reasoning: bool = field(default=True, metadata={"help": "Whether to use Chain of Causation reasoning"})
    reasoning_vocab_size: int = field(default=1000, metadata={"help": "Vocabulary size for reasoning tokens"})
    reasoning_max_length: int = field(default=128, metadata={"help": "Maximum length of reasoning trace"})
    reasoning_hidden_dim: int = field(default=512, metadata={"help": "Hidden dimension for reasoning head"})
    reasoning_num_layers: int = field(default=2, metadata={"help": "Number of transformer layers in reasoning head"})
    reasoning_loss_weight: float = field(default=1.0, metadata={"help": "Weight for reasoning loss"})
    tune_reasoning_head: bool = field(default=True, metadata={"help": "Whether to tune the reasoning head"})
    reasoning_conditioning_type: str = field(default="decoder", metadata={"help": "Where to condition reasoning: 'decoder' or 'dit' or 'both'"})
    # Action decision types: 6种细粒度的决策类型
    action_decision_types: list[str] = field(
        default_factory=lambda: [
            "left_search_grasp_pull",      # 1. 左手搜索抓取拉开，右手不动
            "left_hold_right_search_grasp", # 2. 左手保持，右手搜索抓取
            "right_search_grasp_pull",     # 3. 右手搜索抓取拉开，左手不动
            "right_hold_left_search_grasp", # 4. 右手保持，左手搜索抓取
            "both_search_grasp",           # 5. 双手同时搜索抓取
            "both_hold_lift",              # 6. 双手保持并上抬
        ],
        metadata={"help": "List of action decision types (6 types)"}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Validate multi-head configuration
        if self.use_multi_action_heads:
            if self.split_arm_heads:
                # When splitting arms, validate left + right = total arm dim
                # Note: action_arm_dim should be set to left+right in groot_n1.py
                expected_arm_dim = self.action_left_arm_dim + self.action_right_arm_dim
                if self.action_arm_dim != expected_arm_dim:
                    raise ValueError(
                        f"When split_arm_heads=True, action_arm_dim ({self.action_arm_dim}) must equal "
                        f"action_left_arm_dim ({self.action_left_arm_dim}) + action_right_arm_dim ({self.action_right_arm_dim}) = {expected_arm_dim}"
                    )
                expected_action_dim = self.action_left_arm_dim + self.action_right_arm_dim + self.action_claw_dim
            else:
                expected_action_dim = self.action_arm_dim + self.action_claw_dim
            
            if self.action_dim is not None and self.action_dim != expected_action_dim:
                # If pretrained_action_dim is set, allow mismatch (we'll pad/truncate)
                if self.pretrained_action_dim is None:
                    raise ValueError(
                        f"When using multi-action heads, action_dim ({self.action_dim}) must equal "
                        f"{'left_arm + right_arm + claw' if self.split_arm_heads else 'arm + claw'} = {expected_action_dim}"
                    )
                # If pretrained_action_dim is set, use it for action_encoder
                if self.pretrained_action_dim != expected_action_dim:
                    print(f"⚠️  Pretrained model uses {self.pretrained_action_dim}D, but data uses {expected_action_dim}D. "
                          f"Will pad/truncate actions for compatibility.")


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
        rtc_processor: RTCProcessor | None = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        
        # Use pretrained_action_dim for action_encoder if specified (for compatibility with pretrained models)
        # Otherwise use action_dim
        encoder_action_dim = config.pretrained_action_dim if config.pretrained_action_dim is not None else config.action_dim
        self.encoder_action_dim = encoder_action_dim
        self.actual_action_dim = config.action_dim  # Actual action dimension from data

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=encoder_action_dim,  # Use pretrained dimension for encoder
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        
        # Multi-head action prediction
        if config.use_multi_action_heads:
            if config.split_arm_heads:
                # Split arm into left and right
                if config.use_shared_arm_features:
                    # 使用共享底层特征的decoder，提升左右手协调性
                    self.shared_arm_decoder = SharedBottomArmDecoder(
                        num_categories=config.max_num_embodiments,
                        input_dim=self.hidden_size,
                        hidden_dim=self.hidden_size,
                        left_output_dim=config.action_left_arm_dim,
                        right_output_dim=config.action_right_arm_dim,
                        use_cross_attention=config.use_cross_attention_arms,
                    )
                    self.action_left_arm_decoder = None
                    self.action_right_arm_decoder = None
                    if config.use_cross_attention_arms:
                        print(f"🤝 Using OPTIMAL hybrid architecture:")
                        print(f"   ✅ Shared bottom layer (coordination)")
                        print(f"   ✅ Cross-attention (left↔right awareness)")
                        print(f"   ✅ Separate output layers (independence)")
                        print(f"   ✅ Coordination loss weight={config.arm_coordination_loss_weight}")
                    else:
                        print(f"🤝 Using shared-bottom arm decoder (cross-attention disabled)")
                        print(f"   ⚠️  This is similar to 'single MLP then split'")
                        print(f"   💡 Enable cross-attention for better coordination!")
                else:
                    # 完全独立的decoder（原始实现）
                    self.action_left_arm_decoder = CategorySpecificMLP(
                        num_categories=config.max_num_embodiments,
                        input_dim=self.hidden_size,
                        hidden_dim=self.hidden_size,
                        output_dim=config.action_left_arm_dim,
                    )
                    self.action_right_arm_decoder = CategorySpecificMLP(
                        num_categories=config.max_num_embodiments,
                        input_dim=self.hidden_size,
                        hidden_dim=self.hidden_size,
                        output_dim=config.action_right_arm_dim,
                    )
                    self.shared_arm_decoder = None
                    print(f"🔀 Using independent arm decoders")
                self.action_arm_decoder = None  # Not used when split
            else:
                # Single arm head
                self.action_arm_decoder = CategorySpecificMLP(
                    num_categories=config.max_num_embodiments,
                    input_dim=self.hidden_size,
                    hidden_dim=self.hidden_size,
                    output_dim=config.action_arm_dim,
                )
                self.action_left_arm_decoder = None
                self.action_right_arm_decoder = None
            
            self.action_claw_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=config.action_claw_dim,
            )
            self.action_decoder = None  # Not used in multi-head mode
            
            if config.split_arm_heads:
                total_dim = config.action_left_arm_dim + config.action_right_arm_dim + config.action_claw_dim
                print(f"📊 Multi-head action: left_arm({config.action_left_arm_dim}D, indices 0-{config.action_left_arm_dim-1}) + "
                      f"right_arm({config.action_right_arm_dim}D, indices {config.action_left_arm_dim}-{config.action_left_arm_dim + config.action_right_arm_dim-1}) + "
                      f"claw({config.action_claw_dim}D, indices {config.action_arm_dim}-{config.action_arm_dim + config.action_claw_dim-1}) = {total_dim}D")
                print(f"   action_arm_dim={config.action_arm_dim} (left+right), actual_action_dim={config.action_dim}")
            else:
                print(f"📊 Multi-head action: arm({config.action_arm_dim}D) + claw({config.action_claw_dim}D) = {config.action_arm_dim + config.action_claw_dim}D")
        else:
            self.action_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.action_dim,
            )
            self.action_arm_decoder = None
            self.action_left_arm_decoder = None
            self.action_right_arm_decoder = None
            self.action_claw_decoder = None
        
        # Learnable loss weights (参考 https://arxiv.org/pdf/1705.07115)
        if config.use_learnable_loss_weights and config.use_multi_action_heads:
            if config.split_arm_heads:
                self.task_log_sigma = nn.ParameterDict({
                    "left_arm": nn.Parameter(torch.zeros(())),    # log(σ_left_arm)
                    "right_arm": nn.Parameter(torch.zeros(())),   # log(σ_right_arm)
                    "claw": nn.Parameter(torch.zeros(())),        # log(σ_claw)
                })
                print(f"🎯 Learnable loss weights enabled: left_arm, right_arm, claw")
            else:
                self.task_log_sigma = nn.ParameterDict({
                    "arm": nn.Parameter(torch.zeros(())),    # log(σ_arm)
                    "claw": nn.Parameter(torch.zeros(())),  # log(σ_claw)
                })
                print(f"🎯 Learnable loss weights enabled: arm, claw")
            print(f"   Using uncertainty-based weighting from https://arxiv.org/pdf/1705.07115")
        else:
            self.task_log_sigma = None
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        
        # Chain of Causation (CoC) Reasoning Head
        if config.use_coc_reasoning:
            self.reasoning_head = ReasoningHead(
                backbone_embedding_dim=config.backbone_embedding_dim,
                reasoning_hidden_dim=config.reasoning_hidden_dim,
                reasoning_vocab_size=config.reasoning_vocab_size,
                reasoning_max_length=config.reasoning_max_length,
                num_layers=config.reasoning_num_layers,
            )
            print(f"🧠 Chain of Causation (CoC) Reasoning enabled:")
            print(f"   ✅ Reasoning vocab size: {config.reasoning_vocab_size}")
            print(f"   ✅ Reasoning max length: {config.reasoning_max_length}")
            print(f"   ✅ Reasoning hidden dim: {config.reasoning_hidden_dim}")
            print(f"   ✅ Reasoning conditioning: {config.reasoning_conditioning_type}")
            print(f"   ✅ Action decision types: {config.action_decision_types}")
        else:
            self.reasoning_head = None
        
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)
        self.rtc_processor = rtc_processor


    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            if self.config.use_multi_action_heads:
                if self.config.split_arm_heads:
                    if hasattr(self, 'shared_arm_decoder') and self.shared_arm_decoder is not None:
                        self.shared_arm_decoder.requires_grad_(False)
                    if self.action_left_arm_decoder is not None:
                        self.action_left_arm_decoder.requires_grad_(False)
                    if self.action_right_arm_decoder is not None:
                        self.action_right_arm_decoder.requires_grad_(False)
                else:
                    if self.action_arm_decoder is not None:
                        self.action_arm_decoder.requires_grad_(False)
                if self.action_claw_decoder is not None:
                    self.action_claw_decoder.requires_grad_(False)
            else:
                if self.action_decoder is not None:
                    self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        
        # Handle reasoning head trainability
        if self.reasoning_head is not None:
            if not self.config.tune_reasoning_head:
                self.reasoning_head.requires_grad_(False)
                print(f"Tune reasoning head: False (frozen)")
            else:
                print(f"Tune reasoning head: True (trainable)")
        
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                if self.config.use_multi_action_heads:
                    if self.config.split_arm_heads:
                        if hasattr(self, 'shared_arm_decoder') and self.shared_arm_decoder is not None:
                            self.shared_arm_decoder.eval()
                        if self.action_left_arm_decoder is not None:
                            self.action_left_arm_decoder.eval()
                        if self.action_right_arm_decoder is not None:
                            self.action_right_arm_decoder.eval()
                    else:
                        if self.action_arm_decoder is not None:
                            self.action_arm_decoder.eval()
                    if self.action_claw_decoder is not None:
                        self.action_claw_decoder.eval()
                else:
                    if self.action_decoder is not None:
                        self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        """
            第二阶段: Vision-Language特征增强(vl_self_attention)
            # 关键步骤
            * 1) 对已经融合的视觉-语言特征进行4层自注意力处理
            * 2) 进一步强化视觉和语言之间的关联
            * 3) 为后续的跨模态注意力做准备
            # values:
            * backbone_features: 视觉-语言特征
            * vlln: 视觉-语言特征归一化
            * vl_self_attention: 视觉-语言特征自注意力处理
            * backbone_output: 视觉-语言特征
            * return_dict: 是否返回字典
            * return_dict: 是否返回字典
        """
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)
        
        # Generate reasoning trace if CoC reasoning is enabled
        # 根据论文 Alpamayo-R1 (https://arxiv.org/pdf/2511.00088)，SFT阶段的损失函数为：
        # L_SFT(θ) = -E_{(o, REASON, a) ~ D_CoC} [log π_θ(REASON, a | o)]
        # 这包含两部分：
        # 1. Reasoning trace的交叉熵损失：log π_θ(REASON | o)
        # 2. Action decision的交叉熵损失（CoC-Action Consistency）：确保reasoning trace和action之间的一致性
        reasoning_logits = None
        reasoning_conditioning = None
        action_decision_logits = None
        reasoning_trace_loss = None  # Reasoning trace的交叉熵损失
        action_decision_loss = None  # Action decision的交叉熵损失（CoC-Action Consistency）
        total_reasoning_loss = None  # 总reasoning损失 = reasoning_trace_loss + action_decision_loss
        
        if self.config.use_coc_reasoning and self.reasoning_head is not None:
            backbone_features = backbone_output.backbone_features  # (B, T, backbone_embedding_dim)
            
            # Get reasoning labels from action_input if available (for training)
            reasoning_labels = action_input.get("reasoning_labels", None) if hasattr(action_input, "get") else None
            if reasoning_labels is None and hasattr(action_input, "data"):
                reasoning_labels = action_input.data.get("reasoning_labels", None)
            
            # Get action decision labels from action_input if available (for training)
            action_decision_labels = None
            if hasattr(action_input, "get"):
                action_decision_labels = action_input.get("action_decision_labels", None)
            elif hasattr(action_input, "data"):
                action_decision_labels = action_input.data.get("action_decision_labels", None)
            
            # Generate reasoning
            # 注意：在训练时，action_decision_labels会被用于teacher forcing，确保conditioning使用正确的decision
            reasoning_logits, reasoning_conditioning, action_decision_logits = self.reasoning_head(
                backbone_features, reasoning_labels, action_decision_labels
            )
            
            # 1. 计算Reasoning trace的交叉熵损失
            # L_reasoning = -log π_θ(REASON | o)
            # 这是思维链reasoning trace的交叉熵损失
            if reasoning_labels is not None and reasoning_logits is not None:
                reasoning_trace_loss = F.cross_entropy(
                    reasoning_logits.reshape(-1, reasoning_logits.shape[-1]),
                    reasoning_labels.reshape(-1),
                    ignore_index=-100,  # Ignore padding tokens
                    reduction="mean"
                )
            
            # 2. 计算Action decision的交叉熵损失（CoC-Action Consistency）
            # L_action_decision = -log π_θ(action_decision | o)
            # 这是动作一致性奖励，确保reasoning trace预测的action decision与ground truth一致
            # 这是CoC-Action Consistency的关键组成部分
            # 注意：action_decision_labels已经在上面获取过了（第933-937行），这里直接使用
            if action_decision_labels is not None and action_decision_logits is not None:
                action_decision_loss = F.cross_entropy(
                    action_decision_logits,
                    action_decision_labels,
                    reduction="mean"
                )
            
            # 3. 总reasoning损失 = reasoning trace损失 + action decision损失
            # 这实现了论文中的 L_SFT(θ) = -E[log π_θ(REASON, a | o)]
            if reasoning_trace_loss is not None and action_decision_loss is not None:
                total_reasoning_loss = reasoning_trace_loss + action_decision_loss
            elif reasoning_trace_loss is not None:
                total_reasoning_loss = reasoning_trace_loss
            elif action_decision_loss is not None:
                total_reasoning_loss = action_decision_loss

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        # NOTE: Processor (GrootPackInputsStep) already pads action to max_action_dim (32)
        # So action_input.action is already (B, T, encoder_action_dim=32)
        
        # 1) 获取真实的 action (ground truth)
        actions = action_input.action  # (B, T, encoder_action_dim)
        action_mask = action_input.action_mask  # (B, T, encoder_action_dim) - marks valid dimensions
        
        # Ensure actions match encoder_action_dim (should already be padded by processor)
        if actions.shape[-1] != self.encoder_action_dim:
            if actions.shape[-1] < self.encoder_action_dim:
                # Pad if needed (shouldn't happen if processor works correctly)
                pad_size = self.encoder_action_dim - actions.shape[-1]
                padding = torch.zeros(
                    (actions.shape[0], actions.shape[1], pad_size),
                    device=actions.device,
                    dtype=actions.dtype
                )
                actions = torch.cat([actions, padding], dim=-1)
            else:
                # Truncate if larger (shouldn't happen)
                actions = actions[:, :, :self.encoder_action_dim]
        # 2) 生成随机噪声
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        # 3) 随机采样时间步 t ∈ [0, 1]
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast
        # 4) 创建加噪轨迹（Flow Matching 核心）
        # 当 t=0：纯噪声
        # 当 t=1：真实 action
        noisy_trajectory = (1 - t) * noise + t * actions
        
        # For velocity, extract only the actual action dimensions (first actual_action_dim)
        # This matches the original data dimension before padding
        velocity = actions[:, :, :self.actual_action_dim] - noise[:, :, :self.actual_action_dim]

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        # 5) 编码加噪轨迹为 action_features
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        """
            第三阶段: Vision-Language与State-Action融合(DiT Cross-Attention)
            # 关键步骤
            * 1) 将视觉-语言特征和状态-动作特征拼接在一起
            * 2) 通过DiT的Cross-Attention机制, 让视觉-语言特征和状态-动作特征相互关注
            * 3) 输出: 状态-动作特征
            
            # values:
            * future_tokens: 未来tokens
            * vl_embs: 视觉-语言特征 # Key/Value
            * sa_embs: 状态-动作特征 # Query
            * vl_attn_mask: 视觉-语言特征的注意力掩码
            * model_output: 模型输出
            * return_dict: 是否返回字典
            * return_dict: 是否返回字典
        """
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        # 6) 拼接为 hidden_states
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        # 7) DiT Cross-Attention
        # 注意：如果reasoning_conditioning_type="dit"或"both"，可以在DiT输入前应用conditioning
        # 但目前DiT接口不支持reasoning_conditioning参数，所以只在decoder输入前应用
        # 这是合理的，因为conditioning在decoder输入前应用也能有效引导动作生成
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        # 8. 预测 velocity
        # pred_velocity = self.action_decoder(model_output)
        # 9. 计算损失
        # loss = MSE(pred_velocity, actions - noise)
        
        # Slice out only the action portion of model output
        model_output_actions = model_output[:, -actions.shape[1] :]
        
        # Apply reasoning conditioning to model_output_actions if reasoning is enabled
        # 关键：reasoning_conditioning已经融合了action decision的信息（在ReasoningHead中）
        # 这确保了action decision能够真正引导DiT的动作生成方向
        # 
        # 完整链路：
        # 1. backbone_features → reasoning trace (思维链)
        # 2. reasoning trace → action decision (动作决策)
        # 3. action decision → action_decision_embedding (决策嵌入)
        # 4. action_decision_embedding + base_conditioning → reasoning_conditioning (融合的条件向量)
        # 5. reasoning_conditioning → 投影到decoder维度 → 残差连接到model_output_actions
        # 6. 条件化的model_output_actions → decoder → 动作预测
        #
        # 例如：如果action decision是"left_search_grasp_pull"：
        # - action_decision_embedding会编码"左手搜索抓取拉开，右手不动"的信息
        # - 这个embedding会通过残差连接偏置model_output_actions
        # - 最终decoder会生成偏置左手动作（搜索、抓取、拉开），右手保持静止的动作
        if self.config.use_coc_reasoning and reasoning_conditioning is not None:
            # Project reasoning conditioning to match model_output_actions dimension
            # Use a cached projection layer if available, otherwise create one
            if not hasattr(self, '_reasoning_proj'):
                self._reasoning_proj = nn.Linear(
                    self.config.reasoning_hidden_dim, 
                    model_output_actions.shape[-1]
                ).to(model_output_actions.device)
            # 投影并扩展维度：reasoning_conditioning (B, reasoning_hidden_dim) 
            # → (B, hidden_size) → (B, 1, hidden_size)
            # 然后通过广播自动扩展到 (B, T, hidden_size)
            reasoning_cond_expanded = self._reasoning_proj(reasoning_conditioning).unsqueeze(1)  # (B, 1, hidden_size)
            
            # Add reasoning conditioning to model output (residual connection)
            # This biases the action generation towards the reasoning decision
            # 注意：目前只在decoder输入前应用（reasoning_conditioning_type="decoder"或"both"）
            # 如果设置为"dit"，需要在DiT内部应用，但这需要修改DiT接口
            if self.config.reasoning_conditioning_type in ["decoder", "both"]:
                model_output_actions = model_output_actions + reasoning_cond_expanded  # (B, T, hidden_size)
        
        # Multi-head action prediction
        if self.config.use_multi_action_heads:
            if self.config.split_arm_heads:
                # Split arm into left and right
                if self.config.use_shared_arm_features and hasattr(self, 'shared_arm_decoder') and self.shared_arm_decoder is not None:
                    # 使用共享底层特征的decoder
                    pred_left_arm, pred_right_arm = self.shared_arm_decoder(model_output_actions, embodiment_id)
                else:
                    # 使用独立的decoder
                    pred_left_arm = self.action_left_arm_decoder(model_output_actions, embodiment_id)
                    pred_right_arm = self.action_right_arm_decoder(model_output_actions, embodiment_id)
                pred_claw = self.action_claw_decoder(model_output_actions, embodiment_id)
                pred_actions = torch.cat([pred_left_arm, pred_right_arm, pred_claw], dim=-1)  # (B, T, action_dim)
                
                # Split ground truth velocity into corresponding parts
                # velocity shape: (B, T, actual_action_dim=16)
                # Structure: [left_arm(0-6, 7D), right_arm(7-13, 7D), claw(14-15, 2D)]
                velocity_left_arm = velocity[:, :, :self.config.action_left_arm_dim]  # (B, T, 7) - indices 0-6
                velocity_right_arm = velocity[:, :, self.config.action_left_arm_dim:self.config.action_left_arm_dim + self.config.action_right_arm_dim]  # (B, T, 7) - indices 7-13
                velocity_claw = velocity[:, :, self.config.action_arm_dim:]  # (B, T, 2) - indices 14-15
                
                # Compute loss for each head
                # action_mask shape: (B, T, encoder_action_dim), extract only actual_action_dim
                action_mask = action_input.action_mask[:, :, :self.actual_action_dim]  # (B, T, 16)
                # Split mask for left_arm, right_arm and claw (same structure as velocity)
                action_mask_left_arm = action_mask[:, :, :self.config.action_left_arm_dim]  # (B, T, 7) - indices 0-6
                action_mask_right_arm = action_mask[:, :, self.config.action_left_arm_dim:self.config.action_left_arm_dim + self.config.action_right_arm_dim]  # (B, T, 7) - indices 7-13
                action_mask_claw = action_mask[:, :, self.config.action_arm_dim:]  # (B, T, 2) - indices 14-15
                
                loss_left_arm = F.mse_loss(pred_left_arm, velocity_left_arm, reduction="none") * action_mask_left_arm
                loss_right_arm = F.mse_loss(pred_right_arm, velocity_right_arm, reduction="none") * action_mask_right_arm
                loss_claw = F.mse_loss(pred_claw, velocity_claw, reduction="none") * action_mask_claw
                
                # 协调性损失：鼓励左右手动作的协调性（可选）
                coordination_loss = None
                if self.config.arm_coordination_loss_weight > 0:
                    # 计算左右手速度的差异，鼓励它们在某些维度上保持同步
                    # 这里使用速度差的L2范数作为协调性损失
                    # 注意：不是完全同步，而是鼓励协调（比如拉箱子时左右手应该同步）
                    left_arm_magnitude = torch.norm(pred_left_arm, dim=-1, keepdim=True)  # (B, T, 1)
                    right_arm_magnitude = torch.norm(pred_right_arm, dim=-1, keepdim=True)  # (B, T, 1)
                    # 鼓励左右手的速度幅度相似（但不完全相同）
                    coordination_loss = F.mse_loss(left_arm_magnitude, right_arm_magnitude, reduction="none")
                    # 只对有效的动作维度计算
                    valid_mask = (action_mask_left_arm.sum(dim=-1, keepdim=True) > 0) & (action_mask_right_arm.sum(dim=-1, keepdim=True) > 0)
                    coordination_loss = (coordination_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
                
                # Use learnable weights or fixed weights
                if self.config.use_learnable_loss_weights and self.task_log_sigma is not None:
                    loss_left_arm_mean = loss_left_arm.sum() / action_mask_left_arm.sum()
                    loss_right_arm_mean = loss_right_arm.sum() / action_mask_right_arm.sum()
                    loss_claw_mean = loss_claw.sum() / action_mask_claw.sum()
                    
                    s_left_arm = self.task_log_sigma["left_arm"]
                    s_right_arm = self.task_log_sigma["right_arm"]
                    s_claw = self.task_log_sigma["claw"]
                    precision_left_arm = torch.exp(-2.0 * s_left_arm)
                    precision_right_arm = torch.exp(-2.0 * s_right_arm)
                    precision_claw = torch.exp(-2.0 * s_claw)
                    
                    loss = precision_left_arm * loss_left_arm_mean + precision_right_arm * loss_right_arm_mean + precision_claw * loss_claw_mean + s_left_arm + s_right_arm + s_claw
                    
                    # 添加协调性损失
                    if coordination_loss is not None:
                        loss = loss + self.config.arm_coordination_loss_weight * coordination_loss
                    
                    output_dict = {
                        "loss": loss,
                        "left_arm_loss": loss_left_arm_mean.item(),
                        "right_arm_loss": loss_right_arm_mean.item(),
                        "claw_loss": loss_claw_mean.item(),
                        "sigma_left_arm": torch.exp(s_left_arm).item(),
                        "sigma_right_arm": torch.exp(s_right_arm).item(),
                        "sigma_claw": torch.exp(s_claw).item(),
                        "weight_left_arm": precision_left_arm.item(),
                        "weight_right_arm": precision_right_arm.item(),
                        "weight_claw": precision_claw.item(),
                    }
                    if coordination_loss is not None:
                        output_dict["arm_coordination_loss"] = coordination_loss.item()
                else:
                    # Use fixed weights
                    loss_left_arm_mean = loss_left_arm.sum() / action_mask_left_arm.sum()
                    loss_right_arm_mean = loss_right_arm.sum() / action_mask_right_arm.sum()
                    loss_claw_mean = loss_claw.sum() / action_mask_claw.sum()
                    loss = self.config.left_arm_loss_weight * loss_left_arm_mean + self.config.right_arm_loss_weight * loss_right_arm_mean + self.config.claw_loss_weight * loss_claw_mean
                    
                    # 添加协调性损失
                    if coordination_loss is not None:
                        loss = loss + self.config.arm_coordination_loss_weight * coordination_loss
                    
                    output_dict = {
                        "loss": loss,
                        "left_arm_loss": loss_left_arm_mean.item(),
                        "right_arm_loss": loss_right_arm_mean.item(),
                        "claw_loss": loss_claw_mean.item(),
                    }
                    if coordination_loss is not None:
                        output_dict["arm_coordination_loss"] = coordination_loss.item()
            else:
                # Single arm head (original behavior)
                pred_arm = self.action_arm_decoder(model_output_actions, embodiment_id)
                pred_claw = self.action_claw_decoder(model_output_actions, embodiment_id)
                pred_actions = torch.cat([pred_arm, pred_claw], dim=-1)  # (B, T, action_dim)
                
                # Split ground truth velocity into corresponding parts
                velocity_arm = velocity[:, :, :self.config.action_arm_dim]  # (B, T, action_arm_dim)
                velocity_claw = velocity[:, :, self.config.action_arm_dim:]  # (B, T, action_claw_dim)
                
                # Compute loss for each head
                action_mask = action_input.action_mask[:, :, :self.actual_action_dim]  # (B, T, actual_action_dim)
                # Split mask for arm and claw
                action_mask_arm = action_mask[:, :, :self.config.action_arm_dim]  # (B, T, action_arm_dim)
                action_mask_claw = action_mask[:, :, self.config.action_arm_dim:]  # (B, T, action_claw_dim)
                
                loss_arm = F.mse_loss(pred_arm, velocity_arm, reduction="none") * action_mask_arm
                loss_claw = F.mse_loss(pred_claw, velocity_claw, reduction="none") * action_mask_claw
                
                # Use learnable weights or fixed weights
                if self.config.use_learnable_loss_weights and self.task_log_sigma is not None:
                    loss_arm_mean = loss_arm.sum() / action_mask_arm.sum()
                    loss_claw_mean = loss_claw.sum() / action_mask_claw.sum()
                    
                    s_arm = self.task_log_sigma["arm"]
                    s_claw = self.task_log_sigma["claw"]
                    precision_arm = torch.exp(-2.0 * s_arm)  # 1 / σ²
                    precision_claw = torch.exp(-2.0 * s_claw)
                    
                    loss = precision_arm * loss_arm_mean + precision_claw * loss_claw_mean + s_arm + s_claw
                    
                    output_dict = {
                        "loss": loss,
                        "arm_loss": loss_arm_mean.item(),
                        "claw_loss": loss_claw_mean.item(),
                        "sigma_arm": torch.exp(s_arm).item(),
                        "sigma_claw": torch.exp(s_claw).item(),
                        "weight_arm": precision_arm.item(),
                        "weight_claw": precision_claw.item(),
                    }
                else:
                    # Use fixed weights
                    loss_arm_mean = loss_arm.sum() / action_mask_arm.sum()
                    loss_claw_mean = loss_claw.sum() / action_mask_claw.sum()
                    loss = self.config.arm_loss_weight * loss_arm_mean + self.config.claw_loss_weight * loss_claw_mean
                    
                    output_dict = {
                        "loss": loss,
                        "arm_loss": loss_arm_mean.item(),
                        "claw_loss": loss_claw_mean.item(),
                    }
        else:
            # Single head (original behavior)
            pred = self.action_decoder(model_output_actions, embodiment_id)
            pred_actions = pred
            
            # Slice out only the action portion of pred and target.
            action_mask = action_input.action_mask
            loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
            loss = loss.sum() / action_mask.sum()
            output_dict = {
                "loss": loss,
            }
        
        # Add reasoning loss to total loss
        # 根据论文，总损失 = 动作预测损失 + reasoning_loss_weight * (reasoning_trace_loss + action_decision_loss)
        # 这实现了 L_SFT(θ) = -E[log π_θ(REASON, a | o)]
        if total_reasoning_loss is not None:
            total_loss = output_dict["loss"] + self.config.reasoning_loss_weight * total_reasoning_loss
            output_dict["loss"] = total_loss
            
            # 分别记录各个损失项，便于监控和调试
            if reasoning_trace_loss is not None:
                output_dict["reasoning_trace_loss"] = reasoning_trace_loss.item() if isinstance(reasoning_trace_loss, torch.Tensor) else reasoning_trace_loss
            if action_decision_loss is not None:
                output_dict["action_decision_loss"] = action_decision_loss.item() if isinstance(action_decision_loss, torch.Tensor) else action_decision_loss
                # 记录CoC-Action Consistency损失（用于监控）
                output_dict["coc_action_consistency_loss"] = output_dict["action_decision_loss"]
            
            # 总reasoning损失（用于向后兼容）
            output_dict["reasoning_loss"] = total_reasoning_loss.item() if isinstance(total_reasoning_loss, torch.Tensor) else total_reasoning_loss
            
            # Add action decision prediction for monitoring
            if action_decision_logits is not None:
                # Get predicted action decision
                predicted_decision = torch.argmax(action_decision_logits, dim=-1)  # (B,)
                output_dict["predicted_action_decision"] = predicted_decision.cpu().numpy().tolist()
        
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature, rtc_enabled: bool, **kwargs) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)
        
        # Generate reasoning trace if CoC reasoning is enabled
        reasoning_conditioning = None
        action_decision_logits = None
        
        if self.config.use_coc_reasoning and self.reasoning_head is not None:
            backbone_features = backbone_output.backbone_features  # (B, T, backbone_embedding_dim)
            
            # Generate reasoning (inference mode, no labels)
            _, reasoning_conditioning, action_decision_logits = self.reasoning_head(
                backbone_features, reasoning_labels=None
            )
            
            # Get predicted action decision
            if action_decision_logits is not None:
                predicted_decision_idx = torch.argmax(action_decision_logits, dim=-1)  # (B,)
                decision_map = {
                    0: "left_search_grasp_pull",
                    1: "left_hold_right_search_grasp",
                    2: "right_search_grasp_pull",
                    3: "right_hold_left_search_grasp",
                    4: "both_search_grasp",
                    5: "both_hold_lift",
                }
                predicted_decision = [decision_map[idx.item()] for idx in predicted_decision_idx]
                print(f"🧠 Predicted action decision: {predicted_decision}")

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        # Use encoder_action_dim for internal processing (compatible with pretrained model)
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        # 1. 初始化：从随机噪声开始
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.encoder_action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        # Zero out padded dimensions to match training behavior
        # In training, padded dimensions (after actual_action_dim) are always 0
        if self.encoder_action_dim != self.actual_action_dim:
            actions[:, :, self.actual_action_dim:] = 0.0

        x_t = actions

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        # 2. 迭代去噪（例如 4 步）
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            def denoise_step_partial_call(input_x_t, current_timestep=t_discretized, state_features=state_features, vl_embs=vl_embs, embodiment_id=embodiment_id, reasoning_conditioning=reasoning_conditioning):
                return self.denoise_step(x_t=input_x_t, timestep=current_timestep, vl_embs=vl_embs, state_features=state_features, embodiment_id=embodiment_id, reasoning_conditioning=reasoning_conditioning)

            if rtc_enabled:
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=t_discretized,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)
            # v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.encoder_action_dim != self.actual_action_dim:
                x_t[:, :, self.actual_action_dim:] = 0.0

            # # Record x_t and v_t after Euler step
            # if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
            #     self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
        # 3. 返回最终生成的 action
        actions_output = x_t[:, :, :self.actual_action_dim]
        return BatchFeature(data={"action_pred": actions_output})

    def denoise_step(self, x_t: torch.Tensor, timestep, vl_embs, state_features, embodiment_id, reasoning_conditioning=None) -> torch.Tensor:
        """
        单步预测 velocity
        """
        # 单步调用 _predict_velocity
        batch_size = x_t.shape[0]
        # timesteps_tensor = torch.full(size=(batch_size,), fill_value=timestep.item(), device=x_t.device)
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=timestep, device=x_t.device)
        v_t = self._predict_velocity(vl_embs, state_features, x_t, timesteps_tensor, embodiment_id, reasoning_conditioning=reasoning_conditioning)
        return v_t

    def _predict_velocity(
            self,
            vl_embs: torch.Tensor,
            state_features: torch.Tensor,
            actions: torch.Tensor,
            timesteps_tensor: torch.Tensor,
            embodiment_id: torch.Tensor,
            reasoning_conditioning: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """v_pi(A, o, tau) in the RTC paper: predicts velocity field for the current action chunk."""
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=actions.device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            model_output_actions = model_output[:, -self.action_horizon :]
            
            # Apply reasoning conditioning to model_output_actions if reasoning is enabled
            if self.config.use_coc_reasoning and reasoning_conditioning is not None:
                # Project reasoning conditioning to match model_output_actions dimension
                # Use a cached projection layer if available, otherwise create one
                if not hasattr(self, '_reasoning_proj'):
                    self._reasoning_proj = nn.Linear(
                        self.config.reasoning_hidden_dim, 
                        model_output_actions.shape[-1]
                    ).to(model_output_actions.device)
                reasoning_cond_expanded = self._reasoning_proj(reasoning_conditioning).unsqueeze(1)  # (B, 1, hidden_size)
                
                # Add reasoning conditioning to model output (residual connection)
                # This biases the action generation towards the reasoning decision
                if self.config.reasoning_conditioning_type in ["decoder", "both"]:
                    model_output_actions = model_output_actions + reasoning_cond_expanded

            if self.config.use_multi_action_heads:
                if self.config.split_arm_heads:
                    # Split arm into left and right
                    if self.config.use_shared_arm_features and hasattr(self, 'shared_arm_decoder') and self.shared_arm_decoder is not None:
                        # 使用共享底层特征的decoder
                        pred_left_arm, pred_right_arm = self.shared_arm_decoder(model_output_actions, embodiment_id)
                    else:
                        # 使用独立的decoder
                        pred_left_arm = self.action_left_arm_decoder(model_output_actions, embodiment_id)
                        pred_right_arm = self.action_right_arm_decoder(model_output_actions, embodiment_id)
                    pred_claw = self.action_claw_decoder(model_output_actions, embodiment_id)
                    pred_velocity = torch.cat([pred_left_arm, pred_right_arm, pred_claw], dim=-1)  # (B, T, action_dim)
                else:
                    # Single arm head
                    pred_arm = self.action_arm_decoder(model_output_actions, embodiment_id)
                    pred_claw = self.action_claw_decoder(model_output_actions, embodiment_id)
                    pred_velocity = torch.cat([pred_arm, pred_claw], dim=-1)  # (B, T, action_dim)
            else:
                pred_velocity = self.action_decoder(model_output_actions, embodiment_id)  # (B, T, action_dim)

            # Pad/truncate to encoder_action_dim so the action_encoder input format stays consistent.
            if self.encoder_action_dim != self.actual_action_dim:
                if self.encoder_action_dim > self.actual_action_dim:
                    pad_size = self.encoder_action_dim - self.actual_action_dim
                    padding = torch.zeros(
                        (pred_velocity.shape[0], pred_velocity.shape[1], pad_size),
                        device=pred_velocity.device,
                        dtype=pred_velocity.dtype,
                    )
                    pred_velocity = torch.cat([pred_velocity, padding], dim=-1)
                else:
                    pred_velocity = pred_velocity[:, :, : self.encoder_action_dim]

            return pred_velocity

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
