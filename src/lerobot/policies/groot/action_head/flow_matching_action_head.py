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


class VisionGroundedReasoningHead(nn.Module):
    """
    Vision-Grounded Reasoning Head（视觉基础推理头）
    
    核心设计理念（参考Alpamayo-R1）：
    =============================
    1. **显式生成reasoning trace**：不是隐式的attention，而是真正的文本推理
    2. **从视觉中推理**：模型学会"看"到箱子尺寸，而不是依赖prompt告诉它
    3. **Two-stage设计**：先推理（生成CoT），后行动（生成action）
    
    与Alpamayo-R1的对比：
    ===================
    | 方面 | Alpamayo-R1 | VisionGroundedReasoningHead |
    |------|-------------|----------------------------|
    | VLM | 完整Qwen3-VL | Eagle2 backbone + 轻量级decoder |
    | CoT生成 | 完整VLM自回归 | 轻量级reasoning decoder |
    | Action生成 | Expert + Diffusion | DiT Flow Matching |
    | 推理过程 | 显式文本输出 | 显式文本输出 + conditioning |
    
    推理示例：
    ========
    输入：视觉特征（看到60cm绿色箱子）+ 简洁prompt "Depalletize the box"
    输出CoT：
        <cot_start>
        观察：我看到一个宽大的绿色箱子，尺寸约为40×60×11 cm (型号4611)
        分析：这是一个较大的箱子，需要双手协调抓取
        位置：箱子位于右侧托盘区域
        决策：执行 both_search_grasp → both_hold_lift 动作序列
        <cot_end>
    输出Action：基于CoT的action conditioning → DiT生成动作
    
    架构设计：
    ========
    1. Vision Encoder: 从backbone_features提取视觉关键信息
    2. Reasoning Decoder: 轻量级Transformer decoder生成CoT
    3. Conditioning Extractor: 从CoT hidden states提取action conditioning
    """
    
    # 预定义的reasoning vocabulary（可训练时扩展）
    REASONING_VOCAB = {
        # 特殊tokens
        "<pad>": 0, "<eos>": 1, "<cot_start>": 2, "<cot_end>": 3,
        # 观察相关
        "observe": 4, "see": 5, "detect": 6,
        # 尺寸相关
        "wide": 7, "narrow": 8, "large": 9, "small": 10, 
        "60cm": 11, "30cm": 12, "40x60": 13, "40x30": 14,
        # 颜色相关
        "green": 15, "box": 16, "type": 17,
        # 位置相关
        "left": 18, "right": 19, "center": 20, "top": 21, "bottom": 22,
        # 动作决策相关
        "grasp": 23, "lift": 24, "move": 25, "place": 26,
        "left_arm": 27, "right_arm": 28, "both_arms": 29,
        "search": 30, "hold": 31, "pull": 32, "push": 33,
        # 连接词
        "and": 34, "then": 35, "because": 36, "so": 37,
        # 数字
        "0": 38, "1": 39, "2": 40, "3": 41, "4": 42, "5": 43,
        "6": 44, "7": 45, "8": 46, "9": 47,
    }
    
    # 动作决策类型（用于CoT生成后的分类）
    ACTION_DECISIONS = [
        "left_search_grasp_pull",      # 左手搜索抓取拉开
        "left_hold_right_search_grasp", # 左手保持，右手搜索抓取
        "right_search_grasp_pull",     # 右手搜索抓取拉开
        "right_hold_left_search_grasp", # 右手保持，左手搜索抓取
        "both_search_grasp",           # 双手同时搜索抓取
        "both_hold_lift",              # 双手保持并上抬
    ]
    
    def __init__(
        self,
        backbone_embedding_dim: int = 1536,  # Eagle2输出维度
        reasoning_hidden_dim: int = 512,     # Reasoning decoder隐藏维度
        reasoning_num_layers: int = 4,       # Reasoning decoder层数
        reasoning_num_heads: int = 8,        # 注意力头数
        reasoning_max_length: int = 64,      # 最大CoT长度
        conditioning_dim: int = 512,         # 输出conditioning维度
        dropout: float = 0.1,
        vocab_size: int = 128,               # 简化的reasoning vocabulary大小
    ):
        super().__init__()
        self.backbone_embedding_dim = backbone_embedding_dim
        self.reasoning_hidden_dim = reasoning_hidden_dim
        self.reasoning_max_length = reasoning_max_length
        self.conditioning_dim = conditioning_dim
        self.vocab_size = vocab_size
        
        # ============================================
        # 1. Vision Feature Encoder
        # 从backbone_features提取视觉关键信息
        # ============================================
        self.vision_proj = nn.Sequential(
            nn.Linear(backbone_embedding_dim, reasoning_hidden_dim),
            nn.LayerNorm(reasoning_hidden_dim),
            nn.GELU(),
        )
        
        # Vision query tokens（用于聚合视觉信息）
        self.num_vision_queries = 8  # 8个query聚合不同的视觉信息
        self.vision_queries = nn.Parameter(torch.randn(1, self.num_vision_queries, reasoning_hidden_dim))
        nn.init.normal_(self.vision_queries, std=0.02)
        
        # Vision cross-attention
        self.vision_cross_attn = nn.MultiheadAttention(
            embed_dim=reasoning_hidden_dim,
            num_heads=reasoning_num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.vision_ln = nn.LayerNorm(reasoning_hidden_dim)
        
        # ============================================
        # 2. Reasoning Decoder（轻量级Transformer）
        # 自回归生成CoT reasoning trace
        # ============================================
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, reasoning_hidden_dim)
        self.position_embedding = nn.Embedding(reasoning_max_length, reasoning_hidden_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=reasoning_hidden_dim,
            nhead=reasoning_num_heads,
            dim_feedforward=reasoning_hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.reasoning_decoder = nn.TransformerDecoder(decoder_layer, num_layers=reasoning_num_layers)
        
        # Output head for token prediction
        self.reasoning_output_head = nn.Linear(reasoning_hidden_dim, vocab_size)
        
        # ============================================
        # 3. Action Decision Head
        # 从CoT生成action decision
        # ============================================
        self.action_decision_head = nn.Sequential(
            nn.Linear(reasoning_hidden_dim, reasoning_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reasoning_hidden_dim, len(self.ACTION_DECISIONS)),
        )
        
        # Action decision embedding（用于conditioning）
        self.action_decision_embedding = nn.Embedding(len(self.ACTION_DECISIONS), conditioning_dim)
        
        # ============================================
        # 4. Conditioning Extractor
        # 从CoT hidden states提取最终的action conditioning
        # ============================================
        self.conditioning_proj = nn.Sequential(
            nn.Linear(reasoning_hidden_dim, conditioning_dim),
            nn.LayerNorm(conditioning_dim),
            nn.GELU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )
        
        # 特殊token IDs
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.cot_start_token_id = 2
        self.cot_end_token_id = 3
        
        print(f"🧠 VisionGroundedReasoningHead initialized:")
        print(f"   ✅ Vision-grounded: learns to 'see' box size from visual features")
        print(f"   ✅ Explicit CoT: generates reasoning trace like Alpamayo-R1")
        print(f"   ✅ {reasoning_num_layers} decoder layers, {reasoning_num_heads} heads")
        print(f"   ✅ Max CoT length: {reasoning_max_length} tokens")
        print(f"   ✅ {len(self.ACTION_DECISIONS)} action decision types")
        print(f"   ✅ Output conditioning dim: {conditioning_dim}")
    
    def encode_vision(
        self,
        backbone_features: torch.Tensor,  # (B, T, backbone_embedding_dim)
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        从backbone_features中编码视觉信息
        
        Returns:
            vision_context: (B, num_vision_queries, reasoning_hidden_dim)
        """
        B = backbone_features.shape[0]
        
        # Project to reasoning dimension
        vision_features = self.vision_proj(backbone_features)  # (B, T, reasoning_hidden_dim)
        
        # Cross-attention: queries attend to vision features
        queries = self.vision_queries.expand(B, -1, -1)
        
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        
        vision_context, _ = self.vision_cross_attn(
            query=queries,
            key=vision_features,
            value=vision_features,
            key_padding_mask=key_padding_mask,
        )
        vision_context = self.vision_ln(vision_context)
        
        return vision_context  # (B, num_vision_queries, reasoning_hidden_dim)
    
    def generate_reasoning_trace(
        self,
        vision_context: torch.Tensor,  # (B, num_vision_queries, reasoning_hidden_dim)
        reasoning_labels: torch.Tensor | None = None,  # (B, L) - teacher forcing时的标签
        temperature: float = 1.0,
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        生成reasoning trace（CoT）
        
        训练模式：使用teacher forcing
        推理模式：自回归生成
        
        Returns:
            reasoning_logits: (B, L, vocab_size) - token预测logits
            reasoning_hidden: (B, L, reasoning_hidden_dim) - 用于conditioning
        """
        B = vision_context.shape[0]
        device = vision_context.device
        max_len = max_length or self.reasoning_max_length
        
        if reasoning_labels is not None:
            # Training mode: teacher forcing
            L = reasoning_labels.shape[1]
            
            # Token + position embeddings
            token_embs = self.token_embedding(reasoning_labels)  # (B, L, hidden_dim)
            pos_ids = torch.arange(L, device=device)
            pos_embs = self.position_embedding(pos_ids)  # (L, hidden_dim)
            decoder_input = token_embs + pos_embs
            
            # Causal mask for decoder
            causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)
            
            # Decode with vision context as memory
            reasoning_hidden = self.reasoning_decoder(
                tgt=decoder_input,
                memory=vision_context,
                tgt_mask=causal_mask,
            )  # (B, L, hidden_dim)
            
            # Token prediction
            reasoning_logits = self.reasoning_output_head(reasoning_hidden)  # (B, L, vocab_size)
            
            return reasoning_logits, reasoning_hidden
        
        else:
            # Inference mode: autoregressive generation
            generated_tokens = torch.full((B, 1), self.cot_start_token_id, device=device, dtype=torch.long)
            all_hidden_states = []
            
            for step in range(max_len - 1):
                L = generated_tokens.shape[1]
                
                # Token + position embeddings
                token_embs = self.token_embedding(generated_tokens)
                pos_ids = torch.arange(L, device=device)
                pos_embs = self.position_embedding(pos_ids)
                decoder_input = token_embs + pos_embs
                
                # Causal mask
                causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)
                
                # Decode
                reasoning_hidden = self.reasoning_decoder(
                    tgt=decoder_input,
                    memory=vision_context,
                    tgt_mask=causal_mask,
                )  # (B, L, hidden_dim)
                
                # Get last token's logits
                last_hidden = reasoning_hidden[:, -1, :]  # (B, hidden_dim)
                last_logits = self.reasoning_output_head(last_hidden)  # (B, vocab_size)
                
                # Sample next token
                if temperature == 0.0:
                    next_token = torch.argmax(last_logits, dim=-1)
                else:
                    probs = F.softmax(last_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(1)], dim=1)
                all_hidden_states.append(last_hidden)
                
                # Stop if all sequences have generated EOS
                if (next_token == self.eos_token_id).all() or (next_token == self.cot_end_token_id).all():
                    break
            
            # Stack hidden states
            reasoning_hidden = torch.stack(all_hidden_states, dim=1)  # (B, generated_len, hidden_dim)
            
            # Get final logits for all generated tokens
            reasoning_logits = self.reasoning_output_head(reasoning_hidden)  # (B, generated_len, vocab_size)
            
            return reasoning_logits, reasoning_hidden
    
    def forward(
        self,
        backbone_features: torch.Tensor,  # (B, T, backbone_embedding_dim)
        attention_mask: torch.Tensor | None = None,
        reasoning_labels: torch.Tensor | None = None,  # (B, L) - CoT标签（训练时）
        action_decision_labels: torch.Tensor | None = None,  # (B,) - 动作决策标签（训练时）
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """
        Vision-Grounded Reasoning forward pass
        
        核心流程：
        1. 编码视觉特征 → vision_context
        2. 生成reasoning trace（CoT）
        3. 从CoT提取action decision
        4. 生成最终的action conditioning
        
        Returns:
            reasoning_logits: (B, L, vocab_size) - CoT token预测logits
            conditioning: (B, conditioning_dim) - action conditioning向量
            action_decision_logits: (B, num_decisions) - 动作决策logits
        """
        # 1. Encode vision features
        vision_context = self.encode_vision(backbone_features, attention_mask)
        
        # 2. Generate reasoning trace
        reasoning_logits, reasoning_hidden = self.generate_reasoning_trace(
            vision_context,
            reasoning_labels=reasoning_labels,
        )
        
        # 3. Extract action decision from reasoning
        # Use mean pooling of reasoning hidden states
        reasoning_aggregated = reasoning_hidden.mean(dim=1)  # (B, hidden_dim)
        action_decision_logits = self.action_decision_head(reasoning_aggregated)  # (B, num_decisions)
        
        # 4. Generate conditioning
        # Combine reasoning aggregation with action decision embedding
        base_conditioning = self.conditioning_proj(reasoning_aggregated)  # (B, conditioning_dim)
        
        # Get action decision (predicted or from labels)
        if action_decision_labels is not None:
            decision_idx = action_decision_labels
        else:
            decision_idx = torch.argmax(action_decision_logits, dim=-1)
        
        action_decision_emb = self.action_decision_embedding(decision_idx)  # (B, conditioning_dim)
        
        # Final conditioning: base + action decision
        conditioning = base_conditioning + action_decision_emb
        
        return reasoning_logits, conditioning, action_decision_logits
    
    def decode_reasoning_tokens(self, token_ids: torch.Tensor) -> list[str]:
        """
        将token IDs解码为可读的reasoning text（用于可视化/调试）
        """
        # 反转vocabulary
        id_to_token = {v: k for k, v in self.REASONING_VOCAB.items()}
        
        decoded = []
        for batch_tokens in token_ids:
            tokens = []
            for tid in batch_tokens.tolist():
                if tid == self.eos_token_id or tid == self.cot_end_token_id:
                    break
                token = id_to_token.get(tid, f"<unk_{tid}>")
                tokens.append(token)
            decoded.append(" ".join(tokens))
        
        return decoded




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
    
    # ============================================
    # VisionGroundedReasoningHead Configuration
    # ============================================
    # 使用VisionGroundedReasoningHead从视觉中显式生成CoT推理
    # 参考Alpamayo-R1的two-stage设计（先推理，后行动）
    
    use_coc_reasoning: bool = field(default=True, metadata={"help": "Whether to use Vision-Grounded CoT reasoning"})
    
    # VisionGroundedReasoningHead config
    reasoning_vocab_size: int = field(default=128, metadata={"help": "Vocabulary size for reasoning tokens"})
    reasoning_max_length: int = field(default=64, metadata={"help": "Maximum length of reasoning trace"})
    reasoning_hidden_dim: int = field(default=512, metadata={"help": "Hidden dimension for reasoning head"})
    reasoning_num_layers: int = field(default=4, metadata={"help": "Number of transformer decoder layers in reasoning head"})
    reasoning_num_heads: int = field(default=8, metadata={"help": "Number of attention heads in reasoning decoder"})
    reasoning_loss_weight: float = field(default=1.0, metadata={"help": "Weight for reasoning trace loss"})
    action_decision_loss_weight: float = field(default=1.0, metadata={"help": "Weight for action decision classification loss"})
    tune_reasoning_head: bool = field(default=True, metadata={"help": "Whether to tune the reasoning head"})
    reasoning_conditioning_type: str = field(default="decoder", metadata={"help": "Where to condition reasoning: 'decoder' or 'dit' or 'both'"})
    
    # Action decision types（6种动作决策类型）
    action_decision_types: list[str] = field(
        default_factory=lambda: [
            "left_search_grasp_pull",      # 1. 左手搜索抓取拉开，右手不动
            "left_hold_right_search_grasp", # 2. 左手保持，右手搜索抓取
            "right_search_grasp_pull",     # 3. 右手搜索抓取拉开，左手不动
            "right_hold_left_search_grasp", # 4. 右手保持，左手搜索抓取
            "both_search_grasp",           # 5. 双手同时搜索抓取
            "both_hold_lift",              # 6. 双手保持并上抬
        ],
        metadata={"help": "List of action decision types (6 types)."}
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
        
        # ============================================
        # VisionGroundedReasoningHead
        # ============================================
        # 从视觉中显式生成CoT，模型学会"看"
        # 参考Alpamayo-R1的two-stage设计
        
        if config.use_coc_reasoning:
            # VisionGroundedReasoningHead - 从视觉中显式生成CoT推理
            # 核心优势：
            # 1. 从视觉中显式生成reasoning trace（类似Alpamayo-R1）
            # 2. 模型学会"看"到箱子尺寸，而不是依赖prompt
            # 3. 可解释的推理过程
            # 4. 支持简洁prompt（如"Depalletize the box"）
            self.reasoning_head = VisionGroundedReasoningHead(
                backbone_embedding_dim=config.backbone_embedding_dim,
                reasoning_hidden_dim=config.reasoning_hidden_dim,
                reasoning_num_layers=config.reasoning_num_layers,
                reasoning_num_heads=config.reasoning_num_heads,
                reasoning_max_length=config.reasoning_max_length,
                conditioning_dim=config.reasoning_hidden_dim,
                dropout=0.1,
                vocab_size=config.reasoning_vocab_size,
            )
            print(f"🧠 Vision-Grounded Reasoning enabled (like Alpamayo-R1):")
            print(f"   ✅ Explicit CoT: model learns to 'see' box size from vision")
            print(f"   ✅ Works with simple prompt: 'Depalletize the box'")
            print(f"   ✅ {config.reasoning_num_layers} decoder layers")
            print(f"   ✅ Max CoT length: {config.reasoning_max_length}")
            print(f"   ✅ Conditioning dim: {config.reasoning_hidden_dim}")
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
        
        # Generate language-conditioned features if reasoning head is enabled
        # 关键改进：使用LanguageConditionedHead，直接从backbone_features提取语言条件化信息
        # 
        # 与原来的区别：
        # 1. 不需要分类标签（action_decision_labels）
        # 2. 不生成reasoning trace（没有reasoning_logits）
        # 3. 直接从已编码的language prompt中提取task-relevant信息
        # 
        # 这样30cm和60cm的箱子可以通过language prompt自然区分：
        # - "wider green box of type 4611 (40×60×11 cm)"
        # - "narrower green box of type 4322 (40×30×22 cm)"
        reasoning_logits = None
        reasoning_conditioning = None
        action_decision_logits = None
        reasoning_trace_loss = None
        action_decision_loss = None
        total_reasoning_loss = None
        
        if self.config.use_coc_reasoning and self.reasoning_head is not None:
            backbone_features = backbone_output.backbone_features  # (B, T, backbone_embedding_dim)
            backbone_attention_mask = backbone_output.backbone_attention_mask  # (B, T)
            
            # 获取训练标签（如果有）
            reasoning_labels = None
            action_decision_labels = None
            if hasattr(action_input, "get"):
                reasoning_labels = action_input.get("reasoning_labels", None)
                action_decision_labels = action_input.get("action_decision_labels", None)
            elif hasattr(action_input, "data"):
                reasoning_labels = action_input.data.get("reasoning_labels", None)
                action_decision_labels = action_input.data.get("action_decision_labels", None)
            
            # VisionGroundedReasoningHead - 从视觉中显式生成CoT推理
            # 支持训练时的teacher forcing
            reasoning_logits, reasoning_conditioning, action_decision_logits = self.reasoning_head(
                backbone_features=backbone_features,
                attention_mask=backbone_attention_mask,
                reasoning_labels=reasoning_labels,
                action_decision_labels=action_decision_labels,
            )
            
            # 计算CoT reasoning trace损失
            if reasoning_labels is not None and reasoning_logits is not None:
                # Shift labels for next-token prediction
                # reasoning_logits: (B, L, vocab_size), reasoning_labels: (B, L)
                # 标签向右移一位：预测下一个token
                shift_logits = reasoning_logits[:, :-1, :].contiguous()
                shift_labels = reasoning_labels[:, 1:].contiguous()
                
                reasoning_trace_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=0,  # 忽略pad token
                    reduction="mean"
                )
            
            # 计算action decision分类损失
            if action_decision_labels is not None and action_decision_logits is not None:
                action_decision_loss = F.cross_entropy(
                    action_decision_logits,
                    action_decision_labels,
                    reduction="mean"
                )
            
            # 合并reasoning损失
            if reasoning_trace_loss is not None and action_decision_loss is not None:
                total_reasoning_loss = (
                    self.config.reasoning_loss_weight * reasoning_trace_loss + 
                    self.config.action_decision_loss_weight * action_decision_loss
                )
            elif reasoning_trace_loss is not None:
                total_reasoning_loss = self.config.reasoning_loss_weight * reasoning_trace_loss
            elif action_decision_loss is not None:
                total_reasoning_loss = self.config.action_decision_loss_weight * action_decision_loss

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
        
        # VisionGroundedReasoningHead - 从视觉中显式生成CoT推理
        # 模型学会"看"到箱子尺寸，而不是依赖prompt
        reasoning_conditioning = None
        action_decision_logits = None
        
        if self.config.use_coc_reasoning and self.reasoning_head is not None:
            backbone_features = backbone_output.backbone_features  # (B, T, backbone_embedding_dim)
            backbone_attention_mask = backbone_output.backbone_attention_mask  # (B, T)
            
            # 推理模式：自回归生成CoT
            reasoning_logits, reasoning_conditioning, action_decision_logits = self.reasoning_head(
                backbone_features=backbone_features,
                attention_mask=backbone_attention_mask,
                reasoning_labels=None,  # 推理模式不需要标签
            )
            
            # 打印预测的action decision
            if action_decision_logits is not None:
                predicted_decision_idx = torch.argmax(action_decision_logits, dim=-1)  # (B,)
                decision_names = VisionGroundedReasoningHead.ACTION_DECISIONS
                predicted_decisions = [decision_names[idx.item()] for idx in predicted_decision_idx]
                print(f"🧠 Vision-Grounded Reasoning:")
                print(f"   → Predicted action decision: {predicted_decisions}")

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
