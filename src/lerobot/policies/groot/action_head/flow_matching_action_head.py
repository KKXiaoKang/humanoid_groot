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
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})

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
        
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        
        # For velocity, extract only the actual action dimensions (first actual_action_dim)
        # This matches the original data dimension before padding
        velocity = actions[:, :, :self.actual_action_dim] - noise[:, :, :self.actual_action_dim]

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
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
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        
        # Slice out only the action portion of model output
        model_output_actions = model_output[:, -actions.shape[1] :]
        
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
        
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature, rtc_enabled: bool, **kwargs) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        # Use encoder_action_dim for internal processing (compatible with pretrained model)
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
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

        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            def denoise_step_partial_call(input_x_t, current_timestep=t_discretized, state_features=state_features, vl_embs=vl_embs, embodiment_id=embodiment_id):
                return self.denoise_step(x_t=input_x_t, timestep=current_timestep, vl_embs=vl_embs, state_features=state_features, embodiment_id=embodiment_id)

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

        actions_output = x_t[:, :, :self.actual_action_dim]
        return BatchFeature(data={"action_pred": actions_output})

    def denoise_step(self, x_t: torch.Tensor, timestep, vl_embs, state_features, embodiment_id) -> torch.Tensor:
        """
        单步预测 velocity
        """
        # 单步调用 _predict_velocity
        batch_size = x_t.shape[0]
        # timesteps_tensor = torch.full(size=(batch_size,), fill_value=timestep.item(), device=x_t.device)
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=timestep, device=x_t.device)
        v_t = self._predict_velocity(vl_embs, state_features, x_t, timesteps_tensor, embodiment_id)
        return v_t

    def _predict_velocity(
            self,
            vl_embs: torch.Tensor,
            state_features: torch.Tensor,
            actions: torch.Tensor,
            timesteps_tensor: torch.Tensor,
            embodiment_id: torch.Tensor,
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
