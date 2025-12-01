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
    
    # Loss weights for different action heads
    arm_loss_weight: float = field(default=1.0, metadata={"help": "Arm absolute position loss weight"})
    claw_loss_weight: float = field(default=1.0, metadata={"help": "Claw position loss weight"})
    
    # Learnable uncertainty weights (参考 https://arxiv.org/pdf/1705.07115)
    use_learnable_loss_weights: bool = field(default=False, metadata={"help": "Enable learnable loss weights based on uncertainty"})
    
    # Pretrained action dimension (for compatibility with pretrained models)
    pretrained_action_dim: int = field(default=None, metadata={"help": "Action dimension of pretrained model (for compatibility)"})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Validate multi-head configuration
        if self.use_multi_action_heads:
            expected_action_dim = self.action_arm_dim + self.action_claw_dim
            if self.action_dim is not None and self.action_dim != expected_action_dim:
                # If pretrained_action_dim is set, allow mismatch (we'll pad/truncate)
                if self.pretrained_action_dim is None:
                    raise ValueError(
                        f"When using multi-action heads, action_dim ({self.action_dim}) must equal "
                        f"action_arm_dim ({self.action_arm_dim}) + action_claw_dim ({self.action_claw_dim}) = {expected_action_dim}"
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
            self.action_arm_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=config.action_arm_dim,
            )
            self.action_claw_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=config.action_claw_dim,
            )
            self.action_decoder = None  # Not used in multi-head mode
            print(f"📊 Multi-head action: arm({config.action_arm_dim}D) + claw({config.action_claw_dim}D) = {config.action_arm_dim + config.action_claw_dim}D")
        else:
            self.action_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.action_dim,
            )
            self.action_arm_decoder = None
            self.action_claw_decoder = None
        
        # Learnable loss weights (参考 https://arxiv.org/pdf/1705.07115)
        if config.use_learnable_loss_weights and config.use_multi_action_heads:
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

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            if self.config.use_multi_action_heads:
                self.action_arm_decoder.requires_grad_(False)
                self.action_claw_decoder.requires_grad_(False)
            else:
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
                    self.action_arm_decoder.eval()
                    self.action_claw_decoder.eval()
                else:
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
            pred_arm = self.action_arm_decoder(model_output_actions, embodiment_id)
            pred_claw = self.action_claw_decoder(model_output_actions, embodiment_id)
            pred_actions = torch.cat([pred_arm, pred_claw], dim=-1)  # (B, T, action_dim)
            
            # Split ground truth velocity into corresponding parts
            velocity_arm = velocity[:, :, :self.config.action_arm_dim]  # (B, T, action_arm_dim)
            velocity_claw = velocity[:, :, self.config.action_arm_dim:]  # (B, T, action_claw_dim)
            
            # Compute loss for each head
            # action_mask is (B, T, encoder_action_dim), but we only need the first actual_action_dim
            # Since velocity is already extracted from first actual_action_dim, we use the corresponding mask
            action_mask = action_input.action_mask[:, :, :self.actual_action_dim]  # (B, T, actual_action_dim)
            # Split mask for arm and claw
            action_mask_arm = action_mask[:, :, :self.config.action_arm_dim]  # (B, T, action_arm_dim)
            action_mask_claw = action_mask[:, :, self.config.action_arm_dim:]  # (B, T, action_claw_dim)
            
            loss_arm = F.mse_loss(pred_arm, velocity_arm, reduction="none") * action_mask_arm
            loss_claw = F.mse_loss(pred_claw, velocity_claw, reduction="none") * action_mask_claw
            
            # Use learnable weights or fixed weights
            if self.config.use_learnable_loss_weights and self.task_log_sigma is not None:
                # Loss = Σ [1/(2σ²) * L_i + log(σ)]
                # 这里使用 log(σ) 作为可学习参数，避免 σ 为负
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
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
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

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            
            # Slice out only the action portion of model output
            model_output_actions = model_output[:, -self.action_horizon :]
            
            # Multi-head action prediction
            if self.config.use_multi_action_heads:
                pred_arm = self.action_arm_decoder(model_output_actions, embodiment_id)
                pred_claw = self.action_claw_decoder(model_output_actions, embodiment_id)
                pred_velocity = torch.cat([pred_arm, pred_claw], dim=-1)  # (B, T, action_dim)
            else:
                pred = self.action_decoder(model_output_actions, embodiment_id)
                pred_velocity = pred

            # Update actions using euler integration.
            # If using multi-head, pred_velocity is actual_action_dim, but actions is encoder_action_dim
            if self.encoder_action_dim != self.actual_action_dim:
                # Pad pred_velocity to match encoder_action_dim
                if self.encoder_action_dim > self.actual_action_dim:
                    pad_size = self.encoder_action_dim - self.actual_action_dim
                    padding = torch.zeros(
                        (pred_velocity.shape[0], pred_velocity.shape[1], pad_size),
                        device=pred_velocity.device,
                        dtype=pred_velocity.dtype
                    )
                    pred_velocity_padded = torch.cat([pred_velocity, padding], dim=-1)
                else:
                    pred_velocity_padded = pred_velocity[:, :, :self.encoder_action_dim]
                actions = actions + dt * pred_velocity_padded
            else:
                actions = actions + dt * pred_velocity
        
        # Return only the actual action dimensions
        actions_output = actions[:, :, :self.actual_action_dim]
        return BatchFeature(data={"action_pred": actions_output})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
