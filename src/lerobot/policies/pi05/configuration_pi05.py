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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # ============================================================================
    # π*0.6 RECAP Value Function 配置
    # ============================================================================
    # 训练模式:
    #   "policy" - 标准策略训练 (flow matching + action MSE loss)
    #   "value_function" - 仅训练 Value Function (MSE loss on target_value)
    training_mode: str = "policy"
    # 是否启用 Value Function（用于 RECAP 训练）
    enable_value_function: bool = False
    # Value Function 使用的 Gemma 变体（较小的模型，默认 gemma_300m ≈ 270M 参数）
    value_function_variant: str = "gemma_300m"
    # 是否与策略网络共享 SigLIP 视觉编码器
    value_function_share_vision_encoder: bool = True
    # Value Head MLP 的 dropout 率
    value_head_dropout: float = 0.1
    # Value Function 的学习率（通常与策略网络不同）
    value_function_lr: float = 1e-4
    # Value Function 的权重衰减
    value_function_weight_decay: float = 0.01

    # Advantage 计算配置
    # Advantage 计算的 N-step lookahead（None=预训练模式使用整个 episode, 50=微调模式）
    advantage_n_steps: int | None = None
    # 正优势数据的目标比例（预训练: 0.3, 微调: 0.4）
    advantage_positive_ratio: float = 0.3
    # Advantage conditioning 的 dropout 比例（用于 CFG 训练）
    advantage_conditioning_dropout: float = 0.3

    # 是否启用 advantage conditioning（策略网络条件化在 advantage indicator 上）
    enable_advantage_conditioning: bool = False
    # 推理时的 CFG beta 系数（β > 1 锐化分布，偏向高 advantage 动作）
    cfg_beta: float = 1.5

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.training_mode not in ["policy", "value_function"]:
            raise ValueError(
                f"Invalid training_mode: {self.training_mode}. Must be 'policy' or 'value_function'"
            )

        # 如果 training_mode 是 value_function，自动启用 enable_value_function
        if self.training_mode == "value_function":
            self.enable_value_function = True

        # Validate Value Function configuration
        if self.enable_value_function:
            if self.value_function_variant not in ["gemma_300m", "gemma_2b"]:
                raise ValueError(f"Invalid value_function_variant: {self.value_function_variant}")
            if not 0.0 <= self.advantage_positive_ratio <= 1.0:
                raise ValueError(
                    f"advantage_positive_ratio must be in [0, 1], got {self.advantage_positive_ratio}"
                )
            if not 0.0 <= self.advantage_conditioning_dropout <= 1.0:
                raise ValueError(
                    f"advantage_conditioning_dropout must be in [0, 1], got {self.advantage_conditioning_dropout}"
                )

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        # Value Function 训练模式使用独立的学习率
        lr = self.value_function_lr if self.training_mode == "value_function" else self.optimizer_lr
        return AdamWConfig(
            lr=lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay if self.training_mode != "value_function"
            else self.value_function_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        # Value Function 训练模式使用独立的学习率
        peak_lr = self.value_function_lr if self.training_mode == "value_function" else self.optimizer_lr
        decay_lr = peak_lr * 0.1 if self.training_mode == "value_function" else self.scheduler_decay_lr
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=peak_lr,
            decay_lr=decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
