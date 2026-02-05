#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, ProcessorMixin
else:
    AutoProcessor = None
    ProcessorMixin = object

from lerobot.configs.types import (
    FeatureType,
    NormalizationMode,
    PolicyFeature,
)
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    HF_LEROBOT_HOME,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

# Defaults for Eagle processor locations
DEFAULT_TOKENIZER_ASSETS_REPO = "lerobot/eagle2hg-processor-groot-n1p5"


# ============================================================================
# 6D Rotation Conversion Utilities (for Relative EEF Action)
# ============================================================================

def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    
    Uses Gram-Schmidt orthogonalization to ensure valid rotation matrix.
    
    Args:
        rot6d: 6D rotation vector [R11, R21, R31, R12, R22, R32]
               Shape: (..., 6) or (6,)
    
    Returns:
        3x3 rotation matrix. Shape: (..., 3, 3) or (3, 3)
    """
    # Handle both batched and unbatched inputs
    original_shape = rot6d.shape
    is_batched = len(original_shape) > 1
    if not is_batched:
        rot6d = rot6d.unsqueeze(0)
    
    # Extract first two columns
    col1 = rot6d[..., :3]  # [R11, R21, R31]
    col2 = rot6d[..., 3:6]  # [R12, R22, R32]
    
    # Gram-Schmidt orthogonalization
    # Normalize first column
    col1_norm = torch.norm(col1, dim=-1, keepdim=True)
    col1_normalized = torch.where(
        col1_norm < 1e-8,
        torch.tensor([1.0, 0.0, 0.0], device=rot6d.device, dtype=rot6d.dtype),
        col1 / (col1_norm + 1e-8)
    )
    
    # Orthogonalize and normalize second column
    col2_projected = col2 - (col2 * col1_normalized).sum(dim=-1, keepdim=True) * col1_normalized
    col2_norm = torch.norm(col2_projected, dim=-1, keepdim=True)
    col2_normalized = torch.where(
        col2_norm < 1e-8,
        torch.tensor([0.0, 1.0, 0.0], device=rot6d.device, dtype=rot6d.dtype),
        col2_projected / (col2_norm + 1e-8)
    )
    
    # Third column via cross product
    col3_normalized = torch.cross(col1_normalized, col2_normalized, dim=-1)
    
    # Stack to form rotation matrix
    rotation_matrix = torch.stack([col1_normalized, col2_normalized, col3_normalized], dim=-1)
    
    if not is_batched:
        rotation_matrix = rotation_matrix.squeeze(0)
    
    return rotation_matrix


def matrix_to_rot6d(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to 6D rotation representation.
    
    Args:
        rotation_matrix: 3x3 rotation matrix. Shape: (..., 3, 3) or (3, 3)
    
    Returns:
        6D rotation vector [R11, R21, R31, R12, R22, R32]. Shape: (..., 6) or (6,)
    """
    # Handle both batched and unbatched inputs
    original_shape = rotation_matrix.shape
    is_batched = len(original_shape) > 2
    if not is_batched:
        rotation_matrix = rotation_matrix.unsqueeze(0)
    
    # Extract first two columns
    col1 = rotation_matrix[..., :, 0]  # [R11, R21, R31]
    col2 = rotation_matrix[..., :, 1]  # [R12, R22, R32]
    
    # Concatenate to form 6D representation
    rot6d = torch.cat([col1, col2], dim=-1)
    
    if not is_batched:
        rot6d = rot6d.squeeze(0)
    
    return rot6d


def make_groot_pre_post_processors(
    config: GrootConfig, 
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_num_frames: int | None = None,
    num_processes: int = 1,  # Number of GPUs/processes for multi-GPU training
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create preprocessor and postprocessor for Groot policy.

    This creates a processing pipeline that transforms LeRobot data format into
    the format expected by Isaac-GR00T models:

    Preprocessing steps:
    1. Optional key renaming (dataset-specific key mapping)
    2. Add batch dimension to unbatched data
    3. Pack video/state/action/language/embodiment and apply optional min-max normalization before padding
    4. Encode video+language with Eagle VLM into intermediate eagle_content
    5. Collate eagle_content into batched eagle_* tensors
    6. Move tensors to device (GPU)

    NOTE: We optionally apply min-max normalization to STATE and ACTION using
    dataset-provided statistics prior to padding, mapping values to [-1, 1].
    This mirrors SO100-style preprocessing and keeps scales consistent with GR00T.

    Args:
        config: Groot configuration containing data_config, embodiment_tag, etc.
        dataset_stats: Optional per-key min/max statistics for normalization before padding.
        dataset_num_frames: Optional total number of frames in dataset. Used to detect when first epoch
                           is complete for relative action stats accumulation (Delta eef mode).

    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """
    # Get action space type from config (default to "Absolute joint" for backward compatibility)
    action_space_type = getattr(config, 'action_space_type', "Absolute joint")
    
    # Get relative action reference mode from config (default to "state" for recommended behavior)
    relative_action_reference_mode = getattr(config, 'relative_action_reference_mode', "state")
    
    # Get horizon/dimension parameters from config
    # These should match the config used for the pretrained model
    # Default values match most GR00T configs (state_horizon=1, action_horizon=16)
    state_horizon = 1
    # CRITICAL: Pretrained GR00T models use action_horizon=16 max!
    # The model architecture hardcodes this limit
    action_horizon = min(config.chunk_size, 16)
    max_state_dim = config.max_state_dim
    max_action_dim = config.max_action_dim

    # Pass raw dataset_stats; normalization will occur inside pack step before padding
    padded_stats = dataset_stats or {}

    # Define feature specs for optional normalization steps
    _features: dict[str, PolicyFeature] = {
        # Observation features (only add those we may normalize)
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_horizon, max_state_dim)),
        # Action feature
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_horizon, max_action_dim)),
    }

    # Normalize STATE and ACTION with min_max (SO100-like default)
    _norm_map = {
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
        FeatureType.STATE: NormalizationMode.MIN_MAX,
    }
    
    # Setup partial normalization for eef action spaces
    action_component_indices = None
    
    if action_space_type in ["Delta eef", "Absolute eef"]:
        # For absolute eef pose (20D):
        # - 3维左手 eef position (x, y, z) -> indices 0-2
        # - 6维左手 eef 6D 旋转表示 [R11, R21, R31, R12, R22, R32] -> indices 3-8
        # - 3维右手 eef position (x, y, z) -> indices 9-11
        # - 6维右手 eef 6D 旋转表示 [R11, R21, R31, R12, R22, R32] -> indices 12-17
        # - 1维左夹爪开合程度 -> indices 18
        # - 1维右夹爪开合程度 -> indices 19
        action_component_indices = {
            "left_eef_pos": (0, 3),
            "left_eef_rot6d": (3, 9),
            "right_eef_pos": (9, 12),
            "right_eef_rot6d": (12, 18),
            "left_gripper": (18, 19),
            "right_gripper": (19, 20),
        }
        print(f"✅ Partial normalization enabled for action space: {action_space_type}")
        print(f"   Components: {list(action_component_indices.keys())}")
        print(f"   6D rotation components (left_eef_rot6d, right_eef_rot6d) will use IDENTITY normalization")
    else:
        print(f"📊 Using standard normalization for action space: {action_space_type}")

    # Determine env action dimension from config (simple, object-like PolicyFeature)
    try:
        env_action_dim = int(config.output_features["action"].shape[0])
    except Exception:
        env_action_dim = 0

    input_steps: list[ProcessorStep] = [
        # 1. Rename keys if needed (e.g., dataset-specific camera names)
        # Leave empty for now - add mappings if your dataset uses different key names
        RenameObservationsProcessorStep(rename_map={}),
        # 2. Add batch dimension for single samples
        AddBatchDimensionProcessorStep(),
        # 3. Pack video/state/action/language/embodiment; apply optional min-max normalization before padding
        GrootPackInputsStep(
            state_horizon=state_horizon,
            action_horizon=action_horizon,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            language_key="task",
            formalize_language=False,
            embodiment_tag=config.embodiment_tag,
            normalize_min_max=True,
            stats=padded_stats,
            action_space_type=action_space_type,
            action_component_indices=action_component_indices,
            relative_action_reference_mode=relative_action_reference_mode,  # Reference pose mode for Delta eef
            dataset_num_frames=dataset_num_frames,
            num_processes=num_processes,  # Pass num_processes for multi-GPU training
        ),
        # 4. Eagle encode (creates eagle_content)
        GrootEagleEncodeStep(
            tokenizer_assets_repo=config.tokenizer_assets_repo,
        ),
        # 5. Collate eagle_content -> eagle_* tensors
        GrootEagleCollateStep(
            tokenizer_assets_repo=config.tokenizer_assets_repo,
        ),
        # 6. Move to device
        DeviceProcessorStep(device=config.device),
    ]

    # Postprocessing: slice to env action dim and unnormalize to env scale, then move to CPU
    output_steps: list[ProcessorStep] = [
        GrootActionUnpackUnnormalizeStep(
            env_action_dim=env_action_dim,
            stats=padded_stats,
            normalize_min_max=True,
            action_space_type=action_space_type,
            action_component_indices=action_component_indices,
        ),
        # Finally, move to CPU for env interaction
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


# GR00T specific processor steps


def _to_uint8_np_bhwc(img_t: torch.Tensor) -> np.ndarray:
    # img_t: (B, C, H, W) float in [0,1] or uint8
    if img_t.dtype.is_floating_point:
        img_t = (img_t.clamp(0, 1) * 255.0).to(torch.uint8)
    return rearrange(img_t.cpu().numpy(), "b c h w -> b h w c")


def _build_eagle_processor(tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO) -> ProcessorMixin:
    # Validate that the cache directory is ready. If not, instruct the user.
    cache_dir = HF_LEROBOT_HOME / tokenizer_assets_repo
    required = [
        cache_dir / "processor_config.json",
        cache_dir / "preprocessor_config.json",
        cache_dir / "image_processing_eagle2_5_vl_fast.py",
    ]
    if not all(p.exists() for p in required):
        raise FileNotFoundError(
            f"[GROOT] Eagle processor cache at '{cache_dir}' is not populated. "
            "Vendor files are copied during model creation. Create the policy/model first, "
            "or call ensure_eagle_cache_ready() before building processors."
        )
    proc = AutoProcessor.from_pretrained(str(cache_dir), trust_remote_code=True, use_fast=True)
    proc.tokenizer.padding_side = "left"
    return proc


@dataclass
@ProcessorStepRegistry.register(name="groot_pack_inputs_v3")
class GrootPackInputsStep(ProcessorStep):
    state_horizon: int = 1
    action_horizon: int = 16
    max_state_dim: int = 64
    max_action_dim: int = 32
    language_key: str = "task"
    formalize_language: bool = False
    embodiment_tag: str = "new_embodiment"
    embodiment_mapping: dict[str, int] = field(
        default_factory=lambda: {
            "new_embodiment": 31,  # Match original GR00T EMBODIMENT_TAG_MAPPING
            "oxe_droid": 17,
            "agibot_genie1": 26,
            "gr1": 24,
            "so100": 2,
            "unitree_g1": 3,
        }
    )
    # Min-max normalization (SO100-like) applied BEFORE padding
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None
    # For partial normalization of action (e.g., 6D rotation representation)
    action_space_type: str | None = None  # "Delta eef", "Absolute eef", "Absolute joint", etc.
    action_component_indices: dict[str, tuple[int, int]] | None = None  # e.g., {"left_eef_pos": (0, 3), ...}
    _relative_action_conversion_logged: bool = False  # Track if we've logged the conversion
    
    # Reference pose mode for Delta eef (relative action) training
    # - "state": Use observation.state as reference pose (RECOMMENDED)
    # - "action": Use action[0] as reference pose (legacy behavior)
    relative_action_reference_mode: str = "state"
    
    # Dynamic normalization statistics for relative action position components
    # These are accumulated during training and saved to model config
    # Note: This field accepts a value in __init__ (from JSON config) but is ignored.
    # The actual value should be loaded via load_state_dict() after initialization.
    relative_action_stats: dict[str, dict[str, torch.Tensor]] | None = field(
        default=None, init=True, repr=False
    )  # Accepts from JSON but ignored - should be loaded via load_state_dict
    _relative_stats_initialized: bool = field(default=False, init=False, repr=False)
    _relative_stats_frozen: bool = field(default=False, init=False, repr=False)  # 是否已冻结统计值
    freeze_stats_after_first_epoch: bool = field(
        default=True, 
        metadata={"help": "If True, freeze relative action stats after first epoch (recommended for fixed datasets)"}
    )
    dataset_num_frames: int | None = field(
        default=None,
        metadata={"help": "Total number of frames in dataset. Used to detect when first epoch is complete."}
    )
    num_processes: int = field(
        default=1,
        metadata={"help": "Number of GPUs/processes for multi-GPU training. Used to adjust dataset_num_frames threshold."}
    )

    def __post_init__(self):
        """
        Post-initialization hook to handle relative_action_stats from JSON config.
        
        When loading from JSON config, relative_action_stats may be passed to __init__,
        but it should be ignored (actual loading happens via load_state_dict).
        This method resets relative_action_stats to None if it was passed from JSON.
        """
        # If relative_action_stats was passed from JSON config (not None and not tensor dict),
        # reset it to None - actual loading happens via load_state_dict
        if self.relative_action_stats is not None:
            # Check if it's from JSON (dict with list values) vs actual tensor dict
            # JSON format: {"left_eef_pos": {"min": [0.1, 0.2], "max": [0.3, 0.4], "count": 100}}
            # Tensor format: {"left_eef_pos": {"min": tensor(...), "max": tensor(...), "count": tensor(...)}}
            first_comp = next(iter(self.relative_action_stats.values()), None)
            if first_comp is not None:
                first_stat = next(iter(first_comp.values()), None)
                # If it's a list (from JSON), reset to None (will be loaded via load_state_dict)
                if isinstance(first_stat, list):
                    self.relative_action_stats = None

    def _convert_absolute_to_relative_eef_action(
        self, 
        absolute_action: torch.Tensor,
        current_state: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Convert absolute EEF pose to relative EEF action.
        
        **重要设计变更 (2026-01):**
        使用 current_state（observation.state）作为 reference pose，而不是 action[0]！
        
        这样做的原因：
        1. 符合马尔科夫性质：state 是当前状态，action 是要去的目标
        2. 训练和推理一致：训练时用 dataset 中的 state，推理时用 FK 计算的 state
        3. 自然 cover 跟踪误差：数据集中的 state 和前一帧 action 之间本来就有误差
        
        relative_action = absolute_action - current_state
        表示"从当前实际状态出发，需要移动多少到达目标"
        
        Args:
            absolute_action: Absolute EEF pose tensor. Shape: (B, T, D)
                            Expected structure for 20D eef:
                            [left_pos(3), left_rot6d(6), right_pos(3), right_rot6d(6), gripper(2)]
            current_state: Current observation state (absolute EEF pose). Shape: (B, D)
                          If None, falls back to using action[0] (legacy behavior, not recommended)
        
        Returns:
            Relative EEF action tensor. Shape: (B, T, D)
        """
        if self.action_component_indices is None:
            return absolute_action
        
        b, t, d = absolute_action.shape
        relative_action = absolute_action.clone()
        
        # 根据 relative_action_reference_mode 选择 reference pose 来源
        # - "state": 使用 observation.state 作为 reference（推荐，训练推理一致）
        # - "action": 使用 action[0] 作为 reference（旧方式）
        
        if self.relative_action_reference_mode == "state" and current_state is not None:
            # 模式1: 使用 observation.state 作为 reference pose（推荐）
            # current_state: (B, D) -> (B, 1, D)
            if current_state.dim() == 2:
                ref_pose = current_state.unsqueeze(1)  # (B, 1, D)
            else:
                ref_pose = current_state[:, 0:1, :]  # (B, 1, D)
            
            # Log once that we're using state as reference
            if not self._relative_action_conversion_logged:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"🔄 [Delta EEF] Converting absolute EEF to relative actions using STATE as reference "
                    f"(action shape: {absolute_action.shape}, state shape: {current_state.shape})"
                )
                logger.info(
                    f"   ✅ Mode: relative_action_reference_mode='state' (RECOMMENDED)"
                )
                logger.info(
                    f"   ✅ Training and inference use consistent reference (actual state, not action command)"
                )
                self._relative_action_conversion_logged = True
        
        elif self.relative_action_reference_mode == "action":
            # 模式2: 使用 action[0] 作为 reference pose（旧方式）
            ref_pose = absolute_action[:, 0:1, :]  # (B, 1, D)
            
            if not self._relative_action_conversion_logged:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"🔄 [Delta EEF] Converting absolute EEF to relative actions using ACTION[0] as reference "
                    f"(action shape: {absolute_action.shape})"
                )
                logger.info(
                    f"   ⚠️  Mode: relative_action_reference_mode='action' (legacy)"
                )
                logger.info(
                    f"   ⚠️  Note: This may cause train-inference mismatch if robot has tracking errors!"
                )
                self._relative_action_conversion_logged = True
        
        else:
            # Fallback: state mode but no state provided
            ref_pose = absolute_action[:, 0:1, :]  # (B, 1, D)
            
            if not self._relative_action_conversion_logged:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"⚠️  [Delta EEF] relative_action_reference_mode='state' but NO STATE PROVIDED!"
                )
                logger.warning(
                    f"   Falling back to action[0] as reference. This is NOT recommended."
                )
                self._relative_action_conversion_logged = True
        
        # Process each component
        for component_name, (start_idx, end_idx) in self.action_component_indices.items():
            if end_idx > d:
                continue
            
            # Extract component from reference and all timesteps
            ref_component = ref_pose[:, :, start_idx:end_idx]  # (B, 1, component_dim)
            abs_components = absolute_action[:, :, start_idx:end_idx]  # (B, T, component_dim)
            
            if "pos" in component_name:
                # Position: simple subtraction
                # relative_pos = pos_i - pos_ref
                rel_components = abs_components - ref_component  # Broadcast: (B, T, dim) - (B, 1, dim)
                relative_action[:, :, start_idx:end_idx] = rel_components
                
            elif "rot6d" in component_name:
                # Rotation: SO(3) relative rotation
                # R_rel = R_ref^-1 @ R_i = R_ref^T @ R_i
                
                # Convert 6D to rotation matrices
                # Reshape for batch processing: (B*T, 6) -> (B*T, 3, 3)
                ref_rot6d = ref_component.squeeze(1)  # (B, 6)
                abs_rot6d = abs_components.reshape(b * t, -1)  # (B*T, 6)
                
                # Convert to matrices
                ref_matrices = rot6d_to_matrix(ref_rot6d)  # (B, 3, 3)
                abs_matrices = rot6d_to_matrix(abs_rot6d)  # (B*T, 3, 3)
                
                # Compute relative rotation: R_rel = R_ref^T @ R_i
                ref_matrices_expanded = ref_matrices.unsqueeze(1).expand(-1, t, -1, -1)  # (B, T, 3, 3)
                ref_matrices_flat = ref_matrices_expanded.reshape(b * t, 3, 3)  # (B*T, 3, 3)
                
                # Relative rotation: R_ref^T @ R_i
                rel_matrices = torch.bmm(ref_matrices_flat.transpose(-2, -1), abs_matrices)  # (B*T, 3, 3)
                
                # Convert back to 6D
                rel_rot6d = matrix_to_rot6d(rel_matrices)  # (B*T, 6)
                rel_rot6d = rel_rot6d.reshape(b, t, -1)  # (B, T, 6)
                
                relative_action[:, :, start_idx:end_idx] = rel_rot6d
                
            elif "gripper" in component_name:
                # Gripper: keep as-is (already relative in most cases, or can be kept absolute)
                # For now, keep absolute values (can be changed if needed)
                pass
        
        return relative_action
    
    def update_relative_action_stats(self, relative_action: torch.Tensor):
        """
        更新relative action position组件的统计值（min/max/count）。
        
        这个方法在训练过程中被调用，用于累积relative action的统计值。
        只对position组件（left_eef_pos, right_eef_pos）进行统计，因为：
        1. rotation组件使用IDENTITY归一化，不需要统计
        2. gripper组件可以使用absolute stats
        
        对于固定数据集，建议设置freeze_stats_after_first_epoch=True，
        这样在第一个epoch完成后会停止累积统计值，避免不必要的计算。
        
        Args:
            relative_action: Relative EEF action tensor. Shape: (B, T, D)
        """
        if self.action_space_type != "Delta eef" or self.action_component_indices is None:
            return
        
        # 如果统计值已冻结，跳过更新
        if self._relative_stats_frozen:
            return
        
        if not self._relative_stats_initialized:
            # 初始化统计值字典
            # 从输入tensor获取设备信息，确保统计值在正确的设备上
            device = relative_action.device
            dtype = relative_action.dtype
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info("📊 [Delta EEF] Initializing relative action stats accumulation:")
            logger.info(f"   Action space: {self.action_space_type}")
            logger.info(f"   Device: {device}")
            logger.info(f"   Dtype: {dtype}")
            logger.info(f"   Freeze after first epoch: {self.freeze_stats_after_first_epoch}")
            logger.info(f"   Num processes (GPUs): {self.num_processes}")
            if self.dataset_num_frames is not None:
                threshold = self.dataset_num_frames / self.num_processes
                logger.info(f"   Dataset num frames: {self.dataset_num_frames}")
                logger.info(f"   Threshold per GPU: {threshold:.0f} frames (will freeze after {threshold:.0f} frames processed)")
            else:
                logger.warning("   ⚠️  dataset_num_frames not set - auto-freeze disabled")
            
            self.relative_action_stats = {}
            for comp_name, (start_idx, end_idx) in self.action_component_indices.items():
                if "pos" in comp_name:
                    # 只对position组件初始化统计值
                    comp_dim = end_idx - start_idx
                    self.relative_action_stats[comp_name] = {
                        "min": torch.full((comp_dim,), float('inf'), dtype=dtype, device=device),
                        "max": torch.full((comp_dim,), float('-inf'), dtype=dtype, device=device),
                        "count": torch.tensor(0, dtype=torch.long, device=device),
                    }
                    logger.info(f"   ✅ Tracking {comp_name} (indices [{start_idx}:{end_idx}], dim={comp_dim})")
            self._relative_stats_initialized = True
            logger.info("   🚀 Stats accumulation started - will update during training")
        
        # 更新每个position组件的统计值
        b, t, d = relative_action.shape
        # 注意：dataset_num_frames是数据集的帧数（samples数），不是batch_size * action_horizon的累积
        # 每个batch处理了b个不同的frames，所以应该统计b而不是b*t
        frames_this_batch = b  # 实际处理的frames数
        device = relative_action.device  # 确保使用正确的设备
        
        for comp_name, (start_idx, end_idx) in self.action_component_indices.items():
            if comp_name not in self.relative_action_stats:
                continue
            
            # 提取组件数据 (B, T, comp_dim)
            comp_data = relative_action[:, :, start_idx:end_idx]
            
            # 计算当前batch的min/max（考虑所有timesteps）
            comp_min = comp_data.min(dim=0)[0].min(dim=0)[0]  # (comp_dim,)
            comp_max = comp_data.max(dim=0)[0].max(dim=0)[0]  # (comp_dim,)
            
            # 更新全局min/max
            current_min = self.relative_action_stats[comp_name]["min"]
            current_max = self.relative_action_stats[comp_name]["max"]
            current_count = self.relative_action_stats[comp_name]["count"]
            
            # 确保所有tensor都在同一设备上（防止设备不匹配）
            if current_min.device != device:
                current_min = current_min.to(device=device)
                current_max = current_max.to(device=device)
                current_count = current_count.to(device=device)
                self.relative_action_stats[comp_name]["min"] = current_min
                self.relative_action_stats[comp_name]["max"] = current_max
                self.relative_action_stats[comp_name]["count"] = current_count
            
            # 使用running min/max更新
            new_min = torch.minimum(current_min, comp_min)
            new_max = torch.maximum(current_max, comp_max)
            # 统计实际处理的frames数，而不是batch_size * action_horizon
            new_count = current_count + frames_this_batch
            
            self.relative_action_stats[comp_name]["min"] = new_min
            self.relative_action_stats[comp_name]["max"] = new_max
            self.relative_action_stats[comp_name]["count"] = new_count
            
            # 检查是否应该冻结统计值（第一个epoch完成）
            if self.freeze_stats_after_first_epoch and self.dataset_num_frames is not None:
                # 在多GPU训练时，每个GPU只处理 dataset_num_frames / num_processes 个frames
                # 所以阈值应该是 dataset_num_frames / num_processes
                threshold = self.dataset_num_frames / self.num_processes
                
                # 定期打印进度（每500个frames或每10%）
                progress_pct = (new_count / threshold) * 100
                if (new_count % 500 < frames_this_batch) or (int(progress_pct) % 10 == 0 and int(progress_pct) > 0):
                    # 使用一个简单的机制避免重复打印
                    last_logged_count = getattr(self, f'_last_logged_count_{comp_name}', -1)
                    if abs(new_count - last_logged_count) >= 500:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.debug(f"📊 [Delta EEF] Stats progress for {comp_name}: "
                                   f"frames_processed={new_count:.0f}/{threshold:.0f} ({progress_pct:.1f}%)")
                        setattr(self, f'_last_logged_count_{comp_name}', new_count)
                
                # 标记这个组件是否达到阈值
                if new_count >= threshold:
                    setattr(self, f'_component_reached_threshold_{comp_name}', True)
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"📈 [Delta EEF] First epoch complete for {comp_name}: "
                              f"frames_processed={new_count:.0f}/{threshold:.0f} "
                              f"(total_dataset={self.dataset_num_frames}, num_processes={self.num_processes}, {progress_pct:.1f}%)")
        
        # 在所有组件更新完成后，检查是否所有组件都达到阈值
        if self.freeze_stats_after_first_epoch and self.dataset_num_frames is not None and not self._relative_stats_frozen:
            threshold = self.dataset_num_frames / self.num_processes
            all_reached = True
            for comp_name in self.relative_action_stats.keys():
                count = self.relative_action_stats[comp_name]["count"]
                count_val = count.item() if isinstance(count, torch.Tensor) else count
                if count_val < threshold:
                    all_reached = False
                    break
            
            if all_reached:
                self.freeze_relative_action_stats()
    
    def freeze_relative_action_stats(self):
        """
        冻结relative action统计值，停止累积。
        
        对于固定数据集，在第一个epoch完成后调用此方法可以：
        1. 避免不必要的计算开销
        2. 确保统计值稳定（不会因为浮点误差产生微小变化）
        3. 符合统计学的直觉（统计值应该基于完整数据集计算一次）
        """
        if self._relative_stats_frozen:
            return
        
        self._relative_stats_frozen = True
        
        # 打印统计值摘要
        if self.relative_action_stats is not None:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("🔒 [Delta EEF] Freezing relative action stats after first epoch:")
            for comp_name, stats in self.relative_action_stats.items():
                count = stats["count"].item() if isinstance(stats["count"], torch.Tensor) else stats["count"]
                min_vals = stats["min"].tolist() if isinstance(stats["min"], torch.Tensor) else stats["min"]
                max_vals = stats["max"].tolist() if isinstance(stats["max"], torch.Tensor) else stats["max"]
                logger.info(f"   {comp_name}: count={count}, min={min_vals}, max={max_vals}")
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        def _align_vec(vec: Any, target_dim: int, *, default: float) -> torch.Tensor:
            t = torch.as_tensor(vec)
            t = t.flatten().to(
                dtype=torch.float32,
                device=next(
                    (v.device for v in obs.values() if isinstance(v, torch.Tensor)), torch.device("cpu")
                ),
            )
            d = int(t.shape[-1]) if t.numel() > 0 else 0
            if d == target_dim:
                return t
            if d < target_dim:
                pad = torch.full((target_dim - d,), default, dtype=t.dtype, device=t.device)
                return torch.cat([t, pad], dim=0)
            return t[:target_dim]

        def _min_max_norm(x: torch.Tensor, key: str) -> torch.Tensor:
            if not self.normalize_min_max:
                return x
            if self.stats is None or key not in self.stats:
                return x
            
            # Check if partial normalization is enabled for action
            if (key == "action" and 
                self.action_space_type in ["Delta eef", "Absolute eef"] and 
                self.action_component_indices is not None):
                return _min_max_norm_partial(x, key)
            
            # Standard normalization: apply to entire tensor
            stats_k = self.stats[key]
            last_dim = x.shape[-1]
            min_v = _align_vec(stats_k.get("min", torch.zeros(last_dim)), last_dim, default=0.0)
            max_v = _align_vec(stats_k.get("max", torch.ones(last_dim)), last_dim, default=1.0)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            mapped = 2 * (x - min_v) / safe_denom - 1
            return torch.where(mask, mapped, torch.zeros_like(mapped))
        
        def _min_max_norm_partial(x: torch.Tensor, key: str) -> torch.Tensor:
            """Apply partial min-max normalization for action with component indices.
            
            For relative actions (Delta eef), adjusts normalization ranges for position components
            since relative position distribution differs from absolute position distribution.
            """
            if self.action_component_indices is None:
                # Fallback to standard normalization
                stats_k = self.stats[key]
                last_dim = x.shape[-1]
                min_v = _align_vec(stats_k.get("min", torch.zeros(last_dim)), last_dim, default=0.0)
                max_v = _align_vec(stats_k.get("max", torch.ones(last_dim)), last_dim, default=1.0)
                denom = max_v - min_v
                mask = denom != 0
                safe_denom = torch.where(mask, denom, torch.ones_like(denom))
                mapped = 2 * (x - min_v) / safe_denom - 1
                return torch.where(mask, mapped, torch.zeros_like(mapped))
            
            stats_k = self.stats[key]
            result = x.clone()
            
            # Define which components should use IDENTITY (no normalization)
            rot6d_components = ["left_eef_rot6d", "right_eef_rot6d"]
            
            # Get full stats (from absolute eef pose)
            last_dim = x.shape[-1]
            min_v_full = _align_vec(stats_k.get("min", torch.zeros(last_dim)), last_dim, default=0.0)
            max_v_full = _align_vec(stats_k.get("max", torch.ones(last_dim)), last_dim, default=1.0)
            
            # For relative actions, adjust normalization ranges for position components
            # Relative position: rel_pos = abs_pos - ref_pos
            # Distribution range is approximately 2x the absolute range (centered at 0)
            is_relative_action = (self.action_space_type == "Delta eef")
            
            # Process each component
            for component_name, (start_idx, end_idx) in self.action_component_indices.items():
                if end_idx > last_dim:
                    continue  # Skip if indices exceed tensor dimension
                
                component = x[..., start_idx:end_idx]
                
                # Check if this component should use IDENTITY normalization
                if component_name in rot6d_components:
                    # 6D rotation: use IDENTITY (no normalization)
                    normalized_component = component
                else:
                    # Position or gripper: apply min-max normalization
                    min_v = min_v_full[start_idx:end_idx]
                    max_v = max_v_full[start_idx:end_idx]
                    
                    # For relative actions, adjust normalization range for position components
                    if is_relative_action and "pos" in component_name:
                        # Use dynamically accumulated relative_action_stats if available
                        if (self.relative_action_stats is not None and 
                            component_name in self.relative_action_stats):
                            # Use accumulated relative action stats
                            rel_stats = self.relative_action_stats[component_name]
                            rel_min = rel_stats["min"]
                            rel_max = rel_stats["max"]
                            
                            # Ensure stats are on the correct device and have correct shape
                            if isinstance(rel_min, torch.Tensor):
                                rel_min = rel_min.to(device=component.device, dtype=component.dtype)
                                rel_max = rel_max.to(device=component.device, dtype=component.dtype)
                            else:
                                rel_min = torch.as_tensor(rel_min, device=component.device, dtype=component.dtype)
                                rel_max = torch.as_tensor(rel_max, device=component.device, dtype=component.dtype)
                            
                            # Use accumulated stats directly
                            min_v = rel_min
                            max_v = rel_max
                            
                            # If stats are invalid (all inf), fallback to adjusted absolute range
                            if torch.any(torch.isinf(rel_min)) or torch.any(torch.isinf(rel_max)):
                                abs_range = torch.maximum(torch.abs(min_v_full[start_idx:end_idx]), 
                                                         torch.abs(max_v_full[start_idx:end_idx]))
                                rel_range = abs_range * 1.5
                                min_v = -rel_range
                                max_v = rel_range
                                default_range = torch.ones_like(abs_range) * 1.0
                                min_v = torch.where(abs_range < 0.1, -default_range, min_v)
                                max_v = torch.where(abs_range < 0.1, default_range, max_v)
                        else:
                            # Fallback: use adjusted absolute range (original behavior)
                            # Relative position distribution: approximately centered at 0
                            # Range: [-abs_range, abs_range] where abs_range = max(|min|, |max|)
                            abs_range = torch.maximum(torch.abs(min_v), torch.abs(max_v))
                            # Use a slightly wider range (1.5x) to account for distribution spread
                            rel_range = abs_range * 1.5
                            min_v = -rel_range
                            max_v = rel_range
                            # If abs_range is too small, use a default range (e.g., ±1.0 meter)
                            default_range = torch.ones_like(abs_range) * 1.0
                            min_v = torch.where(abs_range < 0.1, -default_range, min_v)
                            max_v = torch.where(abs_range < 0.1, default_range, max_v)
                    
                    denom = max_v - min_v
                    mask = denom != 0
                    safe_denom = torch.where(mask, denom, torch.ones_like(denom))
                    mapped = 2 * (component - min_v) / safe_denom - 1
                    normalized_component = torch.where(mask, mapped, torch.zeros_like(mapped))
                
                result[..., start_idx:end_idx] = normalized_component
            
            return result

        # 1) Video (B, T=1, V, H, W, C) uint8
        img_keys = sorted([k for k in obs if k.startswith("observation.images.")])
        if not img_keys and "observation.image" in obs:
            img_keys = ["observation.image"]
        if img_keys:
            cams = [_to_uint8_np_bhwc(obs[k]) for k in img_keys]
            video = np.stack(cams, axis=1)  # (B, V, H, W, C)
            video = np.expand_dims(video, axis=1)  # (B, 1, V, H, W, C)
            # GR00T validates that video.shape[3] == 3 (channels), so reorder to (B, T, V, C, H, W)
            video = np.transpose(video, (0, 1, 2, 5, 3, 4))  # (B, 1, V, C, H, W)
            obs["video"] = video
            # Drop raw images to avoid confusion downstream
            for k in img_keys:
                obs.pop(k, None)

        # 2) Language (string)
        lang = comp.get(self.language_key)
        if isinstance(lang, list):
            lang = lang[0] if len(lang) > 0 else None
        if not lang:
            lang = "Perform the task."
        if self.formalize_language:
            lang = (lang or "").lower()
            lang = "".join(ch for ch in lang if ch.isalnum() or ch.isspace())
        comp["language"] = lang

        # 3) State/state_mask -> (B, 1, max_state_dim)
        if "observation.state" in obs:
            state = obs["observation.state"]  # (B, D)
            if state.dim() != 2:
                raise ValueError(f"state must be (B, D), got {tuple(state.shape)}")
            bsz, d = state.shape
            
            # **重要**: 保存原始的 state（归一化之前）用于 Delta eef 的 reference pose
            # 这确保训练和推理时的 reference 来源一致！
            if self.action_space_type == "Delta eef":
                obs["_original_observation_state"] = state.clone()
            
            # Normalize BEFORE padding
            if self.normalize_min_max:
                state = _min_max_norm(state, "observation.state")
            state = state.unsqueeze(1)  # (B, 1, D)
            if d > self.max_state_dim:
                state = state[:, :, : self.max_state_dim]
                d = self.max_state_dim
            elif d < self.max_state_dim:
                pad = torch.zeros(bsz, 1, self.max_state_dim - d, dtype=state.dtype, device=state.device)
                state = torch.cat([state, pad], dim=2)
            state_mask = torch.zeros(bsz, 1, self.max_state_dim, dtype=torch.bool, device=state.device)
            state_mask[:, :, :d] = True
            obs["state"] = state
            obs["state_mask"] = state_mask

        # 4) Action/action_mask -> (B, action_horizon, max_action_dim)
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            # Handle temporal expansion first (before relative action conversion)
            if action.dim() == 2:
                action = action.unsqueeze(1).repeat(1, self.action_horizon, 1)
            elif action.dim() == 3:
                b, t, d = action.shape
                if t < self.action_horizon:
                    last = action[:, -1:, :]
                    pad = last.repeat(1, self.action_horizon - t, 1)
                    action = torch.cat([action, pad], dim=1)
                elif t > self.action_horizon:
                    action = action[:, : self.action_horizon, :]
            else:
                raise ValueError(f"action must be (B, D) or (B, T, D), got {tuple(action.shape)}")

            b, t, d = action.shape
            
            # Convert absolute eef pose to relative eef action if needed
            # This happens BEFORE normalization
            # 
            # **重要**: 使用 observation.state 作为 reference pose！
            # 这确保训练和推理时的 reference 来源一致：
            # - 训练时: state 来自 dataset（机器人录制时的实际状态）
            # - 推理时: state 来自 FK（机器人当前的实际状态）
            if self.action_space_type == "Delta eef" and self.action_component_indices is not None:
                # 获取原始的 observation.state（归一化之前的版本）
                # 注意：这里需要在 state 归一化之前保存原始值
                current_state_for_ref = obs.get("_original_observation_state", None)
                
                action = self._convert_absolute_to_relative_eef_action(action, current_state_for_ref)
                # Update relative action statistics for position components
                # This accumulates stats during training
                # Note: We always update stats when processing data (training or inference)
                # During inference, stats won't change since they're already loaded from checkpoint
                self.update_relative_action_stats(action)
            
            # Normalize AFTER relative action conversion
            if self.normalize_min_max:
                if action.dim() == 2:
                    action = _min_max_norm(action, "action")
                elif action.dim() == 3:
                    b, t, d = action.shape
                    flat = action.reshape(b * t, d)
                    flat = _min_max_norm(flat, "action")
                    action = flat.view(b, t, d)

            if d > self.max_action_dim:
                action = action[:, :, : self.max_action_dim]
                d = self.max_action_dim
            elif d < self.max_action_dim:
                pad = torch.zeros(b, t, self.max_action_dim - d, dtype=action.dtype, device=action.device)
                action = torch.cat([action, pad], dim=2)
            action_mask = torch.zeros(b, t, self.max_action_dim, dtype=torch.bool, device=action.device)
            action_mask[:, :, :d] = True
            transition[TransitionKey.ACTION] = action
            comp["action_mask"] = action_mask

        # 5) Embodiment id as LongTensor (B,)
        emb_id = self.embodiment_mapping.get(self.embodiment_tag, 0)
        # Infer batch size/device from any tensor in obs or action
        bsz = None
        device = torch.device("cpu")
        for v in list(obs.values()) + [transition.get(TransitionKey.ACTION)]:
            if isinstance(v, torch.Tensor):
                bsz = v.shape[0]
                device = v.device
                break
        if bsz is None and "video" in obs and isinstance(obs["video"], np.ndarray):
            bsz = obs["video"].shape[0]
        if bsz is None:
            bsz = 1
        comp["embodiment_id"] = torch.full((bsz,), emb_id, dtype=torch.long, device=device)

        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    # Pipeline API requirement: declare how features change (we keep it simple)
    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        """
        Returns a serializable dictionary of the processor's configuration.

        Excludes 'stats' since they are saved separately via state_dict().
        Includes relative_action_stats if available (for Delta eef mode).
        """
        config = {
            "state_horizon": self.state_horizon,
            "action_horizon": self.action_horizon,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "language_key": self.language_key,
            "formalize_language": self.formalize_language,
            "embodiment_tag": self.embodiment_tag,
            "embodiment_mapping": self.embodiment_mapping,
            "normalize_min_max": self.normalize_min_max,
            "action_space_type": self.action_space_type,
            "relative_action_reference_mode": self.relative_action_reference_mode,  # Reference pose mode for Delta eef
            "freeze_stats_after_first_epoch": self.freeze_stats_after_first_epoch,
            "dataset_num_frames": self.dataset_num_frames,
            "num_processes": self.num_processes,
        }
        
        # Include relative_action_stats if available (for Delta eef mode)
        if self.relative_action_stats is not None:
            # Convert to serializable format (list instead of tensor)
            relative_stats_serializable = {}
            for comp_name, stats in self.relative_action_stats.items():
                relative_stats_serializable[comp_name] = {
                    "min": stats["min"].cpu().tolist() if isinstance(stats["min"], torch.Tensor) else stats["min"],
                    "max": stats["max"].cpu().tolist() if isinstance(stats["max"], torch.Tensor) else stats["max"],
                    "count": int(stats["count"].item() if isinstance(stats["count"], torch.Tensor) else stats["count"]),
                }
            config["relative_action_stats"] = relative_stats_serializable
        
        return config

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Returns normalization statistics as a flat state dictionary.

        This enables saving stats to safetensors files, similar to normalizer_processor.
        Also includes relative_action_stats for Delta eef mode.
        """
        flat: dict[str, torch.Tensor] = {}
        
        # Save standard stats
        if self.stats:
            for key, sub in self.stats.items():
                for stat_name, value in sub.items():
                    tensor = torch.as_tensor(value).cpu()
                    flat[f"{key}.{stat_name}"] = tensor
        
        # Save relative_action_stats for Delta eef mode
        if self.relative_action_stats is not None:
            for comp_name, stats in self.relative_action_stats.items():
                for stat_name, value in stats.items():
                    tensor = torch.as_tensor(value).cpu()
                    flat[f"relative_action.{comp_name}.{stat_name}"] = tensor
        
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """
        Loads normalization statistics from a flat state dictionary.

        This enables loading stats from safetensors files during from_pretrained.
        Also loads relative_action_stats for Delta eef mode.
        """
        if not state:
            return

        reconstructed: dict[str, dict[str, Any]] = {}
        relative_stats_reconstructed: dict[str, dict[str, torch.Tensor]] = {}
        
        for flat_key, tensor in state.items():
            if flat_key.startswith("relative_action."):
                # Handle relative_action_stats
                # Format: "relative_action.{comp_name}.{stat_name}"
                parts = flat_key.split(".")
                if len(parts) == 3:
                    _, comp_name, stat_name = parts
                    if comp_name not in relative_stats_reconstructed:
                        relative_stats_reconstructed[comp_name] = {}
                    relative_stats_reconstructed[comp_name][stat_name] = tensor
            elif "." in flat_key:
                # Handle standard stats
                key, stat_name = flat_key.rsplit(".", 1)
                if key not in reconstructed:
                    reconstructed[key] = {}
                reconstructed[key][stat_name] = tensor

        if reconstructed:
            self.stats = reconstructed
        
        if relative_stats_reconstructed:
            self.relative_action_stats = relative_stats_reconstructed
            self._relative_stats_initialized = True


@dataclass
@ProcessorStepRegistry.register(name="groot_eagle_encode_v3")
class GrootEagleEncodeStep(ProcessorStep):
    tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO
    _proc: ProcessorMixin | None = field(default=None, init=False, repr=False)

    @property
    def proc(self) -> ProcessorMixin:
        if self._proc is None:
            self._proc = _build_eagle_processor(self.tokenizer_assets_repo)
        return self._proc

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        if "video" not in obs:
            return transition

        video = obs["video"]  # (B, T, V, H, W, C) uint8
        lang = comp.get("language", "Perform the task.")
        if isinstance(lang, list):
            lang = lang[0] if len(lang) > 0 else "Perform the task."

        bsz = video.shape[0]
        eagle_contents: list[dict[str, Any]] = []
        for b in range(bsz):
            vt = video[b]  # (T, V, C, H, W) after reorder
            if vt.ndim != 5:
                # Fallback: assume (T, V, H, W, C)
                t, v, h, w, c = vt.shape
                flat = rearrange(vt, "t v h w c -> (t v) h w c")
            else:
                t, v, c, h, w = vt.shape
                flat = rearrange(vt, "t v c h w -> (t v) h w c")
            images = [Image.fromarray(flat[i]) for i in range(t * v)]
            # Format language as string list representation to match Original GROOT
            lang_formatted = str([lang])
            text_content = [{"type": "text", "text": lang_formatted}]
            image_content = [{"type": "image", "image": img} for img in images]
            conv = [{"role": "user", "content": image_content + text_content}]
            text_list = [self.proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)]
            img_inputs, vid_inputs = self.proc.process_vision_info(conv)
            eagle_contents.append(
                {
                    "text_list": text_list,
                    "image_inputs": img_inputs,
                    "video_inputs": vid_inputs,
                }
            )

        comp["eagle_content"] = eagle_contents
        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    # Pipeline API requirement: declare how features change (no schema change here)
    def transform_features(self, features):
        return features


# Original GR00T-style collate: converts eagle_content -> eagle_* tensors
def collate(features: list[dict[str, Any]], eagle_processor: ProcessorMixin) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            text_list: list[str] = []
            image_inputs: list[Any] = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs
            eagle_inputs = eagle_processor(
                text=text_list,
                images=image_inputs,
                images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},
                return_tensors="pt",
                padding=True,
            )
            for k, v in eagle_inputs.items():
                k = "eagle_" + k
                batch[k] = v
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


@dataclass
@ProcessorStepRegistry.register(name="groot_eagle_collate_v3")
class GrootEagleCollateStep(ProcessorStep):
    tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO
    _proc: ProcessorMixin | None = field(default=None, init=False, repr=False)

    @property
    def proc(self) -> ProcessorMixin:
        if self._proc is None:
            self._proc = _build_eagle_processor(self.tokenizer_assets_repo)
        return self._proc

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        contents = comp.get("eagle_content")
        if not contents:
            return transition

        # Build features list as original API expects: one dict per batch item
        features = [{"eagle_content": content} for content in contents]
        batched = collate(features, self.proc)

        # Inject eagle_* tensors and remove the temporary content and raw video to free memory
        for k, v in batched.items():
            comp[k] = v
        comp.pop("eagle_content", None)
        obs.pop(
            "video", None
        )  # The video has been fully encoded into eagle_* tensors, so we don't need the raw video anymore
        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register(name="groot_action_unpack_unnormalize_v1")
class GrootActionUnpackUnnormalizeStep(ProcessorStep):
    env_action_dim: int = 0
    # Apply inverse of min-max normalization if it was used in preprocessor
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None
    # For partial normalization of action (e.g., 6D rotation representation)
    action_space_type: str | None = None  # "Delta eef", "Absolute eef", "Absolute joint", etc.
    action_component_indices: dict[str, tuple[int, int]] | None = None  # e.g., {"left_eef_pos": (0, 3), ...}
    # Dynamic normalization statistics for relative action position components (loaded from checkpoint)
    relative_action_stats: dict[str, dict[str, torch.Tensor]] | None = field(
        default=None, init=False, repr=False
    )  # e.g., {"left_eef_pos": {"min": ..., "max": ..., "count": ...}, ...}

    def _convert_relative_to_absolute_eef_action(
        self, relative_action: torch.Tensor, reference_pose: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Convert relative EEF action to absolute EEF pose.
        
        For inference: uses provided reference pose (current robot eef pose) to convert
        relative action back to absolute pose.
        
        Args:
            relative_action: Relative EEF action tensor. Shape: (B, D) or (B, T, D)
            reference_pose: Reference EEF pose (current robot state). Shape: (B, D) or (B, 1, D)
                           If None, returns relative_action as-is (assumes already absolute)
        
        Returns:
            Absolute EEF pose tensor. Shape: same as relative_action
        """
        if reference_pose is None or self.action_component_indices is None:
            # No reference pose provided or no component indices -> assume already absolute
            return relative_action
        
        if self.action_space_type != "Delta eef":
            # Only convert if action_space_type is "Delta eef"
            return relative_action
        
        # Handle both (B, D) and (B, T, D) shapes
        original_shape = relative_action.shape
        is_temporal = len(original_shape) == 3
        if not is_temporal:
            relative_action = relative_action.unsqueeze(1)  # (B, 1, D)
        
        b, t, d = relative_action.shape
        
        # Ensure reference_pose has correct shape
        if reference_pose.dim() == 2:
            reference_pose = reference_pose.unsqueeze(1)  # (B, 1, D)
        elif reference_pose.dim() == 1:
            reference_pose = reference_pose.unsqueeze(0).unsqueeze(1)  # (1, 1, D)
        
        # Ensure batch size matches
        if reference_pose.shape[0] != b:
            if reference_pose.shape[0] == 1:
                reference_pose = reference_pose.expand(b, -1, -1)
            else:
                raise ValueError(f"Batch size mismatch: reference_pose {reference_pose.shape[0]} vs relative_action {b}")
        
        absolute_action = relative_action.clone()
        
        # Process each component
        for component_name, (start_idx, end_idx) in self.action_component_indices.items():
            if end_idx > d:
                continue
            
            # Extract component from reference and relative action
            ref_component = reference_pose[:, :, start_idx:end_idx]  # (B, 1, component_dim)
            rel_components = relative_action[:, :, start_idx:end_idx]  # (B, T, component_dim)
            
            if "pos" in component_name:
                # Position: absolute_pos = relative_pos + ref_pos
                abs_components = rel_components + ref_component  # Broadcast: (B, T, dim) + (B, 1, dim)
                absolute_action[:, :, start_idx:end_idx] = abs_components
                
            elif "rot6d" in component_name:
                # Rotation: absolute_rot = ref_rot @ relative_rot
                # R_abs = R_ref @ R_rel
                
                # Convert 6D to rotation matrices
                ref_rot6d = ref_component.squeeze(1)  # (B, 6)
                rel_rot6d = rel_components.reshape(b * t, -1)  # (B*T, 6)
                
                # Convert to matrices
                ref_matrices = rot6d_to_matrix(ref_rot6d)  # (B, 3, 3)
                rel_matrices = rot6d_to_matrix(rel_rot6d)  # (B*T, 3, 3)
                
                # Compute absolute rotation: R_abs = R_ref @ R_rel
                ref_matrices_expanded = ref_matrices.unsqueeze(1).expand(-1, t, -1, -1)  # (B, T, 3, 3)
                ref_matrices_flat = ref_matrices_expanded.reshape(b * t, 3, 3)  # (B*T, 3, 3)
                
                # Absolute rotation: R_ref @ R_rel
                abs_matrices = torch.bmm(ref_matrices_flat, rel_matrices)  # (B*T, 3, 3)
                
                # Convert back to 6D
                abs_rot6d = matrix_to_rot6d(abs_matrices)  # (B*T, 6)
                abs_rot6d = abs_rot6d.reshape(b, t, -1)  # (B, T, 6)
                
                absolute_action[:, :, start_idx:end_idx] = abs_rot6d
                
            elif "gripper" in component_name:
                # Gripper: keep as-is (or add reference if needed)
                # For now, keep relative values (can be changed if needed)
                pass
        
        if not is_temporal:
            absolute_action = absolute_action.squeeze(1)  # (B, D)
        
        return absolute_action

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Expect model outputs to be in TransitionKey.ACTION as (B, T, D_model)
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return transition

        # Select last timestep and slice to env dimension
        if action.dim() == 3:
            action = action[:, -1, :]
        # Now action is (B, D_model)
        if self.env_action_dim and action.shape[-1] >= self.env_action_dim:
            action = action[..., : self.env_action_dim]

        # Inverse min-max normalization mirroring _min_max_norm:
        # forward: y = 2 * (x - min) / denom - 1, with y=0 when denom==0
        # inverse: x = (y+1)/2 * denom + min, and when denom==0 -> x = min
        if self.normalize_min_max and self.stats is not None:
            # Check if partial normalization is enabled for action
            if (self.action_space_type in ["Delta eef", "Absolute eef"] and 
                self.action_component_indices is not None):
                action = self._min_max_unnorm_partial(action)
            else:
                # Standard unnormalization: apply to entire tensor
                stats_k = self.stats.get("action", {})
                d = action.shape[-1]
                min_v = torch.as_tensor(
                    stats_k.get("min", torch.zeros(d)), dtype=action.dtype, device=action.device
                )
                max_v = torch.as_tensor(
                    stats_k.get("max", torch.ones(d)), dtype=action.dtype, device=action.device
                )
                if min_v.numel() != d:
                    min_v = torch.nn.functional.pad(min_v.flatten()[:d], (0, max(0, d - min_v.numel())))
                    min_v = min_v.to(action.device, dtype=action.dtype)
                if max_v.numel() != d:
                    max_v = torch.nn.functional.pad(max_v.flatten()[:d], (0, max(0, d - max_v.numel())))
                    max_v = max_v.to(action.device, dtype=action.dtype)
                denom = max_v - min_v
                mask = denom != 0
                safe_denom = torch.where(mask, denom, torch.ones_like(denom))
                inv = (action + 1.0) * 0.5 * safe_denom + min_v
                action = torch.where(mask, inv, min_v)
        
        # Convert relative action to absolute pose if needed (for inference)
        # NOTE: This requires reference pose (current robot eef pose) from observation
        # For now, we skip this conversion if reference pose is not available
        # In practice, the reference pose should be obtained from the environment/robot state
        # and passed to the postprocessor, or computed from joint positions via forward kinematics
        if self.action_space_type == "Delta eef" and self.action_component_indices is not None:
            # Try to get reference pose from observation if available
            # This is a placeholder - in practice, reference pose should be provided by the caller
            # or computed from current robot state
            reference_pose = None
            # TODO: Extract reference pose from observation.state or compute from joint positions
            # For now, if reference_pose is None, we assume action is already absolute (backward compatibility)
            if reference_pose is not None:
                action = self._convert_relative_to_absolute_eef_action(action, reference_pose)
            else:
                # If no reference pose available, log a warning but continue
                # In practice, this should be handled by the inference code
                pass
        
        # Update transition with processed action and return
        transition[TransitionKey.ACTION] = action
        return transition
    
    def _min_max_unnorm_partial(self, x: torch.Tensor) -> torch.Tensor:
        """Apply partial inverse min-max normalization for action with component indices.
        
        For relative actions (Delta eef), uses the same adjusted normalization ranges
        as in the forward normalization to ensure consistency.
        """
        if self.action_component_indices is None:
            # Fallback to standard unnormalization
            stats_k = self.stats.get("action", {})
            d = x.shape[-1]
            min_v = torch.as_tensor(
                stats_k.get("min", torch.zeros(d)), dtype=x.dtype, device=x.device
            )
            max_v = torch.as_tensor(
                stats_k.get("max", torch.ones(d)), dtype=x.dtype, device=x.device
            )
            if min_v.numel() != d:
                min_v = torch.nn.functional.pad(min_v.flatten()[:d], (0, max(0, d - min_v.numel())))
                min_v = min_v.to(x.device, dtype=x.dtype)
            if max_v.numel() != d:
                max_v = torch.nn.functional.pad(max_v.flatten()[:d], (0, max(0, d - max_v.numel())))
                max_v = max_v.to(x.device, dtype=x.dtype)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            inv = (x + 1.0) * 0.5 * safe_denom + min_v
            return torch.where(mask, inv, min_v)
        
        stats_k = self.stats.get("action", {})
        result = x.clone()
        
        # Define which components should use IDENTITY (no normalization)
        rot6d_components = ["left_eef_rot6d", "right_eef_rot6d"]
        
        # Get full stats (from absolute eef pose)
        d = x.shape[-1]
        min_v_full = torch.as_tensor(
            stats_k.get("min", torch.zeros(d)), dtype=x.dtype, device=x.device
        )
        max_v_full = torch.as_tensor(
            stats_k.get("max", torch.ones(d)), dtype=x.dtype, device=x.device
        )
        if min_v_full.numel() != d:
            min_v_full = torch.nn.functional.pad(min_v_full.flatten()[:d], (0, max(0, d - min_v_full.numel())))
            min_v_full = min_v_full.to(x.device, dtype=x.dtype)
        if max_v_full.numel() != d:
            max_v_full = torch.nn.functional.pad(max_v_full.flatten()[:d], (0, max(0, d - max_v_full.numel())))
            max_v_full = max_v_full.to(x.device, dtype=x.dtype)
        
        # For relative actions, adjust normalization ranges for position components (same as forward)
        is_relative_action = (self.action_space_type == "Delta eef")
        
        # Process each component
        for component_name, (start_idx, end_idx) in self.action_component_indices.items():
            if end_idx > d:
                continue  # Skip if indices exceed tensor dimension
            
            component = x[..., start_idx:end_idx]
            
            # Check if this component should use IDENTITY normalization
            if component_name in rot6d_components:
                # 6D rotation: use IDENTITY (no unnormalization)
                unnormalized_component = component
            else:
                # Position or gripper: apply inverse min-max normalization
                min_v = min_v_full[start_idx:end_idx]
                max_v = max_v_full[start_idx:end_idx]
                
                # For relative actions, adjust normalization range for position components (same as forward)
                if is_relative_action and "pos" in component_name:
                    # Use dynamically accumulated relative_action_stats if available
                    if (self.relative_action_stats is not None and 
                        component_name in self.relative_action_stats):
                        # Use accumulated relative action stats
                        rel_stats = self.relative_action_stats[component_name]
                        rel_min = rel_stats["min"]
                        rel_max = rel_stats["max"]
                        
                        # Ensure stats are on the correct device and have correct shape
                        if isinstance(rel_min, torch.Tensor):
                            rel_min = rel_min.to(device=component.device, dtype=component.dtype)
                            rel_max = rel_max.to(device=component.device, dtype=component.dtype)
                        else:
                            rel_min = torch.as_tensor(rel_min, device=component.device, dtype=component.dtype)
                            rel_max = torch.as_tensor(rel_max, device=component.device, dtype=component.dtype)
                        
                        # Use accumulated stats directly
                        min_v = rel_min
                        max_v = rel_max
                        
                        # Debug logging (only log once per component to avoid spam)
                        if not hasattr(self, f'_unnorm_logged_{component_name}'):
                            print(
                                f"🔓 [Delta EEF Unnorm] Using dynamic stats for {component_name}: "
                                f"min={rel_min.tolist()}, max={rel_max.tolist()}"
                            )
                            setattr(self, f'_unnorm_logged_{component_name}', True)
                        
                        # If stats are invalid (all inf), fallback to adjusted absolute range
                        if torch.any(torch.isinf(rel_min)) or torch.any(torch.isinf(rel_max)):
                            abs_range = torch.maximum(torch.abs(min_v_full[start_idx:end_idx]), 
                                                     torch.abs(max_v_full[start_idx:end_idx]))
                            rel_range = abs_range * 1.5
                            min_v = -rel_range
                            max_v = rel_range
                            default_range = torch.ones_like(abs_range) * 1.0
                            min_v = torch.where(abs_range < 0.1, -default_range, min_v)
                            max_v = torch.where(abs_range < 0.1, default_range, max_v)
                    else:
                        # Fallback: use adjusted absolute range (original behavior)
                        # Log warning if relative_action_stats is not available
                        if not hasattr(self, f'_fallback_logged_{component_name}'):
                            print(
                                f"⚠️  [Delta EEF Unnorm] Fallback to adjusted absolute range for {component_name} "
                                f"(relative_action_stats not available or component not found)"
                            )
                            setattr(self, f'_fallback_logged_{component_name}', True)
                        # Relative position distribution: approximately centered at 0
                        # Range: [-abs_range, abs_range] where abs_range = max(|min|, |max|)
                        abs_range = torch.maximum(torch.abs(min_v), torch.abs(max_v))
                        # Use a slightly wider range (1.5x) to account for distribution spread
                        rel_range = abs_range * 1.5
                        min_v = -rel_range
                        max_v = rel_range
                        # If abs_range is too small, use a default range (e.g., ±1.0 meter)
                        default_range = torch.ones_like(abs_range) * 1.0
                        min_v = torch.where(abs_range < 0.1, -default_range, min_v)
                        max_v = torch.where(abs_range < 0.1, default_range, max_v)
                
                denom = max_v - min_v
                mask = denom != 0
                safe_denom = torch.where(mask, denom, torch.ones_like(denom))
                inv = (component + 1.0) * 0.5 * safe_denom + min_v
                unnormalized_component = torch.where(mask, inv, min_v)
            
            result[..., start_idx:end_idx] = unnormalized_component
        
        return result

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        """
        Returns a serializable dictionary of the processor's configuration.

        Excludes 'stats' since they are saved separately via state_dict().
        """
        return {
            "env_action_dim": self.env_action_dim,
            "normalize_min_max": self.normalize_min_max,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Returns normalization statistics as a flat state dictionary.

        This enables saving stats to safetensors files, similar to normalizer_processor.
        """
        if not self.stats:
            return {}

        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                tensor = torch.as_tensor(value).cpu()
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """
        Loads normalization statistics from a flat state dictionary.

        This enables loading stats from safetensors files during from_pretrained.
        """
        if not state:
            return

        reconstructed: dict[str, dict[str, Any]] = {}
        for flat_key, tensor in state.items():
            if "." in flat_key:
                key, stat_name = flat_key.rsplit(".", 1)
                if key not in reconstructed:
                    reconstructed[key] = {}
                reconstructed[key][stat_name] = tensor

        if reconstructed:
            self.stats = reconstructed
