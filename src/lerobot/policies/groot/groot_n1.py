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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
    from transformers.feature_extraction_utils import BatchFeature
else:
    AutoConfig = None
    AutoModel = None
    PretrainedConfig = object
    PreTrainedModel = object
    BatchFeature = None

try:
    import tree
except ImportError:
    try:
        import dm_tree as tree
    except ImportError:
        # Fallback: use a simple recursive function if tree is not available
        def _map_structure(func, structure):
            """Simple recursive map_structure implementation."""
            if isinstance(structure, dict):
                return {k: _map_structure(func, v) for k, v in structure.items()}
            elif isinstance(structure, (list, tuple)):
                return type(structure)(_map_structure(func, item) for item in structure)
            else:
                return func(structure)
        
        class _TreeModule:
            @staticmethod
            def map_structure(func, structure):
                return _map_structure(func, structure)
        
        tree = _TreeModule()

from lerobot.policies.groot.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from lerobot.policies.groot.utils import ensure_eagle_cache_ready
from lerobot.utils.constants import HF_LEROBOT_HOME

DEFAULT_VENDOR_EAGLE_PATH = str((Path(__file__).resolve().parent / "eagle2_hg_model").resolve())
DEFAULT_TOKENIZER_ASSETS_REPO = "lerobot/eagle2hg-processor-groot-n1p5"


class EagleBackbone(nn.Module):
    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str = DEFAULT_VENDOR_EAGLE_PATH,
        tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO,
        project_to_dim: int = 1536,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        # Prefer loading Eagle model config from the cache directory where vendor files were copied.
        vendor_dir = DEFAULT_VENDOR_EAGLE_PATH
        cache_dir = HF_LEROBOT_HOME / tokenizer_assets_repo
        try:
            ensure_eagle_cache_ready(vendor_dir, cache_dir, tokenizer_assets_repo)
        except Exception as exc:  # nosec: B110
            print(f"[GROOT] Warning: failed to prepare Eagle cache for backbone: {exc}")

        config = AutoConfig.from_pretrained(str(cache_dir), trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v for k, v in vl_input.items() if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]

        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        return BatchFeature(
            data={"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}
        )  # [B, T2, hidden_size]


BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00TN15Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00TN15(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00TN15Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00TN15Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        
        # If using multi-action heads, override action_dim to match arm_dim + claw_dim
        action_head_cfg_dict = config.action_head_cfg.copy()
        # Use FlowmatchingActionHeadConfig default (True) if not specified in config
        use_multi_action_heads = action_head_cfg_dict.get("use_multi_action_heads", True)
        
        # Save pretrained action_dim for compatibility (pretrained model uses 32D)
        pretrained_action_dim = action_head_cfg_dict.get("action_dim", 32)
        
        if use_multi_action_heads:
            split_arm_heads = action_head_cfg_dict.get("split_arm_heads", False)
            if split_arm_heads:
                # Split arm into left and right
                action_left_arm_dim = action_head_cfg_dict.get("action_left_arm_dim", 7)
                action_right_arm_dim = action_head_cfg_dict.get("action_right_arm_dim", 7)
                action_claw_dim = action_head_cfg_dict.get("action_claw_dim", 2)
                actual_action_dim = action_left_arm_dim + action_right_arm_dim + action_claw_dim
                # Set action_arm_dim for compatibility (left + right)
                action_head_cfg_dict["action_arm_dim"] = action_left_arm_dim + action_right_arm_dim
                # Ensure split_arm_heads is set in the dict
                action_head_cfg_dict["split_arm_heads"] = True
                print(f"✅ Split arm heads enabled: left_arm({action_left_arm_dim}D) + right_arm({action_right_arm_dim}D) + claw({action_claw_dim}D) = {actual_action_dim}D")
            else:
                # Single arm head
                action_arm_dim = action_head_cfg_dict.get("action_arm_dim", 14)
                action_claw_dim = action_head_cfg_dict.get("action_claw_dim", 2)
                actual_action_dim = action_arm_dim + action_claw_dim
            
            action_head_cfg_dict["action_dim"] = actual_action_dim
            # Set pretrained_action_dim for compatibility with pretrained encoder
            action_head_cfg_dict["pretrained_action_dim"] = pretrained_action_dim
            # Ensure use_multi_action_heads is set in the dict
            action_head_cfg_dict["use_multi_action_heads"] = True
            if pretrained_action_dim != actual_action_dim:
                print(f"🔧 Using pretrained action encoder ({pretrained_action_dim}D) with multi-head output ({actual_action_dim}D)")
        
        action_head_cfg = FlowmatchingActionHeadConfig(**action_head_cfg_dict)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        # Keep config.action_dim for compatibility (may be 32 for pretrained models)
        # The actual output dimension is action_head.actual_action_dim (16 for multi-head)
        # validate_data will use the correct dimension based on context
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            # In inference, action may be omitted or None; validate only when it's a tensor.
            if action is None:
                pass  # allow None during inference
            elif isinstance(action, torch.Tensor):
                shape_ok = (
                    len(action.shape) == 3
                    and action.shape[1] == self.action_horizon
                    and action.shape[2] == self.action_dim
                )
                if not shape_ok:
                    error_msg += f"\n{action.shape=}"
                    detected_error = True
            else:
                # Unexpected non-tensor type provided for action
                error_msg += f"\nInvalid type for action: {type(action)}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature) or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        # For inference, use actual_action_dim if using multi-head (may differ from config.action_dim)
        # For training, we don't check action_pred dimensions (only check for loss key)
        expected_action_dim = self.action_dim
        if not is_training and ACTION_KEY in action_head_outputs:
            # During inference, use actual output dimension from action_head
            # This handles the case where multi-head outputs actual_action_dim (16) 
            # but config.action_dim is pretrained dimension (32)
            if hasattr(self.action_head, 'actual_action_dim'):
                expected_action_dim = self.action_head.actual_action_dim

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == expected_action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            if ACTION_KEY in action_head_outputs:
                error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=} (config)"
            error_msg += f"\n{expected_action_dim=} (expected for validation)"
            if hasattr(self.action_head, 'actual_action_dim'):
                error_msg += f"\n{self.action_head.actual_action_dim=} (actual from action_head)"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Cast floating tensors to a memory-efficient compute dtype when requested.
            # Rationale: Upcasting backbone activations to fp32 significantly increases VRAM.
            # When compute_dtype is bfloat16, prefer bf16 for activations to match AMP behavior.
            if not isinstance(x, torch.Tensor):
                return x
            if torch.is_floating_point(x):
                if getattr(self, "compute_dtype", None) == "bfloat16":
                    return x.to(self.device, dtype=torch.bfloat16)
                # Fallback: preserve previous behavior if not using bf16 compute
                return x.to(self.device, dtype=self.action_head.dtype)
            # Non-floating tensors: move device only
            return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        # Load config directly from config.json file
        import json
        import os
        from pathlib import Path
        
        config_path = Path(local_model_path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {local_model_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Create GR00TN15Config from the loaded dict
        config = GR00TN15Config(**config_dict)
        
        # Create model instance with local_model_path
        pretrained_model = cls(config, local_model_path=local_model_path)
        
        # Load weights manually
        try:
            from transformers import AutoModel
            # Try to load state dict from the model files
            import os
            from safetensors.torch import load_file
            from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME
            
            # Try safetensors first
            safetensors_path = os.path.join(local_model_path, SAFE_WEIGHTS_NAME)
            if not os.path.exists(safetensors_path):
                # Try split safetensors
                import glob
                safetensors_files = glob.glob(os.path.join(local_model_path, "model-*.safetensors"))
                if safetensors_files:
                    state_dict = {}
                    for f in sorted(safetensors_files):
                        state_dict.update(load_file(f))
                    pretrained_model.load_state_dict(state_dict, strict=False)
                else:
                    # Fallback to pytorch model
                    pytorch_path = os.path.join(local_model_path, WEIGHTS_NAME)
                    if os.path.exists(pytorch_path):
                        import torch
                        state_dict = torch.load(pytorch_path, map_location="cpu")
                        pretrained_model.load_state_dict(state_dict, strict=False)
            else:
                state_dict = load_file(safetensors_path)
                pretrained_model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Continuing with randomly initialized weights.")

        pretrained_model.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        
        # Verify that LLM parameters are actually trainable if tune_llm=True
        if tune_llm:
            llm_params = list(pretrained_model.backbone.eagle_model.language_model.parameters())
            trainable_llm_params = [p for p in llm_params if p.requires_grad]
            if trainable_llm_params:
                print(f"✅ Verified: {len(trainable_llm_params)}/{len(llm_params)} LLM parameters are trainable")
            else:
                print(f"⚠️  Warning: No LLM parameters are trainable despite tune_llm=True!")
        else:
            llm_params = list(pretrained_model.backbone.eagle_model.language_model.parameters())
            frozen_llm_params = [p for p in llm_params if not p.requires_grad]
            if len(frozen_llm_params) == len(llm_params):
                print(f"✅ Verified: All {len(llm_params)} LLM parameters are frozen (tune_llm=False)")
            else:
                print(f"⚠️  Warning: Some LLM parameters are still trainable despite tune_llm=False!")
        
        return pretrained_model
