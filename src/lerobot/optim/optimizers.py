#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import abc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import draccus
import torch
from safetensors.torch import load_file, save_file

from lerobot.datasets.utils import flatten_dict, unflatten_dict, write_json
from lerobot.utils.constants import (
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
)
from lerobot.utils.io_utils import deserialize_json_into_object


class CautiousOptimizer(torch.optim.Optimizer):
    """
    Cautious Optimizer wrapper that applies cautious masking to any momentum-based optimizer.
    
    Based on "Cautious Optimizers: Improving Training with One Line of Code" (Liang et al., 2024).
    Paper: https://arxiv.org/pdf/2411.16085
    
    The core idea is to only update parameters when the update direction aligns with the gradient,
    which prevents temporary loss increases and speeds up convergence.
    
    Algorithm (from paper):
        m = (u * g > 0).to(g.dtype)  # Alignment mask
        p.add_(u * m/(m.mean() + eps), alpha=-lr)  # Masked update with scaling
    
    This implementation wraps a base optimizer and intercepts the step() call to apply cautious masking.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, eps: float = 1e-8):
        """
        Args:
            optimizer: The base optimizer to wrap (e.g., AdamW, Adam)
            eps: Small constant for numerical stability in scaling factor
        """
        # Store the base optimizer
        self.optimizer = optimizer
        self.eps = eps
        
        # Forward all attributes from base optimizer for compatibility
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self.defaults = optimizer.defaults
        
        # Log that Cautious Optimizer is enabled
        import logging
        logging.info("🚀 Cautious Optimizer enabled (arXiv:2411.16085)")
        logging.info("   Only updating parameters when update direction aligns with gradient direction")
        logging.info("   This can accelerate convergence and improve training stability for Transformers")
    
    def __getstate__(self):
        return self.optimizer.__getstate__()
    
    def __setstate__(self, state):
        self.optimizer.__setstate__(state)
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state

    def load_state_dict(self, state_dict):
        """Forward load_state_dict to base optimizer."""
        self.optimizer.load_state_dict(state_dict)
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state

    def zero_grad(self, set_to_none: bool = False):
        """Forward zero_grad to base optimizer."""
        return self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self):
        """Forward state_dict to base optimizer."""
        return self.optimizer.state_dict()
    
    def step(self, closure=None):
        """
        Perform a cautious optimization step.
        
        Implements Algorithm 1 from the paper (arXiv:2411.16085):
        m = (u * g > 0).to(g.dtype)  # Alignment mask
        p.add_(u * m/(m.mean() + eps), alpha=-lr)  # Masked update with scaling
        
        Where:
        - u is the update direction from the optimizer (before learning rate)
        - g is the gradient
        - m is the alignment mask
        
        Implementation: We let the optimizer compute and apply its update, then
        compute the actual update vector and check alignment with gradients.
        """
        # Store original parameters and gradients before optimizer step
        old_params = {}
        gradients = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    old_params[id(p)] = p.data.clone()
                    gradients[id(p)] = p.grad.clone()
        
        # Let the base optimizer perform its step (this updates parameters and internal state)
        loss = self.optimizer.step(closure)
        
        # Apply cautious masking: revert and reapply with alignment check
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_id = id(p)
                if param_id not in old_params or param_id not in gradients:
                    continue
                
                old_param = old_params[param_id]
                grad = gradients[param_id]
                
                # Compute the update that was applied by the optimizer
                # This includes learning rate but not weight decay (for AdamW)
                update_applied = p.data - old_param
                
                # For AdamW, we need to account for weight decay
                # Weight decay is applied as: p = (p - lr*u) * (1 - lr*weight_decay)
                # So the actual parameter update before weight decay is:
                weight_decay = group.get('weight_decay', 0)
                if weight_decay > 0:
                    # Reverse the weight decay: p_before_wd = p / (1 - lr * weight_decay)
                    p_before_wd = p.data / (1 - group['lr'] * weight_decay)
                    update_before_wd = p_before_wd - old_param
                else:
                    update_before_wd = update_applied
                
                # Extract update direction u (without learning rate)
                # Since update_before_wd = -lr * u, we have u = -update_before_wd / lr
                lr = group['lr']
                if abs(lr) > 1e-10:
                    u = -update_before_wd / lr
                else:
                    u = -update_before_wd
                
                # Apply cautious masking: m = (u * g > 0) per Algorithm 1
                # This checks if update direction and gradient have the same sign
                alignment_mask = (u * grad > 0).to(grad.dtype)
                
                # Compute mean alignment for normalization
                mask_mean = alignment_mask.mean()
                if mask_mean > 0:
                    # Compute masked and scaled update: u * m / (m.mean() + eps)
                    scaled_u = u * alignment_mask / (mask_mean + self.eps)
                    
                    # Revert to old parameter
                    p.data.copy_(old_param)
                    
                    # Apply cautious update: p = old_param - lr * scaled_u
                    # This matches: p.add_(u * m/(m.mean() + eps), alpha=-lr)
                    p.data.add_(scaled_u, alpha=-lr)
                    
                    # Apply weight decay separately (for AdamW)
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                else:
                    # No aligned dimensions, revert parameter
                    p.data.copy_(old_param)
                    # Still apply weight decay if needed
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
        
        return loss
        
    def add_param_group(self, param_group):
        """Forward add_param_group to base optimizer."""
        return self.optimizer.add_param_group(param_group)


@dataclass
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    lr: float
    weight_decay: float
    grad_clip_norm: float
    use_cautious_optimizer: bool = False
    """Enable Cautious Optimizer wrapper for improved training efficiency.
    
    Based on "Cautious Optimizers: Improving Training with One Line of Code" (arXiv:2411.16085).
    Only updates parameters when update direction aligns with gradient direction.
    Can accelerate convergence by 1.28x-1.47x for Transformer-based models.
    """
    cautious_eps: float = 1e-8
    """Epsilon (ε) for cautious masking normalization factor.
    
    This parameter is used in the masking normalization formula:
        scaled_u = u * m / (m.mean() + cautious_eps)
    
    Where:
    - u: update direction from optimizer
    - m: alignment mask (where u * g > 0)
    - m.mean(): mean of the alignment mask
    
    Purpose:
    1. **Numerical stability**: Prevents division by zero when mask_mean is very small
    2. **Smoothing effect**: Adds a small constant to the denominator to stabilize scaling
    3. **Default value**: 1e-8 (same as typical optimizer eps values)
    
    Usually doesn't need adjustment - the default 1e-8 works well in practice.
    Only modify if you encounter numerical instability issues.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "adam"

    @abc.abstractmethod
    def build(self) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
        """
        Build the optimizer. It can be a single optimizer or a dictionary of optimizers.
        NOTE: Multiple optimizers are useful when you have different models to optimize.
        For example, you can have one optimizer for the policy and another one for the value function
        in reinforcement learning settings.

        Returns:
            The optimizer or a dictionary of optimizers.
        """
        raise NotImplementedError


@OptimizerConfig.register_subclass("adam")
@dataclass
class AdamConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: dict) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        use_cautious = kwargs.pop("use_cautious_optimizer", False)
        cautious_eps = kwargs.pop("cautious_eps", 1e-8)
        kwargs.pop("grad_clip_norm")
        optimizer = torch.optim.Adam(params, **kwargs)
        if use_cautious:
            optimizer = CautiousOptimizer(optimizer, eps=cautious_eps)
        return optimizer


@OptimizerConfig.register_subclass("adamw")
@dataclass
class AdamWConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    grad_clip_norm: float = 10.0

    def build(self, params: dict) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        use_cautious = kwargs.pop("use_cautious_optimizer", False)
        cautious_eps = kwargs.pop("cautious_eps", 1e-8)
        kwargs.pop("grad_clip_norm")
        optimizer = torch.optim.AdamW(params, **kwargs)
        if use_cautious:
            optimizer = CautiousOptimizer(optimizer, eps=cautious_eps)
        return optimizer




@OptimizerConfig.register_subclass("sgd")
@dataclass
class SGDConfig(OptimizerConfig):
    lr: float = 1e-3
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: dict) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        use_cautious = kwargs.pop("use_cautious_optimizer", False)
        cautious_eps = kwargs.pop("cautious_eps", 1e-8)
        kwargs.pop("grad_clip_norm")
        optimizer = torch.optim.SGD(params, **kwargs)
        if use_cautious:
            optimizer = CautiousOptimizer(optimizer, eps=cautious_eps)
        return optimizer


@OptimizerConfig.register_subclass("multi_adam")
@dataclass
class MultiAdamConfig(OptimizerConfig):
    """Configuration for multiple Adam optimizers with different parameter groups.

    This creates a dictionary of Adam optimizers, each with its own hyperparameters.

    Args:
        lr: Default learning rate (used if not specified for a group)
        weight_decay: Default weight decay (used if not specified for a group)
        optimizer_groups: Dictionary mapping parameter group names to their hyperparameters
        grad_clip_norm: Gradient clipping norm
    """

    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    optimizer_groups: dict[str, dict[str, Any]] = field(default_factory=dict)

    def build(self, params_dict: dict[str, list]) -> dict[str, torch.optim.Optimizer]:
        """Build multiple Adam optimizers.

        Args:
            params_dict: Dictionary mapping parameter group names to lists of parameters
                         The keys should match the keys in optimizer_groups

        Returns:
            Dictionary mapping parameter group names to their optimizers
        """
        optimizers = {}

        for name, params in params_dict.items():
            # Get group-specific hyperparameters or use defaults
            group_config = self.optimizer_groups.get(name, {})

            # Create optimizer with merged parameters (defaults + group-specific)
            optimizer_kwargs = {
                "lr": group_config.get("lr", self.lr),
                "betas": group_config.get("betas", (0.9, 0.999)),
                "eps": group_config.get("eps", 1e-5),
                "weight_decay": group_config.get("weight_decay", self.weight_decay),
            }

            optimizers[name] = torch.optim.Adam(params, **optimizer_kwargs)

        return optimizers


def save_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer], save_dir: Path
) -> None:
    """Save optimizer state to disk.

    Args:
        optimizer: Either a single optimizer or a dictionary of optimizers.
        save_dir: Directory to save the optimizer state.
    """
    if isinstance(optimizer, dict):
        # Handle dictionary of optimizers
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            optimizer_dir.mkdir(exist_ok=True, parents=True)
            _save_single_optimizer_state(opt, optimizer_dir)
    else:
        # Handle single optimizer
        _save_single_optimizer_state(optimizer, save_dir)


def _save_single_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> None:
    """Save a single optimizer's state to disk."""
    state = optimizer.state_dict()
    param_groups = state.pop("param_groups")
    flat_state = flatten_dict(state)
    save_file(flat_state, save_dir / OPTIMIZER_STATE)
    write_json(param_groups, save_dir / OPTIMIZER_PARAM_GROUPS)


def load_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer], save_dir: Path
) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
    """Load optimizer state from disk.

    Args:
        optimizer: Either a single optimizer or a dictionary of optimizers.
        save_dir: Directory to load the optimizer state from.

    Returns:
        The updated optimizer(s) with loaded state.
    """
    if isinstance(optimizer, dict):
        # Handle dictionary of optimizers
        loaded_optimizers = {}
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            if optimizer_dir.exists():
                loaded_optimizers[name] = _load_single_optimizer_state(opt, optimizer_dir)
            else:
                loaded_optimizers[name] = opt
        return loaded_optimizers
    else:
        # Handle single optimizer
        return _load_single_optimizer_state(optimizer, save_dir)


def _load_single_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> torch.optim.Optimizer:
    """Load a single optimizer's state from disk."""
    current_state_dict = optimizer.state_dict()
    flat_state = load_file(save_dir / OPTIMIZER_STATE)
    state = unflatten_dict(flat_state)

    # Handle case where 'state' key might not exist (for newly created optimizers)
    if "state" in state:
        loaded_state_dict = {"state": {int(k): v for k, v in state["state"].items()}}
    else:
        loaded_state_dict = {"state": {}}

    if "param_groups" in current_state_dict:
        param_groups = deserialize_json_into_object(
            save_dir / OPTIMIZER_PARAM_GROUPS, current_state_dict["param_groups"]
        )
        loaded_state_dict["param_groups"] = param_groups

    optimizer.load_state_dict(loaded_state_dict)
    return optimizer
