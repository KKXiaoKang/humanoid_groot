#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Reward-/Advantage-aligned BC weighting for GR00T training.

Two modes are supported:

- ``rabc`` (Reward-Aligned Behavior Cloning, paper "SARM"):
    Weight each sample by a clipped soft-weight derived from progress delta
    ``r_i = phi(o_{t+Δ}) - phi(o_t)`` with global running statistics, plus a
    hard threshold ``kappa`` for prior overrides (Eq. 7-9 of SARM paper).

- ``awbc`` (Advantage-Weighted Behavior Cloning, paper "ARM"):
    Use a length-adaptive gain
    ``ΔG_t = (P_{t+H} - P_t) * (L_seq / L_bar)`` and a per-batch
    statistical clamp
    ``w_i = clamp((ΔG_i - (μ-2σ)) / (4σ + ε), 0, 1)``
    with no kappa override and no global divisor on the loss
    (the loss should be aggregated as ``mean(w_i * l_i)``; see
    ``lerobot_train.py``).

Both paths read the same precomputed ``progress_<head_mode>`` parquet column
(the ``compute_rabc_weights.py`` script in the SARM repo also handles ARM
checkpoints by integrating tri-state advantages into a sigmoid'd progress).
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download


def resolve_hf_path(path: str | Path) -> Path:
    """Resolve a path that may be a HuggingFace URL (hf://datasets/...) to a local path."""
    path_str = str(path)
    if path_str.startswith("hf://datasets/"):
        parts = path_str.replace("hf://datasets/", "").split("/")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))
    return Path(path)


class RABCWeights:
    """
    Load precomputed SARM/ARM progress values and compute per-sample BC weights.

    The class supports two modes, selected via ``mode``:

    ``"rabc"`` (default, SARM paper):
        ``r_i = P[t+Δ] - P[t]`` with offline global ``(μ, σ)`` and a kappa-based
        prior override. Returns weights normalized to sum to ``batch_size``
        (so the downstream ``loss = (l*w).sum() / (w.sum()+ε)`` matches the
        SARM paper's Eq. 7).

    ``"awbc"`` (ARM paper):
        ``ΔG_i = (P[t+H] - P[t]) * (L_seq[ep] / L_bar)``,
        ``b_lower = μ - 2σ``, ``b_upper = μ + 2σ`` from the **current batch**,
        ``w_i = clamp((ΔG_i - b_lower) / (b_upper - b_lower + ε), 0, 1)``.
        No kappa, no global divisor — the downstream loss should be
        ``mean(l*w)``.

    Args:
        progress_path: Path to parquet file with precomputed progress values.
        chunk_size: Number of frames ahead used as the action-chunk horizon
            (Δ in RA-BC, H in AW-BC).
        head_mode: Which precomputed head column to use (``"sparse"`` or
            ``"dense"``).
        mode: ``"rabc"`` or ``"awbc"``.
        kappa: Hard threshold for high-quality samples (only used in ``rabc``
            mode; ignored in ``awbc``).
        epsilon: Small constant for numerical stability.
        fallback_weight: Weight to use for frames without valid delta.
            Default ``1.0`` keeps RA-BC backwards compatible; for paper-strict
            AW-BC behaviour use ``0.0``.
        device: Device to return tensors on.
    """

    def __init__(
        self,
        progress_path: str | Path,
        chunk_size: int = 50,
        head_mode: str = "sparse",
        mode: Literal["rabc", "awbc"] = "rabc",
        kappa: float = 0.01,
        epsilon: float = 1e-6,
        fallback_weight: float = 1.0,
        device: torch.device = None,
    ):
        if mode not in ("rabc", "awbc"):
            raise ValueError(f"mode must be 'rabc' or 'awbc', got {mode!r}")

        self.progress_path = resolve_hf_path(progress_path)
        self.chunk_size = chunk_size
        self.head_mode = head_mode
        self.mode = mode
        self.kappa = kappa
        self.epsilon = epsilon
        self.fallback_weight = fallback_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.progress_column = f"progress_{head_mode}"

        logging.info(f"Loading progress values from {self.progress_path} (mode={mode})")
        self.df = pd.read_parquet(self.progress_path)

        if self.progress_column not in self.df.columns:
            available = [c for c in self.df.columns if c.startswith("progress")]
            raise ValueError(
                f"Column '{self.progress_column}' not found. Available progress columns: {available}"
            )

        logging.info(f"Using progress column: {self.progress_column}")

        self.progress_lookup: dict[int, float] = {}
        self.episode_lookup: dict[int, int] = {}

        for _, row in self.df.iterrows():
            global_idx = int(row["index"])
            progress = row[self.progress_column]
            episode_idx = int(row["episode_index"])

            if not np.isnan(progress):
                self.progress_lookup[global_idx] = float(progress)
            self.episode_lookup[global_idx] = episode_idx

        # Per-episode boundaries (also used to derive episode lengths for AW-BC).
        self.episode_boundaries: dict[int, dict[str, int]] = {}
        episode_lengths: list[int] = []
        for episode_idx in self.df["episode_index"].unique():
            ep_df = self.df[self.df["episode_index"] == episode_idx]
            ep_start = int(ep_df["index"].min())
            ep_end = int(ep_df["index"].max()) + 1
            self.episode_boundaries[int(episode_idx)] = {"start": ep_start, "end": ep_end}
            episode_lengths.append(ep_end - ep_start)

        self.episode_lengths: dict[int, int] = {
            ep: bounds["end"] - bounds["start"] for ep, bounds in self.episode_boundaries.items()
        }
        self.mean_episode_length: float = (
            float(np.mean(episode_lengths)) if episode_lengths else 1.0
        )

        logging.info(f"Loaded {len(self.progress_lookup)} frame progress values")
        logging.info(
            f"Chunk/horizon size: {chunk_size}, "
            f"num_episodes={len(self.episode_lengths)}, "
            f"mean_episode_length={self.mean_episode_length:.1f}"
        )

        # Pre-compute global delta stats only when needed (used by RA-BC). AW-BC
        # uses per-batch stats so this is a no-op there.
        self.delta_mean: float = 0.0
        self.delta_std: float = self.epsilon
        if self.mode == "rabc":
            self._compute_global_stats()

    def _compute_global_stats(self) -> None:
        """Compute global mean and std of progress deltas (RA-BC only)."""
        all_deltas: list[float] = []

        for global_idx, progress in self.progress_lookup.items():
            episode_idx = self.episode_lookup.get(global_idx)
            if episode_idx is None:
                continue
            bounds = self.episode_boundaries.get(episode_idx)
            if bounds is None:
                continue

            future_idx = global_idx + self.chunk_size
            if future_idx >= bounds["end"]:
                future_idx = bounds["end"] - 1

            future_progress = self.progress_lookup.get(future_idx)
            if future_progress is not None:
                all_deltas.append(future_progress - progress)

        if all_deltas:
            self.delta_mean = max(float(np.mean(all_deltas)), 0.0)
            self.delta_std = max(float(np.std(all_deltas)), self.epsilon)
            logging.info(
                f"[RA-BC] Progress delta stats: mean={self.delta_mean:.4f}, std={self.delta_std:.4f}"
            )
        else:
            self.delta_mean = 0.0
            self.delta_std = self.epsilon
            logging.warning("[RA-BC] No valid progress deltas found, using default stats")

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Compute per-sample weights for a batch.

        Returns:
            Tuple of (weights tensor of shape (batch_size,), stats dict).
            For ``rabc`` mode the weights are normalized to sum to
            ``batch_size``; for ``awbc`` mode the raw clipped weights are
            returned (the loss layer should aggregate via ``mean``).
        """
        indices = batch.get("index")
        if indices is None:
            logging.warning("RA-BC/AW-BC: batch missing 'index' key, using uniform weights")
            batch_size = self._get_batch_size(batch)
            return torch.ones(batch_size, device=self.device), {"raw_mean_weight": 1.0}

        if isinstance(indices, torch.Tensor):
            indices_list = indices.cpu().numpy().tolist()
        elif isinstance(indices, np.ndarray):
            indices_list = indices.tolist()
        else:
            indices_list = list(indices)

        deltas = np.empty(len(indices_list), dtype=np.float32)
        episode_lens = np.empty(len(indices_list), dtype=np.float32)
        for i, idx in enumerate(indices_list):
            idx_int = int(idx)
            deltas[i] = self._compute_delta(idx_int)
            ep = self.episode_lookup.get(idx_int)
            episode_lens[i] = (
                float(self.episode_lengths.get(ep, self.mean_episode_length))
                if ep is not None
                else self.mean_episode_length
            )

        if self.mode == "awbc":
            weights, stats = self._compute_awbc_weights(deltas, episode_lens)
        else:
            weights, stats = self._compute_rabc_weights(deltas)

        weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)

        if self.mode == "rabc":
            # Backwards-compatible normalization to sum=batch_size so that
            # `loss = (l*w).sum() / (w.sum()+ε)` reproduces SARM paper Eq. 7.
            batch_size = len(weights_tensor)
            weight_sum = weights_tensor.sum() + self.epsilon
            weights_tensor = weights_tensor * batch_size / weight_sum

        return weights_tensor, stats

    def _compute_delta(self, global_idx: int) -> float:
        """Raw progress delta P[t+Δ] - P[t] (no length-adaptive scaling)."""
        current_progress = self.progress_lookup.get(global_idx)
        if current_progress is None:
            return np.nan

        episode_idx = self.episode_lookup.get(global_idx)
        if episode_idx is None:
            return np.nan
        bounds = self.episode_boundaries.get(episode_idx)
        if bounds is None:
            return np.nan

        future_idx = global_idx + self.chunk_size
        if future_idx >= bounds["end"]:
            future_idx = bounds["end"] - 1

        future_progress = self.progress_lookup.get(future_idx)
        if future_progress is None:
            return np.nan

        return future_progress - current_progress

    def _compute_rabc_weights(self, deltas: np.ndarray) -> tuple[np.ndarray, dict]:
        """SARM RA-BC weighting (Eq. 8-9), uses offline global stats."""
        valid_mask = ~np.isnan(deltas)

        lower_bound = self.delta_mean - 2 * self.delta_std
        soft_weights = (deltas - lower_bound) / (4 * self.delta_std + self.epsilon)
        soft_weights = np.clip(soft_weights, 0.0, 1.0)

        weights = np.zeros_like(deltas, dtype=np.float32)

        # ri > kappa  -> 1
        high_quality_mask = deltas > self.kappa
        weights[high_quality_mask] = 1.0

        # 0 <= ri <= kappa  -> soft weight
        moderate_mask = (deltas >= 0) & (deltas <= self.kappa)
        weights[moderate_mask] = soft_weights[moderate_mask]

        # ri < 0   -> 0  (already zero)
        # NaN      -> fallback
        weights[~valid_mask] = self.fallback_weight

        stats = {
            "raw_mean_weight": float(np.nanmean(weights)) if weights.size else 1.0,
            "num_zero_weight": int(np.sum(weights == 0)),
            "num_full_weight": int(np.sum(weights == 1.0)),
            "num_nan_delta": int(np.sum(~valid_mask)),
            "delta_batch_mean": float(np.nanmean(deltas)) if np.any(valid_mask) else 0.0,
            "delta_batch_std": float(np.nanstd(deltas)) if np.any(valid_mask) else 0.0,
        }
        return weights, stats

    def _compute_awbc_weights(
        self, deltas: np.ndarray, episode_lens: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """ARM AW-BC weighting with length-adaptive gain and per-batch stats.

        ΔG_i = (P[t+H] - P[t]) * (L_seq[ep] / L_bar)
        μ, σ from the **current batch** of valid ΔG_i
        b_lower = μ - 2σ,  b_upper = μ + 2σ
        w_i = clamp((ΔG_i - b_lower) / (b_upper - b_lower + ε), 0, 1)
        """
        valid_mask = ~np.isnan(deltas)
        gains = np.full_like(deltas, np.nan, dtype=np.float32)
        if np.any(valid_mask):
            length_ratio = episode_lens[valid_mask] / max(self.mean_episode_length, 1.0)
            gains[valid_mask] = deltas[valid_mask] * length_ratio

        valid_gains = gains[valid_mask]
        if valid_gains.size > 0:
            mu = float(np.mean(valid_gains))
            sigma = max(float(np.std(valid_gains)), self.epsilon)
        else:
            mu = 0.0
            sigma = self.epsilon

        b_lower = mu - 2.0 * sigma
        b_upper = mu + 2.0 * sigma
        denom = (b_upper - b_lower) + self.epsilon

        weights = np.zeros_like(deltas, dtype=np.float32)
        if np.any(valid_mask):
            soft = (gains[valid_mask] - b_lower) / denom
            weights[valid_mask] = np.clip(soft, 0.0, 1.0)

        # Frames without a valid future progress (e.g. episode end / NaN).
        weights[~valid_mask] = self.fallback_weight

        stats = {
            "raw_mean_weight": float(np.nanmean(weights)) if weights.size else 1.0,
            "num_zero_weight": int(np.sum(weights == 0)),
            "num_full_weight": int(np.sum(weights >= 1.0 - 1e-6)),
            "num_nan_delta": int(np.sum(~valid_mask)),
            "delta_batch_mean": float(np.nanmean(deltas)) if np.any(valid_mask) else 0.0,
            "delta_batch_std": float(np.nanstd(deltas)) if np.any(valid_mask) else 0.0,
            "gain_batch_mean": mu,
            "gain_batch_std": sigma,
            "gain_b_lower": b_lower,
            "gain_b_upper": b_upper,
        }
        return weights, stats

    def _get_batch_size(self, batch: dict) -> int:
        for key in ("action", "index"):
            if key in batch:
                val = batch[key]
                if isinstance(val, (torch.Tensor, np.ndarray)):
                    return val.shape[0]
        return 1

    def get_stats(self) -> dict:
        """Return constant statistics about the loaded progress dataset."""
        return {
            "mode": self.mode,
            "num_frames": len(self.progress_lookup),
            "num_episodes": len(self.episode_lengths),
            "mean_episode_length": self.mean_episode_length,
            "chunk_size": self.chunk_size,
            "head_mode": self.head_mode,
            "delta_mean": self.delta_mean,
            "delta_std": self.delta_std,
            "kappa": self.kappa if self.mode == "rabc" else None,
        }
