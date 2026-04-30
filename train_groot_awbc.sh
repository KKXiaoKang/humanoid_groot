#!/bin/bash
# ============================================================================
# GR00T N1.5 + AW-BC (Advantage-Weighted Behavior Cloning, ARM paper)
# ============================================================================
# Pre-requisites:
#   1. ARM/SARM checkpoint trained:
#        SARM_model/outputs/train/sarm_dual_<ts>/checkpoints/<step>/pretrained_model
#   2. progress parquet generated for the SAME LeRobotDataset that you train
#      GR00T on, e.g.:
#        cd SARM_model && PYTHONPATH=src python -m \
#          lerobot.policies.sarm.compute_rabc_weights \
#          --dataset-repo-id <ds> --reward-model-path <ckpt> \
#          --head-mode sparse --output-path <ds>/sarm_progress.parquet \
#          --awbc-progress-renorm
#
# Usage:
#   bash train_groot_awbc.sh                       # all visible GPUs
#   bash train_groot_awbc.sh --gpu 0,1,2,3
#   bash train_groot_awbc.sh --gpu 0 --dryrun      # 50-step sanity check
#   bash train_groot_awbc.sh --gpu 0,1 --rabc_mode=rabc   # extra args are
#       forwarded to lerobot-train, so you can override anything here
# ============================================================================

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

# ----------------------------------------------------------------------------
# Dataset / progress file (the one used to train ARM + generate parquet)
# ----------------------------------------------------------------------------
DATASET_ROOT="/home/zhangtianchu/SARM_model/lerobot_dataset/0415_pick_cube_single_s62_real_clawAsS-filtered"
DATASET_REPO_ID="0415_pick_cube_single_s62_real_clawAsS-filtered"
RABC_PROGRESS_PATH="${DATASET_ROOT}/sarm_progress.parquet"

# ----------------------------------------------------------------------------
# Model / action space (matches the dataset above: 20D bimanual EEF + grippers)
# ----------------------------------------------------------------------------
MANIPULATION_MODE="bimanual"
ACTION_SPACE_TYPE="Delta eef"
RELATIVE_ACTION_REF_MODE="state"
CHUNK_SIZE=32
N_ACTION_STEPS=32

# ----------------------------------------------------------------------------
# AW-BC knobs (paper-aligned defaults)
# ----------------------------------------------------------------------------
USE_RABC="true"
RABC_MODE="awbc"            # "awbc" (ARM paper) | "rabc" (SARM paper)
RABC_HEAD_MODE="sparse"     # which progress column to read
RABC_FALLBACK_WEIGHT=0.0    # paper-strict: drop frames w/o valid ΔG (last-H)

# ----------------------------------------------------------------------------
# Defaults you usually want to tweak
# ----------------------------------------------------------------------------
OUTPUT_DIR_BASE="./outputs/groot_awbc_pick_cube"
JOB_NAME="groot_awbc_pick_cube"
BATCH_SIZE=24
NUM_STEPS=20000
SAVE_FREQ=2000
LOG_FREQ=10
EVAL_FREQ=0
NUM_WORKERS=8

BASE_LR=1e-4
LR_SCALING_MODE="very_conservative"   # linear|sqrt|conservative|very_conservative|fixed_scale
FIXED_SCALE_FACTOR=1.3

# ----------------------------------------------------------------------------
# CLI parsing: --gpu / --dryrun consumed here, everything else forwarded to
# lerobot-train so you can override per-run.
# ----------------------------------------------------------------------------
GPU_IDS=""
DRYRUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu|-g)
      GPU_IDS="$2"; shift 2 ;;
    --dryrun)
      DRYRUN=1; shift ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ "${DRYRUN}" -eq 1 ]]; then
  BATCH_SIZE=8
  NUM_STEPS=50
  SAVE_FREQ=50
  LOG_FREQ=5
  OUTPUT_DIR_BASE="./outputs/groot_awbc_dryrun"
  JOB_NAME="groot_awbc_dryrun"
fi

OUTPUT_DIR="${OUTPUT_DIR_BASE}_$(date +%Y%m%d_%H%M%S)"

# ----------------------------------------------------------------------------
# GPU + LR scaling
# ----------------------------------------------------------------------------
if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | sed '/^$/d' | wc -l)
else
  NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "error: no GPU available" >&2; exit 1
fi

case "${LR_SCALING_MODE}" in
  linear)             SCALED_LR=$(python3 -c "print('{:.6f}'.format(${BASE_LR}*${NUM_GPUS}))") ;;
  sqrt)               SCALED_LR=$(python3 -c "import math;print('{:.6f}'.format(${BASE_LR}*math.sqrt(${NUM_GPUS})))") ;;
  conservative)       SCALED_LR=$(python3 -c "import math;print('{:.6f}'.format(${BASE_LR}*math.pow(${NUM_GPUS},0.4)))") ;;
  very_conservative)  SCALED_LR=$(python3 -c "import math;print('{:.6f}'.format(${BASE_LR}*math.pow(${NUM_GPUS},0.3)))") ;;
  fixed_scale)        SCALED_LR=$(python3 -c "print('{:.6f}'.format(${BASE_LR}*${FIXED_SCALE_FACTOR}))") ;;
  *) echo "unknown LR_SCALING_MODE=${LR_SCALING_MODE}"; exit 1 ;;
esac

# ----------------------------------------------------------------------------
# Pre-flight check: parquet exists & has the right column?
# ----------------------------------------------------------------------------
if [[ ! -f "${RABC_PROGRESS_PATH}" ]]; then
  echo "error: progress parquet not found at ${RABC_PROGRESS_PATH}" >&2
  echo "       run compute_rabc_weights.py first (see header)." >&2
  exit 1
fi
python3 - <<PY
import pandas as pd, sys
df = pd.read_parquet("${RABC_PROGRESS_PATH}")
need = "progress_${RABC_HEAD_MODE}"
if need not in df.columns:
    sys.exit(f"parquet missing column {need!r} (have {[c for c in df.columns if c.startswith('progress')]})")
ep = df.groupby("episode_index")["progress_${RABC_HEAD_MODE}"].agg(["min","max","count"])
print(f"[parquet] {need}: episodes={len(ep)}, "
      f"per-ep P_0 mean={ep['min'].mean():.3f}, "
      f"P_T mean={ep['max'].mean():.3f}, "
      f"avg episode length={ep['count'].mean():.1f}")
PY

LEROBOT_TRAIN="$(command -v lerobot-train || true)"
if [[ -z "${LEROBOT_TRAIN}" ]]; then
  echo "error: lerobot-train not found on PATH" >&2; exit 1
fi

echo "=========================================="
echo "GR00T + AW-BC training"
echo "  num_gpus=${NUM_GPUS}  scaled_lr=${SCALED_LR}  batch_size=${BATCH_SIZE}"
echo "  effective batch size = ${BATCH_SIZE} x ${NUM_GPUS} = $((BATCH_SIZE*NUM_GPUS))"
echo "  steps=${NUM_STEPS}  log_freq=${LOG_FREQ}  save_freq=${SAVE_FREQ}"
echo "  dataset.root=${DATASET_ROOT}"
echo "  rabc_progress_path=${RABC_PROGRESS_PATH}"
echo "  rabc_mode=${RABC_MODE}  head_mode=${RABC_HEAD_MODE}  fallback=${RABC_FALLBACK_WEIGHT}"
echo "  manipulation_mode=${MANIPULATION_MODE}  action_space=${ACTION_SPACE_TYPE}"
echo "  output_dir=${OUTPUT_DIR}"
echo "=========================================="

accelerate launch \
  --multi_gpu \
  --num_processes=${NUM_GPUS} \
  --mixed_precision=bf16 \
  "${LEROBOT_TRAIN}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --resume=false \
  --save_checkpoint=true \
  --batch_size=${BATCH_SIZE} \
  --steps=${NUM_STEPS} \
  --save_freq=${SAVE_FREQ} \
  --log_freq=${LOG_FREQ} \
  --eval_freq=${EVAL_FREQ} \
  --num_workers=${NUM_WORKERS} \
  --seed=42 \
  \
  --policy.type=groot \
  --policy.base_model_path="nvidia/GR00T-N1.5-3B" \
  --policy.push_to_hub=false \
  --policy.tune_llm=true \
  --policy.tune_visual=true \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.use_bf16=true \
  --policy.max_state_dim=64 \
  --policy.max_action_dim=32 \
  --policy.action_space_type="${ACTION_SPACE_TYPE}" \
  --policy.manipulation_mode="${MANIPULATION_MODE}" \
  --policy.relative_action_reference_mode="${RELATIVE_ACTION_REF_MODE}" \
  --policy.optimizer_lr=${SCALED_LR} \
  --policy.warmup_ratio=0.10 \
  --policy.chunk_size=${CHUNK_SIZE} \
  --policy.n_action_steps=${N_ACTION_STEPS} \
  \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend="decord" \
  \
  --use_rabc=${USE_RABC} \
  --rabc_mode=${RABC_MODE} \
  --rabc_head_mode=${RABC_HEAD_MODE} \
  --rabc_fallback_weight=${RABC_FALLBACK_WEIGHT} \
  --rabc_progress_path="${RABC_PROGRESS_PATH}" \
  \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --wandb.project="groot-awbc" \
  "${EXTRA_ARGS[@]}"

echo "=========================================="
echo "Done. checkpoints -> ${OUTPUT_DIR}/checkpoints/"
echo "=========================================="
