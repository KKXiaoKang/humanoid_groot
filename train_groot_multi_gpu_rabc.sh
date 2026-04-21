#!/bin/bash

# ============================================================================
# Groot + RA-BC 多卡训练脚本
# ============================================================================
# 前置条件：
# 1) 已完成 SARM progress 预计算，得到 sarm_progress.parquet
# 2) 当前代码已包含 RA-BC 适配（use_rabc / rabc_* 参数 + Groot reduction="none"）
#
# 使用方法:
#   bash train_groot_multi_gpu_rabc.sh
#   bash train_groot_multi_gpu_rabc.sh --gpu 0,1,2
#   bash train_groot_multi_gpu_rabc.sh --gpu 0,1 --kappa 0.03 --head sparse
# ============================================================================

set -euo pipefail

# --------------------------
# 输出与任务配置
# --------------------------
OUTPUT_DIR="./outputs/groot_rabc_0415_pick_cube_s62"
JOB_NAME="groot_depalletize_rabc"

# --------------------------
# 数据集配置
# --------------------------
# 这里使用你已生成 progress 的同一个数据集目录，避免 index 不匹配。
DATASET_ROOT="/home/lab/humanoid_groot/SARM_LEROBOT/lerobot/lerobot_dataset/0415_pick_cube_single_s62_real_clawAsS-filtered"
DATASET_REPO_ID="0415_pick_cube_single_s62_real_clawAsS-filtered"

# --------------------------
# RA-BC 配置
# --------------------------
USE_RABC=true
RABC_PROGRESS_PATH="/home/lab/humanoid_groot/SARM_LEROBOT/lerobot/lerobot_dataset/0415_pick_cube_single_s62_real_clawAsS-filtered/sarm_progress.parquet"
RABC_HEAD_MODE="sparse"   # sparse 或 dense
RABC_KAPPA="0.01"
RABC_EPSILON="1e-6"

# --------------------------
# GPU 参数
# --------------------------
GPU_IDS_DEFAULT=""
GPU_IDS="$GPU_IDS_DEFAULT"

# 训练参数
BATCH_SIZE=16
NUM_STEPS=20000
SAVE_FREQ=2000
LOG_FREQ=100
EVAL_FREQ=0
NUM_WORKERS=8

# Delta EEF / 操作模式配置
RELATIVE_ACTION_REF_MODE="state"
MANIPULATION_MODE="bimanual"

# 学习率缩放配置
BASE_LR=1e-4
LR_SCALING_MODE="very_conservative"  # linear/sqrt/conservative/very_conservative/fixed_scale
FIXED_SCALE_FACTOR=1.3

RESUME=false
IMAGE_TRANSFORMS_CONFIG_PATH="config/image_transforms.json"

# --------------------------
# 参数解析
# --------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu|-g)
            GPU_IDS="$2"
            shift 2
            ;;
        --kappa)
            RABC_KAPPA="$2"
            shift 2
            ;;
        --head)
            RABC_HEAD_MODE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

if [[ "${RABC_HEAD_MODE}" != "sparse" && "${RABC_HEAD_MODE}" != "dense" ]]; then
    echo "错误: --head 只能是 sparse 或 dense，当前为 ${RABC_HEAD_MODE}"
    exit 1
fi

if [[ ! -f "${RABC_PROGRESS_PATH}" ]]; then
    echo "错误: 找不到 RABC 进度文件: ${RABC_PROGRESS_PATH}"
    exit 1
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "错误: 数据集根目录不存在: ${DATASET_ROOT}"
    exit 1
fi

export TOKENIZERS_PARALLELISM=false

# --------------------------
# GPU 数量计算
# --------------------------
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || nvidia-smi --list-gpus | wc -l)
fi

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "错误: 未检测到可用 GPU"
    exit 1
fi

# --------------------------
# 学习率缩放
# --------------------------
if [ "$LR_SCALING_MODE" = "linear" ]; then
    SCALED_LR=$(python3 -c "print('{:.6f}'.format($BASE_LR * $NUM_GPUS))")
elif [ "$LR_SCALING_MODE" = "sqrt" ]; then
    SCALED_LR=$(python3 -c "import math; print('{:.6f}'.format($BASE_LR * math.sqrt($NUM_GPUS)))")
elif [ "$LR_SCALING_MODE" = "conservative" ]; then
    SCALED_LR=$(python3 -c "import math; print('{:.6f}'.format($BASE_LR * math.pow($NUM_GPUS, 0.4)))")
elif [ "$LR_SCALING_MODE" = "very_conservative" ]; then
    SCALED_LR=$(python3 -c "import math; print('{:.6f}'.format($BASE_LR * math.pow($NUM_GPUS, 0.3)))")
elif [ "$LR_SCALING_MODE" = "fixed_scale" ]; then
    SCALED_LR=$(python3 -c "print('{:.6f}'.format($BASE_LR * $FIXED_SCALE_FACTOR))")
else
    echo "错误: 未知学习率缩放模式 ${LR_SCALING_MODE}"
    exit 1
fi

echo "=========================================="
echo "🚀 开始 Groot + RA-BC 训练"
echo "GPU: ${GPU_IDS:-ALL} (num=${NUM_GPUS})"
echo "Dataset root: ${DATASET_ROOT}"
echo "Dataset repo_id: ${DATASET_REPO_ID}"
echo "RABC: use=${USE_RABC}, head=${RABC_HEAD_MODE}, kappa=${RABC_KAPPA}"
echo "RABC progress: ${RABC_PROGRESS_PATH}"
echo "chunk_size (policy): 32  # 需与 RA-BC 的 Δ 语义一致"
echo "LR: base=${BASE_LR}, scaled=${SCALED_LR}, mode=${LR_SCALING_MODE}"
echo "Effective batch size: ${BATCH_SIZE} x ${NUM_GPUS} = $((BATCH_SIZE * NUM_GPUS))"
echo "=========================================="

accelerate launch \
  --multi_gpu \
  --num_processes=${NUM_GPUS} \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  $(if [ -n "$IMAGE_TRANSFORMS_CONFIG_PATH" ] && [ -f "$IMAGE_TRANSFORMS_CONFIG_PATH" ]; then echo "--config_path=${IMAGE_TRANSFORMS_CONFIG_PATH}"; fi) \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME} \
  --resume=${RESUME} \
  --save_checkpoint=true \
  --batch_size=${BATCH_SIZE} \
  --steps=${NUM_STEPS} \
  --save_freq=${SAVE_FREQ} \
  --log_freq=${LOG_FREQ} \
  --eval_freq=${EVAL_FREQ} \
  --num_workers=${NUM_WORKERS} \
  --seed=42 \
  \
  --use_rabc=${USE_RABC} \
  --rabc_progress_path="${RABC_PROGRESS_PATH}" \
  --rabc_head_mode="${RABC_HEAD_MODE}" \
  --rabc_kappa=${RABC_KAPPA} \
  --rabc_epsilon=${RABC_EPSILON} \
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
  --policy.action_space_type="Delta eef" \
  --policy.manipulation_mode="${MANIPULATION_MODE}" \
  --policy.relative_action_reference_mode="${RELATIVE_ACTION_REF_MODE}" \
  --policy.optimizer_lr=${SCALED_LR} \
  --policy.warmup_ratio=0.10 \
  --policy.chunk_size=32 \
  --policy.n_action_steps=32 \
  \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.video_backend="decord" \
  $(if [ -z "$IMAGE_TRANSFORMS_CONFIG_PATH" ] || [ ! -f "$IMAGE_TRANSFORMS_CONFIG_PATH" ]; then echo "--dataset.image_transforms.enable=True --dataset.image_transforms.max_num_transforms=2 --dataset.image_transforms.random_order=False"; fi) \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --wandb.project="groot-depalletize-rabc"

echo ""
echo "=========================================="
echo "训练完成！"
echo "模型保存在: ${OUTPUT_DIR}"
echo "Checkpoints: ${OUTPUT_DIR}/checkpoints/"
echo "=========================================="
