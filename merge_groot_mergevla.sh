#!/bin/bash
# MergeVLA 风格的 GROOT 模型融合脚本
# 基于论文: https://arxiv.org/pdf/2511.18810
#
# 使用方式：
#   ./merge_groot_mergevla.sh              # 默认使用 GPU 0
#   ./merge_groot_mergevla.sh 4            # 使用 GPU 4
#   ./merge_groot_mergevla.sh 0,1          # 使用多个 GPU（通过 CUDA_VISIBLE_DEVICES）

set -e

# GPU 控制：第一个参数指定 GPU ID
# ⚠️ 重要：始终使用 CUDA_VISIBLE_DEVICES 来确保只使用指定的 GPU
# 这样不会影响其他 GPU 上的任务，也不会被之前的 CUDA_VISIBLE_DEVICES 影响
if [ -n "$1" ]; then
    GPU_ARG="$1"
    # 检查是否包含逗号（多个 GPU）
    if [[ "$GPU_ARG" == *","* ]]; then
        # 多个 GPU：使用 CUDA_VISIBLE_DEVICES
        export CUDA_VISIBLE_DEVICES="$GPU_ARG"
        DEVICE="cuda:0"  # 在可见的 GPU 中，使用第一个
        GPU_INFO="GPU: $GPU_ARG (CUDA_VISIBLE_DEVICES)"
    else
        # 单个 GPU：也使用 CUDA_VISIBLE_DEVICES 来限制只使用这个 GPU
        export CUDA_VISIBLE_DEVICES="$GPU_ARG"
        DEVICE="cuda:0"  # 在可见的 GPU 中，使用第一个（逻辑上就是指定的 GPU）
        GPU_INFO="GPU: $GPU_ARG (CUDA_VISIBLE_DEVICES, device=cuda:0)"
    fi
else
    # 默认使用 cuda:0，但不设置 CUDA_VISIBLE_DEVICES（使用系统默认）
    # 如果之前环境中有 CUDA_VISIBLE_DEVICES，需要清理
    unset CUDA_VISIBLE_DEVICES
    DEVICE="cuda:0"
    GPU_INFO="GPU: $DEVICE (默认，未限制 CUDA_VISIBLE_DEVICES)"
fi

# 模型路径（根据实际情况修改）
NARROWER_PATH="/home/kangkk/humanoid_groot_base/outputs/0112_h100x4_groot_cross_attention_narrower_very_conservative/checkpoints/020000/pretrained_model"
WIDER_PATH="/home/kangkk/humanoid_groot_base/outputs/0113_h100x4_groot_cross_attention_wider_very_conservative_mix_dense/checkpoints/014000/pretrained_model"
BASE_MODEL_PATH="${NARROWER_PATH}"  # 如果 base == narrower，则 τ_narrower = 0
OUTPUT_PATH="./outputs/merged_groot_mergevla/pretrained_model"

echo "=========================================="
echo "🚀 MergeVLA-Style GROOT Model Merging"
echo "=========================================="
echo "Narrower: ${NARROWER_PATH}"
echo "Wider: ${WIDER_PATH}"
echo "Base: ${BASE_MODEL_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "${GPU_INFO}"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "⚠️ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (只使用这些 GPU)"
fi
echo "=========================================="
echo ""

# 运行 MergeVLA 融合
python scripts/train_weight_merge.py \
    --method mergevla \
    --narrower_path "${NARROWER_PATH}" \
    --wider_path "${WIDER_PATH}" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --adapter_type sparse_lora \
    --lora_rank 16 \
    --sparsity 0.5 \
    --adapter_epochs 20 \
    --adapter_lr 1e-3 \
    --num_samples 10 \
    --batch_size 1 \
    --device "${DEVICE}" \
    --use_default_datasets

echo ""
echo "=========================================="
echo "✅ MergeVLA merging completed!"
echo "=========================================="
echo ""
echo "💡 To evaluate the merged model:"
echo "   python scripts/eval_merged_groot_on_dataset.py \\"
echo "       --model-path ${OUTPUT_PATH} \\"
echo "       --dataset-root /path/to/dataset \\"
echo "       --episode 0 \\"
echo "       --action-chunk-size 32 \\"
echo "       --visualize"
