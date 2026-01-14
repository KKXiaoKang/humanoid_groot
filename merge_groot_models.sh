#!/bin/bash
#
# GROOT 模型权重融合快速脚本
#
# 使用方式：
#   ./merge_groot_models.sh                   # 默认使用 Expert Merging（效果最好）⭐
#   ./merge_groot_models.sh expert_merge      # Expert Merging（效果最好，使用训练数据校准）
#   ./merge_groot_models.sh expert_merge 0    # 指定使用 GPU 0
#   ./merge_groot_models.sh expert_merge 0,1  # 指定使用 GPU 0 和 1（通过 CUDA_VISIBLE_DEVICES）
#   ./merge_groot_models.sh task_arithmetic   # Task Arithmetic 融合（无需训练，快速）
#   ./merge_groot_models.sh ties              # TIES 融合
#   ./merge_groot_models.sh dare              # DARE 融合
#   ./merge_groot_models.sh interpolation     # 直接插值（最简单）
#
# GPU 控制：
#   可以通过第二个参数指定 GPU：
#   - 单个 GPU: ./merge_groot_models.sh expert_merge 0
#   - 多个 GPU: ./merge_groot_models.sh expert_merge 0,1  (使用 CUDA_VISIBLE_DEVICES)
#   - 默认: 使用 cuda:0
#
# 校准数据说明：
#   Expert Merging 需要少量校准数据（5-10个样本），可以使用原始训练数据集！
#   默认会自动使用 lerobot_data 文件夹中的数据集
#

set -e

# 默认方法改为 expert_merge（效果最好）
METHOD=${1:-expert_merge}

# 模型路径
NARROWER_PATH="/home/kangkk/humanoid_groot_base/outputs/0112_h100x4_groot_cross_attention_narrower_very_conservative/checkpoints/020000/pretrained_model"
WIDER_PATH="/home/kangkk/humanoid_groot_base/outputs/0113_h100x4_groot_cross_attention_wider_very_conservative_mix_dense/checkpoints/014000/pretrained_model"
OUTPUT_PATH="./outputs/merged_groot/pretrained_model"

# ⚠️ 重要：Base 模型必须与专家模型有相同的架构！
# 使用 narrower 模型作为 base（不能使用 nvidia/GR00T-N1.5-3B，结构不兼容）
BASE_MODEL_PATH="$NARROWER_PATH"

# GPU 控制：第二个参数指定 GPU ID
# ⚠️ 重要：始终使用 CUDA_VISIBLE_DEVICES 来确保只使用指定的 GPU
# 这样不会影响其他 GPU 上的任务，也不会被之前的 CUDA_VISIBLE_DEVICES 影响
if [ -n "$2" ]; then
    GPU_ARG="$2"
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

echo "======================================"
echo "🚀 GROOT 模型权重融合"
echo "   方法: ${METHOD}"
echo "   ${GPU_INFO}"
echo "   输出: ${OUTPUT_PATH}"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "   ⚠️ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (只使用这些 GPU)"
fi
echo "======================================"

case $METHOD in
    expert_merge)
        # ⭐ 默认方法 - 基于论文 "Expert Merging" (arXiv:2509.25712)
        echo "使用 Expert Merging 方法（基于论文推荐参数）⭐..."
        echo ""
        echo "📚 校准数据：将使用 lerobot_data/split_dataset 文件夹中的训练数据集"
        echo "   - 窄箱子数据集 (narrower): four, random, dense, mix"
        echo "   - 宽箱子数据集 (wider): four, random, dense, mix"
        echo "   - 论文推荐: N=5-10 个样本即可 (Table 11-13)"
        echo ""
        echo "⚠️ 论文参数说明 (Table 9, 10):"
        echo "   - γ (regularization_weight) = 0.8: 论文最优值"
        echo "   - N (num_samples) = 5: 性能饱和点"
        echo "   - epochs = 10: 足够收敛"
        echo ""
        echo "⚠️ 重要：只融合 Backbone，Action Head (DiT) 使用 narrower 模型的！"
        echo "   DiT 对参数变化非常敏感，不能直接线性插值。"
        echo ""
        echo "💡 注意：Hidden loss 不下降是正常的！这是多目标优化问题。"
        echo "   真正的评估应该在实际任务上进行。"
        echo ""
        python scripts/train_weight_merge.py \
            --method expert_merge \
            --narrower_path "$NARROWER_PATH" \
            --wider_path "$WIDER_PATH" \
            --base_model_path "$BASE_MODEL_PATH" \
            --use_default_datasets \
            --num_samples 5 \
            --num_epochs 10 \
            --lr 1e-3 \
            --regularization_weight 0.8 \
            --initial_coefficient 0.5 \
            --merge_backbone_only \
            --action_head_source first_expert \
            --device "$DEVICE" \
            --output_path "$OUTPUT_PATH"
        ;;
    
    task_arithmetic)
        echo "使用 Task Arithmetic 方法（无需训练，快速）..."
        echo "⚠️ 重要：只融合 Backbone，Action Head (DiT) 使用 narrower 模型的！"
        python scripts/train_weight_merge.py \
            --method task_arithmetic \
            --narrower_path "$NARROWER_PATH" \
            --wider_path "$WIDER_PATH" \
            --base_model_path "$BASE_MODEL_PATH" \
            --narrower_weight 0.5 \
            --wider_weight 0.5 \
            --skip_action_head \
            --device "$DEVICE" \
            --output_path "$OUTPUT_PATH"
        ;;
    
    ties)
        echo "使用 TIES Merging 方法..."
        echo "⚠️ 重要：只融合 Backbone，Action Head (DiT) 使用 narrower 模型的！"
        python scripts/train_weight_merge.py \
            --method ties \
            --narrower_path "$NARROWER_PATH" \
            --wider_path "$WIDER_PATH" \
            --base_model_path "$BASE_MODEL_PATH" \
            --ties_trim_ratio 0.2 \
            --ties_scale 1.0 \
            --skip_action_head \
            --device "$DEVICE" \
            --output_path "$OUTPUT_PATH"
        ;;
    
    dare)
        echo "使用 DARE Merging 方法..."
        echo "⚠️ 重要：只融合 Backbone，Action Head (DiT) 使用 narrower 模型的！"
        python scripts/train_weight_merge.py \
            --method dare \
            --narrower_path "$NARROWER_PATH" \
            --wider_path "$WIDER_PATH" \
            --base_model_path "$BASE_MODEL_PATH" \
            --dare_drop_rate 0.1 \
            --narrower_weight 0.5 \
            --wider_weight 0.5 \
            --skip_action_head \
            --device "$DEVICE" \
            --output_path "$OUTPUT_PATH"
        ;;
    
    interpolation)
        echo "使用直接插值方法（最简单）..."
        echo "⚠️ 重要：只融合 Backbone，Action Head (DiT) 使用 narrower 模型的！"
        python scripts/train_weight_merge.py \
            --method interpolation \
            --narrower_path "$NARROWER_PATH" \
            --wider_path "$WIDER_PATH" \
            --alpha 0.5 \
            --skip_action_head \
            --device "$DEVICE" \
            --output_path "$OUTPUT_PATH"
        ;;
    
    *)
        echo "未知方法: $METHOD"
        echo "可用方法: task_arithmetic, ties, dare, expert_merge, interpolation"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "✅ 融合完成！模型保存在: $OUTPUT_PATH"
echo ""
echo "💡 评估融合模型："
echo "   python eval/eval_merged_groot.py --model_path $OUTPUT_PATH"
echo "======================================"
