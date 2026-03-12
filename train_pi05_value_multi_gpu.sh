#!/bin/bash
# ============================================================================
# PI05 Value Function 多卡训练脚本
# ============================================================================
# 训练 π*0.6 RECAP Value Function (仅训练 Value Head，不训练策略网络)
#
# 前提条件:
#   1. 数据集已通过 add_reward_to_dataset.py 添加了 reward 和 target_value 字段
#   2. PI05 预训练模型已下载或可通过 HuggingFace 加载
#
# 使用方法:
#   bash train_pi05_value_multi_gpu.sh                    # 使用所有可用GPU
#   bash train_pi05_value_multi_gpu.sh --gpu 0,1,2        # 使用指定的GPU
#   bash train_pi05_value_multi_gpu.sh -g 0,1             # 使用GPU 0和1
#
# 训练流程:
#   1. 加载 PI05 预训练策略网络 (冻结，不训练)
#   2. 初始化 Value Function (小型 VLA: SigLIP + Gemma 300M + MLP Head)
#   3. 对每个样本: Value Function 接收 (图像, 语言指令) → 预测 V(s, ℓ)
#   4. 损失: MSE(V_predicted, target_value)  其中 target_value = Σ r_t (预计算的 Return)
#   5. 仅更新 Value Function 的参数
#
# 重要提示:
#   - training_mode="value_function" 会自动:
#     a) 启用 value function
#     b) 冻结策略网络参数
#     c) forward() 自动路由到 value function 损失计算
#     d) get_optim_params() 只返回 value function 参数
#   - 数据集必须包含 "target_value" 列 (由 add_reward_to_dataset.py 生成)
# ============================================================================

# 设置输出目录
OUTPUT_DIR="./outputs/pi05_value_function_training"
JOB_NAME="pi05_value_function"

# ============================================================================
# 数据集配置
# ============================================================================
# 数据集必须包含 reward 和 target_value 字段
# 使用 add_reward_to_dataset.py 预处理数据集:
#   python scripts/add_reward_to_dataset.py \
#       --dataset_path /path/to/dataset \
#       --reward_type progress_linear \
#       --output_path /path/to/output \
#       --verify
#
# 单数据集配置:
#   DATASET_ROOT 指向数据集目录（包含 meta/ 和 data/ 的目录）
#   DATASET_REPO_ID 为数据集名称
#
# 多数据集配置:
#   DATASET_ROOT 指向包含所有数据集目录的父目录
#   DATASET_REPO_ID 为逗号分隔的多个数据集名称
# ============================================================================

# 数据集路径配置
# DATASET_ROOT 是包含所有数据集目录的父目录
# DATASET_REPO_ID 是逗号分隔的数据集目录名
DATASET_ROOT="/home/kangkk/humanoid_groot/lerobot_data/eef_dataset"
DATASET_REPO_ID="eef_3x2_with_reward,eef_mix_color_with_reward,eef_short_dense_with_reward"

# PI05 预训练模型路径 (可以是 HuggingFace model ID 或本地路径)
# Value Function 训练模式下策略网络会被冻结
# 如果不使用预训练模型，留空 (模型将随机初始化)
PI05_PRETRAINED_PATH=""
# PI05_PRETRAINED_PATH="/path/to/pi05/checkpoint"

# GPU选择配置
GPU_IDS_DEFAULT=""

# 解析命令行参数
GPU_IDS="$GPU_IDS_DEFAULT"
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu|-g)
            GPU_IDS="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dataset_root|-d)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --dataset_repo|-r)
            DATASET_REPO_ID="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# 环境变量设置
export TOKENIZERS_PARALLELISM=false

# 设置CUDA可见设备并计算GPU数量
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
    echo "=========================================="
    echo "指定使用GPU: $GPU_IDS"
    echo "GPU数量: $NUM_GPUS"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "=========================================="
else
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || nvidia-smi --list-gpus | wc -l)
    echo "=========================================="
    echo "使用所有可用GPU"
    echo "检测到GPU数量: $NUM_GPUS"
    echo "=========================================="
fi

# 验证GPU数量
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "错误: 未检测到可用的GPU！"
    exit 1
fi

# ============================================================================
# 训练参数 (4x RTX 5880 48GB 优化配置)
# ============================================================================
BATCH_SIZE=16             # 每个GPU的batch size (VF模型小, 48GB可以支持16)
                          # 有效batch = 16 × 4 = 64
NUM_STEPS=50000           # 训练步数 (~30K帧, 每步64样本, ~21轮遍历)
SAVE_FREQ=2000            # 每N步保存一次checkpoint
LOG_FREQ=50               # 每N步打印一次日志
EVAL_FREQ=0               # 设置为0禁用评估 (Value Function 训练不需要环境评估)
NUM_WORKERS=4             # 数据加载器工作进程数

# ============================================================================
# Value Function 训练超参数
# ============================================================================
# Value Function 学习率 (通常比策略网络更大，因为模型更小)
VALUE_FUNCTION_LR=1e-4
# Value Function 权重衰减
VALUE_FUNCTION_WEIGHT_DECAY=0.01
# Value Function VLM 变体 (gemma_300m 约 270M 参数，比策略网络的 2B 小很多)
VALUE_FUNCTION_VARIANT="gemma_300m"
# 是否共享视觉编码器 (无预训练模型时建议设为 false)
SHARE_VISION_ENCODER=false
# Value Head MLP dropout
VALUE_HEAD_DROPOUT=0.1
# 梯度检查点 (节省内存)
GRADIENT_CHECKPOINTING=true
# 数据精度 (RTX 5880 支持 bf16, 节省内存且训练更快)
DTYPE="bfloat16"

# ============================================================================
# 学习率缩放配置
# ============================================================================
BASE_LR=${VALUE_FUNCTION_LR}
LR_SCALING_MODE="sqrt"  # Value Function 训练推荐 sqrt 缩放

if [ "$LR_SCALING_MODE" = "linear" ]; then
    SCALED_LR=$(python3 -c "print('{:.6f}'.format($BASE_LR * $NUM_GPUS))")
    echo "使用线性缩放: lr = ${BASE_LR} × ${NUM_GPUS} = ${SCALED_LR}"
elif [ "$LR_SCALING_MODE" = "sqrt" ]; then
    SCALED_LR=$(python3 -c "import math; print('{:.6f}'.format($BASE_LR * math.sqrt($NUM_GPUS)))")
    echo "使用平方根缩放: lr = ${BASE_LR} × √${NUM_GPUS} = ${SCALED_LR}"
elif [ "$LR_SCALING_MODE" = "conservative" ]; then
    SCALED_LR=$(python3 -c "import math; print('{:.6f}'.format($BASE_LR * math.pow($NUM_GPUS, 0.4)))")
    echo "使用保守缩放: lr = ${BASE_LR} × ${NUM_GPUS}^0.4 = ${SCALED_LR}"
else
    SCALED_LR=${BASE_LR}
    echo "使用基础学习率: lr = ${SCALED_LR}"
fi

# 是否从checkpoint继续训练
RESUME=false

# ============================================================================
# 打印配置信息
# ============================================================================
echo ""
echo "=========================================="
echo "🧠 PI05 Value Function 训练配置:"
echo "=========================================="
echo "   训练模式: value_function (仅训练 Value Head)"
echo "   策略网络: 冻结 (不参与训练)"
echo "   Value Function VLM: ${VALUE_FUNCTION_VARIANT}"
echo "   共享视觉编码器: ${SHARE_VISION_ENCODER}"
echo "   Value Head Dropout: ${VALUE_HEAD_DROPOUT}"
echo "   梯度检查点: ${GRADIENT_CHECKPOINTING}"
echo "   精度: ${DTYPE}"
echo ""
echo "   数据集: ${DATASET_REPO_ID}"
echo "   数据集路径: ${DATASET_ROOT}"
echo ""
echo "   学习率: ${SCALED_LR} (缩放模式: ${LR_SCALING_MODE})"
echo "   Batch Size: ${BATCH_SIZE} × ${NUM_GPUS} = $((BATCH_SIZE * NUM_GPUS)) (effective)"
echo "   训练步数: ${NUM_STEPS}"
echo "   保存频率: 每 ${SAVE_FREQ} 步"
echo ""
echo "   损失函数: MSE(V_predicted, target_value)"
echo "   target_value = Σ_{t'=t}^T r_{t'} (预计算的 Return)"
echo "=========================================="
echo ""

# ============================================================================
# 构建训练命令
# ============================================================================
# 构建预训练路径参数
PRETRAINED_ARG=""
if [ -n "$PI05_PRETRAINED_PATH" ]; then
    PRETRAINED_ARG="--policy.pretrained_path=${PI05_PRETRAINED_PATH}"
    echo "📦 使用预训练模型: ${PI05_PRETRAINED_PATH}"
else
    echo "📦 从头初始化模型 (不使用预训练权重)"
fi

accelerate launch \
  --multi_gpu \
  --num_processes=${NUM_GPUS} \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
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
  --policy.type=pi05 \
  ${PRETRAINED_ARG} \
  --policy.push_to_hub=false \
  --policy.dtype=${DTYPE} \
  \
  --policy.training_mode=value_function \
  --policy.enable_value_function=true \
  --policy.value_function_variant=${VALUE_FUNCTION_VARIANT} \
  --policy.value_function_share_vision_encoder=${SHARE_VISION_ENCODER} \
  --policy.value_head_dropout=${VALUE_HEAD_DROPOUT} \
  --policy.value_function_lr=${SCALED_LR} \
  --policy.value_function_weight_decay=${VALUE_FUNCTION_WEIGHT_DECAY} \
  --policy.gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
  \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.video_backend="decord" \
  \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --wandb.project="pi05-value-function"

echo ""
echo "=========================================="
echo "🎉 Value Function 训练完成！"
echo "模型保存在: ${OUTPUT_DIR}"
echo "Checkpoints保存在: ${OUTPUT_DIR}/checkpoints/"
echo ""
echo "📌 下一步:"
echo "   1. 使用训练好的 Value Function 计算 Advantage"
echo "   2. 用 Advantage Conditioning 微调策略网络"
echo "=========================================="
