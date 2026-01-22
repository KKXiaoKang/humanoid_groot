#!/bin/bash
# 🧠 Router Network 训练脚本
#
# 训练一个小型神经网络来预测当前输入应该使用哪个专家。
# 这是 MoE (Mixture of Experts) 的标准做法，比启发式路由更可靠。
#
# 使用方式：
#   ./train_router_network.sh                    # 默认使用 GPU 0
#   ./train_router_network.sh 4                  # 使用 GPU 4
#   ./train_router_network.sh 0,1                # 使用 GPU 0 和 1（通过 CUDA_VISIBLE_DEVICES）
#   ./train_router_network.sh --gpu 6,7         # 使用 GPU 6 和 7
#   ./train_router_network.sh -g 0              # 使用 GPU 0

set -e

# ============================================================
# 默认配置（在参数解析之前定义）
# ============================================================
DEFAULT_MODEL_PATH="/home/kangkk/humanoid_groot_base/outputs/0122_merged_groot_mergevla/pretrained_model"

# ============================================================
# 参数解析
# ============================================================
GPU_IDS=""
DEVICE="cuda:0"
MODEL_PATH=""
OUTPUT_PATH=""
EPOCHS=""
BATCH_SIZE=""
LEARNING_RATE=""
SAMPLES_PER_TASK=""
HIDDEN_DIM=""
INTERMEDIATE_DIM=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu|-g)
            GPU_IDS="$2"
            shift 2
            ;;
        --model-path|--model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output-path|--output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate|--learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --samples-per-task|--samples_per_task)
            SAMPLES_PER_TASK="$2"
            shift 2
            ;;
        --hidden-dim|--hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --intermediate-dim|--intermediate_dim)
            INTERMEDIATE_DIM="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --gpu, -g GPU_IDS              指定 GPU ID(s)，例如: 0 或 0,1"
            echo "  --model-path PATH              模型路径（默认: ${DEFAULT_MODEL_PATH}）"
            echo "  --output-path PATH             输出路径（默认: model_path/router_network.pt）"
            echo "  --epochs N                    训练轮数（默认: 20）"
            echo "  --batch-size N                批次大小（默认: 8）"
            echo "  --learning-rate LR            学习率（默认: 1e-3）"
            echo "  --samples-per-task N           每个任务的样本数（默认: 使用所有数据）"
            echo "  --hidden-dim N                隐藏层维度（默认: 2048）"
            echo "  --intermediate-dim N          中间层维度（默认: 256）"
            echo "  --help, -h                    显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                              # 使用默认配置"
            echo "  $0 --gpu 4                     # 使用 GPU 4"
            echo "  $0 --gpu 0,1 --epochs 30        # 使用 GPU 0,1，训练 30 轮"
            echo "  $0 --model-path /path/to/model  # 指定模型路径"
            exit 0
            ;;
        *)
            # 兼容旧的位置参数（单个或多个 GPU ID）
            if [ -z "$GPU_IDS" ]; then
                GPU_IDS="$1"
            fi
            shift
            ;;
    esac
done

# ============================================================
# GPU 配置
# ============================================================
if [ -n "$GPU_IDS" ]; then
    # 检查是否包含逗号（多个 GPU）
    if [[ "$GPU_IDS" == *","* ]]; then
        # 多个 GPU：使用 CUDA_VISIBLE_DEVICES 设置可见的 GPU
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        DEVICE="cuda:0"  # 在可见的 GPU 中，使用第一个作为主设备
        NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
        GPU_INFO="GPU: $GPU_IDS (CUDA_VISIBLE_DEVICES, 将使用 $NUM_GPUS 张 GPU 进行多卡训练)"
        echo "✅ 多卡训练模式: 将使用 $NUM_GPUS 张 GPU"
    else
        # 单个 GPU：使用 CUDA_VISIBLE_DEVICES 来限制只使用这个 GPU
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        DEVICE="cuda:0"  # 在可见的 GPU 中，使用第一个（逻辑上就是指定的 GPU）
        GPU_INFO="GPU: $GPU_IDS (CUDA_VISIBLE_DEVICES, device=cuda:0)"
    fi
else
    # 默认：检测所有可用 GPU，自动启用多卡训练
    # 不设置 CUDA_VISIBLE_DEVICES，让脚本自动检测所有 GPU
    unset CUDA_VISIBLE_DEVICES
    DEVICE="cuda:0"
    # 检测 GPU 数量
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || nvidia-smi --list-gpus | wc -l)
    if [ "$NUM_GPUS" -gt 1 ]; then
        GPU_INFO="GPU: 自动检测到 $NUM_GPUS 张 GPU，将启用多卡训练"
        echo "✅ 自动检测到 $NUM_GPUS 张 GPU，将启用多卡训练模式"
    else
        GPU_INFO="GPU: $DEVICE (单卡模式)"
    fi
fi

# ============================================================
# 训练参数配置（应用默认值，如果未通过命令行指定）
# ============================================================
# 模型路径（默认路径，可通过命令行参数覆盖）
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"

# Router Network 训练参数
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
HIDDEN_DIM="${HIDDEN_DIM:-2048}"
INTERMEDIATE_DIM="${INTERMEDIATE_DIM:-256}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-}"  # 如果为空，使用所有数据；否则指定数量，如 500

# 输出路径（默认保存到模型目录）
OUTPUT_PATH="${OUTPUT_PATH:-}"

# ============================================================
# 显示配置信息
# ============================================================
echo "=========================================="
echo "🧠 Router Network Training"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Output path: ${OUTPUT_PATH:-${MODEL_PATH}/router_network.pt}"
echo "${GPU_INFO}"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
echo ""
echo "Training parameters:"
echo "   Epochs: ${EPOCHS}"
echo "   Batch size: ${BATCH_SIZE}"
echo "   Learning rate: ${LEARNING_RATE}"
echo "   Hidden dim: ${HIDDEN_DIM}"
echo "   Intermediate dim: ${INTERMEDIATE_DIM}"
if [ -n "$SAMPLES_PER_TASK" ]; then
    echo "   Samples per task: ${SAMPLES_PER_TASK}"
else
    echo "   Samples per task: all (使用所有数据)"
fi
echo ""
echo "📋 Dataset configuration (硬编码在 train_router_network.py 中):"
echo "   - narrower: 4 datasets"
echo "   - wider: 4 datasets"
echo "=========================================="
echo ""

# ============================================================
# 检查模型路径是否存在
# ============================================================
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: ${MODEL_PATH}"
    echo "   请检查路径是否正确，或使用 --model-path 参数指定正确的路径"
    exit 1
fi

# ============================================================
# 构建训练命令
# ============================================================
TRAIN_ARGS=(
    --model-path "${MODEL_PATH}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --learning-rate "${LEARNING_RATE}"
    --hidden-dim "${HIDDEN_DIM}"
    --intermediate-dim "${INTERMEDIATE_DIM}"
    --device "${DEVICE}"
)

# 如果指定了输出路径，添加到参数中
if [ -n "$OUTPUT_PATH" ]; then
    TRAIN_ARGS+=( --output-path "${OUTPUT_PATH}" )
fi

# 如果指定了 samples_per_task，添加到参数中
if [ -n "$SAMPLES_PER_TASK" ]; then
    TRAIN_ARGS+=( --samples-per-task "${SAMPLES_PER_TASK}" )
fi

# ============================================================
# 启动训练
# ============================================================
echo "🚀 开始训练 Router Network..."
echo ""

python scripts/train_router_network.py \
    "${TRAIN_ARGS[@]}"

echo ""
echo "=========================================="
echo "✅ Router Network 训练完成！"
echo "=========================================="
echo ""
echo "📁 保存位置: ${OUTPUT_PATH:-${MODEL_PATH}/router_network.pt}"
echo ""
echo "💡 使用训练好的 Router Network 进行推理:"
echo ""
echo "   python scripts/eval_merged_groot_on_dataset.py \\"
echo "       --model-path ${MODEL_PATH} \\"
echo "       --dataset-root /path/to/dataset \\"
echo "       --episode 0 --visualize \\"
echo "       --router-network"
echo ""
echo "   python eval/eval_merged_groot.py \\"
echo "       --model_path ${MODEL_PATH} \\"
echo "       --rtc.enabled=true \\"
echo "       --task=\"Depalletize the box\" \\"
echo "       --use_router_network=true"
echo ""
