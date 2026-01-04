#!/bin/bash

# ============================================================================
# Groot模型多卡训练脚本
# ============================================================================
# 使用 accelerate launch 启动多卡分布式训练
# 
# 使用方法:
#   bash train_groot_multi_gpu.sh                    # 使用所有可用GPU
#   bash train_groot_multi_gpu.sh --gpu 0,1,2        # 使用指定的GPU
#   bash train_groot_multi_gpu.sh -g 0,1             # 使用GPU 0和1
#
# 重要提示 - 学习率缩放:
#   LeRobot不会自动缩放学习率，多卡训练时需要手动缩放
#   - 8卡训练时，有效batch size = batch_size × 8
#   - 学习率应该相应缩放: lr = base_lr × num_gpus (线性缩放)
#   - 脚本会自动根据GPU数量计算并应用缩放后的学习率
#   - 如果效果不理想，可以尝试平方根缩放 (修改 LR_SCALING_MODE="sqrt")
# ============================================================================

# 设置输出目录
OUTPUT_DIR="./outputs/01_03_h100x4_groot_no_down_sample_multi_arm_head_share_features_cross_attention_mix_3x2"
JOB_NAME="groot_depalletize"

# 数据集配置
# ============================================================================
# 单数据集配置（原方式）:
#   - root 应该直接指向数据集目录（包含 meta/ 和 data/ 的目录）
#   - repo_id 为单个数据集名称
# 
# 多数据集配置（新方式）:
#   - root 应该指向包含所有数据集目录的父目录
#   - repo_id 为逗号分隔的多个数据集名称
#   例如: DATASET_REPO_ID="dataset1,dataset2,dataset3"
#   数据集路径: ${DATASET_ROOT}/dataset1/, ${DATASET_ROOT}/dataset2/, ...
# ============================================================================

# 单数据集配置（当前使用）
# DATASET_ROOT="/home/zhicheng/KangKK/humanoid_groot/lerobot_data/1125_groot_train_data_with_task_filtered"
# DATASET_REPO_ID="1125_groot_train_data_with_task_filtered"

# 多数据集配置（使用两个数据集）
DATASET_ROOT="/home/kangkk/humanoid_groot/lerobot_data/v3_0_dataset"
DATASET_REPO_ID="1223_dense,1225_mix,1229_4322,1215_four,1221_random,3X2"


# GPU选择配置
# 方式1: 通过命令行参数指定 (推荐)
#   用法: bash train_groot_multi_gpu.sh --gpu 0,1,2
#   或:    bash train_groot_multi_gpu.sh -g 0,1
# 方式2: 在脚本中直接设置下面的 GPU_IDS_DEFAULT 变量作为默认值
GPU_IDS_DEFAULT=""

# 解析命令行参数
GPU_IDS="$GPU_IDS_DEFAULT"
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu|-g)
            GPU_IDS="$2"
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
    # 计算GPU数量（通过逗号分隔的ID数量）
    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
    echo "=========================================="
    echo "指定使用GPU: $GPU_IDS"
    echo "GPU数量: $NUM_GPUS"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "=========================================="
else
    # 如果没有指定，使用所有GPU
    # 使用Python来准确检测可见的GPU数量（考虑CUDA_VISIBLE_DEVICES）
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

# 训练参数 (可根据GPU内存调整)
# 注意: batch_size 是每个GPU的batch size，总的有效batch size = batch_size * num_gpus
BATCH_SIZE=16          # 每个GPU的batch size
NUM_STEPS=20000        # 训练步数
SAVE_FREQ=2000         # 每8000步保存一次checkpoint
LOG_FREQ=100           # 每100步打印一次日志
EVAL_FREQ=0            # 设置为0禁用评估
NUM_WORKERS=8          # 数据加载器工作进程数（每个GPU）

# 学习率配置
# ============================================================================
# 重要: LeRobot不会自动缩放学习率，需要手动根据GPU数量缩放
# 
# 学习率缩放策略:
#   - linear: BASE_LR × NUM_GPUS (线性缩放，适用于大batch size)
#   - sqrt: BASE_LR × √NUM_GPUS (平方根缩放，适用于中等batch size)
#   - conservative: BASE_LR × NUM_GPUS^0.4 (更保守，比sqrt更小)
#   - very_conservative: BASE_LR × NUM_GPUS^0.3 (非常保守，适合训练不稳定时)
#   - fixed_scale: BASE_LR × FIXED_SCALE_FACTOR (固定缩放因子，手动控制)
# 
# 对于3卡训练，有效batch size = 16 × 3 = 48
# 如果出现loss震荡，建议使用 conservative 或 very_conservative
# ============================================================================
BASE_LR=1e-4           # 单卡时的基础学习率
LR_SCALING_MODE="conservative"  # "linear", "sqrt", "conservative", "very_conservative", "fixed_scale"
FIXED_SCALE_FACTOR=1.3  # 仅在 LR_SCALING_MODE="fixed_scale" 时使用

# 根据GPU数量自动计算缩放后的学习率
if [ "$LR_SCALING_MODE" = "linear" ]; then
    # 线性缩放: lr = base_lr × num_gpus
    SCALED_LR=$(python3 -c "print('{:.6f}'.format($BASE_LR * $NUM_GPUS))")
    echo "使用线性缩放: lr = ${BASE_LR} × ${NUM_GPUS} = ${SCALED_LR}"
elif [ "$LR_SCALING_MODE" = "sqrt" ]; then
    # 平方根缩放: lr = base_lr × √num_gpus
    SCALED_LR=$(python3 -c "import math; print('{:.6f}'.format($BASE_LR * math.sqrt($NUM_GPUS)))")
    echo "使用平方根缩放: lr = ${BASE_LR} × √${NUM_GPUS} = ${SCALED_LR}"
elif [ "$LR_SCALING_MODE" = "conservative" ]; then
    # 保守缩放: lr = base_lr × num_gpus^0.4 (比sqrt更保守)
    SCALED_LR=$(python3 -c "import math; print('{:.6f}'.format($BASE_LR * math.pow($NUM_GPUS, 0.4)))")
    echo "使用保守缩放: lr = ${BASE_LR} × ${NUM_GPUS}^0.4 = ${SCALED_LR}"
elif [ "$LR_SCALING_MODE" = "very_conservative" ]; then
    # 非常保守缩放: lr = base_lr × num_gpus^0.3 (非常保守，适合训练不稳定时)
    SCALED_LR=$(python3 -c "import math; print('{:.6f}'.format($BASE_LR * math.pow($NUM_GPUS, 0.3)))")
    echo "使用非常保守缩放: lr = ${BASE_LR} × ${NUM_GPUS}^0.3 = ${SCALED_LR}"
elif [ "$LR_SCALING_MODE" = "fixed_scale" ]; then
    # 固定缩放: lr = base_lr × fixed_scale_factor
    SCALED_LR=$(python3 -c "print('{:.6f}'.format($BASE_LR * $FIXED_SCALE_FACTOR))")
    echo "使用固定缩放: lr = ${BASE_LR} × ${FIXED_SCALE_FACTOR} = ${SCALED_LR}"
else
    echo "错误: 未知的学习率缩放模式: ${LR_SCALING_MODE}"
    echo "支持的模式: linear, sqrt, conservative, very_conservative, fixed_scale"
    exit 1
fi

# 是否从checkpoint继续训练
RESUME=false

# 使用 accelerate launch 启动多卡训练
# --multi_gpu: 启用多GPU训练（必需）
# --num_processes: 进程数量（等于GPU数量）
# --mixed_precision=bf16: 使用bf16混合精度训练（匹配policy.use_bf16=true）
# 注意: 使用 $(which lerobot-train) 确保使用正确的命令路径
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
  --policy.optimizer_lr=${SCALED_LR} \
  --policy.warmup_ratio=0.10 \
  --policy.chunk_size=32 \
  --policy.n_action_steps=32 \
  \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.video_backend="decord" \
  \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --wandb.project="groot-depalletize"

echo ""
echo "=========================================="
echo "训练完成！"
echo "模型保存在: ${OUTPUT_DIR}"
echo "Checkpoints保存在: ${OUTPUT_DIR}/checkpoints/"
echo "有效batch size: ${BATCH_SIZE} x ${NUM_GPUS} = $((BATCH_SIZE * NUM_GPUS))"
echo "学习率配置: 基础LR=${BASE_LR}, 缩放后LR=${SCALED_LR} (${LR_SCALING_MODE}缩放)"
echo "=========================================="

