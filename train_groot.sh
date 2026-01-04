#!/bin/bash

# ============================================================================
# Groot模型微调训练脚本
# ============================================================================
# 使用本地数据集: /root/lerobot/lerobot_data/1118_sim_depalletize
# 
# 数据集信息:
# - 4个摄像头: cam_head, cam_chest, cam_left, cam_right
# - State维度: 18
# - Action维度: 18
# - 总episodes: 24
# - 总frames: 10223
# ============================================================================

# 设置输出目录
OUTPUT_DIR="./outputs/12_29_groot_no_down_sample_pick_1203_h100_one_gpu_mix_four_4322"
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
DATASET_REPO_ID="1223_dense,1225_mix,1229_4322,1215_four,1221_random"

# GPU选择配置
# 方式1: 通过命令行参数指定 (推荐，会覆盖下面的默认值)
#   用法: bash train_groot.sh --gpu 0,1,2
#   或:    bash train_groot.sh -g 0
# 方式2: 在脚本中直接设置下面的 GPU_IDS_DEFAULT 变量作为默认值
#   例如: GPU_IDS_DEFAULT="0" 使用第0张卡
#        GPU_IDS_DEFAULT="0,1" 使用第0和第1张卡
#        GPU_IDS_DEFAULT="0,1,2,3" 使用前4张卡
# 如果不设置，将使用所有可用的GPU

# 默认GPU设置（如果不想通过命令行指定，可以在这里设置）
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
            # 忽略其他参数（这些参数会被lerobot-train处理）
            shift
            ;;
    esac
done

# 环境变量设置
# 禁用tokenizers并行警告（在使用多进程数据加载时会出现）
export TOKENIZERS_PARALLELISM=false

# 设置CUDA可见设备
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "=========================================="
    echo "指定使用GPU: $GPU_IDS"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "=========================================="
else
    echo "=========================================="
    echo "使用所有可用GPU"
    echo "=========================================="
fi

# 训练参数 (可根据GPU内存调整)
BATCH_SIZE=20          # 如果GPU内存不足，可以减小到2或1
NUM_STEPS=80000       # 训练步数
SAVE_FREQ=8000        # 每2000步保存一次checkpoint
LOG_FREQ=100          # 每100步打印一次日志
EVAL_FREQ=0           # 设置为0禁用评估（因为没有环境配置）
NUM_WORKERS=8         # 数据加载器工作进程数

# ============================================================================
# 微调模式说明:
# ============================================================================
# 当前默认: 只微调 Flow-Matching Action Head (projector + diffusion model)
#   - tune_llm=false: 冻结 LLM backbone
#   - tune_visual=false: 冻结视觉 tower
#   - tune_projector=true: 训练 projector
#   - tune_diffusion_model=true: 训练 diffusion model
#
# 全量微调选项（在下面的命令中修改）:
#   - 将 --policy.tune_llm=false 改为 --policy.tune_llm=true
#   - 将 --policy.tune_visual=false 改为 --policy.tune_visual=true
#   - 全量微调时建议使用更小的学习率 (将 optimizer_lr=1e-4 改为 5e-5)
#   - 全量微调需要更多 GPU 内存，建议减小 batch_size 到 4-8
# ============================================================================

# 是否从checkpoint继续训练
# 设置为 true 时，会从 ${OUTPUT_DIR}/checkpoints/last 加载最新的checkpoint并继续训练
# 设置为 false 时，从头开始训练（如果checkpoint已存在会报错，防止意外覆盖）
RESUME=false

# 运行训练
lerobot-train \
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
  --policy.optimizer_lr=1e-4 \
  --policy.warmup_ratio=0.05 \
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
echo "=========================================="

