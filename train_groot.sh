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
OUTPUT_DIR="./outputs/11_19_groot_depalletize"
JOB_NAME="groot_depalletize"

# 数据集配置
# 注意: root 应该直接指向数据集目录（包含 meta/ 和 data/ 的目录）
DATASET_ROOT="/root/lerobot/lerobot_data/1118_sim_depalletize"
DATASET_REPO_ID="1118_sim_depalletize"

# 环境变量设置
# 禁用tokenizers并行警告（在使用多进程数据加载时会出现）
export TOKENIZERS_PARALLELISM=false

# 训练参数 (可根据GPU内存调整)
BATCH_SIZE=8          # 如果GPU内存不足，可以减小到2或1
NUM_STEPS=100000       # 训练步数
SAVE_FREQ=20000        # 每5000步保存一次checkpoint
LOG_FREQ=100          # 每100步打印一次日志
EVAL_FREQ=0           # 设置为0禁用评估（因为没有环境配置）
NUM_WORKERS=8         # 数据加载器工作进程数

# 运行训练
lerobot-train \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME} \
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
  --policy.tune_llm=false \
  --policy.tune_visual=false \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.use_bf16=true \
  --policy.max_state_dim=64 \
  --policy.max_action_dim=32 \
  --policy.optimizer_lr=1e-4 \
  --policy.warmup_ratio=0.05 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
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

