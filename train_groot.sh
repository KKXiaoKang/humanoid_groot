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
# 使用 v3.0 格式数据集，包含 task 字段
DATASET_ROOT="/home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1125_groot_train_data_with_task"
DATASET_REPO_ID="1125_groot_train_data_with_task"

# 环境变量设置
# 禁用tokenizers并行警告（在使用多进程数据加载时会出现）
export TOKENIZERS_PARALLELISM=false

# 训练参数 (可根据GPU内存调整)
BATCH_SIZE=16          # 如果GPU内存不足，可以减小到2或1
NUM_STEPS=40000       # 训练步数
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

