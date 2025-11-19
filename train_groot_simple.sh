#!/bin/bash

# 简单的Groot训练命令（单行版本）
# 直接复制下面的命令到终端运行，或执行此脚本

lerobot-train \
  --output_dir=./outputs/groot_depalletize \
  --job_name=groot_depalletize \
  --save_checkpoint=true \
  --batch_size=4 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=100 \
  --eval_freq=0 \
  --num_workers=4 \
  --seed=42 \
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
  --dataset.repo_id=1118_sim_depalletize \
  --dataset.root=/root/lerobot/lerobot_data/1118_sim_depalletize \
  --dataset.video_backend="decord" \
  --wandb.enable=false

