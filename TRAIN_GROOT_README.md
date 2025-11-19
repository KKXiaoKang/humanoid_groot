# Groot模型微调训练指南

## 数据集信息

您的数据集位于: `/root/lerobot/lerobot_data/1118_sim_depalletize`

- **摄像头数量**: 4个 (cam_head, cam_chest, cam_left, cam_right)
- **State维度**: 18
- **Action维度**: 18
- **总Episodes**: 24
- **总Frames**: 10223
- **FPS**: 10

## 快速开始

### 1. 单GPU训练

直接运行训练脚本：

```bash
bash train_groot.sh
```

### 2. 多GPU训练（推荐）

如果有多张GPU，可以使用accelerate进行分布式训练：

```bash
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  train_groot.sh
```

或者直接使用accelerate配置：

```bash
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  $(which lerobot-train) \
  --output_dir=./outputs/groot_depalletize \
  --job_name=groot_depalletize \
  --save_checkpoint=true \
  --batch_size=4 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=4 \
  --seed=42 \
  --policy.type=groot \
  --policy.base_model_path="nvidia/GR00T-N1.5-3B" \
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
  --dataset.root=/root/lerobot/lerobot_data \
  --dataset.video_backend="decord" \
  --wandb.enable=false
```

## 参数说明

### 训练参数

- `--batch_size`: 批次大小，根据GPU内存调整（建议4-8）
- `--steps`: 训练总步数（建议20000-50000）
- `--save_freq`: checkpoint保存频率
- `--log_freq`: 日志打印频率
- `--num_workers`: 数据加载器工作进程数

### Groot策略参数

- `--policy.base_model_path`: 基础模型路径（默认: "nvidia/GR00T-N1.5-3B"）
- `--policy.tune_llm`: 是否微调LLM骨干网络（默认: false，节省内存）
- `--policy.tune_visual`: 是否微调视觉编码器（默认: false）
- `--policy.tune_projector`: 是否微调投影层（默认: true）
- `--policy.tune_diffusion_model`: 是否微调扩散模型（默认: true）
- `--policy.use_bf16`: 使用bfloat16精度（默认: true，节省内存）
- `--policy.max_state_dim`: 最大state维度（默认: 64，您的数据是18，所以足够）
- `--policy.max_action_dim`: 最大action维度（默认: 32，您的数据是18，所以足够）
- `--policy.optimizer_lr`: 学习率（默认: 1e-4）
- `--policy.warmup_ratio`: 学习率预热比例（默认: 0.05）

### 数据集参数

- `--dataset.repo_id`: 数据集ID（本地数据集目录名）
- `--dataset.root`: 数据集根目录（包含数据集目录的父目录）
- `--dataset.video_backend`: 视频后端（"decord" 或 "torchvision_av"）

## 内存优化建议

如果遇到GPU内存不足的问题，可以尝试：

1. **减小batch_size**: 从4减到2或1
2. **禁用某些组件的微调**: 
   - `--policy.tune_llm=false` (默认已禁用)
   - `--policy.tune_visual=false` (默认已禁用)
3. **使用gradient checkpointing**: 如果Groot支持的话
4. **减小num_workers**: 从4减到2

## 监控训练

### 使用WandB（可选）

如果想使用Weights & Biases监控训练：

```bash
# 先登录wandb
wandb login

# 在训练命令中添加
--wandb.enable=true \
--wandb.project="groot-depalletize"
```

### 查看日志

训练日志会显示：
- Loss值
- 梯度范数
- 学习率
- 训练速度

## 检查点

训练过程中，checkpoints会保存在：
```
./outputs/groot_depalletize/checkpoints/
```

每个checkpoint包含：
- 模型权重
- 优化器状态
- 学习率调度器状态
- 预处理器和后处理器

## 恢复训练

如果训练中断，可以从checkpoint恢复：

```bash
lerobot-train \
  ... (其他参数) \
  --resume=true \
  --checkpoint_path=./outputs/groot_depalletize/checkpoints/step-XXXXX
```

## 验证训练

训练完成后，可以使用以下命令测试模型：

```bash
# 使用lerobot-eval进行评估（如果有评估环境）
lerobot-eval \
  --policy.path=./outputs/groot_depalletize/checkpoints/step-20000 \
  --dataset.repo_id=1118_sim_depalletize \
  --dataset.root=/root/lerobot/lerobot_data
```

## 常见问题

### 1. Flash Attention错误

如果遇到Flash Attention相关错误，确保已正确安装：

```bash
pip install "flash-attn>=2.5.9,<3.0.0" --no-build-isolation
```

### 2. CUDA内存不足

- 减小batch_size
- 减小num_workers
- 确保使用bf16精度（`--policy.use_bf16=true`）

### 3. 数据集加载错误

确保数据集路径正确：
- `--dataset.root` 应该指向包含数据集目录的父目录
- `--dataset.repo_id` 应该是数据集目录名

## 下一步

训练完成后，您可以：
1. 使用训练好的模型进行推理
2. 在真实机器人上测试
3. 继续微调以改进性能

更多信息请参考: https://huggingface.co/docs/lerobot/groot

