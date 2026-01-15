# MergeVLA 模型崩溃问题分析与解决方案

## 🔍 问题现象

- **Dim_14 (left_claw) MSE: 3639.63** ❌
- **Dim_15 (right_claw) MSE: 3114.78** ❌
- **其他维度 (Dim_0-13, arms) MSE: 0.3-4.2** ✅ 正常

## 🔍 根本原因分析

### 1. ⚠️ Task Masks 没有稀疏化（已修复）

**问题**：
- Task masks 初始化为全 1，经过 sigmoid 后仍然是 100% 激活
- 稀疏 LoRA 实际上变成了普通 LoRA
- 没有实现 MergeVLA 的核心思想（稀疏激活）

**修复**：
- ✅ 已修复：正确初始化 task_masks 以实现稀疏度
- 使用伯努利分布初始化：约 `sparsity` 比例的参数初始化为 5.0（激活），其余为 -5.0（不激活）

### 2. ⚠️ Action Chunk Size 不匹配

**问题**：
- 训练时：`chunk_size=32`, `n_action_steps=32`
- 评估时：`action_chunk_size=16`
- `future_tokens.shape = (64, 1536)`（这是 `num_target_vision_tokens`，与 `action_horizon` 独立）

**影响**：
- 虽然 `future_tokens` 数量不需要匹配 `action_horizon`，但使用不同的 `action_chunk_size` 可能导致：
  - 迭代去噪的步数不匹配
  - Action features 的长度不匹配
  - 模型输出切片位置错误

**解决方案**：
```bash
# 使用与训练时相同的 action_chunk_size
python scripts/eval_merged_groot_on_dataset.py \
    --model-path ./outputs/merged_groot_mergevla/pretrained_model \
    --dataset-root /path/to/dataset \
    --episode 0 \
    --action-chunk-size 32 \  # ⚠️ 改为 32，与训练时一致
    --visualize
```

### 3. ⚠️ 适配层改变了特征分布

**观察**：
- Distribution change: mean_diff=21-24%, std_diff=16.4%
- Feature change: 0.96-0.98（适配层确实在改变特征）

**问题**：
- Flow Matching 的迭代去噪对输入特征分布非常敏感
- 分布变化可能导致迭代去噪过程中的误差累积
- Claw actions 可能对分布变化更敏感（因为它们的值域较小）

**可能的原因**：
1. 适配层改变了特征的统计特性（mean/std）
2. 迭代去噪过程中误差累积
3. Claw actions 的值域较小，更容易受到分布变化的影响

### 4. ⚠️ 稀疏 LoRA 可能不够强

**当前配置**：
- LoRA rank: 16
- Sparsity: 0.5（但之前没有正确实现）
- 参数量: 135,170

**问题**：
- Rank 16 可能不够表达复杂的分布变换
- 稀疏度 0.5 可能太稀疏，无法完全对齐分布

## 💡 解决方案

### 方案 1: 重新训练适配层（使用修复后的稀疏掩码）⭐ 推荐

```bash
# 重新训练，这次稀疏掩码会正确工作
./merge_groot_mergevla.sh 4

# 或者手动运行，增加 LoRA rank 和训练轮数
python scripts/train_weight_merge.py \
    --method mergevla \
    --adapter_type sparse_lora \
    --lora_rank 32 \  # 增加 rank
    --sparsity 0.3 \  # 降低稀疏度（更密集）
    --adapter_epochs 50 \  # 增加训练轮数
    --adapter_lr 1e-3 \
    --num_samples 50 \  # 增加样本数
    --device cuda:4
```

### 方案 2: 使用正确的 action_chunk_size 评估

```bash
# 使用与训练时相同的 action_chunk_size
python scripts/eval_merged_groot_on_dataset.py \
    --model-path ./outputs/merged_groot_mergevla/pretrained_model \
    --dataset-root /path/to/dataset \
    --episode 0 \
    --action-chunk-size 32 \  # ⚠️ 改为 32
    --visualize
```

### 方案 3: 尝试 Expert Merging（不依赖适配层）

如果适配层方法仍然失败，可以尝试 Expert Merging：

```bash
./merge_groot_models.sh expert_merge 4
```

Expert Merging 不依赖适配层，直接学习最优的融合系数，可能更可靠。

### 方案 4: 检查训练数据中 claw actions 的分布

```python
# 检查训练数据中 claw actions 的统计信息
# 如果 claw actions 的分布与 arm actions 差异很大，可能需要特殊处理
```

## 🔧 已修复的问题

1. ✅ **Task Masks 稀疏化初始化**：现在会正确初始化以实现稀疏度
2. ✅ **num_tasks 保存**：merge_config.json 现在会保存 num_tasks

## 📋 下一步行动

1. **重新训练适配层**（使用修复后的代码）
2. **使用正确的 action_chunk_size (32) 评估**
3. **如果仍然失败，尝试 Expert Merging 方法**

## 🔍 诊断命令

```bash
# 检查适配层权重
python -c "
from safetensors.torch import load_file
import glob
from pathlib import Path
import torch

model_path = Path('outputs/merged_groot_mergevla/pretrained_model')
safetensors_files = glob.glob(str(model_path / 'model*.safetensors'))
state_dict = {}
for f in sorted(safetensors_files):
    state_dict.update(load_file(f))

task_masks = state_dict['distribution_adapter.adapter.task_masks']
mask_soft = torch.sigmoid(task_masks)
for t in range(2):
    active_ratio = (mask_soft[t] > 0.5).float().mean().item()
    print(f'Task {t} active ratio: {active_ratio:.2%}')
"
```
