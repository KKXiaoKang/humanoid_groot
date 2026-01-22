# Two-Stage Adapter 方法问题分析

## 问题总结

### 1. 适配层加载失败
- **问题**：训练时 `lora_rank=32`，但评估时默认创建 `rank=16`
- **原因**：`merge_config.json` 中没有保存 `lora_rank`
- **修复**：已在 `TwoStageExpertMerger.save` 中添加 `lora_rank` 保存

### 2. 即使没有适配层，动作仍然震荡
- **现象**：禁用适配层后，Dim_14/15（claw）MSE 仍然非常大（2091, 2954）
- **原因**：**简单插值融合 backbone 导致分布漂移**
  - 融合后的 backbone 输出分布与 narrower 的 action_head 期望的分布不匹配
  - Action head 是在 narrower 的 backbone 输出分布上训练的
  - 融合后的 backbone 输出分布发生了变化，导致 action head 无法正常工作

### 3. 核心问题
**简单插值融合 backbone 不适合 VLA 模型！**

VLA 模型的 backbone 和 action_head 之间有**强耦合**：
- Backbone 输出的特征分布必须与 action_head 训练时的分布一致
- 简单插值会改变特征分布，导致 action_head 崩溃

## 解决方案

### 方案 1：使用 Expert Merging（推荐）⭐

**Expert Merging 不依赖适配层，直接学习最优融合系数**

```bash
./merge_groot_models.sh expert_merge
```

**优势**：
- ✅ 不依赖适配层，更可靠
- ✅ 基于论文方法，有理论保证
- ✅ 直接学习最优融合系数，而不是简单插值

### 方案 2：改进 Two-Stage 方法

**使用 Expert Merging 融合 backbone，而不是简单插值**

当前 Two-Stage 方法的问题：
- Stage 1 使用简单插值（`alpha * narrower + (1-alpha) * wider`）
- 这种方法会导致分布漂移

改进方案：
- Stage 1 使用 Expert Merging 融合 backbone
- Stage 2 训练适配层（可选，如果 Expert Merging 已经足够好）

### 方案 3：使用 Task Arithmetic（快速，无需训练）

```bash
./merge_groot_models.sh task_arithmetic
```

**优势**：
- ✅ 无需训练，快速
- ✅ 不依赖适配层
- ✅ 只融合 backbone，action_head 使用 narrower 的

## 为什么 LoRA 适配层方法可能不可行？

### 理论分析

1. **训练/推理分布不匹配**
   - 训练时：单步 Flow Matching loss
   - 推理时：多步迭代去噪（20步）
   - 适配层在单步训练时学习，但推理时需要多步稳定

2. **梯度传播问题**
   - Flow Matching loss 的梯度可能太弱
   - 即使有梯度，也可能无法有效学习到多步稳定的适配

3. **特征分布变化**
   - 适配层改变了特征分布（mean/std）
   - 在迭代去噪过程中，这种变化会被放大
   - 导致每一步的误差累积，最终崩溃

### 实验证据

- 训练 loss 正常（-2.4），但推理动作震荡
- 即使禁用适配层，动作仍然震荡
- 说明问题在 backbone 融合本身，而不是适配层

## 推荐方案

### ⭐ 首选：Expert Merging

```bash
./merge_groot_models.sh expert_merge
```

**理由**：
1. 不依赖适配层，更可靠
2. 基于论文方法，有理论保证
3. 直接学习最优融合系数

### 备选：Task Arithmetic

```bash
./merge_groot_models.sh task_arithmetic
```

**理由**：
1. 快速，无需训练
2. 不依赖适配层
3. 如果 Expert Merging 效果不好，可以尝试

## 下一步

1. **修复 lora_rank 保存问题**（已完成）
2. **重新训练 Two-Stage Adapter**（如果仍想尝试）
3. **⭐ 推荐：使用 Expert Merging 方法**

## 结论

**简单插值融合 backbone + 适配层的方法可能不适合 VLA 模型。**

建议：
- ✅ 使用 Expert Merging（推荐）
- ✅ 或使用 Task Arithmetic（快速备选）
- ⚠️ Two-Stage Adapter 方法需要进一步改进（使用 Expert Merging 融合 backbone）
