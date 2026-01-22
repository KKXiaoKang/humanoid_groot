# VLA 模型融合问题分析和解决方案

## 问题诊断

### 当前问题
1. **适配层权重几乎为零**：无论学习率如何调整，适配层都无法学习
2. **梯度传播可能有问题**：Flow Matching loss 的梯度可能太弱，无法有效更新适配层
3. **训练和推理分布差异**：适配层在训练时学到的分布对齐，在推理时可能不适用

### 根本原因（基于最新研究）

根据 **MergeVLA (Nov 2025)** 和 **AdaMoE (Oct 2025)** 的研究：

1. **VLA 模型的特殊性**：
   - Action experts 通过 self-attention 产生 **inter-block dependencies**
   - 任务特定信息在层间**纠缠**，难以模块化重组
   - 简单的适配层无法解决这种深层依赖

2. **Flow Matching Loss 的局限性**：
   - 梯度可能太弱，无法有效更新适配层
   - 训练时使用 ground truth，推理时使用预测，分布差异大

## 解决方案

### 方案 1：使用 Expert Merging（推荐）⭐

**优势**：
- 不依赖适配层，直接学习融合系数
- 梯度直接作用于融合系数，传播路径清晰
- 已经在代码中实现，可以直接使用

**使用方法**：
```bash
./merge_groot_models.sh expert_merge
```

**原理**：
- 直接学习 layer-wise 融合系数
- 使用 hidden alignment loss（不需要 action loss）
- 只融合 backbone，action_head 使用 narrower 的

### 方案 2：实现 MergeVLA 风格的 LoRA 适配层

**基于 MergeVLA (Nov 2025) 的方法**：
- 使用稀疏激活的 LoRA adapters + task masks
- 将适配层设计为可组合的模块
- 添加 test-time task router

**实现要点**：
1. 使用 LoRA 而不是全连接层
2. 添加 task-specific masks
3. 使用 cross-attention 而不是 self-attention（如果可能）

### 方案 3：使用 Task Arithmetic（无需训练）

**优势**：
- 无需训练，快速
- 不依赖适配层
- 适合快速实验

**使用方法**：
```bash
./merge_groot_models.sh task_arithmetic
```

## 推荐方案

### 立即尝试：Expert Merging

这是最可靠的方案，因为：
1. ✅ 已经在代码中实现
2. ✅ 不依赖适配层
3. ✅ 梯度传播路径清晰
4. ✅ 适合 VLA 模型

### 如果 Expert Merging 效果不好

考虑实现 MergeVLA 风格的方案：
1. 使用 LoRA 适配层
2. 添加 task routing
3. 使用 cross-attention 架构
