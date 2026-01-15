# 适配层训练问题分析：训练 Loss 正常但推理动作震荡

## 问题描述

训练时 Flow Matching loss 正常（约 -2.3），但推理时动作震荡严重。

## 根本原因分析

### 1. **训练和推理的分布差异** ⚠️ 核心问题

**训练时**：
- 使用 **ground truth action** 和 **随机 timestep**
- Flow Matching loss = MSE(predicted_velocity, target_velocity)
- 只优化**单步** velocity prediction

**推理时**：
- 使用 **预测的 action** 和 **固定的 timestep 序列**（0, 1/N, 2/N, ...）
- 需要**多步迭代去噪**：`x_t = x_t + dt * v_t`
- 每一步都依赖上一步的输出

**关键洞察**：
- 适配层在训练时只看到了 ground truth action 的分布
- 但在推理时，action 的分布是**动态变化的**（每一步都在变化）
- 如果适配层训练不充分，它可能无法处理推理时的分布变化

### 2. **迭代去噪的误差累积**

```python
# 推理时的迭代过程（flow_matching_action_head.py:898）
for t in range(num_steps):
    v_t = denoise_step(x_t, timestep=t, vl_embs=vl_embs, ...)
    x_t = x_t + dt * v_t  # ⚠️ 误差累积！
```

**问题**：
- 训练时只优化单步预测，loss 正常不代表多步迭代稳定
- 每一步的小误差会累积，导致最终动作震荡
- 适配层可能只在单步预测时对齐分布，但在多步迭代时不对齐

### 3. **适配层训练不充分**

虽然 loss 在下降，但适配层可能：
- 没有学到足够的分布对齐（特别是对于迭代去噪的中间状态）
- 残差连接的 `residual_scale` 没有正确学习
- 只对齐了训练数据的分布，没有泛化到推理分布

### 4. **适配层架构问题**

当前使用 MLP adapter + 残差连接：
```python
return x + self.residual_scale * self.adapter(x)
```

**潜在问题**：
- 如果 `residual_scale` 接近 1.0，适配层的影响很小
- 如果适配层输出不稳定，残差连接会放大不稳定性
- 训练时可能学到了"局部最优"，但在推理时失效

## 解决方案

### 方案 1：改进训练策略（推荐）⭐

**核心思想**：让训练过程更接近推理过程

1. **使用迭代去噪损失**：
   - 在训练时也进行多步迭代去噪
   - 计算每一步的 loss，而不是只计算单步 loss
   - 这样可以训练适配层处理迭代过程中的分布变化

2. **使用预测 action 而不是 ground truth**：
   - 在训练时，使用上一步的预测 action 作为下一步的输入
   - 这样可以模拟推理时的分布

3. **增加训练数据量**：
   - 当前使用 100 个样本可能不够
   - 建议增加到 500-1000 个样本

### 方案 2：改进适配层架构

1. **使用更强的适配层**：
   - 增加 MLP 的层数和宽度
   - 使用 LayerNorm 稳定训练

2. **移除残差连接**：
   - 如果残差连接导致不稳定，可以尝试移除
   - 或者使用可学习的门控机制

### 方案 3：使用 Direct Action Loss（如果可能）

**问题**：当前无法使用，因为 `get_action` 有 `@torch.no_grad()` 装饰器

**可能的解决方案**：
- 修改 `get_action` 方法，添加一个 `enable_grad` 参数
- 在训练时启用梯度，在推理时禁用

### 方案 4：诊断和调试

1. **检查适配层权重**：
   ```python
   # 在 eval_merged_groot_on_dataset.py 中添加
   print(f"适配层权重统计:")
   for name, param in adapter.named_parameters():
       print(f"  {name}: mean={param.mean():.6f}, std={param.std():.6f}")
   ```

2. **检查推理时的中间状态**：
   - 打印每一步的 `x_t` 和 `v_t`
   - 检查是否有异常值或 NaN

3. **对比训练和推理的分布**：
   - 在训练时记录 backbone 输出的统计信息
   - 在推理时也记录，对比差异

## 推荐的修复步骤

### 步骤 1：诊断当前状态

1. 检查适配层是否正确加载
2. 检查适配层权重是否合理
3. 检查推理时的中间状态

### 步骤 2：改进训练策略

1. 增加训练数据量（500-1000 样本）
2. 增加训练轮数（100+ epochs）
3. 使用更大的学习率（1e-2 或更大）
4. 添加迭代去噪损失

### 步骤 3：如果仍然失败

1. 尝试不使用适配层的方法（Expert Merging）
2. 或者使用更强的适配层架构
3. 或者修改 `get_action` 方法以支持梯度

## 代码修改建议

### 修改 1：添加迭代去噪损失训练

在 `weight_merge_groot.py` 的 `train_adapter` 方法中：

```python
# 在训练时也进行多步迭代去噪
if use_iterative_loss:
    # 模拟推理过程
    x_t = torch.randn_like(gt_action)  # 初始噪声
    for t in range(num_inference_steps):
        t_cont = t / float(num_inference_steps)
        timestep = int(t_cont * num_timestep_buckets)
        # 预测 velocity
        v_t = self._predict_velocity_with_adapter(x_t, timestep, ...)
        x_t = x_t + dt * v_t
    # 计算最终 loss
    loss = F.mse_loss(x_t, gt_action)
```

### 修改 2：增加训练数据量和轮数

在 `merge_groot_models.sh` 中：

```bash
--num_samples 500 \  # 增加到 500
--adapter_epochs 100 \  # 增加到 100
--adapter_lr 1e-2 \  # 增加到 1e-2
```

### 修改 3：添加诊断代码

在 `eval_merged_groot_on_dataset.py` 的 `get_action_with_adapter` 中：

```python
# 添加诊断
if debug_mode:
    print(f"Backbone features stats: mean={backbone_features.mean():.6f}, std={backbone_features.std():.6f}")
    print(f"Adapted features stats: mean={adapted_features.mean():.6f}, std={adapted_features.std():.6f}")
```

## 总结

### 核心问题

1. **训练和推理的分布差异**：
   - 训练时使用 ground truth action，推理时使用预测的 action
   - 适配层只在训练分布上对齐，无法处理推理时的分布变化

2. **迭代去噪的误差累积**：
   - 训练时只优化单步 velocity prediction
   - 推理时需要多步迭代去噪，每一步的误差会累积
   - 单步 loss 正常不代表多步迭代稳定

3. **适配层训练不充分**：
   - 虽然 loss 在下降，但适配层可能没有学到足够的分布对齐
   - 特别是对于迭代去噪过程中的中间状态

### 解决方案优先级

1. **立即尝试**：增加训练数据量和轮数
   - `--num_samples 500`（增加到 500）
   - `--adapter_epochs 100`（增加到 100）
   - `--adapter_lr 1e-2`（增加到 1e-2）

2. **如果仍然失败**：检查适配层是否正确加载和工作
   - 运行评估脚本，查看诊断输出
   - 检查适配层权重是否合理
   - 检查推理时的中间状态

3. **长期解决方案**：实现迭代去噪损失训练
   - 让训练过程更接近推理过程
   - 训练适配层处理迭代过程中的分布变化

### 诊断步骤

运行评估脚本时，会自动输出诊断信息：
```bash
python scripts/eval_merged_groot_on_dataset.py \
    --model-path /path/to/merged/model \
    --dataset-root /path/to/dataset \
    --episode 0
```

查看输出中的：
- 适配层输入/输出的统计信息
- 特征变化量（|adapted - backbone|）
- 预测动作的统计信息
- 是否有 NaN/Inf 异常值

### 如果问题仍然存在

考虑以下替代方案：
1. 不使用适配层的方法（Expert Merging）
2. 使用更强的适配层架构（更深的 MLP）
3. 修改 `get_action` 方法以支持梯度，使用 Direct Action Loss
