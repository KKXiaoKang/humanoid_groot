# 适配层训练失败导致动作坍塌 - 问题诊断和修复

## 问题现象

**训练时**：
- Loss 正常下降（最终 -2.42）
- 所有批次成功训练（850/850）
- `residual_scale` 有调整（1.063939）

**推理时**：
- **动作完全坍塌**：夹爪维度 MSE 达到 2559 和 3122
- 关节轨迹剧烈震荡
- 整体 MSE 355.98（灾难性失败）

## 根本原因

### 1. **适配层权重几乎为零** ⚠️ 核心问题

从训练日志可以看到：
```
adapter.0.weight: Mean: -0.000044, Std: 0.010915  # 几乎为零！
adapter.3.weight: Mean: 0.000009, Std: 0.009807    # 几乎为零！
```

**这说明适配层根本没有学习到任何有用的变换！**

### 2. **学习率衰减太激进**

- 初始学习率：1e-3
- 使用 CosineAnnealing 衰减
- **最终学习率降到 1e-6**（最后几个 epoch）
- 学习率太低，导致适配层无法更新权重

### 3. **Flow Matching Loss 梯度太弱**

- Flow Matching Loss 对适配层的梯度可能太弱
- 即使有梯度，学习率太低也无法有效更新

### 4. **残差连接掩盖了问题**

- 残差连接：`output = x + residual_scale * adapter(x)`
- 如果 `adapter(x)` 接近零，输出 ≈ x（看起来正常）
- 但实际推理时，分布不匹配导致崩溃

## 修复方案

### 修复 1：使用固定学习率（已实现）⭐

**问题**：CosineAnnealing 让学习率降得太低

**解决方案**：
```python
# 在 weight_merge_groot.py 中
use_constant_lr = True  # 使用固定学习率
if use_constant_lr:
    scheduler = None  # 不使用 scheduler
```

### 修复 2：增大学习率（已更新脚本）

**问题**：1e-3 的学习率可能不够

**解决方案**：
```bash
# 在 merge_groot_models.sh 中
--adapter_lr 1e-2 \      # 从 1e-3 增加到 1e-2
--adapter_epochs 100 \   # 从 50 增加到 100
--num_samples 500 \     # 从 100 增加到 500
```

### 修复 3：添加实时诊断（已实现）

**问题**：无法及时发现适配层没有学习

**解决方案**：
- 每个 epoch 检查适配层权重是否在学习
- 如果权重接近零，立即警告
- 训练结束后详细诊断

## 重新训练步骤

### 步骤 1：使用修复后的代码重新训练

```bash
./merge_groot_models.sh two_stage_adapter
```

**关键改进**：
- ✅ 使用固定学习率（不会衰减到 1e-6）
- ✅ 使用更大的学习率（1e-2）
- ✅ 更多训练数据（500 样本）
- ✅ 更多训练轮数（100 epochs）
- ✅ 实时诊断适配层学习状态

### 步骤 2：检查训练日志

**关键指标**：
1. **适配层权重是否在学习**：
   ```
   🔍 Adapter learning check (Epoch X):
      Layer 0 weight abs mean: X.XXXXXX  # 应该 > 0.05
      Layer 3 weight abs mean: X.XXXXXX  # 应该 > 0.05
   ```

2. **如果看到警告**：
   ```
   ⚠️ CRITICAL: Adapter is NOT learning! Weights are almost zero!
   ```
   说明学习率还是太小，需要进一步增大（1e-1 或更大）

3. **训练结束后的诊断**：
   ```
   adapter.0.weight:
      Weight abs mean: X.XXXXXX  # 应该 > 0.05
   ```
   如果 < 0.01，说明训练失败

### 步骤 3：评估模型

```bash
python scripts/eval_merged_groot_on_dataset.py \
    --model-path /path/to/merged/model \
    --dataset-root /path/to/dataset \
    --episode 0
```

**期望结果**：
- 夹爪维度 MSE < 10（而不是 2559）
- 整体 MSE < 5（而不是 355）
- 动作轨迹平滑，无剧烈震荡

## 如果仍然失败

### 方案 A：进一步增大学习率

```bash
--adapter_lr 1e-1 \  # 增加到 0.1
```

### 方案 B：使用更强的适配层架构

修改 `weight_merge_groot.py` 中的适配层：
- 增加层数（3-4 层 MLP）
- 增加宽度（hidden_size * 3 或 * 4）
- 移除 Dropout（可能阻碍学习）

### 方案 C：尝试不使用适配层的方法

如果适配层方法始终失败，可以尝试：
1. **Expert Merging**：直接学习融合系数
2. **Task Arithmetic**：无需训练，直接融合

## 总结

**核心问题**：学习率衰减太激进 + Flow Matching Loss 梯度弱 → 适配层无法学习

**解决方案**：
1. ✅ 使用固定学习率（不衰减）
2. ✅ 使用更大的学习率（1e-2）
3. ✅ 更多训练数据和轮数
4. ✅ 实时诊断适配层学习状态

**关键指标**：适配层权重绝对值均值应该 > 0.05，否则训练失败。
