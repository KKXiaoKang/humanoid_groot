# State Encoder 影响分析

## 问题描述

对于同一个absolute eef pose模型，无论`use_state_encoder=True`还是`use_state_encoder=False`，推理结果都差不多。这确实很奇怪。

## 可能的原因分析

### 1. **State Encoder输出很小**

从代码看，`CategorySpecificMLP`的初始化：
- `W = 0.02 * torch.randn(...)` - 权重很小（标准差0.02）
- `b = torch.zeros(...)` - bias初始化为0

如果state输入是0，那么：
- `layer1_output = ReLU(W * 0 + b) = ReLU(0) = 0`
- `layer2_output = W2 * 0 + b2 = 0`

所以如果state是0，state_features也会是0，对DiT没有影响。

### 2. **State Normalization的影响**

Processor会对state进行min-max归一化：
- 如果state是0，归一化后可能不是0（取决于stats）
- 但如果state_encoder的权重很小，即使归一化后的state不是0，输出仍然很小

### 3. **State Features在DiT中的贡献很小**

即使state_features不是0，如果它的norm相对于future_tokens和action_features很小，那么：
- 在cross-attention中，state_features作为query的一部分
- 如果state_features的norm很小，它对attention的影响也很小
- 模型主要依赖vision-language特征（vl_embs）进行推理

### 4. **模型训练时State的影响可能很小**

如果模型在训练时：
- 主要依赖RGB图像和语言指令
- State的影响很小（可能state_encoder的梯度更新很小）
- 那么即使推理时state被置零，结果也不会差太多

## 验证方法

我已经添加了诊断代码，会输出：
1. State encoder输出的norm和max值
2. State features在sa_embs中的贡献比例
3. 如果state features贡献 < 1%，会发出警告

## 预期结果

运行推理时，你应该看到类似这样的输出：

```
🔍 [Diagnostic] State encoder output: norm=0.000123, max_abs=0.000456
   ⚠️  State features are very small (norm < 0.1), indicating minimal impact on DiT
🔍 [Diagnostic] sa_embs composition:
   State features norm: 0.000123 (contribution: 0.01%)
   Future tokens norm: 12.345678 (contribution: 45.67%)
   Action features norm: 14.567890 (contribution: 54.32%)
   Total sa_embs norm: 26.913691
   ⚠️  State features contribute < 1% to sa_embs, explaining why results are similar!
```

## 结论

如果state features的贡献 < 1%，那么：
- 即使`use_state_encoder=True`，state的影响也很小
- 即使state被置零，结果也不会差太多
- 这解释了为什么两个模型的推理结果差不多

## 建议

1. **检查state_encoder的权重**：如果权重很小，说明训练时state的影响就很小
2. **对比使用真实state和state=0的结果**：如果差异很小，说明模型主要依赖RGB图像
3. **对于absolute eef pose**：如果模型可以从RGB图像中推断出当前的机器人状态，就不需要state输入
