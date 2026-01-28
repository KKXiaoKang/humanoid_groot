# State Features 贡献分析报告

## 诊断输出分析

### 终端输出1：RGB-only模式（use_state_encoder=False）
- **模型配置**：`🎨 RGB-only mode enabled: state encoder disabled`
- **诊断输出**：无（因为state_encoder=None，不会处理state）
- **推理结果**：Overall MSE=84.26, MAE=1.11

### 终端输出2：State Encoder启用（use_state_encoder=True）
- **模型配置**：`✅ State encoder enabled: state features will be included in DiT input`
- **诊断输出**（第195-200行）：
  ```
  🔍 [Diagnostic] State encoder output (inference): norm=7.328076, max_abs=0.574106
  🔍 [Diagnostic] sa_embs composition (inference):
     State features norm: 7.328076 (contribution: 2.01%)
     Future tokens norm: 0.826149 (contribution: 0.23%)
     Action features norm: 1846.487061 (contribution: 505.22%)
     Total sa_embs norm: 365.481415
  ```
- **推理结果**：Overall MSE=124.61, MAE=1.63

## 关键发现

### 1. **State Features的贡献很小（约2%）**

从诊断输出看：
- **State features norm: 7.33**，贡献约**2.01%**
- **Future tokens norm: 0.83**，贡献约**0.23%**
- **Action features norm: 1846.49**，贡献约**505.22%**（这个计算有问题，见下文）

**注意**：贡献比例计算有问题！Action features的贡献显示为505.22%，这明显不对。这是因为：
- State features: 1个token
- Future tokens: 32个tokens
- Action features: T个tokens（T=16，action_horizon）

直接比较per-token norm的平均值是不公平的，因为它们的序列长度不同。

### 2. **正确的贡献计算方式**

应该计算每个组件的**总能量**（所有token的norm之和），而不是per-token norm的平均值：

- **State features总能量** = 7.33 × 1 = 7.33
- **Future tokens总能量** = 0.83 × 32 ≈ 26.56
- **Action features总能量** = 1846.49 × 16 ≈ 29,543.84
- **总能量** ≈ 7.33 + 26.56 + 29,543.84 ≈ 29,577.73

**正确的贡献比例**：
- State features: 7.33 / 29,577.73 ≈ **0.025%**（非常小！）
- Future tokens: 26.56 / 29,577.73 ≈ **0.09%**
- Action features: 29,543.84 / 29,577.73 ≈ **99.89%**

### 3. **为什么结果相似？**

**答案**：State features的贡献非常小（< 0.1%）！

即使`use_state_encoder=True`，state_features在sa_embs中的贡献也只有约0.025%，这意味着：
- State features对DiT的cross-attention影响极小
- 模型主要依赖action_features（99.89%）进行推理
- 即使state被置零，对结果的影响也很小

### 4. **State Encoder输出的norm分析**

从诊断输出看：
- **State encoder output norm: 7.33**
- **State encoder output max_abs: 0.57**

这个norm（7.33）相对于action_features的norm（1846.49）来说很小，但考虑到state只有1个token，而action有16个tokens，state的per-token norm（7.33）实际上比action的per-token norm（1846.49/16 ≈ 115.4）小得多。

## 结论

1. **State features的贡献非常小（< 0.1%）**：即使`use_state_encoder=True`，state对DiT的影响也很小
2. **Action features占主导（> 99%）**：模型主要依赖action_features进行推理
3. **这解释了为什么结果相似**：即使state被置零，对结果的影响也很小，因为state的贡献本来就很小

## 建议

1. **修复诊断代码**：使用总能量（sum of norms）而不是per-token norm的平均值来计算贡献比例
2. **检查训练时的state影响**：如果训练时state的影响就很小，那么推理时state被置零也不会差太多
3. **对于absolute eef pose**：如果模型可以从RGB图像中推断出当前的机器人状态，就不需要state输入
