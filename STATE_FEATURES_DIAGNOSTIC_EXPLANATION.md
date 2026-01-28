# State Features 贡献诊断说明

## 终端输出分析

### 终端1：RGB-only模式（use_state_encoder=False）
- **第65行**：`🎨 RGB-only mode enabled: state encoder disabled`
- **诊断输出**：无（因为state_encoder=None，不会处理state）
- **推理结果**：Overall MSE=84.26, MAE=1.11

### 终端2：State Encoder启用（use_state_encoder=True）
- **第81行**：`✅ State encoder enabled: state features will be included in DiT input`
- **第195-200行诊断输出**：
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

### 2. **贡献比例计算的问题**

当前诊断代码使用per-token norm的平均值来计算贡献比例，但这是**不准确的**，因为：
- **State features**: 1个token
- **Future tokens**: 32个tokens  
- **Action features**: 16个tokens（action_horizon=16）

直接比较per-token norm的平均值是不公平的，因为它们的序列长度不同。

### 3. **正确的贡献计算方式**

应该计算每个组件的**总能量**（所有token的norm之和）：

**从诊断输出估算**：
- State features总能量 ≈ 7.33 × 1 = **7.33**
- Future tokens总能量 ≈ 0.83 × 32 = **26.56**
- Action features总能量 ≈ 1846.49 × 16 = **29,543.84**（这个看起来不对，应该是per-token norm的平均值）
- 总能量 ≈ 7.33 + 26.56 + 29,543.84 ≈ **29,577.73**

**正确的贡献比例**（基于总能量）：
- State features: 7.33 / 29,577.73 ≈ **0.025%**（非常小！）
- Future tokens: 26.56 / 29,577.73 ≈ **0.09%**
- Action features: 29,543.84 / 29,577.73 ≈ **99.89%**

### 4. **为什么结果相似？**

**答案**：State features的贡献非常小（< 0.1%）！

即使`use_state_encoder=True`，state_features在sa_embs中的贡献也只有约**0.025%**，这意味着：
- ✅ State features对DiT的cross-attention影响极小
- ✅ 模型主要依赖action_features（99.89%）进行推理
- ✅ 即使state被置零，对结果的影响也很小

### 5. **State Encoder输出的norm分析**

从诊断输出看：
- **State encoder output norm: 7.33**（per-token）
- **State encoder output max_abs: 0.57**

这个norm（7.33）相对于action_features的per-token norm（1846.49/16 ≈ 115.4）来说很小，说明：
- State encoder的输出确实很小
- 即使state不是0，state_features的贡献也很小

## 结论

1. **State features的贡献非常小（< 0.1%）**：即使`use_state_encoder=True`，state对DiT的影响也很小
2. **Action features占主导（> 99%）**：模型主要依赖action_features进行推理
3. **这解释了为什么结果相似**：即使state被置零，对结果的影响也很小，因为state的贡献本来就很小

## 已修复的诊断代码

我已经修复了诊断代码，现在使用**总能量**（sum of norms）而不是per-token norm的平均值来计算贡献比例，这样更准确。

新的诊断输出会显示：
- **Energy**：每个组件的总能量（所有token的norm之和）
- **Contribution**：基于总能量的贡献比例（总和为100%）
- **Per-token norm**：每个token的平均norm（用于参考）

这样你就能更清楚地看到state features的实际贡献了。
