# 6D 旋转表示归一化问题分析

## 问题描述

在使用 absolute eef pose 作为 action space 时，action 的格式为：
- 3维左手 eef position (x, y, z)
- 6维左手 eef 6D 旋转表示 [R11, R21, R31, R12, R22, R32]
- 3维右手 eef position (x, y, z)
- 6维右手 eef 6D 旋转表示 [R11, R21, R31, R12, R22, R32]
- 1维左夹爪开合程度
- 1维右夹爪开合程度

**总计 20 维**

## 核心问题

### 6D 旋转表示的几何约束

6D 旋转表示是旋转矩阵的前两列：
```
R = [R11  R12  R13]
    [R21  R22  R23]
    [R31  R32  R33]
```

6D 向量 = [R11, R21, R31, R12, R22, R32]

这6个值必须满足：
1. **正交性约束**：第一列和第二列必须正交
2. **归一化约束**：每列必须是单位向量（或接近单位向量）
3. **右手坐标系约束**：第三列 = 第一列 × 第二列

### 当前归一化方式的问题

当前代码使用 `MEAN_STD` 或 `MIN_MAX` 归一化，对**每个维度独立**进行归一化：

```python
# 在 normalize_processor.py 中
if norm_mode == NormalizationMode.MEAN_STD:
    mean, std = stats["mean"], stats["std"]  # shape: (20,)
    return (tensor - mean) / denom  # 每个维度独立归一化
```

**问题**：
1. 对 6D 旋转的6个维度分别归一化，会破坏它们之间的几何关系
2. 归一化后的向量可能不再满足正交性和归一化约束
3. 在 unnormalize 后，6D 向量可能无法重构出有效的旋转矩阵
4. 这会导致 IK 求解失败或产生不合理的姿态

## 影响分析

### 训练阶段
- 模型学习的是归一化后的 6D 向量
- 损失函数基于归一化后的值计算
- 模型可能学习到不满足旋转矩阵约束的表示

### 推理阶段
- 模型输出归一化后的 6D 向量
- Unnormalize 后，6D 向量可能无法重构有效旋转矩阵
- 即使能重构，旋转矩阵可能不满足正交性和归一化约束
- IK 求解可能失败或产生不合理结果

## 解决方案

### 方案 1：对 6D 旋转使用 IDENTITY 归一化（推荐）

**优点**：
- 简单直接，不需要修改代码
- 保持 6D 旋转的几何约束
- 只对 position 和 gripper 归一化

**实现方式**：
需要修改归一化逻辑，对 action 的不同部分使用不同的归一化方式：
- Position (维度 0-2, 9-11): 使用 MEAN_STD 或 MIN_MAX
- 6D Rotation (维度 3-8, 12-17): 使用 IDENTITY（不归一化）
- Gripper (维度 18-19): 使用 MEAN_STD 或 MIN_MAX

### 方案 2：实现特殊的旋转归一化

**思路**：
- 将 6D 旋转转换为四元数或欧拉角
- 对四元数/欧拉角进行归一化
- 转换回 6D 表示

**缺点**：
- 实现复杂
- 可能引入额外的数值误差
- 需要处理奇点问题

### 方案 3：使用 Gram-Schmidt 正交化后归一化

**思路**：
- 在 unnormalize 后，使用 Gram-Schmidt 正交化确保旋转矩阵有效
- 在 normalize 前，先对 6D 向量进行预处理

**缺点**：
- 需要修改 preprocessor 和 postprocessor
- 可能改变数据的分布

## 推荐实现（方案 1）

### 修改建议

1. **创建自定义的归一化处理器**，能够对 action 的不同部分使用不同的归一化方式

2. **或者修改 `normalize_processor.py`**，添加对 6D 旋转的特殊处理

3. **在配置中指定 action 的组成部分**：
   ```python
   action_components = {
       "left_eef_pos": (0, 3),
       "left_eef_rot6d": (3, 9),
       "right_eef_pos": (9, 12),
       "right_eef_rot6d": (12, 18),
       "left_gripper": (18, 19),
       "right_gripper": (19, 20),
   }
   ```

4. **对不同部分应用不同的归一化**：
   - Position 和 Gripper: MEAN_STD 或 MIN_MAX
   - 6D Rotation: IDENTITY

## 验证方法

1. **检查归一化后的 6D 向量**：
   - 计算第一列和第二列的点积（应该接近0）
   - 计算每列的范数（应该接近1）

2. **检查 unnormalize 后的旋转矩阵**：
   - 使用 `reconstruct_rotation_matrix_6d` 重构旋转矩阵
   - 检查是否满足正交性：R @ R.T ≈ I
   - 检查行列式：det(R) ≈ 1

3. **检查 IK 求解成功率**：
   - 使用 `verify_ik_fk_consistency.py` 验证
   - 检查 IK 求解的成功率
   - 检查 FK 验证的误差

## 当前状态

**⚠️ 警告**：当前代码直接对整个 action tensor 进行归一化，**会对 6D 旋转表示造成问题**。

**建议**：在训练前，先实现方案 1，确保 6D 旋转表示不被归一化，或者使用特殊的归一化方法。

## 参考

- 6D 旋转表示论文：https://arxiv.org/pdf/1812.07035
- 当前归一化实现：`src/lerobot/processor/normalize_processor.py`
- IK/FK 验证脚本：`eval/IK_eef_eval/verify_ik_fk_consistency.py`
