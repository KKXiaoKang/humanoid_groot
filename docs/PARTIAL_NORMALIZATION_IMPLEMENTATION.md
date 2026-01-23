# 分部分归一化实现说明

## 概述

实现了对 absolute eef pose action space 的分部分归一化功能，解决了 6D 旋转表示归一化的问题。

## 实现的功能

### 1. 全局变量控制

在 `src/lerobot/policies/groot/modeling_groot.py` 中定义了全局变量：

```python
GLOBAL_ACTION_SPACE = "Absolute joint"  # 可选值: "Delta eef", "Absolute eef", "Absolute joint"
```

### 2. 自动检测和启用

当 `GLOBAL_ACTION_SPACE` 设置为 `"Delta eef"` 或 `"Absolute eef"` 时，系统会自动启用分部分归一化。

### 3. 分部分归一化策略

对于 absolute eef pose (20维)：

- **Position 部分** (维度 0-2, 9-11): 使用 `MIN_MAX` 归一化
  - `left_eef_pos`: 维度 0-2 (左手位置 x, y, z)
  - `right_eef_pos`: 维度 9-11 (右手位置 x, y, z)

- **6D Rotation 部分** (维度 3-8, 12-17): 使用 `IDENTITY` 归一化（不归一化）
  - `left_eef_rot6d`: 维度 3-8 (左手 6D 旋转表示)
  - `right_eef_rot6d`: 维度 12-17 (右手 6D 旋转表示)

- **Gripper 部分** (维度 18-19): 使用 `MIN_MAX` 归一化
  - `left_gripper`: 维度 18 (左夹爪开合程度)
  - `right_gripper`: 维度 19 (右夹爪开合程度)

## 修改的文件

### 1. `src/lerobot/processor/normalize_processor.py`

- 添加了 `action_space_type` 和 `action_component_indices` 参数到 `_NormalizationMixin`
- 实现了 `_normalize_action_partial()` 方法，支持分部分归一化
- 实现了 `_apply_transform_component()` 辅助方法

### 2. `src/lerobot/policies/groot/processor_groot.py`

- 在 `make_groot_pre_post_processors()` 中读取 `GLOBAL_ACTION_SPACE` 全局变量
- 根据 action space 类型自动设置 `action_component_indices`
- 修改了 `GrootPackInputsStep` 以支持分部分归一化
  - 添加了 `action_space_type` 和 `action_component_indices` 参数
  - 实现了 `_min_max_norm_partial()` 方法
- 修改了 `GrootActionUnpackUnnormalizeStep` 以支持分部分反归一化
  - 添加了 `action_space_type` 和 `action_component_indices` 参数
  - 实现了 `_min_max_unnorm_partial()` 方法

### 3. `src/lerobot/policies/groot/modeling_groot.py`

- 修正了全局变量名：`GLOBAL_ACTION_SAPCE` -> `GLOBAL_ACTION_SPACE`

## 使用方法

### 1. 设置动作模式

在训练或推理前，设置全局变量：

```python
from lerobot.policies.groot.modeling_groot import GLOBAL_ACTION_SPACE

# 设置为 absolute eef pose 模式
GLOBAL_ACTION_SPACE = "Absolute eef"
```

### 2. 确保数据集维度正确

数据集中的 action 和 state 应该是 20 维：
- 3维左手 eef position
- 6维左手 eef 6D 旋转表示
- 3维右手 eef position
- 6维右手 eef 6D 旋转表示
- 1维左夹爪开合程度
- 1维右夹爪开合程度

### 3. 配置检查

确保 `GrootConfig` 中的维度设置正确：

```python
config.max_action_dim = 32  # 可以大于 20，用于 padding
config.max_state_dim = 64   # 可以大于 20，用于 padding
```

实际使用的 action_dim 和 state_dim 会从数据集自动推断。

## 工作原理

1. **训练阶段**：
   - `GrootPackInputsStep` 在 packing 之前对 action 进行分部分归一化
   - Position 和 Gripper 部分被归一化到 [-1, 1]
   - 6D Rotation 部分保持原样（不归一化）

2. **推理阶段**：
   - `GrootActionUnpackUnnormalizeStep` 对模型输出进行分部分反归一化
   - Position 和 Gripper 部分被反归一化回原始范围
   - 6D Rotation 部分保持原样（不反归一化）

## 验证

训练时，系统会打印：

```
✅ Partial normalization enabled for action space: Absolute eef
   Components: ['left_eef_pos', 'left_eef_rot6d', 'right_eef_pos', 'right_eef_rot6d', 'left_gripper', 'right_gripper']
   6D rotation components (left_eef_rot6d, right_eef_rot6d) will use IDENTITY normalization
```

## 注意事项

1. **6D 旋转表示的约束**：
   - 6D 旋转表示必须满足旋转矩阵的几何约束（正交性和归一化）
   - 不归一化 6D 旋转部分可以保持这些约束

2. **向后兼容**：
   - 当 `GLOBAL_ACTION_SPACE` 为 `"Absolute joint"` 或其他值时，使用标准归一化
   - 不会影响现有的 joint space 训练

3. **数据集统计**：
   - 数据集统计信息（mean, std, min, max）仍然需要包含所有 20 维
   - 系统会自动提取每个组件对应的统计信息

## 参考

- 6D 旋转表示论文：https://arxiv.org/pdf/1812.07035
- 问题分析文档：`docs/6D_ROTATION_NORMALIZATION_ISSUE.md`
