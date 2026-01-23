# Absolute EEF Action Space 训练指南

## 快速开始

### 1. 设置动作空间类型

**方法 1: 在训练脚本中设置（推荐）**

在 `train_groot_multi_gpu.sh` 中添加：

```bash
--policy.action_space_type="Absolute eef"
```

**方法 2: 在 Python 代码中设置**

```python
from lerobot.policies.groot.configuration_groot import GrootConfig

config = GrootConfig(
    action_space_type="Absolute eef",
    # ... 其他配置
)
```

### 2. 执行训练

```bash
./train_groot_multi_gpu.sh --gpu 4,5,6,7
```

## 自动检测和配置

系统会自动：

1. **读取配置**：`processor_groot.py` 会从 `GrootConfig` 读取 `action_space_type`
2. **自动配置维度**：根据 `action_space_type` 自动设置 multi-head 维度
   - `Absolute eef`: left_arm=9D, right_arm=9D, claw=2D (总共 20D)
   - `Absolute joint`: left_arm=7D, right_arm=7D, claw=2D (总共 16D)
3. **启用分部分归一化**：当 `action_space_type` 为 `"Delta eef"` 或 `"Absolute eef"` 时，自动启用
4. **从数据集推断维度**：action_dim 和 state_dim 会从数据集的 `meta.features` 自动推断
5. **应用分部分归一化**：
   - Position (0-2, 9-11): MIN_MAX 归一化
   - 6D Rotation (3-8, 12-17): IDENTITY（不归一化）
   - Gripper (18-19): MIN_MAX 归一化

## 训练日志验证

训练开始时会打印：

```
🎯 Auto-configured for Absolute eef action space:
   left_arm=9D, right_arm=9D, claw=2D
✅ Split arm heads enabled: left_arm(9D) + right_arm(9D) + claw(2D) = 20D
✅ Partial normalization enabled for action space: Absolute eef
   Components: ['left_eef_pos', 'left_eef_rot6d', 'right_eef_pos', 'right_eef_rot6d', 'left_gripper', 'right_gripper']
   6D rotation components (left_eef_rot6d, right_eef_rot6d) will use IDENTITY normalization
```

如果看到这些输出，说明：
1. ✅ 动作空间类型已正确设置
2. ✅ Multi-head 维度已自动配置
3. ✅ 分部分归一化已成功启用

## 数据集要求

### 维度要求

数据集中的 action 和 state 必须是 **20 维**（absolute eef pose 格式）：

```
action/state 格式 (20维):
- 维度 0-2:   左手 eef position (x, y, z)
- 维度 3-8:   左手 eef 6D 旋转表示 [R11, R21, R31, R12, R22, R32]
- 维度 9-11:  右手 eef position (x, y, z)
- 维度 12-17: 右手 eef 6D 旋转表示 [R11, R21, R31, R12, R22, R32]
- 维度 18:    左夹爪开合程度
- 维度 19:    右夹爪开合程度
```

### 配置要求

训练脚本中的配置：

```bash
--policy.max_state_dim=64   # 可以大于 20，用于 padding
--policy.max_action_dim=32  # 可以大于 20，用于 padding
```

这些是**最大值**，用于 padding。实际的 action_dim 和 state_dim 会从数据集自动推断（应该是 20）。

## 验证步骤

### 1. 检查数据集维度

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id="your_dataset", root="/path/to/dataset")
print(f"Action shape: {dataset.meta.features['action']['shape']}")
print(f"State shape: {dataset.meta.features['observation.state']['shape']}")
```

应该输出：
```
Action shape: (20,)
State shape: (20,)
```

### 2. 检查训练日志

训练开始时会打印：
- ✅ Partial normalization enabled for action space: Absolute eef
- 组件列表和归一化策略

### 3. 验证归一化行为

可以在训练过程中检查：
- Position 和 Gripper 部分被归一化到 [-1, 1]
- 6D Rotation 部分保持原样（不归一化）

## 常见问题

### Q: 如果数据集不是 20 维怎么办？

A: 需要先转换数据集。使用 `cvt_bag2lerobot_depalletizer_task_eef_pinocchio.py` 脚本，并确保：
- 使用 `--urdf-path` 指定 URDF 文件
- 不使用 `--no-fk` 标志（启用 FK 计算）
- 数据集会生成 20 维的 absolute eef pose 格式

### Q: max_action_dim 和 max_state_dim 需要修改吗？

A: 不需要。这些是最大值，用于 padding。只要它们 >= 20 就可以。训练脚本中设置的 32 和 64 已经足够。

### Q: 如何确认分部分归一化正在工作？

A: 查看训练日志，应该看到：
```
🎯 Auto-configured for Absolute eef action space:
   left_arm=9D, right_arm=9D, claw=2D
✅ Partial normalization enabled for action space: Absolute eef
```

如果没有看到这个输出，检查：
1. `action_space_type` 是否正确设置为 `"Absolute eef"`（在训练脚本或配置中）
2. 配置是否正确传递到 `GrootConfig`

### Q: 训练时出现维度错误怎么办？

A: 检查：
1. 数据集维度是否为 20
2. `max_action_dim` 和 `max_state_dim` 是否 >= 20
3. 数据集统计信息是否包含所有 20 维

## 参考

- 问题分析：`docs/6D_ROTATION_NORMALIZATION_ISSUE.md`
- 实现说明：`docs/PARTIAL_NORMALIZATION_IMPLEMENTATION.md`
