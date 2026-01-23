# Absolute EEF Action Head 配置说明

## 概述

当使用 **Absolute EEF** action space (20D) 时，DiT (Diffusion Transformer) 部分**不需要特殊处理**，但需要正确配置 **multi-head action heads** 的维度。

## DiT 部分

### ✅ 不需要特殊处理

DiT 本身只是预测 velocity（速度场），不直接处理归一化：

1. **归一化已在 processor 中处理**：
   - `GrootPackInputsStep` 中已经应用了分部分归一化
   - 6D rotation 部分使用 IDENTITY（不归一化）
   - Position 和 Gripper 部分使用 MIN_MAX 归一化

2. **DiT 的输入输出**：
   - 输入：经过 `action_encoder` 编码的加噪轨迹特征
   - 输出：预测的 velocity（用于 Flow Matching 去噪）
   - DiT 不关心归一化，因为它处理的是已经归一化的特征

## Multi-Head Action Heads 配置

### 自动配置

系统会根据 `GLOBAL_ACTION_SPACE` **自动配置** multi-head 维度：

#### Absolute EEF (20D)

当 `GLOBAL_ACTION_SPACE = "Absolute eef"` 时，自动设置：

```python
action_left_arm_dim = 9   # 3D pos (0-2) + 6D rot (3-8)
action_right_arm_dim = 9  # 3D pos (9-11) + 6D rot (12-17)
action_claw_dim = 2       # left gripper (18) + right gripper (19)
total = 20D
```

#### Absolute Joint (16D) - 默认

当 `GLOBAL_ACTION_SPACE = "Absolute joint"` 时，使用默认值：

```python
action_left_arm_dim = 7   # 7 joints (0-6)
action_right_arm_dim = 7  # 7 joints (7-13)
action_claw_dim = 2       # left gripper (14) + right gripper (15)
total = 16D
```

### 实现位置

自动配置在 `src/lerobot/policies/groot/groot_n1.py` 中实现：

```python
# Auto-configure dimensions based on GLOBAL_ACTION_SPACE
if GLOBAL_ACTION_SPACE in ["Delta eef", "Absolute eef"]:
    # For eef action space (20D)
    if "action_left_arm_dim" not in action_head_cfg_dict:
        action_head_cfg_dict["action_left_arm_dim"] = 9
    if "action_right_arm_dim" not in action_head_cfg_dict:
        action_head_cfg_dict["action_right_arm_dim"] = 9
    if "action_claw_dim" not in action_head_cfg_dict:
        action_head_cfg_dict["action_claw_dim"] = 2
```

### 手动覆盖

如果需要手动指定维度（不推荐），可以在训练脚本或配置中设置：

```bash
--policy.action_head_cfg.action_left_arm_dim=9
--policy.action_head_cfg.action_right_arm_dim=9
--policy.action_head_cfg.action_claw_dim=2
```

## 训练日志验证

训练开始时会打印：

```
🎯 Auto-configured for Absolute eef action space:
   left_arm=9D (3D pos + 6D rot), right_arm=9D (3D pos + 6D rot), claw=2D
✅ Split arm heads enabled: left_arm(9D) + right_arm(9D) + claw(2D) = 20D
```

## 维度分割逻辑

在 `flow_matching_action_head.py` 的 `forward` 方法中，velocity 会被正确分割：

```python
# For absolute eef (20D):
velocity_left_arm = velocity[:, :, :9]      # indices 0-8 (3D pos + 6D rot)
velocity_right_arm = velocity[:, :, 9:18]   # indices 9-17 (3D pos + 6D rot)
velocity_claw = velocity[:, :, 18:]         # indices 18-19 (left + right gripper)
```

## 总结

### ✅ 需要做的

1. **设置 `GLOBAL_ACTION_SPACE = "Absolute eef"`**（在 `modeling_groot.py` 中）
2. **确保数据集是 20D 格式**（absolute eef pose）
3. **系统会自动配置 multi-head 维度**

### ❌ 不需要做的

1. **不需要修改 DiT 代码**（DiT 不关心归一化）
2. **不需要手动设置 action_head 维度**（会自动配置）
3. **不需要修改 loss 计算**（维度分割逻辑已通用化）

## 参考

- 问题分析：`docs/6D_ROTATION_NORMALIZATION_ISSUE.md`
- 实现说明：`docs/PARTIAL_NORMALIZATION_IMPLEMENTATION.md`
- 训练指南：`docs/ABSOLUTE_EEF_TRAINING_GUIDE.md`
