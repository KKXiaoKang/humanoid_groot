# Action Space 配置重构说明

## 概述

本次重构移除了全局变量 `GLOBAL_ACTION_SPACE`，改为在配置中设置动作空间类型。这样更加清晰和可维护。

## 主要改动

### 1. 移除全局变量

- **移除前**：`src/lerobot/policies/groot/modeling_groot.py` 中定义了 `GLOBAL_ACTION_SPACE = "Absolute joint"`
- **移除后**：不再使用全局变量，改为在配置中设置

### 2. 在 `FlowmatchingActionHeadConfig` 中添加 `action_space_type` 字段

```python
# src/lerobot/policies/groot/action_head/flow_matching_action_head.py
action_space_type: str = field(default="Absolute joint", metadata={"help": "Action space type: 'Absolute joint', 'Absolute eef', or 'Delta eef'"})
```

### 3. 在 `GrootConfig` 中添加 `action_space_type` 字段

```python
# src/lerobot/policies/groot/configuration_groot.py
action_space_type: str = field(default="Absolute joint", metadata={"help": "Action space type: 'Absolute joint', 'Absolute eef', or 'Delta eef'. This will be passed to action_head_cfg."})
```

### 4. 自动配置维度

在 `FlowmatchingActionHeadConfig.__init__` 中，根据 `action_space_type` 自动配置维度：

- **Absolute eef / Delta eef** (20D):
  - `action_left_arm_dim = 9` (3D pos + 6D rot)
  - `action_right_arm_dim = 9` (3D pos + 6D rot)
  - `action_claw_dim = 2` (left + right gripper)

- **Absolute joint** (16D) - 默认:
  - `action_left_arm_dim = 7` (joints)
  - `action_right_arm_dim = 7` (joints)
  - `action_claw_dim = 2` (left + right gripper)

### 5. Processor 自动读取配置

`processor_groot.py` 中的 `make_groot_pre_post_processors` 函数现在从 `GrootConfig` 读取 `action_space_type`：

```python
action_space_type = getattr(config, 'action_space_type', "Absolute joint")
```

## 使用方法

### 方法 1: 在训练脚本中设置（推荐）

```bash
--policy.action_space_type="Absolute eef"
```

### 方法 2: 在 Python 代码中设置

```python
from lerobot.policies.groot.configuration_groot import GrootConfig

config = GrootConfig(
    action_space_type="Absolute eef",
    # ... 其他配置
)
```

### 方法 3: 在配置文件中设置

如果使用配置文件，可以在配置文件中添加：

```json
{
  "policy": {
    "action_space_type": "Absolute eef",
    ...
  }
}
```

## 向后兼容性

- **默认值**：如果不设置 `action_space_type`，默认使用 `"Absolute joint"`，保持向后兼容
- **自动配置**：系统会根据 `action_space_type` 自动配置维度，无需手动设置 `action_left_arm_dim` 等参数

## 验证

训练开始时会打印：

```
🎯 Auto-configured for Absolute eef action space:
   left_arm=9D, right_arm=9D, claw=2D
✅ Split arm heads enabled: left_arm(9D) + right_arm(9D) + claw(2D) = 20D
✅ Partial normalization enabled for action space: Absolute eef
   Components: ['left_eef_pos', 'left_eef_rot6d', 'right_eef_pos', 'right_eef_rot6d', 'left_gripper', 'right_gripper']
   6D rotation components (left_eef_rot6d, right_eef_rot6d) will use IDENTITY normalization
```

## 总结

1. ✅ **移除了全局变量**：不再使用 `GLOBAL_ACTION_SPACE`
2. ✅ **配置化**：`action_space_type` 在 `GrootConfig` 和 `FlowmatchingActionHeadConfig` 中设置
3. ✅ **自动配置**：根据 `action_space_type` 自动配置维度
4. ✅ **向后兼容**：默认使用 `"Absolute joint"`，保持原有行为
5. ✅ **支持两种动作空间**：`Absolute joint` (16D) 和 `Absolute eef` (20D) 都能正常工作
