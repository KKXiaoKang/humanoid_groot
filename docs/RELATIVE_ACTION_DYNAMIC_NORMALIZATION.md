# Relative Action 动态归一化统计收集

## 问题背景

在使用 "Delta eef" action space 训练模型时，遇到了左右手位置预测偏差的问题：
- 左手拆垛时，左手准确，但右手y方向不准
- 右手拆垛时，右手准确，但左手y方向不准
- 双手往前任务时，总有一个手的y方向差异很大

**根本原因**：relative action的position分布与absolute eef pose的分布不同，但归一化时使用的是absolute stats，导致归一化不准确。

## 解决方案

实现了**动态归一化统计收集**功能：
1. **训练过程中**：实时累积relative action position组件的统计值（min/max/count）
2. **保存checkpoint时**：将累积的统计值保存到preprocessor的state_dict和config.json中
3. **推理时**：从保存的统计值中加载，用于归一化relative action position组件

## 实现细节

### 1. 统计值累积 (`update_relative_action_stats`)

在`GrootPackInputsStep`中添加了`update_relative_action_stats`方法，在每次处理batch时：
- 提取relative action的position组件（left_eef_pos, right_eef_pos）
- 计算当前batch的min/max
- 使用running min/max更新全局统计值

```python
def update_relative_action_stats(self, relative_action: torch.Tensor):
    """更新relative action position组件的统计值（min/max/count）"""
    # 只对position组件进行统计
    # rotation组件使用IDENTITY归一化，不需要统计
    # gripper组件可以使用absolute stats
```

### 2. 归一化逻辑更新

在`_min_max_norm_partial`和`_min_max_unnorm_partial`中：
- **优先使用**：动态累积的`relative_action_stats`
- **回退方案**：如果统计值无效（inf），使用调整后的absolute range（1.5x）

```python
# 使用动态累积的relative_action_stats
if (self.relative_action_stats is not None and 
    component_name in self.relative_action_stats):
    rel_stats = self.relative_action_stats[component_name]
    min_v = rel_stats["min"]
    max_v = rel_stats["max"]
else:
    # 回退：使用调整后的absolute range
    abs_range = torch.maximum(torch.abs(min_v), torch.abs(max_v))
    rel_range = abs_range * 1.5
    min_v = -rel_range
    max_v = rel_range
```

### 3. 保存和加载

**保存 (`state_dict`)**：
- 将`relative_action_stats`保存到safetensors文件
- 键名格式：`relative_action.{comp_name}.{stat_name}`

**加载 (`load_state_dict`)**：
- 从safetensors文件加载`relative_action_stats`
- 自动初始化`_relative_stats_initialized`标志

**配置保存 (`get_config`)**：
- 将`relative_action_stats`序列化为JSON格式
- 保存到`policy_preprocessor.json`中

## 使用方法

### 训练时

无需额外配置，功能自动启用：
1. 训练过程中，每次处理batch时自动累积统计值
2. **第一个epoch完成后自动冻结统计值**（默认行为，推荐）
3. 保存checkpoint时，统计值自动保存到preprocessor的state_dict和config.json

```bash
# 正常训练即可
python -m lerobot.scripts.lerobot_train \
    --config-path configs/train_groot.yaml \
    --config-name train_groot
```

### 统计值累积策略

**默认行为（推荐）**：
- `freeze_stats_after_first_epoch=True`（默认）
- 第一个epoch完成后自动冻结统计值
- 避免不必要的计算开销
- 确保统计值稳定（不会因为浮点误差产生微小变化）

**为什么第一个epoch后冻结？**

对于固定数据集：
1. ✅ **统计值已收敛**：一个epoch后，min/max已经遍历了所有数据
2. ✅ **避免数值误差**：继续累积不会改变min/max（数据相同），但可能因为浮点误差产生微小差异
3. ✅ **提高效率**：避免不必要的计算开销
4. ✅ **符合统计学直觉**：统计值应该基于完整数据集计算一次

**如果需要继续累积**（不推荐）：
- 设置`freeze_stats_after_first_epoch=False`
- 统计值会在整个训练过程中持续累积
- 适用于streaming数据集或数据增强场景

### 推理时

统计值会自动从checkpoint加载，无需额外操作：

```python
from lerobot.policies.groot.modeling_groot import GrootPolicy

# 加载模型（自动加载relative_action_stats）
policy = GrootPolicy.from_pretrained(checkpoint_path)

# 正常推理
action = policy.get_action(observation)
```

### 检查统计值

可以使用`check_normalization_stats.py`脚本检查统计值：

```bash
python scripts/check_normalization_stats.py \
    --ckpt-path outputs/train/your_checkpoint/checkpoints/020000/pretrained_model \
    --dataset-root /path/to/dataset \
    --dataset-repo-ids "dataset1,dataset2"
```

## 技术细节

### 统计值结构

```python
relative_action_stats = {
    "left_eef_pos": {
        "min": torch.Tensor([x_min, y_min, z_min]),  # (3,)
        "max": torch.Tensor([x_max, y_max, z_max]),  # (3,)
        "count": torch.tensor(total_samples),  # scalar
    },
    "right_eef_pos": {
        "min": torch.Tensor([x_min, y_min, z_min]),  # (3,)
        "max": torch.Tensor([x_max, y_max, z_max]),  # (3,)
        "count": torch.tensor(total_samples),  # scalar
    },
}
```

### 保存位置

1. **safetensors文件**：`policy_preprocessor_step_X_groot_pack_inputs_v3.safetensors`
   - 键名：`relative_action.{comp_name}.{stat_name}`

2. **JSON配置文件**：`policy_preprocessor.json`
   - 键名：`relative_action_stats`
   - 格式：序列化的列表（非tensor）

### 兼容性

- **向后兼容**：如果checkpoint中没有`relative_action_stats`，会回退到调整后的absolute range（1.5x）
- **向前兼容**：新训练的模型会自动包含`relative_action_stats`

## 验证

训练完成后，检查checkpoint中的统计值：

```python
from lerobot.policies.factory import make_pre_post_processors
from pathlib import Path

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path="path/to/checkpoint",
)

# 检查relative_action_stats
for step in preprocessor.steps:
    if hasattr(step, 'relative_action_stats') and step.relative_action_stats:
        print("Relative action stats:", step.relative_action_stats)
        break
```

## 注意事项

1. **只统计position组件**：rotation组件使用IDENTITY归一化，不需要统计
2. **第一个epoch后冻结**：默认在第一个epoch完成后冻结统计值，避免不必要的计算
3. **设备兼容性**：统计值会自动转换到正确的device和dtype
4. **无效值处理**：如果统计值无效（inf），会自动回退到调整后的absolute range
5. **数据集大小检测**：需要传递`dataset_num_frames`参数才能自动检测第一个epoch完成（训练脚本已自动处理）

## 预期效果

使用动态归一化统计后，应该能够：
1. ✅ 更准确地归一化relative action position组件
2. ✅ 减少左右手位置预测偏差
3. ✅ 提高双手协调任务的性能

## 故障排除

### 问题：统计值没有更新

**检查**：
1. 确认`action_space_type == "Delta eef"`
2. 确认`action_component_indices`已正确设置
3. 检查`update_relative_action_stats`是否被调用

### 问题：推理时统计值未加载

**检查**：
1. 确认checkpoint中包含`relative_action_stats`
2. 检查`load_state_dict`是否成功加载
3. 查看`_relative_stats_initialized`标志

### 问题：归一化结果异常

**检查**：
1. 确认统计值不是inf或nan
2. 检查统计值的device和dtype是否正确
3. 查看是否触发了回退逻辑
