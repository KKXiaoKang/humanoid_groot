# Relative Action 误差累积修复方案使用指南

## 问题背景

在使用 **Delta eef** (relative action) 模式时，由于训练-推理分布不匹配和误差累积，导致模型表现不佳。详细分析请参考 `RELATIVE_ACTION_ERROR_ACCUMULATION_ANALYSIS.md`。

## 已实现的解决方案

### 方案1: 训练时引入噪声（推荐，需要重新训练）

**功能**：在训练时对 reference pose 添加噪声，模拟推理时的跟踪误差，让模型学习对误差的鲁棒性。

**配置参数**：
- `relative_action_reference_noise_std`: Position 噪声标准差（单位：米），建议设置为 0.02-0.03（2-3cm）
- `relative_action_rotation_noise_deg`: Rotation 噪声标准差（单位：度），建议设置为 1-2 度

**使用方法**：

1. 在训练配置中添加噪声参数：

```python
# 在 GrootConfig 或训练脚本中
config.preprocessor_config = {
    "steps": [
        {
            "registry_name": "groot_pack_inputs_v3",
            "config": {
                "action_space_type": "Delta eef",
                "relative_action_reference_noise_std": 0.025,  # 2.5cm 噪声
                "relative_action_rotation_noise_deg": 1.5,    # 1.5度 噪声
                # ... 其他配置
            }
        },
        # ... 其他步骤
    ]
}
```

2. 重新训练模型：

```bash
python train_groot.py --config your_config.yaml
```

**优点**：
- ✅ 从根本上解决分布不匹配问题
- ✅ 让模型学习对 reference pose 误差的鲁棒性
- ✅ 训练和推理的分布更匹配

**缺点**：
- ❌ 需要重新训练模型
- ❌ 训练时间可能略长（但影响不大）

### 方案2: 使用预测的 absolute pose 作为 reference（临时方案，可立即应用）

**功能**：不使用实际机器人状态作为 reference，而是使用上一次预测的 absolute pose，避免误差累积。

**实现状态**：✅ 已实现，在 `eval_depalletize_camera_model_reload_limit_vel_select_eef.py` 中自动启用

**工作原理**：
1. 第一个 chunk：使用实际机器人状态（FK）作为 reference
2. 后续 chunk：使用上一次预测的最后一个 absolute pose 作为 reference
3. 这样可以避免误差累积（因为 reference 是预测值，不是实际值）

**使用方法**：

无需额外配置，代码已自动启用。在 relative action mode 下，推理时会自动使用预测值作为 reference。

**优点**：
- ✅ 不需要重新训练
- ✅ 可以立即应用
- ✅ 避免误差累积

**缺点**：
- ❌ 如果预测有偏差，reference 也会有偏差
- ❌ 可能导致"漂移"（drift）问题
- ❌ 不如方案1从根本上解决问题

## 推荐使用流程

### 短期（立即应用）：
1. **使用方案2**：代码已自动启用，无需额外配置
   - 运行现有的推理脚本即可
   - 观察效果是否改善

### 中期（重新训练）：
2. **实施方案1**：训练时引入噪声
   - 在训练配置中添加噪声参数
   - 重新训练模型
   - 使用新模型进行推理

### 长期（优化）：
3. **如果方案1效果不够好，考虑混合模式**：
   - Position 使用 relative action
   - Rotation 使用 absolute action
   - 需要修改训练和推理代码（参考 `RELATIVE_ACTION_ERROR_ACCUMULATION_ANALYSIS.md`）

## 验证方法

### 1. 检查方案2是否生效：

运行推理脚本，查看日志输出：

```
[INFERENCE] ✅ Using last predicted absolute pose as reference (relative action mode, error accumulation mitigation)
```

如果看到这条日志，说明方案2已启用。

### 2. 对比效果：

- **使用方案2前**：双手容易偏得更岔开，对不准物体
- **使用方案2后**：双手位置应该更稳定，误差累积减少

### 3. 检查方案1配置：

在训练配置中检查是否设置了噪声参数：

```python
"relative_action_reference_noise_std": 0.025,  # 应该 > 0
"relative_action_rotation_noise_deg": 1.5,     # 应该 > 0
```

## 注意事项

1. **方案2的局限性**：
   - 如果模型预测本身有偏差，使用预测值作为 reference 可能会导致"漂移"
   - 建议与方案1结合使用

2. **方案1的噪声设置**：
   - Position 噪声应该匹配实际的跟踪误差（2-3cm）
   - Rotation 噪声应该匹配实际的旋转跟踪误差（1-2度）
   - 噪声太大可能导致训练不稳定，太小可能效果不明显

3. **双手协同任务**：
   - 双手协同任务对误差更敏感
   - 建议优先使用方案1（训练时引入噪声）

## 故障排除

### 问题1: 方案2没有生效

**检查**：
- 确认 `is_relative_action_mode` 为 `True`
- 确认 `action_space_type == "Delta eef"`
- 查看日志是否有相关输出

### 问题2: 方案1训练不稳定

**可能原因**：
- 噪声设置太大
- 建议减小噪声标准差，例如：
  - `relative_action_reference_noise_std`: 0.015 (1.5cm)
  - `relative_action_rotation_noise_deg`: 1.0 (1度)

### 问题3: 效果改善不明显

**可能原因**：
- 方案2只是临时方案，建议使用方案1重新训练
- 噪声设置可能不合适，需要调整
- 可能需要考虑其他因素（如归一化统计、模型架构等）

## 相关文档

- `RELATIVE_ACTION_ERROR_ACCUMULATION_ANALYSIS.md`: 详细的问题分析和解决方案
- `src/lerobot/policies/groot/processor_groot.py`: 方案1的实现代码
- `scripts/eval_depalletize_camera_model_reload_limit_vel_select_eef.py`: 方案2的实现代码
