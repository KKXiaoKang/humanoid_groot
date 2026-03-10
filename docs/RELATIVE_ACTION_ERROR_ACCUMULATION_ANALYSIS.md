# Relative Action 误差累积问题深度分析

## 问题描述

在使用 **Delta eef** (relative action) 模式训练和推理时，模型表现远不如 **Absolute eef** 模式：
- Absolute eef: ✅ 双手可以准确对准物体
- Relative eef: ❌ 双手容易偏得更岔开，对不准物体

## 根本原因分析

### 1. 训练-推理分布不匹配 (Distribution Shift)

#### 训练时：
```python
# 使用 ground truth 的 perfect reference
ref_pose = absolute_action[:, 0:1, :]  # 完美的 ground truth，无误差
relative_action = absolute_action - ref_pose  # 基于 perfect reference
```

**特点**：
- Reference pose 是 ground truth，完全准确
- 模型学习的是"从 perfect reference 到 target"的映射
- Relative action 的分布基于 perfect reference

#### 推理时：
```python
# 使用实际机器人的状态（有 2-3cm 跟踪误差）
current_reference_pose = FK(arm_joint_pos)  # 实际状态，有误差！
absolute_pose = current_reference_pose + relative_action  # 误差会传播
```

**特点**：
- Reference pose 来自实际机器人状态（通过 FK 计算），存在 2-3cm 跟踪误差
- 模型输出的是"从 perfect reference 到 target"的 relative action
- 但这个 relative action 被加到了**有误差的 reference** 上

### 2. 误差累积机制

```
Chunk 1:
  reference_actual = FK(actual_joints)  [误差: 2-3cm]
  → absolute_predicted = reference_actual + relative_predicted
  → 执行后实际到达位置 ≠ absolute_predicted [又有 2-3cm 跟踪误差]
  
Chunk 2:
  reference_actual = FK(new_actual_joints)  [累积误差: 4-6cm]
  → absolute_predicted = reference_actual + relative_predicted
  → 误差继续累积...
  
Chunk N:
  reference_actual = FK(...)  [累积误差: 2N-3N cm]
  → 双手位置偏差越来越大，无法对准物体
```

### 3. 双手协同的放大效应

- 左右手的误差会相互影响
- 如果左手有 3cm 误差，右手也有 3cm 误差，双手的相对位置偏差可能达到 6cm
- 对于需要精确双手协同的任务（如抓取物体），这种累积误差是致命的

### 4. 归一化统计的不匹配

- Relative action 的归一化统计是基于训练数据的 relative action 计算的
- 训练数据的 relative action 分布和推理时的可能不同（因为 reference 不同）
- 这进一步加剧了分布不匹配问题

## 解决方案

### 方案 1: 训练时引入噪声（推荐 ⭐⭐⭐⭐⭐）

**核心思想**：在训练时对 reference pose 添加噪声，模拟推理时的跟踪误差，让模型学习对误差的鲁棒性。

**实现方式**：
1. 在 `_convert_absolute_to_relative_eef_action` 中，对 reference pose 添加噪声
2. 噪声类型：
   - Position: 高斯噪声，标准差 2-3cm（匹配实际跟踪误差）
   - Rotation: 小角度旋转噪声（如 1-2 度）
3. 只在训练时添加噪声，推理时不添加

**优点**：
- 让模型学习对 reference pose 误差的鲁棒性
- 训练和推理的分布更匹配
- 不需要改变推理代码

**缺点**：
- 需要重新训练模型
- 训练时间可能略长（但影响不大）

### 方案 2: 使用预测的 absolute pose 作为 reference（次优 ⭐⭐⭐）

**核心思想**：不使用实际机器人状态作为 reference，而是使用上一次预测的 absolute pose。

**实现方式**：
1. 第一个 chunk：使用实际机器人状态（FK）作为 reference
2. 后续 chunk：使用上一次预测的最后一个 absolute pose 作为 reference
3. 这样可以避免误差累积（因为 reference 是预测值，不是实际值）

**优点**：
- 不需要重新训练
- 可以立即应用

**缺点**：
- 如果预测有偏差，reference 也会有偏差
- 可能导致"漂移"（drift）问题
- 不如方案 1 从根本上解决问题

### 方案 3: 混合模式（Hybrid Mode）⭐⭐⭐⭐

**核心思想**：结合 absolute 和 relative 的优点，使用"相对位置 + 绝对旋转"。

**实现方式**：
1. Position: 使用 relative action（相对位置）
2. Rotation: 使用 absolute action（绝对旋转）
3. 原因：旋转误差累积影响更大，使用绝对旋转可以避免累积误差

**优点**：
- 结合两种模式的优点
- Position 使用 relative 可以更好地处理不同起始位置
- Rotation 使用 absolute 可以避免累积误差

**缺点**：
- 需要修改训练和推理代码
- 需要重新训练模型

### 方案 4: 误差补偿（Error Compensation）⭐⭐⭐

**核心思想**：在推理时，检测并补偿 reference pose 的误差。

**实现方式**：
1. 记录上一次预测的 absolute pose 和实际到达的位置
2. 计算误差：`error = actual_pose - predicted_pose`
3. 在下一个 chunk 的 reference pose 中补偿误差：
   ```python
   compensated_reference = FK(actual_joints) + error
   ```

**优点**：
- 不需要重新训练
- 可以立即应用

**缺点**：
- 需要准确测量实际到达位置（可能需要额外的传感器）
- 误差补偿可能不稳定
- 不如方案 1 从根本上解决问题

## 推荐实施步骤

### 短期（立即应用）：
1. **实施方案 2**：使用预测的 absolute pose 作为 reference
   - 修改 `eval_depalletize_camera_model_reload_limit_vel_select_eef.py`
   - 在推理循环中，保存上一次预测的最后一个 absolute pose
   - 下一个 chunk 使用这个预测值作为 reference

### 中期（重新训练）：
2. **实施方案 1**：训练时引入噪声
   - 修改 `processor_groot.py` 中的 `_convert_absolute_to_relative_eef_action`
   - 添加配置参数：`relative_action_reference_noise_std`（position 噪声标准差）
   - 重新训练模型

### 长期（优化）：
3. **考虑方案 3**：混合模式
   - 如果方案 1 效果不够好，考虑使用混合模式
   - Position 使用 relative，Rotation 使用 absolute

## 代码修改示例

### 方案 1: 训练时引入噪声

```python
def _convert_absolute_to_relative_eef_action(self, absolute_action: torch.Tensor) -> torch.Tensor:
    # ... existing code ...
    
    # Extract reference pose (first timestep of each batch)
    ref_pose = absolute_action[:, 0:1, :]  # (B, 1, D)
    
    # 训练时添加噪声（模拟推理时的跟踪误差）
    if self.training and hasattr(self, 'relative_action_reference_noise_std'):
        noise_std = self.relative_action_reference_noise_std  # 例如 0.02 (2cm)
        
        # 对 position 组件添加高斯噪声
        for component_name, (start_idx, end_idx) in self.action_component_indices.items():
            if "pos" in component_name:
                noise = torch.randn_like(ref_pose[:, :, start_idx:end_idx]) * noise_std
                ref_pose[:, :, start_idx:end_idx] = ref_pose[:, :, start_idx:end_idx] + noise
            
            # 可选：对 rotation 添加小角度噪声
            elif "rot6d" in component_name and hasattr(self, 'relative_action_rotation_noise_deg'):
                # 实现旋转噪声（略复杂，需要转换为旋转矩阵，添加噪声，再转回6D）
                pass
    
    # ... rest of the code ...
```

### 方案 2: 使用预测的 absolute pose 作为 reference

```python
# 在推理循环中
last_predicted_absolute_pose = None  # 保存上一次预测的最后一个 absolute pose

for chunk_idx in range(num_chunks):
    if is_relative_action_mode:
        if last_predicted_absolute_pose is not None:
            # 使用上一次预测的最后一个 absolute pose 作为 reference
            current_reference_pose = last_predicted_absolute_pose.copy()
        else:
            # 第一个 chunk：使用实际机器人状态
            current_reference_pose = FK(arm_joint_pos)
        
        # 转换 relative action 到 absolute pose
        pred_chunk_absolute = convert_relative_to_absolute(
            pred_chunk_relative, 
            current_reference_pose
        )
        
        # 保存最后一个 absolute pose 供下一个 chunk 使用
        last_predicted_absolute_pose = pred_chunk_absolute[-1].copy()
```

## 总结

**核心问题**：训练时使用 perfect reference，推理时使用 imperfect reference（有误差），导致分布不匹配和误差累积。

**最佳解决方案**：在训练时引入噪声，让模型学习对 reference pose 误差的鲁棒性。这是最根本的解决方案，可以从根本上解决分布不匹配问题。

**临时解决方案**：使用预测的 absolute pose 作为 reference，可以立即应用，但不如方案 1 从根本上解决问题。
