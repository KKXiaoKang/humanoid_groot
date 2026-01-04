# 人形双臂Action Decoder使用指南

## 问题总结

### v1.0版本问题
- **配置**：`split_arm_heads=False`（双臂14D共享一个decoder）
- **问题**：左手拉的动作拉不开，可能受右手影响
- **原因**：左右手特征完全耦合，单手操作时另一只手会"拖累"

### v2.0版本问题
- **配置**：`split_arm_heads=True, use_shared_arm_features=False`（完全独立的三个头）
- **问题**：左手做动作，右手忘记联动；flow-matching生成的单手结果不一致
- **原因**：左右手完全独立，缺少协调机制

## 最优方案（v3.0推荐）

### 配置
```python
# 在config.json或配置文件中设置
{
    "split_arm_heads": true,              # 分离左右手
    "use_shared_arm_features": true,      # 共享底层特征
    "use_cross_attention_arms": true,     # 启用交叉注意力（关键！）
    "arm_coordination_loss_weight": 0.2,  # 协调性损失权重（可调）
    "use_learnable_loss_weights": true    # 自适应损失权重
}
```

### 架构特点

1. **共享底层特征** (`use_shared_arm_features=True`)
   - 提取共同的环境和任务信息
   - 让左右手共享上下文理解

2. **交叉注意力机制** (`use_cross_attention_arms=True`) ⭐ **关键！**
   - 左手的query关注右手的key/value
   - 右手的query关注左手的key/value
   - 让左右手能够感知对方状态，实现协调

3. **独立输出层**
   - 保持左右手的独立性
   - 可以分别控制损失权重

4. **协调性损失** (`arm_coordination_loss_weight=0.2`)
   - 鼓励左右手动作幅度相似（但不完全相同）
   - 显式约束协调性，但不强制完全同步

## 如何选择配置

### 场景1：需要高度协调的任务（如拉箱子）
```python
{
    "use_shared_arm_features": true,
    "use_cross_attention_arms": true,      # 必须启用
    "arm_coordination_loss_weight": 0.3,   # 提高权重
}
```

### 场景2：需要独立操作的任务（如单手抓取）
```python
{
    "use_shared_arm_features": true,
    "use_cross_attention_arms": true,      # 仍然启用，但权重较低
    "arm_coordination_loss_weight": 0.1,   # 降低权重
}
```

### 场景3：混合任务（推荐）
```python
{
    "use_shared_arm_features": true,
    "use_cross_attention_arms": true,      # 启用交叉注意力
    "arm_coordination_loss_weight": 0.2,   # 中等权重
    "use_learnable_loss_weights": true     # 让模型自动学习
}
```

## 调参建议

### 1. 协调性损失权重 (`arm_coordination_loss_weight`)

- **0.0-0.1**：强调独立性，适合单手操作任务
- **0.1-0.2**：平衡独立性和协调性（推荐）
- **0.2-0.3**：强调协调性，适合双手协调任务
- **>0.3**：可能过度约束，导致动作僵硬

### 2. 交叉注意力 (`use_cross_attention_arms`)

- **True（推荐）**：让左右手能够感知对方状态
- **False**：类似"合成一个MLP然后split"，协调性较差

### 3. 自适应损失权重 (`use_learnable_loss_weights`)

- **True（推荐）**：让模型自动学习最优权重
- **False**：使用固定权重，需要手动调参

## 训练建议

1. **渐进式训练**
   - 先训练协调性（`arm_coordination_loss_weight=0.3`）
   - 再微调独立性（降低权重到0.1-0.2）

2. **监控指标**
   - `left_arm_loss` / `right_arm_loss`：左右手独立损失
   - `arm_coordination_loss`：协调性损失
   - `weight_left_arm` / `weight_right_arm`：自适应权重

3. **调试技巧**
   - 如果左右手忘记联动：提高`arm_coordination_loss_weight`
   - 如果单手操作受限：降低`arm_coordination_loss_weight`
   - 如果flow-matching结果不一致：确保`use_cross_attention_arms=True`

## 预期效果

使用最优方案后，应该能够：

✅ **解决v1.0问题**：通过分离输出层，左右手可以独立操作  
✅ **解决v2.0问题**：通过交叉注意力，左右手能够感知对方状态  
✅ **提升协调性**：通过协调性损失，显式鼓励协调  
✅ **保持独立性**：通过独立输出层，保持左右手的灵活性  

## 参考研究

- 《Learning Bimanual Manipulation via Action Chunking and Inter-Arm Coordination with Transformers》
- Multi-task learning with shared representations
- Cross-attention mechanisms for coordination

