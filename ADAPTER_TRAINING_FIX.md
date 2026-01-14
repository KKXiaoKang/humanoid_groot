# 🔧 适配层训练问题完整解决方案

## 📋 问题分析

### 根本原因

1. **Flow Matching loss 梯度太弱**
   - 梯度需要穿过整个 DiT (action_head)
   - 经过多层 Transformer，梯度衰减严重
   - 适配层初始化为恒等映射，需要强梯度才能推动学习

2. **适配层容量不足**
   - Linear adapter 表达能力有限
   - 无法学习复杂的分布映射

3. **训练策略不当**
   - 学习率 1e-4 太小
   - 训练数据太少（20 samples）
   - 训练轮数不够（20 epochs）

### 证据

检查适配层权重发现：
```
distribution_adapter.adapter.0.weight:
    Distance from identity: 0.002929  # 几乎等于恒等矩阵！
    ⚠️ Adapter NOT trained!
```

---

## ✅ 完整解决方案

### 方案 1：改进的适配层训练（推荐）⭐

**关键改进**：
1. **MLP adapter**（更强表达能力）
2. **学习率 1e-3**（10x 增大）
3. **50 epochs**（更多训练）
4. **100+ samples**（更多数据）
5. **Warmup scheduler**（更好的训练策略）

**使用方法**：

```bash
# 直接运行（已更新参数）
./merge_groot_models.sh two_stage_adapter 0

# 或者手动指定参数
python scripts/train_weight_merge.py \
    --method two_stage_adapter \
    --narrower_path "$NARROWER_PATH" \
    --wider_path "$WIDER_PATH" \
    --use_default_datasets \
    --alpha 0.5 \
    --adapter_type mlp \
    --adapter_epochs 50 \
    --adapter_lr 1e-3 \
    --num_samples 100 \
    --batch_size 1 \
    --device cuda:0 \
    --output_path ./outputs/merged_groot/pretrained_model
```

**训练后检查**：

```bash
# 检查适配层是否真的被训练了
python3 -c "
import json
from pathlib import Path
from safetensors.torch import load_file
import glob
import numpy as np

model_path = Path('./outputs/merged_groot/pretrained_model')
safetensors_files = glob.glob(str(model_path / 'model*.safetensors'))
state_dict = {}
for f in sorted(safetensors_files):
    state_dict.update(load_file(f))

# 检查 MLP adapter 的第一层权重
key = 'distribution_adapter.adapter.0.weight'
if key in state_dict:
    weight = state_dict[key].cpu().numpy()
    print(f'Adapter weight shape: {weight.shape}')
    print(f'Mean: {weight.mean():.6f}, Std: {weight.std():.6f}')
    print(f'Min: {weight.min():.6f}, Max: {weight.max():.6f}')
    
    # MLP adapter 的第一层是 (hidden_size, hidden_size*2)，不是方阵
    # 所以不能检查与恒等矩阵的距离
    # 但可以检查权重是否远离零
    if abs(weight.mean()) > 0.01 or weight.std() > 0.1:
        print('✅ Adapter weights have changed significantly!')
    else:
        print('⚠️ Adapter weights still close to initialization')

# 检查 residual_scale
key2 = 'distribution_adapter.residual_scale'
if key2 in state_dict:
    scale = state_dict[key2].item()
    print(f'Residual scale: {scale:.6f}')
    if abs(scale - 1.0) > 0.01:
        print('✅ Residual scale adjusted!')
    else:
        print('⚠️ Residual scale unchanged')
"
```

---

### 方案 2：不使用适配层，直接插值（最简单）⭐

**如果适配层训练仍然失败，这是最可靠的方案**：

```bash
# 直接插值 backbone，使用 narrower 的 action_head
./merge_groot_models.sh interpolation 0
```

**原理**：
- 只融合 backbone（50% narrower + 50% wider）
- Action head 使用 narrower 的原始权重（不融合）
- 不需要适配层，避免了所有适配层训练问题

**优点**：
- ✅ 简单可靠
- ✅ 不需要训练
- ✅ 避免了适配层训练的所有问题

**缺点**：
- ⚠️ 可能无法完全解决分布漂移问题
- ⚠️ 性能可能不如训练好的适配层

---

### 方案 3：Expert Merging（只融合 backbone）

```bash
./merge_groot_models.sh expert_merge 0
```

**原理**：
- 使用训练数据学习最优融合系数
- 只融合 backbone，action_head 使用 narrower 的

---

## 🔍 诊断工具

### 检查适配层权重

```bash
python3 << 'EOF'
import json
from pathlib import Path
from safetensors.torch import load_file
import glob
import numpy as np

model_path = Path('./outputs/merged_groot/pretrained_model')
safetensors_files = glob.glob(str(model_path / 'model*.safetensors'))
state_dict = {}
for f in sorted(safetensors_files):
    state_dict.update(load_file(f))

adapter_keys = [k for k in state_dict.keys() if k.startswith('distribution_adapter.')]
print(f'Found {len(adapter_keys)} adapter parameters:')

for key in sorted(adapter_keys):
    weight = state_dict[key]
    weight_np = weight.cpu().numpy()
    print(f'\n{key}:')
    print(f'  Shape: {weight.shape}')
    print(f'  Mean: {weight_np.mean():.6f}, Std: {weight_np.std():.6f}')
    print(f'  Min: {weight_np.min():.6f}, Max: {weight_np.max():.6f}')
    
    # 检查是否接近初始化
    if 'adapter.0.weight' in key:
        if len(weight_np.shape) == 2:
            # MLP: (hidden_size, hidden_size*2)
            if abs(weight_np.mean()) < 0.01 and weight_np.std() < 0.1:
                print('  ⚠️ Still close to initialization!')
            else:
                print('  ✅ Weight has changed significantly')
    
    if 'residual_scale' in key:
        scale = weight_np.item()
        if abs(scale - 1.0) < 0.01:
            print('  ⚠️ Residual scale unchanged!')
        else:
            print(f'  ✅ Residual scale adjusted: {scale:.6f}')
EOF
```

---

## 📊 训练监控

训练时会打印：
- ✅ 梯度大小（应该 > 1e-6）
- ✅ 学习率（使用 warmup）
- ✅ 适配层参数统计（训练后）

**关键指标**：
- `grad_mean > 1e-6`：梯度足够大
- `residual_scale != 1.0`：适配层在学习
- `adapter weight std > 0.1`：权重有变化

---

## 🚀 完整流程

### 1. 训练融合模型

```bash
# 方案 1：改进的适配层训练
./merge_groot_models.sh two_stage_adapter 0

# 方案 2：直接插值（如果方案 1 失败）
./merge_groot_models.sh interpolation 0
```

### 2. 检查适配层权重

```bash
# 运行上面的诊断工具
```

### 3. 评估模型

```bash
python scripts/eval_merged_groot_on_dataset.py \
    --model-path ./outputs/merged_groot/pretrained_model \
    --dataset-root /path/to/dataset \
    --episode 0 \
    --visualize
```

---

## ⚠️ 如果仍然失败

### 可能的原因

1. **Flow Matching loss 梯度确实太弱**
   - 这是 GROOT 架构的固有问题
   - 适配层训练可能不可行

2. **两个专家模型差异太大**
   - Backbone 融合后分布漂移太严重
   - 适配层无法修复

3. **训练数据分布不匹配**
   - 训练数据与推理数据分布不同
   - 适配层过拟合训练数据

### 最终方案

**如果所有适配层方案都失败，建议**：

1. **使用直接插值**（方案 2）
   - 最简单可靠
   - 不需要训练

2. **分别使用两个专家模型**
   - 根据任务选择 narrower 或 wider
   - 避免融合带来的问题

3. **重新训练统一模型**
   - 使用混合数据集训练一个模型
   - 避免模型融合的复杂性

---

## 📝 总结

**推荐顺序**：

1. ✅ **方案 1（改进的适配层训练）**：如果成功，效果最好
2. ✅ **方案 2（直接插值）**：如果方案 1 失败，最简单可靠
3. ✅ **方案 3（Expert Merging）**：如果方案 2 效果不好，可以尝试

**关键参数**：
- `--adapter_type mlp`（必须）
- `--adapter_lr 1e-3`（必须，10x 增大）
- `--adapter_epochs 50`（推荐）
- `--num_samples 100`（推荐）

**如果仍然失败**：使用直接插值（方案 2），这是最可靠的方案。
