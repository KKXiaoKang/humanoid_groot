# GROOT 模型权重融合指南

基于论文 [Expert Merging: Model Merging with Unsupervised Expert Alignment](https://arxiv.org/pdf/2509.25712) 实现的权重融合方案。

## 📋 问题背景

您有两个针对不同任务训练的 GROOT 模型：
- **GROOT_narrower**: 针对窄箱子拆垛任务训练
- **GROOT_wider**: 针对宽箱子拆垛任务训练

两个模型都是从 `nvidia/GR00T-N1.5-3B` 全量微调得到的。

## ⭐ 校准数据说明

**原始训练数据集可以直接作为校准数据使用！**

Expert Merging 方法需要少量校准数据来学习最优的融合系数：
- **数量**：只需要 5-10 个样本
- **来源**：原始训练数据集（`lerobot_data` 文件夹中的数据）
- **要求**：不需要动作标签，只使用观测数据
- **建议**：混合使用两个任务的数据（窄箱子 + 宽箱子）

默认使用的校准数据集：
```
/home/lab/humanoid_groot/lerobot_data/v3_0_dataset/
├── 1215_5w_groot_4311_4322_4611_4633_narrower    # 窄箱子数据
├── 1221_5w_random_height_4322_4611_narrower      # 窄箱子数据
├── 1223_5w_dense_stacking_narrower               # 窄箱子数据
├── 1225_5w_unpack_mix_color_narrower             # 窄箱子数据
├── 1215_5w_groot_4311_4322_4611_4633_wider       # 宽箱子数据
├── 1221_5w_random_height_4322_4611_wider         # 宽箱子数据
├── 1223_5w_dense_stacking_wider                  # 宽箱子数据
└── 1225_5w_unpack_mix_color_wider                # 宽箱子数据
```

## 🎯 方案对比

| 方法 | 需要训练 | 效果 | 速度 | 推荐度 |
|------|----------|------|------|--------|
| **Task Arithmetic** | ❌ | ⭐⭐⭐ | 快 | ⭐⭐⭐⭐ 推荐 |
| **TIES Merging** | ❌ | ⭐⭐⭐⭐ | 快 | ⭐⭐⭐⭐ |
| **DARE** | ❌ | ⭐⭐⭐ | 快 | ⭐⭐⭐ |
| **Expert Merging** | ✅ (少量数据) | ⭐⭐⭐⭐⭐ | 中 | ⭐⭐⭐⭐⭐ 最佳 |
| **Direct Interpolation** | ❌ | ⭐⭐ | 最快 | ⭐⭐ |

## 🚀 快速开始

### 方法 1: Expert Merging（默认，效果最好）⭐

**原理**: 学习最优的 layer-wise 融合系数，通过对齐隐藏状态和 action 输出

```bash
# 最简单的方式：使用默认的训练数据集作为校准数据
./merge_groot_models.sh

# 或者直接运行（自动使用 lerobot_data 中的数据集）
python scripts/train_weight_merge.py \
    --method expert_merge \
    --use_default_datasets \
    --num_samples 10 \
    --num_epochs 10 \
    --output_path ./outputs/merged_groot/pretrained_model

# 指定自定义校准数据集
python scripts/train_weight_merge.py \
    --method expert_merge \
    --data_path /path/to/narrow_data,/path/to/wide_data \
    --num_samples 10 \
    --num_epochs 10 \
    --output_path ./outputs/merged_groot/pretrained_model
```

### 方法 2: Task Arithmetic（无需训练，快速）

**原理**: 计算每个模型相对于 base 模型的变化（Task Vector），然后加权组合。

```bash
./merge_groot_models.sh task_arithmetic

# 或者
python scripts/train_weight_merge.py \
    --method task_arithmetic \
    --narrower_weight 0.5 \
    --wider_weight 0.5 \
    --output_path ./outputs/merged_groot/pretrained_model
```

**公式**:
```
θ_merged = θ_base + α_narrow × (θ_narrow - θ_base) + α_wide × (θ_wide - θ_base)
```

### 方法 3: TIES Merging

**原理**: 剪枝小的参数变化 + 解决符号冲突 + 重缩放

```bash
./merge_groot_models.sh ties
```

### 方法 4: DARE Merging

**原理**: 随机丢弃部分参数变化并重缩放

```bash
./merge_groot_models.sh dare
```

**核心公式**:

1. **Layer-wise Coefficients**:
   ```
   θ_merged^ℓ = θ_base^ℓ + Σ_k α_k^ℓ × τ_k^ℓ
   ```

2. **Hidden Alignment Loss**:
   ```
   L_hid = ||h_merged - h_expert||²
   ```

3. **Action Alignment Loss**:
   ```
   L_action = ||a_merged - a_expert||²
   ```

4. **Coefficient Regularization**:
   ```
   L_reg = (1/KL) × Σ_k Σ_ℓ |α_k^ℓ - α_init|
   ```

5. **Total Loss**:
   ```
   L = Σ_k β_k × (L_hid^k + L_action^k) + γ × L_reg
   ```

### 方法 5: 直接插值（最简单）

```bash
python scripts/train_weight_merge.py \
    --method interpolation \
    --alpha 0.5 \
    --output_path ./outputs/merged_groot/pretrained_model
```

**公式**:
```
θ_merged = α × θ_narrow + (1-α) × θ_wide
```

## 📊 评估融合模型

```bash
# 评估融合后的模型
python eval/eval_merged_groot.py --model_path ./outputs/merged_groot/pretrained_model

# 或使用标准评估脚本
python eval/eval_online.py --policy.path=./outputs/merged_groot/pretrained_model
```

## ⚙️ 高级配置

### Expert Merging 参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 1e-3 | 学习率 |
| `--num_epochs` | 10 | 训练轮数 |
| `--regularization_weight` | 0.8 | 正则化权重 γ |
| `--initial_coefficient` | 0.5 | 初始系数值 |
| `--hidden_weight` | 1.0 | Hidden alignment 损失权重 |
| `--logit_weight` | 1.0 | Action alignment 损失权重 |
| `--narrower_task_weight` | 1.0 | narrower 专家的任务权重 |
| `--wider_task_weight` | 1.0 | wider 专家的任务权重 |
| `--num_samples` | 10 | 每个数据集采样数量 |

### 调整任务优先级

如果希望融合模型更偏向某个任务：

```bash
# 偏向 narrower 任务
python scripts/train_weight_merge.py \
    --method task_arithmetic \
    --narrower_weight 0.7 \
    --wider_weight 0.3 \
    --output_path ./outputs/merged_groot/pretrained_model

# 或使用 Expert Merging
python scripts/train_weight_merge.py \
    --method expert_merge \
    --narrower_task_weight 1.5 \
    --wider_task_weight 0.5 \
    --output_path ./outputs/merged_groot/pretrained_model
```

## 🔧 代码结构

```
src/lerobot/policies/groot/
├── weight_merge_groot.py     # 核心融合逻辑
│   ├── TaskVector            # Task Vector 计算
│   ├── LayerWiseCoefficients # 可学习系数
│   ├── MergedGR00TModel      # 融合模型
│   └── ExpertMerger          # 融合训练器

scripts/
├── train_weight_merge.py     # 训练脚本

eval/
├── eval_merged_groot.py      # 评估脚本
```

## 📚 理论背景

### Task Arithmetic

Task Arithmetic 是一种简单但有效的模型融合方法：

1. **Task Vector**: 模型微调后相对于 base 的参数变化
   ```
   τ = θ_finetuned - θ_base
   ```

2. **融合**: 将多个 Task Vector 加权组合
   ```
   θ_merged = θ_base + Σ_k λ_k × τ_k
   ```

### Expert Merging

Expert Merging 在 Task Arithmetic 基础上，通过学习最优系数来提升效果：

1. **Layer-wise Coefficients**: 每层学习独立的系数，捕获层间差异
2. **Hidden Alignment**: 对齐融合模型和专家模型的内部表示
3. **Logit/Action Alignment**: 对齐输出行为
4. **Coefficient Regularization**: 防止系数过度偏移

### TIES Merging

TIES (TrIm, Elect Sign, and Scale) 通过三个步骤减少参数冲突：

1. **Trim**: 剪枝小的参数变化（噪声）
2. **Elect Sign**: 使用多数投票解决符号冲突
3. **Scale**: 重缩放合并后的参数

### DARE

DARE (Drop And REscale) 通过稀疏化减少干扰：

1. **Drop**: 随机丢弃部分参数变化
2. **Rescale**: 重缩放保留的参数以保持期望值

## 🤔 FAQ

### Q: 应该选择哪种方法？

A: 推荐顺序：
1. **首先尝试 Task Arithmetic** - 快速、无需训练、效果通常不错
2. **如果效果不佳，尝试 TIES** - 可能解决参数冲突问题
3. **如果有校准数据，使用 Expert Merging** - 效果最好

### Q: 融合后模型效果变差怎么办？

A: 
1. 调整权重比例（narrower_weight/wider_weight）
2. 尝试 TIES 方法减少参数冲突
3. 使用 Expert Merging 学习最优系数
4. 增加正则化权重避免过度融合

### Q: Expert Merging 需要多少校准数据？

A: 论文推荐 5-10 个样本即可，无需标签。

### Q: 融合后的模型参数量是多少？

A: 与单个模型完全相同，不会增加参数量。

## 📚 参考文献

1. [Expert Merging](https://arxiv.org/pdf/2509.25712) - Layer-wise coefficient learning
2. [Task Arithmetic](https://arxiv.org/abs/2212.04089) - Task vector merging
3. [TIES Merging](https://arxiv.org/abs/2306.01708) - Conflict resolution
4. [DARE](https://arxiv.org/abs/2311.03099) - Drop and rescale
5. [kai0 Model Arithmetic](https://mmlab.hk/research/kai0) - HKU MMLab
