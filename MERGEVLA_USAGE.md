# MergeVLA 风格的 GROOT 模型融合

基于论文 "MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent"
https://arxiv.org/pdf/2511.18810

## 核心思想

1. **稀疏激活的 LoRA 适配器**：使用任务掩码来稀疏激活不同的 LoRA 参数子集，减少任务间冲突
2. **Cross-attention-only action head**：GROOT 已满足（使用 DiT）
3. **Task Vectors 融合**：计算 τ = θ_expert - θ_base，然后融合

## 使用方法

### 基本用法（推荐）

```bash
# 使用 MergeVLA 方法融合两个全量微调的 GROOT 模型
python scripts/train_weight_merge.py \
    --method mergevla \
    --narrower_path /path/to/narrower/model \
    --wider_path /path/to/wider/model \
    --base_model_path /path/to/narrower/model \  # 如果 base == narrower，则 τ_narrower = 0
    --output_path ./outputs/merged_groot/pretrained_model \
    --adapter_type sparse_lora \  # MergeVLA 风格：稀疏激活的 LoRA
    --lora_rank 16 \
    --sparsity 0.5 \  # 每个任务激活 50% 的参数
    --adapter_epochs 20 \
    --adapter_lr 1e-3 \  # MergeVLA 使用较大的学习率
    --num_samples 10 \  # 每个数据集采样 10 个样本用于训练适配层
    --use_default_datasets  # 使用默认的训练数据集
```

### 参数说明

- `--method mergevla`: 使用 MergeVLA 风格的融合方法
- `--adapter_type sparse_lora`: 稀疏激活的 LoRA（MergeVLA 核心）
- `--sparsity 0.5`: 稀疏度，每个任务激活的参数比例（0.5 = 50%）
- `--merge_action_head`: 可选，是否融合 action_head（GROOT 使用 cross-attention，可以尝试）
- `--narrower_weight` / `--wider_weight`: Task Vector 的权重（默认 0.5）

### 评估融合模型

```bash
python scripts/eval_merged_groot_on_dataset.py \
    --model-path ./outputs/merged_groot/pretrained_model \
    --dataset-root /path/to/dataset \
    --episode 0 \
    --action-chunk-size 32 \
    --visualize
```

## 与旧方法的区别

### MergeVLA vs Two-Stage Adapter (kai0)

| 特性 | MergeVLA | Two-Stage Adapter (kai0) |
|------|----------|--------------------------|
| 适配器类型 | 稀疏激活的 LoRA（任务掩码） | 标准 LoRA / MLP |
| 任务路由 | 支持（通过 task_id） | 不支持 |
| 理论基础 | 论文已发表，可复现 | 无法复现 |
| 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐ |

### MergeVLA vs Expert Merging

| 特性 | MergeVLA | Expert Merging |
|------|----------|----------------|
| 训练需求 | 只需要训练适配层（轻量） | 需要训练所有系数 |
| 适配器 | 稀疏 LoRA（MergeVLA 核心） | 无适配器 |
| 适用场景 | 全量微调的模型 | 从相同 base 微调的模型 |

## 技术细节

### 稀疏 LoRA 适配器

```python
# 为每个任务创建独立的 LoRA 参数
lora_A: (num_tasks, hidden_size, rank)  # 例如 (2, 2048, 16)
lora_B: (num_tasks, rank, hidden_size)  # 例如 (2, 16, 2048)

# 任务掩码：每个任务激活哪些参数
task_masks: (num_tasks, hidden_size)  # 例如 (2, 2048)

# 前向传播时：
# 1. 根据 task_id 选择对应的 LoRA 参数
# 2. 应用任务掩码（稀疏激活）
# 3. 计算 LoRA 输出并添加到输入
```

### Task Vector 计算

```python
# 如果 base_model_path == narrower_path:
τ_narrower = narrower - base = 0  # 因为 base == narrower
τ_wider = wider - base = wider - narrower

# 融合公式：
θ_merged = θ_base + α_narrower * τ_narrower + α_wider * τ_wider
         = θ_narrower + α_wider * (θ_wider - θ_narrower)
```

## 故障排除

### 如果推理时动作震荡

1. **检查适配层是否正常加载**：
   ```bash
   # 查看 merge_config.json 中的 adapter_type 和 lora_rank
   cat outputs/merged_groot/pretrained_model/merge_config.json
   ```

2. **尝试不同的稀疏度**：
   ```bash
   --sparsity 0.3  # 更稀疏（30% 激活）
   --sparsity 0.7  # 更密集（70% 激活）
   ```

3. **增加训练轮数**：
   ```bash
   --adapter_epochs 50  # 增加训练轮数
   ```

4. **检查数据是否包含任务标签**：
   - 确保数据加载器返回 `task_source` 字段（0=narrower, 1=wider）

## 参考文献

- MergeVLA 论文: https://arxiv.org/pdf/2511.18810
- MergeVLA 项目页面: https://mergevla.github.io/
