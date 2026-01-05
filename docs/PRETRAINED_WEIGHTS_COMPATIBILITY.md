# Pretrained权重兼容性分析

## 问题

用户想知道修改 `num_target_vision_tokens` 和 `vl_self_attention_cfg` 是否会破坏 pretrained 权重。

## 权重加载机制

从 `groot_n1.py` 的 `from_pretrained` 方法可以看到：

```python
pretrained_model.load_state_dict(state_dict, strict=False)
```

**关键点**：使用 `strict=False`，这意味着：
- 如果某些层的形状不匹配，会**跳过**这些层
- 跳过的层会使用**随机初始化**
- 不会报错，但会丢失这些层的pretrained权重

## 参数分析

### 1. `num_target_vision_tokens` ✅ 安全修改

**代码位置**：
```python
self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
```

**分析**：
- `future_tokens` 是一个**新初始化的** `nn.Embedding`
- 不依赖pretrained权重
- 修改 `num_target_vision_tokens` 只会改变embedding的大小
- **完全安全**，不会破坏pretrained权重

**修改建议**：
```json
{
  "action_head_cfg": {
    "num_target_vision_tokens": 64  // 从32增加到64，安全
  }
}
```

### 2. `vl_self_attention_cfg` ⚠️ 部分安全

**代码位置**：
```python
self.vl_self_attention = (
    SelfAttentionTransformer(**config.vl_self_attention_cfg) if config.use_vlln else nn.Identity()
)
```

**分析**：
- `vl_self_attention` 是通过 `SelfAttentionTransformer` 创建的
- 如果改变**层数**或**维度**，会创建新的层
- 由于 `strict=False`，如果形状不匹配，pretrained权重会被**跳过**
- 新层会使用**随机初始化**

**可以安全修改的参数**：
- ✅ `dropout`: 不影响权重形状
- ✅ `final_dropout`: 不影响权重形状
- ✅ `attention_bias`: 只影响bias参数，不影响主要权重
- ✅ `activation_fn`: 不影响权重形状
- ✅ `positional_embeddings`: 不影响权重形状

**会破坏pretrained权重的参数**：
- ❌ `num_layers`: 改变层数，会创建新层，pretrained权重会被跳过
- ❌ `num_attention_heads`: 改变注意力头数，权重形状不匹配
- ❌ `attention_head_dim`: 改变头维度，权重形状不匹配
- ❌ `output_dim`: 改变输出维度，权重形状不匹配

## 解决方案

### 方案1: 只修改 `num_target_vision_tokens`（推荐）

**优点**：
- ✅ 完全安全，不会破坏pretrained权重
- ✅ 可以增加视觉token数量，提高对复杂场景的处理能力
- ✅ 实现简单

**缺点**：
- ⚠️ 只增加token数量，不增强视觉-语言融合能力

**实现**：
```json
{
  "action_head_cfg": {
    "num_target_vision_tokens": 64  // 从32增加到64
  }
}
```

### 方案2: 修改 `vl_self_attention_cfg` 但保持兼容

**策略**：只修改不影响权重形状的参数

**可以修改的参数**：
```json
{
  "action_head_cfg": {
    "vl_self_attention_cfg": {
      "num_layers": 4,              // 保持原值，不要改
      "num_attention_heads": 32,   // 保持原值，不要改
      "attention_head_dim": 64,     // 保持原值，不要改
      "dropout": 0.1,               // 可以修改（不影响权重）
      "final_dropout": true,        // 可以修改（不影响权重）
      "positional_embeddings": null // 可以修改（不影响权重）
    }
  }
}
```

### 方案3: 增加层数但使用渐进式训练（高级）

**策略**：增加层数，但使用渐进式训练策略

**步骤**：
1. 先加载pretrained权重（前4层会被加载）
2. 新增的层（5-8层）使用随机初始化
3. 使用较小的学习率，让新层逐步学习
4. 可以冻结前4层，只训练新层

**实现**：
```python
# 1. 修改配置，增加层数
config.vl_self_attention_cfg["num_layers"] = 8  # 从4增加到8

# 2. 加载pretrained权重（前4层会被加载，后4层随机初始化）
model = GR00TN15.from_pretrained(...)

# 3. 冻结前4层，只训练后4层
for i, layer in enumerate(model.action_head.vl_self_attention.transformer_blocks):
    if i < 4:  # 前4层冻结
        layer.requires_grad_(False)
    else:  # 后4层可训练
        layer.requires_grad_(True)
```

**优点**：
- ✅ 可以增强视觉-语言融合能力
- ✅ 保留pretrained权重（前4层）
- ✅ 新层可以逐步学习

**缺点**：
- ⚠️ 需要手动实现冻结逻辑
- ⚠️ 训练时间可能更长

## 推荐方案

### 短期方案（快速解决）

**只修改 `num_target_vision_tokens`**：
```json
{
  "action_head_cfg": {
    "num_target_vision_tokens": 64  // 从32增加到64
  }
}
```

**优点**：
- ✅ 完全安全，不会破坏pretrained权重
- ✅ 实现简单，立即生效
- ✅ 可以增加视觉token数量，提高对复杂场景的处理能力

### 长期方案（根本解决）

**组合使用**：
1. **增加 `num_target_vision_tokens`** (32→64)
2. **增强任务描述**（明确区分"4个箱子"和"6个箱子"）
3. **数据平衡**（过采样6个箱子的数据集）

**如果仍然不够，再考虑**：
- 增加 `vl_self_attention` 层数，但使用渐进式训练策略

## 验证方法

### 检查权重加载情况

```python
# 加载模型
model = GR00TN15.from_pretrained("nvidia/GR00T-N1.5-3B", ...)

# 检查vl_self_attention的权重
for name, param in model.action_head.vl_self_attention.named_parameters():
    print(f"{name}: {param.shape}")

# 如果某些层的权重是随机初始化的，它们的值会很大（因为随机初始化）
# 如果权重是从pretrained加载的，它们的值会相对较小
```

### 检查权重是否匹配

```python
# 加载pretrained权重
pretrained_state_dict = torch.load("pretrained_model.safetensors")

# 检查vl_self_attention的权重
for key in pretrained_state_dict:
    if "vl_self_attention" in key:
        print(f"{key}: {pretrained_state_dict[key].shape}")

# 对比当前模型的权重形状
for name, param in model.action_head.vl_self_attention.named_parameters():
    pretrained_key = f"action_head.{name}"
    if pretrained_key in pretrained_state_dict:
        if param.shape != pretrained_state_dict[pretrained_key].shape:
            print(f"⚠️ 形状不匹配: {name}")
            print(f"   当前: {param.shape}")
            print(f"   Pretrained: {pretrained_state_dict[pretrained_key].shape}")
```

## 总结

| 参数 | 安全修改 | 说明 |
|------|---------|------|
| `num_target_vision_tokens` | ✅ 完全安全 | 新初始化的embedding，不依赖pretrained权重 |
| `vl_self_attention_cfg.num_layers` | ❌ 会破坏 | 改变层数，新层会随机初始化 |
| `vl_self_attention_cfg.num_attention_heads` | ❌ 会破坏 | 改变权重形状，不匹配 |
| `vl_self_attention_cfg.attention_head_dim` | ❌ 会破坏 | 改变权重形状，不匹配 |
| `vl_self_attention_cfg.dropout` | ✅ 安全 | 不影响权重形状 |
| `vl_self_attention_cfg.final_dropout` | ✅ 安全 | 不影响权重形状 |

**推荐**：
1. **优先修改 `num_target_vision_tokens`**（32→64），完全安全
2. **增强任务描述**，明确区分场景
3. **数据平衡**，过采样6个箱子的数据集
4. 如果仍然不够，再考虑增加 `vl_self_attention` 层数，但使用渐进式训练策略

