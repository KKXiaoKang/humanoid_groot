# Flow-Matching 模型在 Groot 中的流程解析

## 概述

本文档详细解释 Flow-Matching 模型在 Groot 中的推理流程，特别是处理 32 维 pretrained 模型与 16 维实际动作维度之间的兼容性问题。

## 核心概念

### 维度说明

- **`encoder_action_dim = 32`**: Pretrained 模型的 action encoder 维度（用于兼容 pretrained 权重）
- **`actual_action_dim = 16`**: 实际的动作维度（14 维手臂关节 + 2 维夹爪位置）
- **Padding**: 将 16 维动作 padding 到 32 维，以满足 pretrained action encoder 的输入要求

## Flow-Matching 基本原理

Flow-Matching 是一种生成模型，通过预测速度场（velocity field）来从噪声逐步还原真实数据：

1. **初始化**: 从噪声分布采样初始状态 `x_0 ~ N(0, I)`
2. **速度预测**: 模型预测速度场 `v_θ(x_t, t)`
3. **积分更新**: 使用 Euler 积分更新 `x_{t+1} = x_t + dt * v_θ(x_t, t)`
4. **迭代**: 重复步骤 2-3 直到 `t=1`，得到最终预测

## Groot 中的完整流程

### 训练阶段（Forward）

```
输入数据 (16维) 
    ↓
Processor Padding (16维 → 32维)
    ↓
[16维真实动作 + 16维零填充] = 32维
    ↓
添加噪声: noisy_trajectory = (1-t) * noise + t * actions
    ↓
Action Encoder (32维 → hidden_dim)
    ↓
DiT Model (预测 velocity)
    ↓
Multi-Head Decoder:
  - Arm Decoder → 14维
  - Claw Decoder → 2维
    ↓
预测 velocity (16维)
    ↓
计算 Loss (只对前16维计算，忽略padding部分)
```

**关键点**:
- 训练时，后 16 维始终为 0（padding）
- Loss 只计算前 16 维的 MSE
- Action encoder 接收 32 维输入，但模型学习的是 16 维的 velocity

### 推理阶段（Get Action）

```
初始化:
  actions = randn(B, T, 32)  # 32维随机噪声
  actions[:, :, 16:] = 0.0   # 清零后16维（匹配训练时的padding）
    ↓
┌─────────────────────────────────────┐
│  Denoising Loop (num_steps 次迭代)  │
└─────────────────────────────────────┘
    ↓
Step t:
  1. Action Encoder(actions, t) 
     → 输入: 32维（前16维是噪声，后16维是0）
     → 输出: hidden features
    ↓
  2. DiT Model(hidden features, t)
     → 预测 velocity field
    ↓
  3. Multi-Head Decoder:
     - Arm Decoder → 14维 velocity
     - Claw Decoder → 2维 velocity
     → 合并: pred_velocity (16维)
    ↓
  4. Padding pred_velocity:
     pred_velocity_padded = [pred_velocity (16维), zeros (16维)]
     → 32维
    ↓
  5. Euler 积分更新:
     actions = actions + dt * pred_velocity_padded
     → actions 前16维被更新，后16维保持为0（因为pred_velocity_padded后16维是0）
    ↓
  6. 清零后16维（确保与训练一致）:
     actions[:, :, 16:] = 0.0
    ↓
重复步骤 1-6 直到 t = num_steps
    ↓
返回: actions[:, :, :16]  # 只返回前16维
```

## 详细流程图

### Mermaid 流程图

```mermaid
graph TD
    A[开始推理] --> B[初始化: actions = randn B,T,32]
    B --> C[清零后16维: actions[:,:,16:] = 0]
    C --> D[Denoising Loop: t = 0 to num_steps]
    
    D --> E[Step t: Action Encoder]
    E --> F[输入: actions B,T,32<br/>前16维: 噪声<br/>后16维: 0]
    F --> G[输出: action_features B,T,hidden_dim]
    
    G --> H[DiT Model]
    H --> I[输入: action_features + state + vision]
    I --> J[输出: model_output]
    
    J --> K[Multi-Head Decoder]
    K --> L[Arm Decoder → 14维]
    K --> M[Claw Decoder → 2维]
    L --> N[合并: pred_velocity 16维]
    M --> N
    
    N --> O[Padding: pred_velocity_padded 32维]
    O --> P[前16维: pred_velocity<br/>后16维: zeros]
    
    P --> Q[Euler 积分: actions += dt * pred_velocity_padded]
    Q --> R[清零后16维: actions[:,:,16:] = 0]
    
    R --> S{t < num_steps?}
    S -->|是| D
    S -->|否| T[返回: actions[:,:,:16] 16维]
    
    style F fill:#e1f5ff
    style N fill:#fff4e1
    style O fill:#ffe1f5
    style T fill:#e1ffe1
```

### ASCII 详细流程图（带维度说明）

```
┌─────────────────────────────────────────────────────────────────┐
│                    推理阶段：Get Action                         │
└─────────────────────────────────────────────────────────────────┘

【初始化阶段】
┌─────────────────────────────────────────────────────────────┐
│ Step 0: 初始化                                               │
├─────────────────────────────────────────────────────────────┤
│ actions = randn(B, T, 32)                                    │
│   └─> [前16维: 随机噪声] [后16维: 随机噪声]                  │
│                                                              │
│ actions[:, :, 16:] = 0.0  # 清零后16维                      │
│   └─> [前16维: 随机噪声] [后16维: 0] ← 匹配训练时的padding   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Denoising Loop: t = 0, 1, 2, ..., num_steps-1              │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  Step t: 单个 Denoising 迭代          │
        └──────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. Action Encoder                                           │
├─────────────────────────────────────────────────────────────┤
│ 输入: actions (B, T, 32)                                    │
│   └─> [前16维: 当前denoised状态] [后16维: 0]                │
│                                                              │
│ action_features = action_encoder(actions, t, cat_ids)      │
│   └─> 输出: (B, T, hidden_dim)                             │
│   └─> 注意: encoder内部使用32维权重，但后16维输入是0         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. DiT Model (Diffusion Transformer)                        │
├─────────────────────────────────────────────────────────────┤
│ 输入:                                                         │
│   - action_features (B, T, hidden_dim)                     │
│   - state_features (B, state_dim, hidden_dim)               │
│   - vision_features (B, vision_tokens, hidden_dim)         │
│                                                              │
│ model_output = DiT(                                          │
│     hidden_states=sa_embs,                                  │
│     encoder_hidden_states=vl_embs,                          │
│     timestep=t                                               │
│ )                                                            │
│   └─> 输出: (B, total_seq_len, hidden_dim)                  │
│                                                              │
│ model_output_actions = model_output[:, -T:]                 │
│   └─> 切片出action部分: (B, T, hidden_dim)                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Multi-Head Decoder                                       │
├─────────────────────────────────────────────────────────────┤
│ pred_arm = action_arm_decoder(model_output_actions, cat_ids)│
│   └─> 输出: (B, T, 14)  # 14维手臂关节velocity             │
│                                                              │
│ pred_claw = action_claw_decoder(model_output_actions, cat_ids)│
│   └─> 输出: (B, T, 2)   # 2维夹爪velocity                   │
│                                                              │
│ pred_velocity = torch.cat([pred_arm, pred_claw], dim=-1)    │
│   └─> 合并: (B, T, 16)  # 16维velocity                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Padding Velocity                                         │
├─────────────────────────────────────────────────────────────┤
│ pred_velocity_padded = [pred_velocity (16维), zeros (16维)] │
│   └─> 输出: (B, T, 32)                                      │
│   └─> [前16维: 预测的velocity] [后16维: 0]                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Euler 积分更新                                            │
├─────────────────────────────────────────────────────────────┤
│ actions = actions + dt * pred_velocity_padded                │
│                                                              │
│ 更新前:                                                       │
│   actions: [前16维: x_t] [后16维: 0]                        │
│                                                              │
│ 更新后:                                                       │
│   actions: [前16维: x_t + dt*v] [后16维: 0]                 │
│   └─> 前16维被velocity更新，后16维保持为0                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. 清零后16维（关键步骤！）                                  │
├─────────────────────────────────────────────────────────────┤
│ actions[:, :, 16:] = 0.0                                    │
│                                                              │
│ 原因: 确保与训练时一致                                       │
│   - 训练时: 后16维始终为0（processor padding）             │
│   - 推理时: 后16维也必须为0（匹配训练输入格式）             │
│   - 如果不清零，action_encoder会收到不一致的输入，导致抖动   │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    ┌──────────────┐
                    │ t < num_steps?│
                    └──────────────┘
                    ↙            ↘
               [是]              [否]
                ↓                 ↓
          [继续循环]      ┌──────────────────────┐
                         │ 返回最终结果          │
                         └──────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. 返回结果                                                  │
├─────────────────────────────────────────────────────────────┤
│ actions_output = actions[:, :, :16]                         │
│   └─> 只返回前16维: (B, T, 16)                              │
│   └─> [14维手臂关节 + 2维夹爪位置]                         │
└─────────────────────────────────────────────────────────────┘
```

## 关键设计点

### 1. Padding 的作用

**训练时**:
- Processor 将 16 维动作 padding 到 32 维
- 后 16 维始终为 0
- Action encoder 接收 32 维输入（兼容 pretrained 权重）

**推理时**:
- 初始化 32 维噪声，但后 16 维清零
- 每次更新后，后 16 维再次清零
- 确保 `action_encoder` 的输入格式与训练时一致

### 2. Velocity 预测

- 模型预测的是 **velocity**（速度场），不是直接的动作值
- Velocity 维度是 16 维（actual_action_dim）
- 通过 Euler 积分累积 velocity 得到最终动作

### 3. 维度转换

```
推理流程中的维度变化:
32维噪声 (初始化)
  ↓ [清零后16维]
32维 (前16维: 噪声, 后16维: 0)
  ↓ [Action Encoder]
hidden_dim
  ↓ [DiT + Multi-Head Decoder]
16维 velocity
  ↓ [Padding]
32维 velocity (前16维: pred, 后16维: 0)
  ↓ [Euler 积分]
32维 actions (前16维: 更新, 后16维: 0)
  ↓ [清零后16维]
32维 actions (前16维: 更新, 后16维: 0)
  ↓ [重复 num_steps 次]
32维 actions (前16维: denoised, 后16维: 0)
  ↓ [切片]
16维最终动作
```

## 为什么需要清零后 16 维？

### 问题场景

如果不清零后 16 维：
1. 初始化时，后 16 维是随机噪声
2. 每次更新，`pred_velocity_padded` 的后 16 维是 0
3. 所以后 16 维会一直保持随机噪声
4. `action_encoder` 接收的输入与训练时不一致（训练时后 16 维是 0）
5. 导致前 16 维的 denoising 质量下降，产生抖动

### 解决方案

在每次 denoising step 后清零后 16 维：
- 确保 `action_encoder` 的输入格式与训练时一致
- 后 16 维始终为 0（匹配训练时的 padding）
- 前 16 维可以正常 denoise

## 代码关键位置

### 1. 初始化（`get_action` 方法）

```python
# 初始化 32 维噪声
actions = torch.randn(size=(batch_size, action_horizon, encoder_action_dim=32))

# 清零后 16 维
if encoder_action_dim != actual_action_dim:
    actions[:, :, actual_action_dim:] = 0.0
```

### 2. Denoising Loop

```python
for t in range(num_steps):
    # 1. Action Encoder (接收 32 维，后 16 维是 0)
    action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
    
    # 2. DiT Model
    model_output = self.model(...)
    
    # 3. Multi-Head Decoder (输出 16 维 velocity)
    pred_arm = self.action_arm_decoder(...)  # 14 维
    pred_claw = self.action_claw_decoder(...)  # 2 维
    pred_velocity = torch.cat([pred_arm, pred_claw], dim=-1)  # 16 维
    
    # 4. Padding 到 32 维
    pred_velocity_padded = torch.cat([pred_velocity, zeros(16维)], dim=-1)
    
    # 5. Euler 积分更新
    actions = actions + dt * pred_velocity_padded
    
    # 6. 清零后 16 维（关键！）
    actions[:, :, actual_action_dim:] = 0.0
```

### 3. 返回结果

```python
# 只返回前 16 维
actions_output = actions[:, :, :actual_action_dim]  # (B, T, 16)
return BatchFeature(data={"action_pred": actions_output})
```

## 训练 vs 推理对比

| 阶段 | Action Encoder 输入 | Velocity 预测 | 输出维度 | Padding 处理 |
|------|-------------------|-------------|---------|-------------|
| **训练** | 32 维（16 维真实 + 16 维零填充） | 16 维 velocity | 16 维 | Processor 自动 padding |
| **推理** | 32 维（16 维噪声 + 16 维零填充） | 16 维 velocity | 16 维 | 手动清零后 16 维 |

## 维度变化详细追踪

### 单个 Denoising Step 的维度变化

```
┌─────────────────────────────────────────────────────────────┐
│ 输入: actions (B, T, 32)                                    │
│   [0:16]  ← 前16维: 当前denoised状态                        │
│   [16:32] ← 后16维: 0 (padding)                            │
└─────────────────────────────────────────────────────────────┘
                    ↓ Action Encoder
┌─────────────────────────────────────────────────────────────┐
│ action_features (B, T, hidden_dim)                          │
│   └─> 32维输入被编码为hidden_dim维特征                       │
└─────────────────────────────────────────────────────────────┘
                    ↓ DiT Model
┌─────────────────────────────────────────────────────────────┐
│ model_output (B, total_seq_len, hidden_dim)                 │
│   └─> 包含state + vision + action的完整序列                 │
└─────────────────────────────────────────────────────────────┘
                    ↓ Slice Action Part
┌─────────────────────────────────────────────────────────────┐
│ model_output_actions (B, T, hidden_dim)                     │
│   └─> 只取action部分                                        │
└─────────────────────────────────────────────────────────────┘
                    ↓ Multi-Head Decoder
┌─────────────────────────────────────────────────────────────┐
│ pred_arm (B, T, 14)  ← 手臂velocity                        │
│ pred_claw (B, T, 2)  ← 夹爪velocity                        │
│                                                              │
│ pred_velocity = cat([pred_arm, pred_claw])                   │
│   └─> (B, T, 16)  ← 16维velocity                           │
└─────────────────────────────────────────────────────────────┘
                    ↓ Padding
┌─────────────────────────────────────────────────────────────┐
│ pred_velocity_padded (B, T, 32)                             │
│   [0:16]  ← pred_velocity (16维)                            │
│   [16:32] ← zeros (16维)                                    │
└─────────────────────────────────────────────────────────────┘
                    ↓ Euler Integration
┌─────────────────────────────────────────────────────────────┐
│ actions = actions + dt * pred_velocity_padded                │
│                                                              │
│ 更新后:                                                       │
│   [0:16]  ← 前16维被velocity更新                            │
│   [16:32] ← 后16维保持为0（因为pred_velocity_padded后16维是0）│
└─────────────────────────────────────────────────────────────┘
                    ↓ Zero Out Padding
┌─────────────────────────────────────────────────────────────┐
│ actions[:, :, 16:] = 0.0  # 确保一致性                       │
│                                                              │
│ 最终:                                                         │
│   [0:16]  ← 更新后的denoised状态                            │
│   [16:32] ← 0 (确保与训练时一致)                             │
└─────────────────────────────────────────────────────────────┘
```

### 完整 Denoising 过程的维度变化

```
t=0:  actions = [噪声(16维), 0(16维)]  → 32维
t=1:  actions = [denoised_1(16维), 0(16维)]  → 32维
t=2:  actions = [denoised_2(16维), 0(16维)]  → 32维
...
t=N:  actions = [denoised_N(16维), 0(16维)]  → 32维
      ↓
返回: actions[:, :, :16]  → 16维最终动作
```

## Padding 的关键作用

### 为什么需要 Padding？

1. **Pretrained 模型兼容性**
   - Pretrained action encoder 期望 32 维输入
   - 直接修改 encoder 维度会破坏 pretrained 权重
   - Padding 允许复用 pretrained encoder

2. **训练时的处理**
   - Processor 自动将 16 维动作 padding 到 32 维
   - 后 16 维始终为 0
   - Action encoder 接收 32 维输入，但只学习前 16 维的 velocity

3. **推理时的处理**
   - 初始化 32 维噪声，但后 16 维清零
   - 每次更新后，后 16 维再次清零
   - 确保 action encoder 的输入格式与训练时完全一致

### Padding 在流程中的位置

```
训练时:
  真实动作(16维) 
    → Processor Padding 
    → [真实动作(16维), 0(16维)] = 32维
    → Action Encoder(32维)
    → 预测velocity(16维)
    → Loss计算(只对16维)

推理时:
  随机噪声(32维)
    → 清零后16维
    → [噪声(16维), 0(16维)] = 32维
    → Action Encoder(32维)
    → 预测velocity(16维)
    → Padding velocity到32维
    → 更新actions(32维)
    → 清零后16维
    → 重复...
    → 返回前16维
```

## 总结

1. **Padding 的作用**: 兼容 pretrained 32 维 action encoder，同时支持 16 维实际动作
2. **清零后 16 维**: 确保推理时 `action_encoder` 的输入格式与训练时一致
3. **Velocity 预测**: 模型预测 16 维 velocity，通过 Euler 积分逐步 denoise
4. **维度流程**: 32 维（内部处理）→ 16 维（velocity 预测）→ 32 维（更新）→ 16 维（最终输出）

这种设计允许模型在保持与 pretrained 权重兼容的同时，学习并预测实际需要的 16 维动作空间。

## 常见问题

### Q1: 为什么不能直接用 16 维？

A: Pretrained action encoder 的权重是 32 维的，直接修改会破坏权重。Padding 允许我们复用 pretrained 权重。

### Q2: 清零后 16 维是否会影响前 16 维的 denoising？

A: 不会。清零后 16 维是为了确保 action encoder 的输入格式与训练时一致。训练时后 16 维就是 0，所以推理时也应该是 0。

### Q3: 为什么预测的是 velocity 而不是直接的动作？

A: Flow-Matching 模型学习的是速度场（velocity field），通过积分速度场来生成动作。这种方式比直接预测动作更稳定，能够生成更平滑的轨迹。

### Q4: 如果不清零后 16 维会怎样？

A: 后 16 维会一直保持随机噪声，导致 action encoder 的输入与训练时不一致，进而影响前 16 维的 denoising 质量，产生抖动。

