# Flow Matching 训练与推理流程详解

## 问题核心

在推理时，作为 `hidden_states` 的 `action_features` 是从哪里来的？因为 action 应该是模型生成的输出，而不是输入。

## 答案概览

**是的，你的理解完全正确！**

- **训练时**：`action_features` 来自**加噪后的真实 action**（`noisy_trajectory`）
- **推理时**：`action_features` 来自**随机噪声**，通过多步迭代去噪逐步生成真实的 action

## 详细流程解析

### 训练流程（`forward` 方法）

```python
def forward(self, backbone_output, action_input):
    # 1. 获取真实的 action（ground truth）
    actions = action_input.action  # (B, T, encoder_action_dim=32)
    
    # 2. 生成随机噪声
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
    
    # 3. 随机采样时间步 t ∈ [0, 1]
    t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
    t = t[:, None, None]  # shape (B,1,1)
    
    # 4. 创建加噪轨迹（Flow Matching 的核心）
    # noisy_trajectory = (1 - t) * noise + t * actions
    # 当 t=0 时：noisy_trajectory = noise（纯噪声）
    # 当 t=1 时：noisy_trajectory = actions（真实 action）
    noisy_trajectory = (1 - t) * noise + t * actions
    
    # 5. 计算目标 velocity（用于监督学习）
    # velocity = actions - noise（从噪声到真实 action 的方向）
    velocity = actions[:, :, :self.actual_action_dim] - noise[:, :, :self.actual_action_dim]
    
    # 6. 将加噪轨迹编码为 action_features
    t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
    action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
    
    # 7. 拼接 state + future_tokens + action_features
    sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
    
    # 8. DiT Cross-Attention
    model_output = self.model(
        hidden_states=sa_embs,  # Query: state + future + action_features
        encoder_hidden_states=vl_embs,  # Key/Value: vision-language
        timestep=t_discretized,
    )
    
    # 9. 解码预测的 velocity
    pred_velocity = self.action_decoder(model_output_actions, embodiment_id)
    
    # 10. 计算损失：预测的 velocity vs 真实的 velocity
    loss = F.mse_loss(pred_velocity, velocity)
```

**关键点**：
- `noisy_trajectory` 是**加噪后的真实 action**，作为 `action_features` 的输入
- 模型学习预测 `velocity = actions - noise`（从噪声到真实 action 的方向）
- 这是**条件生成**：给定 `noisy_trajectory` 和时间步 `t`，预测去噪方向

### 推理流程（`get_action` 方法）

```python
@torch.no_grad()
def get_action(self, backbone_output, action_input, rtc_enabled=False, **kwargs):
    # 1. 获取 vision-language 和 state（这些是已知的）
    vl_embs = backbone_output.backbone_features
    state_features = self.state_encoder(action_input.state, embodiment_id)
    
    # 2. 初始化：从随机噪声开始
    # 这是关键！推理时没有真实的 action，所以从纯噪声开始
    actions = torch.randn(
        size=(batch_size, self.config.action_horizon, self.encoder_action_dim),
        dtype=vl_embs.dtype,
        device=device,
    )
    x_t = actions  # x_t 是当前时刻的 action（初始为纯噪声）
    
    # 3. 迭代去噪（Flow Matching 的推理过程）
    num_steps = self.num_inference_timesteps  # 例如：20 步
    dt = 1.0 / num_steps  # 每步的时间增量
    
    for t in range(num_steps):
        # 3.1 计算当前时间步（从 0 到 1）
        t_cont = t / float(num_steps)  # 例如：0, 0.05, 0.1, ..., 0.95
        t_discretized = int(t_cont * self.num_timestep_buckets)
        
        # 3.2 预测 velocity（去噪方向）
        # 这里使用当前的 x_t（噪声）作为 action_features 的输入
        v_t = self.denoise_step(
            x_t=x_t,  # 当前的 action（噪声）
            timestep=t_discretized,
            vl_embs=vl_embs,
            state_features=state_features,
            embodiment_id=embodiment_id
        )
        
        # 3.3 更新 action（沿着 velocity 方向移动）
        # x_{t+1} = x_t + dt * v_t
        # 这是 Euler 积分，逐步从噪声走向真实 action
        x_t = x_t + dt * v_t
    
    # 4. 返回最终生成的 action
    actions_output = x_t[:, :, :self.actual_action_dim]
    return BatchFeature(data={"action_pred": actions_output})
```

**关键点**：
- 初始 `x_t` 是**纯随机噪声**
- 每一步都使用当前的 `x_t`（噪声）作为 `action_features` 的输入
- 模型预测 `velocity`（去噪方向）
- 通过 `x_t = x_t + dt * v_t` 逐步更新，从噪声走向真实 action

### `denoise_step` 和 `_predict_velocity` 详解

```python
def denoise_step(self, x_t, timestep, vl_embs, state_features, embodiment_id):
    """单步预测 velocity"""
    timesteps_tensor = torch.full(size=(batch_size,), fill_value=timestep, device=x_t.device)
    v_t = self._predict_velocity(vl_embs, state_features, x_t, timesteps_tensor, embodiment_id)
    return v_t

def _predict_velocity(self, vl_embs, state_features, actions, timesteps_tensor, embodiment_id):
    """预测 velocity field"""
    # 1. 将当前的 actions（噪声）编码为 action_features
    action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
    
    # 2. 拼接 state + future_tokens + action_features
    sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
    
    # 3. DiT Cross-Attention
    model_output = self.model(
        hidden_states=sa_embs,  # Query: state + future + action_features（来自噪声）
        encoder_hidden_states=vl_embs,  # Key/Value: vision-language
        timestep=timesteps_tensor,
    )
    
    # 4. 解码预测的 velocity
    pred_velocity = self.action_decoder(model_output_actions, embodiment_id)
    
    return pred_velocity
```

**关键点**：
- `actions` 参数是**当前的 action 状态**（训练时是加噪轨迹，推理时是迭代中的噪声）
- `action_features` 来自这个 `actions`，作为 Query 的一部分
- 模型预测 `velocity`，表示"应该朝哪个方向移动"

## 流程图对比

### 训练流程

```
输入：
  - Vision: 摄像头图像
  - Language: 任务描述
  - State: 机器人状态
  - Action (Ground Truth): 真实的动作序列

流程：
  1. Vision + Language → vl_embs (encoder_hidden_states)
  2. State → state_features
  3. Action + Noise + Time → noisy_trajectory
  4. noisy_trajectory → action_features
  5. [state_features, future_tokens, action_features] → sa_embs (hidden_states)
  6. DiT Cross-Attention: Query(sa_embs) 关注 Key/Value(vl_embs)
  7. model_output → pred_velocity
  8. Loss: pred_velocity vs (actions - noise)
```

### 推理流程

```
输入：
  - Vision: 摄像头图像
  - Language: 任务描述（hard code）
  - State: 机器人传感器状态
  - Action: ❌ 不存在（需要生成）

流程：
  初始化：
    x_t = 随机噪声 (B, T, action_dim)
  
  迭代去噪（num_steps 步）：
    For t in range(num_steps):
      1. Vision + Language → vl_embs (encoder_hidden_states) [只计算一次]
      2. State → state_features [每步相同]
      3. x_t（当前噪声）→ action_features [每步更新]
      4. [state_features, future_tokens, action_features] → sa_embs
      5. DiT Cross-Attention: Query(sa_embs) 关注 Key/Value(vl_embs)
      6. model_output → pred_velocity
      7. x_t = x_t + dt * pred_velocity [更新 action]
  
  输出：
    actions_output = x_t（最终生成的 action）
```

## 关键理解

### 1. **Flow Matching 的本质**

Flow Matching 是一种**生成模型**，类似于 Diffusion Model：

- **训练**：学习从噪声到真实数据的"速度场"（velocity field）
- **推理**：从噪声开始，沿着速度场逐步"流动"到真实数据

### 2. **为什么需要 action_features 作为输入？**

在 Flow Matching 中，模型需要知道：
- **当前位置**：`x_t`（当前的 action 状态，可能是噪声）
- **时间步**：`t`（在去噪过程中的位置）
- **条件信息**：Vision-Language 和 State（指导生成方向）

模型根据这些信息预测**应该朝哪个方向移动**（velocity）。

### 3. **训练和推理的一致性**

训练和推理使用**相同的模型结构**：
- 都使用 `action_encoder` 编码 action
- 都使用 DiT Cross-Attention
- 都使用 `action_decoder` 解码 velocity

区别在于：
- **训练**：`action_features` 来自加噪的真实 action
- **推理**：`action_features` 来自迭代中的噪声（逐步去噪）

### 4. **为什么 Vision-Language 只需要计算一次？**

在推理时，Vision-Language 特征（`vl_embs`）在每次调用 `get_action` 时是**固定的**：
- Vision：当前摄像头图像
- Language：当前任务描述

这些信息在整个去噪过程中不变，所以只需要计算一次，作为 `encoder_hidden_states` 缓存起来。

## 代码位置总结

| 阶段 | 方法 | 关键代码行 | 说明 |
|------|------|-----------|------|
| **训练** | `forward` | 613-625 | 生成加噪轨迹，编码为 action_features |
| **训练** | `forward` | 649-660 | DiT Cross-Attention，预测 velocity |
| **推理** | `get_action` | 846-856 | 初始化随机噪声 |
| **推理** | `get_action` | 861-888 | 迭代去噪循环 |
| **推理** | `denoise_step` | 897-906 | 单步去噪调用 |
| **推理** | `_predict_velocity` | 917-951 | 预测 velocity（与训练时相同） |

## 总结

你的理解完全正确：

1. ✅ **训练时**：`action_features` 来自**加噪后的真实 action**（`noisy_trajectory`）
2. ✅ **推理时**：`action_features` 来自**随机噪声**，通过多步迭代去噪逐步生成真实的 action
3. ✅ **模型作用**：预测 `velocity`（从当前位置到真实 action 的方向）
4. ✅ **去噪过程**：`x_t = x_t + dt * v_t`，逐步从噪声走向真实 action

这是 Flow Matching / Diffusion Model 的标准流程，模型学习的是"速度场"（velocity field），而不是直接预测 action。

