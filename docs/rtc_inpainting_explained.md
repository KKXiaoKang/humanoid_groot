# RTC Inpainting 机制详解

## 1. 整体流程概览

### 1.1 触发条件
当 `action_queue.qsize() <= get_actions_threshold` 时，触发新动作生成。

**关键点**：此时队列中还有未执行完的动作，这些剩余动作将成为下一个chunk生成的引导信号。

### 1.2 数据流
```
当前chunk执行中 → 队列大小 <= 阈值 → 获取剩余动作 → 生成新chunk（带引导）→ 合并到队列
```

## 2. 剩余动作的获取与传递

### 2.1 获取剩余动作

```python
# 在 eval_offline_rtc_resampled.py 第 409-410 行
action_index_before_inference = action_queue.get_action_index()  # 获取当前执行到的索引
prev_actions = action_queue.get_left_over()  # 获取剩余动作
```

**`get_left_over()` 的实现**（action_queue.py 第 119-133 行）：
```python
def get_left_over(self) -> Tensor | None:
    """获取未执行的原始动作（用于RTC inpainting）"""
    with self.lock:
        if self.original_queue is None:
            return None
        # 返回从 original_last_index 开始的所有剩余动作
        return self.original_queue[self.original_last_index :]
```

**关键变量说明**：
- `original_queue`: 存储原始10Hz动作（未重采样）
- `original_last_index`: 当前在10Hz时间尺度上执行到的索引
- `last_index`: 当前在100Hz时间尺度上执行到的索引

**索引更新逻辑**（action_queue.py 第 82-83 行）：
```python
if self.last_index % 10 == 0:
    self.original_last_index = self.last_index // 10
```
每执行10个100Hz动作，更新一次10Hz索引。

### 2.2 传递到Policy

```python
# 在 eval_offline_rtc_resampled.py 第 440-444 行
actions = policy.predict_action_chunk(
    processed_observation,
    inference_delay=inference_delay,        # 推理延迟步数
    prev_chunk_left_over=prev_actions,     # 上一块的剩余动作 ⭐
)
```

## 3. Flow-Matching 中的 RTC Inpainting

### 3.1 Flow-Matching 基础

Flow-Matching 是一种生成模型，通过求解常微分方程（ODE）来生成动作序列：

```
dx/dt = v_θ(x_t, t, o)
```

其中：
- `x_t`: 时刻 t 的动作状态（噪声 → 真实动作）
- `v_θ`: 速度场（神经网络预测）
- `o`: 观测（图像、状态等）
- `t`: 时间参数，从 1（纯噪声）到 0（真实动作）

**去噪过程**（flow_matching_action_head.py 第 569-593 行）：
```python
x_t = torch.randn(...)  # 初始化：纯噪声
dt = 1.0 / num_steps    # 时间步长

for t in range(num_steps):
    t_cont = t / float(num_steps)  # t: 1 → 0
    v_t = denoise_step(x_t, t_cont)  # 预测速度场
    x_t = x_t + dt * v_t  # 欧拉步进：x_{t+1} = x_t + dt * v_t
```

### 3.2 RTC 引导机制

**关键理解**：RTC 不是替换动作，而是**引导速度场**，让生成的动作序列与上一块的剩余动作平滑衔接。

#### 3.2.1 引导公式推导

在 `RTCProcessor.denoise_step()` 中（modeling_rtc.py 第 116-248 行）：

**步骤1：预测下一步状态**
```python
# 第 215 行
x1_t = x_t - time * v_t
```

**数学含义**：
- `x_t`: 当前去噪状态（在时间 t）
- `v_t`: 原始速度场预测
- `x1_t`: 预测的下一步状态（如果按照当前速度场，下一步会到哪里）

**注意**：在 Flow-Matching 中，时间从 1（噪声）到 0（真实），所以：
- `x1_t = x_t - time * v_t` 表示向前预测（向真实动作方向）

**步骤2：计算误差**
```python
# 第 216 行
err = (prev_chunk_left_over - x1_t) * weights
```

**数学含义**：
- `prev_chunk_left_over`: 上一块的剩余动作（目标）
- `x1_t`: 当前预测的下一步状态
- `err`: 预测与目标的误差
- `weights`: 前缀权重（哪些时间步需要对齐）

**步骤3：计算修正梯度**
```python
# 第 218 行
correction = torch.autograd.grad(x1_t, x_t, grad_outputs=err, retain_graph=False)[0]
```

**数学含义**：
- 通过自动微分，计算如何修改 `x_t` 才能让 `x1_t` 更接近 `prev_chunk_left_over`
- `correction = ∂x1_t/∂x_t · err`（梯度 × 误差）

**步骤4：计算引导权重**
```python
# 第 220-226 行
tau = 1 - time  # 归一化时间（在RTC论文中，tau从0到1）
squared_one_minus_tau = (1 - tau) ** 2
inv_r2 = (squared_one_minus_tau + tau**2) / squared_one_minus_tau
c = (1 - tau) / tau
guidance_weight = c * inv_r2
guidance_weight = min(guidance_weight, max_guidance_weight)
```

**数学公式**：
```
τ = 1 - t                    # 归一化时间（0→1）
c = (1-τ) / τ                # 基础权重系数
inv_r² = ((1-τ)² + τ²) / (1-τ)²  # 逆半径平方
λ = c · inv_r²               # 引导权重
λ = min(λ, λ_max)            # 限制最大权重
```

**物理含义**：
- `τ` 接近 0（去噪初期，噪声多）：引导权重小（因为预测不准）
- `τ` 接近 1（去噪后期，接近真实）：引导权重大（因为预测更准）
- `inv_r²` 项考虑了流形的曲率

**步骤5：应用修正**
```python
# 第 228 行
result = v_t - guidance_weight * correction
```

**最终公式**：
```
v_t^guided = v_t - λ · correction
```

其中：
- `v_t`: 原始速度场（无引导）
- `correction`: 修正项（让预测更接近上一块剩余动作）
- `λ`: 引导权重（时间相关的权重）
- `v_t^guided`: 引导后的速度场

### 3.3 前缀权重（Prefix Weights）

**作用**：决定哪些时间步需要与上一块对齐。

```python
# modeling_rtc.py 第 204-209 行
weights = self.get_prefix_weights(
    inference_delay,      # 推理延迟步数
    execution_horizon,    # 执行视野（通常10-16）
    action_chunk_size     # 动作块大小
)
```

**权重调度类型**（modeling_rtc.py 第 250-269 行）：

1. **ZEROS**: 前 `inference_delay` 步权重为1，其余为0
2. **ONES**: 前 `execution_horizon` 步权重为1，其余为0
3. **LINEAR**: 线性衰减
4. **EXP**: 指数衰减（默认）

**示例**（EXP调度，inference_delay=2, execution_horizon=10）：
```
weights = [1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0, ...]
          ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
          inference_delay=2    execution_horizon=10
```

**含义**：
- 前 `inference_delay` 步：完全对齐（权重=1）
- `inference_delay` 到 `execution_horizon` 步：逐渐衰减
- 之后：不引导（权重=0）

## 4. 引导是速度场引导，不是动作替换

### 4.1 为什么是速度场引导？

**关键理解**：Flow-Matching 生成的是**轨迹**，不是单个动作。

如果直接替换动作：
- ❌ 会导致轨迹不连续
- ❌ 违反物理约束（加速度突变）
- ❌ 破坏生成模型的概率流

通过引导速度场：
- ✅ 保持轨迹的平滑性
- ✅ 尊重生成模型的概率分布
- ✅ 自然地在流形上插值

### 4.2 引导过程可视化

```
去噪过程（无引导）：
x_1 (噪声) → v_1 → x_0.9 → v_0.9 → ... → x_0 (真实动作)

去噪过程（有引导）：
x_1 (噪声) → v_1^guided → x_0.9 → v_0.9^guided → ... → x_0 (对齐后的动作)
                ↑                        ↑
            correction              correction
            (让x_0.9接近            (让x_0.8接近
             prev_left_over)         prev_left_over)
```

### 4.3 数学保证

引导后的速度场仍然满足：
- **连续性**：`v_t^guided` 是 `v_t` 的平滑修正
- **可积性**：`∫ v_t^guided dt` 仍然定义良好的轨迹
- **概率性**：修正项通过梯度计算，保持概率流的合理性

## 5. ActionQueue Merge 合并逻辑

### 5.1 合并时机

在生成新动作后（eval_offline_rtc_resampled.py 第 488-490 行）：
```python
action_queue.merge(
    original_actions,        # 原始10Hz动作（用于RTC计算）
    postprocessed_resampled, # 重采样后的100Hz动作（用于执行）
    new_delay,              # 实际推理延迟步数
    action_index_before_inference  # 推理开始时的索引
)
```

### 5.2 RTC 模式下的替换逻辑

```python
# action_queue.py 第 162-181 行
def _replace_actions_queue(self, original_actions, processed_actions, real_delay):
    """RTC模式：替换队列，跳过推理延迟"""
    
    # 原始10Hz动作：跳过 real_delay 个时间步
    self.original_queue = original_actions[real_delay:].clone()
    
    # 重采样后的100Hz动作：跳过 real_delay*10 个时间步
    self.queue = processed_actions[real_delay*10:].clone()
    
    # 重置索引
    self.last_index = 0
    self.original_last_index = 0
```

**为什么跳过 `real_delay` 个动作？**

在推理期间（假设 `real_delay=3` 步）：
- 机器人继续执行旧动作
- 当新动作生成时，已经执行了3个10Hz动作
- 所以新动作的前3步已经"过期"，需要跳过

**时间线示例**：
```
时间轴（10Hz）:  0    1    2    3    4    5    6    7    8    9
旧动作:         [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
执行状态:       ✓    ✓    ✓    → 开始推理 → 推理完成
新动作生成:     [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9]
                ↑    ↑    ↑    ↑
                已执行  需要跳过

跳过后:         [b3, b4, b5, b6, b7, b8, b9]  ← 从b3开始执行
```

### 5.3 非RTC模式下的追加逻辑

```python
# action_queue.py 第 183-204 行
def _append_actions_queue(self, original_actions, processed_actions):
    """非RTC模式：追加到队列"""
    
    if self.queue is None:
        # 首次添加
        self.original_queue = original_actions.clone()
        self.queue = processed_actions.clone()
        return
    
    # 移除已消费的动作
    self.original_queue = self.original_queue[self.last_index:]
    self.queue = self.queue[self.last_index:]
    
    # 追加新动作
    self.original_queue = torch.cat([self.original_queue, original_actions])
    self.queue = torch.cat([self.queue, processed_actions])
    
    # 重置索引
    self.last_index = 0
```

## 6. RTC 技巧如何串联

### 6.1 完整流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 执行阶段                                                    │
│    Actor线程以100Hz执行动作                                    │
│    last_index: 0 → 1 → 2 → ... → 10 → 11 → ...              │
│    original_last_index: 0 → 1 → 2 → ... (每10步更新一次)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 触发条件                                                    │
│    action_queue.qsize() <= threshold (例如15)                │
│    此时队列中还有未执行完的动作                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 获取剩余动作                                                │
│    prev_actions = action_queue.get_left_over()                │
│    返回: original_queue[original_last_index:]                 │
│    例如: [a5, a6, a7, a8, a9] (剩余5个10Hz动作)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 计算推理延迟                                                │
│    inference_delay = ceil(latency / time_per_chunk)           │
│    例如: latency=0.35s, time_per_chunk=0.1s → delay=4        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 生成新动作（带RTC引导）                                      │
│    policy.predict_action_chunk(                              │
│        observation,                                          │
│        inference_delay=4,                                    │
│        prev_chunk_left_over=[a5, a6, a7, a8, a9]             │
│    )                                                          │
│                                                               │
│    在Flow-Matching去噪过程中：                                │
│    - 每个去噪步骤调用 rtc_processor.denoise_step()            │
│    - 计算 correction = grad(x1_t, x_t) · err                  │
│    - 应用 v_t^guided = v_t - λ · correction                  │
│    - 结果：新动作的前几个时间步与 [a5, a6, a7, a8, a9] 对齐  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. 后处理与重采样                                              │
│    - Postprocessor: 反归一化                                 │
│    - Resample: 10Hz → 100Hz (手臂线性插值，爪子零阶保持)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. 合并到队列（替换模式）                                       │
│    action_queue.merge(                                        │
│        original_actions=[b0, b1, ..., b9],                    │
│        processed_actions=[b0_100Hz, ..., b9_100Hz],         │
│        real_delay=4                                          │
│    )                                                          │
│                                                               │
│    替换后：                                                    │
│    original_queue = [b4, b5, b6, b7, b8, b9]  (跳过前4个)    │
│    queue = [b4_100Hz, ..., b9_100Hz]  (跳过前40个)           │
│    last_index = 0  (从头开始执行)                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. 继续执行                                                    │
│    Actor线程继续从新队列取动作执行                              │
│    新动作的前几个时间步已经与上一块剩余动作对齐，实现平滑过渡    │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 关键设计点

1. **双队列设计**：
   - `original_queue`: 10Hz原始动作（用于RTC计算）
   - `queue`: 100Hz重采样动作（用于执行）

2. **索引同步**：
   - `last_index`: 100Hz执行索引
   - `original_last_index`: 10Hz执行索引（每10步更新）

3. **延迟补偿**：
   - 跳过 `real_delay` 个动作，补偿推理时间
   - 确保动作的时效性

4. **引导机制**：
   - 不是硬替换，而是软引导
   - 通过梯度修正速度场
   - 保持轨迹的平滑性和物理合理性

## 7. 数学公式总结

### 7.1 Flow-Matching 基础
```
dx/dt = v_θ(x_t, t, o)
x_{t+1} = x_t + dt · v_t
```

### 7.2 RTC 引导
```
x1_t = x_t - time · v_t                    # 预测下一步
err = (prev_chunk_left_over - x1_t) · w   # 加权误差
correction = ∂x1_t/∂x_t · err             # 梯度修正
λ = min((1-τ)/τ · inv_r², λ_max)          # 引导权重
v_t^guided = v_t - λ · correction         # 引导速度场
```

### 7.3 符号说明

| 符号 | 含义 | 维度 |
|------|------|------|
| `x_t` | 时刻t的动作状态（去噪中） | (B, T, A) |
| `v_t` | 原始速度场 | (B, T, A) |
| `v_t^guided` | 引导后的速度场 | (B, T, A) |
| `prev_chunk_left_over` | 上一块剩余动作 | (B, T_prev, A) |
| `x1_t` | 预测的下一步状态 | (B, T, A) |
| `err` | 预测误差 | (B, T, A) |
| `weights` | 前缀权重 | (T,) |
| `correction` | 修正项 | (B, T, A) |
| `λ` | 引导权重（标量） | () |
| `τ` | 归一化时间 (1-t) | () |
| `inference_delay` | 推理延迟步数 | () |
| `execution_horizon` | 执行视野 | () |

其中：
- B: batch size
- T: 动作块大小（时间步数）
- A: 动作维度
- T_prev: 上一块剩余动作的时间步数

## 8. 总结

RTC Inpainting 的核心思想：
1. **不是替换**，而是**引导**：通过修正速度场，让生成的动作自然对齐
2. **时间相关**：引导权重随去噪进度变化（后期权重大）
3. **空间相关**：前缀权重决定哪些时间步需要对齐
4. **延迟补偿**：跳过推理期间已执行的动作，保持实时性

这种设计既保证了动作的平滑性，又实现了实时控制的需求。

