# π*₀.₆ (RECAP) 完整架构与数据流解析

> **基于论文 [π*₀.₆: a VLA That Learns From Experience](https://www.pi.website/download/pistar06.pdf) 的详细架构分析**

## 核心架构概览

RECAP (RL with Experience and Corrections via Advantage-conditioned Policies) 是一个**离线强化学习（Offline RL）**框架，用于训练 Vision-Language-Action (VLA) 模型。与传统 Actor-Critic 不同，RECAP 使用 **advantage conditioning** 而非策略梯度。

```
┌─────────────────────────────────────────────────────────────────┐
│              π*₀.₆ (RECAP) 完整数据流架构                        │
└─────────────────────────────────────────────────────────────────┘

阶段1: 预训练 (Pre-training)
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 多任务数据集   │ --> │ Value Function │ --> │ Advantage     │
│ (离线)       │     │ 训练          │     │ 计算          │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                      │
                            ▼                      ▼
                    ┌──────────────┐     ┌──────────────┐
                    │ V(s) 预测     │     │ A(o,a) =     │
                    │ (状态价值)    │     │ r_{t:N} +    │
                    │              │     │ V(o_{t+N}) - │
                    │              │     │ V(o_t)       │
                    └──────────────┘     └──────────────┘
                                              │
                                              ▼
                                    ┌──────────────┐
                                    │ Advantage    │
                                    │ Conditioning │
                                    │ Policy       │
                                    │ (π*₀.₆)     │
                                    └──────────────┘

阶段2: 任务特定微调 (Task-specific Fine-tuning)
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 演示数据      │ --> │ 策略微调      │ --> │ 部署执行      │
│ (Demonstrations)│   │ (BC)         │     │ (Autonomous) │
└──────────────┘     └──────────────┘     └──────────────┘
                                              │
                                              ▼
                                    ┌──────────────┐
                                    │ 数据收集      │
                                    │ - 自主rollout │
                                    │ - 专家干预    │
                                    │ - Reward反馈  │
                                    └──────────────┘
                                              │
                                              ▼
                                    ┌──────────────┐
                                    │ Value Function│
                                    │ 微调          │
                                    └──────────────┘
                                              │
                                              ▼
                                    ┌──────────────┐
                                    │ Advantage    │
                                    │ 重新计算      │
                                    └──────────────┘
                                              │
                                              ▼
                                    ┌──────────────┐
                                    │ 策略更新      │
                                    │ (迭代改进)    │
                                    └──────────────┘
```

---

## 详细组件解析

### 1. Actor Policy (策略网络 π*₀.₆)

**身份**：π*₀.₆ VLA 模型（基于 π₀.₆，添加了 advantage conditioning 能力）

**架构组成**：
- **视觉编码器**：SigLIP (400M)
- **语言模型**：Gemma (4B)
- **动作专家**：860M 参数
- **特殊能力**：可以条件化在二值化的 advantage 指标上

**关键特性**：
- 不是传统的 Actor，而是 **advantage-conditioned policy**
- 通过 conditioning 在 advantage 指标上来改进策略，而非策略梯度

---

### 2. Critic Policy (价值函数 V(s))

**身份**：独立的、较小的 VLA 模型，专门用于预测状态价值

**架构组成**：
- **视觉编码器**：SigLIP (400M) - 与策略网络共享
- **语言模型**：Gemma (270M) - **比策略网络小得多**（策略网络是 4B）
- **价值头**：MLP，输出单个标量值 V(s)

**关键特性**：
- ✅ **独立模型**：与策略网络分离，单独训练
- ✅ **预测 V(s)**：状态价值，不是 Q(s,a)
- ✅ **较小规模**：使用较小的 Gemma (270M) 而非 (4B)，降低计算成本

---

## 完整训练流程解析（基于论文 Algorithm 1）

> **参考**：[π*₀.₆: a VLA That Learns From Experience](https://www.pi.website/download/pistar06.pdf)

### 总体流程概览

RECAP 的训练分为两个主要阶段：

1. **预训练（Pre-training）**：在多任务离线数据集上训练通用的 π*₀.₆ 模型
2. **任务特定微调（Task-specific Fine-tuning）**：在特定任务上迭代改进策略

---

### 阶段 1：预训练（Pre-training）

#### 步骤 1.1：准备多任务离线数据集

**数据来源**：
- 多任务、多机器人的离线数据集
- 包含：观察 (o_t)、动作 (a_t)、奖励 (r_t)、任务标签 (ℓ)

**数据格式**：
```python
transition = {
    "observation": o_t,      # 图像 + 状态
    "action": a_t,            # 动作序列
    "reward": r_t,            # 奖励信号（稀疏）
    "task": ℓ,                # 任务标签
    "next_observation": o_{t+1},
    "done": done_flag
}
```

#### 步骤 1.2：训练 Value Function V^π(o_t, ℓ)

**目标**：训练价值函数来评估任务完成进度

**训练方法**（论文 Section III-F）：
- 使用**监督学习**（MSE 损失）
- 目标值：预训练时使用整个 episode 的累积奖励

**训练公式**：
$$
V^\pi(o_t, \ell) \leftarrow \arg\min_V \mathbb{E}[(V(o_t, \ell) - \sum_{t'=t}^T r_{t'})^2]
$$

**伪代码**：
```python
# 伪代码：Value Function 预训练
for batch in offline_dataset:
    observations = batch["observation"]  # (B, ...)
    rewards = batch["reward"]            # (B, T)
    task_labels = batch["task"]          # (B,)
    
    # 1. 预测状态价值
    predicted_V = value_function(observations, task_labels)  # (B,)
    
    # 2. 计算目标值：整个 episode 的累积奖励
    # 预训练时：N = T（整个 episode）
    target_V = []
    for episode in batch:
        episode_return = sum(episode["reward"])  # Σ_{t=0}^T r_t
        target_V.append(episode_return)
    
    # 3. MSE 损失
    value_loss = MSE(predicted_V, target_V)
    
    # 4. 更新 Value Function
    value_function.backward(value_loss)
    value_function.update()
```

**关键点**：
- ✅ 使用整个 episode 的累积奖励作为目标（N=T）
- ✅ 这样可以在单次前向传播中计算所有 advantage
- ✅ 预训练时 advantage 估计方差较高，但在大规模数据上效果良好

#### 步骤 1.3：计算 Advantage 并二值化

**⚠️ 重要澄清**：Advantage 的计算流程是**先计算连续值，然后二值化**，两个阶段都是如此。

**Advantage 计算公式**（论文 Section III-F）：

**预训练时**（连续值计算）：
$$
A^\pi(o_t, a_t, \ell) = \sum_{t'=t}^T r_{t'} - V^\pi(o_t, \ell)
$$

**Advantage 是连续值**：
- 输出范围：$\mathbb{R}$（可以是任何实数）
- 例如：$A = 0.7, 0.5, 0.3, 0.1, -0.2, -0.5$ 等连续值
- **含义**：表示"这个动作比模型预期好多少"（正值=比预期好，负值=比预期差）

**二值化**（论文 Section III-F）：
$$
I_t = \begin{cases}
1 & \text{if } A^\pi(o_t, a_t, \ell) > \epsilon_\ell \\
0 & \text{otherwise}
\end{cases}
$$

**二值化过程**：
1. **先计算连续 Advantage**：$A \in \mathbb{R}$（连续值）
2. **然后二值化**：$I_t \in \{0, 1\}$（离散值）
3. **用于策略训练**：策略 conditioning 在二值化的 $I_t$ 上

**阈值设置**（论文 Section III-F）：
- 预训练时：选择阈值 ε_ℓ 使得**约 30%** 的演示数据有正优势
- 基于随机采样的 10k 数据点计算

**伪代码**：
```python
# 伪代码：Advantage 计算与二值化
for batch in offline_dataset:
    observations = batch["observation"]
    rewards = batch["reward"]
    task_labels = batch["task"]
    
    # 1. 计算 advantage（连续值，预训练时 N=T）
    current_V = value_function(observations, task_labels)  # V(o_t) ∈ [0, 1]
    episode_returns = compute_episode_returns(rewards)  # Σ_{t=0}^T r_t = 0 或 1
    advantages = episode_returns - current_V
    # advantages 是连续值，例如：[0.7, 0.5, 0.3, 0.1, -0.2, -0.5, ...]
    # 含义：正值表示"比预期好"，负值表示"比预期差"
    
    # 2. 二值化（连续值 → 离散值，每个任务单独设置阈值）
    for task_label in unique(task_labels):
        task_mask = (task_labels == task_label)
        task_advantages = advantages[task_mask]  # 连续值
        
        # 选择阈值使得约 30% 的数据有正优势
        threshold = np.percentile(task_advantages, 70)  # 70th percentile
        epsilon_ell[task_label] = threshold
        
        # 二值化：连续值 → 0 或 1
        advantage_indicators[task_mask] = (
            task_advantages > threshold
        ).astype(int)
        # advantage_indicators 是离散值：[1, 1, 1, 0, 0, 0, ...]
    
    batch["advantage"] = advantages  # 保存连续值（可选）
    batch["advantage_indicator"] = advantage_indicators  # 二值化后的离散值
```

#### 步骤 1.4：训练 Advantage-Conditioned Policy

**核心思想**：通过 conditioning 在 advantage 指标上来改进策略

**策略形式**：
$$
\pi_\theta(a_t | o_t, \ell, I_t)
$$

其中 I_t 是二值化的 advantage indicator。

**训练方法**：
- 使用 **flow-matching loss**（因为动作头是 flow-matching 模型）
- 训练时随机 dropout 30% 的 advantage conditioning（用于 CFG）

**伪代码**：
```python
# 伪代码：Policy 预训练
for batch in offline_dataset:
    observations = batch["observation"]
    actions = batch["action"]
    task_labels = batch["task"]
    advantage_indicators = batch["advantage_indicator"]
    
    # 1. Advantage conditioning dropout（30%）
    dropout_mask = (np.random.rand(len(batch)) < 0.3)
    advantage_indicators[dropout_mask] = None  # 无条件
    
    # 2. 训练条件策略：π(a_t | o_t, ℓ, I_t)
    predicted_actions = policy(
        observations, 
        task_labels, 
        advantage_indicator=advantage_indicators
    )
    
    # 3. Flow-matching loss
    policy_loss = flow_matching_loss(predicted_actions, actions)
    
    # 4. 更新策略
    policy.backward(policy_loss)
    policy.update()
```

**Advantage Conditioning Dropout 的作用**（论文 Section III-E）：
- 训练时随机 dropout 30% 的 advantage conditioning
- 这样可以在推理时使用 **CFG（Classifier-Free Guidance）**进行策略改进
- CFG 允许在推理时通过调整 β > 1 来锐化策略分布

---

### 阶段 2：任务特定微调（Task-specific Fine-tuning）

#### 步骤 2.1：初始演示微调（Demonstration Fine-tuning）

**目的**：在任务特定演示数据上微调策略

**方法**：标准行为克隆（BC）

**伪代码**：
```python
# 伪代码：演示微调
for batch in demonstration_dataset:
    observations = batch["observation"]
    actions = batch["action"]  # 专家动作
    task_labels = batch["task"]
    
    # 标准行为克隆（不使用 advantage conditioning）
    predicted_actions = policy(observations, task_labels)
    bc_loss = MSE(predicted_actions, actions)
    
    policy.backward(bc_loss)
    policy.update()
```

#### 步骤 2.2：部署与数据收集（迭代改进循环）

**数据收集类型**（论文 Section III-F）：
1. **自主 rollouts**：策略自主执行收集的数据
2. **专家干预（Corrections）**：专家在策略失败时接管并纠正的数据
3. **奖励反馈**：每个 trial 的稀疏奖励信号

**部署流程**：
```python
# 伪代码：部署与数据收集
collected_data = []

for episode in range(num_episodes):
    observation = env.reset()
    episode_data = []
    
    while not done:
        # 1. 策略推理（可选使用 CFG）
        # 推理时可以使用 CFG 进行策略改进（β ∈ [1.5, 2.5]）
        action = policy.sample(
            observation, 
            task_label,
            advantage_indicator=None,  # 推理时可选
            cfg_beta=1.5  # CFG 系数（可选）
        )
        
        # 2. 执行动作
        next_observation, reward, done = env.step(action)
        
        # 3. 记录 transition
        transition = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
            "task": task_label,
            "expert_correction": False
        }
        
        # 4. 专家干预（可选）
        if expert_intervention_needed:
            corrected_action = expert.teleoperate(observation)
            transition["action"] = corrected_action
            transition["expert_correction"] = True
        
        episode_data.append(transition)
        observation = next_observation
    
    # 5. 记录 episode 奖励
    episode_reward = sum([t["reward"] for t in episode_data])
    collected_data.append({
        "episodes": episode_data,
        "total_reward": episode_reward
    })
```

#### 步骤 2.3：微调 Value Function

**关键差异**（论文 Section III-F）：
- **预训练时**：使用整个 episode 的累积奖励（N=T）
- **微调时**：使用 **N=50** 的固定 lookahead

**什么是固定 Lookahead？**

**Lookahead** 指的是"向前看多少步"来计算累积奖励。固定 lookahead 意味着无论 episode 长度是多少，都只向前看固定的 N=50 步。

**对比**：
- **预训练（N=T）**：向前看到 episode 末尾（可能是 100 步、200 步等）
- **微调（N=50）**：无论 episode 多长，都只向前看 50 步

**Advantage 计算公式（微调时）**：
$$
A^\pi(o_t, a_t, \ell) = \sum_{t'=t}^{t+N-1} r_{t'} + V^\pi(o_{t+N}, \ell) - V^\pi(o_t, \ell)
$$

其中 N=50（固定值）。

**如何实现固定 N=50 Lookahead？**

**实现步骤**：

1. **对于每个时间步 t**：
   - 计算从 t 到 t+49 的累积奖励：$\sum_{t'=t}^{t+49} r_{t'}$
   - 如果 t+50 < episode 长度，加上 $V^\pi(o_{t+50})$（50 步后的状态价值）
   - 如果 t+50 >= episode 长度，只使用实际累积奖励（不加上未来价值）

2. **边界处理**：
   - 当接近 episode 末尾时（t+50 >= T），只使用实际奖励，不加上未来价值

**详细伪代码**：
```python
# 伪代码：固定 N=50 Lookahead 的实现
def compute_n_step_return_fixed_lookahead(observations, rewards, value_function, task_labels, n_steps=50):
    """
    计算固定 N-step return（N=50）
    
    Args:
        observations: 所有时间步的观察 (T, ...)
        rewards: 所有时间步的奖励 (T,)
        value_function: Value Function 模型
        task_labels: 任务标签
        n_steps: 固定 lookahead 步数（N=50）
    
    Returns:
        n_step_returns: 每个时间步的 N-step return (T,)
    """
    T = len(rewards)  # Episode 长度
    n_step_returns = []
    
    for t in range(T):
        # 步骤 1：计算从 t 到 t+n_steps-1 的实际奖励
        # 注意：如果 t+n_steps > T，只取到 T-1
        actual_rewards = rewards[t:min(t+n_steps, T)]
        immediate_return = sum(actual_rewards)  # Σ_{t'=t}^{min(t+N-1, T-1)} r_{t'}
        
        # 步骤 2：如果还有未来状态（t+n_steps < T），加上未来状态价值
        if t + n_steps < T:
            # 有完整的 N 步，加上 V(o_{t+N})
            future_obs = observations[t + n_steps]
            future_value = value_function(future_obs, task_labels[t])
            n_step_return = immediate_return + future_value
        else:
            # 接近 episode 末尾，没有完整的 N 步，只使用实际奖励
            n_step_return = immediate_return
        
        n_step_returns.append(n_step_return)
    
    return n_step_returns

# 使用示例
for batch in collected_data:
    observations = batch["observation"]  # (T, ...)
    rewards = batch["reward"]            # (T,)
    task_labels = batch["task"]         # (T,)
    
    # 1. 预测当前状态价值
    predicted_V = value_function(observations, task_labels)  # (T,)
    
    # 2. 计算目标值（固定 N=50 lookahead）
    target_V = compute_n_step_return_fixed_lookahead(
        observations,
        rewards,
        value_function,
        task_labels,
        n_steps=50  # 固定 lookahead
    )
    
    # 3. MSE 损失
    value_loss = MSE(predicted_V, target_V)
    value_function.backward(value_loss)
    value_function.update()
```

**具体例子：固定 N=50 Lookahead**

假设一个 episode 有 200 个时间步（T=200）：

```python
# Episode 长度：T = 200
# 固定 lookahead：N = 50

# 时间步 t=0
# - 计算奖励：sum(rewards[0:50]) = r_0 + r_1 + ... + r_49
# - 未来状态：observations[50] 存在
# - 加上未来价值：V(o_50)
# - N-step return = sum(rewards[0:50]) + V(o_50)

# 时间步 t=50
# - 计算奖励：sum(rewards[50:100]) = r_50 + r_51 + ... + r_99
# - 未来状态：observations[100] 存在
# - 加上未来价值：V(o_100)
# - N-step return = sum(rewards[50:100]) + V(o_100)

# 时间步 t=150
# - 计算奖励：sum(rewards[150:200]) = r_150 + r_151 + ... + r_199
# - 未来状态：observations[200] 不存在（超出范围）
# - 不加上未来价值（因为 t+50=200 >= T=200）
# - N-step return = sum(rewards[150:200])  # 只使用实际奖励

# 时间步 t=180
# - 计算奖励：sum(rewards[180:200]) = r_180 + r_181 + ... + r_199
# - 未来状态：observations[230] 不存在（超出范围）
# - 不加上未来价值
# - N-step return = sum(rewards[180:200])  # 只使用实际奖励
```

**为什么使用固定 N=50？**

1. **更精确**：关注动作的即时影响（接下来 50 步），而不是整个 episode
2. **更稳定**：方差较低，适合小规模数据
3. **更实用**：在实际部署中，更关心"这个动作对接下来几步的影响"
4. **计算效率**：不需要处理不同长度的 episode

**与预训练的区别**：

| 特性 | 预训练（N=T） | 微调（N=50） |
|------|-------------|-------------|
| **Lookahead** | 整个 episode | 固定 50 步 |
| **计算方式** | $\sum_{t'=t}^T r_{t'}$ | $\sum_{t'=t}^{t+49} r_{t'} + V(o_{t+50})$ |
| **方差** | 较高（episode 长度不同） | 较低（固定 lookahead） |
| **适用场景** | 大规模数据 | 小规模任务特定数据 |

#### 步骤 2.4：重新计算 Advantage 并更新策略

**⚠️ 重要澄清**：微调阶段的 Advantage 计算流程与预训练相同：**先计算连续值，然后二值化**。

**流程**（论文 Algorithm 1）：

1. **重新计算 Advantage（连续值）**：使用微调后的 value function
   - Advantage 计算公式：$A^\pi(o_t, a_t, \ell) = \sum_{t'=t}^{t+N-1} r_{t'} + V^\pi(o_{t+N}, \ell) - V^\pi(o_t, \ell)$
   - 输出：连续值 $A \in \mathbb{R}$（例如：0.7, 0.5, 0.3, 0.1, -0.2 等）

2. **二值化**：设置阈值使得约 40% 的评估 rollouts 有正优势
   - 将连续值转换为二值：$I_t \in \{0, 1\}$

3. **更新策略**：使用 advantage-conditioned policy 训练
   - 策略 conditioning 在二值化的 $I_t$ 上：$\pi_\theta(a_t | o_t, \ell, I_t)$

**阈值设置**（论文 Section III-F）：
- **微调时**：选择阈值使得**约 40%** 的评估 rollouts 有正优势
- **特殊情况**：对于 T-shirt 和 shorts 叠衣服任务，设置为约 10%（因为高质量演示数据训练出的策略较慢但成功率高）

**关键点**：
- ✅ **两个阶段都相同**：预训练和微调都是先计算连续 Advantage，然后二值化
- ✅ **连续值用于计算**：Advantage 本身是连续值，表示"比预期好多少"
- ✅ **二值化用于训练**：策略 conditioning 使用的是二值化的 indicator（0 或 1）

**伪代码**：
```python
# 伪代码：迭代改进（Algorithm 1）
for iteration in range(num_iterations):
    # === 步骤 1：使用更新后的 value function 重新计算 advantage（连续值）===
    for batch in collected_data:
        observations = batch["observation"]
        rewards = batch["reward"]
        task_labels = batch["task"]
        
        # 计算 advantage（连续值，使用固定 N=50 lookahead）
        advantages = compute_advantage_fixed_lookahead(
            observations,
            rewards,
            task_labels,
            value_function,  # 使用微调后的 value function
            n_steps=50  # 固定 lookahead
        )
        # advantages 是连续值，例如：[0.7, 0.5, 0.3, 0.1, -0.2, -0.5, ...]
        batch["advantage"] = advantages
    
    # === 步骤 2：二值化 advantage（连续值 → 离散值）===
    # 设置阈值使得约 40% 的数据有正优势
    advantage_threshold = np.percentile(
        [a for batch in collected_data for a in batch["advantage"]],
        60  # 60th percentile（约 40% 正优势）
    )
    
    for batch in collected_data:
        # 二值化：连续值 → 0 或 1
        batch["advantage_indicator"] = (
            batch["advantage"] > advantage_threshold
        ).astype(int)
        # advantage_indicator 是离散值：[1, 1, 1, 0, 0, 0, ...]
    
    # === 步骤 3：使用 advantage-conditioned policy 更新策略 ===
    for batch in collected_data:
        observations = batch["observation"]
        actions = batch["action"]
        task_labels = batch["task"]
        advantage_indicators = batch["advantage_indicator"]
        
        # Advantage conditioning dropout（30%）
        dropout_mask = (np.random.rand(len(batch)) < 0.3)
        advantage_indicators[dropout_mask] = None
        
        # 训练条件策略
        predicted_actions = policy(
            observations,
            task_labels,
            advantage_indicator=advantage_indicators
        )
        
        # Flow-matching loss
        policy_loss = flow_matching_loss(predicted_actions, actions)
        
        policy.backward(policy_loss)
        policy.update()
    
    # === 步骤 4：重新部署收集数据（可选，多轮迭代）===
    if iteration < num_iterations - 1:
        collected_data = deploy_and_collect(policy)
```

---

### 关键参数总结（论文 Section III-F）

| 参数 | 预训练 | 微调 |
|------|--------|------|
| **Advantage 计算** | N = T（整个 episode） | N = 50（固定 lookahead） |
| **Advantage 阈值** | 约 30% 正优势 | 约 40% 正优势 |
| **Advantage Dropout** | 30% | 30% |
| **CFG β** | - | 1.5 - 2.5（推理时可选） |

---

### 推理时的策略改进（CFG）

**论文 Section III-E**：训练后可以使用 CFG 进一步锐化策略分布

**CFG 公式**：
$$
\hat{\pi}(a_{t:t+H} | o_t, \ell) \propto \pi_{\text{ref}}(a_{t:t+H} | o_t, \ell) \left(\frac{\pi_{\text{ref}}(a_{t:t+H} | I_t, o_t, \ell)}{\pi_{\text{ref}}(a_{t:t+H} | o_t, \ell)}\right)^\beta
$$

**实现方式**（通过 flow-matching 梯度）：
$$
\nabla_a \log \pi_\theta(a_{t:t+H} | o_t, \ell) + \beta \left(\nabla_a \log \pi_\theta(a_{t:t+H} | I_t, o_t, \ell) - \nabla_a \log \pi_\theta(a_{t:t+H} | o_t, \ell)\right)
$$

**关键点**：
- β > 1 时锐化分布，偏向高 advantage 动作
- β 与训练时的 advantage 阈值 ε_ℓ 相关（都用于锐化分布）
- 论文主要依赖训练时的 ε_ℓ，CFG 作为补充（β ∈ [1.5, 2.5]）

---

## 关键问题解答

### Q1: Actor Policy 如何传递 transition 用于 RL buffer 构建？

**答案**：RECAP 使用**离线 RL**，不是在线收集数据到 replay buffer。

**实际流程**：

1. **预训练阶段**：
   - 使用**预先收集的多任务离线数据集**
   - 数据格式：`(o_t, a_t, r_t, o_{t+1}, ℓ)`
   - 直接从数据集采样 batch，**无需 buffer**

2. **微调阶段**：
   - **部署策略**收集新数据（自主执行 + 专家干预）
   - 数据收集到**临时存储**（不是传统 replay buffer）
   - 收集完成后，**一次性处理所有数据**进行训练

```python
# 伪代码：数据收集流程
# 阶段1：部署收集
collected_transitions = []
for episode in deployment_episodes:
    transitions = execute_policy_and_collect(policy)
    collected_transitions.extend(transitions)

# 阶段2：批量处理（不是在线更新）
dataset = create_dataset(collected_transitions)
for batch in dataset:
    train_step(batch)
```

**关键点**：
- ❌ **不是在线 RL**：不维护动态更新的 replay buffer
- ✅ **离线 RL**：收集数据 → 批量训练 → 重新部署
- ✅ **迭代改进**：可以多轮迭代（收集 → 训练 → 部署 → 收集...）

---

### Q2: Critic Policy 如何从 buffer 接收 transition 并进行 Q 值更新？

**答案**：RECAP 使用 **Value Function (V(s))**，不是 Q 值网络。

**实际流程**：

1. **Value Function 训练**：
```python
# 伪代码：Value Function 训练
for batch in dataset:  # 从离线数据集或收集的数据采样
    observations = batch["observation"]  # (B, ...)
    rewards = batch["reward"]            # (B, T)
    task_labels = batch["task"]          # (B,)
    
    # 1. Value Function 前向传播
    predicted_V = value_function(observations, task_labels)  # (B,)
    
    # 2. 计算目标值（n-step return）
    # 预训练：A = Σ_{t=0}^T r_t - V(o_0)
    # 微调：A = Σ_{t=t}^{t+N-1} r_t + V(o_{t+N}) - V(o_t)
    target_V = compute_target_value(rewards, value_function, n_steps)
    
    # 3. 监督学习损失（不是 TD 误差）
    value_loss = MSE(predicted_V, target_V)
    
    # 4. 更新 Value Function（独立更新，不更新策略）
    value_function.backward(value_loss)
    value_function.update()
```

2. **Advantage 计算**（使用更新后的 Value Function）：
```python
# 伪代码：Advantage 计算（固定 N-step lookahead）
def compute_advantage_fixed_lookahead(observations, rewards, value_function, task_labels, n_steps=50):
    """
    计算 advantage：A(o_t, a_t) = r_{t:t+N} + V(o_{t+N}) - V(o_t)
    
    使用固定 N-step lookahead（N=50）
    """
    T = len(rewards)  # Episode 长度
    current_values = value_function(observations, task_labels)  # V(o_t) for all t
    
    # 计算 n-step return（固定 lookahead）
    n_step_returns = []
    for t in range(T):
        # 步骤 1：计算从 t 到 t+n_steps-1 的实际奖励
        actual_rewards = rewards[t:min(t+n_steps, T)]
        immediate_return = sum(actual_rewards)  # Σ_{t'=t}^{min(t+N-1, T-1)} r_{t'}
        
        # 步骤 2：如果还有未来状态（t+n_steps < T），加上未来状态价值
        if t + n_steps < T:
            # 有完整的 N 步，加上 V(o_{t+N})
            future_obs = observations[t + n_steps]
            future_value = value_function(future_obs, task_labels[t])
            n_step_return = immediate_return + future_value
        else:
            # 接近 episode 末尾，没有完整的 N 步，只使用实际奖励
            n_step_return = immediate_return
        
        n_step_returns.append(n_step_return)
    
    # 步骤 3：计算 Advantage
    advantages = np.array(n_step_returns) - current_values  # A = R - V
    
    return advantages

# 使用示例
advantages = compute_advantage_fixed_lookahead(
    observations,
    rewards,
    value_function,
    task_labels,
    n_steps=50  # 固定 lookahead
)
```

**关键点**：
- ✅ **预测 V(s)**：状态价值，不是 Q(s,a)
- ✅ **监督学习**：使用 MSE 损失，不是 TD 误差
- ✅ **独立训练**：Value Function 单独训练，不依赖策略更新
- ✅ **离线计算**：Advantage 在训练前预计算，不是动态更新

---

### Q3: 更新后的 Q 值如何作用到 Actor Policy？如何选择 Q 值最大的动作？

**答案**：RECAP **不使用 Q 值**，而是使用 **Advantage Conditioning**。

**实际流程**：

1. **Advantage 作用到策略**：
```python
# 伪代码：Advantage Conditioning
# 训练时
for batch in dataset:
    observations = batch["observation"]
    actions = batch["action"]
    advantages = batch["advantage"]  # 预计算的
    
    # 二值化 advantage
    advantage_indicators = (advantages > epsilon).astype(int)
    
    # 策略 conditioning 在 advantage indicator 上
    predicted_actions = policy(
        observations,
        task_labels,
        advantage_indicator=advantage_indicators  # 条件输入
    )
    
    # 训练策略（flow-matching loss）
    policy_loss = flow_matching_loss(predicted_actions, actions)
    policy.backward(policy_loss)
    policy.update()
```

2. **推理时的动作选择**：
```python
# 伪代码：推理时动作选择
def sample_action(observation, task_label, use_cfg=True, cfg_beta=1.5):
    """
    推理时采样动作
    """
    if use_cfg:
        # 使用 CFG（Classifier-Free Guidance）进行策略改进
        # 这相当于隐式地使用 advantage 来改进策略
        
        # 1. 采样无条件策略
        unconditional_action = policy.sample(
            observation,
            task_label,
            advantage_indicator=None
        )
        
        # 2. 采样条件策略（高 advantage）
        conditional_action = policy.sample(
            observation,
            task_label,
            advantage_indicator=1  # 正优势
        )
        
        # 3. CFG 组合（β > 1 时锐化分布）
        # 这相当于选择"更好"的动作
        action = unconditional_action + cfg_beta * (
            conditional_action - unconditional_action
        )
    else:
        # 直接采样
        action = policy.sample(observation, task_label)
    
    return action
```

**关键点**：
- ❌ **不选择 Q 值最大的动作**：没有 Q 值，也没有最大化操作
- ✅ **Advantage Conditioning**：通过 conditioning 在 advantage 指标上改进策略
- ✅ **CFG 推理**：推理时使用 Classifier-Free Guidance 隐式地利用 advantage
- ✅ **Flow-matching**：使用扩散模型（flow-matching）生成动作，不是确定性选择

---

### Q4: Offline Policy 如何进行？

**答案**：RECAP 是完全的**离线 RL**方法，但包含**迭代数据收集**。

**完整流程**：

#### 阶段 1：完全离线预训练

```python
# 伪代码：离线预训练
# 1. 加载多任务离线数据集
offline_dataset = load_multi_task_dataset()

# 2. 训练 Value Function（完全离线）
value_function = train_value_function(offline_dataset)

# 3. 计算 advantage（离线计算）
for batch in offline_dataset:
    advantages = compute_advantage(batch, value_function)
    batch["advantage"] = advantages

# 4. 训练 advantage-conditioned policy（完全离线）
policy = train_policy(offline_dataset, advantage_conditioning=True)
```

#### 阶段 2：任务特定微调（迭代改进）

```python
# 伪代码：迭代离线 RL
# 初始策略：预训练的 π*₀.₆
policy = load_pretrained_policy()
value_function = load_pretrained_value_function()

for iteration in range(num_iterations):
    # === 步骤 1：部署收集数据（唯一需要在线交互的部分）===
    collected_data = []
    for episode in range(num_episodes):
        # 部署策略执行任务
        episode_data = deploy_policy(policy, task)
        collected_data.append(episode_data)
    
    # === 步骤 2：离线处理数据 ===
    # 2.1 计算奖励（如果有稀疏奖励）
    for episode in collected_data:
        episode["reward"] = compute_reward(episode)
    
    # 2.2 微调 Value Function（离线）
    value_function = fine_tune_value_function(
        collected_data,
        value_function
    )
    
    # 2.3 计算 advantage（离线）
    for batch in collected_data:
        advantages = compute_advantage(
            batch,
            value_function  # 使用更新后的 value function
        )
        batch["advantage"] = advantages
    
    # 2.4 更新策略（离线）
    policy = fine_tune_policy(
        collected_data,
        policy,
        advantage_conditioning=True
    )
    
    # === 步骤 3：评估（可选）===
    if evaluate:
        performance = evaluate_policy(policy)
        print(f"Iteration {iteration}: {performance}")
```

**关键特性**：

1. **离线训练**：
   - ✅ Value Function 训练：完全离线，使用监督学习
   - ✅ Policy 训练：完全离线，使用 advantage-conditioned behavior cloning
   - ✅ Advantage 计算：离线预计算，不是动态更新

2. **数据收集**：
   - ✅ **部署时收集**：策略部署执行时收集数据
   - ✅ **批量处理**：收集完成后批量训练，不是在线更新
   - ✅ **迭代改进**：可以多轮迭代（收集 → 训练 → 部署）

3. **与传统 Offline RL 的区别**：
   - 传统 Offline RL：完全离线，不收集新数据
   - RECAP：**迭代离线 RL**，允许部署收集新数据，但训练完全离线

---

## 架构对比总结

| 组件 | 传统 Actor-Critic RL | RECAP (π*₀.₆) |
|------|---------------------|---------------|
| **Actor** | 策略网络 π(a\|s) | Advantage-conditioned policy π(a\|o, ℓ, I) |
| **Critic** | Q(s,a) 或 V(s) | Value Function V(s)（独立模型） |
| **数据收集** | 在线交互 + Replay Buffer | 离线数据集 + 部署收集 |
| **Critic 更新** | TD 误差 / GAE | 监督学习（MSE） |
| **Actor 更新** | 策略梯度 | Advantage-conditioned BC |
| **优势使用** | TD 误差 / GAE | 预计算的 advantage + conditioning |
| **动作选择** | 最大化 Q 值或采样 | CFG + Flow-matching 采样 |
| **训练模式** | 在线学习 | 离线 RL + 迭代改进 |

---

## 关键创新点

1. **Advantage Conditioning**：
   - 不是策略梯度，而是通过 conditioning 在 advantage 指标上改进策略
   - 类似于 classifier-free guidance，但用于策略改进

2. **独立 Value Function**：
   - 使用较小的 VLA 模型（270M vs 4B）预测价值
   - 降低计算成本，同时保持准确性

3. **迭代离线 RL**：
   - 结合离线 RL 的稳定性和在线数据收集的适应性
   - 允许策略在实际部署中改进

4. **异构数据融合**：
   - 同时利用演示数据、自主 rollouts 和专家干预
   - 通过 advantage 统一处理不同类型的数据

---

## 参考文献

- [π*₀.₆: a VLA That Learns From Experience](https://www.pi.website/download/pistar06.pdf)
- RECAP: RL with Experience and Corrections via Advantage-conditioned Policies
