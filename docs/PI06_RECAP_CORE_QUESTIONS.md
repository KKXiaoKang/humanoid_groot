# π*0.6 (RECAP) 核心问题详解

> **基于论文和代码库的深入分析**

## 问题 1：预训练阶段的 Reward 设计来源

### 1.1 Reward 的来源

**论文中的说明**（Section III-F）：
- **多任务离线数据集**：预训练使用的是**预先收集的多任务、多机器人平台的离线数据集**
- **任务完成信号**：对于每个任务，reward 通常是**稀疏的（Sparse Reward）**，只在 episode 结束时给出
- **二值化奖励**：大多数任务使用**成功/失败**的二值奖励（1 = 成功，0 = 失败）

**⚠️ 重要澄清：论文使用的是 Sparse Reward（稀疏奖励），不是 Dense Reward（密集奖励）**

### 1.2 Reward 的具体形式

**论文原文明确说明**（Section III-F）：
> "During pre-training, we calculate the advantage estimate as $A^\pi(\mathbf{o}_{t},\mathbf{a}_{t})=\sum_{t'=0}^{T}r'_{t}-V^\pi(\mathbf{o}_{t})$"

**关键点**：
- Reward 是**稀疏的**：大多数时间步 $r_t = 0$，只在 episode 末尾有奖励
- **不是 Dense Reward**：中间步骤**没有**奖励信号（如 0.2, 0.3 等中间奖励）

**Sparse Reward 的具体形式**（对应图片右边的例子）：
```python
# Episode: 叠衣服任务（成功）
rewards = [0, 0, 0, 0, 1]  # 只有最后一步 reward = 1（成功）
# 中间步骤没有 reward（不是 [0, 0, 0.2, 0.3, 1.0]）

# 计算 Return（累积奖励）
R_0 = 0 + 0 + 0 + 0 + 1 = 1  # 从 t=0 到 T 的累积奖励
R_1 = 0 + 0 + 0 + 1 = 1      # 从 t=1 到 T 的累积奖励
R_2 = 0 + 0 + 1 = 1          # 从 t=2 到 T 的累积奖励
R_3 = 0 + 1 = 1              # 从 t=3 到 T 的累积奖励
R_4 = 1                      # 从 t=4 到 T 的累积奖励

# 结果：R = [1, 1, 1, 1, 1]（所有时间步的 Return 都是 1）
```
### 1.3 Reward 的计算方式

**预训练阶段**：
$$
R_t = \sum_{t'=t}^T r_{t'}
$$

其中：
- $r_{t'}$ 是每个时间步的奖励（**稀疏的**，大部分为 0，只在 episode 末尾有奖励）
- $T$ 是 episode 的总长度
- 预训练时使用**整个 episode 的累积奖励**（N = T）

**为什么使用整个 episode？**
- **计算效率**：可以在单次前向传播中计算所有 advantage
- **大规模数据**：预训练数据量大，方差较高但可以接受
- **通用性**：预训练目标是学习通用策略，不需要精确的即时反馈

### 1.4 Reward 设计的关键点

**论文中的实际做法**：
- ✅ **稀疏奖励（Sparse Reward）**：大多数时间步 reward = 0，只在 episode 末尾有奖励
- ✅ **二值化**：成功 = 1，失败 = 0
- ✅ **任务特定**：不同任务有不同的 reward 定义
- ✅ **离线标注**：所有 reward 都是预先标注好的，不是在线计算

**示例**（叠衣服任务）：
```python
# Episode 1: 成功叠衣服（Sparse Reward）
rewards = [0, 0, 0, ..., 0, 1]  # 只有最后一步 reward = 1（成功）
episode_return = sum(rewards) = 1
# 所有时间步的 Return 都是 1（因为只有最后有奖励）

# Episode 2: 失败（衣服掉落）
rewards = [0, 0, 0, ..., 0, 0]  # 最后一步 reward = 0（失败）
episode_return = sum(rewards) = 0
# 所有时间步的 Return 都是 0（因为任务失败）
```

**⚠️ 重要区别**：
- **论文使用**：Sparse Reward（图片右边的形式）→ `r = [0, 0, 0, 0, 1]`，所有 Return 都是 1
- **论文不使用**：Dense Reward（图片左边的形式）→ `r = [0, 0, 0.2, 0.3, 1.0]`，Return 不同

---

## 问题 1.5：Value Function 的架构和训练方式（补充）

### 1.5.1 Value Function 的架构

**⚠️ 重要澄清**：Value Function **不是单纯的 MLP**，而是一个**完整的、较小的 VLA 模型**，最后通过 MLP value head 输出标量值。

**架构组成**（论文 Section III-F）：
```
Value Function = 小型 VLA 模型
├── 视觉编码器：SigLIP (400M) - 与策略网络共享权重
├── 语言模型：Gemma (270M) - 比策略网络小得多（策略网络是 4B）
└── 价值头（Value Head）：MLP，输出单个标量值 V(s)
```

**关键点**：
- ✅ **独立模型**：与策略网络分离，单独训练
- ✅ **较小规模**：使用较小的 Gemma (270M) 而非 (4B)，降低计算成本
- ✅ **共享视觉编码器**：与策略网络共享 SigLIP 视觉编码器（可能冻结或微调）
- ✅ **MLP Value Head**：最后通过 MLP 将深度表示映射为单个标量值

**架构示意图**：
```python
# 伪代码：Value Function 架构
class ValueFunction:
    def __init__(self):
        # 1. 视觉编码器（与策略网络共享）
        self.vision_encoder = SigLIP(400M)  # 共享权重
        
        # 2. 语言模型（较小）
        self.language_model = Gemma(270M)  # 比策略网络小
        
        # 3. Value Head（MLP）
        self.value_head = MLP(
            input_dim=hidden_dim,
            hidden_dims=[512, 256],
            output_dim=1  # 输出单个标量值
        )
    
    def forward(self, observation, task_label):
        # 1. 视觉编码
        vision_features = self.vision_encoder(observation)
        
        # 2. 语言编码
        text_features = self.language_model(task_label)
        
        # 3. 多模态融合（类似 VLA 架构）
        fused_features = fuse(vision_features, text_features)
        
        # 4. 通过语言模型处理
        hidden_state = self.language_model(fused_features)
        
        # 5. 提取深度表示（通常是 [CLS] token 或第一个 token）
        deep_representation = hidden_state[:, 0, :]  # (B, hidden_dim)
        
        # 6. 通过 MLP Value Head 预测价值
        value = self.value_head(deep_representation)  # (B, 1)
        
        return value.squeeze(-1)  # (B,)
```

### 1.5.2 Value Function 的训练方式

**训练方法**：**监督学习（Supervised Learning）**，不是 TD 学习

**训练公式**：
$$
V^\pi(o_t, \ell) \leftarrow \arg\min_V \mathbb{E}[(V(o_t, \ell) - \sum_{t'=t}^T r_{t'})^2]
$$

**训练步骤**：

#### 步骤 1：准备训练数据

```python
# 伪代码：准备 Value Function 训练数据
for episode in offline_dataset:
    observations = episode["observation"]  # (T, ...)
    rewards = episode["reward"]            # (T,)
    task_label = episode["task"]          # 任务标签
    
    # 计算每个时间步的目标值（累积奖励）
    for t in range(T):
        # 预训练时：使用整个 episode 的累积奖励
        target_value = sum(rewards[t:])  # Σ_{t'=t}^T r_{t'}
        
        training_data.append({
            "observation": observations[t],
            "task_label": task_label,
            "target_value": target_value
        })
```

#### 步骤 2：训练 Value Function

```python
# 伪代码：Value Function 训练循环
value_function = ValueFunction()  # 初始化
optimizer = Adam(lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        observations = batch["observation"]  # (B, ...)
        task_labels = batch["task_label"]    # (B,)
        target_values = batch["target_value"]  # (B,)
        
        # 1. 前向传播：预测状态价值
        predicted_values = value_function(observations, task_labels)  # (B,)
        
        # 2. 计算 MSE 损失
        loss = MSE(predicted_values, target_values)
        
        # 3. 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**关键点**：
- ✅ **监督学习**：使用 MSE 损失，不是 TD 误差
- ✅ **目标值**：预训练时使用整个 episode 的累积奖励（N = T）
- ✅ **独立训练**：Value Function 单独训练，不依赖策略更新
- ✅ **离线训练**：所有数据都是预先收集好的，不是在线交互

### 1.5.4 ⚠️ 关键问题：Sparse Reward 下如何训练 Value Function？

**问题**：如果 reward 是 sparse 的（只有最后一帧是 1，其他都是 0），那么同一个 episode 内所有时间步的 Return 都相同，Value Function 如何学习区分不同状态？

**示例**：
```python
# Episode 1: 成功叠衣服
rewards = [0, 0, 0, 0, 1]
# 所有时间步的 Return 都是 1
R_0 = R_1 = R_2 = R_3 = R_4 = 1

# Episode 2: 失败
rewards = [0, 0, 0, 0, 0]
# 所有时间步的 Return 都是 0
R_0 = R_1 = R_2 = R_3 = R_4 = 0
```

**答案：跨 Episode 学习（Cross-Episode Learning）**

虽然**同一个 episode 内的 Return 相同**，但**不同 episode 之间的 Return 不同**。Value Function 通过**跨 episode 的模式学习**来区分状态价值。

#### 学习机制

**1. 数据分布差异**：
- **成功的 episode**：所有状态的目标值都是 1
- **失败的 episode**：所有状态的目标值都是 0
- Value Function 学习：**哪些状态更可能出现在成功的 episode 中**

**2. 状态特征学习**：
```python
# 训练数据示例
training_data = [
    # 成功的 episode 1
    {"obs": "衣服散乱在桌上", "target": 1},      # 成功episode的初始状态
    {"obs": "衣服已摊平", "target": 1},          # 成功episode的中间状态
    {"obs": "衣服正在折叠", "target": 1},        # 成功episode的中间状态
    {"obs": "衣服已折叠完成", "target": 1},      # 成功episode的结束状态
    
    # 失败的 episode 2
    {"obs": "衣服散乱在桌上", "target": 0},      # 失败episode的初始状态
    {"obs": "衣服掉落在地", "target": 0},        # 失败episode的中间状态
    {"obs": "衣服被撕破", "target": 0},          # 失败episode的中间状态
    {"obs": "任务失败", "target": 0},            # 失败episode的结束状态
    
    # 成功的 episode 3
    {"obs": "衣服散乱在桌上", "target": 1},      # 又是成功episode的初始状态
    {"obs": "衣服已摊平", "target": 1},          # 又是成功episode的中间状态
    ...
]

# Value Function 学习到的模式：
# - "衣服散乱在桌上" → 有时出现在成功episode（target=1），有时出现在失败episode（target=0）
#   → 平均价值 ≈ 0.5（如果成功/失败各占一半）
# - "衣服已摊平" → 主要出现在成功episode（target=1）
#   → 平均价值 ≈ 0.8（更接近成功）
# - "衣服掉落在地" → 主要出现在失败episode（target=0）
#   → 平均价值 ≈ 0.2（更接近失败）
# - "衣服已折叠完成" → 只出现在成功episode（target=1）
#   → 平均价值 ≈ 1.0（完全成功）
```

**3. 统计学习过程**：

通过大量数据，Value Function 学习到：
- **接近成功的状态**（在成功 episode 中更常见）→ **高价值**
- **远离成功的状态**（在失败 episode 中更常见）→ **低价值**
- **中间状态**（在成功和失败 episode 中都出现）→ **中等价值**

**数学表达**：
$$
V(o_t) \approx \mathbb{E}_{\text{episodes}}[\text{Return} | \text{状态 } o_t \text{ 出现在该 episode}]
$$

即：Value Function 预测的是"**包含该状态的 episode 的平均 Return**"。

#### 具体例子：叠衣服任务

**训练数据**（假设有 1000 个 episode）：
```python
# 成功的 episode（500个）
success_episodes = [
    {"obs": "衣服散乱", "target": 1},
    {"obs": "衣服摊平", "target": 1},
    {"obs": "衣服折叠中", "target": 1},
    {"obs": "衣服完成", "target": 1},
] * 500

# 失败的 episode（500个）
failure_episodes = [
    {"obs": "衣服散乱", "target": 0},
    {"obs": "衣服掉落", "target": 0},
    {"obs": "衣服撕破", "target": 0},
    {"obs": "任务失败", "target": 0},
] * 500

# Value Function 学习到的价值：
# V("衣服散乱") ≈ (500×1 + 500×0) / 1000 = 0.5
# V("衣服摊平") ≈ (500×1 + 0×0) / 500 = 1.0  # 只在成功episode中出现
# V("衣服掉落") ≈ (0×1 + 500×0) / 500 = 0.0  # 只在失败episode中出现
# V("衣服完成") ≈ (500×1 + 0×0) / 500 = 1.0  # 只在成功episode中出现
```

**关键洞察**：
- ✅ **跨 episode 学习**：Value Function 通过比较不同 episode 来学习状态价值
- ✅ **模式识别**：学习"哪些状态特征与成功相关"
- ✅ **统计平均**：预测的是包含该状态的 episode 的平均 Return

#### 为什么这种方法有效？

1. **大规模数据**：预训练数据量大（多任务、多机器人），有足够的成功/失败 episode 对比
2. **特征丰富**：VLA 模型（视觉编码器 + 语言模型）可以提取丰富的状态特征
3. **模式识别**：深度学习模型擅长从大量数据中学习统计模式

**局限性**：
- ⚠️ **方差较高**：同一个 episode 内所有状态的目标值相同，导致方差较大
- ⚠️ **需要大量数据**：需要足够多的成功/失败 episode 对比才能学习
- ⚠️ **不够精确**：不如 dense reward 精确，但在大规模数据上可接受

**论文中的处理**（Section III-F）：
> "We use this advantage estimate since it allows us to calculate the advantage values on-the-fly during pre-training using a single inference call to the value function. We find empirically that this advantage estimate works well when the policy is trained on large amounts of data from diverse tasks during pre-training."

即：虽然方差较高，但在大规模多样化数据上效果良好。

### 1.5.3 Value Function 的输出范围和含义

**⚠️ 重要澄清**：Value Function 的输出是**连续值（Continuous Value）**，不是离散值。

#### 输出范围

**π*0.6 的 Value Function**：
- **输出类型**：连续标量值（Continuous Scalar）
- **理论范围**：$\mathbb{R}$（所有实数）
- **实际范围**：训练目标在 **[0, 1]** 范围内（因为 sparse reward：成功=1，失败=0）
- **归一化**：**没有显式归一化**，但训练会自然地将输出限制在合理范围内

**训练目标值**：
```python
# Sparse Reward 下的目标值
# 成功的 episode
target_value = sum(rewards[t:]) = 1  # 所有时间步的目标值都是 1

# 失败的 episode
target_value = sum(rewards[t:]) = 0  # 所有时间步的目标值都是 0

# 所以目标值范围是 [0, 1]，不是 [-1, 1]
```

**Value Function 输出**：
```python
# Value Function 的输出（连续值）
value_pred = value_function(observation, task_label)  # Shape: (B, 1)
# 输出范围：理论上可以是任何实数，但训练后通常在 [0, 1] 附近
# 例如：0.2, 0.5, 0.8, 1.0, 0.1 等连续值
```

#### 输出含义

**Value Function 预测的是什么？**

1. **不是任务完成度**：Value Function 预测的是"**从这个状态开始的累积奖励期望**"
2. **不是进度百分比**：不是 0-1 的完成度，而是"**这个状态能获得多少累积奖励**"
3. **是状态价值**：预测"如果从这个状态继续执行，能获得多少累积奖励"

**数学表达**：
$$
V^\pi(o_t, \ell) = \mathbb{E}[\sum_{t'=t}^T r_{t'} | o_t, \ell]
$$

即：Value Function 预测的是"**包含该状态的 episode 的平均累积奖励**"。

**具体例子**：
```python
# 叠衣服任务
# 状态1：衣服散乱在桌上
V("衣服散乱") ≈ 0.5  # 50% 的 episode 会成功，50% 会失败

# 状态2：衣服已摊平
V("衣服已摊平") ≈ 0.8  # 80% 的 episode 会成功，20% 会失败

# 状态3：衣服已折叠完成
V("衣服已折叠完成") ≈ 1.0  # 100% 的 episode 会成功

# 状态4：衣服掉落在地
V("衣服掉落") ≈ 0.1  # 10% 的 episode 会成功，90% 会失败
```

### 1.5.4 Value Function 与 Advantage 计算的关系

**Advantage 计算公式**：
$$
A^\pi(o_t, a_t, \ell) = \sum_{t'=t}^T r_{t'} - V^\pi(o_t, \ell)
$$

**计算流程**：
```python
# 伪代码：使用 Value Function 计算 Advantage
def compute_advantage(observation, rewards, value_function, task_label):
    # 1. 计算实际累积奖励（离散值：0 或 1）
    episode_return = sum(rewards[t:])  # Σ_{t'=t}^T r_{t'} = 0 或 1
    
    # 2. Value Function 预测状态价值（连续值：0 到 1 之间）
    predicted_value = value_function(observation, task_label)  # V(o_t) ∈ [0, 1]
    
    # 3. 计算 Advantage（连续值：可以是负数、零或正数）
    advantage = episode_return - predicted_value  # A = R - V
    # 例如：A = 1 - 0.8 = 0.2（正优势）
    # 例如：A = 0 - 0.3 = -0.3（负优势）
    
    return advantage
```

**为什么需要 Value Function？**
1. **提供基准**：Value Function 预测"这个状态的平均价值"（连续值）
2. **计算 Advantage**：Advantage = 实际回报（离散：0/1）- 预测价值（连续：0-1）
3. **指导学习**：正 Advantage → 好的行为，负 Advantage → 差的行为

**关键点**：
- ✅ **连续值输出**：Value Function 输出连续值，不是离散值
- ✅ **范围 [0, 1]**：训练目标在 [0, 1] 范围内（对应成功/失败）
- ✅ **不是完成度**：预测的是累积奖励期望，不是任务完成百分比
- ✅ **无显式归一化**：没有 clamp 或 sigmoid，但训练会自然限制范围

### 1.5.4 与 Q 值估计的区别

| 特性 | Q 值网络 Q(s,a) | Value Function V(s) |
|------|----------------|---------------------|
| **输入** | (状态, 动作) | (状态, 任务标签) |
| **输出** | Q(s,a) 标量 | V(s) 标量 |
| **训练方式** | TD 学习 | 监督学习（MSE） |
| **目标值** | r + γQ(s',a') | Σ_{t'=t}^T r_{t'} |
| **用途** | 选择最优动作 | 评估状态价值 |

**关键区别**：
- ❌ **不是 Q 值网络**：Value Function 不依赖动作，只评估状态价值
- ✅ **状态价值函数**：预测"从这个状态开始，能获得多少累积奖励"
- ✅ **监督学习**：使用累积奖励作为目标，不是 TD 误差

---

## 问题 2：Advantage 计算、作用机制及具体例子

### 2.1 Advantage 的计算方式

**关键澄清**：Advantage **不是 action head 给出的数值**，而是**通过公式计算的数值**。

**预训练阶段的 Advantage 计算公式**：
$$
A^\pi(o_t, a_t, \ell) = \sum_{t'=t}^T r_{t'} - V^\pi(o_t, \ell)
$$

**计算步骤**：
1. **计算累积奖励**：$\sum_{t'=t}^T r_{t'}$（从当前时间步到 episode 末尾的实际累积奖励）
2. **预测状态价值**：$V^\pi(o_t, \ell)$（Value Function 预测的当前状态价值）
3. **计算 Advantage**：两者相减得到 Advantage

**伪代码**：
```python
# 伪代码：Advantage 计算
def compute_advantage(observation, rewards, value_function, task_label):
    # 1. 计算从当前时间步到 episode 末尾的累积奖励
    episode_return = sum(rewards[t:])  # Σ_{t'=t}^T r_{t'}
    
    # 2. Value Function 预测当前状态价值
    predicted_value = value_function(observation, task_label)  # V(o_t)
    
    # 3. 计算 Advantage
    advantage = episode_return - predicted_value  # A = R - V
    
    return advantage
```

**关键点**：
- ✅ **公式计算**：Advantage 是通过公式计算的，不是模型直接输出
- ✅ **需要 Value Function**：必须先训练好 Value Function 才能计算 Advantage
- ✅ **连续数值**：Advantage 是连续标量值，可以是正数、负数或零

### 2.2 Advantage 如何作用于后续任务特定微调

**作用机制**：

#### 步骤 1：Advantage 二值化
$$
I_t = \begin{cases}
1 & \text{if } A^\pi(o_t, a_t, \ell) > \epsilon_\ell \\
0 & \text{otherwise}
\end{cases}
$$

#### 步骤 2：转换为模型输入
- `I_t = 1`（正优势）→ 模型学习"好的行为"
- `I_t = 0`（负优势）→ 模型学习"差的行为"（或忽略）

#### 步骤 3：策略条件化
$$
\pi_\theta(a_t | o_t, \ell, I_t)
$$

策略网络被条件化在 advantage indicator 上，学习：
- **高优势时**：模仿成功轨迹的动作
- **低优势时**：避免失败轨迹的动作

**在微调阶段的作用**：
1. **数据筛选**：只使用高优势的数据进行训练
2. **策略改进**：通过 advantage conditioning 引导策略学习更好的行为
3. **迭代优化**：多轮迭代，逐步提高策略性能

### 2.3 为什么要设定 Advantage？

**核心原因**：

1. **区分数据质量**：
   - 不是所有演示数据都同样优秀
   - Advantage 量化了"这个轨迹比平均水平好多少"

2. **提供学习信号**：
   - 稀疏奖励（只有成功/失败）提供的信息有限
   - Advantage 提供了**密集的、连续的梯度信息**

3. **指导策略学习**：
   - 通过 advantage conditioning，模型学会"在什么情况下采用什么动作"
   - 类似于"模仿学习 + 质量加权"

4. **支持离线 RL**：
   - 不需要在线交互，只需要离线数据
   - 通过 advantage 实现策略改进

### 2.4 具体例子：叠衣服任务

#### 正例：高 Advantage（I_t = 1）

**场景描述**：
- 机器人成功地将一件 T 恤从篮子中取出
- 正确地将衣服摊平（领口朝上）
- 按照正确顺序折叠（先对折，再对折）
- 最终放在桌面右上角的指定位置

**Advantage 计算**：
```python
# 时间步 t=0（开始取衣服）
observation_0 = "篮子中有T恤，桌面空"
rewards = [0, 0, 0, ..., 0, 1]  # 最后成功
episode_return = sum(rewards) = 1.0
predicted_value = value_function(observation_0, "fold_cloth") = 0.7
advantage_0 = 1.0 - 0.7 = +0.3  # ✅ 正优势

# 时间步 t=50（正在折叠）
observation_50 = "衣服已摊平，正在对折"
predicted_value_50 = value_function(observation_50, "fold_cloth") = 0.8
advantage_50 = 1.0 - 0.8 = +0.2  # ✅ 正优势

# 二值化（假设阈值 ε = 0.1）
I_0 = 1  # advantage > threshold
I_50 = 1  # advantage > threshold
```

**模型学习**：
- 策略网络看到 `I_t = 1`，学习"这是好的行为"
- 模型会模仿这个轨迹的动作序列
- 在类似情况下，模型倾向于采用相同的动作

#### 反例：低 Advantage（I_t = 0）

**场景描述**：
- 机器人取出了 T 恤
- **错误**：将衣服领口朝下放置（应该是朝上）
- 尝试折叠，但因为方向错误导致折叠困难
- 最终衣服掉落，任务失败

**Advantage 计算**：
```python
# 时间步 t=0（开始取衣服）
observation_0 = "篮子中有T恤，桌面空"
rewards = [0, 0, 0, ..., 0, 0]  # 最后失败
episode_return = sum(rewards) = 0.0
predicted_value = value_function(observation_0, "fold_cloth") = 0.7
advantage_0 = 0.0 - 0.7 = -0.7  # ❌ 负优势

# 时间步 t=30（错误放置衣服）
observation_30 = "衣服领口朝下放置"
predicted_value_30 = value_function(observation_30, "fold_cloth") = 0.6
advantage_30 = 0.0 - 0.6 = -0.6  # ❌ 负优势

# 二值化（假设阈值 ε = 0.1）
I_0 = 0  # advantage < threshold
I_30 = 0  # advantage < threshold
```

**模型学习**：
- 策略网络看到 `I_t = 0`，学习"这是差的行为"
- 模型会避免这个轨迹的动作序列
- 在类似情况下，模型倾向于采用不同的动作（如正确放置衣服）

#### 对比总结

| 方面 | 正例（高 Advantage） | 反例（低 Advantage） |
|------|-------------------|-------------------|
| **动作序列** | 正确取衣 → 正确摊平 → 正确折叠 → 成功 | 正确取衣 → **错误摊平** → 折叠困难 → 失败 |
| **Advantage** | +0.3（正） | -0.7（负） |
| **Indicator** | I_t = 1 | I_t = 0 |
| **模型学习** | ✅ 模仿这个行为 | ❌ 避免这个行为 |
| **结果** | 提高成功率 | 降低失败率 |

---

## 问题 3：任务特定微调阶段的 Advantage 重新计算

### 3.1 为什么需要重新计算 Advantage？

**核心原因**：

#### 1. **分布转移（Distribution Shift）**

**预训练阶段**：
- 使用多任务、多机器人的离线数据
- Advantage 计算使用整个 episode（N = T）
- Value Function 是在通用数据上训练的

**微调阶段**：
- 使用特定任务的在线数据
- 数据分布可能与预训练数据不同
- Value Function 需要适应新任务

**示例**：
```python
# 预训练：通用 Value Function
# 训练数据：多种任务（叠衣服、制作咖啡、组装纸箱等）
value_function_pretrain = train_on_multi_task_data()

# 微调：任务特定 Value Function
# 训练数据：只有叠衣服任务的数据
value_function_finetune = fine_tune_on_fold_cloth_data(value_function_pretrain)

# 如果使用预训练的 Value Function 计算 Advantage，可能不准确
# 因为预训练的 Value Function 没有见过这个特定任务的数据分布
```

#### 2. **Advantage 计算方式不同**

**预训练**：
$$
A^\pi(o_t, a_t) = \sum_{t'=t}^T r_{t'} - V^\pi(o_t)
$$
- 使用整个 episode 的累积奖励
- 方差较高，但在大规模数据上可接受

**微调**：
$$
A^\pi(o_t, a_t) = \sum_{t'=t}^{t+N-1} r_{t'} + V^\pi(o_{t+N}) - V^\pi(o_t)
$$
- 使用 N-step lookahead（N = 50）
- 更关注"当前动作对近期成功的影响"
- 方差较低，更精确

#### 3. **Value Function 需要更新**

**原因**：
- 微调阶段收集了新数据（自主 rollouts + 专家干预）
- 这些数据反映了当前策略的实际表现
- Value Function 需要在这些新数据上微调，才能准确估计价值

**流程**：
```
1. 部署策略收集新数据
   ↓
2. 使用预训练的 Value Function 计算初始 Advantage
   ↓
3. 在新数据上微调 Value Function
   ↓
4. 使用微调后的 Value Function 重新计算 Advantage
   ↓
5. 使用新的 Advantage 更新策略
```

### 3.2 Advantage 计算参考的定义

**微调阶段的 Advantage 计算公式**（论文 Section III-F）：
$$
A^\pi(o_t, a_t, \ell) = \sum_{t'=t}^{t+N-1} r_{t'} + V^\pi(o_{t+N}, \ell) - V^\pi(o_t, \ell)
$$

其中：
- $N = 50$（固定 lookahead 步数）
- $r_{t'}$ 是每个时间步的奖励
- $V^\pi(o_{t+N}, \ell)$ 是 N 步后的状态价值（由 Value Function 预测）
- $V^\pi(o_t, \ell)$ 是当前状态价值（由 Value Function 预测）

**计算步骤**：
```python
# 伪代码：微调阶段的 Advantage 计算
def compute_advantage_finetune(observation, rewards, value_function, task_label, n_steps=50):
    # 1. 预测当前状态价值
    current_value = value_function(observation, task_label)  # V(o_t)
    
    # 2. 计算 N-step return
    n_step_return = sum(rewards[t:t+n_steps])  # Σ_{t'=t}^{t+N-1} r_{t'}
    
    # 3. 如果有未来状态，加上未来状态价值
    if t + n_steps < len(observations):
        next_observation = observations[t + n_steps]
        next_value = value_function(next_observation, task_label)  # V(o_{t+N})
        n_step_return += next_value
    
    # 4. 计算 Advantage
    advantage = n_step_return - current_value
    
    return advantage
```

**为什么使用 N-step lookahead？**
- **更精确**：关注动作的即时影响，而不是整个 episode
- **更稳定**：方差较低，适合小规模数据
- **更实用**：在实际部署中，我们更关心"这个动作对接下来几步的影响"

### 3.3 如何动态设定 Advantage 阈值？

**论文中的方法**（Section III-F）：

#### 方法 1：百分位数方法（论文采用）

**预训练阶段**：
- 选择阈值使得**约 30%** 的演示数据有正优势
- 基于随机采样的 10k 数据点计算

**微调阶段**：
- 选择阈值使得**约 40%** 的评估 rollouts 有正优势
- 基于当前迭代收集的所有数据计算

**实现代码**：
```python
# 伪代码：动态阈值设定
def compute_advantage_threshold(advantages, target_percentile=60):
    """
    计算 Advantage 阈值
    
    Args:
        advantages: 所有数据点的 Advantage 值列表
        target_percentile: 目标百分位数（60 = 约40%正优势）
    
    Returns:
        threshold: Advantage 阈值
    """
    # 计算百分位数
    threshold = np.percentile(advantages, target_percentile)
    
    # 二值化
    advantage_indicators = (advantages >= threshold).astype(int)
    
    # 验证：检查正优势的比例
    positive_ratio = np.mean(advantage_indicators)
    print(f"Positive advantage ratio: {positive_ratio:.2%}")
    
    return threshold, advantage_indicators

# 使用示例
advantages = [0.3, -0.2, 0.5, -0.1, 0.4, ...]  # 所有数据点的 Advantage
threshold, indicators = compute_advantage_threshold(advantages, target_percentile=60)
# threshold ≈ 0.15（假设）
# indicators = [1, 0, 1, 0, 1, ...]  # 约40%为1
```

#### 方法 2：任务特定调整

**论文中的特殊情况**（Section III-F）：
- **T-shirt 和 shorts 叠衣服任务**：设置为约 **10%** 正优势
- **原因**：高质量演示数据训练出的策略较慢但成功率高
- **目的**：通过降低阈值，只学习最优秀的行为

**实现代码**：
```python
# 伪代码：任务特定阈值设定
def compute_task_specific_threshold(advantages, task_name):
    """
    根据任务名称设定不同的阈值
    """
    if task_name == "fold_tshirt_shorts":
        # 特殊情况：只选择最优秀的10%数据
        target_percentile = 90  # 90th percentile = 10%正优势
    else:
        # 默认：40%正优势
        target_percentile = 60
    
    threshold = np.percentile(advantages, target_percentile)
    return threshold
```

#### 方法 3：迭代自适应调整

**动态调整策略**：
```python
# 伪代码：迭代自适应阈值
def adaptive_threshold_adjustment(advantages_history, current_advantages):
    """
    根据历史性能动态调整阈值
    """
    # 1. 计算当前性能
    current_performance = evaluate_policy(current_policy)
    
    # 2. 如果性能提升，可以适当提高阈值（更严格）
    if current_performance > previous_performance:
        target_percentile = 65  # 从60提高到65（更严格）
    # 3. 如果性能下降，可以适当降低阈值（更宽松）
    elif current_performance < previous_performance:
        target_percentile = 55  # 从60降低到55（更宽松）
    else:
        target_percentile = 60  # 保持不变
    
    threshold = np.percentile(current_advantages, target_percentile)
    return threshold
```

### 3.4 完整微调流程

**迭代改进流程**（论文 Algorithm 1）：
```python
# 伪代码：完整的微调流程
policy = load_pretrained_policy()
value_function = load_pretrained_value_function()

for iteration in range(num_iterations):
    # === 步骤 1：部署收集数据 ===
    collected_data = deploy_and_collect(policy)
    # 包含：自主 rollouts + 专家干预 + reward 反馈
    
    # === 步骤 2：微调 Value Function ===
    value_function = fine_tune_value_function(
        collected_data,
        value_function,
        n_steps=50  # N-step lookahead
    )
    
    # === 步骤 3：重新计算 Advantage ===
    advantages = []
    for batch in collected_data:
        batch_advantages = compute_advantage_finetune(
            batch["observation"],
            batch["rewards"],
            value_function,  # 使用微调后的 Value Function
            batch["task"],
            n_steps=50
        )
        advantages.extend(batch_advantages)
    
    # === 步骤 4：动态设定阈值 ===
    if task_name == "fold_tshirt_shorts":
        target_percentile = 90  # 10%正优势
    else:
        target_percentile = 60  # 40%正优势
    
    threshold = np.percentile(advantages, target_percentile)
    
    # === 步骤 5：二值化 Advantage ===
    advantage_indicators = (np.array(advantages) >= threshold).astype(int)
    
    # === 步骤 6：更新策略 ===
    for batch in collected_data:
        batch["advantage_indicator"] = advantage_indicators[...]
        
        # Advantage conditioning dropout（30%）
        dropout_mask = (np.random.rand(len(batch)) < 0.3)
        batch["advantage_indicator"][dropout_mask] = None
        
        # 训练条件策略
        predicted_actions = policy(
            batch["observation"],
            batch["task"],
            advantage_indicator=batch["advantage_indicator"]
        )
        
        policy_loss = flow_matching_loss(predicted_actions, batch["action"])
        policy.backward(policy_loss)
        policy.update()
    
    # === 步骤 7：评估（可选）===
    if evaluate:
        performance = evaluate_policy(policy)
        print(f"Iteration {iteration}: Performance = {performance}")
```

---

## 总结

### 问题 1：Reward 设计来源
- ✅ **来源**：多任务离线数据集中的任务完成信号（成功/失败）
- ✅ **计算**：整个 episode 的累积奖励（N = T）
- ✅ **特点**：稀疏奖励，只在 episode 末尾给出

### 问题 2：Advantage 计算和作用机制
- ✅ **计算方式**：通过公式计算（不是 action head 输出）
- ✅ **公式**：$A = \sum r - V(s)$（预训练）或 $A = \sum_{t}^{t+N-1} r + V(s_{t+N}) - V(s)$（微调）
- ✅ **作用**：通过二值化和条件化，指导策略学习好的行为，避免差的行为
- ✅ **例子**：正例（高 Advantage）→ 学习成功轨迹；反例（低 Advantage）→ 避免失败轨迹

### 问题 3：微调阶段的 Advantage 重新计算
- ✅ **原因**：分布转移、计算方式不同、Value Function 需要更新
- ✅ **计算参考**：N-step lookahead（N = 50），使用微调后的 Value Function
- ✅ **阈值设定**：百分位数方法（默认 40% 正优势，特殊情况 10%）

---

## 参考文献

- [π*₀.₆: a VLA That Learns From Experience](https://www.pi.website/download/pistar06.pdf)
- Section III-F: Additional algorithm details
- Section III-E: Using CFG for test-time policy improvement
