# GR-RL Critic Transformer 架构详解

> **基于 [GR-RL 论文](https://arxiv.org/pdf/2512.01801) 的架构分析**

## 核心问题

**GR-RL 的 Critic Transformer 具体模型架构是怎么样的？**

---

## 一、论文中的架构描述

### 1.1 整体架构概述

根据论文 **Section 2: The GR-RL Model**，GR-RL 采用 **Mixture-of-Transformer (MoT) 架构**：

**论文原文**：
> "GR-RL adopts a Mixture-of-Transformer architecture, consisting of a vision-language-action (VLA) model π_θ and a multi-task critic Q_φ with a total of 5B parameters."

**关键信息**：
- ✅ **架构类型**：Mixture-of-Transformer (MoT)
- ✅ **组件**：VLA 模型 π_θ + multi-task critic Q_φ
- ✅ **总参数量**：5B 参数

---

### 1.2 Critic Transformer 的输入输出（基于 Figure 2）

根据论文 **Figure 2: The GR-RL Model** 的描述：

**Critic Transformer 的输入**：
1. **kv cache**：来自 Vision-Language Model 的 key-value cache（共享上下文表示）
2. **proprios**：本体感觉信息（机器人状态）
3. **actions**：动作序列

**Critic Transformer 的输出**：
- **value predictions**：价值预测（Q 值分布）

**训练目标**：
- 使用 **distributional reinforcement learning**
- 通过 **cross entropy** 损失训练
- 预测 **value distribution**（价值分布）

---

## 二、Critic Transformer 架构推断

### 2.1 基于论文信息的架构推断

虽然论文没有详细说明 Critic Transformer 的具体层数和维度，但可以从以下信息推断：

**架构组成**（推断）：

```
Critic Transformer
├── 输入层
│   ├── kv cache（来自 Vision-Language Model）
│   ├── proprios（本体感觉信息）
│   └── actions（动作序列）
│
├── Transformer 编码器
│   ├── Multi-head Attention
│   ├── Feed-Forward Network
│   └── Layer Normalization
│
└── 输出层
    ├── Distributional Value Head
    │   ├── 预测价值分布（而非单个值）
    │   └── 使用 cross entropy 损失
    └── Q(s,a) 输出
```

---

### 2.2 关键架构特点

#### 1. 共享 Vision-Language Model 的 kv cache

**论文描述**：
> "kv cache" 从 Vision-Language Model 传递到 Critic Transformer

**含义**：
- ✅ Critic 和 Actor **共享 Vision-Language Model** 的特征提取
- ✅ Critic 接收 **kv cache**（预计算的上下文表示）
- ✅ 降低计算成本（不需要重新编码视觉和语言输入）

**架构优势**：
- ✅ **参数共享**：减少参数量
- ✅ **计算效率**：复用 Vision-Language Model 的特征
- ✅ **一致性**：Critic 和 Actor 使用相同的视觉-语言表示

---

#### 2. Distributional Critic（分布式 Critic）

**论文原文**：
> "We adopt distributional critics and observe that they give much more robust performance under offline sparse reward scenarios."

**关键特点**：
- ✅ **预测价值分布**：而非单个 Q 值
- ✅ **Cross Entropy 损失**：用于训练分布预测
- ✅ **鲁棒性**：在离线稀疏奖励场景下表现更好

**架构设计**：
```python
# 伪代码：Distributional Critic
class CriticTransformer:
    def forward(self, kv_cache, proprios, actions):
        # 1. 融合输入
        inputs = concat([kv_cache, proprios, actions])
        
        # 2. Transformer 编码
        features = transformer_encoder(inputs)
        
        # 3. Distributional Value Head
        value_distribution = distributional_head(features)
        # 输出：价值分布（例如：51 个 bin 的分布）
        
        return value_distribution
```

---

#### 3. Multi-task Critic

**论文描述**：
> "multi-task critic Q_φ"

**含义**：
- ✅ Critic 可以处理**多个任务**
- ✅ 可能通过**任务条件化**实现（类似 π*₀.₆ 的 task label）

**架构推断**：
```python
# 伪代码：Multi-task Critic
class CriticTransformer:
    def forward(self, kv_cache, proprios, actions, task_label=None):
        # 1. 任务条件化（可选）
        if task_label is not None:
            task_embedding = embed_task(task_label)
            inputs = concat([kv_cache, proprios, actions, task_embedding])
        else:
            inputs = concat([kv_cache, proprios, actions])
        
        # 2. Transformer 编码
        features = transformer_encoder(inputs)
        
        # 3. Distributional Value Head
        value_distribution = distributional_head(features)
        
        return value_distribution
```

---

## 三、与 π*₀.₆ Value Function 的对比

### 3.1 架构对比

| 特性 | GR-RL Critic Transformer | π*₀.₆ Value Function |
|------|-------------------------|---------------------|
| **架构类型** | Transformer（共享 VLM） | 独立小型 VLA 模型 |
| **输入** | kv cache + proprios + actions | obs + task_label |
| **输出** | Q(s,a) 分布 | V(s) 标量 |
| **训练方式** | Distributional RL（TD 学习） | 监督学习（MSE） |
| **参数共享** | ✅ 与 Actor 共享 VLM | ❌ 完全独立 |
| **价值类型** | Q(s,a)（状态-动作价值） | V(s)（状态价值） |

---

### 3.2 关键区别

**GR-RL Critic Transformer**：
- ✅ **Q-function**：预测 Q(s,a)，需要动作输入
- ✅ **共享架构**：与 Actor 共享 Vision-Language Model
- ✅ **Distributional**：预测价值分布，而非单个值
- ✅ **离线 RL**：使用 TD 学习训练

**π*₀.₆ Value Function**：
- ✅ **V-function**：预测 V(s)，不需要动作输入
- ✅ **独立模型**：完全独立的 VLA 模型
- ✅ **标量输出**：预测单个值
- ✅ **监督学习**：使用 MSE 损失训练

---

## 四、Critic Transformer 的训练流程

### 4.1 训练目标

**论文描述**：
> "we train a critic model on both successful and failed trajectories with offline reinforcement learning (RL). Given a sparse reward at the end of the episode, the predicted value naturally reflects the progress of the task."

**训练流程**：

```python
# 伪代码：Critic Transformer 训练
for batch in offline_dataset:
    # 1. 获取输入
    kv_cache = vision_language_model.get_kv_cache(obs, language)
    proprios = batch["proprios"]
    actions = batch["actions"]
    
    # 2. Critic 前向传播
    predicted_q_distribution = critic_transformer(
        kv_cache, proprios, actions
    )
    
    # 3. 计算目标 Q 值分布（使用 TD 学习）
    # 对于稀疏奖励：只有最后一步有奖励
    target_q_distribution = compute_target_distribution(
        rewards, next_obs, next_actions, critic_transformer
    )
    
    # 4. Distributional Loss（Cross Entropy）
    loss = cross_entropy_loss(
        predicted_q_distribution, 
        target_q_distribution
    )
    
    # 5. 更新 Critic
    critic_transformer.backward(loss)
    critic_transformer.update()
```

---

### 4.2 Distributional RL 的优势

**论文原文**：
> "We adopt distributional critics and observe that they give much more robust performance under offline sparse reward scenarios."

**优势**：
1. **鲁棒性**：在离线稀疏奖励场景下表现更好
2. **不确定性估计**：可以估计价值的不确定性
3. **更好的泛化**：分布预测比点估计更稳定

---

## 五、Critic Transformer 的使用方式

### 5.1 Q 值作为 Progress Function

**论文描述**：
> "Given a sparse reward at the end of the episode, the predicted value naturally reflects the progress of the task, which we further use to filter only transitions that contribute positively to the progress and discard the rest."

**使用流程**：

```python
# 伪代码：使用 Q 值过滤演示轨迹
for transition in demonstration_trajectories:
    # 1. 获取输入
    kv_cache = vision_language_model.get_kv_cache(obs, language)
    proprios = transition["proprios"]
    action = transition["action"]
    
    # 2. 预测 Q 值分布
    q_distribution = critic_transformer(kv_cache, proprios, action)
    
    # 3. 计算期望 Q 值（从分布中提取）
    q_value = compute_expected_value(q_distribution)
    
    # 4. 过滤：只保留 Q 值高的 transition
    if q_value > threshold:
        filtered_trajectories.append(transition)
    else:
        # 丢弃低 Q 值的 transition
        pass

# 5. 在过滤后的轨迹上训练策略
policy = train_bc(filtered_trajectories)
```

---

## 六、架构细节推断

### 6.1 Transformer 编码器结构（推断）

基于标准的 Transformer 架构和论文描述：

```python
class CriticTransformer(nn.Module):
    def __init__(self, config):
        # 1. 输入投影层
        self.input_projection = nn.Linear(
            kv_cache_dim + proprios_dim + action_dim,
            hidden_dim
        )
        
        # 2. Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        
        # 3. Distributional Value Head
        self.distributional_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms * num_actions)
            # num_atoms: 分布中的 bin 数量（例如：51）
            # num_actions: 动作空间大小（如果是离散动作）
        )
    
    def forward(self, kv_cache, proprios, actions):
        # 1. 融合输入
        inputs = torch.cat([kv_cache, proprios, actions], dim=-1)
        inputs = self.input_projection(inputs)
        
        # 2. Transformer 编码
        features = self.transformer_encoder(inputs)
        
        # 3. 提取最后一个时间步的特征
        last_features = features[:, -1, :]
        
        # 4. Distributional Value Head
        q_distribution = self.distributional_head(last_features)
        # 输出形状：(batch_size, num_atoms * num_actions)
        
        return q_distribution
```

---

### 6.2 与 Actor 的共享架构

**关键设计**：
- ✅ **共享 Vision-Language Model**：Critic 和 Actor 共享视觉-语言编码器
- ✅ **kv cache 传递**：Critic 接收 Actor 的 kv cache
- ✅ **参数效率**：减少总参数量

**架构图**：
```
Vision-Language Model（共享）
    │
    ├── kv cache ──> Action Diffusion Transformer（Actor）
    │
    └── kv cache ──> Critic Transformer
                        │
                        ├── proprios
                        ├── actions
                        └── Q(s,a) 分布输出
```

---

## 七、总结

### 7.1 Critic Transformer 的关键特点

1. **架构类型**：
   - ✅ Mixture-of-Transformer (MoT)
   - ✅ 与 Actor 共享 Vision-Language Model

2. **输入**：
   - ✅ kv cache（来自 Vision-Language Model）
   - ✅ proprios（本体感觉信息）
   - ✅ actions（动作序列）

3. **输出**：
   - ✅ Q(s,a) 分布（Distributional Critic）
   - ✅ 使用 cross entropy 损失训练

4. **训练方式**：
   - ✅ 离线 RL（TD 学习）
   - ✅ 稀疏奖励信号
   - ✅ 成功和失败轨迹都参与训练

5. **用途**：
   - ✅ Q 值作为 Progress Function
   - ✅ 过滤演示轨迹（只保留高 Q 值的 transition）

---

### 7.2 与 π*₀.₆ Value Function 的区别

| 维度 | GR-RL Critic Transformer | π*₀.₆ Value Function |
|------|-------------------------|---------------------|
| **价值类型** | Q(s,a)（状态-动作价值） | V(s)（状态价值） |
| **输入** | kv cache + proprios + actions | obs + task_label |
| **架构** | Transformer（共享 VLM） | 独立小型 VLA 模型 |
| **输出** | Q 值分布 | V 值标量 |
| **训练** | Distributional RL（TD） | 监督学习（MSE） |
| **用途** | 过滤演示轨迹 | 计算 Advantage |

---

### 7.3 关键洞察

**GR-RL Critic Transformer 的设计理念**：
- ✅ **共享架构**：与 Actor 共享 Vision-Language Model，提高参数效率
- ✅ **Distributional**：预测价值分布，提高鲁棒性
- ✅ **Q-function**：预测 Q(s,a)，需要动作输入
- ✅ **过滤机制**：使用 Q 值过滤演示轨迹，提高数据质量

**与 π*₀.₆ 的区别**：
- ✅ GR-RL：Q-function，共享架构，Distributional
- ✅ π*₀.₆：V-function，独立模型，标量输出

---

## 八、论文中的具体信息总结

### 8.1 明确提到的信息

1. **架构类型**：Mixture-of-Transformer (MoT)
2. **总参数量**：5B 参数
3. **组件**：VLA 模型 π_θ + multi-task critic Q_φ
4. **Critic 类型**：Distributional Critic
5. **训练方式**：离线 RL + 稀疏奖励
6. **输入**：kv cache + proprios + actions（基于 Figure 2）
7. **输出**：value predictions（Q 值分布）

### 8.2 未明确说明的信息

1. **Transformer 层数**：未明确说明
2. **隐藏维度**：未明确说明
3. **注意力头数**：未明确说明
4. **Distributional Head 的具体设计**：未明确说明（bin 数量等）
5. **与 Actor 的参数共享细节**：未明确说明

---

## 九、建议

要获得 GR-RL Critic Transformer 的**精确架构细节**，建议：

1. **查看论文的补充材料**：可能包含更详细的架构描述
2. **查看官方代码**：如果字节跳动开源了代码
3. **联系作者**：直接询问架构细节
4. **参考类似工作**：参考其他 Distributional Critic 的实现

---

## 十、参考

- [GR-RL 论文](https://arxiv.org/pdf/2512.01801)
- Figure 2: The GR-RL Model（论文中的架构图）
- Section 2: The GR-RL Model（论文中的模型描述）
