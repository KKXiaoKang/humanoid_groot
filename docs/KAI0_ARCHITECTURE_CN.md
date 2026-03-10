# Kai0 (χ₀) 后训练框架完整架构解析

> **基于论文 arXiv:2602.09021 和代码库的全面分析**

## 核心定位

**Kai0 (χ₀) 不是传统的Actor-Critic强化学习框架**。它是一个专门为解决**分布不一致性（Distributional Inconsistencies）**问题而设计的**离线模仿学习框架**，通过三个核心技术模块实现生产级鲁棒性：

1. **Model Arithmetic**：权重空间模型融合
2. **Stage Advantage**：阶段感知优势估计
3. **Train-Deploy Alignment**：训练-部署对齐

**关键特性**：
- ✅ 基于行为克隆（BC）和优势加权行为克隆（AWBC）
- ✅ 完全离线学习，无需在线交互
- ✅ 使用预计算的优势标签，而非动态Q值更新
- ✅ 通过prompt conditioning实现优势加权
- ❌ **没有**replay buffer、Q值网络、策略梯度等传统RL组件

## 框架架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kai0 后训练框架架构                           │
└─────────────────────────────────────────────────────────────────┘

数据流：
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ 离线数据集   │ --> │ 优势估计器    │ --> │ AWBC训练     │
│ (LeRobot)   │     │ (Advantage   │     │ (Policy)     │
│             │     │  Estimator)  │     │              │
└─────────────┘     └──────────────┘     └──────────────┘
      │                    │                    │
      │                    │                    │
      ▼                    ▼                    ▼
  原始数据           优势标签              策略网络
  (观察+动作)        (task_index)          (π0/π0.5)
```

## 核心问题解答

### 问题1: Actor Policy如何收集transition数据，buffer传输给Critic Policy用于学习？

**答案：Kai0不使用传统的Actor-Critic架构，也没有transition buffer机制。**

#### 实际的数据收集流程：

1. **离线数据集准备**：
   - 使用LeRobot格式的离线数据集（HDF5/Parquet格式）
   - 数据包含：观察（图像+状态）、动作序列、元数据（episode_index, timestamp等）
   - 数据来源：人类演示、DAgger收集、或其他离线数据源

2. **数据加载机制**（`src/openpi/training/data_loader.py`）：
```python
# 数据加载器创建流程
def create_data_loader(config):
    # 1. 创建数据集（LeRobotDataset或AdvantageLerobotDataset）
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    
    # 2. 应用数据变换（归一化、图像resize、tokenization等）
    dataset = transform_dataset(dataset, data_config)
    
    # 3. 创建PyTorch DataLoader（支持shuffle、多进程）
    data_loader = TorchDataLoader(dataset, batch_size, shuffle=True)
    
    return data_loader
```

3. **训练时的数据流**（`scripts/train.py`）：
```python
# 训练循环
data_loader = create_data_loader(config)
data_iter = iter(data_loader)  # 无限迭代器

for step in range(num_train_steps):
    batch = next(data_iter)  # 从数据集中采样batch
    # batch格式: (Observation, Actions)
    # Observation包含: images, state, prompt等
    # Actions包含: action序列
    
    train_state, info = train_step(config, rng, train_state, batch)
```

**关键点**：
- ❌ **没有在线交互**：所有数据都是预先收集好的离线数据
- ❌ **没有replay buffer**：使用PyTorch DataLoader直接从数据集采样
- ❌ **没有Critic Policy**：这是一个监督学习框架，不是RL
- ✅ **数据随机采样**：通过DataLoader的`shuffle=True`实现随机采样

---

### 问题2: Critic Policy如何针对buffer里面的数据随机排序采样，用于更新Q值网络？

**答案：Kai0没有Critic Policy和Q值网络。它使用优势估计器（Advantage Estimator）来预测优势值，而不是Q值。**

#### 实际的优势估计流程：

1. **优势估计器训练**（Stage 1，`stage_advantage/annotation/`）：
   - **目标**：训练一个模型来预测每个时间步的优势值（progress差异）
   - **架构**：基于π0模型的变体，输出优势/进度值而非动作
   - **损失函数**：回归损失（预测progress差异）

```python
# 优势估计器配置（src/openpi/models/pi0_config.py）
class AdvantageEstimatorConfig(Pi0Config):
    loss_action_weight: float = 0.0  # 禁用动作预测
    loss_value_weight: float = 1.0    # 启用值预测
```

2. **优势标签生成**（Stage 0，`stage_advantage/annotation/gt_label.py`）：
   - 计算每个帧的优势值：`advantage[i] = progress[i + chunk_size] - progress[i]`
   - 将优势值离散化为`task_index`（二进制或多分片）
   - 写入`meta/tasks.jsonl`，映射`task_index`到prompt字符串

3. **优势估计推理**（Stage 2，`stage_advantage/annotation/eval.py`）：
   - 使用训练好的优势估计器对新数据集进行推理不是GAN的判别器（分类任务）
   - 输出：`relative_advantage`, `absolute_value`, `absolute_advantage`
   - 结果写入parquet文件的额外列

**关键点**：
- ❌ **没有Q值网络**：使用优势估计器预测优势，而非Q(s,a)
- ❌ **没有buffer随机采样**：直接从数据集采样，通过DataLoader的shuffle实现
- ✅ **优势加权**：通过prompt conditioning实现，而非显式的样本权重

---

### 问题3: 策略网络Actor如何进行更新？

**答案：策略网络通过标准的行为克隆损失进行更新，使用优势加权的prompt conditioning。**

#### 策略网络更新流程：

1. **损失计算**（`src/openpi/models/pi0.py`）：
```python
@override
def compute_loss(
    self, rng, observation: Observation, actions: Actions, *, train: bool = False
) -> Array:
    # 1. 预处理观察（图像、状态、prompt）
    observation = preprocess_observation(preprocess_rng, observation, train=train)
    
    # 2. 添加噪声（扩散模型训练）
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    
    # 3. 前向传播
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
    
    # 4. Transformer前向传播
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
    )
    
    # 5. 动作预测
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
    
    # 6. 计算MSE损失
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

2. **训练步骤**（`scripts/train.py`）：
```python
def train_step(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    
    observation, actions = batch
    
    # 计算损失和梯度
    loss, grads = nnx.value_and_grad(loss_fn)(
        model, train_rng, observation, actions
    )
    
    # 更新参数
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_state, info
```

3. **优势加权机制（AWBC）**：
   - **不是显式的样本权重**：而是通过**prompt conditioning**实现
   - 每个样本的prompt包含优势标签（如"fold the cloth, Advantage: positive"）
   - 模型学习根据不同的prompt输出不同的动作
   - 高优势样本（positive）的prompt会引导模型学习更好的行为

```python
# PromptFromLeRobotTask变换（src/openpi/transforms.py）
class PromptFromLeRobotTask(DataTransformFn):
    def __call__(self, data):
        task_index = int(data["task_index"])
        prompt = self.tasks[task_index]  # 例如: "fold the cloth, Advantage: positive"
        return {**data, "prompt": prompt}
```

**关键点**：
- ✅ **监督学习**：使用MSE损失进行行为克隆（见```189:214:src/openpi/models/pi0.py```）
- ✅ **优势加权**：通过prompt conditioning隐式实现，而非显式样本权重
- ✅ **扩散模型**：使用扩散过程训练动作预测（时间步t从1到0的去噪过程）
- ❌ **不是RL更新**：没有策略梯度、Q学习或actor-critic更新
- ✅ **训练步骤**：标准梯度下降（见```140:194:scripts/train.py```），使用`nnx.value_and_grad`计算梯度

---

## 详细代码流程分析

### 1. 数据加载与采样流程

**代码路径**：`src/openpi/training/data_loader.py`

```python
# 创建数据加载器（scripts/train.py:228-234）
data_loader = _data_loader.create_data_loader(
    config,
    sharding=data_sharding,
    shuffle=True,  # ✅ 随机打乱
)
data_iter = iter(data_loader)  # 无限迭代器
batch = next(data_iter)  # 每次从数据集采样batch
```

**关键实现**：
- `TorchDataLoader`（```448:536:src/openpi/training/data_loader.py```）：使用PyTorch DataLoader
- `shuffle=True`：每个epoch随机打乱数据顺序
- 无限迭代：数据集遍历完后自动重新开始

**❌ 没有replay buffer**：所有buffer相关代码都在推理模块（`train_deploy_alignment/inference/`），用于时序平滑，不是训练用的replay buffer

### 2. 优势估计器训练流程

**代码路径**：`src/openpi/models_pytorch/pi0_pytorch.py:464-592`

```python
class AdvantageEstimator(PI0Pytorch):
    def forward(self, observation, actions, ...):
        # 1. 动作预测损失（可选，通常loss_action_weight=0.0）
        loss_action = F.mse_loss(u_t, v_t, reduction="none").mean(dim=-1)
        
        # 2. 值预测损失（核心：预测progress差异）
        deep_rep = suffix_out_full[:, 0, :]
        value_pred = self.value_head(deep_rep)  # MLP输出单个值
        progress_tgt = obs_full.progress.float()  # 目标：progress差异
        value_loss = F.mse_loss(value_pred, progress_tgt, reduction="none")
        
        # 3. 总损失
        loss = loss_action * self.loss_action_weight + value_loss * self.loss_value_weight
        return loss
```

**关键点**：
- ✅ **不是Q值网络**：预测的是progress差异（优势），而非Q(s,a)
- ✅ **预计算目标**：`progress_tgt`来自数据集（```97:97:src/openpi/training/advantage_dataset.py```）
- ✅ **监督学习**：使用MSE损失回归，不是TD学习

### 2.1 优势估计器的本质：监督式价值函数学习

**重要观察**：优势估计器确实更像是一个**reward/value函数**，但它使用的是**监督式价值学习（Supervised Value Learning）**方法。

**关键澄清**：虽然优势估计器预测的是连续值（progress/advantage），但**最终用于训练的是离散的task_index**（0或1），通过prompt conditioning实现优势加权。

#### 方法类型分析

**1. 不是传统的Critic网络**：
- ❌ 不是Q(s,a)网络（不依赖动作）
- ❌ 不是通过TD误差更新
- ✅ 是V(s)或progress预测器（只依赖观察）

**2. 不是Discriminator（判别器）**：
- ❌ 不是GAN的判别器（分类任务）
- ❌ 不是对抗训练
- ✅ 是回归任务（预测连续值）

**3. 实际方法类型**：

这种方法最接近以下几种方法的组合：

**a) Reward Modeling / Value Function Pretraining**：
- 使用预定义的ground truth reward（progress）训练value function
- 类似于InstructGPT中的reward model训练方式
- 但这里是预测progress差异，而非人类偏好

**b) Supervised Value Learning**：
- 使用监督学习训练value function
- 目标是从观察预测progress值
- 类似于行为克隆，但预测的是value而非action

**c) Progress Prediction / Task Progress Estimation**：
- 预测任务完成进度（0-1）
- 类似于goal-conditioned RL中的goal progress
- 但这里是预定义的progress标签

#### 具体实现机制

**训练阶段**（```571:578:src/openpi/models_pytorch/pi0_pytorch.py```）：
```python
# 1. 提取深度表示
deep_rep = suffix_out_full[:, 0, :]  # 状态token的表示

# 2. 通过MLP预测progress值
value_pred = self.value_head(deep_rep)  # Shape: (B, 1)

# 3. 与ground truth progress比较不是GAN的判别器（分类任务）
progress_tgt = obs_full.progress.float()  # 来自数据集
value_loss = F.mse_loss(value_pred, progress_tgt, reduction="none")
```

**推理阶段**（```437:439:stage_advantage/annotation/evaluator.py```）：
```python
# 预测当前帧的progress值
absolute_val = self.model.sample_values(device, observation)

# 计算优势：progress[n+50] - progress[n]
absolute_advantage = absolute_value[n+50] - absolute_value[n]
```

#### 与传统方法的对比

| 特性 | 传统Critic (Q-learning) | 传统Reward Model | Kai0优势估计器 |
|------|------------------------|------------------|----------------|
| **输入** | (s, a) | (s, a) | s (观察) |
| **输出** | Q(s,a) | r(s,a) | V(s) = progress |
| **训练方式** | TD学习 | 监督学习（人类标注） | 监督学习（progress标签） |
| **目标** | 最大化累积reward | 预测人类偏好 | 预测任务进度 |
| **更新频率** | 每个step | 批量更新 | 批量更新 |
| **标签来源** | 环境reward | 人类标注 | 预定义progress |

#### 为什么这种方法有效？

1. **Progress作为密集信号**：
   - Progress提供了每个时间步的密集反馈
   - 比稀疏的episode-level reward更稳定
   - 类似于curriculum learning中的progress signal

2. **阶段感知设计**：
   - 将长时域任务分解为阶段
   - 在每个阶段内计算优势，避免跨阶段数值不稳定
   - 类似于hierarchical RL中的sub-goal progress

3. **监督式学习优势**：
   - 不需要在线交互
   - 训练稳定（MSE损失）
   - 可以预计算所有优势标签

#### 方法归类

这种方法可以归类为：

**"Supervised Progress-Based Value Learning"** 或 **"Progress-Guided Advantage Estimation"**

核心思想：
- 使用预定义的progress标签（ground truth）训练value function
- 预测任务完成进度，而非累积reward
- 通过progress差异计算优势，用于AWBC训练

**与传统RL的区别**：
- 传统RL：reward → value function → advantage
- Kai0：progress (ground truth) → value function → advantage → **离散化为task_index** → prompt conditioning

**完整流程**：
```
连续advantage值 → 离散化为task_index (0/1) → 映射为prompt字符串 → prompt conditioning
```

这是一种**离线、监督式的价值学习**方法，结合了：
- Reward modeling的思想（监督学习value function）
- Progress prediction的思想（预测任务进度）
- Advantage-weighted BC的思想（使用优势加权策略训练）
- **Prompt conditioning**：通过语言条件隐式实现优势加权（而非显式样本权重）

### 3. 策略网络训练流程

**代码路径**：`scripts/train.py:140-194`

```python
def train_step(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    observation, actions = batch
    
    # 1. 计算损失（MSE，不是策略梯度）
    def loss_fn(model, rng, observation, actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)  # ✅ 标准监督学习损失
    
    # 2. 计算梯度（标准梯度下降）
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions
    )
    
    # 3. 更新参数（Adam/AdamW优化器）
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_state, info
```

**损失计算**（```189:214:src/openpi/models/pi0.py```）：
```python
def compute_loss(self, rng, observation, actions, *, train=False):
    # 1. 添加噪声（扩散模型）
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    
    # 2. 前向传播
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
    
    # 3. MSE损失（✅ 不是策略梯度）
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

### 4. AWBC优势加权实现

**代码路径**：`src/openpi/transforms.py:342-356`

```python
class PromptFromLeRobotTask(DataTransformFn):
    def __call__(self, data):
        task_index = int(data["task_index"])  # 0或1（低/高优势）
        prompt = self.tasks[task_index]  # 映射到prompt字符串
        # 例如: "fold the cloth, Advantage: positive"
        return {**data, "prompt": prompt}
```

**数据流**：
1. 数据集包含`task_index`列（```89:90:src/openpi/training/advantage_dataset.py```）
2. `PromptFromLeRobotTask`将`task_index`映射为prompt（```342:356:src/openpi/transforms.py```）
3. 模型通过prompt conditioning学习不同优势下的行为
4. **❌ 没有显式样本权重**：优势加权通过语言条件隐式实现

### 5. 优势标签生成流程

**代码路径**：`stage_advantage/annotation/gt_label.py`

```python
def calculate_rewards(data, chunk_size=50, advantage_source="progress"):
    if advantage_source == "absolute_advantage":
        # 使用预计算的绝对优势
        rewards = data['absolute_advantage'].values
    elif advantage_source == "progress":
        # 计算progress差异
        for i in range(n_frames):
            if i + chunk_size < n_frames:
                rewards[i] = progress[i + chunk_size] - progress[i]
    return rewards

# 离散化为task_index（二进制模式）
# task_index=0: 低优势（bottom 70%）
# task_index=1: 高优势（top 30%）
```

**关键点**：
- ✅ **预计算**：优势值在训练前计算好，写入parquet文件
- ✅ **离散化**：连续优势值 → 离散task_index（0或1）
- ✅ **阶段感知**：可按`stage_progress_gt`分阶段计算优势

---

## DAgger：持续自我改进机制

**重要**：虽然Kai0是离线学习框架，但它通过**DAgger（Dataset Aggregation）**机制实现持续自我改进！

### DAgger工作流程

DAgger允许模型在部署过程中**从失败中学习**，实现持续改进：

```
┌─────────────────────────────────────────────────────────────┐
│              DAgger 持续改进循环                              │
└─────────────────────────────────────────────────────────────┘

1. 初始训练
   └─> 使用初始数据集训练策略模型

2. 部署推理
   └─> 模型在真实机器人上执行任务

3. 检测失败/需要改进
   └─> 操作员观察模型行为，发现错误

4. 进入DAgger模式（按'd'键）
   └─> 暂停模型推理
   └─> 切换到人工控制模式

5. 人工演示纠正
   └─> 操作员通过主臂（master arm）演示正确行为
   └─> 系统记录：observation + 人工动作 + intervention=1

6. 保存数据（按's'键）
   └─> 保存为HDF5格式（包含intervention标签）
   └─> 可选：保存视频

7. 恢复推理（按'r'键）
   └─> 退出DAgger模式
   └─> 继续使用模型控制（intervention=0）

8. 重新训练
   └─> 将新收集的DAgger数据合并到数据集
   └─> 重新训练模型
   └─> 回到步骤2，形成改进循环
```

### 代码实现细节

**1. DAgger模式切换**（```720:737:train_deploy_alignment/dagger/agilex/agilex_openpi_dagger_collect.py```）：

```python
# 检查DAgger模式激活
if dagger_mode_active and not dagger_mode_entered:
    print("⏸️  INFERENCE PAUSED - ENTERING DAGGER MODE")
    ros_operator.enter_dagger_mode()  # 切换到人工控制
    dagger_mode_entered = True
elif not dagger_mode_active and dagger_mode_entered:
    print("▶️  DAGGER MODE DEACTIVATED - RESUMING INFERENCE")
    ros_operator.data_collector.stop_collection()
    dagger_mode_entered = False
```

**2. 数据收集**（```515:543:train_deploy_alignment/dagger/arx/arx_openpi_dagger_collect.py```）：

```python
def add_frame(self, observation, action, intervention: int = 1):
    """添加一帧数据
    
    Args:
        intervention: 0=模型执行, 1=人工纠正（默认1）
    """
    # 非阻塞队列，后台线程写入
    self._frame_queue.put_nowait((obs_copy, action_copy, intervention))
```

**3. 数据保存**（```434:471:train_deploy_alignment/dagger/arx/arx_openpi_dagger_collect.py```）：

```python
def _do_save(self):
    """保存当前episode（后台线程执行）"""
    dataset_path = f"episode_{self.episode_idx}"
    
    # 保存HDF5 + intervention标签 + 可选视频
    save_data(timesteps_copy, actions_copy, dataset_path, 
              interventions=interventions_copy)
    
    if export_video:
        save_videos(timesteps_copy, dataset_path, camera_names)
```

**关键特性**：
- ✅ **非阻塞收集**：使用后台线程和队列，不影响实时控制
- ✅ **Intervention标签**：区分模型动作（0）和人工纠正（1）
- ✅ **无缝切换**：推理模式 ↔ DAgger模式，无需重启
- ✅ **数据格式**：HDF5 + 视频，兼容LeRobot格式

### 持续改进循环

```
初始数据集（20小时）
    ↓
训练模型 v1
    ↓
部署到机器人
    ↓
执行任务（可能失败）
    ↓
按'd'进入DAgger模式
    ↓
人工演示纠正（收集新数据）
    ↓
按's'保存数据
    ↓
合并到数据集
    ↓
重新训练 → 模型 v2（改进版）
    ↓
部署模型 v2
    ↓
（继续循环...）
```

### 与传统在线RL的区别

| 特性 | 传统在线RL | Kai0 DAgger |
|------|-----------|-------------|
| **数据收集** | 自动探索+试错 | 人工监督+纠正 |
| **改进方式** | 策略梯度更新 | 重新训练（离线） |
| **失败处理** | 自动从replay buffer学习 | 人工介入演示 |
| **数据质量** | 可能包含大量失败数据 | 人工筛选高质量纠正 |
| **训练频率** | 每个step更新 | 收集足够数据后批量训练 |

### 为什么DAgger有效？

1. **针对性改进**：只在模型失败的地方收集纠正数据
2. **高质量数据**：人工演示确保数据质量
3. **分布对齐**：在真实部署分布上收集数据，解决train-deploy gap
4. **渐进式改进**：每次迭代都基于前一个版本，逐步提升性能

### 实际使用示例

```bash
# 1. 启动策略服务器（GPU机器）
python scripts/serve_policy.py --checkpoint_path <model_checkpoint>

# 2. 在IPC上运行DAgger收集脚本
python train_deploy_alignment/dagger/agilex/agilex_openpi_dagger_collect.py \
    --host <gpu_host_ip> --port 8000 \
    --dataset_name my_dagger_data

# 3. 操作流程：
#    - 按Enter：开始推理（模型控制）
#    - 按'd'：进入DAgger模式（人工控制）
#    - 演示纠正动作
#    - 按's'：保存当前episode
#    - 按'r'：恢复推理模式
#    - 重复...

# 4. 合并数据并重新训练
python scripts/merge_lerobot.py --input <dagger_data> --output <merged_dataset>
uv run scripts/train.py <config> --exp_name=improved_model
```

### DAgger数据使用优势估计器重新标注流程

**关键问题**：DAgger收集的新数据没有GT progress标签，如何用优势估计器标注？

**完整流程**：

```
┌─────────────────────────────────────────────────────────────────┐
│      DAgger数据 → 优势标注 → 重新训练的完整流程                  │
└─────────────────────────────────────────────────────────────────┘

步骤1: DAgger数据收集
  └─> HDF5格式（observations, actions, videos）
  └─> 保存位置：<dataset_dir>/<dataset_name>/episode_*.hdf5

步骤2: 转换为LeRobot格式
  └─> 使用convert_h5_lerobot.py
  └─> 输入：HDF5 + videos
  └─> 输出：LeRobot格式（parquet + videos + meta）

步骤3: 使用优势估计器推理标注
  └─> 使用eval.py对新数据集进行推理
  └─> 输出：包含advantage列的parquet文件

步骤4: 离散化为task_index
  └─> 使用gt_label.py将advantage离散化
  └─> 输出：包含task_index的parquet + tasks.jsonl

步骤5: 合并到原数据集
  └─> 使用merge_lerobot.py合并
  └─> 输出：合并后的数据集

步骤6: 重新训练策略
  └─> 使用合并后的数据集训练AWBC
```

**详细命令**：

```bash
# ===== 步骤1: DAgger数据收集（已完成） =====
# 数据保存在：~/data/dagger/my_dagger_data/episode_*.hdf5

# ===== 步骤2: 转换为LeRobot格式 =====
cd train_deploy_alignment/data_augment/utils
# 安装mini_lerobot依赖
uv pip install -e ../mini_lerobot
export PYTHONPATH="${PYTHONPATH}:$(pwd)/mini_lerobot"

# 转换HDF5 → LeRobot格式
python convert_h5_lerobot.py \
    /path/to/dagger_data \
    /path/to/output \
    my_dagger_data \
    --prompt "fold the cloth" \
    --save-repoid dagger_lerobot \
    --max-workers 8

# 输出：/path/to/output/flatten_fold/dagger_lerobot/
#       ├── data/chunk-*/episode_*.parquet
#       ├── videos/chunk-*/.../episode_*.mp4
#       └── meta/

# ===== 步骤3: 使用优势估计器推理标注 =====
# 更新eval.py中的MODELS_CONFIG_MAP，指向训练好的优势估计器checkpoint
# 然后运行推理：

uv run python stage_advantage/annotation/eval.py \
    Flatten-Fold KAI0 \
    /path/to/output/flatten_fold/dagger_lerobot

# 输出：/path/to/output/flatten_fold/dagger_lerobot/
#       └── data_KAI0_100000/chunk-*/episode_*.parquet
#           （包含relative_advantage, absolute_value, absolute_advantage列）

# ===== 步骤4: 离散化为task_index =====
# 使用gt_label.py将advantage值离散化为task_index

# 方法1: 使用gt_labeling.sh脚本（推荐）
# 编辑gt_labeling.sh，设置DATA_PATH指向dagger_lerobot目录
# 脚本会自动将data_KAI0_100000复制到data/目录并运行gt_label.py

bash stage_advantage/annotation/gt_labeling.sh

# 方法2: 手动操作
# 1. 创建新目录用于标注
mkdir -p /path/to/dagger_labeled
cp -r /path/to/output/flatten_fold/dagger_lerobot/videos /path/to/dagger_labeled/
cp -r /path/to/output/flatten_fold/dagger_lerobot/meta /path/to/dagger_labeled/
# 2. 将data_KAI0_100000复制为data目录（gt_label.py从data/读取）
cp -r /path/to/output/flatten_fold/dagger_lerobot/data_KAI0_100000 \
      /path/to/dagger_labeled/data
# 3. 运行gt_label.py
python stage_advantage/annotation/gt_label.py \
    /path/to/dagger_labeled \
    --threshold 30 \
    --chunk-size 50 \
    --discretion-type binary \
    --advantage-source absolute_advantage \
    --stage-nums 1

# 输出：更新后的parquet文件（包含task_index列）+ meta/tasks.jsonl

# ===== 步骤5: 合并到原数据集 =====
# 将DAgger数据合并到原始训练数据集

python scripts/merge_lerobot.py \
    --src_paths \
        /path/to/original_dataset \
        /path/to/output/flatten_fold/dagger_lerobot \
    --tgt_path /path/to/merged_dataset \
    --repo_id merged_dataset \
    --fps 30 \
    --robot-type agilex \
    --force

# ===== 步骤6: 重新训练策略 =====
# 使用合并后的数据集训练AWBC

# 更新config.py中的repo_id指向合并后的数据集
# 然后训练：

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_flatten_fold_awbc \
    --exp_name=dagger_improved_v2
```

**关键点**：

1. **DAgger数据格式**：HDF5格式，需要转换为LeRobot格式才能使用优势估计器
2. **优势估计器推理**：不需要GT progress标签，直接从观察预测advantage值
3. **优势来源**：使用`--advantage-source absolute_advantage`，从优势估计器的预测值计算
4. **数据目录结构**：`eval.py`输出到`data_KAI0_100000/`，`gt_label.py`需要从`data/`读取，需要复制
5. **数据合并**：使用`merge_lerobot.py`将新数据合并到原数据集，保持格式一致

**优势估计器的作用**：

- ✅ **替代GT标签**：对于没有GT progress的新数据，使用训练好的优势估计器预测advantage
- ✅ **批量标注**：可以快速标注大量DAgger数据，无需人工标注
- ✅ **一致性保证**：使用相同的优势估计器，确保标注方式与训练数据一致
- ✅ **持续改进**：随着优势估计器改进，新数据的标注质量也会提升

**为什么这种方法有效**：

- ✅ **无需GT标签**：优势估计器从观察预测advantage，不需要人工标注progress
- ✅ **自动标注**：批量推理可以快速标注大量DAgger数据
- ✅ **保持一致性**：使用相同的优势估计器，确保标注方式一致
- ✅ **渐进改进**：每次DAgger迭代都基于最新的优势估计器标注

---

## 完整训练流程

### Stage 0: GT数据标注
```bash
# 计算优势值并离散化为task_index
python stage_advantage/annotation/gt_label.py <dataset_path> \
    --threshold 30 --chunk-size 50 --discretion-type binary \
    --advantage-source absolute_advantage
```

### Stage 1: 训练优势估计器
```bash
# 训练优势估计器（预测progress/advantage）
uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
    --exp_name=run1 --save_interval 10000
```

### Stage 2: 优势估计推理
```bash
# 使用训练好的估计器对新数据标注优势值
uv run python stage_advantage/annotation/eval.py Task-A KAI0 /path/to/dataset
```

### Stage 3: AWBC训练
```bash
# 使用优势标注的数据训练策略
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_flatten_fold_awbc --exp_name=run1
```

---

## 数据流详解

### 1. 数据加载流程

```
LeRobot数据集 (Parquet + Videos)
    │
    ├─> LeRobotDataset / AdvantageLerobotDataset
    │       │
    │       ├─> 读取parquet文件
    │       ├─> 解码视频帧
    │       └─> 提取观察、动作、元数据
    │
    ├─> TransformedDataset
    │       │
    │       ├─> RepackTransform (重组数据结构)
    │       ├─> DataTransforms (数据变换)
    │       ├─> Normalize (归一化)
    │       ├─> ModelTransforms (模型输入变换)
    │       └─> PromptFromLeRobotTask (AWBC: 添加prompt)
    │
    └─> TorchDataLoader
            │
            ├─> Shuffle (随机打乱)
            ├─> Batch (批处理)
            └─> 多进程加载
```

### 2. 训练时的数据采样

```python
# 在scripts/train.py中
data_loader = create_data_loader(config, shuffle=True)
data_iter = iter(data_loader)  # 无限迭代器

for step in range(num_train_steps):
    batch = next(data_iter)  # 自动从数据集中随机采样batch
    # 如果数据集遍历完，DataLoader会自动重新开始
    # shuffle=True确保每次epoch的数据顺序不同
```

### 3. AWBC的优势加权机制

```
优势标注数据集
    │
    ├─> task_index列 (0或1，表示低/高优势)
    │
    ├─> meta/tasks.jsonl
    │       │
    │       ├─> task_index=0 -> "fold the cloth, Advantage: negative"
    │       └─> task_index=1 -> "fold the cloth, Advantage: positive"
    │
    └─> PromptFromLeRobotTask变换
            │
            └─> 将task_index映射为prompt
                    │
                    └─> 模型通过prompt conditioning学习
                            │
                            ├─> positive prompt -> 学习高优势行为
                            └─> negative prompt -> 学习低优势行为
```

---

## 与传统RL的区别

| 特性 | 传统Actor-Critic RL | Kai0 (AWBC) |
|------|-------------------|-------------|
| **数据来源** | 在线交互收集 | 离线数据集 |
| **Buffer机制** | Replay Buffer | PyTorch DataLoader |
| **Critic网络** | Q(s,a)或V(s) | 优势估计器（预测progress差异） |
| **Actor更新** | 策略梯度 | 监督学习（MSE损失） |
| **优势使用** | TD误差/GAE | Prompt conditioning |
| **样本权重** | 显式权重 | 隐式（通过prompt） |
| **训练模式** | 在线学习 | 离线学习 |

---

## 代码验证总结

通过全面代码审查（grep搜索RL关键词、分析训练循环、损失函数、数据流），确认：

### ✅ 确认的事实

1. **没有RL组件**：
   - ❌ 无replay buffer（只有推理时的action buffer用于时序平滑）
   - ❌ 无Q值网络（只有优势估计器的value head用于预测progress）
   - ❌ 无策略梯度（使用标准梯度下降）
   - ❌ 无TD误差/GAE（优势是预计算的）

2. **训练机制**：
   - ✅ 标准监督学习：```153:154:scripts/train.py``` - `jnp.mean(chunked_loss)`
   - ✅ MSE损失：```214:214:src/openpi/models/pi0.py``` - `jnp.mean(jnp.square(v_t - u_t))`
   - ✅ 数据采样：```228:234:scripts/train.py``` - PyTorch DataLoader with shuffle

3. **优势加权实现**：
   - ✅ Prompt conditioning：```342:356:src/openpi/transforms.py``` - `PromptFromLeRobotTask`
   - ✅ 优势估计器：```464:592:src/openpi/models_pytorch/pi0_pytorch.py``` - `AdvantageEstimator`类
   - ✅ 预计算标签：```38:69:stage_advantage/annotation/gt_label.py``` - 从progress计算advantage

### 📊 架构对比

| 组件 | 传统Actor-Critic RL | Kai0框架 | 代码证据 |
|------|-------------------|---------|---------|
| **数据收集** | 在线交互+Replay Buffer | 离线数据集+DataLoader | `data_loader.py:448-536` |
| **Critic网络** | Q(s,a)迭代更新 | 优势估计器（预计算） | `pi0_pytorch.py:464-592` |
| **Actor更新** | 策略梯度 | 监督学习MSE | `train.py:140-194` |
| **优势使用** | TD误差/GAE | Prompt conditioning | `transforms.py:342-356` |
| **损失函数** | Policy loss + Value loss | MSE loss | `pi0.py:189-214` |

### 🎯 核心创新点

1. **Stage Advantage**：将长时域任务分解为语义阶段，在每个阶段内计算优势，避免数值不稳定
2. **Prompt-based AWBC**：通过语言条件隐式实现优势加权，而非显式样本权重
3. **Model Arithmetic**：多模型权重融合，无需MoE架构复杂度
4. **Train-Deploy Alignment**：显式处理训练-部署分布差异

### 📝 最终结论

Kai0是一个**工程化的离线模仿学习框架**，通过以下机制实现后训练：

1. **数据收集**：使用离线数据集（LeRobot格式），通过DataLoader随机采样
2. **优势估计**：训练独立的优势估计器预测progress差异，预计算优势标签
3. **优势加权**：通过prompt conditioning实现AWBC（`task_index` → prompt字符串）
4. **策略更新**：使用标准的行为克隆损失（MSE），通过扩散模型训练

**关键理解**：
- ❌ **不是**Actor-Critic RL（无Q值网络、无策略梯度、无replay buffer）
- ✅ **是**基于行为克隆的离线学习框架
- ✅ 优势加权通过**prompt conditioning**实现（语言条件隐式加权）
- ✅ 使用**预计算的优势标签**，而非动态Q值更新
- ✅ 训练流程是**标准监督学习**，使用MSE损失和梯度下降
