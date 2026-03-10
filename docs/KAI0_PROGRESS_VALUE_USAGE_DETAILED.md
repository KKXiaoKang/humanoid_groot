# Kai0 Progress 值（Value）的完整使用流程详解

> **详细解析 Kai0 预测的 Progress 值如何被使用，以及与 π*₀.₆ 的区别**

## 核心问题

**Kai0 预测的是 Progress（任务进度），不是未来期望收益值（如 π*₀.₆ 的 V(s)）。这个 Progress 值用来做什么？**

---

## 一、Progress 值的本质

### 1.1 什么是 Progress？

**Progress（任务进度）**：
- **定义**：任务完成的百分比（0-1 或 -1 到 1）
- **含义**：当前状态距离任务完成的进度
- **来源**：预定义的 ground truth 标签（不是从 reward 计算）

**代码证据**（```574:574:src/openpi/models_pytorch/pi0_pytorch.py```）：
```python
progress_tgt = torch.clamp(obs_full.progress.float(), -1.0, 1.0)
# progress 来自数据集，是预定义的 ground truth
```

### 1.2 与 π*₀.₆ 的 V(s) 的区别

| 特性 | Kai0 Progress | π*₀.₆ V(s) |
|------|--------------|------------|
| **含义** | 任务完成进度（0-1） | 状态价值（累积奖励期望） |
| **来源** | 预定义的 ground truth | 从奖励计算（n-step return） |
| **范围** | [-1, 1]（归一化） | $\mathbb{R}$（未归一化） |
| **依赖** | 不依赖环境奖励 | 依赖环境奖励信号 |

---

## 二、Progress 值的完整使用流程

### 阶段 1：预测 Progress 值

**优势估计器预测 Progress**（```597:644:src/openpi/models_pytorch/pi0_pytorch.py```）：

```python
@torch.no_grad()
def sample_values(self, device, observation):
    """
    预测当前观察的 progress 值
    """
    # 1. VLA 模型前向传播
    suffix_out = self.paligemma_with_expert.forward(...)
    
    # 2. 提取状态 token 表示
    deep_rep = suffix_out[:, 0, :]  # 状态 token 表示
    
    # 3. 通过 MLP Value Head 预测 Progress
    value_pred = self.value_head(deep_rep)  # Shape: (B, 1)，范围：[-1, 1]
    
    return value_pred  # 这就是 Progress 值
```

**输出**：
- `absolute_value`：当前帧的 Progress 值（例如：0.3, 0.5, 0.7 等）

---

### 阶段 2：计算 Advantage（优势）

**关键步骤**：使用 Progress 值计算 Advantage

#### 方式 1：使用 GT Progress（训练数据准备阶段）

**代码位置**（```38:69:stage_advantage/annotation/gt_label.py```）：

```python
def calculate_rewards(data: pd.DataFrame, chunk_size: int = 50, 
                     advantage_source: str = "progress"):
    """
    计算 advantage：progress[i+50] - progress[i]
    """
    if advantage_source == "progress":
        progress = data['progress'].values  # GT progress
        for i in range(n_frames):
            if i + chunk_size < n_frames:
                rewards[i] = progress[i + chunk_size] - progress[i]
            else:
                # 边界处理
                rewards[i] = (progress[-1] - progress[i]) / (len(progress) - i) * chunk_size
    return rewards
```

**公式**：
$$
A_t = \text{Progress}(o_{t+N}) - \text{Progress}(o_t)
$$

其中：
- $N = 50$（chunk_size，固定向前看 50 帧）
- $\text{Progress}(o_t)$ = 当前帧的 Progress
- $\text{Progress}(o_{t+N})$ = 未来 N 帧后的 Progress
- $A_t$ = Advantage（Progress 差异）

**含义**：
- **正值**：未来 Progress 增加，表示"向好的方向发展"
- **负值**：未来 Progress 减少或不变，表示"没有进展或倒退"

---

#### 方式 2：使用预测的 Progress（推理阶段）

**代码位置**（```467:480:stage_advantage/annotation/evaluator.py```）：

```python
# 1. 预测当前帧和未来帧的 Progress
absolute_val_arr = self.model.sample_values(device, absolute_observation)  # 当前帧
future_result = all_results_dict.get(future_frame_idx)  # 未来帧

# 2. 计算 absolute_advantage
if future_frame_idx == frame_idx:
    result["absolute_advantage"] = 0.0
elif future_frame_idx - frame_idx != relative_interval:
    # 归一化处理
    result["absolute_advantage"] = (
        future_result["absolute_value"] - result["absolute_value"]
    ) / (future_frame_idx - frame_idx) * relative_interval
else:
    # 标准计算：未来 Progress - 当前 Progress
    result["absolute_advantage"] = (
        future_result["absolute_value"] - result["absolute_value"]
    )

# 3. 裁剪到 [-1, 1]
result["absolute_advantage"] = max(-1.0, min(1.0, result["absolute_advantage"]))
```

**公式**：
$$
\text{absolute\_advantage}_t = \text{Progress}(o_{t+N}) - \text{Progress}(o_t)
$$

**关键点**：
- ✅ **使用预测的 Progress**：不需要 GT progress 标签
- ✅ **可以用于新数据**：对于没有 GT progress 的新数据，可以使用训练好的优势估计器预测

---

### 阶段 3：离散化为 task_index

**目的**：将连续的 Advantage 值转换为离散的标签（0 或 1）

**代码位置**（```223:263:stage_advantage/annotation/gt_label.py```）：

```python
def assign_task_index(parquet_file: str, threshold_percentile: float, ...):
    """
    将 advantage 离散化为 task_index
    """
    # 1. 计算 advantage（rewards）
    rewards = calculate_rewards(df, chunk_size, advantage_source)
    
    # 2. 离散化（binary mode）
    if discretion_type == "binary":
        # Binary mode: task_index = 0 for rewards below threshold, 1 for >= threshold
        task_index = (rewards >= threshold_percentile).astype(np.int32)
    elif discretion_type == "n_slices":
        # n-slices mode: task_index from 0 to (n_slices-1)
        # ...
    
    # 3. 写入 parquet 文件
    df['task_index'] = task_index
    df.to_parquet(parquet_file, index=False)
```

**离散化过程**：

1. **计算 Advantage 分布**：统计所有帧的 advantage 值
2. **设置阈值**：例如，选择阈值使得约 30% 的数据有正优势
3. **二值化**：
   - `advantage >= threshold` → `task_index = 1`（正优势）
   - `advantage < threshold` → `task_index = 0`（负优势）

**示例**：
```python
# 假设 advantage 分布：[0.7, 0.5, 0.3, 0.1, -0.2, -0.5, ...]
# 阈值：70th percentile = 0.3

# 离散化结果：
# advantage = 0.7 → task_index = 1  (正优势)
# advantage = 0.5 → task_index = 1  (正优势)
# advantage = 0.3 → task_index = 1  (正优势)
# advantage = 0.1 → task_index = 0  (负优势)
# advantage = -0.2 → task_index = 0  (负优势)
# advantage = -0.5 → task_index = 0  (负优势)
```

---

### 阶段 4：映射为 Prompt 字符串

**目的**：将离散的 task_index 映射为语言 prompt，用于策略 conditioning

**代码位置**（```342:356:src/openpi/transforms.py```）：

```python
class PromptFromLeRobotTask(DataTransformFn):
    def __call__(self, data):
        task_index = int(data["task_index"])  # 0 或 1
        prompt = self.tasks[task_index]  # 映射为 prompt 字符串
        # 例如：
        # task_index=0 → "fold the cloth, Advantage: negative"
        # task_index=1 → "fold the cloth, Advantage: positive"
        return {**data, "prompt": prompt}
```

**Prompt 映射**（```190:220:stage_advantage/annotation/gt_label.py```）：

```python
def update_tasks_jsonl(base_path: str, discretion_type: str, n_slices: int = 10):
    """
    更新 tasks.jsonl 文件，映射 task_index → prompt 字符串
    """
    if discretion_type == "binary":
        tasks = [
            {"task_index": 0, "task": "fold the cloth, Advantage: negative"},
            {"task_index": 1, "task": "fold the cloth, Advantage: positive"},
        ]
    elif discretion_type == "n_slices":
        for i in range(n_slices):
            tasks.append({"task_index": i, "task": f"fold the cloth, Advantage: {i}"})
    
    # 写入 tasks.jsonl
    with open(tasks_file, 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')
```

**映射结果**：
- `task_index = 0` → `"fold the cloth, Advantage: negative"`
- `task_index = 1` → `"fold the cloth, Advantage: positive"`

---

### 阶段 5：用于策略训练（Prompt Conditioning）

**目的**：通过 prompt conditioning 实现优势加权，引导策略学习高优势行为

**训练流程**（```67:99:src/openpi/training/advantage_dataset.py```）：

```python
class AdvantageLerobotDataset(LeRobotDataset):
    def __getitem__(self, idx: int) -> dict:
        # 1. 获取样本
        item = self.get_sample_with_imgs_from_idx(idx)
        
        # 2. 获取 task_index
        task_idx = item["task_index"].item()  # 0 或 1
        
        # 3. 映射为 prompt
        episode_level_dict["task"] = self.meta.tasks[task_idx]
        # 例如："fold the cloth, Advantage: positive"
        
        # 4. 返回包含 prompt 的样本
        final_item = {**final_item, **episode_level_dict}
        return final_item
```

**策略训练**（```189:214:src/openpi/models/pi0.py```）：

```python
def compute_loss(self, observation, actions, ...):
    # 1. 观察包含 prompt（例如："fold the cloth, Advantage: positive"）
    observation = preprocess_observation(observation)
    
    # 2. 策略 conditioning 在 prompt 上
    predicted_actions = self.forward(observation)  # 使用 prompt 作为条件
    
    # 3. 计算 MSE 损失
    loss = MSE(predicted_actions, actions)
    
    return loss
```

**关键机制**：
- ✅ **Prompt Conditioning**：策略通过语言 prompt 学习不同的行为
- ✅ **隐式优势加权**：
  - `task_index=1`（positive）→ prompt 引导学习高优势行为
  - `task_index=0`（negative）→ prompt 引导学习低优势行为
- ✅ **监督学习**：使用 MSE 损失，不是策略梯度

---

## 三、完整数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│          Kai0 Progress 值的完整使用流程                          │
└─────────────────────────────────────────────────────────────────┘

阶段 1：预测 Progress 值
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 观察 o_t      │ --> │ 优势估计器    │ --> │ Progress 值   │
│ (图像+语言+状态)│     │ (VLA + MLP) │     │ ([-1, 1])    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                    │
                                                    ▼
阶段 2：计算 Advantage
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Progress[t]   │     │              │     │ Advantage    │
│ Progress[t+N] │ --> │ 计算差异      │ --> │ (连续值)      │
│ (N=50)       │     │              │     │ A = P[t+N]-P[t]│
└──────────────┘     └──────────────┘     └──────────────┘
                                                    │
                                                    ▼
阶段 3：离散化为 task_index
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Advantage     │ --> │ 阈值化        │ --> │ task_index   │
│ (连续值)      │     │ (例如：70th)  │     │ (0 或 1)     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                    │
                                                    ▼
阶段 4：映射为 Prompt
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ task_index   │ --> │ tasks.jsonl  │ --> │ Prompt       │
│ (0 或 1)     │     │ (映射表)     │     │ (字符串)     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                    │
                                                    ▼
阶段 5：策略训练（Prompt Conditioning）
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 观察 + Prompt │ --> │ 策略网络      │ --> │ 动作预测      │
│              │     │ (π0/π0.5)   │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## 四、为什么使用 Progress 而不是未来期望收益？

### 4.1 设计动机

**问题**：长时域 manipulation 任务中 reward 极其稀疏
- 例如：叠衣服任务，只有最后成功才有 reward（+1）
- 中间步骤没有 reward（0）
- Trajectory-level advantage 非常 noisy

**解决方案**：使用 Progress（任务进度）
- ✅ **密集信号**：每个时间步都有 progress 值（0-1）
- ✅ **阶段感知**：可以分解为多个阶段，每个阶段有独立的 progress
- ✅ **稳定训练**：Progress 差异比稀疏 reward 更稳定

### 4.2 与 π*₀.₆ 的对比

| 特性 | Kai0 (Progress) | π*₀.₆ (V(s)) |
|------|----------------|--------------|
| **信号密度** | 密集（每个时间步） | 稀疏（依赖奖励） |
| **来源** | 预定义标签 | 环境奖励 |
| **稳定性** | 高（Progress 平滑变化） | 中等（奖励可能噪声） |
| **适用场景** | 长时域任务 | 短时域任务 |
| **优势计算** | Progress 差异 | Return - V(s) |

---

## 五、实际应用示例

### 5.1 训练数据准备（使用 GT Progress）

```python
# 步骤 1：数据集包含 GT progress
data = {
    "frame_0": {"progress": 0.0},
    "frame_50": {"progress": 0.3},
    "frame_100": {"progress": 0.6},
    "frame_150": {"progress": 0.9},
    "frame_200": {"progress": 1.0},
}

# 步骤 2：计算 advantage
advantage[0] = progress[50] - progress[0] = 0.3 - 0.0 = 0.3  # 正优势
advantage[50] = progress[100] - progress[50] = 0.6 - 0.3 = 0.3  # 正优势
advantage[100] = progress[150] - progress[100] = 0.9 - 0.6 = 0.3  # 正优势
advantage[150] = progress[200] - progress[150] = 1.0 - 0.9 = 0.1  # 正优势（较小）

# 步骤 3：离散化（假设阈值 = 0.2）
task_index[0] = 1  # 0.3 >= 0.2
task_index[50] = 1  # 0.3 >= 0.2
task_index[100] = 1  # 0.3 >= 0.2
task_index[150] = 0  # 0.1 < 0.2

# 步骤 4：映射为 prompt
prompt[0] = "fold the cloth, Advantage: positive"
prompt[50] = "fold the cloth, Advantage: positive"
prompt[100] = "fold the cloth, Advantage: positive"
prompt[150] = "fold the cloth, Advantage: negative"

# 步骤 5：用于策略训练
# 策略学习：在 positive prompt 下，学习高优势行为
# 策略学习：在 negative prompt 下，学习低优势行为
```

### 5.2 推理阶段（使用预测的 Progress）

```python
# 步骤 1：优势估计器预测 Progress
observation_0 = get_observation(frame_0)
progress_0 = advantage_estimator.sample_values(observation_0)  # 预测：0.2

observation_50 = get_observation(frame_50)
progress_50 = advantage_estimator.sample_values(observation_50)  # 预测：0.5

# 步骤 2：计算 advantage
advantage_0 = progress_50 - progress_0 = 0.5 - 0.2 = 0.3

# 步骤 3：离散化（使用训练时的阈值）
task_index_0 = 1 if advantage_0 >= threshold else 0  # 假设 threshold = 0.2

# 步骤 4：映射为 prompt
prompt_0 = "fold the cloth, Advantage: positive"

# 步骤 5：策略推理（使用 prompt）
action = policy.sample(observation_0, prompt=prompt_0)
```

---

## 六、关键洞察总结

### 6.1 Progress 值的用途

1. **计算 Advantage**：
   - 公式：$A_t = \text{Progress}(o_{t+N}) - \text{Progress}(o_t)$
   - 含义：衡量"未来进展"相对于"当前进展"的差异

2. **离散化为 task_index**：
   - 将连续的 Advantage 转换为离散标签（0 或 1）
   - 用于区分"高优势"和"低优势"样本

3. **映射为 Prompt**：
   - 将 task_index 映射为语言 prompt
   - 例如："fold the cloth, Advantage: positive"

4. **策略 Conditioning**：
   - 通过 prompt conditioning 实现优势加权
   - 引导策略学习高优势行为

### 6.2 与 π*₀.₆ 的本质区别

| 维度 | Kai0 | π*₀.₆ |
|------|------|--------|
| **Value 含义** | Progress（任务进度） | V(s)（状态价值） |
| **Value 来源** | 预定义标签 | 环境奖励 |
| **优势计算** | Progress 差异 | Return - V(s) |
| **优势用途** | 离散化 → Prompt → Conditioning | 二值化 → Token → Conditioning |
| **信号密度** | 密集（每个时间步） | 稀疏（依赖奖励） |

### 6.3 为什么这样设计？

**Kai0 的设计理念**：
- ✅ **密集信号**：Progress 提供每个时间步的反馈
- ✅ **阶段感知**：可以分解为多个阶段，每个阶段独立计算
- ✅ **稳定训练**：Progress 差异比稀疏 reward 更稳定
- ✅ **语言引导**：通过 prompt conditioning 利用预训练语言模型的知识

**π*₀.₆ 的设计理念**：
- ✅ **奖励驱动**：直接使用环境奖励信号
- ✅ **价值估计**：预测累积奖励期望
- ✅ **直接 conditioning**：Advantage 直接作为 token 输入

---

## 七、总结

### Progress 值的完整用途链

```
Progress 值（预测或 GT）
    ↓
计算 Advantage（Progress 差异）
    ↓
离散化为 task_index（0 或 1）
    ↓
映射为 Prompt（语言字符串）
    ↓
策略 Conditioning（优势加权）
    ↓
改进策略行为（学习高优势动作）
```

### 关键点

1. **Progress 不是未来期望收益**：
   - Progress = 任务完成进度（0-1）
   - V(s) = 累积奖励期望

2. **Progress 用于计算 Advantage**：
   - Advantage = Progress 差异（未来 - 当前）
   - 不是 Return - V(s)

3. **Advantage 用于策略改进**：
   - 离散化 → Prompt → Conditioning
   - 通过语言引导实现优势加权

4. **设计优势**：
   - 密集信号、稳定训练、阶段感知
   - 适合长时域 manipulation 任务
