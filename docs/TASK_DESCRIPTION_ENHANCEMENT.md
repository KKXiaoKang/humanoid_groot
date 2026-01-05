# 任务描述增强方案

## 问题分析

### 当前问题

当两个场景在视觉上相似但实际不同时，如果使用相同的任务描述，会导致模型混淆：

**场景1（4个箱子）**：
- 任务描述：`"Depalletize a narrower green box of type 4322 (40×30×22 cm) on the left"`
- 视觉特征：传统左拆，4个箱子排列

**场景2（6个箱子，3x2排列）**：
- 任务描述：`"Depalletize a narrower green box of type 4322 (40×30×22 cm) on the left"`
- 视觉特征：箱子在中间时的左拆，6个箱子（3x2排列）

**问题**：
- ✅ 两个场景都是"左拆"，视觉上可能相似
- ✅ 但场景不同（4个箱子 vs 6个箱子）
- ❌ 如果任务描述相同，模型无法区分
- ❌ 模型可能学习到错误的视觉-语言映射

### 为什么会造成困扰？

1. **视觉特征重叠**：
   - 两个场景都是"左拆"，视觉特征可能重叠
   - 模型可能无法仅通过视觉特征区分

2. **语言信息不足**：
   - 任务描述没有明确区分场景
   - 语言模型无法帮助模型区分不同场景

3. **学习目标冲突**：
   - 相同的任务描述 + 不同的视觉场景 = 学习目标冲突
   - 模型可能学习到模糊的映射关系

## 解决方案

### 方案1: 添加场景描述（推荐）

**原理**：在任务描述中明确区分场景（4个箱子 vs 6个箱子）

**实现**：

```python
# 4个箱子的场景
task_description = "Depalletize a narrower green box of type 4322 (40×30×22 cm) on the left from a stack of 4 boxes"

# 6个箱子的场景（3x2排列）
task_description = "Depalletize a narrower green box of type 4322 (40×30×22 cm) on the left from a 3x2 grid of 6 boxes"
```

**优点**：
- ✅ 明确区分场景
- ✅ 利用语言模型的语义理解能力
- ✅ 实现简单，只需修改任务描述生成逻辑

**缺点**：
- ⚠️ 任务描述变长，但通常不是问题

### 方案2: 添加位置描述（更精确）

**原理**：在任务描述中明确描述位置（传统左拆 vs 中间左拆）

**实现**：

```python
# 4个箱子的场景（传统左拆）
task_description = "Depalletize a narrower green box of type 4322 (40×30×22 cm) on the left from a stack of 4 boxes"

# 6个箱子的场景（3x2排列，左拆）
task_description = "Depalletize a narrower green box of type 4322 (40×30×22 cm) from the left side of a 3x2 grid of 6 boxes"
```

**优点**：
- ✅ 更精确的位置描述
- ✅ 明确区分"传统左拆"和"中间左拆"
- ✅ 利用语言模型的语义理解能力

**缺点**：
- ⚠️ 任务描述变长
- ⚠️ 需要更仔细地设计描述

### 方案3: 添加场景标识符（最明确）

**原理**：在任务描述中添加场景标识符

**实现**：

```python
# 4个箱子的场景
task_description = "Depalletize a narrower green box of type 4322 (40×30×22 cm) on the left [4-box stack]"

# 6个箱子的场景（3x2排列）
task_description = "Depalletize a narrower green box of type 4322 (40×30×22 cm) on the left [3x2 grid, 6 boxes]"
```

**优点**：
- ✅ 最明确的场景区分
- ✅ 易于解析和理解
- ✅ 可以用于数据过滤和分析

**缺点**：
- ⚠️ 标识符可能不够自然
- ⚠️ 语言模型可能无法很好地理解标识符

## 推荐方案

### 最佳实践：组合使用方案1和方案2

**4个箱子的场景**：
```python
task_description = "Depalletize a narrower green box of type 4322 (40×30×22 cm) on the left from a stack of 4 boxes"
```

**6个箱子的场景（3x2排列）**：
```python
task_description = "Depalletize a narrower green box of type 4322 (40×30×22 cm) from the left side of a 3x2 grid of 6 boxes"
```

**关键区别**：
1. **场景描述**：`"from a stack of 4 boxes"` vs `"from a 3x2 grid of 6 boxes"`
2. **位置描述**：`"on the left"` vs `"from the left side of"`（更精确）

## 代码实现

### 修改任务描述生成逻辑

在 `cvt_bag2lerobot_depalletizer_task_dense.py` 中修改 `make_task` 函数：

```python
def make_task(task_name, box_key, side: str | None, scene_type: str | None = None):
    """
    根据任务类型、盒子类型、位置和场景类型生成任务描述
    
    Args:
        task_name: 任务类型 ("pick" 或 "unpack")
        box_key: 盒子类型键（如 "4322_green"）
        side: 位置 ("left" 或 "right")，pick任务时可以为None
        scene_type: 场景类型 ("4boxes" 或 "6boxes_3x2")，用于区分不同场景
        
    Returns:
        任务描述字符串
    """
    task_text = f"{TASK_INFO[task_name]} {BOX_INFO[box_key]}"
    
    # 如果是 "pick" 任务，不添加 side words
    if task_name != "pick":
        if side is None:
            raise ValueError(f"unpack task requires side, but got None")
        
        # 根据场景类型添加不同的位置描述
        if scene_type == "6boxes_3x2":
            # 6个箱子的场景（3x2排列）
            task_text += f" from the {side} side of a 3x2 grid of 6 boxes"
        elif scene_type == "4boxes":
            # 4个箱子的场景
            task_text += f" on the {side} from a stack of 4 boxes"
        else:
            # 默认情况（向后兼容）
            task_text += f" {SIDE_WORDS[side]}"
    
    return task_text
```

### 从文件名或路径解析场景类型

```python
def parse_scene_type(bag_path: Path) -> str | None:
    """
    从bag文件路径解析场景类型
    
    Args:
        bag_path: bag文件的路径
        
    Returns:
        场景类型字符串（"4boxes" 或 "6boxes_3x2"），如果无法确定则返回None
    """
    # 方法1: 从目录名解析
    parent_dir = bag_path.parent.name
    
    if "3x2" in parent_dir or "3X2" in parent_dir:
        return "6boxes_3x2"
    elif "four_stacks" in parent_dir or "4boxes" in parent_dir:
        return "4boxes"
    
    # 方法2: 从文件名解析
    filename = bag_path.stem
    if "3x2" in filename or "3X2" in filename:
        return "6boxes_3x2"
    elif "four_stacks" in filename or "4boxes" in filename:
        return "4boxes"
    
    return None
```

### 在数据转换时使用场景类型

```python
def populate_dataset(
    self,
    dataset: LeRobotDataset,
    bag_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
    auto_task_from_path: bool = True,
) -> LeRobotDataset:
    # ... 现有代码 ...
    
    for ep_idx in tqdm.tqdm(episodes):
        ep_path = bag_files[ep_idx]
        
        # 解析场景类型
        scene_type = parse_scene_type(ep_path)
        
        # 根据路径自动确定task（如果启用）
        if auto_task_from_path:
            # 解析任务信息（task_name, box_key, side）
            parsed_task_info = parse_task_from_filename(ep_path, verbose=False)
            
            if parsed_task_info:
                # 使用场景类型生成任务描述
                episode_task = make_task(
                    task_name=parsed_task_info["task_name"],
                    box_key=parsed_task_info["box_key"],
                    side=parsed_task_info["side"],
                    scene_type=scene_type  # 添加场景类型
                )
            else:
                episode_task = task
        else:
            episode_task = task
        
        # ... 其余代码 ...
```

## 任务描述示例

### 4个箱子的场景

```python
# 左拆
"Depalletize a narrower green box of type 4322 (40×30×22 cm) on the left from a stack of 4 boxes"

# 右拆
"Depalletize a narrower green box of type 4322 (40×30×22 cm) on the right from a stack of 4 boxes"

# 抓取
"Pick up a narrower green box of type 4322 (40×30×22 cm) from a stack of 4 boxes"
```

### 6个箱子的场景（3x2排列）

```python
# 左拆
"Depalletize a narrower green box of type 4322 (40×30×22 cm) from the left side of a 3x2 grid of 6 boxes"

# 右拆
"Depalletize a narrower green box of type 4322 (40×30×22 cm) from the right side of a 3x2 grid of 6 boxes"

# 抓取
"Pick up a narrower green box of type 4322 (40×30×22 cm) from a 3x2 grid of 6 boxes"
```

## 效果预期

### 改进前（可能混淆）

```
场景1（4个箱子）: "Depalletize ... on the left"
场景2（6个箱子）: "Depalletize ... on the left"
→ 模型可能混淆，无法区分场景
```

### 改进后（明确区分）

```
场景1（4个箱子）: "Depalletize ... on the left from a stack of 4 boxes"
场景2（6个箱子）: "Depalletize ... from the left side of a 3x2 grid of 6 boxes"
→ 模型可以明确区分场景，利用语言信息帮助理解
```

## 总结

**问题**：相同的任务描述 + 不同的视觉场景 = 模型混淆

**解决方案**：
1. ✅ **在任务描述中明确区分场景**（4个箱子 vs 6个箱子）
2. ✅ **使用更精确的位置描述**（"on the left" vs "from the left side of"）
3. ✅ **利用语言模型的语义理解能力**帮助模型区分场景

**推荐实现**：
- 4个箱子：`"Depalletize ... on the left from a stack of 4 boxes"`
- 6个箱子：`"Depalletize ... from the left side of a 3x2 grid of 6 boxes"`

这样可以：
- ✅ 明确区分场景
- ✅ 利用语言信息帮助模型理解
- ✅ 减少视觉特征重叠带来的混淆
- ✅ 提高模型在混合数据集上的表现

