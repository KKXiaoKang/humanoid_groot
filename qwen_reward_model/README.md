# Qwen Reward Model - 任务完成度评估

这个工具使用 Qwen2.5-VL-7B-Instruct 模型来评估 lerobot v3.0 数据集中每个 episode 的任务完成度。

## 功能

- 加载 lerobot v3.0 数据集
- 使用 Qwen2.5-VL-7B-Instruct 模型对视觉序列进行评估
- 输出 0-1 的任务完成度分数
- 支持批量处理多个 episodes
- 自动从数据集元数据中读取任务描述

## 安装依赖

```bash
pip install transformers torch pillow numpy tqdm
```

如果需要 GPU 支持，请确保安装了正确版本的 PyTorch 和 CUDA。

## 使用方法

### 基本用法

```bash
python process_lerobot_data_reward.py \
    --dataset_path /home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1125_groot_train_data_with_task_filtered \
    --camera_key observation.images.cam_head
```

### 完整参数

```bash
python process_lerobot_data_reward.py \
    --dataset_path /home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1125_groot_train_data_with_task_filtered \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --camera_key observation.images.cam_head \
    --task_description "把灰色箱子拉开，并且机械臂双边夹爪把箱子抬起来" \
    --max_episodes 10 \
    --frames_per_episode 20 \
    --output_path ./reward_scores.json \
    --device auto

python process_lerobot_data_reward.py \
    --dataset_path /home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1125_groot_train_data_with_task_filtered \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --camera_key observation.images.cam_head \
    --task_description "把灰色箱子拉开，并且机械臂双边夹爪把箱子抬起来" \
    --max_episodes 2 \
    --frames_per_episode 20 \
    --output_path ./reward_scores.json \
    --device auto
```

### 参数说明

- `--dataset_path`: lerobot v3.0 数据集路径（必需）
- `--model_name`: 模型名称（默认: Qwen/Qwen2.5-VL-7B-Instruct）
- `--camera_key`: 相机数据键名（默认: observation.images.cam_head）
- `--task_description`: 任务描述（如果为 None，从数据集元数据中读取）
- `--max_episodes`: 最大处理 episode 数量（None 表示处理所有）
- `--frames_per_episode`: 每个 episode 使用的帧数（None 表示使用所有帧）
- `--output_path`: 输出结果保存路径（默认保存到数据集目录）
- `--device`: 设备（auto, cuda, cpu）

## 输出格式

脚本会生成一个 JSON 文件，包含以下信息：

```json
{
  "dataset_path": "...",
  "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
  "camera_key": "observation.images.cam_head",
  "task_description": "...",
  "num_episodes": 10,
  "results": [
    {
      "episode": 0,
      "task": "...",
      "score": 0.85,
      "response": "模型生成的完整响应...",
      "num_frames": 20
    },
    ...
  ],
  "statistics": {
    "mean_score": 0.75,
    "std_score": 0.15,
    "min_score": 0.3,
    "max_score": 1.0
  }
}
```

## 提示模板

默认的提示模板如下：

```
你是一个机器人任务评估器。
目标任务：{task_description}

下面是机器人执行该任务未来的视觉序列。

请根据视觉变化判断机器人是否越来越接近目标。
输出一个从 0 到 1 的分数，其中：
- 1 表示完全成功
- 0 表示完全失败
- 中间分数表示部分进展

请直接输出分数（0-1之间的数字）。
```

## 注意事项

1. **GPU 显存要求**: Qwen2.5-VL-7B-Instruct 模型需要较大的 GPU 显存（建议至少 24GB）
2. **处理时间**: 每个 episode 的处理时间取决于帧数和模型推理速度
3. **分数提取**: 脚本会自动从模型响应中提取分数，如果提取失败会使用默认值 0.5
4. **错误处理**: 如果某个 episode 处理失败，会在结果中记录错误信息，不会中断整个处理过程

## 示例

处理前 5 个 episodes，每个 episode 使用 10 帧：

```bash
python process_lerobot_data_reward.py \
    --dataset_path /home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1125_groot_train_data_with_task_filtered \
    --max_episodes 5 \
    --frames_per_episode 10
```

