# GROOT 异步推理使用指南

本文档介绍如何使用异步推理架构来运行 GROOT policy，从而提升推理性能。

## 架构概述

异步推理架构将推理任务分离到独立的服务器端，客户端专注于观测获取和动作执行：

- **服务器端 (policy_server.py)**: 
  - 运行 GROOT 模型推理
  - 接收客户端观测
  - 返回动作块
  
- **客户端 (eval_depalletize_async.py)**:
  - 从机器人获取观测
  - 发送观测到服务器
  - 接收动作块
  - 执行动作控制

## 优势

相比同步推理（`eval_depalletize_camera_model_reload.py`），异步推理的优势：

1. **更高的吞吐量**: 推理和动作执行可以并行进行
2. **更低的延迟**: 动作队列机制确保机器人始终有动作可执行
3. **更好的资源利用**: 可以将服务器部署在更强的 GPU 机器上
4. **更容易扩展**: 可以支持多个客户端连接到同一个服务器

## 快速开始

### 1. 启动服务器

在一个终端中启动策略服务器：

```bash
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033
```

参数说明：
- `--host`: 服务器绑定地址（默认: 127.0.0.1）
- `--port`: 服务器端口（默认: 8080）
- `--fps`: 目标帧率（默认: 30）
- `--inference_latency`: 推理延迟目标（秒，默认: 0.033）

### 2. 运行客户端

在另一个终端中运行客户端：

```bash
python scripts/eval_depalletize_async.py \
    --server_address=127.0.0.1:8080 \
    --ckpt_path=/path/to/your/checkpoint \
    --action_chunk_size=20 \
    --task_description="Depalletize the box" \
    --fps=30 \
    --chunk_size_threshold=0.5
```

参数说明：
- `--server_address`: 服务器地址（格式: `host:port`）
- `--ckpt_path`: GROOT 模型 checkpoint 路径（**必需**）
- `--action_chunk_size`: 动作块大小（默认: 20）
- `--lerobot_dataset_path`: 数据集路径（用于加载统计信息，可选）
- `--task_description`: 任务描述字符串（默认: "Depalletize the box"）
- `--fps`: 控制频率（FPS，默认: 30.0）
- `--chunk_size_threshold`: 发送新观测的阈值（当队列大小/动作块大小 < 阈值时发送，默认: 0.5）

## 工作原理

### 客户端工作流程

1. **初始化**:
   - 连接 gRPC 服务器
   - 发送策略配置（模型路径、动作块大小等）
   - 初始化 `GrabBoxMpcEnv`

2. **启动接收线程**:
   - 独立线程从服务器接收动作块
   - 将动作添加到队列中

3. **主控制循环**:
   - 获取机器人观测
   - 检查动作队列，如果队列大小低于阈值，发送新观测
   - 从队列获取动作并执行
   - 对动作进行重采样和低通滤波（与同步版本相同）

### 动作队列机制

客户端维护一个动作队列：

- **队列填充**: 接收线程持续从服务器获取动作并添加到队列
- **队列消耗**: 主循环从队列取出动作执行
- **观测发送**: 当队列大小 < `chunk_size_threshold * action_chunk_size` 时，发送新观测

这种机制确保：
- 机器人始终有动作可执行（减少延迟）
- 避免过度推理（节省计算资源）
- 处理网络延迟和推理时间波动

## 与同步推理的对比

| 特性 | 同步推理 | 异步推理 |
|------|---------|---------|
| 架构 | 单线程，阻塞式 | 多线程，非阻塞 |
| 推理时机 | 动作队列为空时立即推理 | 根据队列阈值异步推理 |
| 延迟 | 推理时间直接影响控制延迟 | 队列缓冲减少延迟影响 |
| 吞吐量 | 受推理时间限制 | 推理与执行并行 |
| 适用场景 | 简单部署、调试 | 生产环境、高性能需求 |

## 性能调优

### 性能分析

根据你的性能数据：
- **推理时间**: ~50ms/chunk
- **相机频率**: 30Hz (33.3ms/帧)
- **动作块大小**: 16
- **控制频率**: 30Hz

**计算**:
- 每个动作时间跨度 = 33.3ms
- 一个动作块（16个动作）时间跨度 = 16 × 33.3ms = 533ms
- **推理速度比** = 533ms / 50ms ≈ **10.7倍实时速度** ✅

这意味着推理速度已经非常快，能够轻松满足实时需求！

### 优化策略

#### 1. **最大化推理利用率**（推荐）

由于推理时间（50ms）已经超过 `inference_latency`（33ms），服务器不会睡眠，会立即处理下一个观测。

**优化建议**：
- ✅ **保持或降低 `--inference_latency`** (如 0.050 或 0.060)：由于实际推理是50ms，设置为接近或略高于实际推理时间可以避免不必要的睡眠
- ✅ **增加 `--action_chunk_size`** (从16增加到20-24)：利用推理速度优势，生成更多动作，减少推理频率
- ✅ **降低 `--chunk_size_threshold`** (从0.5降低到0.3-0.4)：更早触发新推理，确保动作队列始终充足

**优化后的配置**：
```bash
# 服务器端
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.050  # 接近实际推理时间（50ms）

# 客户端
python scripts/eval_depalletize_async.py \
    --server_address=127.0.0.1:8080 \
    --ckpt_path=... \
    --action_chunk_size=20 \          # 增加块大小
    --fps=30 \
    --chunk_size_threshold=0.3 \      # 降低阈值，更快触发
    --lerobot_dataset_path=... \
    --task_description="..."
```

#### 2. **进一步提升响应性**

如果你想要更快的响应速度：

- **增加控制频率**（如果机器人支持）：
  - 将 `--fps` 从 30 提升到 50 或更高
  - **注意**：需要确保服务器端 `--fps` 也相应调整

- **进一步降低阈值**：
  - `--chunk_size_threshold=0.2`：几乎持续发送观测，最大化响应速度

#### 3. **进一步优化推理速度**（如果需要）

如果你想进一步提升推理速度本身：

- **使用 TensorRT/ONNX Runtime**：如果模型支持，可以显著加速推理
- **混合精度推理**：使用 FP16 可能带来 1.5-2x 加速
- **批处理**：如果有多个观测，可以批量处理（需要修改代码）
- **模型量化**：INT8 量化可能带来 2-4x 加速（可能影响精度）

### 参数说明

#### 服务器端参数

- `--inference_latency`: 
  - 推理延迟目标（秒）
  - 如果推理时间 < 此值，服务器会睡眠剩余时间
  - 如果推理时间 ≥ 此值，服务器不睡眠，立即处理下一个观测
  - **建议**：设置为略高于实际推理时间（如实际50ms，设置为0.050-0.060）

#### 客户端参数

- `--chunk_size_threshold`: 
  - 较小的值（如 0.2-0.3）: 更频繁发送观测，响应更快，但可能增加计算负担
  - 中等值（如 0.4-0.5）: 平衡响应速度和计算效率（**推荐**）
  - 较大的值（如 0.7-0.8）: 减少观测发送，节省计算，但可能增加延迟
  
- `--fps`: 控制频率。应与服务器端 `--fps` 一致。

- `--action_chunk_size`: 动作块大小。
  - **较小的值（12-16）**: 响应更快，但需要更频繁的推理
  - **较大的值（20-24）**: 提供更多缓冲，减少推理频率，提高效率（**推荐用于快速推理场景**）

## 性能监控

### 关键指标

监控以下指标来评估系统性能：

1. **服务器端日志**：
   - `Preprocessing and inference took X.XXs` - 推理时间（目标：<50ms）
   - `Observation X | Total time: XX.XXms` - 端到端处理时间
   - 如果频繁看到总时间超过目标时间，可能需要优化

2. **客户端日志**：
   - `[CLIENT] 📊 Status Summary` - 每2秒的状态汇总
   - `Server queue: X/Y` - 服务器动作队列状态
   - `Time since last obs: X.XXs` - 观测发送频率
   - `Time since last chunk: X.XXs` - 动作接收频率

3. **理想状态**：
   - 服务器队列大小接近 `action_chunk_size`
   - 动作接收频率稳定
   - 很少看到 "No action available" 警告

### 性能调优检查清单

- [ ] 推理时间稳定在预期范围内（如 ~50ms）
- [ ] 动作队列保持充足（>50% capacity）
- [ ] 很少或没有 "No action available" 警告
- [ ] 服务器 GPU 利用率合理（不会一直100%）
- [ ] 客户端和服务器的网络延迟 < 10ms

## 故障排查

### 连接失败

如果客户端无法连接到服务器：

1. 检查服务器是否正在运行：`ps aux | grep policy_server`
2. 检查端口是否被占用：`netstat -tuln | grep 8080`
3. 检查防火墙设置

### 动作队列为空

如果机器人没有动作执行：

1. 检查服务器日志，确认是否收到观测
2. 检查网络连接是否稳定
3. 检查 `chunk_size_threshold` 是否设置过大

### 性能问题

如果延迟较高：

1. 增加 `--action_chunk_size` 提供更多缓冲
2. 降低 `--chunk_size_threshold` 更频繁发送观测
3. 检查服务器 GPU 利用率
4. 考虑使用更强的 GPU 或优化模型

## 注意事项

1. **数据集路径**: 如果不提供 `--lerobot_dataset_path`，客户端会尝试使用默认路径。确保数据集包含必要的统计信息。

2. **ROS 节点**: 确保 ROS 环境已正确配置，`GrabBoxMpcEnv` 需要 ROS 话题。

3. **模型路径**: checkpoint 路径必须指向有效的 GROOT 模型目录。

4. **网络稳定性**: 异步推理依赖网络连接，确保服务器和客户端之间的网络稳定。

## 示例脚本

完整的运行示例：

```bash
# 终端 1: 启动服务器（在 GPU 机器上）
# 基础配置（适合推理时间 ~50ms）
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.050  # 接近实际推理时间，避免不必要睡眠

# 终端 2: 运行客户端（在机器人控制机器上）
# 优化配置（最大化推理利用率和响应速度）
python scripts/eval_depalletize_async.py \
    --server_address=127.0.0.1:8080 \
    --ckpt_path=/home/lab/lerobot_groot/outputs/train/12_01_groot_full_tune_multi_head_use_learn_weight/checkpoints/016000/pretrained_model \
    --action_chunk_size=20 \          # 增加块大小，减少推理频率
    --lerobot_dataset_path=/home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1128_groot_train_data_with_task_filtered \
    --task_description="Depalletize the green box" \
    --fps=30 \
    --chunk_size_threshold=0.3 \      # 降低阈值，更快触发新推理
    --rotate-head-camera

# 如果需要极致响应速度（牺牲一些计算效率）
# 可以将 --chunk_size_threshold 降低到 0.2
# 可以将 --action_chunk_size 保持为 16 或降低到 12
```

## 下一步

- 尝试调整参数以优化性能
- 监控动作队列大小以找到最佳阈值
- 考虑在生产环境中使用负载均衡来处理多个机器人

