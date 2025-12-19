# 多数据集训练使用说明

现在已经支持同时加载多个LeRobotDataset进行训练。

## 使用方法

### 方式1：通过命令行参数指定多个数据集

修改训练脚本，使用逗号分隔的repo_id列表：

```bash
lerobot-train \
  --dataset.repo_id="1212_5w_groot_train_data_with_task,1215_5w_groot_4311_4322_4611_4633" \
  --dataset.root="/home/lab/lerobot_groot/lerobot_data/v3_0_dataset" \
  # ... 其他参数
```

### 方式2：修改训练脚本

在你的训练脚本中（如`train_groot.sh`），修改数据集配置部分：

```bash
# 单数据集（原方式）
# DATASET_REPO_ID="1125_groot_train_data_with_task_filtered"
# DATASET_ROOT="/path/to/dataset"

# 多数据集（新方式）
DATASET_REPO_ID="1212_5w_groot_train_data_with_task,1215_5w_groot_4311_4322_4611_4633"
DATASET_ROOT="/home/lab/lerobot_groot/lerobot_data/v3_0_dataset"

# 然后在训练命令中使用
lerobot-train \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  # ... 其他参数
```

## 重要说明

1. **数据集路径结构**：
   - `root`参数应该指向包含所有数据集目录的父目录
   - 每个数据集的repo_id应该是`root`目录下的子目录名
   - 例如：如果`root="/home/lab/lerobot_groot/lerobot_data/v3_0_dataset"`，那么数据集应该在：
     - `/home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1212_5w_groot_train_data_with_task/`
     - `/home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1215_5w_groot_4311_4322_4611_4633/`

2. **特征兼容性**：
   - 只有所有数据集共有的特征会被保留
   - 如果某个数据集有其他数据集没有的特征，这些特征会被禁用
   - 训练时会输出警告信息，显示哪些特征被禁用

3. **统计信息聚合**：
   - 所有数据集的统计信息会被自动聚合，用于数据归一化

4. **数据集索引**：
   - 每个样本会包含一个`dataset_index`字段，表示它来自哪个数据集
   - 数据集索引按照repo_id列表的顺序分配（从0开始）

## 示例

对于你的两个数据集：
- `/home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1212_5w_groot_train_data_with_task`
- `/home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1215_5w_groot_4311_4322_4611_4633`

### 示例1：修改训练脚本

在你的`train_groot.sh`或`train_groot_multi_gpu.sh`中，修改数据集配置：

```bash
# 数据集配置
DATASET_ROOT="/home/lab/lerobot_groot/lerobot_data/v3_0_dataset"
DATASET_REPO_ID="1212_5w_groot_train_data_with_task,1215_5w_groot_4311_4322_4611_4633"
```

然后在训练命令中使用：

```bash
lerobot-train \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME} \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=groot \
  # ... 其他训练参数
```

### 示例2：直接使用命令行

```bash
lerobot-train \
  --output_dir="./outputs/multi_dataset_training" \
  --job_name="groot_multi_dataset" \
  --dataset.repo_id="1212_5w_groot_train_data_with_task,1215_5w_groot_4311_4322_4611_4633" \
  --dataset.root="/home/lab/lerobot_groot/lerobot_data/v3_0_dataset" \
  --policy.type=groot \
  --policy.base_model_path="nvidia/GR00T-N1.5-3B" \
  # ... 其他参数
```

