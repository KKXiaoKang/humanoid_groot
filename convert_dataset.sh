#!/bin/bash

# 数据集格式转换脚本：从 v2.1 转换为 v3.0
# 这个脚本会将您的数据集从旧格式转换为新格式（原地转换）

DATASET_ROOT="/root/lerobot/lerobot_data/1118_sim_depalletize"
DATASET_REPO_ID="1118_sim_depalletize"

echo "开始转换数据集格式：v2.1 -> v3.0"
echo "数据集路径: ${DATASET_ROOT}"

python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id=${DATASET_REPO_ID} \
  --root=${DATASET_ROOT} \
  --push-to-hub=false \
  --force-conversion

echo ""
echo "=========================================="
echo "数据集转换完成！"
echo "现在可以运行训练脚本了：bash train_groot.sh"
echo "=========================================="

