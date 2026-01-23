# Absolute EEF 训练快速检查清单

## ✅ 已完成的配置

1. **训练脚本已更新**：`train_groot_multi_gpu.sh` 中已添加：
   ```bash
   --policy.action_space_type="Absolute eef"
   ```

2. **代码已适配**：
   - ✅ 自动配置维度（9+9+2=20D）
   - ✅ 自动启用分部分归一化
   - ✅ 6D rotation 使用 IDENTITY（不归一化）

## 📋 运行前检查清单

### 1. 数据集维度检查

确保数据集是 **20 维**（absolute eef pose 格式）：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id="your_dataset", root="/path/to/dataset")
print(f"Action shape: {dataset.meta.features['action']['shape']}")
print(f"State shape: {dataset.meta.features['observation.state']['shape']}")
```

**应该输出**：
```
Action shape: (20,)
State shape: (20,)
```

### 2. 数据集格式验证

数据集中的 action/state 应该是：
- 维度 0-2:   左手 eef position (x, y, z)
- 维度 3-8:   左手 eef 6D 旋转表示 [R11, R21, R31, R12, R22, R32]
- 维度 9-11:  右手 eef position (x, y, z)
- 维度 12-17: 右手 eef 6D 旋转表示 [R11, R21, R31, R12, R22, R32]
- 维度 18:    左夹爪开合程度
- 维度 19:    右夹爪开合程度

### 3. 训练脚本配置

确认 `train_groot_multi_gpu.sh` 中：
- ✅ `--policy.action_space_type="Absolute eef"` 已添加
- ✅ `--policy.max_action_dim=32` (可以大于 20，用于 padding)
- ✅ `--policy.max_state_dim=64` (可以大于 20，用于 padding)

### 4. 数据集路径

确认数据集路径正确：
```bash
DATASET_ROOT="/home/kangkk/humanoid_groot/lerobot_data/split_dataset/wider"
DATASET_REPO_ID="dense,mix,four,random,4622_rc,4611_mix_fail"
```

## 🚀 直接运行

如果以上检查都通过，可以直接运行：

```bash
./train_groot_multi_gpu.sh --gpu 4,5,6,7
```

## 📊 训练日志验证

训练开始时会看到：

```
🎯 Auto-configured for Absolute eef action space:
   left_arm=9D, right_arm=9D, claw=2D
✅ Split arm heads enabled: left_arm(9D) + right_arm(9D) + claw(2D) = 20D
✅ Partial normalization enabled for action space: Absolute eef
   Components: ['left_eef_pos', 'left_eef_rot6d', 'right_eef_pos', 'right_eef_rot6d', 'left_gripper', 'right_gripper']
   6D rotation components (left_eef_rot6d, right_eef_rot6d) will use IDENTITY normalization
```

如果看到这些输出，说明配置正确！

## ⚠️ 如果数据集不是 20D

如果数据集不是 20D 格式，需要先转换：

1. 使用 `cvt_bag2lerobot_depalletizer_task_eef_pinocchio.py` 脚本
2. 确保使用 `--urdf-path` 指定 URDF 文件
3. 不使用 `--no-fk` 标志（启用 FK 计算）
4. 数据集会生成 20 维的 absolute eef pose 格式

## 🔄 切换回 Absolute Joint

如果想切换回关节空间训练，只需：

1. 修改训练脚本：
   ```bash
   --policy.action_space_type="Absolute joint"  # 或直接删除这一行（默认就是 Absolute joint）
   ```

2. 确保数据集是 16D 格式（7+7+2）
