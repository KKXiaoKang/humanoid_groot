# IK和FK一致性验证脚本

这个脚本用于验证IKAnalytical逆运动学和Pinocchio正运动学的一致性。

## 功能

1. 从LeRobot dataset v3.0读取数据
2. 从state和action中提取absolute eef pose（20维格式）
3. 使用IKAnalytical解IK得到关节角
4. 使用Pinocchio FK验证，计算actual eef pose
5. 对比state eef pose、action eef pose和actual eef pose
6. 生成可视化图表：
   - 3D轨迹可视化
   - 位置对比（x, y, z）
   - 欧拉角对比（roll, pitch, yaw）

## 使用方法

```bash
python eval/IK_eef_eval/verify_ik_fk_consistency.py \
    --dataset-path /home/lab/humanoid_groot/lerobot_data/v3_0_dataset/0122_4322_eef_test \
    --episode-idx 0 \
    --urdf-path /home/lab/kuavo-manip/lerobot_datasets/utils/biped_s60_only_arm.urdf \
    --model-type 60 \
    --robot-version 5_wheel \
    --output-dir ./ik_fk_verification_results
```

## 参数说明

- `--dataset-path`: LeRobot dataset v3.0的路径（必需）
- `--episode-idx`: 要分析的episode索引（默认：0）
- `--urdf-path`: URDF文件路径（默认：`/home/lab/kuavo-manip/lerobot_datasets/utils/biped_s60_only_arm.urdf`）
- `--model-type`: 机器人型号，可选：45, 46, 60（默认：45）
- `--robot-version`: 机器人版本，可选：5_wheel, 4_pro（默认：5_wheel）
- `--max-frames`: 最大分析帧数，None表示全部（默认：None）
- `--output-dir`: 输出目录（默认：`./ik_fk_verification_results`）

## 输出

脚本会在输出目录生成以下文件：

1. `3d_trajectory.png`: 3D轨迹可视化，显示state、action和IK->FK的eef pose轨迹
2. `position_comparison.png`: 位置对比图（x, y, z），横轴为frame number，纵轴为位置值
3. `euler_comparison.png`: 欧拉角对比图（roll, pitch, yaw），横轴为frame number，纵轴为角度值（度）

## 数据格式

数据集中的state和action都是20维格式：
- left_eef(9维): [x, y, z, R11, R21, R31, R12, R22, R32]
- right_eef(9维): [x, y, z, R11, R21, R31, R12, R22, R32]
- gripper(2维): [left_gripper, right_gripper]

其中旋转使用6D表示（旋转矩阵的前两列）。

## 注意事项

1. 确保URDF文件路径正确
2. 确保数据集路径正确且包含v3.0格式的数据
3. 如果IK求解失败，会使用零关节角并打印警告
4. 脚本会计算并打印误差统计信息
