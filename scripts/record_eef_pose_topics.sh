#!/bin/bash

# ROS bag录制脚本 - 录制所有EEF pose相关话题
# 使用方法: ./record_eef_pose_topics.sh [bag_name]
# 如果不提供bag_name，将使用默认名称（带时间戳）

# 获取当前时间戳（用于默认bag文件名）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DEFAULT_BAG_NAME="eef_pose_recording_${TIMESTAMP}"

# 如果提供了bag名称，使用它；否则使用默认名称
BAG_NAME=${1:-${DEFAULT_BAG_NAME}}

echo "=========================================="
echo "🎬 Starting ROS bag recording..."
echo "📦 Bag name: ${BAG_NAME}.bag"
echo "📡 Recording topics:"
echo "=========================================="
echo ""
echo "EEF Pose Topics (PoseStamped):"
echo "  - /policy/eef_pose/real_state_left"
echo "  - /policy/eef_pose/real_state_right"
echo "  - /policy/eef_pose/predicted_absolute_left"
echo "  - /policy/eef_pose/predicted_absolute_right"
echo "  - /policy/eef_pose/reference_left (Delta eef only)"
echo "  - /policy/eef_pose/reference_right (Delta eef only)"
echo "  - /policy/eef_pose/ik_fk_roundtrip_left"
echo "  - /policy/eef_pose/ik_fk_roundtrip_right"
echo ""
echo "EEF Euler Angle Topics (Vector3Stamped: x=roll, y=pitch, z=yaw):"
echo "  - /policy/eef_euler/real_state_left"
echo "  - /policy/eef_euler/real_state_right"
echo "  - /policy/eef_euler/predicted_absolute_left"
echo "  - /policy/eef_euler/predicted_absolute_right"
echo "  - /policy/eef_euler/reference_left (Delta eef only)"
echo "  - /policy/eef_euler/reference_right (Delta eef only)"
echo "  - /policy/eef_euler/ik_fk_roundtrip_left"
echo "  - /policy/eef_euler/ik_fk_roundtrip_right"
echo ""
echo "Robot State Topics:"
echo "  - /kuavo_arm_traj"
echo "  - /sensors_data_raw"
echo "  - /joint_cmd"
echo "  - /leju_claw_command"
echo "  - /leju_claw_state"
echo ""
echo "Optional Visualization Topics:"
echo "  - /policy/action/eef_pose_marker_all"
echo "  - /policy/eef_pose_markers"
echo ""
echo "=========================================="
echo "💡 Press Ctrl+C to stop recording"
echo "=========================================="
echo ""

# 执行rosbag record命令
rosbag record \
  /policy/eef_pose/real_state_left \
  /policy/eef_pose/real_state_right \
  /policy/eef_pose/predicted_absolute_left \
  /policy/eef_pose/predicted_absolute_right \
  /policy/eef_pose/reference_left \
  /policy/eef_pose/reference_right \
  /policy/eef_pose/ik_fk_roundtrip_left \
  /policy/eef_pose/ik_fk_roundtrip_right \
  /policy/eef_euler/real_state_left \
  /policy/eef_euler/real_state_right \
  /policy/eef_euler/predicted_absolute_left \
  /policy/eef_euler/predicted_absolute_right \
  /policy/eef_euler/reference_left \
  /policy/eef_euler/reference_right \
  /policy/eef_euler/ik_fk_roundtrip_left \
  /policy/eef_euler/ik_fk_roundtrip_right \
  /kuavo_arm_traj \
  /sensors_data_raw \
  /joint_cmd \
  /policy/action/eef_pose_marker_all \
  /policy/eef_pose_markers \
  /leju_claw_command \
  /leju_claw_state \
  -O "${BAG_NAME}"

echo ""
echo "=========================================="
echo "✅ Recording completed!"
echo "📦 Bag file saved as: ${BAG_NAME}.bag"
echo "=========================================="
