#! /bin/bash
# 获取脚本所在的上一级目录
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 设置默认参数
ROSBAG_PARAMS=""
TOPICS=""
TASK_NAME="default"
BAG_PREFIX=""
OUTPUT_DIR="$PARENT_DIR/raw_data"

# 预定义要记录的话题
PREDEFINED_TOPICS="/tf /arm_trajectory_visualization /robot_tag_info_odom /robot_tag_info /joint_cmd /cmd_vel /hand_wrench_cmd /humanoid/mpc/targetState /humanoid_mpc_gait_time_name /humanoid_mpc_mode_schedule /humanoid_wbc_observation /camera/rgb/image_raw /camera/color/image_raw /right_cam/color/image_raw /left_cam/color/image_raw /sensors_data_raw /mm_kuavo_arm_traj /cmd_pose /state_estimate/imu_data_filtered/angularVel /state_estimate/imu_data_filtered/linearAccel /kuavo_arm_traj /leju_claw_command /leju_claw_state /mm/two_arm_hand_pose_cmd /two_arm_hand_pose_cmd /kuavo_arm_target_poses"
# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -a|--all-topics)
        ROSBAG_PARAMS+="-a"
        shift
        ;;
        -t|--topics)
        TOPICS=$PREDEFINED_TOPICS
        shift
        ;;
        -n|--name)
        TASK_NAME="$2"
        shift
        shift
        ;;
        -b|--bag-prefix)
        BAG_PREFIX="$2"
        shift
        shift
        ;;
        *)
        echo "Invalid option: $1"   
        exit 1
        ;;
    esac
done

# 检查参数
if [ -n "$TOPICS" ] && [[ $ROSBAG_PARAMS == *"-a"* ]]; then
    echo "Error: Cannot use both -a and -t options together"
    exit 1
fi

if [ -z "$TOPICS" ] && [[ $ROSBAG_PARAMS != *"-a"* ]]; then
    echo "Error: Please specify either -a for all topics or -t for specific topics"
    echo "Example: ./record_sim_episodes.sh -t -n task1"
    echo "Example: ./record_sim_episodes.sh -a -n task1"
    exit 1
fi

# 设置任务特定的输出目录
TASK_DIR="$OUTPUT_DIR/$TASK_NAME"
mkdir -p "$TASK_DIR"

# episode_num 根据文件夹中所有 bag 文件的数量来统计
if [ -d "$TASK_DIR" ]; then
    # 统计文件夹中所有 .bag 文件的数量
    BAG_COUNT=$(find "$TASK_DIR" -maxdepth 1 -name "*.bag" -type f | wc -l)
    EPISODE_NUM=$BAG_COUNT
else
    EPISODE_NUM=0
fi

# 构建bag文件名
if [ -n "$BAG_PREFIX" ]; then
    # BAG_PREFIX 由上层脚本传入，如：pick_width_40
    # 按照期望格式组合：episode_{EPISODE_NUM}_{BAG_PREFIX}.bag
    BAG_NAME="episode_${EPISODE_NUM}_${BAG_PREFIX}"
else
    # 若未指定 BAG_PREFIX，退回到原来的 episode_{EPISODE_NUM}.bag 格式
    BAG_NAME="episode_${EPISODE_NUM}"
fi

echo "============================================="
echo "Recording episode $EPISODE_NUM"
echo "Output: $TASK_DIR/${BAG_NAME}.bag"
echo "============================================="
echo ""

# 启动数据记录节点
rosbag record $ROSBAG_PARAMS -O "$TASK_DIR/${BAG_NAME}" $TOPICS

