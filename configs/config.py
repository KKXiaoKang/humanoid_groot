import cv2
import numpy as np
from kuavo_msgs.msg import sensorsData
from std_msgs.msg import Float64MultiArray
import rospy
from scipy.spatial.transform import Rotation

"""
和topic相关的config放这里 (在bag2lerobot和eval的时候都用到)

Action Mode说明:
- "relative": 相对动作模式，state为60维 (不包含previous_delta_action)
- "delta": Delta动作模式，state为69维 (包含9维previous_delta_action用于闭环反馈)
- "absolute": 绝对动作模式，state为60维 (不包含previous_delta_action)

状态维度说明:
- relative模式: 60维 = 14(arm_joints) + 14(arm_velocities) + 3(lin_acc) + 3(ang_vel) + 6(est_com_vel) + 2(com_z_pitch) + 9(left_hand_pose) + 9(right_hand_pose)
- delta模式: 69维 = 60维基础状态 + 9维previous_delta_action (用于闭环反馈)
- absolute模式: 60维 = 同relative模式
"""
# 默认action_mode，可以通过外部设置覆盖
ACTION_MODE = "absolute"  # "absolute", "delta", "relative"

"""
    TASK_DATA_MODE - 指定任务数据集来源
    vr - 使用vr数据集, 里面相机配置为4相机配置
    strategy - 使用strategy数据集, 里面相机配置为3相机配置
"""
# TASK_DATA_MODE = "strategy" # "VR", "strategy"
TASK_DATA_MODE = "strategy"
"""
    STATE_COMPONENTS - 指定状态空间包含的组件
    可选组件:
    - "J_q": 手臂关节位置 (14维)
    - "IMU": IMU数据 (6维: 3维线加速度 + 3维角速度)
    - "Claw_pos": 夹爪位置状态 (2维)
    - "Com_z_pitch": 质心z位置和pitch角度 (2维，从/humanoid_wbc_observation获取)
    
    示例配置:
    - ["J_q", "IMU", "Claw_pos"]: 22维 (14 + 6 + 2)
    - ["J_q", "Claw_pos", "Com_z_pitch"]: 18维 (14 + 2 + 2)
    - ["J_q"]: 14维 (只有手臂关节)
    - ["Claw_pos"]: 2维 (只有夹爪位置)
    - ["J_q", "IMU"]: 20维 (14 + 6)
    
    注意: 如果ACTION_COMPONENTS包含Cmd_pose_z或Cmd_pose_pitch，则STATE_COMPONENTS必须包含Com_z_pitch
"""
if TASK_DATA_MODE == "strategy":
    STATE_COMPONENTS = ["J_q", "Claw_pos", "Com_z_pitch"]
    # STATE_COMPONENTS = ["J_q", "Claw_pos"]  # 默认16维配置
elif TASK_DATA_MODE == "VR":
    STATE_COMPONENTS = ["J_q", "Claw_pos", "Com_z_pitch"]
    # STATE_COMPONENTS = ["J_q", "Claw_pos"] # VR 使用state16进行学习

"""
    ACTION_COMPONENTS - 指定动作空间包含的组件
    可选组件:
    - "Left_arm": 左臂位置 (7维)
    - "Right_arm": 右臂位置 (7维)
    - "Left_claw": 左爪位置 (1维)
    - "Right_claw": 右爪位置 (1维)
    - "Cmd_pose_z": 命令姿态z位置 (1维)
    - "Cmd_pose_pitch": 命令姿态pitch角度 (1维)
    
    示例配置:
    - ["Left_arm", "Right_arm", "Left_claw", "Right_claw"]: 16维 (7+7+1+1)
    - ["Left_arm", "Right_arm", "Left_claw", "Right_claw", "Cmd_pose_z", "Cmd_pose_pitch"]: 18维 (7+7+1+1+1+1)
    - ["Left_arm", "Right_arm"]: 14维 (只有手臂)
"""
# 默认action组件配置（depalletizer任务通常不需要cmd_pose）
# ACTION_COMPONENTS = ["Left_arm", "Right_arm", "Left_claw", "Right_claw"]
ACTION_COMPONENTS = ["Left_arm", "Right_arm", \
                     # "Left_claw", "Right_claw"]
                     "Left_claw", "Right_claw", "Cmd_pose_z", "Cmd_pose_pitch"]

# 验证：如果action包含cmd_pose，state必须包含com组件
if ("Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS):
    if "Com_z_pitch" not in STATE_COMPONENTS:
        print(f"⚠️  Warning: ACTION_COMPONENTS contains cmd_pose components, but STATE_COMPONENTS does not include Com_z_pitch")
        print(f"   Adding Com_z_pitch to STATE_COMPONENTS automatically...")
        STATE_COMPONENTS = list(STATE_COMPONENTS) + ["Com_z_pitch"]
        print(f"   Updated STATE_COMPONENTS: {STATE_COMPONENTS}")

def euler_to_rotation_matrix_first_two_cols(roll, pitch, yaw):
    """
    将欧拉角(roll, pitch, yaw)转换为旋转矩阵的前两列
    
    Args:
        roll: 绕x轴旋转角度(弧度)
        pitch: 绕y轴旋转角度(弧度) 
        yaw: 绕z轴旋转角度(弧度)
    
    Returns:
        6D向量，包含旋转矩阵前两列的6个元素
    """
    # 使用scipy创建旋转矩阵
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    rotation_matrix = r.as_matrix()
    
    # 取前两列并展平为6D向量
    # 使用'F' (Fortran order，按列展平) 以保持标准6D旋转表示格式
    #        | R11  R12  R13 |
    #    R = | R21  R22  R23 |
    #        | R31  R32  R33 |
    # [R11, R21, R31, R12, R22, R32] 而不是 [R11, R12, R21, R22, R31, R32]
    first_two_cols = rotation_matrix[:, :2].flatten('F')
    return first_two_cols

## ------------------------ camera key mapping -------------------------- ##

# 相机名称到新key格式的映射
CAMERA_KEY_MAPPING = {
    "image": "cam_head",
    "chest_image": "cam_chest",
    "left_shoulder_image": "cam_left",
    "right_shoulder_image": "cam_right",
}

def get_camera_observation_key(camera_name: str, use_image_features: bool = False) -> str:
    """
    根据相机名称获取对应的观测key
    
    Args:
        camera_name: 相机名称 (image, chest_image, left_shoulder_image, right_shoulder_image)
        use_image_features: 是否使用图像特征模式
        
    Returns:
        对应的观测key，格式统一为 observation.images.cam_{name}
        注意：embeds模式也使用 observation.images.* 格式，因为模型统一处理所有 observation.images.* 的key
    """
    # 获取相机的基础名称（cam_head, cam_chest等）
    cam_base_name = CAMERA_KEY_MAPPING.get(camera_name, f"cam_{camera_name}")
    
    # 统一使用 observation.images.* 格式（无论是原始图像还是embeds）
    # 模型会根据特征类型自动识别为视觉特征
    return f"observation.images.{cam_base_name}"

## ------------------------ state space -------------------------- ##

# 定义各个状态组件对应的状态名称
STATE_COMPONENT_DEFINITIONS = {
    "J_q": [
        # 左手七个关节 | 右手七个关节 (14维)
        "arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5", "arm_joint_6", "arm_joint_7",
        "arm_joint_8", "arm_joint_9", "arm_joint_10", "arm_joint_11", "arm_joint_12", "arm_joint_13", "arm_joint_14",
    ],
    "IMU": [
        # IMU 线加速度 (3维)
        "lin_acc_x", "lin_acc_y", "lin_acc_z",  
        # IMU 角速度 (3维)
        "ang_vel_x", "ang_vel_y", "ang_vel_z",  
    ],
    "Claw_pos": [
        # 夹爪状态 (2维)
        "left_claw_state", "right_claw_state",
    ],
    "Com_z_pitch": [
        # 质心z位置和pitch角度 (2维)
        "com_z_position", "com_pitch_eular",
    ],
}

def get_states_names(action_mode="delta", state_components=None):
    """
    根据action_mode和state_components返回对应的状态名称列表
    
    Args:
        action_mode: "absolute", "delta", "relative" (deprecated for depalletizer task)
        state_components: 状态组件列表，如果为None则使用全局STATE_COMPONENTS配置
                         可选值: ["J_q", "IMU", "Claw_pos"] 的组合
        
    Returns:
        list: 状态名称列表
        
    状态维度说明:
    - 根据STATE_COMPONENTS动态组合:
      - ["J_q", "IMU", "Claw_pos"]: 22维 = 14(arm_joints) + 6(IMU) + 2(claw_states)
      - ["J_q", "Claw_pos", "Com_z_pitch"]: 18维 = 14(arm_joints) + 2(claw_states) + 2(com_z_pitch)
      - ["J_q"]: 14维 = 14(arm_joints)
      - ["Claw_pos"]: 2维 = 2(claw_states)
      - ["J_q", "IMU"]: 20维 = 14(arm_joints) + 6(IMU)
    """
    # 如果没有指定state_components，使用全局配置
    if state_components is None:
        state_components = STATE_COMPONENTS
    
    # 确保state_components是列表
    if isinstance(state_components, str):
        state_components = [state_components]
    
    # 根据配置组合状态名称
    states_list = []
    for component in state_components:
        if component in STATE_COMPONENT_DEFINITIONS:
            states_list.extend(STATE_COMPONENT_DEFINITIONS[component])
        else:
            print(f"⚠️  Warning: Unknown state component '{component}'. Available components: {list(STATE_COMPONENT_DEFINITIONS.keys())}")
    
    # 打印配置信息
    if len(states_list) > 0:
        dims_info = []
        for component in state_components:
            if component in STATE_COMPONENT_DEFINITIONS:
                dims_info.append(f"{len(STATE_COMPONENT_DEFINITIONS[component])}({component})")
        print(f"📊 State configuration: {state_components} -> {len(states_list)}D [{'+'.join(dims_info)}]")
    else:
        print(f"⚠️  Warning: No valid state components selected. Using default 22D configuration.")
        # 如果没有任何有效组件，返回默认的22维配置
        states_list = (
            STATE_COMPONENT_DEFINITIONS["J_q"] +
            STATE_COMPONENT_DEFINITIONS["IMU"] +
            STATE_COMPONENT_DEFINITIONS["Claw_pos"]
        )
    
    return states_list

# 定义各个动作组件对应的动作名称
ACTION_COMPONENT_DEFINITIONS = {
    "Left_arm": [
        "arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5", "arm_joint_6", "arm_joint_7",
    ],
    "Right_arm": [
        "arm_joint_8", "arm_joint_9", "arm_joint_10", "arm_joint_11", "arm_joint_12", "arm_joint_13", "arm_joint_14",
    ],
    "Left_claw": [
        "left_claw_position",
    ],
    "Right_claw": [
        "right_claw_position",
    ],
    "Cmd_pose_z": [
        "cmd_pose_z",
    ],
    "Cmd_pose_pitch": [
        "cmd_pose_pitch",
    ],
}

def get_actions_names(action_components=None):
    """
    根据action_components返回对应的动作名称列表
    
    Args:
        action_components: 动作组件列表，如果为None则使用全局ACTION_COMPONENTS配置
                         可选值: ["Left_arm", "Right_arm", "Left_claw", "Right_claw", "Cmd_pose_z", "Cmd_pose_pitch"] 的组合
        
    Returns:
        list: 动作名称列表
        
    动作维度说明:
    - 根据ACTION_COMPONENTS动态组合:
      - ["Left_arm", "Right_arm", "Left_claw", "Right_claw"]: 16维 = 7+7+1+1
      - ["Left_arm", "Right_arm", "Left_claw", "Right_claw", "Cmd_pose_z", "Cmd_pose_pitch"]: 18维 = 7+7+1+1+1+1
      - ["Left_arm", "Right_arm"]: 14维 = 7+7
    """
    # 如果没有指定action_components，使用全局配置
    if action_components is None:
        action_components = ACTION_COMPONENTS
    
    # 确保action_components是列表
    if isinstance(action_components, str):
        action_components = [action_components]
    
    # 根据配置组合动作名称
    actions_list = []
    for component in action_components:
        if component in ACTION_COMPONENT_DEFINITIONS:
            actions_list.extend(ACTION_COMPONENT_DEFINITIONS[component])
        else:
            print(f"⚠️  Warning: Unknown action component '{component}'. Available components: {list(ACTION_COMPONENT_DEFINITIONS.keys())}")
    
    # 打印配置信息
    if len(actions_list) > 0:
        dims_info = []
        for component in action_components:
            if component in ACTION_COMPONENT_DEFINITIONS:
                dims_info.append(f"{len(ACTION_COMPONENT_DEFINITIONS[component])}({component})")
        print(f"🎮 Action configuration: {action_components} -> {len(actions_list)}D [{'+'.join(dims_info)}]")
    else:
        print(f"⚠️  Warning: No valid action components selected. Using default 16D configuration.")
        # 如果没有任何有效组件，返回默认的16维配置
        actions_list = (
            ACTION_COMPONENT_DEFINITIONS["Left_arm"] +
            ACTION_COMPONENT_DEFINITIONS["Right_arm"] +
            ACTION_COMPONENT_DEFINITIONS["Left_claw"] +
            ACTION_COMPONENT_DEFINITIONS["Right_claw"]
        )
    
    return actions_list

# 根据当前ACTION_MODE和STATE_COMPONENTS获取状态名称
states_names = get_states_names(ACTION_MODE, STATE_COMPONENTS)

## ---------------- action space --------------------------- ## 
# 根据ACTION_COMPONENTS动态生成action_names
action_names = get_actions_names(ACTION_COMPONENTS)

def process_Image(msg, data_dict, name, ts=None):
    if msg.encoding != 'rgb8':
        # Handle different encodings here if necessary
        raise ValueError(f"Unsupported encoding: {msg.encoding}. Expected 'rgb8'.")

    # Convert the ROS Image message to a numpy array
    img_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    # If the image is in 'bgr8' format, convert it to 'rgb8'
    if msg.encoding == 'bgr8':
        cv_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    else:
        cv_img = img_arr
    if ts is None:
        ts = msg.header.stamp.to_sec()
    data_dict[name]['data'].append(cv_img)
    data_dict[name]['ts'].append(ts)

    # print(f"Image shape: {cv_img.shape}, Timestamp: {ts}")

def process_MultiArray(msg, data_dict, name, ts=None):
    """
    注： data_dict是外部可变对象
    """
    data = list(msg.data)
    ts = ts

    if ts is None:
        ts = rospy.Time.now().to_sec()

    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

    # print(f"MultiArray data: {data}, Timestamp: {ts}")

def process_wbc_observation(msg, data_dict, name, ts=None):
    data = list(msg.state.value[:12])  # Only extract the first 6 elements
    
    if ts is None:
        ts = rospy.Time.now().to_sec()
    
    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

def process_wbc_observation_z_pitch(msg, data_dict, name, ts=None):
    com_z_position = msg.state.value[8]
    com_pitch_eular = msg.state.value[10]
    data = list([com_z_position, com_pitch_eular])  # Only extract the first 6 elements
    
    if ts is None:
        ts = rospy.Time.now().to_sec()
    
    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

def process_wbc_observation_com_state(msg, data_dict, name, ts=None):
    # 提取质心位置和欧拉角 [x, y, z, roll, pitch, yaw]
    state_values = msg.state.value[6:12]
    x, y, z, yaw, pitch, roll = state_values
    
    # 将欧拉角转换为旋转矩阵前2列
    rotation_cols = euler_to_rotation_matrix_first_two_cols(roll, pitch, yaw)
    
    # 组合位置和旋转矩阵前2列: [x, y, z, R11, R21, R31, R12, R22, R32]
    data = list(np.concatenate([[x, y, z], rotation_cols]))
    
    if ts is None:
        ts = rospy.Time.now().to_sec()
    
    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

def process_wbc_observation_com_vel(msg, data_dict, name, ts=None):
    com_linear_vel = msg.state.value[0:3]
    com_angular_vel = msg.state.value[3:6]
    data = list(np.concatenate([com_linear_vel, com_angular_vel]))
    if ts is None:
        ts = rospy.Time.now().to_sec()
    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

def process_wbc_observation_com_state_euler(msg, data_dict, name, ts=None):
    """提取完整的COM状态：位置和欧拉角 [x, y, z, yaw, pitch, roll]"""
    # 提取质心位置和欧拉角 [x, y, z, roll, pitch, yaw]
    state_values = msg.state.value[6:12]
    x, y, z, yaw, pitch, roll = state_values
    
    # 直接使用欧拉角格式（不转换为旋转矩阵）: [x, y, z, yaw, pitch, roll]
    data = list([x, y, z, yaw, pitch, roll])
    
    if ts is None:
        ts = rospy.Time.now().to_sec()
    
    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

def process_sensorsData(msg, data_dict, name, ts=None):
    arm_begin = 12
    arm_end = 25
    if ts is None:
        ts = msg.header.stamp.to_sec()

    data = msg.joint_data.joint_q
    data = list(data[arm_begin:arm_end+1])
    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

    # print(f"sensorsData: {data}, Timestamp: {msg.header.stamp.to_sec()}")

def process_sensorsData_vel(msg, data_dict, name, ts=None):
    arm_begin = 12
    arm_end = 25
    if ts is None:
        ts = msg.header.stamp.to_sec()

    data = msg.joint_data.joint_v
    data = list(data[arm_begin:arm_end+1])
    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

def process_jointCmd(msg, data_dict, name, ts=None):
    arm_begin = 12
    arm_end = 25

    ts = msg.header.stamp.to_sec()
    joint_q = msg.joint_q
    joint_v = msg.joint_v
    joint_tau = msg.tau

    if ts is None:
        ts = msg.header.stamp.to_sec()

    data_dict[name]['data'].append(list(joint_tau[arm_begin:arm_end+1]))
    data_dict[name]['ts'].append(ts)

def process_JointState(msg, data_dict, name, ts=None):
    """
        deg2rad 手臂关节归一化
    """
    joint_q = np.deg2rad(msg.position)  # Convert degrees to radians
    
    if ts is None:
        ts = msg.header.stamp.to_sec()

    data_dict[name]['data'].append(list(joint_q))
    data_dict[name]['ts'].append(ts)

    # print(f"JointState data: {joint_q}, Timestamp: {ts}")

def process_Twist(msg, data_dict, name, ts=None):
    data = [
        msg.linear.x,
        msg.linear.y,
        msg.linear.z,
        msg.angular.x,
        msg.angular.y,
        msg.angular.z,
    ]

    if ts is None:
        ts = msg.header.stamp.to_sec()

    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

def process_Pose(msg, data_dict, name, ts=None):
    data = [
        msg.linear.x,
        msg.linear.y,
        msg.linear.z,
        msg.angular.x,
        msg.angular.y,
        msg.angular.z,
    ]

    if ts is None:
        ts = msg.header.stamp.to_sec()

    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

def process_Wrench(msg, data_dict, name, ts=None):
    data = [
        msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5],
        msg.data[6], msg.data[7], msg.data[8], msg.data[9], msg.data[10], msg.data[11]
    ]
    if ts is None:
        ts = msg.header.stamp.to_sec()

    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

def process_PoseStamped(msg, data_dict, name, ts=None):
    """
    处理geometry_msgs/PoseStamped消息，将四元数转换为6D旋转表示
    
    Args:
        msg: geometry_msgs/PoseStamped消息
        data_dict: 数据字典
        name: 数据名称
        ts: 时间戳（可选）
    
    Returns:
        9D向量: [x, y, z, R11, R21, R31, R12, R22, R32]
    """
    # 提取位置
    position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    
    # 提取四元数 (x, y, z, w)
    quat = [msg.pose.orientation.x, msg.pose.orientation.y, 
            msg.pose.orientation.z, msg.pose.orientation.w]
    
    # 将四元数转换为旋转矩阵
    r = Rotation.from_quat(quat)
    rotation_matrix = r.as_matrix()
    
    # 取前两列并展平为6D向量 (使用'F' order保持标准6D旋转表示格式)
    rotation_cols = rotation_matrix[:, :2].flatten('F')
    
    # 组合位置和旋转矩阵前2列: [x, y, z, R11, R21, R31, R12, R22, R32]
    data = list(np.concatenate([position, rotation_cols]))
    
    if ts is None:
        ts = msg.header.stamp.to_sec()
    
    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

def process_lejuClawState(msg, data_dict, name, ts=None):
    """
    处理kuavo_msgs/lejuClawState消息，提取左右夹爪状态
    
    Args:
        msg: kuavo_msgs/lejuClawState消息
        data_dict: 数据字典
        name: 数据名称
        ts: 时间戳（可选）
    
    Returns:
        2D向量: [left_claw_position, right_claw_position]
    """
    # 提取左右夹爪位置
    # msg.data.position[0] - 左夹爪状态
    # msg.data.position[1] - 右夹爪状态
    if len(msg.data.position) >= 2:
        left_claw_state = msg.data.position[0]
        right_claw_state = msg.data.position[1]
    else:
        # 如果数据不足，用零填充
        left_claw_state = 0.0
        right_claw_state = 0.0
    
    data = [left_claw_state, right_claw_state]
    
    if ts is None:
        ts = msg.header.stamp.to_sec()
    
    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

def process_lejuClawCommand(msg, data_dict, name, ts=None):
    """
    处理kuavo_msgs/lejuClawCommand消息，提取左右夹爪命令
    
    Args:
        msg: kuavo_msgs/lejuClawCommand消息
        data_dict: 数据字典
        name: 数据名称
        ts: 时间戳（可选）
    
    Returns:
        2D向量: [left_claw_position, right_claw_position]
    """
    # 提取左右夹爪位置
    # msg.data.position[0] - 左夹爪命令
    # msg.data.position[1] - 右夹爪命令
    if len(msg.data.position) >= 2:
        left_claw_cmd = msg.data.position[0]
        right_claw_cmd = msg.data.position[1]
    else:
        # 如果数据不足，用零填充
        left_claw_cmd = 0.0
        right_claw_cmd = 0.0
    
    data = [left_claw_cmd, right_claw_cmd]
    
    if ts is None:
        ts = msg.header.stamp.to_sec()
    
    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

def process_cmd_pose(msg, data_dict, name, ts=None):
    """
    处理geometry_msgs/Twist消息（cmd_pose），提取z位置和pitch角度
    
    Args:
        msg: geometry_msgs/Twist消息
        data_dict: 数据字典
        name: 数据名称
        ts: 时间戳（可选）
    
    Returns:
        2D向量: [cmd_pose_z, cmd_pose_pitch]
    """
    # 提取z位置和pitch角度
    cmd_pose_z = msg.linear.z
    cmd_pose_pitch = msg.angular.y  # pitch对应angular.y
    
    data = [cmd_pose_z, cmd_pose_pitch]
    
    if ts is None:
        ts = rospy.Time.now().to_sec()
    
    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)

def get_topic_info(action_mode="delta", task_data_mode="strategy"):
    """
    根据action_mode返回对应的topic配置
    
    Args:
        action_mode: "absolute", "delta", "relative"
        task_data_mode: "VR" or "strategy"
        
    Returns:
        dict: topic配置字典
    """
    print(f" =================== Action Mode: {action_mode.upper()} ================= ")
    print(f" =================== Task Data Mode: {task_data_mode.upper()} ================= ")
    print(f" =================== State dimensions: {len(get_states_names(action_mode))} ================= ")
    print(f" =================== Action components: {ACTION_COMPONENTS} ================= ")
    print(f" =================== State components: {STATE_COMPONENTS} ================= ")
    
    # 统一转换为大写进行比较，支持大小写不敏感
    task_data_mode_upper = task_data_mode.upper()
    
    if task_data_mode_upper == "VR":
        print(" =================== Set camera topic to /camera/color/image_raw \
               /chest_cam/color/image_raw \ /left_cam/color/image_raw /right_cam/color/image_raw ==================")
        return {
                    # ----------------------------------------- image ----------------------------------------------------- #
                    "image": {
                        "topic": "/camera/color/image_raw",
                        "msg_process_fn": process_Image,
                        "shape": None,
                    },

                    # new image obs
                    "chest_image": {
                        "topic": "/chest_cam/color/image_raw",  # /chest_cam/color/image_raw
                        "msg_process_fn": process_Image,
                        "shape": None,
                    },

                    # left shoulder camera
                    "left_shoulder_image": {
                        "topic": "/left_cam/color/image_raw",  # /left_cam/color/image_raw
                        "msg_process_fn": process_Image,
                        "shape": None,
                    },

                    # right shoulder camera
                    "right_shoulder_image": {
                        "topic": "/right_cam/color/image_raw",
                        "msg_process_fn": process_Image,
                        "shape": None,
                    },

                    # ----------------------------------------- obs ----------------------------------------------------- #
                    # 手臂关节状态
                    "dof_state": {
                        "topic": "/sensors_data_raw",
                        "msg_process_fn": process_sensorsData,
                        "shape": None,
                    },
                    # 手臂关节速度
                    "dof_state_vel": {
                        "topic": "/sensors_data_raw",
                        "msg_process_fn": process_sensorsData_vel,
                        "shape": None,
                    },
                    
                    # imu
                    "ang_vel": {
                        "topic": "/state_estimate/imu_data_filtered/angularVel",
                        "msg_process_fn": process_MultiArray,
                        "shape": None,
                    },

                    "lin_acc": {
                        "topic": "/state_estimate/imu_data_filtered/linearAccel",
                        "msg_process_fn": process_MultiArray,
                        "shape": None,
                    },

                    # 夹爪状态
                    "claw_state": {
                        "topic": "/leju_claw_state",
                        "msg_process_fn": process_lejuClawState,
                        "shape": (2,),
                    },
                    
                    # 质心z位置和pitch角度（从/humanoid_wbc_observation获取）
                    "com_z_pitch": {
                        "topic": "/humanoid_wbc_observation",
                        "msg_process_fn": process_wbc_observation_z_pitch,
                        "shape": (2,),
                    },

                    # ----------------------------------------- action ----------------------------------------------------- #
                    "action_arm": {  # 手臂关节位置
                        "topic": "/kuavo_arm_traj",
                        "msg_process_fn": process_JointState,
                        "shape": (14,),
                    },

                    "action_claw": {  # 夹爪命令
                        "topic": "/leju_claw_command",
                        "msg_process_fn": process_lejuClawCommand,
                        "shape": (2,),
                    },
                    
                    "action_cmd_pose": {  # cmd_pose命令 (z位置和pitch角度)
                        "topic": "/cmd_pose",
                        "msg_process_fn": process_cmd_pose,
                        "shape": (2,),
                    }
        }
    elif task_data_mode_upper == "STRATEGY":
        print(" =================== Set camera topic to /camera/color/image_raw \
               /chest_cam/color/image_raw /left_cam/color/image_raw /right_cam/color/image_raw ==================")
        return {
                    # ----------------------------------------- image ----------------------------------------------------- #
                    # 统一使用4相机配置，即使strategy模式也包含chest_image（可能为空数据）
                    "image": {
                        "topic": "/camera/color/image_raw",
                        "msg_process_fn": process_Image,
                        "shape": None,
                    },

                    # chest camera (strategy模式可能没有数据，但保留占位符以保持4相机配置)
                    "chest_image": {
                        "topic": "/chest_cam/color/image_raw",
                        "msg_process_fn": process_Image,
                        "shape": None,
                    },

                    # left shoulder camera
                    "left_shoulder_image": {
                        "topic": "/left_cam/color/image_raw",  # /left_cam/color/image_raw
                        "msg_process_fn": process_Image,
                        "shape": None,
                    },

                    # right shoulder camera
                    "right_shoulder_image": {
                        "topic": "/right_cam/color/image_raw",
                        "msg_process_fn": process_Image,
                        "shape": None,
                    },

                    # ----------------------------------------- obs ----------------------------------------------------- #
                    # 手臂关节状态
                    "dof_state": {
                        "topic": "/sensors_data_raw",
                        "msg_process_fn": process_sensorsData,
                        "shape": None,
                    },
                    # 手臂关节速度
                    "dof_state_vel": {
                        "topic": "/sensors_data_raw",
                        "msg_process_fn": process_sensorsData_vel,
                        "shape": None,
                    },
                    
                    # imu
                    "ang_vel": {
                        "topic": "/state_estimate/imu_data_filtered/angularVel",
                        "msg_process_fn": process_MultiArray,
                        "shape": None,
                    },

                    "lin_acc": {
                        "topic": "/state_estimate/imu_data_filtered/linearAccel",
                        "msg_process_fn": process_MultiArray,
                        "shape": None,
                    },

                    # 夹爪状态
                    "claw_state": {
                        "topic": "/leju_claw_state",
                        "msg_process_fn": process_lejuClawState,
                        "shape": (2,),
                    },
                    
                    # 质心z位置和pitch角度（从/humanoid_wbc_observation获取）
                    "com_z_pitch": {
                        "topic": "/humanoid_wbc_observation",
                        "msg_process_fn": process_wbc_observation_z_pitch,
                        "shape": (2,),
                    },

                    # ----------------------------------------- action ----------------------------------------------------- #
                    "action_arm": {  # 手臂关节位置
                        "topic": "/mm_kuavo_arm_traj",
                        "msg_process_fn": process_JointState,
                        "shape": (14,),
                    },

                    "action_claw": {  # 夹爪命令
                        "topic": "/leju_claw_command",
                        "msg_process_fn": process_lejuClawCommand,
                        "shape": (2,),
                    },
                    
                    "action_cmd_pose": {  # cmd_pose命令 (z位置和pitch角度)
                        "topic": "/cmd_pose",
                        "msg_process_fn": process_cmd_pose,
                        "shape": (2,),
                    }
        }

def set_action_mode(action_mode):
    """
    动态设置action_mode并更新相关的配置
    
    Args:
        action_mode: "absolute", "delta", "relative"
    """
    global ACTION_MODE, TASK_DATA_MODE, states_names, topic_info
    
    ACTION_MODE = action_mode
    states_names = get_states_names(ACTION_MODE)
    topic_info = get_topic_info(ACTION_MODE)
    
    print(f"✅ Action mode updated to: {ACTION_MODE}")
    print(f"✅ State dimensions: {len(states_names)}")
    print(f"✅ Topic info updated")

# 根据当前ACTION_MODE获取topic配置
topic_info = get_topic_info(ACTION_MODE, TASK_DATA_MODE)