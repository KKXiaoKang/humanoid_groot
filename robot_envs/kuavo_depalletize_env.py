"""
策略网络在此和mujoco里的机器人交互
"""
import time

import numpy as np
from typing import List, Optional, Union, Dict, Callable
import rospy

from sensor_msgs.msg import Image
from ocs2_msgs.msg import mpc_observation
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose, PoseStamped
from std_msgs.msg import Float64MultiArray
from collections import deque
import math
from kuavo_msgs.msg import sensorsData, lejuClawCommand, lejuClawState
from cv_bridge import CvBridge
import cv2
from tqdm import tqdm

# from config import process_Image
from configs.config import topic_info, TASK_DATA_MODE, STATE_COMPONENTS, ACTION_COMPONENTS, CAMERA_COMPONENTS, get_camera_names

class TargetPublisher:
    """
    在这里定义各种发布
    类里包含很多 ros publisher
    """
    def __init__(self):
        # 1. 发布arm的action (control mode = )
        self.arm_action_publisher = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=10)
        # 2. 发布夹爪命令
        self.claw_action_publisher = rospy.Publisher('/leju_claw_command', lejuClawCommand, queue_size=10)
        # 3. 发布cmd_pose命令（如果ACTION_COMPONENTS包含cmd_pose相关组件）
        if "Cmd_pose_z" in ACTION_COMPONENTS or "Cmd_pose_pitch" in ACTION_COMPONENTS:
            from geometry_msgs.msg import Twist
            self.cmd_pose_publisher = rospy.Publisher('/cmd_pose', Twist, queue_size=10)
        else:
            self.cmd_pose_publisher = None

        self.last_action_exec_time = time.time()

    def publish_target_arm_claw(self, arm_action: np.ndarray, claw_action: np.ndarray, control_arm: bool = True, control_claw: bool = True):
        """
        发布arm和夹爪的目标
        Args:
            arm_action: 手臂关节角度 (14维)
            claw_action: 夹爪位置 [left_claw, right_claw] (2维)
            control_arm: 是否控制手臂
            control_claw: 是否控制夹爪
        """
        msg_arm = JointState()
        msg_arm.header.stamp = rospy.Time.now()
        msg_arm.name = [
            "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
            # 左手七个关节
            "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
        ]
        msg_arm.position = np.rad2deg(arm_action.tolist()) # 弧度转角度
        
        # 发布夹爪命令
        msg_claw = lejuClawCommand()
        msg_claw.data.name = ['left_claw', 'right_claw']
        msg_claw.data.position = claw_action.tolist()  # [left_claw, right_claw]
        msg_claw.data.velocity = [90.0, 90.0]  # 默认速度
        msg_claw.data.effort = [1.0, 1.0]  # 默认力矩

        if control_arm:
            self.arm_action_publisher.publish(msg_arm)
        if control_claw:
            self.claw_action_publisher.publish(msg_claw)

        # 拆分左右手臂动作
        left_arm_str = np.array2string(msg_arm.position[:7], precision=8, suppress_small=True, separator=', ')
        right_arm_str = np.array2string(msg_arm.position[7:14], precision=8, suppress_small=True, separator=', ')
        
        # print(f'Published arm actions:')
        # print(f'Left arm (7 joints):  {left_arm_str}')
        # print(f'Right arm (7 joints): {right_arm_str}')
        # print(f'Claw command: [left={claw_action[0]:.2f}, right={claw_action[1]:.2f}]')

ISAAC_SIM_CAMERA_FLAG = False
USE_WBC_OBS = False
class ObsBuffer:
    """
    订阅ros话题，获取当前状态。（这里面的subscriber单独在自己的线程里update）
    """
    def __init__(self):
        
        # image - 根据topic_info动态配置相机
        if ISAAC_SIM_CAMERA_FLAG:
            self.img_topic_map = {
                'image': {
                    'topic': '/camera/rgb/image_raw',
                    'msg_type': Image,
                    'frequency': 30,
                    'callback': self.common_callback,
                    'size_wh': (640, 480)
                }
            }
        else:
            # 从topic_info中读取相机配置，根据CAMERA_COMPONENTS动态设置
            self.img_topic_map = {}
            
            # 根据CAMERA_COMPONENTS获取相机名称列表
            camera_names = get_camera_names(CAMERA_COMPONENTS)
            
            for camera_name in camera_names:
                if camera_name in topic_info:
                    camera_config = topic_info[camera_name]
                    self.img_topic_map[camera_name] = {
                        'topic': camera_config['topic'],
                        'msg_type': Image,
                        'frequency': 30,
                        'callback': self.common_callback,
                        'size_wh': (640, 480)
                    }
            
            print(f"📷 Camera configuration based on CAMERA_COMPONENTS ({CAMERA_COMPONENTS}):")
            print(f"   Detected {len(self.img_topic_map)} cameras: {list(self.img_topic_map.keys())}")

        # obs
        if USE_WBC_OBS:
            self.obs_topic_map = {
                'dof_state': {
                    'topic': '/sensors_data_raw',
                    'msg_type': sensorsData,
                    'frequency': 30,
                    'callback': self.common_callback,
                },
                'ang_vel': {
                    'topic': '/state_estimate/imu_data_filtered/angularVel',
                    'msg_type': Float64MultiArray,
                    'frequency': 500,
                    'callback': self.common_callback,
                },
                'lin_acc': {
                    'topic': '/state_estimate/imu_data_filtered/linearAccel',
                    'msg_type': Float64MultiArray,
                    'frequency': 500,
                    'callback': self.common_callback,
                },
                'humanoid_wbc_observation': {
                    'topic': '/humanoid_wbc_observation',
                    'msg_type': mpc_observation,
                    'frequency': 500,
                    'callback': self.common_callback,
                },
            }
        else:
            self.obs_topic_map = {
                # 手臂关节状态
                'dof_state': {
                    'topic': '/sensors_data_raw',
                    'msg_type': sensorsData,
                    'frequency': 30,
                    'callback': self.common_callback,
                },
                # 手臂关节速度
                'dof_state_vel': {
                    'topic': '/sensors_data_raw',
                    'msg_type': sensorsData,
                    'frequency': 30,
                    'callback': self.common_callback,
                },
                # IMU角速度
                'ang_vel': {
                    'topic': '/state_estimate/imu_data_filtered/angularVel',
                    'msg_type': Float64MultiArray,
                    'frequency': 500,
                    'callback': self.common_callback,
                },
                # IMU线加速度
                'lin_acc': {
                    'topic': '/state_estimate/imu_data_filtered/linearAccel',
                    'msg_type': Float64MultiArray,
                    'frequency': 500,
                    'callback': self.common_callback,
                },
                # 夹爪状态（必需，因为状态空间包含夹爪状态）
                'claw_state': {
                    'topic': '/leju_claw_state',
                    'msg_type': lejuClawState,
                    'frequency': 30,
                    'callback': self.common_callback,
                },
            }
            
            # 如果STATE_COMPONENTS包含Com_z_pitch，添加质心观测
            if "Com_z_pitch" in STATE_COMPONENTS:
                from ocs2_msgs.msg import mpc_observation
                self.obs_topic_map['com_z_pitch'] = {
                    'topic': '/humanoid_wbc_observation',
                    'msg_type': mpc_observation,
                    'frequency': 500,
                    'callback': self.common_callback,
                }

        self.base_action = None
        self.arm_action = None

        # ---------- init obs_buffer_data --------------- #
        self.obs_buffer_data = {key: {"data": deque(maxlen=self.img_topic_map[key]["frequency"]),"ts": deque(maxlen=self.img_topic_map[key]["frequency"]),} \
                                for key in self.img_topic_map}

        self.obs_buffer_data.update({key: {"data": deque(maxlen=self.obs_topic_map[key]["frequency"]),"ts": deque(maxlen=self.obs_topic_map[key]["frequency"]),} \
                                    for key in self.obs_topic_map})

        # print(f'---------------- self.obs_buffer_data {self.obs_buffer_data} ------------')
        self.setup_subscribers()

    def setup_subscribers(self):
        """
        对于每个话题，创建一个ros subscriber
        Returns:

        """
        self.suber_dict = {}
        for obs_name, obs_info in self.obs_topic_map.items():
            topic = obs_info['topic']
            msg_type = obs_info['msg_type']
            frequency = obs_info['frequency']
            callback = obs_info['callback']

            suber = rospy.Subscriber(topic, msg_type, callback, callback_args=obs_name)

            print(callback)
            self.suber_dict[obs_name] = suber

        for obs_name, obs_info in self.img_topic_map.items():
            topic = obs_info['topic']
            msg_type = obs_info['msg_type']
            frequency = obs_info['frequency']
            callback = obs_info['callback']

            # 创建一个subscriber
            suber = rospy.Subscriber(topic, msg_type, callback, callback_args=obs_name)
            self.suber_dict[obs_name] = suber

    # --------- some callback functions --------------- #

    def common_callback(self, msg, name: str):
        # 检查 name 是否在 topic_info 中，如果不在则跳过处理（可能是已删除的topic）
        if name not in topic_info:
            rospy.logwarn(f"Skipping callback for topic '{name}' as it is not in topic_info (may have been removed)")
            return
        process_fn = topic_info[name]['msg_process_fn']
        process_fn(msg, self.obs_buffer_data, name)


    # ----------- 以特殊方式从buffer里获取数据 --------------- #
    def get_latest_k_state(self, k_frames_per_topic):
        """

        Args:
            k_frames_per_topic: 每个话题要取的k是多少

        Returns:

        """
        out = {}
        for name, info in self.obs_topic_map.items():
            k = k_frames_per_topic[name]
            out[name] = {
            "data": np.asarray(list(self.obs_buffer_data[name]["data"])[-k:]),  # 取最后的k个
            "robot_receive_timestamp": np.asarray(list(self.obs_buffer_data[name]["ts"])[-k:])  # 取最后的k个
            }

        return out


    def get_latest_k_img(self, k_frames_per_img_topic):
        """
        获取图像的buffer
        Args:
            k_frames_per_img_topic: 每个话题要取的k是多少

        Returns:

        """
        out = {}
        for name, info in self.img_topic_map.items():
            k = k_frames_per_img_topic[name]
            out[name] = {
                "data": np.asarray(list(self.obs_buffer_data[name]["data"])[-k:]),  # 取最后的k个
                "robot_receive_timestamp": np.asarray(list(self.obs_buffer_data[name]["ts"])[-k:])  # 取最后的k个
            }

        return out

    # ---------------- 一些启动和检查buffer的函数 ---------------- #

    def obs_buffer_is_ready(self):
        """
        所有观测初始化成功的判断
        Args:
            just_img:

        Returns:

        """
        return all([len(self.obs_buffer_data[key]["data"]) == self.img_topic_map[key]["frequency"] for key in self.img_topic_map]) and \
            all([len(self.obs_buffer_data[key]["data"]) == self.obs_topic_map[key]["frequency"] for key in self.obs_topic_map])

    def wait_buffer_ready(self, just_img: bool = False):
        progress_bars = {}
        position = 0

        for key in self.img_topic_map:
            progress_bars[key] = tqdm(
                total=self.img_topic_map[key]["frequency"],
                desc=f"Filling {key}",
                position=position,
                leave=True
            )
            position += 1

        for key in self.obs_topic_map:
            progress_bars[key] = tqdm(
                total=self.obs_topic_map[key]["frequency"],
                desc=f"Filling {key}",
                position=position,
                leave=True
            )
            position += 1

        try:
            while not self.obs_buffer_is_ready():
                for key in self.img_topic_map:
                    current_len = len(self.obs_buffer_data[key]["data"])
                    progress_bars[key].n = current_len
                    progress_bars[key].refresh()

                for key in self.obs_topic_map:
                    current_len = len(self.obs_buffer_data[key]["data"])
                    progress_bars[key].n = current_len
                    progress_bars[key].refresh()

                time.sleep(1)  # 降低CPU负载，提升ctrl+c响应性

        except KeyboardInterrupt:
            print("\n[Interrupted] Exiting by user Ctrl+C.")

        print("All buffers are ready!")
        time.sleep(0.5)

class GrabBoxMpcEnv:
    """
    和mujoco里的机器人交互
    """
    def __init__(self):
        # 在这里直接 init node
        # rospy.init_node('manip', anonymous=True)
        self.target_publisher = TargetPublisher()
        self.obs_buffer = ObsBuffer()
        self.control_frequency = 100  # 这个为策略控制机器人的频率。注意，还有一种频率是data_frequency, 是每个话题自己更新读数的频率
        self.control_dt = 1.0 / self.control_frequency
        self.obs_topic_map = self.obs_buffer.obs_topic_map
        self.img_topic_map = self.obs_buffer.img_topic_map

        self.last_action_exec_time = time.time()

        self.n_obs_steps = 1 # 每次获取obs的历史跨越多少个控制帧

    def get_obs(self):
        """
        订阅ros话题，获取当前状态
        Returns:
        """
        # TODO: 在这里检查buffer是否ready

        # ============= 获取相机的obs ================ #
        k_frames_per_img_topic = {
                name: min(self.img_topic_map[name]["frequency"],
                          math.ceil((self.n_obs_steps + 3) * (self.img_topic_map[name]["frequency"] / self.control_frequency)))
            for name in self.img_topic_map if 'image' in name
        }

        last_img_data = self.obs_buffer.get_latest_k_img(k_frames_per_img_topic)

        # print(last_img_data)

        # 取时间，然后align
        dt = self.control_dt
        # 安全获取时间戳，避免索引错误
        timestamps = []
        for x in last_img_data.values():
            ts = x["robot_receive_timestamp"]
            if len(ts) >= 2:
                timestamps.append(ts[-2])  # 倒数第二个
            elif len(ts) >= 1:
                timestamps.append(ts[-1])  # 如果只有一个，用最后一个
            else:
                print(f"Warning: Empty timestamp array in image data")
                timestamps.append(0.0)  # 默认值
        
        if timestamps:
            last_timestamp = np.min(timestamps)
        else:
            print("Error: No valid timestamps found")
            last_timestamp = 0.0

        # 形成网络观测历史的时间戳
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        # 生成真正准备输入网络的obs
        camera_obs = dict()
        camera_obs_ts = dict()  # 时间戳 (why时间戳is important ??
        for name, value in last_img_data.items():
            # 对于每个topic
            topic_ts = value["robot_receive_timestamp"]
            picked_idx = list()  # 最后选取的帧
            for t in obs_align_timestamps:
                # 对于每个时间戳
                # FIXME: 会有问题是，有可能每帧都是同一个idx
                idx = np.argmin(np.abs(topic_ts - t))
                picked_idx.append(idx)

            camera_obs[name] = value["data"][picked_idx]
            camera_obs_ts[name] = topic_ts[picked_idx]


        # ============= 获取robot的obs ================ #
        # n_obs_steps * (data_freq / ctrl_freq) : 在data源数据中要取多少帧
        k_frames_per_topic = {
                name: min(self.obs_topic_map[name]["frequency"],
                          math.ceil((self.n_obs_steps + 3) * (self.obs_topic_map[name]["frequency"] / self.control_frequency)))
            for name in self.obs_topic_map if 'img' not in name
            }
        last_robot_data = self.obs_buffer.get_latest_k_state(k_frames_per_topic)

        # 生成真正准备输入网络的obs
        robot_obs = dict()
        robot_obs_ts = dict()
        for name, value in last_robot_data.items():
            # 对于每个topic
            topic_ts = value["robot_receive_timestamp"]
            picked_idx = list()
            for t in obs_align_timestamps:
                this_idx = np.argmin(np.abs(topic_ts - t))
                picked_idx.append(this_idx)

            robot_obs[name] = value["data"][picked_idx]
            robot_obs_ts[name] = topic_ts[picked_idx]

        # 把所有的模态data集中起来
        '''
        obs_data = {
            "image": (T,H,W,C),
            "img02": (T,H,W,C),
            "img...": (T,H,W,C),
            "agent_pos": (T,D),
            "ts": (T,)
        }
        '''
        obs_data = dict(camera_obs)
        if USE_WBC_OBS:
            """ 带 wbc obs 观测"""
            all_non_img_states = np.concatenate((robot_obs['dof_state'],
                                                robot_obs['lin_acc'],
                                                robot_obs['ang_vel'],
                                                robot_obs['claw_state'],
                                                robot_obs['humanoid_wbc_observation']), axis=1)
        else:
            """不带 wbc obs - depalletize任务: 根据STATE_COMPONENTS配置动态组合状态""" 
            state_parts = []
            for component in STATE_COMPONENTS:
                if component == "J_q":
                    # 手臂关节位置 (14维)
                    state_parts.append(robot_obs['dof_state'])
                elif component == "IMU":
                    # IMU数据 (6维: 3维线加速度 + 3维角速度)
                    state_parts.append(robot_obs['lin_acc'])
                    state_parts.append(robot_obs['ang_vel'])
                elif component == "Claw_pos":
                    # 夹爪状态 (2维)
                    state_parts.append(robot_obs['claw_state'])
                elif component == "Com_z_pitch":
                    # 质心z位置和pitch角度 (2维)
                    if 'com_z_pitch' in robot_obs:
                        state_parts.append(robot_obs['com_z_pitch'])
                    else:
                        print("⚠️  Warning: Com_z_pitch component in STATE_COMPONENTS but com_z_pitch not in robot_obs")
            
            if state_parts:
                all_non_img_states = np.concatenate(state_parts, axis=1)
            else:
                # 如果没有配置，使用默认的16维配置
                all_non_img_states = np.concatenate((robot_obs['dof_state'],
                                                    robot_obs['claw_state']), axis=1)
        
        # 更新obs_data
        obs_data.update(
            {
                "state": all_non_img_states,  # FIXME: 这里的名字叫agent_pos？
            }
        )

        return obs_data, camera_obs, camera_obs_ts, robot_obs, robot_obs_ts,

    def exec_actions(
        self,
        actions: np.ndarray,
        control_arm: bool = True,
        control_claw: bool = True,
        control_cmd_pose: bool = True,
    ):
        """
        把网络推理出的action变成话题发布
        Args:
            actions: 动作数组，根据ACTION_COMPONENTS动态组合
                    格式根据ACTION_COMPONENTS决定，例如:
                    - ["Left_arm", "Right_arm", "Left_claw", "Right_claw"]: 16维
                    - ["Left_arm", "Right_arm", "Left_claw", "Right_claw", "Cmd_pose_z", "Cmd_pose_pitch"]: 18维
            control_arm: 是否控制手臂
            control_claw: 是否控制夹爪
            control_cmd_pose: 是否控制cmd_pose
        """
        actions = np.asarray(actions)

        # 解析动作数组，根据ACTION_COMPONENTS动态提取
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        
        action_idx = 0
        
        # 提取左臂动作（7维）
        if "Left_arm" in ACTION_COMPONENTS:
            left_arm_action = actions[0, action_idx:action_idx+7]
            action_idx += 7
        else:
            left_arm_action = None
            
        # 提取右臂动作（7维）
        if "Right_arm" in ACTION_COMPONENTS:
            right_arm_action = actions[0, action_idx:action_idx+7]
            action_idx += 7
        else:
            right_arm_action = None
            
        # 组合左右臂动作
        if left_arm_action is not None and right_arm_action is not None:
            arm_action = np.concatenate([left_arm_action, right_arm_action])
        elif left_arm_action is not None:
            arm_action = left_arm_action
        elif right_arm_action is not None:
            arm_action = right_arm_action
        else:
            arm_action = None
            
        # 提取左爪动作（1维）
        if "Left_claw" in ACTION_COMPONENTS:
            left_claw_action = actions[0, action_idx]
            action_idx += 1
        else:
            left_claw_action = None
            
        # 提取右爪动作（1维）
        if "Right_claw" in ACTION_COMPONENTS:
            right_claw_action = actions[0, action_idx]
            action_idx += 1
        else:
            right_claw_action = None
            
        # 组合左右爪动作
        if left_claw_action is not None and right_claw_action is not None:
            claw_action = np.array([left_claw_action, right_claw_action])
        elif left_claw_action is not None:
            claw_action = np.array([left_claw_action, 0.0])
        elif right_claw_action is not None:
            claw_action = np.array([0.0, right_claw_action])
        else:
            claw_action = None

        # 提取cmd_pose_z（1维）
        if "Cmd_pose_z" in ACTION_COMPONENTS:
            cmd_pose_z = actions[0, action_idx]
            action_idx += 1
        else:
            cmd_pose_z = None
            
        # 提取cmd_pose_pitch（1维）
        if "Cmd_pose_pitch" in ACTION_COMPONENTS:
            cmd_pose_pitch = actions[0, action_idx]
            action_idx += 1
        else:
            cmd_pose_pitch = None

        # 发布手臂和夹爪动作
        if arm_action is not None and claw_action is not None:
            # clip 小臂 pitch
            if len(arm_action) >= 14:
                arm_action[3] = np.clip(arm_action[3], np.deg2rad(-130), np.deg2rad(0.0))
                arm_action[10] = np.clip(arm_action[10], np.deg2rad(-130), np.deg2rad(0.0))
            
            self.target_publisher.publish_target_arm_claw(
                arm_action=arm_action,
                claw_action=claw_action,
                control_arm=control_arm,
                control_claw=control_claw
            )

        # 发布cmd_pose动作
        if (cmd_pose_z is not None or cmd_pose_pitch is not None) and control_cmd_pose:
            if self.target_publisher.cmd_pose_publisher is not None:
                from geometry_msgs.msg import Twist
                cmd_pose_msg = Twist()
                if cmd_pose_z is not None:
                    cmd_pose_msg.linear.z = float(cmd_pose_z)
                if cmd_pose_pitch is not None:
                    cmd_pose_msg.angular.y = float(cmd_pose_pitch)
                self.target_publisher.cmd_pose_publisher.publish(cmd_pose_msg)

        # 执行完动作之后，在这里控制时间
        dt = self.control_dt
        duration = time.time() - self.last_action_exec_time
        time_to_sleep = max(0, dt - duration)
        time.sleep(time_to_sleep)
        self.last_action_exec_time = time.time()


