from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.robot_sdk import RobotSDK
from kuavo_humanoid_sdk.kuavo_strategy_pytree.nodes.api import ArmAPI, TorsoAPI
from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.data_type import Pose, Tag, Frame, Transform3D
from kuavo_humanoid_sdk.kuavo_strategy_pytree.nodes.utils import generate_full_bezier_trajectory

import py_trees
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from typing import List
import time
import numpy as np


def generate_arm_traj(arm_api, left_arm_keypoints, right_arm_keypoints):
    left_targets = []
    right_targets = []

    for left_key_pose, right_key_pose in zip(left_arm_keypoints, right_arm_keypoints):
        assert left_key_pose.frame in [Frame.ODOM, Frame.BASE, Frame.TAG], \
            print(
                "在全局控制模式下，left_key_pose.frame must be Frame.ODOM, Frame.BASE or Frame.TAG")
        assert right_key_pose.frame in [Frame.ODOM, Frame.BASE, Frame.TAG], \
            print(
                "在全局控制模式下，right_key_pose.frame must be Frame.ODOM, Frame.BASE or Frame.TAG")
        if Frame.ODOM == left_key_pose.frame:

            transform_base_to_world = arm_api.get_current_transform(source_frame=Frame.ODOM,
                                                                            target_frame=Frame.BASE)
            left_targets.append(transform_base_to_world.apply_to_pose(left_key_pose))
            right_targets.append(transform_base_to_world.apply_to_pose(right_key_pose))

        elif Frame.BASE == left_key_pose.frame:
            left_targets.append(left_key_pose)
            right_targets.append(right_key_pose)


    left_eef_pose_world, right_eef_pose_world = arm_api.get_eef_pose_world()
    left_bezier_trajectory, right_bezier_trajectory = generate_full_bezier_trajectory(
        current_left_pose=left_eef_pose_world,
        current_right_pose=right_eef_pose_world,
        left_keypoints_list=left_targets,
        right_keypoints_list=right_targets,
    )
    return left_bezier_trajectory, right_bezier_trajectory


def transform_tag(latest_tag):
    # ==================== 把 latest_tag 转成y轴向上， z轴向机器人的 ===========
    new_tag_in_real_tag = Pose.from_euler(
        pos=(0, 0, 0),  # 最新标签的位置，单位米
        euler=(90, 0, 0),  # 45, 60
        # euler=(0, 0, 180),  # 49
        degrees=True,  # 使用度数表示
        frame=Frame.TAG  # 使用标签坐标系
    )

    transform_tag_to_odom = Transform3D(
        trans_pose=latest_tag.pose,
        source_frame=Frame.TAG,
        target_frame=Frame.ODOM,
    )

    return transform_tag_to_odom.apply_to_pose(new_tag_in_real_tag)


def transform_pose_from_tag_to_world(tag: Tag, pose: Pose) -> Pose:
    """
    将tag坐标系下的位姿转换到世界坐标系下。

    参数：
        tag (Tag): Tag对象，包含位姿信息。
        pose (Pose): 需要转换的位姿。

    返回：
        Pose: 转换后的Pose对象。
    """
    # 转换stand_pose_in_tag到世界坐标系。注意、需要搞清楚tag的坐标定义和机器人的坐标定义
    transform_tag_to_world = Transform3D(
        trans_pose=tag.pose,
        source_frame=Frame.TAG,  # 源坐标系为Tag坐标系
        target_frame=Frame.ODOM  # 目标坐标系为里程计坐标系
    )
    stand_pose_in_world = transform_tag_to_world.apply_to_pose(
        pose  # 将站立位置转换到里程计坐标系
    )
    return stand_pose_in_world


class NodePercep(Behaviour):
    def __init__(self, name,
                 robot_sdk: RobotSDK,
                 tags_id: List,
                 tag_up_axis: str,
                 ):
        super(NodePercep, self).__init__(name)
        self.bb = py_trees.blackboard.Client(name=self.name)

        for k in [f'latest_tag_{tag_id}' for tag_id in tags_id] + [f'latest_tag_{tag_id}_version' for tag_id in
                                                                   tags_id]:
            self.bb.register_key(key=k, access=py_trees.common.Access.WRITE)

        self.robot_sdk = robot_sdk
        self.tags_id = tags_id
        self.tag_up_axis = tag_up_axis

    def initialise(self):
        self.logger.debug(f"NodePercep::initialise {self.name}")
        # for tag_id in self.tags_id:
        #     setattr(self.bb, f"latest_tag_{tag_id}_version", 0)
        for tag_id in self.tags_id:
            # 初始化 tag 数据占位
            setattr(self.bb, f"latest_tag_{tag_id}", None)  # tag 本身
            setattr(self.bb, f"latest_tag_{tag_id}_version", 0)  # 版本号

    def update(self):
        self.logger.debug(f"NodePercep::update {self.name}")
        tag_pose_filter = TagPoseFilter(self.tag_up_axis)
        for tag_id in self.tags_id:
            target_data = self.robot_sdk.vision.get_data_by_id_from_odom(tag_id)
            if target_data is not None:
                tag_pose = target_data["poses"][0]  # 获取第一个tag的位姿
                pos, quat = tag_pose_filter.tag_pose_filter(tag_pose)
                latest_tag = Tag(
                    id=tag_id,
                    pose=Pose(
                        pos=pos,
                        quat=quat,
                        frame=Frame.ODOM
                    )
                )
                setattr(self.bb, f"latest_tag_{tag_id}", latest_tag)
                current_version = getattr(self.bb, f"latest_tag_{tag_id}_version", 0)
                setattr(self.bb, f"latest_tag_{tag_id}_version", current_version + 1)
        return Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug(f"NodePercep::terminate {self.name} to {new_status}")


class TagPoseFilter:
    _AXIS_FIX = {
        "x": (0.0, -90.0, None),
        "y": (90.0, 0.0, None),
        "z": (0.0, 0.0, None),
    }

    def __init__(self, tag_up_axis: str):
        self.tag_up_axis = tag_up_axis

    def tag_pose_filter(self, tag_pose):
        """返回修正后的 tag 位姿: (pos, quat)"""
        quat = [tag_pose.orientation.x,
                tag_pose.orientation.y,
                tag_pose.orientation.z,
                tag_pose.orientation.w]
        euler = self._quat2euler(quat, degrees=True)

        # 根据 up_axis 修正欧拉角
        if self.tag_up_axis in self._AXIS_FIX:
            roll_fix, pitch_fix, yaw_fix = self._AXIS_FIX[self.tag_up_axis]
            euler[0] = roll_fix if roll_fix is not None else euler[0]
            euler[1] = pitch_fix if pitch_fix is not None else euler[1]
            euler[2] = yaw_fix if yaw_fix is not None else euler[2]

        quat_fixed = self._euler2quat(euler, degrees=True)
        pos = (tag_pose.position.x, tag_pose.position.y, tag_pose.position.z)
        return pos, tuple(quat_fixed)

    def _quat2euler(self, quat, degrees=False):
        x, y, z, w = quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        euler = np.array([roll, pitch, yaw])
        return np.degrees(euler) if degrees else euler

    def _euler2quat(self, euler, degrees=False):
        roll, pitch, yaw = np.radians(euler) if degrees else euler
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        return np.array([x, y, z, w])


class NodeWaitForBlackboards(py_trees.behaviour.Behaviour):
    def __init__(self, keys, name=None,
                 timeout: float = None):
        super().__init__(name or f"WaitForAny({','.join(keys)})")
        self.keys = keys
        self.timeout = timeout
        self.start_t = None
        self.bb = py_trees.blackboard.Client(name=f"{self.name}/reader")
        for k in self.keys:
            self.bb.register_key(key=k, access=py_trees.common.Access.READ)

    def initialise(self):
        self.start_t = time.time()

    def update(self):
        # 遍历 keys，只要有一个可用就成功
        for k in self.keys:
            try:
                val = getattr(self.bb, k)
                if val is not None:
                    return Status.SUCCESS
            except KeyError:
                continue  # key 不存在，继续检查其他的

        # 超时就失败；否则继续等待
        if self.timeout is not None and (time.time() - self.start_t) > self.timeout:
            return Status.FAILURE
        return Status.RUNNING


class NodeTagToTorsoGoal(Behaviour):
    def __init__(self, name, torso_api, tag_id, delta_z):
        super(NodeTagToTorsoGoal, self).__init__(name)
        self.bb = py_trees.blackboard.Client(name='search_pick_tag_TAG2GOAL')
        for k in [f'latest_tag_{tag_id}', f'latest_tag_{tag_id}_version']:
            self.bb.register_key(key=k, access=py_trees.common.Access.READ)

        for k in ['torso_goal']:
            self.bb.register_key(key=k, access=py_trees.common.Access.WRITE)

        self.torso_api = torso_api
        self.tag_id = tag_id
        self.tag_version = -1
        self.delta_z = delta_z

    def initialise(self):
        self.logger.debug(f"NodeTagToTorsoGoal::initialise {self.name}")
        self.tag_version = -1

    def update(self):
        self.logger.debug(f"NodeTagToTorsoGoal::update {self.name}")
        latest_tag = getattr(self.bb, f"latest_tag_{self.tag_id}", None)
        tag_version = getattr(self.bb, f"latest_tag_{self.tag_id}_version", None)

        new_tag_pose = transform_tag(latest_tag)
        latest_tag.pose = new_tag_pose
        if latest_tag is None or tag_version is None:
            return Status.RUNNING

        elif tag_version <= self.tag_version:
            return Status.RUNNING

        lower_limit = max(0.6, latest_tag.pose.pos[2] + 0.1)
        upper_limit = min(1.1, latest_tag.pose.pos[2] + self.delta_z)

        torso_goal = Pose.from_euler(
            pos=(0, 0, np.random.uniform(lower_limit, upper_limit)),
            euler=(0, 0, 0),
            degrees=True,
            frame=Frame.ODOM
        )
        self.bb.torso_goal = torso_goal

        self.tag_version = getattr(self.bb, f"latest_tag_{self.tag_id}_version", 0)
        return Status.SUCCESS


class NodeArmGoal(Behaviour):
    def __init__(self, 
                 name, 
                 arm_api, 
                 left_arm_keypoints, 
                 right_arm_keypoints,
                 is_legged: bool = False,
                ):
        super().__init__(name)
        self.bb = py_trees.blackboard.Client(name=name)
        self.bb.register_key(key="left_arm_eef_traj", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="right_arm_eef_traj", access=py_trees.common.Access.WRITE)
        self.arm_api = arm_api
        self.left_arm_keypoints = left_arm_keypoints
        self.right_arm_keypoints = right_arm_keypoints
        self.is_legged = is_legged

    def initialise(self):
        self.logger.debug(f"NodeArmGoal::initialise {self.name}")

    def update(self):
        self.logger.debug(f"NodeArmGoal::update {self.name}")

        left_targets = []
        right_targets = []

        # print('===== Generating arm trajectory based on keypoints')
        for left_key_pose, right_key_pose in zip(self.left_arm_keypoints, self.right_arm_keypoints):
            assert left_key_pose.frame in [Frame.ODOM, Frame.BASE, Frame.TAG, Frame.WAIST_YAW_LINK], \
                self.logger.error(
                    "在全局控制模式下，left_key_pose.frame must be Frame.ODOM, Frame.BASE or Frame.TAG")
            assert right_key_pose.frame in [Frame.ODOM, Frame.BASE, Frame.TAG, Frame.WAIST_YAW_LINK], \
                self.logger.error(
                    "在全局控制模式下，right_key_pose.frame must be Frame.ODOM, Frame.BASE or Frame.TAG")
            if Frame.ODOM == left_key_pose.frame:
                left_targets.append(left_key_pose)
                right_targets.append(right_key_pose)
            else:
                transform_base_to_world = self.arm_api.get_current_transform(source_frame=Frame.BASE if self.is_legged else Frame.WAIST_YAW_LINK,
                                                                            target_frame=Frame.ODOM)
                left_targets.append(transform_base_to_world.apply_to_pose(left_key_pose))
                right_targets.append(transform_base_to_world.apply_to_pose(right_key_pose))

        left_eef_pose_world, right_eef_pose_world = self.arm_api.get_eef_pose_world()
        left_bezier_trajectory, right_bezier_trajectory = generate_full_bezier_trajectory(
            current_left_pose=left_eef_pose_world,
            current_right_pose=right_eef_pose_world,
            left_keypoints_list=left_targets,
            right_keypoints_list=right_targets,
        )

        self.bb.left_arm_eef_traj = left_bezier_trajectory
        self.bb.right_arm_eef_traj = right_bezier_trajectory

        return Status.SUCCESS


class NodeTagToArmGoal(Behaviour):
    def __init__(self,
                 name,
                 arm_api: ArmAPI,
                 tag_id: int,
                 left_arm_relative_keypoints,
                 right_arm_relative_keypoints,
                 ):
        super(NodeTagToArmGoal, self).__init__(name)
        self.bb = py_trees.blackboard.Client(name='search_pick_tag_TAG2GOAL')

        # 只读
        for k in [f'latest_tag_{tag_id}', f'latest_tag_{tag_id}_version']:
            self.bb.register_key(key=k, access=py_trees.common.Access.READ)
        # 读写
        for k in ['left_arm_eef_traj', 'right_arm_eef_traj']:
            self.bb.register_key(key=k, access=py_trees.common.Access.WRITE)

        self.arm_api = arm_api
        self.tag_id = tag_id
        self.tag_version = -1

        self.left_arm_relative_keypoints = left_arm_relative_keypoints
        self.right_arm_relative_keypoints = right_arm_relative_keypoints

    def initialise(self):
        self.logger.debug(f"NodeTagToArmGoal::initialise {self.name}")
        self.tag_version = -1

    def update(self):
        self.logger.debug(f"NodeTagToArmGoal::update {self.name}")
        latest_tag = getattr(self.bb, f"latest_tag_{self.tag_id}", None)
        tag_version = getattr(self.bb, f"latest_tag_{self.tag_id}_version", None)

        new_tag_pose = transform_tag(latest_tag)
        latest_tag.pose = new_tag_pose
        if latest_tag is None or tag_version is None:
            return Status.RUNNING

        elif tag_version <= self.tag_version:
            return Status.RUNNING

        left_targets = []
        right_targets = []

        if len(self.right_arm_relative_keypoints) == 0:
            for left_key_pose in self.left_arm_relative_keypoints:
                assert left_key_pose.frame in [Frame.ODOM, Frame.BASE, Frame.TAG], \
                    self.logger.error(
                        "在全局控制模式下，left_key_pose.frame must be Frame.ODOM, Frame.BASE or Frame.TAG")
                if Frame.ODOM == left_key_pose.frame:

                    transform_world_to_base = self.arm_api.get_current_transform(source_frame=Frame.ODOM,
                                                                                    target_frame=Frame.BASE)
                    left_targets.append(transform_world_to_base.apply_to_pose(left_key_pose))

                elif Frame.BASE == left_key_pose.frame:
                    left_targets.append(left_key_pose)

                elif Frame.TAG == left_key_pose.frame:
                    tag = latest_tag
                    # print(f"tag.pose: {tag.pose}")
                    # print(f"left_key_pose: {left_key_pose}")
                    transform_source_to_target = Transform3D(
                        trans_pose=tag.pose,
                        source_frame=Frame.TAG,  # 源坐标系为Tag坐标系
                        target_frame=Frame.ODOM  # 目标坐标系为里程计坐标系
                    )
                    left_targets.append(transform_source_to_target.apply_to_pose(left_key_pose))

            left_eef_pose_world, right_eef_pose_world = self.arm_api.get_eef_pose_world()
            for i in range(len(left_targets)):
                right_targets.append(right_eef_pose_world)

        elif len(self.left_arm_relative_keypoints) == 0:
            for right_key_pose in self.right_arm_relative_keypoints:
                assert right_key_pose.frame in [Frame.ODOM, Frame.BASE, Frame.TAG], \
                    self.logger.error(
                        "在全局控制模式下，right_key_pose.frame must be Frame.ODOM, Frame.BASE or Frame.TAG")
                if Frame.ODOM == right_key_pose.frame:

                    transform_world_to_base = self.arm_api.get_current_transform(source_frame=Frame.ODOM,
                                                                                    target_frame=Frame.BASE)
                    right_targets.append(transform_world_to_base.apply_to_pose(right_key_pose))

                elif Frame.BASE == right_key_pose.frame:
                    right_targets.append(right_key_pose)

                elif Frame.TAG == right_key_pose.frame:
                    tag = latest_tag
                    transform_source_to_target = Transform3D(
                        trans_pose=tag.pose,
                        source_frame=Frame.TAG,  # 源坐标系为Tag坐标系
                        target_frame=Frame.ODOM  # 目标坐标系为里程计坐标系
                    )

                    right_targets.append(transform_source_to_target.apply_to_pose(right_key_pose))

            left_eef_pose_world, right_eef_pose_world = self.arm_api.get_eef_pose_world()
            for i in range(len(right_targets)):
                left_targets.append(left_eef_pose_world)

        else:
            for left_key_pose, right_key_pose in zip(self.left_arm_relative_keypoints, self.right_arm_relative_keypoints):
                assert left_key_pose.frame in [Frame.ODOM, Frame.BASE, Frame.TAG], \
                    self.logger.error(
                        "在全局控制模式下，left_key_pose.frame must be Frame.ODOM, Frame.BASE or Frame.TAG")
                assert right_key_pose.frame in [Frame.ODOM, Frame.BASE, Frame.TAG], \
                    self.logger.error(
                        "在全局控制模式下，right_key_pose.frame must be Frame.ODOM, Frame.BASE or Frame.TAG")
                if Frame.ODOM == left_key_pose.frame:

                    transform_world_to_base = self.arm_api.get_current_transform(source_frame=Frame.ODOM,
                                                                                target_frame=Frame.BASE)
                    left_targets.append(transform_world_to_base.apply_to_pose(left_key_pose))
                    right_targets.append(transform_world_to_base.apply_to_pose(right_key_pose))

                elif Frame.BASE == left_key_pose.frame:
                    left_targets.append(left_key_pose)
                    right_targets.append(right_key_pose)

                elif Frame.TAG == left_key_pose.frame:
                    tag = latest_tag
                    transform_source_to_target = Transform3D(
                        trans_pose=tag.pose,
                        source_frame=Frame.TAG,  # 源坐标系为Tag坐标系
                        target_frame=Frame.ODOM  # 目标坐标系为里程计坐标系
                    )

                    left_targets.append(transform_source_to_target.apply_to_pose(left_key_pose))
                    right_targets.append(transform_source_to_target.apply_to_pose(right_key_pose))

            left_eef_pose_world, right_eef_pose_world = self.arm_api.get_eef_pose_world()
        
        left_bezier_trajectory, right_bezier_trajectory = generate_full_bezier_trajectory(
            current_left_pose=left_eef_pose_world,
            current_right_pose=right_eef_pose_world,
            left_keypoints_list=left_targets,
            right_keypoints_list=right_targets,
        )
    

        self.bb.left_arm_eef_traj = left_bezier_trajectory
        self.bb.right_arm_eef_traj = right_bezier_trajectory

        self.tag_version = getattr(self.bb, f"latest_tag_{self.tag_id}_version", 0)
        return Status.SUCCESS

    def terminate(self, new_status):
        self.logger.debug(f"NodeTagToArmGoal::terminate {self.name} to {new_status}")


class NodeFuntion(py_trees.behaviour.Behaviour):
    """
    把一个函数快速包装成一个行为节点
    """

    def __init__(self, fn, name=None):
        super().__init__(name or fn.__name__)
        self.fn = fn

    def update(self):
        # 执行函数，返回值转换成 py_trees 的 Status
        result = self.fn()
        if result is True:
            return Status.SUCCESS
        elif result is False:
            return Status.FAILURE
        else:
            # 如果函数不返回布尔，可以自己定义约定
            return Status.RUNNING


class NodeArm(Behaviour):
    def __init__(self, name, arm_api: ArmAPI, use_ik: bool = True, is_legged: bool = True):
        super(NodeArm, self).__init__(name)
        self.bb = py_trees.blackboard.Client(name=name)
        # 从白板拿到手臂目标轨迹
        for k in ['left_arm_eef_traj', 'right_arm_eef_traj']:
            self.bb.register_key(key=k, access=py_trees.common.Access.READ)

        self.arm_api = arm_api
        self.use_ik = use_ik
        self.is_legged = is_legged

    def initialise(self):
        self.logger.debug(f"NodeArm::initialise {self.name}")
        left_traj = getattr(self.bb, "left_arm_eef_traj", None)
        right_traj = getattr(self.bb, "right_arm_eef_traj", None)
        if left_traj is None or right_traj is None:
            self.logger.error(f"NodeArm::update {self.name} - No arm trajectory found on blackboard")
            return Status.FAILURE

        if self.use_ik:
            # import rospy
            # rospy.loginfo(f"{self.name}: 使用IK轨迹控制")

            # 转换KuavoPose到Pose（如果需要）
            def convert_to_pose(traj):
                """将KuavoPose列表转换为Pose列表"""
                from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.data_type import Pose, Frame
                import numpy as np
                result = []
                for item in traj:
                    if hasattr(item, 'pos'):  # 已经是Pose对象
                        if item.frame == Frame.ODOM:
                            if self.is_legged:
                                transform_odom_to_base = self.arm_api.get_current_transform(source_frame=Frame.ODOM, 
                                                                                            target_frame=Frame.BASE)
                            else:
                                transform_odom_to_base = self.arm_api.get_current_transform(source_frame=Frame.ODOM, 
                                                                                            target_frame=Frame.WAIST_YAW_LINK)
                            result.append(transform_odom_to_base.apply_to_pose(item))
                        else:
                            result.append(item)
                    elif hasattr(item, 'position'):  # 是KuavoPose对象
                        item_original = Pose(
                            pos=item.position,
                            quat=np.array(item.orientation) if isinstance(item.orientation, list) else item.orientation,
                            frame=item.frame
                        )
                        if item_original.frame == Frame.ODOM:
                            if self.is_legged:
                                transform_odom_to_base = self.arm_api.get_current_transform(source_frame=Frame.ODOM, 
                                                                                            target_frame=Frame.BASE)
                            else:
                                transform_odom_to_base = self.arm_api.get_current_transform(source_frame=Frame.ODOM, 
                                                                                            target_frame=Frame.WAIST_YAW_LINK)
                            result.append(transform_odom_to_base.apply_to_pose(item))
                        else:
                            result.append(item_original)
                    else:
                        rospy.logerr(f"Unknown pose type: {type(item)}")
                return result

            left_traj_converted = convert_to_pose(left_traj)
            right_traj_converted = convert_to_pose(right_traj)
            
            self.fut = self.arm_api.move_eef_traj_ik(
                left_traj=left_traj_converted,
                right_traj=right_traj_converted,
                asynchronous=True,
                total_time=1,
                interpolation=True,
                validate_keypoints=False
            )
        else:
            raise NotImplementedError("MPC trajectory control is not supported. Use use_ik=True.")

    def update(self):
        self.logger.debug(f"NodeArm::update {self.name}")

        if not self.fut.done():
            time.sleep(0.01)
            return Status.RUNNING

        # 检查IK执行结果
        if self.use_ik:
            import rospy
            try:
                result = self.fut.result()  # 获取Future返回值
                if result == False:
                    rospy.logerr(f"{self.name}: I轨迹执行失败（关键点IK失败或成功率<80%）")
                    return Status.FAILURE
            except Exception as e:
                rospy.logerr(f"{self.name}: IK轨迹执行异常: {e}")
                return Status.FAILURE
        
        return Status.SUCCESS

    def terminate(self, new_status):
        self.logger.debug(f"NodeArm::terminate {self.name} to {new_status}")


class NodeTorso(Behaviour):
    def __init__(self, name, torso_api: TorsoAPI):
        super(NodeTorso, self).__init__(name)
        self.bb = py_trees.blackboard.Client(name=self.name)
        for k in ['torso_goal']:
            self.bb.register_key(key=k, access=py_trees.common.Access.WRITE)

        self.torso_api = torso_api

    def initialise(self):
        self.logger.debug(f"NodeTorso::initialise {self.name}")

    def update(self):
        self.logger.debug(f"NodeTorso::update {self.name}")
        torso_goal = getattr(self.bb, "torso_goal", None)

        if torso_goal is None:
            self.logger.error(f"NodeTorso::update {self.name} - No torso goal found on blackboard")
            return Status.FAILURE

        self.torso_api.move_torso_pose(desir_torso_pose=torso_goal, asynchronous=True)
        # print(f"NodeTorso::update {self.name} - torso goal: {torso_goal}")
        return Status.SUCCESS

    def terminate(self, new_status):
        self.logger.debug(f"NodeTorso::terminate {self.name} to {new_status}")