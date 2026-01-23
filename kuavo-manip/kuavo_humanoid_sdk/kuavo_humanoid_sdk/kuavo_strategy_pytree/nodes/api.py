from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.robot_sdk import RobotSDK
from kuavo_humanoid_sdk.kuavo_strategy_pytree.utils.utils import normalize_angle
from kuavo_humanoid_sdk.kuavo.core.ros.control import KuavoManipulationMpcCtrlMode
from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.data_type import Pose, Frame, Transform3D

from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Tuple
import time
import numpy as np
from kuavo_ik.ik_library import IKAnalytical


class HeadAPI:
    """
    头部控制API
    """

    def __init__(self, robot_sdk: RobotSDK):
        self.robot_sdk = robot_sdk
        self._pool = ThreadPoolExecutor(max_workers=2)

    def _move_head_traj(self,
                        head_traj: List[Tuple[float, float]] = [],  # 头部目标点列表，格式为[(yaw, pitch), ...
                        ):
        for pair in head_traj:
            yaw, pitch = pair
            result = self.robot_sdk.control.control_head(
                yaw, pitch)
            time.sleep(0.7)
        return result

    def move_head_traj(self,
                       head_traj: List[Tuple[float, float]] = [],  # 头部目标点列表，格式为[(yaw, pitch), ...
                       asynchronous: bool = False,  # 布尔值，指定运动命令是否为异步。默认值为 false，表示函数会阻塞
                       ) -> Future:
        if asynchronous:
            fut = self._pool.submit(self._move_head_traj, head_traj)
            return fut  # 外部拿到 Future

        else:
            result = self._move_head_traj(head_traj)
            return result  # 返回布尔值，True表示成功，False表示失败


class ArmAPI:
    """
    根据手臂和躯干控制接口，封装手臂控制的API
    """

    def __init__(self, robot_sdk: RobotSDK, is_legged: bool = False):
        self.robot_sdk = robot_sdk
        self._pool = ThreadPoolExecutor(max_workers=2)
        self.analytical_ik_solver = IKAnalytical()

        self.is_legged = is_legged

        if self.is_legged:
            self.robot_sdk.control.set_external_control_arm_mode()
        else:
            self.robot_sdk.control.set_arm_only_mode()
        # self.last_delta_q = [None, None]

    def move_eef_pose_ik(self, eef_target_left=None, eef_target_right=None):
        """
        通过ik来控制joint
        """
        # 不能同时为None
        assert eef_target_left is not None or eef_target_right is not None, "eef_target_left和eef_target_right不能同时为None"
        model_type = '45' if self.is_legged else '60'

        target_q_14 = self.get_joint_positions().position
        
        LOWER = np.deg2rad(-110)
        UPPER = np.deg2rad(60)

        if eef_target_left is not None:
            target_q = self.analytical_ik_solver.compute(eef_target_left[:3], eef_target_left[3:7], eef_frame='zarm_l7_link', model_type=model_type)
            # delta_q = target_q - target_q_14[:7]
            # if self.last_delta_q[0] is None:
            #     self.last_delta_q[0] = delta_q[-1]
            # assert ((abs(delta_q[-1] - self.last_delta_q[0]) < np.deg2rad(40)) 
            #         and 
            #         (abs(delta_q[-1] < np.deg2rad(70)) and abs(delta_q[-3] < np.deg2rad(70)))
            #     )
            # self.last_delta_q[0] = delta_q[-1]
            target_q[-1] = np.clip(target_q[-1], LOWER, UPPER)
            target_q_14[:7] = target_q

        if eef_target_right is not None:
            target_q = self.analytical_ik_solver.compute(eef_target_right[:3], eef_target_right[3:7], eef_frame='zarm_r7_link', model_type=model_type)
            # delta_q = target_q - target_q_14[7:14]
            # if self.last_delta_q[1] is None:
            #     self.last_delta_q[1] = delta_q[-1]
            # assert ((abs(delta_q[-1] - self.last_delta_q[1]) < np.deg2rad(40)) and (abs(delta_q[-1] < np.deg2rad(70)) and abs(delta_q[-3] < np.deg2rad(70))))
            # self.last_delta_q[1] = delta_q[-1]
            target_q[-1] = np.clip(target_q[-1], LOWER, UPPER)
            target_q_14[7:14] = target_q

        return self.move_joint_pos(target_q_14)


    def move_joint_pos(self,
                        target_q
                        ):

        res = self.robot_sdk.control.control_arm_joint_positions(joint_positions=target_q)
        return res
    

    def _interpolate_trajectory(self, traj: List[List[float]], dt=0.1) -> List[List[float]]:
        """
        轨迹插值（线性位置 + SLERP四元数）
        
        Args:
            traj: 关键点列表 [[x,y,z,qx,qy,qz,qw], ...]
            dt: 插值时间步长（秒，未使用，保留参数兼容性）
        
        Returns:
            插值后的密集轨迹
        """
        if len(traj) < 2:
            return traj
        
        from scipy.spatial.transform import Rotation, Slerp
        
        dense_traj = []
        for i in range(len(traj) - 1):
            p1 = np.array(traj[i])
            p2 = np.array(traj[i+1])
            
            # 计算两点距离
            distance = np.linalg.norm(p2[:3] - p1[:3])
            
            # 处理重合点
            if distance < 1e-6:
                dense_traj.append(traj[i])
                continue
            
            # 自适应插值密度（每2cm一个点）
            n_steps = max(2, int(distance / 0.01))
            n_steps = min(n_steps, 300)  # 限制最大步数
            
            # 四元数反向检查（选择短路径）
            q1 = p1[3:7]
            q2 = p2[3:7]
            dot = np.dot(q1, q2)
            if dot < 0:
                q2 = -q2  # 反转四元数（表示相同姿态）
            
            # 插值
            for j in range(n_steps):
                alpha = j / n_steps
                
                # 位置线性插值
                pos = p1[:3] + alpha * (p2[:3] - p1[:3])
                
                # 四元数SLERP插值
                r1 = Rotation.from_quat(q1)
                r2 = Rotation.from_quat(q2)
                slerp = Slerp([0, 1], Rotation.concatenate([r1, r2]))
                quat = slerp(alpha).as_quat()
                
                dense_traj.append(np.concatenate([pos, quat]).tolist())
        
        # 添加最后一个点
        dense_traj.append(traj[-1])
        return dense_traj


    def _move_eef_traj_ik(self,
                          left_traj: List[List[float]],
                          right_traj: List[List[float]],
                          total_time: float,
                          interpolation: bool,
                          validate_keypoints: bool):
        """
        IK轨迹执行（内部方法，方案C：分级策略）
        
        Args:
            left_traj: 左手轨迹（Pose对象列表，需转换）
            right_traj: 右手轨迹（Pose对象列表，需转换）
            total_time: 总执行时间
            interpolation: 是否插值
            validate_keypoints: 是否先验证关键点
        
        Returns:
            bool: True=成功, False=失败
        """
        import rospy
        
        # 1. 转换Pose对象为list格式
        def pose_to_list(pose):
            """Pose对象 → [x,y,z,qx,qy,qz,qw]"""
            return list(pose.pos[:3]) + list(pose.quat[:4])
        
        left_traj_list = [pose_to_list(p) for p in left_traj] if left_traj else []
        right_traj_list = [pose_to_list(p) for p in right_traj] if right_traj else []
        
        # 2. 【方案C关键】先验证原始关键点（必须全部成功）
        if validate_keypoints:
            rospy.loginfo("=== 验证原始关键点 ===")
            for i in range(max(len(left_traj_list), len(right_traj_list))):
                left_p = left_traj_list[min(i, len(left_traj_list)-1)] if len(left_traj_list) > 0 else None
                right_p = right_traj_list[min(i, len(right_traj_list)-1)] if len(right_traj_list) > 0 else None
                
                success = self.move_eef_pose_ik(
                    eef_target_left=left_p,
                    eef_target_right=right_p
                )
                
                if not success:
                    rospy.logerr(f"关键点 {i} IK失败，中止轨迹！")
                    return False  # 关键点失败，立即中断
                
                rospy.loginfo(f"关键点 {i} 验证通过")
            
            rospy.loginfo("所有关键点IK验证通过\n")
        
        # 3. 执行插值轨迹（允许部分失败）
        if interpolation:
            # rospy.loginfo("=== 执行插值轨迹 ===")
            left_dense = self._interpolate_trajectory(left_traj_list) if len(left_traj_list) > 0 else []
            right_dense = self._interpolate_trajectory(right_traj_list) if len(right_traj_list) > 0 else []
        else:
            left_dense = left_traj_list
            right_dense = right_traj_list
        
        # 计算时间步长
        num_points = max(len(left_dense), len(right_dense))
        if num_points == 0:
            return True
        
        dt = total_time / num_points

        # 逐点执行IK（插值点允许失败）
        success_count = 0
        skip_count = 0
        
        for i in range(num_points):
            left_p = left_dense[min(i, len(left_dense)-1)] if len(left_dense) > 0 else None
            right_p = right_dense[min(i, len(right_dense)-1)] if len(right_dense) > 0 else None
            
            success = self.move_eef_pose_ik(
                eef_target_left=left_p,
                eef_target_right=right_p
            )
            
            if success:
                success_count += 1
            else:
                skip_count += 1
                rospy.logwarn(f"跳过IK失败的插值点 {i}/{num_points}")
            
            time.sleep(dt)
        
        # 检查成功率
        success_rate = success_count / num_points
        # rospy.loginfo(f"\n=== IK轨迹执行完成 ===")
        # rospy.loginfo(f"总点数: {num_points}, 成功: {success_count}, 跳过: {skip_count}")
        # rospy.loginfo(f"成功率: {success_rate*100:.1f}%")
        
        # 【方案C关键】插值点失败率阈值检查
        if success_rate < 0.8:  # 成功率低于80%
            rospy.logerr(f"插值点成功率过低 ({success_rate*100:.1f}%), 轨迹质量不佳")
            return False
        
        return True


    def move_eef_traj_ik(self,
                         left_traj: List = None,
                         right_traj: List = None,
                         asynchronous: bool = False,
                         total_time: float = 2.0,
                         interpolation: bool = True,
                         validate_keypoints: bool = True):
        """
        IK轨迹跟踪（公开接口，替代MPC版本）
        
        Args:
            left_traj: 左手轨迹（Pose对象列表）
            right_traj: 右手轨迹（Pose对象列表）
            asynchronous: 是否异步执行
            total_time: 总执行时间（秒）
            interpolation: 是否插值（True=平滑运动）
            validate_keypoints: 是否先验证关键点（True=方案C，False=直接执行）
        
        Returns:
            Future | None: 异步返回Future，同步返回None
        """
        if asynchronous:
            # 异步执行
            fut = self._pool.submit(
                self._move_eef_traj_ik,
                left_traj, right_traj, total_time, interpolation, validate_keypoints
            )
            return fut
        else:
            # 同步执行
            self._move_eef_traj_ik(
                left_traj, right_traj, total_time, interpolation, validate_keypoints
            )
            return None


    def get_eef_pose_world(self, frame: Frame = Frame.ODOM, eef: bool = False):
        target_frame = frame
        if eef:
            left_pose = self.robot_sdk.tools.get_link_pose(
                link_name="zarm_l7_end_effector",
                reference_frame=target_frame
            )
            right_pose = self.robot_sdk.tools.get_link_pose(
                link_name="zarm_r7_end_effector",
                reference_frame=target_frame
            )
        else:
            left_pose = self.robot_sdk.tools.get_link_pose(
                link_name="zarm_l7_link",
                reference_frame=target_frame
            )
            right_pose = self.robot_sdk.tools.get_link_pose(
                link_name="zarm_r7_link",
                reference_frame=target_frame
            )
        current_left_pose = Pose(
            pos=left_pose.position,
            quat=left_pose.orientation,
            frame=target_frame
        )
        current_right_pose = Pose(
            pos=right_pose.position,
            quat=right_pose.orientation,
            frame=target_frame
        )

        return current_left_pose, current_right_pose


    def get_joint_positions(self):
        joint_positions = self.robot_sdk.state.arm_joint_state()
        return joint_positions


    def get_current_transform(self, source_frame: Frame, target_frame: Frame) -> Transform3D:
        """
        将tf的变换转换为Transform3D对象。

        参数：
            source_frame (Frame): 源坐标系。
            target_frame (Frame): 目标坐标系。

        返回：
            Transform3D: 转换后的Transform3D对象。
        """
        tf_pose = self.robot_sdk.tools.get_tf_transform(target_frame, source_frame)

        source_to_target_pose = Pose(
            pos=tf_pose.position,
            quat=tf_pose.orientation,
            frame=target_frame
        )

        transform_source_to_target = Transform3D(
            trans_pose=source_to_target_pose,
            source_frame=source_frame,  # 源坐标系为Tag坐标系
            target_frame=target_frame  # 目标坐标系为里程计坐标系
        )

        return transform_source_to_target


class TorsoAPI:
    """
    根据手臂和躯干控制接口，封装躯干控制的API
    """

    def __init__(self, robot_sdk: RobotSDK, yaw_threshold=np.deg2rad(5), pos_threshold=0.15):
        self.robot_sdk = robot_sdk
        self._pool = ThreadPoolExecutor(max_workers=2)

    def _move_torso_pose(self,
                         desir_torso_pose: Pose,
                         control_base: bool):
        if desir_torso_pose is None:
            raise ValueError("desir_torso_pose must not be None")

        self.robot_sdk.control.set_manipulation_mpc_mode(KuavoManipulationMpcCtrlMode.ArmOnly)

        x, y, z = desir_torso_pose.pos.tolist()
        roll, pitch, yaw = desir_torso_pose.get_euler(degrees=False).tolist()

        result = self.robot_sdk.control.control_torso_pose(x, y, z, roll, pitch, yaw)
        return result

    def move_torso_pose(self,
                        desir_torso_pose: Pose,
                        asynchronous: bool = False,
                        control_base: bool = False):
        if asynchronous:
            return self._pool.submit(
                self._move_torso_pose,
                desir_torso_pose,
                control_base,
            )

        result = self._move_torso_pose(desir_torso_pose, control_base)
        return result
