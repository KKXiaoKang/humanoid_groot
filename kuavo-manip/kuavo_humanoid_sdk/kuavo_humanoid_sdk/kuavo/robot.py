#!/usr/bin/env python3
# coding: utf-8
import math
from kuavo_humanoid_sdk.interfaces.robot import RobotBase
from kuavo_humanoid_sdk.common.logger import SDKLogger
from kuavo_humanoid_sdk.interfaces.data_types import KuavoManipulationMpcCtrlMode, KuavoArmCtrlMode
from kuavo_humanoid_sdk.kuavo.core.core import KuavoRobotCore

from kuavo_humanoid_sdk.kuavo.robot_info import KuavoRobotInfo

"""
Kuavo SDK - Kuavo机器人控制的Python接口

本模块提供了通过Python控制Kuavo机器人的主要接口。

KuavoRobot:
    主要的机器人控制接口类,提供以下功能的访问:
    - 机器人信息和状态 (通过 KuavoRobotInfo)
    - 机械臂控制功能
    - 头部控制功能
    - 核心机器人功能 (通过 KuavoRobotCore)
    
该模块需要正确配置的ROS环境才能运行。
"""
__all__ = ["KuavoRobot"]


class KuavoRobot(RobotBase):
    def __init__(self):
        super().__init__(robot_type="kuavo")
        
        self._robot_info = KuavoRobotInfo()
        self._kuavo_core = KuavoRobotCore()
    
    def control_torso_pose(self, x: float, y: float, z: float,
                           roll: float, pitch: float, yaw: float) -> bool:
        """直接控制轮臂机器人躯干的位姿

        Args:
            x, y, z (float): 目标位置（米）
            roll, pitch, yaw (float): 目标欧拉角（弧度）

        Returns:
            bool: 控制命令是否发送成功
        """
        return self._kuavo_core.control_torso_pose(x, y, z, roll, pitch, yaw)
    
    def control_head(self, yaw: float, pitch: float)->bool:
        """控制机器人的头部关节运动。

        Args:
            yaw (float): 头部的偏航角,单位弧度,范围[-1.396, 1.396](-80到80度)。
            pitch (float): 头部的俯仰角,单位弧度,范围[-0.436, 0.436](-25到25度)。

        Returns:
            bool: 如果头部控制成功返回True,否则返回False。
        """

        limited_yaw = yaw
        limited_pitch = pitch

        # Check yaw limits (-80 to 80 degrees)
        if yaw < -math.pi*4/9 or yaw > math.pi*4/9:
            SDKLogger.warn(f"[Robot] yaw {yaw} exceeds limit [-{math.pi*4/9:.3f}, {math.pi*4/9:.3f}] radians (-80 to 80 degrees), will be limited")
            limited_yaw = min(math.pi*4/9, max(-math.pi*4/9, yaw))

        # Check pitch limits (-25 to 25 degrees)
        if pitch < -math.pi/7.2 - 0.001 or pitch > math.pi/7.2 + 0.001:  # -25 to 25 degrees in radians
            SDKLogger.warn(f"[Robot] pitch {pitch} exceeds limit [-{math.pi/7.2:.3f}, {math.pi/7.2:.3f}] radians (-25 to 25 degrees), will be limited")
            limited_pitch = min(math.pi/7.2, max(-math.pi/7.2, pitch))

        result = self._kuavo_core.control_robot_head(yaw=limited_yaw, pitch=limited_pitch)

        return result
    
    def control_arm_joint_positions(self, joint_positions:list)->bool:
        """通过关节位置角度控制手臂

        Args:
            joint_positions (list): 手臂的目标关节位置,单位弧度。

        Returns:
            bool: 如果手臂控制成功返回True,否则返回False。

        Raises:
            ValueError: 如果关节位置列表长度不正确。
            ValueError: 如果关节位置超出[-π, π]范围。
            RuntimeError: 如果在尝试控制手臂时机器人不在stance状态。
        """
        if len(joint_positions) != self._robot_info.arm_joint_dof:
            raise ValueError("Invalid position length. Expected {}, got {}".format(self._robot_info.arm_joint_dof, len(joint_positions)))

        # Check if joint positions are within ±180 degrees (±π radians)
        for pos in joint_positions:
            if abs(pos) > math.pi:
                raise ValueError(f"Joint position {pos} rad exceeds ±π rad (±180 deg) limit")

        return self._kuavo_core.control_robot_arm_joint_positions(joint_data=joint_positions)

    def set_external_control_arm_mode(self) -> bool:
        """切换手臂控制模式到外部控制模式。

        Returns:
            bool: 如果切换手臂控制模式到外部控制模式成功返回True,否则返回False。
        """
        return self._kuavo_core.change_robot_arm_ctrl_mode(KuavoArmCtrlMode.ExternalControl)

    def set_arm_only_mode(self) -> bool:
        """切换手臂控制模式到仅控制手臂模式。

        Returns:
            bool: 如果切换手臂控制模式到仅控制手臂模式成功返回True,否则返回False。
        """
        return self._kuavo_core.change_manipulation_mpc_ctrl_mode(KuavoManipulationMpcCtrlMode.ArmOnly)
    
    def set_manipulation_mpc_mode(self, ctrl_mode: KuavoManipulationMpcCtrlMode) -> bool:
        """设置 Manipulation MPC 模式。

        Returns:
            bool: 如果 Manipulation MPC 模式设置成功返回True,否则返回False。
        """
        return self._kuavo_core.change_manipulation_mpc_ctrl_mode(ctrl_mode)

    # used
    def set_direct_to_wbc(self) -> bool:
        """设置直接流向WBC。

        Returns:
            bool: 如果设置成功返回True,否则返回False。
        """
        return self._kuavo_core.set_direct_to_wbc()
        # return self._kuavo_core.change_manipulation_mpc_control_flow(KuavoManipulationMpcControlFlow.DirectToWbc)

    # used
    def set_arm_quick_mode(self, enable: bool):
        """设置手臂快速模式
        """
        return self._kuavo_core.set_arm_quick_mode(enable)

    # used
    def control_leju_claw(self, postions:list, velocities:list=[90, 90], torques:list=[1.0, 1.0]) ->bool:
        """控制机器人末端夹爪

        Args:
            postions (list): 夹爪位置行程,单位百分比, 0: 张开, 100: 闭合
            velocities (list): 夹爪速度,单位百分比
            torques (list): 夹爪力矩,单位牛顿米

        Returns:
            bool: 如果控制成功返回True,否则返回False。
        """
        return self._kuavo_core.control_leju_claw(postions, velocities, torques)
