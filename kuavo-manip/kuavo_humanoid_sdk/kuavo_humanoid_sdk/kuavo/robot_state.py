#!/usr/bin/env python3
# coding: utf-8
from kuavo_humanoid_sdk.interfaces.data_types import KuavoJointData
from kuavo_humanoid_sdk.kuavo.core.ros.state import KuavoRobotStateCore

class KuavoRobotState:
    def __init__(self, robot_type: str = "kuavo"):
        self._rs_core = KuavoRobotStateCore()

    def arm_joint_state(self) -> KuavoJointData:
        """获取 Kuavo 机器人手臂关节的当前状态。

        获取 Kuavo 机器人手臂关节的当前状态，包括:
            - 关节位置(角度)，单位为弧度
            - 关节速度，单位为弧度/秒
            - 关节扭矩/力矩，单位为牛顿米、安培
            - 关节加速度

        Returns:
            KuavoJointData: 手臂关节数据包含:
                * position: list[float] * arm_dof(14)
                * velocity: list[float] * arm_dof(14)
                * torque: list[float] * arm_dof(14)
                * acceleration: list[float] * arm_dof(14)
        """
        # Get arm joint states from index 12 to 25 (14 arm joints)
        if len(self._rs_core.joint_data.position) == 28:
            arm_joint_indices = range(12, 12+14)
        elif len(self._rs_core.joint_data.position) == 20:
            arm_joint_indices = range(4, 4+14)
        else:
            raise ValueError(f"Joint data length is not 28 or 20: {len(self._rs_core.joint_data.position)}")

        return KuavoJointData(
            position=[self._rs_core.joint_data.position[i] for i in arm_joint_indices],
            velocity=[self._rs_core.joint_data.velocity[i] for i in arm_joint_indices],
            torque=[self._rs_core.joint_data.torque[i] for i in arm_joint_indices],
            acceleration=[self._rs_core.joint_data.acceleration[i] for i in arm_joint_indices]
        )
