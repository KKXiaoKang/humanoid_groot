#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from kuavo_msgs.msg import JoySticks
from sensor_msgs.msg import JointState


class DAgger(object):
    def __init__(self):
        self._relay_active = False
        self._grip_threshold =  0.6
        self._traj_cpp_topic = "/kuavo_arm_traj_cpp"
        self._traj_out_topic = "/kuavo_arm_traj"

        self._pub = rospy.Publisher(
            self._traj_out_topic, JointState, queue_size=1, tcp_nodelay=True
        )
        self._sub_cpp = None

        self.quest_sub = rospy.Subscriber(
            "/quest_joystick_data",
            JoySticks,
            self._joy_callback,
            queue_size=10,
            tcp_nodelay=True,
        )

    def _set_relay_active(self, active):
        if active and not self._relay_active:
            self._relay_active = True
            self._sub_cpp = rospy.Subscriber(
                self._traj_cpp_topic,
                JointState,
                self._traj_cpp_callback,
                queue_size=1,
                tcp_nodelay=True,
            )
            rospy.loginfo(
                "检测到扳机按下，停止模型推理并启用 %s -> %s 转发。",
                self._traj_cpp_topic,
                self._traj_out_topic,
            )
        elif not active and self._relay_active:
            self._relay_active = False
            self._sub_cpp.unregister()
            self._sub_cpp = None
            rospy.loginfo(
                "扳机释放，停止 %s -> %s 转发。",
                self._traj_cpp_topic,
                self._traj_out_topic,
            )
        else:
            return

    def _traj_cpp_callback(self, msg):
        if self._relay_active:
            self._pub.publish(msg)

    def _joy_callback(self, msg):
        trigger_pressed = (
            msg.left_grip >= self._grip_threshold
            or msg.right_grip >= self._grip_threshold
        )
        self._set_relay_active(trigger_pressed)

    def is_relay_active(self):
        """返回是否启用 /kuavo_arm_traj_cpp -> /kuavo_arm_traj 转发。"""
        return self._relay_active

