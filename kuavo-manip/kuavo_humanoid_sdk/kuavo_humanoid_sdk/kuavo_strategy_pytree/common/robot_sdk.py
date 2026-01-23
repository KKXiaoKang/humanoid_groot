import time

from kuavo_humanoid_sdk.kuavo.robot import KuavoRobot
from kuavo_humanoid_sdk.kuavo.robot_state import KuavoRobotState
from kuavo_humanoid_sdk.kuavo.robot_tool import KuavoRobotTools
from kuavo_humanoid_sdk.kuavo.robot_vision import KuavoRobotVision

class RobotSDK:
    def __init__(self):
        """
        初始化机器人SDK

        Args:
            robot: KuavoRobot实例，提供机器人控制能力
            robot_state: KuavoRobotState实例，提供机器人状态信息
            robot_tools: KuavoRobotTools实例，提供坐标转换等工具
        """
        self.control = KuavoRobot()
        self.state = KuavoRobotState()
        self.tools = KuavoRobotTools()
        self.vision = KuavoRobotVision()

        time.sleep(2)  # 等待底层初始化完成
