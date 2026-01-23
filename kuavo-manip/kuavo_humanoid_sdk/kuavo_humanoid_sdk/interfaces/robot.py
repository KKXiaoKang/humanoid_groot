
from abc import ABC, abstractmethod

class RobotBase(ABC):
    def __init__(self, robot_type: str = "kuavo"):
        self._robot_type = robot_type

    @property
    def robot_type(self) -> str:
        return self._robot_type
