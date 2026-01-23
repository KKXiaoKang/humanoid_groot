from __future__ import annotations

from dataclasses import dataclass
from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.data_type import Pose, Frame

TAG_ID = 4

INIT_TORSO_POSE = Pose.from_euler(
        pos=(0, 0, 1.0),
        euler=(0, 0, 0),
        degrees=True,
        frame=Frame.ODOM
    )

@dataclass
class TaskSpec:
    name: str   # "pick", "unpack_left", "unpack_right"
    pick_type: str   # "pick", "unpack"
    pick_order: list[str]   # ["left", "right"]

@dataclass
class KPArgs:
    box_width: float
    box_behind_tag: float 
    box_beneath_tag: float
    box_left_tag: float
    side: str

TASK_SPECS = {
    "pick": TaskSpec(name="pick", pick_order=None, pick_type="pick"),
    "unpack_left": TaskSpec(name="unpack_left", pick_order=["left", "right"], pick_type="unpack"),
    "unpack_right": TaskSpec(name="unpack_right", pick_order=["right", "left"], pick_type="unpack"),
}

BASE_KP_ARGS = KPArgs(
    box_width=0.4,
    box_behind_tag=-0.1,
    box_beneath_tag=0.11,
    box_left_tag=0.0,
    side="both",
)

TASK_ID_MAP = {"1": "pick", "2": "unpack_left", "3": "unpack_right"}
BOX_WIDTH_MAP = {"1": 0.4, "2": 0.6}
BOX_COLOR_MAP = {"1": "green", "2": "grey", "3": "blue"}

TASK_ID_PROMPT = (
    "请选择任务编号：\n"
    "  1) 搬箱\n"
    "  2) 左拆\n"
    "  3) 右拆\n"
    "请输入编号: "
)

BOX_WIDTH_PROMPT = (
    "请选择箱子宽度：\n"
    "  1) 0.4 m\n"
    "  2) 0.6 m\n"
    "请输入编号: "
)

BOX_COLOR_PROMPT = (
    "请选择箱子颜色：\n"
    "  1) 绿色\n"
    "  2) 灰色\n"
    "  3) 蓝色\n"
    "请输入编号: "
)