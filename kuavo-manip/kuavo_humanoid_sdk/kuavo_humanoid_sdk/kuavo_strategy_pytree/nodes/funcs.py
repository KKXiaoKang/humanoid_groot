from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.data_type import Pose, Frame


from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

class ArmAction(Enum):
    INIT = auto()
    PICK = auto()
    FIRST_PICK = auto()
    FIRST_PICK_FAIL = auto()
    FIRST_PICK_RECOVER = auto()
    SECOND_PICK = auto()
    SECOND_PICK_FAIL = auto()
    SECOND_PICK_RECOVER = auto()
    DEPALLETIZE = auto()
    LIFT = auto()
    END = auto()


class RobotPlatform(Enum):
    LEGGED = auto()
    WHEELED = auto()


@dataclass
class ArmKeypointConfig:
    approach_height: float
    target_height: float
    lateral_offset: float = 0.0
    retreat_offset: float = 0.0
    euler: Tuple[float, float, float] = (0, 35, 90)
    frame: Frame = Frame.TAG


ARM_ACTION_CONFIG = {
    ArmAction.PICK: ArmKeypointConfig(0.4, 0.27),
    ArmAction.FIRST_PICK: ArmKeypointConfig(0.4, 0.27, lateral_offset=0.0),
    ArmAction.FIRST_PICK_FAIL: ArmKeypointConfig(0.4, 0.4, lateral_offset=0.0),
    ArmAction.FIRST_PICK_RECOVER: ArmKeypointConfig(0.4, 0.27, lateral_offset=0.0),
    ArmAction.SECOND_PICK: ArmKeypointConfig(0.4, 0.23, lateral_offset=0.02),
    ArmAction.SECOND_PICK_FAIL: ArmKeypointConfig(0.4, 0.4, lateral_offset=0.02),
    ArmAction.SECOND_PICK_RECOVER: ArmKeypointConfig(0.4, 0.23, lateral_offset=0.02),
    ArmAction.DEPALLETIZE: ArmKeypointConfig(0.4, 0.4, retreat_offset=0.1),
    ArmAction.LIFT: ArmKeypointConfig(0.4, 0.42, euler=(0, 40, 90)),
}

ARM_BASE_POSES_WHEELED = {
    ArmAction.INIT: {
        "left": [
            Pose.from_euler((0.2, 0.3, 0.1), (-45, -45, 0), Frame.WAIST_YAW_LINK, True,)
        ],
        "right": [
            Pose.from_euler((0.2, -0.3, 0.1), (45, -45, 0), Frame.WAIST_YAW_LINK, True,)
        ],
    },

    ArmAction.END: {
        "left": [
            Pose.from_euler((0.3, 0.3, 0.05), (0, -45, -90), Frame.WAIST_YAW_LINK, True,),
            Pose.from_euler((0.2, 0.3, 0.05), (-45, -45, 0), Frame.WAIST_YAW_LINK, True,),
        ],
        "right": [
            Pose.from_euler((0.3, -0.3, 0.05), (0, -45, 90), Frame.WAIST_YAW_LINK, True,),
            Pose.from_euler((0.2, -0.3, 0.05), (45, -45, 0), Frame.WAIST_YAW_LINK, True,),
        ],
    },
}

ARM_BASE_POSES_LEGGED = {
    ArmAction.INIT: {
        "left": [
            Pose.from_euler((0.2, 0.3, 0.3), (-45, -45, 0), Frame.BASE, True,)
        ],
        "right": [
            Pose.from_euler((0.2, -0.3, 0.3), (45, -45, 0), Frame.BASE, True,)
        ],
    },

    ArmAction.END: {
        "left": [
            Pose.from_euler((0.2, 0.3, 0.3), (-45, -45, 0), Frame.BASE, True,),
        ],
        "right": [
            Pose.from_euler((0.2, -0.3, 0.3), (45, -45, 0), Frame.BASE, True,),
        ],
    },
}

# 维护一个持久化的offset，在调用END之后清零
_persistent_offset = {
    -1: (0.0, 0.0, 0.0),  # left arm: (x, y, z)
    1: (0.0, 0.0, 0.0),   # right arm: (x, y, z)
}
_offset_initialized = False  # 标记offset是否已经被初始化


def arm_generate_action_keypoints(
    action: ArmAction,
    box_width: float,
    box_behind_tag: float,
    box_beneath_tag: float,
    box_left_tag: float,
    side: str,  # left | right | both
    random_offset: float = 0.0,  # 随机偏移量（前后左右各±random_offset）
):
    cfg = ARM_ACTION_CONFIG[action]

    # 第一次生成随机偏移时写入，后续不再生成，直到调用END之后重置
    global _persistent_offset, _offset_initialized
    if not _offset_initialized and random_offset > 0:
        _persistent_offset = {
            -1: (
                np.random.uniform(-random_offset, random_offset),  # left x
                np.random.uniform(-0.08, 0),  # left y
                np.random.uniform(-random_offset, random_offset),  # left z
            ),
            1: (
                np.random.uniform(-random_offset, random_offset),  # right x
                np.random.uniform(-0.08, 0),  # right y
                np.random.uniform(-random_offset, random_offset),  # right z
            ),
        }
        _offset_initialized = True
    
    # 根据 random_offset 决定使用哪些 offset 分量
    # random_offset > 0: 使用完整的持久化 offset (x, y, z)
    # random_offset == 0: offset_x 和 offset_z 为 0，offset_y 使用持久化的值（如 RECOVER 动作）
    offset = _persistent_offset.copy() if random_offset > 0 else {
        -1: (0.0, _persistent_offset[-1][1] if _offset_initialized else 0.0, 0.0),
        1: (0.0, _persistent_offset[1][1] if _offset_initialized else 0.0, 0.0),
    }

    def make_arm(sign, lateral, retreat):
        offset_x, offset_y, offset_z = offset[sign]
        x = sign * (box_width / 2) - box_left_tag + lateral + retreat + offset_x
        z = -box_behind_tag + offset_z

        # INIT动作只生成一个keypoint（类似FIRST_PICK的第一个位置）
        if action in (ArmAction.FIRST_PICK_FAIL, ArmAction.SECOND_PICK_FAIL):
            return [
                Pose.from_euler(
                    pos=(x, -box_beneath_tag + cfg.approach_height + offset_y, z),
                    euler=cfg.euler,
                    degrees=True,
                    frame=cfg.frame,
                ),
                Pose.from_euler(
                    pos=(x, -box_beneath_tag + cfg.target_height + offset_y, z),
                    euler=cfg.euler,
                    degrees=True,
                    frame=cfg.frame,
                ),
            ]
        elif action in (ArmAction.FIRST_PICK_RECOVER, ArmAction.SECOND_PICK_RECOVER):
            return [
                Pose.from_euler(
                    pos=(x, -box_beneath_tag + cfg.approach_height + offset_y, z),
                    euler=cfg.euler,
                    degrees=True,
                    frame=cfg.frame,
                ),
                Pose.from_euler(
                    pos=(x, -box_beneath_tag + cfg.target_height + offset_y, z),
                    euler=cfg.euler,
                    degrees=True,
                    frame=cfg.frame,
                ),
            ]
        else:
            return [
                Pose.from_euler(
                    pos=(x if action != ArmAction.DEPALLETIZE else x -retreat, -box_beneath_tag + cfg.approach_height, z),
                    euler=cfg.euler,
                    degrees=True,
                    frame=cfg.frame,
                ),
                Pose.from_euler(
                    pos=(x, -box_beneath_tag + cfg.target_height, z),
                    euler=cfg.euler,
                    degrees=True,
                    frame=cfg.frame,
                ),
            ]

    left, right = [], []

    if side in ("left", "both"):
        left = make_arm(-1, cfg.lateral_offset, -cfg.retreat_offset)

    if side in ("right", "both"):
        right = make_arm(1, -cfg.lateral_offset, cfg.retreat_offset)

    return left, right


def arm_generate_base_keypoints(
    platform: RobotPlatform,
    action: ArmAction,
    side: str,
):
    poses = ARM_BASE_POSES_LEGGED.get(action) if platform == RobotPlatform.LEGGED else ARM_BASE_POSES_WHEELED.get(action)
    if poses is None:
        return [], []
    
    left = poses["left"] if side in ("left", "both") else []
    right = poses["right"] if side in ("right", "both") else []
    return left, right


def arm_generate_keypoints(
    action: ArmAction,
    platform: RobotPlatform,
    box_width: float = 0.0,
    box_behind_tag: float = 0.0,
    box_beneath_tag: float = 0.0,
    box_left_tag: float = 0.0,
    side: str = "both",
    random_offset: float = 0.0,  # 随机偏移量（前后左右各±random_offset，仅用于INIT动作）
):
    # 在调用END之后清零持久化offset
    if action == ArmAction.END:
        global _persistent_offset, _offset_initialized
        _persistent_offset = {
            -1: (0.0, 0.0, 0.0),
            1: (0.0, 0.0, 0.0),
        }
        _offset_initialized = False
        left, right = arm_generate_base_keypoints(platform, action, side)
    elif action == ArmAction.INIT:
        left, right = arm_generate_base_keypoints(platform, action, side)
    else:
        left, right = arm_generate_action_keypoints(
            action,
            box_width,
            box_behind_tag,
            box_beneath_tag,
            box_left_tag,
            side,
            random_offset=random_offset,
        )

    return left, right