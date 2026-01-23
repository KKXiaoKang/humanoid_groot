#!/usr/bin/env python3
import time
import argparse
import numpy as np
import py_trees

from kuavo_humanoid_sdk.kuavo_strategy_pytree.nodes.nodes import (
    NodePercep, NodeWaitForBlackboards, NodeFuntion, 
    NodeArm, NodeTagToArmGoal, NodeArmGoal,
    NodeTagToTorsoGoal, NodeTorso,
)

from kuavo_humanoid_sdk.kuavo_strategy_pytree.nodes.api import HeadAPI, ArmAPI, TorsoAPI
from kuavo_humanoid_sdk.kuavo_strategy_pytree.nodes.funcs import arm_generate_keypoints, ArmAction, RobotPlatform
from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.robot_sdk import RobotSDK
from constants import (
    TAG_ID, 
    TASK_SPECS, BASE_KP_ARGS,
    TASK_ID_MAP, BOX_WIDTH_MAP, BOX_COLOR_MAP, 
    TASK_ID_PROMPT, BOX_WIDTH_PROMPT, BOX_COLOR_PROMPT,
)
    
from recorder.record import RecordingController
from dataclasses import asdict


class PlatformConfig:
    """平台配置类，封装不同平台的行为差异"""
    def __init__(self, platform: str):
        self.platform = platform
        self.is_legged = (platform == "legged")
        self.robot_platform = RobotPlatform.LEGGED if self.is_legged else RobotPlatform.WHEELED
        
    def get_arm_api(self, robot_sdk: RobotSDK) -> ArmAPI:
        """根据平台创建 ArmAPI"""
        return ArmAPI(robot_sdk=robot_sdk, is_legged=self.is_legged)
    
    def get_node_arm_kwargs(self) -> dict:
        """返回 NodeArm 需要的平台相关参数"""
        return {"is_legged": self.is_legged}
    
    def get_node_arm_goal_kwargs(self) -> dict:
        """返回 NodeArmGoal 需要的平台相关参数"""
        return {"is_legged": self.is_legged}


robot_sdk = RobotSDK()
head_api = HeadAPI(robot_sdk=robot_sdk)
torso_api = TorsoAPI(robot_sdk=robot_sdk)
# arm_api 将在 main 函数中根据平台动态创建


def select_from_map(prompt, value_map):
    while True:
        user_input = input(prompt).strip()
        if user_input in value_map:
            return value_map[user_input]
        print("无效的输入，请重新输入")


def get_task_params(task_specs, task_id_map, box_width_map, box_color_map):
    task_id = select_from_map(
        TASK_ID_PROMPT,
        task_id_map
    )
    task_spec = task_specs[task_id]

    box_width = select_from_map(
        BOX_WIDTH_PROMPT,
        box_width_map
    )

    box_color = select_from_map(
        BOX_COLOR_PROMPT,
        box_color_map
    )

    return task_spec, box_width, box_color


def build_init(platform_config: PlatformConfig, arm_api: ArmAPI):
    """构建初始化行为树"""
    tag_id = TAG_ID
    
    action = py_trees.composites.Sequence("init_action", memory=True)
    root = py_trees.composites.Parallel(
        "init_root",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )
    
    # 轮臂平台需要感知节点
    if not platform_config.is_legged:
        percep = NodePercep(
            name="percep",
            robot_sdk=robot_sdk,
            tags_id=[tag_id],
            tag_up_axis="z"
        )
        root.add_children([action, percep])
    else:
        root.add_children([action])

    l_kp, r_kp = arm_generate_keypoints(
        ArmAction.INIT, 
        platform_config.robot_platform, 
        **asdict(BASE_KP_ARGS)
    )

    action_children = [
        NodeFuntion(name="set_init_head_pose", fn=lambda: head_api.move_head_traj([(0, np.deg2rad(20))])),
        NodeFuntion(name="set_init_claw", fn=lambda: robot_sdk.control.control_leju_claw([0, 0], [0, 0], [1.0, 1.0])),
    ]
    
    # 足式平台特有：设置直接控制到 WBC
    if platform_config.is_legged:
        action_children.insert(0, NodeFuntion(
            name="set_direct_to_wbc", 
            fn=lambda: robot_sdk.control.set_direct_to_wbc()
        ))
    
    # 轮臂平台特有：初始化躯干姿态和移动到标签位置
    if not platform_config.is_legged:
        try:
            from constants import INIT_TORSO_POSE
            action_children.extend([
                NodeFuntion(name="set_init_torso_pose", fn=lambda: torso_api.move_torso_pose(INIT_TORSO_POSE)),
                NodeWaitForBlackboards(keys=[f"latest_tag_{tag_id}"]),
                NodeTagToTorsoGoal(name='init_torso_pose', torso_api=torso_api, tag_id=tag_id, delta_z=0.3),
                NodeTorso(name='init_torso_pose', torso_api=torso_api),
                # NodeFuntion(name="sleep", fn=lambda: (time.sleep(2), True)[1]),
            ])
        except ImportError:
            # 如果 constants 中没有 INIT_TORSO_POSE，跳过躯干初始化
            pass
    
    # 添加手臂初始化
    action_children.extend([
        NodeArmGoal("init_tag2goal", arm_api, l_kp, r_kp, **platform_config.get_node_arm_goal_kwargs()),
        NodeArm("init_arm", arm_api, use_ik=True, **platform_config.get_node_arm_kwargs()),
    ])
    
    # 轮臂平台特有：设置快速模式
    if not platform_config.is_legged:
        action_children.append(
            NodeFuntion(name="set_arm_quick_mode", fn=lambda: robot_sdk.control.set_arm_quick_mode(True))
        )
    
    action.add_children(action_children)
    return root


def build_fail_stage_1(task_spec, box_width, platform_config: PlatformConfig, arm_api: ArmAPI):
    """构建失败阶段1行为树"""
    tag_id = TAG_ID
    kp_args = {**asdict(BASE_KP_ARGS), 
               "box_width": box_width,
               "box_behind_tag": -0.1 if box_width == 0.4 else -0.15,
               "side": task_spec.pick_order[0] if task_spec.pick_order else "both"}

    action = py_trees.composites.Sequence("init_action", memory=True)
    root = py_trees.composites.Parallel(
        "fail_stage_1_root",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )
    
    percep = NodePercep(
        name="percep",
        robot_sdk=robot_sdk,
        tags_id=[tag_id],
        tag_up_axis="z"
    )
    root.add_children([action, percep])
    
    l_kp, r_kp = arm_generate_keypoints(
        ArmAction.FIRST_PICK_FAIL, 
        platform_config.robot_platform, 
        **kp_args,
        random_offset=0.1
    )
    
    action.add_children([
        NodeTagToArmGoal("init_tag2goal", arm_api, tag_id, l_kp, r_kp),
        NodeArm("init_arm", arm_api, use_ik=True, **platform_config.get_node_arm_kwargs()),
    ])
    
    return root


def build_fail_stage_2(task_spec, box_width, platform_config: PlatformConfig, arm_api: ArmAPI):
    """构建失败阶段2行为树"""
    tag_id = TAG_ID

    percep = NodePercep(
        name="percep",
        robot_sdk=robot_sdk,
        tags_id=[tag_id],
        tag_up_axis="z"
    )

    action = py_trees.composites.Sequence(f"{task_spec.name}_action", memory=True)
    root = py_trees.composites.Parallel(
        f"{task_spec.name}_root",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )
    root.add_children([action, percep])

    search = py_trees.composites.Sequence("search_tag", memory=True)
    search.add_children([
        NodeWaitForBlackboards(
            keys=[f"latest_tag_{tag_id}"]
        )
    ])

    kp_args = {**asdict(BASE_KP_ARGS), "box_width": box_width, "box_behind_tag": -0.1 if box_width == 0.4 else -0.15}
    arm_kwargs = platform_config.get_node_arm_kwargs()


    kp_args_first = {**kp_args, "side": task_spec.pick_order[0]}
    kp_args_second = {**kp_args, "side": task_spec.pick_order[1]}
    l_kp, r_kp = arm_generate_keypoints(ArmAction.FIRST_PICK, platform_config.robot_platform, **kp_args_first)
    first_pick = py_trees.composites.Sequence(f"{task_spec.name}_first_pick", memory=True)
    first_pick.add_children([
        NodeTagToArmGoal(f"{task_spec.name}_first_pick_tag2goal", arm_api, tag_id, l_kp, r_kp),
        NodeArm(f"{task_spec.name}_first_pick_arm", arm_api, use_ik=True, **arm_kwargs),
        NodeFuntion(
            name=f"{task_spec.name}_first_claw_close",
            fn=lambda: robot_sdk.control.control_leju_claw(
                [80, 0], 
                [80, 0], 
                [1.0, 1.0]
            ) if task_spec.pick_order[0] == "left" 
            else robot_sdk.control.control_leju_claw(
                [0, 80], 
                [0, 80], 
                [1.0, 1.0]
            )
        )
    ])

    l_kp, r_kp = arm_generate_keypoints(ArmAction.DEPALLETIZE, platform_config.robot_platform, **kp_args_first)
    depalletize = py_trees.composites.Sequence(f"{task_spec.name}_depalletize", memory=True)
    depalletize.add_children([
        NodeTagToArmGoal(f"{task_spec.name}_depalletize_tag2goal", arm_api, tag_id, l_kp, r_kp),
        NodeArm(f"{task_spec.name}_depalletize_arm", arm_api, use_ik=True, **arm_kwargs),
    ])

    l_kp, r_kp = arm_generate_keypoints(ArmAction.SECOND_PICK_FAIL, platform_config.robot_platform, **kp_args_second, random_offset=0.1)
    second_pick_fail = py_trees.composites.Sequence(f"{task_spec.name}_second_pick_fail", memory=True)
    second_pick_fail.add_children([
        NodeTagToArmGoal(f"{task_spec.name}_second_pick_fail_tag2goal", arm_api, tag_id, l_kp, r_kp),
        NodeArm(f"{task_spec.name}_second_pick_fail_arm", arm_api, use_ik=True, **arm_kwargs),
        # NodeFuntion(name=f"{task_spec.name}_close_both_fail", fn=lambda: robot_sdk.control.control_leju_claw([80, 80], [80, 80], [1.0, 1.0])),
    ])

    action.add_children([
        search,
        first_pick,
        depalletize,
        second_pick_fail,
    ])

    return root

def build_main_stage_2(task_spec, box_width, platform_config: PlatformConfig, arm_api: ArmAPI):
    """构建主任务阶段2行为树"""
    tag_id = TAG_ID

    percep = NodePercep(
        name="percep",
        robot_sdk=robot_sdk,
        tags_id=[tag_id],
        tag_up_axis="z"
    )

    action = py_trees.composites.Sequence(f"{task_spec.name}_action", memory=True)
    root = py_trees.composites.Parallel(
        f"{task_spec.name}_root",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )
    root.add_children([action, percep])

    search = py_trees.composites.Sequence("search_tag", memory=True)
    search.add_children([
        NodeWaitForBlackboards(
            keys=[f"latest_tag_{tag_id}"]
        )
    ])

    kp_args_pick = {**asdict(BASE_KP_ARGS), "box_width": box_width, "side": task_spec.pick_order[1], "box_behind_tag": -0.1 if box_width == 0.4 else -0.15}
    kp_args_lift = {**asdict(BASE_KP_ARGS), "box_width": box_width, "box_behind_tag": -0.1 if box_width == 0.4 else -0.15}
    arm_kwargs = platform_config.get_node_arm_kwargs()

    l_kp, r_kp = arm_generate_keypoints(ArmAction.SECOND_PICK_RECOVER, platform_config.robot_platform, **kp_args_pick)
    second_pick = py_trees.composites.Sequence(f"{task_spec.name}_second_pick", memory=True)
    second_pick.add_children([
        NodeTagToArmGoal(f"{task_spec.name}_second_pick_tag2goal", arm_api, tag_id, l_kp, r_kp),
        NodeArm(f"{task_spec.name}_second_pick_arm", arm_api, use_ik=True, **arm_kwargs),
        NodeFuntion(name=f"{task_spec.name}_close_both", fn=lambda: robot_sdk.control.control_leju_claw([80, 80], [80, 80], [1.0, 1.0])),
    ])

    l_kp, r_kp = arm_generate_keypoints(ArmAction.LIFT, platform_config.robot_platform, **kp_args_lift)
    lift = py_trees.composites.Sequence(f"{task_spec.name}_lift", memory=True)
    lift.add_children([
        NodeTagToArmGoal(f"{task_spec.name}_lift_tag2goal", arm_api, tag_id, l_kp, r_kp),
        NodeArm(f"{task_spec.name}_lift_arm", arm_api, use_ik=True, **arm_kwargs),
    ])


    action.add_children([
        search,
        second_pick,
        lift,
    ])

    return root


def build_main(task_spec, box_width, platform_config: PlatformConfig, arm_api: ArmAPI, fail_stage: int):
    """构建主任务行为树"""
    tag_id = TAG_ID

    percep = NodePercep(
        name="percep",
        robot_sdk=robot_sdk,
        tags_id=[tag_id],
        tag_up_axis="z"
    )

    action = py_trees.composites.Sequence(f"{task_spec.name}_action", memory=True)
    root = py_trees.composites.Parallel(
        f"{task_spec.name}_root",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )
    root.add_children([action, percep])

    search = py_trees.composites.Sequence("search_tag", memory=True)
    search.add_children([
        NodeWaitForBlackboards(
            keys=[f"latest_tag_{tag_id}"]
        )
    ])

    kp_args = {**asdict(BASE_KP_ARGS), "box_width": box_width, "box_behind_tag": -0.1 if box_width == 0.4 else -0.15}
    arm_kwargs = platform_config.get_node_arm_kwargs()

    if task_spec.pick_type == "unpack":
        kp_args_first = {**kp_args, "side": task_spec.pick_order[0]}
        kp_args_second = {**kp_args, "side": task_spec.pick_order[1]}
        l_kp, r_kp = arm_generate_keypoints(ArmAction.FIRST_PICK if fail_stage == 0 else ArmAction.FIRST_PICK_RECOVER, platform_config.robot_platform, **kp_args_first)
        first_pick = py_trees.composites.Sequence(f"{task_spec.name}_first_pick", memory=True)
        first_pick.add_children([
            NodeTagToArmGoal(f"{task_spec.name}_first_pick_tag2goal", arm_api, tag_id, l_kp, r_kp),
            NodeArm(f"{task_spec.name}_first_pick_arm", arm_api, use_ik=True, **arm_kwargs),
            NodeFuntion(
                name=f"{task_spec.name}_first_claw_close",
                fn=lambda: robot_sdk.control.control_leju_claw(
                    [80, 0], 
                    [80, 0], 
                    [1.0, 1.0]
                ) if task_spec.pick_order[0] == "left" 
                else robot_sdk.control.control_leju_claw(
                    [0, 80], 
                    [0, 80], 
                    [1.0, 1.0]
                )
            )
        ])

        l_kp, r_kp = arm_generate_keypoints(ArmAction.DEPALLETIZE, platform_config.robot_platform, **kp_args_first)
        depalletize = py_trees.composites.Sequence(f"{task_spec.name}_depalletize", memory=True)
        depalletize.add_children([
            NodeTagToArmGoal(f"{task_spec.name}_depalletize_tag2goal", arm_api, tag_id, l_kp, r_kp),
            NodeArm(f"{task_spec.name}_depalletize_arm", arm_api, use_ik=True, **arm_kwargs),
        ])

        l_kp, r_kp = arm_generate_keypoints(ArmAction.SECOND_PICK, platform_config.robot_platform, **kp_args_second)
        second_pick = py_trees.composites.Sequence(f"{task_spec.name}_second_pick", memory=True)
        second_pick.add_children([
            NodeTagToArmGoal(f"{task_spec.name}_second_pick_tag2goal", arm_api, tag_id, l_kp, r_kp),
            NodeArm(f"{task_spec.name}_second_pick_arm", arm_api, use_ik=True, **arm_kwargs),
            NodeFuntion(name=f"{task_spec.name}_close_both", fn=lambda: robot_sdk.control.control_leju_claw([80, 80], [80, 80], [1.0, 1.0])),
        ])

        action.add_children([
            search,
            first_pick,
            depalletize,
            second_pick,
        ])

    else:
        l_kp, r_kp = arm_generate_keypoints(ArmAction.PICK, platform_config.robot_platform, **kp_args)
        pick = py_trees.composites.Sequence(f"{task_spec.name}_pick", memory=True)
        pick.add_children([
            NodeTagToArmGoal(f"{task_spec.name}_first_pick_tag2goal", arm_api, tag_id, l_kp, r_kp),
            NodeArm(f"{task_spec.name}_first_pick_arm", arm_api, use_ik=True, **arm_kwargs),
            NodeFuntion(name=f"{task_spec.name}_first_claw_close", fn=lambda: robot_sdk.control.control_leju_claw([80, 80], [80, 80], [1.0, 1.0])),
        ])

        action.add_children([
            search,
            pick,
        ])

    l_kp, r_kp = arm_generate_keypoints(ArmAction.LIFT, platform_config.robot_platform, **kp_args)
    lift = py_trees.composites.Sequence(f"{task_spec.name}_lift", memory=True)
    lift.add_children([
        NodeTagToArmGoal(f"{task_spec.name}_lift_tag2goal", arm_api, tag_id, l_kp, r_kp),
        NodeArm(f"{task_spec.name}_lift_arm", arm_api, use_ik=True, **arm_kwargs),
    ])

    action.add_children([
        lift,
    ])

    return root


def build_end(platform_config: PlatformConfig, arm_api: ArmAPI):
    """构建结束行为树"""
    l_kp, r_kp = arm_generate_keypoints(
        ArmAction.END, 
        platform_config.robot_platform, 
        **asdict(BASE_KP_ARGS)
    )
    seq = py_trees.composites.Sequence("end", memory=True)
    seq.add_children([
        NodeFuntion(name="open_claw", fn=lambda: robot_sdk.control.control_leju_claw([0, 0], [0, 0], [1.0, 1.0])),
        NodeFuntion(name="sleep", fn=lambda: (time.sleep(2), True)[1]),
        NodeArmGoal("end_tag2goal", arm_api, l_kp, r_kp, **platform_config.get_node_arm_goal_kwargs()),
        NodeArm("end_arm", arm_api, use_ik=True, **platform_config.get_node_arm_kwargs()),
    ])
    return seq


def run_tree(tree):
    bt = py_trees.trees.BehaviourTree(tree)
    while True:
        bt.tick()
        if tree.status != py_trees.common.Status.RUNNING:
            return tree.status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default="default")
    parser.add_argument("--episodes", type=int, default=999)
    parser.add_argument("--platform", type=str, default="wheeled", choices=["legged", "wheeled"])
    parser.add_argument("-i", "--interactive", action="store_true", help="每个案例开始前进行获取用户输入")
    parser.add_argument("-F", "--fail-stage", type=int, choices=[0, 1, 2], help="失败模拟阶段, 0=不模拟失败, 1=第一次抓取失败, 2=第二次抓取失败")
    args = parser.parse_args()

    # 创建平台配置和对应的 API
    platform_config = PlatformConfig(args.platform)
    arm_api = platform_config.get_arm_api(robot_sdk)

    recorder = RecordingController(args.name, init_node=False)

    task_spec, box_width, box_color = None, None, None

    for eps in range(args.episodes):
        print(f"\n>>> Episode {eps} START (Platform: {args.platform})")

        if args.interactive or task_spec is None:
            task_spec, box_width, box_color = get_task_params(TASK_SPECS, TASK_ID_MAP, BOX_WIDTH_MAP, BOX_COLOR_MAP)

        recorder.bag_prefix = f"{task_spec.name}_width_{int(box_width*100)}_{box_color}"

        # INIT
        run_tree(build_init(platform_config, arm_api))

        if args.fail_stage == 1:
            run_tree(build_fail_stage_1(task_spec, box_width, platform_config, arm_api))

        # MAIN (record)
        if args.fail_stage == 2 and task_spec.pick_type == "unpack":
            run_tree(build_fail_stage_2(task_spec, box_width, platform_config, arm_api))
            recorder.start_recording()
            time.sleep(2)   # wait for camera to be ready
            run_tree(build_main_stage_2(task_spec, box_width, platform_config, arm_api))
            recorder.stop_recording()
            
            time.sleep(0.5)
        else:
            recorder.start_recording()
            time.sleep(2)   # wait for camera to be ready
            run_tree(build_main(task_spec, box_width, platform_config, arm_api, args.fail_stage))
            recorder.stop_recording()
            
            time.sleep(0.5)

        # END
        run_tree(build_end(platform_config, arm_api))
        
        # 轮臂平台特有：重置躯干姿态
        if not platform_config.is_legged:
            try:
                from constants import INIT_TORSO_POSE
                torso_api.move_torso_pose(INIT_TORSO_POSE)
            except ImportError:
                pass  # 如果 constants 中没有 INIT_TORSO_POSE，跳过

        py_trees.blackboard.Blackboard().clear()
        input(f">>> Episode {eps} END, press Enter to continue...\n")


if __name__ == "__main__":
    main()