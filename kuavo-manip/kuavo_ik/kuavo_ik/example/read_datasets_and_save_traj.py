import json
import os.path

from visualize_traj import plot_trajectories_with_frames
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
import time
from lowpass_filter import LowPassFilter

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ik_library import IKAnalytical, IKNumerical
from fk_tool import FKTool

fk_tool = FKTool()


def read_json_and_do_ik(visualize_in_mujoco=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ds_json_path = os.path.join(current_dir, 'depalletize_traj_both_hands.json')

    with open(ds_json_path, 'r') as f:
        dataset_trajs = json.load(f)

    episodes = dataset_trajs['episodes']

    eps = episodes['0']
    SIDE = 'left'  # 'left' or 'right'
    EEF_FRAME = 'zarm_l7_link'  # 'zarm_l7_link' or 'zarm_r7_end_effector'

    joint_cmds = eps['joint_cmd']
    eef_poses_left = eps['eef_poses_left']
    eef_poses_euler_left = []
    for eef_pose in eef_poses_left:
        x, y, z, qx, qy, qz, qw = eef_pose
        # convert quaternion to roll, pitch, yaw
        r = R.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = r.as_euler('xyz')
        eef_poses_euler_left.append([x, y, z, roll, pitch, yaw])

    eef_poses_euler_left = np.asarray(eef_poses_euler_left)

    # right
    eef_poses_right = eps['eef_poses_right']
    eef_poses_euler_right = []
    for eef_pose in eef_poses_right:
        x, y, z, qx, qy, qz, qw = eef_pose
        # convert quaternion to roll, pitch, yaw
        r = R.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = r.as_euler('xyz')
        eef_poses_euler_right.append([x, y, z, roll, pitch, yaw])

    eef_poses_euler_right = np.asarray(eef_poses_euler_right)

    if SIDE == 'left':
        eef_poses_euler = eef_poses_euler_left
    else:
        eef_poses_euler = eef_poses_euler_right

    # ========= 伪造 ==== fake ========
    # eef_poses_euler = []
    #
    # for i in range(50):
    #     x = 0.4
    #     y = 0.2 + i * 0.01
    #     z = np.sin(i * 2 * np.pi / 50) * 0.1 + 0.4
    #     roll = 0.0
    #     pitch = -np.deg2rad(90)
    #     yaw = 0.0
    #
    #     if SIDE == 'right':
    #         y = -y
    #     eef_poses_euler.append([x, y, z, roll, pitch, yaw])
    # eef_poses_euler = np.asarray(eef_poses_euler)

    # ============ 第一种ik
    eps['eef_from_ik1'] = []
    eps['joint_from_ik1'] = []
    ik_stg1 = IKNumerical(
        urdf_file='/home/oem/Documents/leju/kuavo-ros-control-wzr/src/kuavo_assets/models/biped_s46/urdf/drake/biped_v3_arm.urdf'
    )
    for eef_pose_euler in eef_poses_euler:
        joint_res = ik_stg1.compute(
            eef_pose_euler[0:3],
            R.from_euler('xyz', eef_pose_euler[3:6]).as_quat(),
            eef_frame=EEF_FRAME,
        )

        if SIDE == 'left':
            joint_res = joint_res.tolist() + [0.0] * 7
        else:
            joint_res = [0.0] * 7 + joint_res.tolist()
        pos_xyz, quat_xyzw = fk_tool.compute(joint_res, EEF_FRAME)

        roll, pitch, yaw = R.from_quat(quat_xyzw).as_euler('xyz')
        eps['eef_from_ik1'].append(np.asarray(pos_xyz + (roll, pitch, yaw)))
        eps['joint_from_ik1'].append(joint_res)

    # ============ 第二种ik
    low_pass_filter = LowPassFilter(cutoff_hz=1.0, dt=0.1)
    eps['eef_from_ik2'] = []
    eps['joint_from_ik2'] = []

    for eef_pose_euler in eef_poses_euler:
        joint_res = IKAnalytical.compute(
            eef_pose_euler[0:3] - np.array([0.0, 0.0, 0.0]),
            R.from_euler('xyz', eef_pose_euler[3:6]).as_quat(),
            eef_frame=EEF_FRAME,
        )

        joint_res = low_pass_filter.update(joint_res)

        if SIDE == 'left':
            joint_res = joint_res.tolist() + [0.0] * 7
            # pos_xyz, quat_xyzw = fk_tool.compute(joint_res, 'zarm_l7_link')
        else:
            joint_res = [0.0] * 7 + joint_res.tolist()
        pos_xyz, quat_xyzw = fk_tool.compute(joint_res, EEF_FRAME)

        roll, pitch, yaw = R.from_quat(quat_xyzw).as_euler('xyz')
        eps['eef_from_ik2'].append(np.asarray(pos_xyz + (roll, pitch, yaw)))
        eps['joint_from_ik2'].append(joint_res)

    # ========= MOE ===============
    eps['eef_from_ik3'] = []
    eps['joint_from_ik3'] = []
    ik_stg1 = IKNumerical()
    for eef_pose_euler in eef_poses_euler:
        ref_q = IKAnalytical.compute(
            eef_pose_euler[0:3],
            R.from_euler('xyz', eef_pose_euler[3:6]).as_quat(),
            eef_frame=EEF_FRAME,
        )

        joint_res = ik_stg1.compute(
            eef_pose_euler[0:3],
            R.from_euler('xyz', eef_pose_euler[3:6]).as_quat(),
            ref_q=ref_q,
            eef_frame=EEF_FRAME,
        )

        if SIDE == 'left':
            joint_res = joint_res.tolist() + [0.0] * 7
        else:
            joint_res = [0.0] * 7 + joint_res.tolist()
        pos_xyz, quat_xyzw = fk_tool.compute(joint_res, EEF_FRAME)

        roll, pitch, yaw = R.from_quat(quat_xyzw).as_euler('xyz')
        eps['eef_from_ik3'].append(np.asarray(pos_xyz + (roll, pitch, yaw)))
        eps['joint_from_ik3'].append(joint_res)

    eps['eef_from_ik1'] = np.asarray(eps['eef_from_ik1'])
    eps['eef_from_ik2'] = np.asarray(eps['eef_from_ik2'])
    eps['eef_from_ik3'] = np.asarray(eps['eef_from_ik3'])

    eps['joint_from_ik1'] = np.asarray(eps['joint_from_ik1'])
    eps['joint_from_ik2'] = np.asarray(eps['joint_from_ik2'])
    eps['joint_from_ik3'] = np.asarray(eps['joint_from_ik3'])

    poses_list = [eef_poses_euler,
                  eps['eef_from_ik1'],
                  eps['eef_from_ik2'],
                  eps['eef_from_ik3']
                  ]
    labels = ["traj_reference",
              "traj_ik_one_stg",
              "traj_ik_sishu",
              "traj_ik_moe"
              ]

    # 每隔 20 个点画一个姿态坐标系，frame_length 控制箭头长度
    plot_trajectories_with_frames(
        poses_list,
        labels=labels,
        frame_step=20,
        frame_length=0.02,
    )

    # plot joint trajectories for joint_from_ik1 (first 7 joints) on same plot
    def plot_joint_trajectories(joint_array, joint_labels=None, title='Joint trajectories (ik_sishu)'):
        if joint_array is None or len(joint_array) == 0:
            print('No joint data to plot')
            return
        arr = np.asarray(joint_array)
        # If array has more than 7 columns, take first 7 as robot joints
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n_steps, n_cols = arr.shape
        n_joints = min(7, n_cols)
        t = np.arange(n_steps)
        plt.figure(figsize=(10, 5))
        for j in range(n_joints):
            label = joint_labels[j] if joint_labels is not None and j < len(joint_labels) else f'joint_{j + 1}'
            plt.plot(t, arr[:, j], label=label)
        plt.xlabel('time step')
        plt.ylabel('joint angle (rad)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if SIDE == 'left':
        plot_joint_trajectories(eps['joint_from_ik2'][:, 0:7],
                                joint_labels=['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
                                              'joint_7'])
    else:
        plot_joint_trajectories(eps['joint_from_ik2'][:, 7:14],
                                joint_labels=['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
                                              'joint_7'])


if __name__ == "__main__":
    read_json_and_do_ik()
