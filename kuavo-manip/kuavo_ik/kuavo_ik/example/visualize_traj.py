import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def rpy_to_rot(roll, pitch, yaw):
    """
    Drake 一样的定义：
    R = Rz(yaw) * Ry(pitch) * Rx(roll)
    roll, pitch, yaw: scalar (rad)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ])

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ])

    R = Rz @ Ry @ Rx
    return R


def generate_spiral_trajectory(
    num_points=200,
    radius=0.5,
    z_scale=0.05,
    yaw_scale=1.0,
    pos_offset=(0.0, 0.0, 0.0),
    noise_std=0.0,
):
    """
    生成一条假的 6D 轨迹:
    - 位置: 空间螺旋线
    - 姿态: roll = pitch = 0, yaw = yaw_scale * t
    - pos_offset: 整体平移 (dx, dy, dz)
    - noise_std: 每个点加一点高斯噪声 (位置噪声)
    返回: (N, 6) [x, y, z, roll, pitch, yaw]
    """
    t = np.linspace(0, 4 * np.pi, num_points)

    # 螺旋位置
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = z_scale * t

    # 平移
    dx, dy, dz = pos_offset
    x = x + dx
    y = y + dy
    z = z + dz

    # 加一点噪声
    if noise_std > 0:
        x += np.random.randn(num_points) * noise_std
        y += np.random.randn(num_points) * noise_std
        z += np.random.randn(num_points) * noise_std

    # 姿态
    roll = np.zeros_like(t)
    pitch = np.zeros_like(t)
    yaw = yaw_scale * t

    poses = np.stack([x, y, z, roll, pitch, yaw], axis=1)
    return poses


def plot_trajectories_with_frames(poses_list, labels=None,
                                  frame_step=20, frame_length=0.1):
    """
    在一张图里画多条 6D 轨迹：
    - 用点画位置
    - 用线连接相邻点
    - 每隔 frame_step 个点画姿态坐标系 (X=红, Y=绿, Z=蓝)
    """
    if labels is None:
        labels = [f"traj_{i}" for i in range(len(poses_list))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    all_x, all_y, all_z = [], [], []


    for idx, poses in enumerate(poses_list):
        # print(poses)
        x = poses[:, 0]
        y = poses[:, 1]
        z = poses[:, 2]
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)

        color = colors[idx % len(colors)]
        label = labels[idx]

        # 点
        ax.scatter(x, y, z, s=8, alpha=0.7, color=color)

        # 线（关键：把点连起来）
        ax.plot(x, y, z, linewidth=1.5, alpha=0.7, color=color, label=label)

        # 起点 / 终点
        ax.scatter(x[0], y[0], z[0], s=50, marker="o", color=color)
        ax.scatter(x[-1], y[-1], z[-1], s=50, marker="x", color=color)

        # 姿态坐标系
        for i in range(0, len(poses), frame_step):
            px, py, pz, roll, pitch, yaw = poses[i]
            R = rpy_to_rot(roll, pitch, yaw)
            origin = np.array([px, py, pz])

            x_axis = R[:, 0] * frame_length
            y_axis = R[:, 1] * frame_length
            z_axis = R[:, 2] * frame_length

            ax.quiver(origin[0], origin[1], origin[2],
                      x_axis[0], x_axis[1], x_axis[2],
                      arrow_length_ratio=0.2, color="r")
            ax.quiver(origin[0], origin[1], origin[2],
                      y_axis[0], y_axis[1], y_axis[2],
                      arrow_length_ratio=0.2, color="g")
            ax.quiver(origin[0], origin[1], origin[2],
                      z_axis[0], z_axis[1], z_axis[2],
                      arrow_length_ratio=0.2, color="b")

    # 等比例坐标轴
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    max_range = np.array([
        all_x.max() - all_x.min(),
        all_y.max() - all_y.min(),
        all_z.max() - all_z.min()
    ]).max() / 2.0
    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Comparison of Trajectories with Orientation Frames")
    ax.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    np.random.seed(0)

    num_points = 200

    traj1 = generate_spiral_trajectory(
        num_points=num_points,
        radius=0.5,
        z_scale=0.05,
        yaw_scale=1.0,
        pos_offset=(0.0, 0.0, 0.0),
        noise_std=0.0,
    )

    traj2 = generate_spiral_trajectory(
        num_points=num_points,
        radius=0.55,
        z_scale=0.05,
        yaw_scale=1.05,
        pos_offset=(0.05, -0.05, 0.0),
        noise_std=0.01,
    )

    traj3 = generate_spiral_trajectory(
        num_points=num_points,
        radius=0.5,
        z_scale=0.06,
        yaw_scale=0.95,
        pos_offset=(-0.05, 0.05, 0.02),
        noise_std=0.02,
    )

    poses_list = [traj1, traj2, traj3]
    labels = ["traj_base", "traj_offset_1", "traj_offset_2"]

    plot_trajectories_with_frames(
        poses_list,
        labels=labels,
        frame_step=20,
        frame_length=0.08,
    )
