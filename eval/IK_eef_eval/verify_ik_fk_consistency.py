"""
验证IK和FK一致性的脚本

从LeRobot dataset v3.0读取数据，使用IKAnalytical解IK，然后用Pinocchio FK验证一致性。
"""
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
# 默认使用非GUI后端（Agg），用于保存图片
# 如果需要交互式显示，会在运行时根据参数切换后端
matplotlib.use('Agg')  # 使用Agg后端，不需要GUI库
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import argparse
import time
from tqdm import tqdm

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import pinocchio as pin

# 导入IK相关函数
from kuavo_ik.ik_library import IKAnalytical

# 导入配置（如果可用）
try:
    from configs.config import ROBOT_VERSION
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: configs.config not available. Using default ROBOT_VERSION='5_wheel'")
    CONFIG_AVAILABLE = False
    ROBOT_VERSION = "5_wheel"


def reconstruct_rotation_matrix_6d(rotation_6d: np.ndarray) -> np.ndarray:
    """
    从6D表示重构3x3旋转矩阵
    使用Gram-Schmidt正交化确保是有效的旋转矩阵
    
    Args:
        rotation_6d: 6维向量 [R11, R21, R31, R12, R22, R32]
        
    Returns:
        3x3旋转矩阵
    """
    # 提取前两列
    col1 = rotation_6d[:3]  # [R11, R21, R31]
    col2 = rotation_6d[3:6]  # [R12, R22, R32]
    
    # Gram-Schmidt正交化
    # 第一列归一化
    col1_norm = np.linalg.norm(col1)
    if col1_norm < 1e-8:
        col1_normalized = np.array([1.0, 0.0, 0.0])
    else:
        col1_normalized = col1 / col1_norm
    
    # 第二列正交化并归一化
    col2_projected = col2 - np.dot(col2, col1_normalized) * col1_normalized
    col2_norm = np.linalg.norm(col2_projected)
    if col2_norm < 1e-8:
        col2_normalized = np.array([0.0, 1.0, 0.0])
    else:
        col2_normalized = col2_projected / col2_norm
    
    # 第三列通过叉积得到
    col3_normalized = np.cross(col1_normalized, col2_normalized)
    
    # 组合成旋转矩阵
    rotation_matrix = np.column_stack([col1_normalized, col2_normalized, col3_normalized])
    
    return rotation_matrix


def rotation_matrix_to_6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    将3x3旋转矩阵转换为6D表示
    
    Args:
        rotation_matrix: 3x3旋转矩阵
        
    Returns:
        6维向量 [R11, R21, R31, R12, R22, R32]
    """
    # 提取前两列
    col1 = rotation_matrix[:, 0]  # [R11, R21, R31]
    col2 = rotation_matrix[:, 1]  # [R12, R22, R32]
    
    # 组合成6D表示
    rotation_6d = np.concatenate([col1, col2])
    
    return rotation_6d


class PinocchioFK:
    """使用 Pinocchio 进行正向运动学计算"""
    def __init__(self, 
                 urdf_path: str, 
                 arm_start_idx: int = 0,
                 arm_end_idx: int = 14,
                 left_eef_frame: str = 'zarm_l7_link', 
                 right_eef_frame: str = 'zarm_r7_link',
                 shared_reference_frame: str = 'base_link'):
        """
        初始化 Pinocchio FK 计算器
        
        Args:
            urdf_path: URDF 文件路径
            arm_start_idx: 手臂关节在全关节向量中的起始索引（默认0，因为输入已经是14维关节角数组）
            arm_end_idx: 手臂关节在全关节向量中的结束索引（默认14，因为输入已经是14维关节角数组）
            left_eef_frame: 左臂末端执行器 frame 名称
            right_eef_frame: 右臂末端执行器 frame 名称
            shared_reference_frame: 共享参考 frame 名称（用于计算相对位姿）
        """
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()
        
        # 检查URDF的实际关节数量
        actual_nq = self.pin_model.nq
        print(f"   URDF model has {actual_nq} joints (nq)")
        
        # 如果URDF只有手臂关节（14个），则使用 [0:14]
        # 如果URDF包含完整机器人（更多关节），则使用传入的索引
        if actual_nq == 14:
            # URDF只包含手臂，直接使用 [0:14]
            self.arm_start_idx = 0
            self.arm_end_idx = 14
            print(f"   ⚠️  URDF only contains arms (14 joints), using indices [0:14] instead of [{arm_start_idx}:{arm_end_idx}]")
        else:
            # URDF包含完整机器人，使用传入的索引
            self.arm_start_idx = arm_start_idx
            self.arm_end_idx = arm_end_idx
            # 验证索引是否有效
            if self.arm_end_idx > actual_nq:
                raise ValueError(f"arm_end_idx ({arm_end_idx}) exceeds model nq ({actual_nq})")
            if self.arm_end_idx - self.arm_start_idx != 14:
                raise ValueError(f"Expected 14 arm joints, but indices [{self.arm_start_idx}:{self.arm_end_idx}] gives {self.arm_end_idx - self.arm_start_idx} joints")
        
        # 获取 frame IDs
        self.left_eef_frame_id = self.pin_model.getFrameId(left_eef_frame)
        self.right_eef_frame_id = self.pin_model.getFrameId(right_eef_frame)
        self.shared_reference_frame_id = self.pin_model.getFrameId(shared_reference_frame)
        
        # 初始化全关节位置向量
        self.wb_joint_pos = np.zeros(self.pin_model.nq)
        
        print(f"✅ Pinocchio FK initialized:")
        print(f"   URDF: {urdf_path}")
        print(f"   Model nq: {actual_nq}")
        print(f"   Arm joint indices: {self.arm_start_idx} to {self.arm_end_idx-1} (total: {self.arm_end_idx - self.arm_start_idx} joints)")
        print(f"   Left EEF frame: {left_eef_frame} (ID: {self.left_eef_frame_id})")
        print(f"   Right EEF frame: {right_eef_frame} (ID: {self.right_eef_frame_id})")
        print(f"   Reference frame: {shared_reference_frame} (ID: {self.shared_reference_frame_id})")
    
    def __call__(self, arm_joint_pos: np.ndarray):
        """
        计算左右臂的末端执行器位姿
        
        Args:
            arm_joint_pos: numpy array of shape (14,) - 左臂7个关节 + 右臂7个关节
        
        Returns:
            tuple: (left_eef_pose, right_eef_pose)
                - left_eef_pose: numpy array of shape (9,) - [x, y, z, R11, R21, R31, R12, R22, R32]
                - right_eef_pose: numpy array of shape (9,) - [x, y, z, R11, R21, R31, R12, R22, R32]
        """
        # 确保输入是 numpy array 且长度为14
        arm_joint_pos = np.array(arm_joint_pos, dtype=np.float64)
        if len(arm_joint_pos) != 14:
            raise ValueError(f"Expected 14 joint angles, got {len(arm_joint_pos)}")
        
        # 将手臂关节角度设置到全关节向量中
        self.wb_joint_pos[self.arm_start_idx:self.arm_end_idx] = arm_joint_pos
        
        # 执行正向运动学
        pin.forwardKinematics(self.pin_model, self.pin_data, self.wb_joint_pos)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        # 获取参考 frame 和末端执行器 frame 的位姿
        shared_reference_pose = self.pin_data.oMf[self.shared_reference_frame_id]
        left_eef_frame_pose = self.pin_data.oMf[self.left_eef_frame_id]
        right_eef_frame_pose = self.pin_data.oMf[self.right_eef_frame_id]
        
        # 计算相对于参考 frame 的位姿
        left_eef_pose = shared_reference_pose.inverse() * left_eef_frame_pose
        right_eef_pose = shared_reference_pose.inverse() * right_eef_frame_pose
        
        # 转换为 6D 旋转表示
        return self._se3_to_xyz_rot6d(left_eef_pose), self._se3_to_xyz_rot6d(right_eef_pose)
    
    def _se3_to_xyz_rot6d(self, se3) -> np.ndarray:
        """
        将 Pinocchio SE3 转换为 9维数组: [x, y, z, R11, R21, R31, R12, R22, R32]
        
        Args:
            se3: Pinocchio SE3 对象
        
        Returns:
            numpy array of shape (9,)
        """
        trans = se3.translation
        rot = se3.rotation
        return np.array([
            trans[0], trans[1], trans[2],  # x, y, z
            rot[0, 0], rot[1, 0], rot[2, 0],  # R11, R21, R31 (第一列)
            rot[0, 1], rot[1, 1], rot[2, 1],  # R12, R22, R32 (第二列)
        ], dtype=np.float32)


def rot6d_to_quaternion_xyzw(rot_6d: np.ndarray) -> np.ndarray:
    """
    将6D旋转表示转换为四元数 (x, y, z, w)
    
    Args:
        rot_6d: 6维向量 [R11, R21, R31, R12, R22, R32]
        
    Returns:
        四元数 [x, y, z, w]
    """
    # 重构旋转矩阵
    R_mat = reconstruct_rotation_matrix_6d(rot_6d)
    
    # 转换为四元数
    quat = R.from_matrix(R_mat).as_quat()  # scipy返回的是 (x, y, z, w)
    
    return quat


def rotation_matrix_to_euler_xyz(R_mat: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为欧拉角 (x, y, z)
    
    Args:
        R_mat: 3x3旋转矩阵
        
    Returns:
        欧拉角 [roll, pitch, yaw] (弧度)
    """
    euler = R.from_matrix(R_mat).as_euler('xyz')
    return euler


def load_dataset(dataset_path: str, episode_idx: int = None):
    """
    加载LeRobot dataset v3.0
    
    参考 eval_on_dataset_lowpass_ik.py 的实现方式
    
    Args:
        dataset_path: 数据集路径（应该是包含meta/和data/目录的数据集根目录）
        episode_idx: 要加载的episode索引（可选）
        
    Returns:
        LeRobotDataset对象
    """
    # 对于本地数据集，repo_id应该是一个字符串标识符（不包含"/"）
    # 使用数据集路径的最后一部分作为标识符，或者使用"local"
    dataset_name = Path(dataset_path).name if dataset_path else "local"
    
    print(f"📂 Loading dataset from {dataset_path}")
    if episode_idx is not None:
        print(f"📹 Episode: {episode_idx}")
    
    # 注意：LeRobotDataset的episodes参数主要用于下载时选择文件
    # 但在加载后需要手动过滤数据，因为多个episodes可能存储在同一个parquet文件中
    episodes = [episode_idx] if episode_idx is not None else None
    dataset = LeRobotDataset(repo_id=dataset_name, root=dataset_path, episodes=episodes)
    
    print(f"✅ Dataset loaded: {len(dataset)} frames, {dataset.num_episodes} episodes")
    return dataset


def extract_eef_poses(state: np.ndarray, action: np.ndarray):
    """
    从state和action中提取eef pose
    
    Args:
        state: state数组，shape (20,) - left_eef(9) + right_eef(9) + gripper(2)
        action: action数组，shape (20,) - left_eef(9) + right_eef(9) + gripper(2)
        
    Returns:
        dict包含:
            - state_left_eef: (9,) - [x, y, z, R11, R21, R31, R12, R22, R32]
            - state_right_eef: (9,)
            - action_left_eef: (9,)
            - action_right_eef: (9,)
    """
    # state格式: left_eef(9) + right_eef(9) + gripper(2) = 20维
    state_left_eef = state[0:9]
    state_right_eef = state[9:18]
    
    # action格式: left_eef(9) + right_eef(9) + gripper(2) = 20维
    action_left_eef = action[0:9]
    action_right_eef = action[9:18]
    
    return {
        'state_left_eef': state_left_eef,
        'state_right_eef': state_right_eef,
        'action_left_eef': action_left_eef,
        'action_right_eef': action_right_eef,
    }


def solve_ik_and_verify_fk(
    action_left_eef: np.ndarray,
    action_right_eef: np.ndarray,
    fk_getter: PinocchioFK,
    model_type: str = '45'
):
    """
    使用IKAnalytical解IK，然后用Pinocchio FK验证
    
    Args:
        action_left_eef: action中的左臂eef pose (9维)
        action_right_eef: action中的右臂eef pose (9维)
        fk_getter: PinocchioFK对象
        model_type: 机器人型号 ('45', '46', '60')
        
    Returns:
        dict包含:
            - ik_left_joints: 左臂关节角 (7,)
            - ik_right_joints: 右臂关节角 (7,)
            - fk_left_eef: FK得到的左臂eef pose (9,)
            - fk_right_eef: FK得到的右臂eef pose (9,)
    """
    # 提取位置和旋转
    left_pos = action_left_eef[:3]
    left_rot6d = action_left_eef[3:9]
    right_pos = action_right_eef[:3]
    right_rot6d = action_right_eef[3:9]
    
    # 转换为四元数用于IK
    left_quat = rot6d_to_quaternion_xyzw(left_rot6d)
    right_quat = rot6d_to_quaternion_xyzw(right_rot6d)
    
    # 解IK
    try:
        ik_left_joints = IKAnalytical.compute(
            eef_pos=left_pos,
            eef_quat_xyzw=left_quat,
            eef_frame='zarm_l7_link',
            model_type=model_type,
            limit=True
        )
    except Exception as e:
        print(f"⚠️  Left arm IK failed: {e}")
        ik_left_joints = np.zeros(7)
    
    try:
        ik_right_joints = IKAnalytical.compute(
            eef_pos=right_pos,
            eef_quat_xyzw=right_quat,
            eef_frame='zarm_r7_link',
            model_type=model_type,
            limit=True
        )
    except Exception as e:
        print(f"⚠️  Right arm IK failed: {e}")
        ik_right_joints = np.zeros(7)
    
    # 组合关节角用于FK
    joint_angles = np.concatenate([ik_left_joints, ik_right_joints])  # (14,)
    
    # 使用Pinocchio FK验证
    fk_left_eef, fk_right_eef = fk_getter(joint_angles)
    
    return {
        'ik_left_joints': ik_left_joints,
        'ik_right_joints': ik_right_joints,
        'fk_left_eef': fk_left_eef,
        'fk_right_eef': fk_right_eef,
    }


def visualize_3d_trajectory(
    state_left: np.ndarray,
    state_right: np.ndarray,
    action_left: np.ndarray,
    action_right: np.ndarray,
    fk_left: np.ndarray,
    fk_right: np.ndarray,
    save_path: str = None,
    interactive: bool = False
):
    """
    3D可视化末端轨迹
    
    Args:
        state_left: state中的左臂eef轨迹 (N, 9)
        state_right: state中的右臂eef轨迹 (N, 9)
        action_left: action中的左臂eef轨迹 (N, 9)
        action_right: action中的右臂eef轨迹 (N, 9)
        fk_left: FK得到的左臂eef轨迹 (N, 9)
        fk_right: FK得到的右臂eef轨迹 (N, 9)
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(16, 8))
    
    # 左臂
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(state_left[:, 0], state_left[:, 1], state_left[:, 2], 
             'b-', label='State EEF', linewidth=2, alpha=0.7)
    ax1.plot(action_left[:, 0], action_left[:, 1], action_left[:, 2], 
             'g-', label='Action EEF', linewidth=2, alpha=0.7)
    ax1.plot(fk_left[:, 0], fk_left[:, 1], fk_left[:, 2], 
             'r--', label='IK->FK EEF', linewidth=2, alpha=0.7)
    ax1.scatter(state_left[0, 0], state_left[0, 1], state_left[0, 2], 
                c='blue', s=100, marker='o', label='Start')
    ax1.scatter(state_left[-1, 0], state_left[-1, 1], state_left[-1, 2], 
                c='red', s=100, marker='s', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Left Arm EEF Trajectory (3D)')
    ax1.legend()
    ax1.grid(True)
    
    # 右臂
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(state_right[:, 0], state_right[:, 1], state_right[:, 2], 
             'b-', label='State EEF', linewidth=2, alpha=0.7)
    ax2.plot(action_right[:, 0], action_right[:, 1], action_right[:, 2], 
             'g-', label='Action EEF', linewidth=2, alpha=0.7)
    ax2.plot(fk_right[:, 0], fk_right[:, 1], fk_right[:, 2], 
             'r--', label='IK->FK EEF', linewidth=2, alpha=0.7)
    ax2.scatter(state_right[0, 0], state_right[0, 1], state_right[0, 2], 
                c='blue', s=100, marker='o', label='Start')
    ax2.scatter(state_right[-1, 0], state_right[-1, 1], state_right[-1, 2], 
                c='red', s=100, marker='s', label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Right Arm EEF Trajectory (3D)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved 3D visualization to {save_path}")
    
    # 显示交互式图表
    backend = matplotlib.get_backend()
    is_interactive_backend = backend.lower() not in ['agg', 'pdf', 'svg', 'ps']
    
    if interactive and is_interactive_backend:
        plt.show(block=True)  # block=True 会等待窗口关闭
        print("💡 Interactive 3D plot closed.")
    elif is_interactive_backend:
        plt.show(block=False)
        print("💡 Interactive 3D plot opened. You can zoom, pan, and rotate the view.")
        print("   Close the window to continue or press Ctrl+C to exit.")
    else:
        plt.close()  # 关闭图形以释放内存
        if not save_path:
            print("⚠️  Using non-interactive backend. Use --interactive flag to enable interactive display.")


def visualize_position_comparison(
    state_left: np.ndarray,
    state_right: np.ndarray,
    action_left: np.ndarray,
    action_right: np.ndarray,
    fk_left: np.ndarray,
    fk_right: np.ndarray,
    save_path: str = None,
    interactive: bool = False
):
    """
    可视化位置对比 (x, y, z)
    
    Args:
        state_left: state中的左臂eef轨迹 (N, 9)
        state_right: state中的右臂eef轨迹 (N, 9)
        action_left: action中的左臂eef轨迹 (N, 9)
        action_right: action中的右臂eef轨迹 (N, 9)
        fk_left: FK得到的左臂eef轨迹 (N, 9)
        fk_right: FK得到的右臂eef轨迹 (N, 9)
        save_path: 保存路径
    """
    num_frames = state_left.shape[0]
    frame_indices = np.arange(num_frames)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 位置名称
    pos_names = ['X', 'Y', 'Z']
    
    for arm_idx, (arm_name, state_arm, action_arm, fk_arm) in enumerate([
        ('Left', state_left, action_left, fk_left),
        ('Right', state_right, action_right, fk_right)
    ]):
        for pos_idx, pos_name in enumerate(pos_names):
            ax = axes[arm_idx * 3 + pos_idx]
            
            ax.plot(frame_indices, state_arm[:, pos_idx], 
                   'b-', label='State', linewidth=2, alpha=0.7)
            ax.plot(frame_indices, action_arm[:, pos_idx], 
                   'g-', label='Action', linewidth=2, alpha=0.7)
            ax.plot(frame_indices, fk_arm[:, pos_idx], 
                   'r--', label='IK->FK', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Frame Number')
            ax.set_ylabel(f'{pos_name} Position (m)')
            ax.set_title(f'{arm_name} Arm - {pos_name} Position')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved position comparison to {save_path}")
    
    # 显示交互式图表
    backend = matplotlib.get_backend()
    is_interactive_backend = backend.lower() not in ['agg', 'pdf', 'svg', 'ps']
    
    if interactive and is_interactive_backend:
        plt.show(block=True)  # block=True 会等待窗口关闭
        print("💡 Interactive position plot closed.")
    elif is_interactive_backend:
        plt.show(block=False)
        print("💡 Interactive position plot opened. You can zoom and pan.")
        print("   Close the window to continue or press Ctrl+C to exit.")
    else:
        plt.close()  # 关闭图形以释放内存
        if not save_path:
            print("⚠️  Using non-interactive backend. Use --interactive flag to enable interactive display.")


def visualize_euler_comparison(
    state_left: np.ndarray,
    state_right: np.ndarray,
    action_left: np.ndarray,
    action_right: np.ndarray,
    fk_left: np.ndarray,
    fk_right: np.ndarray,
    save_path: str = None,
    interactive: bool = False
):
    """
    可视化欧拉角对比 (roll, pitch, yaw)
    
    Args:
        state_left: state中的左臂eef轨迹 (N, 9)
        state_right: state中的右臂eef轨迹 (N, 9)
        action_left: action中的左臂eef轨迹 (N, 9)
        action_right: action中的右臂eef轨迹 (N, 9)
        fk_left: FK得到的左臂eef轨迹 (N, 9)
        fk_right: FK得到的右臂eef轨迹 (N, 9)
        save_path: 保存路径
    """
    num_frames = state_left.shape[0]
    frame_indices = np.arange(num_frames)
    
    # 计算欧拉角
    def compute_euler_trajectory(eef_trajectory):
        """从eef轨迹计算欧拉角轨迹"""
        euler_traj = []
        for eef in eef_trajectory:
            rot6d = eef[3:9]
            R_mat = reconstruct_rotation_matrix_6d(rot6d)
            euler = rotation_matrix_to_euler_xyz(R_mat)
            euler_traj.append(euler)
        return np.array(euler_traj)
    
    state_left_euler = compute_euler_trajectory(state_left)
    state_right_euler = compute_euler_trajectory(state_right)
    action_left_euler = compute_euler_trajectory(action_left)
    action_right_euler = compute_euler_trajectory(action_right)
    fk_left_euler = compute_euler_trajectory(fk_left)
    fk_right_euler = compute_euler_trajectory(fk_right)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 欧拉角名称
    euler_names = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    
    for arm_idx, (arm_name, state_euler, action_euler, fk_euler) in enumerate([
        ('Left', state_left_euler, action_left_euler, fk_left_euler),
        ('Right', state_right_euler, action_right_euler, fk_right_euler)
    ]):
        for euler_idx, euler_name in enumerate(euler_names):
            ax = axes[arm_idx * 3 + euler_idx]
            
            ax.plot(frame_indices, np.degrees(state_euler[:, euler_idx]), 
                   'b-', label='State', linewidth=2, alpha=0.7)
            ax.plot(frame_indices, np.degrees(action_euler[:, euler_idx]), 
                   'g-', label='Action', linewidth=2, alpha=0.7)
            ax.plot(frame_indices, np.degrees(fk_euler[:, euler_idx]), 
                   'r--', label='IK->FK', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Frame Number')
            ax.set_ylabel(f'{euler_name} (degrees)')
            ax.set_title(f'{arm_name} Arm - {euler_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved euler comparison to {save_path}")
    
    # 显示交互式图表
    backend = matplotlib.get_backend()
    is_interactive_backend = backend.lower() not in ['agg', 'pdf', 'svg', 'ps']
    
    if interactive and is_interactive_backend:
        plt.show(block=True)  # block=True 会等待窗口关闭
        print("💡 Interactive euler plot closed.")
    elif is_interactive_backend:
        plt.show(block=False)
        print("💡 Interactive euler plot opened. You can zoom and pan.")
        print("   Close the window to continue or press Ctrl+C to exit.")
    else:
        plt.close()  # 关闭图形以释放内存
        if not save_path:
            print("⚠️  Using non-interactive backend. Use --interactive flag to enable interactive display.")


def _se3_to_viser_pose(se3_obj):
    """将Pinocchio SE3转换为viser使用的(position, wxyz)。"""
    translation = se3_obj.translation.astype(np.float32)
    quat_xyzw = R.from_matrix(se3_obj.rotation).as_quat().astype(np.float32)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    return translation, quat_wxyz


def visualize_with_viser(
    urdf_path: str,
    mesh_dir: str,
    model_type: str,
    state_left: np.ndarray,
    state_right: np.ndarray,
    action_left: np.ndarray,
    action_right: np.ndarray,
    fk_left: np.ndarray,
    fk_right: np.ndarray,
    ik_joint_traj: np.ndarray,
    host: str = "0.0.0.0",
    port: int = 8080,
    loop: bool = False,
    fps: float = 20.0,
):
    """
    使用viser进行3D可视化，并显示URDF网格模型。
    说明：
    - 轨迹全部显示（state/action/IK->FK）
    - URDF模型按 IK 关节轨迹播放
    """
    try:
        import viser
        import trimesh
    except ImportError as e:
        print("❌ viser/trimesh 未安装，无法使用 --viser 可视化")
        print(f"   详细错误: {e}")
        print("   安装命令: pip install viser trimesh")
        return

    pin_model = pin.buildModelFromUrdf(urdf_path)
    pin_data = pin_model.createData()
    package_dirs = [mesh_dir, str(Path(urdf_path).parent)]
    geom_model = pin.buildGeomFromUrdf(pin_model, urdf_path, pin.GeometryType.VISUAL, package_dirs)
    geom_data = pin.GeometryData(geom_model)

    server = viser.ViserServer(host=host, port=port)
    print(f"✅ Viser started at: http://{host}:{port}")
    print(f"   Model type: {model_type}")
    print(f"   URDF: {urdf_path}")
    print(f"   Mesh dir: {mesh_dir}")

    # 绘制轨迹（左右手都放在一个场景内）
    server.scene.add_spline_catmull_rom(
        "/traj/left/state",
        positions=state_left[:, :3],
        color=(50, 120, 255),
        line_width=2.0,
    )
    server.scene.add_spline_catmull_rom(
        "/traj/left/action",
        positions=action_left[:, :3],
        color=(50, 200, 50),
        line_width=2.0,
    )
    server.scene.add_spline_catmull_rom(
        "/traj/left/ik_fk",
        positions=fk_left[:, :3],
        color=(255, 50, 50),
        line_width=2.0,
    )
    server.scene.add_spline_catmull_rom(
        "/traj/right/state",
        positions=state_right[:, :3],
        color=(100, 180, 255),
        line_width=2.0,
    )
    server.scene.add_spline_catmull_rom(
        "/traj/right/action",
        positions=action_right[:, :3],
        color=(100, 255, 100),
        line_width=2.0,
    )
    server.scene.add_spline_catmull_rom(
        "/traj/right/ik_fk",
        positions=fk_right[:, :3],
        color=(255, 120, 120),
        line_width=2.0,
    )

    # 标注起点终点
    server.scene.add_point_cloud(
        "/markers/start",
        points=np.array([state_left[0, :3], state_right[0, :3]], dtype=np.float32),
        colors=np.array([[0, 0, 255], [0, 0, 255]], dtype=np.uint8),
        point_size=0.015,
    )
    server.scene.add_point_cloud(
        "/markers/end",
        points=np.array([state_left[-1, :3], state_right[-1, :3]], dtype=np.float32),
        colors=np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8),
        point_size=0.015,
    )

    # 预加载URDF可视mesh
    mesh_handles = []
    mesh_cache = {}
    for geom_idx, geom_obj in enumerate(geom_model.geometryObjects):
        mesh_path = getattr(geom_obj, "meshPath", "")
        if not mesh_path:
            continue
        mesh_path = str(mesh_path)
        if not Path(mesh_path).exists():
            candidate = Path(mesh_dir) / Path(mesh_path).name
            if candidate.exists():
                mesh_path = str(candidate)
            else:
                continue
        if mesh_path not in mesh_cache:
            try:
                mesh_cache[mesh_path] = trimesh.load(mesh_path, force="mesh")
            except Exception:
                continue
        trimesh_mesh = mesh_cache[mesh_path].copy()
        mesh_scale = np.asarray(getattr(geom_obj, "meshScale", np.ones(3)), dtype=np.float32)
        trimesh_mesh.vertices = trimesh_mesh.vertices * mesh_scale.reshape(1, 3)
        handle = server.scene.add_mesh_trimesh(f"/robot/visual_{geom_idx}", trimesh_mesh)
        mesh_handles.append((geom_idx, handle))

    if not mesh_handles:
        print("⚠️ 未加载到任何URDF mesh，请检查 --urdf-path 与 --mesh-dir")
    else:
        print(f"✅ 已加载 {len(mesh_handles)} 个URDF visual mesh")

    # 关节映射：将14维输入映射到pin模型前14个可动关节
    q0 = pin.neutral(pin_model)
    non_fixed_joint_ids = []
    for joint_id in range(1, pin_model.njoints):
        if pin_model.joints[joint_id].nq > 0:
            non_fixed_joint_ids.append(joint_id)
    if len(non_fixed_joint_ids) < 14:
        print(f"⚠️ URDF可动关节数量不足14（实际={len(non_fixed_joint_ids)}），将尽量映射")

    q_indices = [pin_model.idx_qs[jid] for jid in non_fixed_joint_ids[:14]]
    for q_idx in q_indices:
        q0[q_idx] = 0.0

    def render_frame(frame_id: int):
        frame_id = int(np.clip(frame_id, 0, ik_joint_traj.shape[0] - 1))
        q = q0.copy()
        joints14 = ik_joint_traj[frame_id]
        for i, q_idx in enumerate(q_indices):
            if i < len(joints14):
                q[q_idx] = float(joints14[i])
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        pin.updateGeometryPlacements(pin_model, pin_data, geom_model, geom_data)

        for geom_idx, handle in mesh_handles:
            T_world_geom = geom_data.oMg[geom_idx]
            pos, wxyz = _se3_to_viser_pose(T_world_geom)
            handle.position = pos
            handle.wxyz = wxyz

    # GUI控制
    frame_slider = server.gui.add_slider(
        "Frame",
        min=0,
        max=int(ik_joint_traj.shape[0] - 1),
        step=1,
        initial_value=0,
    )
    play_toggle = server.gui.add_checkbox("Play", initial_value=False)
    loop_toggle = server.gui.add_checkbox("Loop", initial_value=loop)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=int(fps))

    @frame_slider.on_update
    def _(_event):
        render_frame(frame_slider.value)

    render_frame(0)
    print("💡 打开网页后可拖动 Frame，或勾选 Play 自动播放。Ctrl+C 退出。")
    try:
        while True:
            if play_toggle.value:
                nxt = frame_slider.value + 1
                if nxt > frame_slider.max:
                    if loop_toggle.value:
                        nxt = frame_slider.min
                    else:
                        nxt = frame_slider.max
                        play_toggle.value = False
                frame_slider.value = int(nxt)
            time.sleep(1.0 / max(1, int(fps_slider.value)))
    except KeyboardInterrupt:
        print("\n🛑 Viser visualization stopped.")


def main():
    parser = argparse.ArgumentParser(description='验证IK和FK一致性')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='LeRobot dataset v3.0路径')
    parser.add_argument('--urdf-path', type=str,
                       default='/home/lab/kuavo-manip/lerobot_datasets/utils/biped_s60_only_arm.urdf',
                       help='URDF文件路径')
    parser.add_argument('--model-type', type=str, default='45',
                       choices=['45', '46', '60', '62'],
                       help='机器人型号')
    parser.add_argument('--episode-idx', type=int, default=0,
                       help='要分析的episode索引')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='最大分析帧数（None表示全部）')
    parser.add_argument('--output-dir', type=str, default='./ik_fk_verification_results',
                       help='输出目录')
    parser.add_argument('--robot-version', type=str, default=None,
                       choices=['5_wheel', '4_pro'],
                       help='机器人版本（用于确定关节索引）。如果不提供，将从configs.config导入ROBOT_VERSION')
    parser.add_argument('--interactive', action='store_true',
                       help='显示交互式图表（默认：保存静态图片后自动关闭）')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存图片文件，只显示交互式图表')
    parser.add_argument('--viser', action='store_true',
                       help='启用viser 3D网页可视化（含URDF+mesh）')
    parser.add_argument('--viser-host', type=str, default='0.0.0.0',
                       help='viser服务监听地址（默认0.0.0.0）')
    parser.add_argument('--viser-port', type=int, default=8080,
                       help='viser服务端口（默认8080）')
    parser.add_argument('--mesh-dir', type=str,
                       default='/home/lab/kuavo-manip/lerobot_datasets/utils/biped_s62_meshes',
                       help='URDF mesh目录（用于viser加载visual mesh）')
    
    args = parser.parse_args()
    
    # 如果需要交互式显示，尝试切换到GUI后端
    if args.interactive or args.no_save:
        try:
            # 尝试切换到Qt5后端
            matplotlib.use('Qt5Agg', force=True)
            print("✅ Using Qt5Agg backend for interactive display")
        except Exception:
            try:
                # 如果Qt5不可用，尝试TkAgg
                matplotlib.use('TkAgg', force=True)
                print("✅ Using TkAgg backend for interactive display")
            except Exception:
                print("⚠️  Warning: No GUI backend available (PyQt5/PySide2/Tkinter not installed)")
                print("   Falling back to non-interactive mode. Images will be saved only.")
                args.interactive = False
                if args.no_save:
                    print("   Error: --no-save requires a GUI backend. Exiting.")
                    return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    dataset = load_dataset(args.dataset_path, episode_idx=args.episode_idx)
    
    # 初始化Pinocchio FK
    # 注意：arm_start_idx 和 arm_end_idx 只是用于从 sensor_data_raw 获取关节角的定义
    # 在 verify_ik_fk_consistency.py 中，我们传入给 PinocchioFK 的已经是14维关节角数组（从IK解出的）
    # 所以应该使用默认的 [0:14]，让 PinocchioFK 根据 URDF 自动处理
    print(f"\n🔧 Initializing Pinocchio FK...")
    print(f"   Note: Input to PinocchioFK is already a 14-dim joint angle array (from IK)")
    print(f"   PinocchioFK will automatically handle indices based on URDF structure")
    fk_getter = PinocchioFK(
        urdf_path=args.urdf_path,
        arm_start_idx=0,  # 默认使用 [0:14]，因为输入已经是14维关节角数组
        arm_end_idx=14,   # PinocchioFK 会根据 URDF 自动调整（如果 URDF 只有14个关节）
        left_eef_frame='zarm_l7_link',
        right_eef_frame='zarm_r7_link',
        shared_reference_frame='base_link'
    )
    
    # 获取episode数据
    print(f"\n📊 Processing episode {args.episode_idx}...")
    
    # 使用episode的索引范围直接切片，比filter快得多
    # 这是必要的，因为v3.0格式中多个episodes可能存储在同一个文件中
    print(f"🔍 Filtering dataset to episode {args.episode_idx}...")
    if args.episode_idx >= len(dataset.meta.episodes):
        raise ValueError(f"Episode {args.episode_idx} out of range. Available episodes: 0-{len(dataset.meta.episodes)-1}")
    
    ep_meta = dataset.meta.episodes[args.episode_idx]
    ep_start = ep_meta["dataset_from_index"]
    ep_end = ep_meta["dataset_to_index"]
    
    # 使用切片而不是filter，这样快得多
    dataset.hf_dataset = dataset.hf_dataset.select(range(ep_start, ep_end))
    print(f"✅ Filtered dataset. Total frames in episode {args.episode_idx}: {len(dataset.hf_dataset)} (indices {ep_start}-{ep_end-1})")
    
    # 确定要处理的帧数
    if args.max_frames:
        num_frames_to_process = min(args.max_frames, len(dataset.hf_dataset))
    else:
        num_frames_to_process = len(dataset.hf_dataset)
    
    episode_frames = list(range(num_frames_to_process))
    print(f"📊 Will process {len(episode_frames)} frames")
    
    # 提取数据
    state_left_list = []
    state_right_list = []
    action_left_list = []
    action_right_list = []
    fk_left_list = []
    fk_right_list = []
    ik_joint_list = []
    
    print(f"\n🔄 Processing {len(episode_frames)} frames...")
    for frame_idx in tqdm(episode_frames):
        # 注意：现在dataset已经被过滤到episode范围，所以frame_idx是相对于episode的
        sample = dataset[frame_idx]
        
        # 提取state和action
        state = sample['observation.state'].numpy()  # (20,)
        action = sample['action'].numpy()  # (20,)
        
        # 提取eef poses
        eef_data = extract_eef_poses(state, action)
        
        # 解IK并验证FK
        ik_fk_result = solve_ik_and_verify_fk(
            eef_data['action_left_eef'],
            eef_data['action_right_eef'],
            fk_getter,
            model_type=args.model_type
        )
        
        # 保存数据
        state_left_list.append(eef_data['state_left_eef'])
        state_right_list.append(eef_data['state_right_eef'])
        action_left_list.append(eef_data['action_left_eef'])
        action_right_list.append(eef_data['action_right_eef'])
        fk_left_list.append(ik_fk_result['fk_left_eef'])
        fk_right_list.append(ik_fk_result['fk_right_eef'])
        ik_joint_list.append(np.concatenate([ik_fk_result['ik_left_joints'], ik_fk_result['ik_right_joints']]))
    
    # 转换为numpy数组
    state_left = np.array(state_left_list)
    state_right = np.array(state_right_list)
    action_left = np.array(action_left_list)
    action_right = np.array(action_right_list)
    fk_left = np.array(fk_left_list)
    fk_right = np.array(fk_right_list)
    ik_joint_traj = np.array(ik_joint_list)
    
    print(f"\n✅ Data processing completed!")
    print(f"   State left shape: {state_left.shape}")
    print(f"   Action left shape: {action_left.shape}")
    print(f"   FK left shape: {fk_left.shape}")
    
    # 计算误差统计
    print(f"\n📈 Computing error statistics...")
    
    # Position errors
    pos_error_action_state_left = np.linalg.norm(action_left[:, :3] - state_left[:, :3], axis=1)
    pos_error_fk_action_left = np.linalg.norm(fk_left[:, :3] - action_left[:, :3], axis=1)
    pos_error_fk_state_left = np.linalg.norm(fk_left[:, :3] - state_left[:, :3], axis=1)
    
    pos_error_action_state_right = np.linalg.norm(action_right[:, :3] - state_right[:, :3], axis=1)
    pos_error_fk_action_right = np.linalg.norm(fk_right[:, :3] - action_right[:, :3], axis=1)
    pos_error_fk_state_right = np.linalg.norm(fk_right[:, :3] - state_right[:, :3], axis=1)
    
    print(f"\n📍 Position Errors (m):")
    print(f"   Left Arm:")
    print(f"     Action vs State: mean={np.mean(pos_error_action_state_left):.4f}, max={np.max(pos_error_action_state_left):.4f}")
    print(f"     FK vs Action: mean={np.mean(pos_error_fk_action_left):.4f}, max={np.max(pos_error_fk_action_left):.4f}")
    print(f"     FK vs State: mean={np.mean(pos_error_fk_state_left):.4f}, max={np.max(pos_error_fk_state_left):.4f}")
    print(f"   Right Arm:")
    print(f"     Action vs State: mean={np.mean(pos_error_action_state_right):.4f}, max={np.max(pos_error_action_state_right):.4f}")
    print(f"     FK vs Action: mean={np.mean(pos_error_fk_action_right):.4f}, max={np.max(pos_error_fk_action_right):.4f}")
    print(f"     FK vs State: mean={np.mean(pos_error_fk_state_right):.4f}, max={np.max(pos_error_fk_state_right):.4f}")
    
    # 可视化
    print(f"\n🎨 Generating visualizations...")
    
    # 确定保存路径和交互模式
    save_3d = None if args.no_save else str(output_dir / '3d_trajectory.png')
    save_pos = None if args.no_save else str(output_dir / 'position_comparison.png')
    save_euler = None if args.no_save else str(output_dir / 'euler_comparison.png')
    
    # 3D轨迹可视化
    print("\n📊 Opening 3D trajectory visualization...")
    visualize_3d_trajectory(
        state_left, state_right,
        action_left, action_right,
        fk_left, fk_right,
        save_path=save_3d,
        interactive=args.interactive
    )
    
    # 位置对比
    print("\n📊 Opening position comparison visualization...")
    visualize_position_comparison(
        state_left, state_right,
        action_left, action_right,
        fk_left, fk_right,
        save_path=save_pos,
        interactive=args.interactive
    )
    
    # 欧拉角对比
    print("\n📊 Opening euler angle comparison visualization...")
    visualize_euler_comparison(
        state_left, state_right,
        action_left, action_right,
        fk_left, fk_right,
        save_path=save_euler,
        interactive=args.interactive
    )
    
    if not args.no_save:
        print(f"\n✅ All visualizations saved to {output_dir}")

    if args.viser:
        print("\n🌐 Launching viser visualization...")
        visualize_with_viser(
            urdf_path=args.urdf_path,
            mesh_dir=args.mesh_dir,
            model_type=args.model_type,
            state_left=state_left,
            state_right=state_right,
            action_left=action_left,
            action_right=action_right,
            fk_left=fk_left,
            fk_right=fk_right,
            ik_joint_traj=ik_joint_traj,
            host=args.viser_host,
            port=args.viser_port,
        )
    
    # 检查后端是否支持交互式显示
    backend = matplotlib.get_backend()
    is_interactive_backend = backend.lower() not in ['agg', 'pdf', 'svg', 'ps']
    
    if args.interactive and is_interactive_backend:
        print("\n💡 All interactive plots have been displayed.")
    elif is_interactive_backend:
        print("\n💡 All plots are open. Close them to finish.")
        # 等待所有窗口关闭
        try:
            input("\nPress Enter to exit after closing all plot windows...")
        except KeyboardInterrupt:
            print("\n\nExiting...")
    else:
        print("\n✅ All visualizations completed. (Using non-interactive backend)")
        if not args.interactive:
            print("💡 Tip: Use --interactive flag to enable interactive display (requires PyQt5/PySide2/Tkinter)")


if __name__ == "__main__":
    main()
