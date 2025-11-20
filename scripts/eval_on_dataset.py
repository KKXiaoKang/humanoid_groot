#!/usr/bin/env python3
"""
Evaluate GrootPolicy Model on Dataset

This script evaluates a GrootPolicy model on a LeRobot dataset and computes error metrics.
It supports optional MuJoCo visualization.

Usage:
    python scripts/eval_on_dataset.py \
        --ckpt-path <checkpoint_path> \
        --dataset-root <dataset_path> \
        --episode <episode_number> \
        [--image-zero]  # Optional: set all images to zero to test model dependency on images
        [--state-zero]  # Optional: set all state inputs to zero to test model dependency on state
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from pathlib import Path
import argparse
import time
from collections import OrderedDict
from tqdm import tqdm

# 使用GrootPolicy模型
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 导入配置模块（如果存在）
try:
    from configs.config import topic_info, TASK_DATA_MODE, get_camera_observation_key, action_names
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: configs.config not available. Using defaults.")
    CONFIG_AVAILABLE = False
    topic_info = {}
    TASK_DATA_MODE = "unknown"
    action_names = []
    def get_camera_observation_key(camera_name, use_image_features=False):
        return f"observation.images.{camera_name}" if use_image_features else f"observation.images.{camera_name}"

# 可选的可视化工具（如果不存在则禁用）
try:
    from visualization_tools.visualizers import RerunVisualizer, KeyboardManager
    RERUN_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: RerunVisualizer not available. Visualization will be disabled.")
    RERUN_AVAILABLE = False
    RerunVisualizer = None
    KeyboardManager = None


def eval_on_dataset(ckpt_path,
                    lerobot_dataset_path,
                    episode,
                    visualize_in_mujoco=False,
                    n_actions=16,
                    show_progress=True,
                    image_zero=False,
                    state_zero=False):
    """
    在数据集上评估模型
    
    Args:
        ckpt_path: 模型checkpoint路径
        lerobot_dataset_path: 数据集根目录
        episode: episode编号
        visualize_in_mujoco: 是否在MuJoCo中可视化执行
        n_actions: action chunk大小
        show_progress: 是否显示进度条
        image_zero: 是否将图像输入置零（用于验证模型对图像的依赖性）
        state_zero: 是否将状态输入置零（用于验证模型对状态的依赖性）
    """
    # ----------- 一些参数 ----------------
    mse_per_action_dim = OrderedDict() # 记录每个动作维度的MSE
    mae_per_action_dim = OrderedDict() # 记录每个动作维度的MAE

    # ------------- 初始化visualizer (可选) -------------
    if RERUN_AVAILABLE:
        vizer = RerunVisualizer()
        kb = KeyboardManager()
        print("✅ RerunVisualizer initialized")
    else:
        vizer = None
        kb = None
        print("⚠️  Running without RerunVisualizer")

    # ------------- 初始化数据集和模型 -------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*80}")
    print(f"🔧 Device: {device}")
    
    # ✅ 使用GrootPolicy加载模型
    print(f"📂 Loading GrootPolicy model from {ckpt_path}...")
    policy = GrootPolicy.from_pretrained(Path(ckpt_path), strict=False)
    policy.config.device = device
    policy.config.n_action_steps = n_actions
    
    print(f"📊 Action chunk size: {n_actions}")
    if image_zero:
        print(f"⚠️  IMAGE ZERO MODE: All image inputs will be set to zero (for dependency testing)")
    if state_zero:
        print(f"⚠️  STATE ZERO MODE: All state inputs will be set to zero (for dependency testing)")
    
    policy.eval().to(device)
    
    # Load dataset statistics for normalization
    print(f"\n📂 Loading dataset for statistics...")
    dataset_for_stats = LeRobotDataset(repo_id=0, root=lerobot_dataset_path)
    dataset_stats = dataset_for_stats.meta.stats if hasattr(dataset_for_stats.meta, 'stats') else None
    print(f"✅ Dataset statistics loaded: {list(dataset_stats.keys()) if dataset_stats else 'None'}")
    
    # 检查action统计信息格式，用于手动反归一化
    if dataset_stats and 'action' in dataset_stats:
        action_stats = dataset_stats['action']
        print(f"📊 Action stats keys: {list(action_stats.keys())}")
        if 'min' in action_stats and 'max' in action_stats:
            action_min = torch.as_tensor(action_stats['min'], dtype=torch.float32)
            action_max = torch.as_tensor(action_stats['max'], dtype=torch.float32)
            print(f"📊 Action normalization range: min={action_min[:5].tolist()}... (shape: {action_min.shape}), max={action_max[:5].tolist()}... (shape: {action_max.shape})")
        else:
            print("⚠️  Warning: Action stats missing 'min' or 'max'. Manual denormalization may not work correctly.")
    
    # Create preprocessor and postprocessor
    print(f"\n🔧 Creating preprocessor and postprocessor...")
    preprocessor, postprocessor = make_groot_pre_post_processors(
        config=policy.config,
        dataset_stats=dataset_stats,
    )
    print("✅ Preprocessor and postprocessor created")
    
    # Debug: Print model configuration
    print(f"🔍 Model configuration input_features keys: {list(policy.config.input_features.keys()) if hasattr(policy.config, 'input_features') else 'N/A'}")
    print(f"🔍 Model configuration output_features keys: {list(policy.config.output_features.keys()) if hasattr(policy.config, 'output_features') else 'N/A'}")
    
    policy.reset()
    print("✅ Model loaded and ready")
    
    # ✅ 使用标准的LeRobotDataset加载数据
    print(f"\n📂 Loading dataset from {lerobot_dataset_path}")
    print(f"📹 Episode: {episode}")
    
    # 注意：LeRobotDataset的episodes参数主要用于下载时选择文件
    # 但在加载后需要手动过滤数据，因为多个episodes可能存储在同一个parquet文件中
    dataset = LeRobotDataset(repo_id=0, root=lerobot_dataset_path, episodes=[episode])
    
    # 使用episode的索引范围直接切片，比filter快得多
    # 这是必要的，因为v3.0格式中多个episodes可能存储在同一个文件中
    print(f"🔍 Filtering dataset to episode {episode}...")
    if episode >= len(dataset.meta.episodes):
        raise ValueError(f"Episode {episode} out of range. Available episodes: 0-{len(dataset.meta.episodes)-1}")
    
    ep_meta = dataset.meta.episodes[episode]
    ep_start = ep_meta["dataset_from_index"]
    ep_end = ep_meta["dataset_to_index"]
    
    # 使用切片而不是filter，这样快得多
    dataset.hf_dataset = dataset.hf_dataset.select(range(ep_start, ep_end))
    print(f"✅ Filtered dataset. Total frames in episode {episode}: {len(dataset.hf_dataset)} (indices {ep_start}-{ep_end-1})")
    
    # 打印相机配置信息
    if CONFIG_AVAILABLE:
        camera_config = {name: info for name, info in topic_info.items() if 'image' in name}
        print(f"\n📷 Camera Configuration (TASK_DATA_MODE: {TASK_DATA_MODE}):")
        print(f"   Detected {len(camera_config)} cameras: {list(camera_config.keys())}")
    else:
        # 从数据集元数据中检测相机
        sample = dataset[0]
        image_keys = [k for k in sample.keys() if 'image' in k.lower()]
        print(f"\n📷 Camera Configuration:")
        print(f"   Detected {len(image_keys)} image keys: {image_keys}")
    
    # 创建dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        pin_memory=(device == "cuda:0"),
        drop_last=False,
    )
    
    print(f"✅ Dataset loaded. Total frames: {dataset.num_frames}")

    # 获取action维度
    first_batch = next(iter(dataloader))
    action_dim = first_batch['action'].shape[1]
    obs_dim = first_batch['observation.state'].shape[1]
    print(f"📊 Action dimension: {action_dim}")
    print(f"📊 Observation dimension: {obs_dim}")
    
    # 重新创建dataloader（因为已经消耗了第一个batch）
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        pin_memory=(device == "cuda:0"),
        drop_last=False,
    )
    
    # 初始化环境（如果需要在mujoco中可视化）
    if visualize_in_mujoco:
        print(f"\n🤖 Initializing MuJoCo environment...")
        # 根据action维度判断使用哪个环境
        # 16维动作 = depalletize任务，使用kuavo_depalletize_env
        # 其他维度 = com控制任务，使用kuavo_com_env
        if action_dim == 16:
            try:
                from robot_envs.kuavo_depalletize_env import GrabBoxMpcEnv
                mujoco_env = GrabBoxMpcEnv()
                print(f"✅ MuJoCo environment initialized (depalletize task)")
                print(f"   - Action dimension: 16 (14 arm joints + 2 claw positions)")
            except ImportError:
                print("⚠️  Warning: robot_envs.kuavo_depalletize_env not available. MuJoCo visualization disabled.")
                visualize_in_mujoco = False
                mujoco_env = None
        else:
            try:
                from robot_envs.kuavo_com_env import GrabBoxMpcEnv
                # GrootPolicy uses absolute actions by default
                mujoco_env = GrabBoxMpcEnv(use_action_history_reference=False)
                print(f"✅ MuJoCo environment initialized (com control task)")
                print(f"   - use_action_history_reference: False (absolute actions)")
            except ImportError:
                print("⚠️  Warning: robot_envs.kuavo_com_env not available. MuJoCo visualization disabled.")
                visualize_in_mujoco = False
                mujoco_env = None
    
    # ========= 可视化数据集里的groundtruth (如果启用RerunVisualizer) =========
    if vizer is not None:
        print(f"\n📊 Visualizing ground truth data...")
        # 加载所有ground truth actions用于可视化
        all_gt_actions = []
        all_gt_states = []
        
        temp_dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=0, batch_size=1, shuffle=False, drop_last=False
        )
        
        for batch in temp_dataloader:
            all_gt_actions.append(batch['action'][0].cpu().numpy())
            all_gt_states.append(batch['observation.state'][0].cpu().numpy())
        
        all_gt_actions = np.array(all_gt_actions)
        all_gt_states = np.array(all_gt_states)
        
        # 可视化ground truth actions
        for dim in range(action_dim):
            vizer.visualize_chunk(
                name=f"chunk/action_dim_{dim}/gt",
                chunk_data=all_gt_actions[:, dim],
                step_id=0,
                width=3.0
            )
        
        # 可视化observations
        for dim in range(obs_dim):
            vizer.visualize_chunk(
                name=f"obs/obs_{dim}",
                chunk_data=all_gt_states[:, dim],
                step_id=0,
                width=3.0
            )
        
        print(f"✅ Ground truth visualization ready")

    # ========= 开始模型推理 =========
    print("\n" + "="*80)
    print("🚀 Starting inference...")
    print("="*80 + "\n")
    
    last_data_step = 0
    predictions = []
    ground_truths = []
    
    # 使用tqdm显示进度（如果启用）
    iterator = tqdm(enumerate(dataloader), total=dataset.num_frames, desc="Processing") if show_progress else enumerate(dataloader)
    
    for data_step, batch in iterator:
        # 暂停控制（如果启用）
        if vizer is not None and kb is not None:
            time.sleep(0.05)
            if kb.paused:
                print(f'===== 暂停中，按下空格开始 =====')
            while kb.paused:
                time.sleep(0.1)
        
        # ✅ 准备observation - 使用Groot预处理器
        # 首先构建原始observation字典
        observation = {
            'observation.state': batch['observation.state'],
        }
        
        # 添加所有图像观测（Groot预处理器会自动处理）
        for key in batch.keys():
            if 'image' in key.lower() and key.startswith('observation'):
                observation[key] = batch[key]
        
        # 如果启用state_zero模式，将状态输入置零（用于验证模型对状态的依赖性）
        if state_zero:
            # 保持相同的形状和设备，但将所有状态值设为0
            observation['observation.state'] = torch.zeros_like(observation['observation.state'])
        
        # 如果启用image_zero模式，将所有图像输入置零（用于验证模型对图像的依赖性）
        if image_zero:
            for key in list(observation.keys()):
                if 'image' in key.lower():
                    # 保持相同的形状和设备，但将所有像素值设为0
                    observation[key] = torch.zeros_like(observation[key])
        
        # 获取ground truth action
        gt_action = batch['action'][0].cpu().numpy()  # (action_dim,)
        
        # 使用预处理器处理输入
        processed_observation = preprocessor(observation)
        
        # 模型推理
        tic = time.time()
        with torch.inference_mode():
            pred_actions = policy.predict_action_chunk(processed_observation)
        
        # pred_actions shape: (batch_size, chunk_size, action_dim)
        # 注意：pred_actions是归一化后的值，范围在[-1, 1]
        # 需要手动反归一化到真实单位，以便与ground truth进行比较
        
        # 准备反归一化参数
        if dataset_stats and 'action' in dataset_stats:
            action_stats = dataset_stats['action']
            if 'min' in action_stats and 'max' in action_stats:
                action_min = torch.as_tensor(action_stats['min'], dtype=torch.float32, device=pred_actions.device)
                action_max = torch.as_tensor(action_stats['max'], dtype=torch.float32, device=pred_actions.device)
                
                # 确保维度匹配
                if action_min.numel() < action_dim:
                    action_min = torch.nn.functional.pad(action_min.flatten()[:action_dim], (0, max(0, action_dim - action_min.numel())))
                if action_max.numel() < action_dim:
                    action_max = torch.nn.functional.pad(action_max.flatten()[:action_dim], (0, max(0, action_dim - action_max.numel())))
                
                action_min = action_min[:action_dim]
                action_max = action_max[:action_dim]
                
                # 反归一化公式：x = (y + 1) / 2 * (max - min) + min
                # 其中y是归一化值[-1, 1]，x是原始值[min, max]
                denom = action_max - action_min
                mask = denom != 0
                safe_denom = torch.where(mask, denom, torch.ones_like(denom))
                
                # 反归一化整个chunk
                pred_actions_unnorm = (pred_actions + 1.0) * 0.5 * safe_denom + action_min
                pred_actions_unnorm = torch.where(mask, pred_actions_unnorm, action_min)
                
                # 选择最后一个时间步作为单步预测
                pred_action_single = pred_actions_unnorm[0, -1, :].cpu().numpy()  # (action_dim,)
                # 整个chunk用于可视化
                pred_chunk = pred_actions_unnorm[0].cpu().numpy()  # (chunk_size, action_dim)
            else:
                # 如果没有统计信息，使用原始值（可能已经是反归一化的）
                pred_action_single = pred_actions[0, -1, :].cpu().numpy()  # (action_dim,)
                pred_chunk = pred_actions[0].cpu().numpy()  # (chunk_size, action_dim)
                print("⚠️  Warning: No action min/max stats found. Using raw predictions (may be normalized).")
        else:
            # 如果没有统计信息，使用原始值
            pred_action_single = pred_actions[0, -1, :].cpu().numpy()  # (action_dim,)
            pred_chunk = pred_actions[0].cpu().numpy()  # (chunk_size, action_dim)
            print("⚠️  Warning: No dataset stats found. Using raw predictions (may be normalized).")
        
        inference_time = time.time() - tic
        
        # 保存预测和真实值
        predictions.append(pred_action_single)
        ground_truths.append(gt_action)
        
        # 计算每个维度的MSE和MAE
        for dim in range(action_dim):
            error = pred_action_single[dim] - gt_action[dim]
            mse = error ** 2
            mae = abs(error)
            
            if dim not in mse_per_action_dim:
                mse_per_action_dim[dim] = []
                mae_per_action_dim[dim] = []
            
            mse_per_action_dim[dim].append(mse)
            mae_per_action_dim[dim].append(mae)
        
        # 可视化（如果启用）
        if vizer is not None:
            # 显示图像 - 动态查找第一个可用的相机图像
            for key in batch.keys():
                if 'image' in key.lower() and key.startswith('observation'):
                    img = batch[key][0]  # (C, H, W)
                    camera_name = key.replace('observation.', '').replace('observation.images.', '')
                    vizer.show_img(
                        name=camera_name,
                        image_data=img.to("cpu"),
                        step_id=data_step
                    )
                    # break  # 只显示第一个找到的相机图像
            
            # 可视化预测的chunk
            for dim in range(action_dim):
                # 可视化MSE
                vizer.visualize_chunk(
                    name=f"mse/action_dim_{dim}",
                    chunk_data=mse_per_action_dim[dim][-1],
                    step_id=data_step,
                    width=3.0,
                )
                
                # 可视化预测chunk
                vizer.visualize_chunk(
                    name=f"chunk/action_dim_{dim}/pred_seg_{data_step}",
                    chunk_data=pred_chunk[:, dim],
                    step_id=data_step,
                    width=2
                )
                
                # 删除上一个chunk的可视化
                if last_data_step != data_step and last_data_step > 0:
                    vizer.del_chunk(
                        name=f"chunk/action_dim_{dim}/pred_seg_{last_data_step}",
                        chunk_data=pred_chunk[:, dim],
                        step_id=last_data_step,
                        width=0.5
                    )
        
        last_data_step = data_step
        
        # ========== 在mujoco里执行动作 (如果启用) =========
        if visualize_in_mujoco and mujoco_env is not None:
            action_np = pred_action_single[np.newaxis, :]  # (1, action_dim)
            
            # 根据action维度选择执行方法
            if action_dim == 16:
                # depalletize任务：16维动作 (14 arm joints + 2 claw positions)
                mujoco_env.exec_actions(
                    actions=action_np,
                    control_arm=True,
                    control_claw=True
                )
            else:
                # com控制任务：使用绝对动作执行（GrootPolicy默认使用绝对动作）
                mujoco_env.exec_absolute_actions(
                    actions=action_np,
                    control_arm=True,
                    control_base=True,
                    control_wrench=False
                )

    # ========= 打印最终统计结果 =========
    print("\n" + "="*80)
    print("📊 Final Statistics")
    print("="*80)
    
    # Action名称定义 - 根据action维度自动选择
    # 优先使用config中的action_names（如果可用且维度匹配）
    if CONFIG_AVAILABLE and action_names and len(action_names) == action_dim:
        eval_action_names = action_names
    elif action_dim == 16:
        # depalletize任务：16维动作 (14 arm joints + 2 claw positions)
        eval_action_names = [f"Arm_joint_{i+1}" for i in range(14)] + ["Left_claw", "Right_claw"]
    elif action_dim == 18:
        # 18维动作：Left_arm(7) + Right_arm(7) + Left_claw(1) + Right_claw(1) + Cmd_pose_z(1) + Cmd_pose_pitch(1)
        # 根据config.py的ACTION_COMPONENT_DEFINITIONS格式
        eval_action_names = (
            [f"arm_joint_{i+1}" for i in range(7)] +  # Left_arm: arm_joint_1-7
            [f"arm_joint_{i+8}" for i in range(7)] +  # Right_arm: arm_joint_8-14
            ["left_claw_position", "right_claw_position", "cmd_pose_z", "cmd_pose_pitch"]
        )
    elif action_dim == 24:
        # com控制任务：24维 = 9 COM + 14 Arm + 1 Gait
        eval_action_names = (
            ["COM_dx", "COM_dy", "COM_dz", "COM_dR11", "COM_dR21", "COM_dR31", "COM_dR12", "COM_dR22", "COM_dR32"] +
            [f"Arm_joint_{i+1}" for i in range(14)] +
            ["Gait_mode"]
        )
    else:
        # 其他维度：使用通用命名
        eval_action_names = [f"Action_dim_{i}" for i in range(action_dim)]
    
    print(f"\n{'Dimension':<20} {'MSE':<15} {'MAE':<15}")
    print("-" * 80)
    
    for dim in range(action_dim):
        mse_mean = np.mean(mse_per_action_dim[dim])
        mae_mean = np.mean(mae_per_action_dim[dim])
        dim_name = eval_action_names[dim] if dim < len(eval_action_names) else f"Dim_{dim}"
        print(f'{dim_name:<20} {mse_mean:<15.8f} {mae_mean:<15.8f}')
    
    overall_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(action_dim)])
    overall_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(action_dim)])
    
    print("-" * 80)
    print(f'{"Overall":<20} {overall_mse:<15.8f} {overall_mae:<15.8f}')
    
    # 分组统计 - 根据action维度选择统计方式
    print("\n📊 Grouped Statistics:")
    print("-" * 80)
    
    if action_dim == 16:
        # depalletize任务：16维 = 14 arm joints + 2 claw positions
        arm_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(14)])
        arm_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(14)])
        print(f'{"Arm (avg)":<20} {arm_mse:<15.8f} {arm_mae:<15.8f}')
        
        claw_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(14, 16)])
        claw_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(14, 16)])
        print(f'{"Claw (avg)":<20} {claw_mse:<15.8f} {claw_mae:<15.8f}')
    elif action_dim == 18:
        # 18维动作：Left_arm(7) + Right_arm(7) + Left_claw(1) + Right_claw(1) + Cmd_pose_z(1) + Cmd_pose_pitch(1)
        left_arm_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(7)])
        left_arm_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(7)])
        print(f'{"Left_arm (avg)":<20} {left_arm_mse:<15.8f} {left_arm_mae:<15.8f}')
        
        right_arm_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(7, 14)])
        right_arm_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(7, 14)])
        print(f'{"Right_arm (avg)":<20} {right_arm_mse:<15.8f} {right_arm_mae:<15.8f}')
        
        arm_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(14)])
        arm_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(14)])
        print(f'{"Arm (avg)":<20} {arm_mse:<15.8f} {arm_mae:<15.8f}')
        
        claw_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(14, 16)])
        claw_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(14, 16)])
        print(f'{"Claw (avg)":<20} {claw_mse:<15.8f} {claw_mae:<15.8f}')
        
        cmd_pose_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(16, 18)])
        cmd_pose_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(16, 18)])
        print(f'{"Cmd_pose (avg)":<20} {cmd_pose_mse:<15.8f} {cmd_pose_mae:<15.8f}')
    elif action_dim == 24:
        # com控制任务：24维 = 9 COM + 14 Arm + 1 Gait
        com_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(9)])
        com_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(9)])
        print(f'{"COM (avg)":<20} {com_mse:<15.8f} {com_mae:<15.8f}')
        
        arm_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in range(9, 23)])
        arm_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in range(9, 23)])
        print(f'{"Arm (avg)":<20} {arm_mse:<15.8f} {arm_mae:<15.8f}')
        
        gait_mse = np.mean(mse_per_action_dim[23])
        gait_mae = np.mean(mae_per_action_dim[23])
        print(f'{"Gait":<20} {gait_mse:<15.8f} {gait_mae:<15.8f}')
    else:
        # 其他维度：尝试根据config推断，或使用通用分组
        if CONFIG_AVAILABLE and action_names and len(action_names) == action_dim:
            # 根据action_names推断分组
            # 查找常见的分组模式
            if any("arm" in name.lower() for name in action_names):
                # 尝试找到arm相关的维度
                arm_dims = [i for i, name in enumerate(action_names) if "arm" in name.lower()]
                if arm_dims:
                    arm_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in arm_dims])
                    arm_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in arm_dims])
                    print(f'{"Arm (avg)":<20} {arm_mse:<15.8f} {arm_mae:<15.8f}')
            
            if any("claw" in name.lower() for name in action_names):
                claw_dims = [i for i, name in enumerate(action_names) if "claw" in name.lower()]
                if claw_dims:
                    claw_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in claw_dims])
                    claw_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in claw_dims])
                    print(f'{"Claw (avg)":<20} {claw_mse:<15.8f} {claw_mae:<15.8f}')
            
            if any("cmd_pose" in name.lower() or "com" in name.lower() for name in action_names):
                cmd_dims = [i for i, name in enumerate(action_names) if "cmd_pose" in name.lower() or "com" in name.lower()]
                if cmd_dims:
                    cmd_mse = np.mean([np.mean(mse_per_action_dim[dim]) for dim in cmd_dims])
                    cmd_mae = np.mean([np.mean(mae_per_action_dim[dim]) for dim in cmd_dims])
                    print(f'{"Cmd/COM (avg)":<20} {cmd_mse:<15.8f} {cmd_mae:<15.8f}')
        else:
            # 无法推断，跳过分组统计
            print("⚠️  Cannot infer action groups for this action dimension. Skipping grouped statistics.")
    
    print("="*80)

    if vizer is not None:
        print("\n[Offline Eval] Visualization active. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n✅ Exiting...")
    else:
        print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate GrootPolicy Model on Dataset',
        epilog='Evaluates a trained GrootPolicy model on a LeRobot dataset.'
    )
    parser.add_argument('--ckpt-path', type=str, required=True,
                       help='Path to the model checkpoint directory')
    parser.add_argument('--dataset-root', '--dataset_root', type=str, required=True,
                       dest='dataset_root',
                       help='Path to the LeRobot dataset root directory')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode number to evaluate (default: 0)')
    parser.add_argument('--action-chunk-size', type=int, default=50,
                       help='Action chunk size (default: 50, should match training config)')
    parser.add_argument('--with-mujoco', action='store_true',
                       help='Visualize and execute in MuJoCo environment')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    parser.add_argument('--image-zero', action='store_true',
                       help='Set all image inputs to zero (for testing model dependency on images)')
    parser.add_argument('--state-zero', action='store_true',
                       help='Set all state inputs to zero (for testing model dependency on state)')

    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎯 GrootPolicy Dataset Evaluation")
    print("="*80)
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Dataset: {args.dataset_root}")
    print(f"Episode: {args.episode}")
    print(f"Action Chunk Size: {args.action_chunk_size}")
    print(f"MuJoCo Visualization: {args.with_mujoco}")
    print(f"Image Zero Mode: {args.image_zero}")
    print(f"State Zero Mode: {args.state_zero}")
    print("="*80)
    
    eval_on_dataset(
        ckpt_path=args.ckpt_path,
        lerobot_dataset_path=args.dataset_root,
        episode=args.episode,
        n_actions=args.action_chunk_size,
        visualize_in_mujoco=args.with_mujoco,
        show_progress=not args.no_progress,
        image_zero=args.image_zero,
        state_zero=args.state_zero
    )
