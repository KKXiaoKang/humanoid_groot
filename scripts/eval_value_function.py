#!/usr/bin/env python
"""
PI05 Value Function 评估脚本

加载训练好的 Value Function checkpoint，从数据集中取样本，
对比模型预测的 V(s) 与真实 target_value，验证推理输出。

使用方法:
    conda run -n lerobot_groot python scripts/eval_value_function.py \
        --checkpoint /home/kangkk/humanoid_groot/outputs/pi05_value_function_training/checkpoints/010000/pretrained_model \
        --dataset eef_3x2_with_reward \
        --dataset-root /home/kangkk/humanoid_groot/lerobot_data/eef_dataset \
        --num-samples 50
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import torch
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="评估 PI05 Value Function 的推理输出")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Value Function checkpoint 路径 (pretrained_model 目录)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eef_3x2_with_reward",
        help="评估用的数据集名称",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/home/kangkk/humanoid_groot/lerobot_data/eef_dataset",
        help="数据集根目录",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="采样数量（如果指定了--episodes则忽略）",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="指定要分析的episode索引列表，例如 --episodes 0 1 2。如果不指定，则分析所有episodes",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="最多分析的episode数量（如果未指定--episodes）",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式模式：每个episode显示后等待用户按键继续",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="推理设备",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="推理 batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    return parser.parse_args()


def check_opencv_gui_support():
    """检查OpenCV是否有GUI支持，并尝试修复。
    
    Returns:
        bool: True如果有GUI支持，False如果没有
    """
    logger.info("=" * 60)
    logger.info("🔍 诊断OpenCV GUI支持...")
    
    # 检查OpenCV构建信息
    build_info = cv2.getBuildInformation()
    
    # 更准确地检查GUI支持
    has_gtk = False
    has_gtk3 = False
    has_qt = False
    has_cocoa = False
    
    # 检查GTK支持
    if 'GTK:' in build_info:
        gtk_section = build_info.split('GTK:')[1].split('\n')[0]
        has_gtk = 'YES' in gtk_section or 'ON' in gtk_section
    if 'GTK 3.x:' in build_info:
        gtk3_section = build_info.split('GTK 3.x:')[1].split('\n')[0]
        has_gtk3 = 'YES' in gtk3_section or 'ON' in gtk3_section
    
    # 检查QT支持
    if 'QT:' in build_info:
        qt_section = build_info.split('QT:')[1].split('\n')[0]
        has_qt = 'YES' in qt_section or 'ON' in qt_section
    
    # 检查Cocoa支持（macOS）
    if 'Cocoa:' in build_info:
        cocoa_section = build_info.split('Cocoa:')[1].split('\n')[0]
        has_cocoa = 'YES' in cocoa_section or 'ON' in cocoa_section
    
    logger.info(f"  GTK支持: {'✅' if has_gtk else '❌'}")
    logger.info(f"  GTK3支持: {'✅' if has_gtk3 else '❌'}")
    logger.info(f"  QT支持: {'✅' if has_qt else '❌'}")
    logger.info(f"  Cocoa支持: {'✅' if has_cocoa else '❌'}")
    
    # 检查DISPLAY环境变量
    display = os.environ.get('DISPLAY')
    if display:
        logger.info(f"  DISPLAY环境变量: {display} ✅")
    else:
        logger.warning(f"  DISPLAY环境变量: 未设置 ❌")
        logger.warning(f"  如果使用SSH，请使用 -X 或 -Y 选项启用X11转发")
        logger.warning(f"  或者设置: export DISPLAY=:0.0")
    
    # 检查是否有任何GUI支持
    has_any_gui = has_gtk or has_gtk3 or has_qt or has_cocoa
    
    if not has_any_gui:
        logger.error("")
        logger.error("❌ OpenCV编译时没有包含任何GUI支持！")
        logger.error("")
        logger.error("解决方案：")
        logger.error("1. 如果使用conda，重新安装带GUI支持的OpenCV：")
        logger.error("   conda uninstall opencv opencv-python")
        logger.error("   conda install -c conda-forge opencv")
        logger.error("")
        logger.error("2. 如果使用pip，先安装系统依赖，然后重新安装：")
        logger.error("   sudo apt-get update")
        logger.error("   sudo apt-get install -y libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev")
        logger.error("   pip uninstall opencv-python opencv-python-headless")
        logger.error("   pip install opencv-python")
        logger.error("")
        logger.error("3. 如果使用SSH，确保X11转发已启用：")
        logger.error("   ssh -X user@host 或 ssh -Y user@host")
        logger.error("   然后检查: echo $DISPLAY")
        return False
    
    # 尝试设置OpenCV后端
    if has_qt:
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        logger.info("  尝试使用QT后端...")
    elif has_gtk3:
        logger.info("  尝试使用GTK3后端...")
    elif has_gtk:
        logger.info("  尝试使用GTK2后端...")
    
    # 尝试创建一个测试窗口
    logger.info("  测试窗口创建...")
    try:
        test_window = "__opencv_gui_test__"
        cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(test_window)
        logger.info("  ✅ OpenCV GUI测试成功！窗口可以正常创建。")
        return True
    except cv2.error as e:
        error_msg = str(e)
        logger.error(f"  ❌ OpenCV GUI测试失败: {error_msg}")
        
        if "not implemented" in error_msg.lower() or "not built" in error_msg.lower():
            logger.error("")
            logger.error("OpenCV没有GUI支持。请按照上面的解决方案重新安装OpenCV。")
        elif "cannot connect to X server" in error_msg.lower():
            logger.error("")
            logger.error("无法连接到X服务器。请检查：")
            logger.error("1. DISPLAY环境变量是否正确设置")
            logger.error("2. X11转发是否启用（如果使用SSH）")
            logger.error("3. X服务器是否正在运行")
        else:
            logger.error(f"  未知错误: {error_msg}")
        
        return False
    except Exception as e:
        logger.error(f"  ❌ 意外错误: {type(e).__name__}: {e}")
        return False


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 检查OpenCV GUI支持
    has_gui = check_opencv_gui_support()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("PI05 Value Function 评估")
    logger.info("=" * 60)
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Dataset:    {args.dataset}")
    logger.info(f"  Root:       {args.dataset_root}")
    logger.info(f"  Samples:    {args.num_samples}")
    logger.info(f"  Device:     {args.device}")
    logger.info("=" * 60)

    # ================================================================
    # 1. 加载模型
    # ================================================================
    logger.info("📦 加载 Value Function 模型...")

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(args.checkpoint, strict=False)
    policy.to(args.device)
    policy.eval()

    # 验证 value function 已加载
    if policy.value_function is None:
        logger.error("❌ Value Function 未加载！请检查 checkpoint 是否包含 value_function 权重。")
        sys.exit(1)

    # 强制使用 float32 精度以避免 bfloat16 相关的 dtype 错误
    # vision_tower 的某些 layer_norm 层不支持 bfloat16
    logger.info("🔧 将 Value Function 转换为 float32 精度...")
    policy.value_function.to(dtype=torch.float32)

    vf_params = sum(p.numel() for p in policy.value_function.parameters())
    vf_trainable = sum(p.numel() for p in policy.value_function.parameters() if p.requires_grad)
    logger.info(f"✅ Value Function 已加载: {vf_params / 1e6:.1f}M 参数 ({vf_trainable / 1e6:.1f}M 可训练)")

    # ================================================================
    # 2. 加载数据集
    # ================================================================
    logger.info("📂 加载数据集...")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds_root = Path(args.dataset_root) / args.dataset
    dataset = LeRobotDataset(repo_id=args.dataset, root=ds_root)

    logger.info(f"  数据集大小: {len(dataset)} frames")
    logger.info(f"  Episodes: {dataset.num_episodes}")

    # 检查 target_value 是否存在
    sample_0 = dataset[0]
    if "target_value" not in sample_0:
        logger.error("❌ 数据集中没有 target_value 字段！请先运行 add_reward_to_dataset.py")
        sys.exit(1)

    logger.info(f"  Sample keys: {list(sample_0.keys())}")

    # ================================================================
    # 3. 创建预处理器
    # ================================================================
    logger.info("🔧 创建预处理器...")

    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    preprocessor, _ = make_pi05_pre_post_processors(
        policy.config,
        dataset_stats=dataset.meta.stats,
    )

    # ================================================================
    # 4. 按episode组织数据并推理
    # ================================================================
    # 确定要分析的episodes
    if args.episodes is not None:
        episode_indices = args.episodes
        logger.info(f"🚀 分析指定的episodes: {episode_indices}")
    else:
        total_episodes = dataset.num_episodes
        max_episodes = args.max_episodes if args.max_episodes else total_episodes
        episode_indices = list(range(min(max_episodes, total_episodes)))
        logger.info(f"🚀 分析前 {len(episode_indices)} 个episodes (共 {total_episodes} 个)")
    
    all_predicted = []
    all_target = []
    all_rewards = []
    all_indices = []
    
    # 按episode组织数据并逐个分析
    for episode_idx in episode_indices:
        logger.info(f"\n{'='*60}")
        logger.info(f"📹 分析 Episode {episode_idx}...")
        logger.info(f"{'='*60}")
        
        # 获取该episode的所有帧索引
        if episode_idx >= len(dataset.meta.episodes):
            logger.warning(f"  Episode {episode_idx} 不存在，跳过")
            continue
        
        ep_meta = dataset.meta.episodes[episode_idx]
        ep_start = ep_meta["dataset_from_index"]
        ep_end = ep_meta["dataset_to_index"]
        episode_frames = list(range(ep_start, ep_end))
        
        if len(episode_frames) == 0:
            logger.warning(f"  Episode {episode_idx} 没有数据，跳过")
            continue
        
        logger.info(f"  Episode {episode_idx} 包含 {len(episode_frames)} 帧 (索引 {ep_start}-{ep_end-1})")
        
        # 存储该episode的数据
        episode_predicted = []
        episode_target = []
        episode_rewards = []
        episode_indices_list = []
        episode_visualization_data = []
        window_name = None  # OpenCV窗口名称，在第一次显示时创建
        
        # 按batch处理该episode的所有帧
        for batch_start in range(0, len(episode_frames), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(episode_frames))
            batch_indices = episode_frames[batch_start:batch_end]

            # 收集 batch
            raw_samples = [dataset[int(idx)] for idx in batch_indices]

            # 预处理每个样本
            processed_samples = []
            for sample in raw_samples:
                processed = preprocessor(sample)
                processed_samples.append(processed)

            # 手动 collate
            batch = {}
            for key in processed_samples[0]:
                vals = [s[key] for s in processed_samples]
                if isinstance(vals[0], torch.Tensor):
                    # 对于图像，检查格式并确保正确
                    if key.startswith("observation.images."):
                        # 预处理器可能返回 [C, H, W] 格式（单个样本）
                        # 我们需要 stack 成 [B, C, H, W]
                        first_val = vals[0]
                        if first_val.dim() == 3:
                            # [C, H, W] -> stack to [B, C, H, W]
                            batch[key] = torch.stack(vals)
                        elif first_val.dim() == 4:
                            # 已经是 [1, C, H, W] 或 [1, H, W, C]（带时间维度）
                            stacked = torch.stack(vals)
                            # 如果 stack 后是 [B, 1, C, H, W] 或 [B, 1, H, W, C]，需要 squeeze 掉时间维度
                            if stacked.dim() == 5 and stacked.shape[1] == 1:
                                # [B, 1, C, H, W] -> [B, C, H, W]
                                stacked = stacked.squeeze(1)
                            # 如果是 [B, H, W, C]，转换为 [B, C, H, W]
                            if stacked.dim() == 4 and stacked.shape[-1] == 3:
                                stacked = stacked.permute(0, 3, 1, 2)
                            batch[key] = stacked
                        else:
                            logger.warning(f"Unexpected image shape for {key}: {first_val.shape}")
                            stacked = torch.stack(vals)
                            # 尝试处理 5 维情况
                            if stacked.dim() == 5 and stacked.shape[1] == 1:
                                stacked = stacked.squeeze(1)
                            batch[key] = stacked
                    else:
                        batch[key] = torch.stack(vals)
                else:
                    batch[key] = vals

            # 获取 target_value（从原始样本中取，因为预处理可能会丢掉它）
            target_values = torch.tensor(
                [float(raw_samples[i].get("target_value", 0.0)) for i in range(len(raw_samples))],
                dtype=torch.float32,
            )
            rewards = torch.tensor(
                [float(raw_samples[i].get("reward", 0.0)) for i in range(len(raw_samples))],
                dtype=torch.float32,
            )

            # 处理时间维度：去除语言 tokens 和 attention_mask 的时间维度
            # 这些字段在数据集中是 [B, T, ...] 格式，但模型期望 [B, ...] 格式
            lang_tokens_key = "observation.language.tokens"
            lang_mask_key = "observation.language.attention_mask"
            
            if lang_tokens_key in batch and batch[lang_tokens_key].dim() == 3:
                # [B, T, seq_len] -> [B, seq_len]
                batch[lang_tokens_key] = batch[lang_tokens_key].squeeze(1)
            if lang_mask_key in batch and batch[lang_mask_key].dim() == 3:
                # [B, T, seq_len] -> [B, seq_len]
                batch[lang_mask_key] = batch[lang_mask_key].squeeze(1)
            
            # 处理 state 的时间维度（如果需要）
            if "observation.state" in batch and batch["observation.state"].dim() == 3:
                # [B, T, state_dim] -> [B, state_dim]
                batch["observation.state"] = batch["observation.state"].squeeze(1)
            
            # 处理 action 的时间维度（如果需要）
            if "action" in batch and batch["action"].dim() == 3:
                # [B, T, action_dim] -> [B, action_dim]
                batch["action"] = batch["action"].squeeze(1)

            # 将 batch 移动到设备
            device_batch = {}
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    device_batch[key] = val.to(args.device)
                else:
                    device_batch[key] = val

            # 推理
            try:
                with torch.no_grad():
                    predicted_values = policy.predict_value(device_batch)  # (B,)
            except Exception as e:
                logger.error(f"推理失败: {e}")
                logger.error(f"Batch keys: {list(device_batch.keys())}")
                for key, val in device_batch.items():
                    if isinstance(val, torch.Tensor):
                        logger.error(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                raise

            pred_np = predicted_values.cpu().float().numpy()
            target_np = target_values.numpy()
            reward_np = rewards.numpy()

            episode_predicted.extend(pred_np.tolist())
            episode_target.extend(target_np.tolist())
            episode_rewards.extend(reward_np.tolist())
            # batch_indices 已经是 list，不需要 tolist()
            episode_indices_list.extend(batch_indices)
            
            # 实时显示每一帧（在推理过程中）
            for i in range(len(batch_indices)):
                sample_idx = int(batch_indices[i])
                raw_sample = raw_samples[i]
                
                # 获取原始图像（未预处理的）
                images_dict = {}
                for cam_key in ['observation.images.cam_head', 'observation.images.cam_left', 'observation.images.cam_right']:
                    if cam_key in raw_sample:
                        img = raw_sample[cam_key]
                        # 如果是tensor，转换为numpy
                        if isinstance(img, torch.Tensor):
                            img = img.cpu().numpy()
                        # 处理时间维度 [T, C, H, W] -> [C, H, W]
                        if img.ndim == 4:
                            if img.shape[0] == 1:
                                img = img[0]  # [1, C, H, W] -> [C, H, W]
                            else:
                                img = img[0]  # 取第一个时间步
                        # 转换为 [H, W, C] 格式用于显示
                        if img.ndim == 3:
                            if img.shape[0] == 3:  # [C, H, W]
                                img = img.transpose(1, 2, 0)  # -> [H, W, C]
                            elif img.shape[2] == 3:  # 已经是 [H, W, C]
                                pass
                        # 归一化到 [0, 1]
                        if img.max() > 1.0:
                            img = np.clip(img / 255.0, 0.0, 1.0)
                        images_dict[cam_key] = img
                
                # 获取state
                state = None
                if 'observation.state' in raw_sample:
                    state = raw_sample['observation.state']
                    if isinstance(state, torch.Tensor):
                        state = state.cpu().numpy()
                    # 处理时间维度
                    if state.ndim == 2 and state.shape[0] == 1:
                        state = state[0]
                
                frame_data = {
                    'index': sample_idx,
                    'frame_idx': len(episode_visualization_data),
                    'images': images_dict,
                    'state': state,
                    'predicted_value': float(pred_np[i]),
                    'target_value': float(target_np[i]),
                    'reward': float(reward_np[i]),
                    'advantage': float(target_np[i] - pred_np[i]),  # 简化的advantage计算
                }
                episode_visualization_data.append(frame_data)

                logger.info(
                    f"  Frame {sample_idx}: V̂={pred_np[i]:.4f}, Target={target_np[i]:.4f}, "
                    f"Reward={reward_np[i]:.4f}, Advantage={target_np[i] - pred_np[i]:.4f}"
                )
                
                # 实时显示/保存当前帧
                if window_name is None and len(episode_visualization_data) == 1:
                    # 第一帧时尝试创建窗口
                    window_name = f"Episode {episode_idx} - Value Function Analysis"
                    if has_gui:
                        try:
                            # 尝试不同的窗口标志
                            try:
                                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                            except:
                                # 如果WINDOW_NORMAL失败，尝试AUTOSIZE
                                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                            
                            # 尝试设置窗口大小
                            try:
                                cv2.resizeWindow(window_name, 1200, 300)
                            except:
                                pass  # 如果resize失败，继续使用默认大小
                            
                            logger.info(f"  ✅ OpenCV窗口已创建: {window_name}")
                        except Exception as e:
                            logger.error(f"  ❌ 创建窗口失败: {e}")
                            logger.error(f"  错误类型: {type(e).__name__}")
                            window_name = None
                            has_gui = False  # 更新全局状态
                            # 创建保存目录
                            save_dir = Path(f"episode_{episode_idx}_frames")
                            save_dir.mkdir(exist_ok=True)
                            logger.info(f"  📁 切换到保存模式，目录: {save_dir.absolute()}")
                    else:
                        # 没有GUI支持，直接使用保存模式
                        window_name = None
                        save_dir = Path(f"episode_{episode_idx}_frames")
                        save_dir.mkdir(exist_ok=True)
                        logger.info(f"  📁 保存目录: {save_dir.absolute()}")
                
                # 显示或保存当前帧
                display_img = create_frame_display(episode_idx, frame_data, len(episode_visualization_data), len(episode_frames))
                
                if window_name is not None:
                    # 有GUI支持，显示窗口
                    try:
                        cv2.imshow(window_name, display_img)
                        cv2.waitKey(1)  # 非阻塞等待，让窗口更新
                    except Exception as e:
                        logger.warning(f"  ⚠️ 显示帧失败，切换到保存模式: {e}")
                        window_name = None  # 切换到保存模式
                        save_dir = Path(f"episode_{episode_idx}_frames")
                        save_dir.mkdir(exist_ok=True)
                else:
                    # 无GUI支持，保存到文件
                    save_dir = Path(f"episode_{episode_idx}_frames")
                    frame_num = len(episode_visualization_data) - 1
                    save_path = save_dir / f"frame_{frame_num:03d}.png"
                    cv2.imwrite(str(save_path), display_img)
                    if (frame_num + 1) % 10 == 0:  # 每10帧打印一次进度
                        logger.info(f"  💾 已保存 {frame_num + 1}/{len(episode_frames)} 帧到文件...")
        
        # 将该episode的数据添加到总数据中
        all_predicted.extend(episode_predicted)
        all_target.extend(episode_target)
        all_rewards.extend(episode_rewards)
        all_indices.extend(episode_indices_list)
        
        # 动态可视化该episode（如果窗口已创建，提供交互式浏览）
        if episode_visualization_data:
            if window_name is not None:
                # 有GUI支持，提供交互式浏览
                logger.info(f"\n🎨 可视化 Episode {episode_idx} ({len(episode_visualization_data)} 帧)...")
                logger.info(f"  所有帧已实时显示在窗口中")
                logger.info(f"  操作说明: 按 'n' 下一帧, 'p' 上一帧, 'q' 退出, 's' 保存当前帧")
                visualize_episode(episode_idx, episode_visualization_data, args.interactive)
            else:
                # 无GUI支持，所有帧已保存到文件
                save_dir = Path(f"episode_{episode_idx}_frames")
                logger.info(f"\n💾 Episode {episode_idx} 所有帧已保存到: {save_dir.absolute()}")
                logger.info(f"  共 {len(episode_visualization_data)} 帧")
                logger.info(f"  可以使用图像查看器查看，或使用以下命令创建视频:")
                logger.info(f"    ffmpeg -r 10 -i {save_dir}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p episode_{episode_idx}.mp4")
        
        # 显示该episode的统计信息
        episode_pred = np.array(episode_predicted)
        episode_tgt = np.array(episode_target)
        episode_rew = np.array(episode_rewards)
        
        logger.info(f"\n📊 Episode {episode_idx} 统计:")
        logger.info(f"  预测值: mean={episode_pred.mean():.4f}, std={episode_pred.std():.4f}")
        logger.info(f"  目标值: mean={episode_tgt.mean():.4f}, std={episode_tgt.std():.4f}")
        logger.info(f"  奖励: mean={episode_rew.mean():.4f}, sum={episode_rew.sum():.4f}")
        
        if args.interactive:
            input(f"\n按 Enter 键继续下一个episode...")

    # ================================================================
    # 5. 汇总统计
    # ================================================================
    predicted = np.array(all_predicted)
    target = np.array(all_target)
    rewards = np.array(all_rewards)

    # 计算误差
    errors = predicted - target
    abs_errors = np.abs(errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_errors)

    # 相关系数
    if np.std(predicted) > 1e-8 and np.std(target) > 1e-8:
        correlation = np.corrcoef(predicted, target)[0, 1]
    else:
        correlation = float("nan")

    # R² score
    ss_res = np.sum((target - predicted) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else float("nan")

    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 Value Function 评估结果")
    logger.info("=" * 60)

    logger.info("")
    logger.info("--- 预测值 V̂(s) 统计 ---")
    logger.info(f"  Mean:   {predicted.mean():.4f}")
    logger.info(f"  Std:    {predicted.std():.4f}")
    logger.info(f"  Min:    {predicted.min():.4f}")
    logger.info(f"  Max:    {predicted.max():.4f}")

    logger.info("")
    logger.info("--- 目标值 target_value 统计 ---")
    logger.info(f"  Mean:   {target.mean():.4f}")
    logger.info(f"  Std:    {target.std():.4f}")
    logger.info(f"  Min:    {target.min():.4f}")
    logger.info(f"  Max:    {target.max():.4f}")

    logger.info("")
    logger.info("--- 即时奖励 reward 统计 ---")
    logger.info(f"  Mean:   {rewards.mean():.4f}")
    logger.info(f"  Std:    {rewards.std():.4f}")
    logger.info(f"  Min:    {rewards.min():.4f}")
    logger.info(f"  Max:    {rewards.max():.4f}")

    logger.info("")
    logger.info("--- 误差指标 ---")
    logger.info(f"  MSE:          {mse:.4f}")
    logger.info(f"  RMSE:         {rmse:.4f}")
    logger.info(f"  MAE:          {mae:.4f}")
    logger.info(f"  Correlation:  {correlation:.4f}")
    logger.info(f"  R² Score:     {r2:.4f}")
    
    # 计算 Advantage 统计
    advantages = target - predicted  # A = R_t - V(s)
    logger.info("")
    logger.info("--- Advantage (A = R_t - V(s)) 统计 ---")
    logger.info(f"  Mean:   {advantages.mean():.4f}  (应该接近0，表示平均预测准确)")
    logger.info(f"  Std:    {advantages.std():.4f}  (标准差，反映预测误差的分布)")
    logger.info(f"  Min:    {advantages.min():.4f}  (最小advantage，最差的预测)")
    logger.info(f"  Max:    {advantages.max():.4f}  (最大advantage，最好的预测)")
    logger.info(f"  正值比例: {(advantages > 0).sum() / len(advantages) * 100:.1f}%  (实际回报 > 预测价值)")
    logger.info(f"  负值比例: {(advantages < 0).sum() / len(advantages) * 100:.1f}%  (实际回报 < 预测价值)")
    logger.info(f"  零值数量: {(advantages == 0).sum()}  (完全匹配的数量)")
    
    # 警告：如果所有advantage都是0，说明有问题
    if np.allclose(advantages, 0, atol=1e-6):
        logger.warning("")
        logger.warning("⚠️  警告：所有 Advantage 都接近 0！")
        logger.warning("  这可能表示：")
        logger.warning("  1. 模型过拟合（完美记忆训练数据）")
        logger.warning("  2. target_value 计算错误")
        logger.warning("  3. 模型输出被限制为常数")
        logger.warning("  4. 数据预处理问题")
    elif np.std(advantages) < 0.01:
        logger.warning("")
        logger.warning("⚠️  警告：Advantage 标准差很小 (< 0.01)！")
        logger.warning("  这可能表示模型预测过于一致，缺乏区分度")
    elif np.abs(advantages.mean()) > 0.1:
        logger.warning("")
        logger.warning("⚠️  警告：Advantage 平均值偏离 0 较大！")
        logger.warning(f"  平均值 = {advantages.mean():.4f}，说明模型存在系统性偏差")

    logger.info("")
    logger.info("--- 逐样本对比 (前20个) ---")
    logger.info(f"  {'Index':>8s}  {'Predicted':>12s}  {'Target':>12s}  {'Error':>12s}  {'Reward':>10s}")
    logger.info(f"  {'-----':>8s}  {'--------':>12s}  {'------':>12s}  {'-----':>12s}  {'------':>10s}")
    for i in range(min(20, len(predicted))):
        logger.info(
            f"  {all_indices[i]:>8d}  {predicted[i]:>12.4f}  {target[i]:>12.4f}  "
            f"{errors[i]:>+12.4f}  {rewards[i]:>10.4f}"
        )

    # ================================================================
    # 6. 质量评估
    # ================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("🏆 质量评估")
    logger.info("=" * 60)

    if r2 > 0.8:
        grade = "优秀 ⭐⭐⭐"
    elif r2 > 0.5:
        grade = "良好 ⭐⭐"
    elif r2 > 0.2:
        grade = "一般 ⭐"
    else:
        grade = "较差 ❌ (可能需要更多训练或调整超参数)"

    logger.info(f"  R² = {r2:.4f} → {grade}")
    logger.info(f"  相关性 = {correlation:.4f}")

    if correlation > 0.7:
        logger.info("  ✅ 模型预测值与目标值有较强正相关，Value Function 学到了有意义的价值估计")
    elif correlation > 0.3:
        logger.info("  ⚠️ 模型预测值与目标值有弱正相关，可能需要更多训练步数")
    else:
        logger.info("  ❌ 模型预测值与目标值相关性很弱，建议检查数据质量或增加训练")

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ 评估完成")
    logger.info("=" * 60)
    
    # ================================================================
    # 7. 汇总统计（所有episodes）
    # ================================================================
    # 注意：可视化已经在每个episode处理时动态显示了


def create_frame_display(episode_idx, data, current_frame_idx, total_frames):
    """创建单个帧的显示图像。
    
    Args:
        episode_idx: episode索引
        data: 单个帧的可视化数据
        current_frame_idx: 当前帧索引（从1开始）
        total_frames: 总帧数
    
    Returns:
        display_img: OpenCV格式的图像 (BGR, uint8)
    """
    # 定义显示参数
    img_height = 160  # 每个相机图像的高度
    img_width = 213   # 每个相机图像的宽度
    info_panel_width = 300  # 信息面板宽度
    state_panel_width = 200  # state面板宽度
    
    # 计算每个帧的显示区域大小
    frame_display_height = img_height + 60  # 图像 + 文字
    frame_display_width = img_width * 3 + state_panel_width + info_panel_width
    
    # 创建显示画布
    display_img = np.ones((frame_display_height, frame_display_width, 3), dtype=np.uint8) * 255
    
    # 1. 显示3个相机视角的图像
    cam_keys = ['observation.images.cam_head', 'observation.images.cam_left', 'observation.images.cam_right']
    cam_names = ['Head', 'Left', 'Right']
    
    for cam_idx, (cam_key, cam_name) in enumerate(zip(cam_keys, cam_names)):
        x_offset = cam_idx * img_width
        
        if cam_key in data['images']:
            img = data['images'][cam_key]
            # 确保图像是uint8格式 [0, 255]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # 如果是RGB，转换为BGR（OpenCV使用BGR）
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 调整图像大小
            img_resized = cv2.resize(img, (img_width, img_height))
            display_img[10:10+img_height, x_offset:x_offset+img_width] = img_resized
        else:
            # 显示"No Image"文字
            cv2.putText(display_img, 'No Image', 
                       (x_offset + 20, img_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # 添加相机名称
        cv2.putText(display_img, f'{cam_name} Cam', 
                   (x_offset + 5, img_height + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 2. 显示state条形图
    state_x_offset = img_width * 3
    if data['state'] is not None:
        state = data['state']
        state_min = state.min()
        state_max = state.max()
        if state_max - state_min > 1e-8:
            state_normalized = (state - state_min) / (state_max - state_min)
        else:
            state_normalized = np.zeros_like(state)
        
        bar_width = max(1, state_panel_width // len(state))
        bar_height = img_height - 20
        
        for i, val in enumerate(state_normalized):
            x1 = state_x_offset + i * bar_width
            x2 = min(state_x_offset + (i + 1) * bar_width, state_x_offset + state_panel_width)
            y1 = 10 + img_height - int(val * bar_height)
            y2 = 10 + img_height
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (70, 130, 180), -1)
        
        # 添加state标题
        cv2.putText(display_img, f'State (dim={len(state)})', 
                   (state_x_offset + 5, img_height + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 3. 显示数值信息
    info_x_offset = state_x_offset + state_panel_width
    
    # 根据advantage值设置颜色
    advantage = data['advantage']
    if advantage > 0:
        text_color = (0, 200, 0)  # 绿色 (BGR)
    else:
        text_color = (0, 0, 200)  # 红色 (BGR)
    
    info_lines = [
        f"Frame: {current_frame_idx}/{total_frames}",
        f"Index: {data['index']}",
        f"V(s): {data['predicted_value']:.4f}",
        f"Target: {data['target_value']:.4f}",
        f"Reward: {data['reward']:.4f}",
        f"Advantage: {advantage:.4f}",
        f"Error: {data['target_value'] - data['predicted_value']:.4f}",
    ]
    
    y_start = 20
    line_height = 25
    for i, line in enumerate(info_lines):
        y_pos = y_start + i * line_height
        color = text_color if i >= 5 else (0, 0, 0)
        cv2.putText(display_img, line, 
                   (info_x_offset + 5, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 添加标题栏
    title_text = f"Episode {episode_idx} - Frame {current_frame_idx}/{total_frames}"
    cv2.putText(display_img, title_text, 
               (10, frame_display_height - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return display_img


def visualize_episode(episode_idx, vis_data, interactive=False):
    """使用OpenCV可视化单个episode的所有帧，动态显示。
    
    注意：这个函数现在主要用于交互式浏览已推理完成的帧。
    实时显示已经在推理循环中完成。
    
    Args:
        episode_idx: episode索引
        vis_data: 包含该episode所有帧的可视化数据列表
        interactive: 是否交互式模式（等待用户按键）
    """
    num_frames = len(vis_data)
    window_name = f"Episode {episode_idx} - Value Function Analysis"
    
    # 检查OpenCV是否有GUI支持
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 300)
        has_gui = True
    except:
        has_gui = False
        logger.warning("  OpenCV没有GUI支持，将自动保存所有帧到文件")
    
    if not has_gui:
        # 如果没有GUI支持，保存所有帧到文件
        output_dir = Path(f"episode_{episode_idx}_frames")
        output_dir.mkdir(exist_ok=True)
        for idx, data in enumerate(vis_data):
            display_img = create_frame_display(episode_idx, data, idx + 1, num_frames)
            output_path = output_dir / f"frame_{idx:03d}.png"
            cv2.imwrite(str(output_path), display_img)
        logger.info(f"  ✅ 已保存 {num_frames} 帧到: {output_dir}")
        return
    
    logger.info(f"  开始显示 {num_frames} 帧...")
    logger.info(f"  操作说明: 按 'n' 下一帧, 'p' 上一帧, 'q' 退出, 's' 保存当前帧")
    
    current_frame_idx = 0
    
    while True:
        if current_frame_idx < 0:
            current_frame_idx = 0
        if current_frame_idx >= num_frames:
            current_frame_idx = num_frames - 1
        
        data = vis_data[current_frame_idx]
        display_img = create_frame_display(episode_idx, data, current_frame_idx + 1, num_frames)
        
        # 显示图像
        cv2.imshow(window_name, display_img)
        
        # 等待按键
        if interactive:
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(100) & 0xFF  # 100ms延迟，自动播放
        
        if key == ord('q') or key == 27:  # 'q' 或 ESC
            break
        elif key == ord('n') or key == 83:  # 'n' 或右箭头
            current_frame_idx += 1
        elif key == ord('p') or key == 81:  # 'p' 或左箭头
            current_frame_idx -= 1
        elif key == ord('s'):  # 's' 保存当前帧
            output_path = Path(f"episode_{episode_idx}_frame_{current_frame_idx}.png")
            cv2.imwrite(str(output_path), display_img)
            logger.info(f"  ✅ 已保存到: {output_path}")
        elif key == ord(' '):  # 空格键暂停/继续
            cv2.waitKey(0)
        
        # 如果到达最后一帧且非交互模式，自动退出
        if not interactive and current_frame_idx >= num_frames - 1:
            cv2.waitKey(1000)  # 显示最后一帧1秒
            break
    
    try:
        cv2.destroyWindow(window_name)
    except:
        pass


def visualize_samples(vis_data):
    """使用OpenCV可视化采样的样本（已废弃，使用visualize_episode代替）。
    
    这个函数保留是为了兼容性，实际使用visualize_episode。
    """
    logger.warning("visualize_samples已废弃，使用visualize_episode代替")
    if vis_data:
        visualize_episode(0, vis_data, interactive=True)


if __name__ == "__main__":
    main()
