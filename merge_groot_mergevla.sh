#!/bin/bash
# MergeVLA 风格的 GROOT 模型融合脚本
# 基于论文: https://arxiv.org/pdf/2511.18810
#
# 使用方式：
#   ./merge_groot_mergevla.sh              # 默认使用 GPU 0（单卡）
#   ./merge_groot_mergevla.sh 4            # 使用 GPU 4（单卡）
#   ./merge_groot_mergevla.sh 0,1          # 使用多个 GPU（通过 CUDA_VISIBLE_DEVICES）
#
# ⭐ 多卡训练（推荐，使用 accelerate）：
#   ./merge_groot_mergevla.sh --multi-gpu --num-gpus 2     # 使用 2 张卡
#   ./merge_groot_mergevla.sh --multi-gpu --gpus 0,1       # 使用 GPU 0 和 1
#   ./merge_groot_mergevla.sh --multi-gpu                  # 使用所有可用 GPU
#
# ⭐ Weights & Biases 实时监控：
#   ./merge_groot_mergevla.sh --wandb                                   # 启用 wandb 监控
#   ./merge_groot_mergevla.sh --wandb --wandb-project my-project        # 自定义项目名
#   ./merge_groot_mergevla.sh --wandb --wandb-run my-run                # 自定义运行名
#   ./merge_groot_mergevla.sh --multi-gpu --gpus 6,7 --wandb            # 多卡 + wandb

set -e

# ============================================================
# 参数解析
# ============================================================
USE_MULTI_GPU=false
NUM_GPUS=""
GPU_IDS=""
DEVICE="cuda:0"

# ⭐ Weights & Biases 实时监控参数
USE_WANDB=false
WANDB_PROJECT="groot-mergevla"
WANDB_RUN_NAME=""
WANDB_ENTITY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --multi-gpu|--multi_gpu)
            USE_MULTI_GPU=true
            shift
            ;;
        --num-gpus|--num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        # ⭐ Weights & Biases 参数
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --wandb-project|--wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-run|--wandb_run)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --wandb-entity|--wandb_entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        *)
            # 兼容旧的位置参数（单个或多个 GPU ID）
            if [ -z "$GPU_IDS" ]; then
                GPU_IDS="$1"
            fi
            shift
            ;;
    esac
done

# ============================================================
# GPU 配置
# ============================================================
if [ "$USE_MULTI_GPU" = true ]; then
    # ⭐ 多卡训练模式（使用 accelerate）
    # ⚠️ 注意：CUDA_VISIBLE_DEVICES 会在启动命令处设置，这里只计算数量和显示信息
    if [ -n "$GPU_IDS" ]; then
        # 计算 GPU 数量（不在这里设置 CUDA_VISIBLE_DEVICES，在启动时设置）
        NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
        GPU_INFO="🚀 多卡训练: 物理 GPU $GPU_IDS -> 逻辑 cuda:0~$((NUM_GPUS-1)) ($NUM_GPUS 张卡)"
    elif [ -n "$NUM_GPUS" ]; then
        # 使用指定数量的 GPU（从 0 开始）
        GPU_INFO="🚀 多卡训练: $NUM_GPUS 张卡（使用 accelerate）"
    else
        # 使用所有可用 GPU
        NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || nvidia-smi --list-gpus | wc -l)
        GPU_INFO="🚀 多卡训练: 所有可用 GPU ($NUM_GPUS 张卡，使用 accelerate)"
    fi
    DEVICE="cuda:0"  # accelerate 会自动分配设备，这个值不影响多卡训练
else
    # 单卡训练模式
    if [ -n "$GPU_IDS" ]; then
        # 检查是否包含逗号（多个 GPU）
        if [[ "$GPU_IDS" == *","* ]]; then
            # 多个 GPU：使用 CUDA_VISIBLE_DEVICES
            export CUDA_VISIBLE_DEVICES="$GPU_IDS"
            DEVICE="cuda:0"  # 在可见的 GPU 中，使用第一个
            GPU_INFO="GPU: $GPU_IDS (CUDA_VISIBLE_DEVICES)"
        else
            # 单个 GPU：也使用 CUDA_VISIBLE_DEVICES 来限制只使用这个 GPU
            export CUDA_VISIBLE_DEVICES="$GPU_IDS"
            DEVICE="cuda:0"  # 在可见的 GPU 中，使用第一个（逻辑上就是指定的 GPU）
            GPU_INFO="GPU: $GPU_IDS (CUDA_VISIBLE_DEVICES, device=cuda:0)"
        fi
    else
        # 默认使用 cuda:0，但不设置 CUDA_VISIBLE_DEVICES（使用系统默认）
        # 如果之前环境中有 CUDA_VISIBLE_DEVICES，需要清理
        unset CUDA_VISIBLE_DEVICES
        DEVICE="cuda:0"
        GPU_INFO="GPU: $DEVICE (默认，未限制 CUDA_VISIBLE_DEVICES)"
    fi
fi

# 模型路径（根据实际情况修改）
NARROWER_PATH="/home/kangkk/humanoid_groot_base/outputs/0112_h100x4_groot_cross_attention_narrower_very_conservative/checkpoints/020000/pretrained_model"
WIDER_PATH="/home/kangkk/humanoid_groot_base/outputs/0113_h100x4_groot_cross_attention_wider_very_conservative_mix_dense/checkpoints/014000/pretrained_model"
BASE_MODEL_PATH="${NARROWER_PATH}"  # 如果 base == narrower，则 τ_narrower = 0
OUTPUT_PATH="./outputs/merged_groot_mergevla/pretrained_model"

echo "=========================================="
echo "🚀 MergeVLA-Style GROOT Model Merging"
echo "=========================================="
echo "Narrower: ${NARROWER_PATH}"
echo "Wider: ${WIDER_PATH}"
echo "Base: ${BASE_MODEL_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "${GPU_INFO}"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "⚠️ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
if [ "$USE_MULTI_GPU" = true ]; then
    echo ""
    echo "💡 多卡训练提示:"
    echo "   - 学习率会自动缩放: lr = base_lr × num_gpus^0.3"
    echo "   - 有效 batch size = batch_size × num_gpus"
fi
echo "=========================================="
echo ""

# 运行 MergeVLA 融合
# ⚠️ 关键参数说明：
#   --adapter_epochs: 增加到 100 以确保充分学习
#   --lora_rank: 增加到 32 以提高适配器容量
#   --sparsity: 降低到 0.3 以激活更多参数（减少稀疏性）
#   --merge_action_head: 尝试融合 action_head（GROOT 使用 cross-attention，更安全）
#   --batch_size: 8 (H100 80GB 可以支持更大的 batch，提高训练稳定性和梯度估计质量)
#                 注意: batch_size=1 会导致 loss 和 grad_norm 波动非常大，训练不稳定
#
# ⭐ Episode-based 采样模式（保持时序连续性）：
#   --episode-based: 按 episode 组织数据，保持动作轨迹的时序关系
#   --num-episodes: 每个数据集采样的 episode 数量（None=全部）
#   这样适配层可以学习到完整的动作轨迹模式：接近→抓取→提起→移动
#
# 💡 采样模式选择：
#   1. --episode-based --num-episodes 5：每个数据集采样 5 个完整 episode（推荐！）
#   2. --use-all-frames：使用所有帧（数据量大，训练慢，但效果最好）
#   3. 不加这些参数：随机帧采样（可能破坏时序关系）
#
# 🔥 MergeVLA Section 4.1: 参数级稀疏掩码融合（默认启用）
#   --use_sparse_merge: 使用公式 S_m = I[|τ_m| > λ|τ_merge - τ_m|] 计算任务掩码
#   --sparse_merge_lambda: 容忍度系数 λ（论文默认 1.0，越大越稀疏）
#   这会过滤掉参数级别的符号冲突，保留对各任务有意义的参数更新
#
# ⭐ MoE 模式（推荐！基于论文 Section 3.2 "Expert Head" 概念）：
#   --use_moe: 保留每个任务独立的 action_head (DiT)，不融合
#              因为 DiT 对参数变化极其敏感，无法直接线性融合
#              推理时通过 Smart Routing 或固定路由选择使用哪个专家
#   
#   ❌ 错误方式：--merge_action_head（线性融合 DiT，会导致性能崩溃）
#   ✅ 正确方式：--use_moe（保留两个独立的 DiT，通过路由选择）

# ============================================================
# 训练参数
# ============================================================
TRAIN_ARGS=(
    --method mergevla
    --narrower_path "${NARROWER_PATH}"
    --wider_path "${WIDER_PATH}"
    --base_model_path "${BASE_MODEL_PATH}"
    --output_path "${OUTPUT_PATH}"
    --adapter_type sparse_lora
    # ⭐ 增强配置：增加 Adapter 容量以提高 wider 任务精度
    --lora_rank 32           # 从 32 增加到 64，参数量翻倍
    --sparsity 0.6           # 从 0.5 增加到 0.6，激活更多参数
    # ⭐ 关键修改：启用 adapter 训练来修正特征分布偏移！
    # 由于融合后的 backbone 是 50/50 混合，对 wider DiT 来说是"陌生"的分布
    # adapter 会学习：
    #   - narrower 任务：50/50 特征 → narrower 特征
    #   - wider 任务：50/50 特征 → wider 特征
    --adapter_epochs 30      # 50 个 epoch
    --adapter_lr 5e-4        # 略微降低学习率，提高稳定性
    # --bypass_adapter  # ⚠️ 不要跳过 adapter！这是修正特征分布的关键
    --batch_size 32
    --device "${DEVICE}"
    --use_default_datasets
    --use_moe
    --episode-based
    --num-episodes 40
    --use_sparse_merge
    --sparse_merge_lambda 1.0
    --warmup_ratio 0.1
    --gradient_accumulation_steps 4
    --max_grad_norm 1.0
    --weight_decay 0.01
)

# ⭐ 添加 Weights & Biases 参数（如果启用）
if [ "$USE_WANDB" = true ]; then
    TRAIN_ARGS+=( --use_wandb )
    TRAIN_ARGS+=( --wandb_project "${WANDB_PROJECT}" )
    if [ -n "$WANDB_RUN_NAME" ]; then
        TRAIN_ARGS+=( --wandb_run_name "${WANDB_RUN_NAME}" )
    fi
    if [ -n "$WANDB_ENTITY" ]; then
        TRAIN_ARGS+=( --wandb_entity "${WANDB_ENTITY}" )
    fi
    echo "📊 Weights & Biases 实时监控已启用"
    echo "   Project: ${WANDB_PROJECT}"
    [ -n "$WANDB_RUN_NAME" ] && echo "   Run: ${WANDB_RUN_NAME}"
    [ -n "$WANDB_ENTITY" ] && echo "   Entity: ${WANDB_ENTITY}"
    echo ""
fi

# ============================================================
# 启动训练
# ============================================================

# ⭐ 关键：设置离线模式，避免多卡训练时多进程同时尝试从 HuggingFace 下载
# 这可以防止 SSL 连接错误和网络竞争条件
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "🔒 已启用 HuggingFace 离线模式 (HF_HUB_OFFLINE=1)"
echo "   请确保模型 nvidia/GR00T-N1.5-3B 已缓存在 ~/.cache/huggingface/hub/"

if [ "$USE_MULTI_GPU" = true ]; then
    # ⭐ 多卡训练：使用 accelerate launch
    echo ""
    echo "🚀 使用 accelerate 启动多卡训练..."
    echo "   GPU 数量: $NUM_GPUS"
    
    # ⚠️ 关键修复：使用 CUDA_VISIBLE_DEVICES 限制可见 GPU
    # accelerate 然后使用逻辑索引 (0, 1, 2...) 访问这些 GPU
    # 例如：CUDA_VISIBLE_DEVICES=6,7 时，GPU 6 变成逻辑 cuda:0，GPU 7 变成 cuda:1
    if [ -n "$GPU_IDS" ]; then
        echo "   物理 GPU IDs: $GPU_IDS"
        echo "   逻辑映射: GPU $GPU_IDS -> cuda:0, cuda:1, ..."
        echo ""
        # ⭐ 只设置 CUDA_VISIBLE_DEVICES，不用 --gpu_ids
        # 因为 CUDA_VISIBLE_DEVICES 已经限制了可见性，accelerate 使用逻辑索引
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch \
            --multi_gpu \
            --num_processes=${NUM_GPUS} \
            --mixed_precision=bf16 \
            scripts/train_weight_merge.py \
            "${TRAIN_ARGS[@]}"
    else
        echo ""
        # 不指定 GPU IDs，让 accelerate 自动选择所有可用 GPU
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch \
            --multi_gpu \
            --num_processes=${NUM_GPUS} \
            --mixed_precision=bf16 \
            scripts/train_weight_merge.py \
            "${TRAIN_ARGS[@]}"
    fi
else
    # 单卡训练：直接使用 python
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/train_weight_merge.py \
        "${TRAIN_ARGS[@]}"
fi

echo ""
echo "=========================================="
echo "✅ MergeVLA (MoE 模式) merging completed!"
echo "=========================================="

# ============================================================
# 🧠 Step 2: 训练 Router Network（智能路由网络）
# ============================================================
echo ""
echo "=========================================="
echo "🧠 Step 2: Training Router Network"
echo "=========================================="
echo ""
echo "训练一个神经网络来自动选择正确的专家..."
echo "这比启发式路由更可靠，是 MoE 的标准做法。"
echo ""

# Router Network 数据集路径
NARROWER_DATASET="/home/lab/humanoid_groot/lerobot_data/v3_0_dataset/1221_5w_random_height_4322_4611_narrower"
WIDER_DATASET="/home/lab/humanoid_groot/lerobot_data/v3_0_dataset/1221_5w_random_height_4322_4611_wider"

# 检查数据集是否存在
if [ -d "$NARROWER_DATASET" ] && [ -d "$WIDER_DATASET" ]; then
    echo "📂 发现训练数据集:"
    echo "   Narrower: $NARROWER_DATASET"
    echo "   Wider: $WIDER_DATASET"
    echo ""
    
    # 训练 Router Network
    python scripts/train_router_network.py \
        --model-path "${OUTPUT_PATH}" \
        --dataset-paths "$NARROWER_DATASET" "$WIDER_DATASET" \
        --task-names narrower wider \
        --epochs 20 \
        --samples-per-task 500 \
        --batch-size 8 \
        --device "${DEVICE}"
    
    echo ""
    echo "✅ Router Network 训练完成！"
    echo "   保存位置: ${OUTPUT_PATH}/router_network.pt"
else
    echo "⚠️ 未找到训练数据集，跳过 Router Network 训练"
    echo "   如需使用智能路由，请手动运行:"
    echo ""
    echo "   python scripts/train_router_network.py \\"
    echo "       --model-path ${OUTPUT_PATH} \\"
    echo "       --dataset-paths /path/to/narrower /path/to/wider \\"
    echo "       --task-names narrower wider \\"
    echo "       --epochs 20"
fi

echo ""
echo "=========================================="
echo "🎉 All Steps Completed!"
echo "=========================================="
echo ""
echo "💡 推理命令（支持多种路由模式）："
echo ""
echo "   # ⭐⭐ 方式 1: Router Network（最智能，推荐！）"
echo "   python scripts/eval_merged_groot_on_dataset.py \\"
echo "       --model-path ${OUTPUT_PATH} \\"
echo "       --dataset-root /path/to/dataset \\"
echo "       --episode 0 --visualize \\"
echo "       --router-network"
echo ""
echo "   # 方式 2: 固定专家（已知任务类型时使用）"
echo "   python scripts/eval_merged_groot_on_dataset.py \\"
echo "       --model-path ${OUTPUT_PATH} \\"
echo "       --dataset-root /path/to/narrower_dataset \\"
echo "       --episode 0 --visualize \\"
echo "       --task-type narrower"
echo ""
echo "   # 方式 3: Action Head Voting（备选）"
echo "   python scripts/eval_merged_groot_on_dataset.py \\"
echo "       --model-path ${OUTPUT_PATH} \\"
echo "       --dataset-root /path/to/dataset \\"
echo "       --episode 0 --visualize \\"
echo "       --action-head-voting"
echo ""
echo "💡 在线推理（eval/eval_merged_groot.py）："
echo ""
echo "   # 使用 narrower 专家"
echo "   python eval/eval_merged_groot.py \\"
echo "       --model_path ${OUTPUT_PATH} \\"
echo "       --rtc.enabled=true \\"
echo "       --task=\"Depalletize the box\" \\"
echo "       --task_type=narrower"
echo ""
echo "   # 使用 wider 专家"
echo "   python eval/eval_merged_groot.py \\"
echo "       --model_path ${OUTPUT_PATH} \\"
echo "       --rtc.enabled=true \\"
echo "       --task=\"Depalletize the box\" \\"
echo "       --task_type=wider"
