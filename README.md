# Groot robot fine-turn
## env prepare
```bash
# 先安装lerobot其他库
conda create -y -n lerobot_groot python=3.10
conda activate lerobot_groot
conda install ffmpeg
pip install -e .
pip install lerobot[groot]

# ubuntu22.04可以直接（pip install flash-attn==2.8.1），但是20.04要抓
pip install lerobot[groot]
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# 必装
pip install transformers
pip install peft

# 推理环境安装
pip3 install rospkg
pip3 install scipy 
cd /home/lab/kuavo-ros-control-rewACT/src/kuavo_humanoid_sdk && ./install.sh
pip3 install websockets 
pip3 install deprecated

# eef action space
conda install pinocchio
pip install PyQt5
pip install PySide2
(lerobot_groot_base) lab@lab [~/humanoid_groot/eval/IK_eef_eval] git:(KangKK/dev/delta_eef_pose_action) ✗ ➜  conda list | grep pinocchio [15:00:04]
libpinocchio              3.9.0                h54b0c19_0    conda-forge
pinocchio                 3.9.0                hff52083_0    conda-forge
pinocchio-python          3.9.0           py310h1e2979e_0    conda-forge
```
## 验证脚本看文末
## 模型架构分析
* ![model_pipeline](./docs/IMG/image_pipeline.png)
### backbone原始定义 - vision encoder
SigLIP Vision Model (SiglipVisionModel)
- 来源：transformers.models.siglip.modeling_siglip.SiglipVisionModel
- 配置：SiglipVisionConfig
- 注意力实现：默认使用 flash_attention_2
```python
            if config.vision_config.model_type == "siglip_vision_model":
                config.vision_config._attn_implementation = "flash_attention_2"
                self.vision_model = SiglipVisionModel(config.vision_config)
            else:
                raise NotImplementedError(f"{config.vision_config.model_type} is not implemented.")
```
* `/home/lab/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5/config.json`
```json
    "vision_config": {
        "_attn_implementation_autoset": true,
        "attention_dropout": 0,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "image_size": 224,
        "intermediate_size": 4304,
        "layer_norm_eps": 0.000001,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 27,
        "patch_size": 14,
        "torch_dtype": "bfloat16"
    }
```
### backbone原始定义 - LLM
* 配置文件存放于`/home/lab/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5/config.json`
* 明确说明了使用`qwen3-1.7B`作为LLM，同时fine-turn使用fp16精度
```json
    "text_config": {
        "_attn_implementation_autoset": true,
        "_name_or_path": "Qwen/Qwen3-1.7B",
        "architectures": [
            "Qwen3ForCausalLM"
        ],
        "attention_bias": false,
        "attention_dropout": 0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 6144,
        "max_position_embeddings": 40960,
        "max_window_layers": 28,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "rms_norm_eps": 0.000001,
        "rope_scaling": null,
        "rope_theta": 1000000,
        "sliding_window": null,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "use_cache": false,
        "use_sliding_window": false,
        "vocab_size": 151680
    },
```
### Groot - action head 定义
* `/home/lab/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e/config.json`定义存放在Groot-N1.5-3B的原始路径下
#### 关于action head的定义
```json
  "action_head_cfg": {
    "action_dim": 32,
    "action_horizon": 16,
    "add_pos_embed": true,
    "backbone_embedding_dim": 2048,
    "diffusion_model_cfg": {
      "attention_head_dim": 48,
      "cross_attention_dim": 2048,
      "dropout": 0.2,
      "final_dropout": true,
      "interleave_self_attention": true,
      "norm_type": "ada_norm",
      "num_attention_heads": 32,
      "num_layers": 16,
      "output_dim": 1024,
      "positional_embeddings": null
    },
```
#### 关于Groot使用backbone的定义
* 重点关注`select_layer`的标签，意味着Groot在使用的backbone_cfg.select_layer：控制保留多少层（12层），实际参数量在0.91B 而不是1.7B
```json
  "backbone_cfg": {
    "eagle_path": "NVEagle/eagle_er-qwen3_1_7B-Siglip2_400M_stage1_5_128gpu_er_v7_1mlp_nops",
    "load_bf16": false,
    "project_to_dim": null,
    "reproject_vision": false,
    "select_layer": 12,
    "tune_llm": false,
    "tune_visual": true,
    "use_flash_attention": true
  },
```
---
## Dataset prepare
* 将数据从v2.1的格式转换为v3.0格式
```bash
./convert_dataset.sh
```
## 可视化转换好的数据
```bash
conda activate lerobot_groot
lerobot-dataset-viz \                                    
    --repo-id /home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1115_depalletize_left_mix_dagger_keyboard \ 
    --episode-index 0
```

## train
* 直接训练，batch_size=8，大概占用26G VRAM的大小
```bash
./train_groot.sh

# 继续训练（从 8000 步继续，训练到 40000 步）
# 修改脚本中的 RESUME=true
RESUME=true bash train_groot.sh
```

## 推理eval
Groot Model Lerobot (0.4.2版本)
- ![IMG](./docs/IMG/image.png)
- 微调40K (loss已经降到很低0.0003)
- State
  - STATE_COMPONENTS = ["J_q", "Claw_pos", "Com_z_pitch"]
- Aciton
  - ACTION_COMPONENTS = ["Left_arm", "Right_arm",  "Left_claw", "Right_claw", "Cmd_pose_z", "Cmd_pose_pitch"]

训练集验证
- 因为predict_action_chunk输出的值是归一化后的-1到1的值，所以实际执行动作的时候还得根据数据集当中的action dim的min和max进行反归一化才能获取到正确的值
eval on dataset
```bash
python scripts/eval_on_dataset_lowpass.py \
    --ckpt-path /home/lab/lerobot_groot/outputs/train/12_20_groot_random_4_box_a100_one_gpu/checkpoints/040000/pretrained_model \
    --dataset-root /home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1221_5w_random_height_4322_4611/  \
    --episode 59 \
    --action-chunk-size 16 \
    --infer-per-frame 16 --task-description "Depalletize the box"
```

### real time sync Eval
```bash
python scripts/eval_depalletize_camera_model_reload_limit_vel.py \
    --eval \
    --ckpt-path /home/lab/lerobot_groot/outputs/train/12_20_groot_random_4_box_a100_one_gpu/checkpoints/048000/pretrained_model \
    --model-type groot \
    --action_chunk_size 16 \
    --task-description "Depalletize the box" --skip-chunk-ratio 0.5 --model-action-dt 0.1 --sync-mode --max-joint-velocity 1.0 --skip-chunk-from-end
```

### Sim仿真验证
* 实验结果标明，Groot对于图像更为依赖，不依赖state，与预期相符
* 实时Sim推理
- 动作准确度以及chunk和预期相符，推理时占用10GB VRAM
* ![eval_real_time](./docs/IMG/image1.png)

### t-SNE分析
* ![t-SNE](./docs/IMG/t-SNE.png)
```bash
# 注意库的安装 plotly sklearn
python scripts/visualize_task_tsne_3d.py \            
    --dataset-path /home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1215_5w_groot_4311_4322_4611_4633 \
    --ckpt-path /home/lab/lerobot_groot/outputs/train/12_23_groot_random_mix_4_box_mix_dense_a100_one_gpu/checkpoints/080000/pretrained_model \
    --output ./t-SNE/tsne_3d_visualization.png
```

### 2026/1/22 - 最新验证脚本
#### 非权重融合
* 具体查看 [GROOT_N1_DETAILED_ARCHITECTURE 架构说明](./docs/GROOT_N1_DETAILED_ARCHITECTURE.md)
* 训练集上的验证
```bash
python scripts/eval_on_dataset_lowpass.py \
    --ckpt-path //media/ubuntu/New/manipulation/humanoid_groot/outputs/narrow/checkpoints/020000/pretrained_model  \
    --dataset-root /home/ubuntu/humanoid_groot/lerobot_data/v3_0_dataset/1221_5w_random_height_4322_4611 \
    --episode 2 \
    --action-chunk-size 16 \
    --infer-per-frame 16 --task-description "Depalletize the box"
```
* 同步推理模式（带动作逐帧采样加速）
```bash
python scripts/eval_depalletize_camera_model_reload_limit_vel_select.py --eval --ckpt-path /home/lab/humanoid_groot/outputs/train/0109_h100x4_groot_cross_attention_mix_vision_token_64_image_enhancement_learnable_weights_arm_coordination/checkpoints/012000/pretrained_model --model-type groot --action_chunk_size 16 --task-description "Depalletize the box" --model-action-dt 0.1 --sync-mode --max-joint-velocity 1.0 --chunk-start 1 --chunk-end 7 --constant-velocity --action-stride 2
```
* RTC推理模式
```bash
# 多模型推理
python eval/eval_multi_model.py --rtc.enabled=true --rtc.execution_horizon=10 --task="Depalletize the box" --duration=30

# 单模型推理
 python eval/eval.py --policy.path=/home/lab/humanoid_groot/outputs/train/0112_h100x4_groot_cross_attention_narrower_very_conservative/checkpoints/020000/pretrained_model --policy.device=cuda --rtc.enabled=true --rtc.execution_horizon=10 --task="Depalletize the box" --duration=30
```

#### 权重融合
* 具体查看 [权重融合架构说明](./docs/MERGEVLA_EXPLANATION.md)
##### 训练LoRA adapt 层
```bash
./merge_groot_mergevla.sh --multi-gpu --gpus 6,7 --wandb \
    --wandb-project groot-mergevla \ 
```
##### 训练 router_network 分类器

**方式 1：使用 Shell 脚本（推荐）** ⭐
```bash
# ⭐ 数据集路径已硬编码到代码中，无需手动指定
# 支持每个任务多个数据集，会自动合并：
# - narrower: 4个数据集
# - wider: 4个数据集

# 使用默认配置（GPU 0，所有数据）：
./train_router_network.sh

# 指定 GPU：
./train_router_network.sh --gpu 4
./train_router_network.sh -g 0,1  # 使用 GPU 0 和 1（会使用第一个）

# 自定义训练参数：
./train_router_network.sh \
    --gpu 6 \
    --epochs 30 \
    --batch-size 16 \
    --learning-rate 2e-3

# 限制样本数量（快速测试）：
./train_router_network.sh \
    --gpu 0 \
    --samples-per-task 500 \
    --epochs 10

# 查看所有选项：
./train_router_network.sh --help
```

**方式 2：直接使用 Python 脚本**
```bash
# 使用所有数据训练（推荐，数据更充分）：
python scripts/train_router_network.py \
    --model-path /home/lab/humanoid_groot/outputs/0122_merged_groot_mergevla/pretrained_model \
    --epochs 10 \
    --batch-size 8 \
    --device cuda:0

# 或者限制每个任务的样本数量（快速测试用）：
python scripts/train_router_network.py \
    --model-path /home/lab/humanoid_groot/outputs/0122_merged_groot_mergevla/pretrained_model \
    --epochs 10 \
    --samples-per-task 500 \
    --batch-size 8 \
    --device cuda:4

# 使用默认模型路径（如果模型路径固定）：
python scripts/train_router_network.py \
    --epochs 10 \
    --batch-size 8 \
    --device cuda:0
```
##### 训练集上的验证
```bash
python scripts/eval_merged_groot_on_dataset.py \
    --model-path /home/lab/humanoid_groot/outputs/0122_merged_groot_mergevla/pretrained_model \
    --dataset-root /home/lab/humanoid_groot/lerobot_data/v3_0_dataset/unpack_4322_short_dense \
    --episode 43 --visualize --action-chunk-size 16 --infer-per-frame 16 \
    --router-network 
```
##### 实时RTC推理验证
```bash
python eval/eval_merged_groot.py \
    --model_path ./outputs/0122_merged_groot_mergevla/pretrained_model \
    --rtc.enabled=true \
    --rtc.execution_horizon=10 \
    --task="Depalletize the box" \
    --duration=30 \
    --use_router_network=true 
```

## eef action sapce
```bash
 python verify_ik_fk_consistency.py \
    --dataset-path /home/lab/humanoid_groot/lerobot_data/v3_0_dataset/0122_4322_eef_test \
    --episode-idx 0 \
    --urdf-path /home/lab/kuavo-manip/lerobot_datasets/utils/biped_s60_only_arm.urdf \
    --model-type 60 \
    --robot-version 5_wheel \
    --output-dir ./ik_fk_verification_results --interactive
```
* ![对比结果](./eval/IK_eef_eval/ik_fk_verification_results/3d_trajectory.png)