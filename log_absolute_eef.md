(lerobot_groot_base) lab@lab [~/humanoid_groot] git:(KangKK/dev/delta_eef_pose_action) ✗ ➜  python scripts/eval_depalletize_camera_model_reload_limit_vel_select_eef.py --eval --ckpt-path /home/lab/humanoid_groot/outputs/train/0124_multi_dataset_h100x4_absolute_eef_4322_2X3_groot_cross-attention_ignore_rotation/checkpoints/020000/pretrained_model --model-type groot --action_chunk_size 16 --task-description "Depalletize the box" --model-action-dt 0.1 --sync-mode --max-joint-velocity 1.0 --chunk-start 1 --chunk-end 7 --constant-velocity --action-stride 1 --pause-before-chunk

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/home/lab/humanoid_groot/scripts/eval_depalletize_camera_model_reload_limit_vel_select_eef.py", line 187, in <module>
    from robot_envs.kuavo_depalletize_env import GrabBoxMpcEnv
  File "/home/lab/humanoid_groot/robot_envs/kuavo_depalletize_env.py", line 19, in <module>
    from cv_bridge import CvBridge
  File "/opt/ros/noetic/lib/python3/dist-packages/cv_bridge/__init__.py", line 6, in <module>
    from cv_bridge.boost.cv_bridge_boost import cvtColorForDisplay, getCvType
AttributeError: _ARRAY_API not found
📊 State configuration: ['J_q', 'Claw_pos'] -> 16D [14(J_q)+2(Claw_pos)]
🎮 Action configuration: ['Left_arm', 'Right_arm', 'Left_claw', 'Right_claw'] -> 16D [7(Left_arm)+7(Right_arm)+1(Left_claw)+1(Right_claw)]
 =================== Action Mode: ABSOLUTE ================= 
 =================== Task Data Mode: STRATEGY ================= 
📊 State configuration: ['J_q', 'Claw_pos'] -> 16D [14(J_q)+2(Claw_pos)]
 =================== State dimensions: 16 ================= 
 =================== Action components: ['Left_arm', 'Right_arm', 'Left_claw', 'Right_claw'] ================= 
 =================== State components: ['J_q', 'Claw_pos'] ================= 
 =================== Set camera topic based on CAMERA_COMPONENTS: ['cam_head', 'cam_left', 'cam_right'] ==================
kuavo-humanoid-sdk log_file: ./log/kuavo_humanoid_sdk/kuavo_humanoid_sdk.log
ERROR:rosout:com_height parameter not found
当前为轮臂模型
2026-01-28 14:27:08,113 - INFO - [WheelArmROSControl] ROS接口初始化成功
2026-01-28 14:27:08,113 - INFO - [WheelArmROSControl] 轮臂ROS控制模块初始化完成
2026-01-28 14:27:08,114 - INFO - Using standard mode, subscribing to: /sensors_data_raw
2026-01-28 14:27:09,336 - ERROR - Service call failed: timeout exceeded while waiting for service /humanoid_get_current_gait_name
2026-01-28 14:27:09,337 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:27:09,437 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:27:09,538 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:27:09,639 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:27:09,739 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:27:09,840 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:27:09,941 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:27:10,041 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:27:10,142 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:27:10,243 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:27:10,344 - WARNING - Timeout waiting for MPC observation data
2026-01-28 14:27:11,553 - ERROR - [Error] get arm ctrl mode: timeout exceeded while waiting for service /humanoid_get_arm_ctrl_mode
2026-01-28 14:27:13,668 - ERROR - Failed to connect to service /get_mm_ctrl_frame: timeout exceeded while waiting for service /get_mm_ctrl_frame
2026-01-28 14:27:15,808 - ERROR - Failed to connect to service /get_mm_wbc_arm_trajectory_control: timeout exceeded while waiting for service /get_mm_wbc_arm_trajectory_control
Warning: TF_OLD_DATA ignoring data from the past (Possible reasons are listed at http://wiki.ros.org/tf/Errors%20explained) for frame base_link (parent dummy_link) at time 1769579082.022296 according to authority /nodelet_manager
         at line 277 in /tmp/binarydeb/ros-noetic-tf2-0.7.10/src/buffer_core.cpp
2026-01-28 14:27:17,833 - DEBUG - Control robot head: 0.0, 20.0
2026-01-28 14:27:19,047 - ERROR - [Error] get arm ctrl mode: timeout exceeded while waiting for service /humanoid_get_arm_ctrl_mode
2026-01-28 14:27:20,266 - ERROR - [Error] get arm ctrl mode: timeout exceeded while waiting for service /humanoid_get_arm_ctrl_mode
2026-01-28 14:27:20,268 - DEBUG - [Core] Change robot arm control  from None to KuavoArmCtrlMode.ExternalControl, retry: 0
2026-01-28 14:27:20,268 - DEBUG - [ROS] Change robot arm control mode: KuavoArmCtrlMode.ExternalControl
 ==== 机器人头部俯仰调节角度: 20 成功 ==== 
 ==== 切换手臂到外部控制模式成功 ==== 
⚡ Using custom MODEL_ACTION_DT: 0.100s (inference frequency: 10.0 Hz)
 ======================  GUI windows disabled ====================== 

📷 Camera Configuration (TASK_DATA_MODE: strategy):
   CAMERA_COMPONENTS: ['cam_head', 'cam_left', 'cam_right']
   Camera names: ['image', 'left_shoulder_image', 'right_shoulder_image']
   Detected 3 cameras in topic_info: ['image', 'left_shoulder_image', 'right_shoulder_image']

================================================================================
🎯 Depalletize Task Evaluation (GrootPolicy)
================================================================================
📂 Checkpoint: /home/lab/humanoid_groot/outputs/train/0124_multi_dataset_h100x4_absolute_eef_4322_2X3_groot_cross-attention_ignore_rotation/checkpoints/020000/pretrained_model
🤖 Model type: groot (using GrootPolicy)
📊 Action chunk size: 16
📦 Action dimension: Supports 16 or 18 (14 arm joints + 2 claw positions [+ 2 cmd_pose])
🖼️  Enable GUI: False
📝 Task description: 'Depalletize the box'
⏭️  Chunk selection: will execute actions from index 1 to 7 (inclusive)
🔄 Sync mode: Enabled
🚦 Max joint velocity limit: 1.00 rad/s
⚙️  Constant velocity mode: Enabled (actions will execute at constant velocity within speed limit)
⏸️  Pause before chunk: Enabled (will pause before executing each chunk for inspection)
🔒 Claw lock mechanism: threshold=50.0, count_threshold=1, locked_value=80.0
================================================================================

🚀 Starting real-time evaluation...
 =================== Loading GrootPolicy =================== 
[GROOT] Flash Attention version: 2.7.4.post1
Loading pretrained dual brain from nvidia/GR00T-N1.5-3B
Tune backbone vision tower: True
Tune backbone LLM: True
Tune action head projector: True
Tune action head DiT: True
Fetching 13 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 9425.40it/s]
`torch_dtype` is deprecated! Use `dtype` instead!
[GROOT] Copying vendor Eagle files to cache: /home/lab/humanoid_groot/src/lerobot/policies/groot/eagle2_hg_model -> /home/lab/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5
[GROOT] Assets repo: lerobot/eagle2hg-processor-groot-n1p5 
 Cache dir: /home/lab/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5
Tune backbone llm: False
Tune backbone visual: True
🔧 Using pretrained action encoder (32D) with multi-head output (16D)
🎯 Auto-configured for Absolute eef action space:
   left_arm=9D, right_arm=9D, claw=2D
⚠️  Pretrained model uses 32D, but data uses 20D. Will pad/truncate actions for compatibility.
Total number of DiT parameters:  550386688
✅ actual_action_dim=20D (based on action_space_type='Absolute eef')
✅ State encoder enabled: state features will be included in DiT input
   ✅ Cross-attention enabled (EEF space): position only, rotation independent
      This preserves 6D rotation geometric constraints (orthonormality)
🤝 Using OPTIMAL hybrid architecture:
   ✅ Shared bottom layer (coordination)
   ✅ Cross-attention (left↔right awareness)
   ✅ Separate output layers (independence)
   ✅ Coordination loss weight=0.2
📊 Multi-head action: left_arm(9D, indices 0-8) + right_arm(9D, indices 9-17) + claw(2D, indices 18-19) = 20D
   action_arm_dim=18 (left+right), actual_action_dim=20 (from action_space_type=Absolute eef)
🎯 Learnable loss weights enabled: left_arm, right_arm, claw
   Using uncertainty-based weighting from https://arxiv.org/pdf/1705.07115
Total number of SelfAttentionTransformer parameters:  201433088
Tune action head projector: True
Tune action head diffusion model: True
Warning: Could not load pretrained weights: Error(s) in loading state_dict for GR00TN15:
        size mismatch for action_head.future_tokens.weight: copying a param with shape torch.Size([32, 1536]) from checkpoint, the shape in current model is torch.Size([64, 1536]).
Continuing with randomly initialized weights.
Tune backbone llm: True
Tune backbone visual: True
Tune action head projector: True
Tune action head diffusion model: True
✅ Verified: 134/134 LLM parameters are trainable
Loading weights from local directory
📝 Using provided task description: 'Depalletize the box'

🔧 Loading preprocessor and postprocessor from checkpoint...
✅ Preprocessor and postprocessor loaded from checkpoint
✅ Found dataset_stats in checkpoint postprocessor
✅ Using dataset_stats from checkpoint: ['action', 'episode_index', 'frame_index', 'index', 'observation.images.cam_head', 'observation.images.cam_left', 'observation.images.cam_right', 'observation.state', 'task_index', 'timestamp']
✅ Partial normalization configuration found:
   - action_space_type: Absolute eef
   - action_component_indices: ['left_eef_pos', 'left_eef_rot6d', 'right_eef_pos', 'right_eef_rot6d', 'left_gripper', 'right_gripper']
   ⚠️  6D rotation components (left_eef_rot6d, right_eef_rot6d) will NOT be unnormalized
🔍 Inferred action_dim=20 from action_space_type=Absolute eef
🔍 Model configuration input_features keys: ['observation.images.cam_right', 'observation.state', 'observation.images.cam_head', 'observation.images.cam_left']
🔍 Model configuration output_features keys: ['action']
🔍 Detected action_dim: 20
🔍 Detected action_space_type: Absolute eef
[LOAD] EEF mode detected: action_space_type=Absolute eef, action_dim=20
[PinocchioFK] ✅ Pinocchio FK initialized:
[PinocchioFK]   URDF: /home/lab/kuavo-manip/lerobot_datasets/utils/biped_s60_only_arm.urdf
[PinocchioFK]   Model nq: 14
[PinocchioFK]   Arm joint indices: 0 to 13 (total: 14 joints)
[PinocchioFK]   Left EEF frame: zarm_l7_link (ID: 15)
[PinocchioFK]   Right EEF frame: zarm_r7_link (ID: 31)
[PinocchioFK]   Reference frame: base_link (ID: 1)
[LOAD] ✅ Forward kinematics initialized for EEF mode
[LOAD]   Will convert joint positions (16D) to EEF pose (20D) for state
📷 Camera configuration based on CAMERA_COMPONENTS (['cam_head', 'cam_left', 'cam_right']):
   Detected 3 cameras: ['image', 'left_shoulder_image', 'right_shoulder_image']
<bound method ObsBuffer.common_callback of <robot_envs.kuavo_depalletize_env.ObsBuffer object at 0x7f01d01261d0>>
<bound method ObsBuffer.common_callback of <robot_envs.kuavo_depalletize_env.ObsBuffer object at 0x7f01d01261d0>>
<bound method ObsBuffer.common_callback of <robot_envs.kuavo_depalletize_env.ObsBuffer object at 0x7f01d01261d0>>
[INFO] [1769581687.455272]: 🔒 Claw lock mechanism initialized: threshold=50.0, count_threshold=1, locked_value=80.0
🤖 Environment initialized for depalletize task
 ======================  Waiting for buffer ready ====================== 
Filling image: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:01<00:00, 29.94it/sAll buffers are ready!_image:  50%|███████████████████████████████████████████████                                               | 15/30 [00:01<00:01, 14.97it/s]
Filling image: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 11.98it/s]
Filling left_shoulder_image:  50%|███████████████████████████████████████████████                                               | 15/30 [00:02<00:02,  5.99it/s]
Filling right_shoulder_image:  53%|█████████████████████████████████████████████████▌                                           | 16/30 [00:02<00:02,  6.39it/s]
Filling dof_state: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 11.98it/s]
Filling dof_state_vel: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 11.98it/s]
Filling claw_state: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 11.98it/s]
 ======================  Buffer ready ====================== 

================================================================================
🔄 Starting inference session #1
📦 First inference: will load bag file for initial trajectory
================================================================================

[INFO] [1769581690.963601]: 🔄 Resetting inference state...
[INFO] [1769581690.965563]:    ✅ Policy reset
[INFO] [1769581690.967153]:    ✅ Claw lock reset
[INFO] [1769581690.968390]:    ⏳ Waiting for buffer to be ready...
Filling image:   0%|                                                                                                                     | 0/30 [00:00<?, ?it/sAll buffers are ready!_image:   0%|                                                                                                       | 0/30 [00:00<?, ?it/s]
Filling image:   0%|                                                                                                                     | 0/30 [00:00<?, ?it/s]
Filling left_shoulder_image:   0%|                                                                                                       | 0/30 [00:00<?, ?it/s]
Filling right_shoulder_image:   0%|                                                                                                      | 0/30 [00:00<?, ?it/s]
Filling dof_state:   0%|                                                                                                                 | 0/30 [00:00<?, ?it/s]
Filling dof_state_vel:   0%|                                                                                                             | 0/30 [00:00<?, ?it/s]
Filling claw_state:   0%|                                                                                                                | 0/30 [00:00<?, ?it/s]
[INFO] [1769581691.473337]:    ✅ Buffer ready
[INFO] [1769581691.475254]: ✅ Inference state reset complete

================================================================================
🎯 DEPALLETIZE TASK CONFIGURATION (GrootPolicy)
================================================================================
🤖 Control arm: True
🦾 Control claw: True
📊 ACTION_COMPONENTS: ['Left_arm', 'Right_arm', 'Left_claw', 'Right_claw']
📊 Action dimension: Will be determined by model output (expected: 28D based on config)
📦 Action chunk size: 16
🎯 Cmd_pose control: Disabled (based on ACTION_COMPONENTS)
⏭️  Chunk selection: will execute actions from index 1 to 7 (inclusive)
🔄 Sync mode: Enabled (inference -> execute chunk -> get_obs -> repeat)
🚦 Max joint velocity limit: 1.00 rad/s
⚙️  Constant velocity mode: Enabled (actions will execute at constant velocity within speed limit)
📝 Task description: 'Depalletize the box'
================================================================================

[INFO] [1769581691.478487]: Initialized ROS publishers for action visualization with chunk size: 16
[INFO] [1769581691.484181]: Initialized action visualization with chunk size: 16
2026-01-28 14:28:12,695 - ERROR - [Error] get arm ctrl mode: timeout exceeded while waiting for service /humanoid_get_arm_ctrl_mode
2026-01-28 14:28:13,916 - ERROR - [Error] get arm ctrl mode: timeout exceeded while waiting for service /humanoid_get_arm_ctrl_mode
2026-01-28 14:28:13,916 - DEBUG - [Core] Change robot arm control  from None to KuavoArmCtrlMode.ExternalControl, retry: 0
2026-01-28 14:28:13,917 - DEBUG - [ROS] Change robot arm control mode: KuavoArmCtrlMode.ExternalControl
[INFO] [1769581694.956686]: Resetting claw lock state before opening claws...
[INFO] [1769581694.958277]: ✅ Claw lock reset complete
[INFO] [1769581694.959930]: Opening claws before reset...
[INFO] [1769581695.963299]: Loading initial arm trajectory from JSON: /home/lab/humanoid_groot/scripts/utils/start_arm_traj.json
[INFO] [1769581695.966239]: Resetting arm from current position to initial position over 5.0s (50 steps)...
[INFO] [1769581700.992429]: Arm reset completed!
轨迹回放 结束, 按回车继续 ==== 轨迹回放成功 ==== 

[INFO] [1769581703.147102]: 🔄 Updating observation data after bag replay...
[INFO] [1769581703.150423]: ✅ Observation data updated with post-bag-replay robot state
[INFO] [1769581703.152444]: call set_arm_quick_mode:True
[INFO] [1769581703.179852]: Successfully enabled arm quick mode
当前机器人模式为: 5_wheel | 控制模式 set_arm_quick_mode 结束, 按回车继续 ==== 切换手臂到wbc轨迹控制模式成功 ==== 


================================================================================
🚀 Starting inference loop...
💡 Press 'q' + Enter to stop current inference and prepare for next run
💡 Press Ctrl+C to exit the program completely
================================================================================

[INFO] [1769581716.800487]: [INFERENCE] ✅ EEF mode detected: action_space_type=Absolute eef, action_dim=20
[INFO] [1769581716.802464]: [INFERENCE] ✅ Converted joint positions (16D) to EEF pose state (20D) using FK
[INFO] [1769581716.804280]: [INFERENCE]   Original joint state shape: (1, 16)
[INFO] [1769581716.806077]: [INFERENCE]   Converted EEF state shape: torch.Size([1, 20])
[INFO] [1769581716.807528]: [INFERENCE]   EEF state components: left_eef(9) + right_eef(9) + gripper(2) = 20D
`use_fast` is set to `True` but the image processor class does not have a fast version.  Falling back to the slow version.
The tokenizer you are loading from '/home/lab/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
The tokenizer you are loading from '/home/lab/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
[INFO] [1769581718.094327]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581718.096140]: [TRANSITION] First inference check: is_eef_mode=True, action_dim_from_chunk=20, action_space_type=Absolute eef
[INFO] [1769581718.097999]: 🔄 First model inference: generating smooth transition from current robot state to first action
[INFO] [1769581718.099590]:    EEF mode detected: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581718.124895]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581718.128962]:    Current arm state: [-0.13108267  0.84227829 -0.22633586 -2.27664515  1.33580762  0.40938443
  0.96440387 -0.13178328 -0.84762207  0.23244085 -2.27550282 -1.33046059
 -0.39981895  0.96897731]... (showing first 3 joints)
[INFO] [1769581718.130392]:    Target arm state: [-0.47989685  0.69367528 -0.49307774 -2.13098987  1.07250675  0.51942757
  1.03169348 -0.13502262 -0.79100083  0.26149193 -2.3784513  -1.32461113
 -0.37674808  0.97068358]... (showing first 3 joints)
[INFO] [1769581718.131920]:    Generating 20 interpolation steps over 0.20s
[INFO] [1769581718.133179]:    Generated transition chunk of size 20
[INFO] [1769581718.134252]:    Combined transition + chunk: 20 + 7 = 27 steps
[INFO] [1769581718.137244]: Executing chunk of size 141 in sync mode

================================================================================
📊 Chunk Information (Step 0)
================================================================================
Chunk size: 141 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.2739, 0.2795, 0.1108]
     Euler (roll, pitch, yaw): [-41.54°, -41.58°, -30.10°]
   Last action:
     Position (xyz): [0.4364, 0.2075, 0.1074]
     Euler (roll, pitch, yaw): [-11.90°, -53.13°, -11.72°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2045, -0.3025, 0.0987]
     Euler (roll, pitch, yaw): [54.47°, -30.65°, 36.48°]
   Last action:
     Position (xyz): [0.2031, -0.3024, 0.1032]
     Euler (roll, pitch, yaw): [52.54°, -32.30°, 34.80°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.14852337  0.83484814 -0.23967296 -2.26936239  1.32264258  0.41488658
  0.96776835 -0.13194525 -0.844791    0.2338934  -2.28065025 -1.33016811
 -0.39866541  0.96906262]
   Last action arm joints (14D): [-0.96855856  0.28590816 -0.74804348 -1.23248195  0.69185704  0.44674919
  0.93056608 -0.14563303 -0.79732694  0.26112682 -2.39450173 -1.30474801
 -0.42999996  0.98092162]

🦀 Claw:
   First action claw (2D): [0.862683   0.73298531]
   Last action claw (2D): [ 0.08947849 -0.49628615]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581721.836527]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581721.839174]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581721.867284]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581721.869744]: Executing chunk of size 43 in sync mode

================================================================================
📊 Chunk Information (Step 141)
================================================================================
Chunk size: 43 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4542, 0.2161, 0.1039]
     Euler (roll, pitch, yaw): [-3.60°, -55.09°, -4.81°]
   Last action:
     Position (xyz): [0.4472, 0.2221, 0.0310]
     Euler (roll, pitch, yaw): [-2.00°, -55.11°, -3.57°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2047, -0.3069, 0.1006]
     Euler (roll, pitch, yaw): [53.58°, -31.32°, 36.71°]
   Last action:
     Position (xyz): [0.2012, -0.3023, 0.0971]
     Euler (roll, pitch, yaw): [54.07°, -30.14°, 35.09°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.97205254  0.2845814  -0.74912863 -1.22274549  0.69003172  0.44655053
  0.92611714 -0.14444178 -0.79792568  0.25990005 -2.39322172 -1.30593788
 -0.42554228  0.98047183]
   Last action arm joints (14D): [-0.88872868  0.23485363 -0.72188738 -0.99418129  0.62741801  0.38769188
  0.69416431 -0.11674051 -0.78872272  0.25401758 -2.38970169 -1.32971989
 -0.393154    0.97658886]

🦀 Claw:
   First action claw (2D): [ 0.11121035 -0.00609159]
   Last action claw (2D): [-0.05068183 -0.30522943]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581723.675200]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581723.677670]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581723.703985]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581723.707595]: Executing chunk of size 32 in sync mode

================================================================================
📊 Chunk Information (Step 184)
================================================================================
Chunk size: 32 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4567, 0.2158, 0.0491]
     Euler (roll, pitch, yaw): [-1.94°, -55.18°, -2.97°]
   Last action:
     Position (xyz): [0.4488, 0.2191, 0.0035]
     Euler (roll, pitch, yaw): [-4.41°, -56.87°, -5.08°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2023, -0.3039, 0.1032]
     Euler (roll, pitch, yaw): [54.15°, -30.51°, 35.75°]
   Last action:
     Position (xyz): [0.2051, -0.3096, 0.1082]
     Euler (roll, pitch, yaw): [54.04°, -31.06°, 35.57°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.8982413   0.23142423 -0.72429065 -0.98815186  0.63030653  0.38924755
  0.69740418 -0.12009576 -0.79110336  0.25350361 -2.39055826 -1.33214093
 -0.3944201   0.9766618 ]
   Last action arm joints (14D): [-0.87438364  0.19368306 -0.71646043 -0.8834947   0.64778411  0.33705752
  0.55955625 -0.14922301 -0.82611374  0.23559445 -2.38678884 -1.37471942
 -0.4013969   0.96961853]

🦀 Claw:
   First action claw (2D): [-0.71133375 -0.78054667]
   Last action claw (2D): [-0.31682849 -0.77919364]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581725.852154]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581725.854102]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581725.882614]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581725.885818]: Executing chunk of size 36 in sync mode

================================================================================
📊 Chunk Information (Step 216)
================================================================================
Chunk size: 36 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4486, 0.2234, 0.0026]
     Euler (roll, pitch, yaw): [-1.19°, -54.41°, -2.35°]
   Last action:
     Position (xyz): [0.4450, 0.2233, 0.0099]
     Euler (roll, pitch, yaw): [-4.79°, -57.75°, -5.35°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.1985, -0.3050, 0.1003]
     Euler (roll, pitch, yaw): [53.27°, -31.19°, 35.39°]
   Last action:
     Position (xyz): [0.2019, -0.3114, 0.1095]
     Euler (roll, pitch, yaw): [52.56°, -31.47°, 36.20°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.87406097  0.19507278 -0.71645072 -0.88353972  0.64605008  0.34012322
  0.56731428 -0.14223661 -0.82198936  0.23571768 -2.38955104 -1.36544236
 -0.40089251  0.97120896]
   Last action arm joints (14D): [-0.86160159  0.22123235 -0.71450213 -0.94621404  0.64870077  0.35128356
  0.57763381 -0.13329955 -0.83346771  0.21908742 -2.39804417 -1.34708882
 -0.39267578  0.98956122]

🦀 Claw:
   First action claw (2D): [-0.35679936 -0.22213459]
   Last action claw (2D): [81.39179993 -0.09998679]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581726.832938]: 🔒 Left claw locked (count: 1, value: 81.24)
[INFO] [1769581727.161049]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581727.162634]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581727.188905]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581727.191854]: Executing chunk of size 32 in sync mode

================================================================================
📊 Chunk Information (Step 252)
================================================================================
Chunk size: 32 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4387, 0.2253, -0.0233]
     Euler (roll, pitch, yaw): [-3.85°, -55.81°, -4.86°]
   Last action:
     Position (xyz): [0.4427, 0.2239, 0.0341]
     Euler (roll, pitch, yaw): [-4.36°, -57.33°, -5.39°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2024, -0.3084, 0.1039]
     Euler (roll, pitch, yaw): [52.73°, -30.96°, 35.31°]
   Last action:
     Position (xyz): [0.2037, -0.3107, 0.1068]
     Euler (roll, pitch, yaw): [53.03°, -31.95°, 36.12°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.85626778  0.21854627 -0.71321914 -0.93681419  0.64719834  0.34719039
  0.57040935 -0.13244257 -0.83124933  0.22078801 -2.39714657 -1.34595359
 -0.39299132  0.98935015]
   Last action arm joints (14D): [-0.87438213  0.25322906 -0.71860333 -1.04338614  0.63770786  0.38778317
  0.67240048 -0.13429428 -0.82632048  0.22673549 -2.38768124 -1.34211203
 -0.38902262  0.97552369]

🦀 Claw:
   First action claw (2D): [81.06374359 -0.11141896]
   Last action claw (2D): [80.80083466 -0.29680133]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581729.446631]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581729.448267]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581729.474142]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581729.480206]: Executing chunk of size 41 in sync mode

================================================================================
📊 Chunk Information (Step 284)
================================================================================
Chunk size: 41 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4418, 0.2109, 0.0611]
     Euler (roll, pitch, yaw): [-1.90°, -57.13°, -2.94°]
   Last action:
     Position (xyz): [0.4379, 0.2186, 0.1170]
     Euler (roll, pitch, yaw): [-5.07°, -55.08°, -5.89°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2006, -0.3062, 0.1012]
     Euler (roll, pitch, yaw): [52.19°, -31.20°, 36.02°]
   Last action:
     Position (xyz): [0.2018, -0.3058, 0.1044]
     Euler (roll, pitch, yaw): [53.68°, -30.78°, 36.08°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.87760518  0.25243672 -0.71946108 -1.04994677  0.63427824  0.3949064
  0.68220021 -0.13237882 -0.82422848  0.2277231  -2.38836232 -1.33817722
 -0.38852285  0.9775925 ]
   Last action arm joints (14D): [-0.99142602  0.31633977 -0.75748054 -1.23955131  0.6601093   0.48859091
  0.94517279 -0.13302192 -0.81024883  0.24177351 -2.3964209  -1.34001126
 -0.39452899  0.98171583]

🦀 Claw:
   First action claw (2D): [81.17565918 -0.1116097 ]
   Last action claw (2D): [8.08026657e+01 4.60892916e-02]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581731.376398]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581731.378019]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581731.405206]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581731.408562]: Executing chunk of size 31 in sync mode

================================================================================
📊 Chunk Information (Step 325)
================================================================================
Chunk size: 31 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4332, 0.2153, 0.1179]
     Euler (roll, pitch, yaw): [-0.78°, -55.85°, -1.95°]
   Last action:
     Position (xyz): [0.4324, 0.2641, 0.1219]
     Euler (roll, pitch, yaw): [-0.98°, -55.07°, -2.22°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2051, -0.3059, 0.1068]
     Euler (roll, pitch, yaw): [52.79°, -32.11°, 36.60°]
   Last action:
     Position (xyz): [0.2031, -0.3086, 0.1082]
     Euler (roll, pitch, yaw): [52.83°, -32.21°, 36.50°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.98935249  0.31596894 -0.75673158 -1.24459088  0.65094635  0.49631042
  0.9496511  -0.1367177  -0.81064935  0.24345293 -2.39488049 -1.33596499
 -0.3952167   0.9821596 ]
   Last action arm joints (14D): [-0.98467789  0.44578232 -0.75695471 -1.29609267  0.71936679  0.47423369
  0.94956444 -0.14308705 -0.82317577  0.2348966  -2.39469073 -1.32860417
 -0.39796709  0.98099803]

🦀 Claw:
   First action claw (2D): [81.03208923 -0.08425117]
   Last action claw (2D): [80.93312073  0.10744929]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581733.168612]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581733.169998]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581733.198251]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581733.200223]: Executing chunk of size 36 in sync mode

================================================================================
📊 Chunk Information (Step 356)
================================================================================
Chunk size: 36 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4274, 0.2244, 0.1263]
     Euler (roll, pitch, yaw): [-2.10°, -55.34°, -3.26°]
   Last action:
     Position (xyz): [0.4314, 0.2977, 0.1223]
     Euler (roll, pitch, yaw): [-3.29°, -52.96°, -4.89°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2074, -0.3005, 0.1105]
     Euler (roll, pitch, yaw): [51.99°, -33.14°, 36.27°]
   Last action:
     Position (xyz): [0.2067, -0.3067, 0.1063]
     Euler (roll, pitch, yaw): [54.01°, -32.64°, 36.37°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.98420557  0.43675269 -0.75673763 -1.29898082  0.7104356   0.48027412
  0.95517403 -0.14824527 -0.82105522  0.23973652 -2.39428916 -1.32455133
 -0.40213469  0.98233884]
   Last action arm joints (14D): [-0.98507023  0.52785875 -0.75397479 -1.29107966  0.80469626  0.39156483
  0.94489567 -0.15809441 -0.81390965  0.25228261 -2.37999148 -1.33610682
 -0.40099378  0.95779584]

🦀 Claw:
   First action claw (2D): [80.87445831 -0.36925077]
   Last action claw (2D): [80.63826752  0.17139316]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581735.261441]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581735.263779]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581735.292529]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581735.295815]: Executing chunk of size 41 in sync mode

================================================================================
📊 Chunk Information (Step 392)
================================================================================
Chunk size: 41 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4250, 0.2815, 0.1237]
     Euler (roll, pitch, yaw): [-3.54°, -55.66°, -2.99°]
   Last action:
     Position (xyz): [0.4250, 0.3061, 0.1089]
     Euler (roll, pitch, yaw): [-5.48°, -53.56°, -5.51°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2042, -0.3003, 0.1096]
     Euler (roll, pitch, yaw): [52.90°, -32.29°, 35.87°]
   Last action:
     Position (xyz): [0.2213, -0.2903, 0.1139]
     Euler (roll, pitch, yaw): [49.03°, -36.08°, 35.09°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.98284024  0.52489522 -0.75314814 -1.29787678  0.80229649  0.40138899
  0.94258626 -0.16071048 -0.81238406  0.25513255 -2.38280409 -1.33293474
 -0.40628545  0.96152155]
   Last action arm joints (14D): [-0.93586945  0.54893522 -0.73009167 -1.31376663  0.83197386  0.38085382
  0.90008513 -0.28932972 -0.76786343  0.3597903  -2.35012083 -1.20094442
 -0.49831099  1.02110029]

🦀 Claw:
   First action claw (2D): [ 8.11059952e+01 -4.10079956e-02]
   Last action claw (2D): [80.99004364 -0.22182465]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581737.357921]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581737.359525]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581737.387855]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581737.390461]: Executing chunk of size 72 in sync mode

================================================================================
📊 Chunk Information (Step 433)
================================================================================
Chunk size: 72 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4280, 0.3220, 0.1260]
     Euler (roll, pitch, yaw): [-3.61°, -54.67°, -6.28°]
   Last action:
     Position (xyz): [0.4259, 0.3188, 0.1086]
     Euler (roll, pitch, yaw): [-7.52°, -54.70°, -7.92°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2067, -0.2951, 0.1056]
     Euler (roll, pitch, yaw): [51.63°, -33.35°, 35.95°]
   Last action:
     Position (xyz): [0.2842, -0.2146, 0.1312]
     Euler (roll, pitch, yaw): [43.44°, -51.68°, 32.45°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.94036284  0.5527521  -0.73184313 -1.31230435  0.83270909  0.37688381
  0.90066175 -0.28022173 -0.76875928  0.35451289 -2.35379409 -1.20530874
 -0.49366531  1.01920528]
   Last action arm joints (14D): [-0.93975076  0.57459938 -0.7292985  -1.29552059  0.86349588  0.32862607
  0.85931584 -0.62859409 -0.47320586  0.62670135 -2.09643208 -0.7683913
 -0.67925583  1.04719755]

🦀 Claw:
   First action claw (2D): [80.73836517 -0.32109618]
   Last action claw (2D): [80.34179688 -0.20061731]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581739.752747]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581739.754463]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581739.786097]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581739.790891]: Executing chunk of size 94 in sync mode

================================================================================
📊 Chunk Information (Step 505)
================================================================================
Chunk size: 94 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4298, 0.2923, 0.1194]
     Euler (roll, pitch, yaw): [-10.12°, -52.84°, -10.16°]
   Last action:
     Position (xyz): [0.4307, 0.2925, 0.1173]
     Euler (roll, pitch, yaw): [-7.70°, -54.15°, -9.76°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.3221, -0.1871, 0.1425]
     Euler (roll, pitch, yaw): [35.51°, -57.19°, 26.58°]
   Last action:
     Position (xyz): [0.4159, -0.1058, 0.1563]
     Euler (roll, pitch, yaw): [-3.57°, -62.03°, -3.70°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.94143884  0.57184555 -0.73029609 -1.29589727  0.8630597   0.33029593
  0.86225762 -0.63515922 -0.46755947  0.62953651 -2.0867341  -0.76140633
 -0.68019962  1.04719755]
   Last action arm joints (14D): [-0.97159281  0.51606121 -0.748601   -1.29301371  0.8120948   0.35256304
  0.90712832 -1.02619019 -0.01551148  0.72614382 -1.23929913 -0.27195895
 -0.69437052  1.04719755]

🦀 Claw:
   First action claw (2D): [80.48464966 -0.31483173]
   Last action claw (2D): [80.3360672  -0.36439896]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581742.339503]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581742.341194]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581742.373671]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581742.378127]: Executing chunk of size 28 in sync mode

================================================================================
📊 Chunk Information (Step 599)
================================================================================
Chunk size: 28 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4325, 0.2931, 0.1169]
     Euler (roll, pitch, yaw): [-7.53°, -53.44°, -8.82°]
   Last action:
     Position (xyz): [0.4314, 0.2895, 0.1205]
     Euler (roll, pitch, yaw): [-7.24°, -53.46°, -8.74°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.4088, -0.1071, 0.1526]
     Euler (roll, pitch, yaw): [-2.04°, -63.23°, -2.52°]
   Last action:
     Position (xyz): [0.4212, -0.1049, 0.0812]
     Euler (roll, pitch, yaw): [-2.99°, -54.78°, -4.24°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.97258733  0.51541767 -0.749081   -1.29044383  0.81445286  0.35361115
  0.90794207 -1.02035038 -0.01753061  0.72533612 -1.24900038 -0.26976842
 -0.69477337  1.04719755]
   Last action arm joints (14D): [-0.98064622  0.51000164 -0.75297204 -1.29381961  0.81508965  0.37069498
  0.92624302 -0.89786008  0.0236157   0.69617046 -1.11242511 -0.32156942
 -0.66829074  0.95915804]

🦀 Claw:
   First action claw (2D): [80.34689331  0.13463795]
   Last action claw (2D): [80.30001068 -0.19384027]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581743.552680]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581743.554198]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581743.589988]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581743.592382]: Executing chunk of size 49 in sync mode

================================================================================
📊 Chunk Information (Step 627)
================================================================================
Chunk size: 49 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4377, 0.2801, 0.1124]
     Euler (roll, pitch, yaw): [-7.21°, -54.51°, -9.35°]
   Last action:
     Position (xyz): [0.4417, 0.2852, 0.1113]
     Euler (roll, pitch, yaw): [-6.81°, -55.13°, -7.61°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.4214, -0.1063, 0.1095]
     Euler (roll, pitch, yaw): [-5.45°, -57.09°, -6.45°]
   Last action:
     Position (xyz): [0.4153, -0.1131, 0.0233]
     Euler (roll, pitch, yaw): [-7.31°, -55.90°, -8.81°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.9808871   0.50408943 -0.75332102 -1.28661942  0.81095502  0.36860152
  0.92079973 -0.90466441  0.02147147  0.69753478 -1.11882746 -0.3166529
 -0.67215755  0.96787265]
   Last action arm joints (14D): [-0.99347174  0.46943959 -0.76016391 -1.21128479  0.81137175  0.35836373
  0.85694644 -0.80213791  0.02403291  0.68666377 -1.03692787 -0.35432804
 -0.6764676   0.76242292]

🦀 Claw:
   First action claw (2D): [8.09881592e+01 4.73380089e-02]
   Last action claw (2D): [8.09304962e+01 6.66528940e-02]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581745.475965]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581745.477663]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581745.518213]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581745.521967]: Executing chunk of size 42 in sync mode

================================================================================
📊 Chunk Information (Step 676)
================================================================================
Chunk size: 42 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4225, 0.2774, 0.1038]
     Euler (roll, pitch, yaw): [-5.43°, -53.82°, -7.48°]
   Last action:
     Position (xyz): [0.4242, 0.2735, 0.1128]
     Euler (roll, pitch, yaw): [-5.34°, -53.24°, -5.34°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.4240, -0.1134, 0.0210]
     Euler (roll, pitch, yaw): [-5.42°, -54.37°, -6.70°]
   Last action:
     Position (xyz): [0.4201, -0.1197, 0.0250]
     Euler (roll, pitch, yaw): [-2.29°, -54.68°, -4.23°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.98750528  0.47066715 -0.75759852 -1.22100324  0.80609398  0.36202676
  0.86253739 -0.80452465  0.02501989  0.68671914 -1.03126291 -0.35755638
 -0.67366221  0.76083914]
   Last action arm joints (14D): [-0.94049043  0.47910758 -0.73735853 -1.34054072  0.76684413  0.44861003
  0.95102241 -0.81713768  0.00694548  0.68994409 -1.02715058 -0.41338153
 -0.62555172  0.76228311]

🦀 Claw:
   First action claw (2D): [ 8.11255875e+01 -1.24394894e-02]
   Last action claw (2D): [80.74960327 80.07967377]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581747.581349]: 🔒 Right claw locked (count: 1, value: 80.24)
[INFO] [1769581747.972516]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581747.975226]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581748.013980]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581748.016179]: Executing chunk of size 34 in sync mode

================================================================================
📊 Chunk Information (Step 718)
================================================================================
Chunk size: 34 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4214, 0.2668, 0.0996]
     Euler (roll, pitch, yaw): [-5.04°, -52.13°, -5.73°]
   Last action:
     Position (xyz): [0.4201, 0.2655, 0.1135]
     Euler (roll, pitch, yaw): [-3.51°, -51.85°, -4.08°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.4181, -0.1200, 0.0052]
     Euler (roll, pitch, yaw): [-2.89°, -54.78°, -4.16°]
   Last action:
     Position (xyz): [0.4202, -0.1269, 0.0558]
     Euler (roll, pitch, yaw): [-1.53°, -53.16°, -2.64°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.93522037  0.47603747 -0.73542164 -1.34083373  0.76145605  0.44787358
  0.95212017 -0.81378536  0.00838986  0.68950092 -1.02031932 -0.41432232
 -0.62302901  0.75289913]
   Last action arm joints (14D): [-0.92915905  0.46714003 -0.73338648 -1.3691866   0.72445805  0.47869524
  1.00544356 -0.84847552 -0.0345043   0.6983554  -1.12420033 -0.38632687
 -0.62934398  0.91028097]

🦀 Claw:
   First action claw (2D): [80.57292175 79.92811584]
   Last action claw (2D): [80.71066284 79.87100983]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581749.658373]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581749.660074]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581749.694018]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581749.697589]: Executing chunk of size 33 in sync mode

================================================================================
📊 Chunk Information (Step 752)
================================================================================
Chunk size: 33 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4130, 0.2562, 0.1166]
     Euler (roll, pitch, yaw): [-2.44°, -52.79°, -4.07°]
   Last action:
     Position (xyz): [0.4084, 0.2539, 0.1374]
     Euler (roll, pitch, yaw): [-2.50°, -53.69°, -3.65°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.4142, -0.1324, 0.0551]
     Euler (roll, pitch, yaw): [-0.38°, -51.67°, -1.98°]
   Last action:
     Position (xyz): [0.4133, -0.1428, 0.1353]
     Euler (roll, pitch, yaw): [-3.06°, -56.35°, -3.27°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.92683127  0.46566138 -0.73254073 -1.37723552  0.71573262  0.48301805
  1.01054689 -0.84453405 -0.03858321  0.69825155 -1.13316415 -0.38701341
 -0.62992928  0.91715149]
   Last action arm joints (14D): [-0.95259813  0.47544816 -0.74259646 -1.46656098  0.68015169  0.54545453
  1.04719755 -0.96570129 -0.13181578  0.73291901 -1.32240674 -0.34812192
 -0.68074025  1.04719755]

🦀 Claw:
   First action claw (2D): [80.74406433 79.9358139 ]
   Last action claw (2D): [80.66560364 79.94437408]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581751.796958]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581751.798510]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581751.831421]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581751.839209]: Executing chunk of size 31 in sync mode

================================================================================
📊 Chunk Information (Step 785)
================================================================================
Chunk size: 31 actions
Action dimension: 16

🎯 Action mode: Absolute eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4131, 0.2493, 0.1325]
     Euler (roll, pitch, yaw): [0.23°, -53.02°, -0.33°]
   Last action:
     Position (xyz): [0.4137, 0.2473, 0.1547]
     Euler (roll, pitch, yaw): [-0.29°, -56.23°, -0.64°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.4123, -0.1482, 0.1037]
     Euler (roll, pitch, yaw): [-1.53°, -54.36°, -1.27°]
   Last action:
     Position (xyz): [0.4100, -0.1526, 0.1410]
     Euler (roll, pitch, yaw): [1.35°, -60.48°, 1.88°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.95267685  0.47223613 -0.7427638  -1.46221192  0.67677419  0.54827819
  1.04719755 -0.95693875 -0.13231825  0.73094415 -1.32028889 -0.35047232
 -0.67595217  1.04719755]
   Last action arm joints (14D): [-1.01184362  0.45951903 -0.7682854  -1.44468205  0.67624552  0.58948488
  1.04719755 -0.96944317 -0.17281499  0.7389064  -1.36804532 -0.39870769
 -0.64634733  1.04719755]

🦀 Claw:
   First action claw (2D): [80.81632996 79.66710663]
   Last action claw (2D): [80.62883759 80.001091  ]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: q

[User] Stopping inference by user request

================================================================================
✅ Inference session #1 stopped by user (q pressed)
[INFO] [1769581753.993677]: Resetting policy state after inference stop...
[INFO] [1769581753.995255]:    ✅ Policy reset complete
[INFO] [1769581753.996842]: Resetting arm position using JSON file...
[INFO] [1769581753.998107]: Resetting claw lock state before opening claws...
[INFO] [1769581753.999268]: 🔓 Resetting claw lock state (was locked: left, right)
[INFO] [1769581754.000378]: ✅ Claw lock reset complete
[INFO] [1769581754.001523]: Opening claws before reset...
[INFO] [1769581755.004928]: Loading initial arm trajectory from JSON: /home/lab/humanoid_groot/scripts/utils/initial_arm_traj.json
[INFO] [1769581755.008321]: Resetting arm from current position to initial position over 5.0s (50 steps)...
[INFO] [1769581760.035428]: Arm reset completed!
💡 Ready for next inference session. Press Enter to start, or Ctrl+C to exit.
================================================================================

Press Enter to start next inference, or 'q'+Enter to exit: q

👋 Exiting program. Goodbye!