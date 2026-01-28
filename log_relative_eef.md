(lerobot_groot_base) lab@lab [~/humanoid_groot] git:(KangKK/dev/delta_eef_pose_action) ✗ ➜  python scripts/eval_depalletize_camera_model_reload_limit_vel_select_eef.py --eval --ckpt-path /home/lab/humanoid_groot/outputs/train/0126_multi_dataset_h100x4_relative_eef_action_4322/checkpoints/020000/pretrained_model --model-type groot --action_chunk_size 16 --task-description "Depalletize the box" --model-action-dt 0.1 --sync-mode --max-joint-velocity 1.0 --chunk-start 1 --chunk-end 7 --constant-velocity --action-stride 1 --pause-before-chunk

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
2026-01-28 14:30:27,912 - INFO - [WheelArmROSControl] ROS接口初始化成功
2026-01-28 14:30:27,913 - INFO - [WheelArmROSControl] 轮臂ROS控制模块初始化完成
2026-01-28 14:30:27,916 - INFO - Using standard mode, subscribing to: /sensors_data_raw
2026-01-28 14:30:29,161 - ERROR - Service call failed: timeout exceeded while waiting for service /humanoid_get_current_gait_name
2026-01-28 14:30:29,161 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:30:29,262 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:30:29,363 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:30:29,463 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:30:29,564 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:30:29,664 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:30:29,765 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:30:29,866 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:30:29,967 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:30:30,068 - DEBUG - Waiting for first MPC observation data...
2026-01-28 14:30:30,169 - WARNING - Timeout waiting for MPC observation data
2026-01-28 14:30:31,378 - ERROR - [Error] get arm ctrl mode: timeout exceeded while waiting for service /humanoid_get_arm_ctrl_mode
2026-01-28 14:30:33,506 - ERROR - Failed to connect to service /get_mm_ctrl_frame: timeout exceeded while waiting for service /get_mm_ctrl_frame
2026-01-28 14:30:35,691 - ERROR - Failed to connect to service /get_mm_wbc_arm_trajectory_control: timeout exceeded while waiting for service /get_mm_wbc_arm_trajectory_control
Warning: TF_OLD_DATA ignoring data from the past (Possible reasons are listed at http://wiki.ros.org/tf/Errors%20explained) for frame base_link (parent dummy_link) at time 1769579082.022296 according to authority /nodelet_manager
         at line 277 in /tmp/binarydeb/ros-noetic-tf2-0.7.10/src/buffer_core.cpp
2026-01-28 14:30:37,742 - DEBUG - Control robot head: 0.0, 20.0
2026-01-28 14:30:38,957 - ERROR - [Error] get arm ctrl mode: timeout exceeded while waiting for service /humanoid_get_arm_ctrl_mode
2026-01-28 14:30:40,170 - ERROR - [Error] get arm ctrl mode: timeout exceeded while waiting for service /humanoid_get_arm_ctrl_mode
2026-01-28 14:30:40,170 - DEBUG - [Core] Change robot arm control  from None to KuavoArmCtrlMode.ExternalControl, retry: 0
2026-01-28 14:30:40,170 - DEBUG - [ROS] Change robot arm control mode: KuavoArmCtrlMode.ExternalControl
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
📂 Checkpoint: /home/lab/humanoid_groot/outputs/train/0126_multi_dataset_h100x4_relative_eef_action_4322/checkpoints/020000/pretrained_model
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
Fetching 13 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 2292.26it/s]
`torch_dtype` is deprecated! Use `dtype` instead!
[GROOT] Copying vendor Eagle files to cache: /home/lab/humanoid_groot/src/lerobot/policies/groot/eagle2_hg_model -> /home/lab/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5
[GROOT] Assets repo: lerobot/eagle2hg-processor-groot-n1p5 
 Cache dir: /home/lab/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5
Tune backbone llm: False
Tune backbone visual: True
🔧 Using pretrained action encoder (32D) with multi-head output (16D)
🎯 Auto-configured for Delta eef action space:
   left_arm=9D, right_arm=9D, claw=2D
⚠️  Pretrained model uses 32D, but data uses 20D. Will pad/truncate actions for compatibility.
Total number of DiT parameters:  550386688
✅ actual_action_dim=20D (based on action_space_type='Delta eef')
✅ State encoder enabled: state features will be included in DiT input
   ✅ Cross-attention enabled (EEF space): position only, rotation independent
      This preserves 6D rotation geometric constraints (orthonormality)
🤝 Using OPTIMAL hybrid architecture:
   ✅ Shared bottom layer (coordination)
   ✅ Cross-attention (left↔right awareness)
   ✅ Separate output layers (independence)
   ✅ Coordination loss weight=0.2
📊 Multi-head action: left_arm(9D, indices 0-8) + right_arm(9D, indices 9-17) + claw(2D, indices 18-19) = 20D
   action_arm_dim=18 (left+right), actual_action_dim=20 (from action_space_type=Delta eef)
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
   - action_space_type: Delta eef
   - action_component_indices: ['left_eef_pos', 'left_eef_rot6d', 'right_eef_pos', 'right_eef_rot6d', 'left_gripper', 'right_gripper']
   ⚠️  6D rotation components (left_eef_rot6d, right_eef_rot6d) will NOT be unnormalized
   🔄 Relative action mode detected: Will convert relative actions to absolute poses during inference
🔍 Inferred action_dim=20 from action_space_type=Delta eef
🔍 Model configuration input_features keys: ['observation.images.cam_right', 'observation.images.cam_head', 'observation.images.cam_left', 'observation.state']
🔍 Model configuration output_features keys: ['action']
🔍 Detected action_dim: 20
🔍 Detected action_space_type: Delta eef
[LOAD] EEF mode detected: action_space_type=Delta eef, action_dim=20
[PinocchioFK] ✅ Pinocchio FK initialized:
[PinocchioFK]   URDF: /home/lab/kuavo-manip/lerobot_datasets/utils/biped_s60_only_arm.urdf
[PinocchioFK]   Model nq: 14
[PinocchioFK]   Arm joint indices: 0 to 13 (total: 14 joints)
[PinocchioFK]   Left EEF frame: zarm_l7_link (ID: 15)
[PinocchioFK]   Right EEF frame: zarm_r7_link (ID: 31)
[PinocchioFK]   Reference frame: base_link (ID: 1)
[LOAD] ✅ Forward kinematics initialized for EEF mode
[LOAD]   Will convert joint positions (16D) to EEF pose (20D) for state
[LOAD]   Will also use FK to compute reference pose for relative action conversion
📷 Camera configuration based on CAMERA_COMPONENTS (['cam_head', 'cam_left', 'cam_right']):
   Detected 3 cameras: ['image', 'left_shoulder_image', 'right_shoulder_image']
<bound method ObsBuffer.common_callback of <robot_envs.kuavo_depalletize_env.ObsBuffer object at 0x7f379412dde0>>
<bound method ObsBuffer.common_callback of <robot_envs.kuavo_depalletize_env.ObsBuffer object at 0x7f379412dde0>>
<bound method ObsBuffer.common_callback of <robot_envs.kuavo_depalletize_env.ObsBuffer object at 0x7f379412dde0>>
[INFO] [1769581888.215767]: 🔒 Claw lock mechanism initialized: threshold=50.0, count_threshold=1, locked_value=80.0
🤖 Environment initialized for depalletize task
 ======================  Waiting for buffer ready ====================== 
Filling image: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:01<00:00, 29.94it/sAll buffers are ready!_image:  47%|███████████████████████████████████████████▊                                                  | 14/30 [00:01<00:01, 13.97it/s]
Filling image: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 11.98it/s]
Filling left_shoulder_image:  47%|███████████████████████████████████████████▊                                                  | 14/30 [00:02<00:02,  5.59it/s]
Filling right_shoulder_image:  47%|███████████████████████████████████████████▍                                                 | 14/30 [00:02<00:02,  5.59it/s]
Filling dof_state: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 11.98it/s]
Filling dof_state_vel: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 11.98it/s]
Filling claw_state: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 11.98it/s]
 ======================  Buffer ready ====================== 

================================================================================
🔄 Starting inference session #1
📦 First inference: will load bag file for initial trajectory
================================================================================

[INFO] [1769581891.723097]: 🔄 Resetting inference state...
[INFO] [1769581891.724251]:    ✅ Policy reset
[INFO] [1769581891.725191]:    ✅ Claw lock reset
[INFO] [1769581891.725994]:    ⏳ Waiting for buffer to be ready...
Filling image:   0%|                                                                                                                     | 0/30 [00:00<?, ?it/sAll buffers are ready!_image:   0%|                                                                                                       | 0/30 [00:00<?, ?it/s]
Filling image:   0%|                                                                                                                     | 0/30 [00:00<?, ?it/s]
Filling left_shoulder_image:   0%|                                                                                                       | 0/30 [00:00<?, ?it/s]
Filling right_shoulder_image:   0%|                                                                                                      | 0/30 [00:00<?, ?it/s]
Filling dof_state:   0%|                                                                                                                 | 0/30 [00:00<?, ?it/s]
Filling dof_state_vel:   0%|                                                                                                             | 0/30 [00:00<?, ?it/s]
Filling claw_state:   0%|                                                                                                                | 0/30 [00:00<?, ?it/s]
[INFO] [1769581892.227797]:    ✅ Buffer ready
[INFO] [1769581892.229008]: ✅ Inference state reset complete

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

[INFO] [1769581892.230766]: Initialized ROS publishers for action visualization with chunk size: 16
[INFO] [1769581892.232781]: Initialized action visualization with chunk size: 16
2026-01-28 14:31:33,439 - ERROR - [Error] get arm ctrl mode: timeout exceeded while waiting for service /humanoid_get_arm_ctrl_mode
2026-01-28 14:31:34,646 - ERROR - [Error] get arm ctrl mode: timeout exceeded while waiting for service /humanoid_get_arm_ctrl_mode
2026-01-28 14:31:34,646 - DEBUG - [Core] Change robot arm control  from None to KuavoArmCtrlMode.ExternalControl, retry: 0
2026-01-28 14:31:34,646 - DEBUG - [ROS] Change robot arm control mode: KuavoArmCtrlMode.ExternalControl
[INFO] [1769581895.655499]: Resetting claw lock state before opening claws...
[INFO] [1769581895.657045]: ✅ Claw lock reset complete
[INFO] [1769581895.657960]: Opening claws before reset...
[INFO] [1769581896.661842]: Loading initial arm trajectory from JSON: /home/lab/humanoid_groot/scripts/utils/start_arm_traj.json
[INFO] [1769581896.664292]: Resetting arm from current position to initial position over 5.0s (50 steps)...
[INFO] [1769581901.689268]: Arm reset completed!
轨迹回放 结束, 按回车继续 ==== 轨迹回放成功 ==== 

[INFO] [1769581908.002786]: 🔄 Updating observation data after bag replay...
[INFO] [1769581908.005443]: ✅ Observation data updated with post-bag-replay robot state
[INFO] [1769581908.006572]: call set_arm_quick_mode:True
[INFO] [1769581908.016118]: Successfully enabled arm quick mode
当前机器人模式为: 5_wheel | 控制模式 set_arm_quick_mode 结束, 按回车继续 ==== 切换手臂到wbc轨迹控制模式成功 ==== 


================================================================================
🚀 Starting inference loop...
💡 Press 'q' + Enter to stop current inference and prepare for next run
💡 Press Ctrl+C to exit the program completely
================================================================================

[INFO] [1769581910.593800]: [INFERENCE] ✅ EEF mode detected: action_space_type=Delta eef, action_dim=20
[INFO] [1769581910.595442]: [INFERENCE] ✅ Converted joint positions (16D) to EEF pose state (20D) using FK
[INFO] [1769581910.596988]: [INFERENCE]   Original joint state shape: (1, 16)
[INFO] [1769581910.598003]: [INFERENCE]   Converted EEF state shape: torch.Size([1, 20])
[INFO] [1769581910.598811]: [INFERENCE]   EEF state components: left_eef(9) + right_eef(9) + gripper(2) = 20D
`use_fast` is set to `True` but the image processor class does not have a fast version.  Falling back to the slow version.
The tokenizer you are loading from '/home/lab/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
The tokenizer you are loading from '/home/lab/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
[INFO] [1769581911.915541]: [INFERENCE] ✅ Computed reference pose from current joint state using FK (relative action mode)
[INFO] [1769581911.924485]: [INFERENCE] Converted relative actions to absolute poses (relative action mode)
[INFO] [1769581911.925701]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581911.926572]: [TRANSITION] First inference check: is_eef_mode=True, action_dim_from_chunk=20, action_space_type=Delta eef
[INFO] [1769581911.927546]: 🔄 First model inference: generating smooth transition from current robot state to first action
[INFO] [1769581911.928451]:    EEF mode detected: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581911.953995]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581911.957189]:    Current arm state: [-0.13049784  0.84113762 -0.22824219 -2.27664633  1.33580929  0.40935523
  0.96440191 -0.13180246 -0.84609912  0.23319933 -2.27550177 -1.33046738
 -0.40019962  0.96898244]... (showing first 3 joints)
[INFO] [1769581911.958416]:    Target arm state: [-0.31092182  0.78450617 -0.35841431 -2.15157555  1.2400661   0.45405509
  0.98996559 -0.18471522 -0.82276441  0.25908575 -2.22518775 -1.31610303
 -0.37418516  0.99143343]... (showing first 3 joints)
[INFO] [1769581911.959310]:    Generating 20 interpolation steps over 0.20s
[INFO] [1769581911.960154]:    Generated transition chunk of size 20
[INFO] [1769581911.961050]:    Combined transition + chunk: 20 + 7 = 27 steps
[INFO] [1769581911.963118]: Executing chunk of size 131 in sync mode

================================================================================
📊 Chunk Information (Step 0)
================================================================================
Chunk size: 131 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.2577, 0.3153, 0.0926]
     Euler (roll, pitch, yaw): [-42.44°, -37.00°, -25.26°]
   Last action:
     Position (xyz): [0.4026, 0.2505, 0.0982]
     Euler (roll, pitch, yaw): [-11.76°, -49.56°, -5.22°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2350, -0.3259, 0.0905]
     Euler (roll, pitch, yaw): [46.37°, -33.33°, 28.60°]
   Last action:
     Position (xyz): [0.2329, -0.3318, 0.0870]
     Euler (roll, pitch, yaw): [45.88°, -33.66°, 27.43°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.13951903  0.83830605 -0.23475079 -2.27039279  1.33102213  0.41159022
  0.9656801  -0.13444809 -0.84493239  0.23449365 -2.27298607 -1.32974916
 -0.3988989   0.97010499]
   Last action arm joints (14D): [-0.84573858  0.44611682 -0.70292373 -1.46412623  0.79401085  0.55301816
  0.99567917 -0.1521738  -0.83124907  0.23236241 -2.22014019 -1.32377811
 -0.36450696  0.9822178 ]

🦀 Claw:
   First action claw (2D): [0.94837159 0.70372187]
   Last action claw (2D): [-0.66959858 -0.4750669 ]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581915.793682]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581915.794969]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581915.823671]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581915.826293]: Executing chunk of size 50 in sync mode

================================================================================
📊 Chunk Information (Step 131)
================================================================================
Chunk size: 50 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4026, 0.2658, 0.1222]
     Euler (roll, pitch, yaw): [-10.78°, -53.31°, -3.96°]
   Last action:
     Position (xyz): [0.4275, 0.2604, 0.0755]
     Euler (roll, pitch, yaw): [0.38°, -55.42°, 4.58°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2416, -0.3306, 0.0839]
     Euler (roll, pitch, yaw): [46.67°, -34.53°, 29.18°]
   Last action:
     Position (xyz): [0.2334, -0.3353, 0.0940]
     Euler (roll, pitch, yaw): [47.06°, -33.61°, 29.44°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.85475165  0.45582061 -0.70554403 -1.4689694   0.80150789  0.55719631
  0.9918437  -0.15780794 -0.82943515  0.23730002 -2.21345825 -1.32108008
 -0.35868807  0.97887115]
   Last action arm joints (14D): [-0.87816469  0.40505466 -0.71743152 -1.26372415  0.73218001  0.5323018
  0.82463796 -0.16343066 -0.85060221  0.22047425 -2.22400238 -1.35024291
 -0.34205989  0.9805082 ]

🦀 Claw:
   First action claw (2D): [-0.02958775 -0.68395138]
   Last action claw (2D): [-0.13970137 -0.71353316]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581917.021947]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581917.023331]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581917.052871]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581917.054747]: Executing chunk of size 47 in sync mode

================================================================================
📊 Chunk Information (Step 181)
================================================================================
Chunk size: 47 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4326, 0.2659, 0.1014]
     Euler (roll, pitch, yaw): [1.59°, -58.54°, 6.62°]
   Last action:
     Position (xyz): [0.4395, 0.2693, 0.0232]
     Euler (roll, pitch, yaw): [3.63°, -58.15°, 8.33°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2412, -0.3334, 0.1059]
     Euler (roll, pitch, yaw): [46.18°, -34.63°, 30.96°]
   Last action:
     Position (xyz): [0.2440, -0.3335, 0.1022]
     Euler (roll, pitch, yaw): [45.86°, -35.02°, 31.00°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.88629156  0.40807213 -0.72032855 -1.26457144  0.73707869  0.53560027
  0.82289015 -0.17271685 -0.85197764  0.22488137 -2.22264197 -1.34875948
 -0.3437943   0.98244826]
   Last action arm joints (14D): [-0.84972141  0.34896798 -0.71059026 -1.04797288  0.76360479  0.44805569
  0.58886336 -0.23915347 -0.8541104   0.263099   -2.19559418 -1.32154946
 -0.34258435  0.9909255 ]

🦀 Claw:
   First action claw (2D): [-0.03742576 -0.95071793]
   Last action claw (2D): [-0.15007257 -1.01718903]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581918.524892]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581918.526117]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581918.554711]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581918.557512]: Executing chunk of size 76 in sync mode

================================================================================
📊 Chunk Information (Step 228)
================================================================================
Chunk size: 76 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4336, 0.2818, 0.0482]
     Euler (roll, pitch, yaw): [4.42°, -60.73°, 9.41°]
   Last action:
     Position (xyz): [0.4381, 0.2861, -0.0268]
     Euler (roll, pitch, yaw): [4.97°, -60.29°, 10.34°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2435, -0.3359, 0.1091]
     Euler (roll, pitch, yaw): [44.25°, -35.80°, 31.02°]
   Last action:
     Position (xyz): [0.2422, -0.3450, 0.1152]
     Euler (roll, pitch, yaw): [43.74°, -35.33°, 31.41°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.85037067  0.35450675 -0.71050421 -1.05719198  0.76508015  0.45092017
  0.5915379  -0.24016192 -0.85550286  0.26251477 -2.19635827 -1.32061838
 -0.34357683  0.99237778]
   Last action arm joints (14D): [-0.81607366  0.32181581 -0.70181486 -0.86437535  0.80661354  0.3436284
  0.38023582 -0.23987917 -0.90181536  0.22165557 -2.20418756 -1.34002312
 -0.33003068  1.02196002]

🦀 Claw:
   First action claw (2D): [ 0.29605627 -0.67822337]
   Last action claw (2D): [ 0.16473532 -0.69506168]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581922.854231]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581922.855731]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581922.883145]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581922.886885]: Executing chunk of size 76 in sync mode

================================================================================
📊 Chunk Information (Step 304)
================================================================================
Chunk size: 76 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4238, 0.2825, -0.0152]
     Euler (roll, pitch, yaw): [1.37°, -62.03°, 7.30°]
   Last action:
     Position (xyz): [0.4431, 0.2775, -0.0028]
     Euler (roll, pitch, yaw): [1.02°, -64.16°, 6.93°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2428, -0.3356, 0.1051]
     Euler (roll, pitch, yaw): [43.41°, -36.97°, 32.76°]
   Last action:
     Position (xyz): [0.2452, -0.3364, 0.1223]
     Euler (roll, pitch, yaw): [42.65°, -37.59°, 32.88°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.81284482  0.32438462 -0.70071885 -0.87418754  0.80598525  0.34607609
  0.38287725 -0.23967372 -0.8996857   0.22340546 -2.20395601 -1.33549195
 -0.32953353  1.0210618 ]
   Last action arm joints (14D): [-0.8454023   0.32731275 -0.7100526  -0.92488783  0.81791854  0.35181628
  0.38080556 -0.2990033  -0.89274171  0.26775686 -2.21281495 -1.28463246
 -0.3611758   1.02668655]

🦀 Claw:
   First action claw (2D): [-0.18165708 -0.73570013]
   Last action claw (2D): [88.30592346 -0.52775741]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581924.211312]: 🔒 Left claw locked (count: 1, value: 87.84)
[INFO] [1769581924.842298]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581924.843568]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581924.872809]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581924.874694]: Executing chunk of size 55 in sync mode

================================================================================
📊 Chunk Information (Step 380)
================================================================================
Chunk size: 55 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4309, 0.2717, 0.0210]
     Euler (roll, pitch, yaw): [-3.21°, -65.75°, 3.43°]
   Last action:
     Position (xyz): [0.4258, 0.2737, 0.0717]
     Euler (roll, pitch, yaw): [-4.49°, -64.76°, 0.99°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2496, -0.3387, 0.1326]
     Euler (roll, pitch, yaw): [41.36°, -39.70°, 33.07°]
   Last action:
     Position (xyz): [0.2502, -0.3416, 0.1208]
     Euler (roll, pitch, yaw): [39.67°, -39.52°, 32.08°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.84377442  0.3297651  -0.70943381 -0.93467693  0.81761001  0.35509307
  0.38470184 -0.30157862 -0.89377475  0.26859739 -2.21230859 -1.28417434
 -0.36202677  1.02668663]
   Last action arm joints (14D): [-0.86628414  0.43857458 -0.71102767 -1.26816389  0.80432445  0.47931849
  0.62522042 -0.30339555 -0.89722563  0.26691145 -2.18285924 -1.23615595
 -0.34494732  1.03153058]

🦀 Claw:
   First action claw (2D): [87.44821167 -0.12079477]
   Last action claw (2D): [88.43205261 -0.20786524]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581926.406783]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581926.407938]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581926.437549]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581926.439384]: Executing chunk of size 44 in sync mode

================================================================================
📊 Chunk Information (Step 435)
================================================================================
Chunk size: 44 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4264, 0.2696, 0.0577]
     Euler (roll, pitch, yaw): [-7.95°, -64.25°, -1.23°]
   Last action:
     Position (xyz): [0.4152, 0.3140, 0.0621]
     Euler (roll, pitch, yaw): [-8.76°, -64.19°, -2.43°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2497, -0.3400, 0.1373]
     Euler (roll, pitch, yaw): [38.71°, -40.50°, 33.26°]
   Last action:
     Position (xyz): [0.2470, -0.3418, 0.1268]
     Euler (roll, pitch, yaw): [40.60°, -40.49°, 34.48°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.86289956  0.43482438 -0.71010681 -1.26296791  0.80754137  0.47490243
  0.61884783 -0.31304729 -0.90147818  0.26984651 -2.18648774 -1.23648013
 -0.34968041  1.03414174]
   Last action arm joints (14D): [-0.81470909  0.54001722 -0.67941861 -1.29806978  0.89040717  0.39084728
  0.58952025 -0.30679219 -0.91092358  0.25792822 -2.20101994 -1.23559393
 -0.33282814  1.02307683]

🦀 Claw:
   First action claw (2D): [87.90731812 -0.14403462]
   Last action claw (2D): [ 8.77778397e+01 -7.77184963e-02]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581928.993208]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581928.994413]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581929.023278]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581929.025255]: Executing chunk of size 52 in sync mode

================================================================================
📊 Chunk Information (Step 479)
================================================================================
Chunk size: 52 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4220, 0.3020, 0.0650]
     Euler (roll, pitch, yaw): [-11.84°, -63.35°, -4.59°]
   Last action:
     Position (xyz): [0.4236, 0.3197, 0.0531]
     Euler (roll, pitch, yaw): [-13.42°, -63.11°, -6.23°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2574, -0.3438, 0.1447]
     Euler (roll, pitch, yaw): [39.31°, -42.03°, 34.55°]
   Last action:
     Position (xyz): [0.2902, -0.3064, 0.1493]
     Euler (roll, pitch, yaw): [32.83°, -48.45°, 31.66°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.81728782  0.53684907 -0.68089207 -1.29534986  0.89158495  0.3906217
  0.59016213 -0.31644391 -0.91341896  0.26241669 -2.1984346  -1.23714881
 -0.33502103  1.02412452]
   Last action arm joints (14D): [-0.83234602  0.52632245 -0.68857763 -1.21466973  0.93130813  0.31347846
  0.5544923  -0.62971911 -0.80818986  0.51376522 -2.07535075 -1.02547634
 -0.46891712  1.04719755]

🦀 Claw:
   First action claw (2D): [85.86023712 -0.46246052]
   Last action claw (2D): [86.18448639 -0.15664697]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581930.359698]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581930.360884]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581930.391016]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581930.393758]: Executing chunk of size 93 in sync mode

================================================================================
📊 Chunk Information (Step 531)
================================================================================
Chunk size: 93 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4276, 0.3308, 0.0614]
     Euler (roll, pitch, yaw): [-13.78°, -62.86°, -7.07°]
   Last action:
     Position (xyz): [0.4240, 0.3224, 0.0537]
     Euler (roll, pitch, yaw): [-13.24°, -63.49°, -5.95°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.2874, -0.2978, 0.1815]
     Euler (roll, pitch, yaw): [35.23°, -54.00°, 36.69°]
   Last action:
     Position (xyz): [0.4116, -0.2002, 0.1940]
     Euler (roll, pitch, yaw): [6.86°, -59.82°, 14.54°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.83444514  0.52786138 -0.68923624 -1.21295867  0.93282919  0.31115235
  0.55514739 -0.63950711 -0.80984972  0.51864564 -2.07776875 -1.02451796
 -0.47300411  1.04719755]
   Last action arm joints (14D): [-0.83517097  0.53127907 -0.68902735 -1.21019328  0.93723799  0.31013613
  0.54650894 -1.1175181  -0.36008035  0.81041366 -1.44294337 -0.58488264
 -0.51696678  1.04719755]

🦀 Claw:
   First action claw (2D): [90.19368744 -0.80297589]
   Last action claw (2D): [90.47087097 -0.3834486 ]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581931.879992]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581931.882699]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581931.916558]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581931.918436]: Executing chunk of size 37 in sync mode

================================================================================
📊 Chunk Information (Step 624)
================================================================================
Chunk size: 37 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4260, 0.3309, 0.0586]
     Euler (roll, pitch, yaw): [-12.67°, -63.75°, -6.30°]
   Last action:
     Position (xyz): [0.4301, 0.3316, 0.0586]
     Euler (roll, pitch, yaw): [-11.96°, -63.67°, -6.43°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.4037, -0.2150, 0.1936]
     Euler (roll, pitch, yaw): [8.23°, -67.50°, 14.07°]
   Last action:
     Position (xyz): [0.4002, -0.2258, 0.1403]
     Euler (roll, pitch, yaw): [3.81°, -63.32°, 10.27°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.83787892  0.53411901 -0.68975095 -1.20790494  0.93765143  0.30650964
  0.5468505  -1.11458662 -0.36914769  0.80972476 -1.45275597 -0.58729775
 -0.52388175  1.04303481]
   Last action arm joints (14D): [-0.86733041  0.54007217 -0.70118586 -1.16456285  0.92954463  0.27215418
  0.5456674  -0.93905617 -0.41513741  0.73940331 -1.50680389 -0.53115805
 -0.51380379  1.00743739]

🦀 Claw:
   First action claw (2D): [ 8.61995087e+01 -3.14891338e-02]
   Last action claw (2D): [86.36940765 -0.2135098 ]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581932.794881]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581932.796005]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581932.828721]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581932.830903]: Executing chunk of size 42 in sync mode

================================================================================
📊 Chunk Information (Step 661)
================================================================================
Chunk size: 42 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4262, 0.3341, 0.0573]
     Euler (roll, pitch, yaw): [-14.92°, -64.16°, -8.14°]
   Last action:
     Position (xyz): [0.4310, 0.3337, 0.0598]
     Euler (roll, pitch, yaw): [-13.89°, -64.44°, -7.59°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.4130, -0.2134, 0.1506]
     Euler (roll, pitch, yaw): [3.54°, -63.79°, 9.49°]
   Last action:
     Position (xyz): [0.4084, -0.2180, 0.0761]
     Euler (roll, pitch, yaw): [4.08°, -59.22°, 9.34°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.8652919   0.54168938 -0.70010482 -1.16741714  0.93282588  0.27126341
  0.5437296  -0.94661081 -0.4091216   0.74228556 -1.497057   -0.5320841
 -0.51471955  1.00580335]
   Last action arm joints (14D): [-0.8730952   0.54355229 -0.70319273 -1.1579195   0.95056645  0.26009046
  0.52600304 -0.82133447 -0.33474233  0.70270871 -1.37332683 -0.5082239
 -0.46176032  0.89203463]

🦀 Claw:
   First action claw (2D): [90.20720673 -0.3862977 ]
   Last action claw (2D): [90.18029785 -0.2037704 ]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581933.862280]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581933.863496]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581933.895675]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581933.897547]: Executing chunk of size 41 in sync mode

================================================================================
📊 Chunk Information (Step 703)
================================================================================
Chunk size: 41 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4389, 0.3320, 0.0639]
     Euler (roll, pitch, yaw): [-13.39°, -63.93°, -7.44°]
   Last action:
     Position (xyz): [0.4330, 0.3341, 0.0539]
     Euler (roll, pitch, yaw): [-9.86°, -63.38°, -4.06°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.4187, -0.2037, 0.0940]
     Euler (roll, pitch, yaw): [-1.68°, -60.42°, 4.36°]
   Last action:
     Position (xyz): [0.4109, -0.2026, 0.0116]
     Euler (roll, pitch, yaw): [-2.66°, -56.00°, 2.95°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.87825127  0.54100467 -0.70567822 -1.15088027  0.95070264  0.25807735
  0.52678361 -0.83079331 -0.32846621  0.70559199 -1.36621644 -0.50075502
 -0.46787858  0.89699732]
   Last action arm joints (14D): [-0.87242294  0.53218983 -0.70430214 -1.12891721  0.93099522  0.26935821
  0.53407855 -0.73894796 -0.23533315  0.68740282 -1.19631332 -0.43787189
 -0.46827058  0.77624386]

🦀 Claw:
   First action claw (2D): [84.12045288 -0.35755634]
   Last action claw (2D): [83.66181946 -0.28668046]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581935.038197]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581935.039391]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581935.073895]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581935.076535]: Executing chunk of size 76 in sync mode

================================================================================
📊 Chunk Information (Step 744)
================================================================================
Chunk size: 76 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4299, 0.3389, 0.0566]
     Euler (roll, pitch, yaw): [-12.30°, -63.90°, -6.47°]
   Last action:
     Position (xyz): [0.4597, 0.3390, 0.0516]
     Euler (roll, pitch, yaw): [-12.09°, -64.04°, -6.07°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.4269, -0.2064, 0.0609]
     Euler (roll, pitch, yaw): [-6.65°, -61.44°, -0.97°]
   Last action:
     Position (xyz): [0.4240, -0.2155, -0.0132]
     Euler (roll, pitch, yaw): [-8.57°, -54.74°, -2.55°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.87185469  0.53381096 -0.70385559 -1.13079673  0.93207134  0.26795941
  0.53377611 -0.7486646  -0.23724806  0.68959249 -1.19747016 -0.43884077
 -0.47147808  0.77797486]
   Last action arm joints (14D): [-0.98303271  0.4629815  -0.75587521 -0.88116674  0.9636181   0.14527274
  0.44937302 -0.76358535 -0.21736561  0.69286713 -1.02831591 -0.46686636
 -0.45846692  0.67525045]

🦀 Claw:
   First action claw (2D): [89.68070984 -0.24368763]
   Last action claw (2D): [90.01941681 -0.10725856]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581938.925097]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581938.926334]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581938.957004]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581938.963412]: Executing chunk of size 72 in sync mode

================================================================================
📊 Chunk Information (Step 820)
================================================================================
Chunk size: 72 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4504, 0.3569, 0.0645]
     Euler (roll, pitch, yaw): [-11.06°, -65.10°, -5.35°]
   Last action:
     Position (xyz): [0.4545, 0.3366, 0.0712]
     Euler (roll, pitch, yaw): [-11.56°, -64.85°, -6.76°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.4506, -0.2002, 0.0190]
     Euler (roll, pitch, yaw): [-7.86°, -60.37°, -4.00°]
   Last action:
     Position (xyz): [0.4252, -0.2164, -0.0498]
     Euler (roll, pitch, yaw): [-8.94°, -60.29°, -5.79°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.98192946  0.46812899 -0.75513839 -0.88760164  0.96498277  0.14645263
  0.45114714 -0.77305923 -0.21290211  0.6947276  -1.01916913 -0.47127984
 -0.45840653  0.66856472]
   Last action arm joints (14D): [-0.98352591  0.49855046 -0.75480402 -0.98705013  0.95255428  0.19827947
  0.50093133 -0.75873957 -0.1694238   0.69262236 -0.86075059 -0.55001562
 -0.43408654  0.40735349]

🦀 Claw:
   First action claw (2D): [89.24204254 -0.25687814]
   Last action claw (2D): [89.96327209 89.05541229]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: 
Continuing with chunk execution...

[INFO] [1769581940.401570]: 🔒 Right claw locked (count: 1, value: 89.06)
[INFO] [1769581940.681627]: ⏭️  Selected actions from index 1 to 7 (inclusive): 7 actions
[INFO] [1769581940.683114]:    EEF mode: Converting action_chunk from 20D EEF space to 16D joint space
[INFO] [1769581940.714174]:    ✅ Converted action_chunk to joint space: (7, 16)
[INFO] [1769581940.717000]: Executing chunk of size 165 in sync mode

================================================================================
📊 Chunk Information (Step 892)
================================================================================
Chunk size: 165 actions
Action dimension: 16

🎯 Action mode: Relative eef

🤚 Left EEF:
   First action:
     Position (xyz): [0.4450, 0.3543, 0.0628]
     Euler (roll, pitch, yaw): [-9.91°, -63.13°, -4.16°]
   Last action:
     Position (xyz): [0.4566, 0.3445, 0.0476]
     Euler (roll, pitch, yaw): [-5.76°, -63.73°, -1.00°]

🤚 Right EEF:
   First action:
     Position (xyz): [0.4553, -0.2027, -0.0251]
     Euler (roll, pitch, yaw): [-8.71°, -65.45°, -6.18°]
   Last action:
     Position (xyz): [0.4494, -0.2144, -0.0156]
     Euler (roll, pitch, yaw): [-7.22°, -64.40°, -5.29°]

🦾 Arm Joints (Joint Space):
   First action arm joints (14D): [-0.98159237  0.50065226 -0.75380265 -0.98855987  0.95352844  0.19801013
  0.5017789  -0.76557365 -0.16557779  0.69360911 -0.85093827 -0.55232214
 -0.43176691  0.40163127]
   Last action arm joints (14D): [-0.96730062  0.47552385 -0.74878085 -0.89149562  0.93293371  0.17713613
  0.45746407 -0.87227904 -0.15000347  0.71331758 -0.78570877 -0.60325303
 -0.40295339  0.36980664]

🦀 Claw:
   First action claw (2D): [87.88930511 -0.25489926]
   Last action claw (2D): [88.40037537 87.49731445]
================================================================================
Press Enter to execute this chunk, or 'q'+Enter to stop: q

[User] Stopping inference by user request

================================================================================
✅ Inference session #1 stopped by user (q pressed)
[INFO] [1769581943.189990]: Resetting policy state after inference stop...
[INFO] [1769581943.191413]:    ✅ Policy reset complete
[INFO] [1769581943.192230]: Resetting arm position using JSON file...
[INFO] [1769581943.193513]: Resetting claw lock state before opening claws...
[INFO] [1769581943.194365]: 🔓 Resetting claw lock state (was locked: left, right)
[INFO] [1769581943.195603]: ✅ Claw lock reset complete
[INFO] [1769581943.196482]: Opening claws before reset...
[INFO] [1769581944.199480]: Loading initial arm trajectory from JSON: /home/lab/humanoid_groot/scripts/utils/initial_arm_traj.json
[INFO] [1769581944.202135]: Resetting arm from current position to initial position over 5.0s (50 steps)...
[INFO] [1769581949.227435]: Arm reset completed!
💡 Ready for next inference session. Press Enter to start, or Ctrl+C to exit.
================================================================================

Press Enter to start next inference, or 'q'+Enter to exit: q

👋 Exiting program. Goodbye!