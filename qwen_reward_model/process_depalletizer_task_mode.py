"""
使用 Qwen2.5-VL 模型根据实时图像序列判断当前应该执行的任务

该脚本会：
1. 订阅 ROS 话题 /camera/color/image_raw 获取实时图像
2. 维护最近5帧的图像历史
3. 使用 Qwen2.5-VL 模型根据图像序列判断当前应该执行的任务
4. 从 tasks.jsonl 读取任务列表（5个任务）
"""

import argparse
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Empty, EmptyResponse
# 延迟导入cv_bridge以避免NumPy版本兼容性问题
CV_BRIDGE_AVAILABLE = False
CvBridge = None
try:
    from cv_bridge import CvBridge
    # 即使导入成功，也可能在运行时崩溃，所以标记为可用但会在使用时测试
    CV_BRIDGE_AVAILABLE = True
except (ImportError, AttributeError, SystemError) as e:
    CV_BRIDGE_AVAILABLE = False
    print(f"⚠️  Warning: cv_bridge import failed: {e}")
    print("  This may be due to NumPy version incompatibility.")
    print("  Will use direct conversion from ROS message data.")
    # 创建一个占位符类
    class CvBridge:
        def imgmsg_to_cv2(self, msg, desired_encoding='passthrough'):
            raise RuntimeError("cv_bridge not available. Please install or fix NumPy compatibility.")
except Exception as e:
    # 捕获其他可能的异常
    CV_BRIDGE_AVAILABLE = False
    print(f"⚠️  Warning: cv_bridge import error: {e}")
    print("  Will use direct conversion from ROS message data.")
    class CvBridge:
        def imgmsg_to_cv2(self, msg, desired_encoding='passthrough'):
            raise RuntimeError("cv_bridge not available.")

from PIL import Image as PILImage
import torch

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Some image format conversions may fail.")

# 使用matplotlib进行显示
try:
    import matplotlib
    matplotlib.use('TkAgg')  # 使用TkAgg后端，更可靠
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. GUI visualization will be disabled.")
except Exception as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"Warning: matplotlib initialization failed: {e}. GUI visualization will be disabled.")

try:
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        QWEN_MODEL_CLASS = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        try:
            from transformers import AutoModel
            QWEN_MODEL_CLASS = AutoModel
        except ImportError:
            from transformers import AutoModelForCausalLM
            QWEN_MODEL_CLASS = AutoModelForCausalLM
except ImportError:
    raise ImportError("transformers library is required. Install with: pip install transformers")


class ImageBuffer:
    """维护图像历史缓冲区"""
    def __init__(self, max_size: int = 5):
        self.buffer = deque(maxlen=max_size)
        self.maxlen = max_size
        self.lock = threading.Lock()
    
    def add_image(self, image: PILImage.Image):
        """添加新图像到缓冲区"""
        with self.lock:
            self.buffer.append(image)
    
    def get_images(self) -> List[PILImage.Image]:
        """获取当前所有图像（线程安全）"""
        with self.lock:
            return list(self.buffer)
    
    def is_ready(self) -> bool:
        """检查缓冲区是否已满（达到最大大小）"""
        with self.lock:
            return len(self.buffer) >= self.buffer.maxlen


class TaskClassifier:
    """使用 Qwen2.5-VL 进行任务分类"""
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        tasks_file: str = None,
        prompt_template: str = None,
        prompt_style: str = "detailed"
    ):
        """
        初始化任务分类器
        
        Args:
            model_name: Qwen2.5-VL 模型名称
            device: 设备（auto, cuda, cpu）
            tasks_file: 任务列表文件路径（tasks.jsonl）
            prompt_template: 自定义 prompt 模板（如果为 None，使用默认模板）
            prompt_style: prompt 风格 ("detailed", "simple", "step_by_step")
        """
        print(f"Loading model: {model_name}")
        self.model, self.processor = self._load_model_and_processor(model_name, device)
        self.device = next(self.model.parameters()).device
        
        # 加载任务列表
        if tasks_file is None:
            # 默认路径
            tasks_file = "/home/lab/lerobot_groot/lerobot_data/1125_groot_train_data_with_task_filtered/meta/tasks.jsonl"
        
        self.tasks = self._load_tasks(tasks_file)
        self.prompt_template = prompt_template
        self.prompt_style = prompt_style
        print(f"Loaded {len(self.tasks)} tasks:")
        for task_idx, task_desc in self.tasks.items():
            print(f"  Task {task_idx}: {task_desc}")
    
    def _load_model_and_processor(self, model_name: str, device: str):
        """加载模型和处理器"""
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            try:
                model = QWEN_MODEL_CLASS.from_pretrained(
                    model_name,
                    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map=device,
                    trust_remote_code=True
                )
            except (ValueError, TypeError) as e:
                print(f"Warning: Direct model class failed, trying AutoModel: {e}")
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    model_name,
                    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map=device,
                    trust_remote_code=True
                )
            
            model.eval()
            print("✅ Model loaded successfully!")
            return model, processor
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _load_tasks(self, tasks_file: str) -> Dict[int, str]:
        """从 tasks.jsonl 加载任务列表"""
        tasks = {}
        try:
            with open(tasks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        task_data = json.loads(line.strip())
                        task_idx = task_data.get('task_index')
                        task_desc = task_data.get('task')
                        if task_idx is not None and task_desc:
                            tasks[task_idx] = task_desc
        except Exception as e:
            print(f"Warning: Failed to load tasks from {tasks_file}: {e}")
            # 使用默认任务
            tasks = {
                0: "Depalletize the green box on the left",
                1: "Depalletize the gray box on the left",
                2: "Depalletize the gray box on the right",
                3: "Depalletize the green box on the right",
                4: "Pick up a single green box"
            }
            print("Using default tasks")
        
        return tasks
    
    def _build_prompt(self) -> str:
        """构建分类 prompt"""
        # 如果提供了自定义模板，直接使用
        if self.prompt_template is not None:
            task_list_text = "\n".join([f"Task {idx}: {desc}" for idx, desc in sorted(self.tasks.items())])
            return self.prompt_template.format(task_list=task_list_text)
        
        # 根据风格选择不同的 prompt
        task_list_text = "\n".join([f"Task {idx}: {desc}" for idx, desc in sorted(self.tasks.items())])
        
        if self.prompt_style == "simple":
            return f"""You are a robot task classifier. Analyze the image sequence and determine which task to execute.

Available tasks:
{task_list_text}

Observe the images carefully. Identify box colors (green/gray) and positions (left/right). Output ONLY the task index number (0-4)."""
        
        elif self.prompt_style == "step_by_step":
            return f"""You are a robot task classifier. Follow these steps to determine the task:

STEP 1: Identify box colors in the scene
- Look for GREEN boxes
- Look for GRAY boxes

STEP 2: Identify box positions
- Determine if boxes are on the LEFT side
- Determine if boxes are on the RIGHT side

STEP 3: Count boxes
- Count how many boxes are visible

STEP 4: Match to task
Available tasks:
{task_list_text}

STEP 5: Output the task index (0-4) that matches your observations.

Output format: Just the number (0, 1, 2, 3, or 4)"""
        
        else:  # detailed (default)
            return f"""You are a precision robot vision system. Your goal is to analyze the provided camera image and determine the correct task index from the list below by following a strict reasoning process.

AVAILABLE TASKS:
{task_list_text}

REASONING PROCESS:

Step 1: Locate the Target Box.
The target box is the one with a square black-and-white tag (QR-like code) on it. All other boxes are context.

Step 2: Analyze the Target Box and its Surroundings.
- What is the color of the tagged box? (Green or Gray)
- Where is the tagged box located within the camera frame? (e.g., primarily in the left half, right half, or center).
- Observe the boxes immediately next to the tagged box. Is there a gray box visible anywhere in the image?

Step 3: Apply the Perspective Inversion Rule.
The task names use "left" and "right" from the robot's point of view. The camera sees an inverted perspective.
- If the target box is in the **LEFT half** of the image, it corresponds to a **"right"** task.
- If the target box is in the **RIGHT half** of the image, it corresponds to a **"left"** task.

Step 4: Match Observations to Task Criteria.
Use your observations from Step 2 and the rule from Step 3 to find the single best match from the criteria below.

TASK IDENTIFICATION CRITERIA:
- **Task 0 (Depalletize the green box on the left)**:
  - The tagged box is GREEN.
  - The tagged box is located in the **RIGHT half** of the image.
  - There are typically other green boxes but no gray boxes nearby.

- **Task 1 (Depalletize the gray box on the left)**:
  - The tagged box is GRAY.
  - The tagged box is located in the **RIGHT half** of the image.

- **Task 2 (Depalletize the gray box on the right)**:
  - The tagged box is GRAY.
  - The tagged box is located in the **LEFT half** of the image.

- **Task 3 (Depalletize the green box on the right)**:
  - The tagged box is GREEN.
  - The tagged box is located in the **LEFT half** of the image.
  - A key distinguishing feature is the presence of a **GRAY box** next to the green boxes.

- **Task 4 (Pick up a single green box)**:
  - The scene shows **only ONE green box remaining on the top-most layer**.

Step 5: State Your Final Answer.
Explain your reasoning, then conclude with a sentence like "The final answer is Task X." where X is the index.

OUTPUT FORMAT:
Explain your reasoning, then conclude with a sentence like "The final answer is Task X." where X is the index."""
    
    def classify_task(self, images: List[PILImage.Image], verbose: bool = True) -> Dict[str, Any]:
        """
        根据图像序列分类任务
        
        Args:
            images: 图像列表（最多5帧）
            verbose: 是否打印详细信息
        
        Returns:
            包含任务索引、任务描述、置信度等的字典
        """
        if len(images) == 0:
            return {
                "task_index": None,
                "task_description": None,
                "confidence": 0.0,
                "response": "No images provided",
                "error": "No images"
            }
        
        # 构建提示文本
        prompt = self._build_prompt()
        
        # 准备对话格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in images
                ] + [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        try:
            # 处理输入
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=text,
                images=images,
                padding=True,
                return_tensors="pt"
            )
            
            # 移动到设备
            if hasattr(inputs, 'to'):
                inputs = inputs.to(self.device)
            else:
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # 生成响应
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )
            
            # 解码响应
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            if verbose:
                print(f"Model response: {response}")
            
            # 提取任务索引
            task_index = self._extract_task_index(response)
            
            if task_index is not None and task_index in self.tasks:
                task_description = self.tasks[task_index]
                return {
                    "task_index": task_index,
                    "task_description": task_description,
                    "confidence": 1.0,  # 可以后续改进为实际置信度
                    "response": response,
                    "num_images": len(images)
                }
            else:
                return {
                    "task_index": None,
                    "task_description": None,
                    "confidence": 0.0,
                    "response": response,
                    "error": f"Could not extract valid task index from response: {response}"
                }
        
        except Exception as e:
            print(f"Error during classification: {e}")
            import traceback
            traceback.print_exc()
            return {
                "task_index": None,
                "task_description": None,
                "confidence": 0.0,
                "response": f"Error: {str(e)}",
                "error": str(e)
            }
    
    def _extract_task_index(self, response: str) -> Optional[int]:
        """从模型响应中提取任务索引"""
        if not response:
            return None
        
        # 清理响应文本
        response_clean = response.strip()
        
        # 尝试提取数字（0-4）
        patterns = [
            r'\b([0-4])\b',  # 直接匹配 0-4
            r'task\s*[：:]\s*([0-4])',  # "task: 0"
            r'task\s*([0-4])',  # "task 0"
            r'([0-4])\s*$',  # 行尾的数字
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_clean, re.IGNORECASE)
            if matches:
                try:
                    task_idx = int(matches[0])
                    if 0 <= task_idx <= 4:
                        return task_idx
                except ValueError:
                    continue
        
        # 如果找不到，尝试查找第一个数字
        numbers = re.findall(r'\d+', response_clean)
        if numbers:
            try:
                task_idx = int(numbers[0])
                if 0 <= task_idx <= 4:
                    return task_idx
            except ValueError:
                pass
        
        return None


class ROSImageSubscriber:
    """ROS 图像订阅器"""
    def __init__(
        self,
        topic: str = "/camera/color/image_raw",
        image_buffer: ImageBuffer = None,
        classification_interval: float = 2.0,
        enable_gui: bool = True
    ):
        """
        初始化 ROS 图像订阅器
        
        Args:
            topic: ROS 话题名称
            image_buffer: 图像缓冲区
            classification_interval: 分类间隔（秒）
            enable_gui: 是否启用 GUI 可视化
        """
        self.topic = topic
        self.image_buffer = image_buffer if image_buffer else ImageBuffer(max_size=5)
        
        # 由于NumPy版本兼容性问题，cv_bridge在运行时会导致segmentation fault
        # 因此默认不使用cv_bridge，直接使用备用转换方法
        # 只有在环境变量USE_CV_BRIDGE=1时才尝试使用cv_bridge
        self.bridge = None
        self.use_cv_bridge = False
        
        # 检查是否用户明确要求使用cv_bridge
        use_cv_bridge_env = os.environ.get('USE_CV_BRIDGE', '0')
        if use_cv_bridge_env.lower() in ['1', 'true', 'yes']:
            if CV_BRIDGE_AVAILABLE and CvBridge is not None:
                try:
                    self.bridge = CvBridge()
                    self.use_cv_bridge = True
                    print("⚠️  Warning: cv_bridge enabled via USE_CV_BRIDGE=1")
                    print("   This may cause segmentation fault due to NumPy incompatibility!")
                except Exception as e:
                    print(f"⚠️  Warning: cv_bridge initialization failed: {e}")
                    print("   Will use direct conversion from ROS message data.")
                    self.use_cv_bridge = False
            else:
                print("⚠️  Warning: USE_CV_BRIDGE=1 but cv_bridge is not available")
                print("   Will use direct conversion from ROS message data.")
        else:
            # 默认使用直接转换方法（避免segmentation fault）
            print("ℹ️  Using direct conversion from ROS message data (default)")
            print("   Set USE_CV_BRIDGE=1 to enable cv_bridge (may cause crashes)")
        
        self.classification_interval = classification_interval
        self.last_classification_time = 0
        self.classifier = None
        self.running = False
        # GUI需要matplotlib可用
        self.enable_gui = enable_gui and MATPLOTLIB_AVAILABLE
        
        # 可视化相关
        self.current_task_result = None
        self.latest_cv_image = None
        self.latest_cv_image_lock = threading.Lock()
        self.window_name = "Task Classifier - Camera View"
        self._image_info_printed = False  # 用于控制调试信息输出
        self._display_error_printed = False  # 用于控制显示错误信息输出
        self._last_display_time = 0
        self._display_interval = 0.1  # 限制显示频率，每100ms最多显示一次
        self._frame_count = 0
        self._display_every_n_frames = 3  # 每3帧显示一次，减少开销
        
        # matplotlib相关
        self.fig = None
        self.ax = None
        
        # ROS 初始化
        if not rospy.get_node_uri():
            rospy.init_node('task_classifier_node', anonymous=True)
        
        # 创建服务，用于重置状态
        self.reset_service = rospy.Service('~reset', Empty, self.handle_reset_service)
        print(f"✅ Reset service available at: {rospy.get_name()}/reset")
        
        # 创建订阅者
        self.subscriber = rospy.Subscriber(
            topic,
            Image,
            self.image_callback,
            queue_size=1
        )
        
        print(f"✅ Subscribed to ROS topic: {topic}")
        
        # 如果启用 GUI，创建matplotlib窗口
        if self.enable_gui:
            try:
                self.fig, self.ax = plt.subplots(figsize=(12, 8))
                self.fig.canvas.manager.set_window_title(self.window_name)
                self.ax.axis('off')
                # 设置非阻塞模式
                plt.ion()  # 开启交互模式
                plt.show(block=False)
                print("✅ GUI visualization enabled (using matplotlib)")
            except Exception as e:
                print(f"⚠️  Warning: Failed to create matplotlib window: {e}")
                print("   Disabling GUI visualization. The program will continue without GUI.")
                self.enable_gui = False
    
    def set_classifier(self, classifier: TaskClassifier):
        """设置任务分类器"""
        self.classifier = classifier
    
    def handle_reset_service(self, req):
        """Handles requests to reset the classifier's state."""
        rospy.loginfo("Received request to reset task classifier state.")
        self.reset_state()
        return EmptyResponse()

    def reset_state(self):
        """Clears the image buffer and resets classification state."""
        # Clear the image buffer
        with self.image_buffer.lock:
            self.image_buffer.buffer.clear()
        
        # Reset the last classification result shown on GUI
        self.current_task_result = None
        
        # Allow for immediate re-classification once buffer is ready
        self.last_classification_time = 0
        
        rospy.loginfo("✅ Classifier state has been reset. Buffer is now empty.")
    
    def _convert_ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        """
        将ROS Image消息转换为numpy数组（RGB格式）
        备用方法，不依赖cv_bridge
        """
        # 根据编码格式确定通道数和每像素字节数
        if msg.encoding in ['rgb8', 'bgr8']:
            channels = 3
            bytes_per_pixel = 3
        elif msg.encoding in ['rgba8', 'bgra8']:
            channels = 4
            bytes_per_pixel = 4
        elif msg.encoding in ['mono8', '8UC1']:
            channels = 1
            bytes_per_pixel = 1
        elif msg.encoding in ['mono16', '16UC1']:
            channels = 1
            bytes_per_pixel = 2
        else:
            # 默认尝试3通道
            channels = 3
            bytes_per_pixel = 3
            print(f"⚠️  Warning: Unknown encoding '{msg.encoding}', assuming 3 channels")
        
        # 处理16位图像
        if msg.encoding in ['mono16', '16UC1']:
            img_data = np.frombuffer(msg.data, dtype=np.uint16)
        else:
            img_data = np.frombuffer(msg.data, dtype=np.uint8)
        
        # 计算每行的期望字节数
        expected_row_size = msg.width * bytes_per_pixel
        
        # 检查step字段（每行的实际字节数，可能包含padding）
        if hasattr(msg, 'step') and msg.step > 0:
            actual_row_size = msg.step
        else:
            actual_row_size = expected_row_size
        
        # 如果step与期望值不同，说明有padding，需要特殊处理
        if actual_row_size != expected_row_size:
            # 有padding的情况：需要逐行提取数据
            img_arr = np.zeros((msg.height, msg.width, channels) if channels > 1 else (msg.height, msg.width), 
                             dtype=img_data.dtype)
            for row in range(msg.height):
                start_idx = row * actual_row_size
                end_idx = start_idx + expected_row_size
                row_data = img_data[start_idx:end_idx]
                if channels == 1:
                    img_arr[row, :] = row_data.reshape(msg.width)
                else:
                    img_arr[row, :, :] = row_data.reshape(msg.width, channels)
        else:
            # 没有padding，直接reshape
            if channels == 1:
                img_arr = img_data.reshape(msg.height, msg.width)
            else:
                img_arr = img_data.reshape(msg.height, msg.width, channels)
        
        # 转换为RGB格式
        if msg.encoding == 'bgr8':
            if CV2_AVAILABLE:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            else:
                # 手动转换BGR到RGB
                img_arr = img_arr[:, :, ::-1]
        elif msg.encoding == 'bgra8':
            if CV2_AVAILABLE:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2RGB)
            else:
                # 手动转换BGRA到RGB
                img_arr = img_arr[:, :, [2, 1, 0]]
        elif msg.encoding == 'rgba8':
            if CV2_AVAILABLE:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
            else:
                # 手动转换RGBA到RGB
                img_arr = img_arr[:, :, :3]
        elif msg.encoding in ['mono8', '8UC1', 'mono16', '16UC1']:
            # 灰度图转RGB
            if CV2_AVAILABLE:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
            else:
                # 手动转换灰度到RGB
                img_arr = np.stack([img_arr, img_arr, img_arr], axis=-1)
        
        # 确保是uint8类型
        if img_arr.dtype != np.uint8:
            if img_arr.max() > 255:
                # 16位图像，缩放到8位
                img_arr = (img_arr / 256).astype(np.uint8)
            else:
                img_arr = img_arr.astype(np.uint8)
        
        return img_arr
    
    def image_callback(self, msg: Image):
        """图像回调函数"""
        try:
            # 转换 ROS Image 消息为 numpy 数组（RGB格式）
            cv_image = None
            
            # 由于NumPy兼容性问题，cv_bridge会导致segmentation fault
            # 因此默认直接使用备用转换方法
            # 只有在明确启用且没有NumPy问题时才使用cv_bridge
            if self.use_cv_bridge and self.bridge is not None:
                # 注意：即使这里尝试使用，也可能导致segmentation fault
                # 所以默认情况下不会执行到这里
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                except Exception as e:
                    print(f"⚠️  cv_bridge conversion failed: {e}")
                    print("   Switching to direct conversion method.")
                    self.use_cv_bridge = False
                    cv_image = None
            else:
                cv_image = None
            
            # 使用直接转换方法（默认或作为备用）
            if cv_image is None:
                cv_image = self._convert_ros_image_to_numpy(msg)
            
            # 验证图像数据
            if cv_image is None or cv_image.size == 0:
                print(f"⚠️  Warning: Invalid image data (shape: {cv_image.shape if cv_image is not None else 'None'})")
                return
            
            # 打印图像信息（仅第一次）
            if not self._image_info_printed:
                print(f"ℹ️  Image info: encoding={msg.encoding}, shape={cv_image.shape}, dtype={cv_image.dtype}, "
                      f"min={cv_image.min()}, max={cv_image.max()}, step={getattr(msg, 'step', 'N/A')}")
                self._image_info_printed = True
            
            # 确保图像是3通道RGB格式
            if len(cv_image.shape) == 2:
                # 灰度图，转换为RGB
                if CV2_AVAILABLE:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
                else:
                    cv_image = np.stack([cv_image, cv_image, cv_image], axis=-1)
            elif len(cv_image.shape) == 3 and cv_image.shape[2] != 3:
                # 如果不是3通道，转换为3通道
                if cv_image.shape[2] == 4:
                    # RGBA转RGB
                    if CV2_AVAILABLE:
                        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)
                    else:
                        cv_image = cv_image[:, :, :3]
                else:
                    print(f"⚠️  Warning: Unexpected image shape: {cv_image.shape}")
                    return
            
            # 根据用户要求，将图像旋转180度
            if CV2_AVAILABLE:
                cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
            else:
                # 如果OpenCV不可用，使用NumPy进行旋转
                cv_image = np.rot90(cv_image, 2)

            # 保存图像数据用于显示（不在回调中直接操作matplotlib，避免线程问题）
            if self.enable_gui:
                self._frame_count += 1
                current_time = time.time()
                
                # 限制更新频率：时间间隔和帧数双重限制
                if (self._frame_count % self._display_every_n_frames == 0 and 
                    current_time - self._last_display_time >= self._display_interval):
                    try:
                        # 确保图像是连续的
                        if not cv_image.flags['C_CONTIGUOUS']:
                            cv_image = np.ascontiguousarray(cv_image)
                        
                        # 只保存图像数据，不在这里操作matplotlib（线程安全）
                        with self.latest_cv_image_lock:
                            self.latest_cv_image = cv_image.copy()
                        
                        self._last_display_time = current_time
                    except Exception as e:
                        if not self._display_error_printed:
                            print(f"⚠️  Warning: Failed to save image for display: {e}")
                            self._display_error_printed = True
            
            # 转换为 PIL Image
            pil_image = PILImage.fromarray(cv_image)
            
            # 添加到缓冲区
            self.image_buffer.add_image(pil_image)
            
            # 检查是否需要分类
            current_time = time.time()
            if (self.classifier is not None and 
                self.image_buffer.is_ready() and
                current_time - self.last_classification_time >= self.classification_interval):
                
                self.last_classification_time = current_time
                self.classify_current_task()
        
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
    
    def classify_current_task(self):
        """对当前图像序列进行分类"""
        images = self.image_buffer.get_images()
        if len(images) > 0 and self.classifier is not None:
            print(f"\n🔍 Classifying task with {len(images)} images...")
            result = self.classifier.classify_task(images, verbose=True)
            
            # 保存任务结果用于可视化
            self.current_task_result = result
            
            if result["task_index"] is not None:
                print(f"✅ Predicted Task: {result['task_index']} - {result['task_description']}")
                print(f"   Confidence: {result['confidence']:.2f}")
            else:
                print(f"❌ Failed to classify task: {result.get('error', 'Unknown error')}")
                print(f"   Response: {result['response']}")
    
    def _update_display_timer_callback(self, event):
        """ROS定时器回调，在主线程中更新matplotlib显示（线程安全）"""
        if not self.enable_gui or self.fig is None:
            return
        
        try:
            # 检查窗口是否关闭
            if not plt.fignum_exists(self.fig.number):
                print("\nMatplotlib window closed. Shutting down...")
                rospy.signal_shutdown("Window closed")
                return
            
            # 获取最新的图像数据
            current_image = None
            with self.latest_cv_image_lock:
                if self.latest_cv_image is not None:
                    current_image = self.latest_cv_image
                    self.latest_cv_image = None  # 消费图像，避免重复显示
            
            # 如果有新图像，更新显示
            if current_image is not None:
                self._draw_task_info_matplotlib(current_image)
                # 刷新显示（在主线程中，安全）
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
        except Exception as e:
            # 静默处理错误，避免频繁打印
            pass
    
    def _draw_task_info_matplotlib(self, image: np.ndarray):
        """使用matplotlib绘制任务信息（RGB格式）- 必须在主线程中调用"""
        if image is None or image.size == 0:
            return
        
        try:
            # 确保图像是连续的
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
            
            h, w = image.shape[:2]
            
            if h <= 0 or w <= 0 or len(image.shape) != 3:
                return
            
            # 清除之前的文本和图像
            self.ax.clear()
            self.ax.axis('off')
            
            # 显示图像
            self.ax.imshow(image)
            
            # 添加任务信息文本（使用相对坐标，0-1之间）
            if self.current_task_result is not None:
                result = self.current_task_result
                if result.get("task_index") is not None:
                    task_text = f"Task {result['task_index']}: {result.get('task_description', '')[:50]}"
                    self.ax.text(0.02, 0.05, task_text, transform=self.ax.transAxes,
                               fontsize=14, color='green', weight='bold',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    conf_text = f"Confidence: {result.get('confidence', 0):.2f}"
                    self.ax.text(0.02, 0.10, conf_text, transform=self.ax.transAxes,
                               fontsize=12, color='yellow',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                else:
                    error_msg = result.get('error', 'Unknown error')
                    if len(error_msg) > 50:
                        error_msg = error_msg[:47] + "..."
                    self.ax.text(0.02, 0.05, "Classification Failed", transform=self.ax.transAxes,
                               fontsize=14, color='red', weight='bold',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    self.ax.text(0.02, 0.10, error_msg, transform=self.ax.transAxes,
                               fontsize=10, color='red',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            else:
                buffer_status = f"Buffer: {len(self.image_buffer.get_images())}/{self.image_buffer.maxlen}"
                self.ax.text(0.02, 0.05, "Waiting for classification...", transform=self.ax.transAxes,
                           fontsize=14, color='yellow', weight='bold',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                self.ax.text(0.02, 0.10, buffer_status, transform=self.ax.transAxes,
                           fontsize=10, color='gray',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            # Frame counter for debugging
            self.ax.text(0.02, 0.98, f"Frame: {self._frame_count}", transform=self.ax.transAxes,
                       fontsize=10, color='white', ha='left', va='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

            # 底部提示
            self.ax.text(0.98, 0.98, "Close window to quit", transform=self.ax.transAxes,
                       fontsize=10, color='gray', ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        except Exception as e:
            if not self._display_error_printed:
                print(f"⚠️  Warning: Failed to draw task info: {e}")
                import traceback
                traceback.print_exc()
                self._display_error_printed = True
    
    
    
    def spin(self):
        """运行 ROS 节点，并在主线程中处理 GUI 更新"""
        self.running = True
        print("🔄 Starting ROS node...")

        # 如果禁用 GUI，则使用 rospy.spin() 阻塞
        if not self.enable_gui:
            rospy.spin()
            self.running = False
            return

        # 如果启用 GUI，则运行自定义循环以在主线程中处理 matplotlib 更新
        rate = rospy.Rate(30)  # 30 Hz
        try:
            while not rospy.is_shutdown():
                self._update_display_timer_callback(None)  # 调用 GUI 更新逻辑
                rate.sleep()
        finally:
            self.running = False
            # 清理 matplotlib 窗口
            if self.fig is not None:
                try:
                    plt.close(self.fig)
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="使用 Qwen2.5-VL 根据实时图像判断任务")
    parser.add_argument(
        "--topic",
        type=str,
        default="/camera/color/image_raw",
        help="ROS 图像话题名称"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen2.5-VL 模型名称"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备（auto, cuda, cpu）"
    )
    parser.add_argument(
        "--tasks_file",
        type=str,
        default=None,
        help="任务列表文件路径（tasks.jsonl）"
    )
    parser.add_argument(
        "--history_size",
        type=int,
        default=5,
        help="图像历史缓冲区大小（默认5帧）"
    )
    parser.add_argument(
        "--classification_interval",
        type=float,
        default=2.0,
        help="分类间隔（秒，默认2.0秒）"
    )
    parser.add_argument(
        "--enable_gui",
        action="store_true",
        default=True,
        help="启用 GUI 可视化窗口（默认启用）"
    )
    parser.add_argument(
        "--disable_gui",
        action="store_true",
        help="禁用 GUI 可视化窗口"
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="detailed",
        choices=["detailed", "simple", "step_by_step"],
        help="Prompt 风格：detailed（详细，默认）、simple（简单）、step_by_step（分步）"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="自定义 prompt 模板文件路径（可选，会覆盖 prompt_style）"
    )
    
    args = parser.parse_args()
    
    # 处理 GUI 选项
    enable_gui = args.enable_gui and not args.disable_gui
    
    if enable_gui and not MATPLOTLIB_AVAILABLE:
        print("⚠️  Warning: GUI requested but matplotlib not available. Disabling GUI.")
        print("   Install matplotlib to enable GUI visualization: pip install matplotlib")
        enable_gui = False
    
    # 读取自定义 prompt 模板（如果有）
    prompt_template = None
    if args.prompt_template:
        try:
            with open(args.prompt_template, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            print(f"✅ Loaded custom prompt template from: {args.prompt_template}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to load prompt template: {e}")
            print("   Using default prompt style instead.")
    
    # 创建图像缓冲区
    image_buffer = ImageBuffer(max_size=args.history_size)
    
    # 创建任务分类器
    print("Initializing task classifier...")
    classifier = TaskClassifier(
        model_name=args.model_name,
        device=args.device,
        tasks_file=args.tasks_file,
        prompt_template=prompt_template,
        prompt_style=args.prompt_style
    )
    
    # 创建 ROS 订阅器
    subscriber = ROSImageSubscriber(
        topic=args.topic,
        image_buffer=image_buffer,
        classification_interval=args.classification_interval,
        enable_gui=enable_gui
    )
    subscriber.set_classifier(classifier)
    
    # 运行
    try:
        print("\n" + "="*60)
        print("Task Classifier is running...")
        print(f"  Topic: {args.topic}")
        print(f"  History size: {args.history_size} frames")
        print(f"  Classification interval: {args.classification_interval} seconds")
        print(f"  GUI visualization: {'Enabled' if enable_gui else 'Disabled'}")
        print("="*60)
        if enable_gui:
            print("\n💡 GUI Window opened. Close the matplotlib window to quit.")
        print("\nWaiting for images...")
        subscriber.spin()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except rospy.ROSInterruptException:
        print("\n\nROS interrupted")


if __name__ == "__main__":
    main()
