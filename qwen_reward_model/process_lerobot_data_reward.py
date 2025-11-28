"""
使用 Qwen2.5-VL-7B-Instruct 模型评估 lerobot v3.0 数据集的任务完成度

该脚本会：
1. 加载 lerobot v3.0 数据集
2. 对每个episode的视觉序列使用 Qwen2.5-VL-7B-Instruct 进行评估
3. 输出任务完成度分数（0-1）
"""

import argparse
import json
import os
import base64
import io
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. MP4 generation will be disabled.")
try:
    from transformers import AutoProcessor
    # 尝试导入 Qwen2.5-VL 的专用类
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        QWEN_MODEL_CLASS = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        try:
            # 如果上面的导入失败，尝试使用 AutoModel
            from transformers import AutoModel
            QWEN_MODEL_CLASS = AutoModel
        except ImportError:
            # 最后尝试 AutoModelForCausalLM
            from transformers import AutoModelForCausalLM
            QWEN_MODEL_CLASS = AutoModelForCausalLM
except ImportError:
    raise ImportError("transformers library is required. Install with: pip install transformers")
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """将 PyTorch tensor (C, H, W) 转换为 PIL Image"""
    # 确保 tensor 在 CPU 上
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 转换为 numpy array
    if tensor.dim() == 3:
        # (C, H, W) -> (H, W, C)
        arr = tensor.permute(1, 2, 0).numpy()
    else:
        arr = tensor.numpy()
    
    # 归一化到 [0, 255]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    
    # 确保值在有效范围内
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)


def load_model_and_processor(model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
    """
    加载 Qwen2.5-VL 模型和处理器
    """
    print(f"Loading model: {model_name}")
    
    # 检查 transformers 版本
    try:
        import transformers
        version = transformers.__version__
        print(f"Transformers version: {version}")
        # Qwen2.5-VL 需要较新的 transformers 版本
        from packaging import version as pkg_version
        if pkg_version.parse(version) < pkg_version.parse("4.40.0"):
            print("⚠️  警告: transformers 版本可能过旧，Qwen2.5-VL 需要 >= 4.40.0")
            print("   建议升级: pip install --upgrade transformers")
    except Exception:
        pass
    
    # 检查下载源
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    if "hf-mirror.com" in hf_endpoint:
        print(f"📡 使用 Hugging Face 镜像源: {hf_endpoint}")
    else:
        print(f"📡 使用 Hugging Face 官方源: {hf_endpoint}")
        print("   提示：如果在国内下载较慢，可以设置环境变量：")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
    
    try:
        # 加载处理器
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # 加载模型 - 尝试使用正确的模型类
        print("Loading model (this may take a while)...")
        try:
            # 首先尝试使用专用类
            model = QWEN_MODEL_CLASS.from_pretrained(
                model_name,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=device,
                trust_remote_code=True
            )
        except (ValueError, TypeError) as e:
            # 如果专用类失败，尝试使用 AutoModel
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
        print("\n可能的解决方案：")
        print("1. 升级 transformers 到最新版本（Qwen2.5-VL 需要较新版本）：")
        print("   pip install --upgrade transformers")
        print("   建议版本 >= 4.40.0")
        print("2. 如果使用国内网络，可以设置镜像：")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
        print("3. 检查 transformers 版本：")
        print("   python -c 'import transformers; print(transformers.__version__)'")
        print("4. 如果问题仍然存在，可能需要从源码安装 transformers：")
        print("   pip install git+https://github.com/huggingface/transformers.git")
        raise


def escape_html(text: str) -> str:
    """转义 HTML 特殊字符"""
    if text is None:
        return ""
    return (str(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


def generate_html_report(output_data: Dict[str, Any], html_path: str):
    """
    生成 HTML 可视化报告
    
    Args:
        output_data: 评估结果数据
        html_path: HTML 文件保存路径
    """
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen Reward Model 评估报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-card .label {{
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 8px;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .episodes {{
            padding: 30px;
        }}
        .episode-card {{
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            margin-bottom: 30px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .episode-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }}
        .episode-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .episode-header h2 {{
            font-size: 1.5em;
        }}
        .score-badge {{
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 1.3em;
            font-weight: bold;
        }}
        .score-badge.success {{
            background: rgba(40, 167, 69, 0.3);
        }}
        .score-badge.failed {{
            background: rgba(220, 53, 69, 0.3);
        }}
        .episode-content {{
            padding: 25px;
        }}
        .section {{
            margin-bottom: 25px;
        }}
        .section-title {{
            font-size: 1.2em;
            font-weight: bold;
            color: #495057;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
        }}
        .prompt-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            line-height: 1.6;
        }}
        .response-box {{
            background: #e7f3ff;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            line-height: 1.6;
        }}
        .error-box {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            color: #856404;
        }}
        .images-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .image-item {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            border: 2px solid #e9ecef;
            transition: transform 0.2s;
        }}
        .image-item:hover {{
            transform: scale(1.05);
            border-color: #667eea;
        }}
        .image-item img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            margin-bottom: 8px;
        }}
        .image-item .image-info {{
            font-size: 0.85em;
            color: #6c757d;
        }}
        .processed-text {{
            background: #f1f3f5;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            max-height: 200px;
            overflow-y: auto;
            word-break: break-all;
        }}
        .task-info {{
            background: #e7f5e7;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            margin-bottom: 20px;
        }}
        .task-info strong {{
            color: #155724;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Qwen Reward Model 评估报告</h1>
            <div class="subtitle">任务完成度评估结果可视化</div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="label">数据集路径</div>
                <div class="value" style="font-size: 0.9em; word-break: break-all;">{escape_html(output_data.get('dataset_path', 'N/A'))}</div>
            </div>
            <div class="stat-card">
                <div class="label">模型名称</div>
                <div class="value" style="font-size: 1em;">{escape_html(output_data.get('model_name', 'N/A'))}</div>
            </div>
            <div class="stat-card">
                <div class="label">总 Episodes</div>
                <div class="value">{output_data.get('num_episodes', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="label">成功评估</div>
                <div class="value" style="color: #28a745;">{output_data['statistics'].get('successful_episodes', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="label">失败评估</div>
                <div class="value" style="color: #dc3545;">{output_data['statistics'].get('failed_episodes', 0)}</div>
            </div>
"""
    
    # 添加统计信息
    stats = output_data['statistics']
    if stats.get('mean_score') is not None:
        html_content += f"""
            <div class="stat-card">
                <div class="label">平均分数</div>
                <div class="value">{stats['mean_score']:.3f}</div>
            </div>
            <div class="stat-card">
                <div class="label">标准差</div>
                <div class="value">{stats['std_score']:.3f}</div>
            </div>
            <div class="stat-card">
                <div class="label">最低分数</div>
                <div class="value">{stats['min_score']:.3f}</div>
            </div>
            <div class="stat-card">
                <div class="label">最高分数</div>
                <div class="value">{stats['max_score']:.3f}</div>
            </div>
"""
    
    html_content += """
        </div>
        
        <div class="episodes">
"""
    
    # 添加每个 episode 的结果
    for result in output_data.get('results', []):
        episode_idx = result.get('episode', 'N/A')
        score = result.get('score')
        response = result.get('response', 'N/A')
        prompt = result.get('prompt', 'N/A')
        processed_text = result.get('processed_text', '')
        images = result.get('images', [])
        task = result.get('task', 'N/A')
        error = result.get('error')
        
        # 确定分数显示样式
        if score is not None:
            score_class = "success" if score >= 0.7 else ("failed" if score < 0.3 else "")
            score_display = f"{score:.3f}"
        else:
            score_class = "failed"
            score_display = "失败"
        
        html_content += f"""
            <div class="episode-card">
                <div class="episode-header">
                    <h2>Episode {episode_idx}</h2>
                    <div class="score-badge {score_class}">{score_display}</div>
                </div>
                <div class="episode-content">
                    <div class="task-info">
                        <strong>任务描述：</strong>{escape_html(task)}
                    </div>
"""
        
        # 如果有错误，显示错误信息
        if error:
            html_content += f"""
                    <div class="section">
                        <div class="section-title">❌ 错误信息</div>
                        <div class="error-box">{escape_html(error)}</div>
                    </div>
"""
        
        # 显示提示文本
        html_content += f"""
                    <div class="section">
                        <div class="section-title">📝 提示文本 (Prompt)</div>
                        <div class="prompt-box">{escape_html(prompt)}</div>
                    </div>
"""
        
        # 显示处理后的文本（如果有）
        if processed_text:
            processed_text_escaped = escape_html(processed_text[:500])
            if len(processed_text) > 500:
                processed_text_escaped += "..."
            html_content += f"""
                    <div class="section">
                        <div class="section-title">🔤 处理后的文本 (包含图像占位符)</div>
                        <div class="processed-text">{processed_text_escaped}</div>
                    </div>
"""
        
        # 显示图像
        if images:
            html_content += f"""
                    <div class="section">
                        <div class="section-title">🖼️ 使用的图像 ({len(images)} 张)</div>
                        <div class="images-grid">
"""
            for img_info in images:
                idx = img_info.get('index', 0)
                size = img_info.get('size', [0, 0])
                thumbnail = img_info.get('thumbnail_base64', '')
                html_content += f"""
                            <div class="image-item">
                                <img src="{thumbnail}" alt="Image {idx}" />
                                <div class="image-info">
                                    <div>图像 #{idx}</div>
                                    <div>尺寸: {size[0]}×{size[1]}</div>
                                </div>
                            </div>
"""
            html_content += """
                        </div>
                    </div>
"""
        
        # 显示模型响应
        html_content += f"""
                    <div class="section">
                        <div class="section-title">💬 模型响应 (Response)</div>
                        <div class="response-box">{escape_html(response)}</div>
                    </div>
"""
        
        html_content += """
                </div>
            </div>
"""
    
    html_content += """
        </div>
    </div>
</body>
</html>
"""
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated: {html_path}")


def extract_score_from_response(response: str, verbose: bool = True) -> float:
    """从模型响应中提取 0-1 的分数"""
    if not response:
        print("Warning: Empty response")
        return 0.5
    
    # 清理响应文本，移除可能的标记和特殊字符
    response_clean = response.strip()
    
    # 尝试多种模式匹配，按优先级排序
    patterns = [
        # 优先匹配明确的分数格式
        r'分数[：:]\s*(\d+\.?\d*)',  # "分数：0.8"
        r'(\d+\.?\d*)\s*/\s*1',  # "0.8/1"
        r'(\d+\.?\d*)\s*分',  # "0.8分"
        r'(\d+\.?\d*)\s*out\s*of\s*1',  # "0.8 out of 1"
        # 匹配 0-1 之间的数字（更严格的模式）
        r'\b(0\.\d+)\b',  # "0.75" 这样的格式
        r'\b(1\.0)\b',  # "1.0"
        r'\b(0)\b',  # "0"
        r'\b(1)\b',  # "1"
        # 最后尝试匹配任何数字
        r'(\d+\.?\d*)',  # 任何数字
    ]
    
    # 首先尝试提取明确的分数
    for pattern in patterns:
        matches = re.findall(pattern, response_clean, re.IGNORECASE)
        if matches:
            try:
                score = float(matches[0])
                # 确保分数在 0-1 范围内
                if score > 1.0:
                    # 如果是大于1的数字，可能是百分比形式
                    if score <= 100:
                        score = score / 100.0
                    else:
                        score = 1.0
                score = max(0.0, min(1.0, score))
                # 减少输出，只在调试模式下打印
                # print(f"✅ Extracted score {score} using pattern: {pattern}")
                return score
            except ValueError:
                continue
    
    # 如果找不到数字，尝试根据关键词判断
    response_lower = response_clean.lower()
    if any(word in response_lower for word in ['完全成功', '完全完成', 'fully successful', 'completely successful', '100%', '100 percent']):
        # print("✅ Extracted score 1.0 from keywords")
        return 1.0
    elif any(word in response_lower for word in ['完全失败', '完全未完成', 'completely failed', 'no progress', '0%', '0 percent', 'failed']):
        # print("✅ Extracted score 0.0 from keywords")
        return 0.0
    elif any(word in response_lower for word in ['部分', 'partial', 'some progress', '50%', '50 percent']):
        # print("✅ Extracted score 0.5 from keywords")
        return 0.5
    
    # 默认返回 0.5，但打印警告
    if verbose:
        print(f"⚠️  Warning: Could not extract score from response. Using default 0.5")
        print(f"   Response: {response[:300]}")
    return 0.5


def evaluate_episode(
    model,  # 使用动态类型，因为可能是不同的模型类
    processor: AutoProcessor,
    images: List[Image.Image],
    task_description: str,
    prompt_template: str = None,
    device: str = "cuda",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    评估单个episode的任务完成度
    
    Args:
        model: Qwen2.5-VL 模型
        processor: 处理器
        images: 图像列表（视觉序列）
        task_description: 任务描述
        prompt_template: 提示模板（如果为None，使用默认模板）
        device: 设备
    
    Returns:
        包含评估结果的字典
    """
    # 如果 prompt_template 为 None，使用默认模板
    if prompt_template is None:
        prompt_template = """你是一个机器人任务评估器。请评估机器人执行任务的完成程度。

目标任务：{task_description}

下面是一系列按时间顺序排列的视觉图像，展示了机器人执行任务的过程。请仔细观察这些图像，判断任务完成的进度。

评估标准：
- 观察图像序列中的视觉变化
- 判断机器人是否在朝着目标前进
- 评估任务完成的百分比

请输出一个 0 到 1 之间的数字分数：
- 1.0 = 任务完全成功，目标已达成
- 0.8-0.9 = 任务基本完成，接近成功
- 0.5-0.7 = 任务部分完成，有一定进展
- 0.2-0.4 = 任务进展很小，刚开始
- 0.0-0.1 = 任务完全失败，没有进展

请只输出一个数字（例如：0.75），不要输出其他文字。"""
    
    # 构建完整的提示
    full_prompt = prompt_template.format(task_description=task_description)
    
    # 准备对话格式（Qwen2.5-VL 使用的格式）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img} for img in images
            ] + [
                {"type": "text", "text": full_prompt}
            ]
        }
    ]
    
    try:
        # 正确的方式：先使用 apply_chat_template 处理 messages，这会插入图像占位符
        # 然后使用 processor 处理文本和图像
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 使用 processor 处理文本和图像
        # processor 会自动识别文本中的图像占位符，并用实际的图像特征替换
        inputs = processor(
            text=text,
            images=images,  # 直接传入图像列表
            padding=True,
            return_tensors="pt"
        )
        
        # 移动到正确的设备
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        else:
            # 如果 inputs 是字典，手动移动每个tensor
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # 生成响应
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        # 解码响应 - 只解码新生成的部分
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
    except Exception as e:
        # 如果上面的方法失败，尝试备用方法
        print(f"Warning: Standard method failed ({e}), trying alternative method...")
        try:
            # 备用方法：尝试只传入第一张图像（可能多图像有问题）
            if len(images) > 1:
                print(f"Warning: Trying with single image instead of {len(images)} images...")
                # 只使用第一张和最后一张图像
                single_images = [images[0], images[-1]] if len(images) > 1 else images
                messages_single = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img} for img in single_images
                        ] + [
                            {"type": "text", "text": full_prompt}
                        ]
                    }
                ]
                text = processor.apply_chat_template(messages_single, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=text,
                    images=single_images,
                    padding=True,
                    return_tensors="pt"
                )
            else:
                # 如果只有一张图像，直接使用
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=text,
                    images=images,
                    padding=True,
                    return_tensors="pt"
                )
            
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
        except Exception as e2:
            print(f"❌ All methods failed. Last error: {e2}")
            import traceback
            traceback.print_exc()
            return {
                "score": None,
                "response": f"Error: {str(e2)}",
                "num_frames": len(images)
            }
    
    # 提取分数
    if verbose:
        print(f"\n🔍 Debug: Model raw response (first 500 chars): {response[:500]}")
        print(f"🔍 Debug: Number of images: {len(images)}")
    
    if response and not response.startswith("Error:"):
        score = extract_score_from_response(response, verbose=verbose)
        if verbose:
            print(f"🔍 Debug: Extracted score: {score}")
    else:
        # 如果有错误，不提取分数
        score = None
        if verbose:
            print(f"⚠️  Warning: Could not get valid response. Response: {response[:200]}")
    
    # 保存图像信息（包括缩略图）
    image_info = []
    for i, img in enumerate(images):
        # 创建缩略图（最大尺寸 224x224，保持宽高比）
        thumbnail_size = (224, 224)
        img_thumbnail = img.copy()
        img_thumbnail.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
        
        # 将缩略图转换为 base64 编码的字符串
        buffer = io.BytesIO()
        img_thumbnail.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        image_info.append({
            "index": i,
            "size": img.size,  # (width, height) 原始尺寸
            "mode": img.mode,  # RGB, etc.
            "format": img.format if hasattr(img, 'format') else None,
            "thumbnail_base64": f"data:image/png;base64,{img_base64}"  # base64 编码的缩略图
        })
    
    # 保存处理后的文本（包含图像占位符）
    processed_text = text if 'text' in locals() else None
    
    return {
        "score": score,
        "response": response,
        "num_frames": len(images),
        "prompt": full_prompt,  # 保存完整的提示文本
        "processed_text": processed_text,  # 保存处理后的文本（包含图像占位符）
        "images": image_info  # 保存图像元数据和缩略图
    }


def process_dataset(
    dataset_path: str,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    camera_key: str = "observation.images.cam_head",
    task_description: str = None,
    prompt_template: str = None,
    episode: int = None,
    img_start_frame: int = None,
    img_end_frame: int = None,
    output_path: str = None,
    device: str = "auto"
):
    """
    处理整个数据集，评估每个episode的任务完成度
    
    Args:
        dataset_path: 数据集路径
        model_name: 模型名称
        camera_key: 相机数据键名
        task_description: 任务描述（如果为None，从数据集元数据中读取）
        prompt_template: 提示模板
        episode: 要处理的episode索引（None表示处理所有episode）
        img_start_frame: 图像帧的起始索引（相对于episode，None表示从episode开始）
        img_end_frame: 图像帧的结束索引（相对于episode，None表示到episode结束）
        output_path: 输出结果保存路径
        device: 设备
    """
    # 默认提示模板
    if prompt_template is None:
        prompt_template = """你是一个机器人任务评估器。请评估机器人执行任务的完成程度。

目标任务：{task_description}

下面是一系列按时间顺序排列的视觉图像，展示了机器人执行任务的过程。请仔细观察这些图像，判断任务完成的进度。

评估标准：
- 观察图像序列中的视觉变化
- 判断机器人是否在朝着目标前进
- 评估任务完成的百分比

请输出一个 0 到 1 之间的数字分数：
- 1.0 = 任务完全成功，目标已达成
- 0.8-0.9 = 任务基本完成，接近成功
- 0.5-0.7 = 任务部分完成，有一定进展
- 0.2-0.4 = 任务进展很小，刚开始
- 0.0-0.1 = 任务完全失败，没有进展

请只输出一个数字（例如：0.75），不要输出其他文字。"""
    
    # 加载数据集
    print(f"Loading dataset from: {dataset_path}")
    dataset = LeRobotDataset(repo_id=0, root=dataset_path)
    
    print(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    print(f"Camera keys available: {dataset.meta.camera_keys}")
    
    # 检查相机键是否存在
    if camera_key not in dataset.meta.camera_keys:
        print(f"Warning: {camera_key} not found in camera keys. Available keys: {dataset.meta.camera_keys}")
        # 尝试使用第一个可用的相机
        if len(dataset.meta.camera_keys) > 0:
            camera_key = dataset.meta.camera_keys[0]
            print(f"Using {camera_key} instead")
        else:
            raise ValueError("No camera keys found in dataset")
    
    # 加载模型
    model, processor = load_model_and_processor(model_name, device)
    
    # 确定要处理的episode列表
    if episode is not None:
        # 处理指定的episode
        if episode < 0 or episode >= dataset.num_episodes:
            raise ValueError(f"Episode index {episode} is out of range [0, {dataset.num_episodes-1}]")
        episode_indices = [episode]
        print(f"Processing episode {episode} only")
    else:
        # 处理所有episode
        episode_indices = list(range(dataset.num_episodes))
        print(f"Processing all {dataset.num_episodes} episodes")
    
    # 获取任务描述
    if task_description is None:
        try:
            # 尝试从数据集元数据中读取任务
            if hasattr(dataset.meta, 'tasks') and len(dataset.meta.tasks) > 0:
                # tasks 是一个 DataFrame，任务名称在索引中
                if hasattr(dataset.meta.tasks, 'index') and len(dataset.meta.tasks.index) > 0:
                    # 使用索引中的第一个任务名称
                    task_description = dataset.meta.tasks.index[0]
                elif hasattr(dataset.meta.tasks, 'iloc'):
                    # 尝试从第一行获取
                    first_row = dataset.meta.tasks.iloc[0]
                    if hasattr(first_row, 'name'):
                        task_description = first_row.name
                    elif isinstance(first_row, dict) and 'task' in first_row:
                        task_description = first_row['task']
                    else:
                        task_description = "把灰色箱子拉开，并且机械臂双边夹爪把箱子抬起来"
                else:
                    task_description = "把灰色箱子拉开，并且机械臂双边夹爪把箱子抬起来"
                print(f"Using task description from dataset: {task_description}")
            else:
                # 使用用户提供的默认任务描述
                task_description = "把灰色箱子拉开，并且机械臂双边夹爪把箱子抬起来"
                print(f"Using default task description: {task_description}")
        except Exception as e:
            print(f"Warning: Failed to get task from dataset metadata: {e}")
            task_description = "把灰色箱子拉开，并且机械臂双边夹爪把箱子抬起来"
            print(f"Using default task description: {task_description}")
    
    # 存储结果
    results = []
    
    # 处理每个episode
    for episode_idx in tqdm(episode_indices, desc="Processing episodes"):
        try:
            # 获取episode的帧范围
            ep_meta = dataset.meta.episodes[episode_idx]
            ep_start = ep_meta["dataset_from_index"]
            ep_end = ep_meta["dataset_to_index"]
            
            # 获取任务描述（如果有多个任务）
            ep_task = task_description
            if hasattr(dataset.meta, 'tasks') and 'task_index' in ep_meta:
                try:
                    task_idx = ep_meta.get('task_index', 0)
                    if hasattr(dataset.meta.tasks, 'index') and task_idx < len(dataset.meta.tasks.index):
                        # 从索引获取任务名称
                        ep_task = dataset.meta.tasks.index[task_idx]
                    elif hasattr(dataset.meta.tasks, 'iloc'):
                        first_row = dataset.meta.tasks.iloc[task_idx]
                        if hasattr(first_row, 'name'):
                            ep_task = first_row.name
                        elif isinstance(first_row, dict) and 'task' in first_row:
                            ep_task = first_row['task']
                except Exception as e:
                    print(f"Warning: Failed to get task for episode {episode_idx}: {e}")
                    ep_task = task_description
            
            # 收集图像
            # 确定帧范围（相对于episode的起始帧）
            if img_start_frame is not None and img_end_frame is not None:
                # 使用指定的帧区间（相对于episode开始）
                if img_start_frame < 0 or img_end_frame < 0:
                    raise ValueError("img_start_frame and img_end_frame must be non-negative")
                if img_start_frame >= img_end_frame:
                    raise ValueError("img_start_frame must be less than img_end_frame")
                
                # 计算实际的帧索引
                actual_start = ep_start + img_start_frame
                actual_end = ep_start + img_end_frame
                
                # 确保不超出episode范围
                actual_start = max(ep_start, actual_start)
                actual_end = min(ep_end, actual_end)
                
                if actual_start >= actual_end:
                    raise ValueError(f"Invalid frame range: start={actual_start}, end={actual_end} (episode range: {ep_start}-{ep_end})")
                
                indices = range(actual_start, actual_end)
                print(f"  Using frames {img_start_frame} to {img_end_frame} (absolute: {actual_start} to {actual_end-1})")
            elif img_start_frame is not None:
                # 只指定了起始帧
                actual_start = ep_start + img_start_frame
                actual_start = max(ep_start, actual_start)
                actual_end = ep_end
                indices = range(actual_start, actual_end)
                print(f"  Using frames from {img_start_frame} to end (absolute: {actual_start} to {actual_end-1})")
            elif img_end_frame is not None:
                # 只指定了结束帧
                actual_start = ep_start
                actual_end = ep_start + img_end_frame
                actual_end = min(ep_end, actual_end)
                indices = range(actual_start, actual_end)
                print(f"  Using frames from start to {img_end_frame} (absolute: {actual_start} to {actual_end-1})")
            else:
                # 使用所有帧
                indices = range(ep_start, ep_end)
                print(f"  Using all frames (absolute: {ep_start} to {ep_end-1})")
            
            images = []
            for idx in indices:
                try:
                    frame_data = dataset[idx]
                    if camera_key in frame_data:
                        img_tensor = frame_data[camera_key]
                        # 转换为 PIL Image
                        img = tensor_to_pil_image(img_tensor)
                        images.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load frame {idx}: {e}")
                    continue
            
            if len(images) == 0:
                print(f"Warning: No images loaded for episode {episode_idx}")
                results.append({
                    "episode": episode_idx,
                    "score": None,
                    "error": "No images loaded",
                    "num_frames": 0,
                    "prompt": prompt_template.format(task_description=ep_task),
                    "images": []
                })
                continue
            
            # 评估episode
            result = evaluate_episode(
                model=model,
                processor=processor,
                images=images,
                task_description=ep_task,
                prompt_template=prompt_template,
                device=next(model.parameters()).device
            )
            
            result["episode"] = episode_idx
            result["task"] = ep_task
            results.append(result)
            
            if result['score'] is not None:
                print(f"Episode {episode_idx}: Score = {result['score']:.3f}, Frames = {result['num_frames']}")
                print(f"  Prompt: {result.get('prompt', 'N/A')[:100]}...")
                print(f"  Images used: {result.get('num_frames', 0)} images")
            else:
                print(f"Episode {episode_idx}: Failed to get score. Response: {result['response'][:100]}...")
                print(f"  Prompt: {result.get('prompt', 'N/A')[:100]}...")
                print(f"  Images used: {result.get('num_frames', 0)} images")
            
        except Exception as e:
            print(f"Error processing episode {episode_idx}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "episode": episode_idx,
                "score": None,
                "error": str(e),
                "num_frames": 0,
                "prompt": prompt_template.format(task_description=ep_task) if 'ep_task' in locals() else None,
                "images": []
            })
    
    # 保存结果
    if output_path is None:
        output_path = os.path.join(dataset_path, "reward_scores.json")
    
    output_data = {
        "dataset_path": dataset_path,
        "model_name": model_name,
        "camera_key": camera_key,
        "task_description": task_description,
        "num_episodes": len(results),
        "results": results,
        "statistics": {
            "mean_score": np.mean([r["score"] for r in results if r["score"] is not None]) if len([r for r in results if r["score"] is not None]) > 0 else None,
            "std_score": np.std([r["score"] for r in results if r["score"] is not None]) if len([r for r in results if r["score"] is not None]) > 1 else 0.0,
            "min_score": np.min([r["score"] for r in results if r["score"] is not None]) if len([r for r in results if r["score"] is not None]) > 0 else None,
            "max_score": np.max([r["score"] for r in results if r["score"] is not None]) if len([r for r in results if r["score"] is not None]) > 0 else None,
            "successful_episodes": len([r for r in results if r["score"] is not None]),
            "failed_episodes": len([r for r in results if r["score"] is None]),
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 生成 HTML 可视化报告
    html_path = output_path.replace('.json', '.html')
    generate_html_report(output_data, html_path)
    
    print(f"\nResults saved to: {output_path}")
    print(f"HTML report saved to: {html_path}")
    print(f"Statistics:")
    if output_data['statistics']['mean_score'] is not None:
        print(f"  Mean score: {output_data['statistics']['mean_score']:.3f}")
        print(f"  Std score: {output_data['statistics']['std_score']:.3f}")
        print(f"  Min score: {output_data['statistics']['min_score']:.3f}")
        print(f"  Max score: {output_data['statistics']['max_score']:.3f}")
    else:
        print("  ⚠️  No successful evaluations!")
    print(f"  Successful episodes: {output_data['statistics']['successful_episodes']}")
    print(f"  Failed episodes: {output_data['statistics']['failed_episodes']}")
    
    return results


def create_reward_visualization_video(
    images: List[Image.Image],
    scores: List[float],
    frame_indices: List[int],
    output_path: str,
    fps: int = 5,
    task_description: str = "Task"
) -> str:
    """
    创建包含图像序列和分数曲线的MP4视频
    
    Args:
        images: 图像列表
        scores: 对应的分数列表
        frame_indices: 帧索引列表
        output_path: 输出视频路径
        fps: 视频帧率
        task_description: 任务描述
    
    Returns:
        输出视频路径
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for video generation. Install with: pip install opencv-python")
    
    if len(images) != len(scores):
        raise ValueError(f"Number of images ({len(images)}) must match number of scores ({len(scores)})")
    
    # 创建临时目录保存帧
    temp_dir = Path(output_path).parent / "temp_frames"
    temp_dir.mkdir(exist_ok=True)
    
    # 准备分数曲线数据
    # 确保分数列表与图像对齐
    score_timeline = scores.copy()
    
    # 创建图形用于绘制分数曲线
    fig_width = 12
    fig_height = 6
    dpi = 100
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    
    print(f"Creating reward visualization video with {len(images)} frames...")
    
    try:
        for i, (img, score) in enumerate(tqdm(zip(images, scores), total=len(images), desc="Generating video frames")):
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
            fig.suptitle(f'Reward Visualization - Frame {frame_indices[i] if i < len(frame_indices) else i}', 
                        fontsize=14, fontweight='bold')
            
            # 左侧：显示当前图像
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title(f'Current Frame\nScore: {score:.3f}', fontsize=12, fontweight='bold')
            
            # 右侧：绘制分数曲线
            current_scores = score_timeline[:i+1]
            current_frames = frame_indices[:i+1] if len(frame_indices) > i else list(range(i+1))
            
            ax2.plot(current_frames, current_scores, 'b-', linewidth=2, label='Reward Score')
            ax2.scatter(current_frames[-1], current_scores[-1], color='red', s=100, zorder=5, 
                       label=f'Current: {current_scores[-1]:.3f}')
            ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.5)')
            ax2.fill_between(current_frames, current_scores, 0.5, where=[s >= 0.5 for s in current_scores], 
                           alpha=0.3, color='green', label='Above Baseline')
            ax2.fill_between(current_frames, current_scores, 0.5, where=[s < 0.5 for s in current_scores], 
                           alpha=0.3, color='red', label='Below Baseline')
            
            ax2.set_xlabel('Frame Index', fontsize=11)
            ax2.set_ylabel('Reward Score', fontsize=11)
            ax2.set_title('Reward Score Timeline', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best', fontsize=9)
            
            # 添加统计信息
            if len(current_scores) > 0:
                mean_score = np.mean(current_scores)
                max_score = np.max(current_scores)
                min_score = np.min(current_scores)
                stats_text = f'Mean: {mean_score:.3f}\nMax: {max_score:.3f}\nMin: {min_score:.3f}'
                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # 保存为图像
            frame_path = temp_dir / f"frame_{i:05d}.png"
            plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            # 读取保存的帧
            frame_img = cv2.imread(str(frame_path))
            if frame_img is None:
                print(f"Warning: Failed to read frame {i}")
                continue
            
            # 初始化视频写入器（使用第一帧的尺寸）
            if video_writer is None:
                height, width = frame_img.shape[:2]
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            video_writer.write(frame_img)
        
        if video_writer is not None:
            video_writer.release()
            print(f"✅ Video saved to: {output_path}")
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return output_path
        
    except Exception as e:
        if video_writer is not None:
            video_writer.release()
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


def evaluate_episode_with_sliding_window(
    model,
    processor: AutoProcessor,
    dataset,
    episode_idx: int,
    camera_key: str,
    task_description: str,
    prompt_template: str = None,
    window_size: int = 20,
    stride: int = 1,
    device: str = "cuda"
) -> Tuple[List[float], List[Image.Image], List[int]]:
    """
    使用滑动窗口评估episode，每次评估window_size帧
    
    Args:
        model: Qwen2.5-VL 模型
        processor: 处理器
        dataset: LeRobotDataset
        episode_idx: episode索引
        camera_key: 相机数据键名
        task_description: 任务描述
        prompt_template: 提示模板（如果为None，使用默认模板）
        window_size: 窗口大小（每次评估的帧数）
        stride: 滑动步长
        device: 设备
    
    Returns:
        (scores, images, frame_indices): 分数列表、图像列表、帧索引列表
    """
    # 如果 prompt_template 为 None，使用默认模板
    if prompt_template is None:
        prompt_template = """你是一个机器人任务评估器。请评估机器人执行任务的完成程度。

目标任务：{task_description}

下面是一系列按时间顺序排列的视觉图像，展示了机器人执行任务的过程。请仔细观察这些图像，判断任务完成的进度。

评估标准：
- 观察图像序列中的视觉变化
- 判断机器人是否在朝着目标前进
- 评估任务完成的百分比

请输出一个 0 到 1 之间的数字分数：
- 1.0 = 任务完全成功，目标已达成
- 0.8-0.9 = 任务基本完成，接近成功
- 0.5-0.7 = 任务部分完成，有一定进展
- 0.2-0.4 = 任务进展很小，刚开始
- 0.0-0.1 = 任务完全失败，没有进展

请只输出一个数字（例如：0.75），不要输出其他文字。"""
    
    # 获取episode的帧范围
    ep_meta = dataset.meta.episodes[episode_idx]
    ep_start = ep_meta["dataset_from_index"]
    ep_end = ep_meta["dataset_to_index"]
    num_frames = ep_end - ep_start
    
    print(f"Episode {episode_idx}: {num_frames} frames, using sliding window (size={window_size}, stride={stride})")
    
    scores = []
    all_images = []
    frame_indices = []
    
    # 计算需要处理的窗口数量
    num_windows = (num_frames - window_size) // stride + 1
    print(f"Total windows to process: {num_windows}")
    
    # 滑动窗口处理（使用 tqdm 显示进度）
    for window_idx, window_start in enumerate(tqdm(range(0, num_frames - window_size + 1, stride), 
                                                    desc=f"Sliding window evaluation", 
                                                    total=num_windows)):
        window_end = window_start + window_size
        window_indices = range(ep_start + window_start, ep_start + window_end)
        
        # 收集窗口内的图像
        window_images = []
        for idx in window_indices:
            try:
                frame_data = dataset[idx]
                if camera_key in frame_data:
                    img_tensor = frame_data[camera_key]
                    img = tensor_to_pil_image(img_tensor)
                    window_images.append(img)
            except Exception as e:
                print(f"Warning: Failed to load frame {idx}: {e}")
                continue
        
        if len(window_images) == 0:
            print(f"Warning: No images in window [{window_start}, {window_end})")
            continue
        
        # 评估当前窗口（减少调试输出，只在每10个窗口或最后一个窗口打印详细信息）
        verbose = (window_idx % 10 == 0) or (window_idx == num_windows - 1)
        
        result = evaluate_episode(
            model=model,
            processor=processor,
            images=window_images,
            task_description=task_description,
            prompt_template=prompt_template,
            device=device,
            verbose=verbose
        )
        
        score = result.get('score')
        if score is not None:
            scores.append(score)
            # 使用窗口的最后一帧作为代表图像
            all_images.append(window_images[-1])
            frame_indices.append(ep_start + window_end - 1)
            
            # 只在 verbose 模式下打印详细信息
            if verbose:
                print(f"Window [{window_start}-{window_end-1}]: Score = {score:.3f}")
        else:
            print(f"Warning: Failed to get score for window [{window_start}, {window_end})")
    
    print(f"\n✅ Completed sliding window evaluation: {len(scores)} scores collected")
    return scores, all_images, frame_indices


def main():
    parser = argparse.ArgumentParser(description="使用 Qwen2.5-VL 评估 lerobot 数据集的任务完成度")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1125_groot_train_data_with_task_filtered",
        help="数据集路径"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="模型名称"
    )
    parser.add_argument(
        "--camera_key",
        type=str,
        default="observation.images.cam_head",
        help="相机数据键名"
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default=None,
        help="任务描述（如果为None，从数据集元数据中读取）"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="要处理的episode索引（None表示处理所有episode）"
    )
    parser.add_argument(
        "--img_start_frame",
        type=int,
        default=None,
        help="图像帧的起始索引（相对于episode开始，None表示从episode开始）"
    )
    parser.add_argument(
        "--img_end_frame",
        type=int,
        default=None,
        help="图像帧的结束索引（相对于episode开始，None表示到episode结束）"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="输出结果保存路径（默认保存到数据集目录）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备（auto, cuda, cpu）"
    )
    parser.add_argument(
        "--sliding_window",
        action="store_true",
        help="使用滑动窗口模式（每次评估20帧，生成reward曲线视频）"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="滑动窗口大小（默认20帧）"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="滑动窗口步长（默认1帧）"
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=5,
        help="输出视频的帧率（默认5fps）"
    )
    
    args = parser.parse_args()
    
    # 执行处理
    if args.sliding_window:
        # 滑动窗口模式：生成reward可视化视频
        if args.episode is None:
            raise ValueError("--episode is required when using --sliding_window mode")
        
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for video generation. Install with: pip install opencv-python")
        
        # 加载数据集
        print(f"Loading dataset from: {args.dataset_path}")
        dataset = LeRobotDataset(repo_id=0, root=args.dataset_path)
        
        # 检查episode范围
        if args.episode < 0 or args.episode >= dataset.num_episodes:
            raise ValueError(f"Episode index {args.episode} is out of range [0, {dataset.num_episodes-1}]")
        
        # 获取任务描述
        task_description = args.task_description
        if task_description is None:
            try:
                if hasattr(dataset.meta, 'tasks') and len(dataset.meta.tasks) > 0:
                    # tasks 是一个 DataFrame，任务名称在索引中
                    if hasattr(dataset.meta.tasks, 'iloc'):
                        # 尝试从索引获取任务名称
                        task_description = dataset.meta.tasks.index[0]
                    elif hasattr(dataset.meta.tasks, 'index') and len(dataset.meta.tasks.index) > 0:
                        task_description = dataset.meta.tasks.index[0]
                    else:
                        # 如果索引不可用，尝试从第一行获取
                        first_row = dataset.meta.tasks.iloc[0] if hasattr(dataset.meta.tasks, 'iloc') else dataset.meta.tasks[0]
                        if hasattr(first_row, 'name'):
                            task_description = first_row.name
                        elif isinstance(first_row, dict) and 'task' in first_row:
                            task_description = first_row['task']
                        else:
                            task_description = "把灰色箱子拉开，并且机械臂双边夹爪把箱子抬起来"
                else:
                    task_description = "把灰色箱子拉开，并且机械臂双边夹爪把箱子抬起来"
            except Exception as e:
                print(f"Warning: Failed to get task from dataset metadata: {e}")
                task_description = "把灰色箱子拉开，并且机械臂双边夹爪把箱子抬起来"
        
        print(f"Using task description: {task_description}")
        
        # 加载模型
        model, processor = load_model_and_processor(args.model_name, args.device)
        
        # 使用滑动窗口评估
        scores, images, frame_indices = evaluate_episode_with_sliding_window(
            model=model,
            processor=processor,
            dataset=dataset,
            episode_idx=args.episode,
            camera_key=args.camera_key,
            task_description=task_description,
            prompt_template=None,
            window_size=args.window_size,
            stride=args.stride,
            device=next(model.parameters()).device
        )
        
        # 生成视频
        if args.output_path is None:
            video_path = f"./reward_episode_{args.episode}_visualization.mp4"
        else:
            video_path = args.output_path.replace('.json', '.mp4')
        
        create_reward_visualization_video(
            images=images,
            scores=scores,
            frame_indices=frame_indices,
            output_path=video_path,
            fps=args.video_fps,
            task_description=task_description
        )
        
        # 保存分数数据
        json_path = video_path.replace('.mp4', '_scores.json')
        score_data = {
            "episode": args.episode,
            "task_description": task_description,
            "window_size": args.window_size,
            "stride": args.stride,
            "num_windows": len(scores),
            "scores": scores,
            "frame_indices": frame_indices,
            "statistics": {
                "mean_score": np.mean(scores) if len(scores) > 0 else None,
                "std_score": np.std(scores) if len(scores) > 1 else 0.0,
                "min_score": np.min(scores) if len(scores) > 0 else None,
                "max_score": np.max(scores) if len(scores) > 0 else None,
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(score_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Reward visualization video saved to: {video_path}")
        print(f"✅ Score data saved to: {json_path}")
        print(f"Statistics:")
        if score_data['statistics']['mean_score'] is not None:
            print(f"  Mean score: {score_data['statistics']['mean_score']:.3f}")
            print(f"  Std score: {score_data['statistics']['std_score']:.3f}")
            print(f"  Min score: {score_data['statistics']['min_score']:.3f}")
            print(f"  Max score: {score_data['statistics']['max_score']:.3f}")
    else:
        # 正常模式
        process_dataset(
            dataset_path=args.dataset_path,
            model_name=args.model_name,
            camera_key=args.camera_key,
            task_description=args.task_description,
            episode=args.episode,
            img_start_frame=args.img_start_frame,
            img_end_frame=args.img_end_frame,
            output_path=args.output_path,
            device=args.device
        )


if __name__ == "__main__":
    main()
