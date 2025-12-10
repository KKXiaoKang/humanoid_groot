#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    3路奥比中光相机CPU负载测试
    功能：
    1. 三个阶段测试：
       - 第一阶段：静置设备，不启动相机
       - 第二阶段：启动相机获取视频流但不推流
       - 第三阶段：启动相机并推流（使用FFmpeg进行CPU编码推流）
    2. 实时监控CPU使用率（整体和单核）
    3. 实时监控CPU频率和降频检测（检测推流时是否发生CPU降频）
    4. 对比三个阶段CPU负载差异
    5. 记录CPU负载和频率数据并生成详细报告
    6. 降频检测：当CPU频率低于最大频率的95%时，判定为降频
    
    推流配置：
    - 使用FFmpeg进行推流，采用CPU编码（libx264）以增加CPU负载
    - 推流类型：UDP（默认，推流到本地回环）、RTMP、NULL（编码但不保存，用于CPU负载测试）
    - 可通过stream_config配置推流参数（编码预设、质量等）
"""

import rospy
import cv2
import numpy as np
import psutil
import threading
import time
import json
import subprocess
import signal
from datetime import datetime
from collections import deque
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os


class CameraStreamTester:
    def __init__(self):
        """初始化测试器"""
        rospy.init_node('camera_cpu_load_test', anonymous=True)
        
        # 相机话题配置
        self.camera_topics = {
            'camera': '/camera/color/image_raw',
            'left_cam': '/left_cam/color/image_raw',
            'right_cam': '/right_cam/color/image_raw'
        }
        
        # 推流配置
        self.stream_config = {
            'use_ffmpeg': True,  # 使用ffmpeg推流
            'stream_type': 'udp',  # 'udp', 'rtmp', 'null' (null表示丢弃但编码，用于测试CPU负载)
            'udp_port_base': 5000,  # UDP推流端口基址（camera:5000, left_cam:5001, right_cam:5002）
            'rtmp_url': None,  # RTMP推流地址（如果使用RTMP）
            'save_video': True,  # 是否保存视频文件
            'video_dir': './test_videos',
            'ffmpeg_preset': 'ultrafast',  # x264编码预设：ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
            'ffmpeg_crf': 23,  # 视频质量（18-28，越小质量越好但CPU负载越高）
            'fps': 30  # 推流帧率
        }
        
        # 初始化
        self.bridge = CvBridge()
        self.running = False
        
        # 图像缓存（线程安全）
        self.image_lock = threading.Lock()
        self.latest_images = {}
        
        # 相机启动配置
        self.camera_launch_cmd = [
            'bash', '-c',
            'cd /home/kuavo/kuavo_ros_application && roslaunch dynamic_biped sensor_orb_and_rs.launch'
        ]
        self.camera_process = None
        
        # CPU监控数据（三个阶段）
        self.idle_cpu_data = deque(maxlen=1000)      # 第一阶段：静置数据
        self.baseline_cpu_data = deque(maxlen=1000)   # 第二阶段：相机但不推流数据
        self.streaming_cpu_data = deque(maxlen=2000)  # 第三阶段：推流数据
        self.monitoring = False
        self.monitor_thread = None
        
        # 视频写入器
        self.video_writers = {}
        self.frame_counts = {}
        
        # FFmpeg推流进程
        self.ffmpeg_processes = {}
        self.ffmpeg_stdin_pipes = {}
        
        # CPU核心数量
        self.cpu_count = psutil.cpu_count()
        
        # CPU频率监控相关
        self.cpu_freq_available = self._check_cpu_freq_available()
        self.cpu_max_freqs = self._get_cpu_max_freqs()  # 每个核心的最大频率（Hz）
        self.cpu_min_freqs = self._get_cpu_min_freqs()  # 每个核心的最小频率（Hz）
        
        # 第一阶段统计（静置）
        self.idle_stats = {
            'avg_cpu': 0.0,
            'max_cpu': 0.0,
            'min_cpu': 100.0,
            'per_core_avg': [0.0] * self.cpu_count,
            'per_core_max': [0.0] * self.cpu_count,
            'per_core_freq_avg': [0.0] * self.cpu_count,  # 平均频率（MHz）
            'per_core_freq_min': [float('inf')] * self.cpu_count,  # 最小频率（MHz）
            'per_core_freq_max': [0.0] * self.cpu_count,  # 最大频率（MHz）
            'throttle_count': 0,  # 降频次数
            'samples': 0
        }
        
        # 第二阶段统计（相机但不推流）
        self.baseline_stats = {
            'avg_cpu': 0.0,
            'max_cpu': 0.0,
            'min_cpu': 100.0,
            'per_core_avg': [0.0] * self.cpu_count,
            'per_core_max': [0.0] * self.cpu_count,
            'per_core_freq_avg': [0.0] * self.cpu_count,
            'per_core_freq_min': [float('inf')] * self.cpu_count,
            'per_core_freq_max': [0.0] * self.cpu_count,
            'throttle_count': 0,
            'samples': 0
        }
        
        # 第三阶段统计（推流）
        self.streaming_stats = {
            'start_time': None,
            'end_time': None,
            'total_frames': {},
            'avg_cpu': 0.0,
            'max_cpu': 0.0,
            'min_cpu': 100.0,
            'per_core_avg': [0.0] * self.cpu_count,
            'per_core_max': [0.0] * self.cpu_count,
            'per_core_min': [100.0] * self.cpu_count,
            'per_core_freq_avg': [0.0] * self.cpu_count,
            'per_core_freq_min': [float('inf')] * self.cpu_count,
            'per_core_freq_max': [0.0] * self.cpu_count,
            'throttle_count': 0,  # 降频次数
            'throttle_events': [],  # 降频事件记录
            'cpu_samples': 0
        }
        
        # 创建视频保存目录
        if self.stream_config['save_video']:
            os.makedirs(self.stream_config['video_dir'], exist_ok=True)
        
        # 检查FFmpeg是否可用
        if self.stream_config.get('use_ffmpeg', False):
            if not self._check_ffmpeg_available():
                rospy.logwarn("FFmpeg不可用，推流功能将被禁用")
                self.stream_config['use_ffmpeg'] = False
            else:
                rospy.loginfo("FFmpeg可用，推流功能已启用")
    
    def _check_ffmpeg_available(self):
        """检查FFmpeg是否可用"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False
    
    def _check_cpu_freq_available(self):
        """检查CPU频率信息是否可用"""
        try:
            freq_file = f"/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
            if os.path.exists(freq_file):
                return True
        except:
            pass
        return False
    
    def _get_cpu_max_freqs(self):
        """获取每个CPU核心的最大频率（Hz）"""
        max_freqs = []
        for i in range(self.cpu_count):
            try:
                freq_file = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_max_freq"
                if os.path.exists(freq_file):
                    with open(freq_file, 'r') as f:
                        max_freqs.append(int(f.read().strip()))
                else:
                    max_freqs.append(0)
            except:
                max_freqs.append(0)
        return max_freqs
    
    def _get_cpu_min_freqs(self):
        """获取每个CPU核心的最小频率（Hz）"""
        min_freqs = []
        for i in range(self.cpu_count):
            try:
                freq_file = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_min_freq"
                if os.path.exists(freq_file):
                    with open(freq_file, 'r') as f:
                        min_freqs.append(int(f.read().strip()))
                else:
                    min_freqs.append(0)
            except:
                min_freqs.append(0)
        return min_freqs
    
    def _get_cpu_current_freqs(self):
        """获取每个CPU核心的当前频率（Hz），返回列表"""
        freqs = []
        for i in range(self.cpu_count):
            try:
                freq_file = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq"
                if os.path.exists(freq_file):
                    with open(freq_file, 'r') as f:
                        freqs.append(int(f.read().strip()))
                else:
                    freqs.append(0)
            except:
                freqs.append(0)
        return freqs
    
    def _check_throttling(self, current_freqs):
        """检查是否有降频发生
        返回: (是否降频, 降频的核心列表)
        """
        throttled_cores = []
        for i, (current_freq, max_freq) in enumerate(zip(current_freqs, self.cpu_max_freqs)):
            if max_freq > 0 and current_freq < max_freq * 0.95:  # 如果当前频率低于最大频率的95%，认为降频
                throttled_cores.append(i)
        return len(throttled_cores) > 0, throttled_cores
    
    def image_callback(self, topic_name, msg):
        """图像回调函数"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.image_lock:
                self.latest_images[topic_name] = cv_image
                if topic_name not in self.frame_counts:
                    self.frame_counts[topic_name] = 0
                self.frame_counts[topic_name] += 1
            
            # 推流或保存视频
            if self.running:
                self.process_frame(topic_name, cv_image)
        
        except Exception as e:
            rospy.logerr(f"Error processing image from {topic_name}: {str(e)}")
    
    def _start_ffmpeg_stream(self, topic_name, width, height):
        """启动FFmpeg推流进程"""
        if topic_name in self.ffmpeg_processes:
            return  # 已经启动
        
        stream_type = self.stream_config.get('stream_type', 'udp')
        fps = self.stream_config.get('fps', 30)
        preset = self.stream_config.get('ffmpeg_preset', 'ultrafast')
        crf = self.stream_config.get('ffmpeg_crf', 23)
        
        # 构建FFmpeg命令
        if stream_type == 'udp':
            # UDP推流到本地回环
            port = self.stream_config['udp_port_base']
            if topic_name == 'camera':
                port = self.stream_config['udp_port_base']
            elif topic_name == 'left_cam':
                port = self.stream_config['udp_port_base'] + 1
            elif topic_name == 'right_cam':
                port = self.stream_config['udp_port_base'] + 2
            
            output_url = f"udp://127.0.0.1:{port}"
            rospy.loginfo(f"启动FFmpeg推流 ({topic_name}): UDP -> {output_url}")
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'bgr24',
                '-r', str(fps),
                '-i', '-',  # 从stdin读取
                '-an',  # 无音频
                '-vcodec', 'libx264',  # 使用CPU编码（x264）增加CPU负载
                '-preset', preset,
                '-crf', str(crf),
                '-tune', 'zerolatency',  # 低延迟
                '-f', 'mpegts',
                output_url
            ]
            
        elif stream_type == 'rtmp':
            # RTMP推流
            rtmp_url = self.stream_config.get('rtmp_url', f'rtmp://localhost:1935/live/{topic_name}')
            rospy.loginfo(f"启动FFmpeg推流 ({topic_name}): RTMP -> {rtmp_url}")
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'bgr24',
                '-r', str(fps),
                '-i', '-',
                '-an',
                '-vcodec', 'libx264',
                '-preset', preset,
                '-crf', str(crf),
                '-tune', 'zerolatency',
                '-f', 'flv',
                rtmp_url
            ]
            
        elif stream_type == 'null':
            # 推流到null（丢弃数据，但进行编码，用于测试CPU负载）
            rospy.loginfo(f"启动FFmpeg推流 ({topic_name}): NULL（编码但不保存，用于CPU负载测试）")
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'bgr24',
                '-r', str(fps),
                '-i', '-',
                '-an',
                '-vcodec', 'libx264',
                '-preset', preset,
                '-crf', str(crf),
                '-f', 'null',
                '-'
            ]
        else:
            rospy.logerr(f"不支持的推流类型: {stream_type}")
            return
        
        try:
            # 启动FFmpeg进程
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            self.ffmpeg_processes[topic_name] = process
            self.ffmpeg_stdin_pipes[topic_name] = process.stdin
            
            rospy.loginfo(f"FFmpeg进程已启动 ({topic_name}), PID: {process.pid}")
            
            # 检查进程是否正常启动
            time.sleep(0.5)
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                rospy.logerr(f"FFmpeg进程意外退出 ({topic_name})")
                rospy.logerr(f"stderr: {stderr.decode()}")
                del self.ffmpeg_processes[topic_name]
                if topic_name in self.ffmpeg_stdin_pipes:
                    del self.ffmpeg_stdin_pipes[topic_name]
                return False
            
            return True
            
        except Exception as e:
            rospy.logerr(f"启动FFmpeg失败 ({topic_name}): {str(e)}")
            return False
    
    def _stop_ffmpeg_stream(self, topic_name):
        """停止FFmpeg推流进程"""
        if topic_name not in self.ffmpeg_processes:
            return
        
        try:
            process = self.ffmpeg_processes[topic_name]
            
            # 关闭stdin
            if topic_name in self.ffmpeg_stdin_pipes:
                try:
                    self.ffmpeg_stdin_pipes[topic_name].close()
                except:
                    pass
                del self.ffmpeg_stdin_pipes[topic_name]
            
            # 等待进程结束（最多5秒）
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 强制终止
                process.kill()
                process.wait()
            
            del self.ffmpeg_processes[topic_name]
            rospy.loginfo(f"FFmpeg推流已停止 ({topic_name})")
            
        except Exception as e:
            rospy.logerr(f"停止FFmpeg推流失败 ({topic_name}): {str(e)}")
    
    def process_frame(self, topic_name, frame):
        """处理帧：推流或保存"""
        height, width = frame.shape[:2]
        
        # FFmpeg推流
        if self.stream_config.get('use_ffmpeg', False):
            if topic_name not in self.ffmpeg_processes:
                if not self._start_ffmpeg_stream(topic_name, width, height):
                    rospy.logwarn(f"FFmpeg推流启动失败 ({topic_name})，跳过推流")
                    return
            
            # 写入帧到FFmpeg stdin
            if topic_name in self.ffmpeg_stdin_pipes:
                try:
                    # 确保帧数据是连续的（numpy数组）
                    frame_contiguous = np.ascontiguousarray(frame)
                    self.ffmpeg_stdin_pipes[topic_name].write(frame_contiguous.tobytes())
                except BrokenPipeError:
                    rospy.logwarn(f"FFmpeg管道已断开 ({topic_name})，尝试重启...")
                    self._stop_ffmpeg_stream(topic_name)
                    if self.running:  # 如果还在运行，尝试重启
                        self._start_ffmpeg_stream(topic_name, width, height)
                except Exception as e:
                    rospy.logerr(f"写入FFmpeg失败 ({topic_name}): {str(e)}")
        
        # 保存视频文件
        if self.stream_config['save_video']:
            if topic_name not in self.video_writers:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = os.path.join(
                    self.stream_config['video_dir'],
                    f"{topic_name}_{timestamp}.mp4"
                )
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writers[topic_name] = cv2.VideoWriter(
                    video_path, fourcc, 30.0, (width, height)
                )
                rospy.loginfo(f"Started recording: {video_path}")
            
            if topic_name in self.video_writers:
                self.video_writers[topic_name].write(frame)
    
    def monitor_cpu(self, phase='idle'):
        """CPU监控线程
        phase: 'idle' - 第一阶段静置, 'baseline' - 第二阶段相机不推流, 'streaming' - 第三阶段推流
        """
        phase_names = {
            'idle': '第一阶段-静置',
            'baseline': '第二阶段-相机不推流',
            'streaming': '第三阶段-推流'
        }
        phase_name = phase_names.get(phase, phase)
        rospy.loginfo(f"CPU监控已启动 ({phase_name})")
        if self.cpu_freq_available:
            rospy.loginfo(f"CPU频率监控已启用")
        else:
            rospy.logwarn(f"CPU频率监控不可用（可能没有权限或系统不支持）")
        
        while self.monitoring:
            # 获取CPU使用率（所有核心的平均值）
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 获取每个核心的使用率
            cpu_per_core = psutil.cpu_percent(percpu=True, interval=0.1)
            
            # 获取内存使用率
            memory = psutil.virtual_memory()
            
            # 获取CPU频率
            cpu_freqs = []
            is_throttling = False
            throttled_cores = []
            if self.cpu_freq_available:
                cpu_freqs = self._get_cpu_current_freqs()
                is_throttling, throttled_cores = self._check_throttling(cpu_freqs)
            
            timestamp = time.time()
            
            cpu_info = {
                'timestamp': timestamp,
                'cpu_percent': cpu_percent,
                'cpu_per_core': cpu_per_core,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'cpu_freqs_mhz': [f / 1000.0 for f in cpu_freqs] if cpu_freqs else [],
                'is_throttling': is_throttling,
                'throttled_cores': throttled_cores
            }
            
            # 保存到对应的数据队列并更新统计
            if phase == 'idle':
                self.idle_cpu_data.append(cpu_info)
                self._update_idle_stats(cpu_percent, cpu_per_core, cpu_freqs, is_throttling, throttled_cores, timestamp)
            elif phase == 'baseline':
                self.baseline_cpu_data.append(cpu_info)
                self._update_baseline_stats(cpu_percent, cpu_per_core, cpu_freqs, is_throttling, throttled_cores, timestamp)
            elif phase == 'streaming':
                self.streaming_cpu_data.append(cpu_info)
                self._update_streaming_stats(cpu_percent, cpu_per_core, cpu_freqs, is_throttling, throttled_cores, timestamp)
            
            # 实时打印
            core_str = " | ".join([f"C{i}:{c:.1f}%" for i, c in enumerate(cpu_per_core)])
            freq_str = ""
            if self.cpu_freq_available and cpu_freqs:
                freq_str = " | " + " | ".join([f"F{i}:{f/1000:.0f}MHz" for i, f in enumerate(cpu_freqs)])
            
            throttle_warning = ""
            if is_throttling:
                throttle_warning = f" | ⚠降频: 核心{throttled_cores}"
            
            frame_info = ""
            if phase in ['baseline', 'streaming']:
                frame_info = f" | 帧数 - camera: {self.frame_counts.get('camera', 0)}, " \
                           f"left: {self.frame_counts.get('left_cam', 0)}, " \
                           f"right: {self.frame_counts.get('right_cam', 0)}"
            
            rospy.loginfo(
                f"[{phase_name}] CPU总: {cpu_percent:.2f}% | "
                f"内存: {memory.percent:.2f}% | "
                f"各核心: {core_str}{freq_str}{throttle_warning}{frame_info}"
            )
            
            time.sleep(0.5)  # 每0.5秒采样一次
    
    def _update_idle_stats(self, cpu_percent, cpu_per_core, cpu_freqs, is_throttling, throttled_cores, timestamp):
        """更新第一阶段（静置）的统计信息"""
        self.idle_stats['avg_cpu'] = (
            (self.idle_stats['avg_cpu'] * self.idle_stats['samples'] + cpu_percent) /
            (self.idle_stats['samples'] + 1)
        )
        self.idle_stats['samples'] += 1
        self.idle_stats['max_cpu'] = max(self.idle_stats['max_cpu'], cpu_percent)
        self.idle_stats['min_cpu'] = min(self.idle_stats['min_cpu'], cpu_percent)
        
        # 更新单核CPU统计
        for i, core_percent in enumerate(cpu_per_core):
            if i < len(self.idle_stats['per_core_avg']):
                self.idle_stats['per_core_avg'][i] = (
                    (self.idle_stats['per_core_avg'][i] * (self.idle_stats['samples'] - 1) + core_percent) /
                    self.idle_stats['samples']
                )
                self.idle_stats['per_core_max'][i] = max(
                    self.idle_stats['per_core_max'][i], core_percent
                )
        
        # 更新频率统计
        if cpu_freqs:
            for i, freq_hz in enumerate(cpu_freqs):
                if i < len(self.idle_stats['per_core_freq_avg']):
                    freq_mhz = freq_hz / 1000.0
                    self.idle_stats['per_core_freq_avg'][i] = (
                        (self.idle_stats['per_core_freq_avg'][i] * (self.idle_stats['samples'] - 1) + freq_mhz) /
                        self.idle_stats['samples']
                    )
                    self.idle_stats['per_core_freq_min'][i] = min(
                        self.idle_stats['per_core_freq_min'][i], freq_mhz
                    )
                    self.idle_stats['per_core_freq_max'][i] = max(
                        self.idle_stats['per_core_freq_max'][i], freq_mhz
                    )
        
        # 记录降频事件
        if is_throttling:
            self.idle_stats['throttle_count'] += 1
    
    def _update_streaming_stats(self, cpu_percent, cpu_per_core, cpu_freqs, is_throttling, throttled_cores, timestamp):
        """更新第三阶段（推流）的统计信息"""
        # 更新整体CPU统计
        self.streaming_stats['avg_cpu'] = (
            (self.streaming_stats['avg_cpu'] * self.streaming_stats['cpu_samples'] + cpu_percent) /
            (self.streaming_stats['cpu_samples'] + 1)
        )
        self.streaming_stats['cpu_samples'] += 1
        self.streaming_stats['max_cpu'] = max(self.streaming_stats['max_cpu'], cpu_percent)
        self.streaming_stats['min_cpu'] = min(self.streaming_stats['min_cpu'], cpu_percent)
        
        # 更新单核CPU统计
        for i, core_percent in enumerate(cpu_per_core):
            if i < len(self.streaming_stats['per_core_avg']):
                self.streaming_stats['per_core_avg'][i] = (
                    (self.streaming_stats['per_core_avg'][i] * (self.streaming_stats['cpu_samples'] - 1) + core_percent) /
                    self.streaming_stats['cpu_samples']
                )
                self.streaming_stats['per_core_max'][i] = max(
                    self.streaming_stats['per_core_max'][i], core_percent
                )
                self.streaming_stats['per_core_min'][i] = min(
                    self.streaming_stats['per_core_min'][i], core_percent
                )
        
        # 更新频率统计
        if cpu_freqs:
            for i, freq_hz in enumerate(cpu_freqs):
                if i < len(self.streaming_stats['per_core_freq_avg']):
                    freq_mhz = freq_hz / 1000.0
                    self.streaming_stats['per_core_freq_avg'][i] = (
                        (self.streaming_stats['per_core_freq_avg'][i] * (self.streaming_stats['cpu_samples'] - 1) + freq_mhz) /
                        self.streaming_stats['cpu_samples']
                    )
                    self.streaming_stats['per_core_freq_min'][i] = min(
                        self.streaming_stats['per_core_freq_min'][i], freq_mhz
                    )
                    self.streaming_stats['per_core_freq_max'][i] = max(
                        self.streaming_stats['per_core_freq_max'][i], freq_mhz
                    )
        
        # 记录降频事件
        if is_throttling:
            self.streaming_stats['throttle_count'] += 1
            self.streaming_stats['throttle_events'].append({
                'timestamp': timestamp,
                'throttled_cores': throttled_cores,
                'cpu_percent': cpu_percent,
                'cpu_freqs_mhz': [f / 1000.0 for f in cpu_freqs] if cpu_freqs else []
            })
    
    def _update_baseline_stats(self, cpu_percent, cpu_per_core, cpu_freqs, is_throttling, throttled_cores, timestamp):
        """更新基准测试的统计信息"""
        # 更新整体CPU统计
        self.baseline_stats['avg_cpu'] = (
            (self.baseline_stats['avg_cpu'] * self.baseline_stats['samples'] + cpu_percent) /
            (self.baseline_stats['samples'] + 1)
        )
        self.baseline_stats['samples'] += 1
        self.baseline_stats['max_cpu'] = max(self.baseline_stats['max_cpu'], cpu_percent)
        self.baseline_stats['min_cpu'] = min(self.baseline_stats['min_cpu'], cpu_percent)
        
        # 更新单核CPU统计
        for i, core_percent in enumerate(cpu_per_core):
            if i < len(self.baseline_stats['per_core_avg']):
                self.baseline_stats['per_core_avg'][i] = (
                    (self.baseline_stats['per_core_avg'][i] * (self.baseline_stats['samples'] - 1) + core_percent) /
                    self.baseline_stats['samples']
                )
                self.baseline_stats['per_core_max'][i] = max(
                    self.baseline_stats['per_core_max'][i], core_percent
                )
        
        # 更新频率统计
        if cpu_freqs:
            for i, freq_hz in enumerate(cpu_freqs):
                if i < len(self.baseline_stats['per_core_freq_avg']):
                    freq_mhz = freq_hz / 1000.0
                    self.baseline_stats['per_core_freq_avg'][i] = (
                        (self.baseline_stats['per_core_freq_avg'][i] * (self.baseline_stats['samples'] - 1) + freq_mhz) /
                        self.baseline_stats['samples']
                    )
                    self.baseline_stats['per_core_freq_min'][i] = min(
                        self.baseline_stats['per_core_freq_min'][i], freq_mhz
                    )
                    self.baseline_stats['per_core_freq_max'][i] = max(
                        self.baseline_stats['per_core_freq_max'][i], freq_mhz
                    )
        
        # 记录降频事件
        if is_throttling:
            self.baseline_stats['throttle_count'] += 1
    
    def start_camera(self):
        """启动相机进程"""
        if self.camera_process is not None:
            rospy.logwarn("相机进程已存在，跳过启动")
            return
        
        rospy.loginfo("正在启动相机...")
        rospy.loginfo(f"执行命令: {' '.join(self.camera_launch_cmd)}")
        
        try:
            self.camera_process = subprocess.Popen(
                self.camera_launch_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # 创建新的进程组
            )
            rospy.loginfo(f"相机进程已启动，PID: {self.camera_process.pid}")
            
            # 等待相机启动（给一些时间让ROS节点启动）
            rospy.loginfo("等待相机初始化...")
            rospy.sleep(10)  # 等待10秒让相机节点完全启动
            
            # 检查进程是否还在运行
            if self.camera_process.poll() is not None:
                stdout, stderr = self.camera_process.communicate()
                rospy.logerr(f"相机进程意外退出")
                rospy.logerr(f"stdout: {stdout.decode()}")
                rospy.logerr(f"stderr: {stderr.decode()}")
                self.camera_process = None
                return False
            
            rospy.loginfo("相机启动成功")
            return True
            
        except Exception as e:
            rospy.logerr(f"启动相机失败: {str(e)}")
            self.camera_process = None
            return False
    
    def stop_camera(self):
        """停止相机进程"""
        if self.camera_process is None:
            rospy.logwarn("相机进程不存在，跳过停止")
            return
        
        rospy.loginfo(f"正在停止相机进程 (PID: {self.camera_process.pid})...")
        
        try:
            # 发送SIGTERM信号给整个进程组
            os.killpg(os.getpgid(self.camera_process.pid), signal.SIGTERM)
            
            # 等待进程结束（最多等待10秒）
            try:
                self.camera_process.wait(timeout=10)
                rospy.loginfo("相机进程已正常停止")
            except subprocess.TimeoutExpired:
                # 如果10秒后还没结束，强制杀死
                rospy.logwarn("相机进程未在10秒内结束，强制终止...")
                os.killpg(os.getpgid(self.camera_process.pid), signal.SIGKILL)
                self.camera_process.wait()
                rospy.loginfo("相机进程已强制终止")
            
        except ProcessLookupError:
            rospy.logwarn("相机进程已不存在")
        except Exception as e:
            rospy.logerr(f"停止相机进程时出错: {str(e)}")
        finally:
            self.camera_process = None
            rospy.sleep(2)  # 等待资源释放
    
    def idle_test(self, duration=30):
        """第一阶段测试：静置设备，不启动相机"""
        rospy.loginfo("="*80)
        rospy.loginfo(f"第一阶段测试：静置设备（不启动相机）")
        rospy.loginfo(f"测试时长: {duration} 秒")
        rospy.loginfo("="*80)
        
        # 启动CPU监控（静置模式）
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=lambda: self.monitor_cpu(phase='idle'), daemon=True)
        self.monitor_thread.start()
        
        # 运行指定时长
        start_time = time.time()
        try:
            while not rospy.is_shutdown() and (time.time() - start_time) < duration:
                rospy.sleep(0.1)
        except KeyboardInterrupt:
            rospy.loginfo("第一阶段测试被用户中断")
        
        # 停止监控
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        rospy.loginfo("第一阶段测试完成")
        rospy.loginfo("="*80)
        rospy.sleep(2)  # 等待2秒后开始下一阶段
    
    def baseline_test(self, duration=30):
        """第二阶段测试：启动相机获取视频流但不推流"""
        rospy.loginfo("="*80)
        rospy.loginfo(f"第二阶段测试：启动相机但不推流")
        rospy.loginfo(f"测试时长: {duration} 秒")
        rospy.loginfo("="*80)
        
        # 启动相机
        if not self.start_camera():
            rospy.logerr("无法启动相机，跳过第二阶段测试")
            return
        
        # 订阅所有相机话题
        self.subscribers = {}
        for name, topic in self.camera_topics.items():
            self.subscribers[name] = rospy.Subscriber(
                topic,
                Image,
                lambda msg, n=name: self.image_callback(n, msg),
                queue_size=1
            )
            rospy.loginfo(f"已订阅话题: {topic}")
        
        # 等待话题连接
        rospy.sleep(2)
        
        # 启动CPU监控（基准测试模式）
        self.monitoring = True
        self.running = False  # 不推流
        self.monitor_thread = threading.Thread(target=lambda: self.monitor_cpu(phase='baseline'), daemon=True)
        self.monitor_thread.start()
        
        # 运行指定时长
        start_time = time.time()
        try:
            while not rospy.is_shutdown() and (time.time() - start_time) < duration:
                rospy.sleep(0.1)
        except KeyboardInterrupt:
            rospy.loginfo("第二阶段测试被用户中断")
        
        # 停止监控
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        rospy.loginfo("第二阶段测试完成")
        rospy.loginfo("="*80)
        rospy.sleep(2)  # 等待2秒后开始推流测试
    
    def streaming_test(self, duration=60):
        """第三阶段测试：启动相机并推流"""
        rospy.loginfo("="*80)
        rospy.loginfo(f"第三阶段测试：启动相机并推流")
        rospy.loginfo(f"测试时长: {duration} 秒")
        rospy.loginfo("="*80)
        
        # 确保相机已启动（如果第二阶段已启动，这里不需要重复启动）
        if self.camera_process is None:
            if not self.start_camera():
                rospy.logerr("无法启动相机，跳过第三阶段测试")
                return
        
        # 确保已订阅话题（如果第二阶段已订阅，这里不需要重复订阅）
        if not hasattr(self, 'subscribers') or not self.subscribers:
            self.subscribers = {}
            for name, topic in self.camera_topics.items():
                self.subscribers[name] = rospy.Subscriber(
                    topic,
                    Image,
                    lambda msg, n=name: self.image_callback(n, msg),
                    queue_size=1
                )
                rospy.loginfo(f"已订阅话题: {topic}")
            rospy.sleep(2)
        
        # 启动CPU监控（推流测试模式）
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=lambda: self.monitor_cpu(phase='streaming'), daemon=True)
        self.monitor_thread.start()
        
        # 开始推流
        self.running = True
        self.streaming_stats['start_time'] = datetime.now()
        
        # 运行指定时长
        start_time = time.time()
        try:
            while not rospy.is_shutdown() and (time.time() - start_time) < duration:
                rospy.sleep(0.1)
        except KeyboardInterrupt:
            rospy.loginfo("第三阶段测试被用户中断")
        
        # 停止测试
        self.stop_test()
    
    def start_test(self, idle_duration=30, baseline_duration=30, streaming_duration=60, 
                   run_idle=True, run_baseline=True, run_streaming=True):
        """开始完整的三阶段测试"""
        try:
            # 第一阶段：静置测试
            if run_idle:
                self.idle_test(duration=idle_duration)
            
            # 第二阶段：相机但不推流
            if run_baseline:
                self.baseline_test(duration=baseline_duration)
            
            # 第三阶段：推流测试
            if run_streaming:
                self.streaming_test(duration=streaming_duration)
            else:
                # 如果不运行第三阶段，需要停止相机
                self.stop_camera()
                
        except Exception as e:
            rospy.logerr(f"测试过程中出错: {str(e)}")
            self.stop_test()
    
    def stop_test(self):
        """停止测试"""
        rospy.loginfo("正在停止测试...")
        
        self.running = False
        self.monitoring = False
        
        # 等待监控线程结束
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        # 停止FFmpeg推流
        for topic_name in list(self.ffmpeg_processes.keys()):
            self._stop_ffmpeg_stream(topic_name)
        
        # 关闭视频写入器
        for writer in self.video_writers.values():
            writer.release()
        
        # 停止相机
        self.stop_camera()
        
        # 记录结束时间
        if self.streaming_stats['start_time']:
            self.streaming_stats['end_time'] = datetime.now()
        
        # 生成报告
        self.generate_report()
    
    def generate_report(self):
        """生成详细的三阶段测试报告"""
        streaming_duration = 0
        if self.streaming_stats['start_time'] and self.streaming_stats['end_time']:
            streaming_duration = (self.streaming_stats['end_time'] - self.streaming_stats['start_time']).total_seconds()
        
        # 构建各阶段的单核统计信息
        idle_per_core_stats = []
        baseline_per_core_stats = []
        streaming_per_core_stats = []
        
        for i in range(self.cpu_count):
            if self.idle_stats['samples'] > 0:
                core_stat = {
                    'core_id': i,
                    'average_percent': round(self.idle_stats['per_core_avg'][i], 2),
                    'max_percent': round(self.idle_stats['per_core_max'][i], 2)
                }
                if self.cpu_freq_available:
                    core_stat.update({
                        'max_freq_mhz': round(self.cpu_max_freqs[i] / 1000.0, 2) if i < len(self.cpu_max_freqs) else 0,
                        'avg_freq_mhz': round(self.idle_stats['per_core_freq_avg'][i], 2) if i < len(self.idle_stats['per_core_freq_avg']) else 0,
                        'min_freq_mhz': round(self.idle_stats['per_core_freq_min'][i], 2) if i < len(self.idle_stats['per_core_freq_min']) and self.idle_stats['per_core_freq_min'][i] != float('inf') else 0,
                        'max_freq_mhz_observed': round(self.idle_stats['per_core_freq_max'][i], 2) if i < len(self.idle_stats['per_core_freq_max']) else 0
                    })
                idle_per_core_stats.append(core_stat)
            
            if self.baseline_stats['samples'] > 0:
                core_stat = {
                    'core_id': i,
                    'average_percent': round(self.baseline_stats['per_core_avg'][i], 2),
                    'max_percent': round(self.baseline_stats['per_core_max'][i], 2)
                }
                if self.cpu_freq_available:
                    core_stat.update({
                        'max_freq_mhz': round(self.cpu_max_freqs[i] / 1000.0, 2) if i < len(self.cpu_max_freqs) else 0,
                        'avg_freq_mhz': round(self.baseline_stats['per_core_freq_avg'][i], 2) if i < len(self.baseline_stats['per_core_freq_avg']) else 0,
                        'min_freq_mhz': round(self.baseline_stats['per_core_freq_min'][i], 2) if i < len(self.baseline_stats['per_core_freq_min']) and self.baseline_stats['per_core_freq_min'][i] != float('inf') else 0,
                        'max_freq_mhz_observed': round(self.baseline_stats['per_core_freq_max'][i], 2) if i < len(self.baseline_stats['per_core_freq_max']) else 0
                    })
                baseline_per_core_stats.append(core_stat)
            
            if self.streaming_stats['cpu_samples'] > 0:
                core_stat = {
                    'core_id': i,
                    'average_percent': round(self.streaming_stats['per_core_avg'][i], 2),
                    'max_percent': round(self.streaming_stats['per_core_max'][i], 2),
                    'min_percent': round(self.streaming_stats['per_core_min'][i], 2)
                }
                if self.cpu_freq_available:
                    max_freq_mhz = round(self.cpu_max_freqs[i] / 1000.0, 2) if i < len(self.cpu_max_freqs) else 0
                    avg_freq_mhz = round(self.streaming_stats['per_core_freq_avg'][i], 2) if i < len(self.streaming_stats['per_core_freq_avg']) else 0
                    min_freq_mhz = round(self.streaming_stats['per_core_freq_min'][i], 2) if i < len(self.streaming_stats['per_core_freq_min']) and self.streaming_stats['per_core_freq_min'][i] != float('inf') else 0
                    max_observed_freq = round(self.streaming_stats['per_core_freq_max'][i], 2) if i < len(self.streaming_stats['per_core_freq_max']) else 0
                    
                    # 计算频率下降百分比
                    freq_drop_percent = 0
                    if max_freq_mhz > 0 and avg_freq_mhz < max_freq_mhz * 0.95:
                        freq_drop_percent = round((1 - avg_freq_mhz / max_freq_mhz) * 100, 2)
                    
                    core_stat.update({
                        'max_freq_mhz': max_freq_mhz,
                        'avg_freq_mhz': avg_freq_mhz,
                        'min_freq_mhz': min_freq_mhz,
                        'max_freq_mhz_observed': max_observed_freq,
                        'freq_drop_percent': freq_drop_percent,
                        'is_throttled': freq_drop_percent > 5  # 如果平均频率低于最大频率5%以上，认为降频
                    })
                streaming_per_core_stats.append(core_stat)
        
        # 计算影响分析
        impact_analysis = {}
        if self.idle_stats['samples'] > 0 and self.baseline_stats['samples'] > 0:
            impact_analysis['camera_impact'] = {
                'cpu_increase_avg': round(self.baseline_stats['avg_cpu'] - self.idle_stats['avg_cpu'], 2),
                'cpu_increase_percent': round(
                    ((self.baseline_stats['avg_cpu'] - self.idle_stats['avg_cpu']) / max(self.idle_stats['avg_cpu'], 0.1)) * 100, 2
                ),
                'per_core_increase': [
                    round(self.baseline_stats['per_core_avg'][i] - self.idle_stats['per_core_avg'][i], 2)
                    for i in range(self.cpu_count)
                ]
            }
        
        if self.baseline_stats['samples'] > 0 and self.streaming_stats['cpu_samples'] > 0:
            impact_analysis['streaming_impact'] = {
                'cpu_increase_avg': round(self.streaming_stats['avg_cpu'] - self.baseline_stats['avg_cpu'], 2),
                'cpu_increase_percent': round(
                    ((self.streaming_stats['avg_cpu'] - self.baseline_stats['avg_cpu']) / max(self.baseline_stats['avg_cpu'], 0.1)) * 100, 2
                ),
                'per_core_increase': [
                    round(self.streaming_stats['per_core_avg'][i] - self.baseline_stats['per_core_avg'][i], 2)
                    for i in range(self.cpu_count)
                ]
            }
        
        report = {
            'test_info': {
                'cpu_count': self.cpu_count,
                'streaming_duration_seconds': streaming_duration,
                'streaming_start_time': self.streaming_stats['start_time'].isoformat() if self.streaming_stats['start_time'] else None,
                'streaming_end_time': self.streaming_stats['end_time'].isoformat() if self.streaming_stats['end_time'] else None,
                'cpu_freq_monitoring_available': self.cpu_freq_available,
                'cpu_max_freqs_mhz': [f / 1000.0 for f in self.cpu_max_freqs] if self.cpu_freq_available else [],
                'cpu_min_freqs_mhz': [f / 1000.0 for f in self.cpu_min_freqs] if self.cpu_freq_available else []
            },
            'camera_info': {
                'topics': self.camera_topics,
                'frames_received': dict(self.frame_counts)
            },
            'streaming_config': {
                'use_ffmpeg': self.stream_config.get('use_ffmpeg', False),
                'stream_type': self.stream_config.get('stream_type', 'udp'),
                'ffmpeg_preset': self.stream_config.get('ffmpeg_preset', 'ultrafast'),
                'ffmpeg_crf': self.stream_config.get('ffmpeg_crf', 23),
                'fps': self.stream_config.get('fps', 30),
                'udp_ports': {
                    'camera': self.stream_config.get('udp_port_base', 5000),
                    'left_cam': self.stream_config.get('udp_port_base', 5000) + 1,
                    'right_cam': self.stream_config.get('udp_port_base', 5000) + 2
                } if self.stream_config.get('stream_type') == 'udp' else None
            },
            'phase1_idle_statistics': {
                'description': '第一阶段：静置设备，不启动相机',
                'average_cpu_percent': round(self.idle_stats['avg_cpu'], 2),
                'max_cpu_percent': round(self.idle_stats['max_cpu'], 2),
                'min_cpu_percent': round(self.idle_stats['min_cpu'], 2),
                'per_core_statistics': idle_per_core_stats,
                'total_samples': self.idle_stats['samples'],
                'throttle_count': self.idle_stats['throttle_count'],
                'cpu_freq_monitoring': self.cpu_freq_available
            },
            'phase2_baseline_statistics': {
                'description': '第二阶段：启动相机但不推流',
                'average_cpu_percent': round(self.baseline_stats['avg_cpu'], 2),
                'max_cpu_percent': round(self.baseline_stats['max_cpu'], 2),
                'min_cpu_percent': round(self.baseline_stats['min_cpu'], 2),
                'per_core_statistics': baseline_per_core_stats,
                'total_samples': self.baseline_stats['samples'],
                'throttle_count': self.baseline_stats['throttle_count'],
                'cpu_freq_monitoring': self.cpu_freq_available
            },
            'phase3_streaming_statistics': {
                'description': '第三阶段：启动相机并推流',
                'average_cpu_percent': round(self.streaming_stats['avg_cpu'], 2),
                'max_cpu_percent': round(self.streaming_stats['max_cpu'], 2),
                'min_cpu_percent': round(self.streaming_stats['min_cpu'], 2),
                'per_core_statistics': streaming_per_core_stats,
                'total_samples': self.streaming_stats['cpu_samples'],
                'throttle_count': self.streaming_stats['throttle_count'],
                'throttle_events': self.streaming_stats['throttle_events'][:100],  # 只保存前100个降频事件
                'cpu_freq_monitoring': self.cpu_freq_available
            },
            'impact_analysis': impact_analysis,
            'idle_timeline': [
                {
                    'timestamp': d['timestamp'],
                    'cpu_percent': d['cpu_percent'],
                    'cpu_per_core': d['cpu_per_core'],
                    'memory_percent': d['memory_percent'],
                    'cpu_freqs_mhz': d.get('cpu_freqs_mhz', []),
                    'is_throttling': d.get('is_throttling', False),
                    'throttled_cores': d.get('throttled_cores', [])
                }
                for d in self.idle_cpu_data
            ],
            'baseline_timeline': [
                {
                    'timestamp': d['timestamp'],
                    'cpu_percent': d['cpu_percent'],
                    'cpu_per_core': d['cpu_per_core'],
                    'memory_percent': d['memory_percent'],
                    'cpu_freqs_mhz': d.get('cpu_freqs_mhz', []),
                    'is_throttling': d.get('is_throttling', False),
                    'throttled_cores': d.get('throttled_cores', [])
                }
                for d in self.baseline_cpu_data
            ],
            'streaming_timeline': [
                {
                    'timestamp': d['timestamp'],
                    'cpu_percent': d['cpu_percent'],
                    'cpu_per_core': d['cpu_per_core'],
                    'memory_percent': d['memory_percent'],
                    'cpu_freqs_mhz': d.get('cpu_freqs_mhz', []),
                    'is_throttling': d.get('is_throttling', False),
                    'throttled_cores': d.get('throttled_cores', [])
                }
                for d in self.streaming_cpu_data
            ]
        }
        
        # 保存JSON报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"./cpu_load_test_report_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 打印详细摘要
        print("\n" + "="*80)
        print("CPU负载测试报告（三阶段测试）")
        print("="*80)
        print(f"CPU核心数: {self.cpu_count}")
        
        # 显示推流配置
        if self.stream_config.get('use_ffmpeg', False):
            stream_type = self.stream_config.get('stream_type', 'udp')
            print(f"\n推流配置:")
            print(f"  推流方式: FFmpeg ({stream_type})")
            print(f"  编码器: libx264 (CPU编码)")
            print(f"  编码预设: {self.stream_config.get('ffmpeg_preset', 'ultrafast')}")
            print(f"  视频质量(CRF): {self.stream_config.get('ffmpeg_crf', 23)}")
            print(f"  帧率: {self.stream_config.get('fps', 30)} FPS")
            if stream_type == 'udp':
                print(f"  UDP端口: camera={self.stream_config.get('udp_port_base', 5000)}, "
                      f"left_cam={self.stream_config.get('udp_port_base', 5000)+1}, "
                      f"right_cam={self.stream_config.get('udp_port_base', 5000)+2}")
            elif stream_type == 'null':
                print(f"  注意: 推流到NULL（编码但不保存），用于CPU负载测试")
        
        # 第一阶段结果
        if self.idle_stats['samples'] > 0:
            print("\n" + "-"*80)
            print("第一阶段：静置设备（不启动相机）")
            print("-"*80)
            print(f"  平均CPU使用率: {self.idle_stats['avg_cpu']:.2f}%")
            print(f"  最大CPU使用率: {self.idle_stats['max_cpu']:.2f}%")
            print(f"  最小CPU使用率: {self.idle_stats['min_cpu']:.2f}%")
            print(f"\n  各核心平均使用率:")
            for i in range(self.cpu_count):
                freq_info = ""
                if self.cpu_freq_available and i < len(self.idle_stats['per_core_freq_avg']):
                    max_freq = self.cpu_max_freqs[i] / 1000.0 if i < len(self.cpu_max_freqs) else 0
                    avg_freq = self.idle_stats['per_core_freq_avg'][i]
                    min_freq = self.idle_stats['per_core_freq_min'][i] if self.idle_stats['per_core_freq_min'][i] != float('inf') else 0
                    freq_info = f" | 频率: {avg_freq:.0f}MHz (最大:{max_freq:.0f}MHz, 最小:{min_freq:.0f}MHz)"
                print(f"    核心 {i}: {self.idle_stats['per_core_avg'][i]:.2f}% "
                      f"(最大: {self.idle_stats['per_core_max'][i]:.2f}%){freq_info}")
            if self.cpu_freq_available:
                print(f"\n  降频检测: 检测到 {self.idle_stats['throttle_count']} 次降频事件")
        
        # 第二阶段结果
        if self.baseline_stats['samples'] > 0:
            print("\n" + "-"*80)
            print("第二阶段：启动相机但不推流")
            print("-"*80)
            print(f"  平均CPU使用率: {self.baseline_stats['avg_cpu']:.2f}%")
            print(f"  最大CPU使用率: {self.baseline_stats['max_cpu']:.2f}%")
            print(f"  最小CPU使用率: {self.baseline_stats['min_cpu']:.2f}%")
            print(f"\n  各核心平均使用率:")
            for i in range(self.cpu_count):
                freq_info = ""
                if self.cpu_freq_available and i < len(self.baseline_stats['per_core_freq_avg']):
                    max_freq = self.cpu_max_freqs[i] / 1000.0 if i < len(self.cpu_max_freqs) else 0
                    avg_freq = self.baseline_stats['per_core_freq_avg'][i]
                    min_freq = self.baseline_stats['per_core_freq_min'][i] if self.baseline_stats['per_core_freq_min'][i] != float('inf') else 0
                    freq_info = f" | 频率: {avg_freq:.0f}MHz (最大:{max_freq:.0f}MHz, 最小:{min_freq:.0f}MHz)"
                print(f"    核心 {i}: {self.baseline_stats['per_core_avg'][i]:.2f}% "
                      f"(最大: {self.baseline_stats['per_core_max'][i]:.2f}%){freq_info}")
            if self.cpu_freq_available:
                print(f"\n  降频检测: 检测到 {self.baseline_stats['throttle_count']} 次降频事件")
            
            # 相机对CPU的影响
            if self.idle_stats['samples'] > 0 and 'camera_impact' in impact_analysis:
                cam_impact = impact_analysis['camera_impact']
                print(f"\n  相机启动导致的CPU增加:")
                print(f"    平均增加: {cam_impact['cpu_increase_avg']:+.2f}%")
                print(f"    增加百分比: {cam_impact['cpu_increase_percent']:+.2f}%")
                print(f"    各核心增加:")
                for i in range(self.cpu_count):
                    print(f"      核心 {i}: {cam_impact['per_core_increase'][i]:+.2f}%")
        
        # 第三阶段结果
        if self.streaming_stats['cpu_samples'] > 0:
            print("\n" + "-"*80)
            print("第三阶段：启动相机并推流")
            print("-"*80)
            print(f"  测试时长: {streaming_duration:.2f} 秒")
            print(f"  平均CPU使用率: {self.streaming_stats['avg_cpu']:.2f}%")
            print(f"  最大CPU使用率: {self.streaming_stats['max_cpu']:.2f}%")
            print(f"  最小CPU使用率: {self.streaming_stats['min_cpu']:.2f}%")
            print(f"\n  各核心平均使用率和频率:")
            throttled_cores = []
            for i in range(self.cpu_count):
                freq_info = ""
                is_throttled = False
                if self.cpu_freq_available and i < len(self.streaming_stats['per_core_freq_avg']):
                    max_freq = self.cpu_max_freqs[i] / 1000.0 if i < len(self.cpu_max_freqs) else 0
                    avg_freq = self.streaming_stats['per_core_freq_avg'][i]
                    min_freq = self.streaming_stats['per_core_freq_min'][i] if self.streaming_stats['per_core_freq_min'][i] != float('inf') else 0
                    freq_drop = 0
                    if max_freq > 0:
                        freq_drop = (1 - avg_freq / max_freq) * 100
                        if freq_drop > 5:  # 如果平均频率低于最大频率5%以上
                            is_throttled = True
                            throttled_cores.append(i)
                    
                    throttle_marker = " ⚠降频" if is_throttled else ""
                    freq_info = f" | 频率: {avg_freq:.0f}MHz (最大:{max_freq:.0f}MHz, 最小:{min_freq:.0f}MHz, 下降:{freq_drop:.1f}%){throttle_marker}"
                print(f"    核心 {i}: {self.streaming_stats['per_core_avg'][i]:.2f}% "
                      f"(最大: {self.streaming_stats['per_core_max'][i]:.2f}%, "
                      f"最小: {self.streaming_stats['per_core_min'][i]:.2f}%){freq_info}")
            
            # 降频检测结果
            if self.cpu_freq_available:
                print(f"\n  CPU频率监控结果:")
                print(f"    降频事件总数: {self.streaming_stats['throttle_count']} 次")
                if throttled_cores:
                    print(f"    ⚠ 警告: 检测到以下核心在推流时发生降频: {throttled_cores}")
                    print(f"    这些核心的平均运行频率低于最大频率5%以上，可能影响推流性能")
                else:
                    print(f"    ✓ 未检测到明显的CPU降频，推流时CPU频率保持正常")
            
            # 推流对CPU的影响
            if self.baseline_stats['samples'] > 0 and 'streaming_impact' in impact_analysis:
                stream_impact = impact_analysis['streaming_impact']
                print(f"\n  推流导致的CPU增加:")
                print(f"    平均增加: {stream_impact['cpu_increase_avg']:+.2f}%")
                print(f"    增加百分比: {stream_impact['cpu_increase_percent']:+.2f}%")
                print(f"    各核心增加:")
                for i in range(self.cpu_count):
                    print(f"      核心 {i}: {stream_impact['per_core_increase'][i]:+.2f}%")
                
                # 判断推流是否对CPU造成显著影响
                if abs(stream_impact['cpu_increase_avg']) < 5:
                    print(f"\n  ⚠ 注意: 推流对CPU负载的影响较小（<5%），可能推流未正常工作或系统负载较低")
                elif stream_impact['cpu_increase_avg'] > 0:
                    print(f"\n  ✓ 推流确实对CPU负载造成了影响，平均增加了 {stream_impact['cpu_increase_avg']:.2f}%")
        
        print(f"\n各相机接收帧数:")
        for name, count in self.frame_counts.items():
            fps = count / streaming_duration if streaming_duration > 0 else 0
            print(f"  {name}: {count} 帧 (平均 {fps:.2f} FPS)")
        
        print(f"\n详细报告已保存至: {report_path}")
        print("="*80 + "\n")
        
        # CPU使用率评估
        if self.streaming_stats['cpu_samples'] > 0:
            print("性能评估:")
            print("-"*80)
            if self.streaming_stats['avg_cpu'] < 50:
                print("✓ CPU负载评估: 良好 (平均使用率 < 50%)")
            elif self.streaming_stats['avg_cpu'] < 80:
                print("⚠ CPU负载评估: 中等 (平均使用率 50-80%)")
            else:
                print("✗ CPU负载评估: 较高 (平均使用率 > 80%)")
            
            if self.streaming_stats['max_cpu'] > 95:
                print("⚠ 警告: 最大CPU使用率超过95%，可能存在性能瓶颈")
            
            # CPU频率和降频评估
            if self.cpu_freq_available:
                print("\nCPU频率评估:")
                throttled_cores = []
                for i in range(self.cpu_count):
                    if i < len(self.streaming_stats['per_core_freq_avg']) and i < len(self.cpu_max_freqs):
                        max_freq = self.cpu_max_freqs[i] / 1000.0
                        avg_freq = self.streaming_stats['per_core_freq_avg'][i]
                        if max_freq > 0:
                            freq_drop = (1 - avg_freq / max_freq) * 100
                            if freq_drop > 5:
                                throttled_cores.append(i)
                
                if throttled_cores:
                    print(f"✗ CPU降频检测: 检测到核心 {throttled_cores} 在推流时发生降频")
                    print(f"  降频事件总数: {self.streaming_stats['throttle_count']} 次")
                    print(f"  ⚠ 警告: CPU降频可能导致推流性能下降，建议:")
                    print(f"    - 检查系统温度是否过高")
                    print(f"    - 检查电源管理设置")
                    print(f"    - 考虑优化推流参数或降低负载")
                else:
                    print(f"✓ CPU频率评估: 正常 (未检测到明显降频)")
                    if self.streaming_stats['throttle_count'] > 0:
                        print(f"  注: 检测到 {self.streaming_stats['throttle_count']} 次短暂降频事件，但平均频率正常")
            
            # 找出负载最高的核心
            max_core_idx = np.argmax(self.streaming_stats['per_core_avg'])
            print(f"\n负载最高的核心: 核心 {max_core_idx} (平均: {self.streaming_stats['per_core_avg'][max_core_idx]:.2f}%)")
        
        print("="*80 + "\n")


def main():
    """主函数"""
    tester = CameraStreamTester()
    
    # 从ROS参数获取测试配置
    idle_duration = rospy.get_param('~idle_duration', 30)
    baseline_duration = rospy.get_param('~baseline_duration', 30)
    streaming_duration = rospy.get_param('~streaming_duration', 60)
    run_idle = rospy.get_param('~run_idle', True)
    run_baseline = rospy.get_param('~run_baseline', True)
    run_streaming = rospy.get_param('~run_streaming', True)
    
    try:
        tester.start_test(
            idle_duration=idle_duration,
            baseline_duration=baseline_duration,
            streaming_duration=streaming_duration,
            run_idle=run_idle,
            run_baseline=run_baseline,
            run_streaming=run_streaming
        )
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Test failed: {str(e)}")
        tester.stop_test()


if __name__ == '__main__':
    main()
