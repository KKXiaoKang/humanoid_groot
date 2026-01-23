#!/usr/bin/env python3

import subprocess
import signal
import sys
import os
import time
import argparse
import termios
import tty
import select
import threading
import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

def get_key():
    """非阻塞方式读取单个按键"""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None

def print_message(msg, old_settings):
    """打印消息并确保正确的换行"""
    # 临时恢复终端设置以正确打印
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    print("\r" + msg + "\n", end='', flush=True)
    # 重新设置raw模式
    # tty.setraw(sys.stdin.fileno())

def read_process_output(process, old_settings):
    """读取进程输出的线程函数"""
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            # 临时恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print(output.strip(), flush=True)
            # 重新设置raw模式
            # tty.setraw(sys.stdin.fileno())

def stop_recording(process, old_settings):
    """停止录制进程"""
    if process is not None:
        # 发送SIGINT信号给进程组
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        process.wait()
        print_message("\nRecording stopped successfully", old_settings)
        return None
    return process

import rosgraph
import subprocess
import json
import time

def read_process_output(process, controller, old_settings):
    """读取录制脚本输出，同时捕获 bag 文件路径"""
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            line = output.strip()

            # 捕获 Output: xxx.bag
            if line.startswith("Output:"):
                bag_path = line.replace("Output:", "").strip()
                controller.last_bag_file = bag_path

            # 打印输出
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print(line, flush=True)

class BagHealthChecker:
    def __init__(self, required_topics, min_rates):
        self.required_topics = required_topics
        self.min_rates = min_rates

    def check_bag(self, bag_file):
        """用正则解析 rosbag info 输出，检查 topic 是否存在及频率"""
        cmd = ["rosbag", "info", bag_file, "--freq"]
        output = subprocess.check_output(cmd).decode("utf-8")

        bag_topics = {}

        # /camera/color/image_raw     126 msgs @  15.1 Hz : sensor_msgs/Image
        import re
        pattern = re.compile(r'(\S+)\s+\d+\s+msgs\s+@\s+([\d\.]+)\s+Hz')
        for line in output.splitlines():
            m = pattern.search(line)
            if m:
                topic_name = m.group(1)
                freq = float(m.group(2))
                bag_topics[topic_name] = freq

        # 检查缺失和低频
        missing_topics = []
        low_rate_topics = []

        for t in self.required_topics:
            if t not in bag_topics:
                missing_topics.append(t)
            else:
                req_freq = self.min_rates.get(t)
                if req_freq and bag_topics[t] < req_freq:
                    low_rate_topics.append((t, bag_topics[t], req_freq))

        # 多行报告
        if not missing_topics and not low_rate_topics:
            return True, "bag_health_ok"

        lines = ["Bag health check failed:"]
        if missing_topics:
            lines.append("Missing topics:")
            for t in missing_topics:
                lines.append(f"  - {t}")
        if low_rate_topics:
            lines.append("Low rate topics:")
            for t, r, req in low_rate_topics:
                lines.append(f"  - {t}: {r:.1f} Hz (required {req} Hz)")

        return False, "\n".join(lines)

class RecordingController:
    def __init__(self, task_name='default', init_node=False):
        self.task_name = task_name
        self.bag_prefix = None  # bag文件名前缀，如果为None则使用默认的episode_编号格式
        self.current_process = None
        self.output_thread = None
        
        # 保存终端设置
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        # 初始化ROS节点
        if init_node:
            rospy.init_node('recording_controller', anonymous=False)

        # 创建服务
        self.start_service = rospy.Service('start_recording', Trigger, self.handle_start_recording)
        self.stop_service = rospy.Service('stop_recording', Trigger, self.handle_stop_recording)
        
        # 创建状态发布者
        self.status_pub = rospy.Publisher('recording_status', String, queue_size=10)
        self.last_bag_file = None
        self.health_checker = BagHealthChecker(
            required_topics=[
                "/camera/color/image_raw",
                "/right_cam/color/image_raw",
                "/left_cam/color/image_raw",
                "/sensors_data_raw",
                "/kuavo_arm_traj",
                "/leju_claw_state",
                "/leju_claw_command",
            ],
            min_rates={
                "/camera/color/image_raw": 25,
                "/right_cam/color/image_raw": 25,
                "/left_cam/color/image_raw": 25,
                "/sensors_data_raw": 480,
                "/kuavo_arm_traj": 80,
                "/leju_claw_state": 480,
                "/leju_claw_command": 80,
            }
        )
        
    def handle_start_recording(self, req):
        """处理开始录制的服务请求"""
        if self.current_process is None:
            self.start_recording()
            return TriggerResponse(success=True, message="Recording started successfully")
        return TriggerResponse(success=False, message="Recording is already in progress")
    
    def handle_stop_recording(self, req):
        """处理停止录制的服务请求"""
        if self.current_process is not None:
            self.stop_recording()
            return TriggerResponse(success=True, message="Recording stopped successfully")
        return TriggerResponse(success=False, message="No recording in progress")
    
    def start_recording(self):
        """开始录制"""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        print_message("\nStarting recording...", self.old_settings)
        script_path = os.path.join(cur_dir, "record_episodes.sh")
        if not os.access(script_path, os.X_OK):
            os.chmod(script_path, 0o755)
        
        # 构建命令参数
        cmd = [script_path, "-t", "-n", self.task_name]
        if self.bag_prefix:
            cmd.extend(["-b", self.bag_prefix])
            
        self.current_process = subprocess.Popen(cmd,
                                             preexec_fn=os.setsid,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT,
                                             universal_newlines=True,
                                             bufsize=1)
        
        self.output_thread = threading.Thread(target=read_process_output, args=(self.current_process, self, self.old_settings))
        self.output_thread.daemon = True
        self.output_thread.start()
        self.status_pub.publish("recording_started")
    
    def stop_recording(self):
        """停止录制"""
        if self.current_process is not None:
            print_message("\nStopping recording...", self.old_settings)
            self.current_process = stop_recording(self.current_process, self.old_settings)
            if self.output_thread:
                self.output_thread.join(timeout=1.0)
            self.status_pub.publish("recording_stopped")
            if self.last_bag_file is None:
                ok = False
                report = "Bag health check error: no bag file detected"
            else:
                ok, report = self.health_checker.check_bag(self.last_bag_file)

            self.status_pub.publish(report)

            GREEN = "\033[92m"
            RED = "\033[91m"
            RESET = "\033[0m"
            msg = f"{GREEN}Bag health check: OK{RESET}" if ok else f"{RED}{report}{RESET}"

            print_message(msg, self.old_settings)
    
    def run(self):
        """运行控制器"""
        try:
            # 设置终端为raw模式
            tty.setraw(sys.stdin.fileno())
            
            print_message(f"Recording Control Panel (Task: {self.task_name})", self.old_settings)
            print_message("Press 'c' to start recording", self.old_settings)
            print_message("Press 's' to stop recording", self.old_settings)
            print_message("Press 'q' to quit", self.old_settings)
            
            rate = rospy.Rate(10)  # 10Hz
            while not rospy.is_shutdown():
                key = get_key()
                if key:
                    if key == 'c':
                        if self.current_process is None:
                            self.start_recording()
                        else:
                            print_message("\nRecording is already in progress", self.old_settings)
                    
                    elif key == 's':
                        if self.current_process is not None:
                            self.stop_recording()
                        else:
                            print_message("\nNo recording in progress", self.old_settings)
                    
                    elif key == 'q':
                        if self.current_process is not None:
                            self.stop_recording()
                        print_message("\nExiting...", self.old_settings)
                        break
                
                rate.sleep()
                
        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Control ROS bag recording with keyboard input and ROS services')
    parser.add_argument('-n', '--name', type=str, default='default',
                      help='Task name for the recording (default: default)')
    args = parser.parse_args()
    
    controller = RecordingController(args.name)
    controller.run()

if __name__ == "__main__":
    main()