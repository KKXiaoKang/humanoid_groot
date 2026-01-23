#!/usr/bin/env python3
# coding: utf-8
from kuavo_humanoid_sdk.common.logger import SDKLogger
from kuavo_humanoid_sdk.interfaces.data_types import KuavoPose
from kuavo_humanoid_sdk.kuavo.core.ros.tools import KuavoRobotToolsCore

class KuavoRobotTools:
    """机器人工具类,提供坐标系转换接口。
    
    该类封装了不同机器人坐标系之间的坐标变换查询功能,支持多种返回数据格式。
    """
    
    def __init__(self):
        self.tools_core = KuavoRobotToolsCore()

    def get_tf_transform(self, target_frame: str, source_frame: str, 
                       return_type: str = "pose_quaternion"):
        """获取指定坐标系之间的变换。
        
        Args:
            target_frame (str): 目标坐标系名称
            source_frame (str): 源坐标系名称
            return_type (str, optional): 返回数据格式类型，默认为"pose_quaternion"
        
        Returns:
            变换数据,如果失败则返回None
        """
        return self.tools_core._get_tf_tree_transform(target_frame, source_frame, return_type=return_type)

    def get_link_pose(self, link_name: str, reference_frame: str = "base_link"):
        """获取指定机械臂关节链接的位置

        Args:
            link_name (str): 关节链接名称，如"zarm_l1_link"
            reference_frame (str): 参考坐标系，默认为base_link

        Returns:
            Tuple[float, float, float] | None: 三维位置坐标(x,y,z)，失败返回None
        """
        try:
            # 获取从参考坐标系到目标链接的变换
            transform = self.tools_core._get_tf_tree_transform(
                reference_frame,
                link_name,
                return_type="pose_quaternion"
            )
            if transform:
                return KuavoPose(
                    position=transform.position,
                    orientation=transform.orientation
                )
            return None
        except Exception as e:
            SDKLogger.error(f"获取{link_name}位置失败: {str(e)}")
            return None