from setuptools import setup, find_packages
import os

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="kuavo_ik",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pydrake",
    ],
    # 包含资源文件（URDF和meshes）
    package_data={
        "kuavo_ik": [
            "drake_urdf/urdf/*.urdf",
            "meshes/*.STL",
        ],
    },
    include_package_data=True,
)