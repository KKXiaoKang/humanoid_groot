#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复适配层训练问题的完整方案

问题分析：
1. Flow Matching loss 对适配层的梯度太弱（需要穿过整个 DiT）
2. Linear adapter 表达能力不够
3. 学习率和训练策略需要优化

解决方案：
1. 使用 MLP adapter（更强表达能力）
2. 增大学习率（10x）
3. 增加训练数据（100+ samples）
4. 增加训练轮数（50 epochs）
5. 使用 warmup scheduler
6. 添加梯度监控和早停机制
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
from safetensors.torch import load_file, save_file
import glob
import numpy as np

from lerobot.policies.groot.weight_merge_groot import TwoStageExpertMerger, DistributionAdapter
from lerobot.policies.groot.groot_n1 import BACKBONE_FEATURE_KEY


class ImprovedDistributionAdapter(nn.Module):
    """
    改进的适配层：更强的表达能力和更好的初始化
    
    关键改进：
    1. 使用 MLP 而不是 Linear（更强表达能力）
    2. 残差连接（更容易学习）
    3. 更好的初始化策略
    """
    
    def __init__(
        self,
        hidden_size: int = 2048,
        adapter_type: str = "mlp",  # 强制使用 mlp
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_type = adapter_type
        
        # 强制使用 MLP（即使指定 linear 也使用 mlp）
        if adapter_type == "linear":
            print("   ⚠️ Linear adapter 表达能力不够，自动切换到 MLP")
            adapter_type = "mlp"
        
        if adapter_type == "mlp":
            # 两层 MLP + 残差连接
            self.adapter = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            
            # 关键：更好的初始化
            # 第一层：Xavier 初始化
            nn.init.xavier_uniform_(self.adapter[0].weight, gain=0.1)  # 小 gain，接近恒等映射
            nn.init.zeros_(self.adapter[0].bias)
            
            # 最后一层：初始化为接近零（残差风格）
            nn.init.zeros_(self.adapter[-2].weight)
            nn.init.zeros_(self.adapter[-2].bias)
            
        elif adapter_type == "layernorm_only":
            self.adapter = nn.LayerNorm(hidden_size)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        # 残差连接的系数（可学习，初始化为 1.0）
        self.residual_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差连接 + 适配层"""
        if self.adapter_type == "mlp":
            return x + self.residual_scale * self.adapter(x)
        else:
            return self.adapter(x)


def create_improved_adapter_training_script():
    """
    创建改进的适配层训练脚本
    
    关键改进：
    1. 使用 MLP adapter
    2. 学习率 1e-3（10x 增大）
    3. 50 epochs
    4. 100+ samples per dataset
    5. Warmup scheduler
    6. 梯度监控
    """
    
    script_content = '''#!/bin/bash
# 改进的适配层训练脚本

set -e

METHOD="two_stage_adapter"
NARROWER_PATH="/home/kangkk/humanoid_groot_base/outputs/0112_h100x4_groot_cross_attention_narrower_very_conservative/checkpoints/020000/pretrained_model"
WIDER_PATH="/home/kangkk/humanoid_groot_base/outputs/0113_h100x4_groot_cross_attention_wider_very_conservative_mix_dense/checkpoints/014000/pretrained_model"
OUTPUT_PATH="./outputs/merged_groot/pretrained_model"
DEVICE=${1:-cuda:0}

echo "======================================"
echo "🚀 改进的适配层训练方案"
echo "======================================"
echo ""
echo "关键改进："
echo "  1. 使用 MLP adapter（更强表达能力）"
echo "  2. 学习率 1e-3（10x 增大）"
echo "  3. 50 epochs（更多训练）"
echo "  4. 100 samples per dataset（更多数据）"
echo "  5. Warmup scheduler"
echo ""

python scripts/train_weight_merge.py \\
    --method two_stage_adapter \\
    --narrower_path "$NARROWER_PATH" \\
    --wider_path "$WIDER_PATH" \\
    --use_default_datasets \\
    --alpha 0.5 \\
    --adapter_type mlp \\
    --adapter_epochs 50 \\
    --adapter_lr 1e-3 \\
    --num_samples 100 \\
    --batch_size 1 \\
    --device "$DEVICE" \\
    --output_path "$OUTPUT_PATH"

echo ""
echo "✅ 训练完成！"
echo "   检查适配层权重是否真的被训练了："
echo "   python3 -c \\"
echo "   \\"
echo "   import json"
echo "   from pathlib import Path"
echo "   from safetensors.torch import load_file"
echo "   import glob"
echo "   import numpy as np"
echo "   "
echo "   model_path = Path('$OUTPUT_PATH')"
echo "   safetensors_files = glob.glob(str(model_path / 'model*.safetensors'))"
echo "   state_dict = {}"
echo "   for f in sorted(safetensors_files):"
echo "       state_dict.update(load_file(f))"
echo "   "
echo "   key = 'distribution_adapter.adapter.0.weight'"
echo "   if key in state_dict:"
echo "       weight = state_dict[key].cpu().numpy()"
echo "       eye = np.eye(weight.shape[0])"
echo "       diff = np.abs(weight - eye).mean()"
echo "       print(f'Distance from identity: {diff:.6f}')"
echo "       if diff > 0.1:"
echo "           print('✅ Adapter trained successfully!')"
echo "       else:"
echo "           print('⚠️ Adapter still close to identity')"
'''
    
    script_path = Path("scripts/train_adapter_improved.sh")
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    
    print(f"✅ Created improved training script: {script_path}")


if __name__ == "__main__":
    create_improved_adapter_training_script()
