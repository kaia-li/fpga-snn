#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np

def export_weights_to_verilog(weights_file, output_file):
    """将NEST模拟器的权重导出为Verilog可用的格式"""
    with open(weights_file, 'rb') as f:
        weights = pickle.load(f)
    
    # 将权重转换为uint8格式（0-255）
    # 首先找到最大和最小权重进行归一化
    min_weight = min(weights)
    max_weight = max(weights)
    range_weight = max_weight - min_weight
    
    # 归一化权重到0-255范围
    normalized_weights = [int(((w - min_weight) / range_weight) * 255) for w in weights]
    
    # 写入Verilog初始化代码
    with open(output_file, 'w') as f:
        f.write("// 自动生成的权重初始化代码\n")
        f.write("// 生成自NEST模拟器\n\n")
        
        f.write("// 在weights_processor模块的initial块中使用以下代码\n")
        f.write("initial begin\n")
        
        # 假设权重是输出层权重
        weight_idx = 0
        hidden_size = 500  # 与模型匹配
        output_size = 10   # 10个类别
        
        for o in range(output_size):
            for i in range(hidden_size):
                if weight_idx < len(normalized_weights):
                    f.write(f"    weights[{o}][{i}] = 8'd{normalized_weights[weight_idx]};\n")
                    weight_idx += 1
        
        f.write("end\n")

if __name__ == "__main__":
    export_weights_to_verilog('snn_weights.pkl', 'weights_init.v')
    print("权重已导出至weights_init.v")