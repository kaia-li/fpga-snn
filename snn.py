#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nest
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import os
import pickle

# 设置NEST
nest.set_verbosity("M_WARNING")
nest.ResetKernel()

# 加载CIFAR-10数据集
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

# 将图像编码为脉冲
def encode_image_to_spikes(image, duration=100.0, max_rate=100.0):
    # 将像素值转换为发放率
    rates = image.reshape(-1) * max_rate
    # 创建泊松发生器
    pg = nest.Create('poisson_generator', rates.size)
    # 设置发放率
    for i, rate in enumerate(rates):
        nest.SetStatus([pg[i]], {'rate': rate})
    
    return pg

# 构建两层SNN网络（感受野拓扑结构）
def create_snn_network(input_size, hidden_size, output_size):
    # 输入层（泊松发生器）
    input_layer = nest.Create('poisson_generator', input_size)
    
    # 隐藏层（带漏电的积分发放神经元）
    hidden_layer = nest.Create('iaf_psc_alpha', hidden_size)
    
    # 输出层（带漏电的积分发放神经元）
    output_layer = nest.Create('iaf_psc_alpha', output_size)
    
    # 创建脉冲记录器
    spike_recorder = nest.Create('spike_recorder')
    nest.Connect(output_layer, spike_recorder)
    
    # 设置神经元参数
    nest.SetStatus(hidden_layer, {
        'tau_m': 20.0,      # 膜时间常数
        'V_th': -55.0,      # 阈值电压
        'V_reset': -70.0,   # 重置电压
        'E_L': -65.0,       # 静息电位
        't_ref': 2.0        # 不应期
    })
    
    nest.SetStatus(output_layer, {
        'tau_m': 20.0,
        'V_th': -55.0,
        'V_reset': -70.0,
        'E_L': -65.0,
        't_ref': 2.0
    })
    
    return input_layer, hidden_layer, output_layer, spike_recorder

# 创建感受野拓扑连接
def create_receptive_field_connections(input_layer, hidden_layer, rf_size=5, stride=3):
    # 输入是32x32x3的CIFAR-10图像
    input_width, input_height, channels = 32, 32, 3
    input_size = input_width * input_height * channels
    
    # 创建连接
    conn_spec = {'rule': 'pairwise_bernoulli', 'p': 0.1}  # 随机连接概率
    syn_spec = {'weight': 10.0, 'delay': 1.0}
    
    # 实现感受野拓扑结构
    for h_idx, h_neuron in enumerate(hidden_layer):
        # 计算感受野中心位置
        center_x = (h_idx % (input_width // stride)) * stride
        center_y = (h_idx // (input_width // stride)) * stride
        
        # 连接感受野内的所有输入神经元到此隐藏层神经元
        for c in range(channels):
            for i in range(max(0, center_x - rf_size//2), min(input_width, center_x + rf_size//2 + 1)):
                for j in range(max(0, center_y - rf_size//2), min(input_height, center_y + rf_size//2 + 1)):
                    input_idx = (j * input_width + i) * channels + c
                    if input_idx < len(input_layer):
                        nest.Connect([input_layer[input_idx]], [h_neuron], syn_spec=syn_spec)
    
    # 连接隐藏层到输出层（全连接）
    nest.Connect(hidden_layer, output_layer, conn_spec='all_to_all', 
                 syn_spec={'weight': 5.0, 'delay': 1.0})

# 训练网络（STDP学习规则）
def train_network(input_layer, hidden_layer, output_layer, x_train, y_train, num_samples=1000):
    # 实现STDP学习
    nest.CopyModel('stdp_synapse', 'stdp_synapse_rec', {
        'tau_plus': 20.0,
        'lambda': 0.01,
        'alpha': 2.0,
        'mu_plus': 0.0,
        'mu_minus': 0.0,
        'Wmax': 100.0
    })
    
    # 重新连接隐藏层到输出层，使用STDP突触
    nest.Connect(hidden_layer, output_layer, conn_spec='all_to_all', 
                 syn_spec={'model': 'stdp_synapse_rec'})
    
    # 训练循环
    for i in range(min(num_samples, len(x_train))):
        image = x_train[i]
        label = y_train[i][0]
        
        # 编码图像为脉冲
        rates = image.reshape(-1) * 100.0  # 最大发放率100Hz
        for j, rate in enumerate(rates):
            if j < len(input_layer):
                nest.SetStatus([input_layer[j]], {'rate': rate})
        
        # 模拟网络
        nest.Simulate(100.0)  # 模拟100ms

# 评估网络
def evaluate_network(input_layer, output_layer, spike_recorder, x_test, y_test, num_samples=100):
    correct = 0
    
    for i in range(min(num_samples, len(x_test))):
        image = x_test[i]
        label = y_test[i][0]
        
        # 重置记录器
        nest.SetStatus(spike_recorder, {'n_events': 0})
        
        # 编码图像为脉冲
        rates = image.reshape(-1) * 100.0
        for j, rate in enumerate(rates):
            if j < len(input_layer):
                nest.SetStatus([input_layer[j]], {'rate': rate})
        
        # 模拟网络
        nest.Simulate(100.0)
        
        # 获取输出层脉冲
        events = nest.GetStatus(spike_recorder, keys='events')[0]
        senders = events['senders']
        
        # 统计每个输出神经元的脉冲数
        spike_counts = np.zeros(len(output_layer))
        for s in senders:
            if s in output_layer:
                idx = output_layer.index(s)
                spike_counts[idx] += 1
        
        # 预测类别（脉冲最多的神经元）
        if len(spike_counts) > 0:
            predicted = np.argmax(spike_counts)
            if predicted == label:
                correct += 1
    
    accuracy = correct / min(num_samples, len(x_test))
    return accuracy

# 主函数
def main():
    print("开始加载CIFAR-10数据集...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print("数据集加载完成")
    
    # 参数设置
    input_size = 32 * 32 * 3  # CIFAR-10图像尺寸
    hidden_size = 500
    output_size = 10  # 10个类别
    
    print("创建SNN网络...")
    input_layer, hidden_layer, output_layer, spike_recorder = create_snn_network(
        input_size, hidden_size, output_size)
    
    print("创建感受野拓扑连接...")
    create_receptive_field_connections(input_layer, hidden_layer)
    
    print("开始训练网络...")
    train_network(input_layer, hidden_layer, output_layer, x_train, y_train, num_samples=500)
    
    print("评估网络性能...")
    accuracy = evaluate_network(input_layer, output_layer, spike_recorder, x_test, y_test, num_samples=100)
    print(f"测试准确率: {accuracy * 100:.2f}%")
    
    # 保存模型参数（用于FPGA实现）
    print("保存模型参数...")
    weights = []
    connections = nest.GetConnections(hidden_layer, output_layer)
    for conn in connections:
        weights.append(nest.GetStatus([conn], 'weight')[0])
    
    with open('snn_weights.pkl', 'wb') as f:
        pickle.dump(weights, f)
    
    print("完成！模型参数已保存至snn_weights.pkl")

if __name__ == "__main__":
    main()