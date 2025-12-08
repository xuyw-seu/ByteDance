#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络流量折线图绘制工具
用于绘制网络流量数据的折线图，区分请求和响应数据
主要功能：
1. 从文本文件中读取网络流量数据
2. 分离请求数据（正值）和响应数据（负值）
3. 绘制请求和响应数据的折线图
4. 支持自定义图表样式，包括颜色、标签、网格等
5. 保存绘制的图表为图片文件
6. 支持不同类型网络流量数据的可视化
"""
import matplotlib.pyplot as plt


def plot_positive_negative_lines(filename, s_dir):
    """
    绘制网络流量数据的折线图，区分请求和响应数据
    :param filename: 数据文件路径
    :param s_dir: 图表保存路径
    """
    positive_values = []  # 存储请求数据
    negative_values = []  # 存储响应数据

    with open(filename, "r") as file:
        for line in file:
            timestamp, value = line.strip().split()
            timestamp = float(timestamp)
            value = int(value)
            if value > 0:
                positive_values.append((timestamp, value))
            else:
                negative_values.append((timestamp, value))

    # 分离时间戳和数据值
    positive_timestamps, positive_data = zip(*positive_values)
    negative_timestamps, negative_data = zip(*negative_values)

    # 绘制图表
    plt.plot(positive_timestamps, positive_data, label='input', color='blue')  # 请求数据用蓝色
    plt.plot(negative_timestamps, negative_data, label='output', color='red')  # 响应数据用红色

    # 设置图表标签
    plt.xlabel('Timestamp')
    plt.ylabel('Packet Len')

    # 设置图例
    plt.legend()
    
    # 设置网格线（仅x轴）
    plt.grid(axis='x', linestyle='--', linewidth=1)
    
    # 保存图表
    plt.savefig(s_dir)
    
    # 显示图表
    plt.show()
    plt.close()

if __name__ == '__main__':
    # 示例使用
    # file_path = r"D:\处理的数据集\TLS1.3\total\163.com-10.txt"  # TLS1.3流量数据
    # file_path = r"D:\处理的数据集\trojan\Trojan-data-clean-eq\Amazon +trojan-12.txt"  # Trojan流量数据
    file_path = r"D:\处理的数据集\QUIC\test\total\GoogleDoc-31.txt"  # QUIC流量数据
    
    save_path = r"C:\Users\Administrator\Desktop\论文图片\len2.png"  # 图表保存路径
    plot_positive_negative_lines(file_path, save_path)