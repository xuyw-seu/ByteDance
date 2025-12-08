#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混淆矩阵生成器与流量数据可视化工具
用于生成和显示网络流量数据的混淆矩阵，支持请求和响应数据的可视化
主要功能：
1. 从文本文件中读取网络流量数据
2. 将请求和响应数据转换为混淆矩阵格式
3. 生成并显示混淆矩阵，使用不同颜色方案（红色表示响应，蓝色表示请求）
4. 支持数据预处理，如空值处理和数据截断
5. 支持不同维度数据的可视化
"""
import matplotlib.pyplot as plt
import numpy as np

def make_res(res):
    """
    生成响应数据的混淆矩阵
    :param res: 响应数据列表
    """
    y_axis_data = []
    for j in res:
        if j == '':
            j = 0
        y_axis_data.append(int(j))

    A = np.array(y_axis_data).reshape(10, 50)
    ax = plt.matshow(A, cmap=plt.cm.Reds)
    plt.colorbar(ax.colorbar, fraction=0.025)
    plt.show()


def make_req(req):
    """
    生成请求数据的混淆矩阵
    :param req: 请求数据列表
    """
    x_axis_data = []
    for j in req:
        if j == '':
            j = 0
        x_axis_data.append(int(j))

    B = np.array(x_axis_data).reshape(10, 50)
    bx = plt.matshow(B, cmap=plt.cm.Blues)
    plt.colorbar(bx.colorbar, fraction=0.025)
    plt.show()

def read_txt(file_path):
    """
    读取并处理网络流量文本数据
    :param file_path: 数据文件路径
    :return: (请求数据列表, 响应数据列表)
    """
    # 注意：这里硬编码了文件路径，实际使用时应替换为传入的file_path参数
    dir = r"D:\处理的数据集\TOR\tor-time-burst-50-10\163-0.txt"
    # dir = r"D:\处理的数据集\TOR\header-500-100\163-0.txt"
    
    try:
        with open(dir, 'r') as file:
            res_list = []
            req_list = []
            
            for line in file:
                # 移除行尾的换行符
                line = line.strip()
                # 使用制表符分割列数据
                columns = line.split('\t')  # 假设使用制表符分隔列
                columns = columns[:-1]  # 移除最后一列（标签列）
                
                # 将列分为请求列和响应列
                req_columns = columns[:(len(columns) // 2)]
                res_columns = columns[(len(columns) // 2):]
                
                # 处理请求列数据
                for i, data in enumerate(req_columns):
                    if data == '':
                        sub = [0]
                    else:
                        sub = data.strip().split(' ')
                    
                    # 确保数据长度为5
                    if len(sub) < 5:
                        for _ in range(5 - len(sub)):
                            sub.append(0)
                    else:
                        sub = sub[:5]
                    req_list.extend(sub)

                # 处理响应列数据
                for i, data in enumerate(res_columns):
                    if data == '':
                        sub = [0]
                    else:
                        sub = data.strip().split(' ')
                    
                    # 确保数据长度为5
                    if len(sub) < 5:
                        for _ in range(5 - len(sub)):
                            sub.append(0)
                    else:
                        sub = sub[:5]
                    res_list.extend(sub)
        
        return req_list, res_list
    except FileNotFoundError:
        print(f"File not found.")
        return [], []

if __name__ == '__main__':
    req, res = read_txt("")
    print(req, '\n', res)
    make_res(res)
    make_req(req)