#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络流量请求响应比例计算器
用于计算网络流量数据中请求和响应的字节比例
主要功能：
1. 计算单个文件前n行数据中请求和响应的字节比例
2. 计算目录中所有文件的平均请求和响应字节比例
3. 支持不同行数范围的比例计算
4. 将结果写入输出文件
5. 支持多种网络流量数据类型（QUIC, Tor, TLS等）
"""
import os
import time


def calculate_ratio(file_path, n):
    """
    计算单个文件前n行数据中请求和响应的字节比例
    :param file_path: 文件路径
    :param n: 读取的行数
    :return: (请求字节比例, 响应字节比例)
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()[:n]

    count_positive = 0  # 请求字节数
    count_negative = 0  # 响应字节数
    total_len = 0       # 总字节数

    for line in lines:
        _, second_column = map(float, line.strip().split('\t'))
        total_len += abs(second_column)
        if second_column > 0:
            count_positive += abs(second_column)
        elif second_column < 0:
            count_negative += abs(second_column)

    total_lines = len(lines)

    if total_lines > 0:
        ratio_positive = count_positive / total_len
        ratio_negative = count_negative / total_len
        return ratio_positive, ratio_negative
    else:
        return 0, 0


def calculate_average_ratio(folder_path, n, output_file):
    """
    计算目录中所有文件的平均请求和响应字节比例
    :param folder_path: 目录路径
    :param n: 读取的行数
    :param output_file: 输出文件路径
    :return: (平均请求字节比例, 平均响应字节比例)
    """
    total_ratio_positive = 0
    total_ratio_negative = 0
    file_count = 0
    i = 0
    
    with open(output_file, 'a') as output:
        for filename in os.listdir(folder_path):
            print("No {} ， name : {}".format(i, filename))
            i += 1
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)

                ratio_positive, ratio_negative = calculate_ratio(file_path, n)

                total_ratio_positive += ratio_positive
                total_ratio_negative += ratio_negative
                file_count += 1

        if file_count > 0:
            average_ratio_positive = total_ratio_positive / file_count
            average_ratio_negative = total_ratio_negative / file_count

            # 写入平均比例到输出文件
            output.write(f"{n}\tAverage\t{average_ratio_positive:.4f}\t{average_ratio_negative:.4f}\n")

            return average_ratio_positive, average_ratio_negative
        else:
            return 0, 0

if __name__ == '__main__':
    # 示例使用
    file_path = r"D:\处理的数据集\QUIC\test\total"  # 数据文件目录
    save_path = r"D:\实验绘图\数据\tor-len.txt"    # 结果输出文件
    n_list = [20, 50, 100, 500, 1000, 2000, 4000]    # 不同行数范围
    
    a = time.time()
    for i, value in enumerate(n_list):
        print("第 {} 个， 时间为 : {}".format(i, time.time() - a))
        pos, neg = calculate_average_ratio(file_path, value, save_path)
        print(f"请求比例: {pos:.4f}, 响应比例: {neg:.4f}")
