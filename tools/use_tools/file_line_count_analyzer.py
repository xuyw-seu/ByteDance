#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件行数统计分析工具
用于统计和分析目录中文本文件的行数分布
主要功能：
1. 统计单个文件的行数
2. 按照指定的行数范围对目录中的文件进行分类统计
3. 计算各类行数范围的文件数量和占比
4. 输出统计结果，包括行数范围、文件个数和比例
5. 支持不同类型网络流量数据文件的行数分析
"""
import os
import glob


def count_lines(file_path):
    """
    统计单个文件的行数
    :param file_path: 文件路径
    :return: 文件行数
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for line in file)


def count_files_within_ranges(directory, line_ranges):
    """
    按照行数范围统计目录中的文件数量
    :param directory: 目录路径
    :param line_ranges: 行数范围列表，格式为[(min1, max1), (min2, max2), ...]
    :return: 行数范围统计结果字典
    """
    txt_files = glob.glob(os.path.join(directory, '*.txt'))

    counts = {range_: 0 for range_ in line_ranges}
    counts["大于4000"] = 0

    for file_path in txt_files:
        num_lines = count_lines(file_path)

        if num_lines > 15000:
            counts["大于4000"] += 1
        else:
            for line_range in line_ranges:
                if line_range[0] < num_lines <= line_range[1]:
                    counts[line_range] += 1

    return counts


def print_results(counts, total_files):
    """
    输出行数统计结果
    :param counts: 行数范围统计结果字典
    :param total_files: 总文件数量
    """
    print("文件行数范围\t文件个数\t比例")
    print("-----------------------------")
    for line_range, file_count in counts.items():
        if isinstance(line_range, tuple):
            range_str = f"{line_range[0]}-{line_range[1]}"
        else:
            range_str = line_range
        percentage = (file_count / total_files) * 100 if total_files > 0 else 0
        print(f"{range_str}\t\t{file_count}\t\t{percentage:.2f}%")


if __name__ == "__main__":
    # tls1.3 < 100 = 92.27%, tor > 2000 = 69.89%, trojan < 500 = 61.12%
    # directory_path = r"D:\处理的数据集\TLS1.3\total"  # 替换为你的目录路径
    # directory_path = r"D:\处理的数据集\TOR\tor-24-double-column-less-2000"
    # directory_path = r"D:\处理的数据集\trojan\Trojan-data-clean-eq"
    # directory_path = r"D:\处理的数据集\QUIC\test\total"
    directory_path = r"D:\处理的数据集\mobile_2025\src_data_RAW_2"
    line_ranges = [(0, 10), (10, 20), (20, 50), (50, 100), (100, 150), (150, 200), (200, 250), 
                   (250, 300), (300, 350), (350, 400), (400, 450), (450, 500), (500, 1000),
                   (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000, 3500), 
                   (3500, 4000), (4000, 5000), (5000, 6000), (6000, 7000), (7000, 8000),
                   (8000, 9000), (9000, 10000), (10000, 11000), (11000, 12000), (12000, 13000), 
                   (13000, 14000)]

    counts = count_files_within_ranges(directory_path, line_ranges)
    total_files = sum(counts.values())
    print_results(counts, total_files)
