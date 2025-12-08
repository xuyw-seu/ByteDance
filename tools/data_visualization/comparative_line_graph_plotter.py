#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比折线图绘制工具
用于绘制两个文本文件数据的对比折线图，特别适用于网络流量相关的比例数据
主要功能：
1. 从两个文本文件中读取浮点数值数据
2. 绘制对比折线图，支持自定义样式
3. 设置专业的图表格式，包括字体、大小、颜色等
4. 保存图表为PDF格式，适用于学术论文
5. 支持LaTeX数学符号（如ρ）
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'

def plot_two_txt_files(file1_path, file2_path):
    """
    从两个仅包含一列 float 数值的 txt 文件中各读取前 500 行，
    并在同一个图中绘制两条折线，分别用红色和蓝色表示。

    :param file1_path: 第一个 txt 文件路径（蓝色）
    :param file2_path: 第二个 txt 文件路径（红色）
    """

    def read_floats(file_path, max_lines=500):
        values = []
        with open(file_path, 'r') as file:
            for _ in range(max_lines):
                line = file.readline()
                if not line:
                    break
                try:
                    values.append(float(line.strip()))
                except ValueError:
                    print(f"跳过无效行：{line.strip()}")
        return values

    y1 = read_floats(file1_path)
    y2 = read_floats(file2_path)

    x1 = list(range(len(y1)))
    x2 = list(range(len(y2)))

    plt.figure(figsize=(10, 5.7), facecolor='white')

    plt.plot(x1, y1, color='blue', label='ByteDance', linewidth=1.5)
    plt.plot(x2, y2, color='red', label='ByteDance w/o PDGC', linewidth=1.5)

    # 设置坐标轴字体大小
    plt.xlabel('Train Iterations', fontsize=21)
    plt.ylabel(r'$\rho$', fontsize=21)

    # 设置刻度字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # 设置图例字体大小
    # 设置图例：黑色直角边框
    legend = plt.legend(fontsize=21, fancybox=False, edgecolor='black')
    legend.get_frame().set_linewidth(1.5)

    # 设置网格和边距    plt.grid(True)
    plt.grid(False)  # 禁用网格线（去掉底纹）
    plt.tight_layout()

    # 保存 PDF 图像
    plt.savefig(r"C:\Users\Administrator\Desktop\BD\ratio.pdf", format='pdf', dpi=300)
    plt.show()
    plt.close()

if __name__ == '__main__':
    plot_two_txt_files(r"E:\论文复现\多模态-PEAN\ByteDance\task2\tls1.3\result\gradient_ratio_tls_nn_2_pdgc.txt",
                       r"E:\论文复现\多模态-PEAN\ByteDance\task2\tls1.3\result\gradient_ratio_tls_nn_2.txt")

