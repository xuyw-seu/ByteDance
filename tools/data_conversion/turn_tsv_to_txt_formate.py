#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSV/TXT Data Conversion and Preprocessing Tool

This script handles conversion, cleaning, and preprocessing of TSV/TXT formatted network traffic data.
It prepares data for machine learning model training by formatting, filtering, and organizing.

Core Modules:
- Data Conversion: Extract, merge, and reorder columns
- Data Cleaning: Remove outliers and map values
- Data Sampling: Extract subsets and shuffle data
- Data Validation: Check file dimensions and values

Main Functions:
- extract_columns_to_txt: Extract specified columns and format
- delete_rows_with_values_above_threshold: Remove rows with outliers
- replace_column_values: Map labels to numerical values
- shuffle_lines_in_file: Randomize data order

Dependencies: os, random, time, numpy, pandas, csv
"""
import os
import random
import time

import numpy as np
import pandas as pd
import csv
import numpy as np


def extract_columns_to_txt(input_file, output_file, column_indices):
    try:
        # 打开输入文件以读取
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 提取指定列的数据
        extracted_data = []
        for line in lines:
            parts = line.strip().split('\t')
            selected_columns = [parts[i][:800] for i in column_indices]
            index = 0
            for j in selected_columns:
                if len(j) > 25:
                    spaced_str = ' '.join(j[i:i + 2] for i in range(0, len(j), 2))
                    selected_columns[index] = spaced_str
                    index += 1
            extracted_data.append('\t'.join(selected_columns))

        # 打开输出文件以写入
        with open(output_file, 'w', encoding='utf-8') as f:
            for data in extracted_data:
                f.write(data + '\n')

        print(f"成功提取并保存列 {column_indices} 到 {output_file}")
    except Exception as e:
        print(f"提取并保存数据时发生错误: {e}")

def get_file_dimensions(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            num_lines = len(lines)
            if num_lines > 0:
                num_columns = len(lines[0].strip().split('\t'))  # 假定列之间使用制表符分隔
            else:
                num_columns = 0
        return num_lines, num_columns
    except Exception as e:
        print(f"无法读取文件维度: {e}")
        return 0, 0


def merge_columns_and_keep_tab(input_file, output_file, merge_column_indices):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        merged_data = []
        for line in lines:
            parts = line.strip().split('\t')  # 假定列之间使用制表符分隔
            merged_column = ' '.join([parts[i] for i in merge_column_indices])
            other_columns = [parts[i] for i in range(len(parts)) if i not in merge_column_indices]
            merged_line = '\t'.join(other_columns + [merged_column])
            merged_data.append(merged_line)

        with open(output_file, 'w', encoding='utf-8') as file:
            for data in merged_data:
                file.write(data + '\n')

        print(f"成功合并列 {merge_column_indices} 并保存到 {output_file}")
    except Exception as e:
        print(f"合并并保存数据时发生错误: {e}")


def swap_columns(input_file, output_file, column_indices):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        swapped_data = []
        for line in lines:
            parts = line.strip().split('\t')  # 假定列之间使用制表符分隔
            # 重新排列列的顺序
            reordered_columns = [parts[i] for i in column_indices]
            swapped_line = '\t'.join(reordered_columns)
            swapped_data.append(swapped_line)

        with open(output_file, 'w', encoding='utf-8') as file:
            for data in swapped_data:
                file.write(data + '\n')

        print(f"成功交换列 {column_indices} 的顺序并保存到 {output_file}")
    except Exception as e:
        print(f"交换列顺序并保存数据时发生错误: {e}")

def replace_column_values(input_file, output_file, column_index, value_replacements):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        replaced_data = []
        for line in lines:
            parts = line.strip().split('\t')  # 假定列之间使用制表符分隔
            if 0 <= column_index < len(parts):
                column_value = parts[column_index]
                if column_value in value_replacements:
                    parts[column_index] = value_replacements[column_value]
            replaced_line = '\t'.join(parts)
            replaced_data.append(replaced_line)

        with open(output_file, 'w', encoding='utf-8') as file:
            for data in replaced_data:
                file.write(data + '\n')

        print(f"成功替换列 {column_index} 的值并保存到 {output_file}")
    except Exception as e:
        print(f"替换列值并保存数据时发生错误: {e}")

def extract_first_n_lines(input_file, output_file, n):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        first_n_lines = lines[:n]

        with open(output_file, 'w', encoding='utf-8') as file:
            for line in first_n_lines:
                file.write(line)

        print(f"成功提取前 {n} 行并保存到 {output_file}")
    except Exception as e:
        print(f"提取并保存数据时发生错误: {e}")

def extract_columns_from_file(file_name, column_indices):
    try:
        with open(file_name, 'r') as file:
            for line in file:
                # 分割行中的值（假设以空格或制表符分隔）
                values = line.strip().split('\t')
                if values[0] == "packet1":
                    continue
                extracted_values = [values[i] for i in column_indices]
                #print('\t'.join(extracted_values))  # 以制表符分隔打印提取的值
                for i in extracted_values:
                    if int(i) > 1500:
                        print("99999999999", i)
                    if int(i) == 0 :
                        print("888888888",i)

    except FileNotFoundError:
        print(f"File '{file_name}' not found.")


def delete_rows_with_values_above_threshold(input_file, output_file, column_indices):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # 分割行中的值（假设以空格或制表符分隔）
                values = line.strip().split('\t')
                should_delete = False

                # 检查每个指定列
                for index in column_indices:
                    if index < len(values):
                        try:
                            value = int(values[index])
                            if value > 2000 or value == 0:
                                should_delete = True
                                break
                        except ValueError:
                            # 如果值无法转换为整数，跳过该行
                            continue

                if not should_delete:
                    outfile.write(line)

    except FileNotFoundError:
        print(f"File '{input_file}' not found.")

def shuffle_lines_in_file(input_file, output_file):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            lines = infile.readlines()
            random.shuffle(lines)
            outfile.writelines(lines)

    except FileNotFoundError:
        print(f"File '{input_file}' not found.")

if __name__ == '__main__':
    input = r"C:\Users\Administrator\Desktop\tt\data.tsv"
    output = r"C:\Users\Administrator\Desktop\tt\data_1.txt"
    output2 = r"C:\Users\Administrator\Desktop\tt\data_2.txt"
    output3 = r"C:\Users\Administrator\Desktop\tt\data_3.txt"
    output4 = r"C:\Users\Administrator\Desktop\tt\data_4.txt"
    output5 = r"C:\Users\Administrator\Desktop\tt\data_5.txt"
    output6 = r"C:\Users\Administrator\Desktop\tt\data_6.txt"
    output7 = r"C:\Users\Administrator\Desktop\tt\data_7.txt"
    column_list = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0]
    # extract_columns_to_txt(input,output,column_list)
    #
    # m_list = [10, 11, 12, 13, 14, 15, 16, 17 ,18 , 19]
    # merge_columns_and_keep_tab(output6,output2,m_list)
    #
    # s_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10]
    # swap_columns(output2,output3,s_list)
    #
    # lable_list = os.listdir(r"D:\处理的数据集\self-tor-24-session")
    # map_dict = { key: str(value) for value, key in enumerate(lable_list)}
    # print(map_dict.keys())
    #
    #
    # replace_column_values(output3,output4,11,map_dict)
    #
    # extract_first_n_lines(output4,output5,5000)

    # delete_rows_with_values_above_threshold(output,output6,[10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    # extract_columns_from_file(output5,[10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    # shuffle_lines_in_file(output4,output7)
    #
    # num_lines, num_columns = get_file_dimensions(output7)
    # print(f"文件行数: {num_lines}")
    # print(f"文件列数: {num_columns}")