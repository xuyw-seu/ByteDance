#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
突发检测增强版网络流量时间序列处理器
用于处理网络流量的时间序列数据，专注于时间序列突发检测和时间突变分析
主要功能：
1. 将二维列表转换为单行文本格式
2. 支持时间序列突发检测和分析（find_burst_end_indices函数）
3. 支持时间突变检测（find_timestamp_transitions函数）
4. 多种时间序列处理方法（基于突发、基于时间、基于序列长度等）
5. 提取请求和响应方向的流量特征
6. 统计和分析时间序列长度
7. 合并和打乱文本文件
8. 支持区间统计和元素计数
9. 支持多种网络流量类型的处理（mobile, TLS, Tor, Trojan, QUIC等）
10. 支持时间序列分组和合并
"""
import os
import time
import random


def write_2d_list_as_single_line(data, time_data, lable=None, filename=None):
    try:
        if lable.isdigit():
            with open(filename, 'w') as file:
                # len_list = [100,100,100,100,100,100,100,100,100,100]
                for row in data:
                    # 使用空格分隔列内元素，然后使用制表符分隔列之间
                    row_str = " ".join(str(abs(column)) for column in row)
                    file.write(row_str + '\t')

                # r = 50 / time_data[0][1]
                for row1 in time_data:
                    row_str1 = " ".join(str(abs(column)) for column in row1)
                    file.write(row_str1 + '\t')

                # row_len = " ".join(str(round(i * r)) for i in time_data)
                # file.write(row_len + '\t' + lable)
                file.write(lable)
        else:
            print("999999999999999999999999")

    except Exception as e:
        print(f"Error writing to file: {str(e)}")

def write_2d_list_as_single_line_single(data, time_data, lable=None, filename=None):
    try:
        if lable.isdigit():
            with open(filename, 'w') as file:
                # len_list = [100,100,100,100,100,100,100,100,100,100]
                for row in data:
                    # 使用空格分隔列内元素，然后使用制表符分隔列之间
                    row_str = " ".join(str(abs(column)) for column in row)
                    file.write(row_str + '\t')

                # r = 50 / time_data[0][1]
                for row1 in time_data:
                    row_str1 = " ".join(str(abs(column)) for column in row1)
                    file.write(row_str1 + '\t')

                # row_len = " ".join(str(round(i * r)) for i in time_data)
                # file.write(row_len + '\t' + lable)
                file.write(lable)
        else:
            print("999999999999999999999999")

    except Exception as e:
        print(f"Error writing to file: {str(e)}")

def process_cell_file_single(file_path, lable, save_dir, max_pcap_num, column_num):
    if os.path.exists(save_dir):
        return
    # 替换成你的文件路径
    data_list = []
    time_list = []
    new_data_list_1 = []
    new_data_list_0 = []
    sub_list_1 = []
    sub_list_0 = []
    sub_list = []
    new_data_list = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')  # 使用制表符分隔列
                if len(columns) >= 2:
                    # 获取第二列数据，并转化为整数
                    data = int(columns[1])
                    data_list.append(data)
                    time_list.append(float(columns[0]))

            rows = column_num
            if len(data_list) < column_num:
                print("less 10")
            if len(data_list) > max_pcap_num:
                num_elements_per_row = int(max_pcap_num / rows)
                unuesd_elements = max_pcap_num - rows * num_elements_per_row
                time_plot_list = [num_elements_per_row + unuesd_elements if _ == (rows - 1) else num_elements_per_row
                                  for _
                                  in range(rows)]
            else:
                num_elements_per_row = int(max_pcap_num / rows)
                time_plot_list = []
                last_len = len(data_list)
                for i in range(rows):
                    if last_len < num_elements_per_row:
                        time_plot_list.append(last_len)
                        last_len = 0
                    else:
                        time_plot_list.append(num_elements_per_row)
                        last_len = last_len - num_elements_per_row
                # time_plot_list = [num_elements_per_row + unuesd_elements if _ == (rows - 1) else num_elements_per_row
                #                   for _
                #                   in range(rows)]
            # 遍历 plot_list 中的值
            index = 0

            for num_elements_per_row in time_plot_list:
                # 从 time_list 中提取相应数量的元素并添加到 result 中
                for i in range(num_elements_per_row):
                    sub_list.append(data_list[index + i] + 1500)
                    sub_list_1.append(0)

                dev_3 = sub_list[:]
                dev_1 = sub_list_1[:]
                new_data_list_1.append(dev_1)
                new_data_list.append(dev_3)
                sub_list.clear()
                sub_list_1.clear()
                index += num_elements_per_row
            write_2d_list_as_single_line(new_data_list, new_data_list_1, lable, save_dir)

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")

def process_cell_file_no_t(file_path, lable, save_dir, max_pcap_num, column_num):
    if os.path.exists(save_dir):
        return
    # 替换成你的文件路径
    data_list = []
    time_list = []
    new_data_list_1 = []
    new_data_list_0 = []
    sub_list_1 = []
    sub_list_0 = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')  # 使用制表符分隔列
                if len(columns) >= 2:
                    # 获取第二列数据，并转化为整数
                    data = int(columns[1])
                    data_list.append(data)
                    time_list.append(float(columns[0]))

            rows = column_num
            if len(data_list) < column_num:
                print("less 10")
            if len(data_list) > max_pcap_num:
                num_elements_per_row = int(max_pcap_num / rows)
                unuesd_elements = max_pcap_num - rows * num_elements_per_row
                time_plot_list = [num_elements_per_row + unuesd_elements if _ == (rows - 1) else num_elements_per_row
                                  for _
                                  in range(rows)]
            else:
                # num_elements_per_row = int(len(data_list) / rows)
                # unuesd_elements = len(data_list) - rows * num_elements_per_row
                # time_plot_list = [num_elements_per_row + unuesd_elements if _ == (rows - 1) else num_elements_per_row
                #                   for _
                #                   in range(rows)]
                num_elements_per_row = int(max_pcap_num / rows)
                time_plot_list = []
                last_len = len(data_list)
                for i in range(rows):
                    if last_len < num_elements_per_row:
                        time_plot_list.append(last_len)
                        last_len = 0
                    else:
                        time_plot_list.append(num_elements_per_row)
                        last_len = last_len - num_elements_per_row
                # time_plot_list = [num_elements_per_row + unuesd_elements if _ == (rows - 1) else num_elements_per_row
                #                   for _
                #                   in range(rows)]
            # 遍历 plot_list 中的值
            index = 0

            for num_elements_per_row in time_plot_list:
                # 从 time_list 中提取相应数量的元素并添加到 result 中
                for i in range(num_elements_per_row):
                    if data_list[index + i] < 0:
                        sub_list_0.append(data_list[index + i])
                    else:
                        sub_list_1.append(data_list[index + i])
                dev_1 = sub_list_1[:]
                dev_2 = sub_list_0[:]
                new_data_list_1.append(dev_1)
                new_data_list_0.append(dev_2)
                sub_list_0.clear()
                sub_list_1.clear()
                index += num_elements_per_row
            # data_list = [data_list[i:i + num_elements_per_row] for i in
            #              range(0, len(data_list) - unuesd_elements, num_elements_per_row)]
            # time_list = [time_list[i:i + num_elements_per_row] for i in
            #              range(0, len(time_list) - unuesd_elements, num_elements_per_row)]

            write_2d_list_as_single_line(new_data_list_1, new_data_list_0, lable, save_dir)

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")



def find_burst_end_indices(lst):

    burst_end_indices = []
    i = 0
    while i < len(lst):
        if lst[i] > 0:  # 找到正数（请求）作为 burst 的起始
            response_index = None

            has_neg = False
            if i + 1 == len(lst):
                response_index = len(lst) - 1

            # 从当前位置开始查找对应的负数（响应）
            for j in range(i + 1, len(lst)):
                if j == len(lst) - 1:
                    response_index = j
                if int(lst[j]) < 0:
                    has_neg = True
                if not has_neg:
                    continue
                if has_neg and int(lst[j]) > 0:
                    response_index = j - 1
                    break

            # 如果找到对应的负数，将结束索引添加到结果中
            if response_index is not None:
                burst_end_indices.append(response_index)
                i = response_index + 1  # 跳过已找到的 burst
            else:
                i += 1  # 如果没有找到对应的负数，继续查找下一个正数

        else:
            i += 1


    n_list = []
    for g in range(len(burst_end_indices)):
        if g == 0:
            n_list.append(int(burst_end_indices[g]) + 1)
        else:
            n_list.append(int(burst_end_indices[g]) - int(burst_end_indices[g - 1]))
    return n_list



def process_cell_file_burst(file_path, lable, save_dir, max_pcap_num, column_num):
    if os.path.exists(save_dir):
        return
    # 替换成你的文件路径
    data_list = []
    time_list = []
    new_data_list_1 = []
    new_data_list_0 = []
    sub_list_1 = []
    sub_list_0 = []
    plot = max_pcap_num // column_num

    time_plot_list = [0 for q in range(column_num)]
    try:
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')  # 使用制表符分隔列
                if len(columns) >= 2:
                    if len(data_list) > max_pcap_num:
                        break
                    # 获取第二列数据，并转化为整数
                    data = int(columns[1])
                    data_list.append(data)
                    time_list.append(float(columns[0]))

            burst_plot_list = find_burst_end_indices(data_list)
            burst_num = len(burst_plot_list)

            if burst_num > column_num:
                time_plot_list = merge_min_elements(burst_plot_list,column_num)
            else:
                for i in range(len(burst_plot_list)):
                    time_plot_list[i] = burst_plot_list[i]

            # 遍历 plot_list 中的值
            index = 0
            for num_elements_per_row in time_plot_list:
                # 从 time_list 中提取相应数量的元素并添加到 result 中
                for i in range(num_elements_per_row):
                    if data_list[index + i] < 0:
                        sub_list_0.append(data_list[index + i])
                    else:
                        sub_list_1.append(data_list[index + i])
                dev_1 = sub_list_1[:]
                dev_2 = sub_list_0[:]
                new_data_list_1.append(dev_1)
                new_data_list_0.append(dev_2)
                sub_list_0.clear()
                sub_list_1.clear()
                index += num_elements_per_row

            write_2d_list_as_single_line(new_data_list_1, new_data_list_0, lable, save_dir)

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")

def find_timestamp_transitions(timestamp_list, column_num, threshold=1.2):
    transitions = []

    current_length = 0

    for i in range(1, len(timestamp_list)):
        if i == 1:
            time_diff = timestamp_list[i]
        else:
            if timestamp_list[i - 1] - timestamp_list[i - 2] == 0:
                time_diff = 1
            else:
                time_diff = (timestamp_list[i] - timestamp_list[i - 1]) / (timestamp_list[i - 1] - timestamp_list[i - 2])

        if time_diff > threshold:
            # 如果时间差大于阈值，表示发生了时间突变
            transitions.append(i)
        else:
            # 如果时间差小于等于阈值，增加当前突变序列的长度
            current_length += 1
    n_list = []
    if len(transitions) != 0:
        for g in range(len(transitions)):
            if g == 0:
                n_list.append(int(transitions[g]))
            else:
                n_list.append(int(transitions[g]) - int(transitions[g - 1]))
        n_list.append(len(timestamp_list) - transitions[-1])
    else:
        n_list.append(len(timestamp_list))
    # print("time transition :", transitions)
    # print("time transition :", n_list)
    if len(n_list) > column_num:
        num_elements_per_row = int(len(timestamp_list) / column_num)
        unuesd_elements = len(timestamp_list) - column_num * num_elements_per_row
        n_list = [num_elements_per_row + unuesd_elements if _ == (column_num - 1) else num_elements_per_row
                          for _
                          in range(column_num)]
    return n_list

def get_sub_plot(time_list, time_burst_list, n):
    scale_list = []
    total_list = []
    for i in time_burst_list:
        data = round(i * n // sum(time_burst_list))
        if data == 0:
            data = 1
        scale_list.append(data)
    last_group = n - sum(scale_list)
    if last_group < 0:
        max_index = scale_list.index(max(scale_list))
        scale_list[max_index] += last_group
    else:
        min_index = scale_list.index(min(scale_list))
        scale_list[min_index] += last_group
    g = 0
    for j, v in enumerate(scale_list):
        if v <= 0:
            continue
        time_plot = (time_list[sum(time_burst_list[0:j + 1]) - 1] - time_list[sum(time_burst_list[0:j])]) / v
        # print(time_plot,time_list[sum(time_burst_list[0:j + 1]) - 1], time_list[sum(time_burst_list[0:j])], v)
        time_plot_list = [0 for q in range(v)]
        sub_time_list = [(l - time_list[sum(time_burst_list[0:j])]) for l in time_list[g:]]
        g += time_burst_list[j]
        for index, value in enumerate(sub_time_list):
            for k in range(v):
                if k * time_plot <= value < time_plot * (k + 1):
                    time_plot_list[k] += 1
            if value == time_plot * v:
                time_plot_list[v - 1] += 1
        total_list.extend(time_plot_list)
        # print(sub_time_list)

    # print(scale_list,time_burst_list,total_list)

    if all(x == scale_list[0] for x in scale_list):
        return time_burst_list
    else:
        return total_list


def process_cell_file_time_burst(file_path, lable, save_dir, max_pcap_num, column_num):
    if os.path.exists(save_dir):
        return
    # 替换成你的文件路径
    data_list = []
    time_list = []
    new_data_list_1 = []
    new_data_list_0 = []
    sub_list_1 = []
    sub_list_0 = []
    plot = max_pcap_num // column_num

    time_plot_list = [0 for q in range(column_num)]
    max_len = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')  # 使用制表符分隔列
                if len(columns) >= 2:
                    if len(data_list) > max_pcap_num:
                        break
                    # 获取第二列数据，并转化为整数
                    data = int(columns[1])
                    data_list.append(data)
                    time_list.append(float(columns[0]))

            sequence_lengths = find_timestamp_transitions(time_list,column_num,threshold=3)

            time_plot_list = get_sub_plot(time_list,sequence_lengths,column_num)
            print(time_plot_list)

            max_len.extend(time_plot_list)

            # 遍历 plot_list 中的值
            index = 0
            for num_elements_per_row in time_plot_list:
                # 从 time_list 中提取相应数量的元素并添加到 result 中
                for i in range(num_elements_per_row):
                    # print(len(data_list), index + i,time_plot_list)
                    if data_list[index + i] < 0:
                        sub_list_0.append(data_list[index + i])
                    else:
                        sub_list_1.append(data_list[index + i])
                dev_1 = sub_list_1[:]
                dev_2 = sub_list_0[:]
                new_data_list_1.append(dev_1)
                new_data_list_0.append(dev_2)
                sub_list_0.clear()
                sub_list_1.clear()
                index += num_elements_per_row

            write_2d_list_as_single_line(new_data_list_1, new_data_list_0, lable, save_dir)
        return max_len
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")



def count_elements_in_intervals(data_list, intervals):
    """
    统计一个列表中元素在各区间的个数，包括大于最后一个区间的元素个数

    参数：
    - data_list: 包含int类型元素的列表
    - intervals: 区间的上限值，以列表形式提供

    返回值：
    一个字典，包含各个区间和其内元素个数的映射关系，以及大于最后一个区间的元素个数
    """
    result = {str(interval): 0 for interval in intervals}
    result["greater_than_last_interval"] = 0
    last_interval = None

    for element in data_list:
        inside_interval = False
        for interval in intervals:
            if last_interval is None:
                if element <= interval:
                    result[str(interval)] += 1
                    inside_interval = True
                    break
            else:
                if last_interval < element <= interval:
                    result[str(interval)] += 1
                    inside_interval = True
                    break
            last_interval = interval

        if not inside_interval and last_interval is not None and element > last_interval:
            result["greater_than_last_interval"] += 1

    return result

def process_cell_file_t(file_path, lable, save_dir, max_pcap_num, column_num):
    if os.path.exists(save_dir):
        return
    # 替换成你的文件路径
    data_list = []
    time_list = []
    new_data_list_1 = []
    new_data_list_0 = []
    sub_list_1 = []
    sub_list_0 = []

    time_plot_list = [0 for q in range(column_num)]
    try:
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')  # 使用制表符分隔列
                if len(columns) >= 2:
                    # 获取第二列数据，并转化为整数
                    data = int(columns[1])
                    data_list.append(data)
                    time_list.append(float(columns[0]))

            if len(data_list) > max_pcap_num:
                time_plot = time_list[max_pcap_num] / column_num
            else:
                time_plot = time_list[len(time_list) - 1] / column_num


            for index, value in enumerate(time_list):
                for j in range(column_num):
                    if j * time_plot <= value < time_plot * (j + 1):
                        time_plot_list[j] += 1
                if value == time_plot*column_num:
                    time_plot_list[column_num - 1] += 1

            # 遍历 plot_list 中的值
            index = 0
            for num_elements_per_row in time_plot_list:
                # 从 time_list 中提取相应数量的元素并添加到 result 中
                for i in range(num_elements_per_row):
                    if data_list[index + i] < 0:
                        sub_list_0.append(data_list[index + i])
                    else:
                        sub_list_1.append(data_list[index + i])
                dev_1 = sub_list_1[:]
                dev_2 = sub_list_0[:]
                new_data_list_1.append(dev_1)
                new_data_list_0.append(dev_2)
                sub_list_0.clear()
                sub_list_1.clear()
                index += num_elements_per_row

            write_2d_list_as_single_line(new_data_list_1, new_data_list_0, lable, save_dir)

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")




def get_text_len(file_path):
    try:
        with open(file_path, 'r') as file:
            line_num = 0
            for line in file:
                line_num += 1
                c_l = 0
                # 移除行尾的换行符
                # 使用制表符或其他分隔符分割列数据
                columns = line.split('\t')  # 假设使用制表符分隔列
                for c in columns:
                    c_l += 1
            print("row number is :{}, column number is :{}".format(line_num, c_l))
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")


def merge_txt_files_to_single_file(source_dir, output_file):
    merged_content = ""
    print("start merge >>>>>")
    i = 0
    for filename in os.listdir(source_dir):
        # print("merge {} ".format(i))
        i += 1

        file_path = os.path.join(source_dir, filename)
        with open(file_path, 'r') as file:
            content = file.read()
            merged_content += content + "\n"

    with open(output_file, 'w') as output:
        output.write(merged_content)


def shuffle_lines_in_file(input_file, output_file):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            lines = infile.readlines()
            random.shuffle(lines)
            outfile.writelines(lines)

    except FileNotFoundError:
        print(f"File '{input_file}' not found.")


def write_numbers_to_file(start, end, output_file):
    with open(output_file, 'w') as file:
        for num in range(start, end + 1):
            file.write(str(num) + '\n')



def find_longest_sequence_length(file_path):
    # 初始化最长序列长度为0
    max_sequence_length = 0

    # 打开文件并逐行读取
    with open(file_path, 'r') as file:
        for line in file:
            # 分割每一行的列，并遍历每一列
            columns = line.strip().split('\t')
            for column in columns:
                # 将每一列的元素转换为整数列表
                int_sequence = column.split(' ')
                # 更新最长序列长度
                max_sequence_length = max(max_sequence_length, len(int_sequence))
    print("max len list is:{}".format(max_sequence_length))
    return max_sequence_length

def merge_min_elements(original_list, n):
    m = len(original_list)

    if m <= n:
        # 如果原始列表元素个数小于等于目标列表元素个数，直接返回原始列表的拷贝
        return original_list.copy()

    # 找到最小的 m-n 个元素的索引
    min_indices = sorted(range(m), key=lambda i: original_list[i])[:m-n]

    # 构建新的列表，将最小的 m-n 个元素合并到左边或右边的元素上
    new_list = original_list.copy()
    for i in sorted(min_indices, reverse=True):
        if i == 0:
            # 如果最小的元素是第一个数，合并到右边
            new_list[i] += new_list.pop(i + 1)
        else:
            # 合并到左边
            new_list[i - 1] += new_list.pop(i)
    return new_list

def test():
    file_path = r"E:\数据集\80_new\0-0"  # 替换成你的文件路径
    dir_path = r"D:\处理的数据集\mobile_app\mobile_eq"
    save_path = r"D:\处理的数据集\mobile_app\mobile_burst_eq_100-10"
    # dir_path = r"D:\处理的数据集\QUIC\test\total"
    # save_path = r"D:\处理的数据集\QUIC\test\quic-2000"
    # dir_path = r"D:\处理的数据集\TLS1.3\total"
    # save_path = r"D:\处理的数据集\TLS1.3\TLS1.3-2000"
    if not os.path.exists(save_path):
        # 如果文件夹不存在，使用 os.makedirs() 创建它
        os.makedirs(save_path)
        print(f"文件夹 {save_path} 已创建")
    else:
        print(f"文件夹 {save_path} 已存在")
    t_list = os.listdir(dir_path)
    num = 0
    a = time.time()
    time_list = []
    # lable_list1 = os.listdir(r"D:\数据集\mKCP\mKCP_total")
    # lable_list2 = os.listdir(r"D:\数据集\mKCP\mKCP_total2")
    # lable_list3 = os.listdir(r"D:\数据集\mKCP\mKCP_total3")
    # lable_list = lable_list1 + lable_list2 + lable_list3
    # lable_list = os.listdir(r"D:\数据集\self_tor_24")
    # lable_list = os.listdir(r"D:\数据集\TLS1.3")
    # lable_list = os.listdir(r"D:\数据集\QUIC Dataset\pretraining\pretraining")
    # lable_list = os.listdir(r"D:\数据集\trojan-clean")

    lable_list = os.listdir(r"D:\数据集\mobile app\22-cleaned")
    lable_dict = {item: str(index) for index, item in enumerate(lable_list)}
    # lable_dict = {"GoogleDoc": "0",
    #               "GoogleDrive": "1",
    #               "GoogleMusic": "2",
    #               "GoogleSearch": "3",
    #               "Youtube": "4", }

    l_n = 0
    #
    for p in t_list:
        num += 1
        t_path = os.path.join(dir_path, p)
        lable = lable_dict[p.split('-')[0] + "-" + p.split('-')[1]]
        # lable = p.split('-')[0]
        s_dir = os.path.join(save_path, p)
        # 5, 30, 50, 100, 500
        process_cell_file_burst(t_path, lable, s_dir,100,10)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))

    # last_txt = r"D:\处理的数据集\trojan\trojan-eq-4000-50.txt"
    # s_txt = r"D:\处理的数据集\trojan\trojan-eq-4000-50-s.txt"

    last_txt = r"D:\处理的数据集\mobile_app\mobile_burst_eq_100-10.txt"
    s_txt = r"D:\处理的数据集\mobile_app\mobile_burst_eq_100-10-s.txt"

    # s1_txt = r"D:\处理的数据集\mKCP\mkcp-20-5-s-1.txt"
    # last_txt = r"D:\处理的数据集\QUIC\test\quic-2000.txt"
    # last_txt = r"D:\处理的数据集\TLS1.3\TLS1.3-2000.txt"
    merge_txt_files_to_single_file(save_path,last_txt)
    # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)
    find_longest_sequence_length(s_txt)

def train_tls():
    # 示例用法
    file_path = r"E:\数据集\80_new\0-0"  # 替换成你的文件路径

    dir_path = r"D:\处理的数据集\TLS1.3\total"
    save_path = r"D:\处理的数据集\TLS1.3\tls1.3_time_burst_100-10"

    # dir_path = r"D:\处理的数据集\QUIC\test\total"
    # save_path = r"D:\处理的数据集\QUIC\test\quic-2000"
    # dir_path = r"D:\处理的数据集\TLS1.3\total"
    # save_path = r"D:\处理的数据集\TLS1.3\TLS1.3-2000"
    if not os.path.exists(save_path):
        # 如果文件夹不存在，使用 os.makedirs() 创建它
        os.makedirs(save_path)
        print(f"文件夹 {save_path} 已创建")
    else:
        print(f"文件夹 {save_path} 已存在")
    t_list = os.listdir(dir_path)
    num = 0
    a = time.time()
    time_list = []
    # lable_list1 = os.listdir(r"D:\数据集\mKCP\mKCP_total")
    # lable_list2 = os.listdir(r"D:\数据集\mKCP\mKCP_total2")
    # lable_list3 = os.listdir(r"D:\数据集\mKCP\mKCP_total3")
    # lable_list = lable_list1 + lable_list2 + lable_list3
    lable_list = os.listdir(r"D:\数据集\TLS1.3")
    lable_dict = {item: str(index) for index, item in enumerate(lable_list)}
    # lable_dict = {"GoogleDoc": "0",
    #               "GoogleDrive": "1",
    #               "GoogleMusic": "2",
    #               "GoogleSearch": "3",
    #               "Youtube": "4", }

    l_n = 0
    #
    for p in t_list:
        num += 1
        t_path = os.path.join(dir_path, p)
        lable = lable_dict[p.split('-')[0]]
        # lable = p.split('-')[0]
        s_dir = os.path.join(save_path, p)
        process_cell_file_time_burst(t_path, lable, s_dir,100,10)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))

    # print(l_n)
    last_txt = r"D:\处理的数据集\TLS1.3\tls1.3_time_burst_100-10.txt"
    s_txt = r"D:\处理的数据集\TLS1.3\tls1.3_time_burst_100-10-s.txt"
    # last_txt = r"D:\处理的数据集\QUIC\test\quic-2000.txt"
    # last_txt = r"D:\处理的数据集\TLS1.3\TLS1.3-2000.txt"
    merge_txt_files_to_single_file(save_path,last_txt)
    # # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)
    find_longest_sequence_length(s_txt)

def train_quic():
    dir_path = r"D:\处理的数据集\QUIC\test\total"
    save_path = r"D:\处理的数据集\QUIC\test\quic-time-burst-500-50"

    if not os.path.exists(save_path):
        # 如果文件夹不存在，使用 os.makedirs() 创建它
        os.makedirs(save_path)
        print(f"文件夹 {save_path} 已创建")
    else:
        print(f"文件夹 {save_path} 已存在")
    t_list = os.listdir(dir_path)
    num = 0
    a = time.time()

    lable_list = os.listdir(r"D:\数据集\QUIC Dataset\pretraining\pretraining")
    lable_dict = {item: str(index) for index, item in enumerate(lable_list)}

    #
    for p in t_list:
        num += 1
        t_path = os.path.join(dir_path, p)
        lable = lable_dict[p.split('-')[0]]
        s_dir = os.path.join(save_path, p)
        process_cell_file_time_burst(t_path, lable, s_dir,500,50)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))

    last_txt = r"D:\处理的数据集\QUIC\test\quic-time-burst-500-50.txt"
    s_txt = r"D:\处理的数据集\QUIC\test\quic-time-burst-500-50-s.txt"
    merge_txt_files_to_single_file(save_path,last_txt)

    shuffle_lines_in_file(last_txt, s_txt)
    get_text_len(s_txt)
    find_longest_sequence_length(s_txt)

def train_trojan():
    # 示例用法
    file_path = r"E:\数据集\80_new\0-0"  # 替换成你的文件路径
    dir_path = r"D:\处理的数据集\trojan\Trojan-data-clean-eq"
    save_path = r"D:\处理的数据集\trojan\Trojan-eq-burst-50-5"
    if not os.path.exists(save_path):
        # 如果文件夹不存在，使用 os.makedirs() 创建它
        os.makedirs(save_path)
        print(f"文件夹 {save_path} 已创建")
    else:
        print(f"文件夹 {save_path} 已存在")
    t_list = os.listdir(dir_path)
    num = 0
    a = time.time()

    lable_list = os.listdir(r"D:\数据集\trojan-clean")
    lable_dict = {item: str(index) for index, item in enumerate(lable_list)}

    #
    for p in t_list:
        num += 1
        t_path = os.path.join(dir_path, p)
        lable = lable_dict[p.split('-')[0]]
        # lable = p.split('-')[0]
        s_dir = os.path.join(save_path, p)
        # 5, 30, 50, 100, 500
        process_cell_file_burst(t_path, lable, s_dir,50,5)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))

    last_txt = r"D:\处理的数据集\trojan\trojan-eq-burst-50-5.txt"
    s_txt = r"D:\处理的数据集\trojan\trojan-eq-burst-50-5-s.txt"

    merge_txt_files_to_single_file(save_path,last_txt)
    # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)

def train_tor():
    dir_path = r"D:\处理的数据集\TOR\tor-24-double-column-less-2000"
    save_path = r"D:\处理的数据集\TOR\tor-time-burst-4000-100"

    if not os.path.exists(save_path):
        # 如果文件夹不存在，使用 os.makedirs() 创建它
        os.makedirs(save_path)
        print(f"文件夹 {save_path} 已创建")
    else:
        print(f"文件夹 {save_path} 已存在")
    t_list = os.listdir(dir_path)
    num = 0
    a = time.time()

    lable_list = os.listdir(r"D:\数据集\self_tor_24")
    lable_dict = {item: str(index) for index, item in enumerate(lable_list)}

    #
    for p in t_list:
        num += 1
        t_path = os.path.join(dir_path, p)
        lable = lable_dict[p.split('-')[0]]
        # lable = p.split('-')[0]
        s_dir = os.path.join(save_path, p)
        # 5, 30, 50, 100, 500
        process_cell_file_time_burst(t_path, lable, s_dir,4000,100)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))

    last_txt = r"D:\处理的数据集\TOR\tor-time-burst-4000-100.txt"
    s_txt = r"D:\处理的数据集\TOR\tor-time-burst-4000-100-s.txt"

    merge_txt_files_to_single_file(save_path,last_txt)
    # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)
    find_longest_sequence_length(s_txt)

def train_mobile():
    dir_path = r"D:\处理的数据集\mobile_app\mobile_data_no_60"
    save_path = r"D:\处理的数据集\mobile_app\mobile_time-burst_20-5"

    if not os.path.exists(save_path):
        # 如果文件夹不存在，使用 os.makedirs() 创建它
        os.makedirs(save_path)
        print(f"文件夹 {save_path} 已创建")
    else:
        print(f"文件夹 {save_path} 已存在")
    t_list = os.listdir(dir_path)
    num = 0
    a = time.time()

    lable_list = os.listdir(r"D:\数据集\mobile app\22-cleaned")
    lable_dict = {item: str(index) for index, item in enumerate(lable_list)}

    #
    for p in t_list:
        num += 1
        t_path = os.path.join(dir_path, p)
        lable = lable_dict[p.split('-')[0] + "-" + p.split('-')[1]]
        # lable = p.split('-')[0]
        s_dir = os.path.join(save_path, p)
        # 5, 30, 50, 100, 500
        process_cell_file_time_burst(t_path, lable, s_dir,20,5)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))



    last_txt = r"D:\处理的数据集\mobile_app\mobile_time-burst_20-5.txt"
    s_txt = r"D:\处理的数据集\mobile_app\mobile_time-burst_20-5-s.txt"

    merge_txt_files_to_single_file(save_path,last_txt)
    # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)
    find_longest_sequence_length(s_txt)

def train_mobile_8():
    dir_path = r"D:\处理的数据集\mobile_app_8_class\mobile_data"
    save_path = r"D:\处理的数据集\mobile_app_8_class\mobile_8_time-burst_100-30"

    if not os.path.exists(save_path):
        # 如果文件夹不存在，使用 os.makedirs() 创建它
        os.makedirs(save_path)
        print(f"文件夹 {save_path} 已创建")
    else:
        print(f"文件夹 {save_path} 已存在")
    t_list = os.listdir(dir_path)
    num = 0
    a = time.time()

    lable_list = os.listdir(r"D:\数据集\mobile app\22-8")
    lable_dict = {item: str(index) for index, item in enumerate(lable_list)}

    len_list = []
    #
    for p in t_list:
        num += 1
        t_path = os.path.join(dir_path, p)
        lable = lable_dict[p.split('-')[0]]
        # lable = p.split('-')[0]
        s_dir = os.path.join(save_path, p)
        # 5, 30, 50, 100, 500
        max_len = process_cell_file_time_burst(t_path, lable, s_dir,100,30)
        len_list.extend(max_len)
        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))
    last_txt = r"D:\处理的数据集\mobile_app_8_class\mobile_8_time-burst_100-30.txt"
    s_txt = r"D:\处理的数据集\mobile_app_8_class\mobile_8_time-burst_100-30-s.txt"

    merge_txt_files_to_single_file(save_path,last_txt)
    # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)
    find_longest_sequence_length(s_txt)
    print("max len :{}".format(max(len_list)))
    print(count_elements_in_intervals(len_list,[5, 10, 15, 20, 50]))


if __name__ == '__main__':
    train_mobile_8()

