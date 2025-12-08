#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版网络流量时间序列处理器
用于处理网络流量的时间序列数据，支持更复杂的数据格式和处理逻辑
主要功能：
1. 将多维度列表转换为单行文本格式（包含长度、字节、方向和时间数据）
2. 支持带字节数据的时间序列处理
3. 处理和分割时间序列数据
4. 提取请求和响应方向的流量特征
5. 统计和分析时间序列长度
6. 合并和打乱文本文件
7. 过滤和清理数据（如移除空白行、过滤特定值）
8. 支持多种网络流量类型的处理（mobile, TLS, Tor, Trojan, QUIC, mKCP等）
9. 支持时间序列突发检测和分析
"""
import os
import time
import random
from itertools import islice


# data:请求方向， time_data:响应方向，byte：字符
def write_2d_list_as_single_line(len_data, byte_data, dir_data, time_data, max_num,  lable=None, filename=None):
    try:
        with open(filename, 'w') as file:
            # len_list = [100,100,100,100,100,100,100,100,100,100]

            if len(byte_data) < max_num:
                byte_data.extend(['0'] * (max_num - len(byte_data)))

            for row0 in byte_data:
                # 使用空格分隔列内元素，然后使用制表符分隔列之间
                file.write(row0 + '\t')


            row_str = " ".join(str(abs(column)) for column in len_data)
            file.write(row_str + '\t')

            row_str = " ".join(str(abs(column)) for column in dir_data)
            file.write(row_str + '\t')

            row_str = " ".join(str(abs(column)) for column in time_data)
            file.write(row_str + '\t')


            file.write(lable)

    except Exception as e:
        print(f"Error writing to file: {str(e)}")


def  process_cell_file_no_t(file_path, lable, save_dir, max_pcap_num):
    if os.path.exists(save_dir):
        return
    # 替换成你的文件路径
    data_list = []
    time_list = []
    byte_list = []
    dir_list = []
    try:
        with open(file_path, 'r') as file:
            for line in islice(file, max_pcap_num):
                columns = line.strip().split('\t')  # 使用制表符分隔列
                if len(columns) == 3:
                    # 获取第二列数据，并转化为整数
                    time_list.append(float(columns[0]))
                    data_list.append(int(columns[1]))
                    byte_list.append(str(columns[2]))
                    if int(columns[1]) < 0:
                        dir_list.append(2)
                    else:
                        dir_list.append(1)


            data_list = data_list[:max_pcap_num]
            byte_list = byte_list[:max_pcap_num]
            dir_list = dir_list[:max_pcap_num]
            time_list = time_list[:max_pcap_num]


            write_2d_list_as_single_line(data_list,byte_list, dir_list, time_list , max_pcap_num, lable, save_dir)

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")

def write_2d_list_as_single_line_3(len_data, byte_data, time_data, max_num,  lable=None, filename=None):
    try:
        with open(filename, 'w') as file:
            # len_list = [100,100,100,100,100,100,100,100,100,100]

            if len(byte_data) < max_num:
                byte_data.extend(['0'] * (max_num - len(byte_data)))

            for row0 in byte_data:
                # 使用空格分隔列内元素，然后使用制表符分隔列之间
                file.write(row0 + '\t')

            row_str = " ".join(str(column + 2000) for column in len_data)
            file.write(row_str + '\t')

            pre = 0
            data_str = ''
            for column in time_data:
                data = int(abs(column) * 1000)
                if data > 60000:
                    data = pre
                if data < 60000:
                    pre = data
                data_str += str(data) + ' '

            data_str = data_str.strip()
            # row_str = " ".join(str(int(abs(column) * 10000)) for column in time_data)
            file.write(data_str + '\t')

            file.write(lable)

    except Exception as e:
        print(f"Error writing to file: {str(e)}")

def process_cell_file_no_t_3(file_path, lable, save_dir, max_pcap_num):
    if os.path.exists(save_dir):
        return
    # 替换成你的文件路径
    data_list = []
    time_list = []
    byte_list = []
    dir_list = []
    try:
        with open(file_path, 'r') as file:
            for line in islice(file, max_pcap_num):
                columns = line.strip().split('\t')  # 使用制表符分隔列
                if len(columns) == 3:
                    # 获取第二列数据，并转化为整数
                    time_list.append(float(columns[0]))
                    data_list.append(int(columns[1]))
                    byte_list.append(str(columns[2]))


            data_list = data_list[:max_pcap_num]
            byte_list = byte_list[:max_pcap_num]
            time_list = time_list[:max_pcap_num]


            write_2d_list_as_single_line_3(data_list, byte_list, time_list , max_pcap_num, lable, save_dir)

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")


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

def process_cell_file_no_t_single(file_path, lable, save_dir, max_pcap_num, column_num):
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


def process_cell_file(file_path, lable=None, save_dir=None, plot=2, total_time=40):
    # 替换成你的文件路径
    data_list = []
    time_list = []
    new_data_list_1 = []
    new_data_list_0 = []
    sub_list_1 = []
    sub_list_0 = []
    rows = total_time // plot
    time_plot_list = [0 for q in range(rows)]

    try:
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')  # 使用制表符分隔列
                if len(columns) >= 2:
                    # 获取第二列数据，并转化为整数
                    data = int(columns[1])
                    data_list.append(data)
                    time_list.append(float(columns[0]))
                    for j in range(rows):
                        if j * plot <= float(columns[0]) < plot * (j + 1):
                            time_plot_list[j] += 1

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


def get_max_time_len(file_path):
    # 替换成你的文件路径
    data_list = []
    time_list = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')  # 使用制表符分隔列
                if len(columns) >= 2:
                    # 获取第二列数据，并转化为整数
                    time_len = columns[-2].split(' ')
                    for i in time_len:
                        time_list.append(int(i))
        over_1000 = 0
        under_1000 = 0
        over_2000 = 0
        total = 0
        for i in time_list:
            total += i
            if i >= 3500:
                over_2000 += 1
            if 1000 < i < 2000:
                over_1000 += 1
            if i <= 1000:
                under_1000 += 1
        print(max(time_list), over_1000, under_1000, over_2000, total // len(time_list), total)

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")


def get_max_time_len_2(file_path):
    # 替换成你的文件路径
    data_list = []
    time_list = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')  # 使用制表符分隔列
                if len(columns) >= 20:
                    for c in range(20):
                        time_len = columns[c].split(' ')
                        for i in time_len:
                            if i.isdigit():
                                time_list.append(int(i))

        print(max(time_list))

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
    i = 0
    for filename in os.listdir(source_dir):
        print("merge {} ".format(i))
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


def get_time_len(file_path):
    try:
        with open(file_path, 'r') as file:
            line_num = 0
            c_l = 0  # 初始化 c_l
            lines = file.readlines()  # 将文件内容读取到 lines 变量
            time = lines[-1].split('\t')[0]
            print("time is : {}".format(time))
            return float(time)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")


def get_header_max_len(file_path):
    try:
        len_list_req = []
        len_list_res = []
        req_list = [i for i in range(10)]
        # res_list = [(i + 10) for i in range(10)]
        with open(file_path, 'r') as file:
            line_num = 0
            for line in file:
                colums = line.split('\t')
                for i in req_list:
                    len_list_req.append(len(colums[i].split(' ')))
                    len_list_res.append(len(colums[i + 10].split(' ')))
            j = 0
            q = 0
            x = 0
            for i in len_list_req:
                if i < 100:
                    j += 1
                elif 100 <= i < 200:
                    q += 1
                else:
                    x += 1
            print("max req number is :{},{},{},{}, max res number is :{}".format(max(len_list_req), j, q, x,
                                                                                 max(len_list_res)))
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")


def remove_blank_lines(input_file, output_file):
    try:
        with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
            for line in input_f:
                if line.strip():  # 如果行不是空白行
                    output_f.write(line)
        return True
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

def remove_txt_files_60(src_folder, dest_folder):
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 遍历源文件夹下的所有文件
    for filename in os.listdir(src_folder):
        # 构建源文件路径和目标文件路径
        src_path = os.path.join(src_folder, filename)
        dest_path = os.path.join(dest_folder, filename)

        # 检查文件是否为 txt 文件
        if filename.endswith('.txt'):
            with open(src_path, 'r') as source_file:
                lines = source_file.readlines()


            # # 过滤出第二列不为 60 的行
            # filtered_lines = []
            # for line in lines:
            #     # 使用制表符 '\t' 分割行，然后检查第二列是否不等于 '60'
            #     if int(line.split('\t')[1]) == 60 or int(line.split('\t')[1]) == -60:
            #         continue
            #     filtered_lines.append(line)

            filtered_lines = [line for line in lines if (int(line.split('\t')[1]) != 60 and int(line.split('\t')[1]) != -60)]

            # 写入结果到目标文件
            with open(dest_path, 'w') as dest_file:
                dest_file.writelines(filtered_lines)

            print(f"Processed file '{filename}' and saved to '{dest_folder}'")

def train_mobile():
    # 示例用法
    file_path = r"E:\数据集\80_new\0-0"  # 替换成你的文件路径
    dir_path = r"D:\处理的数据集\mobile_app\mobile_eq"
    save_path = r"D:\处理的数据集\mobile_app\mobile_eq_100-20"
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
        process_cell_file_no_t(t_path, lable, s_dir,100,20)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))

    # last_txt = r"D:\处理的数据集\trojan\trojan-eq-4000-50.txt"
    # s_txt = r"D:\处理的数据集\trojan\trojan-eq-4000-50-s.txt"

    last_txt = r"D:\处理的数据集\mobile_app\mobile_eq_100-20.txt"
    s_txt = r"D:\处理的数据集\mobile_app\mobile_eq_100-20-s.txt"

    # s1_txt = r"D:\处理的数据集\mKCP\mkcp-20-5-s-1.txt"
    # last_txt = r"D:\处理的数据集\QUIC\test\quic-2000.txt"
    # last_txt = r"D:\处理的数据集\TLS1.3\TLS1.3-2000.txt"
    merge_txt_files_to_single_file(save_path,last_txt)
    # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)
    get_max_time_len_2(s_txt)

def train_tls():
    # dir_path = r"D:\处理的数据集\P-DTSF\p-tls1.3-data"
    dir_path = r"D:\处理的数据集\tls1.3-50\src_data_tls.13_50"
    # data-n-m, 数据集-包个数-分组个数
    # save_path = r"D:\处理的数据集\P-DTSF\pint-less-tls1.3-20"
    save_path = r"D:\处理的数据集\tls1.3-50\new-less-tls1.3-20"
    if not os.path.exists(save_path):
        # 如果文件夹不存在，使用 os.makedirs() 创建它
        os.makedirs(save_path)
        print(f"文件夹 {save_path} 已创建")
    else:
        print(f"文件夹 {save_path} 已存在")
    t_list = os.listdir(dir_path)
    num = 0
    a = time.time()

    lable_list = os.listdir(r"D:\数据集\TLS1.3")

    lable_dict = {item: str(index) for index, item in enumerate(lable_list)}
    #
    for p in t_list:
        num += 1
        t_path = os.path.join(dir_path, p)
        # lable = lable_dict[p.split('-')[0] + "-" + p.split('-')[1]]
        lable = lable_dict[p.split('-')[0]]
        s_dir = os.path.join(save_path, p)
        # 5, 30, 50, 100, 500
        process_cell_file_no_t_3(t_path, lable, s_dir,20)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))

    # last_txt = r"D:\处理的数据集\P-DTSF\pint-less-tls1.3-20.txt"
    # s_txt = r"D:\处理的数据集\P-DTSF\pint-less-tls1.3-20-s.txt"
    last_txt = r"D:\处理的数据集\tls1.3-50\new-less-tls1.3-20.txt"
    s_txt = r"D:\处理的数据集\tls1.3-50\new-less-tls1.3-20-s.txt"

    merge_txt_files_to_single_file(save_path,last_txt)
    # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)
    get_max_time_len_2(s_txt)

def train_tor():
    # dir_path = r"D:\处理的数据集\P-DTSF-tor\p-tor-data"
    dir_path = r"D:\处理的数据集\P-DTSF-tor\p-tor-data"
    # dir_path = r"D:\处理的数据集\P-DTSF-tor\p-tor-cell-data"
    save_path = r"D:\处理的数据集\P-DTSF-tor\pint-tor-10"

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
        # lable = lable_dict[p.split('-')[0] + "-" + p.split('-')[1]]
        lable = lable_dict[p.split('-')[0]]
        s_dir = os.path.join(save_path, p)
        # 5, 30, 50, 100, 500
        process_cell_file_no_t_3(t_path, lable, s_dir, 10)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))

    last_txt = r"D:\处理的数据集\P-DTSF-tor\pint-tor-10.txt"
    s_txt = r"D:\处理的数据集\P-DTSF-tor\pint-tor-10-s.txt"

    merge_txt_files_to_single_file(save_path,last_txt)
    # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)
    get_max_time_len_2(s_txt)

def train_trojan():
    # 示例用法
    file_path = r"E:\数据集\80_new\0-0"  # 替换成你的文件路径
    # dir_path = r"D:\处理的数据集\trojan\Trojan-data-clean-eq"
    dir_path = r"D:\处理的数据集\trojan\pint-trojan-3-data"
    # save_path = r"D:\处理的数据集\trojan\Trojan-single-500-50"
    save_path = r"D:\处理的数据集\trojan\pint-trojan-100"
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
        process_cell_file_no_t_3(t_path, lable, s_dir, 100)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))

    last_txt = r"D:\处理的数据集\trojan\pint-trojan-100.txt"
    s_txt = r"D:\处理的数据集\trojan\pint-trojan-100-s.txt"

    merge_txt_files_to_single_file(save_path,last_txt)
    # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)
    get_max_time_len_2(s_txt)

def train_quic():
    dir_path = r"D:\处理的数据集\QUIC\test\total"
    save_path = r"D:\处理的数据集\QUIC\test\quic-burst-100-10"

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
        process_cell_file_no_t(t_path, lable, s_dir,100,10)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))

    last_txt = r"D:\处理的数据集\QUIC\test\quic-burst-100-10.txt"
    s_txt = r"D:\处理的数据集\QUIC\test\quic-burst-100-10-s.txt"
    merge_txt_files_to_single_file(save_path,last_txt)

    shuffle_lines_in_file(last_txt, s_txt)
    get_text_len(s_txt)

def train_mkcp():
    dir_path = r"D:\处理的数据集\mKCP-NEW\mkcp-new-src-8"
    save_path = r"D:\处理的数据集\mKCP-NEW\mkcp-new-8-100-10"

    if not os.path.exists(save_path):
        # 如果文件夹不存在，使用 os.makedirs() 创建它
        os.makedirs(save_path)
        print(f"文件夹 {save_path} 已创建")
    else:
        print(f"文件夹 {save_path} 已存在")
    t_list = os.listdir(dir_path)
    num = 0
    a = time.time()
    # lable_list1 = os.listdir(r"D:\数据集\mKCP\mKCP_total")
    # lable_list2 = os.listdir(r"D:\数据集\mKCP\mKCP_total2")
    # lable_list3 = os.listdir(r"D:\数据集\mKCP\mKCP_total3")
    # lable_list = lable_list1 + lable_list2 + lable_list3
    lable_list = os.listdir(r"D:\数据集\mKCP-new")
    lable_dict = {item: str(index) for index, item in enumerate(lable_list)}

    #
    for p in t_list:
        num += 1
        t_path = os.path.join(dir_path, p)
        # lable = lable_dict[p.split('-')[0] + "-" + p.split('-')[1]]
        lable = lable_dict[p.split('-')[0]]
        s_dir = os.path.join(save_path, p)
        # 5, 30, 50, 100, 500
        process_cell_file_no_t(t_path, lable, s_dir,100,10)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))

    last_txt = r"D:\处理的数据集\mKCP-NEW\mkcp-new-8-100-10.txt"
    s_txt = r"D:\处理的数据集\mKCP-NEW\mkcp-new-8-100-10-s.txt"

    merge_txt_files_to_single_file(save_path,last_txt)
    # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)
    get_max_time_len_2(s_txt)

def train_mobile_2():
    dir_path = r"D:\处理的数据集\mobile_app_2\mobile_data"
    save_path = r"D:\处理的数据集\mobile_app_2\mobile_2-20-5"

    if not os.path.exists(save_path):
        # 如果文件夹不存在，使用 os.makedirs() 创建它
        os.makedirs(save_path)
        print(f"文件夹 {save_path} 已创建")
    else:
        print(f"文件夹 {save_path} 已存在")
    t_list = os.listdir(dir_path)
    num = 0
    a = time.time()
    lable_list = os.listdir(r"D:\数据集\mobile_app_2")

    lable_dict = {item: str(index) for index, item in enumerate(lable_list)}

    #
    for p in t_list:
        num += 1
        t_path = os.path.join(dir_path, p)
        lable = lable_dict[p.split('-')[0]]
        # lable = p.split('-')[0]
        s_dir = os.path.join(save_path, p)
        # 5, 30, 50, 100, 500
        process_cell_file_no_t(t_path, lable, s_dir,20,5)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))

    last_txt = r"D:\处理的数据集\mobile_app_2\mobile_2-20-5.txt"
    s_txt = r"D:\处理的数据集\mobile_app_2\mobile_2-20-5-s.txt"

    merge_txt_files_to_single_file(save_path,last_txt)
    # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)
    get_max_time_len_2(s_txt)

def train_mobile_8():
    dir_path = r"D:\处理的数据集\mobile_app_8_class\mobile_data"
    save_path = r"D:\处理的数据集\mobile_app_8_class\mobile_8_100-10"

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
        process_cell_file_no_t(t_path, lable, s_dir,50)
        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))
    last_txt = r"D:\处理的数据集\mobile_app_8_class\mobile_8_100-10.txt"
    s_txt = r"D:\处理的数据集\mobile_app_8_class\mobile_8_100-10-s.txt"

    merge_txt_files_to_single_file(save_path,last_txt)
    # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)

def train_mobile_2025():
    dir_path = r"D:\处理的数据集\mobile_2025\src_data"
    # data-n-m, 数据集-包个数-分组个数
    save_path = r"D:\处理的数据集\mobile_2025\mobile_5g_20"
    if not os.path.exists(save_path):
        # 如果文件夹不存在，使用 os.makedirs() 创建它
        os.makedirs(save_path)
        print(f"文件夹 {save_path} 已创建")
    else:
        print(f"文件夹 {save_path} 已存在")
    t_list = os.listdir(dir_path)
    num = 0
    a = time.time()

    lable_list = os.listdir(r"D:\数据集\mobile_app_8_2025")

    lable_dict = {item: str(index) for index, item in enumerate(lable_list)}
    #
    for p in t_list:
        num += 1
        t_path = os.path.join(dir_path, p)
        # lable = lable_dict[p.split('-')[0] + "-" + p.split('-')[1]]
        lable = lable_dict[p.split('-')[0]]
        s_dir = os.path.join(save_path, p)
        # 5, 30, 50, 100, 500
        process_cell_file_no_t_3(t_path, lable, s_dir,20)

        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))

    last_txt = r"D:\处理的数据集\mobile_2025\mobile_5g_20_l.txt"
    s_txt = r"D:\处理的数据集\mobile_2025\mobile_5g_20_ls.txt"

    merge_txt_files_to_single_file(save_path,last_txt)
    # # #
    shuffle_lines_in_file(last_txt, s_txt)
    #
    get_text_len(s_txt)
    get_max_time_len_2(s_txt)

if __name__ == '__main__':
    # train_tor()
    # train_trojan()
    train_tls()

