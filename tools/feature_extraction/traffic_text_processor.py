#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本数据处理器
用于处理网络流量文本数据，包括特征提取、数据转换、文件合并和并行处理等功能
主要功能：
1. 并行处理多个文件
2. 将二维列表转换为单行文本格式
3. 处理和分析文本文件内容
4. 提取QUIC协议特征
5. 合并和打乱文本文件
6. 统计和分析时间序列数据
"""
import os
import time
import random
import h5py
from tt1 import filter_and_save_txt_file
import multiprocessing as mp


def parallel(flist, n_jobs=20):
    try:
        with mp.Pool(n_jobs) as p:
            res = p.map(filter_and_save_txt_file, flist)
            p.close()
            p.join()
        return res
    except Exception as e:
        print('异常说明', e)

def write_2d_list_as_single_line(filename, data, time_data, lable=None):
    try:
        if lable.isdigit():
            # for row0 in time_data:
            #     if row0[1] == 0:
            #         return
            with open(filename, 'w') as file:
                # len_list = [100,100,100,100,100,100,100,100,100,100]
                for row in data:
                    # 使用空格分隔列内元素，然后使用制表符分隔列之间
                    row_str = " ".join(str(column) for column in row)
                    file.write(row_str + '\t')

                # r = 50 / time_data[0][1]
                for row1 in time_data:
                    row_str = " ".join(str(round(column * 1)) for column in row1)
                    file.write(row_str + '\t')

                # row_len = " ".join(str(round(i * r)) for i in time_data)
                # file.write(row_len + '\t' + lable)
                file.write(lable)
        else:
            print("999999999999999999999999")

    except Exception as e:
        print(f"Error writing to file: {str(e)}")


def process_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            line_num = 0
            for i in file:
                line_num += 1

            line_num_part = int(line_num / 10)
            print("line num is : {}".format(line_num / 10))
            dev_par_len = line_num_part
            line_num = 0
            # 重新定位文件指针到文件的开头
            file.seek(0)

            total_list = []

            len_dir_list = []
            for line in file:
                if line_num < line_num_part:
                    # 移除行尾的换行符
                    line = line.strip()
                    # 使用制表符或其他分隔符分割列数据
                    columns = line.split('\t')  # 假设使用制表符分隔列
                    len_dir_list.append(int(columns[1]) + 2000)
                    line_num += 1
                else:
                    total_list.append(len_dir_list[:])
                    len_dir_list.clear()
                    line_num_part += dev_par_len
                #     line_num -= 1
                # line_num += 1

            print("total list len is: {}".format(len(total_list)))
            print(total_list)
            for i in total_list:
                print(len(i))

            write_2d_list_as_single_line(r"C:\Users\Administrator\Desktop\jjj\11.txt", total_list, '2')


    except FileNotFoundError:
        print(f"File '{file_path}' not found.")


def process_cell_file(file_path, lable, save_dir):
    # 替换成你的文件路径
    data_list = []
    time_list = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')  # 使用制表符分隔列
                if len(columns) >= 2:
                    # 获取第二列数据，并转化为整数
                    data = int(columns[1]) + 2000
                    data_list.append(data)
                    time_list.append(round(float(columns[0]) * 100))

            rows = 10
            num_elements_per_row = int(len(data_list) / rows)
            unuesd_elements = len(data_list) - rows * num_elements_per_row
            data_list = [data_list[i:i + num_elements_per_row] for i in
                         range(0, len(data_list) - unuesd_elements, num_elements_per_row)]
            time_list = [time_list[i:i + num_elements_per_row] for i in
                         range(0, len(time_list) - unuesd_elements, num_elements_per_row)]
            write_2d_list_as_single_line(save_dir, data_list, time_list, lable)
            # if time_list[1] != 0:
            #     r = 100 / time_list[1]
            #     result_list = [x * r for x in time_list]
            #     return max(result_list)
            # else:
            #     return 0
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
                    for c in range(10):
                        c += 10
                        time_len = columns[c].split(' ')
                        for i in time_len:
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
                line = line.strip()
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
        if filename.endswith(".txt"):
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
            # for line in lines:
            #     line_num += 1
            #     # 移除行尾的换行符
            #     line = line.strip()
            #     # 使用制表符或其他分隔符分割列数据
            #     columns = line.split('\t')  # 假设使用制表符分隔列
            #     for c in columns:
            #         c_l += 1
            # print("row number is :{}, column number is :{}".format(line_num, c_l))
            print("time is : {}".format(time))
            return float(time)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")


def quic_extract_feature_to_two_columns(input_filename, output_filename):
    try:
        # 打开输入文件
        with open(input_filename, 'r') as input_file:
            # 逐行读取文件内容
            lines = input_file.readlines()

        # 提取后3列数据，假设列之间使用空格分隔
        new_lines = []
        for line in lines:
            columns = line.split('\t')
            if len(columns) >= 3:
                if int(columns[3]) == 1:
                    p_len = columns[2]
                elif int(columns[3]) == 0:
                    p_len = '-' + columns[2]
                else:
                    p_len = ''
                    print("error")
                last_three_columns = columns[-3:-1]
                # print(last_three_columns)
                last_three_columns[1] = p_len + '\n'
                new_line = '\t'.join(last_three_columns)
                new_lines.append(new_line)

        # 打开输出文件并将提取的数据写入
        with open(output_filename, 'w') as output_file:
            output_file.writelines(new_lines)

        print("提取并保存成功")
    except Exception as e:
        print(f"发生错误: {e}")


def print_numbers_to_file(n, filename):
    with open(filename, 'w') as file:
        for i in range(1, n+1):
            file.write(str(i) + '\n')

if __name__ == '__main__':
    # 示例用法
    # file_path = r"E:\数据集\80_new\0-0"  # 替换成你的文件路径
    # dir_path = r"D:\处理的数据集\tor-24-double-column"
    # save_path = r"D:\处理的数据集\tor-24-double-column-less-2000"
    # if not os.path.exists(save_path):
    #     # 如果文件夹不存在，使用 os.makedirs() 创建它
    #     os.makedirs(save_path)
    #     print(f"文件夹 {save_path} 已创建")
    # else:
    #     print(f"文件夹 {save_path} 已存在")
    # t_list = os.listdir(dir_path)
    # num = 0
    # a = time.time()
    # time_list = []
    # pral_dict = {}
    # flist = []
    # for p in t_list:
    #     num += 1
    #     t_path = os.path.join(dir_path,p)
    #     lable = p.split('-')[0]
    #     s_dir = os.path.join(save_path,p)
    #     pral_dict[t_path] = s_dir
    #     # process_cell_file(t_path,lable,s_dir)
    #     # quic_extract_feature_to_two_columns(t_path,s_dir)
    #     print("第 {} 个， 时间为 ：{}".format(num,time.time() - a))
    #
    # for k, v in pral_dict.items():
    #     flist.append((k,v))
    # parallel(flist)

    print_numbers_to_file(64980,"E:\论文复现\多模态-PEAN\PEAN-main - re\Config\o6.txt")

    # print("max time is : {}".format(max(time_list)))
    #
    # len_1 = 0
    # len_2 = 0
    # len_3 = 0
    # len_4 = 0
    # for i in time_list:
    #     if i < 20:
    #         len_1 += 1
    #     elif 20 <= i < 30 :
    #         len_2 += 1
    #     elif 30 <= i <40 :
    #         len_3 += 1
    #     else:
    #         len_4 += 1
    # print(len_1,len_2,len_3,len_4)
    # merge_txt_files_to_single_file(save_path,r"C:\Users\Administrator\Desktop\jjj\data_t_m_50_25.txt")
    #
    # shuffle_lines_in_file(r"C:\Users\Administrator\Desktop\jjj\data_t_m_50_25.txt", r"C:\Users\Administrator\Desktop\jjj\data_t_m_50_25_s.txt")
    # # write_numbers_to_file(0,4000,r"C:\Users\Administrator\Desktop\jjj\vo.txt")
    #
    # get_max_time_len_2(r"C:\Users\Administrator\Desktop\jjj\data_t_m_50_25_s.txt")



    # s_dir = r"D:\数据集\QUIC Dataset\pretraining\pretraining\Google Doc\GoogleDoc-31.txt"
    # t_dir = r"D:\处理的数据集\QUIC\test.txt"
    # quic_extract_feature_to_two_columns(s_dir,t_dir)

    # get_text_len(r"D:\处理的数据集\QUIC\test.txt")

    # get_time_len(r"C:\Users\Administrator\Desktop\jjj\80_new\0-0")

