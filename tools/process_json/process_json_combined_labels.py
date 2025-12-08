#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON Combined Labels Processor

Core Functions:
- Process network traffic JSON files
- Calculate relative time features
- Combine BF_label and BF_activity for rich labeling
- Classify files by keywords (audiocall/chat/videocall)
- Basic JSON parsing and organization

Dependencies: json, os, shutil, time
"""
import json
import os
import shutil
import time


def save_dict_to_json(data, file_path):
    try:
        with open(file_path, 'w') as output_file:
            json.dump(data, output_file, indent=4)
        print(f"Data has been saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving data to {file_path}: {e}")


def read_json(file_path, save_path, file_index):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            total_dict = {}
            new_data_dict = {}

            for k, v in data.items():

                packet_data_dict = v['packet_data']
                f_time = packet_data_dict['timestamp'][0]
                time_list = [(j - f_time) for j in packet_data_dict['timestamp']]
                dir_list = packet_data_dict['packet_dir']
                len_list = [(i+60) for i in packet_data_dict['L4_payload_bytes']]
                tcp_window_list = packet_data_dict['TCP_win_size']
                raw_bytes_list = packet_data_dict['L4_raw_payload']
                flow_meta_dict = v['flow_metadata']
                app_label = flow_meta_dict['BF_label']
                active_label = flow_meta_dict['BF_activity']
                mix_lable = app_label + "_" + active_label

                new_data_dict["time_list"] = time_list
                new_data_dict["dir_list"] = dir_list
                new_data_dict["len_list"] = len_list
                new_data_dict["tcp_window_list"] = tcp_window_list
                new_data_dict["raw_bytes_list"] = raw_bytes_list
                new_data_dict["mix_label"] = mix_lable
                new_name = str(file_index) + "-" + k + "-" + mix_lable + ".json"
                s_path = os.path.dirname(save_path)
                s_path = os.path.join(s_path,new_name)
                total_dict[k] = new_data_dict
                save_dict_to_json(total_dict, s_path)
                new_data_dict.clear()
                total_dict.clear()

            return 0
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def copy_files_to_different_folders(src_folder, dest_base_folder):
    # 遍历源文件夹下的所有文件
    for filename in os.listdir(src_folder):
        # 构建源文件路径
        src_path = os.path.join(src_folder, filename)

        # 检查文件是否为文件夹
        if os.path.isdir(src_path):
            continue  # 如果是文件夹，跳过

        # 根据文件名的特定条件决定目标文件夹
        if 'audiocall' in filename:
            dest_folder = os.path.join(dest_base_folder, os.path.basename(src_folder) + '-' + 'ac')
        elif 'chat' in filename:
            dest_folder = os.path.join(dest_base_folder, os.path.basename(src_folder) + '-' + 'chat')
        elif 'videocall' in filename:
            dest_folder = os.path.join(dest_base_folder, os.path.basename(src_folder) + '-' + 'vc')
        else:
            dest_folder = os.path.join(dest_base_folder, 'unknow')

        # 创建目标文件夹（如果不存在）
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # 构建目标文件路径
        dest_path = os.path.join(dest_folder, filename)

        # 复制文件到目标文件夹
        shutil.copy2(src_path, dest_path)
        print(f"File '{filename}' copied to '{dest_folder}'")


def rename_files_as_forld_name(folder_path):
    # 获取文件夹下的所有文件
    files = os.listdir(folder_path)
    index = 0

    for filename in files:
        # 构建新的文件名
        base, extension = os.path.splitext(filename)
        if extension == ".exe":
            return
        name = os.path.basename(folder_path)
        # 如果文件名符合规定的格式，则进行重命名

        new_filename = f"{name}-{index}" + ".json"
        index += 1
        new_path = os.path.join(folder_path, new_filename)

        # 检查新文件名是否已存在
        if not os.path.exists(new_path):
            old_path = os.path.join(folder_path, filename)
            # 执行重命名
            os.rename(old_path, new_path)
            print(f"重命名文件：{filename} -> {new_filename}")
        else:
            print(f"文件已存在，未重命名：{new_filename}")

if __name__ == '__main__':

    s_dir = r"D:\数据集\mobile app\22\GotoMeeting"
    t_dir = r"D:\数据集\mobile app\22-new\GotoMeeting"
    if not os.path.exists(t_dir):
        # 如果文件夹不存在，使用 os.makedirs() 创建它
        os.makedirs(t_dir)
        print(f"文件夹 {t_dir} 已创建")

    dir_list = os.listdir(s_dir)
    a = time.time()
    index = 0

    for i in dir_list:
        print("No {}, time : {}".format(index,time.time() - a))
        index += 1
        js_data = os.path.join(s_dir,i)
        tg_data = os.path.join(t_dir,i)
        read_json(js_data, tg_data, index)


    # s_dir = r"D:\数据集\mobile app\22-cleaned"
    # t_dir = r"D:\数据集\mobile app\22-cleaned"
    # if not os.path.exists(t_dir):
    #     # 如果文件夹不存在，使用 os.makedirs() 创建它
    #     os.makedirs(t_dir)
    #     print(f"文件夹 {t_dir} 已创建")
    #
    # dir_list = os.listdir(s_dir)
    # a = time.time()
    # index = 0
    #
    # for i in dir_list:
    #     print("No {}, time : {}".format(index,time.time() - a))
    #     index += 1
    #     js_data = os.path.join(s_dir,i)
    #     rename_files_as_forld_name(js_data)
    #     # copy_files_to_different_folders(js_data,t_dir)



