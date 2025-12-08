#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5文件处理器
用于处理HDF5文件，包括文件结构查看和数据集转换为文本格式
主要功能：
1. 递归遍历和打印HDF5文件的结构
2. 查看HDF5文件中的数据集信息（形状、数据类型等）
3. 将HDF5数据集转换为文本文件格式
4. 访问和读取HDF5文件中的数据集
5. 支持复杂HDF5文件结构的处理
"""
import h5py


def read_h5_structure(filename):
    try:
        with h5py.File(filename, 'r') as file:
            # 递归遍历HDF5文件的结构
            def print_h5_structure(group, level=0):
                for name, item in group.items():
                    if isinstance(item, h5py.Group):
                        print(f"{'  ' * level}Group: {name}/")
                        print_h5_structure(item, level + 1)
                    else:
                        print(f"{'  ' * level}Dataset: {name} (Shape: {item.shape}, Data Type: {item.dtype})")

            print(f"File: {filename}")
            print_h5_structure(file)

            # 访问数据集
            data_group = file['data']

            axis0_data = data_group['axis0'][:]
            axis1_data = data_group['axis1']
            block0_items_data = data_group['block0_items'][:]
            block0_values_data = data_group['block0_values'][:]

            print(axis0_data, axis1_data)

    except Exception as e:
        print(f"发生错误: {e}")


def write_hdf5_dataset_to_txt(hdf5_file_path, dataset_name, output_file_path):
    try:
        # 打开HDF5文件
        with h5py.File(hdf5_file_path, 'r') as file:
            # 检查数据集是否存在
            if dataset_name in file:
                # 访问数据集
                data = file[dataset_name][:]

                # 写入数据到文本文件
                with open(output_file_path, 'w') as output_file:
                    for value in data:
                        output_file.write(str(value) + '\t')  # 使用制表符分隔不同列
                    output_file.write('\n')  # 换行以分隔不同行
                print(f"数据已写入到 {output_file_path}")
            else:
                print(f"数据集 '{dataset_name}' 不存在于文件 '{hdf5_file_path}'")
    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == '__main__':
    file = r"D:\数据集\VPN\VNAT_Dataframe_release_1.h5"
    out_file = r"D:\数据集\VPN\VNAT_Dataframe_release_1.txt"
    read_h5_structure(file)

    write_hdf5_dataset_to_txt(file,"data/block0_values", out_file)