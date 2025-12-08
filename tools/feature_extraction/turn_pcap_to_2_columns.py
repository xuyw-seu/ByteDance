#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCAP to Two-Column Feature Format Converter

This script converts PCAP files to a two-column feature format for network traffic analysis.
It uses parallel processing to handle large datasets efficiently.

Core Features:
- Parallel processing with configurable worker count (default: 8)
- Preserves directory structure and file naming conventions
- Supports batch processing across multiple categories
- Calls external functions from tt1.py for feature extraction

Processing Flow:
1. Traverse source directory with category subfolders
2. Map PCAP files to output paths with formatted names
3. Process files in parallel using multiprocessing.Pool
4. Extract features and save in two-column format

Dependencies: scapy, multiprocessing, tt1.py
"""
import multiprocessing as mp
from multiprocessing import  Process
import os
from scapy.all import *
from scapy.layers.inet import TCP, IP
from traffic_feature_extractor import extract_feature_from_pcap,extract_3_colum_feature_from_pcap, extract_3_colum_feature_from_pcap_tor_cell,extract_2_feature_from_json, extract_2_column_raw_feature_from_pcap


def parallel(flist, n_jobs=8):
    try:
        with mp.Pool(n_jobs) as p:
            res = p.map(extract_3_colum_feature_from_pcap, flist)
            p.close()
            p.join()
        return res
    except Exception as e:
        print('异常说明', e)





if __name__ == '__main__':

    # 示例用法
    # dir_path = r"D:\数据集\mKCP-new"
    # save_path = r"D:\处理的数据集\mKCP-NEW\mkcp-new-src-8"
    # dir_path = r"D:\数据集\TLS1.3"
    dir_path = r"D:\数据集\TLS1.3"
    save_path = r"D:\处理的数据集\tls1.3-50\src_data_tls.13_50"
    # dir_path = r"D:\数据集\self_tor_24"
    # save_path = r"D:\处理的数据集\P-DTSF-tor\p-tor-less-data"
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
    pcap_dict = {}
    tu_list = []
    for p in t_list:
        num += 1
        t_path = os.path.join(dir_path,p)
        lable = p.split('-')[0]
        s_dir = os.path.join(save_path, p)
        p_list = os.listdir(t_path)
        for q in p_list[:700]:
            p_path = os.path.join(t_path,q)
            # p_lable = q.split('-')[1].split('.')[0]
            p_lable = q.split('.')[0]
            # p_lable = q.split('.')[0]
            ps_dir = s_dir + '-' + p_lable + '.txt'
            pcap_dict[p_path] = ps_dir
        for k, v in pcap_dict.items():
            tu_list.append(k + '%%' +v)
        pcap_dict.clear()
        print("第 {} 个， 时间为 ：{}".format(num, time.time() - a))
    parallel(tu_list)
