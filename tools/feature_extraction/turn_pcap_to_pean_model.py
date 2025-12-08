#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCAP to PEAN Model Format Converter

This script converts PCAP files to the format required by the PEAN (Packet Embedding with Attention Networks) model.
It uses multiprocessing to efficiently process large datasets with parallel execution.

Core Features:
- Parallel processing with configurable worker count
- Preserves directory structure and assigns labels based on folders
- Supports batch processing of multiple categories
- Calls external functions from tt1.py for actual conversion

Processing Flow:
1. Traverse source directory with category subfolders
2. Map PCAP files to output paths
3. Assign labels based on folder names
4. Process files in parallel using multiprocessing.Pool

Dependencies: scapy, multiprocessing, tt1.py
"""
from scapy.all import *
from scapy.layers.inet import TCP, IP
import multiprocessing as mp
from traffic_feature_extractor import turn_pcap_to_pean_txt
from traffic_feature_extractor import turn_txt_to_pean_txt_len

def parallel(flist, n_jobs=4):
    try:
        with mp.Pool(n_jobs) as p:
            res = p.map(turn_pcap_to_pean_txt, flist)
            p.close()
            p.join()
        return res
    except Exception as e:
        print('异常说明', e)


if __name__ == '__main__':
    # dir_path = r"D:\数据集\trojan-clean - eq"
    # save_path = r"D:\处理的数据集\PEAN_data\trojan_source_32"
    dir_path = r"D:\数据集\TLS1.3"
    save_path = r"D:\处理的数据集\PEAN_data\tls_source_20_400"
    # dir_path = r"D:\数据集\self_tor_24"
    # save_path = r"D:\处理的数据集\PEAN_data\tor_source_32"
    # turn_pcap_to_tsv(s_dir,save_dir, '22')
    if not os.path.exists(save_path):
        # 如果文件夹不存在，使用 os.makedirs() 创建它
        os.makedirs(save_path)
        print(f"文件夹 {save_path} 已创建")
    else:
        print(f"文件夹 {save_path} 已存在")
    t_list = os.listdir(dir_path)
    num = 0
    a = time.time()
    lable_list = os.listdir(r"D:\数据集\mKCP\mKCP_total")
    lable_list += os.listdir(r"D:\数据集\mKCP\mKCP_total2")
    lable_list += os.listdir(r"D:\数据集\mKCP\mKCP_total3")
    lable_list = os.listdir(r"D:\数据集\trojan-clean - eq")
    lable_list = os.listdir(r"D:\数据集\TLS1.3")
    # lable_list = os.listdir(r"D:\数据集\self_tor_24")
    lable_dict = {item: str(index) for index, item in enumerate(lable_list)}
    pcap_dict = {}
    tu_list = []
    for p in t_list:
        num += 1
        t_path = os.path.join(dir_path,p)
        lable = lable_dict[p.split('-')[0]]
        s_dir = os.path.join(save_path, p)
        p_list = os.listdir(t_path)
        for q in p_list:
            p_path = os.path.join(t_path,q)
            p_no = q.split('.')[0]
            ps_dir = s_dir + '-' + p_no + '.txt'
            pcap_dict[p_path] = ps_dir
        for k, v in pcap_dict.items():
            tu_list.append(k + '%%' + v + '%%' + lable)
        pcap_dict.clear()
        print("第 {} 个， 时间为 ：{}".format(num,time.time() - a))
    parallel(tu_list)
    # tu_list.clear()