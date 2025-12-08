#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCAP File Cleaning and Processing Tool

This script provides functions for filtering, cleaning, and processing PCAP (Packet Capture) files.
It helps prepare network traffic data for analysis by removing unnecessary packets and standardizing file formats.

Key Features:
- Filter PCAP files to retain only useful packets
- Remove DNS, empty payload, and corrupted packets
- Standardize file naming conventions
- Process large PCAP files with progress tracking
- Support for batch processing

Dependencies:
- scapy: For PCAP file parsing and manipulation
- tqdm: For progress bar display

Usage:
python clean_pcap.py
"""
import os

from scapy.all import *
from scapy.layers.dns import DNS
from scapy.layers.inet import *
from tqdm import tqdm
import time


def filter_and_save_pcap(input_pcap, output_pcap):
    # 读取原始pcap文件
    packets = rdpcap(input_pcap)

    # 过滤条件：UDP、DNS报文，以及载荷不为0的TCP报文
    filtered_packets = []

    # 计算总包数
    total_packets = len(packets)

    # 使用tqdm显示进度条
    with tqdm(total=total_packets, desc="Filtering Packets", unit="packet") as pbar:
        for packet in packets:
            if (UDP in packet or (TCP in packet and packet[TCP].payload and len(packet[TCP].payload) > 0)) or \
                    (DNS in packet):
                filtered_packets.append(packet)
            pbar.update(1)  # 更新进度条
            time.sleep(0.1)  # 为了更好的显示进度，稍微延迟一下

    # 创建新的pcap文件并保存
    wrpcap(output_pcap, filtered_packets)


def flowProcess(s_pcap,t_pcap):
    if os.path.exists(t_pcap):
        return
    packets = rdpcap(s_pcap)  # 提取pcap
    # 用于统计不要的报文数
    IPv4_num = 0
    IPv6_num = 0
    arp = 0
    other = 0
    dns = 0
    wrong_data = 0
    payload_zero = 0


    # 创建写入器
    writer = PcapWriter(t_pcap, append=True)
    # print('E:/数据集/ISCX2016/NonVPN-PCAPs-01' + t_name2[1] +"_flow"+ t_name2[1] + '%d' % total + '.pcap')

    for data in packets:
        # 只分理出IPv4的数据报
        if data.payload.name == 'IP':
            IPv4_num = IPv4_num + 1
            if data.payload.payload.payload.name == 'DNS':
                dns = dns + 1
                continue
            elif data.payload.payload.name == 'UDP' and data.payload.payload.len - 8 == 0:
                payload_zero = payload_zero + 1
                continue
            elif data.payload.payload.name == 'TCP' and (data.payload.len - data.payload.ihl * 4 - data.payload.payload.dataofs * 4 == 0 or len(data[TCP].payload) == 0 or
            len(data[TCP].payload) == 1):
                payload_zero = payload_zero + 1
                continue
            else:
                try:
                    # 读取当前数据包的信息
                    writer.write(data)  # 写入流
                    writer.flush()
                except AttributeError:
                    wrong_data = wrong_data + 1
        else:
            other = other + 1

    writer.close()
    print("Ipv4数目：", IPv4_num, "Ipv6数目：", IPv6_num, 'arp数目:', arp, ' other数目:', other, ' 数据包损坏数目:', wrong_data, '载荷为0的数目为：', payload_zero, 'DNS数目为：', dns)
    # end for


def rename_files(folder_path):
    # 获取文件夹下的所有文件
    files = os.listdir(folder_path)

    for filename in files:
        # 构建新的文件名
        base, extension = os.path.splitext(filename)
        if extension == ".log":
            return
        parts = base.split('_')
        no = extension.split('.pcap')

        if no[1] == '':
            num = "0"
        else:
            num = str(no[1])
        # 如果文件名符合规定的格式，则进行重命名
        if len(parts) == 3:
            new_filename = f"{parts[0]}_{parts[1]}_{parts[2]}_{num}" + ".pcap"
        elif len(parts) == 4:
            new_filename = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_{num}" + ".pcap"
        elif len(parts) == 5:
            new_filename = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}" + ".pcap"

        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        # 检查新文件名是否已存在
        if not os.path.exists(new_path):
            old_path = os.path.join(folder_path, filename)
            # 执行重命名
            os.rename(old_path, new_path)
            print(f"重命名文件：{filename} -> {new_filename}")
        else:
            print(f"文件已存在，未重命名：{new_filename}")

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

        new_filename = f"{name}-{index}" + ".pcap"
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
    in_path = r"D:\数据集\DDOS-2"
    out_path = r"D:\数据集\trojan\YouTube+trojan"
    s_path = r"C:\Users\Administrator\Desktop\trojan"
    t_path = r"D:\数据集\trojan"
    list = os.listdir(in_path)
    index = 0
    a = time.time()
    # for i in list:
    #     print("第 ： {} 个， 时间：{}".format(index,time.time() - a))
    #     index += 1
    #     s = os.path.join(in_path,i)
    #     o = os.path.join(out_path,i)
    #     flowProcess(s,o)

    # flowProcess("D:\数据集\DDOS\DDoS-HTTP_Flood\DDoS-HTTP_Flood\DDoS-HTTP_Flood_2.pcap","D:\数据集\DDOS\DDoS-HTTP_Flood\DDoS-HTTP_Flood\DDoS-HTTP_Flood_clean.pcap")

    # for i in list:
    #     print("第 ： {} 个， 时间：{}".format(index,time.time() - a))
    #     index += 1
    #     s = os.path.join(in_path,i)
    #     rename_files_as_forld_name(s)

    rename_files_as_forld_name(r"D:\数据集\DDOS-2\DDoS-SlowLoris")
