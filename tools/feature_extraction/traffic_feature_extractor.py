#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Traffic Feature Extraction Library

This module provides a collection of functions for extracting features from PCAP files and JSON traffic data.
It is a core dependency for several other scripts in the network traffic processing pipeline.

Core Functions:
- PCAP Feature Extraction:
  - extract_3_colum_feature_from_pcap: Extract 3-column features (time, length, hex data)
  - extract_3_colum_feature_from_pcap_tor_cell: Extract Tor cell features
  - extract_2_column_raw_feature_from_pcap: Extract 2-column raw features
  - turn_pcap_to_pean_txt: Convert PCAP to PEAN model format
  - extract_feature_from_pcap: Basic feature extraction
  - split_1000_from_pcap: Split PCAP into chunks
  - filter_and_save_txt_file: Filter TXT files by value

- JSON Feature Extraction:
  - extract_2_feature_from_json: Extract time and length features from JSON

- Helper Functions:
  - is_ack_packet: Check if a packet is an ACK packet
  - turn_txt_to_pean_txt_len: Convert TXT to PEAN model format with lengths

Dependencies: scapy, json, os, time
"""
import json
import os
import time

from scapy.layers.inet import TCP, UDP, IP
from scapy.packet import Raw
from scapy.utils import rdpcap, PcapWriter


def filter_and_save_txt_file(data):
    input_file_path, output_file_path = data
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            columns = line.split()  # 假设列之间使用空格分隔
            if len(columns) == 2:
                try:
                    column2_value = abs(int(columns[1]))
                    if column2_value <= 2000:
                        outfile.write(line)
                except ValueError:
                    # 忽略无法解析为整数的行
                    pass


def is_ack_packet(packet):
    if packet.haslayer(TCP):
        tcp_flags = packet[TCP].flags
        # TCP flag for ACK is 16
        if tcp_flags & 16:
            return True
    return False

def extract_3_colum_feature_from_pcap(data_tuple):
    try:
        pcap_path = data_tuple.split('%%')[0]
        save_path = data_tuple.split('%%')[1]
        # 创建TSV文件并写入标题行
        tsv_file = save_path
        # 使用 with 语句打开文件，确保在退出代码块时文件被关闭
        if os.path.exists(tsv_file):
            print("exist, return! ")
            return
        # 读取PCAP文件
        pcap_file = pcap_path
        # packets = rdpcap(pcap_file,count=200)
        packets = rdpcap(pcap_file)

        # 检查数据包数量是否为0或小于10
        if len(packets) == 0 or len(packets) < 5:
            print("The number of packets is 0 .")
            return

        with open(tsv_file, 'w') as file:


            # # 计算第一个数据包的时间戳
            # first_packet_time = packets[0].time
            # s_ip = packets[0].getlayer('IP')
            # f_src_ip = s_ip.src
            # f_dst_ip = s_ip.dst

            # 计算第一个数据包的时间戳
            first_packet_time = None
            s_ip = None
            f_src_ip = None
            f_dst_ip = None

            unknow_dir_packet = 0

            f = True

            # 遍历前10个数据包并将它们写入TSV文件
            for i, packet in enumerate(packets):


                if packet.haslayer(IP):
                    if f:
                        first_packet_time = packet.time
                        s_ip = packet.getlayer('IP')
                        f_src_ip = s_ip.src
                        f_dst_ip = s_ip.dst
                        f = False
                    relative_timestamp = packet.time - first_packet_time
                    pay_len = 0
                    ip_layer = packet[IP]
                    src_ip = ip_layer.src
                    ip_layer.src = 0
                    ip_layer.dst = 0
                    if packet.haslayer(TCP):
                        tcp_layer = packet[TCP]
                        pay_len = len(tcp_layer.payload)

                        # 将源端口和目的端口设置为0
                        tcp_layer.sport = 0
                        tcp_layer.dport = 0

                        # 数据, 52 字节， 20 + 32
                        raw_data = bytes(ip_layer)[:80]

                    elif packet.haslayer(UDP):
                        udp_layer = packet[UDP]
                        pay_len = len(udp_layer.payload)
                        # 将源端口和目的端口设置为0
                        udp_layer.sport = 0
                        udp_layer.dport = 0
                        # 20 + 8
                        raw_data = bytes(ip_layer)[:30]

                    if pay_len == 0 or pay_len > 2000 or pay_len == 6:
                        continue

                    # if is_ack_packet(packet):
                    #     continue

                    if src_ip == f_src_ip:
                        pay_len = pay_len
                    elif src_ip == f_dst_ip:
                        pay_len = -pay_len
                    else:
                        unknow_dir_packet += 1
                    hex_data = ' '.join(['%02X' % byte for byte in raw_data])
                    line = f"{relative_timestamp}\t{pay_len}\t{hex_data}\n"
                    # line = f"{relative_timestamp}\t{pay_len}\n"
                    file.write(line)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def extract_3_colum_feature_from_pcap_tor_cell(data_tuple):
    try:
        pcap_path = data_tuple.split('%%')[0]
        save_path = data_tuple.split('%%')[1]
        # 创建TSV文件并写入标题行
        tsv_file = save_path
        # 使用 with 语句打开文件，确保在退出代码块时文件被关闭
        if os.path.exists(tsv_file):
            print("exist, return! ")
            return
        # 读取PCAP文件
        pcap_file = pcap_path
        packets = rdpcap(pcap_file,count=2000)

        # 检查数据包数量是否为0或小于10
        if len(packets) == 0 or len(packets) < 4:
            print("The number of packets is 0 .")
            return

        with open(tsv_file, 'w') as file:
            # 计算第一个数据包的时间戳
            first_packet_time = packets[0].time
            s_ip = packets[0].getlayer('IP')
            f_src_ip = s_ip.src
            f_dst_ip = s_ip.dst

            unknow_dir_packet = 0

            # 遍历前10个数据包并将它们写入TSV文件
            for i, packet in enumerate(packets):
                relative_timestamp = packet.time - first_packet_time
                if packet.haslayer(IP):
                    pay_len = 0
                    ip_layer = packet[IP]
                    src_ip = ip_layer.src
                    ip_layer.src = 0
                    ip_layer.dst = 0
                    if packet.haslayer(TCP):
                        tcp_layer = packet[TCP]
                        pay_len = len(tcp_layer.payload)

                        # 将源端口和目的端口设置为0
                        tcp_layer.sport = 0
                        tcp_layer.dport = 0

                        # 数据, 52 字节， 20 + 32
                        raw_data = bytes(ip_layer)[:80]

                    elif packet.haslayer(UDP):
                        udp_layer = packet[UDP]
                        pay_len = len(udp_layer.payload)
                        # 将源端口和目的端口设置为0
                        udp_layer.sport = 0
                        udp_layer.dport = 0
                        # 20 + 8
                        raw_data = bytes(ip_layer)[:30]


                    cell_len = 512
                    cell_num = pay_len // cell_len

                    if cell_num < 1:
                        continue

                    if src_ip == f_src_ip:
                        cell_len = cell_len
                    elif src_ip == f_dst_ip:
                        cell_len = -cell_len
                    else:
                        unknow_dir_packet += 1
                    hex_data = ' '.join(['%02X' % byte for byte in raw_data])
                    for i in range(cell_num):
                        line = f"{relative_timestamp}\t{cell_len}\t{hex_data}\n"
                        file.write(line)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def extract_feature_from_pcap(data_tuple):
    try:
        pcap_path = data_tuple.split('%%')[0]
        save_path = data_tuple.split('%%')[1]
        # 创建TSV文件并写入标题行
        tsv_file = save_path
        # 使用 with 语句打开文件，确保在退出代码块时文件被关闭
        if os.path.exists(tsv_file):
            print("exist, return! ")
            return
        # 读取PCAP文件
        pcap_file = pcap_path
        print(pcap_path)
        packets = rdpcap(pcap_file,count=200)

        # 检查数据包数量是否为0或小于10
        if len(packets) == 0 or len(packets) < 10:
            print("The number of packets is 0 .")
            return
        packet_num = 0

        src_ip_list = ["192.254.0.3", "192.254.0.4", "192.254.0.5", "192.254.0.6"]
        dst_ip_list = ["192.254.0.11", "192.254.0.12", "192.254.0.13", "192.254.0.14"]
        mkcp_port_list = [50004, 40004, 50002, 40002, 40003, 50003]

        with open(tsv_file, 'w') as file:

            # 计算第一个数据包的时间戳
            first_packet_time = packets[0].time

            s_ip = packets[0].getlayer('IP')
            f_src_ip = s_ip.src
            f_dst_ip = s_ip.dst

            unknow_dir_packet = 0

            # 遍历前10个数据包并将它们写入TSV文件
            for i, packet in enumerate(packets):
                # if packet.haslayer(TCP) and packet.haslayer(Raw):
                # if packet.haslayer(UDP) or packet.haslayer(TCP):
                #
                #     relative_timestamp = packet.time - first_packet_time
                #
                #     ip = packet.getlayer('IP')
                #     src_ip = ip.src
                #     dst_ip = ip.dst
                #
                #     p_len = len(packet)
                #
                #     if src_ip == f_src_ip:
                #         payload_length = p_len
                #     elif src_ip == f_dst_ip:
                #         payload_length = -p_len
                #
                #     line = f"{relative_timestamp}\t{payload_length}\n"
                #     file.write(line)

                    # 获取源IP、目标IP、源端口和目标端口
                    if packet.haslayer('UDP'):
                        relative_timestamp = packet.time - first_packet_time
                        tcp = packet.getlayer('UDP')
                        ip = packet.getlayer('IP')
                        src_ip = ip.src

                        source_port = tcp.sport
                        dest_port = tcp.dport
                        payload_length = len(tcp.payload)
                        if src_ip == f_src_ip:
                            payload_length = payload_length
                        elif src_ip == f_dst_ip:
                            payload_length = -payload_length
                        else:
                            unknow_dir_packet += 1
                        # # 判断数据包的方向
                        # if source_port == 443:
                        #     payload_length = -payload_length
                        # elif dest_port == 443:
                        #     payload_length = payload_length
                        # else:
                        #     unknow_dir_packet += 1

                        # # 判断数据包的方向
                        # if source_port in mkcp_port_list:
                        #     payload_length = -payload_length
                        # elif dest_port in mkcp_port_list:
                        #     payload_length = payload_length
                        # else:
                        #     unknow_dir_packet += 1

                        # if src_ip in src_ip_list:
                        #     payload_length = payload_length
                        # elif src_ip in dst_ip_list:
                        #     payload_length = -payload_length
                        # else:
                        #     unknow_dir_packet += 1
                        # 将信息写入文本文件，以制表符分隔
                        line = f"{relative_timestamp}\t{payload_length}\n"
                        file.write(line)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def split_1000_from_pcap(pcap_path, save_path):
    try:
        # pcap_path = data_tuple.split('%%')[0]
        # save_path = data_tuple.split('%%')[1]
        # 创建TSV文件并写入标题行
        tsv_file = save_path
        # 使用 with 语句打开文件，确保在退出代码块时文件被关闭
        # if os.path.exists(tsv_file):
        #     print("exist, return! ")
        #     return
        # 读取PCAP文件
        pcap_file = pcap_path
        print(pcap_path)
        packets = rdpcap(pcap_file,count=2000000)


        offset = len(packets) // 500

        print(offset)

        # 检查数据包数量是否为0或小于10
        if len(packets) == 0 or len(packets) < 500:
            print("The number of packets is 0 .")
            return
        src_ip_list = ["192.254.0.3", "192.254.0.4", "192.254.0.5", "192.254.0.6"]
        dst_ip_list = ["192.254.0.11", "192.254.0.12", "192.254.0.13", "192.254.0.14"]
        port_list = [50004, 40004, 50002, 40002, 40003, 50003]

        for i in range(500):
            print(" -------------第 {} 个-----------------".format(i) )
            writer = PcapWriter(os.path.join(save_path,os.path.basename(pcap_path).split('.')[0] + '-' + str(i) + '.pcap'), append = True)
            start = False
            for j in range(offset):
                print("单个pcap 第 {} 个".format(j))
                if packets[j + i * offset].haslayer('UDP'):
                    tcp = packets[j + i * offset].getlayer('UDP')
                    source_port = tcp.sport
                    dest_port = tcp.dport
                    # 判断数据包的方向
                    if dest_port in port_list or start:
                        writer.write(packets[j + i * offset])
                        start = True
                else:
                    continue
            writer.close()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def turn_pcap_to_pean_txt(data):
    try:
        pcap_path = data.split('%%')[0]
        save_path = data.split('%%')[1]
        label = data.split('%%')[2]
        # 读取PCAP文件
        pcap_file = pcap_path
        print(pcap_path)
        if os.path.exists(save_path):
            return
    # 默认：10
        packets = rdpcap(pcap_file,count=20)

        # 检查数据包数量是否为0或小于10
        if len(packets) == 0 or len(packets) < 20:
            print("The number of packets is 0 or less than 10.")
            return

        # 创建TSV文件并写入标题行
        tsv_file = save_path

        # 使用 with 语句打开文件，确保在退出代码块时文件被关闭


        with open(tsv_file, 'w') as file:

            # 计算第一个数据包的时间戳
            first_packet_time = packets[0].time

            # 创建一个列表来存储行的数据
            row_data = []

            len_data = []

            # 遍历前10个数据包并将它们写入TSV文件
            for i, packet in enumerate(packets[:20]):
                if packet.haslayer(TCP) and packet.haslayer(Raw):
                    # 获取TCP层数据
                    tcp_layer = packet[TCP]

                    ip_layer = packet[IP]

                    ip_layer.src = 0
                    ip_layer.dst = 0

                    # 将源端口和目的端口设置为0
                    tcp_layer.sport = 0
                    tcp_layer.dport = 0
                    payload_length = len(packet[Raw])

                    # 获取TCP头和应用层负载数据
                    tcp_data = bytes(ip_layer)[:400]
                    len_data.append(str(payload_length))

                else:
                    len_data.append(str(0))
                    payload_data = b'\x00' * 400
                hex_data = ' '.join(['%02X' % byte for byte in tcp_data])
                file.write(hex_data + '\t')  # 使用制表符分隔包


                # row_data.append(f"{tcp_data.hex()}\t{payload_length}\t")

            # 将行数据写入文件
            file.write(' '.join(len_data) + "\t" + label)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def extract_2_column_raw_feature_from_pcap(data_tuple):
    try:
        pcap_path = data_tuple.split('%%')[0]
        save_path = data_tuple.split('%%')[1]
        # 创建TSV文件并写入标题行
        tsv_file = save_path
        # 使用 with 语句打开文件，确保在退出代码块时文件被关闭
        if os.path.exists(tsv_file):
            print("exist, return! ")
            return
        # 读取PCAP文件
        pcap_file = pcap_path
        packets = rdpcap(pcap_file,count=200)

        # 检查数据包数量是否为0或小于10
        if len(packets) == 0 or len(packets) < 20:
            print("The number of packets is 0 .")
            return

        with open(tsv_file, 'w') as file:

            # 计算第一个数据包的时间戳
            first_packet_time = packets[0].time

            # 遍历前10个数据包并将它们写入TSV文件
            for i, packet in enumerate(packets):
                relative_timestamp = packet.time - first_packet_time

                if packet.haslayer(IP):
                    pay_len = 0
                    ip_layer = packet[IP]
                    ip_layer.src = 0
                    ip_layer.dst = 0
                    if packet.haslayer(TCP):

                        tcp_layer = packet[TCP]
                        pay_len = len(tcp_layer.payload)

                        # 将源端口和目的端口设置为0
                        tcp_layer.sport = 0
                        tcp_layer.dport = 0

                        # 数据, 52 字节， 20 + 32
                        raw_data = bytes(ip_layer)[:52]

                    elif packet.haslayer(UDP):
                        udp_layer = packet[UDP]
                        pay_len = len(udp_layer.payload)
                        # 将源端口和目的端口设置为0
                        udp_layer.sport = 0
                        udp_layer.dport = 0
                        # 20 + 8
                        raw_data = bytes(ip_layer)[:28]

                    if pay_len == 0:
                        continue
                    hex_data = ' '.join(['%02X' % byte for byte in raw_data])
                    line = f"{relative_timestamp}\t{hex_data}\n"
                    file.write(line)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def turn_txt_to_pean_txt_len(data):
    try:
        input_file_path = data.split('%%')[0]
        output_file_path = data.split('%%')[1]
        byte_data = [str(i * 100) for i in range(10)]
        label = data.split('%%')[2]
        print(output_file_path, label)
        len_data = []
        index = 0
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line in infile:
                if index >= 10:
                    break
                columns = line.split('\t')  # 假设列之间使用空格分隔
                if len(columns) == 4:
                    len_data.append(str(int(columns[2])))
                index += 1
            true_byte_data = '\t'.join(byte_data)
            true_len_data = ' '.join(len_data)
            outfile.write(true_byte_data + '\t' + true_len_data + '\t' + label)
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def extract_2_feature_from_json(data_tuple):
    try:
        pcap_path = data_tuple.split('%%')[0]
        save_path = data_tuple.split('%%')[1]
        # 创建TSV文件并写入标题行
        tsv_file = save_path
        # 使用 with 语句打开文件，确保在退出代码块时文件被关闭
        if os.path.exists(tsv_file):
            print("exist, return! ")
            return
        with open(pcap_path, 'r') as file:
            data = json.load(file)

        for k, v in data.items():
            time_list = v["time_list"]
            dir_list = v["dir_list"]
            len_list = v["len_list"]

            if len(time_list) < 20:
                print("The number of packets is less 20 .")
                return

            with open(tsv_file, 'w') as file:
                # 遍历前10个数据包并将它们写入TSV文件
                for i in range(len(time_list)):
                    if int(dir_list[i]) == 0:
                        line = f"{time_list[i]}\t{len_list[i]}\n"
                    elif int(dir_list[i]) == 1:
                        line = f"{time_list[i]}\t{-len_list[i]}\n"
                    else:
                        return
                    file.write(line)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    s_path = r"D:\数据集\mKCP\mKCP_total3"
    save_dir = r"D:\处理的数据集\mKCP\mKCP_source2"
    # d_list = os.listdir(s_path)
    # a = time.time()
    # i = 0
    # for p in d_list:
    #     i += 1
    #     p_dir = os.path.join(s_path,p)
    #     tp_list = os.listdir(p_dir)
    #     save_path = os.path.join(save_dir,p)
    #     os.makedirs(save_path,exist_ok=True)
    #     for pcap in tp_list:
    #         t_path = os.path.join(p_dir,pcap)
    #         if os.path.isfile(t_path):
    #             split_1000_from_pcap(t_path,save_path)
    #             print("第 {} 个， 时间是 ： {}".format(i,time.time() - a) )



    input_dir = r"D:\数据集\mobile_app_8_2025\instagram\04010921.instagram.pcap"
    output_dir = r"C:\Users\Administrator\Desktop\22\2.txt"
    # os.makedirs(output_dir,exist_ok=True)
    data = input_dir + "%%" + output_dir + "%%" + "1"
    extract_3_colum_feature_from_pcap(data)