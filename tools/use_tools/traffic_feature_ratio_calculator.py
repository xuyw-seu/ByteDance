#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流量特征比例计算器
用于计算网络流量数据中请求和响应的比例特征
主要功能：
1. 从文本文件中读取网络流量数据
2. 分离请求和响应数据
3. 计算请求和响应的比例特征
4. 支持两种比例计算方式：
   - 总体比例：所有数据的总长度与预期长度的比例
   - 平均比例：每个子列表的比例的平均值
"""

def caculate_ratio(file_path, per_num):
    try:
        with open(file_path, 'r') as file:
            res_list = []
            req_list = []
            for line in file:
                columns = line.strip().split('\t')  # 假设使用制表符分隔列
                columns = columns[:-1]
                # print(columns)
                for i,data in enumerate(columns):
                    sub = data.strip().split(' ')
                    sub_len = 0
                    for j in sub:
                        if j == '':
                            continue
                        sub_len += 1
                    if i < (len(columns) // 2):
                        req_list.append(sub_len)
                    else:
                        res_list.append(sub_len)
            # print(req_list, '\n', res_list, '\n', sum(req_list))
            req_ratio = sum(req_list) / (len(req_list) * per_num)
            res_ratio = sum(res_list) / (len(res_list) * per_num)

        return req_ratio,res_ratio
    except FileNotFoundError:
        print(f"File not found.")

def caculate_ratio_avg(file_path, per_num):
    try:
        with open(file_path, 'r') as file:
            res_list = []
            req_list = []
            total_ration = 0
            n = 0
            for line in file:
                columns = line.strip().split('\t')  # 假设使用制表符分隔列
                columns = columns[:-1]
                # print(columns)
                for i,data in enumerate(columns):
                    sub = data.strip().split(' ')
                    sub_len = 0
                    for j in sub:
                        if j == '':
                            continue
                        sub_len += 1
                    sub_ratio = sub_len / per_num
                    if i < (len(columns) // 2):
                        req_list.append(sub_ratio)
                    else:
                        res_list.append(sub_ratio)
            # print(req_list, '\n', res_list, '\n', sum(req_list))
            req_ratio = sum(req_list) / (len(req_list))
            res_ratio = sum(res_list) / (len(res_list))

        return req_ratio,res_ratio
    except FileNotFoundError:
        print(f"File not found.")


if __name__ == '__main__':
    file_path = r"D:\处理的数据集\TLS1.3\tls1.3-header-10-100-s.txt" # 0.09509932483918414 0.17471566826787227
    # file_path = r"D:\处理的数据集\TOR\tor-header-100-4000-s.txt" # 0.04054331550802139 0.6665018285319994
    # file_path = r"D:\处理的数据集\mobile_app_8_class\mobile_8_100-10-s.txt" # 0.2702875312533971 0.2871852621852622
    # file_path = r"D:\处理的数据集\QUIC\test\quic-500-50-s.txt" # 0.34427488740487655 0.6387345861158565


    file_path = r"E:\论文复现\多模态-PEAN\PEAN-main - re\TrafficData\trojan-eq-500-50-s.txt" # 0.18896223291672087 0.4834198100663645
    file_path = r"E:\论文复现\多模态-PEAN\PEAN-main - re\TrafficData\trojan-eq-500-100-s.txt"
    # file_path = r"D:\处理的数据集\mKCP-NEW\mkcp-new-16-100-10-s.txt" # 0.353322632423756 0.6412427135253865
    req, res = caculate_ratio_avg(file_path,5)
    print(req,res)