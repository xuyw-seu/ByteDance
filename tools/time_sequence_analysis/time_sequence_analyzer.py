#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间序列分析器
用于网络流量时间序列数据的分析，主要功能包括：
1. 检测时间序列中的突变点（find_timestamp_transitions）
2. 根据突变区间生成时间子图分布（get_sub_plot）

适用场景：
- 网络流量时间序列分析
- 流量模式识别
- 时间突变检测
- 流量子图生成
"""
def find_timestamp_transitions(timestamp_list, threshold=100):
    transitions = []

    current_length = 0

    for i in range(1, len(timestamp_list)):
        if i == 1:
            time_diff = timestamp_list[i]
        else:
            time_diff = (timestamp_list[i] - timestamp_list[i - 1]) / (timestamp_list[i - 1] - timestamp_list[i - 2])

        if time_diff > threshold:
            # 如果时间差大于阈值，表示发生了时间突变
            transitions.append(i)
        else:
            # 如果时间差小于等于阈值，增加当前突变序列的长度
            current_length += 1

    n_list = []
    for g in range(len(transitions)):
        if g == 0:
            n_list.append(int(transitions[g]))
        else:
            n_list.append(int(transitions[g]) - int(transitions[g - 1]))

    n_list.append(len(timestamp_list) - transitions[-1])
    return transitions, n_list


def get_sub_plot(time_list, time_burst_list, n):
    scale_list = []
    total_list = []
    for i in time_burst_list:
        data = round(i * n // sum(time_burst_list))
        if data == 0:
            data = 1
        scale_list.append(data)
    last_group = n - sum(scale_list)
    if last_group != 0:
        max_index = scale_list.index(max(scale_list))
        scale_list[max_index] += last_group

    g = 0
    for j, v in enumerate(scale_list):
        time_plot = (time_list[sum(time_burst_list[0:j + 1]) - 1] - time_list[sum(time_burst_list[0:j])]) / v
        print(time_plot,time_list[sum(time_burst_list[0:j + 1]) - 1], time_list[sum(time_burst_list[0:j])], time_burst_list[j])
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
        print(sub_time_list)
    print(scale_list,total_list)
# 示例
timestamp_list = [0, 1, 2, 5, 8, 10, 12, 15, 18, 20, 1000, 2000, 3000,4000,5000, 100000, 200000]
threshold = 10
transitions,sequence_lengths = find_timestamp_transitions(timestamp_list, threshold)

print("时间突变开始位置：", transitions)
print("突变序列的长度：", sequence_lengths)

get_sub_plot(timestamp_list,sequence_lengths,5)
