import dpkt
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import TensorDataset, DataLoader
import json


def pcap2npy_GLADS(paths, Np=32, Nb=768):
    out = np.zeros((len(paths), Np * 8 + Nb), dtype=np.float32)
    for idx, path in enumerate(paths):
        # print(path)
        with open(path, 'rb') as file:
            packets = dpkt.pcap.Reader(file)
            ptype = packets.datalink()  # 数据连接类型，DLT_EN10MB表示包含链路层
            src_ip = 0  # 记录源IP，用于计算数据包方向
            last_ts = 0  # 记录上一个包的时间戳，用于记录到达间隔时间
            total_payload_len = 0
            # print(path)
            for num, (ts, buf) in enumerate(packets):
                ip = dpkt.ethernet.Ethernet(buf).data if ptype == dpkt.pcap.DLT_EN10MB else dpkt.ip.IP(buf)
                #ip = dpkt.ethernet.Ethernet(buf).data
                trans = ip.data  # 传输层数据
                if not isinstance(trans, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    continue

                if num >= Np and total_payload_len >= Nb:
                    break

                if src_ip == 0:  # 记录第一个包的源ip和时间戳
                    src_ip = ip.src
                    last_ts = ts

                now_idx = min(num, Np) * 8 + total_payload_len
                if num < Np:
                    out[idx, now_idx] = len(trans.data)
                    out[idx, now_idx + 1] = trans.win if isinstance(trans, dpkt.tcp.TCP) else 0
                    out[idx, now_idx + 2] = ts - last_ts
                    out[idx, now_idx + 3] = (ip.src != src_ip)
                    last_ts = ts
                    out[idx, now_idx:now_idx + 4] = np.log(1 + out[idx, now_idx:now_idx + 4])
                    out[idx, now_idx + 4:now_idx + 8] = np.array([1, 0, 1, 0])
                    now_idx += 8
                if total_payload_len < Nb:
                    for i in range(min(128, len(trans.data))):
                        out[idx, now_idx + i] = (trans.data[i] & 0xff) / 255
                        total_payload_len += 1
                        if (total_payload_len >= Nb):
                            break
                    if total_payload_len % 4 != 0:
                        total_payload_len = ((total_payload_len // 4) + 1) * 4
    return out

def json2npy_GLADS(paths, Np=32, Nb=768):
    out = np.zeros((len(paths), Np * 8 + Nb), dtype=np.float32)
    for idx, path in tqdm(enumerate(paths), total=len(paths)):
        #print(path)

        with open(path, 'r') as json_file:
            json_dict = json.load(json_file)
            feature_dict = list(json_dict.values())[0]
            #print(len(feature_dict['raw_bytes_list'][0]))

            dir_list = feature_dict['dir_list']
            time_list = feature_dict['time_list']
            raw_byte_list = feature_dict['raw_bytes_list']
            window_list = feature_dict['tcp_window_list']
            total_payload_len = 0
            # print(path)
            for num in range(len(dir_list)): #num遍历每个包

                if num >= Np and total_payload_len >= Nb:
                    break

                now_idx = min(num, Np) * 8 + total_payload_len
                if num < Np:
                    out[idx, now_idx] = len(raw_byte_list[num])
                    out[idx, now_idx + 1] = window_list[num]
                    out[idx, now_idx + 2] = time_list[num] - time_list[max(0, num - 1)]
                    out[idx, now_idx + 3] = dir_list[num]

                    out[idx, now_idx:now_idx + 4] = np.log(1 + out[idx, now_idx:now_idx + 4])
                    out[idx, now_idx + 4:now_idx + 8] = np.array([1, 0, 1, 0])
                    now_idx += 8
                if total_payload_len < Nb:
                    for i in range(min(128, len(raw_byte_list[num]))):
                        out[idx, now_idx + i] = (raw_byte_list[num][i] & 0xff) / 255
                        total_payload_len += 1
                        if (total_payload_len >= Nb):
                            break
                    if total_payload_len % 4 != 0:
                        total_payload_len = ((total_payload_len // 4) + 1) * 4
    return out

def path2label(path, start, num, label_dict):
    out = ''
    sp = path.lower().split('\\')
    for i in range(start, start + num):
        label_dict[i - start].setdefault(sp[i], len(label_dict[i - start]))
        out = out + str(label_dict[i - start][sp[i]]) + '_'

    return out[:-1]

def str2npy(input):
    return np.array([[int(t) for t in x.split('_')] for x in input])

def KFoldIdxGen(base, start, num, n_split, shuffle=False, random_state=None, sample_num=0, upper=0, lower=0):
    path_arr = []
    all_labels = []
    label_dict = [dict() for _ in range(num)]
    for root, ds, fs in os.walk(base):
        if len(fs) == 0:
            continue
        label_str = path2label(root, start, num, label_dict)
        all_labels.append(label_str)
        sample_list = fs
        for f in sample_list:
            full_name = os.path.join(root, f)
            path_arr.append(full_name)

    print([len(x) for x in label_dict])
 

    encoded_label, unique_label = pd.factorize(np.array(all_labels))

    int2str = dict(zip(encoded_label, unique_label))
    str2int = dict(zip(unique_label, encoded_label))

    path_ndr = np.array(path_arr)
    label_encoded = np.array([str2int[path2label(x, start, num, label_dict)] for x in path_ndr])

    skf = StratifiedKFold(n_splits=n_split, shuffle=shuffle, random_state=random_state)
    gen = skf.split(path_ndr, label_encoded)

    return gen, path_ndr, str2npy([int2str[x] for x in label_encoded])

def npy2dataloader(data, labels, batch_size=128, shuffle=False):
    data_t = torch.tensor(data, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(data_t, labels_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)