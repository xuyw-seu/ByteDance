import torch
import torch.nn as nn
from transformers import AutoConfig,AutoModelWithLMHead,AutoModel
import TRF
import numpy as np


class Network1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network1, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 256, bidirectional=True)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(hidden_size*2, 256, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # Taking only the last timestep's output
        out = self.linear(out)
        out = self.relu(out)
        return out

class Network2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network2, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size*2, bidirectional=True)
        self.linear = nn.Linear(hidden_size*4, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # Taking only the last timestep's output
        out = self.linear(out)
        out = self.relu(out)
        return out

class Network3(nn.Module):
    def __init__(self, input_channels):
        super(Network3, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5)
        self.batchnorm4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)
        self.linear1 = nn.Linear(3456, 2560)
        self.relu5 = nn.ReLU()
        self.linear2 = nn.Linear(2560, 256)
        self.relu6 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        out = self.conv4(out)
        out = self.batchnorm4(out)
        out = self.relu4(out)
        out = self.maxpool4(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.relu5(out)
        out = self.linear2(out)
        out = self.relu6(out)
        return out

class ModelConfig():
    def __init__(self, config):
        myconfig = AutoConfig.for_model('bert').from_json_file(config.pretrainModel_json)
        model_mlm = AutoModelWithLMHead.from_config(myconfig)
        model_mlm = model_mlm.to(config.device)
        model_mlm.load_state_dict(torch.load(config.pretrainModel_path))
        model_mlm_dict = model_mlm.state_dict()  # get pre-trained parameters

        self.model_bert = AutoModel.from_config(myconfig)
        model_bert_dict = self.model_bert.state_dict()

        model_mlm_dict = {k: v for k, v in model_mlm_dict.items() if k in model_bert_dict}
        model_bert_dict.update(model_mlm_dict)
        self.model_bert.load_state_dict(model_bert_dict)

class MFF(nn.Module):
    def __init__(self, config):
        super(MFF, self).__init__()
        self.config = config
        self.mode = config.mode
        self.emb_size = config.embedding_size

        self.net1 = Network1(64, 256, 256)

        # self.net2 = Network2(2, 256, 256)

        self.net3 = Network3(1)


        self.fc1 = nn.Linear(256 * 3, 256)
        self.fc2 = nn.Linear(256, config.num_classes)
        self.relu = nn.ReLU()

        self.s_fc = nn.Linear(256, config.num_classes)

        # self.length_embedding = nn.Embedding(60000, config.length_emb_size, padding_idx=0)

        self.lstm_lll = nn.LSTM(2, 128, config.num_layers,
                          bidirectional=True, batch_first=True, dropout=config.dropout)


    def forward(self, x):
        config = self.config
        traffic_bytes_idss = x[0].float()
        length_seq = x[1]
        time_seq = x[2]
        flow_byte = x[3].float()
        ts_seq = torch.cat((length_seq, time_seq), dim=2)


        if config.feature == "ensemble":
            out_raw_packet = self.net1(traffic_bytes_idss)
            out_ts, h = self.lstm_lll(ts_seq)
            out_ts = out_ts[:, -1, :]
            out_raw_flow = self.net3(flow_byte)
            # print(length_seq, traffic_bytes_idss)

        elif config.feature == "raw_p":
            out_raw_packet = self.net1(traffic_bytes_idss)
        elif config.feature == "raw_f":
            out_raw_flow = self.net3(flow_byte)
        elif config.feature == "ts":
            # length_seq = self.length_embedding(length_seq)
            out_ts, h = self.lstm_lll(ts_seq)
            out_ts = out_ts[:, -1, :]

            # out_ts = self.net2(ts_seq)


        if config.feature == "ensemble":
            c_tensor = torch.cat((out_raw_packet, out_ts, out_raw_flow), dim=1)
            f_out = self.fc1(c_tensor)
            f_out = self.relu(f_out)
            f_out = self.fc2(f_out)
            return f_out, None, None

        elif config.feature == "raw_p":
            f_out = self.s_fc(out_raw_packet)
            return f_out, None, None
        elif config.feature == "raw_f":
            f_out = self.s_fc(out_raw_flow)
            return f_out, None, None
        elif config.feature == "ts":
            f_out = self.s_fc(out_ts)
            return f_out, None, None
