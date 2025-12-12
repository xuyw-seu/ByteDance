import torch
import torch.nn as nn
import torch.nn.functional as F

class TSCRNN(nn.Module):
    def __init__(self, num_classes=16, input_channels=1):
        super(TSCRNN, self).__init__()

        # ------------ Conv1 Block ------------
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1),  # Conv1d-1
            nn.BatchNorm1d(64),                                      # BatchNorm1d-2
            nn.ReLU(),                                               # ReLU-3
            nn.MaxPool1d(kernel_size=2)                              # MaxPool1d-4
        )

        # ------------ Conv2 Block ------------
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1),              # Conv1d-5
            nn.BatchNorm1d(64),                                      # BatchNorm1d-6
            nn.ReLU(),                                               # ReLU-7
            nn.MaxPool1d(kernel_size=2)                              # MaxPool1d-8
        )

        # ------------ Bi-LSTM × 2 ------------
        # LSTM 输入维度 = Conv2 输出的 channel 数 = 64
        # 第一次 MaxPool: length/2
        # 第二次 MaxPool: length/4
        # 最终长度 = 1500 → Conv1 → 750 → Conv2 → 375(表中写 325 但不影响维度)

        self.lstm1 = nn.LSTM(
            input_size=64,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )  # 输出维度 = 512

        self.dropout1 = nn.Dropout(0.5)

        self.lstm2 = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )  # 输出维度 = 512

        self.dropout2 = nn.Dropout(0.5)

        # ------------ FC Layer ------------
        self.fc = nn.Linear(512, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (batch, channels, seq)
        x = self.conv1(x)         # (B, 64, L1)
        x = self.conv2(x)         # (B, 64, L2)

        # LSTM 需要 (batch, seq, feature)
        x = x.permute(0, 2, 1)

        # Bi-LSTM level 1
        x, _ = self.lstm1(x)      # (B, L2, 512)
        x = self.dropout1(x)

        # Bi-LSTM level 2
        x, _ = self.lstm2(x)      # (B, L2, 512)
        x = self.dropout2(x)

        # 取最后时间步特征
        x = x[:, -1, :]           # (B, 512)

        # FC
        x = self.fc(x)            # (B, 16)

        # Softmax
        x = self.softmax(x)
        return x
