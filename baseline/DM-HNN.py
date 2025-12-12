import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelWithLMHead, AutoModel
import numpy as np
import TRF


# ============================================================
# 1. CNN1D 特征抽取模型
# ============================================================
class CNN1DModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN1DModel, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)

        # 池化
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # 全连接层（根据你原来的3840→1280→256）
        self.fc1 = nn.Linear(3840, 1280)
        self.fc2 = nn.Linear(256, num_classes)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv1 + Pool
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)

        # Conv2 + Pool
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # FC
        x = self.relu(self.fc1(x))
        return x


# ============================================================
# 2. 单层 Autoencoder
# ============================================================
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = self.decoder(x)
        return x


# ============================================================
# 3. 堆叠自动编码器 SAE（Stacked Autoencoder）
# ============================================================
class StackedAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        """
        hidden_sizes: 例如 [128, 64, 32]
        """
        super(StackedAutoencoder, self).__init__()
        self.layers = nn.ModuleList([
            Autoencoder(input_size, hidden_size)
            for hidden_size in hidden_sizes
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ============================================================
# 4. DMHNN 模型主体
# ============================================================
class DMHNN(nn.Module):
    def __init__(self, config):
        super(DMHNN, self).__init__()

        self.config = config
        self.mode = config.mode
        self.emb_size = config.embedding_size

        # ========== Length Embedding + GRU ==========
        self.length_embedding = nn.Embedding(
            num_embeddings=60000,
            embedding_dim=config.length_emb_size,
            padding_idx=0
        )

        self.lenlstm = nn.GRU(
            input_size=config.length_emb_size,
            hidden_size=config.lenlstmhidden_size,
            num_layers=config.num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=config.dropout
        )

        # ========== Byte embedding ==========
        self.emb = nn.Embedding(config.n_vocab, self.emb_size, padding_idx=0)

        # ========== SAE（Stacked Autoencoder）==========
        hidden_sizes = [128, 64, 32]
        self.sae = StackedAutoencoder(40, hidden_sizes)

        # ========== Fully Connected ==========
        self.fc = nn.Linear(40 + config.lenlstmhidden_size, config.num_classes)
        self.fc01 = nn.Linear(40, config.num_classes)
        self.fc02 = nn.Linear(config.lenlstmhidden_size, config.num_classes)

    def forward(self, x):
        traffic_bytes_idss = x[0].float()
        length_seq = x[1]

        # ========== 1. SAE 输入 ==========
        hidden_feature = traffic_bytes_idss[:, 0, :]
        hidden_feature = hidden_feature.reshape(hidden_feature.shape[0], -1)
        out1 = self.sae(hidden_feature)  # (batch, 40)

        # ========== 2. Length embedding + GRU ==========
        emb_len = self.length_embedding(length_seq)
        emb_len = emb_len.reshape(-1, self.config.pad_len_seq, self.config.length_emb_size)

        output, (final_hidden_state, final_cell_state) = self.lenlstm(emb_len)
        out2 = output[:, -1, :]  # (batch, lenlstm_hidden)

        # ========== 3. Fusion ==========
        middle_layer = torch.cat((out1, out2), dim=1)

        # ========== 4. Classifier ==========
        final_output = self.fc(middle_layer)

        return final_output, out1, out2
