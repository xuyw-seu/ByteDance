"""
ByteDance Model - Multimodal Traffic Classification Model

This implementation is based on the PEAN (Packet Embedding with Attention Networks) model,
specifically the "mix-fuse-3" feature configuration for multimodal traffic classification.

The model processes three types of features:
- Length sequences (packet lengths)
- Raw byte sequences (packet payloads)
- Time sequences (packet timestamps)

Paper: [Please add paper reference here]
Authors: [Please add authors here]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from timm.models.layers import trunc_normal_, lecun_normal_
from ByteTransformer import ByteBlockTransformerEncoder


# =============================================================================
# TRF Module - Transformer-based Feature Extractors
# =============================================================================

class Scaled_Dot_Product_Attention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context, attention


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(num_head * self.dim_head, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5
        context, alpha = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out, alpha


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out, alpha = self.attention(x)
        out = self.feed_forward(out)
        return out, alpha


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(x.device)
        out = self.dropout(out)
        return out


class Model(nn.Module):
    """Transformer model for length/time sequences"""
    def __init__(self, config, if_cls_token=False):
        super(Model, self).__init__()
        self.dim_model = config.embedding_size
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = config.trf_heads
        self.num_encoder = config.trf_layers
        self.if_cls_token = if_cls_token

        self.postion_embedding = Positional_Encoding(
            self.dim_model,
            config.max_packet_num + int(self.if_cls_token),
            config.dropout,
            config.device
        )
        self.encoder = Encoder(self.dim_model, self.num_head, self.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(self.num_encoder)
        ])

        self.tanh = nn.Tanh()

        if self.if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim_model))
            trunc_normal_(self.cls_token, std=.02)
            self.cls_token.requires_grad_(True)

    def forward(self, x):
        if self.if_cls_token:
            B, N, D = x.shape
            cls_token = self.cls_token.expand(B, -1, -1).to(x.device)
            x = torch.cat((cls_token, x), dim=1)

        out = self.postion_embedding(x)
        for encoder in self.encoders:
            out, alpha = encoder(out)

        if self.if_cls_token:
            cls_output = out[:, 0, :]
            other_output = out[:, 1:, :]
            return cls_output, other_output

        return out


class Model_RAW(nn.Module):
    """Transformer model for raw byte sequences"""
    def __init__(self, config, if_cls_token=False):
        super(Model_RAW, self).__init__()
        self.dim_model = 80  # Fixed dimension for raw bytes
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = config.trf_heads
        self.num_encoder = config.trf_layers_res
        self.if_cls_token = if_cls_token

        self.postion_embedding = Positional_Encoding(
            self.dim_model,
            config.max_packet_num + int(self.if_cls_token),
            config.dropout,
            config.device
        )
        self.encoder = Encoder(self.dim_model, self.num_head, self.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(self.num_encoder)
        ])

        self.tanh = nn.Tanh()
        if self.if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim_model))
            trunc_normal_(self.cls_token, std=.02)
            self.cls_token.requires_grad_(True)

    def forward(self, x):
        if self.if_cls_token:
            B, N, D = x.shape
            cls_token = self.cls_token.expand(B, -1, -1).to(x.device)
            x = torch.cat((cls_token, x), dim=1)

        out = self.postion_embedding(x)
        for encoder in self.encoders:
            out, alpha = encoder(out)

        if self.if_cls_token:
            cls_output = out[:, 0, :]
            other_output = out[:, 1:, :]
            return cls_output, other_output

        return out


class Model_Time(nn.Module):
    """Transformer model for time sequences"""
    def __init__(self, config, if_cls_token=False):
        super(Model_Time, self).__init__()
        self.dim_model = config.embedding_size
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = config.trf_heads
        self.num_encoder = config.trf_layers_res
        self.if_cls_token = if_cls_token

        self.postion_embedding = Positional_Encoding(
            self.dim_model,
            config.max_packet_num + int(self.if_cls_token),
            config.dropout,
            config.device
        )
        self.encoder = Encoder(self.dim_model, self.num_head, self.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(self.num_encoder)
        ])

        self.tanh = nn.Tanh()

        if self.if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim_model))
            trunc_normal_(self.cls_token, std=.02)
            self.cls_token.requires_grad_(True)

    def forward(self, x):
        if self.if_cls_token:
            B, N, D = x.shape
            cls_token = self.cls_token.expand(B, -1, -1).to(x.device)
            x = torch.cat((cls_token, x), dim=1)

        out = self.postion_embedding(x)
        for encoder in self.encoders:
            out, alpha = encoder(out)

        if self.if_cls_token:
            cls_output = out[:, 0, :]
            other_output = out[:, 1:, :]
            return cls_output, other_output

        return out


# =============================================================================
# ByteDance Model - Main Classification Model
# =============================================================================

class ByteDance(nn.Module):
    """
    ByteDance: Multimodal Traffic Classification Model

    This model processes three modalities of network traffic:
    1. Packet length sequences
    2. Raw byte sequences (packet payloads)
    3. Packet timestamp sequences

    The model fuses these modalities using transformer encoders and GRU networks
    for comprehensive traffic classification.
    """

    def __init__(self, config):
        """
        Initialize ByteDance model

        Args:
            config: Configuration object containing model hyperparameters
        """
        super(ByteDance, self).__init__()
        self.config = config
        self.emb_size = config.embedding_size

        # Length sequence embedding and transformer
        self.length_embedding = nn.Embedding(4000, config.embedding_size, padding_idx=0)
        self.TRF_LEN = Model(config=self.config, if_cls_token=True)
        self.TRF_RAW = Model_RAW(config=self.config, if_cls_token=True)

        # Byte-level transformer encoder
        self.local_encoder = ByteBlockTransformerEncoder(
            vocab_size=256,
            embed_dim=80,
            nhead=8,
            num_layers=1,
            pooling='mean'
        )

        # Feature fusion using bidirectional GRU
        self.fuse = nn.GRU(
            208, 256, 1,  # input_size=208 (128+80), hidden_size=256, num_layers=1
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout
        )

        # Final classification layer
        self.fuse_fc = nn.Linear(512, config.num_classes)  # 256*2 for bidirectional

    def forward(self, x):
        """
        Forward pass through ByteDance model

        Args:
            x: Tuple of (len_seq, byte_seq)
                - len_seq: Packet length sequences [batch_size, max_packet_num]
                - byte_seq: Raw byte sequences [batch_size, max_packet_num, 80]

        Returns:
            tuple: (final_output, out_raw_cls, out_len_cls)
                - final_output: Classification logits [batch_size, num_classes]
                - out_raw_cls: Raw byte CLS token [batch_size, 80]
                - out_len_cls: Length CLS token [batch_size, embedding_size]
        """
        len_seq, byte_seq = x

        # Process length sequences
        hidden_feature = self.length_embedding(len_seq)
        hidden_feature = hidden_feature.reshape(-1, self.config.max_packet_num, self.config.embedding_size)
        out_len_cls, out_len = self.TRF_LEN(hidden_feature)

        # Process raw byte sequences with local encoder
        block_boundaries = [20, 52, 80]  # Three variable-length blocks
        output_feature3 = byte_seq

        # Reshape to (batch_size * max_packet_num, 80) for individual processing
        x_reshaped_3 = output_feature3.view(-1, output_feature3.size(2))

        # Feature extraction with ByteBlockTransformerEncoder
        features = self.local_encoder(x_reshaped_3, block_boundaries=block_boundaries)

        # Reshape back to (batch_size, max_packet_num, embed_dim)
        features = features.view(output_feature3.size(0), output_feature3.size(1), features.size(1))

        # Get CLS token and sequence features from raw transformer
        out_raw_cls, out_raw = self.TRF_RAW(features)

        # Concatenate length and raw byte features
        concatenated_tensor = torch.cat((out_len, out_raw), dim=2)

        # Fuse features with bidirectional GRU
        middle_layer, _ = self.fuse(concatenated_tensor)
        f_out = middle_layer[:, -1, :]  # Take the last timestep output

        # Final classification
        final_output = self.fuse_fc(f_out)

        return final_output, out_raw_cls, out_len_cls


# =============================================================================
# Configuration Class
# =============================================================================

class Config:
    """
    Configuration class for ByteDance model (mix-local implementation)
    """
    def __init__(self,
                 embedding_size=128,
                 max_packet_num=20,
                 num_classes=10,
                 trf_heads=8,
                 trf_layers=2,
                 trf_layers_res=2,
                 dropout=0.1,
                 device='cpu'):  # Changed default to 'cpu' for testing
        self.embedding_size = embedding_size
        self.max_packet_num = max_packet_num
        self.num_classes = num_classes
        self.trf_heads = trf_heads
        self.trf_layers = trf_layers
        self.trf_layers_res = trf_layers_res
        self.dropout = dropout
        self.device = device


if __name__ == "__main__":
    # Example usage
    config = Config(device='cpu')  # Use CPU for testing
    model = ByteDance(config).to('cpu')  # Ensure model is on CPU

    # Example input shapes
    batch_size = 32
    max_packet_num = 20

    len_seq = torch.randint(0, 4000, (batch_size, max_packet_num))  # Length sequences
    byte_seq = torch.randint(0, 256, (batch_size, max_packet_num, 80))  # Raw byte features (0-255)

    # Forward pass
    output, raw_cls, len_cls = model((len_seq, byte_seq))

    print(f"Output shape: {output.shape}")  # [batch_size, num_classes]
    print(f"Raw CLS shape: {raw_cls.shape}")  # [batch_size, 80]
    print(f"Length CLS shape: {len_cls.shape}")  # [batch_size, embedding_size]
    print("ByteDance model test passed!")
