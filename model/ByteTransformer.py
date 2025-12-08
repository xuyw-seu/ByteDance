import torch
import torch.nn as nn


class ByteBlockTransformerEncoder(nn.Module):
    def __init__(self,
                 vocab_size=256,
                 embed_dim=128,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=512,
                 dropout=0.1,
                 pooling='mean'):  # pooling: 'mean' | 'cls' | None
        super().__init__()
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, block_boundaries):
        """
        x: (batch, seq_len), 每个值 ∈ [0,255]
        block_boundaries: list of int, 表示每块的结束位置，例如 [20, 50, 90]
        """
        batch_size, seq_len = x.shape
        assert block_boundaries[-1] == seq_len, "block_boundaries最后一个值应等于seq_len"

        # Embedding + Positional Encoding
        x_embed = self.embedding(x)
        x_embed = self.pos_encoder(x_embed)

        # 构造注意力mask（block-local）
        attn_mask = self.build_variable_block_mask(seq_len, block_boundaries).to(x.device)

        # Transformer Encoder
        out = self.transformer(x_embed, mask=attn_mask)  # (batch, seq_len, embed_dim)

        if self.pooling == 'mean':
            return out.mean(dim=1)  # (batch, embed_dim)
        elif self.pooling == 'cls':
            return out[:, 0, :]
        else:
            return out  # (batch, seq_len, embed_dim)

    def build_variable_block_mask(self, seq_len, boundaries):
        """构建多块局部注意力mask（变长）"""
        mask = torch.full((seq_len, seq_len), float('-inf'))
        start = 0
        for end in boundaries:
            mask[start:end, start:end] = 0  # 每块允许内部注意
            start = end
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # a = x
        # b = x + self.pe[:, :x.size(1)]
        return x



if __name__ == '__main__':
    model = ByteBlockTransformerEncoder(
        vocab_size=256,
        embed_dim=128,
        nhead=8,
        num_layers=4,
        pooling='mean'
    )

    # 示例输入：3块不等长，总长度 90
    x = torch.randint(0, 256, (32, 90))  # (batch, seq_len)
    block_boundaries = [20, 50, 90]  # 三块：0-19, 20-49, 50-89

    out = model(x, block_boundaries)  # (32, 128)

    print(out.shape)
