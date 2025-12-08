import torch
from local_attention import LocalTransformer

# 输入二维向量 batchsize * len, 输出3维

model = LocalTransformer(
    num_tokens = 256, # 字典的长度， 也是输出最后的维度
    dim = 128,   # 字典嵌入后的维度
    depth = 6,
    max_seq_len = 8192,
    causal = True,
    local_attn_window_size = 32
).cuda()

x = torch.randint(0, 256, (128, 80)).cuda()

logits = model(x) # (1, 8192, 256)

print("99")