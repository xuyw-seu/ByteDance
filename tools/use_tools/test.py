import torch
import torch.nn as nn

# 创建一个可训练的嵌入层
# embedding_layer = nn.Embedding(num_embeddings=10000, embedding_dim=128)  # 10类别，每个类别嵌入为4维向量
#
# # 获取嵌入向量
# input_indices = torch.LongTensor([9999, 3, 5, 9])  # 输入索引
# embeddings = embedding_layer(input_indices)
#
#
# # 创建两个大小为128x1x1280的张量
# tensor1 = torch.randn(128,  1280)
# tensor2 = torch.randn(128,  1280)
#
# # 使用torch.cat将它们拼接在一起
# combined_tensor = torch.cat((tensor1.unsqueeze(1), tensor2.unsqueeze(1)), dim=1)
#
# # 打印拼接后的张量的形状
# print(combined_tensor.shape)


# def write_integers_to_file(n, filename):
#     with open(filename, 'w') as file:
#         for i in range(n + 1):
#             file.write(str(i) + '\n')
#
# write_integers_to_file(65482,r"D:\处理的数据集\vo4.txt")
# 默认情况下，嵌入向量是可训练的，可以在模型训练中更新

# l1 = [1,2,3]
# l2 = ["wo"]
# l3 = ["ni"]
#
# l4 = l1 + l2 + l3
# print(l4)

# import torch
#
# # 假设有两个向量 tensor1 和 tensor2，形状为 (128, 10, 200)
# tensor1 = torch.randn(2, 12)
# tensor2 = torch.randn(2, 12)
#
# # 将每个向量沿着第二个维度拼接
# concatenated_tensors = torch.cat((tensor1.unsqueeze(2), tensor2.unsqueeze(2)), dim=2)
#
# # 交替拼接
# result_tensor = concatenated_tensors.view(2, 6, 4)
#
# # 输出结果的大小为 (128, 20, 200)
# print(result_tensor.size())
# 创建一个形状为128x1x50的示例子张量
sub_tensor = torch.rand((128, 1, 50))

# 初始化一个形状为128x10x50的全零张量
new_tensor = torch.zeros((128, 10, 50))

# 遍历第二个维度，将每个子张量拼接到新张量中
for i in range(new_tensor.size(1)):
    new_tensor[:, i:i+1, :] = sub_tensor

# 打印结果
print("新张量大小:", new_tensor.size())