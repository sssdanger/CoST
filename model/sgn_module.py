import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

class GraphAttentionEmbedding(nn.Module):
    def __init__(self, node_features, n_layers, device, n_heads=2, dropout=0.1):
        super(GraphAttentionEmbedding, self).__init__()
        # self.attention_model = MultiHeadAttention(input_dim=768, num_heads=n_heads)
        self.attention_models = torch.nn.ModuleList([MultiHeadAttention(input_dim=768, num_heads=n_heads) for _ in range(2)])

    def compute_embedding(self, q, k):

        q = q.unsqueeze(0).unsqueeze(0)
        # print(q.shape)
        sub_graph_embedding = self.attention_models[0](q, k, k)
        # print(sub_graph_embedding.shape)
        # sub_graph_embedding = self.attention_models[1](sub_graph_embedding, k, k)
        sub_graph_embedding = sub_graph_embedding.squeeze(0).squeeze(0)
        return sub_graph_embedding

    pass


def embedding_module(node_features, n_layers, device, n_heads=2, dropout=0.1):
    return GraphAttentionEmbedding(node_features=node_features, n_layers=n_layers, device=device, n_heads=n_heads,
                                   dropout=dropout)
class lstmEmbedding(nn.Module):
    def __init__(self, device):
        super(lstmEmbedding, self).__init__()
        # self.attention_model = MultiHeadAttention(input_dim=768, num_heads=n_heads)
        self.lstm = torch.nn.LSTM(768, 768)

    def compute_embedding(self, i, h, c):
        i = i.unsqueeze(0).float()
        h = h.unsqueeze(0).float()
        c = c.unsqueeze(0).float()
        o, (nh, nc) = self.lstm(i, (h, c))
        o = o.squeeze(0)
        nh = nh.squeeze(0)
        nc = nc.squeeze(0)
        return o, nh, nc

    pass
def lstm_module(device):
    return lstmEmbedding(device=device)

# class MultiHeadAttention(nn.Module):
#     def __init__(self, input_dim, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = input_dim // num_heads
#
#         self.query_linear = nn.Linear(input_dim, input_dim)
#         self.key_linear = nn.Linear(input_dim, input_dim)
#         self.value_linear = nn.Linear(input_dim, input_dim)
#         self.output_linear = nn.Linear(input_dim, input_dim)
#
#     def forward(self, query, key, value, mask=None):
#         batch_size = query.size(0)
#
#         # 线性变换得到 Q、K、V
#         query = self.query_linear(query)
#         key = self.key_linear(key)
#         value = self.value_linear(value)
#
#         # 分割输入的 Q、K、V，并计算注意力分数
#         query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#
#         scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
#             torch.tensor(self.head_dim, dtype=torch.float32))
#
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float("-inf"))
#
#         attention_weights = torch.softmax(scores, dim=-1)
#
#         # 注意力加权和
#         attention_output = torch.matmul(attention_weights, value)
#
#         # 将多头注意力的输出进行拼接和线性变换
#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1,
#                                                                               self.num_heads * self.head_dim)
#         output = self.output_linear(attention_output)
#
#         return output


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        self.output_linear = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换得到 Q、K、V
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # 分割输入的 Q、K、V，并计算注意力分数
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        # 注意力加权和
        attention_output = torch.matmul(self.dropout(attention_weights), value)

        # 将多头注意力的输出进行拼接和线性变换
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                              self.num_heads * self.head_dim)
        output = self.output_linear(attention_output)

        return output
