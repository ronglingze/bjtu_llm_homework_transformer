import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    """
    实现缩放点积注意力

    Args:
        Q: Query tensor [batch_size, n_heads, seq_len, d_k]
        K: Key tensor [batch_size, n_heads, seq_len, d_k]
        V: Value tensor [batch_size, n_heads, seq_len, d_v]
        mask: 可选的掩码张量
        dropout: 可选的dropout层

    Returns:
        attention_output: 注意力输出 [batch_size, n_heads, seq_len, d_v]
        attention_weights: 注意力权重 [batch_size, n_heads, seq_len, seq_len]
    """
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    attention_output = torch.matmul(attention_weights, V)

    return attention_output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model: 模型的特征维度
            n_heads: 注意力头的数量
            dropout: dropout概率
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        将输入按头数分割

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            分割后的张量 [batch_size, n_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        将多头输出合并

        Args:
            x: 输入张量 [batch_size, n_heads, seq_len, d_v]

        Returns:
            合并后的张量 [batch_size, seq_len, d_model]
        """
        batch_size, n_heads, seq_len, d_v = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, n_heads * d_v)

    def forward(self, query, key, value, mask=None):
        """
        前向传播

        Args:
            query: Query张量 [batch_size, seq_len, d_model]
            key: Key张量 [batch_size, seq_len, d_model]
            value: Value张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len] 或 [batch_size, 1, seq_len, seq_len]

        Returns:
            output: 注意力输出 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        if mask is not None:
            mask = mask.unsqueeze(1)

        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout)

        output = self.combine_heads(attention_output)

        output = self.W_o(output)

        return output, attention_weights
