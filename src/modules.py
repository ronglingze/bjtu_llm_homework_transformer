import torch
import torch.nn as nn
import math


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈神经网络实现
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型的特征维度
            d_ff: 前馈网络的隐藏层维度
            dropout: dropout概率
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            output: 输出张量 [batch_size, seq_len, d_model]
        """
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.w_2(x)

        return x


class PositionalEncoding(nn.Module):
    """
    位置编码实现
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: 模型的特征维度
            max_len: 最大序列长度
            dropout: dropout概率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            output: 添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Embeddings(nn.Module):
    """
    词嵌入层实现
    """

    def __init__(self, vocab_size, d_model, padding_idx=0):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 嵌入维度
            padding_idx: padding token的索引
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入token索引 [batch_size, seq_len]

        Returns:
            output: 嵌入向量 [batch_size, seq_len, d_model]
        """
        return self.lut(x) * math.sqrt(self.d_model)


class ProjectionLayer(nn.Module):
    """
    投影层，用于将模型输出投影回词汇表空间
    """

    def __init__(self, d_model, vocab_size):
        """
        Args:
            d_model: 模型的特征维度
            vocab_size: 词汇表大小
        """
        super(ProjectionLayer, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            output: 投影后的logits [batch_size, seq_len, vocab_size]
        """
        return self.proj(x)


class LayerNormalization(nn.Module):
    """
    层归一化实现
    """

    def __init__(self, d_model, eps=1e-6):
        """
        Args:
            d_model: 模型特征维度
            eps: 数值稳定性的小常数
        """
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            output: 归一化后的张量 [batch_size, seq_len, d_model]
        """
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, unbiased=False, keepdim=True)

        normalized = (x - mean) / torch.sqrt(variance + self.eps)

        output = normalized * self.gamma + self.beta

        return output
