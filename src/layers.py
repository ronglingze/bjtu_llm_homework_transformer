import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .modules import PositionwiseFeedForward, LayerNormalization


class EncoderLayer(nn.Module):
    """
    Transformer编码器层实现
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型的特征维度
            n_heads: 注意力头的数量
            d_ff: 前馈网络的隐藏层维度
            dropout: dropout概率
        """
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 自注意力掩码 [batch_size, seq_len, seq_len]

        Returns:
            output: 编码器层输出 [batch_size, seq_len, d_model]
        """
        norm_x = self.norm1(x)
        attention_output, _ = self.self_attention(query=norm_x, key=norm_x, value=norm_x, mask=mask)
        x = x + self.dropout1(attention_output)

        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout2(ff_output)

        return x


class DecoderLayer(nn.Module):
    """
    Transformer解码器层实现
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型的特征维度
            n_heads: 注意力头的数量
            d_ff: 前馈网络的隐藏层维度
            dropout: dropout概率
        """
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, tgt_seq_len, d_model]
            enc_output: 编码器输出 [batch_size, src_seq_len, d_model]
            self_mask: 自注意力掩码 [batch_size, tgt_seq_len, tgt_seq_len]
            cross_mask: 交叉注意力掩码 [batch_size, 1, src_seq_len]

        Returns:
            output: 解码器层输出 [batch_size, tgt_seq_len, d_model]
            self_attention_weights: 自注意力权重 [batch_size, n_heads, tgt_seq_len, tgt_seq_len]
            cross_attention_weights: 交叉注意力权重 [batch_size, n_heads, tgt_seq_len, src_seq_len]
        """
        norm_x = self.norm1(x)
        self_attention_output, self_attention_weights = self.self_attention(query=norm_x, key=norm_x, value=norm_x, mask=self_mask)
        x = x + self.dropout1(self_attention_output)

        norm_x = self.norm2(x)
        cross_attention_output, cross_attention_weights = self.cross_attention(query=norm_x, key=enc_output, value=enc_output, mask=cross_mask)
        x = x + self.dropout2(cross_attention_output)

        norm_x = self.norm3(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout3(ff_output)

        return x, self_attention_weights, cross_attention_weights


class Encoder(nn.Module):
    """
    Transformer编码器实现
    """

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len=5000, dropout=0.1, padding_idx=0):
        """
        Args:
            vocab_size: 源语言词汇表大小
            d_model: 模型的特征维度
            n_layers: 编码器层数
            n_heads: 注意力头的数量
            d_ff: 前馈网络的隐藏层维度
            max_len: 最大序列长度
            dropout: dropout概率
            padding_idx: padding token的索引
        """
        super(Encoder, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        from .modules import PositionalEncoding
        self.position_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask=None):
        """
        前向传播

        Args:
            x: 输入token索引 [batch_size, src_seq_len]
            mask: 填充掩码 [batch_size, 1, src_seq_len]

        Returns:
            output: 编码器输出 [batch_size, src_seq_len, d_model]
        """
        x = self.token_embedding(x) * (self.token_embedding.embedding_dim**0.5)

        x = self.position_encoding(x)

        attention_mask = mask

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)

        return x


class Decoder(nn.Module):
    """
    Transformer解码器实现
    """

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len=5000, dropout=0.1, padding_idx=0):
        """
        Args:
            vocab_size: 目标语言词汇表大小
            d_model: 模型的特征维度
            n_layers: 解码器层数
            n_heads: 注意力头的数量
            d_ff: 前馈网络的隐藏层维度
            max_len: 最大序列长度
            dropout: dropout概率
            padding_idx: padding token的索引
        """
        super(Decoder, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        from .modules import PositionalEncoding
        self.position_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

        self.norm = LayerNormalization(d_model)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        前向传播

        Args:
            x: 输入token索引 [batch_size, tgt_seq_len]
            enc_output: 编码器输出 [batch_size, src_seq_len, d_model]
            tgt_mask: 目标序列掩码 [batch_size, 1, tgt_seq_len]
            src_mask: 源序列掩码 [batch_size, 1, src_seq_len]

        Returns:
            output: 解码器输出 [batch_size, tgt_seq_len, d_model]
            self_attention_weights: 自注意力权重列表
            cross_attention_weights: 交叉注意力权重列表
        """
        x = self.token_embedding(x) * (self.token_embedding.embedding_dim**0.5)

        x = self.position_encoding(x)

        if tgt_mask is not None:
            seq_len = x.size(1)
            subsequent_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            subsequent_mask = subsequent_mask.unsqueeze(0).to(x.device)
            tgt_attention_mask = tgt_mask.expand(-1, seq_len, -1)
            self_attention_mask = tgt_attention_mask & ~subsequent_mask
        else:
            self_attention_mask = None

        cross_attention_mask = src_mask

        self_attention_weights = []
        cross_attention_weights = []

        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(x, enc_output, self_attention_mask, cross_attention_mask)
            self_attention_weights.append(self_attn_weights)
            cross_attention_weights.append(cross_attn_weights)

        x = self.norm(x)

        return x, self_attention_weights, cross_attention_weights