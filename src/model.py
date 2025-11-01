import torch.nn as nn
from .layers import Encoder, Decoder
from .modules import ProjectionLayer


class Transformer(nn.Module):
    """
    完整的Transformer模型实现
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_len=5000, dropout=0.1, src_padding_idx=0, tgt_padding_idx=0, share_embedding=False, tie_projection_weights=False):
        """
        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型的特征维度
            n_layers: 编码器/解码器层数
            n_heads: 注意力头的数量
            d_ff: 前馈网络的隐藏层维度
            max_len: 最大序列长度
            dropout: dropout概率
            src_padding_idx: 源语言padding token的索引
            tgt_padding_idx: 目标语言padding token的索引
            share_embedding: 是否共享源语言和目标语言的嵌入层
            tie_projection_weights: 是否共享解码器嵌入层和输出投影层权重
        """
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.encoder = Encoder(vocab_size=src_vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=dropout, padding_idx=src_padding_idx)

        self.decoder = Decoder(vocab_size=tgt_vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=dropout, padding_idx=tgt_padding_idx)

        self.projection = ProjectionLayer(d_model, tgt_vocab_size)

        if share_embedding and src_vocab_size == tgt_vocab_size:
            self.decoder.token_embedding.weight = self.encoder.token_embedding.weight

        if tie_projection_weights:
            self.projection.proj.weight = self.decoder.token_embedding.weight

        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx

    def create_padding_mask(self, x, padding_idx):
        """
        创建填充掩码

        Args:
            x: 输入张量 [batch_size, seq_len]
            padding_idx: padding token的索引

        Returns:
            mask: 填充掩码 [batch_size, 1, seq_len]
        """
        return (x != padding_idx).unsqueeze(1)

    def encode(self, src, src_mask=None):
        """
        编码器前向传播

        Args:
            src: 源序列 [batch_size, src_seq_len]
            src_mask: 源序列掩码 [batch_size, 1, src_seq_len]

        Returns:
            enc_output: 编码器输出 [batch_size, src_seq_len, d_model]
        """
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        """
        解码器前向传播

        Args:
            tgt: 目标序列 [batch_size, tgt_seq_len]
            enc_output: 编码器输出 [batch_size, src_seq_len, d_model]
            tgt_mask: 目标序列掩码 [batch_size, 1, tgt_seq_len]
            src_mask: 源序列掩码 [batch_size, 1, src_seq_len]

        Returns:
            dec_output: 解码器输出 [batch_size, tgt_seq_len, d_model]
            self_attention_weights: 自注意力权重列表
            cross_attention_weights: 交叉注意力权重列表
        """
        return self.decoder(tgt, enc_output, tgt_mask, src_mask)

    def project(self, x):
        """
        投影到词汇表空间

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            logits: 输出logits [batch_size, seq_len, vocab_size]
        """
        return self.projection(x)

    def forward(self, src, tgt):
        """
        完整的前向传播

        Args:
            src: 源序列 [batch_size, src_seq_len]
            tgt: 目标序列 [batch_size, tgt_seq_len]

        Returns:
            output: 模型输出logits [batch_size, tgt_seq_len, tgt_vocab_size]
            self_attention_weights: 自注意力权重列表
            cross_attention_weights: 交叉注意力权重列表
        """
        src_mask = self.create_padding_mask(src, self.src_padding_idx)
        tgt_mask = self.create_padding_mask(tgt, self.tgt_padding_idx)

        enc_output = self.encode(src, src_mask)

        dec_output, self_attention_weights, cross_attention_weights = self.decode(tgt, enc_output, tgt_mask, src_mask)

        output = self.project(dec_output)

        return output, self_attention_weights, cross_attention_weights


class TransformerConfig:
    """
    Transformer模型配置类
    """

    def __init__(self):
        self.d_model = 512
        self.n_layers = 6
        self.n_heads = 8
        self.d_ff = 2048
        self.dropout = 0.1

        self.max_len = 5000
        self.src_padding_idx = 0
        self.tgt_padding_idx = 0

        self.label_smoothing = 0.1
        self.warmup_steps = 4000
        self.share_embedding = False
        self.tie_projection_weights = False

    def create_model(self, src_vocab_size, tgt_vocab_size):
        """
        根据配置创建模型

        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小

        Returns:
            model: Transformer模型实例
        """
        return Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, d_model=self.d_model, n_layers=self.n_layers, n_heads=self.n_heads, d_ff=self.d_ff, max_len=self.max_len, dropout=self.dropout,
                           src_padding_idx=self.src_padding_idx, tgt_padding_idx=self.tgt_padding_idx, share_embedding=self.share_embedding, tie_projection_weights=self.tie_projection_weights)

    @classmethod
    def base(cls):
        """基础配置"""
        config = cls()
        return config

    @classmethod
    def small(cls):
        """小模型配置"""
        config = cls()
        config.d_model = 256
        config.n_layers = 3
        config.n_heads = 4
        config.d_ff = 1024
        return config

    @classmethod
    def large(cls):
        """大模型配置"""
        config = cls()
        config.d_model = 1024
        config.n_layers = 12
        config.n_heads = 16
        config.d_ff = 4096
        return config


def count_parameters(model):
    """
    计算模型参数数量

    Args:
        model: PyTorch模型

    Returns:
        total_params: 总参数数量
        trainable_params: 可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def initialize_model(model, init_type='normal', init_gain=0.02):
    """
    初始化模型权重

    Args:
        model: PyTorch模型
        init_type: 初始化类型 ('normal', 'xavier', 'kaiming')
        init_gain: 初始化增益
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(init_func)
    return model