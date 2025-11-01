"""
Transformer实现模块

这个模块包含了完整的Transformer模型实现，包括：
- 多头注意力机制
- 位置编码
- 前馈神经网络
- 编码器和解码器层
- 完整的Transformer模型
- 工具函数和优化器
"""

from .attention import MultiHeadAttention, scaled_dot_product_attention
from .modules import PositionwiseFeedForward, PositionalEncoding, Embeddings, ProjectionLayer, LayerNormalization
from .layers import EncoderLayer, DecoderLayer, Encoder, Decoder
from .model import Transformer, TransformerConfig, count_parameters, initialize_model
from .utils import (
    create_padding_mask, create_subsequent_mask, create_combined_mask,
    LabelSmoothingLoss, NoamOpt, get_optimizer, get_lr_scheduler,
    greedy_decode, beam_search_decode, calculate_bleu_score,
    save_checkpoint, load_checkpoint
)

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    # 注意力机制
    'MultiHeadAttention',
    'scaled_dot_product_attention',

    # 基础模块
    'PositionwiseFeedForward',
    'PositionalEncoding',
    'Embeddings',
    'ProjectionLayer',
    'LayerNormalization',

    # 层实现
    'EncoderLayer',
    'DecoderLayer',
    'Encoder',
    'Decoder',

    # 模型实现
    'Transformer',
    'TransformerConfig',
    'count_parameters',
    'initialize_model',

    # 工具函数
    'create_padding_mask',
    'create_subsequent_mask',
    'create_combined_mask',
    'LabelSmoothingLoss',
    'NoamOpt',
    'get_optimizer',
    'get_lr_scheduler',
    'greedy_decode',
    'beam_search_decode',
    'calculate_bleu_score',
    'save_checkpoint',
    'load_checkpoint',
]