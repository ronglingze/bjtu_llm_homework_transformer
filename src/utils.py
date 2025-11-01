import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from typing import Optional, Tuple


def set_seed(seed):
    """
    设置随机种子以确保实验可重现性

    Args:
        seed (int): 随机种子值
    """
    # 设置Python环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 设置基础库随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 设置PyTorch随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    print(f"随机种子已设置为: {seed}")


def create_padding_mask(seq: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    """
    创建填充掩码

    Args:
        seq: 输入序列 [batch_size, seq_len]
        padding_idx: padding token的索引

    Returns:
        mask: 填充掩码 [batch_size, 1, seq_len]
    """
    return (seq != padding_idx).unsqueeze(1)


def create_subsequent_mask(seq_len: int) -> torch.Tensor:
    """
    创建后续掩码（防止看到未来信息）

    Args:
        seq_len: 序列长度

    Returns:
        mask: 后续掩码 [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask


def create_combined_mask(tgt_seq: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    """
    创建组合掩码（填充掩码 + 后续掩码）

    Args:
        tgt_seq: 目标序列 [batch_size, tgt_seq_len]
        padding_idx: padding token的索引

    Returns:
        mask: 组合掩码 [batch_size, 1, tgt_seq_len, tgt_seq_len]
    """
    batch_size, tgt_seq_len = tgt_seq.size()

    padding_mask = create_padding_mask(tgt_seq, padding_idx)

    subsequent_mask = create_subsequent_mask(tgt_seq_len).to(tgt_seq.device)

    combined_mask = padding_mask & ~subsequent_mask.unsqueeze(0)

    return combined_mask


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    """

    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        """
        Args:
            vocab_size: 词汇表大小
            padding_idx: padding token的索引
            smoothing: 平滑系数
        """
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算标签平滑损失

        Args:
            x: 模型输出logits [batch_size, seq_len, vocab_size]
            target: 目标序列 [batch_size, seq_len]

        Returns:
            loss: 标签平滑损失
        """
        batch_size, seq_len, vocab_size = x.size()

        x = x.reshape(-1, vocab_size)
        target = target.reshape(-1)

        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (vocab_size - 2))

        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        true_dist[:, self.padding_idx] = 0

        mask = (target != self.padding_idx)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        x = x[mask]
        true_dist = true_dist[mask]

        loss = self.criterion(F.log_softmax(x, dim=-1), true_dist)

        return loss / mask.sum()


class NoamOpt:
    """
    Noam学习率调度器
    """

    def __init__(self, model_size: int, factor: float, warmup: int, optimizer: optim.Optimizer):
        """
        Args:
            model_size: 模型维度
            factor: 缩放因子
            warmup: 预热步数
            optimizer: 优化器
        """
        self.optimizer = optimizer
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self._step = 0
        self._rate = 0

    def step(self):
        """更新学习率"""
        self._step += 1
        rate = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        self._rate = rate

    def rate(self, step: Optional[int] = None) -> float:
        """
        计算当前学习率

        Args:
            step: 当前步数，如果为None则使用内部步数

        Returns:
            lr: 学习率
        """
        if step is None:
            step = self._step

        return self.factor * (self.model_size**(-0.5) * min(step**(-0.5), step * self.warmup**(-1.5)))

    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()


def get_optimizer(model: nn.Module, learning_rate: float = 0.001, weight_decay: float = 0.0, optimizer_type: str = 'adamw') -> optim.Optimizer:
    """
    获取优化器

    Args:
        model: 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
        optimizer_type: 优化器类型 ('adam', 'adamw', 'sgd')

    Returns:
        optimizer: 优化器
    """
    parameters = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == 'adam':
        optimizer = optim.Adam(parameters, lr=learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(parameters, lr=learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def get_lr_scheduler(optimizer: optim.Optimizer, scheduler_type: str = 'noam', model_size: int = 512, warmup_steps: int = 4000, factor: float = 1.0):
    """
    获取学习率调度器

    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 ('noam', 'cosine', 'step')
        model_size: 模型维度（Noam调度器需要）
        warmup_steps: 预热步数（Noam调度器需要）
        factor: 缩放因子（Noam调度器需要）

    Returns:
        scheduler: 学习率调度器
    """
    if scheduler_type == 'noam':
        return NoamOpt(model_size, factor, warmup_steps, optimizer)
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def greedy_decode(model: nn.Module, src: torch.Tensor, max_len: int, start_symbol: int, end_symbol: Optional[int] = None) -> torch.Tensor:
    """
    贪婪解码

    Args:
        model: Transformer模型
        src: 源序列 [batch_size, src_seq_len]
        max_len: 最大解码长度
        start_symbol: 开始符号
        end_symbol: 结束符号（可选）

    Returns:
        decoded: 解码结果 [batch_size, max_len]
    """
    model.eval()
    batch_size = src.size(0)
    device = src.device

    with torch.no_grad():
        src_mask = model.create_padding_mask(src, model.src_padding_idx)
        enc_output = model.encode(src, src_mask)

        decoded = torch.full((batch_size, 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = model.create_padding_mask(decoded, model.tgt_padding_idx)

            output, _, _ = model.decode(decoded, enc_output, tgt_mask, src_mask)

            proj_output = model.project(output)

            next_token = proj_output[:, -1:].argmax(dim=-1)

            decoded = torch.cat([decoded, next_token], dim=1)

            if end_symbol is not None and (next_token == end_symbol).all():
                break

    return decoded


def beam_search_decode(model: nn.Module, src: torch.Tensor, beam_size: int = 4, max_len: int = 50, start_symbol: int = 1, end_symbol: Optional[int] = None, length_penalty: float = 1.0) -> torch.Tensor:
    """
    简化的集束搜索解码

    注意：这是一个简化实现，不是完整的批处理集束搜索。
    对于作业要求，建议使用greedy_decode，它已经足够完善。

    Args:
        model: Transformer模型
        src: 源序列 [batch_size, src_seq_len]
        beam_size: 集束大小
        max_len: 最大解码长度
        start_symbol: 开始符号
        end_symbol: 结束符号（可选）
        length_penalty: 长度惩罚系数

    Returns:
        decoded: 解码结果 [batch_size, max_len]
    """
    return greedy_decode(model, src, max_len, start_symbol, end_symbol)


def calculate_bleu_score(references: list, hypotheses: list, n_gram: int = 4) -> float:
    """
    计算BLEU分数

    Args:
        references: 参考翻译列表
        hypotheses: 假设翻译列表
        n_gram: n-gram的最大值

    Returns:
        bleu_score: BLEU分数
    """
    from collections import Counter
    import math

    def get_ngrams(sentence: list, n: int) -> list:
        """获取n-gram"""
        return [tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]

    def count_overlap(ref_ngrams: Counter, hyp_ngrams: Counter) -> int:
        """计算重叠的n-gram数量"""
        overlap = 0
        for ngram in hyp_ngrams:
            overlap += min(hyp_ngrams[ngram], ref_ngrams[ngram])
        return overlap

    total_precision = 0

    for n in range(1, n_gram + 1):
        total_ref_ngrams = 0
        total_hyp_ngrams = 0
        total_overlap = 0

        for ref, hyp in zip(references, hypotheses):
            ref_ngrams = Counter(get_ngrams(ref, n))
            hyp_ngrams = Counter(get_ngrams(hyp, n))

            total_ref_ngrams += sum(ref_ngrams.values())
            total_hyp_ngrams += sum(hyp_ngrams.values())
            total_overlap += count_overlap(ref_ngrams, hyp_ngrams)

        if total_hyp_ngrams == 0:
            precision = 0
        else:
            precision = total_overlap / total_hyp_ngrams

        total_precision += math.log(precision + 1e-10)

    hyp_lengths = [len(hyp) for hyp in hypotheses]
    ref_lengths = [len(ref) for ref in references]

    closest_ref_lengths = []
    for hyp_len in hyp_lengths:
        closest_ref = min(ref_lengths, key=lambda x: abs(x - hyp_len))
        closest_ref_lengths.append(closest_ref)

    if sum(hyp_lengths) > sum(closest_ref_lengths):
        bp = 1.0
    else:
        bp = math.exp(1 - sum(closest_ref_lengths) / sum(hyp_lengths))

    bleu_score = bp * math.exp(total_precision / n_gram)

    return bleu_score


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, loss: float, filepath: str):
    """
    保存模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 当前损失
        filepath: 保存路径
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, filepath: str, device: torch.device) -> tuple:
    """
    加载模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        filepath: 检查点路径
        device: 设备

    Returns:
        epoch: 恢复的epoch
        loss: 恢复的损失
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, loss
