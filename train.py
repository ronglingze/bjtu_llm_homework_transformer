#!/usr/bin/env python3
"""
Transformer模型训练脚本
"""

import os
import sys
import argparse
import yaml
import time
import logging
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.model import Transformer, TransformerConfig, count_parameters
from src.utils import LabelSmoothingLoss, NoamOpt, save_checkpoint, get_optimizer, load_checkpoint, set_seed
from src.data import get_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('training.log'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)




def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch, gradient_clip=1.0, scaler=None, log_interval=100):
    """
    训练一个epoch (
    """
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                output, _, _ = model(src, tgt_input)
                loss = criterion(output, tgt_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
        else:
            output, _, _ = model(src, tgt_input)
            loss = criterion(output, tgt_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

            optimizer.step()
            scheduler.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device):
    """
    验证模型 (
    """
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    with torch.no_grad():
        for batch in val_loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            output, _, _ = model(src, tgt_input)
            loss = criterion(output, tgt_target)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def plot_curves(train_losses, val_losses, output_dir):
    """
    绘制训练和验证损失曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        output_dir: 输出目录
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(plot_path)
    logger.info(f"Loss curves saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Transformer模型训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据和分词器缓存目录')  #
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cpu/cuda/auto)')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--mixed_precision', action='store_true', help='使用混合精度训练')
    parser.add_argument('--log_interval', type=int, default=100, help='日志间隔')
    parser.add_argument('--save_interval', type=int, default=1, help='保存间隔')
    parser.add_argument('--dummy_data', action='store_true', help='(已弃用) 使用虚拟数据')  #
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于确保实验可重现性')

    args = parser.parse_args()

    # 设置随机种子以确保实验可重现性
    set_seed(args.seed)
    logger.info(f"随机种子已设置为: {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f'使用设备: {device}')

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
    })
    logger.info('配置信息:')
    for key, value in config.items():
        logger.info(f'  {key}: {value}')

    logger.info(f"正在从 {args.data_dir} 加载/创建数据集...")
    train_loader, val_loader, src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx = get_dataloaders(config=config, data_dir=args.data_dir)
    logger.info("数据加载完成。")

    model_config = TransformerConfig()
    for key, value in config.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)

    model_config.src_padding_idx = src_pad_idx
    model_config.tgt_padding_idx = tgt_pad_idx

    model = model_config.create_model(src_vocab_size, tgt_vocab_size)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info(f'模型参数总数: {total_params:,}')
    logger.info(f'可训练参数: {trainable_params:,}')

    criterion = LabelSmoothingLoss(vocab_size=tgt_vocab_size, padding_idx=tgt_pad_idx, smoothing=config.get('label_smoothing', 0.1)).to(device)

    optimizer = get_optimizer(model, learning_rate=args.learning_rate, weight_decay=config.get('weight_decay', 0.01), optimizer_type='adamw')

    scheduler = NoamOpt(model_size=model_config.d_model, factor=1.0, warmup=config.get('warmup_steps', 4000), optimizer=optimizer)

    scaler = GradScaler() if args.mixed_precision else None

    start_epoch = 0
    best_val_loss = float('inf')

    train_loss_history = []
    val_loss_history = []

    if args.resume:
        logger.info(f'从检查点恢复训练: {args.resume}')
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1
        logger.info(f'恢复到epoch {start_epoch}, 损失: {best_val_loss:.4f}')

    logger.info('开始训练...')
    training_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch, gradient_clip=config.get('gradient_clip', 1.0), scaler=scaler, log_interval=args.log_interval)

        val_loss = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start_time

        # 记录损失历史
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        logger.info(f'Epoch {epoch} 完成:')
        logger.info(f'  训练损失: {train_loss:.4f}')
        logger.info(f'  验证损失: {val_loss:.4f}')
        logger.info(f'  学习率: {scheduler.rate():.6f}')
        logger.info(f'  耗时: {epoch_time:.2f}秒')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            logger.info(f'保存最佳模型: {best_model_path}')

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            logger.info(f'保存检查点: {checkpoint_path}')

    total_training_time = time.time() - training_start_time
    logger.info(f'训练完成! 总耗时: {total_training_time:.2f}秒')

    plot_curves(train_loss_history, val_loss_history, args.output_dir)


if __name__ == '__main__':
    main()
