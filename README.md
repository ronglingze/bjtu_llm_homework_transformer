# Transformer 机器翻译项目

基于Transformer架构的英德机器翻译实现，使用IWSLT2017数据集进行训练。

## 项目结构

```
llm_homework/
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── model.py           # Transformer模型定义
│   ├── attention.py       # 注意力机制实现
│   ├── layers.py          # Transformer层组件
│   ├── modules.py         # 基础模块
│   ├── data.py            # 数据加载和预处理
│   └── utils.py           # 工具函数
├── configs/               # 配置文件目录
│   └── base.yaml          # 基础配置文件
├── scripts/               # 脚本目录
│   └── run.sh             # 训练脚本
├── data/                  # 数据目录（自动下载）
├── results/               # 训练结果目录
├── train.py               # 主训练脚本
├── translate.py           # 翻译脚本
├── requirements.txt       # 依赖包列表
└── README.md             # 项目说明文档
```

## 环境要求

- Python 3.8+
- PyTorch 1.12+

### 硬件要求

**最低配置：**
- CPU: 4核心以上
- 内存: 8GB RAM
- 存储: 10GB 可用空间（用于数据集和模型）

**推荐配置：**
- GPU: NVIDIA GPU with 8GB+ VRAM (支持CUDA 11.0+)
- CPU: 8核心以上
- 内存: 16GB+ RAM
- 存储: 20GB+ 可用空间

**性能说明：**
- CPU训练: 单epoch约需15-30分钟
- GPU训练: 单epoch约需2-5分钟
- 完整训练（10 epochs）: CPU约需3-5小时，GPU约需20-50分钟

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

### 1. 准备配置文件

项目提供了基础的配置文件 `configs/base.yaml`，包含以下默认配置：

```yaml
# 模型架构配置
d_model: 512                    # 模型维度
n_layers: 6                     # 编码器和解码器层数
n_heads: 8                      # 多头注意力头数
d_ff: 2048                      # 前馈网络维度
dropout: 0.1                    # Dropout率

# 词汇表和序列配置
vocab_size: 10000               # 词汇表大小
max_len: 5000                   # 最大序列长度
src_padding_idx: 1              # 源语言padding索引
tgt_padding_idx: 1              # 目标语言padding索引

# 训练优化配置
weight_decay: 0.01              # 权重衰减
label_smoothing: 0.1            # 标签平滑
warmup_steps: 4000              # 预热步数
gradient_clip: 1.0              # 梯度裁剪

# 权重共享配置
share_embedding: false           # 是否共享源目标和目标嵌入
tie_projection_weights: true    # 是否绑定投影权重
```

你也可以创建自定义配置文件来调整模型参数。

### 2. 使用训练脚本（推荐）

项目提供了便捷的训练脚本 `scripts/run.sh`，使用方法如下：

```bash
# 使用默认配置训练
bash scripts/run.sh

# 指定配置文件、输出目录、训练轮数、批大小和随机种子
bash scripts/run.sh configs/base.yaml results 10 32 42
```

参数说明：
- 第1个参数: 配置文件路径（默认: configs/base.yaml）
- 第2个参数: 输出目录（默认: results）
- 第3个参数: 训练轮数（默认: 10）
- 第4个参数: 批大小（默认: 32）
- 第5个参数: 随机种子（默认: 42）

脚本功能：
- 自动设置CUDA环境变量和Python路径
- 检查配置文件和依赖环境
- 创建必要的目录结构
- 启用混合精度训练
- 设置详细的日志间隔和保存间隔
- 支持随机种子设置确保可重现性

### 3. 直接使用Python训练

#### 基础训练命令

```bash
python train.py --config configs/base.yaml --data_dir ./data --output_dir ./results
```

#### GPU训练命令（推荐）

```bash
python train.py --config configs/base.yaml --data_dir ./data --output_dir ./results --device cuda --batch_size 64
```

#### 混合精度训练命令（适用于大模型）

```bash
python train.py --config configs/base.yaml --data_dir ./data --output_dir ./results --device cuda --mixed_precision --batch_size 128
```

#### 使用虚拟数据快速测试

```bash
python train.py --config configs/base.yaml --data_dir ./data --output_dir ./results --dummy_data --epochs 1
```

#### 完整重现实验命令

为确保实验可重现性，请使用以下确切命令：

```bash
python train.py \
    --config configs/base.yaml \
    --data_dir ./data \
    --output_dir ./results \
    --epochs 10 \
    --batch_size 32 \
    --seed 42
```

该命令会自动设置所有必要的随机种子（Python、NumPy、PyTorch）确保实验结果可重现。

### 4. 模型翻译

训练完成后，您可以使用 `translate.py` 脚本进行英德翻译：

#### 基础翻译命令

```bash
python translate.py --text "Hello world"
```

#### 完整翻译命令

```bash
python translate.py \
    --text "How are you today?" \
    --config configs/base.yaml \
    --data_dir ./data \
    --model_path ./results/best_model.pth \
    --device cuda \
    --seed 42
```

#### 翻译示例

```bash
# 翻译单个句子
python translate.py --text "Hello world"
python translate.py --text "I love machine learning"
python translate.py --text "The weather is nice today"

# 指定设备进行翻译
python translate.py --text "Hello world" --device cpu
python translate.py --text "Hello world" --device cuda
```

#### 主要参数说明：
- `--text`: 要翻译的英文文本（必需）
- `--config`: 配置文件路径（默认：configs/base.yaml）
- `--data_dir`: 数据目录，包含分词器文件（默认：./data）
- `--model_path`: 训练好的模型路径（默认：./results/best_model.pth）
- `--device`: 计算设备（auto/cuda/cpu，默认：auto）
- `--seed`: 随机种子（默认：42）

#### 前置条件

使用翻译功能前，请确保：
1. 已经完成模型训练，并且 `./results/best_model.pth` 文件存在
2. 分词器文件存在：`./data/tokenizer-en.json` 和 `./data/tokenizer-de.json`
3. 相关依赖包已安装

#### 主要参数说明：
- `--config`: 配置文件路径（必需）
- `--data_dir`: 数据和分词器缓存目录（默认：./data）
- `--output_dir`: 模型输出目录（默认：./results）
- `--epochs`: 训练轮数（默认：10）
- `--batch_size`: 批大小（默认：32，GPU可设置64-128）
- `--learning_rate`: 学习率（默认：0.001）
- `--device`: 设备选择（cpu/cuda/auto，默认：auto）
- `--mixed_precision`: 使用混合精度训练（减少显存占用）
- `--dummy_data`: 使用虚拟数据（用于快速测试）
- `--seed`: 随机种子（确保实验可重现）
- `--log_interval`: 日志输出间隔（默认：100）
- `--save_interval`: 模型保存间隔（默认：5）
- `--resume`: 从检查点恢复训练

## 功能特性

- **完整的Transformer架构**：包含多头注意力、前馈网络、位置编码等
- **自动数据处理**：自动下载和预处理IWSLT2017数据集
- **自定义分词器**：使用WordPiece算法训练源语言和目标语言分词器
- **灵活的配置系统**：通过YAML文件配置模型和训练参数
- **模型翻译功能**：支持使用训练好的模型进行英德翻译
- **高级优化技术**：
  - 标签平滑损失函数
  - 学习率预热和调度器
  - 梯度裁剪
  - 权重衰减
  - 混合精度训练
- **训练脚本**：提供便捷的bash脚本用于快速启动训练
- **虚拟数据模式**：支持使用虚拟数据进行快速测试
- **模型保存和恢复**：支持训练过程中保存检查点并从中恢复
- **可重现训练**：支持随机种子设置确保实验可重现
- **详细的日志记录**：包含训练损失、学习率变化等详细信息

## 模型架构

本项目实现了标准的Transformer架构，包括：

- **编码器**：N层相同的层堆叠，每层包含多头自注意力机制和前馈网络
- **解码器**：N层相同的层堆叠，每层包含多头自注意力、编码器-解码器注意力和前馈网络
- **注意力机制**：缩放点积注意力，支持多头注意力
- **位置编码**：使用正弦和余弦函数的位置编码
- **掩码机制**：支持填充掩码和后续掩码

## 数据集

项目使用IWSLT2017英德翻译数据集：
- 训练集：约200,000句对
- 验证集：约7,000句对
- 测试集：约7,000句对

数据会自动下载到指定的 `data_dir` 目录中。

## 训练日志

训练过程中会生成详细的日志，包括：
- 每个epoch的训练和验证损失
- 学习率变化
- 模型参数统计
- 训练时间统计

日志会同时输出到控制台和保存到 `training.log` 文件中。

## 实验重现

为确保实验结果的可重现性，请按照以下步骤操作：

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv transformer_env
source transformer_env/bin/activate  # Linux/Mac
# transformer_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据集准备

首次运行会自动下载IWSLT2017数据集（约500MB），确保：
- 网络连接稳定
- 磁盘空间充足（至少2GB）

### 3. 完整重现命令

使用以下命令确保完全相同的实验条件：

```bash
# 使用项目提供的配置文件和随机种子
python train.py \
    --config configs/base.yaml \
    --data_dir ./data \
    --output_dir ./results \
    --epochs 10 \
    --batch_size 32 \
    --seed 42
```

或者使用提供的训练脚本：

```bash
bash scripts/run.sh configs/base.yaml results 10 32 42
```

### 4. 预期输出

训练完成后，您应该在 `./results` 目录下看到：
- `best_model.pth`: 最佳模型权重
- `checkpoint_epoch_*.pth`: 各epoch检查点
- `training.log`: 详细训练日志
- `tokenizer-en.json`: 英文分词器
- `tokenizer-de.json`: 德文分词器

### 5. 性能基准

使用默认配置和随机种子42，预期结果：
- 最终验证损失: ~2.8-3.2
- 训练时间: CPU 3-5小时，GPU 20-50分钟
- 显存使用: 约6-8GB（GPU训练）

## 注意事项

1. 首次运行时会自动下载数据集，需要稳定的网络连接
2. 建议使用GPU进行训练以获得更好的性能
3. 可以通过调整配置文件中的参数来优化模型性能
4. 混合精度训练可以减少显存使用并加速训练
5. 使用 `--dummy_data` 参数可以快速测试训练流程，无需下载数据集
6. 训练过程中的详细日志会保存到 `training.log` 文件中
7. 分词器文件会自动保存在 `data/` 目录中，下次运行时直接加载
8. 使用翻译功能前，请确保模型训练已完成且模型文件存在
9. 翻译质量取决于训练数据的质量和训练轮数，建议使用完整的训练周期

## 许可证

本项目仅用于学习和研究目的。