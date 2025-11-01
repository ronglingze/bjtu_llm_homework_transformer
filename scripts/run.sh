#!/bin/bash

# Transformer机器翻译训练脚本

# 设置CUDA环境变量
export CUDA_VISIBLE_DEVICES=0

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建必要的目录
mkdir -p results
mkdir -p logs

# 获取参数
CONFIG_FILE=${1:-"configs/base.yaml"}
OUTPUT_DIR=${2:-"results"}
EPOCHS=${3:-10}
BATCH_SIZE=${4:-32}
SEED=${5:-42}

echo "********************************************"
echo "开始训练Transformer模型..."
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "训练轮数: $EPOCHS"
echo "批大小: $BATCH_SIZE"
echo "随机种子: $SEED"
echo "********************************************"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在!"
    exit 1
fi

# 检查Python环境
python -c "import torch" || {
    echo "错误: PyTorch未安装或环境有问题: pip install -r requirements.txt"
    exit 1
}

# 开始训练
python train.py \
    --config "$CONFIG_FILE" \
    --data_dir ./data \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --device cuda \
    --mixed_precision \
    --log_interval 50 \
    --save_interval 2

echo "训练完成! 结果保存在 $OUTPUT_DIR 目录中。"