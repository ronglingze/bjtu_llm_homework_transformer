import torch
import argparse
import yaml
import os
import sys
from tokenizers import Tokenizer
from src.model import Transformer, TransformerConfig
from src.utils import greedy_decode, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def translate(model, text, tokenizer_src, tokenizer_tgt, config, device):
    """
    使用训练好的模型进行翻译

    Args:
        model: 训练好的Transformer模型
        text: 待翻译的源文本
        tokenizer_src: 源语言分词器
        tokenizer_tgt: 目标语言分词器
        config: 模型配置
        device: 计算设备

    Returns:
        str: 翻译后的文本
    """
    model.eval()

    sos_id = tokenizer_src.token_to_id("[SOS]")
    eos_id = tokenizer_src.token_to_id("[EOS]")

    src_tokens = [sos_id] + tokenizer_src.encode(text).ids + [eos_id]
    src = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)

    start_symbol = tokenizer_tgt.token_to_id("[SOS]")

    decoded_ids = greedy_decode(model, src, max_len=config.max_len, start_symbol=start_symbol, end_symbol=tokenizer_tgt.token_to_id("[EOS]")).squeeze(0)

    output_text = tokenizer_tgt.decode(decoded_ids.tolist(), skip_special_tokens=True)

    return output_text


def main():
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='')
    parser.add_argument('--data_dir', type=str, default='./data', help='')
    parser.add_argument('--model_path', type=str, default='./results/best_model.pth', help='')
    parser.add_argument('--text', type=str, required=True, help='')
    parser.add_argument('--device', type=str, default='auto', help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"随机种子已设置为: {args.seed}")

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"使用设备: {device}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    model_config = TransformerConfig()
    for key, value in config_dict.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)

    print("正在加载分词器...")
    try:
        tokenizer_src = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer-en.json"))
        tokenizer_tgt = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer-de.json"))
        print(f"成功加载分词器: 英文词汇表大小 {tokenizer_src.get_vocab_size()}, 德文词汇表大小 {tokenizer_tgt.get_vocab_size()}")
    except FileNotFoundError:
        print(f"错误: 找不到分词器文件!")
        print(f"请确保目录 {args.data_dir} 中存在 tokenizer-en.json 和 tokenizer-de.json 文件")
        print(f"如果尚未训练分词器，请先运行训练脚本")
        sys.exit(1)

    model_config.src_padding_idx = tokenizer_src.token_to_id("[PAD]")
    model_config.tgt_padding_idx = tokenizer_tgt.token_to_id("[PAD]")

    model = model_config.create_model(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())

    print(f"正在加载模型: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型参数，训练轮数: {checkpoint.get('epoch', '未知')}")
    except FileNotFoundError:
        print(f"错误: 找不到模型文件!")
        print(f"请确保文件 {args.model_path} 存在")
        print(f"如果尚未训练模型，请先运行训练脚本")
        sys.exit(1)

    model = model.to(device)

    print("模型加载完成!")
    print("-" * 50)
    print(f"输入文本 (英语): {args.text}")
    output_text = translate(model, args.text, tokenizer_src, tokenizer_tgt, model_config, device)
    print(f"输出文本 (德语): {output_text}")
    print("-" * 50)


if __name__ == '__main__':
    main()
