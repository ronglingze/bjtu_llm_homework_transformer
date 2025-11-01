import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from functools import partial

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]


def train_tokenizer(dataset, lang, vocab_size, data_dir, unk_token="[UNK]", pad_token="[PAD]", sos_token="[SOS]", eos_token="[EOS]"):
    """
    在数据集上训练一个新的WordPiece分词器
    """
    tokenizer_path = os.path.join(data_dir, f"tokenizer-{lang}.json")

    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer: {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)

    print(f"Training new tokenizer for '{lang}'...")
    tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield [ex[lang] for ex in dataset[i:i + batch_size]['translation']]

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    os.makedirs(data_dir, exist_ok=True)
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    return tokenizer


def collate_batch(batch, src_pad_idx, tgt_pad_idx):
    """
    自定义批处理函数，用于Dataloader
    """
    src_list, tgt_list = [], []
    for item in batch:
        src_list.append(item['src_ids'])
        tgt_list.append(item['tgt_ids'])

    src_padded = pad_sequence(src_list, batch_first=True, padding_value=src_pad_idx)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=tgt_pad_idx)

    return {'src': src_padded, 'tgt': tgt_padded}


def get_dataloaders(config: dict, data_dir: str):
    """
    主函数：加载IWSLT2017，训练分词器，并创建Dataloaders
    """

    print("Loading raw dataset...")
    raw_dataset = load_dataset("iwslt2017", "iwslt2017-en-de", cache_dir=data_dir)
    train_data = raw_dataset['train']
    val_data = raw_dataset['validation']

    config_vocab_size = config.get('vocab_size', 10000)
    tokenizer_src = train_tokenizer(train_data, 'en', config_vocab_size, data_dir, *SPECIAL_TOKENS)
    tokenizer_tgt = train_tokenizer(train_data, 'de', config_vocab_size, data_dir, *SPECIAL_TOKENS)

    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    src_pad_idx = tokenizer_src.token_to_id("[PAD]")
    tgt_pad_idx = tokenizer_tgt.token_to_id("[PAD]")

    sos_id_src = tokenizer_src.token_to_id("[SOS]")
    eos_id_src = tokenizer_src.token_to_id("[EOS]")
    sos_id_tgt = tokenizer_tgt.token_to_id("[SOS]")
    eos_id_tgt = tokenizer_tgt.token_to_id("[EOS]")

    print(f"Source (en) vocab size: {src_vocab_size}")
    print(f"Target (de) vocab size: {tgt_vocab_size}")
    print(f"Source PAD ID: {src_pad_idx}")
    print(f"Target PAD ID: {tgt_pad_idx}")

    def tokenize_and_format(examples):
        src_texts = [ex['en'] for ex in examples['translation']]
        tgt_texts = [ex['de'] for ex in examples['translation']]

        src_tokenized = tokenizer_src.encode_batch(src_texts)
        tgt_tokenized = tokenizer_tgt.encode_batch(tgt_texts)

        src_ids = [[sos_id_src] + s.ids + [eos_id_src] for s in src_tokenized]
        tgt_ids = [[sos_id_tgt] + t.ids + [eos_id_tgt] for t in tgt_tokenized]

        return {'src_ids': src_ids, 'tgt_ids': tgt_ids}

    print("Tokenizing datasets...")
    tokenized_train_ds = train_data.map(tokenize_and_format, batched=True, remove_columns=train_data.column_names)
    tokenized_val_ds = val_data.map(tokenize_and_format, batched=True, remove_columns=val_data.column_names)

    tokenized_train_ds.set_format(type='torch')
    tokenized_val_ds.set_format(type='torch')

    print("Creating DataLoaders...")

    collate_fn_with_padding = partial(collate_batch, src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)

    train_loader = DataLoader(tokenized_train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn_with_padding, num_workers=4)
    val_loader = DataLoader(tokenized_val_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn_with_padding, num_workers=4)

    return train_loader, val_loader, src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx
