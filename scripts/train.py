"""
训练脚本：在 WikiText-2 上训练 Transformer 语言模型
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math

from model import Transformer
from model.transformer import make_src_mask


class WikiTextDataset(Dataset):
    """WikiText-2 数据集"""
    
    def __init__(self, filepath, seq_len=256):
        self.seq_len = seq_len
        self.vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>"}
        self.data = self._load_and_tokenize(filepath)
        
    def _load_and_tokenize(self, filepath):
        """加载文本并构建词表"""
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 构建词表
        word_counts = {}
        for line in lines:
            words = line.strip().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # 只保留出现频率 >= 2 的词
        for word, count in word_counts.items():
            if count >= 2 and word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.idx2word[idx] = word
        
        print(f"词表大小: {len(self.vocab)}")
        
        # 转换为 token ID 序列
        tokens = [self.vocab["<bos>"]]
        for line in lines:
            words = line.strip().split()
            for word in words:
                tokens.append(self.vocab.get(word, self.vocab["<unk>"]))
            tokens.append(self.vocab["<eos>"])
        
        return tokens
    
    def __len__(self):
        return len(self.data) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        
        # 输入：前 seq_len 个 token
        x = self.data[start:end]
        # 目标：后 seq_len 个 token（向右移一位）
        y = self.data[start+1:end+1] if end+1 <= len(self.data) else self.data[start:end] + [self.vocab["<pad>"]]
        
        # 补齐长度
        if len(x) < self.seq_len:
            x = x + [self.vocab["<pad>"]] * (self.seq_len - len(x))
        if len(y) < self.seq_len:
            y = y + [self.vocab["<pad>"]] * (self.seq_len - len(y))
        
        return torch.tensor(x), torch.tensor(y)


def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    progress = tqdm(dataloader, desc="Training")
    for src, tgt in progress:
        src = src.to(device)
        tgt = tgt.to(device)
        
        # 创建掩码
        src_mask = make_src_mask(src, pad_idx=0)
        
        # 前向传播
        output = model(src, tgt, src_mask=src_mask)
        
        # 计算损失（忽略 padding 位置）
        # output: (batch, seq_len, vocab_size)
        # tgt: (batch, seq_len)
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item() * src.size(0) * src.size(1)
        total_tokens += (tgt != 0).sum().item()
        
        progress.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设备
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(config["seed"])
    
    # 数据
    print("加载数据...")
    train_dataset = WikiTextDataset(
        os.path.join(config["data"]["data_dir"], "train.txt"),
        seq_len=config["model"]["max_len"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"]
    )
    
    # 模型
    print("构建模型...")
    model = Transformer(
        src_vocab_size=len(train_dataset.vocab),
        tgt_vocab_size=len(train_dataset.vocab),
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        d_ff=config["model"]["d_ff"],
        max_len=config["model"]["max_len"],
        dropout=config["model"]["dropout"]
    ).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 创建保存目录
    os.makedirs(config["train"]["save_dir"], exist_ok=True)
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(config["train"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['train']['epochs']}")
        print("-" * 50)
        
        avg_loss, perplexity = train_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_grad=config["train"]["clip_grad"]
        )
        
        print(f"  Loss: {avg_loss:.4f} | PPL: {perplexity:.2f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
                'vocab': train_dataset.vocab
            }
            torch.save(
                checkpoint,
                os.path.join(config["train"]["save_dir"], "best_model.pt")
            )
            print(f"  ✓ 保存最佳模型 (Loss: {best_loss:.4f})")
        
        # 每 5 个 epoch 保存一次
        if (epoch + 1) % 5 == 0:
            torch.save(
                checkpoint,
                os.path.join(config["train"]["save_dir"], f"model_epoch_{epoch+1}.pt")
            )
    
    print(f"\n训练完成！最佳 Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
