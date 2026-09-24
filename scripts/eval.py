"""
评估脚本：计算测试集困惑度 (Perplexity)
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import math

from model import Transformer
from model.transformer import make_src_mask
from scripts.train import WikiTextDataset


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            src_mask = make_src_mask(src, pad_idx=0)
            
            output = model(src, tgt, src_mask=src_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            
            total_loss += loss.item() * src.size(0) * src.size(1)
            total_tokens += (tgt != 0).sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    print("加载测试数据...")
    test_dataset = WikiTextDataset(
        os.path.join(config["data"]["data_dir"], "test.txt"),
        seq_len=config["model"]["max_len"]
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False
    )
    
    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = Transformer(
        src_vocab_size=len(checkpoint['vocab']),
        tgt_vocab_size=len(checkpoint['vocab']),
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        d_ff=config["model"]["d_ff"],
        max_len=config["model"]["max_len"],
        dropout=config["model"]["dropout"]
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    avg_loss, perplexity = evaluate(model, test_loader, criterion, device)
    
    print(f"\n{'='*50}")
    print(f"测试结果")
    print(f"{'='*50}")
    print(f"  平均 Loss: {avg_loss:.4f}")
    print(f"  困惑度 (PPL): {perplexity:.2f}")
    print(f"{'='*50}")
    
    # 生成示例文本
    print("\n生成示例：")
    model.eval()
    with torch.no_grad():
        # 输入一个起始句子
        start_tokens = [checkpoint['vocab'].get(w, 1) for w in ["The", "cat", "is"]]
        input_ids = torch.tensor([start_tokens]).to(device)
        
        for _ in range(20):
            src_mask = make_src_mask(input_ids)
            output = model(input_ids, input_ids, src_mask=src_mask)
            next_token = output[0, -1].argmax().item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(device)], dim=1)
            
            if next_token == 3:  # <eos>
                break
        
        # 解码
        idx2word = {v: k for k, v in checkpoint['vocab'].items()}
        generated = " ".join([idx2word.get(t.item(), "<unk>") for t in input_ids[0]])
        print(f"  {generated}")


if __name__ == "__main__":
    main()
