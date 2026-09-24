import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """词嵌入层：把词 ID 变成向量"""
    
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        # x: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    """
    位置编码：给每个位置一个唯一的向量表示
    
    使用正弦和余弦函数，让模型知道词的顺序。
    公式：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 预先计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # div_term = 1 / (10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 偶数位置用 sin，奇数位置用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # pe: (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        # 把位置编码加到词嵌入上
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
