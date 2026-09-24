import torch
import torch.nn as nn

from .embedding import TokenEmbedding, PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    
    包含：
    - 源序列嵌入 + 位置编码
    - 目标序列嵌入 + 位置编码
    - 编码器
    - 解码器
    - 输出线性层 + softmax
    """
    
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()
        
        # 嵌入层
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # 编码器和解码器
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        # 输出层（映射到词表大小）
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """Xavier 初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: (batch_size, src_len) 源序列
            tgt: (batch_size, tgt_len) 目标序列
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（通常是下三角掩码）
        """
        # 1. 源序列嵌入 + 位置编码
        src_emb = self.pos_encoding(self.src_embedding(src))
        
        # 2. 编码器
        encoder_output = self.encoder(src_emb, src_mask)
        
        # 3. 目标序列嵌入 + 位置编码
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))
        
        # 4. 解码器
        decoder_output = self.decoder(
            tgt_emb, encoder_output,
            self_mask=tgt_mask,
            cross_mask=src_mask
        )
        
        # 5. 输出层
        output = self.output_linear(decoder_output)
        
        return output
    
    def encode(self, src, src_mask=None):
        """仅编码（用于推理时缓存）"""
        src_emb = self.pos_encoding(self.src_embedding(src))
        return self.encoder(src_emb, src_mask)
    
    def decode(self, tgt, encoder_output, tgt_mask=None, cross_mask=None):
        """仅解码（用于自回归生成）"""
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))
        decoder_output = self.decoder(
            tgt_emb, encoder_output,
            self_mask=tgt_mask,
            cross_mask=cross_mask
        )
        return self.output_linear(decoder_output)


def make_src_mask(src, pad_idx=0):
    """创建源序列掩码（遮挡填充位置）"""
    # src: (batch_size, src_len)
    mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
    return mask


def make_tgt_mask(tgt, pad_idx=0):
    """创建目标序列掩码（遮挡填充 + 未来信息）"""
    batch_size, tgt_len = tgt.shape
    
    # 填充掩码
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
    
    # 下三角掩码（遮挡未来信息）
    sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
    sub_mask = sub_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, tgt_len)
    
    # 合并
    mask = pad_mask & sub_mask
    return mask
