import torch.nn as nn

from .attention import MultiHeadAttention, FeedForward


class DecoderLayer(nn.Module):
    """
    单个解码器层
    
    结构：掩码自注意力 -> 残差+归一化 -> 编码器-解码器注意力 -> 残差+归一化 -> 前馈网络 -> 残差+归一化
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        """
        Args:
            x: (batch_size, tgt_len, d_model) 解码器输入
            encoder_output: (batch_size, src_len, d_model) 编码器输出
        """
        # 1. 掩码自注意力（只能看已生成的词）
        self_attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 2. 编码器-解码器注意力（看源序列）
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Decoder(nn.Module):
    """堆叠多个解码器层"""
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)
        return self.norm(x)
