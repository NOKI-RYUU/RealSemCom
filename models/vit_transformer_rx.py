import torch
import torch.nn as nn


class ViTTransformerRX(nn.Module):
    """接收端 第一层 ViT 计算 `F_recovered`，增加残差门控"""
    def __init__(self, d_model=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.gate = nn.Parameter(torch.Tensor([0.5]))  # 可学习权重
        self.norm = nn.LayerNorm(d_model)

    def forward(self, offset_feature, ref_feature):
        """
        offset_feature: 信道传输后的 `F_offset`
        ref_feature: 知识库中最近邻 (Query)
        """
        attn_output, _ = self.cross_attn(ref_feature.unsqueeze(0), offset_feature.unsqueeze(0), offset_feature.unsqueeze(0))
        residual_output = self.gate * attn_output + (1 - self.gate) * offset_feature.unsqueeze(0)  # 残差门控
        return self.norm(residual_output.squeeze(0))
