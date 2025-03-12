import torch
import torch.nn as nn


class ViTTransformerTX(nn.Module):
    """发送端 第二层 ViT 计算偏位特征，加入门控机制"""
    def __init__(self, d_model=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.gate = nn.Parameter(torch.Tensor([0.5]))  # 可学习权重
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_feature, ref_feature):
        """
        enc_feature: 第一层 ViT 输出 (Key, Value)
        ref_feature: 知识库中最近邻 (Query)
        """
        attn_output, _ = self.cross_attn(ref_feature.unsqueeze(0), enc_feature.unsqueeze(0), enc_feature.unsqueeze(0))
        gated_output = self.gate * attn_output + (1 - self.gate) * enc_feature.unsqueeze(0)  # 门控融合
        return self.norm(gated_output.squeeze(0))
