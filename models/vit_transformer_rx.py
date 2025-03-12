import torch
import torch.nn as nn


class ViTTransformerRX(nn.Module):
    """接收端 第一层 ViT 计算 `F_recovered`"""
    def __init__(self, model_name="vit_base_patch16_224", d_model=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, offset_feature, ref_feature):
        """
        offset_feature: 信道传输后的 `F_offset`
        ref_feature: 知识库中最近邻 (Query)
        """
        attn_output, _ = self.cross_attn(ref_feature.unsqueeze(0), offset_feature.unsqueeze(0), offset_feature.unsqueeze(0))
        return self.norm(attn_output.squeeze(0))
