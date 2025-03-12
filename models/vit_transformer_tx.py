import torch
import torch.nn as nn
import timm


class ViTTransformerTX(nn.Module):
    """发送端 第二层 ViT 计算偏位特征"""
    def __init__(self, model_name="vit_base_patch16_224", d_model=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_feature, ref_feature):
        """
        enc_feature: 第一层 ViT 输出 (Key, Value)
        ref_feature: 知识库中最近邻 (Query)
        """
        attn_output, _ = self.cross_attn(ref_feature.unsqueeze(0), enc_feature.unsqueeze(0), enc_feature.unsqueeze(0))
        return self.norm(attn_output.squeeze(0))
