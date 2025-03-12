import torch
import torch.nn as nn
import timm


class ViTTransformerRX(nn.Module):
    """接收端 第一层 ViT 计算 `F_recovered`"""
    def __init__(self, model_name="vit_base_patch16_224"):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)

    def forward(self, offset_feature, ref_feature):
        """
        offset_feature: 信道传输后的 `F_offset`
        ref_feature: 知识库中最近邻 (Query)
        """
        return self.vit(ref_feature, offset_feature, offset_feature)
