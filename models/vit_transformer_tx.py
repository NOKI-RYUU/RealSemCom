import torch
import torch.nn as nn
import timm


class ViTTransformerTX(nn.Module):
    """发送端 第二层 ViT 计算偏位特征"""
    def __init__(self, model_name="vit_base_patch16_224"):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)

    def forward(self, enc_feature, ref_feature):
        """
        enc_feature: 第一层 ViT 输出 (Key, Value)
        ref_feature: 知识库中最近邻 (Query)
        """
        return self.vit(ref_feature, enc_feature, enc_feature)
