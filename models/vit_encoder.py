import torch
import torch.nn as nn
import timm


class ViTEncoder(nn.Module):
    """第一层 ViT 编码器（冻结参数）"""
    def __init__(self, model_name="vit_base_patch16_224", freeze=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False  # 冻结参数

    def forward(self, x):
        return self.vit(x)
