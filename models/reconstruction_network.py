import torch
import torch.nn as nn
import timm


class ReconstructionNetwork(nn.Module):
    """接收端 第二层 ViT 进行最终图像重构"""
    def __init__(self, model_name="vit_base_patch16_224"):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)

    def forward(self, recovered_feature):
        return self.vit(recovered_feature)
