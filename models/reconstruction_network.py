import torch
import torch.nn as nn
import timm


class ReconstructionNetwork(nn.Module):
    """ViT + ResNet 进行最终图像重构，提升细节保留能力"""
    def __init__(self, d_model=768, num_heads=8):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.norm = nn.LayerNorm(d_model)

        # MLP 变换回 CNN 兼容的形状
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 7 * 7 * 512),
            nn.ReLU()
        )

        # 使用 ResNet 残差连接增强细节
        self.resnet = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        )

        # CNN 上采样回 224x224
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 归一化到 [0,1]
        )

    def forward(self, recovered_feature):
        """
        recovered_feature: (batch_size, 768) -> (batch_size, 3, 224, 224)
        """
        vit_out = self.vit(recovered_feature)  # ViT 提取全局特征
        norm_out = self.norm(vit_out)  # LayerNorm 归一化

        # MLP 变换回 CNN 输入形状
        mlp_out = self.mlp(norm_out)  # (batch_size, 7*7*512)
        mlp_out = mlp_out.view(-1, 512, 7, 7)  # 变换成 CNN 格式 (batch_size, 512, 7, 7)

        # **使用 ResNet 细节增强**
        resnet_out = self.resnet(mlp_out) + mlp_out  # 残差连接

        # CNN 逐步上采样回 224x224
        output = self.upsample(resnet_out)  # (batch_size, 3, 224, 224)

        return output
