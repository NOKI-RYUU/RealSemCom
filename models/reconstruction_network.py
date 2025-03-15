import torch
import torch.nn as nn
import timm


class ReconstructionNetwork(nn.Module):
    """ViT + ResNet 进行最终图像重构，确保输出与原图像大小一致"""
    def __init__(self, d_model=768, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # **MLP 变换回 CNN 兼容的形状**
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 7 * 7 * 512),  # 变成 (Batch, 7*7*512)
            nn.ReLU()
        )

        # **使用 ResNet 细节增强**
        self.resnet = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        )

        # **CNN 逐步上采样回 224x224**
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出 3x224x224
            nn.Sigmoid()  # 归一化到 [0,1]
        )

    def forward(self, recovered_feature):
        """
        recovered_feature: (batch_size, 768) -> (batch_size, 3, 224, 224)
        """
        norm_out = self.norm(recovered_feature)  # 归一化

        # **MLP 变换回 CNN 输入形状**
        mlp_out = self.mlp(norm_out)  # (batch_size, 7*7*512)
        mlp_out = mlp_out.view(-1, 512, 7, 7)  # 变换成 CNN 格式 (batch_size, 512, 7, 7)

        # **使用 ResNet 细节增强**
        resnet_out = self.resnet(mlp_out) + mlp_out  # 残差连接

        # **CNN 逐步上采样回 224x224**
        output = self.upsample(resnet_out)  # (batch_size, 3, 224, 224)

        return output
