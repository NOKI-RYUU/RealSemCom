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
            nn.Linear(d_model, 14 * 14 * 512),  # 变换成 (Batch, 14*14*512)
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

        # **增加一次 `Upsample`，确保从 14x14 开始**
        self.initial_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # 14x14 -> 28x28

        # **CNN 逐步上采样回 224x224**
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28 -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56x56 -> 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112x112 -> 224x224
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # 进一步细化
            nn.Sigmoid()  # 归一化到 [0,1]
        )

    def forward(self, recovered_feature):
        """
        recovered_feature: (batch_size, 768) -> (batch_size, 3, 224, 224)
        """
        norm_out = self.norm(recovered_feature)  # 归一化

        # **MLP 变换回 CNN 输入形状**
        mlp_out = self.mlp(norm_out)  # (batch_size, 14*14*512)
        mlp_out = mlp_out.view(-1, 512, 14, 14)  # 变换成 CNN 格式 (batch_size, 512, 14, 14)

        # **额外增加一次 `Upsample` 让 14x14 变成 28x28**
        upsample_out = self.initial_upsample(mlp_out)

        # **使用 ResNet 细节增强**
        resnet_out = self.resnet(upsample_out) + upsample_out  # 残差连接

        # **CNN 逐步上采样回 224x224**
        output = self.upsample(resnet_out)  # (batch_size, 3, 224, 224)

        # **✅ 断言检查，确保输出尺寸正确**
        assert output.shape[-1] == 224 and output.shape[-2] == 224, \
            f"❌ 输出尺寸错误: 期望 (batch_size, 3, 224, 224)，但得到 {output.shape}"

        return output
