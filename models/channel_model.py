import torch
import torch.nn as nn
import timm


class ReconstructionNetwork(nn.Module):
    """ViT + CNN 进行最终图像重构，加入 LayerNorm + Skip Connection"""
    def __init__(self, d_model=768, num_heads=8):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.norm = nn.LayerNorm(d_model)
        
        # CNN 负责细节增强
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

        self.skip_connection = nn.Identity()  # 跳跃连接

    def forward(self, recovered_feature):
        """
        recovered_feature: 从 ViTTransformerRX 输出的 F_recovered
        """
        vit_out = self.vit(recovered_feature)  # ViT 提取全局特征
        norm_out = self.norm(vit_out)  # LayerNorm 归一化
        
        # 跳跃连接（保留 ViT 语义信息）
        combined_feature = self.skip_connection(norm_out + recovered_feature) 
        
        # CNN 细节增强
        output = self.cnn(combined_feature.unsqueeze(1)).squeeze(1)

        return output
