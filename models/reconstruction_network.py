import torch
import torch.nn as nn


class ReconstructionNetwork(nn.Module):
    """接收端 第二层 ViT 进行最终图像重构"""
    def __init__(self, d_model=768, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.skip_connection = nn.Identity()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, recovered_feature):
        attn_output, _ = self.multihead_attn(recovered_feature.unsqueeze(0), recovered_feature.unsqueeze(0), recovered_feature.unsqueeze(0))
        x = self.norm1(attn_output.squeeze(0))
        x = self.fc1(x)
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.skip_connection(x + recovered_feature)  # Skip Connection
        return self.softmax(x)
