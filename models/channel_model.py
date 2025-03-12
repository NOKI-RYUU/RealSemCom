import torch
import torch.nn as nn


class ChannelModel(nn.Module):
    """语义信道模型"""
    def __init__(self, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return x + noise  # 简单 AWGN 信道
