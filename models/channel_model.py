import torch
import torch.nn as nn
import numpy as np


class ChannelModel(nn.Module):
    """信道模型"""
    def __init__(self, noise_std=0.1, fading=True):
        super().__init__()
        self.noise_std = noise_std
        self.fading = fading

    def forward(self, x):
        if self.fading:
            h = torch.randn_like(x) * np.sqrt(0.5) + 1j * torch.randn_like(x) * np.sqrt(0.5)  # Rayleigh 信道
            x = x * torch.abs(h)  # 乘以信道增益

        noise = torch.randn_like(x) * self.noise_std  # AWGN 噪声
        return x + noise
