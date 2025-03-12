import torch
import torch.nn as nn
import numpy as np


class ChannelModel(nn.Module):
    """信道模型，支持 AWGN、Rayleigh 衰落和 Rician 衰落"""
    def __init__(self, noise_std=0.1, channel_type="AWGN", k_factor=3.0):
        """
        :param noise_std: 高斯噪声标准差（AWGN & Rician）
        :param channel_type: "AWGN" / "Rayleigh" / "Rician"
        :param k_factor: Rician 信道的 K 因子（直射路径功率 / 多径功率）
        """
        super().__init__()
        self.noise_std = noise_std
        self.channel_type = channel_type
        self.k_factor = k_factor  # Rician 信道 K 因子

    def forward(self, x):
        """
        :param x: 输入信号 (Batch, Feature_dim)
        :return: 通过信道后的信号
        """
        batch_size, feature_dim = x.shape

        # AWGN 信道（仅加性高斯噪声）
        if self.channel_type == "AWGN":
            noise = torch.randn_like(x) * self.noise_std
            return x + noise

        # Rayleigh 信道（无直射路径，仅多径衰落）
        elif self.channel_type == "Rayleigh":
            h_real = torch.randn(batch_size, feature_dim, device=x.device) * np.sqrt(0.5)
            h_imag = torch.randn(batch_size, feature_dim, device=x.device) * np.sqrt(0.5)
            h = torch.sqrt(h_real**2 + h_imag**2)  # Rayleigh 分布
            noise = torch.randn_like(x) * self.noise_std
            return h * x + noise  # 信道衰落 + 噪声

        # Rician 信道（有直射路径 + 多径）
        elif self.channel_type == "Rician":
            K = self.k_factor  # 直射路径和多径传播的功率比
            h_los = torch.ones(batch_size, feature_dim, device=x.device) * np.sqrt(K / (K + 1))  # 直射路径
            h_nlos_real = torch.randn(batch_size, feature_dim, device=x.device) * np.sqrt(0.5 / (K + 1))
            h_nlos_imag = torch.randn(batch_size, feature_dim, device=x.device) * np.sqrt(0.5 / (K + 1))
            h_nlos = torch.sqrt(h_nlos_real**2 + h_nlos_imag**2)  # 多径
            h = h_los + h_nlos  # Rician 信道系数
            noise = torch.randn_like(x) * self.noise_std
            return h * x + noise  # 信道衰落 + 噪声

        else:
            raise ValueError("Unsupported channel type. Choose from 'AWGN', 'Rayleigh', 'Rician'.")
