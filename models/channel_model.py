import torch
import torch.nn as nn
import math


class ChannelModel(nn.Module):
    """信道模型，支持 AWGN、Rayleigh、Rician，训练时支持随机 SNR"""
    def __init__(self, channel_type="AWGN", k_factor=5.0):
        """
        :param channel_type: "AWGN", "Rayleigh", or "Rician"
        :param k_factor: K-factor for Rician fading
        """
        super().__init__()
        self.channel_type = channel_type
        self.k_factor = k_factor

    def power_normalize(self, x):
        """信号功率归一化"""
        x_square = torch.mul(x, x)
        power = math.sqrt(2) * x_square.mean(dim=1, keepdim=True).sqrt()
        return torch.div(x, power)

    def snr_to_noise(self, snr):
        """计算 SNR 对应的噪声标准差"""
        snr = 10 ** (snr / 10)
        return 1 / ((2 * snr) ** 0.5)

    def forward(self, Tx_sig, snr):
        """
        :param Tx_sig: 发送信号
        :param snr: 当前 batch 选定的 SNR
        """
        noise_std = self.snr_to_noise(snr)
        Tx_sig = self.power_normalize(Tx_sig)

        if self.channel_type == "AWGN":
            noise = torch.normal(0, noise_std, size=Tx_sig.shape, device=Tx_sig.device)
            return Tx_sig + noise

        elif self.channel_type == "Rayleigh":
            H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(Tx_sig.device)
            H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(Tx_sig.device)
            H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(Tx_sig.device)
            Tx_sig = torch.matmul(Tx_sig.view(Tx_sig.shape[0], -1, 2), H)
            noise = torch.normal(0, noise_std, size=Tx_sig.shape, device=Tx_sig.device)
            Rx_sig = Tx_sig + noise
            return torch.matmul(Rx_sig, torch.inverse(H)).view(Tx_sig.shape)

        elif self.channel_type == "Rician":
            K = 10**(self.k_factor / 10)
            mean = math.sqrt(K / (K + 1))
            std = math.sqrt(1 / (K + 1)) * math.sqrt(1/2)
            H_real = torch.normal(mean, std, size=[1]).to(Tx_sig.device)
            H_imag = torch.normal(0, std, size=[1]).to(Tx_sig.device)
            H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(Tx_sig.device)
            Tx_sig = torch.matmul(Tx_sig.view(Tx_sig.shape[0], -1, 2), H)
            noise = torch.normal(0, noise_std, size=Tx_sig.shape, device=Tx_sig.device)
            Rx_sig = Tx_sig + noise
            return torch.matmul(Rx_sig, torch.inverse(H)).view(Tx_sig.shape)

        else:
            raise ValueError("Unsupported channel type. Choose from 'AWGN', 'Rayleigh', 'Rician'.")
