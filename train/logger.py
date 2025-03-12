import os
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """TensorBoard 训练日志管理"""
    def __init__(self, log_dir="logs/"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """记录标量数据，如损失"""
        self.writer.add_scalar(tag, value, step)

    def log_image(self, tag, image, step):
        """记录图片数据"""
        self.writer.add_image(tag, image, step)

    def close(self):
        """关闭 TensorBoard 记录"""
        self.writer.close()
