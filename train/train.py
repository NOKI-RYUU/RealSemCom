import sys
import os
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from models.vit_encoder import ViTEncoder
from models.vit_transformer_tx import ViTTransformerTX
from models.vit_transformer_rx import ViTTransformerRX
from models.reconstruction_network import ReconstructionNetwork
from models.channel_model import ChannelModel
from data.dataset import ImageDataset
from utils import save_checkpoint, save_best_model, load_checkpoint
from logger import Logger
from torch.optim.lr_scheduler import CosineAnnealingLR


# **✅ 统一路径管理**
parser = argparse.ArgumentParser(description="Train Semantic Communication System")
parser.add_argument("--data_path", type=str, default="data/coco_dataset_original/train2017", help="Path to training images")
parser.add_argument("--faiss_index", type=str, default="data/knowledge_bases_coco_crop_background/knowledge_base_threshold_0.3/faiss_index/knowledge_base.index", help="Path to FAISS index")
parser.add_argument("--faiss_vectors", type=str, default="data/knowledge_bases_coco_crop_background/encoded_features/image_features.npy", help="Path to FAISS feature vectors")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--alpha", type=float, default=0.8, help="Weight for MSE in loss function (1-alpha for SSIM)")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Path to save checkpoints")
parser.add_argument("--max_checkpoints", type=int, default=5, help="Maximum number of checkpoints to keep")
parser.add_argument("--channel_type", type=str, choices=["AWGN", "Rayleigh", "Rician"], default="AWGN", help="Type of channel model")
parser.add_argument("--k_factor", type=float, default=5.0, help="K-factor for Rician fading")
parser.add_argument("--scheduler_tmax", type=int, default=100, help="T_max parameter for CosineAnnealingLR")
parser.add_argument("--scheduler_eta_min", type=float, default=1e-6, help="Minimum learning rate for CosineAnnealingLR")
args = parser.parse_args()

# **✅ 训练参数**
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
ALPHA = args.alpha  # SSIM / MSE 组合权重
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = args.checkpoint_dir
MAX_CHECKPOINTS = args.max_checkpoints
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")

# **✅ 初始化模型**
encoder = ViTEncoder().to(DEVICE)
tx_vit = ViTTransformerTX().to(DEVICE)
rx_vit = ViTTransformerRX().to(DEVICE)
decoder = ReconstructionNetwork().to(DEVICE)
channel = ChannelModel(channel_type=args.channel_type, k_factor=args.k_factor).to(DEVICE)

# **✅ 预加载 FAISS 数据，提高查询效率**
dataset = ImageDataset(args.data_path, args.faiss_index, args.faiss_vectors)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32, pin_memory=True)

# **✅ 采用 AdamW + 学习率调度**
optimizer = optim.AdamW(
    list(tx_vit.parameters()) + list(rx_vit.parameters()) + list(decoder.parameters()),
    lr=LEARNING_RATE, weight_decay=1e-4
)
scheduler = CosineAnnealingLR(optimizer, T_max=args.scheduler_tmax, eta_min=args.scheduler_eta_min)

# **✅ 组合 SSIM + MSE Loss**
mse_loss = nn.MSELoss()
ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

def combined_loss(reconstructed, target, alpha=ALPHA):
    """联合 MSE + SSIM Loss"""
    mse = mse_loss(reconstructed, target)
    ssim = 1 - ssim_loss(reconstructed, target)  # SSIM 越大越好
    return alpha * mse + (1 - alpha) * ssim

# **✅ 训练日志（TensorBoard）**
logger = Logger()

# **✅ 训练循环**
best_loss = float("inf")

for epoch in range(EPOCHS):
    epoch_loss = 0
    snr = random.randint(0, 10)  # **✅ 随机 SNR**
    print(f"📡 Epoch {epoch+1}: 使用 SNR={snr}")

    for batch_idx, (image, ref_feature) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        image, ref_feature = image.to(DEVICE, non_blocking=True), ref_feature.to(DEVICE, non_blocking=True)

        enc_feature = encoder(image)
        offset_feature = tx_vit(enc_feature, ref_feature)
        channel_feature = channel(offset_feature, snr)
        recovered_feature = rx_vit(channel_feature, ref_feature)
        reconstructed_image = decoder(recovered_feature)

        loss = combined_loss(reconstructed_image, image, alpha=ALPHA)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()  # **✅ 更新学习率**
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"📉 Epoch {epoch+1} Loss: {avg_epoch_loss:.5f}")

    # **✅ 保存最佳模型**
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        save_best_model(tx_vit, optimizer, epoch, best_loss, "best_model_tx.pth")
        save_best_model(rx_vit, optimizer, epoch, best_loss, "best_model_rx.pth")
        save_best_model(decoder, optimizer, epoch, best_loss, "best_model_decoder.pth")

# **✅ 关闭日志**
logger.close()
