import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.vit_encoder import ViTEncoder
from models.vit_transformer_tx import ViTTransformerTX
from models.vit_transformer_rx import ViTTransformerRX
from models.reconstruction_network import ReconstructionNetwork
from models.channel_model import ChannelModel
from data.dataset import ImageDataset
from train.utils import save_checkpoint, save_best_model, load_checkpoint
from train.logger import Logger

# **统一路径管理**
parser = argparse.ArgumentParser(description="Train Semantic Communication System")
parser.add_argument("--data_path", type=str, default="data/images", help="Path to training images")
parser.add_argument("--faiss_index", type=str, default="knowledge_base/faiss_index/full_image.index", help="Path to FAISS index")
parser.add_argument("--faiss_vectors", type=str, default="knowledge_base/full_image_vectors.npy", help="Path to FAISS feature vectors")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Path to save checkpoints")
parser.add_argument("--max_checkpoints", type=int, default=5, help="Maximum number of checkpoints to keep")
parser.add_argument("--channel_type", type=str, choices=["AWGN", "Rayleigh", "Rician"], default="AWGN", help="Type of channel model")
parser.add_argument("--k_factor", type=float, default=5.0, help="K-factor for Rician fading")
args = parser.parse_args()

# **训练参数**
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = args.checkpoint_dir
MAX_CHECKPOINTS = args.max_checkpoints
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")

# **初始化模型**
encoder = ViTEncoder().to(DEVICE)
tx_vit = ViTTransformerTX().to(DEVICE)
rx_vit = ViTTransformerRX().to(DEVICE)
decoder = ReconstructionNetwork().to(DEVICE)
channel = ChannelModel(noise_std=0.1, channel_type=args.channel_type, k_factor=args.k_factor).to(DEVICE)

# **加载数据**
dataset = ImageDataset(args.data_path, args.faiss_index, args.faiss_vectors)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# **初始化优化器**
optimizer = optim.Adam(
    list(tx_vit.parameters()) + list(rx_vit.parameters()) + list(decoder.parameters()),
    lr=LEARNING_RATE,
)
criterion = nn.MSELoss()

# **训练日志（TensorBoard）**
logger = Logger()

# **尝试加载上次训练的 checkpoint**
if os.path.exists(LATEST_CHECKPOINT):
    tx_vit, optimizer = load_checkpoint(tx_vit, optimizer, LATEST_CHECKPOINT)
    rx_vit, optimizer = load_checkpoint(rx_vit, optimizer, LATEST_CHECKPOINT)
    decoder, optimizer = load_checkpoint(decoder, optimizer, LATEST_CHECKPOINT)
    print(f"✅ 成功加载 {LATEST_CHECKPOINT}，继续训练...")

# **训练循环**
best_loss = float("inf")

for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_idx, (image, ref_feature) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        image, ref_feature = image.to(DEVICE), ref_feature.to(DEVICE)

        # 发送端
        enc_feature = encoder(image)
        offset_feature = tx_vit(enc_feature, ref_feature)

        # 信道模型
        channel_feature = channel(offset_feature)

        # 接收端
        recovered_feature = rx_vit(channel_feature, ref_feature)
        reconstructed_image = decoder(recovered_feature)

        # 计算损失
        loss = criterion(reconstructed_image, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 记录日志
        logger.log_scalar("Loss/Batch", loss.item(), epoch * len(dataloader) + batch_idx)

    # 记录每轮损失
    avg_epoch_loss = epoch_loss / len(dataloader)
    logger.log_scalar("Loss/Epoch", avg_epoch_loss, epoch)
    print(f"📉 Epoch {epoch+1} Loss: {avg_epoch_loss:.5f}")

    # **保存 Checkpoint**
    if epoch % 5 == 0:
        save_checkpoint(tx_vit, optimizer, epoch, avg_epoch_loss, CHECKPOINT_DIR, MAX_CHECKPOINTS)
        save_checkpoint(rx_vit, optimizer, epoch, avg_epoch_loss, CHECKPOINT_DIR, MAX_CHECKPOINTS)
        save_checkpoint(decoder, optimizer, epoch, avg_epoch_loss, CHECKPOINT_DIR, MAX_CHECKPOINTS)

    # **保存最佳模型**
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        save_best_model(tx_vit, optimizer, epoch, best_loss, "best_model_tx.pth")
        save_best_model(rx_vit, optimizer, epoch, best_loss, "best_model_rx.pth")
        save_best_model(decoder, optimizer, epoch, best_loss, "best_model_decoder.pth")

# **关闭日志**
logger.close()
