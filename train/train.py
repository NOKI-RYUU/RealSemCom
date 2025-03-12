import os
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

# 训练参数
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CHECKPOINTS = 5  # 只保留最近 5 个 checkpoint
CHECKPOINT_DIR = "checkpoints/"
BEST_MODEL_PATHS = {
    "tx_vit": "best_model_tx.pth",
    "rx_vit": "best_model_rx.pth",
    "decoder": "best_model_decoder.pth",
}
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")

# 初始化模型
encoder = ViTEncoder().to(DEVICE)
tx_vit = ViTTransformerTX().to(DEVICE)
rx_vit = ViTTransformerRX().to(DEVICE)
decoder = ReconstructionNetwork().to(DEVICE)
channel = ChannelModel().to(DEVICE)

# 加载数据
dataset = ImageDataset("data/images")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化优化器
optimizer = optim.Adam(
    list(tx_vit.parameters()) + list(rx_vit.parameters()) + list(decoder.parameters()),
    lr=LEARNING_RATE,
)
criterion = nn.MSELoss()

# 训练日志（TensorBoard）
logger = Logger()

# 最佳损失初始化
best_loss = float("inf")

# 尝试加载上次训练的 checkpoint
if os.path.exists(LATEST_CHECKPOINT):
    tx_vit, optimizer = load_checkpoint(tx_vit, optimizer, LATEST_CHECKPOINT)
    rx_vit, optimizer = load_checkpoint(rx_vit, optimizer, LATEST_CHECKPOINT)
    decoder, optimizer = load_checkpoint(decoder, optimizer, LATEST_CHECKPOINT)
    print(f"✅ 成功加载 {LATEST_CHECKPOINT}，继续训练...")

# 训练循环
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

    # 保存 checkpoint（每 5 轮保存一次）
    if epoch % 5 == 0:
        save_checkpoint(tx_vit, optimizer, epoch, avg_epoch_loss, checkpoint_dir=CHECKPOINT_DIR, max_checkpoints=MAX_CHECKPOINTS)
        save_checkpoint(rx_vit, optimizer, epoch, avg_epoch_loss, checkpoint_dir=CHECKPOINT_DIR, max_checkpoints=MAX_CHECKPOINTS)
        save_checkpoint(decoder, optimizer, epoch, avg_epoch_loss, checkpoint_dir=CHECKPOINT_DIR, max_checkpoints=MAX_CHECKPOINTS)

        # 保存最新 checkpoint 以支持断点续训
        save_checkpoint(tx_vit, optimizer, epoch, avg_epoch_loss, checkpoint_dir=CHECKPOINT_DIR, max_checkpoints=1)

    # 更新最佳模型
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        save_best_model(tx_vit, optimizer, epoch, best_loss, best_model_path=BEST_MODEL_PATHS["tx_vit"])
        save_best_model(rx_vit, optimizer, epoch, best_loss, best_model_path=BEST_MODEL_PATHS["rx_vit"])
        save_best_model(decoder, optimizer, epoch, best_loss, best_model_path=BEST_MODEL_PATHS["decoder"])

# 关闭日志
logger.close()
