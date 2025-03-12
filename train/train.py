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
from train.utils import save_checkpoint, load_checkpoint
from train.logger import Logger

# 训练参数
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
encoder = ViTEncoder().to(DEVICE)
tx_vit = ViTTransformerTX().to(DEVICE)
rx_vit = ViTTransformerRX().to(DEVICE)
decoder = ReconstructionNetwork().to(DEVICE)
channel = ChannelModel().to(DEVICE)

# 训练
dataset = ImageDataset("data/images")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = optim.Adam(list(tx_vit.parameters()) + list(rx_vit.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# 日志管理
logger = Logger()

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
    print(f"Epoch {epoch+1} Loss: {avg_epoch_loss}")

    # 保存模型
    if epoch % 10 == 0:
        save_checkpoint(tx_vit, optimizer, epoch, avg_epoch_loss, "best_model_tx.pth")
        save_checkpoint(rx_vit, optimizer, epoch, avg_epoch_loss, "best_model_rx.pth")
        save_checkpoint(decoder, optimizer, epoch, avg_epoch_loss, "best_model_decoder.pth")

# 关闭日志
logger.close()
