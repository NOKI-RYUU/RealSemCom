import os
import torch
import faiss
import numpy as np


def find_nearest_feature(feature, index_path, vectors_path):
    """查找知识库中最近邻的特征"""
    index = faiss.read_index(index_path)
    knowledge_vectors = np.load(vectors_path)
    distances, indices = index.search(feature.reshape(1, -1), k=1)
    return knowledge_vectors[indices[0][0]]


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints/", max_checkpoints=5):
    """保存 Checkpoint，最多保留 max_checkpoints 个"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Checkpoint 已保存: {checkpoint_path}")

    # 清理旧的 checkpoint（最多保留 max_checkpoints 个）
    checkpoints = sorted(os.listdir(checkpoint_dir))
    if len(checkpoints) > max_checkpoints:
        os.remove(os.path.join(checkpoint_dir, checkpoints[0]))


def save_best_model(model, optimizer, epoch, loss, best_model_path="best_model.pth"):
    """保存最佳模型"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, best_model_path)
    print(f"🏆 最佳模型已更新: {best_model_path}（Loss: {loss:.5f}）")


def load_checkpoint(model, optimizer, load_path="best_model.pth"):
    """加载模型"""
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"🔄 加载模型: {load_path}（Epoch: {checkpoint['epoch']}）")
    return model, optimizer
