import os
import torch
import numpy as np
import faiss


def find_nearest_feature(feature, index_path, vectors_path):
    """查找知识库中最近邻的特征"""
    index = faiss.read_index(index_path)
    knowledge_vectors = np.load(vectors_path)

    distances, indices = index.search(feature.reshape(1, -1), k=1)
    return knowledge_vectors[indices[0][0]]


def save_checkpoint(model, optimizer, epoch, loss, save_path="best_model.pth"):
    """保存模型"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, save_path)
    print(f"✅ 模型已保存: {save_path}")


def load_checkpoint(model, optimizer, load_path="best_model.pth"):
    """加载模型"""
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"🔄 加载模型: {load_path}（Epoch: {checkpoint['epoch']}）")
    return model, optimizer
