import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import faiss


class ImageDataset(Dataset):
    """训练数据加载 + FAISS 查找最近邻"""
    def __init__(self, dataset_path, index_path, vectors_path):
        """
        :param dataset_path: 训练图像存放目录
        :param index_path: FAISS 索引路径
        :param vectors_path: 知识库特征向量路径
        """
        self.image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith(('.jpg', '.png'))]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"错误: 数据目录 '{dataset_path}' 为空！请检查数据路径。")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 确保 FAISS 数据库存在
        if not os.path.exists(index_path) or not os.path.exists(vectors_path):
            raise ValueError(f"错误: FAISS 索引或向量文件不存在！\n检查: {index_path} 和 {vectors_path}")

        self.index = faiss.read_index(index_path)
        self.knowledge_vectors = np.load(vectors_path)

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)

    def __getitem__(self, index):
        """返回图像 Tensor + 最近邻特征"""
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        distances, indices = self.index.search(image_tensor.flatten().reshape(1, -1), k=1)
        ref_feature = self.knowledge_vectors[indices[0][0]]

        return image_tensor, torch.tensor(ref_feature)
