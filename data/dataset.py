import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import faiss


class ImageDataset(Dataset):
    """数据集，自动查找知识库最近邻"""
    def __init__(self, dataset_path):
        self.image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 加载知识库
        self.index = faiss.read_index("knowledge_base/faiss_index/full_image.index")
        self.knowledge_vectors = np.load("knowledge_base/full_image_vectors.npy")

    def find_nearest_feature(self, feature):
        """查找知识库中最接近的原位特征"""
        distances, indices = self.index.search(feature.reshape(1, -1), k=1)
        return self.knowledge_vectors[indices[0][0]]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """返回：图像, 最近邻特征"""
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        return image_tensor, torch.tensor(self.find_nearest_feature(image_tensor.numpy()))
