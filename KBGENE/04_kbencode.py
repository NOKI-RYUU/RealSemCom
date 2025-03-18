import os
import torch
import numpy as np
import json
import timm
import shutil
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from datetime import datetime


class ImageEncoder:
    def __init__(self, dataset_path, output_root, model_name="vit_base_patch16_224", device="cuda"):
        """
        初始化图像编码器
        :param dataset_path: 数据集路径，每个类别是一个子文件夹
        :param output_root: 存放已编码特征的路径
        :param model_name: ViT 模型
        :param device: 设备 (cuda/cpu)
        """
        self.device = device
        self.dataset_path = dataset_path
        self.output_path = os.path.join(output_root, "encoded_features")
        os.makedirs(self.output_path, exist_ok=True)

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.metadata = []  # 存储图片路径和类别信息
        self.features = []  # 存储特征向量

        self.log_file = os.path.join(self.output_path, "encoding_log.txt")
        with open(self.log_file, "w") as log:
            log.write(f"编码任务开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"数据集路径: {self.dataset_path}\n\n")

    def log(self, message):
        """打印并记录日志"""
        print(message)
        with open(self.log_file, "a") as log:
            log.write(message + "\n")

    def encode_image(self, image):
        """使用 ViT 提取图像特征"""
        image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(image)
        return feature.cpu().numpy()

    def process_dataset(self):
        """遍历数据集并编码所有图像"""
        dataset = ImageFolder(root=self.dataset_path, transform=self.transform)
        self.log(f"发现 {len(dataset.imgs)} 张图片，开始编码...")

        for img_path, class_idx in tqdm(dataset.imgs, desc="编码图像", unit="img"):
            image = dataset.loader(img_path)  # 读取图像
            image = self.transform(image)  # 预处理
            feature = self.encode_image(image)  # 提取特征

            # 存储数据
            self.features.append(feature.flatten())
            self.metadata.append({
                "image_path": img_path,
                "category": dataset.classes[class_idx]  # 记录类别名称
            })

        # 保存特征向量和元数据
        np.save(os.path.join(self.output_path, "image_features.npy"), np.array(self.features))
        with open(os.path.join(self.output_path, "image_metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)

        self.log(f"编码完成！已保存 {len(self.features)} 条数据。")


if __name__ == "__main__":
    dataset_path = "coco_dataset/coco_cropped_parts"  # 数据集路径
    output_root = "coco_dataset/knowledge_bases_coco_crop"  # 存放编码后的数据
    encoder = ImageEncoder(dataset_path, output_root, model_name="vit_base_patch16_224", device="cuda")
    encoder.process_dataset()
