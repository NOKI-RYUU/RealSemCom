import os
import torch
import numpy as np
import json
import timm
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datetime import datetime


class ImageEncoder:
    def __init__(self, dataset_path, output_root, model_name="vit_base_patch16_224", device="cuda"):
        """
        初始化图像编码器
        :param dataset_path: 数据集路径（当前文件夹直接包含图片）
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

        self.metadata = []  # 存储图片路径信息
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
        image_files = [f for f in os.listdir(self.dataset_path) if os.path.isfile(os.path.join(self.dataset_path, f))]

        self.log(f"发现 {len(image_files)} 张图片，开始编码...")

        for img_name in tqdm(image_files, desc="编码图像", unit="img"):
            img_path = os.path.join(self.dataset_path, img_name)

            try:
                image = Image.open(img_path).convert("RGB")  # 读取图像并转换为RGB格式
                image = self.transform(image)  # 预处理
                feature = self.encode_image(image)  # 提取特征

                # 存储数据
                self.features.append(feature.flatten())
                self.metadata.append({
                    "image_path": img_path
                })
            except Exception as e:
                self.log(f"处理 {img_path} 时出错: {e}")

        # 保存特征向量和元数据
        np.save(os.path.join(self.output_path, "image_features.npy"), np.array(self.features))
        with open(os.path.join(self.output_path, "image_metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)

        self.log(f"编码完成！已保存 {len(self.features)} 条数据。")


if __name__ == "__main__":
    dataset_path = "coco_dataset/train2017"  # 数据集路径
    output_root = "coco_dataset/knowledge_bases_coco_crop_background"  # 存放编码后的数据
    encoder = ImageEncoder(dataset_path, output_root, model_name="vit_base_patch16_224", device="cuda")
    encoder.process_dataset()
