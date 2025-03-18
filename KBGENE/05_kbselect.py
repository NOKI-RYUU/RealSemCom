import os
import torch
import numpy as np
import faiss
import json
import shutil
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeBaseBuilder:
    def __init__(self, encoded_path, output_root, threshold=0.3):
        """
        初始化知识库构建类
        :param encoded_path: 预编码特征路径
        :param output_root: 存放知识库的根目录
        :param threshold: 语义相似度阈值
        """
        self.encoded_path = encoded_path
        self.output_path = os.path.join(output_root, f"knowledge_base_threshold_{threshold}")
        self.image_output_path = os.path.join(self.output_path, "images")
        self.faiss_output_path = os.path.join(self.output_path, "faiss_index")
        os.makedirs(self.image_output_path, exist_ok=True)
        os.makedirs(self.faiss_output_path, exist_ok=True)

        self.threshold = threshold

        # 读取已编码的特征
        self.features = np.load(os.path.join(self.encoded_path, "image_features.npy"))
        with open(os.path.join(self.encoded_path, "image_metadata.json"), "r") as f:
            self.metadata = json.load(f)

        # 组织类别数据
        self.category_data = {}
        for idx, data in enumerate(self.metadata):
            category = data["category"]
            if category not in self.category_data:
                self.category_data[category] = {"features": [], "images": []}
            self.category_data[category]["features"].append(self.features[idx])
            self.category_data[category]["images"].append(data["image_path"])

        self.log_file = os.path.join(self.output_path, "log.txt")
        with open(self.log_file, "w") as log:
            log.write(f"知识库构建开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"语义相似度阈值: {self.threshold}\n\n")

    def log(self, message):
        """打印并记录日志"""
        print(message)
        with open(self.log_file, "a") as log:
            log.write(message + "\n")

    def build(self):
        """构建类别内筛选的知识库"""
        self.log(f"发现 {len(self.category_data)} 个类别，开始筛选...")

        category_index = {}  # 存储类别映射

        for category, data in tqdm(self.category_data.items(), desc="处理类别", unit="category"):
            # 打印开始处理该类别的消息
            self.log(f"开始处理类别 {category}...")

            features = np.array(data["features"])  # 该类别所有特征
            images = data["images"]  # 该类别所有图片路径

            selected_features = []
            selected_images = []

            # 使用 tqdm 显示类别内部的处理进度
            for i, feature in tqdm(enumerate(features), desc=f"处理 {category} 内的图片", unit="image", leave=False):
                if selected_features:
                    similarities = cosine_similarity([feature], selected_features)
                    max_similarity = similarities.max()
                else:
                    max_similarity = 0  # 第一个样本直接加入

                if max_similarity < self.threshold:
                    selected_features.append(feature)
                    selected_images.append(images[i])

                    # 复制图片到新的分类文件夹
                    category_output_path = os.path.join(self.image_output_path, category)
                    os.makedirs(category_output_path, exist_ok=True)
                    new_img_path = os.path.join(category_output_path, os.path.basename(images[i]))
                    shutil.copy(images[i], new_img_path)

            # 构建 FAISS 索引
            if selected_features:
                selected_features = np.array(selected_features)
                faiss_index = faiss.IndexFlatL2(selected_features.shape[1])
                faiss_index.add(selected_features)
                faiss.write_index(faiss_index, os.path.join(self.faiss_output_path, f"{category}.index"))

                # 记录类别索引
                category_index[category] = selected_images

                self.log(f"{category} 知识库构建完成，共 {len(selected_features)} 张图片")

        # 保存类别索引
        category_index_path = os.path.join(self.output_path, "category_index.json")
        with open(category_index_path, "w") as f:
            json.dump(category_index, f, indent=4)

        self.log(f"类别索引已保存至: {category_index_path}")
        self.log(f"知识库构建完成！共处理 {len(self.category_data)} 个类别")


if __name__ == "__main__":
    builder = KnowledgeBaseBuilder("coco_dataset/knowledge_bases_coco_crop/encoded_features", "coco_dataset/knowledge_bases_coco_crop", threshold=0.3)
    builder.build()
