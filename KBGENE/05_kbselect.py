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

        # 读取预编码特征
        self.features = np.load(os.path.join(self.encoded_path, "image_features.npy"))
        with open(os.path.join(self.encoded_path, "image_metadata.json"), "r") as f:
            self.metadata = json.load(f)

        self.category_index = {}
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
        """构建知识库"""
        self.log(f"处理 {len(self.metadata)} 张图片...")
        for idx, data in tqdm(enumerate(self.metadata), total=len(self.metadata), desc="🔍 筛选知识库", unit="img"):
            category = data["category"]
            img_path = data["image_path"]
            feature = self.features[idx]

            if category not in self.category_index:
                self.category_index[category] = {"images": [], "features": []}

            similarities = cosine_similarity([feature], self.category_index[category]["features"]) if self.category_index[category]["features"] else [0]
            if max(similarities) < self.threshold:
                self.category_index[category]["features"].append(feature)
                new_img_path = os.path.join(self.image_output_path, category, os.path.basename(img_path))
                os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
                shutil.copy(img_path, new_img_path)
                self.category_index[category]["images"].append(new_img_path)

        # 创建 FAISS 索引
        for category, data in self.category_index.items():
            faiss_index = faiss.IndexFlatL2(len(data["features"][0]))
            faiss_index.add(np.array(data["features"]))
            faiss.write_index(faiss_index, os.path.join(self.faiss_output_path, f"{category}.index"))

        self.log(f"知识库构建完成！")

if __name__ == "__main__":
    builder = KnowledgeBaseBuilder("./knowledge_bases_coco_crop/encoded_features", "./knowledge_bases_coco_crop", threshold=0.3)
    builder.build()
