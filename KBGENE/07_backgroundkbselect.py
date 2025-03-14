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
        """构建知识库，筛选具有代表性的图片"""
        self.log(f"发现 {len(self.features)} 张图片，开始筛选...")

        selected_features = []
        selected_images = []

        for i, feature in tqdm(enumerate(self.features), total=len(self.features), desc="🔍 筛选图片", unit="img"):
            if selected_features:
                similarities = cosine_similarity([feature], selected_features)
                max_similarity = similarities.max()
            else:
                max_similarity = 0  # 第一张图片直接加入

            if max_similarity < self.threshold:
                selected_features.append(feature)
                selected_images.append(self.metadata[i]["image_path"])

                # 复制图片到新知识库
                new_img_path = os.path.join(self.image_output_path, os.path.basename(self.metadata[i]["image_path"]))
                shutil.copy(self.metadata[i]["image_path"], new_img_path)

        # 构建 FAISS 索引
        if selected_features:
            selected_features = np.array(selected_features)
            faiss_index = faiss.IndexFlatL2(selected_features.shape[1])
            faiss_index.add(selected_features)
            faiss.write_index(faiss_index, os.path.join(self.faiss_output_path, "knowledge_base.index"))

            # 记录索引
            index_data = {"selected_images": selected_images}
            with open(os.path.join(self.output_path, "image_index.json"), "w") as f:
                json.dump(index_data, f, indent=4)

            self.log(f"知识库构建完成，共 {len(selected_features)} 张图片")

        else:
            self.log("没有符合条件的图片！")


if __name__ == "__main__":
    builder = KnowledgeBaseBuilder(
        "coco_dataset/knowledge_bases_coco_crop_background/encoded_features",
        "coco_dataset/knowledge_bases_coco_crop_background",
        threshold=0.3
    )
    builder.build()
