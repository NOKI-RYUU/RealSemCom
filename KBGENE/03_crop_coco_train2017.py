import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 文件路径
coco_json = "coco_dataset/coco_sorted_by_category.json"
image_dir = "coco_dataset/train2017"  # 或 train2017
output_dir = "coco_dataset/coco_cropped_parts"
checkpoint_file = "coco_dataset/coco_checkpoint.json"

# 读取已完成任务（支持断点续跑）
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        completed_tasks = set(map(tuple, json.load(f)))  # 读取已处理的 (image, bbox)
else:
    completed_tasks = set()

# 读取 COCO 数据
with open(coco_json, "r") as f:
    category_data = json.load(f)

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)


def process_image(category_id, img_name, bbox, idx):
    """裁剪图像，保留 bbox 内容，其他部分设为透明"""
    img_path = os.path.join(image_dir, img_name)

    # 确保相同图片的多个 bbox 都会处理
    task_id = (img_name, tuple(bbox))
    if task_id in completed_tasks:
        return task_id

    output_filename = f"{os.path.splitext(img_name)[0]}_{category_id}_{idx}.png"
    output_path = os.path.join(output_dir, f"category_{category_id}", output_filename)

    if not os.path.exists(img_path):
        return None

    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    # 确保通道为 4（BGRA）
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    x1, y1, x2, y2 = map(int, bbox)

    # 创建全透明图像
    transparent_image = np.zeros_like(image, dtype=np.uint8)

    # 复制 bbox 区域
    transparent_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    # 保存裁剪后的图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, transparent_image)

    return task_id


# 任务队列
tasks = []
for category_id, items in category_data.items():
    for idx, item in enumerate(items):
        img_name = item["image"]
        bbox = item["bbox"]
        tasks.append((category_id, img_name, bbox, idx))

# 多线程处理
MAX_WORKERS = 8  # 根据服务器 CPU 调整
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_image, *task): task for task in tasks}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
        result = future.result()
        if result:
            completed_tasks.add(result)

        # 定期保存进度
        if len(completed_tasks) % 10000 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump(list(completed_tasks), f)

# 最终保存
with open(checkpoint_file, "w") as f:
    json.dump(list(completed_tasks), f)

print("All images processed and saved.")
