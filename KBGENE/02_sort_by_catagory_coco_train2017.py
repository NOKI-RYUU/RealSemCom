import json

# 读取 JSON 数据
input_json = "coco_dataset/coco_detections.json"
output_json = "coco_dataset/coco_sorted_by_category.json"

with open(input_json, "r") as f:
    all_detections = json.load(f)

# 选取前 118287 项
selected_detections = dict(list(all_detections.items())[:118287])

# 按照 category_id 整理
category_dict = {}

for img_name, detections in selected_detections.items():
    for obj in detections:
        category_id = obj["category_id"]
        bbox = obj["bbox"]

        if category_id not in category_dict:
            category_dict[category_id] = []

        category_dict[category_id].append({"image": img_name, "bbox": bbox})

# 按 category_id 排序
sorted_category_dict = dict(sorted(category_dict.items()))

# 保存整理后的 JSON
with open(output_json, "w") as f:
    json.dump(sorted_category_dict, f, indent=4)

print(f"Processing complete. Results saved to {output_json}")
