from ultralytics import YOLO
import cv2
import json
import os
from tqdm import tqdm

model = YOLO('yolov8n.pt')

coco_dirs = {
    "train": "coco_dataset/train2017",
    "val": "coco_dataset/val2017"
}

output_json = "coco_dataset/coco_detections.json"

if os.path.exists(output_json):
    with open(output_json, "r") as f:
        all_detections = json.load(f)
else:
    all_detections = {}

for split, image_dir in coco_dirs.items():
    if not os.path.exists(image_dir):
        continue

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    image_files = [img for img in image_files if img not in all_detections]

    if not image_files:
        continue

    for idx, img_name in enumerate(tqdm(image_files, desc=f"Processing {split}", leave=True, dynamic_ncols=True)):
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        results = model(image, verbose=False)

        image_detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])

                image_detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "category_id": cls
                })

        all_detections[img_name] = image_detections

        if (idx + 1) % 10000 == 0:
            with open(output_json, "w") as f:
                json.dump(all_detections, f, indent=4)

with open(output_json, "w") as f:
    json.dump(all_detections, f, indent=4)

print(f"Processing complete. Results saved to {output_json}")
