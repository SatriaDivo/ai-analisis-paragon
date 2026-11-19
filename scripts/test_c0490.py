import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

model_path = Path("testing/spraybot_seg_video_finetuned.pt")
img_path = Path(r"dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")

model = YOLO(str(model_path))
img = cv2.imread(str(img_path))
h, w = img.shape[:2]
print(f"Image shape: {w}x{h}")

# Test with different confidences
for conf in [0.10, 0.20, 0.30, 0.40, 0.50]:
    results = model.predict(img, conf=conf, verbose=False)[0]
    num_boxes = len(results.boxes) if results.boxes is not None else 0
    print(f"\n--- Conf threshold: {conf} -> Found {num_boxes} detections ---")
    if num_boxes > 0:
        for i in range(num_boxes):
            b_conf = float(results.boxes.conf[i])
            box = results.boxes.xyxy[i].cpu().numpy().astype(int)
            print(f"  Det {i}: conf={b_conf:.3f}, box={box.tolist()}")
            if results.masks is not None and len(results.masks) > i:
                pts = results.masks.xy[i]
                print(f"    Mask points: {len(pts)}, min_y={int(np.min(pts[:, 1]))}, max_y={int(np.max(pts[:, 1]))}, min_x={int(np.min(pts[:, 0]))}, max_x={int(np.max(pts[:, 0]))}")

