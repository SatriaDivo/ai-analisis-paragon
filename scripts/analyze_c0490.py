import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

img_p = Path(r"dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")
img = cv2.imread(str(img_p))
h, w = img.shape[:2]

# Let's inspect the image properties
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Image: {img_p.name}, shape: {w}x{h}")
print(f"Min gray: {gray.min()}, Max gray: {gray.max()}, Mean: {gray.mean():.1f}, Median: {np.median(gray):.1f}")

# Background estimation from top area (e.g. y < 300)
bg_sample = gray[:300, 500:1500]
bg_median = np.median(bg_sample)
print(f"Background fabric median: {bg_median:.1f}")

# Differential
diff = np.clip(gray.astype(np.float32) - bg_median, 0, 255).astype(np.uint8)

# Check spray plume zone:
# Where is the high diff?
spray_zone = diff > 15
y_indices, x_indices = np.where(spray_zone)
if len(x_indices) > 0:
    print(f"Spray zone pixels with diff > 15: {len(x_indices)}")
    print(f"  x range: {x_indices.min()} to {x_indices.max()}")
    print(f"  y range: {y_indices.min()} to {y_indices.max()}")

