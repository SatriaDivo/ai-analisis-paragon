import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

img_p = Path(r"dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")
img_bgr = cv2.imread(str(img_p))
h, w = img_bgr.shape[:2]

# Hand is on the left
# Let's find the nozzle:
# In C0490, nozzle is around x=150..250, y=550..620
# Let's inspect the differential with local background:
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# Estimate background from the right/top side
bg_level = np.median(blurred[100:300, 1000:1800])
diff = np.clip(blurred.astype(np.float32) - bg_level, 0, 255)

print(f"Background level: {bg_level}")
print(f"Diff in spray area (y=500..650, x=200..1200) max: {diff[500:650, 200:1200].max()}, mean: {diff[500:650, 200:1200].mean():.1f}")
print(f"Diff in top curtain area (y=50..300, x=500..1500) max: {diff[50:300, 500:1500].max()}, mean: {diff[50:300, 500:1500].mean():.1f}")

