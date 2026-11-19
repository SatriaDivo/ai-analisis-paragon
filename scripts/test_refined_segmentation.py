import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

img_p = Path(r"dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")
img_bgr = cv2.imread(str(img_p))
h, w = img_bgr.shape[:2]

# Method: Local Contrast & High-Frequency Texture / Spectral Density of the Mist
# 1. Glove / Hand is on the left (x < 250, bright pixels):
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
glove_mask = (img_bgr[:, :, 0] > 140) & (img_bgr[:, :, 2] > 140) & (gray > 140)
y_g, x_g = np.where(glove_mask[:, :400])

if len(x_g) > 0:
    nozzle_x = int(x_g.max()) + 15
    # Find the top of the bottle near the glove:
    nozzle_y = int(np.percentile(y_g, 15)) - 10
else:
    nozzle_x = 210
    nozzle_y = 600

print(f"Estimated Nozzle: ({nozzle_x}, {nozzle_y})")

# 2. Extract Plume using Local Background Estimation around the spray corridor
# The spray shoots horizontally from nozzle_x to ~1200, within y in [nozzle_y - 120, nozzle_y + 120]
corridor_y1 = max(0, nozzle_y - 140)
corridor_y2 = min(h, nozzle_y + 140)
corridor_x1 = nozzle_x
corridor_x2 = min(w, nozzle_x + 1100)

corridor = gray[corridor_y1:corridor_y2, corridor_x1:corridor_x2]

# Estimate local background from slices just above and below the corridor
above_bg = gray[max(0, corridor_y1 - 80):corridor_y1, corridor_x1:corridor_x2]
below_bg = gray[corridor_y2:min(h, corridor_y2 + 80), corridor_x1:corridor_x2]

bg_ref = np.zeros_like(corridor, dtype=np.float32)
# Interpolate background vertically
for col in range(corridor.shape[1]):
    val_top = np.median(above_bg[:, col]) if above_bg.shape[0] > 0 else 75.0
    val_bot = np.median(below_bg[:, col]) if below_bg.shape[0] > 0 else 75.0
    bg_ref[:, col] = np.linspace(val_top, val_bot, corridor.shape[0])

diff_corridor = np.clip(corridor.astype(np.float32) - bg_ref, 0, 255)
blurred_diff = cv2.GaussianBlur(diff_corridor, (9, 9), 2)

# Adaptive thresholding for faint mist
thresh = max(4.0, float(np.percentile(blurred_diff, 82)))
print(f"Adaptive threshold in spray corridor: {thresh:.2f}")

binary_plume = (blurred_diff >= thresh).astype(np.uint8)

# Connect plume from nozzle
cv2.line(binary_plume, (0, nozzle_y - corridor_y1), (30, nozzle_y - corridor_y1), 1, 15)

# Morphological closing
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
closed = cv2.morphologyEx(binary_plume, cv2.MORPH_CLOSE, k)

# Connected components from nozzle
num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
full_mask = np.zeros((h, w), dtype=np.uint8)

noz_local_y = nozzle_y - corridor_y1
noz_local_x = 0

best_label = -1
max_area = 0
for i in range(1, num_l):
    left = stats[i, cv2.CC_STAT_LEFT]
    top = stats[i, cv2.CC_STAT_TOP]
    height = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    if left < 40 and area > 1000:
        if area > max_area:
            max_area = area
            best_label = i

if best_label > 0:
    corridor_mask = (labels == best_label).astype(np.uint8) * 255
    # Smooth
    corridor_mask = cv2.GaussianBlur(corridor_mask, (15, 15), 0)
    corridor_mask = (corridor_mask > 80).astype(np.uint8) * 255
    full_mask[corridor_y1:corridor_y2, corridor_x1:corridor_x2] = corridor_mask

# Draw visualization
vis = img_bgr.copy()
overlay = np.zeros_like(img_bgr)
overlay[full_mask > 0] = (0, 255, 0)
vis = cv2.addWeighted(vis, 1.0, overlay, 0.45, 0)

cnts, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if cnts:
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    
cv2.circle(vis, (nozzle_x, nozzle_y), 7, (0, 0, 255), -1)
cv2.putText(vis, f"Nosel ({nozzle_x},{nozzle_y})", (nozzle_x - 30, nozzle_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

out_test = Path("testing/output/c0490_refined_plume.jpg")
cv2.imwrite(str(out_test), vis)
print(f"Refined segmentation saved to: {out_test}")

