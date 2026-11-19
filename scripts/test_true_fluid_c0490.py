import cv2
import numpy as np
from pathlib import Path

f_base = Path("dataset/dataset_video_frames_2/C0490/C0490_f00000_t00.00s.jpg")
f_act = Path("dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")

im_base = cv2.imread(str(f_base))
im_act = cv2.imread(str(f_act))
h, w = im_act.shape[:2]

# Compute background differential
gray_base = cv2.cvtColor(im_base, cv2.COLOR_BGR2GRAY)
gray_act = cv2.cvtColor(im_act, cv2.COLOR_BGR2GRAY)

diff = cv2.absdiff(gray_act, gray_base)
blurred_diff = cv2.GaussianBlur(diff, (9, 9), 0)

# Threshold fluid activity
_, mask = cv2.threshold(blurred_diff, 8, 255, cv2.THRESH_BINARY)

# Find the hand position to protect bottle/hand from being segmented as fluid:
# Hand has movement too, but hand is at the left (x < 250 or detect glove color)
# In C0490, where is the nozzle?
# Let's find where the glove ends:
glove_pixels = (im_act[:, :, 0] > 120) & (im_act[:, :, 2] > 120) & (im_act[:, :, 1] < 150)
y_g, x_g = np.where(glove_pixels)
nozzle_x = int(x_g.max()) if len(x_g) > 0 else 220
nozzle_y = int(np.median(y_g)) if len(y_g) > 0 else 580

print(f"Detected Nozzle at: ({nozzle_x}, {nozzle_y})")

# Protect hand/bottle:
mask[:, :nozzle_x] = 0

# Morphological clean up
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_cnts = [c for c in cnts if cv2.contourArea(c) > 2000]

vis = im_act.copy()
if valid_cnts:
    c_main = max(valid_cnts, key=cv2.contourArea)
    overlay = np.zeros_like(im_act)
    cv2.drawContours(overlay, [c_main], -1, (0, 255, 0), -1)
    vis = cv2.addWeighted(vis, 1.0, overlay, 0.45, 0)
    cv2.drawContours(vis, [c_main], -1, (0, 255, 0), 2)
    cv2.circle(vis, (nozzle_x, nozzle_y), 8, (0, 0, 255), -1)
    cv2.putText(vis, f"Nozzle ({nozzle_x},{nozzle_y})", (nozzle_x - 30, nozzle_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    print("Contour area of fluid plume:", cv2.contourArea(c_main))

out_path = Path("testing/output/c0490_ground_truth_segmentation.jpg")
out_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_path), vis)
print(f"Saved true fluid segmentation to: {out_path}")

