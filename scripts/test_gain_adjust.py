import cv2
import numpy as np
from pathlib import Path

d2 = Path("dataset/dataset_video_frames_2/C0467")
imgs = sorted(list(d2.glob("*.jpg")))
f0 = imgs[0]
f_mid = imgs[len(imgs)//2]

im0 = cv2.imread(str(f0))
im_mid = cv2.imread(str(f_mid))

g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
g_mid = cv2.cvtColor(im_mid, cv2.COLOR_BGR2GRAY)

# Estimate global exposure difference from the top-right corner (free of spray)
bg0 = np.median(g0[:200, 1200:])
bg_mid = np.median(g_mid[:200, 1200:])
gain_offset = bg_mid - bg0
print(f"Global exposure offset: {gain_offset:.1f}")

# Adjust g_mid to match baseline exposure
g_mid_adjusted = np.clip(g_mid.astype(np.float32) - gain_offset, 0, 255).astype(np.uint8)

diff = cv2.absdiff(g_mid_adjusted, g0)
print(f"Diff without adjustment max: {cv2.absdiff(g_mid, g0).max()}, mean: {cv2.absdiff(g_mid, g0).mean():.1f}")
print(f"Diff with adjustment max: {diff.max()}, mean: {diff.mean():.1f}")

_, bin_mask = cv2.threshold(cv2.GaussianBlur(diff, (9, 9), 0), 10, 255, cv2.THRESH_BINARY)
bin_mask[:, :250] = 0
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k)
cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid = [c for c in cnts if cv2.contourArea(c) > 2000]
print("Valid contours with adjustment:", len(valid))
if valid:
    c = max(valid, key=cv2.contourArea)
    print("Contour area:", cv2.contourArea(c))
    x, y, w, h = cv2.boundingRect(c)
    print(f"Bounding rect: x={x}, y={y}, w={w}, h={h}")

