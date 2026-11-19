import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

f_base = Path("dataset/dataset_video_frames_2/C0490/C0490_f00000_t00.00s.jpg")
f_act = Path("dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")

im_base = cv2.imread(str(f_base))
im_act = cv2.imread(str(f_act))

print("Base shape:", im_base.shape if im_base is not None else None)
print("Act shape:", im_act.shape if im_act is not None else None)

# Absolute difference in gray
gray_base = cv2.cvtColor(im_base, cv2.COLOR_BGR2GRAY)
gray_act = cv2.cvtColor(im_act, cv2.COLOR_BGR2GRAY)

diff = cv2.absdiff(gray_act, gray_base)
print("Diff max:", diff.max(), "mean:", diff.mean())

# Look at diff > 10
_, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
# Exclude the hand on the left (x < 650)
mask_fluid = mask.copy()
mask_fluid[:, :650] = 0

contours, _ = cv2.findContours(mask_fluid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Contours in fluid zone:", len(contours))
valid = [c for c in contours if cv2.contourArea(c) > 500]
print("Valid contours > 500px:", len(valid))
if valid:
    c_main = max(valid, key=cv2.contourArea)
    print("Main contour area:", cv2.contourArea(c_main))
    x, y, w, h = cv2.boundingRect(c_main)
    print(f"Bounding rect: x={x}, y={y}, w={w}, h={h}")

