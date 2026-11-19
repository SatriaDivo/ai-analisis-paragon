import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

f_act = Path("dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")
img = cv2.imread(str(f_act))
h, w = img.shape[:2]

# Hand & spray analysis
# Let's convert to LAB and HSV
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_chan = lab[:, :, 0]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
v_chan = hsv[:, :, 2]
s_chan = hsv[:, :, 1]

# In C0490, hand is holding a purple/white glove
# The spray plume is white mist
# Let's inspect rows around y=550..650
spray_row = img[580:620, :, :]
print("Row 580-620 mean BGR along x:")
for x in range(200, 1600, 100):
    patch = img[580:620, x:x+50]
    print(f"  x={x}..{x+50}: mean BGR={patch.mean(axis=(0,1)).astype(int)}")

