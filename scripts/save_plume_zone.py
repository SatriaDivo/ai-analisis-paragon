import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

f_act = Path("dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")
img = cv2.imread(str(f_act))
h, w = img.shape[:2]

# In C0490:
# Where is the nozzle?
# The hand holding the can is around x=500..650, y=550..700
# Let's inspect the hand and spray area in detail
crop = img[400:800, 400:1300]
cv2.imwrite("output/c0490_plume_zone.jpg", crop)
print("Plume zone saved.")

