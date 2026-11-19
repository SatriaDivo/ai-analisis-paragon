import cv2
import numpy as np

img = cv2.imread(r"dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")
h, w = img.shape[:2]

# Hand is on the left
# Let's see the hand and spray coordinates
# In Panel 1:
# Hand is around x=0..600, y=500..750
# Spray plume goes from x~600 to x~1100, y~550..650
# Let's crop the spray area:
spray_crop = img[500:750, 450:1200]
print("Spray crop shape:", spray_crop.shape)
print("Spray crop mean BGR:", spray_crop.mean(axis=(0,1)).astype(int))

# Top area:
top_crop = img[50:350, 500:1500]
print("Top crop mean BGR:", top_crop.mean(axis=(0,1)).astype(int))
print("Top crop std:", top_crop.std(axis=(0,1)).astype(int))

