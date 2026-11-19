import cv2
import numpy as np

img = cv2.imread(r"dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")
h, w = img.shape[:2]

# Let's save a downscaled thumbnail or inspect left side:
# In the user screenshot:
# Panel 1: Hand is on the far left!
# Let's check mean BGR of x=0..200, y=400..800
hand_zone = img[400:800, 0:300]
print("Hand zone (0..300) mean BGR:", hand_zone.mean(axis=(0,1)).astype(int))

# And the right side:
right_zone = img[400:800, 1500:1920]
print("Right zone (1500..1920) mean BGR:", right_zone.mean(axis=(0,1)).astype(int))

