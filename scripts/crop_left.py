import cv2
import numpy as np

img = cv2.imread(r"dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")
h, w = img.shape[:2]

# Let's crop a window around x=200..600, y=400..900 and find the can
# The Gatsby can is usually metallic/blue or black
# The hand is wearing a purple glove
# Let's save crops to disk to inspect exact coordinates
cv2.imwrite("testing/output/c0490_left_half.jpg", img[:, :1000])
print("Left half saved.")

