import cv2
import numpy as np

img = cv2.imread(r"dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")
h, w = img.shape[:2]

# Where are pixels with B > 150 or R > 150?
bright_mask = (img[:, :, 0] > 150) & (img[:, :, 2] > 150)
y_idx, x_idx = np.where(bright_mask)
print(f"Total bright pixels: {len(x_idx)}")
if len(x_idx) > 0:
    print(f"x range: {x_idx.min()}..{x_idx.max()}")
    print(f"y range: {y_idx.min()}..{y_idx.max()}")
    # Most common x
    print("x percentiles (10, 50, 90):", np.percentile(x_idx, [10, 50, 90]))
    print("y percentiles (10, 50, 90):", np.percentile(y_idx, [10, 50, 90]))

