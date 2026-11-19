import cv2
import numpy as np

img = cv2.imread(r"dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold high brightness (glove is white/light purple)
bright = gray > 180
y_b, x_b = np.where(bright)
print(f"Bright pixels (>180): {len(x_b)}")
if len(x_b) > 0:
    print(f"  x: {x_b.min()}..{x_b.max()}, y: {y_b.min()}..{y_b.max()}")
    print("  Median position of bright region:", int(np.median(x_b)), int(np.median(y_b)))

