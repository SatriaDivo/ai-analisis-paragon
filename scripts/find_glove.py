import cv2
import numpy as np

img = cv2.imread(r"dataset/dataset_video_frames_2/C0490/C0490_f00150_t03.00s.jpg")
h, w = img.shape[:2]

# In Panel 1:
# The glove is lavender/purple: B > 150, R > 150, G < 180 or HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Lavender/purple glove:
glove_mask = (img[:, :, 0] > 140) & (img[:, :, 2] > 140) & (img[:, :, 1] < 170)
y_g, x_g = np.where(glove_mask)
if len(x_g) > 0:
    print(f"Glove bounding box: x={x_g.min()}..{x_g.max()}, y={y_g.min()}..{y_g.max()}")
    # Nozzle is at the top-right of the glove
    nozzle_x_est = x_g.max()
    nozzle_y_est = y_g.min()
    print(f"Estimated nozzle position near: ({nozzle_x_est}, {nozzle_y_est})")

# Let's check the spray mist right in front of the nozzle
spray_x_start = x_g.max() if len(x_g) > 0 else 600
spray_y_center = (y_g.min() + y_g.max()) // 2 if len(y_g) > 0 else 600

# Sample the mist
mist_sample = img[y_g.min()-50:y_g.max()+50, spray_x_start:spray_x_start+400]
print("Mist sample shape:", mist_sample.shape)
print("Mist sample mean BGR:", mist_sample.mean(axis=(0,1)).astype(int))

