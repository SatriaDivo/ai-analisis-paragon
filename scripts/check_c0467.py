import cv2
from pathlib import Path

d2 = Path("dataset/dataset_video_frames_2/C0467")
imgs = sorted(list(d2.glob("*.jpg")))
f0 = imgs[0]
f_mid = imgs[len(imgs)//2]

im0 = cv2.imread(str(f0))
im_mid = cv2.imread(str(f_mid))

print("f0 shape:", im0.shape, "mean BGR:", im0.mean(axis=(0,1)).astype(int))
print("f_mid shape:", im_mid.shape, "mean BGR:", im_mid.mean(axis=(0,1)).astype(int))

