import cv2

d2 = r"dataset/dataset_video_frames_2/C0467"
im0 = cv2.imread(f"{d2}/C0467_f00000_t00.00s.jpg")
im_mid = cv2.imread(f"{d2}/C0467_f02005_t40.10s.jpg")

cv2.imwrite("testing/output/c0467_f0.jpg", cv2.resize(im0, (960, 540)))
cv2.imwrite("testing/output/c0467_fmid.jpg", cv2.resize(im_mid, (960, 540)))
print("Saved C0467 preview images.")

