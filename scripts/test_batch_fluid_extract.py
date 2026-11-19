import cv2
import numpy as np
from pathlib import Path

def extract_fluid_polygon(base_img_path, act_img_path, thresh=8):
    im_base = cv2.imread(str(base_img_path))
    im_act = cv2.imread(str(act_img_path))
    if im_base is None or im_act is None:
        return None, 0
    h, w = im_act.shape[:2]
    
    gray_b = cv2.cvtColor(im_base, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.cvtColor(im_act, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray_a, gray_b)
    diff_blur = cv2.GaussianBlur(diff, (9, 9), 0)
    
    _, bin_mask = cv2.threshold(diff_blur, thresh, 255, cv2.THRESH_BINARY)
    
    # Hand/glove/bottle protection:
    # Most nozzles are at x between 70 and 260
    # Find bright glove pixels in left zone (x < 450)
    glove = (im_act[:, :, 0] > 140) & (im_act[:, :, 2] > 140)
    y_g, x_g = np.where(glove[:, :450])
    if len(x_g) > 0:
        hand_cut_x = min(350, int(x_g.max()) + 10)
    else:
        hand_cut_x = 120
        
    bin_mask[:, :hand_cut_x] = 0
    
    # Exclude table at bottom if y >= 820 in 1080p frame (for Dataset 1)
    # Check bottom table brightness
    bin_mask[820:, :] = 0
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k)
    
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if cv2.contourArea(c) > 1500]
    if not valid:
        return None, 0
        
    c_main = max(valid, key=cv2.contourArea)
    area = cv2.contourArea(c_main)
    
    # Convert contour to normalized YOLO string
    epsilon = 0.002 * cv2.arcLength(c_main, True)
    approx = cv2.approxPolyDP(c_main, epsilon, True)
    if len(approx) < 3:
        approx = c_main
    pts = approx.reshape(-1, 2)
    norm_pts = []
    for x, y in pts:
        nx = np.clip(x / w, 0.0, 1.0)
        ny = np.clip(y / h, 0.0, 1.0)
        norm_pts.extend([f"{nx:.6f}", f"{ny:.6f}"])
        
    yolo_line = "0 " + " ".join(norm_pts) + "\n"
    return yolo_line, area

# Test on 3 videos from Dataset 1
d1 = Path("dataset/dataset_video_frames")
for v in ["C0350-Gas-Gatsby-1", "C0352-Normal-Kahf-1", "C0365-Normal-Wardah_SprayOn_Serum-1.1"]:
    f_base = d1 / v / f"{v}_f00000_t00.00s.jpg"
    imgs = sorted(list((d1 / v).glob("*.jpg")))
    f_act = imgs[len(imgs)//2]
    yolo_l, area = extract_fluid_polygon(f_base, f_act)
    print(f"D1 {v}: plume area = {area:.0f} px, polygon points = {len(yolo_l.split())//2 if yolo_l else 0}")

# Test on 3 videos from Dataset 2
d2 = Path("dataset/dataset_video_frames_2")
for v in ["C0467", "C0471", "C0490"]:
    f_base = d2 / v / f"{v}_f00000_t00.00s.jpg"
    imgs = sorted(list((d2 / v).glob("*.jpg")))
    f_act = imgs[len(imgs)//2]
    yolo_l, area = extract_fluid_polygon(f_base, f_act)
    print(f"D2 {v}: plume area = {area:.0f} px, polygon points = {len(yolo_l.split())//2 if yolo_l else 0}")

