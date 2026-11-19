import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_fluid_extraction(video_name, frame_name, dataset_dir):
    f_base = dataset_dir / video_name / f"{video_name}_f00000_t00.00s.jpg"
    f_act = dataset_dir / video_name / frame_name
    
    if not f_base.exists() or not f_act.exists():
        print(f"File not found for {video_name}")
        return
        
    im_base = cv2.imread(str(f_base))
    im_act = cv2.imread(str(f_act))
    h, w = im_act.shape[:2]
    
    gray_b = cv2.cvtColor(im_base, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.cvtColor(im_act, cv2.COLOR_BGR2GRAY)
    
    # Differential
    diff = cv2.absdiff(gray_a, gray_b)
    diff_blur = cv2.GaussianBlur(diff, (9, 9), 0)
    
    # Threshold for fluid
    _, bin_mask = cv2.threshold(diff_blur, 8, 255, cv2.THRESH_BINARY)
    
    # Glove / hand exclusion:
    # Find hand on the left side
    glove = (im_act[:, :, 0] > 140) & (im_act[:, :, 2] > 140)
    y_g, x_g = np.where(glove[:, :400])
    if len(x_g) > 0:
        hand_cut_x = int(x_g.max()) + 15
    else:
        hand_cut_x = 220
        
    bin_mask[:, :hand_cut_x] = 0
    
    # Morph close
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k)
    
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if cv2.contourArea(c) > 2000]
    
    print(f"[{video_name} - {frame_name}] Resolution: {w}x{h}, Valid fluid contours: {len(valid)}")
    if valid:
        c_main = max(valid, key=cv2.contourArea)
        area = cv2.contourArea(c_main)
        x, y, cw, ch = cv2.boundingRect(c_main)
        print(f"  Plume Area: {area:.0f} px, Bounding Box: x={x}, y={y}, w={cw}, h={ch}")

# Test Dataset 1 (1080p frame from dataset_video_frames)
test_fluid_extraction("C0350-Gas-Gatsby-1", "C0350-Gas-Gatsby-1_f00405_t08.10s.jpg", Path("dataset/dataset_video_frames"))

# Test Dataset 2 (1080p frame from dataset_video_frames_2)
test_fluid_extraction("C0490", "C0490_f00150_t03.00s.jpg", Path("dataset/dataset_video_frames_2"))

