import cv2
import numpy as np
from pathlib import Path

def inspect_video_timeline(v_folder):
    imgs = sorted(list(v_folder.glob("*.jpg")))
    if not imgs:
        return
    f0 = cv2.imread(str(imgs[0]))
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    
    # Sample every 10 frames
    diff_timeline = []
    for img_p in imgs[::10]:
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # Background exposure compensation
        bg0 = np.median(g0[:200, 1200:])
        bg_curr = np.median(g[:200, 1200:])
        adj_g = np.clip(g.astype(np.float32) - (bg_curr - bg0), 0, 255).astype(np.uint8)
        
        # Spray corridor diff (x=200..1200, y=400..800)
        corridor_diff = cv2.absdiff(adj_g[400:800, 200:1200], g0[400:800, 200:1200])
        val = np.mean(corridor_diff > 12)
        diff_timeline.append((img_p.name, val))
        
    # Find peak spray
    diff_timeline.sort(key=lambda x: x[1], reverse=True)
    best = diff_timeline[0] if diff_timeline else ("None", 0)
    print(f"{v_folder.name:<10}: Peak spray at {best[0]} (fluid ratio: {best[1]*100:.1f}%)")

d2 = Path("dataset/dataset_video_frames_2")
for v in sorted(list(d2.glob("*"))):
    if v.is_dir():
        inspect_video_timeline(v)

