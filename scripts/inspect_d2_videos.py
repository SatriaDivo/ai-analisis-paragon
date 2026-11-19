import cv2
from pathlib import Path

vids = sorted(list(Path("dataset/dataset_video_frames_2").glob("*")))
print(f"Total video folders in dataset_video_frames_2: {len(vids)}")
for v in vids:
    if v.is_dir():
        imgs = sorted(list(v.glob("*.jpg")))
        mid_idx = len(imgs) // 2
        f_mid = imgs[mid_idx] if imgs else None
        print(f"  {v.name:<10}: {len(imgs):4d} frames | Mid frame: {f_mid.name if f_mid else 'None'}")

