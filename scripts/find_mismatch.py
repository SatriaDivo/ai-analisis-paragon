from pathlib import Path
import cv2

all_vids = list(Path("dataset/dataset_video_frames").glob("*")) + list(Path("dataset/dataset_video_frames_2").glob("*"))

for v in all_vids:
    if v.is_dir():
        imgs = sorted(list(v.glob("*.jpg")))
        if len(imgs) >= 2:
            im0 = cv2.imread(str(imgs[0]))
            im_last = cv2.imread(str(imgs[-1]))
            if im0 is not None and im_last is not None:
                if im0.shape != im_last.shape:
                    print(f"MISMATCH in {v.name}: {imgs[0].name} is {im0.shape}, {imgs[-1].name} is {im_last.shape}")

