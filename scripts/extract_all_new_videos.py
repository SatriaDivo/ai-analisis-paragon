"""
Script Ekstraksi Otomatis Seluruh Dataset Video Baru:
1. dataset_new_video_09-11-2026 (58 Video MP4) -> dataset_video_frames_09-11-2026
2. dataset_new_video_13-11-2026 (43 Video MP4) -> dataset_video_frames_13-11-2026
Resolusi Penuh 1080p, Step=5 (Sampling 10 FPS dari 50 FPS), ThreadPool Paralel
"""

import time
from pathlib import Path
from extract_video_dataset import process_all_videos

def main():
    ROOT = Path("D:/Tugas Kuliah/Semester 6/MBKM - Magang/Paragon Spray/AI Analysis")
    DATASET_DIR = ROOT / "dataset"
    
    # 1. Dataset 09-11-2026
    vids_09 = DATASET_DIR / "dataset_new_video_09-11-2026"
    out_09 = DATASET_DIR / "dataset_video_frames_09-11-2026"
    
    # 2. Dataset 13-11-2026
    vids_13 = DATASET_DIR / "dataset_new_video_13-11-2026"
    out_13 = DATASET_DIR / "dataset_video_frames_13-11-2026"
    
    print("=" * 80)
    print("EKSTRAKSI BATCH KONSOLIDASI SELURUH 101 VIDEO STUDIO BARU")
    print("=" * 80)
    
    t0 = time.time()
    
    print("\n[PROSES 1/2] Mengekstrak 58 Video dari dataset_new_video_09-11-2026...")
    cnt_09 = process_all_videos(
        video_dir=vids_09,
        output_dir=out_09,
        step=5,
        quality=95,
        crop_height=None,
        overwrite=False,
        max_workers=6
    )
    
    print("\n[PROSES 2/2] Mengekstrak 43 Video dari dataset_new_video_13-11-2026...")
    cnt_13 = process_all_videos(
        video_dir=vids_13,
        output_dir=out_13,
        step=5,
        quality=95,
        crop_height=None,
        overwrite=False,
        max_workers=6
    )
    
    total_time = (time.time() - t0) / 60.0
    print("=" * 80)
    print(f"SEMUA EKSTRAKSI SELESAI DALAM {total_time:.2f} MENIT!")
    print(f"  - Frames 09-11-2026 : {cnt_09} frame")
    print(f"  - Frames 13-11-2026 : {cnt_13} frame")
    print(f"  - Total Frame Baru  : {cnt_09 + cnt_13} frame 1080p")
    print("=" * 80)

if __name__ == "__main__":
    main()
