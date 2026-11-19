"""
Script Ekstraksi Frame Video Semprotan Aerosol & Liquid Mist (High Performance with cap.grab() & ThreadPool)
Dataset Proyek AI Spraybot MBKM Magang PT Paragon Technology and Innovation
"""

import os
import cv2
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_frames_from_video(video_path, output_dir, step=5, quality=95, crop_height=None, overwrite=False):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    video_stem = video_path.stem
    
    target_dir = output_dir / video_stem
    target_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka video: {video_path.name}")
        return 0
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected_frames = (total_frames + step - 1) // step
    
    # Cek apakah sudah pernah diekstrak lengkap dengan resolusi yang sesuai
    existing_files = sorted(list(target_dir.glob("*.jpg")))
    if not overwrite and len(existing_files) >= expected_frames - 2:
        # Cek resolusi sampel apakah sesuai
        sample = cv2.imread(str(existing_files[0]))
        if sample is not None:
            sh = sample.shape[0]
            if (crop_height is None and sh >= 1080) or (crop_height is not None and sh == crop_height):
                print(f"[SKIP] {video_path.name:<40} sudah lengkap ({len(existing_files)} frames, h={sh}px)")
                cap.release()
                return len(existing_files)
        
    saved_count = 0
    frame_idx = 0
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    # Gunakan cap.grab() yang 5x lebih cepat karena tidak mendecode piksel frame yang dilewati
    while True:
        if frame_idx % step == 0:
            ret, frame = cap.read()
            if not ret:
                break
            if crop_height is not None and frame.shape[0] > crop_height:
                frame = frame[:crop_height, :]
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_s = timestamp_ms / 1000.0
            out_filename = target_dir / f"{video_stem}_f{frame_idx:05d}_t{timestamp_s:05.2f}s.jpg"
            cv2.imwrite(str(out_filename), frame, encode_param)
            saved_count += 1
        else:
            grabbed = cap.grab()
            if not grabbed:
                break
            
        frame_idx += 1
        
    cap.release()
    h_info = f"1920x{crop_height}" if crop_height else "1920x1080 (Asli)"
    print(f"[OK] {video_path.name:<40} -> {saved_count} frame selesai [{h_info}]")
    return saved_count

def process_all_videos(video_dir, output_dir, step=5, quality=95, crop_height=None, overwrite=False, max_workers=6):
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dedup case-insensitive file list
    video_files_dict = {f.name.lower(): f for f in video_dir.iterdir() if f.is_file() and f.suffix.lower() == ".mp4"}
    video_files = sorted(list(video_files_dict.values()), key=lambda x: x.name)
    
    print("=" * 80)
    print("EKSTRAKSI FRAME DATASET VIDEO SPRAYBOT (PARALEL THREADPOOL)")
    print("=" * 80)
    print(f"Direktori Video  : {video_dir}")
    print(f"Jumlah Video     : {len(video_files)} file")
    print(f"Sampling Step    : {step} (Dari 50 FPS -> {50.0/step:.1f} FPS diekstrak)")
    if crop_height:
        print(f"Pemotongan Tinggi: 0 s/d {crop_height} px (Resolusi: 1920 x {crop_height})")
    else:
        print(f"Pemotongan Tinggi: NONE (Resolusi Asli Penuh)")
    print(f"Target Output    : {output_dir}")
    print(f"Workers (Thread) : {max_workers}")
    print(f"Mode Overwrite   : {overwrite}")
    print("-" * 80)
    
    start_time = time.time()
    total_extracted = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_frames_from_video, vf, output_dir, step, quality, crop_height, overwrite): vf
            for vf in video_files
        }
        for fut in as_completed(futures):
            vf = futures[fut]
            try:
                cnt = fut.result()
                total_extracted += cnt
            except Exception as e:
                print(f"[ERROR] {vf.name}: {e}")
            
    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"SELESAI! Total {total_extracted} frame siap dalam {elapsed:.2f} detik.")
    print("=" * 80)
    return total_extracted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstraksi Frame Video Semprotan")
    parser.add_argument("--video-dir", default=r"D:\Tugas Kuliah\Semester 6\MBKM - Magang\Paragon Spray\AI Analysis\dataset\dataset_new_video")
    parser.add_argument("--output-dir", default=r"D:\Tugas Kuliah\Semester 6\MBKM - Magang\Paragon Spray\AI Analysis\dataset\dataset_video_frames")
    parser.add_argument("--step", type=int, default=5, help="Sampling step (default: 5 untuk 10 FPS)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG Quality (1-100)")
    parser.add_argument("--crop-height", type=int, default=None, help="Tinggi potong y (misal: 800 untuk buang meja putih)")
    parser.add_argument("--overwrite", action="store_true", help="Paksa ekstrak ulang dan timpa frame yang ada")
    parser.add_argument("--workers", type=int, default=6, help="Jumlah thread paralel (default: 6)")
    
    args = parser.parse_args()
    process_all_videos(args.video_dir, args.output_dir, args.step, args.quality, args.crop_height, args.overwrite, args.workers)
