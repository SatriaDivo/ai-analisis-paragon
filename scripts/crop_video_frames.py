"""
Script Pemotong Bagian Bawah Frame Video (Crop Bottom Meja/Tembok Putih)
Dataset Proyek AI Spraybot MBKM Magang Industri PT Paragon Technology and Innovation

Fungsi:
- Memotong baris piksel bawah (y >= crop_height) yang berisi meja laboratorium / tembok putih.
- Mempertahankan zona semprotan aerosol murni (y = 0 s/d crop_height) dengan latar belakang kain hitam pekat.
- Menggunakan multithreading cepat untuk memproses ribuan frame dalam hitungan detik.
"""

import os
import cv2
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_frame(src_path, dst_path, crop_height=800, quality=95):
    try:
        img = cv2.imread(str(src_path))
        if img is None:
            return False, f"Gagal membaca: {src_path.name}"
        
        h, w = img.shape[:2]
        if h <= crop_height:
            # Sudah terpotong atau lebih kecil
            cropped = img
        else:
            cropped = img[0:crop_height, :]
            
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), cropped, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return True, src_path.name
    except Exception as e:
        return False, f"Error {src_path.name}: {str(e)}"

def crop_dataset_frames(source_dir, target_dir, crop_height=800, quality=95, num_workers=8):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    print("=" * 80)
    print("AI SPRAYBOT — BATCH FRAME CROPPING (ELIMINASI TEMBOK/MEJA PUTIH)")
    print("=" * 80)
    print(f"Direktori Sumber : {source_dir}")
    print(f"Direktori Target : {target_dir}")
    print(f"Tinggi Potongan  : 0 s/d {crop_height} px (Resolusi Baru: 1920 x {crop_height})")
    print(f"Kualitas JPEG    : {quality}")
    print(f"Jumlah Thread    : {num_workers}")
    print("-" * 80)
    
    # Kumpulkan semua file gambar
    video_folders = sorted([f for f in source_dir.glob("*") if f.is_dir()])
    tasks = []
    
    for v_fld in video_folders:
        dst_fld = target_dir / v_fld.name
        dst_fld.mkdir(parents=True, exist_ok=True)
        
        for img_p in sorted(v_fld.glob("*.jpg")):
            dst_p = dst_fld / img_p.name
            tasks.append((img_p, dst_p))
            
    total_images = len(tasks)
    print(f"Ditemukan {len(video_folders)} folder video dengan total {total_images} frame gambar.")
    if total_images == 0:
        print("[PERINGATAN] Tidak ada file .jpg yang ditemukan!")
        return
        
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    # Eksekusi dengan ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_frame, src, dst, crop_height, quality): src 
            for src, dst in tasks
        }
        
        for i, fut in enumerate(as_completed(futures), 1):
            success, msg = fut.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"[GAGAL] {msg}")
                
            if i % 500 == 0 or i == total_images:
                elapsed = time.time() - start_time
                fps = i / elapsed if elapsed > 0 else 0
                pct = (i / total_images) * 100
                print(f"[{i:04d}/{total_images:04d}] ({pct:5.1f}%) Diproses | {fps:5.1f} frame/dtk | {elapsed:5.1f}s")
                
    total_elapsed = time.time() - start_time
    print("=" * 80)
    print(f"SELESAI DALAM {total_elapsed:.2f} DETIK!")
    print(f"  - Berhasil dipotong : {success_count} frame")
    print(f"  - Gagal             : {fail_count} frame")
    print(f"  - Lokasi Output     : {target_dir}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop area bawah meja/tembok putih pada dataset video frame.")
    parser.add_argument("--source-dir", default=r"dataset/dataset_video_frames", help="Folder sumber frame")
    parser.add_argument("--target-dir", default=r"dataset/dataset_video_frames_cropped", help="Folder target frame terpotong")
    parser.add_argument("--crop-height", type=int, default=800, help="Tinggi pemotongan piksel y (default: 800)")
    parser.add_argument("--quality", type=int, default=95, help="Kualitas JPEG (default: 95)")
    parser.add_argument("--workers", type=int, default=8, help="Jumlah worker thread (default: 8)")
    
    args = parser.parse_args()
    crop_dataset_frames(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        crop_height=args.crop_height,
        quality=args.quality,
        num_workers=args.workers
    )

