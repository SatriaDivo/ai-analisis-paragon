"""
Script Pembuat Dataset YOLOv8-Seg Terpadu & Organik (122 Video Studio Paragon)
Versi 2.0 (Clean Natural Boundaries - Zero Artificial Cutoff)

Perbaikan Kritis:
1. Menghapus hardcode x_min = 120 dan segitiga sintetis apex yang memotong moncong botol.
2. Menggunakan Adaptive Noise-Floor Thresholding (mean + 2.5 * std pada dark cloth).
3. Masking sarung tangan nitrile biru berbasis warna HSV tanpa memotong uap di moncong botol.
4. Mendukung semprotan gas aerosol rapat maupun mist pump halus secara merata.
"""

import os
import cv2
import json
import shutil
import random
import numpy as np
from pathlib import Path

ROOT = Path("D:/Tugas Kuliah/Semester 6/MBKM - Magang/Paragon Spray/AI Analysis")
os.chdir(ROOT)

DATASET_DIR = ROOT / "dataset"
D1_DIR = DATASET_DIR / "dataset_video_frames_03-11-2026"
D2_DIR = DATASET_DIR / "dataset_video_frames_09-11-2026"
D3_DIR = DATASET_DIR / "dataset_video_frames_13-11-2026"
TARGET_YOLO = DATASET_DIR / "dataset_yolo_clean"


def extract_plume_polygon(base_bgr, act_bgr, v_name=""):
    """
    Mengekstraksi poligon uap fluida semprotan secara alami dan organik.
    Bebas potongan garis lurus vertikal (hard knife cut) atau batas buatan x_min.
    """
    if act_bgr.shape != base_bgr.shape:
        act_bgr = cv2.resize(act_bgr, (base_bgr.shape[1], base_bgr.shape[0]))
    h, w = act_bgr.shape[:2]
    
    gray_b = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.cvtColor(act_bgr, cv2.COLOR_BGR2GRAY)
    
    # Kompensasi pencahayaan global (area kain pojok kanan atas)
    bg_ref_b = np.median(gray_b[:150, -300:])
    bg_ref_a = np.median(gray_a[:150, -300:])
    offset = bg_ref_a - bg_ref_b
    gray_a_adj = np.clip(gray_a.astype(np.float32) - offset, 0, 255).astype(np.uint8)
    
    diff = cv2.absdiff(gray_a_adj, gray_b)
    diff_blur = cv2.GaussianBlur(diff, (7, 7), 0)
    
    # Adaptive noise-floor thresholding
    bg_noise = np.std(diff_blur[:150, -300:])
    bg_mean = np.mean(diff_blur[:150, -300:])
    thresh_val = max(6.0, min(14.0, bg_mean + 2.5 * bg_noise))
    
    _, fluid_bin = cv2.threshold(diff_blur, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Hapus sarung tangan nitrile (biru/cyan/violet)
    hsv = cv2.cvtColor(act_bgr, cv2.COLOR_BGR2HSV)
    glove = cv2.inRange(hsv, (85, 30, 30), (140, 255, 255))
    glove_dil = cv2.dilate(glove, np.ones((11, 11), np.uint8))
    fluid_bin[glove_dil > 0] = 0
    
    # Filter warna netral: partikel semprotan memiliki R ~= G ~= B
    # Singkirkan artefak berwarna tinggi (seperti stiker botol atau pantulan warna)
    b_ch, g_ch, r_ch = cv2.split(act_bgr.astype(np.int16))
    color_diff = np.maximum(np.abs(b_ch - g_ch), np.abs(g_ch - r_ch))
    is_colored = color_diff > 40
    fluid_bin[is_colored] = 0
    
    # Nolkan hanya area lantai/meja ekstrem bawah
    fluid_bin[980:, :] = 0
    
    # Morphological closing untuk menghubungkan partikel droplet halus
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(fluid_bin, cv2.MORPH_CLOSE, k)
    
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if cv2.contourArea(c) > 1500]
    if not valid:
        return ""
        
    c_main = max(valid, key=cv2.contourArea)
    epsilon = 0.0018 * cv2.arcLength(c_main, True)
    approx = cv2.approxPolyDP(c_main, epsilon, True)
    if len(approx) < 3:
        approx = c_main
        
    pts = approx.reshape(-1, 2)
    norm_pts = []
    for x, y in pts:
        nx = np.clip(x / w, 0.0, 1.0)
        ny = np.clip(y / h, 0.0, 1.0)
        norm_pts.extend([f"{nx:.6f}", f"{ny:.6f}"])
        
    return "0 " + " ".join(norm_pts) + "\n"


def main():
    print("=" * 80)
    print("PEMBUATAN DATASET YOLOV8-SEG TERPADU v2.0 (122 VIDEO STUDIO PARAGON)")
    print("Mode: Natural Continuous Plumes (Tanpa Pemotongan 120px / Bebas Pisau Buatan)")
    print("=" * 80)

    # Bersihkan folder target agar dataset 100% fresh
    if TARGET_YOLO.exists():
        shutil.rmtree(TARGET_YOLO)

    for split in ["train", "val"]:
        (TARGET_YOLO / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_YOLO / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Kumpulkan semua folder video dari ketiga sumber
    vids_d1 = sorted([(f, "D1_03-11") for f in D1_DIR.glob("*") if f.is_dir()])
    vids_d2 = sorted([(f, "D2_09-11") for f in D2_DIR.glob("*") if f.is_dir()])
    vids_d3 = sorted([(f, "D3_13-11") for f in D3_DIR.glob("*") if f.is_dir()])

    all_vids = vids_d1 + vids_d2 + vids_d3
    print(f"Total video terdeteksi: {len(all_vids)} video ({len(vids_d1)} dari 03-11, {len(vids_d2)} dari 09-11, {len(vids_d3)} dari 13-11)")

    # Pembagian Video-Level (Anti Data Leakage: 80% train, 20% val)
    random.seed(42)
    random.shuffle(all_vids)

    val_count = max(8, int(len(all_vids) * 0.20))
    val_vids = set([v[0].name for v in all_vids[:val_count]])
    train_vids = set([v[0].name for v in all_vids[val_count:]])

    print(f"Video Train: {len(train_vids)} video")
    print(f"Video Val  : {len(val_vids)} video ({val_count} video)")
    print("-" * 80)

    total_train_imgs = 0
    total_val_imgs = 0
    total_neg_samples = 0
    total_pos_samples = 0

    print("Memulai ekstraksi anotasi ground-truth organik per video...")
    for idx, (v_path, s_type) in enumerate(all_vids, 1):
        v_name = v_path.name
        split = "val" if v_name in val_vids else "train"
        
        all_imgs = sorted(list(v_path.glob("*.jpg")))
        if len(all_imgs) < 5:
            continue
            
        base_img = cv2.imread(str(all_imgs[0]))
        if base_img is None:
            continue
            
        # 1. Sampel Negatif (t=00.00s & t=00.10s) -> label kosong (mengajari AI latar belakang bersih)
        for neg_idx in [0, min(1, len(all_imgs)-1)]:
            neg_p = all_imgs[neg_idx]
            neg_img = cv2.imread(str(neg_p))
            if neg_img is not None and neg_idx > 0:
                diff_check = cv2.absdiff(neg_img, base_img)
                if np.mean(diff_check) > 10.0:
                    continue  # Lewati jika sudah ada uap semprotan

            dst_img = TARGET_YOLO / "images" / split / f"{v_name}_{neg_p.name}"
            dst_lbl = TARGET_YOLO / "labels" / split / f"{v_name}_{neg_p.stem}.txt"
            
            shutil.copy2(neg_p, dst_img)
            dst_lbl.write_text("")  # Label kosong (Background sample)
            total_neg_samples += 1
            if split == "train":
                total_train_imgs += 1
            else:
                total_val_imgs += 1
                
        # 2. Sampel Positif Aktif
        active_candidates = all_imgs[5:]
        step = max(1, len(active_candidates) // 25)
        sampled_active = active_candidates[::step][:30]
        
        pos_count_vid = 0
        for act_p in sampled_active:
            act_img = cv2.imread(str(act_p))
            if act_img is None:
                continue
                
            yolo_line = extract_plume_polygon(base_img, act_img, v_name=v_name)
            if yolo_line:  # Hanya simpan jika cairan aktif keluar
                dst_img = TARGET_YOLO / "images" / split / f"{v_name}_{act_p.name}"
                dst_lbl = TARGET_YOLO / "labels" / split / f"{v_name}_{act_p.stem}.txt"
                
                shutil.copy2(act_p, dst_img)
                dst_lbl.write_text(yolo_line)
                pos_count_vid += 1
                total_pos_samples += 1
                if split == "train":
                    total_train_imgs += 1
                else:
                    total_val_imgs += 1
                    
        if idx % 10 == 0 or idx == len(all_vids):
            print(f"[{idx:3d}/{len(all_vids):3d}] Diproses: {v_name:<38} | {s_type} -> {pos_count_vid:2d} Pos ({split.upper()})")

    # Buat file data.yaml resmi
    data_yaml_content = f"""path: {TARGET_YOLO.as_posix()}
train: images/train
val: images/val

names:
  0: spray_plume
"""

    yaml_path = TARGET_YOLO / "data.yaml"
    yaml_path.write_text(data_yaml_content)

    print("=" * 80)
    print("DATASET YOLOV8-SEG TERPADU v2.0 SELESAI DIBANGUN!")
    print(f"  - Total Citra Train : {total_train_imgs} citra")
    print(f"  - Total Citra Val   : {total_val_imgs} citra")
    print(f"  - Sampel Positif    : {total_pos_samples} frame cairan")
    print(f"  - Sampel Negatif    : {total_neg_samples} frame latar/botol kosong")
    print(f"  - File Konfigurasi  : {yaml_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
