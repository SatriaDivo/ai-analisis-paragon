"""
testing/uji_model.py — Script Pengujian Model AI Spraybot Terbaru (YOLOv8s-Seg)
Riset MBKM Magang Industri PT Paragon Technology and Innovation & PT Makerindo Prima Solusi

Cara Penggunaan:
  1. Uji satu gambar tertentu:
     python testing/uji_model.py --image "dataset/dataset_video_frames_cropped/C0350-Gas-Gatsby-1/C0350-Gas-Gatsby-1_f00405_t08.10s.jpg"

  2. Uji seluruh gambar dalam satu folder:
     python testing/uji_model.py --folder "dataset/dataset_video_frames_cropped/C0352-Normal-Kahf-1" --max-images 10

  3. Opsi tambahan:
     --conf 0.20          (Threshold confidence, default: 0.15)
     --px-per-cm 20.46    (Skala kalibrasi piksel per cm lab Paragon)
     --model "path/model" (Default: weights/spraybot_seg_video_finetuned.pt)
"""

import os
import sys
import cv2
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# Resolusi path root proyek
FILE_DIR = Path(__file__).resolve().parent
TESTING_DIR = FILE_DIR.parent if FILE_DIR.name == "scripts" else FILE_DIR
ROOT_DIR = TESTING_DIR.parent
LOCAL_WEIGHTS = TESTING_DIR / "weights" / "spraybot_seg_finetuned.pt"
GLOBAL_WEIGHTS = ROOT_DIR / "weights" / "spraybot_seg_finetuned.pt"
DEFAULT_WEIGHTS = LOCAL_WEIGHTS if LOCAL_WEIGHTS.exists() else GLOBAL_WEIGHTS
DEFAULT_OUTPUT_DIR = TESTING_DIR / "output"

def load_spray_model(weights_path=None):
    if weights_path is None:
        weights_path = DEFAULT_WEIGHTS
    weights_path = Path(weights_path)
    if not weights_path.exists():
        fallback = ROOT_DIR / "runs" / "finetune_video_dataset" / "spraybot_video_finetuned_run" / "weights" / "best.pt"
        if fallback.exists():
            weights_path = fallback
        else:
            raise FileNotFoundError(f"Model bobot tidak ditemukan di: {weights_path}")
            
    print(f"[MODEL] Memuat bobot AI: {weights_path.name}")
    model = YOLO(str(weights_path))
    return model

def calculate_spray_geometry(pts, img_shape, px_per_cm=20.46):
    """Menghitung parameter geometri fisik semprotan aerosol."""
    h, w = img_shape[:2]
    pts = np.asarray(pts, dtype=np.float32)
    
    # 1. Posisi Titik Awal Nosel (titik paling kiri poligon)
    min_x_idx = np.argmin(pts[:, 0])
    nozzle_pt = (int(pts[min_x_idx, 0]), int(pts[min_x_idx, 1]))
    
    # 2. Penetrasi / Jangkauan Horizontal (Reach)
    max_x = np.max(pts[:, 0])
    reach_px = float(max_x - nozzle_pt[0])
    reach_cm = reach_px / px_per_cm
    
    # 3. Lebar Maksimum Vertikal (Max Width)
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    width_px = float(max_y - min_y)
    width_cm = width_px / px_per_cm
    
    # 4. Luas Area Poligon Semprotan
    area_px2 = float(cv2.contourArea(pts.astype(np.int32)))
    area_cm2 = area_px2 / (px_per_cm ** 2)
    
    # 5. Sudut Sebaran Kerucut (Cone Angle)
    # Dihitung dari titik nosel ke batas atas dan batas bawah semprotan pada 50% jarak jangkauan
    mid_x = nozzle_pt[0] + (reach_px * 0.5)
    near_mid = pts[np.abs(pts[:, 0] - mid_x) < (reach_px * 0.15)]
    if len(near_mid) >= 2:
        top_y = np.min(near_mid[:, 1])
        bot_y = np.max(near_mid[:, 1])
        angle_rad = 2 * np.arctan2((bot_y - top_y) / 2.0, (mid_x - nozzle_pt[0]))
        cone_angle_deg = float(np.degrees(angle_rad))
    else:
        cone_angle_deg = float(np.degrees(2 * np.arctan2(width_px / 2.0, max(1.0, reach_px))))
        
    return {
        "nozzle": nozzle_pt,
        "reach_px": round(reach_px, 1),
        "reach_cm": round(reach_cm, 2),
        "width_px": round(width_px, 1),
        "width_cm": round(width_cm, 2),
        "area_cm2": round(area_cm2, 2),
        "cone_angle_deg": round(cone_angle_deg, 1)
    }

def analyze_single_image(model, img_path, conf=0.15, px_per_cm=20.46, save_dir=None):
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"[ERROR] Gambar tidak ditemukan: {img_path}")
        return None
        
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[ERROR] Gagal membaca file citra: {img_path.name}")
        return None
        
    h, w = img_bgr.shape[:2]
    
    # Inferensi YOLOv8-Seg
    t_start = time.time()
    results = model.predict(source=img_bgr, conf=conf, device=0 if torch_has_cuda() else "cpu", verbose=False)
    t_infer = (time.time() - t_start) * 1000.0
    res = results[0]
    
    found_plume = res.masks is not None and len(res.masks) > 0
    
    # Kanvas Visualisasi Overlay
    vis = img_bgr.copy()
    metrics = {
        "filename": img_path.name,
        "path": str(img_path),
        "resolution": f"{w}x{h}",
        "detected": found_plume,
        "infer_time_ms": round(t_infer, 1),
        "conf": 0.0,
        "reach_cm": 0.0,
        "width_cm": 0.0,
        "area_cm2": 0.0,
        "cone_angle_deg": 0.0,
        "nozzle_coord": "(0, 0)"
    }
    
    if found_plume:
        # Pilih mask dengan confidence tertinggi
        best_idx = int(np.argmax(res.boxes.conf.cpu().numpy()))
        best_conf = float(res.boxes.conf[best_idx].cpu().numpy())
        pts = res.masks.xy[best_idx]
        
        geo = calculate_spray_geometry(pts, img_bgr.shape, px_per_cm)
        metrics.update({
            "conf": round(best_conf, 3),
            "reach_cm": geo["reach_cm"],
            "width_cm": geo["width_cm"],
            "area_cm2": geo["area_cm2"],
            "cone_angle_deg": geo["cone_angle_deg"],
            "nozzle_coord": f"({geo['nozzle'][0]}, {geo['nozzle'][1]})"
        })
        
        # Gambar Mask Hijau Transparan
        overlay = np.zeros_like(img_bgr)
        pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_int], (0, 255, 0))
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.40, 0)
        cv2.polylines(vis, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Bounding Box
        box = res.boxes.xyxy[best_idx].cpu().numpy().astype(int)
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        
        # Titik Nosel
        nx, ny = geo["nozzle"]
        cv2.circle(vis, (nx, ny), 7, (0, 0, 255), -1)
        cv2.putText(vis, f"Nozzle ({nx},{ny})", (max(10, nx - 40), max(25, ny - 12)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        
        # Header Info Metrik di Gambar
        info_lines = [
            f"Model: {Path(model.ckpt_path).name} | Conf: {best_conf:.1%}",
            f"Jangkauan (Reach): {geo['reach_cm']} cm ({geo['reach_px']} px)",
            f"Lebar Max (Width): {geo['width_cm']} cm ({geo['width_px']} px)",
            f"Sudut Sebaran: {geo['cone_angle_deg']} deg | Luas Plume: {geo['area_cm2']} cm2"
        ]
        
        card_w = 520
        card_h = len(info_lines) * 26 + 18
        card_x, card_y = w - card_w - 20, 20
        cv2.rectangle(vis, (card_x, card_y), (card_x + card_w, card_y + card_h), (15, 15, 15), -1)
        cv2.rectangle(vis, (card_x, card_y), (card_x + card_w, card_y + card_h), (0, 255, 0), 2)
        
        for li, txt in enumerate(info_lines):
            cv2.putText(vis, txt, (card_x + 14, card_y + 24 + li * 26), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # Tanda jika tidak ada semprotan (Sampel Negatif)
        cv2.putText(vis, "STATUS: TIDAK TERDETEKSI SEMPROTAN (SAMPEL NEGATIF)", (50, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
    # Simpan hasil visualisasi jika save_dir ditentukan
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"pred_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        metrics["output_image"] = str(out_path)
        
    return metrics, vis

def torch_has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def print_result_card(m):
    print("=" * 65)
    print(f"HASIL DETEKSI AI: {m['filename']}")
    print("-" * 65)
    print(f"  - Status Semprotan  : {'✅ TERDETEKSI (VALID PLUME)' if m['detected'] else '❌ TIDAK TERDETEKSI (SAMPEL NEGATIF)'}")
    if m['detected']:
        print(f"  - Confidence Score  : {m['conf']:.1%}")
        print(f"  - Jangkauan (Reach) : {m['reach_cm']} cm")
        print(f"  - Lebar Maksimum    : {m['width_cm']} cm")
        print(f"  - Luas Area Plume   : {m['area_cm2']} cm²")
        print(f"  - Sudut Semprotan   : {m['cone_angle_deg']}°")
        print(f"  - Koordinat Nosel   : {m['nozzle_coord']}")
    print(f"  - Waktu Inferensi   : {m['infer_time_ms']} ms")
    if "output_image" in m:
        print(f"  - Gambar Tersimpan  : {m['output_image']}")
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="Uji Model YOLOv8-Seg Semprotan Aerosol Paragon")
    parser.add_argument("--image", type=str, default=None, help="Path ke satu file gambar")
    parser.add_argument("--folder", type=str, default=None, help="Path ke folder berisi banyak gambar")
    parser.add_argument("--model", type=str, default=None, help="Path bobot model .pt kustom")
    parser.add_argument("--conf", type=float, default=0.15, help="Ambang batas confidence (default: 0.15)")
    parser.add_argument("--px-per-cm", type=float, default=20.46, help="Skala kalibrasi px/cm (default: 20.46)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Direktori penyimpanan hasil")
    parser.add_argument("--max-images", type=int, default=20, help="Batas maksimal gambar jika mode folder")
    parser.add_argument("--show", action="store_true", help="Tampilkan visualisasi di layar pop-up")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = load_spray_model(args.model)
    
    # 1. Mode Gambar Tunggal
    if args.image:
        m, vis = analyze_single_image(model, args.image, conf=args.conf, px_per_cm=args.px_per_cm, save_dir=output_dir)
        if m:
            print_result_card(m)
            if args.show:
                cv2.imshow(f"Hasil Uji - {m['filename']}", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
    # 2. Mode Batch Folder
    elif args.folder:
        folder_p = Path(args.folder)
        img_files = sorted(list(folder_p.glob("*.jpg")) + list(folder_p.glob("*.png")))[:args.max_images]
        print(f"\n[BATCH] Memproses {len(img_files)} citra dari folder: {folder_p}")
        
        all_metrics = []
        for i, img_p in enumerate(img_files, 1):
            print(f"  [{i:02d}/{len(img_files):02d}] Menguji: {img_p.name} ...")
            m, _ = analyze_single_image(model, img_p, conf=args.conf, px_per_cm=args.px_per_cm, save_dir=output_dir / "overlays")
            if m:
                all_metrics.append(m)
                
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            csv_path = output_dir / "laporan_uji_batch.csv"
            df.to_csv(csv_path, index=False)
            print("=" * 65)
            print(f"SELESAI! {len(all_metrics)} citra telah diuji.")
            print(f"Laporan CSV tersimpan di: {csv_path}")
            print(f"Gambar overlay tersimpan di: {output_dir / 'overlays'}")
            print("=" * 65)
            
    else:
        # Default jika dijalankan tanpa argumen: Jalankan sampel otomatis
        default_sample = ROOT_DIR / "dataset" / "dataset_video_frames_cropped" / "C0350-Gas-Gatsby-1" / "C0350-Gas-Gatsby-1_f00405_t08.10s.jpg"
        print(f"\n[INFO] Menjalankan uji sampel bawaan pada: {default_sample.name}")
        m, _ = analyze_single_image(model, default_sample, conf=args.conf, px_per_cm=args.px_per_cm, save_dir=output_dir)
        if m:
            print_result_card(m)

if __name__ == "__main__":
    main()

