"""
Script Pelatihan Resmi AI Spraybot v2.0 (Clean Natural Fluid Fine-Tuning)
Dataset: dataset/dataset_yolo_clean (122 Video Studio Paragon)
Base Weights: weights/training_awal/spraybot_seg_video_finetuned.pt (Stable Baseline)
Hyperparameters Khusus Fluida Semprotan:
- mosaic = 0.0 (Matikan mosaic agar uap semprotan tidak terpotong 4 kuadran)
- erasing = 0.0 (Matikan random erasing agar tidak melubangi uap cairan)
- mixup = 0.0
- retina_masks = True (Gunakan resolusi penuh untuk kontur halus)
- lr0 = 0.001 (Learning rate stabil untuk transfer learning)
- lrf = 0.01
- fliplr = 0.0, flipud = 0.0 (Konsisten kiri ke kanan)
- imgsz = 1024, batch = 8, epochs = 30
"""

import os
import shutil
import time
import torch
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

def main():
    ROOT = Path("D:/Tugas Kuliah/Semester 6/MBKM - Magang/Paragon Spray/AI Analysis")
    os.chdir(ROOT)

    DATA_YAML = ROOT / "dataset" / "dataset_yolo_clean" / "data.yaml"
    
    # Gunakan bobot baseline yang stabil (Model Awal) sebagai titik tolak transfer learning
    BASE_MODEL = ROOT / "weights" / "training_awal" / "spraybot_seg_video_finetuned.pt"
    if not BASE_MODEL.exists():
        BASE_MODEL = ROOT / "weights" / "spraybot_seg_video_finetuned.pt"
    if not BASE_MODEL.exists():
        BASE_MODEL = ROOT / "weights" / "yolov8s-seg.pt"

    RUNS_DIR = ROOT / "runs" / "spraybot_clean_training"
    RUN_NAME = "spraybot_seg_finetune_122vids_clean"
    WEIGHTS_DIR = ROOT / "weights"
    TESTING_DIR = ROOT / "testing"
    WEB_MODELS_DIR = Path("D:/Tugas Kuliah/Semester 6/MBKM - Magang/Paragon Spray/Spray-Website/models/segmentation")

    print("=" * 80)
    print("PELATIHAN RESMI AI SPRAYBOT v2.0 — CLEAN FLUID FINE-TUNING (122 VIDEO STUDIO)")
    print("=" * 80)
    print(f"Data YAML          : {DATA_YAML}")
    print(f"Base Model Weights : {BASE_MODEL}")
    print(f"Output Run         : {RUNS_DIR / RUN_NAME}")
    print(f"CUDA Available     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device         : {torch.cuda.get_device_name(0)}")
        print(f"Dedicated VRAM     : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("-" * 80)

    # Inisialisasi Model
    model = YOLO(str(BASE_MODEL))

    EPOCHS = 30
    IMG_SIZE = 1024
    BATCH_SIZE = 8
    WORKERS = 0  # 0 untuk Windows multiprocessing safety

    print(f"Konfigurasi Hyperparameters Khusus Fluida:")
    print(f"  • Epochs         : {EPOCHS} (Patience: 12)")
    print(f"  • Image Size     : {IMG_SIZE}")
    print(f"  • Batch Size     : {BATCH_SIZE}")
    print(f"  • Learning Rate  : lr0=0.001, lrf=0.01 (Stable Transfer Learning)")
    print(f"  • Augmentations  : mosaic=0.0 (OFF), erasing=0.0 (OFF), mixup=0.0 (OFF)")
    print(f"  • Retina Masks   : True (Resolusi Penuh Kontur Halus)")
    print(f"  • Arah Semprot   : fliplr=0.0, flipud=0.0 (Kiri -> Kanan)")
    print("-" * 80)
    print("Memulai pelatihan model...")

    start_time = time.time()

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=2.0,
        mosaic=0.0,
        erasing=0.0,
        mixup=0.0,
        retina_masks=True,
        close_mosaic=0,
        patience=12,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        hsv_h=0.015,
        hsv_s=0.2,
        hsv_v=0.2,
        translate=0.05,
        scale=0.10,
        verbose=True
    )

    elapsed_min = (time.time() - start_time) / 60.0
    print("=" * 80)
    print(f"PELATIHAN SELESAI DALAM {elapsed_min:.2f} MENIT!")
    print("=" * 80)

    # Cetak ringkasan evaluasi kuantitatif dari results.csv
    res_csv = RUNS_DIR / RUN_NAME / "results.csv"
    if res_csv.exists():
        df = pd.read_csv(res_csv)
        df.columns = df.columns.str.strip()
        last_row = df.iloc[-1]
        
        map50_b = last_row.get("metrics/mAP50(B)", 0) * 100
        map50_95_b = last_row.get("metrics/mAP50-95(B)", 0) * 100
        prec_b = last_row.get("metrics/precision(B)", 0) * 100
        rec_b = last_row.get("metrics/recall(B)", 0) * 100
        
        map50_m = last_row.get("metrics/mAP50(M)", 0) * 100
        map50_95_m = last_row.get("metrics/mAP50-95(M)", 0) * 100
        prec_m = last_row.get("metrics/precision(M)", 0) * 100
        rec_m = last_row.get("metrics/recall(M)", 0) * 100

        print("\nHASIL KUANTITATIF EVALUASI CHECKPOINT AKHIR:")
        print("+" + "-"*40 + "+" + "-"*16 + "+")
        print(f"| {'METRIK EVALUASI':<38} | {'NILAI PERSEN':<14} |")
        print("+" + "-"*40 + "+" + "-"*16 + "+")
        print(f"| {'mAP50 (Box / Plume)':<38} | {map50_b:>12.2f}% |")
        print(f"| {'mAP50-95 (Box)':<38} | {map50_95_b:>12.2f}% |")
        print(f"| {'Precision (Box)':<38} | {prec_b:>12.2f}% |")
        print(f"| {'Recall (Box)':<38} | {rec_b:>12.2f}% |")
        print("+" + "-"*40 + "+" + "-"*16 + "+")
        print(f"| {'mAP50 (Mask / Segmentasi)':<38} | {map50_m:>12.2f}% |")
        print(f"| {'mAP50-95 (Mask)':<38} | {map50_95_m:>12.2f}% |")
        print(f"| {'Precision (Mask)':<38} | {prec_m:>12.2f}% |")
        print(f"| {'Recall (Mask)':<38} | {rec_m:>12.2f}% |")
        print("+" + "-"*40 + "+" + "-"*16 + "+")

    # Salin bobot model terbaik
    best_pt = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    if best_pt.exists():
        (WEIGHTS_DIR / "finetuned").mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, WEIGHTS_DIR / "finetuned" / "spraybot_seg_finetune_clean_best.pt")
        shutil.copy2(best_pt, WEIGHTS_DIR / "spraybot_seg_finetuned.pt")
        shutil.copy2(best_pt, TESTING_DIR / "weights" / "spraybot_seg_finetuned.pt")
        if WEB_MODELS_DIR.exists():
            shutil.copy2(best_pt, WEB_MODELS_DIR / "spraybot_seg_finetuned.pt")
            shutil.copy2(best_pt, WEB_MODELS_DIR / "spraybot_seg_video_finetuned.pt")
            
        print(f"\n[SUKSES] Bobot model terbaik disinkronkan ke seluruh direktori produksi!")
    else:
        print("\n[PERINGATAN] File best.pt tidak ditemukan!")

if __name__ == "__main__":
    main()
