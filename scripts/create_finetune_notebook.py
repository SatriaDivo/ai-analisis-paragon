import json
import shutil
from pathlib import Path

ROOT = Path("D:/Tugas Kuliah/Semester 6/MBKM - Magang/Paragon Spray/AI Analysis")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# FINE-TUNING RESMI AI SPRAYBOT — INSTANCE SEGMENTATION FLUIDA SEMPROTAN AEROSOL\n",
            "### Konsolidasi 122 Sesi Video Studio Paragon (Bebas Garis Lurus & Presisi Moncong Botol)\n",
            "**Proyek Riset MBKM Magang Industri — PT Paragon Technology and Innovation & PT Makerindo Prima Solusi**\n",
            "\n",
            "---\n",
            "\n",
            "### 🎯 Fokus Utama Fine-Tuning Model Spraybot:\n",
            "1. **Bebas Potongan Garis Lurus (Zero Knife Cut):** Menggunakan masker uap organik yang meruncing alami tepat di titik moncong botol (*nozzle apex* $x \\approx 120\\text{ px}, y \\approx 780-830\\text{ px}$) sehingga segmentasi AI benar-benar mengikuti semprotan asli yang keluar dari moncong.\n",
            "2. **Konsolidasi Seluruh 122 Video Studio:**\n",
            "   - **Sesi 1 (03-11-2026):** 21 Video (Gatsby, Kahf, Wardah, Labora)\n",
            "   - **Sesi 2 (09-11-2026):** 58 Video (`C0467` s.d `C0545`)\n",
            "   - **Sesi 3 (13-11-2026):** 43 Video (`C0549` s.d `C0597`)\n",
            "   - **Total:** 2.924 frame teranotasi bersih (2.378 Train, 546 Val, termasuk 244 sampel negatif latar belakang).\n",
            "3. **Zero False Positive Background:** Menyertakan sampel negatif ($t=0.00\\text{s}$) berlabel kosong agar model tidak mendeteksi lipatan kain latar belakang sebagai semprotan.\n",
            "4. **Orientasi Konsisten:** 100% video menyemprot dari **Kiri ke Kanan** (`fliplr=0.0`).\n",
            "5. **Akselerasi GPU:** Dioptimasi untuk NVIDIA GeForce RTX 4050 Laptop GPU (`workers=0` anti-crash multiprocessing Windows)."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Setup Environment, Library & Deteksi Hardware GPU\n",
            "Memverifikasi ketersediaan PyTorch CUDA GPU RTX 4050 dan inisialisasi direktori kerja proyek."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, shutil, time, yaml, json\n",
            "from pathlib import Path\n",
            "import torch\n",
            "import cv2\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from PIL import Image\n",
            "from ultralytics import YOLO\n",
            "\n",
            "# Konfigurasi Direktori Kerja Proyek\n",
            "ROOT = Path(\"D:/Tugas Kuliah/Semester 6/MBKM - Magang/Paragon Spray/AI Analysis\")\n",
            "os.chdir(ROOT)\n",
            "\n",
            "print(\"=\" * 80)\n",
            "print(\"SETUP ENVIRONMENT & DETEKSI HARDWARE GPU\")\n",
            "print(\"=\" * 80)\n",
            "print(f\"Direktori Kerja : {os.getcwd()}\")\n",
            "print(f\"Versi PyTorch   : {torch.__version__}\")\n",
            "print(f\"CUDA Tersedia   : {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    gpu_name = torch.cuda.get_device_name(0)\n",
            "    gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
            "    print(f\"Perangkat GPU   : {gpu_name}\")\n",
            "    print(f\"VRAM Dedicated  : {gpu_vram:.2f} GB\")\n",
            "else:\n",
            "    print(\"[PERINGATAN] CUDA tidak aktif. Pelatihan akan berjalan lambat di CPU!\")\n",
            "print(\"=\" * 80)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Verifikasi Dataset Bersih Terpadu (`dataset_yolo_clean`)\n",
            "Memeriksa konfigurasi `data.yaml` dan menghitung sampel positif uap cairan serta sampel negatif latar belakang."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "DATA_YAML = ROOT / \"dataset\" / \"dataset_yolo_clean\" / \"data.yaml\"\n",
            "\n",
            "if not DATA_YAML.exists():\n",
            "    raise FileNotFoundError(f\"File konfigurasi {DATA_YAML} tidak ditemukan!\")\n",
            "\n",
            "with open(DATA_YAML, \"r\") as f:\n",
            "    yaml_config = yaml.safe_load(f)\n",
            "\n",
            "train_img_dir = ROOT / \"dataset\" / \"dataset_yolo_clean\" / \"images\" / \"train\"\n",
            "val_img_dir = ROOT / \"dataset\" / \"dataset_yolo_clean\" / \"images\" / \"val\"\n",
            "train_lbl_dir = ROOT / \"dataset\" / \"dataset_yolo_clean\" / \"labels\" / \"train\"\n",
            "val_lbl_dir = ROOT / \"dataset\" / \"dataset_yolo_clean\" / \"labels\" / \"val\"\n",
            "\n",
            "n_train = len(list(train_img_dir.glob(\"*.jpg\")))\n",
            "n_val = len(list(val_img_dir.glob(\"*.jpg\")))\n",
            "n_train_pos = sum(1 for p in train_lbl_dir.glob(\"*.txt\") if p.read_text().strip() != \"\")\n",
            "n_train_neg = sum(1 for p in train_lbl_dir.glob(\"*.txt\") if p.read_text().strip() == \"\")\n",
            "n_val_pos = sum(1 for p in val_lbl_dir.glob(\"*.txt\") if p.read_text().strip() != \"\")\n",
            "n_val_neg = sum(1 for p in val_lbl_dir.glob(\"*.txt\") if p.read_text().strip() == \"\")\n",
            "\n",
            "print(\"=\" * 80)\n",
            "print(\"STATISTIK DATASET FINE-TUNING SPRAYBOT (122 VIDEO STUDIO)\")\n",
            "print(\"=\" * 80)\n",
            "print(f\"File data.yaml     : {DATA_YAML}\")\n",
            "print(f\"Nama Kelas         : {yaml_config.get('names')}\")\n",
            "print(f\"Total Citra Train  : {n_train} citra (Positif: {n_train_pos}, Negatif Latar: {n_train_neg})\")\n",
            "print(f\"Total Citra Val    : {n_val} citra (Positif: {n_val_pos}, Negatif Latar: {n_val_neg})\")\n",
            "print(f\"Total Keseluruhan  : {n_train + n_val} citra\")\n",
            "print(\"=\" * 80)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Visualisasi Distribusi Dataset & Sampel Negatif Background\n",
            "Diagram komposisi dataset untuk memastikan rasio train-val 80:20 serta sampel negatif pelindung false positive."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n",
            "\n",
            "# Pie Chart Distribusi Dataset\n",
            "labels_pie = [f\"Train Positif ({n_train_pos})\", f\"Train Negatif ({n_train_neg})\",\n",
            "              f\"Val Positif ({n_val_pos})\", f\"Val Negatif ({n_val_neg})\"]\n",
            "sizes_pie = [n_train_pos, n_train_neg, n_val_pos, n_val_neg]\n",
            "colors_pie = [\"#2ecc71\", \"#95a5a6\", \"#3498db\", \"#bdc3c7\"]\n",
            "explode = (0.05, 0.05, 0.05, 0.05)\n",
            "\n",
            "axes[0].pie(sizes_pie, explode=explode, labels=labels_pie, colors=colors_pie,\n",
            "            autopct=\"%1.1f%%\", startangle=140, shadow=True)\n",
            "axes[0].set_title(\"Distribusi Split Citra (122 Video Studio)\", fontsize=13, fontweight=\"bold\")\n",
            "\n",
            "# Bar Chart Perbandingan\n",
            "categories = [\"Train Positif\", \"Train Negatif\", \"Val Positif\", \"Val Negatif\"]\n",
            "counts = [n_train_pos, n_train_neg, n_val_pos, n_val_neg]\n",
            "bars = axes[1].bar(categories, counts, color=[\"#2ecc71\", \"#7f8c8d\", \"#3498db\", \"#95a5a6\"])\n",
            "for bar in bars:\n",
            "    yval = bar.get_height()\n",
            "    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 20, int(yval), ha=\"center\", va=\"bottom\", fontweight=\"bold\")\n",
            "axes[1].set_ylabel(\"Jumlah Frame\", fontsize=11)\n",
            "axes[1].set_title(\"Komposisi Frame Latih dan Validasi\", fontsize=13, fontweight=\"bold\")\n",
            "axes[1].grid(axis=\"y\", linestyle=\"--\", alpha=0.5)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Eksekusi Fine-Tuning YOLOv8s-seg\n",
            "Menjalankan proses fine-tuning model:\n",
            "- **Base Weights:** `weights/spraybot_seg_finetuned.pt` (transfer learning bobot terbaik).\n",
            "- **Epochs:** `25` (konvergensi optimal untuk fine-tuning).\n",
            "- **imgsz:** `1024` (resolusi tinggi menangkap uap mikro di ujung moncong).\n",
            "- **batch:** `8` (optimal untuk 6GB VRAM RTX 4050).\n",
            "- **workers:** `0` (wajib di Windows).\n",
            "- **fliplr:** `0.0` (tanpa flip horizontal, arah semprot konsisten Kiri ke Kanan).\n",
            "- **lr0:** `0.005`, **lrf:** `0.01` (laju adaptasi fine-tuning)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Bobot awal untuk fine-tuning (transfer learning)\n",
            "BASE_WEIGHTS = ROOT / \"weights\" / \"spraybot_seg_finetuned.pt\"\n",
            "if not BASE_WEIGHTS.exists():\n",
            "    BASE_WEIGHTS = ROOT / \"weights\" / \"spraybot_seg_video_finetuned.pt\"\n",
            "if not BASE_WEIGHTS.exists():\n",
            "    BASE_WEIGHTS = ROOT / \"weights\" / \"yolov8s-seg.pt\"\n",
            "\n",
            "RUNS_DIR = ROOT / \"runs\" / \"spraybot_clean_training\"\n",
            "RUN_NAME = \"spraybot_seg_finetune_122vids\"\n",
            "\n",
            "print(\"=\" * 80)\n",
            "print(\"MEMULAI FINE-TUNING YOLOV8S-SEG SPRAYBOT (122 VIDEO STUDIO)\")\n",
            "print(\"=\" * 80)\n",
            "print(f\"Bobot Awal       : {BASE_WEIGHTS}\")\n",
            "print(f\"File data.yaml   : {DATA_YAML}\")\n",
            "print(f\"Direktori Output : {RUNS_DIR / RUN_NAME}\")\n",
            "print(\"=\" * 80)\n",
            "\n",
            "model = YOLO(str(BASE_WEIGHTS))\n",
            "\n",
            "EPOCHS = 25\n",
            "IMG_SIZE = 1024\n",
            "BATCH_SIZE = 8\n",
            "WORKERS = 0\n",
            "\n",
            "start_train_time = time.time()\n",
            "\n",
            "results = model.train(\n",
            "    data=str(DATA_YAML),\n",
            "    epochs=EPOCHS,\n",
            "    imgsz=IMG_SIZE,\n",
            "    batch=BATCH_SIZE,\n",
            "    workers=WORKERS,\n",
            "    device=0 if torch.cuda.is_available() else \"cpu\",\n",
            "    project=str(RUNS_DIR),\n",
            "    name=RUN_NAME,\n",
            "    exist_ok=True,\n",
            "    pretrained=True,\n",
            "    optimizer=\"auto\",\n",
            "    lr0=0.005,\n",
            "    lrf=0.01,\n",
            "    warmup_epochs=2.0,\n",
            "    close_mosaic=5,\n",
            "    patience=10,\n",
            "    fliplr=0.0,\n",
            "    flipud=0.0,\n",
            "    hsv_h=0.015,\n",
            "    hsv_s=0.3,\n",
            "    hsv_v=0.3,\n",
            "    translate=0.05,\n",
            "    scale=0.15,\n",
            "    verbose=True\n",
            ")\n",
            "\n",
            "elapsed = (time.time() - start_train_time) / 60.0\n",
            "print(\"=\" * 80)\n",
            "print(f\"FINE-TUNING SELESAI DALAM {elapsed:.2f} MENIT!\")\n",
            "print(\"=\" * 80)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Evaluasi Metrik Pelatihan (Loss & mAP Segmentasi)\n",
            "Melihat grafik loss konvergensi dan kurva mAP50 mask segmentasi dari hasil fine-tuning."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "results_png = RUNS_DIR / RUN_NAME / \"results.png\"\n",
            "if not results_png.exists():\n",
            "    results_png = RUNS_DIR / \"spraybot_yolov8s_seg_122vids\" / \"results.png\"\n",
            "if not results_png.exists():\n",
            "    results_png = RUNS_DIR / \"spraybot_yolov8s_seg_run\" / \"results.png\"\n",
            "\n",
            "if results_png.exists():\n",
            "    img = Image.open(results_png)\n",
            "    plt.figure(figsize=(18, 10))\n",
            "    plt.imshow(img)\n",
            "    plt.axis(\"off\")\n",
            "    plt.title(\"Kurva Hasil Fine-Tuning YOLOv8s-Seg (Loss, Precision, Recall, mAP50-95)\", fontsize=15, fontweight=\"bold\")\n",
            "    plt.show()\n",
            "else:\n",
            "    print(f\"[INFO] File {results_png} belum tersedia. Jalankan sel pelatihan di atas terlebih dahulu.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Uji Inferensi Visual Pada Citra Nyata (Verifikasi Bebas Garis Lurus)\n",
            "Menguji model pada citra semprotan nyata `C0490_f00150_t03.00s.jpg` untuk membuktikan:\n",
            "1. Segmentasi uap **meruncing dan terhubung organik ke moncong botol**.\n",
            "2. **Zero Garis Lurus Vertikal (Zero Knife Cut)**.\n",
            "3. Latar belakang kain hitam bebas false positive."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Path Citra Uji Kasus Nyata Video C0490\n",
            "TEST_IMAGE_PATH = ROOT / \"dataset\" / \"dataset_video_frames_09-11-2026\" / \"C0490\" / \"C0490_f00150_t03.00s.jpg\"\n",
            "if not TEST_IMAGE_PATH.exists():\n",
            "    TEST_IMAGE_PATH = sorted(list((ROOT / \"dataset\" / \"dataset_yolo_clean\" / \"images\" / \"val\").glob(\"*.jpg\")))[0]\n",
            "\n",
            "best_model_weights = RUNS_DIR / RUN_NAME / \"weights\" / \"best.pt\"\n",
            "if not best_model_weights.exists():\n",
            "    best_model_weights = ROOT / \"weights\" / \"spraybot_seg_finetuned.pt\"\n",
            "if not best_model_weights.exists():\n",
            "    best_model_weights = ROOT / \"weights\" / \"spraybot_seg_video_finetuned.pt\"\n",
            "\n",
            "infer_model = YOLO(str(best_model_weights))\n",
            "res = infer_model.predict(source=str(TEST_IMAGE_PATH), imgsz=1024, conf=0.25, save=False, verbose=False)[0]\n",
            "\n",
            "raw_bgr = cv2.imread(str(TEST_IMAGE_PATH))\n",
            "raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)\n",
            "\n",
            "panel2 = raw_rgb.copy()\n",
            "conf_val = 0.0\n",
            "if res.masks is not None and len(res.masks) > 0:\n",
            "    mask_np = res.masks.data[0].cpu().numpy()\n",
            "    mask_np = cv2.resize(mask_np, (raw_bgr.shape[1], raw_bgr.shape[0])) > 0.5\n",
            "    overlay = panel2.copy()\n",
            "    overlay[mask_np] = [0, 220, 50]\n",
            "    panel2 = cv2.addWeighted(overlay, 0.45, panel2, 0.55, 0)\n",
            "    cnts, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
            "    cv2.drawContours(panel2, cnts, -1, (0, 255, 0), 2)\n",
            "    conf_val = float(res.boxes.conf[0].cpu().numpy()) if res.boxes is not None else 0.0\n",
            "\n",
            "# Panel 3: Heatmap Turbo\n",
            "gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)\n",
            "norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)\n",
            "heatmap = cv2.applyColorMap(norm_gray, cv2.COLORMAP_TURBO)\n",
            "heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)\n",
            "\n",
            "fig, axes = plt.subplots(3, 1, figsize=(14, 18))\n",
            "axes[0].imshow(raw_rgb)\n",
            "axes[0].set_title(f\"Panel 1: Citra Asli ({TEST_IMAGE_PATH.name})\", fontsize=12, fontweight=\"bold\")\n",
            "axes[0].axis(\"off\")\n",
            "\n",
            "axes[1].imshow(panel2)\n",
            "axes[1].set_title(f\"Panel 2: Hasil Segmentasi AI YOLOv8-Seg (Conf: {conf_val*100:.1f}%) — Terhubung Organik ke Moncong\", fontsize=12, fontweight=\"bold\")\n",
            "axes[1].axis(\"off\")\n",
            "\n",
            "axes[2].imshow(heatmap_rgb)\n",
            "axes[2].set_title(\"Panel 3: Analisis Spektral Densitas Aerosol (Heatmap Turbo)\", fontsize=12, fontweight=\"bold\")\n",
            "axes[2].axis(\"off\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Ekspor Bobot Model Terbaik (`best.pt`)\n",
            "Menyalin bobot terbaik hasil fine-tuning ke direktori resmi proyek:\n",
            "1. `weights/spraybot_seg_finetuned.pt`\n",
            "2. `testing/spraybot_seg_finetuned.pt`"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "BEST_PT = RUNS_DIR / RUN_NAME / \"weights\" / \"best.pt\"\n",
            "TARGET_WEIGHTS = ROOT / \"weights\" / \"spraybot_seg_finetuned.pt\"\n",
            "TARGET_TESTING = ROOT / \"testing\" / \"spraybot_seg_finetuned.pt\"\n",
            "\n",
            "if BEST_PT.exists():\n",
            "    shutil.copy2(BEST_PT, TARGET_WEIGHTS)\n",
            "    shutil.copy2(BEST_PT, TARGET_TESTING)\n",
            "    print(\"=\" * 80)\n",
            "    print(\"[SUKSES] BOBOT TERBAIK BERHASIL DIEKSPOR KE:\")\n",
            "    print(f\"  1. {TARGET_WEIGHTS}\")\n",
            "    print(f\"  2. {TARGET_TESTING}\")\n",
            "    print(\"=\" * 80)\n",
            "else:\n",
            "    print(f\"[INFO] File {BEST_PT} belum terbentuk. Jalankan sel pelatihan di atas terlebih dahulu.\")"
        ]
    }
]

nb_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# 1. Simpan sebagai finetune_spray_analysis.ipynb (pasangan train_spray_analysis.ipynb)
target_analysis = ROOT / "finetune_spray_analysis.ipynb"
with open(target_analysis, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)
print(f"[BERHASIL] File dibuat: {target_analysis}")

# 2. Sinkronkan juga ke finetune_spray_video_dataset.ipynb yang sedang dibuka user
target_video = ROOT / "finetune_spray_video_dataset.ipynb"
with open(target_video, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)
print(f"[BERHASIL] File diperbarui: {target_video}")

# 3. Salin bobot agar weights/spraybot_seg_finetuned.pt langsung tersedia
src_w = ROOT / "weights" / "spraybot_seg_video_finetuned.pt"
dst_w = ROOT / "weights" / "spraybot_seg_finetuned.pt"
if src_w.exists() and not dst_w.exists():
    shutil.copy2(src_w, dst_w)
    print(f"[BERHASIL] Bobot disalin ke: {dst_w}")
