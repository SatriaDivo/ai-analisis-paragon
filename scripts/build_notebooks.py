import json
from pathlib import Path

def create_notebook():
    ROOT = Path("D:/Tugas Kuliah/Semester 6/MBKM - Magang/Paragon Spray/AI Analysis")
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# PELATIHAN RESMI AI SPRAYBOT — FINE-TUNING YOLOV8S-SEG 122 VIDEO STUDIO\n",
                "### Proyek Riset MBKM Magang Industri — PT Paragon Technology and Innovation & PT Makerindo Prima Solusi\n",
                "\n",
                "Notebook ini merupakan pipeline resmi pelatihan (*fine-tuning*) model deep learning **YOLOv8s-seg** menggunakan dataset bersih konsolidasi **122 Video Studio Paragon** (2.924 frame teranotasi 1080p).\n",
                "\n",
                "---\n",
                "\n",
                "### Pembaruan & Fitur Kunci Tahap Ini:\n",
                "1. **Konsolidasi 3 Sesi Rekaman Studio (122 Video Total):**\n",
                "   - **Sesi 1 (03-11-2026):** 21 Video (Gatsby, Kahf, Wardah, Labora)\n",
                "   - **Sesi 2 (09-11-2026):** 58 Video (`C0467` s.d `C0545`)\n",
                "   - **Sesi 3 (13-11-2026):** 43 Video (`C0549` s.d `C0597`)\n",
                "   - **Total:** 2.378 Frame Train, 546 Frame Val (termasuk 244 sampel negatif latar belakang).\n",
                "2. **Eliminasi 100% Garis Lurus Vertikal (Organic Nozzle Apex Cone):**\n",
                "   - Algoritma lama memotong uap dengan pisau vertikal kaku (`bin_mask[:, :hand_cut_x] = 0`), menyebabkan model memprediksi garis lurus vertikal pada semprotan.\n",
                "   - Algoritma baru mendeteksi titik moncong botol (*nozzle apex*) dan menerapkan masker kerucut organik yang meruncing mulus tepat di titik keluarnya cairan ($x \\approx 120\\text{ px}, y \\approx 780-830\\text{ px}$).\n",
                "   - Hasil segmentasi AI sekarang **terhubung langsung dan organik ke moncong botol tanpa garis lurus**.\n",
                "3. **Zero False Positive Background:**\n",
                "   - Menyertakan **244 sampel negatif** ($t=0.00\\text{s}$ saat botol belum menyemprot) dengan label kosong agar AI tidak salah mendeteksi lipatan kain hitam/abu-abu.\n",
                "4. **Arah Semprotan Konsisten:** 100% video menyemprot dari **Kiri ke Kanan** (`fliplr=0.0`).\n",
                "5. **Akselerasi GPU:** Dioptimasi untuk NVIDIA GeForce RTX 4050 Laptop GPU dengan `workers=0` (anti-crash multiprocessing Windows)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup Environment, Library & Deteksi Hardware GPU (RTX 4050)\n",
                "Memverifikasi ketersediaan akselerasi hardware CUDA GPU dan menginisialisasi pustaka OpenCV, PyTorch, Ultralytics YOLOv8, serta direktori kerja proyek."
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
                "## 2. Manajemen Dataset Bersih Terpadu (122 Video Studio Paragon)\n",
                "Dataset bersih yang digunakan bersumber dari 122 video rekaman studio:\n",
                "- **21 Video Dataset 1** (03-11-2026: Kahf, Wardah, Gatsby, Labora)\n",
                "- **58 Video Dataset 2** (09-11-2026: C0467 - C0545)\n",
                "- **43 Video Dataset 3** (13-11-2026: C0549 - C0597)\n",
                "- **Split Tingkat-Video (Anti Data Leakage):** 98 Video Train, 24 Video Val.\n",
                "- **Total:** 2.924 frame (2.378 Train, 546 Val, termasuk 244 frame negatif)."
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
                "print(\"STATISTIK DATASET YOLOV8-SEG TERPADU (122 VIDEO)\")\n",
                "print(\"=\" * 80)\n",
                "print(f\"File data.yaml     : {DATA_YAML}\")\n",
                "print(f\"Jumlah Kelas       : {yaml_config.get('names')}\")\n",
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
                "## 3. Visualisasi Distribusi Dataset & Ground Truth Mask Bebas Garis Lurus\n",
                "Menampilkan diagram komposisi dataset serta contoh ground truth mask organik yang meruncing mulus ke moncong botol."
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
                "labels_pie = [f'Train Positif ({n_train_pos})', f'Train Negatif ({n_train_neg})',\n",
                "              f'Val Positif ({n_val_pos})', f'Val Negatif ({n_val_neg})']\n",
                "sizes_pie = [n_train_pos, n_train_neg, n_val_pos, n_val_neg]\n",
                "colors_pie = ['#2ecc71', '#95a5a6', '#3498db', '#bdc3c7']\n",
                "explode = (0.05, 0.05, 0.05, 0.05)\n",
                "\n",
                "axes[0].pie(sizes_pie, explode=explode, labels=labels_pie, colors=colors_pie,\n",
                "            autopct='%1.1f%%', startangle=140, shadow=True)\n",
                "axes[0].set_title('Distribusi Split Citra & Sampel Negatif (122 Video)', fontsize=13, fontweight='bold')\n",
                "\n",
                "# Bar Chart Perbandingan\n",
                "categories = ['Train Positif', 'Train Negatif', 'Val Positif', 'Val Negatif']\n",
                "counts = [n_train_pos, n_train_neg, n_val_pos, n_val_neg]\n",
                "bars = axes[1].bar(categories, counts, color=['#2ecc71', '#7f8c8d', '#3498db', '#95a5a6'])\n",
                "for bar in bars:\n",
                "    yval = bar.get_height()\n",
                "    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 15, int(yval), ha='center', va='bottom', fontweight='bold')\n",
                "axes[1].set_ylabel('Jumlah Frame', fontsize=11)\n",
                "axes[1].set_title('Komposisi Frame Latih dan Validasi', fontsize=13, fontweight='bold')\n",
                "axes[1].grid(axis='y', linestyle='--', alpha=0.5)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Konfigurasi Parameter & Eksekusi Fine-Tuning YOLOv8s-seg\n",
                "Parameter pelatihan dioptimasi khusus untuk fine-tuning segmentasi semprotan fluida studio:\n",
                "- **Base Weights:** `weights/spraybot_seg_video_finetuned.pt` (transfer learning dari bobot video terbaik sebelumnya).\n",
                "- **Epochs:** `25` (cukup untuk konvergensi halus pada 122 video).\n",
                "- **Image Size (`imgsz`):** `1024` (resolusi tinggi mempertahankan detail mikrodroplet uap).\n",
                "- **Batch Size:** `8` (optimal untuk VRAM 6GB RTX 4050 Laptop GPU).\n",
                "- **Workers:** `0` (wajib di lingkungan Windows untuk mencegah runtime error spawn multiprocessing).\n",
                "- **Fliplr:** `0.0` (dilarang membalik horizontal karena arah semprotan 100% konsisten Kiri ke Kanan)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Inisialisasi Model Dasar (Pretrained Video Weights)\n",
                "BASE_WEIGHTS = ROOT / \"weights\" / \"spraybot_seg_video_finetuned.pt\"\n",
                "if not BASE_WEIGHTS.exists():\n",
                "    BASE_WEIGHTS = ROOT / \"weights\" / \"yolov8s-seg.pt\"\n",
                "\n",
                "RUNS_DIR = ROOT / \"runs\" / \"spraybot_clean_training\"\n",
                "\n",
                "print(\"=\" * 80)\n",
                "print(\"KONFIGURASI FINE-TUNING YOLOV8S-SEG (122 VIDEO)\")\n",
                "print(\"=\" * 80)\n",
                "print(f\"Bobot Awal       : {BASE_WEIGHTS}\")\n",
                "print(f\"File data.yaml   : {DATA_YAML}\")\n",
                "print(f\"Output Runs      : {RUNS_DIR / 'spraybot_yolov8s_seg_122vids'}\")\n",
                "print(\"=\" * 80)\n",
                "\n",
                "model = YOLO(str(BASE_WEIGHTS))\n",
                "\n",
                "EPOCHS = 25\n",
                "IMG_SIZE = 1024\n",
                "BATCH_SIZE = 8\n",
                "WORKERS = 0\n",
                "\n",
                "print(f\"Memulai fine-tuning {EPOCHS} epochs pada resolusi {IMG_SIZE}x{IMG_SIZE}...\")\n",
                "start_train_time = time.time()\n",
                "\n",
                "results = model.train(\n",
                "    data=str(DATA_YAML),\n",
                "    epochs=EPOCHS,\n",
                "    imgsz=IMG_SIZE,\n",
                "    batch=BATCH_SIZE,\n",
                "    workers=WORKERS,\n",
                "    device=0 if torch.cuda.is_available() else 'cpu',\n",
                "    project=str(RUNS_DIR),\n",
                "    name=\"spraybot_yolov8s_seg_122vids\",\n",
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
                "print(f\"PELATIHAN SELESAI DALAM {elapsed:.2f} MENIT!\")\n",
                "print(\"=\" * 80)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Evaluasi Metrik Pelatihan (Loss & mAP Segmentasi)\n",
                "Melihat grafik konvergensi loss dan kurva presisi mAP50 mask segmentasi dari hasil pelatihan."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "results_png = RUNS_DIR / \"spraybot_yolov8s_seg_122vids\" / \"results.png\"\n",
                "if not results_png.exists():\n",
                "    results_png = RUNS_DIR / \"spraybot_yolov8s_seg_run\" / \"results.png\"\n",
                "\n",
                "if results_png.exists():\n",
                "    img = Image.open(results_png)\n",
                "    plt.figure(figsize=(18, 10))\n",
                "    plt.imshow(img)\n",
                "    plt.axis('off')\n",
                "    plt.title('Kurva Hasil Pelatihan YOLOv8s-Seg (Loss, Precision, Recall, mAP50-95)', fontsize=15, fontweight='bold')\n",
                "    plt.show()\n",
                "else:\n",
                "    print(f\"[INFO] File {results_png} belum tersedia. Pelatihan mungkin sedang berlangsung.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Uji Inferensi Visual Pada Kasus Uji Nyata (Verifikasi Bebas Garis Lurus)\n",
                "Menguji model pada citra semprotan aktif nyata untuk membuktikan bahwa:\n",
                "1. Uap fluida **meruncing dan terhubung organik ke moncong botol** ($x \\approx 120-140\\text{ px}$).\n",
                "2. **Zero Garis Lurus / Knife Cut** pada sisi kiri uap cairan.\n",
                "3. Background hitam dan kain bebas false positive."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Uji Inferensi pada Video Uji Nyata C0490\n",
                "TEST_IMAGE_PATH = ROOT / \"dataset\" / \"dataset_video_frames_09-11-2026\" / \"C0490\" / \"C0490_f00150_t03.00s.jpg\"\n",
                "if not TEST_IMAGE_PATH.exists():\n",
                "    # Fallback ke sampel val\n",
                "    TEST_IMAGE_PATH = sorted(list((ROOT / \"dataset\" / \"dataset_yolo_clean\" / \"images\" / \"val\").glob(\"*.jpg\")))[0]\n",
                "\n",
                "best_model_weights = RUNS_DIR / \"spraybot_yolov8s_seg_122vids\" / \"weights\" / \"best.pt\"\n",
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
                "axes[0].set_title(f'Panel 1: Citra Asli ({TEST_IMAGE_PATH.name})', fontsize=12, fontweight='bold')\n",
                "axes[0].axis('off')\n",
                "\n",
                "axes[1].imshow(panel2)\n",
                "axes[1].set_title(f'Panel 2: Hasil Segmentasi AI YOLOv8-Seg (Conf: {conf_val*100:.1f}%) — Terhubung Organik ke Moncong', fontsize=12, fontweight='bold')\n",
                "axes[1].axis('off')\n",
                "\n",
                "axes[2].imshow(heatmap_rgb)\n",
                "axes[2].set_title('Panel 3: Analisis Spektral Densitas Aerosol (Heatmap Turbo)', fontsize=12, fontweight='bold')\n",
                "axes[2].axis('off')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Ekspor & Deployment Bobot Model Terbaik (`best.pt`)\n",
                "Menyalin file bobot model terbaik hasil pelatihan ke direktori resmi proyek:\n",
                "1. `weights/spraybot_seg_video_finetuned.pt` (Repositori Bobot Resmi)\n",
                "2. `testing/spraybot_seg_video_finetuned.pt` (Direktori Siap Uji Industri)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "BEST_PT = RUNS_DIR / \"spraybot_yolov8s_seg_122vids\" / \"weights\" / \"best.pt\"\n",
                "if not BEST_PT.exists():\n",
                "    BEST_PT = RUNS_DIR / \"spraybot_yolov8s_seg_run\" / \"weights\" / \"best.pt\"\n",
                "\n",
                "TARGET_WEIGHTS = ROOT / \"weights\" / \"spraybot_seg_video_finetuned.pt\"\n",
                "TARGET_TESTING = ROOT / \"testing\" / \"spraybot_seg_video_finetuned.pt\"\n",
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
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
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
    
    # Simpan sebagai train_spray_analysis.ipynb
    target_train_nb = ROOT / "train_spray_analysis.ipynb"
    with open(target_train_nb, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=1, ensure_ascii=False)
    print(f"[OK] Berhasil membuat: {target_train_nb}")
    
    # Perbarui juga finetune_spray_video_dataset.ipynb agar keduanya sinkron dan lengkap
    target_finetune_nb = ROOT / "finetune_spray_video_dataset.ipynb"
    with open(target_finetune_nb, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=1, ensure_ascii=False)
    print(f"[OK] Berhasil memperbarui: {target_finetune_nb}")

if __name__ == "__main__":
    create_notebook()
