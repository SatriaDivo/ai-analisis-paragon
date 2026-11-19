# ⚖️ `weights/` — Direktori Bobot Model AI Spraybot

> Direktori ini menyimpan **seluruh file bobot model resmi (`.pt`)** yang telah dirapikan ke dalam sub-folder fungsional sesuai fase pengembangan model AI Spraybot Paragon.

---

## 📂 Struktur Lengkap Direktori `weights/`

```text
weights/
│
├── 📁 finetuned/                        # ⭐ MODEL HASIL FINE-TUNING TERBARU (AKTIF)
│   └── spraybot_seg_finetuned.pt        # Model 122 video studio (23,87 MB, Epoch 24 best)
│
├── 📁 training_awal/                    # 📦 MODEL TRAINING AWAL (SEBELUM FINETUNING)
│   └── spraybot_seg_video_finetuned.pt  # Model 38 video organik baseline (22,77 MB)
│
├── 📁 pretrained/                       # 🔵 BOBOT DASAR PRE-TRAINED RESMI (COCO)
│   ├── yolov8s-seg.pt                   # Base weights untuk fine-tuning Small (23,91 MB)
│   ├── yolov8n-seg.pt                   # Base weights untuk Nano segmentation (7,07 MB)
│   └── yolov8n.pt                       # Base weights untuk Nano detection (6,55 MB)
│
│── Salinan Cepat (Quick Access):
├── spraybot_seg_finetuned.pt            # Salinan model aktif (kompatibel notebook & script)
├── spraybot_seg_video_finetuned.pt      # Salinan model baseline
├── yolov8s-seg.pt                       # Salinan base model Small
├── yolov8n-seg.pt                       # Salinan base model Nano
├── yolov8n.pt                           # Salinan base model Nano detect
│
└── README.md                            # Dokumentasi induk ini
```

---

## 📋 Detail Tiap Kategori Model

### 1. `weights/finetuned/` — Model Hasil Fine-Tuning 122 Video (Model Produksi)
* **File:** `spraybot_seg_finetuned.pt` (23.875.764 bytes / 23,87 MB)
* **Tanggal:** 18 November 2026 (Selesai 15:13 WIB)
* **Dataset:** 122 Video Studio (2.924 frame bersih, `dataset_yolo_clean`)
* **Performa Evaluasi:**
  - mAP50 (Box): **79.14%**
  - Recall (Box): **74.63%**
  - Precision (Mask): **77.39%** (Peak: **78.14%**)
  - mAP50 (Mask): **68.98% (~69.0%)**
  - Recall (Mask): **64.26%**
* **Keunggulan:** Bebas 100% potongan pisau lurus (*zero knife-cut*), zero false positive pada kain latar belakang, confidence hingga **85.4%**.

### 2. `weights/training_awal/` — Model Training Awal (Sebelum Fine-Tuning 122 Video)
* **File:** `spraybot_seg_video_finetuned.pt` (23.877.620 bytes / 22,77 MB)
* **Tanggal:** 12 November 2026
* **Dataset:** 38 Video Studio Sesi 1 (03-11-2026)
* **Performa Evaluasi:** mAP50 (Box) 67.50%, mAP50 (Mask) 55.95%, Precision (Mask) 73.34%.
* **Fungsi:** Model acuan dasar (*checkpoint*) yang digunakan sebagai bobot awal sebelum dilanjutkan ke fine-tuning 122 video.

### 3. `weights/pretrained/` — Bobot Resmi Ultralytics COCO (Base)
* **`yolov8s-seg.pt`** (23,91 MB): Bobot segmentasi Small resmi COCO. Digunakan saat pertama kali melatih model dari nol.
* **`yolov8n-seg.pt`** (7,07 MB): Bobot segmentasi Nano resmi COCO.
* **`yolov8n.pt`** (6,55 MB): Bobot deteksi kotak Nano resmi COCO.

---

## 🗑️ Catatan Pembersihan File Usang (Dihapus)

File-file berikut telah dibersihkan/dihapus karena tidak terpakai, cacat algoritma lama, atau merupakan duplikat tak berlabel:
1. `best.pt` — Duplikat tanpa label dari prototipe v1.
2. `yolo26n.pt` — File eksperimen coba-coba tidak terpakai.
3. `spraybot_seg_v1_best.pt` — Prototipe v1 nano lama dengan cacat potongan garis lurus vertikal parah.
4. `spraybot_seg_v3_best.pt` — Prototipe v3 nano lama.
5. `spraybot_seg_v3_small_best.pt` — Prototipe v3 small lama (15/10/2026).
6. `spraybot_seg_new_finetuned.pt` — Fine-tuning lama (03/11/2026) sebelum pembersihan label organik.
7. Duplikat root `yolov8n-seg.pt`, `yolov8n.pt`, dan `yolov8s-seg.pt` di root direktori proyek.
