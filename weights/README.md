# ⚖️ `weights/` — Direktori Bobot Model AI Spraybot

> Direktori ini menyimpan **seluruh file bobot model resmi (`.pt`)** yang telah dirapikan ke dalam sub-folder fungsional sesuai fase pengembangan model AI Spraybot Paragon.

---

## 📂 Struktur Lengkap Direktori `weights/`

```text
weights/
│
├── 📁 training_awal/                   # ⭐ MODEL AKTIF/PRODUKSI (Lolos Quality Gate)
│   └── spraybot_seg_video_finetuned.pt # Model 38 video organik (23.877.620 bytes, 22,77 MB)
│
├── 📁 finetuned/                        # 📦 MODEL HASIL FINE-TUNING CLEAN v2.0 (Eksperimen)
│   ├── spraybot_seg_finetuned.pt        # Model Clean v2.0 — Epoch akhir (23.876.788 bytes)
│   └── spraybot_seg_finetune_clean_best.pt  # Model Clean v2.0 — Best Epoch 28 (23.876.788 bytes)
│
├── 📁 pretrained/                       # 🔵 BOBOT DASAR PRE-TRAINED RESMI (COCO) [gitignored]
│   ├── yolov8s-seg.pt                   # Base weights untuk fine-tuning Small (23,91 MB)
│   ├── yolov8n-seg.pt                   # Base weights untuk Nano segmentation (7,07 MB)
│   └── yolov8n.pt                       # Base weights untuk Nano detection (6,55 MB)
│
│── Salinan Cepat (Quick Access):
├── spraybot_seg_finetuned.pt            # Salinan model Clean v2.0 (23.876.788 bytes)
├── spraybot_seg_video_finetuned.pt      # ⭐ Salinan model AKTIF di website (23.877.620 bytes)
│
└── README.md                            # Dokumentasi induk ini
```

---

## 📋 Detail Tiap Kategori Model

### 1. `weights/training_awal/` — Model Training Awal ⭐ **AKTIF DI WEBSITE**
* **File:** `spraybot_seg_video_finetuned.pt` (23.877.620 bytes / 22,77 MB)
* **Tanggal:** 12 November 2026
* **Dataset:** 38 Video Studio Sesi 1 (03-11-2026)
* **Status:** ✅ **Model Produksi Aktif** — dideploy ke `Spray-Website/models/segmentation/`
* **Performa Evaluasi (Lolos Quality Gate):**
  - mAP50 (Box): **79.14%** 🟢
  - Recall (Box): **74.63%** 🟢
  - Precision (Box): **74.23%** 🟢
  - Precision (Mask): **77.39%** 🟢
  - mAP50 (Mask): **68.98%** 🟡
  - Recall (Mask): **64.26%** 🟡
  - Confidence pada frame uji (C0490): **63.6%**
* **Keunggulan:** Zero false positive pada kain latar belakang, kontur organik nozzle apex.

### 2. `weights/finetuned/` — Model Clean v2.0 (Eksperimen, Belum Lolos Gate)
* **File:** `spraybot_seg_finetuned.pt` dan `spraybot_seg_finetune_clean_best.pt` (keduanya 23.876.788 bytes)
* **Tanggal:** 19 November 2026 03:51 WIB (selesai pelatihan)
* **Dataset:** Dataset v2.0 — 3.493 frame dari 122 video (`dataset_yolo_clean`)
* **Run:** `runs/spraybot_clean_training/spraybot_seg_finetune_122vids_clean/` (30 epoch, 646.30 menit)
* **Best Epoch:** Epoch 28
* **Hyperparameter:** `mosaic=0.0`, `erasing=0.0`, `mixup=0.0`, `retina_masks=True`, `lr0=0.001`, `imgsz=1024`, `batch=8`, `patience=12`
* **Performa Evaluasi:**
  - mAP50 (Box): **54.63%** ⚠️
  - Recall (Box): **49.84%** ⚠️
  - Precision (Box): **56.86%** ⚠️
  - **Precision (Mask): 73.43%** 🟢
  - mAP50 (Mask): **56.14%** ⚠️
  - Recall (Mask): **44.43%** ⚠️
  - Confidence pada frame uji (C0490): **45.0%** (lebih rendah dari Model Aktif)
  - Mask area: **231.014 px** (uap lebih lebar/tipis vs 84.429 px model aktif)
* **Catatan:** Precision Mask (73.43%) unggul atas model lama yang rusak, namun belum menggantikan Model Aktif di produksi.

### 3. `weights/pretrained/` — Bobot Resmi Ultralytics COCO (Base) [gitignored]
* **`yolov8s-seg.pt`** (23,91 MB): Bobot segmentasi Small resmi COCO. Digunakan saat pertama kali melatih model dari nol.
* **`yolov8n-seg.pt`** (7,07 MB): Bobot segmentasi Nano resmi COCO.
* **`yolov8n.pt`** (6,55 MB): Bobot deteksi kotak Nano resmi COCO.
* ⚠️ Folder `pretrained/` dikecualikan dari Git (`.gitignore`) karena ukuran file besar.

---

## 🗑️ Catatan Pembersihan File Usang (Dihapus)

File-file berikut telah dibersihkan/dihapus karena tidak terpakai, cacat algoritma lama, atau merupakan duplikat tak berlabel:
1. `best.pt` — Duplikat tanpa label dari prototipe v1.
2. `yolo26n.pt` — File eksperimen coba-coba tidak terpakai.
3. `spraybot_seg_v1_best.pt` — Prototipe v1 nano lama dengan cacat potongan garis lurus vertikal parah.
4. `spraybot_seg_v3_best.pt` — Prototipe v3 nano lama.
5. `spraybot_seg_v3_small_best.pt` — Prototipe v3 small lama (15/10/2026).
6. `spraybot_seg_new_finetuned.pt` — Fine-tuning lama (03/11/2026) sebelum pembersihan label organik.
7. `runs/.../spraybot_seg_finetune_122vids/weights/best.pt` — Model lama dengan augmentasi destruktif (`mosaic=1.0`, `erasing=0.4`) yang menyebabkan uap berlubang dan bbox terpotong di x=120.

---

*Diperbarui: 19 November 2026 | Model Aktif: `training_awal/spraybot_seg_video_finetuned.pt` (79.14% mAP50 Box)*
