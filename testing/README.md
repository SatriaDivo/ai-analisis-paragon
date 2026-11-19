# 🧪 `testing/` — Direktori Pengujian & Validasi Model AI Spraybot

> Direktori ini menyimpan **skrip pengujian, notebook interaktif, bobot pengujian aktif, serta arsip hasil visual pengujian model per tanggal run**.

---

## 📂 Struktur Lengkap Direktori `testing/`

Semua komponen pengujian kini terpisah rapi ke dalam folder fungsional:

```text
testing/
│
├── 📁 scripts/                          # 📜 SKRIP & NOTEBOOK PENGUJIAN
│   ├── README.md                        # Dokumentasi skrip & cara penggunaan
│   ├── uji_satu_gambar.ipynb            # Notebook uji interaktif 1 gambar (3-view panel)
│   ├── uji_satu_folder.ipynb            # Notebook uji interaktif 1 folder video
│   └── uji_model.py                     # Script Python inferensi batch serbaguna
│
├── 📁 weights/                          # ⚖️ BOBOT MODEL AKTIF UNTUK PENGUJIAN
│   └── spraybot_seg_finetuned.pt        # Salinan model aktif terkini (kompatibel script)
│
├── 📁 output/                           # 📤 FOLDER DEFAULT OUTPUT PENGUJIAN RUNTIME
│   └── (Menyimpan output gambar/plot saat script dijalankan secara interaktif)
│
├── 📁 testing_18-11-2026/               # ⭐ ARSIP PENGUJIAN RUN 18 NOVEMBER 2026
│   ├── images/                          # 🖼️ Gambar visualisasi hasil pengujian
│   │   ├── uji_C0490_peak.jpg           # Uji frame puncak Kahf C0490 (Conf: 79.8%)
│   │   ├── uji_C0365_mid.jpg            # Uji frame matang Wardah C0365 (Conf: 85.4%)
│   │   └── uji_C0490_bg_neg.jpg         # Uji frame latar belakang kosong (0% FP)
│   ├── weights/                         # ⚖️ Model checkpoint saat tanggal ini
│   │   └── spraybot_seg_finetuned.pt    # Bobot epoch 24 best (23,87 MB)
│   ├── reports/                         # 📊 Laporan data kuantitatif
│   │   └── test_results_summary.json    # Ringkasan confidence, area, bbox
│   └── README.md                        # Laporan detail evaluasi 18-11-2026
│
├── 📁 testing_11-11-2026/               # 📦 ARSIP PENGUJIAN RUN 11 NOVEMBER 2026
│   ├── images/                          # 🖼️ Gambar visualisasi hasil pengujian lawas
│   │   ├── bukti_perbandingan_nyata.jpg # Komparasi sebelum vs sesudah cone mask
│   │   ├── hasil_inferensi_organik_c0490.jpg
│   │   ├── panel_uji_c0490_organik.jpg
│   │   ├── komparasi_baru_vs_heatmap.jpg
│   │   └── ... (8 file uji rintisan apex cone & background)
│   ├── weights/                         # ⚖️ Model checkpoint saat tanggal ini
│   │   └── spraybot_seg_video_finetuned.pt # Bobot 38 video organik awal (22,77 MB)
│   ├── output/                          # 📦 Dump output inferensi lawas
│   └── README.md                        # Laporan detail evaluasi 11-11-2026
│
└── README.md                            # Dokumentasi induk ini
```

---

## 📋 Ringkasan Arsip Pengujian per Tanggal

| Folder Tanggal | Model yang Diuji | Status / Hasil Kunci | Folder Gambar | Folder Bobot | Laporan |
|---|---|---|---|---|---|
| **`testing_18-11-2026/`** | `spraybot_seg_finetune_122vids` / `finetuned/spraybot_seg_finetuned.pt` (Epoch 24) ⚠️ **DEPRECATED** | Confidence 79.8–85.4%, Zero FP BG, Zero knife-cut. Model ini sudah digantikan oleh Model Aktif. | `images/` | `weights/` | `reports/` |
| **`testing_11-11-2026/`** | `training_awal/spraybot_seg_video_finetuned.pt` (38 Video) ⭐ **MODEL AKTIF** | **mAP50 Box 79.14%**, Confidence 63.6% pada C0490. Lolos Quality Gate, dideploy ke website. | `images/` | `weights/` | `output/` |

---

## 📖 Makna & Definisi Tiap Metrik Evaluasi

| Metrik Evaluasi | Nilai Uji (18-11-2026) | Makna Matematis & Fungsi Praktis dalam Pengujian |
|---|---|---|
| **`mAP50 (Box / Plume)`** | **79.14%** (🟢 $\ge 72\%$) | **Akurasi Deteksi Kotak (IoU $\ge 0.50$):** Mengukur ketepatan AI dalam menemukan lokasi gumpalan semprotan secara global di dalam frame tanpa meleset. |
| **`mAP50-95 (Box)`** | **48.46%** | **Keketatan Batas Kotak (IoU 0.50–0.95):** Menguji seberapa presisi margin kotak terhadap semprotan nyata pada berbagai tingkat ambang batas ketat. |
| **`Precision (Box)`** | **74.23%** (🟢 $\ge 72\%$) | **Ketepatan Prediksi Kotak:** Dari seluruh kotak yang diprediksi model, ~74.23% benar-benar berisi semprotan nyata (sangat minim deteksi palsu pada kain latar). |
| **`Recall (Box)`** | **74.63%** (🟢 $\ge 72\%$) | **Sensitivitas Tangkapan Kotak:** Dari seluruh uap yang ada di frame, model berhasil menangkap 74.63%-nya (hanya uap awal tipis yang terlewat). |
| **`mAP50 (Mask / Segmentasi)`** | **68.98%** (🟡 Mendekati 72%) | **Standar Emas Akurasi Fluida (IoU $\ge 0.50$):** Mengukur kesesuaian bentuk masker/poligon fluida yang digambar AI terhadap bentuk uap semprotan sesungguhnya. Naik **+13.03%** dari model awal. |
| **`mAP50-95 (Mask)`** | **36.38%** | **Akurasi Kontur Halus (IoU 0.50–0.95):** Mengukur ketelitian piksel batas terluar masker terhadap droplet aerosol halus di tepi uap. |
| **`Precision (Mask)`** | **77.39%** (🟢 $\ge 72\%$) | **Kebersihan Masker:** Dari seluruh piksel yang diwarnai AI, 77.39% murni uap cairan semprotan (tidak melebar ke latar belakang kain atau botol). |
| **`Recall (Mask)`** | **64.26%** (🟡 Naik +13.2%) | **Kelengkapan Masker:** Persentase partikel uap nyata yang berhasil dicakup masker AI. Menunjukkan semprotan tertangkap lebih tebal dan panjang. |

---

## 🚀 Cara Menjalankan Pengujian Baru

### 1. Uji Satu Gambar Interaktif (Panel 3-View)
Buka [testing/scripts/uji_satu_gambar.ipynb](file:///d:/Tugas%20Kuliah/Semester%206/MBKM%20-%20Magang/Paragon%20Spray/AI%20Analysis/testing/scripts/uji_satu_gambar.ipynb), pilih preset atau masukkan path citra semprotan, lalu jalankan semua sel. Gambar hasil uji otomatis tersimpan ke `testing/output/`.

### 2. Uji Satu Folder Video
Buka [testing/scripts/uji_satu_folder.ipynb](file:///d:/Tugas%20Kuliah/Semester%206/MBKM%20-%20Magang/Paragon%20Spray/AI%20Analysis/testing/scripts/uji_satu_folder.ipynb) untuk menguji seluruh frame video secara berurutan dan mengamati perkembangan kepulan uap.

### 3. Uji via Command Line (CLI)
```bash
python testing/scripts/uji_model.py --image "dataset/dataset_video_frames_09-11-2026/C0490/C0490_f00150_t03.00s.jpg"
```
Output berupa citra overlay dan metrik (reach cm, width cm, dsb.) akan langsung ditampilkan dan disimpan ke folder output.
