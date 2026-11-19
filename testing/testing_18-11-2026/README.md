# 🧪 Pengujian Model — 18 November 2026

Dokumentasi hasil pengujian model segmentasi YOLOv8s-seg hasil fine-tuning 122 video studio (`spraybot_seg_finetune_122vids`).

---

## 📂 Struktur Folder Pengujian Tanggal 18-11-2026
```text
testing_18-11-2026/
├── images/                       # 🖼️ Seluruh visualisasi hasil pengujian (.jpg)
│   ├── uji_C0490_peak.jpg        # Panel visual uji C0490 (Confidence: 79.8%)
│   ├── uji_C0365_mid.jpg         # Panel visual uji C0365 (Confidence: 85.4%)
│   └── uji_C0490_bg_neg.jpg      # Panel visual uji background kosong (0% FP)
├── weights/                      # ⚖️ Bobot model saat pengujian tanggal ini
│   └── spraybot_seg_finetuned.pt # Bobot checkpoint terbaik (Epoch 24, 23,87 MB)
├── reports/                      # 📊 Laporan data kuantitatif
│   └── test_results_summary.json # Ringkasan deteksi, confidence, area, bbox
└── README.md                     # Dokumentasi pengujian ini
```

---

## 📌 Informasi Run Model
- **Tanggal Run / Uji**: 18 November 2026 (15:13 WIB)
- **Model Checkpoint**: `weights/spraybot_seg_finetuned.pt` (Epoch 24 best, 23,87 MB)
- **Dataset Latih**: 122 video studio (2.924 frame bersih, `dataset_yolo_clean`)
- **Fitur Khusus**: *Organic Apex Cone Mask* (zero knife-cut), Video-level train/val split.

---

## 📊 Hasil Uji Benchmark Frame

| Uji Kasus | Video / Frame Sumber | Tipe Pengujian | Plume Terdeteksi | Confidence Score | Luas Area Plume | File Gambar di `images/` |
|---|---|---|---|---|---|---|
| **C0490 Peak** | `C0490_f00150_t03.00s.jpg` | Plume Puncak (Sesi 2 - Kahf) | ✅ Ya | **79.8%** | ~87.419 px | `images/uji_C0490_peak.jpg` |
| **C0365 Mid** | `C0365..._f00620_t12.40s.jpg` | Plume Matang (Sesi 1 - Wardah) | ✅ Ya | **85.4%** | ~64.120 px | `images/uji_C0365_mid.jpg` |
| **C0490 BG** | `C0490_f00000_t00.00s.jpg` | Negatif Background (t=0s) | ❌ Tidak (Bersih) | **0.0%** | 0 px | `images/uji_C0490_bg_neg.jpg` |

---

## 🔍 Observasi Kualitas Segmentasi
1. **Zero Knife-Cut**: Batas kiri mask menempel dan meruncing secara alami ke moncong botol (*nozzle apex*), tidak ada lagi potongan garis lurus vertikal.
2. **Zero False Positive pada Background**: Frame t=0.00s tidak menghasilkan deteksi palsu sama sekali (100% bersih).
3. **Puncak Confidence Tinggi**: Pada semprotan aktif stabil C0365, confidence score tembus **85.4%**.
4. **Data JSON**: Ringkasan data kuantitatif tersimpan di `reports/test_results_summary.json`.
