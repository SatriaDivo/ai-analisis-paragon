# 🧪 Pengujian Model — 11-12 November 2026 (Baseline)

Dokumentasi hasil pengujian model baseline awal 38 video organik (`spraybot_seg_video_finetuned.pt`).

---

## 📂 Struktur Folder Pengujian Tanggal 11-11-2026
```text
testing_11-11-2026/
├── images/                             # 🖼️ Seluruh visualisasi hasil pengujian (.jpg)
│   ├── bukti_perbandingan_nyata.jpg    # Komparasi sebelum vs sesudah cone mask
│   ├── hasil_inferensi_organik_c0490.jpg # Uji awal video C0490
│   ├── panel_uji_c0490_organik.jpg     # Panel 3-view awal C0490
│   ├── komparasi_baru_vs_heatmap.jpg   # Uji korelasi kontur dengan spektral turbo
│   ├── debug_crop_act.jpg              # Debug crop frame aktif
│   ├── debug_crop_diff.jpg             # Debug frame differencing
│   ├── debug_new_organic_label.jpg     # Debug label poligon organik
│   ├── test_nozzle_connection.jpg      # Uji kontinuitas plume ke nozzle
│   ├── test_perfect_cone_apex.jpg      # Uji apex cone murni
│   ├── test_pure_natural_contour.jpg   # Uji kontur alami
│   ├── test_pure_organic_plume.jpg     # Uji plume organik
│   └── test_pure_spray_noglove.jpg     # Uji plume tanpa interferensi sarung tangan
├── weights/                            # ⚖️ Bobot model saat pengujian tanggal ini
│   └── spraybot_seg_video_finetuned.pt # Bobot 38 video organik (22,77 MB)
├── output/                             # 📦 Dump output inferensi lawas
└── README.md                           # Dokumentasi pengujian ini
```

---

## 📌 Informasi Run Model
- **Tanggal Run / Uji**: 11 – 12 November 2026
- **Model Checkpoint**: `weights/spraybot_seg_video_finetuned.pt` (38 video organik, 22,77 MB)
- **Dataset Latih**: 38 video studio (Sesi 1: 03-11-2026)
- **Capaian Metrik**: mAP50(B) 67.50%, mAP50(M) 55.95%, Precision(M) 73.34%.
- **Inovasi Pertama**: Eksperimen transisi dari bounding box kaku ke masker organik moncong botol.
