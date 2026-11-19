# 📤 `output/` — Direktori Artefak Visualisasi & Eksplorasi

> Direktori ini menyimpan **hasil visualisasi audit, eksplorasi pipeline, dan artefak intermediate** yang dihasilkan selama proses riset dan pengembangan model AI Spraybot.

---

## 📂 Daftar Isi

```
output/
├── finetune_video_dataset/              # Sub-folder output fine-tuning lama
│
│── ARTEFAK AUDIT & EKSPLORASI:
├── video_catalog.json                   # Katalog metadata semua video dataset
├── c0490_plume_zone.jpg                 # Visualisasi zona plume C0490
├── audit_anti_bottle_verification.png   # Audit verifikasi anti-botol false positive
├── audit_perfect_video_plumes.png       # Audit plume sempurna dari video
├── audit_video_plume_bursts.png         # Audit burst semprotan per video
├── audit_video_plume_extraction.png     # Audit hasil ekstraksi plume
├── cell10_saved_output.png              # Output cell 10 notebook (tersimpan)
├── cell18_saved_output.png              # Output cell 18 notebook (tersimpan)
├── cell6_saved_output.png               # Output cell 6 notebook (tersimpan)
├── color_enhancement_4panel_sample.png  # Uji 4-panel enhancement warna (gambar)
├── color_enhancement_4panel_video_sample.png  # Uji 4-panel enhancement (video)
├── flawless_video_side_by_side.png      # Perbandingan video sempurna
├── panel1a_test.png                     # Test panel visualisasi
├── test_exact_physical_side_by_side.png # Test komparasi fisik eksak
├── test_heatmap_guided_side_by_side.png # Test heatmap-guided segmentasi
└── test_perfect_heatmap_guided.png      # Test heatmap-guided sempurna
```

---

## 📋 Detail File Penting

### `video_catalog.json`
Katalog JSON yang berisi metadata terstruktur dari seluruh video dataset studio Paragon.

**Struktur Data:**
```json
{
  "C0350-Gas-Gatsby-1": {
    "session": "03-11-2026",
    "product": "Gatsby Gas",
    "frame_count": 240,
    "resolution": "1920x1080",
    "fps": 50,
    "nozzle_x_approx": 50,
    "nozzle_y_approx": 720
  },
  "C0490": {
    "session": "09-11-2026",
    "frame_count": 240,
    ...
  }
}
```

---

### Artefak Audit Plume

| File | Deskripsi |
|------|-----------|
| `audit_anti_bottle_verification.png` | Panel multi-video membuktikan model tidak mendeteksi botol atau tangan sebagai semprotan (zero false positive pada objek keras) |
| `audit_perfect_video_plumes.png` | Galeri plume terbaik dari berbagai video — digunakan untuk memverifikasi konsistensi anotasi organik |
| `audit_video_plume_bursts.png` | Visualisasi burst awal semprotan (momen t=0.5s hingga t=2.0s) untuk memahami dinamika uap |
| `audit_video_plume_extraction.png` | Hasil ekstraksi plume pipeline awal dibandingkan dengan ground truth visual |

---

### Artefak Heatmap & Enhancement

| File | Deskripsi |
|------|-----------|
| `color_enhancement_4panel_sample.png` | Eksperimen 4-panel: (1) Raw, (2) CLAHE Enhanced, (3) Heatmap Turbo, (4) Segmentasi overlay |
| `color_enhancement_4panel_video_sample.png` | Versi video dari eksperimen 4-panel enhancement |
| `test_heatmap_guided_side_by_side.png` | Perbandingan segmentasi berbasis heatmap vs segmentasi murni frame-differencing |
| `test_perfect_heatmap_guided.png` | Hasil terbaik dari pendekatan heatmap-guided segmentation |

---

### `c0490_plume_zone.jpg`
Visualisasi zona plume yang terdeteksi dari video benchmark C0490. Menampilkan:
- Kontur batas uap aktual
- Zona nozzle apex (titik awal plume)
- Gradasi densitas aerosol

---

### Output Notebook (Cell Snapshots)

| File | Konten |
|------|--------|
| `cell6_saved_output.png` | Snapshot output sel ke-6 notebook pelatihan (distribusi dataset) |
| `cell10_saved_output.png` | Snapshot output sel ke-10 (visualisasi konvergensi loss) |
| `cell18_saved_output.png` | Snapshot output sel ke-18 (panel inferensi akhir) |

---

### `finetune_video_dataset/`
Sub-folder output dari proses fine-tuning versi lama (iterasi sebelum pipeline organik Clean v2.0). Disimpan sebagai arsip referensi historis.

---

## 📝 Panduan Penggunaan Output

### Menemukan Visualisasi Spesifik
```python
from pathlib import Path
output_dir = Path("D:/Tugas Kuliah/Semester 6/MBKM - Magang/Paragon Spray/AI Analysis/output")

# Lihat semua file PNG
for f in sorted(output_dir.glob("*.png")):
    print(f.name)

# Buka gambar audit terbaik
from PIL import Image
img = Image.open(output_dir / "audit_perfect_video_plumes.png")
img.show()
```

### Memperbarui Katalog Video
```python
import json
from pathlib import Path

catalog_path = Path("output/video_catalog.json")
with open(catalog_path, "r") as f:
    catalog = json.load(f)

# Tambahkan entry video baru
catalog["C0598"] = {
    "session": "20-11-2026",
    "product": "New Product",
    ...
}

with open(catalog_path, "w") as f:
    json.dump(catalog, f, indent=2)
```

---

## ⚠️ Catatan

> [!NOTE]
> Folder `output/` bersifat **eksploratif dan non-kritis**. File di sini adalah artefak riset yang dapat dihapus jika disk penuh — semua data penting ada di `dataset/`, `weights/`, dan `runs/`.

> [!TIP]
> Jika ingin menyimpan visualisasi inferensi baru secara otomatis, ubah parameter `save=True` pada `model.predict()` dan arahkan `project` ke folder `output/`.
