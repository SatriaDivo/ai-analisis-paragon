# 📜 `testing/scripts/` — Skrip & Notebook Pengujian Model

Direktori ini berisi seluruh skrip eksekusi dan notebook interaktif untuk menguji performa model AI segmentasi fluida semprotan.

---

## 📂 Daftar File Skrip & Notebook

| File | Tipe | Fungsi Utama | Cara Menjalankan |
|---|---|---|---|
| `uji_satu_gambar.ipynb` | Jupyter Notebook | Uji inferensi 1 gambar semprotan secara interaktif dengan **Panel 3-View** (Citra Asli + Segmentasi Masker Hijau + Spektral Densitas Heatmap Turbo). | Buka di VS Code, pilih kernel Python, jalankan per sel. |
| `uji_satu_folder.ipynb` | Jupyter Notebook | Uji inferensi batch terhadap seluruh frame dalam 1 folder video untuk meninjau perkembangan dinamika semprotan. | Buka di VS Code, tentukan path folder video, jalankan per sel. |
| `uji_model.py` | Python CLI Script | Runner inferensi command line serbaguna (mendukung opsi `--image`, `--folder`, kalibrasi `--px-per-cm`, dan ekspor laporan CSV). | `python testing/scripts/uji_model.py --image <path>` |

---

## ⚙️ Konfigurasi Path Otomatis
Seluruh skrip dan notebook di folder ini telah dilengkapi logika penentuan path root dinamis (`Path.cwd()`), sehingga dapat dijalankan dari:
- Direktori workspace utama (`AI Analysis/`)
- Maupun langsung dari dalam folder `testing/` atau `testing/scripts/`.
Model bobot default yang dimuat adalah model terbaru di `weights/spraybot_seg_finetuned.pt`.
