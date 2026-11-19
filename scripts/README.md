# 🛠️ `scripts/` — Direktori Script Pipeline AI Spraybot

> Direktori ini berisi seluruh **script Python utility** yang membentuk pipeline pengolahan data, ekstraksi frame, pembangunan dataset, dan training model AI Spraybot.

---

## 📂 Daftar Script & Fungsinya

### 🟢 Script AKTIF (Pipeline Produksi)

| Script | Fungsi | Dijalankan Kapan |
|--------|--------|-----------------|
| `extract_all_new_videos.py` | Ekstraksi batch frame dari 101 video baru (Sesi 2 & 3) ke folder frame | **Sekali**, saat ada video baru |
| `build_yolo_clean_dataset.py` | Build dataset YOLO bersih dari 122 video (3 sesi) dengan masker organik bebas garis lurus | **Setiap kali** dataset ingin diperbarui |
| `train_yolo_clean.py` | Script fine-tuning YOLOv8s-seg via `.py` (alternatif notebook) | Opsional (gunakan notebook lebih diutamakan) |
| `create_finetune_notebook.py` | Generator otomatis `finetune_spray_analysis.ipynb` | Saat notebook perlu dibuat ulang |

---

### 🟡 Script Diagnostik & Eksplorasi (Utility)

| Script | Fungsi |
|--------|--------|
| `analyze_c0490.py` | Analisis mendalam frame video C0490 (uji kasus benchmark utama) |
| `check_c0467.py` | Pemeriksaan awal video C0467 (video pertama Sesi 2) |
| `check_hand_zone.py` | Debug zona tangan/sarung tangan di frame semprotan |
| `crop_left.py` | Percobaan pemotongan sisi kiri frame |
| `crop_test_c0490.py` | Test crop pada video C0490 |
| `crop_video_frames.py` | Crop frame dari 1920×1080 → 1920×800 (meja dihapus) |
| `diff_test_c0490.py` | Test frame differencing pada C0490 |
| `find_bright.py` | Deteksi area terang berlebih di frame |
| `find_glove.py` / `find_glove2.py` | Deteksi sarung tangan operator berdasarkan warna |
| `find_mismatch.py` | Mencari frame dengan resolusi tidak sesuai (penyebab crash `cv2.absdiff`) |
| `find_nozzle_exact.py` | Deteksi koordinat tepat titik moncong botol (nozzle apex) |
| `inspect_d2_videos.py` | Inspeksi awal video Dataset Sesi 2 |
| `inspect_plume_profile.py` | Analisis profil uap/plume dari frame |
| `inspect_uji_gambar.py` | Inspeksi folder gambar uji |
| `save_c0467_preview.py` | Simpan preview frame dari C0467 |
| `save_plume_zone.py` | Simpan visualisasi zona plume yang terdeteksi |
| `scan_spray_timelines.py` | Scan timeline semprotan di seluruh video dataset |
| `test_batch_fluid_extract.py` | Uji ekstraksi fluida secara batch |
| `test_c0490.py` | Uji pipeline pada video C0490 |
| `test_fluid_extraction.py` | Uji fungsi ekstraksi fluida |
| `test_gain_adjust.py` | Uji kompensasi gain/pencahayaan |
| `test_plume_extract_c0490.py` | Uji ekstraksi plume khusus C0490 |
| `test_refined_segmentation.py` | Uji pipeline segmentasi yang sudah diperhalus |
| `test_true_fluid_c0490.py` | Uji deteksi fluida murni (tanpa artefak sarung tangan) |
| `update_uji_gambar_focus.py` | Update fokus pengujian gambar uji |

---

## 📄 Detail Script Utama (Pipeline Produksi)

---

### `extract_all_new_videos.py`

**Fungsi:** Mengekstraksi frame dari seluruh video baru Sesi 2 (09-11-2026) dan Sesi 3 (13-11-2026) secara batch.

**Input:**
- `dataset/dataset_new_video_09-11-2026/` → 58 video (C0467–C0545)
- `dataset/dataset_new_video_13-11-2026/` → 43 video (C0549–C0597)

**Output:**
- `dataset/dataset_video_frames_09-11-2026/` → **9.877 frame** dalam 58 folder
- `dataset/dataset_video_frames_13-11-2026/` → **16.181 frame** dalam 43 folder

**Parameter Kunci:**
```python
STEP = 5           # Ambil 1 frame setiap 5 frame (10 fps efektif dari 50 fps sumber)
FORMAT = "{vid}_f{idx:05d}_t{t:.2f}s.jpg"  # Penamaan file frame
```

**Cara Jalankan:**
```bash
python scripts/extract_all_new_videos.py
```

**Waktu Eksekusi:** ±5–8 menit (101 video, RTX 4050, SSD)

---

### `build_yolo_clean_dataset.py`

**Fungsi:** Membangun `dataset/dataset_yolo_clean` dengan anotasi organik otomatis dari 122 video (3 sesi). Ini adalah **jantung pipeline anotasi otomatis**.

**Input:**
- `dataset/dataset_video_frames_03-11-2026/` (21 video, 4.236 frame)
- `dataset/dataset_video_frames_09-11-2026/` (58 video, 9.877 frame)
- `dataset/dataset_video_frames_13-11-2026/` (43 video, 16.181 frame)

**Output:**
- `dataset/dataset_yolo_clean/images/train/` — 2.378 frame citra
- `dataset/dataset_yolo_clean/images/val/` — 546 frame citra
- `dataset/dataset_yolo_clean/labels/train/` — 2.378 label YOLO-seg
- `dataset/dataset_yolo_clean/labels/val/` — 546 label YOLO-seg
- `dataset/dataset_yolo_clean/data.yaml` — Konfigurasi dataset

**Algoritma Inti (`extract_plume_polygon`):**

```python
def extract_plume_polygon(base_bgr, act_bgr, v_name=""):
    # 1. Kompensasi pencahayaan (median area kanan atas 150×300px)
    offset = median(gray_act[:150,-300:]) - median(gray_base[:150,-300:])
    gray_act_adj = clip(gray_act - offset, 0, 255)
    
    # 2. Frame differencing
    diff = absdiff(gray_act_adj, gray_base)
    diff_blur = GaussianBlur(diff, (7,7), 0)
    
    # 3. Threshold: 14px (Sesi 1), 18px (Sesi 2 & 3)
    _, mist_mask = threshold(diff_blur, thresh_val, 255, BINARY)
    
    # 4. Deteksi nozzle apex per kolom
    for x in range(x_min, x_max+40, 2):
        if sum(core_diff[500:950, x] > 0) >= 3:
            best_x, best_y = x, 500 + median(ys)
            break
    
    # 5. Organic Cone Mask (meruncing di titik apex)
    apex_pts = [(best_x, best_y-2), (best_x+350, best_y-300),
                (w, 0), (w, h), (best_x+350, best_y+300), (best_x, best_y+2)]
    cone_mask = fillPoly(zeros, apex_pts, 255)
    
    # 6. Interseksi + Morphological Close
    spray_clean = bitwise_and(mist_mask, cone_mask)
    closed = morphologyEx(spray_clean, CLOSE, ellipse_13x13)
    
    # 7. Kontur terbesar → Poligon YOLO-seg
    c_main = max(contours, key=contourArea)
    return "0 " + " ".join(normalized_polygon_points) + "\n"
```

> [!IMPORTANT]
> **Perbedaan Konfigurasi Sesi:**
> - **Sesi 1 (03-11):** `thresh=14`, `x_min=30`, `x_max=100`, `y_default=720`, `floor_cut=810`
> - **Sesi 2 & 3 (09-11, 13-11):** `thresh=18`, `x_min=120`, `x_max=240`, `y_default=780`, `floor_cut=1020`

**Cara Jalankan:**
```bash
python scripts/build_yolo_clean_dataset.py
```

**Waktu Eksekusi:** ±15–25 menit (122 video, proses full)

---

### `train_yolo_clean.py`

**Fungsi:** Menjalankan fine-tuning YOLOv8s-seg via script Python (alternatif dari notebook).

> [!NOTE]
> Disarankan menggunakan **`finetune_spray_analysis.ipynb`** untuk kemudahan monitoring interaktif. Script ini digunakan jika ingin menjalankan training tanpa Jupyter.

**Parameter:**
```python
BASE_MODEL = "weights/spraybot_seg_finetuned.pt"
DATA_YAML  = "dataset/dataset_yolo_clean/data.yaml"
RUN_NAME   = "spraybot_seg_finetune_122vids"
EPOCHS     = 25
IMG_SIZE   = 1024
BATCH_SIZE = 8
WORKERS    = 0        # Windows: WAJIB 0!
LR0        = 0.005    # Fine-tuning: LR lebih rendah
LRF        = 0.01
FLIPLR     = 0.0      # Larangan flip horizontal
```

**Cara Jalankan:**
```bash
python scripts/train_yolo_clean.py
```

---

### `build_notebooks.py`

**Fungsi:** Generator otomatis konten Jupyter Notebook training (versi lama). Sudah digantikan oleh `create_finetune_notebook.py`.

---

### `create_finetune_notebook.py`

**Fungsi:** Generator notebook resmi `finetune_spray_analysis.ipynb` dan `finetune_spray_video_dataset.ipynb`.

**Output yang Dibuat:**
1. `finetune_spray_analysis.ipynb` — Notebook utama fine-tuning
2. `finetune_spray_video_dataset.ipynb` — Copy/alias untuk referensi
3. `weights/spraybot_seg_finetuned.pt` — Bobot di-copy dari `spraybot_seg_video_finetuned.pt` jika belum ada

**Cara Jalankan:**
```bash
python scripts/create_finetune_notebook.py
```

---

## ⚠️ Aturan Penting

> [!CAUTION]
> **WAJIB `workers=0` di Windows!** Jika `workers > 0`, proses Python akan crash dengan `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase` akibat mekanisme `spawn` multiprocessing Windows.

> [!WARNING]
> **WAJIB `fliplr=0.0`!** Seluruh dataset direkam dengan orientasi semprotan **Kiri → Kanan** secara konsisten. Jika flip horizontal diaktifkan, model akan belajar arah yang salah dan menyebabkan prediksi cermin/terbalik.

> [!IMPORTANT]
> **Gunakan `if __name__ == "__main__":`** di semua script yang memanggil `model.train()` agar kompatibel dengan sistem spawn multiprocessing Windows.

---

## 🔬 Riwayat Perbaikan Bug Kritis

### Bug 1: OpenCV Matrix Size Mismatch (`cv2.absdiff`)
**Error:** `cv2.error: (-209:Sizes of input arguments do not match)`  
**Penyebab:** Video `C0363` memiliki frame dengan resolusi campuran (mixed resolution).  
**Solusi:**
```python
if act_bgr.shape != base_bgr.shape:
    act_bgr = cv2.resize(act_bgr, (base_bgr.shape[1], base_bgr.shape[0]))
```

### Bug 2: Potongan Garis Lurus Vertikal (Knife Cut)
**Penyebab:** Logika lama menggunakan `glove = (B > 130) & (R > 130)` untuk mendeteksi sarung tangan putih, lalu memotong paksa dengan `bin_mask[:, :hand_cut_x] = 0`. Uap aerosol putih juga terkena deteksi ini.  
**Solusi:** Dihapus sepenuhnya, diganti dengan **Organic Nozzle Apex Cone Mask** yang meruncing alami.

### Bug 3: Windows Multiprocessing Crash
**Error:** `RuntimeError: An attempt has been made to start a new process...`  
**Solusi:** Selalu gunakan `workers=0` dan pastikan training dipanggil di dalam `if __name__ == "__main__":`.
