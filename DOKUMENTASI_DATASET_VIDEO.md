# DOKUMENTASI DATASET VIDEO SEMPROTAN AEROSOL & LIQUID MIST
## Proyek AI Spraybot — Riset MBKM Magang Industri PT Paragon Technology and Innovation & PT Makerindo Prima Solusi

> **Status Dataset:** 100% Terverifikasi & Seluruh Frame Selesai Diekstrak  
> **Direktori Sumber Video:** `dataset/dataset_new_video/`  
> **Direktori Frame Dataset:** `dataset/dataset_video_frames/`  
> **Script Ekstraksi Otomatis:** `scripts/extract_video_dataset.py`  

---

## 1. Ringkasan Eksekutif Dataset

Dataset ini merupakan kumpulan rekaman video resolusi tinggi (*High Definition*) berkecepatan tinggi (**50.0 FPS**) yang mendokumentasikan fenomena semprotan fluida (*spray plume dynamics*) dari berbagai produk kosmetik, perawatan kulit, dan wewangian produksi PT Paragon Technology and Innovation serta produk pembanding industri.

| Parameter Teknis | Nilai Spesifikasi | Keterangan |
|---|---|---|
| **Total File Video** | **21 File MP4** | 21 sesi uji semprotan independen |
| **Resolusi Spasial** | **1920 x 1080 (Full HD 1080p)** | Rasio 16:9 aspek rasio standar |
| **Kecepatan Tangkapan (FPS)** | **50.0 FPS** | High-speed shutter capture |
| **Format Kontainer & Codec** | **MP4 (H.264 / AVC)** | Kompatibel dengan OpenCV & PyTorch |
| **Total Durasi Rekaman** | **422.88 Detik (~7.05 Menit)** | Meliputi semprotan jet & kabut intermiten |
| **Total Frame Video Asli** | **21,144 Frame** | Frame mentah kontinu 50 FPS |
| **Sampling Ekstraksi (Step 5)** | **10.0 Frame per Detik (10 FPS)** | Menghindari redundancy & overfitting |
| **Total Frame Diekstrak** | **4,236 Frame Gambar (.jpg)** | 100% siap untuk dataset AI YOLO / SAM |
| **Total Ukuran File Video** | **~3,401 MB (3.32 GB)** | Disimpan lokal di `dataset/dataset_new_video` |

---

## 2. Anatomi & Struktur Penamaan File Video

Setiap file video dinamai secara terstruktur menggunakan pola standar industri:

```text
[KODE_KAMERA]-[TIPE_MEKANISME]-[MEREK_PRODUK]-[NOMOR_UJI/REPLIKASI].MP4
```

### Komponen Penamaan:
1. **`[KODE_KAMERA]` (`C0350` s/d `C0382`)**:  
   Identitas unik sekuensial rekaman dari kamera studio resolusi tinggi.
2. **`[TIPE_MEKANISME]`**:  
   * **`Gas`**: Semprotan aerosol bertekanan gas pendorong (*pressurized propellant*). Semburan berbentuk semburan jet kontinu berdensitas tinggi dengan kecepatan awal tinggi.
   * **`Normal`**: Semprotan pompa cairan mekanis non-gas (*finger-actuated mechanical pump / atomizer*). Semburan berbentuk kepulan kabut intermiten (*pulsed mist puff*) yang melambat di udara.
3. **`[MEREK_PRODUK]`**:  
   * **`Gatsby`**: Kaleng aerosol pria (produk pembanding kontrol gas).
   * **`Kahf`**: Botol semprot pria Paragon (formula wewangian cair Varian 1).
   * **`Kahf2`**: Botol semprot pria Paragon (formula wewangian cair Varian 2).
   * **`Wardah_Scentsation`**: Botol atomizer body mist wanita.
   * **`Wardah_SprayOn_Serum`**: Botol pompa serum wajah dengan partikel mikro.
   * **`Labora-Sensitive-Skincare`**: Kaleng aerosol skincare lembut untuk kulit sensitif.
4. **`[NOMOR_UJI/REPLIKASI]`**:  
   * `1`, `2`, `3`: Replikasi uji semprotan berulang pada spesimen yang sama.
   * `1.1`, `1.2`, `1.3`: Uji replikasi ke-1, 2, dan 3 pada **Botol Spesimen 1**.
   * `2.1`, `2.2`, `2.3`: Uji replikasi ke-1, 2, dan 3 pada **Botol Spesimen 2**.
   * `3.1`, `3.2`, `3.3`: Uji replikasi ke-1, 2, dan 3 pada **Botol Spesimen 3**.

---

## 3. Tabel Master Katalog Lengkap 21 File Video

| No | Nama File Video Asli | Kode | Mekanisme | Merek & Produk | Replikasi | Durasi | Frame Asli | Frame Ekstrak | Ukuran (MB) | Karakteristik Semprotan |
|:---:|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 01 | `C0350-Gas-Gatsby-1.MP4` | `C0350` | **Gas** | Gatsby | Take 1 | 11.52s | 576 | **116** | 128.27 MB | Aerosol deodorant gas bertekanan tinggi kaleng Gatsby. Semburan jet horizontal berkecepatan tinggi dengan kepulan uap pekat berkelanjutan (continuous jet). Replikasi 1. |
| 02 | `C0352-Normal-Kahf-1.MP4` | `C0352` | **Normal** | Kahf | Take 1 | 26.4s | 1320 | **264** | 192.27 MB | Body spray cairan Kahf (botol normal pompa/mist). Semburan semprotan kabut halus intermiten (pulsed mist puff) dengan ekspansi kerucut alami. Replikasi 1. |
| 03 | `C03540-Normal-Kahf-2.MP4` | `C03540` | **Normal** | Kahf | Take 2 | 24.0s | 1200 | **240** | 192.27 MB | Body spray cairan Kahf (botol normal pompa/mist). Semburan semprotan kabut halus intermiten berulang untuk uji konsistensi plume. Replikasi 2. |
| 04 | `C0355-Gas-Gatsby-2.MP4` | `C0355` | **Gas** | Gatsby | Take 2 | 13.44s | 672 | **135** | 128.27 MB | Aerosol deodorant gas kaleng Gatsby. Semburan jet horizontal berkecepatan tinggi, diuji ulang untuk validasi kestabilan aerosol stream. Replikasi 2. |
| 05 | `C0358-Normal-Wardah_Scentsation-1.MP4` | `C0358` | **Normal** | Wardah Scentsation | Take 1 | 24.96s | 1248 | **250** | 192.27 MB | Wardah Scentsation Body Mist (botol atomizer plastik transparan). Kabut wangi partikel halus berjangkauan menengah. Replikasi 1. |
| 06 | `C0360-Normal-Wardah_Scentsation-2.MP4` | `C0360` | **Normal** | Wardah Scentsation | Take 2 | 24.48s | 1224 | **245** | 192.27 MB | Wardah Scentsation Body Mist. Kabut semprotan intermiten, menunjukkan turbulensi udara pada fase dispersi. Replikasi 2. |
| 07 | `C0363-Normal-Wardah_Scentsation-3.MP4` | `C0363` | **Normal** | Wardah Scentsation | Take 3 | 23.52s | 1176 | **236** | 192.27 MB | Wardah Scentsation Body Mist. Uji semprotan ketiga untuk analisis variasi sudut sebaran (spray angle) dan jangkauan kabut. Replikasi 3. |
| 08 | `C0364-Gas-Gatsby-3.MP4` | `C0364` | **Gas** | Gatsby | Take 3 | 17.28s | 864 | **173** | 128.27 MB | Aerosol deodorant gas kaleng Gatsby. Evaluasi semprotan gas fase akhir dan peluruhan turbulensi plume. Replikasi 3. |
| 09 | `C0365-Normal-Wardah_SprayOn_Serum-1.1.MP4` | `C0365` | **Normal** | Wardah SprayOn Serum | Take 1.1 | 18.72s | 936 | **188** | 192.27 MB | Wardah Spray-On Serum Wajah (Spesimen Botol 1). Pompa presisi mikro untuk formula cair serum kental ringan. Replikasi 1. |
| 10 | `C0368-Normal-Wardah_SprayOn_Serum-1.2.MP4` | `C0368` | **Normal** | Wardah SprayOn Serum | Take 1.2 | 14.4s | 720 | **144** | 128.27 MB | Wardah Spray-On Serum Wajah (Spesimen Botol 1). Semburan semprotan mikro kedua untuk mengukur repeatability droplet. Replikasi 2. |
| 11 | `C0369-Normal-Wardah_SprayOn_Serum-1.3.MP4` | `C0369` | **Normal** | Wardah SprayOn Serum | Take 1.3 | 16.32s | 816 | **164** | 128.27 MB | Wardah Spray-On Serum Wajah (Spesimen Botol 1). Semburan semprotan mikro ketiga untuk validasi stabilitas nosel. Replikasi 3. |
| 12 | `C0370-Normal-Wardah_SprayOn_Serum-2.1.MP4` | `C0370` | **Normal** | Wardah SprayOn Serum | Take 2.1 | 16.32s | 816 | **164** | 128.27 MB | Wardah Spray-On Serum Wajah (Spesimen Botol 2). Uji variasi antar-botol kemasan serum. Replikasi 1. |
| 13 | `C0372-Normal-Wardah_SprayOn_Serum-2.2.MP4` | `C0372` | **Normal** | Wardah SprayOn Serum | Take 2.2 | 15.84s | 792 | **159** | 128.27 MB | Wardah Spray-On Serum Wajah (Spesimen Botol 2). Semburan kabut serum mikro kedua. Replikasi 2. |
| 14 | `C0373-Normal-Wardah_SprayOn_Serum-2.3.MP4` | `C0373` | **Normal** | Wardah SprayOn Serum | Take 2.3 | 15.36s | 768 | **154** | 128.27 MB | Wardah Spray-On Serum Wajah (Spesimen Botol 2). Semburan kabut serum mikro ketiga. Replikasi 3. |
| 15 | `C0374-Normal-Wardah_SprayOn_Serum-3.1.MP4` | `C0374` | **Normal** | Wardah SprayOn Serum | Take 3.1 | 14.4s | 720 | **144** | 128.27 MB | Wardah Spray-On Serum Wajah (Spesimen Botol 3). Uji toleransi nosel botol ketiga untuk memverifikasi keseragaman fabrikasi kemasan. Replikasi 1. |
| 16 | `C0375-Normal-Wardah_SprayOn_Serum-3.2.MP4` | `C0375` | **Normal** | Wardah SprayOn Serum | Take 3.2 | 14.4s | 720 | **144** | 128.27 MB | Wardah Spray-On Serum Wajah (Spesimen Botol 3). Semburan kabut serum mikro kedua botol 3. Replikasi 2. |
| 17 | `C0376-Normal-Wardah_SprayOn_Serum-3.3.MP4` | `C0376` | **Normal** | Wardah SprayOn Serum | Take 3.3 | 13.92s | 696 | **140** | 128.27 MB | Wardah Spray-On Serum Wajah (Spesimen Botol 3). Semburan kabut serum mikro ketiga botol 3. Replikasi 3. |
| 18 | `C0378-Gas-Labora-Sensitive-Skincare-1.MP4` | `C0378` | **Gas** | Labora | Take Sensitive | 31.2s | 1560 | **312** | 256.28 MB | Labora Sensitive Skincare (Aerosol Canister Bertekanan Gas). Menghasilkan micro-fine thermal mist/skincare spray ultra-halus untuk kulit sensitif. Replikasi 1. |
| 19 | `C0379-Normal-Kahf2-1.MP4` | `C0379` | **Normal** | Kahf2 | Take 1 | 28.8s | 1440 | **288** | 256.28 MB | Body Spray Kahf Varian 2 (Formula Baru / Botol Alternatif). Semprotan kabut wangi intermiten dengan dinamika partikel berbeda dari Varian 1. Replikasi 1. |
| 20 | `C0381-Normal-Kahf2-2.MP4` | `C0381` | **Normal** | Kahf2 | Take 2 | 28.8s | 1440 | **288** | 256.28 MB | Body Spray Kahf Varian 2. Pengujian semprotan berulang untuk mengamati profil sebaran spray plume. Replikasi 2. |
| 21 | `C0382-Normal-Kahf2-3.MP4` | `C0382` | **Normal** | Kahf2 | Take 3 | 28.8s | 1440 | **288** | 256.28 MB | Body Spray Kahf Varian 2. Pengujian semprotan ketiga untuk mengukur konsistensi volume dan durasi semprotan. Replikasi 3. |

---

## 4. Klasifikasi Berdasarkan Kategori & Mekanisme Semprotan

### A. Kategori Aerosol Gas Tekanan Tinggi (`Gas`) — Total 4 Video (735 Frame Diekstrak)
Karakteristik: Tekanan propelan gas tinggi, kecepatan semprotan tinggi (*jet velocity*), pola semburan berbentuk kerucut sempit kontinu (*continuous core jet*), partikel padat dan stabil sepanjang durasi penekanan aktuator.

1. **Gatsby Aerosol Deodorant (3 Video):**
   - `C0350-Gas-Gatsby-1.MP4` (11.52 detik, 116 frame terekstrak)
   - `C0355-Gas-Gatsby-2.MP4` (13.44 detik, 135 frame terekstrak)
   - `C0364-Gas-Gatsby-3.MP4` (17.28 detik, 173 frame terekstrak)
2. **Labora Sensitive Skincare Fine Mist (1 Video):**
   - `C0378-Gas-Labora-Sensitive-Skincare-1.MP4` (31.20 detik, 312 frame terekstrak) — Partikel aerosol ultra-lembut berkecepatan sedang khusus perawatan kulit wajah sensitif.

### B. Kategori Pompa Cairan / Atomizer Normal (`Normal`) — Total 17 Video (3,501 Frame Diekstrak)
Karakteristik: Pompa manual intermiten (*mechanical finger pump*), menghasilkan semprotan kabut (*mist puff*) yang mengembang luas, kecepatan semprotan meluruh dengan cepat di udara (*rapid deceleration*), membentuk riak turbulensi yang khas.

1. **Kahf Body Spray Varian 1 (2 Video):**
   - `C0352-Normal-Kahf-1.MP4` (26.40 detik, 264 frame terekstrak)
   - `C03540-Normal-Kahf-2.MP4` (24.00 detik, 240 frame terekstrak)
2. **Kahf Body Spray Varian 2 (3 Video):**
   - `C0379-Normal-Kahf2-1.MP4` (28.80 detik, 288 frame terekstrak)
   - `C0381-Normal-Kahf2-2.MP4` (28.80 detik, 288 frame terekstrak)
   - `C0382-Normal-Kahf2-3.MP4` (28.80 detik, 288 frame terekstrak)
3. **Wardah Scentsation Body Mist (3 Video):**
   - `C0358-Normal-Wardah_Scentsation-1.MP4` (24.96 detik, 250 frame terekstrak)
   - `C0360-Normal-Wardah_Scentsation-2.MP4` (24.48 detik, 245 frame terekstrak)
   - `C0363-Normal-Wardah_Scentsation-3.MP4` (23.52 detik, 236 frame terekstrak)
4. **Wardah Spray-On Serum Wajah — Multi-Specimen Rigorous Test (9 Video):**
   - **Botol Spesimen 1:** `C0365` (188 f), `C0368` (144 f), `C0369` (164 f)
   - **Botol Spesimen 2:** `C0370` (164 f), `C0372` (159 f), `C0373` (154 f)
   - **Botol Spesimen 3:** `C0374` (144 f), `C0375` (144 f), `C0376` (140 f)

---

## 5. Struktur Penyimpanan Dataset Frame Hasil Ekstraksi

Seluruh frame diekstrak dan ditata secara modular pada subfolder masing-masing di dalam direktori `dataset/dataset_video_frames/`:

```text
dataset/
├── dataset_new_video/                     # Direktori 21 file video asli (.MP4)
│   ├── C0350-Gas-Gatsby-1.MP4
│   ├── C0352-Normal-Kahf-1.MP4
│   └── ... (21 video)
│
└── dataset_video_frames/                    # Direktori master hasil ekstraksi frame
    ├── C0350-Gas-Gatsby-1/
    │   ├── C0350-Gas-Gatsby-1_f00000_t00.00s.jpg
    │   ├── C0350-Gas-Gatsby-1_f00005_t00.10s.jpg
    │   ├── C0350-Gas-Gatsby-1_f00010_t00.20s.jpg
    │   └── ... (116 frame)
    ├── C0352-Normal-Kahf-1/               (264 frame)
    ├── C03540-Normal-Kahf-2/              (240 frame)
    ├── C0355-Gas-Gatsby-2/                (135 frame)
    ├── C0358-Normal-Wardah_Scentsation-1/ (250 frame)
    ├── C0360-Normal-Wardah_Scentsation-2/ (245 frame)
    ├── C0363-Normal-Wardah_Scentsation-3/ (236 frame)
    ├── C0364-Gas-Gatsby-3/                (173 frame)
    ├── C0365-Normal-Wardah_SprayOn_Serum-1.1/ (188 frame)
    ├── C0368-Normal-Wardah_SprayOn_Serum-1.2/ (144 frame)
    ├── C0369-Normal-Wardah_SprayOn_Serum-1.3/ (164 frame)
    ├── C0370-Normal-Wardah_SprayOn_Serum-2.1/ (164 frame)
    ├── C0372-Normal-Wardah_SprayOn_Serum-2.2/ (159 frame)
    ├── C0373-Normal-Wardah_SprayOn_Serum-2.3/ (154 frame)
    ├── C0374-Normal-Wardah_SprayOn_Serum-3.1/ (144 frame)
    ├── C0375-Normal-Wardah_SprayOn_Serum-3.2/ (144 frame)
    ├── C0376-Normal-Wardah_SprayOn_Serum-3.3/ (140 frame)
    ├── C0378-Gas-Labora-Sensitive-Skincare-1/ (312 frame)
    ├── C0379-Normal-Kahf2-1/              (288 frame)
    ├── C0381-Normal-Kahf2-2/              (288 frame)
    └── C0382-Normal-Kahf2-3/              (288 frame)
```

### Format Penamaan File Frame:
`[Nama_Video]_f[Index_Frame_5_Digit]_t[Timestamp_Detik]s.jpg`  
*Contoh:* `C0350-Gas-Gatsby-1_f00090_t01.80s.jpg` = Cuplikan dari video C0350, frame ke-90 pada detik 1.80.

---

## 6. Panduan Menjalankan Script Ekstraksi Ulang

Jika di kemudian hari diperlukan ekstraksi dengan sampling frame yang berbeda (misalnya mengekstrak 100% seluruh frame `step=1`, atau `step=10`), script `scripts/extract_video_dataset.py` dapat dijalankan langsung melalui terminal:

```bash
# 1. Ekstraksi standar (10 FPS / step=5):
.venv\Scripts\python scripts/extract_video_dataset.py --step 5 --quality 95

# 2. Ekstraksi 100% seluruh frame asli tanpa jeda (50 FPS / step=1):
.venv\Scripts\python scripts/extract_video_dataset.py --step 1 --quality 95

# 3. Ekstraksi ringan (5 FPS / step=10):
.venv\Scripts\python scripts/extract_video_dataset.py --step 10 --quality 90
```

---
*Dokumentasi ini dibuat otomatis dan terverifikasi secara presisi untuk keperluan riset AI Spraybot MBKM Magang Industri Paragon.*