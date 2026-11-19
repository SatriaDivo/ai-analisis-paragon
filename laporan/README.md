# 📁 Laporan & Dokumen Teknis Riset AI Spraybot

Direktori ini berisi seluruh dokumen laporan kemajuan riset, laporan teknis, dan dokumen referensi konseptual terkait sistem analisis semprotan fluida (*spray plume characterization*) berbasis Computer Vision & Deep Learning (YOLOv8-seg).

---

## 📑 Daftar Dokumen Utama

### 1. Dokumen Teknis Resmi (Konseptual & Teori)
Dokumen teknis komprehensif tanpa memasukkan persentase hasil run (bersifat murni teoritis, formulasi matematika, dan penjelasan istilah teknik evaluasi):

- **Format Microsoft Word (.docx)**:
  👉 [Dokumen_Teknis_Metrik_dan_Evaluasi_AI_YOLO.docx](Dokumen_Teknis_Metrik_dan_Evaluasi_AI_YOLO.docx) *(51 KB, siap unduh dan baca di Microsoft Word / Google Docs)*
- **Format Markdown (.md)**:
  👉 [DOKUMEN_TEKNIS_METRIK_AI.md](DOKUMEN_TEKNIS_METRIK_AI.md) *(Pratinjau cepat di VS Code atau Git)*

#### Cakupan Bab Dokumen Teknis:
1. **Bab 1**: Pendahuluan & Paradigma Computer Vision (*Classification* vs *Object Detection* vs *Instance Segmentation*, Bounding Box vs Mask, Karakteristik Fluida Semprotan)
2. **Bab 2**: Landasan Pengukuran Spasial (Ground Truth vs Prediksi, Formulasi IoU / Jaccard Index, $IoU_{box}$ vs $IoU_{mask}$, Ambang Strictness IoU)
3. **Bab 3**: Metrik Evaluasi Klasifikasi & Deteksi Biner (True Positive, False Positive, False Negative, True Negative, Precision, Recall, F1-Score, Trade-off Dilemma, P-R Curve)
4. **Bab 4**: Mean Average Precision (mAP) sebagai Standar Emas (AP & AUC Interpolasi Kontinu, mAP, mAP@50, mAP@50-95 COCO, Komparasi mAP Box vs Mask)
5. **Bab 5**: Fungsi Kerugian (*Loss Functions*) Arsitektur YOLOv8 ($L_{box}$ CIoU, $L_{dfl}$ Distribution Focal, $L_{seg}$ BCE Mask, $L_{cls}$ Classification)
6. **Bab 6**: Parameter Operasional Inferensi & Post-Processing (*Confidence Threshold*, *Non-Maximum Suppression / NMS*, *Image Size / imgsz*, *Mask Ratio*)
7. **Bab 7**: Matriks Evaluasi Lanjutan & Relevansi Fisik Industri (*Confusion Matrix*, *Background Negative Handling*, Implikasi terhadap *Spray Angle* & *Penetration Length*, Glosarium Istilah Singkat)

---

### 2. Arsip Laporan Kemajuan Magang Industri (MBKM)
Laporan kemajuan berkala pelaksanaan MBKM Magang Industri di PT Paragon Technology and Innovation:

| File Laporan | Tanggal | Deskripsi |
|---|---|---|
| `Laporan_Kemajuan_Paragon_Spray_2026-10-14.docx` | 14 Oktober 2026 | Laporan kemajuan sesi awal riset |
| `Laporan_Kemajuan_Paragon_Spray_Semprotan_Memanjang.docx` | 14 Oktober 2026 | Analisis karakteristik semprotan memanjang |
| `Laporan_Kemajuan_Paragon_Spray_v2_Anotasi.docx` | 14 Oktober 2026 | Standarisasi dan metodologi anotasi v2 |
| `Laporan_Kemajuan_Paragon_Spray_v2_Hasil_Pelatihan.docx` | 14 Oktober 2026 | Hasil pelatihan model baseline v2 |

---
*Diperbarui: 19 November 2026 | Model Aktif: `training_awal/spraybot_seg_video_finetuned.pt` | Dataset v2.0: 3.493 frame*
