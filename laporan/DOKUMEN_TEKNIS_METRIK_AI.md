# DOKUMEN TEKNIS METRIK DAN EVALUASI COMPUTER VISION
## Arsitektur Object Detection & Instance Segmentation (YOLOv8) pada Analisis Karakterisasi Spray Fluida

---

**Informasi Dokumen**
- **Klasifikasi**: Dokumen Teknis & Glosarium Konseptual (*Technical Reference Manual*)
- **Domain**: Computer Vision, Deep Learning, Object Detection, Instance Segmentation
- **Arsitektur Referensi**: YOLOv8 (You Only Look Once versi 8 - Segmentasi)
- **Konteks Rekayasa**: Karakterisasi Morfologi dan Dinamika Fluida Semprotan (*Fluid Spray Pattern & Plume Characterization*)
- **Status Dokumen**: *Official Technical Documentation* (Non-Reporting Reference)

---

## DAFTAR ISI

1. [BAB 1: PENDAHULUAN & PARADIGMA COMPUTER VISION](#bab-1-pendahuluan--paradigma-computer-vision)
   - 1.1 Perbedaan Klasifikasi Citra, Object Detection, dan Instance Segmentation
   - 1.2 Anatomi Output Segmentasi: Bounding Box vs. Polygon Mask
   - 1.3 Karakteristik Khusus Objek Fluida Semprotan (*Spray Plume & Cone*)
2. [BAB 2: LANDASAN PENGUKURAN SPASIAL (SPATIAL OVERLAP METRICS)](#bab-2-landasan-pengukuran-spasial-spatial-overlap-metrics)
   - 2.1 Ground Truth vs. Prediksi Model
   - 2.2 Intersection over Union (IoU) / Jaccard Index
   - 2.3 IoU Bounding Box ($IoU_{box}$) vs. IoU Segmentation Mask ($IoU_{mask}$)
   - 2.4 Tingkat Ketat Spasial (*Strictness Level*) Berdasarkan Ambang Batas IoU
3. [BAB 3: METRIK EVALUASI KLASIFIKASI & DETEKSI BINER](#bab-3-metrik-evaluasi-klasifikasi--deteksi-biner)
   - 3.1 Taksonomi Deteksi: True Positive (TP), False Positive (FP), False Negative (FN), True Negative (TN)
   - 3.2 Precision (Presisi)
   - 3.3 Recall (Sensitivitas / Tingkat Temuan)
   - 3.4 Kompromi Precision vs. Recall (*Trade-off Dilemma*)
   - 3.5 F1-Score (Rata-rata Harmonik)
   - 3.6 Kurva Precision-Recall (*P-R Curve*)
4. [BAB 4: MEAN AVERAGE PRECISION (mAP) SEBAGAI STANDAR EMAS](#bab-4-mean-average-precision-map-sebagai-standar-emas)
   - 4.1 Average Precision (AP) dan Perhitungan Area Under Curve (AUC)
   - 4.2 Mean Average Precision (mAP)
   - 4.3 mAP@50 (mAP pada Ambang IoU 0.50)
   - 4.4 mAP@50-95 (Standar COCO: Ambang IoU 0.50 hingga 0.95)
   - 4.5 Analisis Komparatif: mAP (Box) vs. mAP (Mask / Segmentation)
5. [BAB 5: FUNGSI KERUGIAN (LOSS FUNCTIONS) DALAM ARSITEKTUR YOLOV8](#bab-5-fungsi-kerugian-loss-functions-dalam-arsitektur-yolov8)
   - 5.1 Peran Loss Function dalam Optimasi Backpropagation
   - 5.2 Bounding Box Loss ($L_{box}$ / CIoU Loss)
   - 5.3 Distribution Focal Loss ($L_{dfl}$)
   - 5.4 Segmentation Mask Loss ($L_{seg}$ / BCE Mask Loss)
   - 5.5 Classification Loss ($L_{cls}$ / BCE Cls Loss)
6. [BAB 6: PARAMETER OPERASIONAL INFERENSI & POST-PROCESSING](#bab-6-parameter-operasional-inferensi--post-processing)
   - 6.1 Confidence Threshold ($conf$)
   - 6.2 Non-Maximum Suppression (NMS) & IoU Threshold ($iou$)
   - 6.3 Resolusi Spasial Citra Input ($imgsz$)
   - 6.4 Mask Ratio dan Downsampling Prototype Mask
7. [BAB 7: MATRIKS EVALUASI LANJUTAN & RELEVANSI FISIK INDUSTRI](#bab-7-matriks-evaluasi-lanjutan--relevansi-fisik-industri)
   - 7.1 Confusion Matrix & Normalized Confusion Matrix
   - 7.2 Latar Belakang Negatif (*Background False Positives vs. Background False Negatives*)
   - 7.3 Dampak Kesalahan Metrik terhadap Ekstraksi Parameter Fisik Fluida
   - 7.4 Glosarium Istilah Singkat (*Quick Reference Table*)

---

## BAB 1: PENDAHULUAN & PARADIGMA COMPUTER VISION

### 1.1 Perbedaan Klasifikasi Citra, Object Detection, dan Instance Segmentation

Dalam bidang penglihatan komputer (*computer vision*), terdapat evolusi metodologis dalam mengenali dan melokalisasi informasi visual:

```
+------------------------------------------------------------------------------------+
|                               EVOLUSI TUGAS VISI KOMPUTER                          |
+------------------------------------------------------------------------------------+
| 1. Image Classification   : "Ada semprotan di dalam gambar ini." (Tanpa lokasi)    |
| 2. Object Detection       : "Semprotan berada di kotak [x, y, w, h]."              |
| 3. Semantic Segmentation  : "Piksel-piksel ini adalah semprotan, sisanya latar."   |
| 4. Instance Segmentation  : "Piksel-piksel ini semprotan A, piksel ini semprotan B"|
+------------------------------------------------------------------------------------+
```

1. **Image Classification (Klasifikasi Citra)**:
   Model hanya memprediksi label kategori dari keseluruhan citra secara global tanpa mengetahui letak koordinat spasial objek.
2. **Object Detection (Deteksi Objek)**:
   Model mengidentifikasi keberadaan objek sekaligus melokalisasi posisinya menggunakan representasi kotak pembatas berupa *Bounding Box* tegak lurus sumbu (*Axis-Aligned Bounding Box / AABB*). Output berupa koordinat 4-tupel: $(x_{center}, y_{center}, width, height)$.
3. **Semantic Segmentation (Segmentasi Semantik)**:
   Model melakukan pelabelan tingkat piksel (*pixel-level labeling*) untuk setiap piksel pada gambar. Namun, jika ada dua objek berdampingan dari kelas yang sama, model menganggap keduanya sebagai satu kesatuan massa tanpa membedakan entitas individu.
4. **Instance Segmentation (Segmentasi Instansia)**:
   Paradigma tertinggi yang menggabungkan deteksi objek dan segmentasi. Model melokalisasi setiap objek secara individual (dengan *bounding box*) dan secara simultan memetakan batas kontur piksel presisi (*pixel-precise mask*) untuk masing-masing instansia objek secara terpisah.

---

### 1.2 Anatomi Output Segmentasi: Bounding Box vs. Polygon Mask

Model segmentasi instansia modern (seperti YOLOv8-seg) menghasilkan dua lapis representasi spasial secara paralel:

```
+-----------------------------------------------------------------------------+
|               BOUNDING BOX                    |       POLYGON MASK          |
+-----------------------------------------------+-----------------------------+
| [Bentuk]                                      | [Bentuk]                    |
|  +-------------------------------+            |          . . . .            |
|  |       (Latar Belakang Kosong) |            |       . '       ' .         |
|  |           / \                 |            |      /   SEM-    \        |
|  |          /   \  Objek Semprot |            |     /   PROTAN    \       |
|  |         /     \               |            |    /               \      |
|  |        /       \              |            |   /                 \     |
|  +-------------------------------+            |  ' . . . . . . . . . '      |
|                                               |                             |
| [Karakteristik]                               | [Karakteristik]             |
| - Koordinat 4 titik persegi panjang           | - Array biner piksel 2D /   |
| - Mengikutsertakan piksel latar belakang      |   koordinat poligon tertutup|
| - Cepat dihitung, namun aproksimasi kasar     | - Mengisolasi batas riil    |
| - Digunakan untuk lokalisasi global           | - Basis hitungan geometri   |
+-----------------------------------------------+-----------------------------+
```

1. **Bounding Box (Kotak Pembatas Spasial)**:
   - Representasi geometris ortogonal yang melingkupi batas terluar objek secara minimal.
   - Bersifat aproksimatif karena mencakup area non-objek (ruang kosong latar belakang) terutama pada objek yang berbentuk konikal, menyudut, atau asimetris.
2. **Segmentation Mask (Topeng Segmentasi Piksel)**:
   - Matriks biner $M \in \{0, 1\}^{H \times W}$ di mana nilai $1$ merepresentasikan piksel milik objek dan nilai $0$ adalah latar belakang.
   - Menangkap geometri sebenarnya dari kontur permukaan objek dengan akurasi sub-milimeter visual.

---

### 1.3 Karakteristik Khusus Objek Fluida Semprotan (*Spray Plume & Cone*)

Objek semprotan fluida (*spray plume / spray cone*) memiliki karakteristik visual yang sangat berbeda dari objek kaku (*rigid objects*) seperti mobil atau manusia:

1. **Batas Gradual dan Transparan (*Soft / Fuzzy Boundary*)**:
   Tepian semprotan bukan berupa garis tegas, melainkan kabut butiran tetesan (*droplets*) mikro yang densitasnya menurun secara bertahap mendekati udara bebas.
2. **Dinamika Bentuk Temporal (*Non-Rigid Geometry*)**:
   Geometri semprotan berubah secara dinamis setiap milidetik tergantung pada tekanan aktuator nozzle, fase semprotan (*developing stage*, *fully-developed stage*, hingga *collapse stage*), dan turbulensi udara.
3. **Pentingnya Kontur Halus untuk Analisis Fisik**:
   Dalam rekayasa fluida, luasan piksel mask semprotan digunakan untuk menghitung sudut semprot (*spray angle*), panjang penetrasi semprotan (*spray penetration length*), dan simetri penyebaran kabut. Kegagalan segmentasi piksel akan merusak akurasi pengukuran sudut dan luasan.

---

## BAB 2: LANDASAN PENGUKURAN SPASIAL (SPATIAL OVERLAP METRICS)

Evaluasi model deteksi dan segmentasi tidak dapat menggunakan akurasi klasifikasi biasa karena model harus dinilai berdasarkan **ketepatan lokasi spasial** di samping ketepatan kelas.

```
       Ground Truth (Anotasi Ahli)             Prediksi Model AI
       +-----------------------+               +-----------------------+
       |   Area Sebenarnya     |               |   Area Hasil Model    |
       |          A            |               |          B            |
       +-----------------------+               +-----------------------+
```

---

### 2.1 Ground Truth vs. Prediksi Model

- **Ground Truth ($GT$ / $A$)**:
  Data kebenaran mutlak yang dianotasi secara manual oleh manusia ahli. Berupa koordinat batas kotak atau poligon piksel sebenarnya dari objek.
- **Prediksi Model ($Pred$ / $B$)**:
  Hasil inferensi algoritma visi komputer, mencakup lokasi koordinat, estimasi kontur piksel, skor keyakinan (*confidence score*), dan prediksi label kelas.

---

### 2.2 Intersection over Union (IoU) / Jaccard Index

**Intersection over Union (IoU)**, atau dikenal dalam teori himpunan sebagai *Jaccard Index*, adalah ukuran kuantitatif dari tingkat tumpang tindih (*overlap*) antara area ground truth dengan area prediksi.

#### Formulasi Matematika:

$$\text{IoU} = \frac{\text{Area of Overlap (Irisan)}}{\text{Area of Union (Gabungan)}} = \frac{|A \cap B|}{|A \cup B|}$$

Di mana:
- $|A \cap B|$ adalah luas area irisan (perpotongan) di mana kedua bidang saling bertumpuk secara bersamaan.
- $|A \cup B| = |A| + |B| - |A \cap B|$ adalah total luas area gabungan yang dicakup oleh salah satu atau kedua bidang tersebut.

```
+-----------------------------------------------------------------------------+
| VISUALISASI PERHITUNGAN INTERSECTION OVER UNION (IoU)                       |
+-----------------------------------------------------------------------------+
|                                                                             |
|   +-------------------+                                                     |
|   | Ground Truth (A)  |                                                     |
|   |         +---------+---------+                                           |
|   |         | ####### |         |        Area Berarsir (#######) : Overlap  |
|   +---------+---------+         |        Total Seluruh Bidang    : Union    |
|             | Prediksi (B)      |                                           |
|             +-------------------+        Nilai IoU = Overlap / Union        |
|                                                                             |
+-----------------------------------------------------------------------------+
```

#### Sifat Nilai IoU:
- $0 \le \text{IoU} \le 1$
- $\text{IoU} = 0$: Tidak ada tumpang tindih sama sekali (prediksi meleset total).
- $\text{IoU} = 1$: Prediksi berhimpit sempurna dengan ground truth (overlap 100%).

---

### 2.3 IoU Bounding Box ($IoU_{box}$) vs. IoU Segmentation Mask ($IoU_{mask}$)

Dalam model multi-task seperti YOLOv8-seg, IoU dihitung secara independen pada dua domain geometri:

| Parameter | $IoU_{box}$ (Bounding Box) | $IoU_{mask}$ (Segmentation Mask) |
| :--- | :--- | :--- |
| **Domain Geometri** | Luas segiempat 2D kontinu $[\text{pixel}^2]$ | Himpunan piksel diskret biner $\sum_{(x,y)} P(x,y)$ |
| **Metode Hitung** | Geometri analitik perpotongan koordinat $[x_1, y_1, x_2, y_2]$ | Operasi logika piksel $\frac{\sum (M_{GT} \land M_{Pred})}{\sum (M_{GT} \lor M_{Pred})}$ |
| **Sensitivitas Batas** | Rendah (hanya melihat batas terluar persegi panjang) | Sangat Tinggi (setiap piksel di batas lekukan dinilai) |
| **Karakter Nilai** | Cenderung lebih tinggi karena toleransi kotak lebar | Cenderung lebih rendah karena batas bergelombang |

---

### 2.4 Tingkat Ketat Spasial (*Strictness Level*) Berdasarkan Ambang Batas IoU

Untuk menentukan apakah suatu deteksi dinyatakan "berhasil" atau "gagal", sistem menerapkan ambang batas IoU threshold ($\tau_{IoU}$):

$$\text{Status Deteksi} = \begin{cases} \text{Valid (Tumpang Tindih Diterima)}, & \text{jika } \text{IoU} \ge \tau_{IoU} \\ \text{Salah / Meleset (Tolak)}, & \text{jika } \text{IoU} < \tau_{IoU} \end{cases}$$

- **Ambang Batas $\tau_{IoU} = 0.50$ (Toleransi Moderat)**:
  Standar evaluasi dasar (PASCAL VOC). Deteksi dinyatakan benar jika separuh dari area prediksi tumpang tindih dengan ground truth.
- **Ambang Batas $\tau_{IoU} = 0.75$ (Ketat)**:
  Digunakan untuk mengevaluasi apakah model mampu menempel rapat pada batas fisik objek.
- **Ambang Batas $\tau_{IoU} = 0.90 - 0.95$ (Sangat Ekstrem / Presisi Tinggi)**:
  Menuntut kecocokan piksel hampir tanpa toleransi kesalahan tepi.

---

## BAB 3: METRIK EVALUASI KLASIFIKASI & DETEKSI BINER

Setelah tumpang tindih spasial dinilai dengan IoU, setiap keluaran model diklasifikasikan ke dalam taksonomi deteksi biner.

---

### 3.1 Taksonomi Deteksi: TP, FP, FN, dan TN

```
+------------------------------------+------------------------------------+
|                                    |            KONDISI AKTUAL          |
|                                    |       (Ground Truth Lapangan)      |
|                                    +------------------+-----------------+
|                                    | Ada Objek (Pos)  | Tdk Ada Obj(Neg)|
+------------------+-----------------+------------------+-----------------+
| KEPUTUSAN        | Deteksi Dibuat  |  TRUE POSITIVE   | FALSE POSITIVE  |
| MODEL PREDIKSI   | (Prediksi Pos)  |      (TP)        |      (FP)       |
| (Inferensi AI)   +-----------------+------------------+-----------------+
|                  | Tdk Ada Deteksi |  FALSE NEGATIVE  |  TRUE NEGATIVE  |
|                  | (Prediksi Neg)  |      (FN)        |      (TN)       |
+------------------+-----------------+------------------+-----------------+
```

1. **True Positive (TP)**:
   Model mendeteksi adanya objek, dan prediksi tersebut memang benar ada di ground truth dengan tumpang tindih spasial memenuhi syarat ($\text{IoU} \ge \tau_{IoU}$).
   *Analogi Spray*: Model menemukan semprotan fluida, dan lokasinya pas pada kabut semprotan yang sebenarnya.

2. **False Positive (FP) / Alarm Palsu (*Type I Error*)**:
   Model memprediksi adanya objek, tetapi di lokasi tersebut tidak ada objek pada ground truth, ATAU ada objek namun lokasi spasialnya meleset di bawah ambang batas ($\text{IoU} < \tau_{IoU}$), ATAU merupakan deteksi ganda terhadap objek yang sama.
   *Analogi Spray*: Pantulan cahaya pada latar belakang atau bayangan mesin dikira semprotan oleh model.

3. **False Negative (FN) / Luput (*Type II Error*)**:
   Objek sebenarnya ada pada ground truth, namun model gagal mendeteksinya sama sekali (terlewat).
   *Analogi Spray*: Ada kabut semprotan tipis di nozzle, namun model mengabaikannya dan menganggap hanya latar kosong.

4. **True Negative (TN)**:
   Area latar belakang yang memang tidak ada objek dan model dengan benar tidak mengeluarkan deteksi apa pun di area tersebut. Dalam deteksi objek berbasis *bounding box*, jumlah kemungkinan kotak latar belakang tak berhingga, sehingga TN biasanya tidak dihitung secara eksplisit dalam formulasi metrik utama.

---

### 3.2 Precision (Presisi)

**Precision** mengukur tingkat kemurnian atau keandalan dari deteksi yang dihasilkan oleh model.

#### Formulasi:

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{\text{Jumlah Deteksi Benar}}{\text{Total Seluruh Deteksi yang Dibuat Model}}$$

#### Makna Konseptual:
- Menjawab pertanyaan: *"Ketika model mengklaim menemukan semprotan fluida, berapa persen klaim tersebut yang benar-benar akurat dan bukan alarm palsu?"*
- Nilai Precision yang tinggi berarti model sangat berhati-hati (*conservative*) dan jarang menghasilkan deteksi salah/sampah.

---

### 3.3 Recall (Sensitivitas / Tingkat Temuan)

**Recall** mengukur kelengkapan model dalam menemukan seluruh objek yang sesungguhnya ada di dunia nyata.

#### Formulasi:

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{\text{Jumlah Deteksi Benar}}{\text{Total Seluruh Objek yang Sebenarnya Ada di Lapangan}}$$

#### Makna Konseptual:
- Menjawab pertanyaan: *"Dari seluruh semprotan yang muncul di depan kamera, berapa persen yang berhasil ditemukan oleh model dan tidak terlewat?"*
- Nilai Recall yang tinggi berarti model sangat tanggap (*aggressive*) dan tidak membiarkan ada semprotan yang lolos tanpa terdeteksi.

---

### 3.4 Kompromi Precision vs. Recall (*Trade-off Dilemma*)

Terdapat hubungan tarik-menarik (*trade-off*) yang inheren antara Precision dan Recall yang dikendalikan oleh ambang batas keyakinan (*confidence threshold*):

```
+-----------------------------------------------------------------------------+
| AMBANG KEYAKINAN TINGGI (Conf = 0.85)   | AMBANG KEYAKINAN RENDAH (Conf = 0.15)|
+-----------------------------------------+-----------------------------------+
| - Model hanya memprediksi jika yakin    | - Model memprediksi semua kandidat|
| - Sedikit alarm palsu (FP sangat kecil) | - Tidak ada objek lolos (FN kecil)|
| - Akibat: Sering melewatkan objek tipis | - Akibat: Banyak alarm palsu (FP) |
| - Hasil: PRECISION TINGGI, RECALL RENDAH| - Hasil: RECALL TINGGI, PRECISION |
|                                         |          RENDAH                   |
+-----------------------------------------+-----------------------------------+
```

---

### 3.5 F1-Score (Rata-rata Harmonik)

F1-Score adalah ukuran tunggal yang menggabungkan Precision dan Recall menggunakan rata-rata harmonik (*harmonic mean*). Rata-rata harmonik digunakan (bukan rata-rata aritmatika biasa) agar jika salah satu metrik bernilai sangat rendah, F1-Score akan langsung jatuh terhukum.

#### Formulasi:

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

F1-Score mencapai nilai optimal ketika terdapat keseimbangan yang baik antara kemampuan model menangkap seluruh semprotan (Recall) tanpa menimbulkan gangguan alarm palsu (Precision).

---

### 3.6 Kurva Precision-Recall (*P-R Curve*)

Kurva Precision-Recall memetakan nilai Precision pada sumbu vertikal (Y) terhadap nilai Recall pada sumbu horizontal (X) untuk setiap variasi ambang batas keyakinan (*confidence threshold*) dari $1.0$ turun ke $0.0$.

```
Precision
  ^
1.0 |-------------------\
    |                    \
    |                     \   Kurva Ideal (Mendekati Pojok Kanan Atas)
    |                      \
    |                       \
    |                        \
    |                         \
0.0 +---------------------------> Recall
   0.0                         1.0
```

- **Kurva Sempurna**: Berada di pojok kanan atas di mana Precision bernilai 1.0 pada Recall bernilai 1.0.
- **AUC (Area Under Curve)**: Luas area di bawah kurva P-R merepresentasikan performa menyeluruh model di seluruh rentang operasionalnya.

---

## BAB 4: MEAN AVERAGE PRECISION (mAP) SEBAGAI STANDAR EMAS

Dalam literatur evaluasi visi komputer (seperti pada benchmark Pascal VOC dan MS COCO), **mAP (Mean Average Precision)** adalah standar metrik paling komprehensif karena mengevaluasi model pada seluruh spektrum ambang keyakinan dan variasi ketatnya batas spasial.

---

### 4.1 Average Precision (AP) dan Perhitungan Area Under Curve (AUC)

Average Precision (AP) pada dasarnya adalah nilai integral luas area di bawah kurva Precision-Recall:

$$\text{AP} = \int_{0}^{1} p(r) \, dr$$

Dalam praktiknya, kurva P-R memiliki pola zig-zag akibat fluktuasi deteksi. Standar modern (COCO/YOLO) menggunakan **Continuous All-Point Interpolation**, di mana nilai precision pada recall tertentu diambil dari precision tertinggi yang dicapai pada recall di sebelah kanannya:

$$p_{\text{interp}}(r) = \max_{\tilde{r} \ge r} p(\tilde{r})$$

Dengan metode ini, efek penurunan sesaat tidak menghukum model secara tidak adil, dan nilai AP mencerminkan kapabilitas global kurva.

---

### 4.2 Mean Average Precision (mAP)

Jika dataset memiliki $N$ buah kelas kategori objek yang berbeda (misalnya: semprotan, botol, nozzle, tangan):

$$\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i$$

*Catatan Rekayasa*: Dalam kasus deteksi spray tunggal (*single-class problem* di mana hanya ada satu kelas target, misal `spray_plume`), maka nilai $\text{mAP}$ secara matematis identik dengan nilai $\text{AP}$ dari kelas semprotan tersebut: $\text{mAP} = \text{AP}_{\text{spray}}$.

---

### 4.3 mAP@50 (mAP pada Ambang IoU 0.50)

- **Notasi**: `mAP@0.5`, `mAP50`, atau `mAP (IoU=0.50)`.
- **Definisi**: Nilai Mean Average Precision yang dihitung dengan menetapkan syarat ketatnya tumpang tindih spasial pada ambang batas tunggal $\tau_{IoU} = 0.50$.
- **Interpretasi Fisik**:
  - Suatu deteksi dianggap benar (TP) jika minimal 50% luasannya beririsan dengan ground truth.
  - Mengukur kemampuan model dalam **menemukan keberadaan dan perkiraan lokasi kasar** objek semprotan.
  - Sangat toleran terhadap ketidaksempurnaan garis batas di ujung kabut semprotan.

---

### 4.4 mAP@50-95 (Standar COCO: Ambang IoU 0.50 hingga 0.95)

- **Notasi**: `mAP@0.5:0.95`, `mAP50-95`, atau `COCO Primary Metric`.
- **Definisi**: Rata-rata dari nilai mAP yang dihitung pada 10 ambang batas IoU diskret yang berbeda, mulai dari 0.50 hingga 0.95 dengan interval langkah sebesar 0.05.

#### Formulasi:

$$\text{mAP@50-95} = \frac{1}{10} \sum_{k=0}^{9} \text{mAP}_{(\tau_{IoU} = 0.50 + 0.05 \times k)}$$

$$\text{mAP@50-95} = \frac{\text{mAP}_{50} + \text{mAP}_{55} + \text{mAP}_{60} + \text{mAP}_{65} + \text{mAP}_{70} + \text{mAP}_{75} + \text{mAP}_{80} + \text{mAP}_{85} + \text{mAP}_{90} + \text{mAP}_{95}}{10}$$

#### Interpretasi Fisik:
- Metrik ini menghukum model yang hanya bisa membuat perkiraan kasar.
- Pada ambang IoU 0.85 hingga 0.95, model dituntut memiliki batas kotak atau piksel yang merekat sangat presisi pada tepi fisik semprotan.
- Nilai mAP@50-95 secara inheren **selalu lebih rendah** daripada mAP@50, karena nilai pada ambang IoU 0.90 dan 0.95 sangat sulit dicapai oleh algoritma visi komputer mana pun di dunia.

---

### 4.5 Analisis Komparatif: mAP (Box) vs. mAP (Mask / Segmentation)

Dalam log pelatihan dan evaluasi YOLOv8-seg, metrik selalu dilaporkan secara berdampingan untuk dua modalitas:

```
+-----------------------------------------------------------------------------+
|               mAP (Box / Plume)               |      mAP (Mask / Segmentasi)|
+-----------------------------------------------+-----------------------------+
| - Dihitung berdasarkan tumpang tindih kotak   | - Dihitung berdasarkan piksel|
|   persegi panjang (Bounding Box IoU).         |   poligon (Mask IoU).       |
| - Merefleksikan keberhasilan lokalisasi global| - Merefleksikan ketepatan   |
|   (apakah model tahu di mana semprotan ada).  |   garis kontur morfologi kabut|
| - Biasanya menghasilkan angka lebih tinggi    | - Menghadapi tantangan piksel|
|   karena kotak toleran terhadap kabut tipis.  |   transparan di tepi semprotan|
| - Parameter kunci untuk deteksi aktivasi spray| - Parameter kunci kalkulasi |
|   (Spray Occurrence / On-Off Detection).      |   Spray Angle & Spray Area. |
+-----------------------------------------------+-----------------------------+
```

---

## BAB 5: FUNGSI KERUGIAN (LOSS FUNCTIONS) DALAM ARSITEKTUR YOLOV8

Saat pelatihan model (*training*), algoritma backpropagation menggunakan fungsi penalti matematis (*Loss Function*) untuk menghitung seberapa jauh tebakan model dari kebenaran ground truth. Bobot model diperbarui untuk meminimalkan nilai kerugian total ($L_{total}$).

$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}}$$

---

### 5.1 Peran Loss Function dalam Optimasi Backpropagation

Jika nilai loss menurun (*loss convergence*), berarti prediksi model semakin mendekati data anotasi acuan. Sebaliknya, jika loss stagnan atau naik pada data validasi (*overfitting*), model mengalami degradasi kemampuan generalisasi.

---

### 5.2 Bounding Box Loss ($L_{box}$ / CIoU Loss)

YOLOv8 menggunakan **Complete IoU (CIoU) Loss** untuk meregresi koordinat kotak pembatas. CIoU tidak hanya menghitung tumpang tindih luas, melainkan menggabungkan tiga faktor geometris penting:

1. **Overlap Area**: Tingkat tumpang tindih dasar ($IoU$).
2. **Central Point Distance**: Jarak Euclidean ternormalisasi antara titik pusat kotak prediksi ($b$) dan kotak ground truth ($b^{gt}$).
3. **Aspect Ratio Consistency**: Konsistensi rasio aspek perbandingan lebar dan tinggi ($\frac{w}{h}$).

#### Formulasi:

$$\mathcal{L}_{\text{box}} = \mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

Di mana:
- $\rho(b, b^{gt})$ adalah jarak Euclidean titik tengah kedua kotak.
- $c$ adalah panjang diagonal dari kotak terkecil yang dapat melingkupi kedua kotak sekaligus (*enclosing box*).
- $v = \frac{4}{\pi^2} \left( \arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h} \right)^2$ mengukur perbedaan proporsi bentuk.
- $\alpha$ adalah parameter pembobot kompensasi.

*Manfaat*: Memastikan kotak prediksi tidak hanya beririsan, tetapi juga memiliki pusat massa dan perbandingan panjang-lebar yang identik dengan semprotan riil.

---

### 5.3 Distribution Focal Loss ($L_{dfl}$)

Pada arsitektur YOLOv8 tanpa *anchor* (*anchor-free*), koordinat tepi kotak pembatas tidak diregresi sebagai satu angka skalar kaku, melainkan sebagai **distribusi probabilitas umum** sepanjang rentang nilai diskret.

#### Konsep:
- Mengakui bahwa batas semprotan fluida bersifat kabur (*uncertain*).
- $L_{dfl}$ memaksa model memusatkan probabilitas tertinggi di sekitar lokasi koordinat tepi yang sebenarnya menggunakan formulasi Focal Loss yang dimodifikasi.

#### Formulasi:

$$\mathcal{L}_{\text{dfl}} = - \left( (y_{i+1} - y) \log(S_i) + (y - y_i) \log(S_{i+1}) \right)$$

Di mana $y$ adalah nilai koordinat kontinu yang diapit oleh dua titik nilai diskret $y_i$ dan $y_{i+1}$, serta $S$ adalah probabilitas softmax.

---

### 5.4 Segmentation Mask Loss ($L_{seg}$ / Binary Cross-Entropy Mask Loss)

Untuk memprediksi topeng piksel, YOLOv8 menggunakan arsitektur berbasis *Prototyping*:
1. Head segmentasi menghasilkan kumpulan basis prototipe topeng global (*Protonet*).
2. Untuk setiap deteksi, model memprediksi sekumpulan koefisien mask ($k$ koefisien).
3. Matriks piksel akhir dihasilkan dari perkalian linear antara prototipe dengan koefisien, lalu dilewatkan ke fungsi sigmoid.

Perbedaan piksel-demi-piksel antara mask prediksi ($p_{i,j}$) dan mask ground truth ($y_{i,j}$) dihukum menggunakan **Binary Cross-Entropy (BCE) Loss**:

$$\mathcal{L}_{\text{seg}} = - \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \left[ y_{i,j} \log(p_{i,j}) + (1 - y_{i,j}) \log(1 - p_{i,j}) \right]$$

*Manfaat*: Memaksa model menyalakan nilai $1$ pada piksel di dalam pola kabut dan mematikan nilai menjadi $0$ pada piksel latar belakang di luar kabut.

---

### 5.5 Classification Loss ($L_{cls}$ / BCE Cls Loss)

Mengukur kesalahan model dalam membedakan kelas objek. Menggunakan Binary Cross-Entropy untuk menentukan apakah objek di dalam kotak tersebut merupakan kelas target yang benar atau kelas lain.

$$\mathcal{L}_{\text{cls}} = - \left[ y \log(\hat{p}) + (1 - y) \log(1 - \hat{p}) \right]$$

Dalam sistem deteksi tunggal (*single-class spray*), $L_{cls}$ berfungsi memastikan bahwa model yakin bahwa objek yang terdeteksi adalah semprotan fluida, bukan objek pengganggu.

---

## BAB 6: PARAMETER OPERASIONAL INFERENSI & POST-PROCESSING

Saat model yang telah dilatih diterapkan pada sistem inferensi (seperti pada web API atau video pipeline), terdapat sekumpulan hiperparameter pasca-pemrosesan (*post-processing hyperparameters*) yang menentukan perilaku operasional deteksi:

---

### 6.1 Confidence Threshold ($conf$ / $thresh$)

- **Definisi**: Ambang batas keyakinan probabilitas minimum yang harus dipenuhi oleh kandidat deteksi agar diakui sebagai deteksi resmi.
- **Rentang Nilai**: $0.00 \le conf \le 1.00$ (Umumnya diatur antara $0.25$ hingga $0.50$).
- **Pengaruh Operasional**:
  - **Terlalu Rendah ($conf < 0.20$)**: Model sangat sensitif, mendeteksi semprotan tipis di udara, namun rentan memunculkan artefak latar belakang / debu sebagai spray (FP tinggi).
  - **Terlalu Tinggi ($conf > 0.60$)**: Deteksi sangat bersih dan bebas noise, namun kabut semprotan fase awal (*developing stage*) yang masih tipis berisiko tidak terdeteksi (FN tinggi).

---

### 6.2 Non-Maximum Suppression (NMS) & IoU Threshold ($iou$)

Arsitektur detektor modern menghasilkan ratusan hingga ribuan kandidat kotak deteksi yang saling tumpang tindih untuk satu objek semprotan yang sama. **Non-Maximum Suppression (NMS)** adalah algoritma pembersihan untuk menyaring kandidat tersebut hingga menyisakan tepat satu deteksi terbaik.

```
+-----------------------------------------------------------------------------+
| ALGORITMA NON-MAXIMUM SUPPRESSION (NMS)                                     |
+-----------------------------------------------------------------------------+
|                                                                             |
|  1. Kumpulkan seluruh kandidat deteksi semprotan.                           |
|  2. Urutkan kandidat berdasarkan skor keyakinan tertinggi (Confidence).    |
|  3. Pilih kandidat dengan skor tertinggi sebagai "Deteksi Terpilih".        |
|  4. Hitung IoU antara Deteksi Terpilih dengan semua kandidat lainnya.       |
|  5. Jika IoU > Ambang Batas NMS (IoU Threshold): HAPUS kandidat tsb         |
|     (karena dianggap mendeteksi objek yang sama / redundan).                |
|  6. Ulangi proses untuk kandidat tersisa hingga semua objek tersaring.      |
|                                                                             |
+-----------------------------------------------------------------------------+
```

- **Ambang Batas NMS ($\tau_{nms}$)**:
  - Nilai default standar biasanya $0.45 - 0.70$.
  - Jika disetel terlalu rendah, dua semprotan yang berdekatan dapat salah terhapus dan dianggap sebagai satu objek.
  - Jika disetel terlalu tinggi, satu semprotan dapat menghasilkan dua kotak deteksi yang saling menumpuk.

---

### 6.3 Resolusi Spasial Citra Input ($imgsz$)

- **Definisi**: Dimensi resolusi piksel di mana citra mentah di-*rescale* sebelum diproses oleh lapisan konvolusi neural network (misal: $640 \times 640$, $1024 \times 1024$, atau $1280 \times 1280$).
- **Dampak pada Analisis Fluida**:
  - Resolusi lebih besar memperjelas partikel kabut halus di tepi nozzle, meningkatkan akurasi $IoU_{mask}$.
  - Namun, resolusi tinggi meningkatkan beban komputasi memori GPU secara kuadratik ($\mathcal{O}(H \times W)$) dan menurunkan kecepatan frame-per-detik (FPS).

---

### 6.4 Mask Ratio dan Downsampling Prototype Mask

Pada YOLOv8-seg, topeng prototipe umumnya diproses pada resolusi seperempat dari resolusi input citra (faktor downsampling 4x) untuk menghemat VRAM dan menjaga inferensi tetap *real-time*. Pada tahap akhir, mask diinterpolasi kembali secara bilinear ke resolusi asli. Memahami konsep ini penting agar tidak salah mengira bahwa tepi piksel tangga (*aliasing*) adalah kesalahan model.

---

## BAB 7: MATRIKS EVALUASI LANJUTAN & RELEVANSI FISIK INDUSTRI

---

### 7.1 Confusion Matrix & Normalized Confusion Matrix

**Confusion Matrix (Matriks Kebingungan)** adalah tabel kontingensi dua dimensi yang memetakan jumlah objek aktual (Ground Truth) pada baris terhadap jumlah objek yang diprediksi oleh model pada kolom.

```
                         KOLOM: PREDIKSI MODEL
                     +--------------------+--------------------+
                     |    Spray Plume     |     Background     |
+----+---------------+--------------------+--------------------+
| B  |  Spray Plume  |   True Positive    |   False Negative   |
| A  |  (Aktual Ada) | (Terdeteksi Benar) |  (Luput / Hilang)  |
| R  +---------------+--------------------+--------------------+
| I  |  Background   |   False Positive   |   True Negative    |
| S  |  (Kosong)     |   (Alarm Palsu)    | (Latar Terjaga)    |
+----+---------------+--------------------+--------------------+
```

- **Normalized Confusion Matrix**:
  Matriks di mana nilai setiap baris dinormalisasi menjadi rentang desimal $0.00$ hingga $1.00$ (atau persentase 0% - 100%).
  - Angka pada diagonal utama menunjukkan persentase akurasi penangkapan masing-masing kelas.
  - Elemen di luar diagonal utama langsung mengindikasikan letak kesalahan klasifikasi.

---

### 7.2 Latar Belakang Negatif (*Background False Positives vs. Background Negatives*)

Dalam domain semprotan kosmetik dan farmasi, interaksi antara objek target dan latar belakang memiliki konsekuensi teknis:

1. **Background False Positive (Latar Dituduh Spray)**:
   Terjadi jika model salah mengira pilar mekanik mesin, selang, atau pantulan cahaya lampu lab sebagai semprotan.
   *Dampak*: Sistem inferensi akan mencatat seolah-olah terjadi kebocoran semprotan di saat nozzle sedang mati.
2. **Background False Negative (Spray Dianggap Latar Belakang)**:
   Terjadi pada fase akhir semprotan saat kabut mulai menipis dan melayang di udara bebas (*spray dissipation*).
   *Dampak*: Durasi semprotan tercatat lebih pendek dari durasi aktual di lapangan.

---

### 7.3 Dampak Kesalahan Metrik terhadap Ekstraksi Parameter Fisik Fluida

Evaluasi model AI bukan sekadar urusan angka metrik perangkat lunak, melainkan berimplikasi langsung pada validitas data rekayasa mekanika fluida:

```
+-----------------------------------------------------------------------------+
| HUBUNGAN METRIK COMPUTER VISION TERHADAP PARAMETER TEKNIK FLUIDA            |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Metrik AI                Parameter Fisik Spray          Konsekuensi Eror   |
|  -------------------------------------------------------------------------  |
|  Low Box Recall    -----> Spray Event Trigger      -----> Semprotan tidak   |
|                           (Deteksi Siklus Semprot)        terekam           |
|                                                                             |
|  Low Mask IoU      -----> Spray Angle / Sudut      -----> Karakter sudut    |
|  (Kontur Meleset)         (Derajat Payung Semprot)        menyimpang        |
|                                                                             |
|  Low Mask Prec.    -----> Spray Area / Volume      -----> Estimasi luasan   |
|  (Area Menggelembung)     (Penyebaran Droplet)            terlalu besar     |
|                                                                             |
|  Low Box Prec.     -----> False Trigger            -----> Menghitung sem-   |
|                           (Siklus Semprot Palsu)          protan fiktif     |
|                                                                             |
+-----------------------------------------------------------------------------+
```

1. **Pengaruh Akurasi Mask terhadap Sudut Semprot (*Spray Angle*)**:
   Sudut semprot dihitung dengan menarik garis singgung dari titik asal nozzle ke batas kontur terluar plume semprotan pada jarak aksial tertentu. Jika model memiliki nilai $IoU_{mask}$ yang rendah (kontur tepi tidak presisi), kemiringan garis singgung akan bergeser beberapa derajat, menghasilkan kesalahan fatal pada standarisasi nozzle industri.
2. **Pengaruh terhadap Panjang Penetrasi Semprot (*Penetration Length*)**:
   Panjang penetrasi diukur dari ujung nozzle ke piksel terjauh yang dicapai kabut semprotan sepanjang sumbu utama semprotan. Ketidaktepatan bounding box bagian bawah menyebabkan kesalahan perhitungan laju kecepatan semprotan fluida.

---

### 7.4 Glosarium Istilah Singkat (*Quick Reference Table*)

Sebagai rujukan cepat (*cheat sheet*) bagi para rekayasawan dan pengembang sistem:

| Istilah / Akronim | Nama Lengkap | Definisi Singkat & Makna Fungsional |
| :--- | :--- | :--- |
| **IoU** | *Intersection over Union* | Rasio tumpang tindih luas irisan dibagi luas gabungan antara anotasi acuan dan prediksi model ($0.0 - 1.0$). |
| **TP** | *True Positive* | Deteksi yang berhasil menemukan objek aktual dengan kriteria tumpang tindih spasial memenuhi syarat ($\text{IoU} \ge \tau$). |
| **FP** | *False Positive* | Alarm palsu; deteksi yang salah sasaran, meleset dari objek, atau mendeteksi latar kosong. |
| **FN** | *False Negative* | Objek riil yang luput dan gagal ditemukan oleh model sama sekali. |
| **Precision** | *Precision (Presisi)* | Tingkat kemurnian hasil; persentase deteksi model yang terbukti benar ($\frac{\text{TP}}{\text{TP} + \text{FP}}$). |
| **Recall** | *Recall (Sensitivitas)* | Tingkat kelengkapan tangkapan; persentase objek riil yang berhasil ditemukan ($\frac{\text{TP}}{\text{TP} + \text{FN}}$). |
| **F1-Score** | *Harmonic Mean of P & R* | Skor keseimbangan optimal antara Presisi dan Sensitivitas ($2 \frac{P \cdot R}{P + R}$). |
| **AP** | *Average Precision* | Luas area di bawah kurva Precision-Recall untuk satu kategori objek tertentu. |
| **mAP** | *Mean Average Precision* | Rata-rata dari nilai AP di seluruh kategori objek yang ada pada dataset. |
| **mAP50** | *mAP at IoU 0.50* | mAP yang dihitung dengan kriteria tumpang tindih longgar ($\ge 50\%$). Menguji kemampuan lokalisasi global. |
| **mAP50-95** | *COCO mAP (0.50:0.95)* | Rata-rata mAP pada 10 ambang IoU (0.50 s.d. 0.95). Menguji kesempurnaan dan kepresisian batas kontur. |
| **Box Metrics** | *Bounding Box Metrics* | Evaluasi performa yang didasarkan pada koordinat segiempat pembatas luar objek. |
| **Mask Metrics** | *Segmentation Mask Metrics*| Evaluasi performa yang didasarkan pada kecocokan piksel-demi-piksel topeng poligon objek. |
| **Loss** | *Cost / Error Function* | Besaran penalti kesalahan matematis saat pelatihan yang diminimalkan oleh backpropagation. |
| **CIoU Loss** | *Complete IoU Loss* | Loss regresi koordinat box yang memperhitungkan overlap, jarak titik pusat, dan rasio aspek bentuk. |
| **DFL Loss** | *Distribution Focal Loss* | Loss regresi koordinat tepi berbasis distribusi probabilitas untuk menangani batas objek kabur. |
| **BCE Loss** | *Binary Cross-Entropy Loss*| Loss fungsi logaritmik untuk klasifikasi biner dan pembentukan piksel mask (on/off). |
| **Conf Thresh** | *Confidence Threshold* | Batas keyakinan minimum agar prediksi diizinkan lolos ke tahap evaluasi akhir. |
| **NMS** | *Non-Maximum Suppression* | Algoritma pemangkasan deteksi ganda/redundant yang menumpuk pada satu objek yang sama. |
| **Confusion Matrix** | *Matriks Kontingensi* | Tabel pemetaan silang yang memperlihatkan distribusi keberhasilan dan kekeliruan prediksi antar kelas. |

---

*Dokumen ini diterbitkan sebagai panduan rekayasa resmi (Technical Reference) arsitektur AI pada proyek sistem analisis visi komputer semprotan fluida.*
