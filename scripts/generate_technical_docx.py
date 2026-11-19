"""
Script untuk meng-generate dokumen teknis 'Dokumen_Teknis_Metrik_dan_Evaluasi_AI_YOLO.docx'
Menggunakan python-docx dengan format profesional:
- Margin 2.54 cm (1 inch)
- Font: Calibri / Arial
- Heading Hierarchy dengan warna korporat/teknis
- Tabel ber-header gelap dengan zebra striping ringan
- Callout box & rumus terstruktur
- Non-reporting (penjelasan konseptual dan teknis tanpa persentase run)
"""

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from pathlib import Path

def set_cell_background(cell, hex_color):
    """Set background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner padding for table cells in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_callout(doc, text, title="KONSEP KUNCI"):
    """Tambahkan kotak callout/catatan penting."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    # Border kiri biru tebal
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:top w:val="none"/><w:left w:val="single" w:sz="36" w:space="0" w:color="1B365D"/><w:bottom w:val="none"/><w:right w:val="none"/></w:tcBorders>')
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_title = p.add_run(f"📌 {title}: ")
    run_title.bold = True
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(10)
    run_title.font.color.rgb = RGBColor(27, 54, 93)
    
    run_text = p.add_run(text)
    run_text.font.name = "Calibri"
    run_text.font.size = Pt(9.5)
    run_text.font.italic = True
    run_text.font.color.rgb = RGBColor(45, 55, 72)
    
    # Spasi setelah callout
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)

def build_document(output_path: str):
    doc = Document()
    
    # Setup Page Margins: 1 inch (25.4 mm)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Header & Footer
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("Dokumen Teknis Metrik & Evaluasi AI | Arsitektur YOLOv8-Seg")
        hrun.font.name = "Calibri"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(120, 144, 156)
        
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("Dokumen Referensi Rekayasa Komputer Visi — Paragon Spray AI Analysis")
        frun.font.name = "Calibri"
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = RGBColor(120, 144, 156)

    # Palette Warna
    COLOR_PRIMARY = RGBColor(27, 54, 93)      # Deep Navy
    COLOR_SECONDARY = RGBColor(43, 108, 176)  # Blue Accent
    COLOR_BODY = RGBColor(45, 55, 72)         # Charcoal Dark
    
    # ==========================
    # JUDUL DOKUMEN & METADATA
    # ==========================
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("DOKUMEN TEKNIS METRIK DAN EVALUASI COMPUTER VISION")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(22)
    run_title.bold = True
    run_title.font.color.rgb = COLOR_PRIMARY
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(14)
    run_sub = p_sub.add_run("Arsitektur Object Detection & Instance Segmentation (YOLOv8) pada Karakterisasi Semprotan Fluida (Spray Plume Analysis)")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = COLOR_SECONDARY
    
    # Garis pemisah horizontal
    p_line = doc.add_paragraph()
    p_line.paragraph_format.space_after = Pt(12)
    r_line = p_line.add_run("―" * 58)
    r_line.font.color.rgb = RGBColor(203, 213, 225)
    
    # Metadata Table
    meta_tbl = doc.add_table(rows=5, cols=2)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Klasifikasi Dokumen", "Dokumen Referensi Teknis & Glosarium Konseptual (Non-Reporting Reference)"),
        ("Domain Rekayasa", "Computer Vision, Deep Learning, Pattern Recognition & Fluid Mechanics"),
        ("Arsitektur Model", "YOLOv8 Segmentation (Ultralytics Framework)"),
        ("Fokus Kajian", "Formulasi Matematika, Spasial Overlap, Ambang Keyakinan & Penalti Kerugian (Loss)"),
        ("Sifat Evaluasi", "Dokumen Teoritis Universal (Tanpa Skor Persentase Run Spesifik)")
    ]
    for idx, (label, val) in enumerate(meta_data):
        row = meta_tbl.rows[idx]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        set_cell_background(c0, "F8FAFC")
        set_cell_margins(c0, top=60, bottom=60, left=100, right=100)
        set_cell_margins(c1, top=60, bottom=60, left=100, right=100)
        
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(label)
        r0.bold = True
        r0.font.name = "Calibri"
        r0.font.size = Pt(9.5)
        r0.font.color.rgb = COLOR_PRIMARY
        
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(val)
        r1.font.name = "Calibri"
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = COLOR_BODY

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(16)
    
    # ==========================
    # DAFTAR ISI
    # ==========================
    p_toc_head = doc.add_paragraph()
    p_toc_head.paragraph_format.space_before = Pt(14)
    p_toc_head.paragraph_format.space_after = Pt(6)
    r_toc_h = p_toc_head.add_run("DAFTAR ISI DOKUMEN")
    r_toc_h.bold = True
    r_toc_h.font.name = "Calibri"
    r_toc_h.font.size = Pt(14)
    r_toc_h.font.color.rgb = COLOR_PRIMARY
    
    toc_items = [
        ("BAB 1: PENDAHULUAN & PARADIGMA COMPUTER VISION", [
            "1.1 Perbedaan Klasifikasi Citra, Object Detection, dan Instance Segmentation",
            "1.2 Anatomi Output Segmentasi: Bounding Box vs. Polygon Mask",
            "1.3 Karakteristik Khusus Objek Fluida Semprotan (Spray Plume & Cone)"
        ]),
        ("BAB 2: LANDASAN PENGUKURAN SPASIAL (SPATIAL OVERLAP METRICS)", [
            "2.1 Ground Truth vs. Prediksi Model",
            "2.2 Intersection over Union (IoU) / Jaccard Index",
            "2.3 IoU Bounding Box (IoU_box) vs. IoU Segmentation Mask (IoU_mask)",
            "2.4 Tingkat Ketat Spasial (Strictness Level) Berdasarkan Ambang Batas IoU"
        ]),
        ("BAB 3: METRIK EVALUASI KLASIFIKASI & DETEKSI BINER", [
            "3.1 Taksonomi Deteksi: True Positive (TP), False Positive (FP), False Negative (FN), True Negative (TN)",
            "3.2 Precision (Presisi) & Keandalan Deteksi",
            "3.3 Recall (Sensitivitas / Kelengkapan Temuan)",
            "3.4 Kompromi Precision vs. Recall (Trade-off Dilemma)",
            "3.5 F1-Score (Rata-rata Harmonik Seimbang)",
            "3.6 Kurva Precision-Recall (P-R Curve)"
        ]),
        ("BAB 4: MEAN AVERAGE PRECISION (mAP) SEBAGAI STANDAR EMAS", [
            "4.1 Average Precision (AP) dan Perhitungan Area Under Curve (AUC)",
            "4.2 Mean Average Precision (mAP) Lintas Kategori",
            "4.3 mAP@50 (mAP pada Ambang IoU 0.50)",
            "4.4 mAP@50-95 (Standar COCO: Rata-rata 10 Ambang IoU 0.50 s.d. 0.95)",
            "4.5 Analisis Komparatif: mAP (Box) vs. mAP (Mask / Segmentation)"
        ]),
        ("BAB 5: FUNGSI KERUGIAN (LOSS FUNCTIONS) DALAM ARSITEKTUR YOLOV8", [
            "5.1 Peran Loss Function dalam Optimasi Backpropagation",
            "5.2 Bounding Box Loss (L_box / CIoU Loss)",
            "5.3 Distribution Focal Loss (L_dfl)",
            "5.4 Segmentation Mask Loss (L_seg / BCE Mask Loss)",
            "5.5 Classification Loss (L_cls / BCE Cls Loss)"
        ]),
        ("BAB 6: PARAMETER OPERASIONAL INFERENSI & POST-PROCESSING", [
            "6.1 Confidence Threshold (conf / thresh)",
            "6.2 Non-Maximum Suppression (NMS) & IoU Threshold",
            "6.3 Resolusi Spasial Citra Input (imgsz)",
            "6.4 Mask Ratio dan Downsampling Prototype Mask"
        ]),
        ("BAB 7: MATRIKS EVALUASI LANJUTAN & RELEVANSI FISIK INDUSTRI", [
            "7.1 Confusion Matrix & Normalized Confusion Matrix",
            "7.2 Deteksi Latar Belakang (Background False Positives vs. False Negatives)",
            "7.3 Dampak Kesalahan Metrik terhadap Ekstraksi Parameter Fisik Fluida",
            "7.4 Glosarium Istilah Singkat (Quick Reference Cheat-Sheet)"
        ])
    ]
    
    for chapter_title, sub_items in toc_items:
        p_ch = doc.add_paragraph()
        p_ch.paragraph_format.space_before = Pt(4)
        p_ch.paragraph_format.space_after = Pt(2)
        r_ch = p_ch.add_run(chapter_title)
        r_ch.bold = True
        r_ch.font.name = "Calibri"
        r_ch.font.size = Pt(10)
        r_ch.font.color.rgb = COLOR_PRIMARY
        
        for sub in sub_items:
            p_sub = doc.add_paragraph()
            p_sub.paragraph_format.left_indent = Inches(0.25)
            p_sub.paragraph_format.space_before = Pt(0)
            p_sub.paragraph_format.space_after = Pt(1.5)
            r_sub = p_sub.add_run(sub)
            r_sub.font.name = "Calibri"
            r_sub.font.size = Pt(9)
            r_sub.font.color.rgb = COLOR_BODY
            
    doc.add_page_break()
    
    # Helper format headings
    def add_h1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(18)
        h.paragraph_format.space_after = Pt(6)
        r = h.add_run(text)
        r.bold = True
        r.font.name = "Calibri"
        r.font.size = Pt(15)
        r.font.color.rgb = COLOR_PRIMARY
        return h

    def add_h2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        r = h.add_run(text)
        r.bold = True
        r.font.name = "Calibri"
        r.font.size = Pt(12)
        r.font.color.rgb = COLOR_SECONDARY
        return h

    def add_p(text, bold_prefix=None, space_after=6):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.bold = True
            rb.font.name = "Calibri"
            rb.font.size = Pt(10)
            rb.font.color.rgb = COLOR_PRIMARY
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10)
        r.font.color.rgb = COLOR_BODY
        return p

    def add_bullet(text, bold_prefix=None):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.bold = True
            rb.font.name = "Calibri"
            rb.font.size = Pt(10)
            rb.font.color.rgb = COLOR_PRIMARY
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10)
        r.font.color.rgb = COLOR_BODY
        return p

    # ==========================
    # BAB 1
    # ==========================
    add_h1("BAB 1: PENDAHULUAN & PARADIGMA COMPUTER VISION")
    
    add_h2("1.1 Perbedaan Klasifikasi Citra, Object Detection, dan Instance Segmentation")
    add_p("Dalam bidang visi komputer modern, pemahaman terhadap citra digital berkembang melalui beberapa tingkatan abstraksi analitis:")
    add_bullet(" Model hanya menjawab keberadaan kategori objek dalam keseluruhan gambar secara global (misal: 'Ada semprotan' atau 'Tidak ada semprotan'). Model tidak memiliki pemahaman di mana koordinat letak objek tersebut.", "1. Image Classification (Klasifikasi Citra):")
    add_bullet(" Model mengenali keberadaan objek sekaligus melokalisasi posisinya menggunakan kotak pembatas segiempat tegak lurus (Axis-Aligned Bounding Box). Output dinyatakan dalam koordinat 4-tupel: (x_center, y_center, width, height) beserta skor probabilitas kelas.", "2. Object Detection (Deteksi Objek):")
    add_bullet(" Model mengklasifikasikan setiap piksel individual ke dalam kategori kelas tertentu. Kelemahan pendekatan ini adalah ketidakmampuannya membedakan dua entitas individu dari kelas yang sama jika keduanya saling bersentuhan atau tumpang tindih.", "3. Semantic Segmentation (Segmentasi Semantik):")
    add_bullet(" Paradigma hibrida terlengkap yang mendeteksi setiap objek secara individual (bounding box) dan secara simultan memetakan batas piksel presisi (pixel mask) yang mengisolasi geometri kontur objek tersebut dari latar belakang.", "4. Instance Segmentation (Segmentasi Instansia):")

    add_h2("1.2 Anatomi Output Segmentasi: Bounding Box vs. Polygon Mask")
    add_p("Pada arsitektur deteksi instansia kontemporer (seperti YOLOv8-seg), model menghasilkan dua bentuk output spasial secara paralel untuk setiap objek semprotan yang teridentifikasi:")
    add_bullet(" Kotak segiempat ortogonal yang melingkupi batas terluar objek secara minimal. Bersifat sebagai estimasi makro/global yang cepat dihitung, namun memiliki kelemahan inheren karena selalu mengikutsertakan area kosong latar belakang pada objek yang bentuknya tidak persegi.", "Bounding Box (Kotak Pembatas):")
    add_bullet(" Matriks biner spasial berukuran resolusi citra atau representasi koordinat poligon tertutup yang hanya bernilai aktif pada piksel yang benar-benar ditempati oleh partikel objek. Ini mengisolasi bentuk sejati (true morphology) dari fluida.", "Polygon Mask (Topeng Segmentasi Piksel):")

    add_callout(doc, "Bounding Box merepresentasikan lokalisasi global dan kehadiran objek, sedangkan Segmentation Mask merepresentasikan bentuk geometris detail yang esensial untuk perhitungan luas fisik, sudut semprotan, dan keliling plume.", "Dualitas Representasi Spasial")

    add_h2("1.3 Karakteristik Khusus Objek Fluida Semprotan (Spray Plume & Cone)")
    add_p("Objek fluida semprotan memiliki properti optik dan morfologi yang sangat menantang bagi model visi komputer standar:")
    add_bullet(" Semprotan aerosol dan cairan tidak memiliki garis batas tegas seperti benda padat (mobil, meja). Tepi semprotan merupakan transisi gradasi partikel droplet mikro yang kepadatannya berkurang menuju udara bebas.", "1. Batas Bertahap dan Semi-Transparan (Soft Boundary):")
    add_bullet(" Bentuk semprotan berfluktuasi drastis setiap milidetik tergantung pada dinamika tekanan aktuator nozzle, fase pembentukan (developing stage), fase stabil (fully-developed), hingga fase runtuh/disipasi (collapse stage).", "2. Deformasi Temporal Ekstrem (Non-Rigid Geometry):")
    add_bullet(" Analisis teknik fluida sangat bergantung pada luasan piksel mask untuk menghitung parameter kritis seperti Spray Angle, Penetration Length, dan Spray Area. Kesalahan segmentasi batas akan merusak validitas pengukuran teknik tersebut.", "3. Ketergantungan Ekstraksi Fitur Fisik:")

    # ==========================
    # BAB 2
    # ==========================
    add_h1("BAB 2: LANDASAN PENGUKURAN SPASIAL (SPATIAL OVERLAP METRICS)")
    
    add_h2("2.1 Ground Truth vs. Prediksi Model")
    add_p("Evaluasi kinerja model visi komputer dilakukan dengan membandingkan dua entitas data spasial:")
    add_bullet(" Anotasi acuan kebenaran mutlak yang dibuat secara cermat oleh analis manusia ahli. Berisi koordinat batas kotak (bounding box) dan poligon kontur semprotan yang sesungguhnya ada pada citra uji.", "Ground Truth (GT / Anotasi Acuan):")
    add_bullet(" Hasil inferensi langsung dari jaringan saraf tiruan (neural network), yang mencakup koordinat kotak prediksi, mask poligon, label kelas, dan skor keyakinan probabilitas (confidence score).", "Prediksi Model (Prediction):")

    add_h2("2.2 Intersection over Union (IoU) / Jaccard Index")
    add_p("Intersection over Union (IoU), dikenal pula dalam statistik sebagai Indeks Jaccard, adalah metrik kuantitatif utama untuk mengukur derajat kesamaan dan tumpang tindih spasial antara area prediksi dengan area acuan ground truth.")
    add_p("Formulasi Matematis:\nIoU = Luas Area Irisan (Intersection) / Luas Area Gabungan (Union) = |A ∩ B| / |A ∪ B|\nDi mana |A ∩ B| adalah luas bidang yang ditempati bersamaan oleh prediksi dan ground truth, sedangkan |A ∪ B| = |A| + |B| - |A ∩ B| adalah total luasan gabungan keduanya.")
    add_p("Karakteristik Nilai IoU:\n• Nilai berkisar antara 0.0 (tidak ada tumpang tindih sama sekali, prediksi meleset total) hingga 1.0 (prediksi berhimpit 100% sempurna dengan ground truth).\n• IoU mengabaikan ukuran absolut citra; nilai IoU 0.8 pada objek kecil memiliki arti kecocokan geometris yang sama dengan IoU 0.8 pada objek besar.")

    add_h2("2.3 IoU Bounding Box (IoU_box) vs. IoU Segmentation Mask (IoU_mask)")
    add_p("Pada model multi-tugas seperti YOLOv8-seg, perhitungan IoU dilakukan secara independen pada dua tingkat abstraksi:")
    
    # Table IoU comparison
    tbl_iou = doc.add_table(rows=5, cols=3)
    tbl_iou.alignment = WD_TABLE_ALIGNMENT.CENTER
    iou_headers = ["Dimensi Komparasi", "IoU Bounding Box (IoU_box)", "IoU Segmentation Mask (IoU_mask)"]
    for i, h in enumerate(iou_headers):
        c = tbl_iou.rows[0].cells[i]
        set_cell_background(c, "1B365D")
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Calibri"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    iou_rows = [
        ("Domain Geometri", "Ruang kontinu 2D segiempat [x1, y1, x2, y2]", "Ruang diskret matriks piksel biner H x W"),
        ("Metode Perhitungan", "Perpotongan matematis koordinat analitik batas", "Operasi logika biner: SUM(M_gt AND M_pred) / SUM(M_gt OR M_pred)"),
        ("Sensitivitas Bentuk", "Rendah; hanya menguji selubung persegi terluar", "Sangat Tinggi; menguji setiap piksel kontur lekukan"),
        ("Kecenderungan Nilai", "Cenderung lebih tinggi (toleran terhadap ruang kosong)", "Cenderung lebih rendah (terhukum oleh partikel kabur di batas tepi)")
    ]
    for r_idx, row_data in enumerate(iou_rows, start=1):
        row = tbl_iou.rows[r_idx]
        bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.name = "Calibri"
            r.font.size = Pt(9)
            r.font.color.rgb = COLOR_BODY

    p_sp2 = doc.add_paragraph()
    p_sp2.paragraph_format.space_after = Pt(8)

    add_h2("2.4 Tingkat Ketat Spasial (Strictness Level) Berdasarkan Ambang Batas IoU")
    add_p("Untuk memutuskan apakah suatu keluaran prediksi dianggap 'sukses melokalisasi objek', ditetapkan nilai ambang batas kesepakatan spasial (IoU Threshold / tau_IoU):")
    add_bullet(" Standar industri klasik (PASCAL VOC). Deteksi dinyatakan benar jika separuh area prediksi saling bertumpuk dengan ground truth. Sangat cocok untuk mengonfirmasi ada/tidaknya objek secara umum.", "tau_IoU = 0.50 (Toleransi Moderat):")
    add_bullet(" Menuntut kecocokan batas yang ketat. Menguji apakah posisi prediksi menempel rapat pada bentuk aktual tanpa pergeseran pusat massa yang signifikan.", "tau_IoU = 0.75 (Toleransi Ketat):")
    add_bullet(" Tuntutan kecocokan geometris ekstrem. Sangat jarang dicapai secara konsisten pada objek semprotan karena sifat batas droplet yang bergradasi dan kabur.", "tau_IoU = 0.90 - 0.95 (Ultra-Presisi):")

    # ==========================
    # BAB 3
    # ==========================
    add_h1("BAB 3: METRIK EVALUASI KLASIFIKASI & DETEKSI BINER")
    
    add_h2("3.1 Taksonomi Deteksi: True Positive, False Positive, False Negative, dan True Negative")
    add_p("Berdasarkan perbandingan antara prediksi model dan ground truth dengan mempertimbangkan ambang batas IoU (tau_IoU), setiap hasil deteksi diklasifikasikan ke dalam 4 kuadran kontingensi:")
    add_bullet(" Model memprediksi adanya objek semprotan, dan prediksi tersebut cocok dengan semprotan acuan pada ground truth dengan tingkat tumpang tindih spasial memenuhi syarat (IoU >= tau_IoU).", "1. True Positive (TP) / Deteksi Benar:")
    add_bullet(" Model memprediksi adanya semprotan di suatu lokasi, namun pada lokasi tersebut tidak ada semprotan aktual (mendeteksi latar kosong), ATAU ada semprotan namun tingkat tumpang tindihnya meleset di bawah ambang batas (IoU < tau_IoU), ATAU merupakan deteksi ganda kedua dari objek yang sama.", "2. False Positive (FP) / Alarm Palsu (Type I Error):")
    add_bullet(" Objek semprotan aktual ada pada citra acuan ground truth, namun model gagal mendeteksinya sama sekali (terlewat / undetected).", "3. False Negative (FN) / Luput (Type II Error):")
    add_bullet(" Area latar belakang kosong yang secara benar tidak diprediksi sebagai semprotan oleh model. Dalam evaluasi deteksi objek berbasis bounding box, TN umumnya tidak dihitung secara eksplisit karena jumlah kemungkinan kotak latar kosong tidak terhingga.", "4. True Negative (TN) / Latar Terjaga:")

    add_h2("3.2 Precision (Presisi) & Keandalan Deteksi")
    add_p("Precision mengukur tingkat kemurnian dan validitas dari seluruh deteksi positif yang dikeluarkan oleh model.")
    add_p("Formulasi:\nPrecision = TP / (TP + FP) = Jumlah Deteksi Benar / Total Seluruh Deteksi yang Dibuat Model")
    add_p("Interpretasi Konseptual:\n• Menjawab pertanyaan: 'Dari seluruh kotak/mask yang diklaim oleh model sebagai semprotan fluida, berapa fraksi yang benar-benar semprotan asli dan bukan noise/pantulan cahaya?'\n• Nilai Precision yang tinggi mengindikasikan bahwa model sangat andal; setiap kali model mendeteksi semprotan, analis dapat yakin bahwa hasil tersebut bukan alarm palsu.")

    add_h2("3.3 Recall (Sensitivitas / Kelengkapan Temuan)")
    add_p("Recall, dikenal pula sebagai Sensitivitas (Sensitivity) atau True Positive Rate (TPR), mengukur kelengkapan model dalam menangkap seluruh objek yang ada di dunia nyata.")
    add_p("Formulasi:\nRecall = TP / (TP + FN) = Jumlah Deteksi Benar / Total Seluruh Objek yang Sebenarnya Ada di Lapangan")
    add_p("Interpretasi Konseptual:\n• Menjawab pertanyaan: 'Dari seluruh kejadian semprotan yang benar-benar muncul di depan kamera, berapa fraksi yang berhasil ditangkap oleh model tanpa ada yang terlewat?'\n• Nilai Recall yang tinggi berarti model tidak pernah 'buta' terhadap semprotan; seluruh fase semprotan berhasil diidentifikasi.")

    add_h2("3.4 Kompromi Precision vs. Recall (Trade-off Dilemma)")
    add_p("Dalam sistem inferensi visi komputer, Precision dan Recall berada dalam posisi kompromi yang dikendalikan oleh ambang batas keyakinan (Confidence Threshold):")
    add_bullet(" Model hanya mengeluarkan prediksi jika tingkat keyakinan probabilitasnya sangat tinggi. Dampaknya, False Positive mendekati nol (Precision sangat tinggi), namun semprotan fase awal yang tipis akan terabaikan (False Negative bertambah, Recall menurun).", "Ambang Keyakinan Tinggi (Model Sangat Konservatif):")
    add_bullet(" Model mengeluarkan prediksi bahkan untuk kandidat dengan keyakinan rendah. Dampaknya, hampir tidak ada semprotan yang luput (Recall sangat tinggi), namun model rentan menganggap bayangan atau latar belakang sebagai semprotan (False Positive meningkat, Precision anjlok).", "Ambang Keyakinan Rendah (Model Sangat Agresif):")

    add_h2("3.5 F1-Score (Rata-rata Harmonik Seimbang)")
    add_p("F1-Score adalah metrik tunggal yang mengukur titik temu optimal antara Precision dan Recall menggunakan rata-rata harmonik (harmonic mean). Rata-rata harmonik digunakan karena memberi penalti berat jika salah satu dari Precision atau Recall bernilai sangat rendah.")
    add_p("Formulasi:\nF1-Score = 2 * (Precision * Recall) / (Precision + Recall) = 2*TP / (2*TP + FP + FN)")
    add_p("F1-Score mencapai nilai tertinggi ketika model berhasil menemukan sebagian besar objek (Recall tinggi) sekaligus mempertahankan tingkat kepalsuan deteksi yang sangat rendah (Precision tinggi).")

    add_h2("3.6 Kurva Precision-Recall (P-R Curve)")
    add_p("Kurva Precision-Recall menggambarkan perubahan nilai Precision (pada sumbu vertikal Y) terhadap nilai Recall (pada sumbu horizontal X) ketika ambang batas keyakinan model divariasikan secara kontinu dari 1.0 turun hingga 0.0.")
    add_p("Karakteristik Kurva P-R:\n• Kurva ideal meregang ke sudut kanan atas (Precision = 1.0 pada Recall = 1.0).\n• Area di bawah kurva (Area Under Curve / AUC) mencerminkan kualitas intrinsik arsitektur model terlepas dari pemilihan nilai threshold spesifik saat deployment.")

    # ==========================
    # BAB 4
    # ==========================
    add_h1("BAB 4: MEAN AVERAGE PRECISION (mAP) SEBAGAI STANDAR EMAS")
    
    add_h2("4.1 Average Precision (AP) dan Perhitungan Area Under Curve (AUC)")
    add_p("Average Precision (AP) merangkum bentuk keseluruhan kurva Precision-Recall menjadi satu skor numerik skalar tunggal, yang secara konseptual setara dengan menghitung integral luas di bawah kurva P-R:")
    add_p("AP = Integral dari r=0 hingga r=1 p(r) dr")
    add_p("Dalam implementasi standar COCO dan YOLOv8, kurva P-R dihaluskan menggunakan interpolasi kontinu (all-point continuous interpolation), di mana precision pada suatu tingkat recall diambil dari nilai precision maksimum di sebelah kanan titik recall tersebut:")
    add_p("p_interp(r) = MAX_{r_tilde >= r} p(r_tilde)\nMetode ini menghilangkan fluktuasi zig-zag yang tidak adil dan menghasilkan pengukuran kapasitas model yang objektif.")

    add_h2("4.2 Mean Average Precision (mAP) Lintas Kategori")
    add_p("Mean Average Precision (mAP) adalah nilai rata-rata dari skor AP yang dihitung di seluruh kelas kategori objek dalam suatu dataset:")
    add_p("mAP = (1 / N_class) * SUM_{i=1..N_class} AP_i")
    add_p("Catatan Desain Rekayasa:\nPada domain analisis karakterisasi semprotan fluida yang bersifat single-class (hanya mendeteksi kelas 'spray_plume'), nilai mAP secara matematis identik dengan nilai AP dari kelas semprotan tersebut: mAP = AP_spray.")

    add_h2("4.3 mAP@50 (mAP pada Ambang IoU 0.50)")
    add_p("mAP@50 (sering dinotasikan sebagai mAP@0.5 atau mAP50) adalah Mean Average Precision yang dihitung pada ambang batas IoU tunggal tau_IoU = 0.50.")
    add_bullet(" Menguji kapabilitas dasar lokalisasi model: apakah model mampu menemukan lokasi kasar semprotan fluida di ruang citra dengan tumpang tindih minimal 50%.", "Tujuan Pengukuran:")
    add_bullet(" Sangat toleran terhadap ketidaksempurnaan garis kontur di batas kabut semprotan yang tipis.", "Sifat Toleransi:")
    add_bullet(" Menjadi indikator utama apakah sistem inferensi mampu berfungsi sebagai pemicu pendeteksi semprotan (spray trigger detector) yang andal.", "Relevansi Operasional:")

    add_h2("4.4 mAP@50-95 (Standar COCO: Ambang IoU 0.50 hingga 0.95)")
    add_p("mAP@50-95 (dikenal sebagai Primary COCO Metric) adalah rata-rata aritmatika dari nilai mAP yang dievaluasi pada 10 ambang batas IoU diskret yang berbeda, mulai dari 0.50 hingga 0.95 dengan interval langkah 0.05:")
    add_p("mAP@50-95 = (1/10) * [ mAP@0.50 + mAP@0.55 + mAP@0.60 + mAP@0.65 + mAP@0.70 + mAP@0.75 + mAP@0.80 + mAP@0.85 + mAP@0.90 + mAP@0.95 ]")
    add_p("Signifikansi Teknis:\n• Nilai mAP@50-95 secara inheren SELALU LEBIH RENDAH daripada mAP@50 karena mengharuskan model mempertahankan ketepatan spasial pada ambang batas sangat ketat (IoU 0.85, 0.90, dan 0.95).\n• Metrik ini mencerminkan keandalan batas morfologis halus model. Jika model menghasilkan bentuk yang sedikit membengkak atau menyusut dibanding bentuk asli semprotan, nilai mAP@50-95 akan langsung anjlok meskipun nilai mAP@50 tetap tinggi.")

    add_h2("4.5 Analisis Komparatif: mAP (Box) vs. mAP (Mask / Segmentation)")
    add_p("Dalam arsitektur YOLOv8 segmentasi, metrik mAP dilaporkan secara terpisah pada dua domain keluaran:")
    add_bullet(" Mengukur kualitas kotak pembatas segiempat terluar. Nilainya cenderung lebih tinggi karena kotak memiliki toleransi luas yang lebih longgar.", "mAP (Box / Plume):")
    add_bullet(" Mengukur ketepatan piksel-demi-piksel dari topeng poligon semprotan. Nilainya cenderung lebih rendah karena mengevaluasi setiap lekukan kontur kabut semprotan yang semi-transparan.", "mAP (Mask / Segmentasi):")

    add_callout(doc, "mAP (Box) yang tinggi menjamin bahwa siklus semprotan terdeteksi tanpa luput (spray event detection). Sedangkan mAP (Mask) yang tinggi menjamin bahwa perhitungan sudut semprotan (spray angle) dan luasan dispersi kabut (spray area) memiliki akurasi geometri yang valid secara teknik fluida.", "Peran Komplementer Box vs Mask")

    # ==========================
    # BAB 5
    # ==========================
    add_h1("BAB 5: FUNGSI KERUGIAN (LOSS FUNCTIONS) DALAM ARSITEKTUR YOLOV8")
    
    add_h2("5.1 Peran Loss Function dalam Optimasi Backpropagation")
    add_p("Selama fase pelatihan (training), fungsi kerugian (Loss Function) mengkuantifikasi penyimpangan antara keluaran model dengan anotasi acuan. Algoritma optimasi (seperti SGD atau AdamW) menggunakan gradien dari fungsi loss ini untuk memperbarui bobot sinapsis neural network ke arah kesalahan minimum:")
    add_p("L_total = lambda_box * L_box + lambda_seg * L_seg + lambda_cls * L_cls + lambda_dfl * L_dfl")

    add_h2("5.2 Bounding Box Loss (L_box / CIoU Loss)")
    add_p("YOLOv8 menggunakan Complete Intersection over Union (CIoU) Loss untuk melatih koordinat kotak pembatas. CIoU menyempurnakan IoU biasa dengan mengintegrasikan tiga aspek geometri secara simultan:")
    add_bullet(" Memaksimalkan tumpang tindih luasan antara kotak prediksi dan ground truth.", "1. Faktor Luas Overlap:")
    add_bullet(" Menghukum jarak Euclidean antara titik pusat kotak prediksi dan titik pusat kotak acuan, dinormalisasi terhadap panjang diagonal kotak pembungkus terkecil.", "2. Faktor Jarak Titik Pusat (Central Point Distance):")
    add_bullet(" Menghukum ketidakkonsistenan rasio aspek perbandingan antara lebar dan tinggi (width/height).", "3. Faktor Kesamaan Bentuk (Aspect Ratio Consistency):")
    add_p("Formulasi:\nL_box = L_CIoU = 1 - IoU + (rho^2(b, b_gt) / c^2) + alpha * v\nDi mana rho adalah jarak titik tengah, c adalah diagonal kotak pelingkup, dan v mengukur deviasi rasio aspek.")

    add_h2("5.3 Distribution Focal Loss (L_dfl)")
    add_p("Pada arsitektur YOLOv8 yang bertipe anchor-free, koordinat tepi kotak pembatas tidak diregresikan sebagai angka skalar tunggal yang kaku, melainkan dimodelkan sebagai distribusi probabilitas pada rentang nilai diskret.")
    add_p("Manfaat untuk Fluida Spray:\n• Karena batas kabut semprotan bersifat kabur dan tidak berpijak pada tepi yang keras, pemodelan berbasis distribusi memberikan fleksibilitas probabilistik.\n• L_dfl memfokuskan pembelajaran agar probabilitas terkonsentrasi rapat di sekitar koordinat tepi aktual ground truth.")

    add_h2("5.4 Segmentation Mask Loss (L_seg / BCE Mask Loss)")
    add_p("YOLOv8-seg memisahkan pembuatan mask menjadi dua cabang: Protonet (menghasilkan prototipe mask dasar) dan cabang koefisien per-deteksi. Mask akhir diperoleh dari kombinasi linear keduanya yang kemudian dilewatkan ke fungsi aktivasi sigmoid.")
    add_p("Perbedaan antara mask biner prediksi p(i,j) dengan mask ground truth y(i,j) dihukum menggunakan Binary Cross-Entropy (BCE) Loss untuk setiap piksel spasial:\nL_seg = - (1 / (H*W)) * SUM_{i,j} [ y(i,j) * log(p(i,j)) + (1 - y(i,j)) * log(1 - p(i,j)) ]\nLoss ini memaksa model menyalakan nilai 1.0 pada piksel droplet semprotan dan mematikan nilai menjadi 0.0 pada latar belakang.")

    add_h2("5.5 Classification Loss (L_cls / BCE Cls Loss)")
    add_p("Classification Loss mengukur kesalahan model dalam memprediksi kelas objek yang berada di dalam kotak pembatas. Menggunakan formulasi Binary Cross-Entropy (BCE), loss ini melatih model agar memiliki probabilitas tinggi terhadap kelas target ('spray_plume') dan probabilitas rendah terhadap kelas lain atau latar belakang.")

    # ==========================
    # BAB 6
    # ==========================
    add_h1("BAB 6: PARAMETER OPERASIONAL INFERENSI & POST-PROCESSING")
    
    add_h2("6.1 Confidence Threshold (conf / thresh)")
    add_p("Confidence Threshold adalah batas ambang probabilitas minimum yang harus dilampaui oleh suatu kandidat deteksi agar diakui sebagai deteksi valid pada tahap keluaran akhir.")
    add_bullet(" Mengizinkan kandidat berprobabilitas rendah lolos. Berguna saat ingin menangkap semprotan tipis di awal aktuasi, namun meningkatkan risiko timbulnya noise alarm palsu (False Positive).", "Ambang Rendah (misal: conf = 0.15 - 0.25):")
    add_bullet(" Hanya meloloskan deteksi yang sangat diyakini model. Menghilangkan seluruh artefak latar belakang, namun berisiko memotong bagian awal dan akhir semprotan yang kerapatan droplet-nya masih rendah (False Negative).", "Ambang Tinggi (misal: conf = 0.50 - 0.70):")

    add_h2("6.2 Non-Maximum Suppression (NMS) & IoU Threshold")
    add_p("Selama proses forward pass, jaringan konvolusi menghasilkan puluhan hingga ratusan usulan kotak deteksi untuk satu objek semprotan yang sama. Non-Maximum Suppression (NMS) adalah algoritma pasca-pemrosesan wajib untuk mereduksi duplikasi tersebut:")
    add_bullet(" Semua kandidat kotak diurutkan berdasarkan skor keyakinan tertinggi.", "1. Pengurutan:")
    add_bullet(" Kotak dengan skor keyakinan terbesar dipilih sebagai deteksi utama.", "2. Pemilihan Utama:")
    add_bullet(" Nilai IoU dihitung antara kotak utama tersebut dengan seluruh kotak kandidat lainnya.", "3. Evaluasi Tumpang Tindih:")
    add_bullet(" Jika IoU melampaui NMS IoU Threshold (misal: 0.70), kandidat tersebut dihapus permanen karena dianggap sebagai deteksi ganda redundan dari semprotan yang sama.", "4. Eliminasi Redundansi:")

    add_h2("6.3 Resolusi Spasial Citra Input (imgsz)")
    add_p("Parameter imgsz menentukan dimensi matriks piksel citra sebelum diproses oleh model (misal: 640x640, 1024x1024, atau 1280x1280 piksel).")
    add_bullet(" Resolusi yang lebih tinggi mempertahankan detail partikel kabut droplet mikro dan batas nozzle, menghasilkan prediksi mask yang jauh lebih tajam.", "Keuntungan Resolusi Tinggi:")
    add_bullet(" Beban memori GPU dan waktu komputasi meningkat secara kuadratik seiring bertambahnya dimensi piksel, yang dapat menurunkan laju frame-per-detik (FPS) pada sistem inferensi video.", "Kompromi Kecepatan:")

    add_h2("6.4 Mask Ratio dan Downsampling Prototype Mask")
    add_p("Pada arsitektur YOLOv8-seg, topeng prototipe Protonet diproses pada resolusi 1/4 dari ukuran input citra (faktor downsampling 4x) untuk menjaga efisiensi komputasi dan memori real-time. Pada tahap akhir, topeng tersebut di-upscale kembali menggunakan interpolasi bilinear. Mengetahui aspek teknis ini penting agar analis memahami bahwa sedikit kehalusan batas piksel merupakan trade-off desain arsitektural demi kecepatan inferensi tinggi.")

    # ==========================
    # BAB 7
    # ==========================
    add_h1("BAB 7: MATRIKS EVALUASI LANJUTAN & RELEVANSI FISIK INDUSTRI")
    
    add_h2("7.1 Confusion Matrix & Normalized Confusion Matrix")
    add_p("Confusion Matrix adalah tabel kontingensi dua dimensi yang menyilangkan label kondisi aktual (baris) dengan label hasil prediksi model (kolom):")
    add_bullet(" Menampilkan jumlah absolut kejadian (frekuensi sampel uji) pada setiap kuadran.", "Confusion Matrix Absolut:")
    add_bullet(" Menampilkan rasio persentase yang dinormalisasi per baris (0.00 hingga 1.00 atau 0% hingga 100%). Diagonal utama mencerminkan akurasi keberhasilan penangkapan objek aktual, sedangkan elemen di luar diagonal memperlihatkan secara transparan kemana kesalahan klasifikasi terdistribusi.", "Normalized Confusion Matrix:")

    add_h2("7.2 Deteksi Latar Belakang (Background False Positives vs. False Negatives)")
    add_p("Pada domain karakterisasi semprotan kosmetik dan farmasi, interaksi model dengan latar belakang memiliki konsekuensi operasional:")
    add_bullet(" Terjadi ketika bayangan aktuator, pipa mesin, atau pantulan cahaya latar disalahartikan sebagai semprotan. Konsekuensi: Sistem mencatat terjadinya penyemprotan palsu di saat aktuator sedang dalam kondisi diam.", "Background False Positive (Latar Dituduh Spray):")
    add_bullet(" Terjadi ketika semprotan aktual tidak terdeteksi dan dianggap sebagai latar belakang kosong oleh model (biasanya terjadi saat kabut sangat tipis pada fase disipasi akhir). Konsekuensi: Durasi waktu penyemprotan terhitung lebih singkat dari durasi aktual.", "Background False Negative (Spray Dianggap Latar):")

    add_h2("7.3 Dampak Kesalahan Metrik terhadap Ekstraksi Parameter Fisik Fluida")
    add_p("Evaluasi model visi komputer bukan sekadar urusan metrik perangkat lunak, melainkan berimplikasi langsung terhadap validitas pengukuran parameter fisik fluida semprotan:")
    add_bullet(" Sudut bukaan semprotan dihitung dari garis singgung titik asal nozzle terhadap kontur terluar plume semprotan pada jarak aksial tertentu. Jika model memiliki nilai mask IoU rendah (kontur tepi tidak presisi), kemiringan garis singgung akan menyimpang beberapa derajat, merusak standardisasi kualitas nozzle semprotan.", "1. Pengaruh Mask IoU terhadap Sudut Semprot (Spray Angle):")
    add_bullet(" Panjang penetrasi semprotan diukur dari nozzle ke batas terdepan semprotan. Ketidaktepatan bounding box bagian bawah menyebabkan kesalahan perhitungan laju kecepatan propagasi fluida di udara.", "2. Pengaruh Box Metrik terhadap Panjang Penetrasi (Penetration Length):")
    add_bullet(" Luasan piksel mask semprotan digunakan untuk mengestimasi volume dispersi kabut cairan. Nilai mask precision yang rendah (mask menggelembung) akan menghasilkan estimasi volume semprotan yang berlebih.", "3. Pengaruh Mask Precision terhadap Estimasi Luas (Spray Area):")

    add_h2("7.4 Glosarium Istilah Singkat (Quick Reference Cheat-Sheet)")
    add_p("Tabel rujukan cepat parameter dan terminologi visi komputer untuk arsitektur YOLOv8 segmentasi:")
    
    # Table Glossary
    tbl_glo = doc.add_table(rows=18, cols=3)
    tbl_glo.alignment = WD_TABLE_ALIGNMENT.CENTER
    glo_headers = ["Istilah / Simbol", "Nama Lengkap", "Definisi Singkat & Makna Operasional"]
    for i, h in enumerate(glo_headers):
        c = tbl_glo.rows[0].cells[i]
        set_cell_background(c, "1B365D")
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Calibri"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    glo_rows = [
        ("IoU", "Intersection over Union", "Rasio tumpang tindih luas irisan dibagi luas gabungan antara ground truth dan prediksi (0.0 - 1.0)."),
        ("TP", "True Positive", "Deteksi yang berhasil menemukan semprotan aktual dengan tumpang tindih spasial valid (IoU >= tau)."),
        ("FP", "False Positive", "Alarm palsu; model mendeteksi semprotan pada latar kosong atau dengan tumpang tindih meleset."),
        ("FN", "False Negative", "Semprotan aktual yang luput dan gagal ditemukan oleh model sama sekali."),
        ("Precision", "Presisi", "Tingkat kemurnian hasil; persentase deteksi model yang terbukti benar (TP / (TP + FP))."),
        ("Recall", "Sensitivitas", "Tingkat kelengkapan; persentase semprotan aktual di lapangan yang berhasil ditemukan (TP / (TP + FN))."),
        ("F1-Score", "Harmonic Mean of P & R", "Skor keseimbangan optimal antara Presisi dan Sensitivitas (2*P*R / (P+R))."),
        ("AP", "Average Precision", "Luas area di bawah kurva Precision-Recall untuk satu kategori objek tertentu."),
        ("mAP", "Mean Average Precision", "Rata-rata nilai AP di seluruh kategori objek yang dievaluasi dalam dataset."),
        ("mAP@50", "mAP at IoU 0.50", "mAP yang dihitung dengan syarat tumpang tindih moderat (>= 50%). Menguji lokalisasi global."),
        ("mAP@50-95", "COCO Primary Metric", "Rata-rata mAP pada 10 ambang batas IoU (0.50 s.d. 0.95). Menguji kepresisian garis tepi kontur."),
        ("Box Metrics", "Bounding Box Metrics", "Evaluasi performa yang didasarkan pada batas luar segiempat pembatas objek."),
        ("Mask Metrics", "Segmentation Mask Metrics", "Evaluasi performa yang didasarkan pada kecocokan piksel-demi-piksel topeng poligon objek."),
        ("CIoU Loss", "Complete IoU Loss", "Fungsi penalti regresi kotak pembatas (overlap, jarak titik pusat, dan rasio aspek)."),
        ("DFL Loss", "Distribution Focal Loss", "Fungsi penalti regresi koordinat tepi berbasis distribusi probabilitas untuk batas kabur."),
        ("BCE Loss", "Binary Cross-Entropy Loss", "Fungsi penalti logaritmik untuk klasifikasi biner dan aktivasi piksel topeng (on/off)."),
        ("Conf Thresh", "Confidence Threshold", "Ambang probabilitas minimum agar kandidat deteksi diizinkan lolos ke hasil akhir.")
    ]
    for r_idx, row_data in enumerate(glo_rows, start=1):
        row = tbl_glo.rows[r_idx]
        bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=50, bottom=50, left=70, right=70)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.name = "Calibri"
            r.font.size = Pt(8.5)
            r.font.color.rgb = COLOR_BODY

    # Simpan dokumen
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    print(f"[SUCCESS] Dokumen teknis Word (.docx) berhasil dibuat di: {output_path}")

if __name__ == "__main__":
    import sys
    out_file = sys.argv[1] if len(sys.argv) > 1 else "laporan/Dokumen_Teknis_Metrik_dan_Evaluasi_AI_YOLO.docx"
    build_document(out_file)
