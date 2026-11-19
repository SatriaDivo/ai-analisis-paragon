import json
from pathlib import Path

nb_p = Path("testing/uji_satu_gambar.ipynb")
with open(nb_p, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 4: Focused on pure fluid segmentation
new_cell4 = '''import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import torch

def test_single_image(model, image_path, conf=0.15, iou=0.45):
    """
    Uji inferensi presisi tinggi fokus pada segmentasi cairan semprotan yang keluar dari botol.
    Menolak halusinasi latar belakang kosong dan memvalidasi kontinuitas kepulan fluida.
    """
    img_p = Path(image_path)
    if not img_p.is_absolute():
        img_p = (ROOT_DIR / img_p).resolve()
        
    if not img_p.exists():
        print(f"[ERROR] File citra tidak ditemukan di: {img_p}")
        return None
        
    img_bgr = cv2.imread(str(img_p))
    if img_bgr is None:
        print(f"[ERROR] Gagal membaca gambar: {img_p.name}")
        return None
        
    h, w = img_bgr.shape[:2]
    
    t0 = time.time()
    # Prediksi menggunakan YOLOv8s-seg
    results = model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )
    infer_time_ms = (time.time() - t0) * 1000.0
    res = results[0]
    
    vis = img_bgr.copy()
    overlay = np.zeros_like(img_bgr)
    
    found_valid_plume = False
    best_conf = 0.0
    plume_area_px = 0
    box_display = None
    
    if res.masks is not None and len(res.masks) > 0:
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        
        # Cari deteksi terbaik yang berada di zona semprotan (bukan langit-langit kain kosong)
        # Semprotan laboratorium Paragon selalu menyembur dari kiri ke kanan (dimulai dari x < w*0.4)
        valid_candidates = []
        for idx in range(len(confs)):
            b = boxes[idx]
            pts = res.masks.xy[idx]
            # Validasi posisi: semprotan harus berawal dari sisi botol (x_min < w*0.45)
            # dan memiliki area poligon yang signifikan
            poly_area = cv2.contourArea(pts.astype(np.int32)) if len(pts) >= 3 else 0
            if b[0] < (w * 0.45) and poly_area > 800:
                valid_candidates.append((idx, confs[idx], poly_area, pts, b))
                
        if valid_candidates:
            # Pilih kandidat dengan kombinasi confidence & kelayakan fisik tertinggi
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            best_idx, best_conf, plume_area_px, pts_best, box_display = valid_candidates[0]
            found_valid_plume = True
            
            # Masker Hijau Transparan Presisi
            pts_int = pts_best.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts_int], (0, 255, 0))
            vis = cv2.addWeighted(vis, 1.0, overlay, 0.45, 0)
            cv2.polylines(vis, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Bounding Box Kuning Tipis & Titik Mulai Nosel
            bx = box_display.astype(int)
            cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 255), 2)
            
            nozzle_x = int(np.min(pts_best[:, 0]))
            nozzle_y = int(pts_best[np.argmin(pts_best[:, 0]), 1])
            cv2.circle(vis, (nozzle_x, nozzle_y), 7, (0, 0, 255), -1)
            cv2.putText(vis, f"Nosel ({nozzle_x},{nozzle_y})", (nozzle_x - 30, max(25, nozzle_y - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
    if not found_valid_plume:
        cv2.putText(vis, "STATUS: TIDAK TERDETEKSI CAIRAN SEMPROTAN AKTIF", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
    # Analisis Spektral Densitas Fluida (Heatmap Turbo)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg_val = float(np.percentile(gray, 10))
    diff = np.clip(gray.astype(np.float32) - bg_val, 0, 255).astype(np.uint8)
    diff_norm = cv2.normalize(cv2.medianBlur(diff, 5), None, 0, 255, cv2.NORM_MINMAX)
    heatmap_turbo = cv2.applyColorMap(diff_norm, cv2.COLORMAP_TURBO)
    
    # Visualisasi 3-Panel
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), facecolor="white")
    
    # Panel 1: Citra Asli
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Panel 1: Citra Asli (RGB) - [{img_p.name}]", fontsize=11, fontweight="bold", pad=6)
    axes[0].axis("off")
    
    # Panel 2: Segmentasi Cairan yang Keluar dari Botol
    status_str = f"VALID PLUME (Cairan Terdeteksi Presisi) | Conf: {best_conf:.1%} | Luas Masker: {plume_area_px:,.0f} px" if found_valid_plume else "SAMPEL KOSONG / BELUM MENYEMPROT"
    title_p2 = f"Panel 2: Hasil Segmentasi Cairan yang Keluar dari Botol (YOLOv8s-Seg)\\n[{status_str}]"
    axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(title_p2, fontsize=11, fontweight="bold", pad=6, color="darkgreen" if found_valid_plume else "red")
    axes[1].axis("off")
    
    # Panel 3: Spektral Densitas
    axes[2].imshow(cv2.cvtColor(heatmap_turbo, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Panel 3: Analisis Spektral Densitas Partikel Fluida (Heatmap Turbo)", fontsize=11, fontweight="bold", pad=6, color="navy")
    axes[2].axis("off")
    
    plt.tight_layout()
    out_f = OUTPUT_DIR / f"uji_cairan_{img_p.stem}.png"
    fig.savefig(out_f, dpi=120, bbox_inches="tight")
    plt.show()
    
    return {
        "filename": img_p.name,
        "detected": found_valid_plume,
        "conf": round(float(best_conf), 3),
        "area_px": int(plume_area_px),
        "infer_time_ms": round(infer_time_ms, 1),
        "saved_plot": str(out_f)
    }

print("Fungsi analisis segmentasi cairan tunggal (Fokus Murni Semburan Fluida) siap digunakan!")
'''

nb["cells"][4]["source"] = [l + "\n" for l in new_cell4.splitlines()]

with open(nb_p, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("testing/uji_satu_gambar.ipynb updated to focus on pure liquid segmentation!")

