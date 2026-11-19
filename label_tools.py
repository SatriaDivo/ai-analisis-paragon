"""
label_tools.py — alat bantu review polygon spray (jalur "A": label manual terverifikasi).

Alur:
  1) export  : jalankan checkpoint historis pada sebagian gambar -> DRAF polygon format LabelMe
               di dataset_label_drafts/ (gambar disalin ke sana; dataset/ asli tidak disentuh).
  2) Anda koreksi di LabelMe (lihat petunjuk yang dicetak). Draf berlabel "draft" TIDAK dianggap
     label. Hanya shape yang Anda ubah labelnya menjadi "spray_plume" DAN gambar yang Anda centang
     flag "reviewed" yang akan diimpor.
  3) import  : ubah hasil review menjadi label YOLO-seg di dataset_label_staging/.
  4) promote : validasi polygon staging dan salin ke dataset_manual_labels/ lengkap dengan
               berkas <stem>.reviewed.json (mengikat hash sumber & hash label) agar lolos gate training.

Jalankan dari dalam repo (folder dataset/ ditemukan otomatis ke atas), contoh:
  python label_tools.py export --per-group 12
  labelme dataset_label_drafts --labels dataset_label_drafts/labels.txt --flags dataset_label_drafts/flags.txt
  python label_tools.py import
  python label_tools.py promote
"""
import argparse, csv, hashlib, json, re, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

CLASS_LABEL = "spray_plume"
DRAFT_LABEL = "draft"
REVIEW_FLAG = "reviewed"


def find_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    while not (p / "dataset").is_dir():
        if p.parent == p:
            sys.exit("Folder dataset/ tidak ditemukan. Jalankan dari dalam repo proyek.")
        p = p.parent
    return p


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_date(name):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else "unknown"


def pick_evenly(files, n):
    """Pilih n berkas berjarak merata dari daftar terurut (agar bervariasi sepanjang sesi)."""
    if n <= 0 or n >= len(files):
        return list(files)
    idx = sorted(set(int(round(i)) for i in np.linspace(0, len(files) - 1, n)))
    return [files[i] for i in idx]


def simplify(poly, max_points_ratio):
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    eps = max_points_ratio * cv2.arcLength(pts, True)
    approx = cv2.approxPolyDP(pts, eps, True).reshape(-1, 2)
    return approx if len(approx) >= 3 else pts.reshape(-1, 2)


def is_valid_polygon_file(path: Path) -> bool:
    """Periksa validitas sintaks format polygon YOLO."""
    try:
        lines = [line.strip().split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return False
        for tokens in lines:
            if tokens[0] != "0" or len(tokens) < 7 or len(tokens) % 2 != 1:
                return False
            if not all(0.0 <= float(value) <= 1.0 for value in tokens[1:]):
                return False
        return True
    except (OSError, ValueError):
        return False


# ----------------------------------------------------------------------------- export
def cmd_export(args):
    root = find_root()
    ds, out = root / "dataset", root / args.out
    ckpt = Path(args.checkpoint) if args.checkpoint else (
        root / "runs" / "notebook" / "spray_training_runs" / "spraybot_seg_v1" / "weights" / "best.pt")
    if not ckpt.is_file():
        sys.exit(f"Checkpoint tidak ditemukan: {ckpt}")
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        sys.exit("Paket ultralytics tidak terpasang.")

    by_date = {}
    for p in sorted(ds.glob("*.png")):
        by_date.setdefault(capture_date(p.name), []).append(p)
    if not by_date:
        sys.exit("Tidak ada *.png di dataset/")
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ckpt))
    manifest = []
    for date, files in sorted(by_date.items()):
        chosen = pick_evenly(files, args.per_group)
        print(f"{date}: {len(files)} gambar -> {len(chosen)} dipilih")
        for src in chosen:
            dst = out / src.name
            json_path = out / (src.stem + ".json")
            if json_path.exists() and not args.overwrite:
                shapes_count = 0
                try:
                    existing_data = json.loads(json_path.read_text(encoding="utf8"))
                    shapes_count = len(existing_data.get("shapes", []))
                except Exception:
                    pass
                manifest.append({"image": src.name, "group": date, "source_sha256": sha256(src), "draft_shapes": shapes_count})
                continue
            shutil.copy2(src, dst)
            img = cv2.imread(str(dst))
            h, w = img.shape[:2]
            res = model.predict(str(dst), conf=args.conf, verbose=False)[0]
            polys = [] if res.masks is None else [p for p in res.masks.xy if len(p) >= 3]
            shapes = [{"label": DRAFT_LABEL, "points": simplify(p, args.simplify).round(1).tolist(),
                       "group_id": None, "description": "draf model historis - periksa, koreksi, lalu ganti label",
                       "shape_type": "polygon", "flags": {}} for p in polys]
            json_path.write_text(json.dumps({
                "version": "5.0.0", "flags": {REVIEW_FLAG: False}, "shapes": shapes,
                "imagePath": src.name, "imageData": None, "imageHeight": h, "imageWidth": w},
                ensure_ascii=False, indent=1), encoding="utf8")
            manifest.append({"image": src.name, "group": date, "source_sha256": sha256(src), "draft_shapes": len(shapes)})
    (out / "labels.txt").write_text(f"{CLASS_LABEL}\n{DRAFT_LABEL}\n", encoding="utf8")
    (out / "flags.txt").write_text(f"{REVIEW_FLAG}\n", encoding="utf8")
    with open(out / "draft_manifest.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "group", "source_sha256", "draft_shapes"])
        wr.writeheader(); wr.writerows(manifest)
    print(f"\n{len(manifest)} draf ditulis ke {out}")
    print("\nBuka dengan:")
    print(f"  labelme {args.out} --labels {args.out}/labels.txt --flags {args.out}/flags.txt")
    print("\nUntuk SETIAP gambar:")
    print(f"  1. Hapus shape yang bukan kabut (busur penggaris derajat, garis pembatas papan, dll.).")
    print(f"  2. Koreksi tepi polygon kabut; tambah polygon baru bila perlu.")
    print(f"  3. Ganti label shape kabut yang sudah benar dari '{DRAFT_LABEL}' menjadi '{CLASS_LABEL}'.")
    print(f"  4. Centang flag '{REVIEW_FLAG}', lalu simpan (Ctrl+S).")
    print(f"Shape yang masih '{DRAFT_LABEL}' atau gambar tanpa flag '{REVIEW_FLAG}' akan diabaikan.")


# ----------------------------------------------------------------------------- import
def cmd_import(args):
    root = find_root()
    ds, drafts, stage = root / "dataset", root / args.drafts, root / args.staging
    if not drafts.is_dir():
        sys.exit(f"{drafts} tidak ada. Jalankan 'export' dulu.")
    stage.mkdir(parents=True, exist_ok=True)
    rows = []
    for jp in sorted(drafts.glob("*.json")):
        rec = json.loads(jp.read_text(encoding="utf8"))
        name = rec.get("imagePath", jp.stem + ".png")
        row = {"image": name, "status": "", "polygons": 0, "note": ""}
        rows.append(row)
        if not rec.get("flags", {}).get(REVIEW_FLAG):
            row["status"], row["note"] = "skipped", f"flag '{REVIEW_FLAG}' belum dicentang"; continue
        src, copy = ds / name, drafts / name
        if not src.is_file() or not copy.is_file():
            row["status"], row["note"] = "rejected", "gambar sumber atau salinan tidak ada"; continue
        if sha256(src) != sha256(copy):
            row["status"], row["note"] = "rejected", "salinan gambar berbeda dari dataset/ (hash tidak cocok)"; continue
        w, h = rec["imageWidth"], rec["imageHeight"]
        if (w, h) != tuple(cv2.imread(str(src)).shape[1::-1]):
            row["status"], row["note"] = "rejected", "ukuran gambar di JSON tidak cocok"; continue
        lines, bad = [], 0
        for s in rec.get("shapes", []):
            if s.get("label") != CLASS_LABEL:
                continue
            pts = np.asarray(s["points"], dtype=float)
            if s.get("shape_type") != "polygon" or len(pts) < 3:
                bad += 1; continue
            pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1); pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
            if cv2.contourArea((pts * 1000).astype(np.float32).reshape(-1, 1, 2)) <= 0:
                bad += 1; continue
            lines.append("0 " + " ".join(f"{v:.6f}" for v in pts.reshape(-1)) + "\n")
        if not lines and not args.allow_empty:
            row["status"], row["note"] = "skipped", f"tidak ada shape berlabel '{CLASS_LABEL}' (pakai --allow-empty bila memang tanpa kabut)"; continue
        (stage / (Path(name).stem + ".txt")).write_text("".join(lines), encoding="utf8")
        row["status"], row["polygons"] = "staged", len(lines)
        if bad: row["note"] = f"{bad} shape tidak valid diabaikan"
    with open(stage / "staging_report.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "status", "polygons", "note"])
        wr.writeheader(); wr.writerows(rows)
    counts = {k: sum(r["status"] == k for r in rows) for k in ("staged", "skipped", "rejected")}
    print("Hasil impor:", counts, "->", stage)
    for r in rows:
        if r["status"] != "staged":
            print(f"  [{r['status']}] {r['image']}: {r['note']}")
    print("\nCATATAN: label ini baru berstatus STAGING. Jalankan 'python label_tools.py promote' atau fungsi "
          "promote_staging() di notebook untuk memindahkan ke dataset_manual_labels/ beserta <stem>.reviewed.json.")
    return rows


# ----------------------------------------------------------------------------- promote
def cmd_promote(args):
    root = find_root()
    ds = root / "dataset"
    stage = root / args.staging
    dest = root / args.dest
    if not stage.is_dir():
        sys.exit(f"Folder staging '{stage}' tidak ada. Jalankan 'import' dulu.")
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    promoted_count = 0
    now_str = datetime.now(timezone.utc).isoformat()
    for txt in sorted(stage.glob("*.txt")):
        stem = txt.stem
        img_name = stem + ".png"
        src_img = ds / img_name
        row = {"image": img_name, "status": "", "note": ""}
        rows.append(row)
        if not src_img.is_file():
            row["status"], row["note"] = "rejected", f"citra sumber '{img_name}' tidak ditemukan di dataset/"
            continue
        if not is_valid_polygon_file(txt):
            row["status"], row["note"] = "rejected", "sintaks polygon YOLO tidak valid"
            continue

        target_txt = dest / txt.name
        shutil.copy2(txt, target_txt)
        img_hash = sha256(src_img)
        lbl_hash = sha256(target_txt)

        review_meta = {
            "verified": True,
            "reviewer_decision": "approved",
            "source_hash": img_hash,
            "label_hash": lbl_hash,
            "reviewer": getattr(args, "reviewer", "operator"),
            "timestamp": now_str,
            "note": "Anotasi manual divalidasi via label_tools"
        }
        review_path = dest / (stem + ".reviewed.json")
        review_path.write_text(json.dumps(review_meta, ensure_ascii=False, indent=2), encoding="utf8")
        row["status"], row["note"] = "promoted", "lolos verifikasi & berkas .reviewed.json dibuat"
        promoted_count += 1

    report_path = dest / "promotion_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "status", "note"])
        wr.writeheader(); wr.writerows(rows)

    counts = {k: sum(r["status"] == k for r in rows) for k in ("promoted", "rejected")}
    print(f"Hasil promosi ke {dest}:", counts)
    for r in rows:
        if r["status"] != "promoted":
            print(f"  [{r['status']}] {r['image']}: {r['note']}")
    print(f"\n{promoted_count} label berhasil diverifikasi dan diikat hash untuk pelatihan.")
    return rows


# ----------------------------------------------------------------------------- programmatic helpers
def export_drafts(root=None, checkpoint=None, per_group=12, conf=0.25, simplify_ratio=0.004,
                  out="dataset_label_drafts", overwrite=False):
    root_p = Path(root or find_root())
    args = argparse.Namespace(
        out=out, checkpoint=checkpoint, per_group=per_group, conf=conf,
        simplify=simplify_ratio, overwrite=overwrite
    )
    return cmd_export(args)


def import_staging(root=None, drafts="dataset_label_drafts", staging="dataset_label_staging",
                   allow_empty=False):
    root_p = Path(root or find_root())
    args = argparse.Namespace(
        drafts=drafts, staging=staging, allow_empty=allow_empty
    )
    return cmd_import(args)


def promote_staging(root=None, staging="dataset_label_staging", dest="dataset_manual_labels",
                    reviewer="operator"):
    root_p = Path(root or find_root())
    args = argparse.Namespace(
        staging=staging, dest=dest, reviewer=reviewer
    )
    return cmd_promote(args)


def get_review_status(root=None, drafts="dataset_label_drafts", staging="dataset_label_staging",
                      manual="dataset_manual_labels"):
    root_p = Path(root or find_root())
    d_dir = root_p / drafts
    s_dir = root_p / staging
    m_dir = root_p / manual

    d_total = len(list(d_dir.glob("*.json"))) if d_dir.is_dir() else 0
    d_reviewed = 0
    if d_dir.is_dir():
        for jp in d_dir.glob("*.json"):
            try:
                rec = json.loads(jp.read_text(encoding="utf8"))
                if rec.get("flags", {}).get(REVIEW_FLAG):
                    d_reviewed += 1
            except Exception:
                pass

    s_total = len(list(s_dir.glob("*.txt"))) if s_dir.is_dir() else 0
    m_total = len(list(m_dir.glob("*.reviewed.json"))) if m_dir.is_dir() else 0

    return {
        "drafts_dir_exists": d_dir.is_dir(),
        "drafts_total": d_total,
        "drafts_reviewed": d_reviewed,
        "staging_dir_exists": s_dir.is_dir(),
        "staging_labels": s_total,
        "manual_dir_exists": m_dir.is_dir(),
        "approved_manual_labels": m_total
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="buat draf polygon LabelMe dari checkpoint historis")
    e.add_argument("--per-group", type=int, default=12, help="jumlah gambar per tanggal tangkapan (default 12)")
    e.add_argument("--conf", type=float, default=0.25)
    e.add_argument("--simplify", type=float, default=0.004, help="rasio penyederhanaan polygon (default 0.004)")
    e.add_argument("--checkpoint", default=None)
    e.add_argument("--out", default="dataset_label_drafts")
    e.add_argument("--overwrite", action="store_true", help="timpa draf yang sudah ada (menghapus koreksi Anda!)")
    e.set_defaults(fn=cmd_export)
    i = sub.add_parser("import", help="ubah hasil review LabelMe menjadi label YOLO-seg (staging)")
    i.add_argument("--drafts", default="dataset_label_drafts")
    i.add_argument("--staging", default="dataset_label_staging")
    i.add_argument("--allow-empty", action="store_true")
    i.set_defaults(fn=cmd_import)
    p = sub.add_parser("promote", help="promosikan label staging yang valid ke dataset_manual_labels dengan <stem>.reviewed.json")
    p.add_argument("--staging", default="dataset_label_staging")
    p.add_argument("--dest", default="dataset_manual_labels")
    p.add_argument("--reviewer", default="operator")
    p.set_defaults(fn=cmd_promote)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
