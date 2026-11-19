import json
from pathlib import Path

nb_path = Path("finetune_spray_analysis.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cell8_source = [
    "# Bobot awal untuk fine-tuning (Gunakan Model Awal yang stabil)\n",
    "BASE_WEIGHTS = ROOT / 'weights' / 'training_awal' / 'spraybot_seg_video_finetuned.pt'\n",
    "if not BASE_WEIGHTS.exists():\n",
    "    BASE_WEIGHTS = ROOT / 'weights' / 'spraybot_seg_video_finetuned.pt'\n",
    "if not BASE_WEIGHTS.exists():\n",
    "    BASE_WEIGHTS = ROOT / 'weights' / 'yolov8s-seg.pt'\n",
    "\n",
    "RUNS_DIR = ROOT / 'runs' / 'spraybot_clean_training'\n",
    "RUN_NAME = 'spraybot_seg_finetune_122vids_clean'\n",
    "\n",
    "print('=' * 80)\n",
    "print('MEMULAI CLEAN FINE-TUNING YOLOV8S-SEG SPRAYBOT (122 VIDEO STUDIO)')\n",
    "print('Mode: Natural Organic Boundaries (Zero 120px Knife-Cut, Retina Masks ON)')\n",
    "print('=' * 80)\n",
    "print(f'Bobot Awal       : {BASE_WEIGHTS}')\n",
    "print(f'File data.yaml   : {DATA_YAML}')\n",
    "print(f'Direktori Output : {RUNS_DIR / RUN_NAME}')\n",
    "print('=' * 80)\n",
    "\n",
    "model = YOLO(str(BASE_WEIGHTS))\n",
    "\n",
    "EPOCHS = 30\n",
    "IMG_SIZE = 1024\n",
    "BATCH_SIZE = 8\n",
    "WORKERS = 0\n",
    "\n",
    "start_train_time = time.time()\n",
    "\n",
    "results = model.train(\n",
    "    data=str(DATA_YAML),\n",
    "    epochs=EPOCHS,\n",
    "    imgsz=IMG_SIZE,\n",
    "    batch=BATCH_SIZE,\n",
    "    workers=WORKERS,\n",
    "    device=0 if torch.cuda.is_available() else 'cpu',\n",
    "    project=str(RUNS_DIR),\n",
    "    name=RUN_NAME,\n",
    "    exist_ok=True,\n",
    "    pretrained=True,\n",
    "    optimizer='auto',\n",
    "    lr0=0.001,\n",
    "    lrf=0.01,\n",
    "    warmup_epochs=2.0,\n",
    "    mosaic=0.0,\n",
    "    erasing=0.0,\n",
    "    mixup=0.0,\n",
    "    retina_masks=True,\n",
    "    close_mosaic=0,\n",
    "    patience=12,\n",
    "    fliplr=0.0,\n",
    "    flipud=0.0,\n",
    "    degrees=0.0,\n",
    "    hsv_h=0.015,\n",
    "    hsv_s=0.2,\n",
    "    hsv_v=0.2,\n",
    "    translate=0.05,\n",
    "    scale=0.10,\n",
    "    verbose=True\n",
    ")\n",
    "\n",
    "elapsed = (time.time() - start_train_time) / 60.0\n",
    "print('=' * 80)\n",
    "print(f'FINE-TUNING SELESAI DALAM {elapsed:.2f} MENIT!')\n",
    "print('=' * 80)\n"
]

# Update Cell 8
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if "model.train" in src:
        c["source"] = new_cell8_source
        print(f"Cell {i} updated!")

# Update Cell 10 paths to point to spraybot_seg_finetune_122vids_clean
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if "spraybot_seg_finetune_122vids" in src and "model.train" not in src:
        new_src = [line.replace("spraybot_seg_finetune_122vids", "spraybot_seg_finetune_122vids_clean") for line in c.get("source", [])]
        c["source"] = new_src
        print(f"Cell {i} updated to spraybot_seg_finetune_122vids_clean!")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully updated!")
