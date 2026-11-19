import json
from pathlib import Path

def inspect_nb(path):
    print("="*60)
    print("Inspecting:", path.name)
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if any(k in src.lower() for k in ["evaluasi", "results.png", "results.csv", "metrik", "val"]):
            print(f"Cell {i} ({c['cell_type']}):")
            print(src[:400])
            print("-" * 40)

inspect_nb(Path(r"D:\Tugas Kuliah\Semester 6\MBKM - Magang\Paragon Spray\AI Analysis\finetune_spray_analysis.ipynb"))
inspect_nb(Path(r"D:\Tugas Kuliah\Semester 6\MBKM - Magang\Paragon Spray\AI Analysis\train_spray_analysis.ipynb"))
