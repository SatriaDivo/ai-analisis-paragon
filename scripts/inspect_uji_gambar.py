import json

with open("testing/uji_satu_gambar.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    print(f"Cell {idx} ({c['cell_type']}): {src.splitlines()[0] if src.splitlines() else ''}")

