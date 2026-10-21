import csv
from pathlib import Path

root = Path("/data/chi-gpu1/ge96xah/data/ESD")

for split in ["train", "evaluation", "test"]:
    with open(root / "esd_csvs" / f"{split}.csv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["lang"] == "en"]
    missing = [r["relative_audio_path"] for r in rows if not (root / r["relative_audio_path"]).exists()]
    with open(root / f"metadata_{split}.csv", "w", encoding="utf-8", newline="") as f:
        f.write("audio_file|text\n")
        for r in rows:
            f.write(f"{root / r['relative_audio_path']}|{r['transcription'].strip()}\n")
    print(split, len(rows), "rows,", len(missing), "missing files")
