"""Moves the pairs of a pool to another machine: the pair tables and the anchor table, with every file they name.

A pair table names the winner, the loser, the prompt and the condition vector of every pair; the anchor table names the
clip and the prompt of every own neutral clip. This script copies exactly those files into one archive, under their
paths relative to --root, and writes the tables with the paths rewritten to --to (the folder on the other machine that
the archive is unpacked into). The vector sources (column source) are ESD paths relative to the ESD root and stay as
they are. A list of the files with their SHA-256 goes into the archive (pack_files_<archive name>.csv), so that the
unpacked copy can be checked; --check reads the lists of every archive unpacked into the folder.
usage: python pack_pairs.py <archive .tar> --root <local folder the paths lie under> --to <folder on the other machine>
           <table.csv> [...]
       on the other machine: tar -xf <archive> -C <folder>; then python pack_pairs.py --check <folder>
"""
import argparse
import hashlib
import io
import tarfile
from pathlib import Path, PureWindowsPath

import pandas as pd

COLUMNS = ["winner", "loser", "prompt_wav", "emo_vec", "clip"]  # the columns of make_pairs.py and make_anchor_table.py that name files
p = argparse.ArgumentParser()
p.add_argument("archive")
p.add_argument("tables", nargs="*")
p.add_argument("--root", default="")
p.add_argument("--to", default="")
p.add_argument("--check", action="store_true")  # on the other machine: the archive's path is the unpacked folder
args = p.parse_intermixed_args()  # the tables may follow the options
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()

if args.check:
    folder = Path(args.archive)
    rows = pd.concat([pd.read_csv(f) for f in sorted(folder.glob("pack_files_*.csv"))], ignore_index=True)  # every archive unpacked here
    bad = [r.rel for r in rows.itertuples() if not (folder / r.rel).exists() or sha(folder / r.rel) != r.sha256]
    print(f"{len(rows)} files listed; missing or changed: {len(bad)}" + (f" (first: {bad[:3]})" if bad else ""))
    raise SystemExit(1 if bad else 0)

root = Path(args.root).resolve()
norm = lambda s: Path(str(PureWindowsPath(s)) if "\\" in str(s) else str(s)).resolve()
files, out_tables = {}, {}
for t in args.tables:
    d = pd.read_csv(t)
    for c in COLUMNS:
        if c not in d.columns:
            continue
        new = []
        for v in d[c]:
            if not isinstance(v, str) or not v:
                new.append(v)
                continue
            f = norm(v)
            rel = f.relative_to(root).as_posix()  # fails if a file lies outside --root
            files[rel] = f
            new.append(f"{args.to.rstrip('/')}/{rel}")
        d[c] = new
    out_tables[Path(t).name] = d
with tarfile.open(args.archive, "w") as tar:
    listing = []
    for rel, f in sorted(files.items()):
        tar.add(str(f), arcname=rel)
        listing.append(dict(rel=rel, bytes=f.stat().st_size, sha256=sha(f)))
    for name, d in out_tables.items():  # the rewritten tables, next to the files
        data = d.to_csv(index=False).encode("utf-8")
        info = tarfile.TarInfo(f"tables/{name}")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    data = pd.DataFrame(listing).to_csv(index=False).encode("utf-8")
    info = tarfile.TarInfo(f"pack_files_{Path(args.archive).stem}.csv")  # one listing per archive, so that two archives unpacked into one folder are both checked
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
print(f"{args.archive}: {len(files)} files ({sum(f.stat().st_size for f in files.values()) / 2**30:.2f} GB) and {len(out_tables)} tables rewritten to {args.to}")
