#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 harelab-ucsc
"""Cut the published birdseye-review sample out of a full dataset root.

Copies a contiguous frame window of one flight (images, labels.txt rows, the
whole flight_data.db with relative save_loc paths), the matching rows of every
dets/val_*.txt of one run, the run's PREREGISTRATION.json and the selection
file it seals, then writes README.txt and MANIFEST.sha256.

Usage:
    python3 scripts/make_sample.py --src /path/to/dataset --out birdseye-review-sample
"""
import argparse
import hashlib
import shutil
import sqlite3
import sys
from pathlib import Path, PurePosixPath


def frame_index(line: str):
    """Leading frame index of a '<frame_idx> <image_path> x,y,c' row, or None."""
    tok = line.split(None, 1)
    if not tok:
        return None
    try:
        return int(tok[0])
    except ValueError:
        return None


def copy_images(src_flight: Path, out_flight: Path, basenames: dict, first: int, last: int) -> int:
    n = 0
    for fi in range(first, last + 1):
        name = basenames.get(fi)
        if name is None:
            sys.exit(f"frame {fi} has no row in cam0_images_flight_data")
        src = src_flight / name
        if not src.is_file():
            sys.exit(f"missing source image for frame {fi}: {src}")
        shutil.copy2(src, out_flight / name)
        n += 1
    return n


def copy_labels(src: Path, dst: Path, flight: str, first: int, last: int) -> int:
    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            fi = frame_index(line)
            if fi is None or not first <= fi <= last:
                continue
            parts = line.split()
            parts[1] = f"{flight}/{PurePosixPath(parts[1]).name}"
            fout.write(" ".join(parts) + "\n")
            n += 1
    return n


def copy_db(src: Path, dst: Path, flight: str) -> tuple:
    s, d = sqlite3.connect(f"file:{src}?mode=ro", uri=True), sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        s.close()
        d.close()
    con = sqlite3.connect(dst)
    try:
        rows = con.execute("select rowid, save_loc from cam0_images_flight_data").fetchall()
        con.executemany("UPDATE cam0_images_flight_data SET save_loc = ? WHERE rowid = ?",
                        [(f"{flight}/{PurePosixPath(loc).name}", rid) for rid, loc in rows])
        con.commit()
        con.execute("VACUUM")
        n_tags = con.execute("select count(*) from clicks_flight_data").fetchone()[0]
    finally:
        con.close()
    if b"/home/" in dst.read_bytes():
        sys.exit(f"{dst} still contains an absolute /home/ path after rewriting save_loc")
    return len(rows), n_tags


def copy_dets(src_dets: Path, out_dets: Path, first: int, last: int) -> list:
    out = []
    for f in sorted(src_dets.glob("val_*.txt")):
        n = 0
        with open(f) as fin, open(out_dets / f.name, "w") as fout:
            for line in fin:
                fi = frame_index(line)
                if fi is not None and first <= fi <= last:
                    fout.write(line)
                    n += 1
        out.append((f.name, n))
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="dataset root")
    ap.add_argument("--out", type=Path, required=True, help="output directory (must not exist)")
    ap.add_argument("--flight", default="2025_04_16/pieranch_rect")
    ap.add_argument("--run", default="results/17_holdout_pieranch")
    ap.add_argument("--selection", default="results/15_model_selection/selection_pooled.json")
    ap.add_argument("--first", type=int, default=600)
    ap.add_argument("--last", type=int, default=641)
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} already exists")
    if a.first > a.last:
        sys.exit("--first must not exceed --last")

    src_flight, out_flight = a.src / a.flight, a.out / a.flight
    src_run, out_run = a.src / a.run, a.out / a.run
    out_flight.mkdir(parents=True)
    (out_run / "dets").mkdir(parents=True)

    # frame_idx is the positional row of cam0_images_flight_data, read without
    # ORDER BY exactly as geo_eval.Projector does.
    con = sqlite3.connect(f"file:{src_flight / 'flight_data.db'}?mode=ro", uri=True)
    try:
        locs = [r[0] for r in con.execute("select save_loc from cam0_images_flight_data")]
    finally:
        con.close()
    basenames = {i: PurePosixPath(loc).name for i, loc in enumerate(locs)}

    n_img = copy_images(src_flight, out_flight, basenames, a.first, a.last)
    n_lab = copy_labels(src_flight / "labels.txt", out_flight / "labels.txt", a.flight, a.first, a.last)
    n_pose, n_tags = copy_db(src_flight / "flight_data.db", out_flight / "flight_data.db", a.flight)
    dets = copy_dets(src_run / "dets", out_run / "dets", a.first, a.last)
    shutil.copy2(src_run / "PREREGISTRATION.json", out_run / "PREREGISTRATION.json")
    (a.out / a.selection).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.src / a.selection, a.out / a.selection)

    det_lines = "\n".join(f"    {a.run}/dets/{name}  ({n} rows)" for name, n in dets)
    (a.out / "README.txt").write_text(f"""birdseye-review sample data
===========================

Source flight: {a.flight}
Frame window:  {a.first}-{a.last} ({n_img} frames)
Run:           {a.run}

Files
    {a.flight}/cam0_*.png  {n_img} rectified 1920x1200 frames
    {a.flight}/labels.txt  {n_lab} rows: <frame_idx> <image_path> <x>,<y>,<conf>
    {a.flight}/flight_data.db  camera poses, RTK geotags, camera parameters
{det_lines}
    {a.run}/PREREGISTRATION.json  sealed model-selection record
    {a.selection}  the selection file PREREGISTRATION.json seals (byte-identical)
    MANIFEST.sha256  checksums of every other file

flight_data.db is whole (all {n_pose:,} poses, all {n_tags} RTK geotags); frame indices
are positional into it and must not be renumbered.

Run (from this directory, with birdseye-review installed):
    birdseye-review --reviewer you@example.org {a.flight} {a.run}

Sample data: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), harelab-ucsc
""")

    files = sorted(p.relative_to(a.out).as_posix() for p in a.out.rglob("*") if p.is_file())
    (a.out / "MANIFEST.sha256").write_text(
        "".join(f"{sha256(a.out / rel)}  {rel}\n" for rel in files))
    print(f"wrote {a.out}: {n_img} frames, {n_lab} label rows, {len(dets)} dets files, "
          f"{len(files) + 1} files")


if __name__ == "__main__":
    main()
