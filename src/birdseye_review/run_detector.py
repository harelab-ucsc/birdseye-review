# SPDX-License-Identifier: MIT
# Copyright (c) 2026 harelab-ucsc
"""
Run the EfficientNetV2B3 detector over a flight and emit RAW per-frame detections.

WHY NOT THE ONBOARD results.txt
-------------------------------
Detections come from this script, not from the birdseye onboard pipeline's
results.txt, which this tool does not read: it is not a detection log. The
onboard semi-supervised
block reprojects *every point of every cluster accumulated so far* into *every*
frame, so rows-per-frame grows 13x across a flight (measured: 243 -> 3,159)
and the same physical detection appears dozens of times. Back-projecting it
cannot recover the original 3D pointcloud: each copy lands on a different
frame's ground plane, so positions smear. Every confidence in it is also 1.0,
so no PR curve or geospatial mAP can be computed from it.

This script produces one row per detection, in the frame that detected it,
with the real heatmap confidence.

Output format matches labels.txt so the existing tooling reads it:
    {frame_index} {image_path} {x},{y},{confidence}

Usage:
    python -m birdseye_review.run_detector --dir 2025_04_16/pieranch_rect \\
        --weights .../models/birdseye_224_224_045.weights.h5 \\
        --thresh 0.03 --out detections_045.txt
"""
import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

from .geo_eval import Projector

# Tiling and post-processing follow harelab-ucsc/birdseye: birdseye/birdseye.py
TILE = (224, 224)
OVERLAP = (30, 40)
DEFAULT_THRESH = 0.03      # on the UNNORMALISED heatmap
MIN_AREA = 10
RADALT_MIN = 3.0           # frames logged below this altitude are on the ground
NOISE_PX = 81.5            # projection-uncertainty buffer


def build_model(weights: Path):
    """Rebuild the architecture the checkpoints were SAVED from.

    Architecture is heatmap_head_deep (harelab-ucsc/birdseye):
    EfficientNetV2B3 -> 5x Conv2DTranspose (512,256,128,64,32) -> 1x1 sigmoid.

    The layer names here (`input_layer`, `functional`) match the saved graph,
    so the load is clean with zero variable warnings. The onboard generator's
    own layer names produce bit-identical heatmaps (max|diff| = 0.000000); a
    hybrid-architecture checkpoint (2x Tconv + Resizing) cannot load as `deep`.
    """
    import tensorflow as tf
    from keras.applications import EfficientNetV2B3
    h, w = TILE
    inp = tf.keras.Input(shape=(h, w, 3), name="input_layer")
    x = tf.keras.ops.cast(inp, "float32")
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    bb = EfficientNetV2B3(input_shape=(h, w, 3), include_top=False, weights="imagenet")
    bb._name = "functional"
    bb.trainable = False
    x = bb(x, training=False)
    for f in (512, 256, 128, 64, 32):
        x = tf.keras.layers.Conv2DTranspose(f, 3, strides=2, padding="same",
                                            activation="relu")(x)
    out = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)
    m = tf.keras.Model(inputs=inp, outputs=out)
    m.load_weights(str(weights))
    return m


def predict_heatmap(frame: np.ndarray, model, batch_size: int = 32) -> np.ndarray:
    """Tiled heatmap inference. Note the trailing right/bottom margins are never
    covered -- `range(0, h-th+1, stride)` -- which is inherited deliberately so
    detections match the original pipeline's field of view."""
    h, w = frame.shape[:2]
    th, tw = TILE
    sy, sx = th - OVERLAP[0], tw - OVERLAP[1]
    tiles, pos = [], []
    for y in range(0, h - th + 1, sy):
        for x in range(0, w - tw + 1, sx):
            tiles.append(frame[y:y + th, x:x + tw])
            pos.append((y, x))
    if not tiles:
        return np.zeros((h, w), np.float32)
    preds = model.predict(np.array(tiles, dtype=np.uint8), batch_size=batch_size, verbose=0)
    acc = np.zeros((h, w), np.float32)
    wgt = np.zeros((h, w), np.float32)
    for i, (y, x) in enumerate(pos):
        acc[y:y + th, x:x + tw] += preds[i].squeeze()
        wgt[y:y + th, x:x + tw] += 1.0
    return np.divide(acc, wgt, out=np.zeros_like(acc), where=wgt > 0)


def detections_from_heatmap(hm: np.ndarray, thresh: float) -> List[Tuple[int, int, float]]:
    """Contour centroids of the thresholded heatmap, plus each one's confidence."""
    blur = cv2.GaussianBlur(hm, (5, 5), sigmaX=1)
    _, binary = cv2.threshold(blur, thresh, 1.0, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours((binary * 255).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) <= MIN_AREA:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        # Peak response inside the contour -- the detection's confidence.
        mask = np.zeros(hm.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 1, -1)
        out.append((cx, cy, float(blur[mask == 1].max())))
    return out


def inner_bound(res) -> np.ndarray:
    w, h = res
    return np.array([[NOISE_PX, NOISE_PX], [w - NOISE_PX, NOISE_PX],
                     [w - NOISE_PX, h - NOISE_PX], [NOISE_PX, h - NOISE_PX]])


def inside(pt, poly) -> bool:
    """In-view polygon test (harelab-ucsc/birdseye: birdseye/birdseye.py),
    signed-area test on all four edges."""
    x, y = pt[0], pt[1]
    s = None
    for i in range(4):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % 4]
        c = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if c == 0:
            continue
        if s is None:
            s = c > 0
        elif (c > 0) != s:
            return False
    return True


def select_frames(proj, d: Path, frame_start: int = 0, stride_frames: int = 1,
                  max_frames: int = None) -> List[Tuple[int, object]]:
    """The frames this detector will actually process, as (frame_idx, pose_row).

    THE ONE DEFINITION. Anything that needs to know "which frames did the
    detector see" -- above all a recall DENOMINATOR -- must call this rather
    than re-deriving it: dividing by every geotag in the flight while
    processing a 250-frame block that can only see a handful caps attainable
    recall near 0.06 regardless of the detector.

    Applies, in order: the contiguous-block slice, the stride, the cap, the
    radalt gate (RADALT_MIN), and local image availability -- a frame
    whose PNG is missing produces no detections and must not be charged as a
    miss either.
    """
    index = {p.name: p for p in Path(d).glob("*.png")}
    rows = list(enumerate(proj.poses))[frame_start:]
    if stride_frames > 1:
        rows = rows[::stride_frames]
    if max_frames:
        rows = rows[:max_frames]
    out = []
    for fi, row in rows:
        if abs(float(row[8])) <= RADALT_MIN:
            continue
        if index.get(Path(str(row[9])).name) is None:
            continue
        out.append((fi, row))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--thresh", type=float, default=DEFAULT_THRESH)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap for validation sweeps; frames are taken in order")
    ap.add_argument("--stride-frames", type=int, default=1,
                    help="process every Nth frame. Leave at 1 when the output feeds "
                         "DBSCAN(min_samples=10): subsampling divides points-per-burrow "
                         "and real burrows get rejected as noise.")
    ap.add_argument("--frame-start", type=int, default=0,
                    help="first frame of a contiguous block")
    ap.add_argument("--thresh-sweep", default=None,
                    help="comma-separated thresholds; extracts all of them from a SINGLE "
                         "inference pass and writes <out>_t<v>.txt per value. Costs the "
                         "same as one threshold because extraction is trivial beside the "
                         "54-tile-per-frame inference.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--rectify", action="store_true", default=True)
    args = ap.parse_args()

    d = Path(args.dir)
    proj = Projector(d / "flight_data.db")
    model = build_model(args.weights)
    map1, map2 = cv2.initUndistortRectifyMap(
        proj.K, proj.D, None, proj.K, (proj.res[0], proj.res[1]), cv2.CV_32FC1)
    poly = inner_bound(proj.res)

    # Local imagery: the save_loc paths in the DB are from the capture machine.
    index = {p.name: p for p in d.glob("*.png")}
    # select_frames() already applies the radalt gate and the image-availability
    # check, so the loop below re-tests them only to keep its n_skip tally
    # meaningful; the SET of frames is decided in exactly one place.
    rows = select_frames(proj, d, args.frame_start, args.stride_frames, args.max_frames)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_det = n_frames = n_skip = 0
    t0 = time.time()
    # Write to a temp file and rename only on success, so the output file EXISTS
    # ONLY IF IT IS COMPLETE. A caller that treats `det.exists()` as "already
    # computed, skip the detector" would otherwise accept a truncated file from
    # a killed run as cache. rename(2) is atomic within a filesystem, and the
    # temp file sits in the destination directory to guarantee that.
    # --thresh-sweep: extract at SEVERAL thresholds from ONE forward pass.
    # Inference is 54 EfficientNetV2-B3 tiles per frame; detections_from_heatmap
    # is a Gaussian blur plus findContours, negligible beside it. So the whole
    # threshold axis costs the same as one threshold, and 8 thresholds x 8
    # checkpoints x 3 flights drops from ~9.8 h of re-inference to ~1.4 h.
    # This is what makes an EXACT threshold sweep affordable, rather than
    # approximating it by filtering cached peak confidences -- which cannot
    # model contour splitting or centroid drift.
    sweep = ([float(x) for x in args.thresh_sweep.split(",")]
             if args.thresh_sweep else [args.thresh])
    outs = {t: (args.out if len(sweep) == 1
                else args.out.with_name(f"{args.out.stem}_t{t:g}{args.out.suffix}"))
            for t in sweep}
    tmps = {t: p.with_suffix(p.suffix + ".partial") for t, p in outs.items()}
    n_det_by_t = {t: 0 for t in sweep}
    handles = {t: open(p, "w") for t, p in tmps.items()}
    try:
        for n, (fi, row) in enumerate(rows, 1):
                radalt = abs(float(row[8]))
                if radalt <= RADALT_MIN:
                    n_skip += 1
                    continue
                img_path = index.get(Path(str(row[9])).name)
                if img_path is None:
                    n_skip += 1
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    n_skip += 1
                    continue
                rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR) \
                    if args.rectify else img
                hm = predict_heatmap(rect, model, args.batch_size)
                for t in sweep:
                    for cx, cy, conf in detections_from_heatmap(hm, t):
                        if not inside((cx, cy), poly):
                            continue
                        handles[t].write(f"{fi} {img_path} {cx},{cy},{conf:.6f}\n")
                        n_det_by_t[t] += 1
                n_det = n_det_by_t[sweep[0]]
                n_frames += 1
                if n % 25 == 0:
                    el = time.time() - t0
                    print(f"    {n}/{len(rows)} frames  {n_det:,} dets @t{sweep[0]:g}  "
                          f"{el/60:.1f} min  eta {(len(rows)-n)*el/n/60:.1f} min", flush=True)
    finally:
        for h in handles.values():
            h.close()

    # Only now, with every frame written and every handle closed, do the outputs
    # become visible under their real names.
    for t in sweep:
        tmps[t].replace(outs[t])

    print(f"\n  {args.dir}  weights={args.weights.name}")
    print(f"  frames processed {n_frames}  skipped {n_skip}")
    for t in sweep:
        print(f"    t={t:<6g} {n_det_by_t[t]:>8,} dets  -> {outs[t]}")
    print(f"  done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
