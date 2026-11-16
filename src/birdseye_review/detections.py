# SPDX-License-Identifier: MIT
# Copyright (c) 2026 harelab-ucsc
"""Detection/label file parsing, image lookup, and the many-to-one spatial
matcher used by the review GUI.

File format (labels.txt and every dets/val_*.txt), one point per line:

    <frame_idx> <image_path> <x>,<y>,<confidence>
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single detection or ground-truth point."""
    frame_idx: int
    image_path: str
    x: float
    y: float
    confidence: float = 1.0


@dataclass
class FrameData:
    """All detections/labels for a single frame."""
    frame_idx: int
    image_path: str
    points: List[Detection]


def parse_annotations(filepath: Path) -> Dict[int, FrameData]:
    """Parse a 'frame_idx image_path x,y,confidence' file into frames.

    Some flight logs carry one row per frame with a placeholder third field
    ('0.0', 'Test', ...) instead of point coordinates. Lines that don't parse
    as 'x,y,confidence' are skipped rather than raising, and a file with no
    parseable lines simply contributes no points."""
    frames: Dict[int, FrameData] = {}
    total_lines = skipped = 0
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            parts = line.split()
            if len(parts) < 3:
                skipped += 1
                continue
            try:
                frame_idx = int(parts[0])
                image_path = parts[1]
                x_str, y_str, conf_str = parts[2].split(",")
                det = Detection(frame_idx, image_path, float(x_str), float(y_str), float(conf_str))
            except ValueError:
                skipped += 1
                continue
            frames.setdefault(frame_idx, FrameData(frame_idx, image_path, [])).points.append(det)
    if total_lines and skipped == total_lines:
        print(f"  Note: {filepath} has no parseable 'x,y,confidence' rows -- skipped entirely.")
    elif skipped:
        print(f"  Note: {filepath}: skipped {skipped}/{total_lines} unparseable lines.")
    return frames


# ---------------------------------------------------------------------------
# Image access
# ---------------------------------------------------------------------------

def build_file_index(base: Path) -> Dict[str, Path]:
    """One-time filename -> path index, so image lookup never re-walks the tree."""
    index: Dict[str, Path] = {}
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(p.name, p)
    return index


def load_image(image_path: str, file_index: Optional[Dict[str, Path]] = None) -> Optional[np.ndarray]:
    """Load an image, falling back to a precomputed basename index when the
    recorded path (often from the machine that captured the flight) doesn't
    exist locally."""
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            return img
    if file_index is not None:
        candidate = file_index.get(Path(image_path).name)
        if candidate is not None:
            return cv2.imread(str(candidate))
    return None


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_spatial_many_to_one(detections: List[Detection], ground_truth: List[Detection],
                              spatial_threshold: float
                              ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Per-point existence-based matching, used by the review GUI's live
    per-frame recompute. A detection counts as matched if ANY ground-truth
    point lies within spatial_threshold, and a ground-truth point counts as
    matched (found) if ANY detection lies within spatial_threshold --
    computed independently per side, so several nearby detections of one
    real target (e.g. overlapping/duplicate detections of the same burrow)
    can all legitimately count as matched, rather than only the single
    nearest one winning an exclusive assignment.

    Returns (matches, unmatched_det_idx, unmatched_gt_idx). 'matches'
    carries one (det_idx, gt_idx, score) tuple per matched detection -- its
    NEAREST ground-truth point -- but a ground-truth point's own matched/
    unmatched status is deliberately NOT derived from 'matches': a
    detection's nearest label isn't necessarily the only label it's within
    threshold of, so gt-side matching is its own independent existence
    check, not inferred from the detection side's nearest-pick."""
    matches, unmatched_det, unmatched_gt, _ = match_spatial_many_to_one_layered(
        detections, ground_truth, spatial_threshold)
    return matches, unmatched_det, unmatched_gt


def match_spatial_many_to_one_layered(detections: List[Detection], ground_truth: List[Detection],
                                      spatial_threshold: float,
                                      det_layer: Optional[Sequence[int]] = None,
                                      n_layers: int = 0
                                      ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int],
                                                 List[int]]:
    """match_spatial_many_to_one, plus a per-ground-truth bitmask of which
    LAYERS had at least one detection within spatial_threshold of it.

    WHY THIS IS ONE FUNCTION AND NOT ONE CALL PER LAYER. The review GUI's
    layered mode overlays N detection files (one per CNN checkpoint) and
    matches their UNION against labels.txt once, then splits the result per
    layer. That is exact, not an approximation, and the reason is visible in
    the loop below:

      - The DETECTION side is computed independently per detection -- the
        `for i, det` body reads no other detection -- so a detection's
        matched/unmatched status is identical whether it is matched alone or
        inside a union with other checkpoints' output.
      - The GROUND-TRUTH side is *not* independent (`matched_gt` accumulates
        across detections), which is exactly why the per-layer mask has to be
        collected here rather than reconstructed afterwards. It is OR'd
        inside the existing `if d <= spatial_threshold` branch, so it costs
        no extra distance computations at all.

    `det_layer[i]` is the layer index of detections[i]; bit L of
    gt_layer_mask[j] is set iff some detection from layer L is within
    threshold of ground_truth[j]. With det_layer=None the mask is all zeros
    and this is exactly the single-source matcher.

    The unlayered name above is a thin wrapper over this, deliberately:
    duplicating the loop body would let the two drift apart."""
    n_det, n_gt = len(detections), len(ground_truth)
    if n_det == 0 or n_gt == 0:
        return [], list(range(n_det)), list(range(n_gt)), [0] * n_gt

    matches: List[Tuple[int, int, float]] = []
    matched_gt: set = set()
    gt_layer_mask = [0] * n_gt
    for i, det in enumerate(detections):
        bit = (1 << det_layer[i]) if det_layer is not None and det_layer[i] >= 0 else 0
        best_j, best_d = None, spatial_threshold
        for j, gt in enumerate(ground_truth):
            d = math.hypot(det.x - gt.x, det.y - gt.y)
            if d <= spatial_threshold:
                matched_gt.add(j)
                gt_layer_mask[j] |= bit
                if d <= best_d:
                    best_j, best_d = j, d
        if best_j is not None:
            # At spatial_threshold == 0.0 (reachable by dragging the Spatial
            # Thresh slider to its minimum) a match can only occur at
            # best_d == 0.0, an exact detection/label coincidence; guard the
            # 0/0 division and score it as perfect.
            score = 1.0 - best_d / spatial_threshold if spatial_threshold > 0 else 1.0
            matches.append((i, best_j, score))

    matched_det = {i for i, _, _ in matches}
    unmatched_det = [i for i in range(n_det) if i not in matched_det]
    unmatched_gt = [j for j in range(n_gt) if j not in matched_gt]
    return matches, unmatched_det, unmatched_gt, gt_layer_mask
