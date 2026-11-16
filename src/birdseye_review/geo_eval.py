# SPDX-License-Identifier: MIT
# Copyright (c) 2026 harelab-ucsc
"""
Pose-database projection and geotag clustering primitives.

Projection and boresight code is ported from harelab-ucsc/birdseye (MIT).

`Projector` reads a flight's `flight_data.db` (camera poses, RTK geotags,
camera parameters) and back-projects image points to UTM ground coordinates.
`s2d` clusters back-projected detections with DBSCAN and matches geotags to
cluster core points, the comparison the onboard pipeline prints:

    geotags = SELECT x,y FROM clicks_flight_data
    clusters = DBSCAN(eps=0.20, min_samples=10) over detections[:, :2], UTM metres
    hits = geotags within 0.35 m of a detection-cluster CORE point
    clusters hit = distinct clusters receiving >= 1 geotag

PROVENANCE OF THE MATH
----------------------
Ported from harelab-ucsc/birdseye: birdseye/birdseye.py so the numbers are
comparable with the onboard pipeline:
  _2Dto3D            ray/ground-plane intersection
  poseRowToTransform (utilities.py; quaternion stored q,u,a,t = scipy xyzw)
  correct_altitude   radalt *= cos(roll)cos(pitch)
  ned_to_enu_se3     similarity transform, not a bare rotation
  boresight          FLIGHT-DATE SPECIFIC, see BORESIGHT_CAMPAIGNS below
  dbscan_predict     geotag -> cluster via core points
  diff               the 's2d' branch
"""
import json
import math
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .detections import parse_annotations

# Boresight correction, per campaign (harelab-ucsc/birdseye: birdseye/birdseye.py).
#
# The onboard code carries three hand-tuned per-campaign branches with only one
# active at a time, so a single constant applied to every flight is wrong for
# flights from another campaign. Measured by back-projecting every labels.txt
# row -- which IS the projected RTK survey, so a correct camera model must land
# it exactly on a geotag:
#
#   flight                        date        median err   correct branch
#   2025_04_16/pieranch_rect      2025-04-16     0.000 m   rr=4
#   another flight (same campaign)               0.000 m   rr=4
#   another flight (later campaign)            * 0.152 m * rr=11 (with rr=4)
#
# With rr=11 that later flight also goes to 0.000 m. The error is not a pure
# translation (|mean vector| 0.101 m against mean |offset| 0.154 m), because a
# roll error scales with attitude and altitude -- so it cannot be fixed by
# subtracting an offset, and it is invisible to every metric computed here.
#
# Hence: select by flight date, and REFUSE rather than guess when no campaign
# covers a flight. Note the deliberate 2025/05/03 .. 2025/05/20 gap -- the
# onboard code has no branch for it either, and a flight in that window must
# fail loudly instead of silently borrowing a neighbour's calibration.
BORESIGHT_CAMPAIGNS = (
    {"from": "2025-03-18", "to": "2025-04-02",
     "rr": -11, "rp": 2, "ry": 0, "mod": 0.125, "tx": 0.0, "ty": 0.0, "tz": 0.0,
     "label": "2025/03/18"},
    {"from": "2025-04-03", "to": "2025-05-02",
     "rr": 4, "rp": -5, "ry": 0, "mod": 0.125, "tx": 0.0, "ty": 0.0, "tz": 0.0,
     "label": "2025/04/03 .. 2025/05/02"},
    {"from": "2025-05-21", "to": "9999-12-31",
     "rr": 11, "rp": -5, "ry": 0, "mod": 0.125, "tx": 0.0, "ty": 0.0, "tz": 0.0,
     "label": "2025/05/21 and after"},
)

# Max distance a reprojected labels.txt row may sit from its nearest geotag.
# The three correctly-calibrated flights all come out at EXACTLY 0.000 m, and
# the one miscalibrated flight at 0.152 m, so anything in between is a wide
# no-man's-land and this cut is not delicate. It is deliberately far below the
# smallest matching tolerance in use (DBSCAN eps 0.10 m) so that a calibration
# error can never be mistaken for a clustering result.
SELFCHECK_TOL_M = 0.02
# Set False (or BIRDSEYE_NO_BORESIGHT_SELFCHECK=1) only if a caller genuinely
# has no labels.txt and the warning is noise. Leaving it on is what makes the
# class of bug above impossible to reintroduce silently.
SELFCHECK = os.environ.get("BIRDSEYE_NO_BORESIGHT_SELFCHECK", "") == ""
_SELFCHECK_DONE: dict = {}


def boresight_for_date(day: str) -> dict:
    """The campaign covering `day` (YYYY-MM-DD), or SystemExit naming the gap."""
    for c in BORESIGHT_CAMPAIGNS:
        if c["from"] <= day <= c["to"]:
            return c
    ranges = ", ".join(f'{c["from"]}..{c["to"]}' for c in BORESIGHT_CAMPAIGNS)
    raise SystemExit(
        f"NO BORESIGHT CALIBRATION COVERS {day}. Known campaigns: {ranges}. "
        f"The onboard birdseye code has no branch for this date either. Applying a "
        f"neighbouring campaign's boresight silently displaces every "
        f"back-projected point -- measured at 0.152 m for a 3-week gap, which "
        f"is larger than the smallest DBSCAN eps in use. Derive the boresight "
        f"for this campaign and add it to BORESIGHT_CAMPAIGNS.")


def flight_date_from_db(db: Path) -> str:
    """Flight date as YYYY-MM-DD, from the first image's capture timestamp.

    Taken from the unix timestamp embedded in save_loc rather than from the
    directory name, so a renamed or copied flight directory cannot silently
    select the wrong calibration. Falls back to the directory name only if no
    timestamp can be parsed.
    """
    try:
        con = sqlite3.connect(str(db))
        row = con.execute(
            "select save_loc from cam0_images_flight_data limit 1").fetchone()
        con.close()
        m = re.search(r"_(\d{10})(?:\.\d+)?\.png$", str(row[0]))
        if m:
            return datetime.fromtimestamp(int(m.group(1))).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001 -- fall through to the path-derived date
        pass
    for part in reversed(Path(db).resolve().parts):
        m = re.fullmatch(r"(\d{4})_(\d{2})_(\d{2})", part)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    raise SystemExit(f"cannot determine the flight date for {db}, so the "
                     f"boresight campaign cannot be selected")

# DBSCAN and matching defaults of the onboard pipeline
DBSCAN_EPS = 0.20          # metres, cluster detections
DBSCAN_MIN_SAMPLES = 10    # a burrow seen in fewer frames is dropped as noise
EPS_PREDICT_DETS = 0.35    # metres, geotag -> detection cluster
EPS_PREDICT_CLICKS = 0.05  # metres, geotag -> click cluster ("viewed")

# Axis swap ENU-body -> OpenCV camera (onboard birdseye convention)
AXIS_SWAP = np.array([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def ned_to_enu_se3(pose_ned: np.ndarray) -> np.ndarray:
    """Similarity transform T P T^T, not a plain rotation."""
    Rm = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
    T = np.eye(4)
    T[:3, :3] = Rm
    return T @ pose_ned @ T.T


def pose_row_to_transform(pose) -> np.ndarray:
    """Columns are stored q,u,a,t and handed to scipy as-is,
    i.e. scalar-LAST (x,y,z,w)."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(pose[3:7], dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(pose[0:3], dtype=float)
    return T


def corrected_radalt(pose_row) -> float:
    """Tilt-corrects the radar slant range to a vertical."""
    e = R.from_quat(np.asarray(pose_row[3:7], dtype=float)).as_euler("xyz")
    v = float(pose_row[8]) * math.cos(e[0]) * math.cos(e[1])
    return abs(v)


class Projector:
    """Camera model for one flight, from parameters_flight_data."""

    def __init__(self, db: Path, selfcheck: Optional[bool] = None):
        db = Path(db)
        self.db = db
        self.flight_date = flight_date_from_db(db)
        self.boresight = boresight_for_date(self.flight_date)
        con = sqlite3.connect(str(db))
        rows = {r[0]: r for r in con.execute(
            "select sensorID, resolution, intrinsics1, intrinsics2, extrinsics "
            "from parameters_flight_data")}
        cam = rows["cam0"]
        res = json.loads(cam[1])
        k = json.loads(cam[2])
        self.res = (int(res[0]), int(res[1]))
        self.K = np.array([[k[0], 0.0, k[2]], [0.0, k[1], k[3]], [0.0, 0.0, 1.0]])
        self.D = np.array(json.loads(cam[3]), dtype=float)
        T_IC = np.array(json.loads(cam[4]), dtype=float).reshape(4, 4)
        self.T_IC = ned_to_enu_se3(T_IC)

        b = self.boresight
        r_adj = R.from_euler("xyz", [b["rr"] * b["mod"], b["rp"] * b["mod"],
                                     b["ry"] * b["mod"]], degrees=True).as_matrix()
        t_adj = np.array([b["tx"], b["ty"], b["tz"]])
        self.T_IC[:3, 3] = r_adj @ self.T_IC[:3, 3]
        self.T_IC[:3, 3] = t_adj + self.T_IC[:3, 3]
        self.T_IC[:3, :3] = r_adj @ self.T_IC[:3, :3]

        self.poses = con.execute(
            "select x, y, z, q, u, a, t, rtk_status, radalt, save_loc "
            "from cam0_images_flight_data").fetchall()
        self.clicks = np.array(con.execute(
            "select x, y from clicks_flight_data").fetchall(), dtype=float)
        con.close()

        # Verify the calibration against the flight's own labels.txt. This is
        # the check that would have caught the frozen-boresight bug the moment
        # it was introduced, so it runs by default rather than on request.
        if selfcheck is None:
            selfcheck = SELFCHECK
        if selfcheck:
            self.verify_reprojection()

    def verify_reprojection(self, tol_m: float = None) -> Optional[float]:
        """Max distance from a reprojected labels.txt row to its nearest geotag.

        labels.txt is not an independent annotation -- it is the same RTK geotag
        survey projected into each frame by birdseye.annotate(). So a correct
        camera model must land every label row essentially exactly back on a
        geotag, and any material error is a camera-model error. That makes this
        a free, exact, per-flight calibration test with no tuning in it.

        Returns the max error, or None when the flight has no labels.txt (some
        do not, and their absence is not an error). Raises when the error
        exceeds the tolerance.
        """
        tol = SELFCHECK_TOL_M if tol_m is None else tol_m
        key = (str(self.db.resolve()), self.boresight["label"], tol)
        if key in _SELFCHECK_DONE:
            return _SELFCHECK_DONE[key]
        labels = self.db.parent / "labels.txt"
        if not labels.exists() or not len(self.clicks):
            _SELFCHECK_DONE[key] = None
            return None
        frames = parse_annotations(labels)
        by_i = {i: r for i, r in enumerate(self.poses)}
        worst = 0.0
        for fi, fd in frames.items():
            row = by_i.get(fi)
            if row is None or not fd.points:
                continue
            xy = np.array([[pt.x, pt.y] for pt in fd.points], dtype=float)
            T_WC, radalt = self.frame_pose(row)
            if radalt <= 0:
                continue
            p3 = self.to_3d(xy, T_WC, radalt)
            if len(p3) != len(xy):
                continue
            d = np.linalg.norm(p3[:, None, :2] - self.clicks[None, :, :2],
                               axis=2).min(axis=1)
            worst = max(worst, float(d.max()))
        _SELFCHECK_DONE[key] = worst
        if worst > tol:
            raise SystemExit(
                f"BORESIGHT SELF-CHECK FAILED for {self.db.parent} "
                f"(flight date {self.flight_date}, campaign "
                f"{self.boresight['label']}): reprojecting labels.txt lands "
                f"{worst:.4f} m from the nearest geotag, tolerance {tol} m.\n"
                f"labels.txt IS the projected RTK survey, so this is a camera "
                f"model error, not a detection error. A correctly calibrated "
                f"flight comes out at 0.000 m. Every geo metric computed for "
                f"this flight would be displaced by roughly this distance -- "
                f"comparable to the DBSCAN eps values in use -- so no number "
                f"from it is trustworthy until the boresight is corrected. "
                f"Check BORESIGHT_CAMPAIGNS against harelab-ucsc/birdseye: birdseye/birdseye.py.")
        return worst

    def frame_pose(self, row) -> Tuple[np.ndarray, float]:
        T_WC = pose_row_to_transform(row) @ self.T_IC
        return T_WC, corrected_radalt(row)

    def to_3d(self, pts2d: np.ndarray, T_WC: np.ndarray, radalt: float) -> np.ndarray:
        """Vectorised `_2Dto3D`. Ray from the camera centre through each
        pixel, terminated on the plane z = camera_z - radalt."""
        if len(pts2d) == 0:
            return np.zeros((0, 3))
        ones = np.ones((len(pts2d), 1))
        p = np.hstack([pts2d, ones])                        # N x 3
        pc = (np.linalg.inv(self.K) @ p.T)                  # 3 x N
        pc = np.vstack([pc, np.ones((1, pc.shape[1]))])     # 4 x N
        pc = AXIS_SWAP @ pc
        pw = T_WC @ pc                                      # 4 x N
        vec = pw - T_WC[:, 3][:, None]
        unit = vec / np.linalg.norm(vec, axis=0, keepdims=True)
        denom = unit[2, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = radalt / denom
        p3d = T_WC[:, 3][:, None] - scale * unit
        out = p3d[:3, :].T
        return out[np.isfinite(out).all(axis=1)]

    def to_2d(self, pts3d: np.ndarray, T_WC: np.ndarray) -> np.ndarray:
        """World (UTM) -> pixel."""
        if len(pts3d) == 0:
            return np.zeros((0, 2))
        P = np.hstack([pts3d[:, :3], np.ones((len(pts3d), 1))])
        cam = np.linalg.inv(T_WC) @ P.T
        cam = AXIS_SWAP @ cam
        proj = self.K @ cam[:3, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            proj = proj / proj[2]
        return proj[:2].T


def dbscan_predict(clustering, fit_data: np.ndarray, new_data: np.ndarray,
                   eps: float) -> np.ndarray:
    """A point matches if any CORE sample lies within eps."""
    mask = np.zeros_like(clustering.labels_, dtype=bool)
    mask[clustering.core_sample_indices_] = True
    core_points, core_labels = fit_data[mask], clustering.labels_[mask]
    if len(core_points) == 0:
        return np.full(len(new_data), -1)
    nn = NearestNeighbors(radius=eps).fit(core_points)
    out = []
    for idx in nn.radius_neighbors(new_data, return_distance=False):
        out.append(-1 if len(idx) == 0 else
                   Counter(core_labels[idx]).most_common(1)[0][0])
    return np.array(out)


def s2d(ref: np.ndarray, test: np.ndarray, eps: float, eps_predict: float,
        min_samples: int) -> Dict:
    """The onboard 's2d' comparison, with the results returned instead
    of printed."""
    ref2, test2 = np.asarray(ref)[:, :2], np.asarray(test)[:, :2]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(test2)
    ref_labels = dbscan_predict(clustering, test2, ref2, eps_predict)
    clusts = set(clustering.labels_) - {-1}
    hit_clusters = set(ref_labels.tolist()) - {-1}
    per_cluster = Counter(l for l in ref_labels.tolist() if l != -1)
    return {
        "n_ref": int(len(ref2)),
        "n_test_points": int(len(test2)),
        "n_clusters": int(len(clusts)),
        "n_noise": int((clustering.labels_ == -1).sum()),
        "hits": int((ref_labels != -1).sum()),
        "n_clusters_hit": int(len(hit_clusters)),
        "clusters_with_multiple_geotags": int(sum(1 for v in per_cluster.values() if v > 1)),
        "labels": clustering.labels_,
        "ref_labels": ref_labels,
    }

