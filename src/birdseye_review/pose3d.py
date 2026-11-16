# SPDX-License-Identifier: MIT
# Copyright (c) 2026 harelab-ucsc
"""
Interactive 3D flight-pose view for review_gui.py
=================================================

`review_gui.py` reviews detections in pure 2D pixel space: it never knows
where the aircraft was, which way it was looking, or how the frame on screen
relates to the geotag survey on the ground. Every flight directory that has a
`flight_data.db` already carries that information, and `geo_eval.py` already
carries the verified projection math (ported from harelab-ucsc/birdseye) to
interpret it. This module
joins the two and draws the result.

WHAT IT DRAWS (all in UTM metres, recentred on the track -- see LOCAL FRAME)

    ground grid     10 m, at the current frame's ground-plane height
    camera track    the whole flight, dim ahead of the current frame and
                    bright behind it, so direction of travel is readable
    current pose    a camera frustum: four rays from the projection centre
                    out to the frame's ground footprint corners
    footprint       that frame's image corners ray-cast to the ground plane,
                    i.e. exactly what the camera can see right now
    detections      this frame's cluster centroids back-projected to the
                    ground, in review_gui's own judgment colours
    cloud           EVERY other frame's detections too, each back-projected
                    through its own pose and drawn translucent in the same
                    colours -- so the review state of the whole flight is
                    visible at once. Points within GEOTAG_HIT_M of a survey
                    geotag are drawn heavier. See build_cloud(), and the
                    smearing caveat in its docstring.
    geotags         the `clicks_flight_data` survey, as static ground markers

WHY IT IS HAND-ROLLED INSTEAD OF matplotlib
-------------------------------------------
The 3D view is drawn with numpy+cv2 so it needs no matplotlib 3D backend.
Beyond that, review_gui.py's mainloop is `cv2.imshow` + `cv2.waitKey(20)`; dropping a
Qt/Tk event loop into it is the single most reliable way to break the one
interactive tool this project has. A numpy+cv2 renderer in a second
`cv2.namedWindow` is serviced by the *same* `waitKey`, needs no new
dependency, and cannot deadlock the main window.

WHY LAYERS, NOT A DEPTH SORT
----------------------------
A general painter's algorithm would sort every primitive by view depth each
frame. At 2,120 track points that is thousands of Python-level sorts at ~50
render ticks/sec. This scene does not need it: it is a near-planar ground
sheet with a track and a frustum above it, so a fixed layer order (grid,
geotags, footprint, track, frustum, detections, marker, text) is *always*
correct for it. Only the small point sets are depth-ordered internally, where
the cost is trivial. If the scene ever gains genuinely interpenetrating
geometry, that assumption is the thing to revisit.

LOCAL FRAME
-----------
UTM coordinates here are ~5.6e5 easting / ~4.1e6 northing. Differencing those
inside a projection costs most of float64's precision on a scene only ~100 m
across, so everything is recentred on the track centroid at load time and the
renderer never sees a raw UTM number.

WHERE IT DRAWS
--------------
Two viewports, chosen by review_gui's [v] cycle: the dead band under the frame
image (see App.inline_pose_rect -- widening the side panel left a permanently
black strip there), or a separate resizable cv2 window. The same PoseScene
serves both; only the (w, h) passed to render() differs, which is exactly why
the framing has to be fitted per viewport rather than fixed once. Inline is
composited inside App.render(), so it is structurally frame-accurate with the
image above it.

Nothing in review_gui.py imports this module at startup -- it is imported
lazily, the first time [v] is pressed, so a failure here can only ever disable
the 3D view, never stop the GUI from launching. See App.cycle_pose_view().
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

WINDOW = "detection review -- 3D flight pose"

# Colours are BGR, matching review_gui's convention.
# Every colour here is checked against COLOR["bg"] by smoke_test_gui.py using
# the same mean-|delta-luma| >= 40 bar the [t] pool badges are held to. `grid`
# (29) and `track_future` (39) originally failed it, which is what made the far
# grid and the not-yet-flown track read as empty black.
COLOR = {
    "bg": (26, 26, 26),
    "grid": (74, 74, 74),
    "grid_major": (105, 105, 105),
    "track_past": (200, 200, 90),
    # Bright enough to see, still an unmistakable step down from track_past
    # (delta-luma 81 vs 137) -- that step IS the current-frame indicator, and it
    # is what visibly sweeps along the track during playback.
    "track_future": (115, 115, 90),
    "track_comet": (140, 255, 255),   # the last COMET_FRAMES before now
    "frustum": (0, 220, 255),
    "footprint": (0, 170, 200),
    "geotag": (200, 120, 255),
    # Geo-cluster centroid states. The four judgment colours are review_gui's
    # own (COLOR["unreviewed"]/["confirmed_real"]/["hard_negative"]) so the 3D
    # view and the 2D frame speak one language; they are restated here rather
    # than imported because pose3d must not depend on review_gui.
    "centroid_none": (170, 170, 170),   # [o] off: geometry only, no status claimed
    "conflict": (200, 0, 200),          # both verdicts present -- the rule decided it
    "camera": (255, 255, 255),
    "text": (220, 220, 220),
    "dim": (150, 150, 150),
    "warn": (80, 140, 255),
    "axis_e": (80, 80, 255),
    "axis_n": (80, 220, 80),
    "axis_u": (255, 180, 80),
}

GRID_SPACING_M = 10.0
GRID_MAJOR_EVERY = 5          # every 5th grid line is drawn brighter
GRID_MAX_LINES = 81           # cap, so a long survey can't emit thousands of lines
NEAR_PLANE_M = 0.5
# HORIZONTAL field of view, i.e. focal length is derived from the viewport's
# WIDTH. Pinning the vertical field instead (as this did originally) gives a
# wide, short viewport -- exactly the inline strip under the main image -- a
# ~132 degree horizontal field and severe wide-angle distortion at the edges.
# Fixing the lens by width does not decide the standoff: see OrbitCamera.fit(),
# which still has to respect whichever axis binds or content falls off the top
# and bottom of a short strip.
HFOV_DEG = 55.0
FIT_MARGIN = 1.12             # a little air around the fitted bounding sphere
ELEVATION_LIMIT_DEG = 89.0
# Opening elevation as a function of viewport aspect. A roughly square survey
# letterboxes badly into a 4:1 strip; dropping the camera low foreshortens the
# depth axis so the same plan view projects into a wide, short trapezoid that
# suits the strip.
ELEVATION_FOR_ASPECT = (50.0, 12.0, 45.0)   # (numerator, min, max): clamp(n/aspect, min, max)
COMET_FRAMES = 30             # recent track drawn brighter/thicker, so motion reads at a glance
# Directory-wide detection cloud (see PoseScene.build_cloud). Every point is
# back-projected through its OWN frame's pose, so the whole flight's review
# state shows at once instead of just the frame on screen.
CLOUD_ALPHA = 0.62            # translucency of the off-frame layer
# Marker sizes in pixels. A single pixel at the old 0.45 alpha was technically
# present but hard to pick out; a 2x2 block reads as a mark without becoming a
# blob at survey zoom, and geotag hits get 3x3 so they stand out by SIZE
# rather than by colour -- the colours have to stay the ones the 2D view uses.
# Grown with one cv2.dilate rather than one scatter per offset, so the cost is
# a function of canvas area (fixed) instead of point count (87k, and 4-8
# writes each cost ~10 ms/tick).
CLOUD_MARKER_PX = 2
CLOUD_HIT_PX = 3
# The per-detector cloud steps BACK when the geo-cluster centroids are up:
# 105,682 union points against ~560 centroids, so at equal weight the cloud
# is the only thing the eye can find. Dimming is a blend weight, not a
# colour change -- the per-layer palette still reads, it just recedes.
CLOUD_ALPHA_DIMMED = 0.24
CLOUD_MARKER_PX_DIMMED = 1
CLOUD_MAX_POINTS = 400_000    # deterministic-stride subsample above this; the HUD says so
# Metres. geo_eval.EPS_PREDICT_DETS -- the same radius the birdseye onboard
# geotag-to-detection matching uses, so "hit a geotag" here means the same
# thing it means in the geospatial metrics.
GEOTAG_HIT_M = 0.35

# Geo-cluster centroids: one marker per DBSCAN cluster in UTM, i.e. one per
# physical spot the detector claims, rather than one per per-frame
# re-projection. A per-frame centroid cloud over all checkpoints would
# superimpose them into one undifferentiated smear; geo clusters are one point
# per spot, so they carry the review-state recolouring without that failure.
CENTROID_R = 4                # filled dot, in screen px
CENTROID_RING_R = 7           # conflict ring, drawn outside the dot
CENTROID_REACH_RING_R = 11    # drawn on the widest-reaching clusters
# A cluster whose verdict would propagate to at least this many review
# clusters gets the outer ring. Chaining is the reason the reach spread is
# 16 (median) to 18,146 (max) on the full 2,120-frame sample flight, and a 3D view that draws a
# 18,146-cluster blob the same size as a 1-cluster spot hides exactly the
# thing this view is being opened to see.
CENTROID_REACH_RING_MIN = 200
ZOOM_STEP = 1.15
ZOOM_DRAG_PX_PER_STEP = 40.0  # vertical drag distance for one ZOOM_STEP
ORBIT_DEG_PER_PX = 0.4
PAN_FRACTION_PER_PX = 0.0015  # of the current distance, so panning scales with zoom


def _fit_text(text: str, max_px: int, font_scale: float,
              font: int = cv2.FONT_HERSHEY_SIMPLEX, thickness: int = 1) -> str:
    """Truncate `text` to what actually fits in `max_px`, measured rather than
    guessed from a character count.

    A deliberate twin of review_gui.fit_text -- duplicated, not imported,
    because pose3d must stay free of a back-dependency on review_gui
    (review_gui imports THIS module lazily, and smoke_test_gui.py imports it
    standalone). Keep the two in step if either changes.
    """
    if cv2.getTextSize(text, font, font_scale, thickness)[0][0] <= max_px:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cv2.getTextSize(text[:mid], font, font_scale, thickness)[0][0] <= max_px:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


class PoseUnavailable(Exception):
    """No usable pose data for this flight directory. Raised (not returned as
    a sentinel) so every failure mode -- missing db, unreadable db, unexpected
    schema -- arrives at the caller as one thing to catch, carrying a message
    fit to show the user verbatim in the GUI's status line."""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_projector(flight_dir: Path):
    """geo_eval.Projector for `flight_dir`, or PoseUnavailable.

    geo_eval is imported here rather than at module scope so that a project
    without scipy/sklearn installed still gets a clean, explanatory failure
    on [v] instead of an ImportError at review_gui startup.
    """
    db = Path(flight_dir) / "flight_data.db"
    if not db.exists():
        raise PoseUnavailable(f"no flight_data.db in {Path(flight_dir).name} -- no pose data for this flight")
    try:
        from .geo_eval import Projector
    except Exception as e:  # noqa: BLE001 -- surfaced to the user, not swallowed
        raise PoseUnavailable(f"could not import geo_eval ({type(e).__name__}: {e})") from e
    try:
        proj = Projector(db)
    except Exception as e:  # noqa: BLE001
        raise PoseUnavailable(f"could not read {db.name} ({type(e).__name__}: {e})") from e
    if not proj.poses:
        raise PoseUnavailable(f"{db.name} has no rows in cam0_images_flight_data")
    return proj


def build_frame_index(proj, max_frame_idx: int) -> Tuple[Dict[int, int], Optional[str]]:
    """frame_idx -> row index into proj.poses, plus a warning to print once.

    Every other consumer of a run (geo_eval.Projector, run_detector) assumes
    `frame_idx` IS the positional row index. That
    is verified here rather than trusted: pose rows carry `save_loc`, and
    review_gui's frame indices come from labels.txt, so a mismatch is
    detectable. Verified true on the full 2,120-frame sample flight (2,120
    poses, 2,120 PNGs, basenames in sorted order) -- but a directory whose
    labels.txt was written against a different image set would silently
    mis-attribute every pose, which is the kind of error that looks plausible
    on screen.

    `max_frame_idx` is the largest frame index that will be asked for, NOT the
    number of frames. Those differ whenever detections cover a subset of the
    flight, and conflating them would return None for every frame above the
    count, silently dropping part of the detection cloud.
    """
    n_poses = len(proj.poses)
    positional = {i: i for i in range(n_poses)}
    # Only a frame index the pose table CANNOT cover is a problem. Having more
    # poses than frames is the normal case, not an anomaly, so warning on any
    # inequality would fire on a healthy flight and train the reader to ignore
    # the line.
    if max_frame_idx < n_poses:
        return positional, None
    return positional, (f"! frame indices run to {max_frame_idx} but there are only "
                        f"{n_poses} pose rows -- those frames have no pose")


def basename_index(proj) -> Dict[str, int]:
    """save_loc basename -> pose row index, for cross-checking / fallback."""
    out = {}
    for i, row in enumerate(proj.poses):
        loc = row[9]
        if loc:
            out[Path(str(loc)).name] = i
    return out


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class OrbitCamera:
    """A turntable camera over a local ENU frame: azimuth/elevation on a
    sphere of radius `distance` about `target`.

    Deliberately NOT geo_eval.Projector.to_2d -- that is the *drone's*
    camera, with the flight's real intrinsics and boresight. This is the
    viewer's camera, and conflating the two would make the view silently
    depend on the aircraft's calibration.
    """

    def __init__(self, target: np.ndarray, distance: float):
        self.target = np.asarray(target, dtype=float).copy()
        self.distance = float(distance)
        self.azimuth = -60.0
        self.elevation = 35.0
        self._home = (self.target.copy(), self.distance, self.azimuth, self.elevation)
        # Viewport the current framing was computed for. A later render at a
        # different size rescales `distance` off this (see on_viewport()),
        # rather than refitting -- refitting would throw away a manual zoom
        # every time the window is dragged.
        self._viewport: Optional[Tuple[int, int]] = None
        # Principal point as (x, y) fractions of the viewport. Not always the
        # centre: on the inline strip the HUD occupies a band across the top,
        # and shifting the vanishing point below it lets the scene use the
        # whole remaining area instead of being fitted into the symmetric
        # leftover, which on a 250px strip is barely 90px tall.
        self.principal = (0.5, 0.5)

    def mark_home(self) -> None:
        """Adopt the current framing as what reset() returns to."""
        self._home = (self.target.copy(), self.distance, self.azimuth, self.elevation)

    def reset(self) -> None:
        t, d, a, e = self._home
        self.target, self.distance, self.azimuth, self.elevation = t.copy(), d, a, e

    @staticmethod
    def half_angles(w: int, h: int) -> Tuple[float, float]:
        """(horizontal, vertical) half-field angles in radians for a viewport.
        Horizontal is the fixed constant; vertical falls out of the aspect,
        because the focal length is derived from width (see HFOV_DEG)."""
        half_h = math.radians(HFOV_DEG) * 0.5
        f = 0.5 * max(1, w) / math.tan(half_h)
        return half_h, math.atan(0.5 * max(1, h) / f)

    def fit(self, w: int, h: int, radius: float, points: Optional[np.ndarray] = None,
            rect: Optional[Tuple[int, int, int, int]] = None) -> None:
        """Frame the content into `rect` (default: the whole viewport).

        The sphere bound alone is far too pessimistic here. The scene is a
        near-flat sheet viewed obliquely, so its on-screen vertical extent is
        heavily foreshortened -- fitting a sphere of the survey's radius to
        the binding axis of a 1100x250 strip put the survey at 20% of the
        available width, a tiny blob in a sea of black. So the sphere is only
        the starting guess (it is guaranteed not to clip), and a few
        refinement passes then pull the camera in until the ACTUAL projected
        bounding box fills `rect`. Projected size goes as 1/distance for a
        distant scene, so the correction is very nearly exact and converges
        in two or three passes.
        """
        x0, y0, x1, y1 = rect if rect else (0, 0, w, h)
        rw, rh = max(1, x1 - x0), max(1, y1 - y0)
        self.principal = ((x0 + x1) * 0.5 / max(1, w), (y0 + y1) * 0.5 / max(1, h))
        half_h, half_v = self.half_angles(w, h)
        theta = max(1e-3, min(half_h, half_v))
        self.distance = max(1.0, min(100_000.0, FIT_MARGIN * max(1e-3, radius) / math.sin(theta)))
        self._viewport = (int(w), int(h))
        if points is None or len(points) == 0:
            return
        cx, cy = self.principal[0] * w, self.principal[1] * h
        for _ in range(4):
            px, depth = self.project(points, w, h)
            ok = (depth > NEAR_PLANE_M) & np.isfinite(px).all(axis=1)
            if not ok.any():
                return
            dx = np.abs(px[ok, 0] - cx).max()
            dy = np.abs(px[ok, 1] - cy).max()
            need = max(dx / (rw * 0.5), dy / (rh * 0.5)) * FIT_MARGIN
            if not np.isfinite(need) or need <= 0:
                return
            self.dolly(need)
            if 0.98 < need < 1.02:
                break

    def default_elevation(self, w: int, h: int) -> float:
        n, lo, hi = ELEVATION_FOR_ASPECT
        aspect = max(0.05, w / max(1, h))
        return max(lo, min(hi, n / aspect))

    def on_viewport(self, w: int, h: int) -> None:
        """Called every render. When the viewport changes shape, rescale the
        standoff by the ratio of the binding half-angles so the same world
        extent stays framed -- a window drag reframes instead of cropping, and
        a manual zoom or orbit survives it untouched."""
        wh = (int(w), int(h))
        if self._viewport == wh:
            return
        if self._viewport is not None:
            old_theta = min(*self.half_angles(*self._viewport))
            new_theta = min(*self.half_angles(*wh))
            if new_theta > 1e-6 and old_theta > 1e-6:
                self.dolly(math.sin(old_theta) / math.sin(new_theta))
        self._viewport = wh

    def orbit(self, d_az_deg: float, d_el_deg: float) -> None:
        self.azimuth = (self.azimuth + d_az_deg) % 360.0
        # Clamped short of the poles: at exactly +-90 the world-up vector is
        # parallel to the view direction and the look-at basis is undefined.
        self.elevation = max(-ELEVATION_LIMIT_DEG, min(ELEVATION_LIMIT_DEG, self.elevation + d_el_deg))

    def dolly(self, factor: float) -> None:
        self.distance = max(1.0, min(100_000.0, self.distance * factor))

    def pan(self, dx_px: float, dy_px: float) -> None:
        """Slide the target in the camera's own screen plane, scaled by
        distance so a drag moves the scene by the same on-screen amount
        however far out you are zoomed."""
        right, up, _ = self._basis()
        k = self.distance * PAN_FRACTION_PER_PX
        self.target += (-dx_px * k) * right + (dy_px * k) * up

    def eye(self) -> np.ndarray:
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        offset = np.array([math.cos(el) * math.cos(az),
                           math.cos(el) * math.sin(az),
                           math.sin(el)], dtype=float)
        return self.target + self.distance * offset

    def _basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(right, up, back) orthonormal camera axes; `back` points from the
        target toward the eye, so a point in front of the camera has a
        negative coordinate along it."""
        back = self.eye() - self.target
        n = np.linalg.norm(back)
        back = back / n if n else np.array([0.0, 0.0, 1.0])
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, back)
        n = np.linalg.norm(right)
        # Only reachable if elevation clamping is ever loosened; kept so the
        # basis is total rather than conditionally valid.
        right = right / n if n else np.array([1.0, 0.0, 0.0])
        up = np.cross(back, right)
        return right, up, back

    def project(self, pts: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """(N,3) local-frame points -> ((N,2) float pixels, (N,) depth in
        metres). Depth <= NEAR_PLANE_M means behind/at the camera; those
        pixels are meaningless and every caller must mask on depth, not on
        the pixel values."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            return np.zeros((0, 2)), np.zeros(0)
        right, up, back = self._basis()
        rel = pts - self.eye()
        x = rel @ right
        y = rel @ up
        depth = -(rel @ back)
        f = 0.5 * w / math.tan(math.radians(HFOV_DEG) * 0.5)
        safe = np.where(depth > NEAR_PLANE_M, depth, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            sx = w * self.principal[0] + f * x / safe
            sy = h * self.principal[1] - f * y / safe
        return np.stack([sx, sy], axis=1), depth


def _clip_to_near(p_in: np.ndarray, p_out: np.ndarray, cam: OrbitCamera) -> np.ndarray:
    """Move `p_out` (behind the near plane) onto it, along the segment from
    `p_in` (in front). Without this a line with one endpoint behind the camera
    projects to a wild off-screen coordinate and draws as a stray streak
    across the view."""
    _, _, back = cam._basis()
    eye = cam.eye()
    d_in = -float((p_in - eye) @ back)
    d_out = -float((p_out - eye) @ back)
    denom = d_in - d_out
    if abs(denom) < 1e-9:
        return p_in
    t = (d_in - NEAR_PLANE_M) / denom
    return p_in + (p_out - p_in) * max(0.0, min(1.0, t))


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class PoseScene:
    """Static-per-flight geometry plus whatever the current frame contributes.

    Construction does all the per-flight work (reading the db, recentring the
    track, projecting the geotags); per-frame state arrives through
    set_frame(), which is cheap enough to call every render tick.
    """

    def __init__(self, flight_dir: Path, rel_name: str, max_frame_idx: int):
        self.rel_name = rel_name
        self.proj = load_projector(flight_dir)
        self.frame_to_row, self.warning = build_frame_index(self.proj, max_frame_idx)
        self.by_basename = basename_index(self.proj)

        track = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in self.proj.poses], dtype=float)
        self.origin = track.mean(axis=0)
        self.track = track - self.origin
        self.rtk = np.array([int(r[7]) if r[7] is not None else -1 for r in self.proj.poses])

        clicks = np.asarray(self.proj.clicks, dtype=float).reshape(-1, 2) if len(self.proj.clicks) else np.zeros((0, 2))
        # The survey table stores easting/northing only. Geotags are ground
        # truth *on the ground*, so they are placed on the ground plane rather
        # than given a fabricated height.
        self.ground_z = float(np.median(self.track[:, 2])) - self._median_radalt()
        self.geotags = (np.column_stack([clicks[:, 0] - self.origin[0],
                                         clicks[:, 1] - self.origin[1],
                                         np.full(len(clicks), self.ground_z)])
                        if len(clicks) else np.zeros((0, 3)))

        # Bounding radius of everything worth seeing (track + geotags) about
        # the survey centre. OrbitCamera.fit() turns this into a standoff once
        # the viewport is known -- which it is not at construction time, since
        # the same scene renders into both a 900x700 window and a ~1100x250
        # inline strip.
        spread = self.track[:, :2]
        if len(self.geotags):
            spread = np.vstack([spread, self.geotags[:, :2]])
        self.survey_radius = (float(np.linalg.norm(spread, axis=1).max())
                              if len(spread) else 50.0)
        self.camera = OrbitCamera(np.array([0.0, 0.0, self.ground_z]),
                                   max(30.0, self.survey_radius * 2.0))
        self._fitted = False   # first render() fits to the real viewport

        # When True the view rides along with the aircraft: set_frame() keeps
        # the orbit target on the current footprint. Off by default, because
        # the first thing you want on opening the window is the whole survey
        # for orientation; [g] cycles between the two.
        self.follow = False
        self.basename_misses = 0
        # Directory-wide detection cloud. Built once per directory by
        # build_cloud(); colours are refreshed separately (and far more often,
        # as judgments change) by set_cloud_colors().
        self.cloud_xyz = np.zeros((0, 3))
        self.cloud_frame = np.zeros(0, dtype=np.int64)
        self.cloud_colors = np.zeros((0, 3), np.uint8)
        self.cloud_geotag_hit = np.zeros(0, dtype=bool)
        self.cloud_note = ""
        self.cloud_layer = np.zeros(0, dtype=np.int16)
        self.cloud_layer_visible = np.ones(1, dtype=bool)
        self.show_cloud = True
        # Geo-cluster centroids, set by set_centroids(). Separate from the
        # cloud in every way that matters: ~560 points not ~105k, so they are
        # drawn with real cv2 calls (depth-sorted, ringed) rather than a
        # numpy scatter, and they recolour as review proceeds while the cloud
        # never does.
        self.centroids = np.zeros((0, 3))
        self.centroid_colors = np.zeros((0, 3), np.uint8)
        self.centroid_conflict = np.zeros(0, dtype=bool)
        self.centroid_wide = np.zeros(0, dtype=bool)
        self.centroid_tally: Dict[str, int] = {}
        self.centroid_rule: Optional[str] = None
        self.centroid_note = ""
        self.show_centroids = True
        self._cloud_proj_key = None      # camera+viewport the cached projection belongs to
        self._cloud_proj: Optional[Tuple[np.ndarray, np.ndarray]] = None
        # Per-frame state, set by set_frame().
        self.frame_idx: Optional[int] = None
        self.row_idx: Optional[int] = None
        self.cam_pos: Optional[np.ndarray] = None
        self.footprint: Optional[np.ndarray] = None
        self.detections: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
        self.selected: Optional[np.ndarray] = None
        self.frame_note: str = ""
        self.pose_text: List[str] = []

    def _median_radalt(self) -> float:
        vals = [abs(float(r[8])) for r in self.proj.poses if r[8] is not None]
        return float(np.median(vals)) if vals else 0.0

    # -- per-frame ---------------------------------------------------------

    def row_for(self, frame_idx: int, image_path: Optional[str]) -> Optional[int]:
        """Pose row for a GUI frame. Basename wins over the positional index
        when the two disagree -- the filename is evidence, the index is a
        convention -- but the positional index is the fallback for a
        directory whose paths have been rewritten (review_gui already has a
        stale-path problem it solves with its own image index)."""
        if image_path:
            hit = self.by_basename.get(Path(image_path).name)
            if hit is not None:
                return hit
            self.basename_misses += 1
        return self.frame_to_row.get(frame_idx)

    def set_frame(self, frame_idx: int, image_path: Optional[str],
                  detections: Sequence[Tuple[float, float, Tuple[int, int, int]]] = (),
                  selected: Optional[Tuple[float, float]] = None) -> None:
        """Point the scene at one frame. `detections` are (x, y, bgr) in
        native image pixels; they are ray-cast onto this frame's ground plane
        here rather than by the caller, so review_gui never has to know the
        projection math."""
        self.frame_idx = frame_idx
        self.detections, self.selected, self.frame_note = [], None, ""
        self.row_idx = self.row_for(frame_idx, image_path)
        if self.row_idx is None:
            self.cam_pos, self.footprint, self.pose_text = None, None, []
            self.frame_note = f"no pose row for frame {frame_idx}"
            return
        row = self.proj.poses[self.row_idx]
        T_WC, radalt = self.proj.frame_pose(row)
        self.cam_pos = np.array([float(row[0]), float(row[1]), float(row[2])]) - self.origin
        # The ground plane this frame was actually projected onto (to_3d
        # terminates its rays at z = camera_z - radalt). The grid and the
        # geotags follow it, so they always sit in the same plane as the
        # footprint and the back-projected detections rather than floating
        # above or below them wherever the terrain differs from the
        # flight-wide median used at construction.
        self.ground_z = float(self.cam_pos[2]) - radalt
        if len(self.geotags):
            self.geotags[:, 2] = self.ground_z

        w, h = self.proj.res
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
        fp = self.proj.to_3d(corners, T_WC, radalt)
        # to_3d drops non-finite rows, so a grazing/degenerate pose can return
        # fewer than 4 corners -- in which case there is no quad to draw.
        self.footprint = (fp - self.origin) if len(fp) == 4 else None

        if detections:
            pts = np.array([[d[0], d[1]] for d in detections], dtype=float)
            p3 = self.proj.to_3d(pts, T_WC, radalt)
            if len(p3) == len(detections):
                self.detections = [(p3[i] - self.origin, detections[i][2]) for i in range(len(p3))]
            else:
                # Partial results can't be matched back to their colours, so
                # drop the layer rather than mislabel it.
                self.frame_note = "some detections did not project"
        if selected is not None:
            p3 = self.proj.to_3d(np.array([selected], dtype=float), T_WC, radalt)
            if len(p3) == 1:
                self.selected = p3[0] - self.origin

        if self.follow:
            # Track the ground point under the camera, not the camera itself:
            # orbiting about the footprint keeps what the frame actually shows
            # in the middle of the view. Only the target moves -- distance,
            # azimuth and elevation are left alone, so a manual orbit or zoom
            # persists as the aircraft advances during playback.
            self.camera.target = np.array([self.cam_pos[0], self.cam_pos[1], self.ground_z])

        e = _euler_deg(row)
        self.pose_text = [
            f"frame {frame_idx}  (pose row {self.row_idx} of {len(self.proj.poses)})",
            f"UTM  E {float(row[0]):.2f}   N {float(row[1]):.2f}   alt {float(row[2]):.2f} m",
            f"roll {e[0]:+.2f}  pitch {e[1]:+.2f}  yaw {e[2]:+.2f}  (deg)",
            f"radalt {abs(float(row[8])):.2f} m raw -> {radalt:.2f} m tilt-corrected"
            f"   rtk_status {row[7]}",
        ]

    # -- directory-wide detection cloud -------------------------------------

    def build_cloud(self, frames: np.ndarray, xy: np.ndarray,
                    image_paths: Optional[Dict[int, str]] = None,
                    layer: Optional[np.ndarray] = None) -> np.ndarray:
        """Back-project EVERY detection in the directory, each through its own
        frame's pose, and keep the result as a static cloud.

        `frames` (N,) and `xy` (N,2) are parallel; the returned (N,) boolean
        mask says which entries made it, so the caller's own parallel arrays
        (keys, colours) stay index-aligned. Returns all-False on failure --
        the cloud is an extra layer, never a reason for the view to break.

        Cost is dominated by one Projector.to_3d call per frame, not per
        point: measured 0.18 s for 191,202 points across 1,355 frames of the sample flight,
        frames, so this runs eagerly on open rather than needing the chunked
        treatment review_gui gives its matcher.

        A caveat worth knowing when reading the result: a physical target seen
        in 200 frames contributes 200 points, each landing on that frame's own
        ground plane, so the cloud SMEARS rather than converging to one dot
        per target. That is a property of per-frame ground-plane
        back-projection, not a bug here.
        """
        n = len(frames)
        valid = np.zeros(n, dtype=bool)
        self.cloud_xyz = np.zeros((0, 3))
        self.cloud_frame = np.zeros(0, dtype=np.int64)
        self.cloud_geotag_hit = np.zeros(0, dtype=bool)
        self.cloud_note = ""
        if n == 0:
            return valid
        stride = max(1, int(math.ceil(n / CLOUD_MAX_POINTS)))
        if stride > 1:
            self.cloud_note = f"cloud subsampled 1/{stride} ({n} detections)"
        take = np.zeros(n, dtype=bool)
        take[::stride] = True

        out = np.zeros((n, 3))
        order = np.argsort(frames, kind="stable")
        bounds = np.flatnonzero(np.diff(frames[order])) + 1
        for chunk in np.split(order, bounds):
            chunk = chunk[take[chunk]]
            if not len(chunk):
                continue
            fi = int(frames[chunk[0]])
            # row_for(), not frame_to_row directly: it prefers the image
            # basename, which is the evidence, over the positional convention.
            row_i = self.row_for(fi, (image_paths or {}).get(fi))
            if row_i is None:
                continue
            T_WC, radalt = self.proj.frame_pose(self.proj.poses[row_i])
            p3 = self.proj.to_3d(xy[chunk], T_WC, radalt)
            # to_3d silently drops non-finite rows, which would desync this
            # frame's points from the caller's arrays. A degenerate pose is
            # rare and losing one frame's contribution is invisible in a
            # cloud this size, so drop the whole frame rather than guess.
            if len(p3) != len(chunk):
                continue
            out[chunk] = p3 - self.origin
            valid[chunk] = True

        self.cloud_xyz = out[valid]
        self.cloud_frame = np.asarray(frames, dtype=np.int64)[valid]
        self.cloud_colors = np.zeros((len(self.cloud_xyz), 3), np.uint8)
        self.cloud_geotag_hit = self._geotag_hits(self.cloud_xyz)
        # Model-layer id per point, parallel to cloud_xyz. Slot 0 is reserved
        # for the populations that belong to no layer (judgment centroids,
        # label-matched greys), which are always visible.
        if layer is not None and len(layer) == len(valid):
            self.cloud_layer = np.asarray(layer, dtype=np.int16)[valid]
        else:
            self.cloud_layer = np.zeros(len(self.cloud_xyz), dtype=np.int16)
        n_slots = int(self.cloud_layer.max()) + 1 if len(self.cloud_layer) else 1
        self.cloud_layer_visible = np.ones(max(1, n_slots), dtype=bool)
        return valid

    def set_layer_visible(self, mask: Sequence[bool]) -> None:
        """Which model layers the cloud shows. Slot 0 (the non-layer
        populations) is forced visible.

        A bool write and nothing else: the mask is applied AFTER
        _cloud_projection in _draw_cloud, so toggling a layer costs one boolean
        AND over the cloud and NEVER invalidates the projection cache. Slicing
        the point array instead would change len(cloud_xyz), which is part of
        that cache key, and turn every toggle into a full reprojection."""
        m = np.asarray(mask, dtype=bool)
        out = np.ones(max(1, len(m)), dtype=bool)
        out[:len(m)] = m
        out[0] = True
        self.cloud_layer_visible = out

    def _geotag_hits(self, xyz: np.ndarray) -> np.ndarray:
        """Which cloud points land within GEOTAG_HIT_M of a survey geotag,
        matched in the ground plane exactly as geo_eval does (its DBSCAN and
        geotag matching both work on the UTM x/y of dets_3D, never z)."""
        if not len(xyz) or not len(self.geotags):
            return np.zeros(len(xyz), dtype=bool)
        try:
            from scipy.spatial import cKDTree
        except Exception:  # noqa: BLE001 -- the cloud still draws, just unmarked
            return np.zeros(len(xyz), dtype=bool)
        tree = cKDTree(self.geotags[:, :2])
        return tree.query(xyz[:, :2], distance_upper_bound=GEOTAG_HIT_M)[0] < np.inf

    def set_cloud_colors(self, colors: np.ndarray) -> None:
        """Per-point BGR, index-aligned with the surviving cloud points.
        Split from build_cloud because judgments change constantly while the
        geometry never does -- recolouring is a cheap array write, rebuilding
        would be a re-projection of the whole directory."""
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if len(colors) == len(self.cloud_xyz):
            self.cloud_colors = colors

    def set_centroids(self, xy: np.ndarray, colors: np.ndarray,
                      conflict: Optional[np.ndarray] = None,
                      wide: Optional[np.ndarray] = None,
                      tally: Optional[Dict[str, int]] = None,
                      rule: Optional[str] = None, note: str = "") -> None:
        """One marker per geo cluster, in ABSOLUTE UTM easting/northing.

        `xy` is (M,2) UTM -- the caller holds GeoScaffold.points, which are
        absolute, and recentring is this class's business (self.origin) so no
        caller has to know about it. Placed on the ground plane for the same
        reason the geotags are: a geo cluster is a claim about a spot on the
        ground, and giving it a fabricated height would invite reading the
        cloud's z-smear as terrain.

        `colors` is (M,3) BGR, index-aligned. `conflict` marks clusters
        holding BOTH a positive and a negative human verdict -- the ones whose
        colour is a function of the [o] convention rather than of agreement,
        which is worth ~14 precision points across 24 clusters on the sample flight and
        so must not be silently absorbed into the fill. `wide` marks clusters
        whose reach exceeds CENTROID_REACH_RING_MIN.

        Cheap by construction: no projection here, just an array write, so
        this can be called on every judgment and on every [o] cycle."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if len(colors) != len(xy):
            return
        self.centroids = (np.column_stack([xy[:, 0] - self.origin[0],
                                           xy[:, 1] - self.origin[1],
                                           np.full(len(xy), self.ground_z)])
                          if len(xy) else np.zeros((0, 3)))
        self.centroid_colors = colors
        self.centroid_conflict = (np.asarray(conflict, dtype=bool)
                                  if conflict is not None and len(conflict) == len(xy)
                                  else np.zeros(len(xy), dtype=bool))
        self.centroid_wide = (np.asarray(wide, dtype=bool)
                              if wide is not None and len(wide) == len(xy)
                              else np.zeros(len(xy), dtype=bool))
        self.centroid_tally = dict(tally or {})
        self.centroid_rule = rule
        self.centroid_note = note

    def centroids_primary(self) -> bool:
        """Whether the centroids are the layer being read, which is what
        dims the cloud. False when there are none or they are toggled off, so
        [y] off restores the cloud to full weight rather than leaving it
        dimmed for a population that is not on screen."""
        return bool(self.show_centroids and len(self.centroids))

    def content_radius(self) -> float:
        """Bounding radius of whatever the current framing is meant to show:
        the flight's own footprint when following, the whole survey when not."""
        if not self.follow:
            return self.survey_radius
        if self.footprint is not None and len(self.footprint):
            centre = self.footprint[:, :2].mean(axis=0)
            reach = float(np.linalg.norm(self.footprint[:, :2] - centre, axis=1).max())
        else:
            reach = 15.0
        if self.cam_pos is not None:
            # Include the aircraft itself, not just the ground patch -- the
            # frustum is the thing being looked at, and cropping its apex is
            # exactly what made following look broken.
            reach = max(reach, float(self.cam_pos[2]) - self.ground_z)
        return max(5.0, reach)

    def fit_points(self) -> np.ndarray:
        """Representative points the framing must contain: the whole survey
        normally, just this frame's footprint and aircraft when following.
        The track is subsampled -- 2,120 points add nothing to a bounding box
        that 200 already pin down, and fit() projects this set four times."""
        if self.follow:
            parts = []
            if self.footprint is not None and len(self.footprint):
                parts.append(self.footprint)
            if self.cam_pos is not None:
                parts.append(self.cam_pos.reshape(1, 3))
            return np.vstack(parts) if parts else np.zeros((0, 3))
        parts = []
        if len(self.track):
            parts.append(self.track[:: max(1, len(self.track) // 200)])
        if len(self.geotags):
            parts.append(self.geotags)
        return np.vstack(parts) if parts else np.zeros((0, 3))

    def scene_rect(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """The area the 3D scene may use, i.e. the viewport minus the band the
        HUD writes into. Shared by the fit and the renderer so the scene is
        never framed into space the text is about to cover."""
        band = self._hud_band(w, h)
        return (4, band, max(8, w - 4), max(band + 8, h - 4))

    def frame_view(self, w: int, h: int) -> None:
        """Point and stand the camera off so the current mode's content fits
        this viewport, at an elevation suited to its aspect."""
        cam = self.camera
        if self.follow and self.cam_pos is not None:
            cam.target = np.array([self.cam_pos[0], self.cam_pos[1], self.ground_z])
        elif not self.follow:
            cam.target = np.array([0.0, 0.0, self.ground_z])
        cam.elevation = cam.default_elevation(w, h)
        cam.fit(w, h, self.content_radius(), self.fit_points(), self.scene_rect(w, h))
        cam.mark_home()
        self._fitted = True

    def set_follow(self, on: bool) -> None:
        """Switch between survey framing and aircraft-following framing, and
        reframe for whichever viewport was last rendered into."""
        self.follow = on
        vp = self.camera._viewport or (900, 700)
        self.frame_view(*vp)

    # -- rendering ---------------------------------------------------------

    def render(self, w: int, h: int) -> np.ndarray:
        """Pure: returns a BGR canvas, touches no cv2 window. Same contract as
        review_gui.App.render(), and for the same reason -- it makes the whole
        view testable with no display attached."""
        w, h = max(160, int(w)), max(120, int(h))
        # First render decides the framing (the viewport is not known at
        # construction: the same scene serves a ~1100x250 inline strip and a
        # 900x700 window). Later renders only rescale if the shape changed.
        if not self._fitted:
            self.frame_view(w, h)
        else:
            self.camera.on_viewport(w, h)
        canvas = np.full((h, w, 3), COLOR["bg"], np.uint8)
        self._draw_grid(canvas, w, h)
        self._draw_cloud(canvas, w, h)
        self._draw_points(canvas, w, h, self.geotags, COLOR["geotag"], radius=3, marker="x")
        self._draw_centroids(canvas, w, h)
        if self.footprint is not None:
            self._draw_loop(canvas, w, h, self.footprint, COLOR["footprint"], 1)
        self._draw_track(canvas, w, h)
        if self.cam_pos is not None and self.footprint is not None:
            for corner in self.footprint:
                self._draw_line(canvas, w, h, self.cam_pos, corner, COLOR["frustum"], 1)
        for p, color in self.detections:
            self._draw_points(canvas, w, h, p.reshape(1, 3), color, radius=4, marker="dot")
        if self.selected is not None:
            self._draw_points(canvas, w, h, self.selected.reshape(1, 3), (0, 255, 255), radius=9, marker="ring")
        if self.cam_pos is not None:
            self._draw_aircraft(canvas, w, h)
        self._draw_axes(canvas, w, h)
        self._draw_text(canvas, w, h)
        return canvas

    # -- primitives --------------------------------------------------------

    def _seg_px(self, w: int, h: int, a: np.ndarray, b: np.ndarray):
        """Both endpoints of a segment as integer pixels, near-plane clipped,
        or None if the whole segment is behind the camera."""
        cam = self.camera
        pts, depth = cam.project(np.stack([a, b]), w, h)
        in_a, in_b = depth[0] > NEAR_PLANE_M, depth[1] > NEAR_PLANE_M
        if not in_a and not in_b:
            return None
        if not in_a:
            a = _clip_to_near(b, a, cam)
        elif not in_b:
            b = _clip_to_near(a, b, cam)
        pts, depth = cam.project(np.stack([a, b]), w, h)
        if not np.isfinite(pts).all():
            return None
        return (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1]))

    def _draw_line(self, canvas, w, h, a, b, color, thickness=1) -> None:
        seg = self._seg_px(w, h, np.asarray(a, float), np.asarray(b, float))
        if seg is not None:
            cv2.line(canvas, seg[0], seg[1], color, thickness, cv2.LINE_AA)

    def _draw_loop(self, canvas, w, h, pts: np.ndarray, color, thickness=1) -> None:
        for i in range(len(pts)):
            self._draw_line(canvas, w, h, pts[i], pts[(i + 1) % len(pts)], color, thickness)

    def _draw_points(self, canvas, w, h, pts: np.ndarray, color, radius: int, marker: str) -> None:
        if len(pts) == 0:
            return
        px, depth = self.camera.project(pts, w, h)
        ok = (depth > NEAR_PLANE_M) & np.isfinite(px).all(axis=1)
        # Reject absurd coordinates before int(): a point just past the near
        # plane can project to ~1e9 px, and cv2 raises on an out-of-int32 draw.
        ok &= (np.abs(px[:, 0]) < 1e5) & (np.abs(px[:, 1]) < 1e5)
        # Far to near, so nearer markers land on top within this one layer.
        order = np.argsort(-depth[ok])
        for i in np.flatnonzero(ok)[order]:
            c = (int(px[i][0]), int(px[i][1]))
            if marker == "ring":
                cv2.circle(canvas, c, radius, color, 2, cv2.LINE_AA)
            elif marker == "x":
                cv2.drawMarker(canvas, c, color, cv2.MARKER_TILTED_CROSS, radius * 2, 1)
            else:
                cv2.circle(canvas, c, radius, color, -1, cv2.LINE_AA)

    def _draw_grid(self, canvas, w, h) -> None:
        if len(self.track) == 0:
            return
        lo = np.floor(self.track[:, :2].min(axis=0) / GRID_SPACING_M) * GRID_SPACING_M - GRID_SPACING_M
        hi = np.ceil(self.track[:, :2].max(axis=0) / GRID_SPACING_M) * GRID_SPACING_M + GRID_SPACING_M
        z = self.ground_z
        for axis in (0, 1):
            n = int((hi[axis] - lo[axis]) / GRID_SPACING_M) + 1
            step = max(1, int(math.ceil(n / GRID_MAX_LINES)))
            other = 1 - axis
            for k in range(0, n, step):
                v = lo[axis] + k * GRID_SPACING_M
                a = np.zeros(3); b = np.zeros(3)
                a[axis] = b[axis] = v
                a[other], b[other] = lo[other], hi[other]
                a[2] = b[2] = z
                major = (k % (GRID_MAJOR_EVERY * step)) == 0
                self._draw_line(canvas, w, h, a, b, COLOR["grid_major"] if major else COLOR["grid"], 1)

    def _draw_track(self, canvas, w, h) -> None:
        if len(self.track) < 2:
            return
        px, depth = self.camera.project(self.track, w, h)
        ok = (depth > NEAR_PLANE_M) & np.isfinite(px).all(axis=1)
        ok &= (np.abs(px[:, 0]) < 1e5) & (np.abs(px[:, 1]) < 1e5)
        pts = px.astype(np.int32)
        cut = self.row_idx if self.row_idx is not None else len(self.track)
        comet = max(0, cut - COMET_FRAMES)
        # Three bands, drawn in order so the brightest lands on top:
        # already-flown, the last COMET_FRAMES before now, and not-yet-flown.
        # The comet is what makes playback legible at survey zoom -- the
        # aircraft itself moves sub-pixel per frame, but a moving 30-frame
        # streak does not.
        bands = ((0, comet + 1, COLOR["track_past"], 1),
                 (comet, cut + 1, COLOR["track_comet"], 2),
                 (cut, len(self.track), COLOR["track_future"], 1))
        # Split into runs of consecutive VISIBLE points and hand each to
        # cv2.polylines in one call -- 2,120 individual cv2.line calls per
        # tick would be the renderer's whole budget.
        for lo, hi in _runs(ok):
            for a, b, color, thick in bands:
                a, b = max(lo, a), min(hi, b)
                if b - a >= 2:
                    cv2.polylines(canvas, [pts[a:b]], False, color, thick, cv2.LINE_AA)

    def _cloud_buffers(self, w: int, h: int):
        """Zeroed scratch buffers for the cloud layer, reused across ticks.

        Three allocations of a canvas-sized array every tick is pure waste
        when the size almost never changes; cv2 needs real arrays to write
        into, so they are cleared rather than re-created."""
        want = (h, w)
        if getattr(self, "_cloud_buf_shape", None) != want:
            self._cloud_buf_shape = want
            self._cloud_buf = (np.zeros((h, w, 3), np.uint8),
                               np.zeros((h, w, 3), np.uint8),
                               np.zeros((h, w), np.uint8))
        for b in self._cloud_buf:
            b[...] = 0
        return self._cloud_buf

    def _cloud_projection(self, w: int, h: int):
        """Projected cloud pixels, cached against the camera pose and
        viewport. ~10 ms for 191k points on the sample flight, so recomputing it on
        every one of ~50 render ticks a second would dominate the frame
        budget -- but the camera is usually still, and when it is not (a
        follow-mode playback moves the target every frame) playback caps out
        at 16.7 fps anyway."""
        cam = self.camera
        key = (round(cam.azimuth, 4), round(cam.elevation, 4), round(cam.distance, 4),
               tuple(np.round(cam.target, 4)), cam.principal, w, h, len(self.cloud_xyz))
        if key != self._cloud_proj_key:
            self._cloud_proj = cam.project(self.cloud_xyz, w, h)
            self._cloud_proj_key = key
        return self._cloud_proj

    def _draw_cloud(self, canvas, w, h) -> None:
        """The whole directory's detections, translucent, in their 2D judgment
        colours -- so the review state of the entire flight is visible at once
        instead of only the frame on screen.

        Drawn as a blended overlay rather than per-point cv2 calls: 191k
        cv2.circle invocations would take seconds, while a numpy scatter into
        an overlay plus one addWeighted takes ~5 ms. Points on the CURRENT
        frame are excluded -- render() draws those solid and larger, and
        overplotting them with translucent dots would only mush them.
        """
        if not self.show_cloud or not len(self.cloud_xyz) or not len(self.cloud_colors):
            return
        px, depth = self._cloud_projection(w, h)
        xs, ys = px[:, 0], px[:, 1]
        # No isfinite() check: project() makes sx/sy NaN exactly where
        # depth <= NEAR_PLANE_M, and NaN (like inf) fails every one of the
        # bounds comparisons below, so the bounds test already excludes them.
        # The explicit check was costing 0.8 ms/tick to re-establish that.
        ok = (depth > NEAR_PLANE_M) & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        # Layer visibility, applied POST-projection on purpose -- see
        # set_layer_visible. ~0.3 ms over a 400k cloud, against ~10 ms to
        # reproject it.
        cl = getattr(self, "cloud_layer", None)
        if cl is not None and len(cl) == len(ok):
            vis = self.cloud_layer_visible
            ok &= vis[np.clip(cl, 0, len(vis) - 1)]
        if self.frame_idx is not None:
            ok &= self.cloud_frame != self.frame_idx
        if not ok.any():
            return
        xi = xs[ok].astype(np.int32)
        yi = ys[ok].astype(np.int32)
        colors = self.cloud_colors[ok]
        layer, hits, mono = self._cloud_buffers(w, h)
        layer[yi, xi] = colors
        mono[yi, xi] = 255
        # Step back when the centroids are the layer being read. Both the
        # blend weight and the marker size drop: alpha alone still leaves a
        # 105k-point field of 2px blocks dense enough to bury 560 centroids.
        dim = self.centroids_primary()
        mk = CLOUD_MARKER_PX_DIMMED if dim else CLOUD_MARKER_PX
        alpha = CLOUD_ALPHA_DIMMED if dim else CLOUD_ALPHA
        cv2.dilate(layer, _kernel(mk), dst=layer)
        cv2.dilate(mono, _kernel(mk), dst=mono)
        # Geotag hits get a bigger block, at the SAME colour -- the emphasis
        # has to be size, not hue. Grown separately and merged brightest-wins,
        # so a hit is never shrunk back by a neighbouring plain marker.
        hit = self.cloud_geotag_hit[ok] if len(self.cloud_geotag_hit) == len(ok) else None
        if hit is not None and hit.any():
            hits[yi[hit], xi[hit]] = colors[hit]
            cv2.dilate(hits, _kernel(CLOUD_HIT_PX), dst=hits)
            np.maximum(layer, hits, out=layer)
            cv2.dilate(mono, _kernel(CLOUD_HIT_PX), dst=mono)
        # Blend the whole canvas, then copy back only the painted pixels.
        # Boolean-indexing the blend in numpy instead cost 5.6 ms/tick; these
        # are two C-level passes over a small canvas.
        blended = cv2.addWeighted(layer, alpha, canvas, 1.0 - alpha, 0.0)
        cv2.copyTo(blended, mono, canvas)

    def _draw_centroids(self, canvas, w, h) -> None:
        """One marker per geo cluster, in its review-status colour.

        Real cv2 calls rather than the cloud's numpy-scatter-plus-blend: there
        are hundreds of these, not hundreds of thousands, and they are the
        layer being read, so they get opaque fills, depth ordering and rings.

        Drawn AFTER the cloud and the geotags and BEFORE the current frame's
        own detections, which stay on top -- the frame you are reviewing must
        never be occluded by the directory-wide summary.
        """
        if not self.centroids_primary() or not len(self.centroid_colors):
            return
        pts = self.centroids
        px, depth = self.camera.project(pts, w, h)
        ok = (depth > NEAR_PLANE_M) & np.isfinite(px).all(axis=1)
        ok &= (np.abs(px[:, 0]) < 1e5) & (np.abs(px[:, 1]) < 1e5)
        if not ok.any():
            return
        # Far to near, so nearer centroids land on top of further ones.
        order = np.argsort(-depth[ok])
        idx = np.flatnonzero(ok)[order]
        for i in idx:
            c = (int(px[i][0]), int(px[i][1]))
            col = tuple(int(v) for v in self.centroid_colors[i])
            if self.centroid_wide[i]:
                # Reach ring first, so the fill and the conflict ring sit
                # inside it. Same hue: the emphasis is size, exactly as it is
                # for the cloud's geotag hits.
                cv2.circle(canvas, c, CENTROID_REACH_RING_R, col, 1, cv2.LINE_AA)
            if self.centroid_conflict[i]:
                cv2.circle(canvas, c, CENTROID_RING_R, COLOR["conflict"], 2, cv2.LINE_AA)
            cv2.circle(canvas, c, CENTROID_R, col, -1, cv2.LINE_AA)
            # A thin dark keyline, so a centroid stays legible where it lands
            # on top of a same-coloured patch of dimmed cloud.
            cv2.circle(canvas, c, CENTROID_R, COLOR["bg"], 1, cv2.LINE_AA)

    def _draw_aircraft(self, canvas, w, h) -> None:
        """The current pose, sized in SCREEN pixels rather than metres, plus a
        drop line to the ground plane and a ground tick.

        At whole-survey zoom the aircraft moves well under a pixel per frame,
        so a marker that scales with the scene shrinks to nothing and playback
        looks frozen. This one is the same size at any zoom, and the drop line
        pins it to a specific spot on the grid -- at an oblique elevation a
        floating dot is genuinely ambiguous about where it is.
        """
        ground = np.array([self.cam_pos[0], self.cam_pos[1], self.ground_z])
        self._draw_line(canvas, w, h, ground, self.cam_pos, COLOR["camera"], 1)
        px, depth = self.camera.project(np.stack([self.cam_pos, ground]), w, h)
        for i, (r_out, r_in) in enumerate(((7, 3), (4, 0))):   # aircraft, then its ground tick
            if depth[i] <= NEAR_PLANE_M or not np.isfinite(px[i]).all() or np.abs(px[i]).max() >= 1e5:
                continue
            c = (int(px[i][0]), int(px[i][1]))
            cv2.circle(canvas, c, r_out, COLOR["camera"], 2, cv2.LINE_AA)
            if r_in:
                cv2.circle(canvas, c, r_in, COLOR["camera"], -1, cv2.LINE_AA)

    def _draw_axes(self, canvas, w, h) -> None:
        """An ENU triad at the orbit target. Without it, an orbited view has no
        cue for which way is north.

        Anchored to the target rather than the survey centre, and scaled to
        the camera's standoff rather than a fixed 10 m: in follow mode the
        survey centre is off somewhere behind you, and a fixed-length triad is
        either a full-screen cross or an invisible speck depending on zoom."""
        base = self.camera.target.astype(float).copy()
        arm = max(1.0, self.camera.distance * 0.07)
        for vec, color, label in ((np.array([arm, 0, 0]), COLOR["axis_e"], "E"),
                                  (np.array([0, arm, 0]), COLOR["axis_n"], "N"),
                                  (np.array([0, 0, arm]), COLOR["axis_u"], "U")):
            self._draw_line(canvas, w, h, base, base + vec, color, 2)
            px, depth = self.camera.project((base + vec * 1.2).reshape(1, 3), w, h)
            if depth[0] > NEAR_PLANE_M and np.isfinite(px[0]).all() and np.abs(px[0]).max() < 1e5:
                cv2.putText(canvas, label, (int(px[0][0]), int(px[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def hud_lines(self) -> Tuple[List[Tuple[str, Tuple[int, int, int]]],
                                 List[Tuple[str, Tuple[int, int, int]]],
                                 List[Tuple[str, Tuple[int, int, int]]]]:
        """(tier1, tier2, tier3) HUD content, most important first.

        Split out from the drawing so the layout can drop whole tiers when the
        viewport is short, and so a test can measure the strings that were
        actually emitted. Warnings sit in tier 1 with the frame identity: a
        mis-attributed pose or an out-of-window boresight must never be the
        thing that gets dropped to make room for a legend.
        """
        warn = COLOR["warn"]
        head = f"frame {self.frame_idx}" if self.frame_idx is not None else self.rel_name
        mode = "FOLLOW" if self.follow else "SURVEY"
        tier1: List[Tuple[str, Tuple[int, int, int]]] = [(f"[{mode}]  {head}", COLOR["text"])]
        miss = (f"! {self.basename_misses} frame(s) matched by row index, not filename "
                f"-- poses may be mis-attributed") if self.basename_misses else ""
        for note in (self.warning, miss, self.frame_note, self.boresight_note()):
            if note:
                tier1.append((note, warn))
        tier1.append(("drag orbit | right-drag pan | ctrl-drag or +/- zoom | g view | d cloud | "
                      "y centroids | v cycles", COLOR["dim"]))
        tier2 = [(line, COLOR["text"]) for line in self.pose_text[1:]] if self.pose_text else []
        # Three colour-coded lines, no more: this column sets the HUD band
        # height, and on a 250px inline strip every extra line is 7% of the
        # room the scene has to live in. The directory name is deliberately
        # absent -- review_gui's own status bar and side panel already carry it.
        tier3 = [
            (f"geotags {len(self.geotags)}", COLOR["geotag"]),
            (f"track {len(self.track)}", COLOR["track_past"]),
            ("footprint / frustum", COLOR["frustum"]),
        ]
        if len(self.cloud_xyz):
            n_hit = int(self.cloud_geotag_hit.sum())
            tier3.append((f"cloud {len(self.cloud_xyz)} ({n_hit} on geotags)"
                          if self.show_cloud else
                          f"cloud OFF ({len(self.cloud_xyz)} pts, [d])", COLOR["dim"]))
        if self.cloud_note:
            tier1.append((self.cloud_note, warn))
        # Centroid line LAST in tier 3 but first in importance once it is
        # populated, so it survives the tier-dropping a short viewport does.
        # The rule is named on it because the colours MEAN nothing without it
        # -- with [o] off these are grey by design, not unreviewed.
        if len(self.centroids):
            if not self.show_centroids:
                tier3.append((f"centroids OFF ({len(self.centroids)}, [y])", COLOR["dim"]))
            elif self.centroid_rule is None:
                tier3.append((f"centroids {len(self.centroids)} -- no status "
                              f"([o] picks a rule)", COLOR["centroid_none"]))
            else:
                bits = "  ".join(f"{k} {v}" for k, v in self.centroid_tally.items() if v)
                tier3.append((f"centroids {len(self.centroids)} [{self.centroid_rule}]"
                              + (f"  {bits}" if bits else ""), COLOR["text"]))
            if self.centroid_note:
                tier3.append((self.centroid_note, warn))
        return tier1, tier2, tier3

    def _hud_metrics(self, w: int, h: int) -> Tuple[float, int, bool]:
        """(font scale, line pitch, wide-layout?) for this viewport."""
        scale = max(0.34, min(0.5, h / 1500.0 + 0.3))
        return scale, int(round(scale * 38)), (w >= 700 and h < 420)

    def _hud_band(self, w: int, h: int) -> int:
        """Height reserved at the top for HUD text. In the wide/inline layout
        the HUD is three columns across the top, so the scene gets everything
        below it; in a tall view the HUD overlays the corners and the scene
        may use the full height."""
        scale, pitch, wide = self._hud_metrics(w, h)
        if not wide:
            return 0
        t1, t2, t3 = self.hud_lines()
        return 8 + pitch * max(len(t1), len(t2), len(t3)) + 4

    def _draw_text(self, canvas, w, h) -> None:
        """Lay the HUD out for THIS viewport.

        The font tracks the viewport, every line is width-fitted, tiers are
        dropped rather than overlapped, and a wide-short strip puts the
        secondary text in a second column instead of demanding vertical space
        it does not have.
        """
        scale, pitch, wide = self._hud_metrics(w, h)
        pad = 8
        tier1, tier2, tier3 = self.hud_lines()

        def column(lines, x, y, max_px, budget_bottom):
            for text, color in lines:
                if y > budget_bottom:
                    return y
                cv2.putText(canvas, _fit_text(text, max_px, scale), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
                y += pitch
            return y

        if wide:
            # Three columns across the top: identity+warnings, pose numbers,
            # legend+controls. Uses the strip's one plentiful dimension.
            col_w = (w - pad * 4) // 3
            for i, lines in enumerate((tier1, tier2, tier3)):
                column(lines, pad + i * (col_w + pad), pad + pitch, col_w - pad, h - pad)
            return
        y = column(tier1, pad, pad + pitch, w - pad * 2, h - pad)
        y = column(tier2, pad, y, w - pad * 2, h - pad)
        # Tier 3 is bottom-anchored when there is room for all of it, and
        # simply continues under tier 2 when there is not.
        bottom_start = h - pad - pitch * (len(tier3) - 1)
        column(tier3, pad, max(y, bottom_start), w - pad * 2, h - pad)

    def boresight_note(self) -> str:
        """This scene's boresight caveat. See boresight_note_for()."""
        return boresight_note_for(self.rel_name)


def boresight_note_for(rel_name: str) -> str:
    """This flight's boresight caveat, or "" when a calibration covers it.

    geo_eval's boresight is FLIGHT-DATE SPECIFIC (harelab-ucsc/birdseye:
    birdseye/birdseye.py). Getting
    it wrong silently translates every back-projected point -- measured at
    0.152 m for a three-week gap, larger than the smallest DBSCAN eps in use --
    so an uncovered flight says so on screen rather than presenting an
    approximate footprint as exact.

    Module-level rather than a PoseScene method so review_gui can ask without
    constructing a scene: the model-layer accumulation tier reprojects through
    this same boresight and owes the reader the same caveat, but it draws in
    the 2D frame view and never opens the 3D window.

    Keep the exception handler narrow enough that a renamed geo_eval constant
    fails loudly here instead of quietly returning "no caveat" for every flight.
    """
    try:
        from .geo_eval import BORESIGHT_CAMPAIGNS
    except ImportError:
        return "! boresight campaigns unavailable -- cannot verify this flight's calibration"
    date = rel_name.split("/")[0].replace("_", "-")
    if len(date) != len("YYYY-MM-DD"):
        return ""
    for c in BORESIGHT_CAMPAIGNS:
        if c["from"] <= date <= c["to"]:
            return ""
    ranges = ", ".join(f'{c["from"]}..{c["to"]}' for c in BORESIGHT_CAMPAIGNS)
    return (f"! no boresight calibration covers {date} (campaigns: {ranges}) "
            f"-- footprint/detections are approximate")


def _euler_deg(row) -> Tuple[float, float, float]:
    from scipy.spatial.transform import Rotation as R
    e = R.from_quat(np.asarray(row[3:7], dtype=float)).as_euler("xyz", degrees=True)
    return float(e[0]), float(e[1]), float(e[2])


_KERNELS: Dict[int, np.ndarray] = {}


def _kernel(size: int) -> np.ndarray:
    """Cached square dilation kernel.

    Growing markers with one cv2.dilate beats scattering each point once per
    pixel offset: the cost stops scaling with the 87k-point cloud (measured
    ~10 ms/tick for 4-8 writes each) and becomes a fixed function of canvas
    area. Dilation takes the per-channel maximum, so two differently-coloured
    markers that overlap blend towards the brighter one -- invisible at two or
    three pixels, and the alternative is a per-point Python loop.
    """
    k = _KERNELS.get(size)
    if k is None:
        k = _KERNELS[size] = np.ones((max(1, size), max(1, size)), np.uint8)
    return k


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[lo, hi) index ranges of consecutive True in a boolean mask."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------

class PoseView:
    """The scene plus mouse state. Owns its own cv2 mouse callback, bound to
    its own window, so it shares nothing with review_gui's on_mouse."""

    def __init__(self, scene: PoseScene):
        self.scene = scene
        self._drag: Optional[str] = None
        self._last: Tuple[int, int] = (0, 0)

    def on_mouse(self, event: int, x: int, y: int, flags: int, userdata) -> None:
        """Drag gestures only -- deliberately NOT the mouse wheel.

        This OpenCV is built against Qt5, and its window's own viewport
        (DefaultViewPort::wheelEvent in window_QT.cpp) zooms the whole
        displayed image on a wheel event. It forwards the event to this
        callback AND scales the view; a setMouseCallback handler receives
        events but cannot consume them, so there is no way to claim the wheel
        for the 3D camera. Handling it here as well only zoomed two things at
        once -- and, worse, the Qt viewport's zoom silently invalidates the
        window-to-canvas mapping every click in the frame image relies on.

        So the wheel is left entirely to Qt and the 3D view is zoomed by drag
        or by [+]/[-] instead, which also removes any reason to scroll over
        the window in the first place.
        """
        cam = self.scene.camera
        if event == cv2.EVENT_LBUTTONDOWN:
            # Ctrl+left is the zoom gesture for anyone without a middle button.
            self._drag = "zoom" if flags & cv2.EVENT_FLAG_CTRLKEY else "orbit"
            self._last = (x, y)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self._drag, self._last = "zoom", (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._drag, self._last = "pan", (x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            self._drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag:
            dx, dy = x - self._last[0], y - self._last[1]
            self._last = (x, y)
            if self._drag == "orbit":
                cam.orbit(-dx * ORBIT_DEG_PER_PX, dy * ORBIT_DEG_PER_PX)
            elif self._drag == "zoom":
                # Drag down = pull back, matching the [-] direction.
                cam.dolly(ZOOM_STEP ** (dy / ZOOM_DRAG_PX_PER_STEP))
            else:
                cam.pan(dx, dy)

    def on_key(self, ch: str) -> bool:
        """Keys forwarded from review_gui's own waitKey. Returns True if the
        key was consumed.

        Note these are GLOBAL, not focus-scoped: cv2.waitKey() reports the
        keystroke but not which highgui window had focus, so a 3D binding
        that reused a main-window letter would fire in both places at once.
        Hence [g] to refit rather than the more obvious [f], which is already
        the main window's play-speed cycle.
        """
        if ch == "d":
            self.scene.show_cloud = not self.scene.show_cloud
        elif ch == "y":
            # [y] and not [c]/[t]: unshifted (cv2.waitKey drops Shift here, so
            # a shifted binding is silently dead), and free -- review_gui's
            # main map already owns every other letter.
            self.scene.show_centroids = not self.scene.show_centroids
        elif ch == "g":
            self.scene.set_follow(not self.scene.follow)
        elif ch in ("+", "="):
            self.scene.camera.dolly(1.0 / ZOOM_STEP)
        elif ch == "-":
            self.scene.camera.dolly(ZOOM_STEP)
        else:
            return False
        return True

    def render(self, w: int, h: int) -> np.ndarray:
        return self.scene.render(w, h)
