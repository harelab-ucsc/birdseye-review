# SPDX-License-Identifier: MIT
# Copyright (c) 2026 harelab-ucsc
"""
Geospatially-clustered precision and recall (the paper's geospatial
precision/recall), computed live against the adjudication state so
`review_gui.py` can show them in its HUD.

WHY THIS EXISTS
---------------
`geo_eval.py` provides the metric-space primitives. It has no notion of the
human adjudication, and at a 1 cm dedup it takes 20 s / 24 GB -- neither of
which fits inside a review loop. This module computes the same quantities,
adds the adjudication terms, and does it fast enough to sit behind a keypress.

There are TWO precision arms: field geotags do not exhaustively enumerate the
visually detectable targets, so the model fires on real burrow features nobody
geotagged and the naive statistic understates it. So do both arms, always:

  GEO PRE   geometry only.
  GEO POST  adjudication applied.

DEFINITIONS
-----------
Precision is over detection CLUSTERS, recall over GEOTAGS -- one cluster can
absorb several geotags -- so the two ratios do not share a denominator and
their "F1" would be a category error. Not computed here; see geo_prf's
docstring.

  PRE   TP  cluster with >=1 geotag within eps_predict of a cluster CORE point
        FP  cluster with none
        FN  geotag not assigned to any cluster

  POST  as PRE, then human verdicts override geometry in both directions:
          hard_negative -> FP,  confirmed_real -> TP
        `uncertain` and `removed` are excluded from the vote entirely; a cluster
        whose only verdicts are those falls back to its PRE geometry. Where a
        cluster holds BOTH verdicts the positive wins by default -- see
        VERDICT_RULES.

RECALL HAS TWO ARMS TOO, AND POST IS A BOUND
--------------------------------------------
Recall is computed per arm. POST precision defines a TP as
(human-adjudicated positive) OR (geometrically matched to a geotag), which
ENLARGES the positive class beyond the geotag survey; recall's denominator IS
the positive class. A single recall shared by both arms would score them under
two different definitions of "true positive", which is not a confusion matrix.

  R_PRE  = n_hit / n_den
  R_POST = (n_hit + c) / (n_den + c + a)

  c  adjudicated-positive clusters that matched no geotag -- real targets the
     survey omitted, found because the detector fired there
  a  spatially deduplicated human-added markers ("added" judgments) -- real
     targets the survey omitted AND the detector missed

R_POST IS AN UPPER BOUND, NOT A POINT ESTIMATE. An unmarked target is only
discoverable where the detector fired, so c is censored and can only raise the
ratio; writing the unobservable remainder as m, R_true = (n_hit+c)/(n_den+c+a+m)
<= R_POST. Added markers are the only observable slice of m. R_PRE remains the
unbiased point estimate under the assumption that survey omission is
independent of detectability -- defensible here because the geotags are a
walking RTK field survey (bE_config/data.csv) while the detector reads aerial
imagery, so the two miss-mechanisms are physically unrelated. Report the pair.
See post_recall() for the full derivation and review_gui.corrected_prf for the
same treatment in pixel space.

UNIT CAVEATS, WHICH DO NOT CANCEL: c counts CLUSTERS while recall counts
GEOTAGS and one cluster can hold five, so c under-counts and pushes R_POST down
toward R_PRE (conservative). a groups clicks at eps and can merge two adjacent
added burrows, so a under-counts too -- but a is in the DENOMINATOR, so that
pushes R_POST up (anti-conservative).

Each arm is reported against two denominators: all geotags, and geotags in view
under the in-view polygon test (image bounds inset by the 81.5 px
projection-noise buffer).

THE CONFLICTED-CLUSTER PROBLEM -- read before quoting POST
-----------------------------------------------------------
DBSCAN at eps=0.20 m chains: median cluster extent on the full 2,120-frame sample flight is 0.44 m and one
cluster holds five geotags, so a real burrow and a hard-negative rock 30 cm
apart are ONE cluster. Measured there: 26 of 95 judged clusters contain both
`confirmed_real` and `hard_negative` members. Which verdict wins is therefore
worth ~14 precision points on its own -- POST runs 0.476 / 0.411 / 0.321 across
positive_wins / majority / hard_negative_wins -- decided by a tie-break rather
than by data. (Those counts move as review continues; the spread is the point,
not the exact figures.)

THERE IS NO PRIMARY RULE. `geo_prf` returns `post` and `recall.post` keyed by
rule, all three, always, and every display shows all three: the HUD draws one
row each, [p] one line each, the PDF one table row each, and geo_metrics.json
one object each. There is no flag that selects one rule and no side-channel
for the others -- an appendix is exactly where a reader skips the number that
undercuts the headline. See VERDICT_RULES for the full argument.
`n_conflicted` is part of the output for the same reason.

Note the spread applies to POST RECALL too, not just precision: `c` counts
confirmed clusters, so it moves with the tie-break.

COST, measured on the full 2,120-frame sample flight 2025_04_16/pieranch_rect
(690,162 raw accumulated-reprojection rows, 161 geotags)
-----------------------------------------------------------------------------
    dedup    points    DBSCAN    peak RSS   clusters   geotags hit   clusters hit
     1 cm   332,330     19.9 s     24.2 GB       170           103             67
     2 cm   178,538      4.2 s      3.8 GB       169           103             68
     3 cm   113,169      1.6 s      1.2 GB       168           103             68
     5 cm    57,077     0.45 s      0.6 GB       166           103             68

1 cm is the reference setting. It is also unusable interactively. Geotags-hit
is 103 at EVERY resolution -- recall does not move at all -- and only the
cluster count drifts, by 1.2% at 3 cm. Hence GUI_DEDUP_CM = 3.0 for the live
HUD and REF_DEDUP_CM = 1.0 for reference figures. `--dedup-cm` is the knob; the
value used is recorded in every JSON this module writes.

Do NOT reach for subsampling instead. Striding divides points-per-burrow and
`min_samples=10` then rejects real burrows as noise. Grid dedup is not
subsampling: it removes redundant copies of the SAME 3D point that an
accumulated reprojection creates, which is why recall survives it untouched.

A CAVEAT ABOUT min_samples ON THIS INPUT
-----------------------------------------
geo_eval.py documents min_samples=10 as "a burrow seen in fewer frames is
dropped as noise". That is not what it does here. min_samples counts POINTS, not
distinct frames, and after dedup the two have come apart: measured on the full 2,120-frame sample flight
at the 3 cm default, only 34 of 113,169 points are noise while 29 of 168
clusters are supported by fewer than 10 distinct source frames (minimum 2);
at the 1 cm reference it is 16 of 332,330 and 30 of 170. The parameter is very nearly
inert on this input. `n_clusters_under_min_frames` is reported so that stays
visible; do not describe the filter as frame-based in any UI string.

PROVENANCE
----------
Nothing about the projection or clustering math is reimplemented here --
Projector, dbscan_predict and the four distance constants are imported from
geo_eval.py, and the in-view polygon test from run_detector.py. This module only
adds the adjudication layer, the in-view denominator, and the accounting.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .geo_eval import (
    DBSCAN_EPS, DBSCAN_MIN_SAMPLES, EPS_PREDICT_DETS,
    Projector, dbscan_predict,
)
from .run_detector import inner_bound

# Grid dedup resolution. See the cost table above: 1 cm is geo_eval.py's
# reference and reproduces its artifact; 3 cm is 12x faster for a 1.2% drift in
# cluster count and none at all in geotags hit.
REF_DEDUP_CM = 1.0
GUI_DEDUP_CM = 3.0

# Frames logged below this altitude are on the ground (onboard birdseye gate).
MIN_RADALT_M = 3.0

# Which verdict wins when one geo cluster holds contradictory human judgments.
#
# THERE IS NO DEFAULT AND NO WAY TO SELECT ONE. All three are always computed
# and all three are always displayed, everywhere -- HUD, [p], geo_metrics.json
# and the PDF report. This is deliberate and it is not a hedge:
#
#   * The choice is worth ~14 precision points on the full 2,120-frame sample flight (0.476 / 0.411 /
#     0.321). That is larger than most effects this project reports.
#   * It is decided by a TIE-BREAK RULE, NOT BY DATA. A conflicted cluster is an
#     artefact of DBSCAN chaining at eps=0.20m -- two physically distinct spots
#     30cm apart merged into one -- so the two verdicts are not disagreeing
#     about the same object at all. No amount of further review resolves it;
#     only a smaller eps would, and that has its own costs.
#   * A number whose value is set by an arbitrary convention, quoted without
#     that convention's alternatives, is not a measurement.
#
# So the spread IS the result. There is no flag selecting a primary arm and no
# side-channel for the others. The surest way to stop one number being quoted
# alone is to never compute it alone.
#
# Ordered MOST-POSITIVE FIRST, which is also the order every display uses:
# positive_wins >= majority >= hard_negative_wins by construction, so a reader
# can see at a glance that the list is a monotone bracket rather than three
# unrelated estimates. The smoke test asserts that ordering.
VERDICT_RULES = ("positive_wins", "majority", "hard_negative_wins")

# Short labels for the rules, for displays where a full name will not fit.
VERDICT_RULE_ABBR = {"positive_wins": "pos", "majority": "maj",
                     "hard_negative_wins": "neg"}

POSITIVE_JUDGMENT = "confirmed_real"
NEGATIVE_JUDGMENT = "hard_negative"
# Excluded from the PRECISION vote entirely -- an abstention, not a negative. A
# cluster whose only verdicts are these falls back to its PRE geometry.
#
# "added" is here for a different reason than the rest: an added marker is not a
# detection at all (the model never fired there), so it has nothing to say about
# whether a cluster is a true positive. It is NOT thereby excluded from the
# accounting -- collect_added() consumes it separately as the `a` term in the
# POST recall denominator, where it is the only thing constraining that arm's
# upper bound from above. See post_recall().
ABSTAIN_JUDGMENTS = ("uncertain", "removed", "unreviewed", "added")


# ---------------------------------------------------------------------------
# The in-view polygon test (image bounds inset by the 81.5 px projection-noise buffer)
# ---------------------------------------------------------------------------

def inside_vec(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised `run_detector.inside`: which of `pts` (N,2) lie inside the
    convex polygon `poly` (K,2).

    This is the in-view polygon test -- the cross product r_edge x r_tag evaluated on a
    walk through every boundary segment, with a consistent sign meaning "inside".
    run_detector.inside does one point at a time in Python; the in-view
    denominator needs 161 geotags x 1,355 frames, so it gets a vectorised twin.
    Asserted equivalent point-for-point in smoke_test_gui.check_geo_metrics.

    Collinear points (c == 0) are skipped there, i.e. a point exactly on an edge
    does not by itself decide the verdict. Reproduced here so the two agree on
    that edge case rather than only on the interior.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    poly = np.asarray(poly, dtype=float)
    if not len(pts):
        return np.zeros(0, dtype=bool)
    k = len(poly)
    pos = np.zeros(len(pts), dtype=bool)   # saw a strictly positive cross
    neg = np.zeros(len(pts), dtype=bool)   # saw a strictly negative cross
    for i in range(k):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % k]
        c = (x2 - x1) * (pts[:, 1] - y1) - (y2 - y1) * (pts[:, 0] - x1)
        pos |= c > 0
        neg |= c < 0
    return ~(pos & neg)


def geotags_in_view(proj: Projector, frame_rows: Sequence,
                    geotags: np.ndarray, by_frame: bool = False) -> np.ndarray:
    """(G,) bool -- which geotags project inside at least one processed frame.

    `by_frame=True` returns the full (F, G) matrix instead, un-collapsed and
    with the early exit disabled. Same test, same code path -- callers that need
    to ask "how many geotags would a WINDOW of frames see" must not re-derive
    the visibility rule, because a recall denominator that disagrees with the
    visibility test caps or inflates attainable recall.

    This is a visibility test. The alternative -- a 0.05 m projection
    ROUND-TRIP test (geo_eval.EPS_PREDICT_CLICKS) -- is not a visibility test,
    and it lifts recall by 10.4 points on the full 2,120-frame sample flight.
    So use the polygon test and report both denominators.

    Geotags are given as UTM x/y; each is placed on the frame's own ground plane
    before projection, matching how the onboard birdseye code synthesises click z.
    """
    geotags = np.asarray(geotags, dtype=float).reshape(-1, 2)
    frame_rows = list(frame_rows)
    seen = np.zeros(len(geotags), dtype=bool)
    per_frame = (np.zeros((len(frame_rows), len(geotags)), dtype=bool)
                 if by_frame else None)
    if not len(geotags):
        return per_frame if by_frame else seen
    poly = inner_bound(proj.res)
    for i, row in enumerate(frame_rows):
        T_WC, radalt = proj.frame_pose(row)
        if radalt <= MIN_RADALT_M:
            continue
        ground_z = T_WC[2, 3] - radalt
        p3 = np.column_stack([geotags, np.full(len(geotags), ground_z)])
        px = proj.to_2d(p3, T_WC)
        ok = np.isfinite(px).all(axis=1)
        if ok.any():
            hit = np.zeros(len(geotags), dtype=bool)
            hit[ok] = inside_vec(px[ok], poly)
            seen |= hit
            if by_frame:
                per_frame[i] = hit
        # The early exit is only sound when collapsing: the by-frame matrix has
        # to be complete for every frame, not just up to saturation.
        if seen.all() and not by_frame:
            break
    return per_frame if by_frame else seen


# ---------------------------------------------------------------------------
# The scaffold: expensive, geometry only, no adjudication in it
# ---------------------------------------------------------------------------

@dataclass
class GeoScaffold:
    """Everything geometric, computed once. Carries no judgment state, so it
    survives every commit and only has to be rebuilt when the detections or the
    clustering parameters change."""
    points: np.ndarray                 # (N,3) deduplicated UTM
    labels: np.ndarray                 # (N,) DBSCAN label, -1 = noise
    core_mask: np.ndarray              # (N,) bool, DBSCAN core samples
    n_clusters: int
    n_noise: int
    geotags: np.ndarray                # (G,2) UTM
    geotag_label: np.ndarray           # (G,) owning cluster, -1 = unmatched
    in_view: np.ndarray                # (G,) bool, in-view polygon test
    cell_to_point: Dict[Tuple[int, int, int], int]   # grid cell -> row of `points`
    cluster_frames: Dict[int, int]     # cluster -> distinct source frames
    params: dict
    unavailable: Optional[str] = None  # set instead of raising, see build()

    @property
    def ok(self) -> bool:
        return self.unavailable is None


def _empty_scaffold(reason: str) -> GeoScaffold:
    return GeoScaffold(
        points=np.zeros((0, 3)), labels=np.zeros(0, dtype=int),
        core_mask=np.zeros(0, dtype=bool), n_clusters=0,
        n_noise=0, geotags=np.zeros((0, 2)), geotag_label=np.zeros(0, dtype=int),
        in_view=np.zeros(0, dtype=bool), cell_to_point={}, cluster_frames={},
        params={}, unavailable=reason)


def _frame_to_utm(proj: Projector, row, xy: np.ndarray) -> Optional[np.ndarray]:
    """(K,3) UTM for K pixel points in ONE frame, or None if the frame is
    unusable.

    The single definition of "how a pixel in this frame becomes a UTM point":
    the radalt gate (MIN_RADALT_M) plus to_3d plus the length check that
    catches to_3d silently dropping non-finite rows. Three callers need exactly
    this sequence -- backproject (the scaffold), locate (the judgment join) and
    collect_added (the added-marker fold) -- and they must not be able to
    disagree about it, because the whole join from a judged point back to its
    geo cluster is an exact grid-cell lookup that only works if the arithmetic
    upstream of it was bit-for-bit the same.

    Returns None rather than a partial array: a degenerate pose is rare, and
    dropping the frame whole beats guessing which rows to_3d discarded (that
    guess would desync the caller's parallel per-point arrays).
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if row is None or not len(xy):
        return None
    T_WC, radalt = proj.frame_pose(row)
    if radalt <= MIN_RADALT_M:
        return None
    p3 = proj.to_3d(xy, T_WC, radalt)
    if len(p3) != len(xy):
        return None
    return p3


def backproject(proj: Projector, frames: Dict[int, object],
                image_paths: Optional[Dict[int, str]] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project every raw detection through its OWN frame's pose.

    `frames` is detections.parse_annotations output -- the GUI already
    holds it as DirectoryReview.dets_fd, so passing it in avoids a second
    parse.

    Applies the radalt gate (MIN_RADALT_M) so every caller agrees on which
    frames contribute -- via _frame_to_utm, the
    shared definition. Returns (P (M,3) UTM, src (M,) frame index),
    UNdeduplicated -- dedup happens in build_scaffold, which needs the pre-dedup
    rows to build its cell index.
    """
    by_index = {i: row for i, row in enumerate(proj.poses)}
    pts: List[np.ndarray] = []
    src: List[np.ndarray] = []
    for fi in sorted(frames):
        row = by_index.get(fi)
        if row is None:
            continue
        pf = frames[fi].points
        if not pf:
            continue
        xy = np.array([[p.x, p.y] for p in pf], dtype=float)
        p3 = _frame_to_utm(proj, row, xy)
        if p3 is None:
            continue
        pts.append(p3)
        src.append(np.full(len(p3), fi, dtype=np.int64))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    return np.vstack(pts), np.concatenate(src)


def build_scaffold(flight_dir: Path, frames: Dict[int, object],
                   eps: float = DBSCAN_EPS,
                   min_samples: int = DBSCAN_MIN_SAMPLES,
                   eps_predict: float = EPS_PREDICT_DETS,
                   dedup_cm: float = GUI_DEDUP_CM,
                   proj: Optional[Projector] = None) -> GeoScaffold:
    """The expensive half. Never raises -- a missing flight_data.db returns a
    scaffold with `unavailable` set, because 7 of the 30 labelled directories
    have no pose data and the metric is an extra layer, never a reason for the
    review loop to break."""
    try:
        if proj is None:
            db = Path(flight_dir) / "flight_data.db"
            if not db.exists():
                return _empty_scaffold(f"no flight_data.db in {Path(flight_dir).name}")
            proj = Projector(db)
    except Exception as e:  # noqa: BLE001 -- see docstring
        return _empty_scaffold(f"pose data unreadable: {e}")

    geotags = np.asarray(proj.clicks, dtype=float).reshape(-1, 2)
    if not len(geotags):
        return _empty_scaffold("flight has no geotag survey (clicks_flight_data empty)")

    try:
        allp, allsrc = backproject(proj, frames)
    except Exception as e:  # noqa: BLE001
        return _empty_scaffold(f"back-projection failed: {e}")
    if not len(allp):
        return _empty_scaffold("no detections back-projected")

    # Grid dedup. An accumulated reprojection re-emits every cluster point into
    # every later frame, so the same physical 3D point arrives dozens of times;
    # the duplicates inflate local density (changing what min_samples means) and
    # blow up DBSCAN's neighbour lists. See the cost table in the module
    # docstring -- this is what makes the whole thing interactive.
    step = dedup_cm / 100.0
    grid = np.round(allp / step).astype(np.int64)
    uniq, first_idx = np.unique(grid, axis=0, return_index=True)
    points = allp[first_idx]
    src = allsrc[first_idx]

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])
    labels = clustering.labels_
    geotag_label = dbscan_predict(clustering, points[:, :2], geotags, eps_predict)

    # Which rows are DBSCAN CORE samples. The scaffold does not retain the
    # fitted estimator (it is not picklable-cheap and nothing else needs it),
    # but collect_added has to ask "did the detector fire here" using the same
    # core-point radius rule dbscan_predict implements, so keep the mask.
    core_mask = np.zeros(len(points), dtype=bool)
    core_mask[clustering.core_sample_indices_] = True

    # Distinct SOURCE FRAMES per cluster. min_samples thresholds points, not
    # frames, and after dedup those have come apart -- see the module docstring.
    cluster_frames: Dict[int, int] = {}
    for lab in np.unique(labels):
        if lab == -1:
            continue
        cluster_frames[int(lab)] = int(len(np.unique(src[labels == lab])))

    cell_to_point = {tuple(int(v) for v in cell): i for i, cell in enumerate(uniq)}

    return GeoScaffold(
        points=points, labels=labels, core_mask=core_mask,
        n_clusters=int(len(set(labels.tolist()) - {-1})),
        n_noise=int((labels == -1).sum()),
        geotags=geotags, geotag_label=geotag_label,
        in_view=geotags_in_view(proj, proj.poses, geotags),
        cell_to_point=cell_to_point, cluster_frames=cluster_frames,
        params={"eps": eps, "min_samples": min_samples,
                "eps_predict": eps_predict, "dedup_cm": dedup_cm,
                "min_radalt_m": MIN_RADALT_M, "rtk_only": False,
                "n_raw_backprojections": int(len(allp)),
                "boresight": dict(proj.boresight),
                "flight_date": proj.flight_date},
    )


def locate(scaffold: GeoScaffold, proj: Projector, frame_row, frame_idx: int,
           xy: np.ndarray) -> np.ndarray:
    """(K,) cluster label for K pixel points in one frame, -1 if unresolvable.

    The join from a judged GUI cluster to a geo cluster. It reruns exactly the
    arithmetic build_scaffold used -- same pose, same to_3d, same grid rounding
    -- so a point that contributed to the scaffold lands in its own cell by
    construction. That is why this is an index lookup and not a radius search:
    no tie-break, no majority vote over neighbours, nothing to get subtly wrong.
    """
    p3 = _frame_to_utm(proj, frame_row, xy)
    if p3 is None:
        return np.full(len(xy), -1, dtype=int)
    step = scaffold.params["dedup_cm"] / 100.0
    cells = np.round(p3 / step).astype(np.int64)
    out = np.full(len(cells), -1, dtype=int)
    for i, cell in enumerate(cells):
        j = scaffold.cell_to_point.get((int(cell[0]), int(cell[1]), int(cell[2])))
        if j is not None:
            out[i] = scaffold.labels[j]
    return out


# ---------------------------------------------------------------------------
# The adjudication layer: cheap, rebuilt whenever JudgmentStore.version ticks
# ---------------------------------------------------------------------------

@dataclass
class GeoVerdicts:
    """Per-geo-cluster tallies of the human verdicts landing in it."""
    votes: Dict[int, Dict[str, int]] = field(default_factory=dict)
    n_unresolved: int = 0     # judged clusters that reached no geo cluster
    n_judged_keys: int = 0

    def tally(self, cluster: int, judgment: str) -> None:
        self.votes.setdefault(cluster, {}).setdefault(judgment, 0)
        self.votes[cluster][judgment] += 1

    @property
    def n_conflicted(self) -> int:
        return sum(1 for v in self.votes.values()
                   if v.get(POSITIVE_JUDGMENT, 0) and v.get(NEGATIVE_JUDGMENT, 0))

    def verdict(self, cluster: int, rule: str) -> Optional[str]:
        """The cluster's human verdict under one tie-break rule, or None to fall
        back to geometry.

        `rule` is REQUIRED -- there is no default, deliberately. A caller that
        wants "the" verdict has to say which convention it means, and geo_prf
        answers by asking for all three. See VERDICT_RULES.

        `positive_wins` treats a human `confirmed_real` as decisive: the cluster
        demonstrably contains a real target. `hard_negative_wins` is the
        opposite, and the literal reading of the original FP clause. `majority`
        lets the weight of evidence decide, with ties falling back to geometry.
        On the full 2,120-frame sample flight these span ~14 precision points across 24 conflicted
        clusters.
        """
        v = self.votes.get(cluster)
        if not v:
            return None
        pos, neg = v.get(POSITIVE_JUDGMENT, 0), v.get(NEGATIVE_JUDGMENT, 0)
        if not pos and not neg:
            return None            # only abstentions -- geometry stands
        if rule == "positive_wins":
            return POSITIVE_JUDGMENT if pos else NEGATIVE_JUDGMENT
        if rule == "hard_negative_wins":
            return NEGATIVE_JUDGMENT if neg else POSITIVE_JUDGMENT
        if rule == "majority":
            if pos == neg:
                return None        # tie -- geometry stands
            return POSITIVE_JUDGMENT if pos > neg else NEGATIVE_JUDGMENT
        raise ValueError(f"unknown verdict rule {rule!r} (expected one of {VERDICT_RULES})")


def collect_verdicts(scaffold: GeoScaffold, proj: Projector,
                     judged: Iterable[Tuple[int, float, float, str]],
                     frame_row_for) -> GeoVerdicts:
    """Fold the GUI's judgments onto geo clusters.

    `judged` yields (frame_idx, x, y, judgment) for every RAW member detection
    of every judged GUI cluster -- fanned out by the caller exactly as
    JudgmentStore.export() fans a centroid judgment out to its cluster members,
    so the two agree on what a judgment covers. `frame_row_for(frame_idx)`
    returns the pose row, or None.
    """
    out = GeoVerdicts()
    by_frame: Dict[int, List[Tuple[float, float, str]]] = {}
    for frame_idx, x, y, judgment in judged:
        if judgment in ABSTAIN_JUDGMENTS:
            continue
        by_frame.setdefault(int(frame_idx), []).append((float(x), float(y), judgment))
    for frame_idx, items in by_frame.items():
        out.n_judged_keys += len(items)
        row = frame_row_for(frame_idx)
        if row is None:
            out.n_unresolved += len(items)
            continue
        xy = np.array([[x, y] for x, y, _ in items], dtype=float)
        labs = locate(scaffold, proj, row, frame_idx, xy)
        for (_, _, judgment), lab in zip(items, labs):
            if lab < 0:
                out.n_unresolved += 1
            else:
                out.tally(int(lab), judgment)
    return out


@dataclass
class AddedMarkers:
    """Human-added markers ("added" judgments, placed by right-click), spatially
    deduplicated into `a` -- the observable part of the undiscovered-miss
    population.

    An added marker is a real target the detector NEVER FIRED ON. It is
    therefore not a detection, takes no part in the precision vote (see
    ABSTAIN_JUDGMENTS), and enters the accounting in exactly one place: the
    POST recall DENOMINATOR, as a false negative that the geotag survey also
    missed. It is the only term that constrains post_recall's upper bound from
    above -- see that function's docstring.
    """
    n_raw: int = 0              # clicks, before any filtering or dedup
    n_units: int = 0            # `a` for the all-geotags denominator
    n_units_in_view: int = 0    # `a` for the in-view polygon-test denominator
    n_near_geotag: int = 0      # dropped: already an FN geotag in the denominator
    n_in_cluster: int = 0       # dropped: the detector DID fire there
    n_unresolved: int = 0       # dropped: no pose / radalt gate / to_3d failure
    eps_group: float = 0.0

    def as_dict(self) -> dict:
        return {"n_added_raw": self.n_raw,
                "a_added_units": self.n_units,
                "a_added_units_in_view": self.n_units_in_view,
                "n_added_near_geotag": self.n_near_geotag,
                "n_added_in_cluster": self.n_in_cluster,
                "n_added_unresolved": self.n_unresolved,
                "added_group_eps_m": self.eps_group}


def collect_added(scaffold: GeoScaffold, proj: Optional[Projector],
                  added: Iterable[Tuple[int, float, float]],
                  frame_row_for,
                  eps_group: Optional[float] = None,
                  eps_predict: Optional[float] = None) -> AddedMarkers:
    """Fold human-added markers onto the map and count the distinct targets.

    `added` yields (frame_idx, x, y) PIXEL coordinates -- deliberately the same
    shape collect_verdicts takes minus the judgment, so the two folds read
    alike. `frame_row_for(frame_idx)` returns the pose row, or None.

    The four filters, in the order they are applied (so the buckets are
    disjoint and every raw click is accounted for exactly once):

    1. UNRESOLVED -- no pose, below the radalt gate, or to_3d dropped the row.
       A marker that cannot be placed on the ground cannot be counted.
    2. NEAR A GEOTAG (within eps_predict) -- that target is ALREADY in the
       recall denominator, and already counted as a false negative if no
       cluster reached it. Counting the click too would charge one miss twice.
       This is the single most important filter here.
    3. INSIDE A CLUSTER (within eps_predict of a DBSCAN core point, the same
       radius rule dbscan_predict implements) -- the detector DID fire there,
       so this is not an undiscovered miss. If the human also confirmed that
       cluster it is already counted in `c`.
    4. DEDUP -- what survives is grouped with DBSCAN(eps_group, min_samples=1).

    WHY DBSCAN AND NOT THE 3 CM GRID DEDUP build_scaffold USES
    ----------------------------------------------------------
    The grid works there only because an accumulated reprojection re-emits LITERALLY IDENTICAL
    accumulated 3D points (see the module docstring) -- which is exactly why
    recall survives it untouched. Two human clicks on the same burrow in frames
    700 and 830 are two INDEPENDENT projections whose disagreement is
    decimetres, so a grid would count one burrow dozens of times and inflate
    `a` catastrophically. eps_group defaults to the scaffold's own DBSCAN eps
    (0.20 m), so `a` and `c` share a spatial unit.

    min_samples=1 because a human click is authoritative evidence, unlike a
    detector point that needs density to be believed -- and with it there is no
    noise label, so every survivor lands in exactly one group.

    The in-view polygon test is applied to the PIXEL coordinates directly (no UTM round-trip, so
    no projection error). A click inside the raw frame but in inner_bound's
    margin sits where the detector was never allowed to fire; charging it as a
    miss would penalise the model for precisely the region recall_in_view
    exists to exclude. A group counts as in view if ANY member click is -- the
    same "seen in at least one frame" semantics geotags_in_view uses.
    """
    out = AddedMarkers()
    if eps_group is None:
        eps_group = float(scaffold.params.get("eps", DBSCAN_EPS))
    if eps_predict is None:
        eps_predict = float(scaffold.params.get("eps_predict", EPS_PREDICT_DETS))
    out.eps_group = float(eps_group)

    by_frame: Dict[int, List[Tuple[float, float]]] = {}
    for frame_idx, x, y in added:
        out.n_raw += 1
        by_frame.setdefault(int(frame_idx), []).append((float(x), float(y)))
    if not out.n_raw or not scaffold.ok or proj is None:
        out.n_unresolved = out.n_raw
        return out

    poly = inner_bound(proj.res)
    utm: List[np.ndarray] = []
    seen: List[np.ndarray] = []
    for frame_idx, pts in by_frame.items():
        xy = np.asarray(pts, dtype=float).reshape(-1, 2)
        p3 = _frame_to_utm(proj, frame_row_for(frame_idx), xy)
        if p3 is None:
            out.n_unresolved += len(xy)
            continue
        utm.append(p3[:, :2])
        seen.append(inside_vec(xy, poly))
    if not utm:
        return out
    pts_xy = np.vstack(utm)
    in_view = np.concatenate(seen)

    keep = np.ones(len(pts_xy), dtype=bool)
    if len(scaffold.geotags):
        nn = NearestNeighbors(radius=eps_predict).fit(scaffold.geotags[:, :2])
        near = np.array([len(i) > 0 for i in
                         nn.radius_neighbors(pts_xy, return_distance=False)])
        out.n_near_geotag = int(near.sum())
        keep &= ~near
    core_pts = scaffold.points[scaffold.core_mask][:, :2]
    if len(core_pts) and keep.any():
        nn = NearestNeighbors(radius=eps_predict).fit(core_pts)
        inc = np.zeros(len(pts_xy), dtype=bool)
        inc[keep] = np.array([len(i) > 0 for i in
                              nn.radius_neighbors(pts_xy[keep],
                                                  return_distance=False)])
        out.n_in_cluster = int(inc.sum())
        keep &= ~inc
    if not keep.any():
        return out

    groups = DBSCAN(eps=eps_group, min_samples=1).fit(pts_xy[keep]).labels_
    out.n_units = int(len(np.unique(groups)))
    kept_in_view = in_view[keep]
    out.n_units_in_view = int(len({int(g) for g, v in zip(groups, kept_in_view) if v}))
    return out


# ---------------------------------------------------------------------------
# The accounting
# ---------------------------------------------------------------------------

def _ratio(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def post_recall(n_hit: int, n_den: int, c: int, a: int) -> Tuple[int, int, float]:
    """(numerator, denominator, ratio) for the ADJUDICATED recall arm:

        R_POST = (n_hit + c) / (n_den + c + a)

    where `n_hit` is survey positives the detector found, `n_den` is survey
    positives, `c` is adjudicated positives that matched no survey unit, and
    `a` is deduplicated human-added markers.

    WHY THIS EXISTS AT ALL -- the arms cannot share one recall
    ----------------------------------------------------------
    POST precision defines a true positive as (human-adjudicated positive) OR
    (geometrically matched to a geotag). That definition ENLARGES the positive
    class beyond the geotag survey. Recall's denominator IS the positive class.
    So pinning POST recall to the PRE value would score the two arms under two
    different definitions of "true
    positive", which is not a confusion matrix. Both `c` and `a` therefore have
    to move it.

    IT IS AN UPPER BOUND, NOT A POINT ESTIMATE
    ------------------------------------------
    An unmarked real target is only DISCOVERABLE where the detector fired, so
    every unit in `c` is by construction a true positive and never a false
    negative: `c` is a censored sample of the omitted-positive population and
    can only raise the ratio. Writing the unobservable misses as `m`,

        R_true = (n_hit + c) / (n_den + c + a + m)   <=  R_POST,   m >= 0

    so R_POST is exactly the m == 0 case. Human-added markers (`a`) are the
    only observable slice of `m` and are the only term that pulls the bound
    back toward the truth. PRE recall stays the unbiased point estimate under
    the assumption that survey omission is independent of detectability -- see
    GEO_METRICS.md for why that assumption is defensible here (the geotags are
    a walking field survey, the detector reads aerial imagery, and the two
    miss-mechanisms are physically unrelated).

    Consequences worth knowing before writing a test against this:
      * R_POST == R_PRE  when c == 0 and a == 0
      * R_POST >= R_PRE  ONLY when a == 0. Once a reviewer places added
        markers R_POST can legitimately fall BELOW R_PRE -- those markers are
        misses the survey never recorded. An unconditional `>=` assertion is
        wrong the day someone right-clicks.
      * R_POST is non-decreasing in c, strictly decreasing in a, and <= 1.
    """
    num = int(n_hit) + int(c)
    den = int(n_den) + int(c) + int(a)
    return num, den, _ratio(num, den)


def geo_prf(scaffold: GeoScaffold, verdicts: Optional[GeoVerdicts] = None,
            added: Optional[AddedMarkers] = None) -> dict:
    """Both arms -- precision AND recall -- with POST computed under ALL THREE
    tie-break rules. Returns a plain dict so it round-trips through JSON and
    smoke tests can assert on exact integers.

    THERE IS NO PRIMARY RULE, BY DESIGN
    -----------------------------------
    Every rule is always reported. Which
    verdict wins in a conflicted cluster is decided by a TIE-BREAK RULE, NOT BY
    DATA, and on this dataset the choice is worth ~14 precision points. A single
    POST number therefore cannot
    be quoted honestly without its spread, and the surest way to stop it being
    quoted alone is to never compute it alone.

    `post` is consequently a dict keyed by rule, ordered most-positive first
    (VERDICT_RULES), and every display built on it shows all three.

    `added=None` means a == 0.

    No F1. Precision is over clusters and recall over geotags -- one cluster can
    absorb five geotags on this data -- so their harmonic mean is a category
    error, not a summary. The pixel-space HUD reports F1 because there both
    ratios share a per-detection denominator; here they do not.
    """
    if not scaffold.ok:
        return {"available": False, "reason": scaffold.unavailable}

    labels = np.unique(scaffold.labels)
    clusters = [int(c) for c in labels if c != -1]
    matched = set(int(c) for c in np.unique(scaffold.geotag_label) if c != -1)

    pre_tp = sum(1 for c in clusters if c in matched)
    pre_fp = len(clusters) - pre_tp

    verdicts = verdicts or GeoVerdicts()
    added = added or AddedMarkers()

    # Recall's PRE terms are geometry only and shared by every rule.
    hit = scaffold.geotag_label != -1
    n_geotags = int(len(scaffold.geotags))
    n_hit = int(hit.sum())
    in_view = scaffold.in_view
    n_in_view = int(in_view.sum())
    n_hit_in_view = int((hit & in_view).sum())

    post: Dict[str, dict] = {}
    recall_post: Dict[str, dict] = {}
    for rule in VERDICT_RULES:
        post_tp = 0
        # `n_discovered` (c in the docs) -- clusters a human confirmed that NO
        # geotag reached: real targets the survey omitted. Counted in the same
        # pass as post_tp because it is the same test with one extra clause.
        # Note c_in_view == c by construction: a cluster exists only because the
        # detector fired on it in some processed frame, so it was in view.
        #
        # c depends on the rule, so POST RECALL HAS A SPREAD TOO -- which the
        # old sensitivity() missed entirely, reporting only precision.
        n_discovered = 0
        for c in clusters:
            v = verdicts.verdict(c, rule)
            if v == POSITIVE_JUDGMENT:
                post_tp += 1
                if c not in matched:
                    n_discovered += 1
            elif v == NEGATIVE_JUDGMENT:
                pass                   # forced FP regardless of geometry
            elif c in matched:
                post_tp += 1
        post[rule] = {"tp": post_tp, "fp": len(clusters) - post_tp,
                      "precision": _ratio(post_tp, len(clusters))}

        r_all = post_recall(n_hit, n_geotags, n_discovered, added.n_units)
        r_view = post_recall(n_hit_in_view, n_in_view, n_discovered,
                             added.n_units_in_view)
        recall_post[rule] = {
            "is_bound": True,
            "c_confirmed_unmatched_clusters": n_discovered,
            "tp_geotags": r_all[0], "n_geotags": r_all[1], "recall_all_bound": r_all[2],
            "tp_geotags_in_view": r_view[0], "n_in_view": r_view[1],
            "recall_in_view_bound": r_view[2],
        }

    # A rule cannot change WHETHER a cluster was judged, only which way -- so
    # coverage is rule-independent and any rule answers it.
    judged_clusters = [c for c in clusters
                       if verdicts.verdict(c, VERDICT_RULES[0]) is not None]
    under_min = sum(1 for c in clusters
                    if scaffold.cluster_frames.get(c, 0) < scaffold.params.get("min_samples", 0))

    return {
        "available": True,
        "verdict_rules": list(VERDICT_RULES),
        "n_clusters": len(clusters),
        "n_noise_points": scaffold.n_noise,
        "n_points": int(len(scaffold.points)),
        "pre": {"tp": pre_tp, "fp": pre_fp, "precision": _ratio(pre_tp, len(clusters))},
        # Keyed by rule, ordered most-positive first. There is no "the" POST
        # number; the spread across these three IS the result.
        "post": post,
        "recall": {
            # -- PRE. These keys keep their meaning and values: they are the
            # geometry-only arm, and geo_metrics.json consumers that predate the
            # POST arm still read them correctly. Do not repoint them at the
            # adjudicated numbers.
            "arm": "pre",
            "tp_geotags": n_hit,
            "fn_geotags": n_geotags - n_hit,
            "n_geotags": n_geotags,
            "n_in_view": n_in_view,
            "tp_geotags_in_view": n_hit_in_view,
            "recall_all": _ratio(n_hit, n_geotags),
            "recall_in_view": _ratio(n_hit_in_view, n_in_view),
            "note": "PRE recall: geometry only. The unbiased point estimate "
                    "under the assumption that survey omission is independent "
                    "of detectability. See recall.post for the adjudicated arm.",
            # Keyed by rule, like `post` above and for the same reason: c is a
            # count of confirmed clusters, so it moves with the tie-break too.
            "post": recall_post,
            "post_common": {
                "is_bound": True,
                **added.as_dict(),
                "note": "UPPER BOUND on recall, not a point estimate. POST "
                        "precision defines TP as (human-adjudicated positive) "
                        "OR (geometrically matched), which enlarges the "
                        "positive class, so the recall denominator must grow "
                        "with it -- the two arms cannot share one recall. c is "
                        "censored: unlabelled positives are discoverable only "
                        "where the detector fired, so c can only raise the "
                        "ratio. a (added markers) is the only observable part "
                        "of the undiscovered-miss population and pulls it back "
                        "down. With a == 0 this is a bound and nothing more.",
                "unit_caveat": "c counts CLUSTERS, recall counts GEOTAGS, and "
                               "one cluster can hold five geotags -- so c "
                               "under-counts discovered burrows and the bound "
                               "is conservative in that direction. a groups "
                               "clicks at eps and can merge two adjacent added "
                               "burrows, which is anti-conservative. The two "
                               "do not cancel.",
            },
        },
        "adjudication": {
            "n_clusters_judged": len(judged_clusters),
            "coverage": _ratio(len(judged_clusters), len(clusters)),
            "n_conflicted": verdicts.n_conflicted,
            "n_judged_points": verdicts.n_judged_keys,
            "n_unresolved_points": verdicts.n_unresolved,
        },
        "diagnostics": {
            "n_clusters_under_min_frames": under_min,
            "note": "min_samples thresholds POINTS, not distinct frames; on this "
                    "input the two have come apart -- see module docstring",
        },
        "params": dict(scaffold.params),
    }


def post_spread(summary: dict, key: str = "precision") -> Tuple[float, float]:
    """(min, max) of a POST quantity across the three tie-break rules.

    The honest error bar on the adjudicated number. `key` is "precision" (from
    `post`) or "recall_all_bound"/"recall_in_view_bound" (from `recall.post`).
    Replaces the old sensitivity() helper, which computed precision only and
    treated the other two rules as a side-channel to the "real" answer.
    """
    src = summary["post"] if key == "precision" else summary["recall"]["post"]
    vals = [src[r][key] for r in summary["verdict_rules"]]
    return min(vals), max(vals)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

DEFINITIONS = {
    "unit_precision": "detection cluster",
    "unit_recall": "geotag",
    "pre": {
        "tp": "cluster with >=1 geotag within eps_predict of a cluster core point",
        "fp": "cluster with no such geotag",
        "fn": "geotag assigned to no cluster",
    },
    "post": {
        "tp": "PRE TP, or adjudicated confirmed_real",
        "fp": "PRE FP, or adjudicated hard_negative",
        "override": "human verdict overrides geometry in both directions",
        "conflict": "a cluster holding both verdicts has NO single answer: the "
                    "result is reported under all three tie-break rules, always. "
                    "There is no default and no way to select one -- the choice "
                    "is a convention, not data, and is worth ~14 precision "
                    "points here, so the spread IS the result",
        "excluded": "uncertain, removed, unreviewed and added are excluded from "
                    "the vote; a cluster with only those falls back to PRE "
                    "geometry. `added` abstains because an added marker is not "
                    "a detection at all -- it is consumed by collect_added for "
                    "the POST recall denominator instead",
    },
    "recall": {
        "pre": "n_hit / n_den -- geometry only. The unbiased "
               "point estimate under the assumption that survey omission is "
               "independent of detectability",
        "post": "(n_hit + c) / (n_den + c + a); c = adjudicated-positive "
                "clusters matching no geotag, a = spatially deduplicated "
                "human-added markers. Reported per tie-break rule: c counts "
                "confirmed clusters, so POST RECALL HAS A SPREAD TOO",
        "post_is_a_bound": "with a == 0 this is a strict UPPER BOUND on recall, "
                           "not a point estimate: unlabelled positives are "
                           "discoverable only where the detector fired",
        "why_two": "POST precision redefines TP to include adjudicated "
                   "positives, which enlarges the positive class; the recall "
                   "denominator must grow with it, or the two arms score under "
                   "two different definitions of 'true positive'",
        "denominators": "each arm is reported over all geotags and over geotags "
                        "in view under the in-view polygon test (image bounds "
                        "inset by the 81.5 px projection-noise buffer)",
    },
    "f1": "not reported: precision and recall have different denominators "
          "(clusters vs geotags), so their harmonic mean is a category error",
}


def write_json(path: Path, summary: dict, verdicts: Optional[GeoVerdicts] = None,
               scaffold: Optional[GeoScaffold] = None,
               extra: Optional[dict] = None) -> Path:
    """Persist the metrics. Anything this module puts on screen must also land
    on disk, so every figure can be cited and regenerated later."""
    doc = dict(summary)
    doc["definitions"] = DEFINITIONS
    # No sensitivity side-channel any more: `post` and `recall.post` are already
    # keyed by tie-break rule, so the spread is in the primary result rather
    # than in an appendix a reader could skip.
    if extra:
        doc.update(extra)
    path = Path(path)
    path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    return path
