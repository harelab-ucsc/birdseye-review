# SPDX-License-Identifier: MIT
# Copyright (c) 2026 harelab-ucsc
"""
Interactive Detection Review GUI
=================================

A human-in-the-loop review tool for a UAV flight's detections. For each flight
directory (a `labels.txt` ground truth, reviewed against a run's
per-checkpoint `dets/` detection files), this shows every frame with its
labels (context) and detections overlaid, distinguishing detections that
already matched a label (context, already-confirmed-real) from *unmatched*
detections -- the ambiguous pool that's either a valid target the annotator
didn't mark, or the model hallucinating on a tricky texture.

Nearby unmatched detections are pre-clustered (complete-linkage,
--cluster-radius, default 30px -- every pairwise distance within a cluster
is bounded by the radius) so a single physical mound's worth of detections
is one click, not dozens: a solid, judgment-colored centroid marker stands
for the cluster (a "xN" label marks multi-member clusters). Click a centroid
to select it (opens a magnified inset + info panel), then press a key to
judge it -- the judgment fans back out to every raw detection in that
cluster on export:
    c = confirmed real (agrees with the model -- an unmarked but valid target)
    h = hard negative   (the model fell for a tricky texture -- a real FP)
    u = uncertain        (don't know -- logged but not exported)
    r = reset to unreviewed
Right-click empty ground to add a target the model missed entirely.

`[o]`/`[m]`/`[a]` are the LABEL PROPAGATOR. A geo cluster (DBSCAN at 0.20 m in
UTM, built from every frame's detections) spans frames by construction, so a
verdict you give in one frame says something about the same physical spot in
every other frame that saw it. `[o]` chooses the tie-break convention for a
cluster holding both verdicts -- off, `positive_wins`, `majority`,
`hard_negative_wins` -- and starts OFF, because that choice is worth ~15
precision points and nobody should be shown a convention they did not pick.
`[m]` rings each cluster with its candidate label; no ring means nothing
propagates. `[a]` commits the frame's candidates: it never overwrites a
judgment you already made, SKIPS a cluster with no propagated label rather
than guessing, needs two presses, and is undone wholesale by `[z]`.

Everything it writes carries `"source": "human"` -- every input is a human
verdict and no model is in the loop -- plus a `"propagated"` block naming the
rule, the geo cluster and the votes, so which convention produced a label is
always recoverable.

`[p]` shows this session's statistics (pixel and geospatial PRE/POST per
model, run parameters, definitions); `[x]` exports the same content as a PDF.
`[?]` lists every key and mouse gesture.

`[v]` cycles an interactive 3D view of the flight -- off, inline, its own
window, off -- showing the camera track, the current frame's pose and ground
footprint, this frame's detections back-projected to the ground plane, and the
geotag survey. `[d]` additionally overlays the WHOLE directory's detections,
each back-projected through its own frame's pose, one palette colour per
model layer, with the ones landing within 0.35m of a survey geotag drawn
heavier. Drag to orbit, right-drag to pan, ctrl-drag or `+`/`-` to zoom,
`[g]` to switch between the whole-survey view and following the aircraft.
Inline draws it into the dead band under the frame image (space
`canvas_h = max(window_h, panel_w)` already left black; the image is never
resized), and falls back to the separate window when the window's aspect
leaves no usable band. During `[5]`/`[6]` playback the 3D camera follows the
aircraft and restores your framing afterwards -- without that the motion is
sub-pixel at survey zoom. It needs a `flight_data.db` in the flight directory;
without one `[v]` says so and changes nothing. See pose3d.py -- it is imported
lazily, so nothing about it can affect this tool's startup.

Outputs. Every commit is appended to `review_log.jsonl` (the append-only
source of truth), and immediately re-exported as derived sidecar files in the
same 3-column format as `labels.txt` and the run's `dets/` files
(`<frame_idx> <image_path> <x>,<y>,<value>`), directly loadable by
`parse_annotations` or a `load_labels`-style training loader:
    hard_negatives.txt        -- confirmed hard negatives (the "penalty" signal)
    confirmed_detections.txt  -- confirmed-real unmatched detections
    added_targets.txt         -- targets the reviewer added by right-click
    by_layer/                 -- the same sidecars split per checkpoint
    geo_metrics.json          -- the geospatial precision/recall on screen
    review_report.pdf         -- the [x] export
All of them go under `<run>/review/<flight_slug>/`. This tool never mutates
`labels.txt` or the run's `dets/` files. Matching is done live, per frame,
by `match_spatial_many_to_one` (pure coordinates, no image I/O): several
nearby detections of the same real target may all legitimately count as
matched, rather than only the single nearest one "winning" an exclusive
assignment and the rest falling into the unmatched review pool -- see
match_spatial_many_to_one's docstring.

`labels.txt` hardcodes confidence to 1.0; the run's `dets/` files carry real
confidences, which is what the "original_confidence"/value field records.

Usage (from the dataset root, e.g. the extracted sample):
    birdseye-review 2025_04_16/pieranch_rect results/17_holdout_pieranch
    birdseye-review --base-dir /path/to/dataset 2025_04_16/pieranch_rect results/17_holdout_pieranch
    birdseye-review --dry-run 2025_04_16/pieranch_rect results/17_holdout_pieranch
"""

import argparse
import csv
import functools
import hashlib
import json
import math
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

from .detections import (
    Detection, FrameData, parse_annotations, build_file_index, load_image,
    match_spatial_many_to_one, match_spatial_many_to_one_layered,
)

# Mirrored from geo_eval.py/geo_metrics.py so build_arg_parser() -- which the
# smoke tests call -- does not have to import sklearn. Asserted equal to the
# originals in smoke_test_gui.check_geo_metrics, so they cannot drift.
_GEO_EPS = 0.20
_GEO_MIN_SAMPLES = 10
_GEO_EPS_PREDICT = 0.35
_GEO_GUI_DEDUP_CM = 3.0
# Tie-break rule abbreviations, most-positive first, mirroring
# geo_metrics.VERDICT_RULE_ABBR. Needed at module scope because hud_rows()
# builds one POST row per rule for the union block AND for every model group,
# and the pending state has to emit the same number of rows as the ready one
# -- there is no "primary" rule to draw just one of.
_GEO_RULE_ABBRS = ("pos", "maj", "neg")
# The rule NAMES, most-positive first, mirroring geo_metrics.VERDICT_RULES.
# Needed at module scope for the same reason as the abbreviations: [o] cycles
# them and hud_columns_clear() measures the longest one, both of which happen
# before geo_metrics (and sklearn behind it) is imported. Asserted equal to
# the original in smoke_test_gui.check_geo_metrics.
_GEO_VERDICT_RULES = ("positive_wins", "majority", "hard_negative_wins")
# The detector's own heatmap post-processing, mirrored from run_detector.py so
# [p] and the PDF can state how the points on screen were produced without
# importing it (it pulls in tensorflow). Asserted equal to the originals in
# smoke_test_gui, so they cannot drift.
_DETECTOR_DEFAULT_THRESH = 0.03
_DETECTOR_MIN_AREA = 10

# How often render() re-lists the run's dets/ to notice a checkpoint that
# finished mid-session -- a directory listing on every ~25ms render tick is
# real syscall traffic for a file that appears at most once every few minutes.
LAYER_POLL_SECONDS = 2.0

WINDOW = "Detection Review"
TRACKBAR_SPATIAL = "Spatial Thresh (px)"
TRACKBAR_CLUSTER = "Cluster Radius (px)"
# How long a slider's value must sit still before it triggers a recompute.
# The worst real directory takes ~13-16s to re-match+re-cluster (a fully
# blocking call) -- a short window would let one continuous drag gesture
# queue a pileup of these, effectively freezing the app; 0.6s costs a
# slightly-less-than-instant feel on cheap directories to protect the
# expensive ones from that.
SLIDER_DEBOUNCE_SECONDS = 0.6
# Wall-clock budget of actual matching+clustering work per render() tick
# while a chunked recompute is in flight -- bounds how choppy the live
# progress banner gets (a denser chunk of frames just yields sooner) without
# meaningfully slowing the total time to finish versus running it in one
# blocking call.
CHUNK_TIME_BUDGET_SECONDS = 0.03
# Lines scrolled per j/k press on the [?] help / [p] stats overlays (see
# App._draw_paginated). Plain 'j'/'k', not arrow keys: on at least one real
# OpenCV highgui backend (the same one behind the [m]/[M] key-collision fix
# above), special keys like arrows are reported inconsistently across
# platforms/backends, while plain ASCII letters are not.
SCROLL_STEP_LINES = 5
# Video play/rewind ([5]/[6]/[f]): wall-clock seconds per frame-advance
# while playing, cycled by [f]. Same time.monotonic()-driven-per-render()-tick
# idiom as SLIDER_DEBOUNCE_SECONDS/CHUNK_TIME_BUDGET_SECONDS above -- see
# App._play_tick().
PLAY_INTERVALS_SECONDS = (0.5, 0.25, 0.12, 0.06)

# [a] bulk-accept: how long the first press stays armed before the confirm
# expires. Long enough to read the preview, short enough that an [a] pressed
# and forgotten can't be completed by an unrelated [a] minutes later.
ACCEPT_ARM_SECONDS = 6.0
# Value of the "source" field in review_log.jsonl. A record without the field
# means SOURCE_HUMAN, which is what judgment_source() below encodes.
#
# Nothing in this tool writes SOURCE_CLASSIFIER. It marks rows an automated
# classifier wrote, where adjudicated precision computed over them would be
# the model grading its own output. The constant exists because
# review_log.jsonl is APPEND-ONLY and may carry such rows, which have to read
# back correctly and be identifiable as machine calls. [a] propagates HUMAN
# verdicts and stamps SOURCE_HUMAN -- see accept_propagated_frame().
SOURCE_HUMAN = "human"
SOURCE_CLASSIFIER = "classifier"


def judgment_source(rec: Optional[dict]) -> str:
    """A judgment record's provenance, defaulting to SOURCE_HUMAN for a
    record without a "source" field."""
    if not rec:
        return SOURCE_HUMAN
    return rec.get("source", SOURCE_HUMAN)


# Zoom-inset magnification. At 3x the inset is never drawn at its nominal
# 600px -- _draw_panel's fixed bottom budget clamps it to ~390px on a
# 1000px-tall window, i.e. an effective ~1.95x. See also
# PANEL_MAX_SCREEN_FRACTION / fit_zoom_scale(), which walk this back down on a
# monitor too narrow to spare the panel width.
DEFAULT_ZOOM_SCALE = 4
# The HUD's height is DERIVED, not written down -- see PANEL_HUD_H and
# PANEL_HUD_MIN_H further down, which cannot be defined until HUD_ADV and the
# group-height helpers exist.
#
# There is no fixed HUD height ceiling: the HUD carries a PER-MODEL block --
# PIXEL PRE/POST plus GEO PRE and one POST row per tie-break rule, for every
# active layer -- so there is no worst case to bound: thirty checkpoints are
# 30 x 118px of rows and the body scrolls. What a ceiling would stand in for
# is asserted directly instead: that the zoom inset stays usable (by
# panel_layout()'s probe at the shortest supported panel) and that the metric
# table stays readable (by hud_columns_clear() at PANEL_FLOOR_W).
# ONE hint line: content a reader consults once and then knows lives in [?],
# and this line promises that [?] is complete: so _help_lines must actually
# be complete, and its FIRST PAGE must carry the keys used hourly.
# smoke_test_gui.check_keybinding_docs asserts exactly that.
PANEL_HINT_LINE = "[?] all keys & mouse gestures        [q] save + quit"
# What the info block says when nothing is selected. A LIST, so it goes through
# panel_layout's info band and its row bound like any other info content --
# see _draw_panel.
PANEL_IDLE_HINT = (
    "click an orange point, or press",
    "[space] for the next unreviewed one.",
    "right-click empty ground to add a",
    "target the model missed entirely.",
)
PANEL_HINT_H = 26
# How much of [?] is readable without scrolling. _help_lines is ~160 lines at
# line_h=26, i.e. five pages on a 1000px window; anything a reviewer needs
# mid-review has to be inside this prefix.
PANEL_HELP_QUICK_LINES = 30
PANEL_INFO_LINE_H = 20    # selected-point info block line pitch
# Inline 3D view (see App.inline_pose_rect). The band under the frame image
# exists because canvas_h = max(window_h, panel_w) while the image is
# letterboxed to 1920x1200 -- the 820px panel leaves dead space below it.
# _draw_status writes two baselines at canvas_h-12 and canvas_h-32,
# so the bottom 48px stay reserved for it.
POSE_STATUS_RESERVE_H = 48
POSE_INLINE_DIVIDER_H = 2
# Under this the strip is not worth having: the 3D HUD alone needs ~66px and
# the scene would get less than the frustum is tall. [v] opens the separate
# window instead rather than presenting a squashed band.
MIN_INLINE_POSE_H = 180
# The side panel may never eat more than this fraction of the screen width --
# past it the frame view, which is the thing being reviewed, starts to starve.
PANEL_MAX_SCREEN_FRACTION = 0.5
# ...but the fit never reduces below this: under 3x the inset is too small to
# judge a point from.
MIN_FITTED_ZOOM_SCALE = 3


def fit_text(text: str, max_px: int, font_scale: float,
             font: int = cv2.FONT_HERSHEY_SIMPLEX, thickness: int = 1) -> str:
    """Truncate `text` to what actually fits in `max_px`, measured with
    cv2.getTextSize rather than guessed from a character count, so a fixed
    slice neither chops the tail off HUD lines nor leaves half the panel
    blank. Returns the text as-is when it already fits."""
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


# The narrowest side panel this tool ever renders. Named because the HUD's
# readability guard is asserted at exactly this width -- below it the panel
# cannot render its own contents at all (see fit_zoom_scale).
PANEL_FLOOR_W = 420


def compute_panel_width(zoom_crop_px: int, zoom_scale: int) -> int:
    return max(zoom_crop_px * zoom_scale + 20, PANEL_FLOOR_W)


def fit_zoom_scale(zoom_crop_px: int, zoom_scale: int, screen_w: Optional[int]) -> int:
    """Largest scale <= zoom_scale whose panel fits PANEL_MAX_SCREEN_FRACTION
    of the screen. Pure arithmetic (screen_w injected, not queried) so it is
    directly testable; returns zoom_scale as-is when the screen size is
    unknown, and never goes below MIN_FITTED_ZOOM_SCALE (nor below the
    requested scale, if that is already lower). Note the 420px floor in
    compute_panel_width() means a small enough screen can still end up with a
    panel over the fraction -- that floor wins, deliberately: below it the
    panel cannot render its own contents at all."""
    if not screen_w or screen_w <= 0:
        return max(1, zoom_scale)
    floor = min(MIN_FITTED_ZOOM_SCALE, max(1, zoom_scale))
    budget = int(screen_w * PANEL_MAX_SCREEN_FRACTION)
    scale = max(1, zoom_scale)
    while scale > floor and compute_panel_width(zoom_crop_px, scale) > budget:
        scale -= 1
    return scale


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """precision, recall, f1 -- zero-division-safe: reports 0.0 rather than
    raising when a count is 0."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def corrected_prf(tp0: int, fp0: int, fn0: int, tp1: int, fp1: int,
                  n_confirmed: int = 0, n_added: int = 0) -> Tuple[float, float, float]:
    """Precision/recall/F1 for the *post-adjudication* estimate -- scored
    against the POST arm's OWN positive class, not the naive one.

        precision = tp1 / (tp1 + fp1)
        recall    = (tp0 + n_confirmed) / (tp0 + fn0 + n_confirmed + n_added)

    Recall is NOT pinned to the naive value. POST precision defines a true
    positive as (human-adjudicated positive) OR (matched to a label), which
    *enlarges the positive class*; recall's denominator IS the positive class.
    Holding it at the naive value would score the two arms under two different
    definitions of "true positive", which is not a confusion matrix.

    `n_confirmed` is corrected_counts()'s confirmed-real raw-detection count --
    unmatched by construction, since those centroids come out of
    `unmatched_clusters_by_frame` -- and is the `c` term. `n_added` is
    spawned_total(), the `a` term.

    THE RETURNED RECALL IS AN UPPER BOUND, NOT A POINT ESTIMATE, and `n_added
    == 0` is the case where it is *only* a bound: an unlabelled positive is
    discoverable only where the detector fired, so `n_confirmed` can never
    correspond to a false negative and can only raise the ratio. Added markers
    are the one observable slice of the misses nobody recorded, and the only
    term that pulls the bound back down -- so once `n_added > 0` this can
    legitimately fall BELOW the naive recall. PRE recall (`prf(tp0, fp0, fn0)`)
    stays the unbiased point estimate under the assumption that annotation
    omission is independent of detectability. Report the pair; F1 here inherits
    the bound for the same reason recall does.

    The function exists so POST recall is never re-pinned to PRE: feeding tp1
    into a generic prf() would silently drop `n_added` from the denominator.

    No pixel-space dedup, deliberately. labels.txt is per-frame and
    naive_counts() sums per-frame, so one burrow legitimately contributes one
    label per frame and a spawned point is exactly one per-frame missing label
    -- the units already line up. Geo space does need dedup, because there a
    burrow is one UTM target however many frames saw it; see
    geo_metrics.collect_added. Kept in sync with geo_metrics.post_recall by an
    explicit cross-space assertion in smoke_test_gui (review_gui cannot import
    geo_metrics at module scope, so the formula is necessarily duplicated)."""
    precision = tp1 / (tp1 + fp1) if (tp1 + fp1) else 0.0
    num = tp0 + n_confirmed
    den = tp0 + fn0 + n_confirmed + n_added
    recall = num / den if den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def judgment_color(judgment: str) -> Tuple[int, int, int]:
    """COLOR lookup that cannot raise. Judgment strings reach the render
    loop straight from review_log.jsonl, which other tools may also write --
    an unknown value would otherwise be a KeyError on every render tick, i.e.
    a dead GUI. The lookup itself is total instead of chasing each new name."""
    return COLOR.get(judgment, COLOR["unreviewed"])


class _GeoImportFailed:
    """Stand-in for a GeoScaffold when geo_metrics itself could not be
    imported. Keeps `.ok` / `.unavailable` valid for every caller so an
    optional metric's absence never becomes an AttributeError in the render
    loop."""

    def __init__(self, reason: str) -> None:
        self.unavailable = reason

    @property
    def ok(self) -> bool:
        return False


def hud_match_params_line(spatial_threshold: float, cluster_radius: float) -> str:
    """Pure text-builder for the live HUD's matching-parameters line -- a
    standalone function (not inlined in the cv2.putText call) so a test can
    assert on the exact string instead of scraping pixels, mirroring
    _selected_point_info_lines()/_stats_overlay_lines()'s existing split.
    Surfaces exactly the two live values that decide the matched/unmatched
    split and the review unit (see App._apply_slider_recompute -- these are
    also the same two values the sliders control and the [p] overlay's
    PARAMS section echoes as its 'live:' line), plus a reminder that
    matching here is many-to-one, not an exclusive one-to-one assignment (see
    REVIEW_GUI.md's "Matching" section) -- the HUD is the one place PRE/POST
    are shown, so the parameters that produced them belong right next to
    them, not buried in a flag only visible at startup."""
    return f"match: spatial<={spatial_threshold:.0f}px  cluster<={cluster_radius:.0f}px  (many-to-one)"


def hud_params_line(spatial_threshold: float, cluster_radius: float,
                    geo_eps: float, geo_match: float,
                    prop_rule: Optional[str]) -> str:
    """The pinned params rule row's text. Pure builder, same convention as
    hud_match_params_line/play_status_line, so a test asserts the exact string
    rather than scraping pixels -- and so hud_columns_clear() can measure the
    real worst case instead of an approximation of it.

    ONLY GLOBAL PARAMETERS BELONG HERE. `eps` and `match` come straight from
    the flags and describe geometry, so they are the same for every model.
    `cov` and `a` are NOT here: both are adjudication-dependent and therefore
    PER MODEL (a layer's coverage is structurally lower than the union's --
    see LayerGeoSummary.n_noise_points), so one number here would be wrong for
    every model. They live in the per-model groups and in [p].

    `prop` is the [o] propagation rule, or `off`. It sits with the parameters
    because it is live state that decides what [m] paints and what [a] would
    write, not a caveat about something already done."""
    return (f"m2o {spatial_threshold:.0f}/{cluster_radius:.0f}px"
            f"  geo eps {geo_eps:.2f}m  match {geo_match:.2f}m"
            f"  prop:{prop_rule or 'off'}")






def play_status_line(play_direction: int, play_speed_idx: int) -> str:
    """Pure text-builder for the live HUD's video play/rewind indicator --
    same rationale as hud_match_params_line above. Empty string when
    stopped (drawn as a blank line so the HUD's height stays constant
    whether or not play is active, rather than the panel layout jumping).
    Persistent in the HUD rather than set_status(), which expires after a
    few seconds -- play state needs to stay visible for as long as it's
    actually running."""
    if play_direction == 0:
        return ""
    fps = 1.0 / PLAY_INTERVALS_SECONDS[play_speed_idx]
    arrow, verb = (">", "playing") if play_direction == 1 else ("<", "rewinding")
    return f"{arrow} {verb} ({fps:.1f} fps)"


JUDGMENTS = ("unreviewed", "confirmed_real", "hard_negative", "uncertain", "added", "removed")
# judgment -> exported sidecar filename. "added" is a human-spawned point
# (the model never detected it at all -- a pure recall-side signal, distinct
# from confirmed_real which recovers a detection the model already made).
# "unreviewed"/"uncertain"/"removed" are logged but never exported.
EXPORTABLE = {
    "confirmed_real": "confirmed_detections.txt",
    "hard_negative": "hard_negatives.txt",
    "added": "added_targets.txt",
}

COLOR = {
    "label": (255, 255, 0),        # cyan ring, context
    "matched": (215, 215, 215),     # light gray ring, context (bumped from a dim 140,140,140 -- was easy to miss against grass)
    "unreviewed": (0, 140, 255),   # orange filled, interactive
    "confirmed_real": (0, 200, 0), # green filled
    "hard_negative": (0, 0, 255),  # red filled
    "uncertain": (200, 0, 200),    # purple filled
    "spawned": (255, 255, 255),    # white star: a human-added point, model never detected it
    # "added"/"removed" (JUDGMENTS) are only ever committed against spawned-
    # point keys, drawn separately via cv2.drawMarker, never through
    # COLOR[j] -- but a cluster centroid's key CAN collide with a spawned/
    # removed point's exact coordinate at cluster_radius<=0 (verified: a
    # spawn placed on top of a then-matched detection, later reclassified
    # unmatched by a spatial-threshold change, collides exactly). These
    # entries exist purely so that narrow, real collision degrades to a
    # sensible fallback color instead of a KeyError, without needing every
    # COLOR[j] call site to special-case it.
    "added": (255, 255, 255),      # same as "spawned" -- semantically the same judgment
    "removed": (215, 215, 215),    # same as "matched" -- not an active review target
    "selected": (0, 255, 255),     # yellow ring, drawn last
    "text": (255, 255, 255),
    "panel_bg": (30, 30, 30),
    "error": (0, 0, 255),
    "hud_bg": (20, 20, 20),
    "hud_border": (90, 90, 90),
    "bar_frame": (0, 200, 255),     # amber-ish: frame-position progress bar
    "bar_review": (0, 200, 0),      # green: unmatched-point review-coverage bar
    "pre": (140, 140, 255),         # muted red-ish: pre-adjudication (naive) stats
    "post": (140, 255, 140),        # muted green: post-adjudication (corrected) stats
    # Cyan: the geospatial block is a THIRD metric space (UTM clusters vs
    # geotags), not a variant of the pixel-space PRE/POST above it, so it must
    # not borrow either of their colours.
    "geo": (230, 210, 120),
    # Magenta, deliberately unlike any judgment colour: the generic caveat-row
    # colour. The [m] ring is painted in the PROPAGATED JUDGMENT's own colour,
    # so a candidate and a commit speak one language.
    "caveat": (255, 120, 255),
}

# ---------------------------------------------------------------------------
# Per-layer palette.
#
# THE HUE CIRCLE IS ALREADY FULLY SPOKEN FOR. Every entry in COLOR above is
# load-bearing and semantic: green means confirmed_real, red means
# hard_negative, orange means unreviewed, and so on. The comment at COLOR's
# "geo" entry -- that the geospatial block must not borrow PRE/POST's colours
# because it is a different metric space -- is a SEMANTIC argument, and it
# applies here with full force: a layer marker that reads as "green" will be
# read as a confirmed judgment.
#
# So layer identity is carried on TWO channels that the judgment language does
# not use:
#
#   hue x shape, inside a lightness/chroma band no COLOR entry occupies.
#
# The band is high-lightness, low-chroma -- "chalk". Judgment colours are
# saturated (C* 55-100); these sit at C* 36, which reads as a tint rather than
# as a verdict even at a glance. Tier is then a THIRD, orthogonal channel
# (chroma x size x alpha), so "which model" and "accumulated vs this frame"
# never compete for the same visual dimension.
#
# The first eight hues are the Okabe-Ito colourblind-safe anchors. THAT SET
# CAPS AT EIGHT and this project routinely shows 30 layers, so beyond eight the
# shape channel -- not the hue -- is what carries identity, and the legend is
# the authority. The help text says exactly that rather than overclaiming.
# ---------------------------------------------------------------------------

LAYER_L, LAYER_C = 82.0, 36.0        # CIELAB L*, C* -- the chalk band
# Okabe-Ito hue angles (degrees, CIELAB), the colourblind-safe eight.
LAYER_HUES_CVD = (41.0, 202.0, 120.0, 60.0, 240.0, 27.0, 12.0, 300.0)
LAYER_SHAPES = ("disc", "square", "diamond", "tri_up")
# Minimum CIE76 distance from any semantic COLOR entry. Enforced by an
# assertion at import, not left as a comment -- the same discipline
# hud_columns_clear() uses for the HUD's column grid.
LAYER_PALETTE_MIN_DE = 22.0
# Minimum CIE76 between two layers THAT SHARE A SHAPE. Not between any two
# layers: 30 hues at a fixed chroma simply are not 12 deltaE apart anywhere on
# the circle (measured: the best achievable is 3.4), and asserting otherwise
# would be a bar this palette fails by arithmetic rather than by design. What
# identity actually rests on is the PAIR (hue, shape), so that is what is
# gated. Measured: 10.7 at 12 layers, 8.6 at 30.
#
# BE HONEST ABOUT WHAT THIS BUYS. Up to ~12 layers -- the holdout run's size --
# hue plus shape distinguishes layers on sight. At 30 -- the full screen -- it
# groups them; it does not uniquely identify them, and the legend is the
# authority. The help text says so, and layer_cvd_safe_count() reports how many
# of the Okabe-Ito anchors actually survived rather than claiming eight.
LAYER_PAIR_MIN_DE = 8.0
# A candidate hue is admitted only if its chalk-band colour actually clears
# LAYER_PALETTE_MIN_DE against every semantic COLOR entry -- MEASURED, not
# guessed from hue angles. The first version of this excluded a band around
# pre/post by eye and the gate below caught it immediately: in the chalk band
# the real nearest neighbours are `geo` (L*79 C*28) and `label` (L*91 C*50),
# not pre/post. Measuring instead of guessing also means a future COLOR entry
# reshapes the sweep automatically rather than silently colliding.
LAYER_HUE_STEP_DEG = 2.0
# Chrome, not judgments: the panel/HUD greys, plain text, and the two
# "not an active review target" greys. A layer tint reading as greyish
# misleads nobody, and excluding them is what leaves the circle usable at all.
_LAYER_DE_EXEMPT = frozenset({
    "text", "panel_bg", "hud_bg", "hud_border", "matched", "removed", "spawned", "added",
})
# Tier geometry / emphasis. The accumulation tier is a 1px scatter so that
# density reads as density; the current frame gets a real glyph.
# Marker size for the accumulation tier, in DISPLAY pixels. 1px at alpha 0.18
# was the first attempt and it was invisible on screen -- 8,000 single pixels
# at 18% over sunlit grass is not a layer, it is noise below the threshold of
# sight. Grown with one cv2.dilate over the whole buffer rather than a bigger
# scatter, so the cost is a function of canvas area, not point count.
LAYER_ACCUM_PX = 2
LAYER_FRAME_GLYPH_PX = 4
LAYER_ACCUM_ALPHA = 0.45
LAYER_FRAME_ALPHA = 0.85
# Radius of the per-layer "fan" around a shared spot, in display pixels.
# WHY A FAN. Every layer that fires at one physical spot lands on essentially
# the same pixel, and the consensus centroid (the interactive review target) is
# drawn over the top of all of them -- so stacked glyphs showed nothing at all
# and a layered frame looked identical to an unlayered one. Fanning each layer
# to a fixed clock position around the spot turns "which models agree here"
# into something readable at a glance, which is the whole question this screen
# exists to answer. The angle comes from the layer's INDEX, not its position
# among the visible ones, so a layer keeps its clock position as others are
# toggled and the arrangement stays learnable.
LAYER_FAN_R = 14
LAYER_SPARSE_ALPHA_BOOST = 1.6       # a sparse layer emits ~1 det/frame; without
                                     # this its accumulation tier is invisible
# Above this many VISIBLE layers the accumulation tier turns itself off: more
# than a handful of translucent hues over grass is not a picture anyone can
# read, and the limit protects readability by declining to draw rather than
# by drawing something else.
LAYER_ACCUM_PERLAYER_MAX = 6
# World-space grid cell for the accumulation tier's cull, in metres. 4 m is
# comfortably under a frame footprint, so a frame touches a handful of cells.
LAYER_GRID_M = 4.0


def draw_layer_glyph(img, org: Tuple[int, int], style: "LayerStyle", size: int,
                     color: Tuple[int, int, int]) -> None:
    """One layer marker: hue says which model, shape says which model too (the
    redundant channel that keeps 30 layers apart when hue alone cannot), and
    hollow/struck say the checkpoint is degenerate.

    A degenerate checkpoint is DRAWN, never dropped -- 19 of the 30 screened
    are degenerate, and that is the finding, not noise to filter."""
    x, y = org
    r = max(1, size // 2)
    th = 1 if style.hollow else -1
    if style.shape == "square":
        cv2.rectangle(img, (x - r, y - r), (x + r, y + r), color, th)
    elif style.shape == "diamond":
        pts = np.array([[x, y - r], [x + r, y], [x, y + r], [x - r, y]], np.int32)
        cv2.fillPoly(img, [pts], color) if th < 0 else cv2.polylines(img, [pts], True, color, 1)
    elif style.shape == "tri_up":
        pts = np.array([[x, y - r], [x + r, y + r], [x - r, y + r]], np.int32)
        cv2.fillPoly(img, [pts], color) if th < 0 else cv2.polylines(img, [pts], True, color, 1)
    else:
        cv2.circle(img, (x, y), r, color, th, cv2.LINE_AA)
    if style.strike:
        cv2.line(img, (x - r - 1, y), (x + r + 1, y), color, 1)


def _srgb_from_lch(L: float, C: float, h_deg: float) -> Tuple[int, int, int]:
    """CIELCh -> BGR uint8, D65. Plain arithmetic rather than a dependency:
    cv2.cvtColor's Lab is 8-bit-quantised and would blunt the deltaE gate the
    assertion below relies on."""
    h = math.radians(h_deg)
    a, b = C * math.cos(h), C * math.sin(h)
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0

    def finv(t: float) -> float:
        return t ** 3 if t > 6.0 / 29.0 else 3.0 * (6.0 / 29.0) ** 2 * (t - 4.0 / 29.0)

    X, Y, Z = 0.95047 * finv(fx), 1.00000 * finv(fy), 1.08883 * finv(fz)
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    bl = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def enc(u: float) -> int:
        u = max(0.0, min(1.0, u))
        u = 1.055 * (u ** (1 / 2.4)) - 0.055 if u > 0.0031308 else 12.92 * u
        return int(round(max(0.0, min(1.0, u)) * 255))

    return (enc(bl), enc(g), enc(r))          # BGR, as cv2 wants


def _bgr_to_lab(bgr: Sequence[int]) -> Tuple[float, float, float]:
    """BGR uint8 -> CIELAB, D65. Inverse of the above, same precision."""
    def dec(v: int) -> float:
        u = v / 255.0
        return ((u + 0.055) / 1.055) ** 2.4 if u > 0.04045 else u / 12.92

    b, g, r = (dec(int(c)) for c in bgr)
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b

    def f(t: float) -> float:
        return t ** (1 / 3) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29

    fx, fy, fz = f(X / 0.95047), f(Y / 1.0), f(Z / 1.08883)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def _delta_e(c1: Sequence[int], c2: Sequence[int]) -> float:
    """CIE76. Crude next to CIEDE2000, but this is a SEPARATION gate, and CIE76
    is the conservative direction: it under-reports differences involving
    chroma, so a palette that clears this bar clears CIEDE2000 too."""
    l1, a1, b1 = _bgr_to_lab(c1)
    l2, a2, b2 = _bgr_to_lab(c2)
    return math.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


def _hue_admissible(h: float) -> bool:
    """Does the chalk-band colour at this hue clear every semantic COLOR entry?"""
    c = _srgb_from_lch(LAYER_L, LAYER_C, h)
    for name, sem in COLOR.items():
        if name in _LAYER_DE_EXEMPT:
            continue
        if _delta_e(c, sem) < LAYER_PALETTE_MIN_DE:
            return False
    return True


@functools.lru_cache(maxsize=None)
def _layer_hue_sequence(n: int) -> Tuple[float, ...]:
    """n hue angles, all admissible, spread as far apart as the survivors allow.

    Order: the Okabe-Ito colourblind-safe anchors THAT SURVIVE the admissibility
    filter, then farthest-point sampling over the rest of the circle. Some
    Okabe-Ito anchors do not survive -- 202 deg sits on top of COLOR["geo"] in
    this band -- which is precisely why the shape channel exists and why the
    help text points at the legend rather than promising eight distinguishable
    hues.

    PREFIX-STABLE in n (_layer_hue_sequence(30)[:8] == _layer_hue_sequence(8)),
    so a checkpoint finishing mid-session and appearing as a 12th layer never
    recolours the eleven already on screen."""
    out: List[float] = [h for h in LAYER_HUES_CVD if _hue_admissible(h)][:n]
    if len(out) < n:
        steps = int(round(360.0 / LAYER_HUE_STEP_DEG))
        pool = [i * LAYER_HUE_STEP_DEG for i in range(steps)]
        pool = [h for h in pool if _hue_admissible(h)]
        pool = [h for h in pool if all(abs((h - o + 180.0) % 360.0 - 180.0) > 1e-9 for o in out)]
        while len(out) < n and pool:
            best, best_d = None, -1.0
            for h in pool:
                d = min(abs((h - o + 180.0) % 360.0 - 180.0) for o in out) if out else 360.0
                if d > best_d:
                    best, best_d = h, d
            out.append(best)
            pool.remove(best)
    return tuple(out[:n])


def layer_cvd_safe_count() -> int:
    """How many of the Okabe-Ito anchors survived. Reported honestly in [?]
    rather than claiming eight unconditionally."""
    return sum(1 for h in LAYER_HUES_CVD if _hue_admissible(h))


def layer_palette(n: int) -> List[Tuple[int, int, int]]:
    """n chalk-band BGR colours, prefix-stable in n."""
    return [_srgb_from_lch(LAYER_L, LAYER_C, h) for h in _layer_hue_sequence(n)]


def layer_accent(bgr: Sequence[int]) -> Tuple[int, int, int]:
    """The current-frame tier's colour for a layer: same hue, chroma x1.9 and
    L* -18. Same identity, unmistakably nearer the front."""
    L, a, b = _bgr_to_lab(bgr)
    C = math.hypot(a, b) * 1.9
    h = math.degrees(math.atan2(b, a))
    return _srgb_from_lch(max(0.0, L - 18.0), C, h)


@dataclass(frozen=True)
class LayerStyle:
    index: int
    color: Tuple[int, int, int]        # accumulation tier (chalk)
    accent: Tuple[int, int, int]       # current-frame tier (saturated)
    shape: str
    hollow: bool                       # sparse or collapsed
    strike: bool                       # collapsed


def layer_style(layer_id: str, all_ids: Sequence[str], verdict: str = "healthy") -> LayerStyle:
    """This layer's visual identity.

    Keyed on the layer's position in the SORTED id list, not on load order, so
    a checkpoint that finishes mid-session and appears as a 12th layer does not
    recolour the eleven already on screen. Asserted in the smoke test.

    Health is a MODIFIER, never a hue -- a hue means "which model", and
    overloading it with "and this one is dead" would make the two unreadable
    together. Sparse draws hollow, collapsed draws hollow and struck through;
    both still draw, always."""
    ids = sorted(all_ids)
    i = ids.index(layer_id) if layer_id in ids else len(ids)
    pal = layer_palette(max(len(ids), i + 1))
    c = pal[i % len(pal)]
    return LayerStyle(index=i, color=c, accent=layer_accent(c),
                      shape=LAYER_SHAPES[(i // len(LAYER_HUES_CVD)) % len(LAYER_SHAPES)],
                      hollow=verdict in ("collapsed", "sparse"),
                      strike=verdict == "collapsed")


def _min_delta_e_vs_COLOR(palette: Sequence[Sequence[int]]) -> float:
    """Closest any palette entry comes to a semantic COLOR entry. The
    near-neutral entries (panel/HUD greys, text, matched) are excluded: they
    are chrome, never a judgment, and a layer tint being "greyish" misleads
    nobody."""
    worst = 1e9
    for c in palette:
        for name, sem in COLOR.items():
            if name in _LAYER_DE_EXEMPT:
                continue
            worst = min(worst, _delta_e(c, sem))
    return worst


# The gate, enforced rather than commented -- same discipline as the HUD
# budget assertion further down. If a future hue is added and this fires, MOVE
# THE HUE; do not lower the bar. A layer colour that reads as a judgment colour
# turns every screenshot of this tool into a misreading.
def _min_same_shape_delta_e(n: int) -> float:
    """Closest two layers sharing a shape come, over an n-layer palette."""
    pal = layer_palette(n)
    shapes = [LAYER_SHAPES[(i // len(LAYER_HUES_CVD)) % len(LAYER_SHAPES)] for i in range(n)]
    worst = 1e9
    for i in range(n):
        for j in range(i + 1, n):
            if shapes[i] == shapes[j]:
                worst = min(worst, _delta_e(pal[i], pal[j]))
    return worst if worst < 1e9 else 1e9


assert _min_delta_e_vs_COLOR(layer_palette(30)) >= LAYER_PALETTE_MIN_DE, (
    f"a layer colour collides with a semantic COLOR entry "
    f"(min deltaE {_min_delta_e_vs_COLOR(layer_palette(30)):.1f} < {LAYER_PALETTE_MIN_DE}); "
    "see the geo/PRE-POST rationale in COLOR -- move the hue, do not lower the gate")
assert _min_same_shape_delta_e(30) >= LAYER_PAIR_MIN_DE, (
    f"two same-shape layer colours are only {_min_same_shape_delta_e(30):.1f} deltaE apart "
    f"(< {LAYER_PAIR_MIN_DE}); add a shape to LAYER_SHAPES rather than lowering this")


# --- HUD layout -------------------------------------------------------------
# Column offsets from the HUD's left pad, in pixels, for the metric table's
# P / R / F1 cells. Hershey is a PROPORTIONAL font, so space-padding a single
# putText string does NOT align columns. Each cell therefore gets its own
# putText at a fixed x.
#
# Sized against the MINIMUM panel: compute_panel_width floors at 420, so
# bar_w = 388. The widest cell is a bounded recall like "0.7911<=" at 49px, so
# the F1 column ends at 286+49 = 335 and still clears 388 with room for the
# section label at x=0 to run to ~145px.
HUD_COL_X = (150, 218, 286)
HUD_COL_HEADS = ("P", "R", "F1")
# Left pads of the two painters that draw metric columns. They DIFFER, which is
# why LAYER_COL_X below is derived rather than being HUD_COL_X reused: the HUD
# pads by 16 and the model table by 6, so sharing the tuple verbatim would sit
# the table's P/R/F1 ten pixels left of the HUD's and the two blocks would not
# read as one grid.
HUD_PAD_X = 16
LAYER_PAD_X = 6
LAYER_COL_X = tuple(x + HUD_PAD_X - LAYER_PAD_X for x in HUD_COL_X)
assert all(LAYER_PAD_X + a == HUD_PAD_X + b for a, b in zip(LAYER_COL_X, HUD_COL_X)), (
    "the model table's columns do not land on the same screen x as the HUD's")

# Row advances, in draw order. Summed and asserted against the panel budget
# below, so the budget is a check rather than hand-arithmetic that every added
# line would silently invalidate.
HUD_ADV = {
    "top": 16,        # y0 -> first baseline
    "label": 8,       # a text line sitting directly above its own progress bar
    "bar": 20,
    "rule": 16,       # section rule + its label/column heads
    "metric": 17,     # a PRE/POST row
    "note": 16,       # dim trailer/aux line
    "warn": 17,
    "descent": 12,    # room under the last baseline
}
# The geo block draws FOUR metric rows (PRE + one per tie-break rule). The
# `op:` line and the pixel counts trailer live in [p]'s SUMMARY block, because
# `c` differs per rule and a single "+N c" would lie. See
# geo_metrics.VERDICT_RULES for why all three are mandatory.


@dataclass
class HudRow:
    """One drawn row of the HUD, as data.

    Split out so the smoke test can assert on layout -- column alignment, the
    height budget, which rows appear -- without scraping pixels, the same
    reason hud_match_params_line and _stats_overlay_lines are pure builders.
    `cells` is a list of (x_offset_from_pad, text, colour); `kind` selects the
    painter in _draw_hud.
    """
    kind: str                      # "text" | "bar" | "rule" | "cells"
    advance: int
    cells: List[Tuple[int, str, Tuple[int, int, int]]] = field(default_factory=list)
    scale: float = 0.36
    frac: float = 0.0
    color: Tuple[int, int, int] = (255, 255, 255)
    rule: bool = False             # draw a separator line above this row
    # Which part of the HUD this row belongs to. The HUD is a scrolling
    # viewport now: `pinned` rows sit at the top and never scroll, `body` rows
    # are the per-model groups and do, and `footer` rows are bottom-anchored.
    # The caveat row is a footer row on purpose -- a caveat you can scroll out
    # of view is a caveat that will be missed, which is the same reasoning
    # status_segments uses to give it priority 0.
    zone: str = "body"             # "pinned" | "body" | "footer"
    group: int = -1                # which model group; -1 for chrome


@dataclass(frozen=True)
class HudModelStat:
    """One model layer's numbers as the HUD needs them: BOTH metric spaces,
    BOTH arms, and all three geo tie-break rules.

    `None` anywhere means NOT COMPUTED YET, never zero. The geo arm is built
    off-thread (a scaffold per layer), so a row can legitimately have its
    pixel numbers and not yet its geo ones. Rendering a pending cell as
    0.0000 would be read as a measurement, and leaving it blank would be read
    as "not applicable" -- which is what geo's empty F1 column already means.
    So pending gets its own glyph.
    """
    layer_id: str
    label: str
    verdict: str                                    # healthy | collapsed | sparse | unknown
    color: Tuple[int, int, int] = (255, 255, 255)   # the layer's own identity colour
    n_dets: int = 0
    pix_pre: Optional[Tuple[float, float, float]] = None
    pix_post: Optional[Tuple[float, float, float]] = None
    geo_pre: Optional[Tuple[float, float]] = None   # (P, R); no F1 -- see _metric_cells
    # One (P, R) per geo_metrics.VERDICT_RULES, most-positive first. A None
    # entry is a rule not yet computed; the tuple length never changes, so the
    # row count does not move when results land.
    geo_post: Tuple[Optional[Tuple[float, float]], ...] = (None, None, None)
    geo_note: str = ""                              # "computing..." / a reason


HUD_PENDING = "computing..."


def _pending_cells(label: str, color: Tuple[int, int, int], text: str = HUD_PENDING
                   ) -> List[Tuple[int, str, Tuple[int, int, int]]]:
    """A metric row whose values are not in yet.

    Emits ONLY the label and one word at the P column, deliberately: _draw_hud
    fits each cell to the gap before the next STOP IN THAT ROW, so with no cell
    at HUD_COL_X[1] the word gets the rest of the width and renders in full
    instead of being clipped to a 68px column."""
    return [(0, label, color), (HUD_COL_X[0], text, (150, 150, 150))]


def _metric_cells(label: str, values: Sequence[Optional[str]],
                  color: Tuple[int, int, int]) -> List[Tuple[int, str, Tuple[int, int, int]]]:
    """A metric row: label at x=0, then one cell per HUD_COL_X. A None value
    leaves its column empty -- geo reports no F1 (precision is per cluster and
    recall per geotag, so their harmonic mean is a category error), and the
    column must stay blank rather than collapse, or geo's R would sit under
    pixel's F1 and invite exactly the cross-space comparison the three-metric-
    space split exists to prevent."""
    cells = [(0, label, color)]
    for x, v in zip(HUD_COL_X, values):
        if v is not None:
            cells.append((x, v, color))
    return cells


def status_segments(*, dir_idx: int, n_dirs: int, rel_name: str, frame_idx,
                    frame_pos: int, n_frames: int, frame_rev: int, frame_tot: int,
                    dir_rev: int, dir_tot: int, layers_on: int = 0, layers_total: int = 0,
                    solo: Optional[str] = None, accum: str = "",
                    caveat: str = "") -> List[Tuple[str, int]]:
    """The status line as (text, priority) segments, left to right.

    Priority 0 is never dropped; higher numbers are dropped first, right to
    left, until the line fits. That replaces a `text[:main_w // 7 + 40]`
    character slice, which guessed a fixed character width for a PROPORTIONAL
    font and chopped mid-word.

    THE CAVEAT IS PRIORITY 0, alongside the flight name. A statement that the
    numbers on screen are approximate outranks any count -- it changes what all
    of them mean.
    """
    segs: List[Tuple[str, int]] = []
    if caveat:
        segs.append((caveat, 0))
    segs.append((f"dir {dir_idx + 1}/{n_dirs}: {rel_name}", 0))
    segs.append((f"frame {frame_idx} ({frame_pos + 1}/{n_frames})", 1))
    if solo:
        segs.append((f"SOLO {solo}", 1))
    if layers_total:
        segs.append((f"layers {layers_on}/{layers_total}", 2))
    if accum:
        segs.append((f"accum {accum}", 3))
    segs.append((f"dir clusters {dir_rev}/{dir_tot} reviewed", 2))
    segs.append((f"frame clusters {frame_rev}/{frame_tot} reviewed", 3))
    return segs


def fit_status(segments: Sequence[Tuple[str, int]], max_px: int,
               font_scale: float = 0.45) -> str:
    """Join the segments, dropping WHOLE segments by descending priority (and
    right to left within a priority) until the line fits. Never chops a word."""
    keep = list(segments)
    while keep:
        text = "  |  ".join(t for t, _ in keep)
        if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] <= max_px:
            return text
        worst = max((pri for _, pri in keep), default=0)
        if worst == 0:
            # Only priority-0 segments left and still too wide: truncate rather
            # than drop something that must not be dropped.
            return fit_text(text, max_px, font_scale)
        for i in range(len(keep) - 1, -1, -1):
            if keep[i][1] == worst:
                keep.pop(i)
                break
    return ""


def hud_rows(*, frame_idx, frame_pos: int, n_frames: int,
             reviewed: int, n_clusters: int, n_raw: int,
             spatial_threshold: float, cluster_radius: float,
             geo_eps: float = _GEO_EPS, geo_match: float = _GEO_EPS_PREDICT,
             prop_note: Optional[str] = None,
             overrides: Optional[dict] = None,
             n_layers: int = 0,
             layer_note: str = "",
             model_stats: Sequence["HudModelStat"] = (),
             play_line: str = "") -> List[HudRow]:
    """The whole HUD as an ordered list of rows -- pure, no cv2, no App.

    The layout it describes:

        frame 812 (813/1355)                          <- pinned
        [====================]
        reviewed 90/168 clusters (54%)   [12345 raw]
        [===============     ]
        -- m2o 80/30px  eps 0.20m  match 0.35m  cov 57%  a=0  prop:pos --
        -- 030 t.04   healthy   44012d --        P        R        F1
           PIXEL PRE                            0.1888   0.8137   0.3064
           PIXEL POST                           0.2011   0.8203<= 0.3232<=
           GEO PRE                              0.1888   0.8137
           GEO POST pos                         0.2011   0.8156<=
           GEO POST maj                         0.1954   0.8149<=
           GEO POST neg                         0.1888   0.8137<=
        -- 014   healthy   19204d --             P        R        F1
           PIXEL PRE                            0.5221   0.5031   0.5124
           GEO PRE                              computing...
         ^ models 2-6/13   [,][.] scroll
           ! OVERRIDE ...                             <- footer

    EVERY METRIC ROW IS PER MODEL. There is no row scoring the union of every
    active checkpoint's detections: the union is a REVIEW-BURDEN measure and
    nobody's precision, it would read as an anchor that invites exactly the
    comparison it cannot support, and the per-model table says everything it
    would say. The union matching itself is still the detection substrate and
    the judgment path -- it simply has no scorecard.

    PRE and POST are ROWS OF ONE TABLE sharing column positions, and the pixel
    and geo blocks share those same columns, so the two comparisons the reviewer
    actually makes -- PRE vs POST within a space, and one space against the
    other -- are reading rather than arithmetic. They stay in COLOR["pre"] /
    COLOR["post"], the colour language the [p] overlay and the exported report
    also use.

    GEO GETS THREE POST ROWS, one per tie-break rule, ordered most-positive
    first. Not a display preference: a conflicted cluster's verdict is decided
    by a CONVENTION rather than by data, the choice is worth ~14 precision
    points here, and there is no primary rule to fall back on. Reading the three
    rows top to bottom IS reading the error bar. See geo_metrics.VERDICT_RULES.

    ONE PINNED PARAMS RULE ROW carries the live matching values (both spaces)
    and `prop:` -- the [o] propagation rule. It is not a metric: it says which
    slider settings and which tie-break convention decide every number below
    it, so the per-model groups are never unparameterised. `prop:` rides here
    rather than in the footer because it changes what [m] paints and what [a]
    would write, which is live state, not a caveat. It carries NO P/R/F1 heads
    -- see the comment on the row for the measurement that forces that.
    """
    note = (150, 150, 150)
    rows = [
        HudRow("text", HUD_ADV["label"], [(0, f"frame {frame_idx} ({frame_pos + 1}/{n_frames})",
                                           COLOR["text"])], scale=0.42, zone="pinned"),
        HudRow("bar", HUD_ADV["bar"], frac=(frame_pos + 1) / n_frames if n_frames else 0.0,
               color=COLOR["bar_frame"], zone="pinned"),
        HudRow("text", HUD_ADV["label"],
               [(0, f"reviewed {reviewed}/{n_clusters} clusters "
                    f"({(reviewed / n_clusters * 100) if n_clusters else 0:.0f}%)   [{n_raw} raw]",
                 COLOR["text"])], scale=0.42, zone="pinned"),
        HudRow("bar", HUD_ADV["bar"], frac=reviewed / n_clusters if n_clusters else 0.0,
               color=COLOR["bar_review"], zone="pinned"),
    ]

    head = [(x, h, note) for x, h in zip(HUD_COL_X, HUD_COL_HEADS)]

    # -- ONE pinned params rule row, and no union metrics ------------------
    #
    # The rule row carries the LIVE PARAMETERS: which slider settings decide
    # the matched/unmatched split and the review unit for every number below,
    # which is why REVIEW_GUI.md calls the section rule the thing you read the
    # table against -- both spaces' params on one line, plus the [o]
    # propagation rule, which belongs with them because it decides what [m]
    # paints and what [a] would write rather than being a caveat.
    params = hud_params_line(spatial_threshold, cluster_radius,
                             geo_eps, geo_match, prop_note)
    # NO P/R/F1 HEADS ON THIS ROW, and that is a measurement rather than a
    # preference. _draw_hud fits each cell to the gap before the next cell IN
    # ITS OWN ROW, so heads here would cap the label at HUD_COL_X[0] - 6 =
    # 144px -- and this line is 290px at the reference setting, 364px with the
    # longest rule name. It would be silently truncated, which is the exact
    # failure hud_columns_clear exists to catch. Heads-less, the row gets the
    # whole bar (400px at PANEL_FLOOR_W) and fits.
    #
    # Nothing is lost: the heads REPEAT on every model group's own rule row,
    # which is already the documented reason they repeat -- heads printed once
    # at the top are heads you have scrolled past.
    rows.append(HudRow("rule", HUD_ADV["rule"], rule=True, scale=0.34, zone="pinned",
                       cells=[(0, params, note)]))

    g = 0

    # -- one group per active model layer ----------------------------------
    #
    # PIXEL PRE/POST and GEO PRE + one POST row per tie-break rule, for EVERY
    # active layer. The whole point of reviewing every checkpoint at once is
    # that one adjudication scores all of them, and that comparison is only
    # readable if the numbers sit in one grid -- so every row here uses the
    # same _metric_cells stops as the union block above it.
    #
    # The P/R/F1 heads REPEAT on each group's rule row. Heads printed once at
    # the top are heads you have scrolled past, and riding them on the rule
    # costs no vertical budget (the separator is drawn in the gap above the
    # baseline).
    for st in model_stats:
        tag = {"collapsed": "COLLAPSED", "sparse": "SPARSE"}.get(st.verdict, "")
        # The title lives in the 150px before the P column, so it carries only
        # identity: label, health, detection count. `geo_note` deliberately
        # does NOT go here -- "computing..." is already said by each pending
        # cell, and appending it pushed the title into the P head where the
        # painter truncated it mid-word.
        title = f"{st.label}   {tag + '  ' if tag else ''}{st.n_dets}d"
        # The group header carries the LAYER'S OWN COLOUR, so the HUD block and
        # the model window's swatch speak one identity language. The metric
        # rows keep the PRE/POST arm colours: arm and identity are different
        # axes and overloading one colour with both makes neither readable.
        rows.append(HudRow("rule", HUD_ADV["rule"], rule=True, scale=0.34, group=g,
                           cells=[(0, title, st.color)] + head))
        rows.append(HudRow("cells", HUD_ADV["metric"], scale=0.40, group=g,
                           cells=_metric_cells(
                               "PIXEL PRE",
                               [f"{st.pix_pre[0]:.4f}", f"{st.pix_pre[1]:.4f}",
                                f"{st.pix_pre[2]:.4f}"], COLOR["pre"])
                           if st.pix_pre else _pending_cells("PIXEL PRE", COLOR["pre"])))
        rows.append(HudRow("cells", HUD_ADV["metric"], scale=0.40, group=g,
                           cells=_metric_cells(
                               "PIXEL POST",
                               [f"{st.pix_post[0]:.4f}", f"{st.pix_post[1]:.4f}<=",
                                f"{st.pix_post[2]:.4f}<="], COLOR["post"])
                           if st.pix_post else _pending_cells("PIXEL POST", COLOR["post"])))
        rows.append(HudRow("cells", HUD_ADV["metric"], scale=0.40, group=g,
                           cells=_metric_cells(
                               "GEO PRE",
                               [f"{st.geo_pre[0]:.4f}", f"{st.geo_pre[1]:.4f}", None],
                               COLOR["pre"])
                           if st.geo_pre else _pending_cells(
                               "GEO PRE", COLOR["pre"],
                               # A REASON, not a spinner, when the arm is
                               # unavailable rather than merely late.
                               text=st.geo_note or HUD_PENDING)))
        # ALL THREE RULES, ALWAYS, and the tuple length is fixed even while
        # they are still computing -- so the row count never changes as
        # results land and nothing reflows under the cursor. There is no
        # primary rule; reading the three top to bottom IS reading the error
        # bar. See geo_metrics.VERDICT_RULES.
        for abbr, pr in zip(_GEO_RULE_ABBRS, st.geo_post):
            rows.append(HudRow("cells", HUD_ADV["metric"], scale=0.40, group=g,
                               cells=_metric_cells(
                                   f"GEO POST {abbr}",
                                   [f"{pr[0]:.4f}", f"{pr[1]:.4f}<=", None], COLOR["post"])
                               if pr else _pending_cells(f"GEO POST {abbr}", COLOR["post"])))
        g += 1

    # ONE caveat row, not one row per caveat. The HUD budget is fixed and the
    # override line would otherwise push the worst case to 252 of 240 -- the
    # module-scope assertion catches that, and raising the budget is not an
    # option because it comes straight off the zoom inset. [p] carries the full
    # list, so truncation here loses nothing that is not one keypress away.
    caveats = []
    if overrides:
        # A deviation from the run whose artifacts are on screen changes what
        # every number above means, so it cannot live only in [p].
        caveats.append("OVERRIDE " + " ".join(
            f"{k.split('_')[0]} {v['run']:g}->{v['live']:g}" for k, v in overrides.items()))
    if layer_note:
        # Folded into the SAME single caveat row rather than taking one of its
        # own -- the HUD budget is enforced at import and a new row would have
        # to be paid for out of the zoom inset.
        caveats.append(layer_note)
    if caveats:
        rows.append(HudRow("text", HUD_ADV["warn"], scale=0.36, zone="footer", cells=[
            (0, "! " + "  ! ".join(caveats) + ("   [p]" if len(caveats) > 1 else ""),
             COLOR["error"] if overrides else COLOR["caveat"])]))
    if play_line:
        rows.append(HudRow("text", 0, scale=0.42, zone="footer",
                           cells=[(0, play_line, COLOR["selected"])]))
    return rows


@dataclass(frozen=True)
class LayerGeoSummary:
    """One layer's geo scaffold, REDUCED to what re-scoring actually needs.

    WHY REDUCE AT ALL. A GeoScaffold holds `points` (N x 3 float), `labels`,
    `core_mask` and a `cell_to_point` dict -- ~25 MB for the full sample flight's union.
    Holding one per layer is affordable at 12 layers and wasteful at 30, and
    the transient DBSCAN neighbour graph is the real peak anyway, so the
    worker builds them one at a time and keeps only this.

    WHY THE JOIN CACHE IS THE WHOLE TRICK. geo_metrics.locate() is expensive
    PER FRAME, not per point -- it recomputes the frame pose and to_3d for
    every frame it is handed -- so collect_verdicts over a few thousand judged
    points spanning ~1,300 frames costs ~0.075s per layer, i.e. ~1s per
    keypress across twelve layers. Unacceptable for something that reruns on
    every commit.

    But the pixel -> 3cm cell -> cluster mapping is PURE GEOMETRY: it does not
    depend on any judgment. So it is computed once per layer for that layer's
    own detections (0.12-0.21s measured) and re-scoring becomes dict lookups,
    under 10 ms. `(frame_idx, x, y)` is unique within a layer file -- verified
    on val_030_t0.04: 44,012 rows, 44,012 distinct keys.

    `member_label` maps that key to a cluster label, with -1 for "DBSCAN noise
    or unresolvable". geo_metrics.locate conflates those two, and this keeps
    the conflation rather than inventing a distinction its caller cannot act
    on -- but see n_noise_points, which is what makes per-layer coverage
    legible: 11-22% of a layer's raw rows are noise, so per-layer adjudication
    coverage is structurally lower than the union's for a reason that is not a
    defect.
    """
    layer_id: str
    ok: bool = True
    unavailable: Optional[str] = None
    member_label: Dict[Tuple[int, float, float], int] = field(default_factory=dict)
    # Everything geo_prf() reads off a scaffold, and nothing else. Verified
    # against its body: it touches .ok/.unavailable, np.unique(labels),
    # geotag_label, in_view, len(geotags), n_noise, len(points),
    # cluster_frames and params -- never points, core_mask or cell_to_point.
    labels_unique: Optional["np.ndarray"] = None
    geotag_label: Optional["np.ndarray"] = None
    in_view: Optional["np.ndarray"] = None
    geotags: Optional["np.ndarray"] = None
    n_noise: int = 0
    n_points: int = 0
    n_noise_points: int = 0
    cluster_frames: Dict[int, int] = field(default_factory=dict)
    params: Dict[str, object] = field(default_factory=dict)
    # Core points in 2D, for collect_added's "did THIS model fire here" test.
    # Per-layer by construction, which is why geo `a` is per-layer -- see
    # _layer_geo_summaries.
    core_xy: Optional["np.ndarray"] = None


class _ScaffoldView:
    """Duck-typed stand-in exposing exactly the attributes geo_prf() and
    collect_added() read, so a reduced summary can be scored by the SAME
    functions as a live scaffold. That identity is the point: no second
    implementation of the PRE/POST arithmetic can drift from the first."""

    def __init__(self, sm: "LayerGeoSummary"):
        self._sm = sm
        self.labels = sm.labels_unique
        self.geotag_label = sm.geotag_label
        self.in_view = sm.in_view
        self.geotags = sm.geotags
        self.n_noise = sm.n_noise
        self.cluster_frames = sm.cluster_frames
        self.params = sm.params
        self.unavailable = sm.unavailable
        # collect_added asks for core points via points[core_mask]; hand it
        # the core points directly with an all-True mask.
        self.points = sm.core_xy
        self.core_mask = None if sm.core_xy is None else \
            np.ones(len(sm.core_xy), dtype=bool)
        self.n_points = sm.n_points

    @property
    def ok(self) -> bool:
        return bool(self._sm.ok)

    def __len__(self) -> int:
        return self._sm.n_points


def _reduce_layer_scaffold(geo_metrics, layer_id: str, sc, frames, proj) -> LayerGeoSummary:
    """Build a LayerGeoSummary from a live scaffold, then let the scaffold go.

    Runs on the worker thread. The join is computed for EVERY detection of the
    layer, not only the judged ones, because which points are judged changes
    on every commit while this mapping does not."""
    if not getattr(sc, "ok", False):
        return LayerGeoSummary(layer_id=layer_id, ok=False,
                               unavailable=getattr(sc, "unavailable", "unavailable"))
    member_label: Dict[Tuple[int, float, float], int] = {}
    n_noise_points = 0
    by_index = {i: row for i, row in enumerate(proj.poses)} if proj is not None else {}
    for fi, fd in frames.items():
        row = by_index.get(int(fi))
        if row is None or not fd.points:
            continue
        xy = np.array([[pt.x, pt.y] for pt in fd.points], dtype=float)
        try:
            labs = geo_metrics.locate(sc, proj, row, int(fi), xy)
        except Exception:  # noqa: BLE001 -- one bad frame must not lose the layer
            continue
        for pt, lab in zip(fd.points, labs):
            lab = int(lab)
            member_label[(int(pt.frame_idx), float(pt.x), float(pt.y))] = lab
            if lab < 0:
                n_noise_points += 1
    core = sc.points[sc.core_mask][:, :2].astype(np.float32) \
        if getattr(sc, "core_mask", None) is not None and len(sc.points) else None
    return LayerGeoSummary(
        layer_id=layer_id, ok=True, member_label=member_label,
        labels_unique=np.unique(sc.labels), geotag_label=sc.geotag_label,
        in_view=sc.in_view, geotags=sc.geotags,
        n_noise=int(sc.n_noise), n_points=int(len(sc.points)),
        n_noise_points=n_noise_points,
        cluster_frames=dict(sc.cluster_frames), params=dict(sc.params),
        core_xy=core)


_MODEL_SORT_KEY = {
    "f1": lambda r: -r.post[2], "recall": lambda r: -r.post[1],
    "precision": lambda r: -r.post[0], "dets": lambda r: r.n_dets,
}

MODEL_SORT_CYCLE = ("f1", "recall", "precision", "dets")


def model_display_order(layers: Sequence["LayerSpec"], *,
                        metrics: Optional[Mapping[str, "ModelMetricRow"]] = None,
                        sort: str = "f1",
                        sel_counts: Optional[Mapping[str, int]] = None) -> List[str]:
    """The ONE presentation sequence: layer_ids in the order they are shown.

    Pure -- no App, no DirectoryReview, no cv2 -- so the smoke test can assert
    the ordering rules directly instead of inferring them from a rendered
    table, and so [ and ] (LayerState.move_focus) step through exactly the
    sequence on screen -- one ordering with one owner.

    THIS IS THE PRESENTATION SEQUENCE, NOT THE INDEX SPACE. LayerState.ids
    stays in disk order for ever, because render_key() builds the static
    composite's bitmask from enumerate(self.ids) and dr.cluster_layer_counts
    tuples are indexed by dr.layers position. Sorting `ids` would silently
    re-key the composite cache every tick and mis-attribute per-layer counts.

    Rules:
      * DEGENERATE CHECKPOINTS SORT LAST BUT ARE NEVER DROPPED. The health key
        is explicit rather than left to the metric because a collapsed
        checkpoint holds the highest precision-like score anywhere in the
        30-checkpoint screen -- sorting on the number alone would float a dead
        model to the top.
      * With a cluster selected, layers that fired on it partition stably to
        the front: on 30 layers the answer to "which models found this spot"
        would otherwise sit below the fold.
    """
    specs = list(layers)
    if not specs:
        return []
    met = dict(metrics or {})
    key = _MODEL_SORT_KEY.get(sort, _MODEL_SORT_KEY["f1"])

    def metric_key(sp) -> float:
        row = met.get(sp.layer_id)
        # A layer with no scorecard yet (or whose geo arm is still building)
        # must sort to a CONSTANT, not to a value that moves as results land --
        # otherwise the table reorders under the cursor mid-gesture, which is
        # the same class of defect as focusing on click.
        return key(row) if row is not None else 0.0

    specs.sort(key=lambda sp: (sp.degenerate, metric_key(sp)))
    if sel_counts is not None:
        specs.sort(key=lambda sp: not sel_counts.get(sp.layer_id, 0))
    return [sp.layer_id for sp in specs]


def hud_rows_height(rows: Sequence[HudRow]) -> int:
    """How tall a HUD row list actually draws, in pixels below its box top.

    The one place this arithmetic lives, so the HUD's background box is
    measured from the rows it actually draws."""
    return HUD_ADV["top"] + sum(r.advance for r in rows) + HUD_ADV["descent"]


HUD_GROUP_ROWS = 7          # rule + PIXEL PRE/POST + GEO PRE + 3 GEO POST


def hud_group_height() -> int:
    """One complete model group, in pixels. DERIVED, not written down."""
    return HUD_ADV["rule"] + (HUD_GROUP_ROWS - 1) * HUD_ADV["metric"]


def hud_chrome_height() -> int:
    """Everything the HUD draws that is NOT a model group: the pinned block
    (two labelled bars plus the params rule row) and the worst-case
    bottom-anchored footer (a caveat row and a play line).

    The import-time probes below check this against the zoom inset's budget."""
    return (HUD_ADV["top"]
            + 2 * (HUD_ADV["label"] + HUD_ADV["bar"])      # pinned: frame + review bars
            + HUD_ADV["rule"]                              # pinned: the params rule row
            + HUD_ADV["warn"]                              # footer: the one caveat row
            + HUD_ADV["descent"])


def hud_groups_fitting(band_h: int) -> int:
    """How many WHOLE model groups a band of this height can show."""
    return max(0, (band_h - hud_chrome_height()) // hud_group_height())


@dataclass(frozen=True)
class HudScroll:
    """A HUD row list sliced to a band: what is pinned, what scrolls, what is
    anchored to the bottom, and what is off-screen in each direction."""
    pinned: List[HudRow] = field(default_factory=list)
    body: List[HudRow] = field(default_factory=list)
    footer: List[HudRow] = field(default_factory=list)
    scroll: int = 0                # the clamped group offset actually applied
    n_groups: int = 0
    groups_shown: int = 0
    more_above: bool = False
    more_below: bool = False

    def all_rows(self) -> List[HudRow]:
        return list(self.pinned) + list(self.body) + list(self.footer)


def hud_scroll_view(rows: Sequence[HudRow], *, band_h: int, scroll: int = 0) -> HudScroll:
    """Slice a HUD row list into a scrolling viewport. Pure, no cv2.

    Guarantees, each of which a fixed-height HUD could not
    have and which the smoke test asserts by sweeping this function:

      * pinned and footer rows are ALWAYS present, whatever the height. The
        footer carries the caveat row, and a caveat that can scroll out of
        view is a caveat that will be missed.
      * the body is sliced on GROUP BOUNDARIES -- a group's header is on
        screen if and only if all of its rows are. Half a group is unreadable,
        and readability is the requirement.
      * `scroll` is clamped so the last screenful is exactly reachable and
        scrolling past the end is impossible. The clamp lives HERE and the
        caller stores what comes back, which is the lesson of
        LayerState.scroll_by: a builder that clamps a local copy lets the
        stored value drift past the end, and the next scroll in the other
        direction then spends its whole step getting back into range and
        looks dead.
      * every group is reachable at SOME scroll offset. Degenerate
        checkpoints sort last, so an unreachable tail would hide exactly the
        models the 30-checkpoint screen exists to expose.
    """
    pinned = [r for r in rows if r.zone == "pinned"]
    footer = [r for r in rows if r.zone == "footer"]
    body = [r for r in rows if r.zone == "body"]
    # Preserve emission order within each group.
    groups: List[List[HudRow]] = []
    for r in body:
        if not groups or (groups[-1] and groups[-1][0].group != r.group):
            groups.append([])
        groups[-1].append(r)

    fixed = (HUD_ADV["top"] + sum(r.advance for r in pinned)
             + sum(r.advance for r in footer) + HUD_ADV["descent"])
    avail = max(0, band_h - fixed)

    shown: List[List[HudRow]] = []
    n = 0
    if groups:
        # Uniform group height is not assumed. Every group is a model group
        # today and they are the same height, but a pending geo arm and a
        # future extra row are both one edit away, and a walk that measures
        # each group cannot be wrong about one that does not match.
        i = max(0, min(int(scroll), max(0, len(groups) - 1)))
        # Walk back if the tail does not fill the band, so the last screenful
        # is reachable exactly rather than only approached.
        def fits(start: int) -> int:
            used, k = 0, 0
            for grp in groups[start:]:
                h = sum(r.advance for r in grp)
                if used + h > avail:
                    break
                used += h
                k += 1
            return k
        n = fits(i)
        while i > 0 and fits(i - 1) >= n + 1:
            i -= 1
            n = fits(i)
        shown = groups[i:i + n]
        scroll = i
    else:
        scroll = 0
    return HudScroll(pinned=pinned, body=[r for grp in shown for r in grp], footer=footer,
                     scroll=scroll, n_groups=len(groups), groups_shown=n,
                     more_above=scroll > 0, more_below=(scroll + n) < len(groups))


def hud_columns_clear(panel_w: int) -> Optional[str]:
    """Does a WORST-CASE HUD group fit its column grid at this panel width?

    A scrolling viewport has no total height to bound, and height is not what
    makes the table readable. What makes it readable is that no cell runs into
    the next column: _draw_hud fits each cell to the gap before the next stop,
    so an over-long cell is silently TRUNCATED rather than overlapped, and a
    test on heights cannot see "0.7911<=" arriving as "0.79..".

    Uses cv2 only to MEASURE text, which is what fit_status and
    check_keybinding_docs already do. Returns None when clear, else a message
    naming the first collision.
    """
    bar_w = panel_w - 2 * HUD_PAD_X
    # The worst case is a per-model label -- derived here rather than assumed,
    # because the point of this guard is that nothing about column fit is
    # taken on trust.
    worst_label = max(
        [HUD_PENDING]
        + [f"GEO POST {a}" for a in _GEO_RULE_ABBRS]
        + ["PIXEL PRE", "PIXEL POST", "GEO PRE"],
        key=lambda t: cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0])
    # Rows that DO carry the P/R/F1 heads: every cell is capped by the next
    # column stop.
    checks = [(0, worst_label, 0.40), (0, "030 t0.035   COLLAPSED  234646d", 0.34)]
    for x, head in zip(HUD_COL_X, HUD_COL_HEADS):
        checks.append((x, head, 0.34))
    for x in HUD_COL_X:
        checks.append((x, "0.7911<=", 0.40))
    stops = sorted(set([0] + list(HUD_COL_X)))
    for x, text, scale in checks:
        w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        nxt = next((o for o in stops if o > x), None)
        avail = (nxt - x - 6) if nxt is not None else (bar_w - x)
        if w > avail:
            return (f"at panel_w={panel_w} the cell {text!r} at x={x} needs {w}px but has "
                    f"{avail}px before the next column -- it would be truncated")
    # The PINNED PARAMS ROW is measured separately and against the WHOLE BAR,
    # because it carries no heads -- _draw_hud derives its stops from the row's
    # own cells, so a lone cell at dx=0 has no next stop. Measuring it against
    # HUD_COL_X[0] would be stricter than the painter and would reject a row
    # that renders fine; not measuring it at all is how the union geo rule row
    # came to be truncated unnoticed. Worst case is the longest rule name.
    worst_params = hud_params_line(80, 30, _GEO_EPS, _GEO_EPS_PREDICT,
                                   max(_GEO_VERDICT_RULES, key=len))
    w = cv2.getTextSize(worst_params, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0][0]
    if w > bar_w:
        return (f"at panel_w={panel_w} the pinned params row {worst_params!r} needs {w}px "
                f"but the bar is {bar_w}px -- it would be truncated")
    return None


# How many whole model groups the HUD ASKS for before the zoom inset grows to
# its target. Two, not one: one group answers "how is this model doing" and
# two answer "how does it compare", which is the question reviewing every
# checkpoint at once exists to serve. Groups three onward are one keypress
# away at zero pixel cost -- which is the argument that replaces the old
# "do NOT raise PANEL_HUD_H": HUD rows scroll, inset pixels do not.
PANEL_HUD_TARGET_GROUPS = 2
# The floor: chrome plus ONE complete model group. A HUD that cannot show one
# whole model is not worth the pixels it costs.
PANEL_HUD_MIN_H = hud_chrome_height() + hud_group_height()
# The target the layout grows it to before the inset competes.
PANEL_HUD_H = hud_chrome_height() + PANEL_HUD_TARGET_GROUPS * hud_group_height()

# THE READABILITY GUARD, asserted at import: no cell runs into the next column
# at the narrowest panel this tool renders. _draw_hud fits each cell to the
# gap before the next stop, so an over-long cell is silently TRUNCATED rather
# than overlapped -- "0.7911<=" arriving as "0.79.." is invisible to any
# assertion about heights.
assert hud_columns_clear(PANEL_FLOOR_W) is None, hud_columns_clear(PANEL_FLOOR_W)
assert PANEL_HUD_MIN_H >= hud_chrome_height() + hud_group_height(), \
    "the HUD floor does not fit one complete model group"


# ---------------------------------------------------------------------------
# The layer legend: a CLICKABLE widget.
#
# The geometry and the hit boxes are built here as PURE DATA -- exactly the
# discipline hud_rows() follows -- and the smoke test asserts on the row
# model, including that no two hit boxes overlap. A widget whose hit boxes are
# only ever tested by clicking it is a widget nobody can safely change.
#
# The widget has its OWN WINDOW (App.open_model_window): a WINDOW_AUTOSIZE
# window's canvas coordinates ARE its mouse coordinates, so there is no rect
# predicate and no rescale between the renderer and the hit test. Two
# independent answers to "where is it" is how a click lands on the wrong row;
# there is only one.
# ---------------------------------------------------------------------------

LAYER_ROW_H = 18
LAYER_HEADER_H = 22
LAYER_FOOTER_H = 18          # scroll strip + the ALL/NONE/ACC buttons; ALWAYS drawn
# NO ROW CAP: panel_layout() sizes the table from the space actually
# available, so a large panel shows as many of 30 models as fit.
# Column offsets from the widget's left pad. Negative offsets are measured from
# the RIGHT edge. Fixed, so hit boxes are arithmetic and the builder needs no
# cv2 -- the same reason HUD_COL_X exists.
LAYER_X_SWATCH = 6
LAYER_X_GLYPH = 20
LAYER_X_LABEL = 34
# The model window shows the full layer_id ("val_030_t0.04") and spells the
# health verdict out; there are no metric columns here -- those live in the
# HUD -- which is where the room comes from.
LAYER_X_HEALTH = 130
LAYER_X_COUNT = 232
# Fixed, not right-anchored: anchored to the right edge the row would read as
# two disconnected halves on a wide panel. This keeps the table a single
# compact block at any panel width, and the whole row is the click target.
LAYER_X_CHECK = 316
LAYER_BTN_W = 46
LAYER_BTN_GAP = 4
# Header cells for the window's two status readouts.
MODEL_X_ACCUM = 232
MODEL_X_SORT = 340

# The model window. Its own OpenCV window, sized to its content.
MODEL_WINDOW = "model layers -- detection review"
MODEL_WIN_W = 420
MODEL_WIN_PAD = 6
# Above this the footer's scroll strip takes over. 900px is well past what 30
# checkpoints need (see model_window_size), so scrolling is the exception here
# rather than the rule it was in the panel band.
MODEL_WIN_MAX_H = 900
# Below this the metric columns do not fit beside the label and checkbox, and
# the table falls back to the plain colour/toggle legend. At the 420px floor
# panel the F1 cell ends at 345 and the checkbox occupies 388-408, so this
# cannot trigger today -- it exists so that a narrower panel degrades rather
# than overlaps.
LAYER_METRICS_MIN_W = 380

@dataclass
class LegendRow:
    """One drawn row of the layer legend, as data.

    `hit` is (x0, y0, x1, y1) RELATIVE TO THE WIDGET ORIGIN, so the builder
    stays pure and the caller adds the widget's own offset exactly once.
    `action` is "" for a row that is not clickable.
    """
    kind: str                      # "header" | "btn" | "layer" | "scroll" | "note"
    y: int
    h: int
    hit: Tuple[int, int, int, int] = (0, 0, 0, 0)
    action: str = ""               # toggle | solo | all | none | accum | up | down
    tag: str = ""
    cells: List[Tuple[int, str, Tuple[int, int, int]]] = field(default_factory=list)
    swatch: Optional[Tuple[int, int, int]] = None
    shape: str = ""
    hollow: bool = False
    strike: bool = False
    emphasis: bool = False
    # Is this CONTROL usable -- not whether the layer it names is visible.
    # Those were briefly the same field, which made an off layer unclickable
    # and therefore impossible to turn back on. Visibility rides on `on_state`
    # and the [x]/[ ] cell; `enabled` is only ever False for a control with
    # nothing behind it (the ACC button with no pose data).
    enabled: bool = True
    on_state: bool = False
    fired: bool = False        # this layer fired on the selected cluster


def layer_rows_shown(n_layers: int, max_h: int) -> int:
    """How many layer rows fit. Pure arithmetic, shared by the layout and the
    scroll clamp so the two cannot disagree about what is reachable."""
    # The footer is ALWAYS subtracted: it carries the ALL/NONE/ACC buttons,
    # not just a scroll strip, so it draws whether or not the list is
    # truncated, and an untruncated table must not claim a row the footer
    # draws over.
    return min(n_layers, layer_row_capacity(max_h))


def model_window_size(n_layers: int) -> Tuple[int, int]:
    """(w, h) for a model window holding `n_layers` rows without scrolling.

    The point of the separate window, quantified: this returns 420x592 for the
    30-checkpoint model-selection screen, so every checkpoint is on screen at
    once. The in-panel table it replaces showed 5 of 30 at the default window
    size and could not be made to show more without taking the pixels off the
    zoom inset."""
    h = (LAYER_HEADER_H + max(1, n_layers) * LAYER_ROW_H + LAYER_FOOTER_H
         + 2 * MODEL_WIN_PAD)
    return MODEL_WIN_W, min(h, MODEL_WIN_MAX_H)


def layer_row_capacity(max_h: int) -> int:
    """How many rows of ANY kind the table body can hold -- NOT capped by the
    layer count.

    The distinction matters: layer_rows_shown() is capped at n_layers, so it
    cannot say whether there is spare room, and the focused model's extra PRE
    row has to be paid for out of real capacity. Using the capped number, a
    600px table holding 14 layers reported "14 shown of 14 capacity" and
    dropped a model to make room that it already had; using it the other way,
    a table sized for exactly 12 drew 12 layers plus a PRE row and pushed its
    own footer 18px outside the widget."""
    body = max_h - LAYER_HEADER_H - LAYER_FOOTER_H
    return max(0, body // LAYER_ROW_H)


# ---------------------------------------------------------------------------
# The side panel's vertical layout.
#
# WHY THIS EXISTS AS ONE PURE FUNCTION. Laying the panel out bottom-up from
# worst-case reserves scattered across several methods leaves dead box,
# paints backgrounds that do not match what is drawn, lets unbounded text run
# over neighbouring blocks, and computes the same arithmetic in two places
# that then disagree.
#
# So: one pure function owns every y-coordinate in the panel, the drawing code
# only consumes bands, and the smoke test sweeps it from 320px to 2000px
# asserting that no two bands ever overlap. Same discipline as hud_rows() and
# model_window_rows() -- geometry as data, asserted on the model, not scraped
# off pixels.
# ---------------------------------------------------------------------------

PANEL_TOP_PAD = 20            # first drawable y
PANEL_GAP = 10                # between adjacent blocks
PANEL_INSET_GAP = 20          # under the zoom inset (also keeps check_inset's
                              # contiguous-row scan from running into the next block)
PANEL_HINT_H = 26
# The zoom inset never drops below this: a 60px floor would be an effective
# 0.6x "magnification" of a 100px crop -- worse than useless.
INSET_FLOOR_H = 240
# ...and grows to this before the model table takes any surplus.
# NOT a round number: smoke_test_gui.check_inset requires the inset to exceed
# `canvas_h - OLD_PANEL_OVERHEAD_PX` = 392px on its 1000px reference window, and
# that contract has to hold WITH LAYERS ON.
INSET_TARGET_H = 400
PANEL_INFO_MIN_LINES = 2      # coordinates + judgment: the irreducible pair
# Five base lines + model agreement + three propagation lines (geo cluster,
# reach, propagates) when the geo arm is ready -- see _selected_point_info_lines.
PANEL_INFO_MAX_LINES = 9
# PANEL_HUD_MIN_H / PANEL_HUD_H are derived beside hud_chrome_height(), which
# they are built from, so they know what the HUD actually draws.
# What gets sacrificed, in order, when even the floors do not fit. Exported so
# the smoke test can assert WHICH degradation happened rather than merely that
# nothing overlapped.
PANEL_SHRINK_ORDER = ("table", "info", "hud", "inset")


@dataclass(frozen=True)
class PanelBand:
    """A vertical slice of the panel. Empty (`h == 0`) means "not drawn"."""
    y0: int = 0
    y1: int = 0

    @property
    def h(self) -> int:
        return max(0, self.y1 - self.y0)

    def __bool__(self) -> bool:
        return self.h > 0


@dataclass(frozen=True)
class PanelLayout:
    inset: PanelBand
    info: PanelBand
    hud: PanelBand
    hint: PanelBand
    info_rows: int = 0             # the bound the info loop never had
    hud_groups: int = 0            # whole model groups the HUD band can show
    dropped: Tuple[str, ...] = ()  # blocks the shrink ladder gave up, in order

    def ordered(self) -> List[Tuple[str, PanelBand]]:
        """Every band top to bottom, in draw order, empties included."""
        return [("inset", self.inset), ("info", self.info), ("hud", self.hud),
                ("hint", self.hint)]


def panel_layout(*, canvas_h: int, hud_h: int, inset_want: int = 0,
                 info_lines: int = 0, has_selection: bool = False) -> PanelLayout:
    """Every y-coordinate in the side panel. Pure: no cv2, no App, no drawing.

    Floors are granted first, then surplus is handed out in a fixed priority
    order, then -- only if the floors themselves do not fit -- blocks are given
    up along PANEL_SHRINK_ORDER.

    THE MODEL TABLE IS GONE FROM THIS FUNCTION, and with it `n_models` and
    `table_expanded`. The per-checkpoint list moved to its own window ([e]) and
    its numbers moved into the HUD, so THE PANEL'S GEOMETRY NO LONGER DEPENDS
    ON HOW MANY CHECKPOINTS EXIST. That is the substantive simplification here:
    a 30-layer run and a 1-layer run lay the panel out identically.

    THE GROWTH ORDER IS THE DESIGN DECISION, so it is spelled out rather than
    buried: info, then the HUD to one whole model group, then the inset to its
    target, then the HUD to PANEL_HUD_H, then the inset to what it can use,
    then the HUD takes whatever is left.

    The HUD is guaranteed ONE COMPLETE MODEL GROUP before the inset's target
    and a second one after it. That is the argument that replaced "do NOT raise
    PANEL_HUD_H": model groups three onward are one keypress away at zero pixel
    cost (the body scrolls), whereas inset pixels are recoverable by no
    keystroke at all -- so the inset wins the middle of the order and the HUD
    takes the tail, which is why a tall window ends up showing many groups
    without ever starving the inset.
    """
    hint_h = PANEL_HINT_H
    budget = canvas_h - PANEL_TOP_PAD - hint_h
    dropped: List[str] = []

    want_info = min(max(info_lines, 0), PANEL_INFO_MAX_LINES) * PANEL_INFO_LINE_H
    min_info = min(PANEL_INFO_MIN_LINES * PANEL_INFO_LINE_H, want_info)
    min_inset = INSET_FLOOR_H if has_selection else 0
    want_inset = max(inset_want, INSET_TARGET_H) if has_selection else 0
    min_hud = min(hud_h, PANEL_HUD_MIN_H)

    cur = {"inset": min_inset, "info": min_info, "hud": min_hud}

    def gaps(c) -> int:
        n = sum(1 for k in ("inset", "info", "hud") if c[k] > 0)
        extra = PANEL_INSET_GAP - PANEL_GAP if c["inset"] > 0 else 0
        return max(0, n - 1) * PANEL_GAP + extra

    def used(c) -> int:
        return sum(c.values()) + gaps(c)

    # -- the shrink ladder, when even the floors do not fit ------------------
    for name in PANEL_SHRINK_ORDER:
        if used(cur) <= budget:
            break
        if name == "info" and cur["info"]:
            cur["info"] = 0
            dropped.append("info")
        elif name == "hud" and cur["hud"]:
            cur["hud"] = 0
            dropped.append("hud")
        elif name == "inset" and cur["inset"]:
            cur["inset"] = max(1, budget - used({**cur, "inset": 0}))
            dropped.append("inset")

    # -- growth, in priority order ------------------------------------------
    one_group = min(hud_h, PANEL_HUD_MIN_H)
    for key, target in (("info", want_info),
                        ("hud", one_group),
                        ("inset", INSET_TARGET_H if has_selection else 0),
                        ("hud", min(hud_h, PANEL_HUD_H)),
                        ("inset", want_inset),
                        ("hud", hud_h)):
        if not cur[key] or cur[key] >= target:
            continue
        cur[key] = min(target, cur[key] + max(0, budget - used(cur)))

    # Snap to whole rows. A band 10px taller than an exact number of lines is
    # 10px of nothing: it cannot draw another row and it pushes everything
    # below it down. The remainder goes back to the inset, which is the one
    # block that uses arbitrary heights.
    if cur["info"]:
        cur["info"] -= cur["info"] % PANEL_INFO_LINE_H
    # Snap the HUD DOWN to chrome + a whole number of model groups, never up --
    # snapping is only ever allowed to give space back; growing past what the
    # ladder granted would push the hint off the bottom of a short canvas.
    if cur["hud"] > hud_chrome_height():
        k = hud_groups_fitting(cur["hud"])
        snapped = hud_chrome_height() + k * hud_group_height()
        if hud_chrome_height() <= snapped <= cur["hud"]:
            cur["hud"] = snapped
    if cur["inset"]:
        cur["inset"] += max(0, budget - used(cur))

    # -- lay the bands out top to bottom ------------------------------------
    y = PANEL_TOP_PAD
    bands = {}
    for k in ("inset", "info", "hud"):
        h = cur[k]
        bands[k] = PanelBand(y, y + h) if h else PanelBand(y, y)
        if h:
            y += h + (PANEL_INSET_GAP if k == "inset" else PANEL_GAP)
    return PanelLayout(
        inset=bands["inset"], info=bands["info"], hud=bands["hud"],
        hint=PanelBand(canvas_h - hint_h, canvas_h),
        info_rows=cur["info"] // PANEL_INFO_LINE_H,
        hud_groups=hud_groups_fitting(cur["hud"]) if cur["hud"] else 0,
        dropped=tuple(dropped))


# The panel's load-bearing guarantees, asserted by RUNNING THE REAL LAYOUT at
# two named heights rather than by hand-arithmetic over coupled constants.
# The guarantee is about the zoom inset, so assert the zoom inset.
_MIN_CANVAS_H = 820      # the shortest panel this tool supports
_REF_CANVAS_H = 1000     # the window smoke_test_gui.check_inset measures at

# There is no `n_models` to sweep here -- the panel's geometry is independent
# of the checkpoint count because the model list has its own window. The HUD
# asks for a fixed target instead.
_probe_min = panel_layout(canvas_h=_MIN_CANVAS_H, hud_h=10_000, inset_want=10_000,
                          info_lines=PANEL_INFO_MAX_LINES, has_selection=True)
assert _probe_min.inset.h >= INSET_FLOOR_H, (
    f"on the shortest supported panel ({_MIN_CANVAS_H}px) the zoom inset drops to "
    f"{_probe_min.inset.h}px, under INSET_FLOOR_H={INSET_FLOOR_H} -- shrink a block above "
    f"it, do not lower the floor")
assert _probe_min.hud_groups >= 1, (
    f"the shortest supported panel shows {_probe_min.hud_groups} whole model groups -- "
    f"the HUD must always be able to show one complete model")
assert not _probe_min.dropped, f"the shortest supported panel already drops {_probe_min.dropped}"

_probe_ref = panel_layout(canvas_h=_REF_CANVAS_H, hud_h=10_000, inset_want=10_000,
                          info_lines=PANEL_INFO_MAX_LINES, has_selection=True)
assert _probe_ref.inset.h >= INSET_TARGET_H, (
    f"the enlarged zoom inset does not survive with the per-model HUD "
    f"({_probe_ref.inset.h}px < {INSET_TARGET_H}) -- see smoke_test_gui.check_inset, "
    f"which measures exactly this")
# ONE group is what is GUARANTEED here, not PANEL_HUD_TARGET_GROUPS. This
# probe is the worst case on purpose -- max info lines AND a selection -- and
# at 1000px a 400px inset plus 160px of info leaves room for one group, not
# two. The growth order ASKS for two; whether it gets them is a property of
# the window, so the guarantee asserted is the one that always holds.
assert _probe_ref.hud_groups >= 1, (
    f"the reference window shows {_probe_ref.hud_groups} whole model groups")

# ...and on a tall window the target is actually reached, with the inset still
# at its target. This is the "a tall window shows what it can" property, and
# the pair of assertions together is what stops either block starving the
# other as the row model changes.
_probe_tall = panel_layout(canvas_h=1440, hud_h=10_000, inset_want=10_000,
                           info_lines=PANEL_INFO_MAX_LINES, has_selection=True)
assert _probe_tall.hud_groups >= PANEL_HUD_TARGET_GROUPS, (
    f"a 1440px panel shows only {_probe_tall.hud_groups} whole model groups, under the "
    f"{PANEL_HUD_TARGET_GROUPS} the growth order promises")
assert _probe_tall.inset.h >= INSET_TARGET_H, (
    f"a 1440px panel starves the zoom inset to {_probe_tall.inset.h}px while growing the HUD")


def model_window_rows(*, layers: Sequence["LayerSpec"], order: Sequence[str],
                      visible: Set[str], solo: Optional[str] = None,
                      focus: Optional[str] = None, scroll: int = 0,
                      width: int = 420, max_h: int = MODEL_WIN_MAX_H,
                      accum_mode: str = "off", accum_available: bool = True,
                      accum_effective: Optional[str] = None,
                      note: str = "", sort: str = "f1",
                      counts_this_frame: Optional[Mapping[str, int]] = None,
                      totals: Optional[Mapping[str, int]] = None,
                      sel_counts: Optional[Mapping[str, int]] = None,
                      sel_conf: Optional[Mapping[str, float]] = None) -> List[LegendRow]:
    """The MODEL WINDOW as an ordered list of rows -- pure, no cv2, no App.

        MODELS 7/12 on   SEL 8/12 fired          accum per_layer   sort f1
        * val_030_t0.04   healthy      4182d                        [x]
          val_013         COLLAPSED      31d                        [ ]
        ^ 1-12/12 v                              [ALL][NONE][ACC][SORT]

    This is a TOGGLE CONTROL, not a metric table. The per-model numbers live
    in the HUD, where they get the full column grid and all three geo
    tie-break rules, and this gets the width and the height it needs to be
    clickable -- model_window_size(30) fits all thirty checkpoints of the
    model-selection screen with no scrolling at all.

    IT WALKS `order`, NOT `layers`. `order` is the one presentation sequence
    (model_display_order -> LayerState.set_order), and both this builder and
    the hit test read it, so [ and ] step through the sequence on screen and
    a click always lands on the row the reviewer can see.

    `hit` boxes are relative to the widget origin, so the caller adds its own
    offset exactly once -- for this window that offset is MODEL_WIN_PAD, and
    because the window is AUTOSIZE its canvas coordinates ARE its mouse
    coordinates.
    """
    counts = counts_this_frame or {}
    tot = totals or {}
    selc = sel_counts or {}
    ids = [s.layer_id for s in layers]
    by_id = {s.layer_id: s for s in layers}
    seq = [lid for lid in order if lid in by_id] or ids
    rows: List[LegendRow] = []
    y = 0
    note_c = (150, 150, 150)

    n_on = len(visible & set(ids))
    eff = accum_effective or accum_mode
    head = f"MODELS {n_on}/{len(ids)} on"
    if sel_counts is not None:
        head += f"  SEL {sum(1 for v in selc.values() if v)}/{len(ids)} fired"
    if solo:
        head += f"  SOLO {by_id[solo].label if solo in by_id else solo}"
    acc = f"accum {eff}" if accum_available else "accum n/a"
    if accum_available and eff != accum_mode:
        acc += f" (from {accum_mode})"
    # The header names the COLUMNS; the accum/sort readouts ride the footer,
    # which has the room. Putting them here collided with the count column --
    # MODEL_X_ACCUM and LAYER_X_COUNT are the same x.
    rows.append(LegendRow("header", y, LAYER_HEADER_H, cells=[
        (0, head, COLOR["text"]),
        (LAYER_X_COUNT, "here/all", note_c),
        (LAYER_X_CHECK, "on", note_c)]))
    y += LAYER_HEADER_H

    shown = layer_rows_shown(len(seq), max_h)
    truncated = shown < len(seq)
    scroll = max(0, min(scroll, max(0, len(seq) - shown)))
    window = seq[scroll:scroll + shown]

    for lid in window:
        spec = by_id[lid]
        st = layer_style(spec.layer_id, ids, spec.verdict)
        on = spec.layer_id in visible
        # SPELLED OUT, and upper-case for the degenerate ones. The panel band
        # only had room for a one-glyph badge ("X"/"~"/"h"), which meant the
        # single most important structural finding in the 30-checkpoint screen
        # -- that 19 of 30 checkpoints are degenerate -- was carried by a
        # tilde. Degenerate checkpoints are always shown and always labelled;
        # they are never filtered out.
        badge, bcol = {"collapsed": ("COLLAPSED", COLOR["error"]),
                       "sparse": ("SPARSE", COLOR["caveat"]),
                       "healthy": ("healthy", note_c)}.get(spec.verdict, ("unknown", note_c))
        if spec.arms:
            badge += "*"
        fired = selc.get(spec.layer_id, 0)
        if sel_counts is not None:
            cnt = (f"{fired}d {sel_conf.get(spec.layer_id, 0.0):.2f}".replace("0.", ".")
                   if fired else "--")
        else:
            # ONE COLUMN, ONE UNIT: "<this frame>/<directory total>". It used
            # to print the per-frame count when the layer fired here and the
            # DIRECTORY TOTAL with a "d" suffix when it did not, so the column
            # read "2", "4", "5305d", "20" down the page -- two different
            # quantities in one column, told apart only by a suffix. Both
            # numbers are worth having, so show both, always.
            cnt = f"{counts.get(spec.layer_id, 0)}/{tot.get(spec.layer_id, 0)}"
        dim = sel_counts is not None and not fired
        lcol = COLOR["selected"] if spec.layer_id == focus else (
            note_c if (dim or not on) else COLOR["text"])
        rows.append(LegendRow(
            "layer", y, LAYER_ROW_H,
            hit=(0, y, width, y + LAYER_ROW_H), action="toggle", tag=spec.layer_id,
            swatch=st.color, shape=st.shape, hollow=st.hollow, strike=st.strike,
            emphasis=(spec.layer_id == focus or spec.layer_id == solo),
            on_state=on, fired=bool(fired),
            cells=[(LAYER_X_LABEL, spec.layer_id, lcol),
                   (LAYER_X_HEALTH, badge, bcol),
                   (LAYER_X_COUNT, cnt, note_c if dim else lcol),
                   (LAYER_X_CHECK, "[x]" if on else "[ ]",
                    COLOR["text"] if on else note_c)]))
        y += LAYER_ROW_H

    # The footer always draws: it carries the buttons.
    lo, hi = scroll + 1, min(scroll + shown, len(seq))
    fcells: List[Tuple[int, str, Tuple[int, int, int]]] = []
    if truncated:
        fcells.append((0, f"^ {lo}-{hi}/{len(seq)} v", note_c))
    # x=92 only when the scroll indicator occupies x=0. Otherwise start at 0
    # and get the whole 206px before the buttons -- the worst-case readout
    # ("accum off (from per_layer)   sort precision") fits there and would be
    # truncated at 92. It was measured at 192px when the worst case read
    # "accum density (from per_layer)"; the current worst case is shorter, so
    # the margin only grew.
    tx = 92 if truncated else 0
    if note:
        fcells.append((tx, fit_text(note, max(0, width - 300), 0.34), COLOR["error"]))
    else:
        fcells.append((tx, f"{acc}   sort {sort}", note_c))
    if truncated:
        rows.append(LegendRow("scroll", y, LAYER_FOOTER_H, hit=(0, y, 40, y + LAYER_FOOTER_H),
                              action="up", cells=fcells))
        rows.append(LegendRow("scroll", y, LAYER_FOOTER_H, hit=(42, y, 82, y + LAYER_FOOTER_H),
                              action="down"))
    else:
        rows.append(LegendRow("scroll", y, LAYER_FOOTER_H, cells=fcells))
    btns = (("ALL", "all", True), ("NONE", "none", True),
            ("ACC", "accum", accum_available), ("SORT", "sort", True))
    bx = width - (len(btns) * LAYER_BTN_W + (len(btns) - 1) * LAYER_BTN_GAP) - 6
    for i, (lbl, act, ok) in enumerate(btns):
        x0 = bx + i * (LAYER_BTN_W + LAYER_BTN_GAP)
        rows.append(LegendRow("btn", y, LAYER_FOOTER_H,
                              hit=(x0, y, x0 + LAYER_BTN_W, y + LAYER_FOOTER_H),
                              action=act, enabled=ok,
                              cells=[(x0, lbl, COLOR["text"] if ok else note_c)]))
    return rows


class LayerState:
    """Live display state for the model layers: which are on, which is soloed,
    which has focus, the accumulation mode, and the legend scroll.

    Purely a DISPLAY concern. Nothing here ever reaches the clustering -- see
    recompute_layered_matching_steps -- which is what guarantees that toggling
    a layer cannot move a centroid and therefore cannot re-key a judgment.
    """

    def __init__(self, layers: Sequence["LayerSpec"] = ()):
        self.set_layers(layers)

    def set_layers(self, layers: Sequence["LayerSpec"]) -> None:
        self.layers: Tuple["LayerSpec", ...] = tuple(layers)
        self.ids: List[str] = [s.layer_id for s in self.layers]
        # Everything on by default: "show all data across all runs, do not
        # curate" -- including the degenerate checkpoints, which are the whole
        # reason this screen is worth having.
        self.visible: Set[str] = set(self.ids)
        # THE PRESENTATION SEQUENCE, kept apart from `ids` on purpose. `ids` is
        # the INDEX SPACE (render_key()'s bitmask, visible_indices(), the
        # per-layer count tuples) and must stay in disk order for ever;
        # `order` is what the screen shows and what [ and ] walk. Conflating
        # them is the bug this pair exists to prevent -- see set_order().
        self.order: List[str] = list(self.ids)
        self.solo: Optional[str] = None
        self._pre_solo: Optional[Set[str]] = None
        self.focus: Optional[str] = self.ids[0] if self.ids else None
        self.scroll = 0
        self.expanded = len(self.ids) <= 4
        self.accum_mode = "off"          # off | per_layer
        self.enabled = bool(self.ids)

    # -- queries ----------------------------------------------------------
    def effective_accum(self, available: bool = True) -> str:
        """The accumulation mode actually used.

        Above LAYER_ACCUM_PERLAYER_MAX visible layers this returns "off":
        eleven translucent smears over grass is not a picture anyone can read.

        It does NOT fall back to a combined "how many models agree here" view:
        that would superimpose every checkpoint into one undifferentiated
        cloud, which is exactly the union reprojection the pane avoids.
        Turning the tier OFF keeps the guard honest -- and it is announced in
        the status bar, never silent, with soloing (or hiding layers down to
        the limit) restoring it immediately."""
        if not available or self.accum_mode == "off":
            return "off" if not available else self.accum_mode
        if len(self.visible) > LAYER_ACCUM_PERLAYER_MAX:
            return "off"
        return self.accum_mode

    def set_order(self, order: Sequence[str]) -> None:
        """Adopt a new presentation sequence (from model_display_order()).

        NEVER re-sort `self.ids` to achieve this. `render_key()` builds the
        static composite's bitmask from enumerate(self.ids) and
        dr.cluster_layer_counts tuples are indexed by dr.layers position, so
        re-ordering the index space would re-key the composite cache on every
        tick and mis-attribute per-layer detection counts. The assertion below
        is the guard: `order` must be a PERMUTATION of `ids`, nothing more.

        The focus rides along by IDENTITY, not by position -- the focused model
        keeps the focus when the sort or the selection re-orders the list, and
        the scroll is re-clamped so the focused row stays on screen, so a
        selection re-partitioning the table never makes the focus jump.
        """
        order = list(order)
        assert sorted(order) == sorted(self.ids), (
            "model_display_order returned something that is not a permutation of "
            f"LayerState.ids ({len(order)} vs {len(self.ids)}) -- `order` is the "
            "presentation sequence, it may never add, drop or invent a layer")
        self.order = order
        if self.focus is not None and self.focus not in order:
            self.focus = order[0] if order else None

    def sequence(self) -> List[str]:
        """The sequence every presentation surface walks -- the model window,
        [ and ], the scroll clamp and the HUD's per-model groups. One list, so
        the screen order and the focus order cannot come apart."""
        return self.order or list(self.ids)

    def render_key(self) -> Tuple:
        """Hashable summary of everything that changes what gets DRAWN. Feeds
        the static-composite cache key."""
        mask = 0
        for i, lid in enumerate(self.ids):
            if lid in self.visible:
                mask |= 1 << i
        return (mask, self.accum_mode, self.solo or "", self.enabled)

    def visible_indices(self) -> List[int]:
        return [i for i, lid in enumerate(self.ids) if lid in self.visible]

    # -- mutations --------------------------------------------------------
    def toggle(self, layer_id: str) -> bool:
        if layer_id in self.visible:
            self.visible.discard(layer_id)
        else:
            self.visible.add(layer_id)
        self.solo = None
        return layer_id in self.visible

    def set_solo(self, layer_id: str) -> bool:
        """Solo, or un-solo back to whatever was on before."""
        if self.solo == layer_id:
            self.visible = set(self._pre_solo if self._pre_solo is not None else self.ids)
            self.solo, self._pre_solo = None, None
            return False
        if self.solo is None:
            self._pre_solo = set(self.visible)
        self.solo = layer_id
        self.visible = {layer_id}
        return True

    def all_on(self) -> None:
        self.visible = set(self.ids)
        self.solo, self._pre_solo = None, None

    def none_on(self) -> None:
        self.visible = set()
        self.solo, self._pre_solo = None, None

    def cycle_accum(self) -> str:
        # Two states: there is no combined/union accumulation view. See
        # effective_accum().
        order = ("off", "per_layer")
        self.accum_mode = order[(order.index(self.accum_mode) + 1) % len(order)]
        return self.accum_mode

    def scroll_by(self, rows: int, n: int, shown: int) -> int:
        """Move the table's scroll, clamped against the LIVE visible-row count.

        The clamp lives here because it is state, not presentation:
        model_window_rows clamps a LOCAL copy, so without this the attribute
        drifted past the end of the list and a subsequent "up" click spent its
        whole step getting back into range and appeared to do nothing."""
        self.scroll = max(0, min(self.scroll + rows, max(0, n - max(1, shown))))
        return self.scroll

    def move_focus(self, delta: int, shown: int = 0) -> Optional[str]:
        """Move the focus and scroll the legend just enough to keep it visible.

        The scroll adjustment lives HERE, not in the row builder: it is a
        response to the focus moving, and doing it on every render pinned the
        legend to the focused layer and made the scroll buttons useless.

        WALKS self.sequence(), NOT self.ids. Indexing `ids` here while the
        table drew a sorted order is the whole defect: [ and ] stepped through
        disk order, so they appeared to skip rows, and the follow-scroll below
        computed its window from an index into the wrong list and jumped to an
        unrelated row -- worst with a cluster selected, since the fired-first
        partition then re-ordered on every selection change."""
        seq = self.sequence()
        if not seq:
            return None
        i = seq.index(self.focus) if self.focus in seq else 0
        j = max(0, min(len(seq) - 1, i + delta))
        self.focus = seq[j]
        if shown > 0:
            self.scroll = max(0, min(self.scroll, max(0, len(seq) - shown)))
            if j < self.scroll:
                self.scroll = j
            elif j >= self.scroll + shown:
                self.scroll = j - shown + 1
        return self.focus


def layer_hit_test(rows: Sequence[LegendRow], x: int, y: int) -> Optional[LegendRow]:
    """First row whose hit box contains (x, y), or None.

    Buttons are emitted after the header row they sit on and layer rows span
    the full width, so an earlier-wins scan resolves the overlap between a
    header and its buttons the way a reader expects. The smoke test asserts
    that no two hit boxes of the SAME kind overlap, which is the property that
    actually matters."""
    for r in rows:
        if not r.action or not r.enabled:
            continue
        x0, y0, x1, y1 = r.hit
        if x0 <= x < x1 and y0 <= y < y1:
            return r
    return None




# ---------------------------------------------------------------------------
# Judgment persistence: review_log.jsonl is the sole source of truth; the two
# .txt sidecars are pure derived exports, regenerated on every commit.
# ---------------------------------------------------------------------------

class JudgmentStore:
    def __init__(self, directory: Path, rel_name: str, reviewer: str, spatial_threshold: float,
                 dry_run: bool, param_overrides: Optional[dict] = None,
                 log_dir: Optional[Path] = None, schema_version: int = 1,
                 record_extra: Optional[dict] = None):
        # `directory` is the FLIGHT directory and supplies every record's
        # "directory" field, which review_report.py and other readers key on.
        #
        # `log_dir` is WHERE THE LOG AND SIDECARS ARE WRITTEN, and defaults to
        # the flight directory. Layered review passes
        # <run>/review/<flight_slug>/ instead, so adjudications of a run's
        # dets/ files can never touch any review_log.jsonl in the flight
        # directory. That separation is structural, not a convention: the
        # three export paths below are only ever built from self.log_dir.
        self.directory = directory
        self.rel_name = rel_name
        self.reviewer = reviewer
        self.spatial_threshold = spatial_threshold
        self.dry_run = dry_run
        self.param_overrides = dict(param_overrides or {})
        self.log_dir = Path(log_dir) if log_dir is not None else Path(directory)
        self.schema_version = schema_version
        self.record_extra = dict(record_extra or {})
        self.log_path = self.log_dir / "review_log.jsonl"
        self.export_paths = {judgment: self.log_dir / fname for judgment, fname in EXPORTABLE.items()}
        # <run>/review/<slug>/ may not exist yet, and both commit()'s append
        # and export()'s write_text would raise on the first judgment. Never
        # under --dry-run, which must leave no trace at all.
        if not dry_run and not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        # key -> latest record for that (frame_idx, x, y)
        self.judgments: Dict[Tuple[int, float, float], dict] = {}
        # Bumped on every judgment write (see commit()). This is the ONLY
        # authoritative mutation point for self.judgments, which is what
        # makes DirectoryReview's stats memo exact rather than best-effort:
        # App.commit/spawn_point/remove_spawned/the [r] handler/
        # migrate_legacy_judgments all route through commit().
        self.version = 0
        self._load()

    def _load(self) -> None:
        if not self.log_path.exists():
            return
        n_bad = 0
        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (int(rec["frame_idx"]), float(rec["x"]), float(rec["y"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    n_bad += 1
                    continue
                self.judgments[key] = rec
        if n_bad:
            print(f"  Note: {self.log_path} had {n_bad} unparseable line(s), skipped.")

    def get(self, frame_idx: int, x: float, y: float) -> str:
        # .get("judgment"), not rec["judgment"]: _load only validates
        # frame_idx/x/y, so a record written by another tool (or a future
        # schema) can reach here
        # without the key. Raising from here would crash the render loop.
        rec = self.judgments.get((frame_idx, x, y))
        return rec.get("judgment", "unreviewed") if rec else "unreviewed"

    def commit(self, frame_idx: int, x: float, y: float, confidence: float,
               judgment: str, image_path: str, extra: Optional[dict] = None) -> None:
        rec = {
            "schema_version": self.schema_version,
            "timestamp": datetime.now().astimezone().isoformat(),
            "reviewer": self.reviewer,
            "directory": self.rel_name,
            "frame_idx": frame_idx,
            "image_path": image_path,
            "x": x, "y": y,
            "original_confidence": confidence,
            "judgment": judgment,
            "spatial_threshold": self.spatial_threshold,
        }
        # A judgment made under parameters that DIFFER from the run whose
        # artifacts are on screen is a different measurement, and the log is the
        # only place that survives the session. Stamped per record rather than
        # once per file because the sliders can move mid-session.
        if self.param_overrides:
            rec["param_overrides"] = self.param_overrides
        # Session-constant provenance (the run, the ordered layer set, the
        # clustering radius) before the per-call fields, so a per-call value
        # always wins.
        if self.record_extra:
            rec.update(self.record_extra)
        if extra:
            rec.update(extra)
        self.judgments[(frame_idx, x, y)] = rec
        self.version += 1  # invalidates DirectoryReview's stats memo
        if not self.dry_run:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def export(self, member_map: Dict[Tuple[int, float, float], List[Detection]]) -> Dict[str, int]:
        """Regenerate the exportable .txt sidecars (one per EXPORTABLE
        judgment) from judgments whose key still resolves against the
        currently-loaded cluster/spawned pool (DirectoryReview.member_map).
        A judgment made once on a cluster centroid is written out as ONE
        LINE PER RAW MEMBER DETECTION (each with its own x/y/confidence,
        not the centroid's mean) -- so the exported files stay at full
        per-detection resolution even though review happens at the cluster
        level. A spawned point's map entry is a singleton list (itself --
        nothing to aggregate). Returns counts keyed by judgment name (raw
        lines written, i.e. already inflated by cluster size), plus
        "orphaned" (clusters/points whose key does not resolve, e.g. the
        underlying data was modified)."""
        buckets: Dict[str, List[str]] = {judgment: [] for judgment in EXPORTABLE}
        orphaned = 0
        for key, rec in self.judgments.items():
            judgment = rec.get("judgment")
            if judgment not in EXPORTABLE:
                continue
            members = member_map.get(key)
            if members is None:
                orphaned += 1
                continue
            for m in members:
                buckets[judgment].append(f"{m.frame_idx} {rec['image_path']} {m.x},{m.y},{m.confidence}\n")
        if not self.dry_run:
            for judgment, lines in buckets.items():
                self.export_paths[judgment].write_text("".join(lines))
        counts = {judgment: len(lines) for judgment, lines in buckets.items()}
        counts["orphaned"] = orphaned
        return counts


class LayeredJudgmentStore(JudgmentStore):
    """A JudgmentStore for a layered (multi-model) review.

    Identical bookkeeping -- same key, same log format, same version counter --
    with two differences in what `export()` produces, both forced by the fact
    that one consensus cluster's members come from several detection files:

    1. THE UNION SIDECARS ARE DEDUPED on (frame_idx, x, y). Measured on the
       full sample flight's holdout run: 7,738 of the 224,758 detections across the 11 layers
       are EXACT coordinate coincidences between layers (max multiplicity 6),
       because several checkpoints legitimately fire on the same pixel. The
       base class writes one line per member, so without dedup a training pool
       built from these files would carry 3.4% phantom duplicates -- and they
       would be duplicates of exactly the easy, unanimous detections, which is
       the worst possible sampling bias to introduce silently.

    2. A PER-LAYER BREAKDOWN is written under by_layer/<layer_id>/. This is
       what makes "adjudicate once, score eleven" consumable outside this GUI:
       each directory is a legitimate per-checkpoint confirmed/hard-negative
       pool, carrying only that checkpoint's own detections, from one human
       pass over the union.
    """

    def __init__(self, *a, layers: Sequence["LayerSpec"] = (), **kw):
        self.layers = tuple(layers)
        super().__init__(*a, **kw)

    def export(self, member_map: Dict[Tuple[int, float, float], List[Detection]]) -> Dict[str, int]:
        buckets: Dict[str, Dict[Tuple[int, float, float], str]] = {j: {} for j in EXPORTABLE}
        per_layer: Dict[int, Dict[str, List[str]]] = {
            i: {j: [] for j in EXPORTABLE} for i in range(len(self.layers))}
        orphaned = 0
        for key, rec in self.judgments.items():
            judgment = rec.get("judgment")
            if judgment not in EXPORTABLE:
                continue
            members = member_map.get(key)
            if members is None:
                orphaned += 1
                continue
            for m in members:
                line = f"{m.frame_idx} {rec['image_path']} {m.x},{m.y},{m.confidence}\n"
                # dict keyed on the coordinate == dedup, order-preserving
                buckets[judgment][(m.frame_idx, m.x, m.y)] = line
                li = getattr(m, "layer", -1)
                if 0 <= li < len(self.layers):
                    per_layer[li][judgment].append(line)
        if not self.dry_run:
            for judgment, lines in buckets.items():
                self.export_paths[judgment].write_text("".join(lines.values()))
            for i, spec in enumerate(self.layers):
                if not any(per_layer[i].values()):
                    continue
                d = self.log_dir / "by_layer" / spec.layer_id
                d.mkdir(parents=True, exist_ok=True)
                for judgment, lines in per_layer[i].items():
                    (d / EXPORTABLE[judgment]).write_text("".join(lines))
        counts = {judgment: len(lines) for judgment, lines in buckets.items()}
        counts["orphaned"] = orphaned
        counts["by_layer"] = {
            self.layers[i].layer_id: {j: len(v) for j, v in per_layer[i].items()}
            for i in range(len(self.layers))}
        return counts


# ---------------------------------------------------------------------------
# Per-directory review state: live-matched (no cached artifacts) from
# labels.txt + the run's dets/ files via detections.parse_annotations.
# ---------------------------------------------------------------------------

@dataclass
class DirectoryReview:
    path: Path
    rel_name: str
    frames_all: List[int]                              # frames with a detection, for linear nav
    labels_by_frame: Dict[int, List[Detection]] = field(default_factory=dict)   # context only
    matched_by_frame: Dict[int, List[Detection]] = field(default_factory=dict)  # context only
    unmatched_by_frame: Dict[int, List[Detection]] = field(default_factory=dict)  # RAW detections; naive stats only
    unmatched_clusters_by_frame: Dict[int, List[Detection]] = field(default_factory=dict)  # centroids -- review targets
    cluster_members: Dict[Tuple[int, float, float], List[Detection]] = field(default_factory=dict)  # centroid key -> raw members
    point_to_cluster_key: Dict[Tuple[int, float, float], Tuple[int, float, float]] = field(default_factory=dict)  # raw key -> owning centroid key
    unmatched_labels_by_frame: Dict[int, List[Detection]] = field(default_factory=dict)  # naive-recall FN pool
    spawned_by_frame: Dict[int, List[Detection]] = field(default_factory=dict)  # human-added, model never detected
    image_path_by_frame: Dict[int, str] = field(default_factory=dict)
    store: Optional[JudgmentStore] = None
    # Cached parse of labels.txt + the run's dets/ -- populated once at initial
    # load, never mutated afterward. Lets a live slider change re-run
    # matching+clustering (recompute_matching()) without re-reading disk;
    # evicting this DirectoryReview from App.dir_cache is exactly
    # equivalent to forcing a fresh reparse, so no separate invalidation
    # bookkeeping is needed anywhere else.
    dets_fd: Dict[int, FrameData] = field(default_factory=dict)
    labels_fd: Dict[int, FrameData] = field(default_factory=dict)
    # Bumped whenever recompute_matching_steps() rebuilds the matching/
    # clustering dicts. Together with store.version this forms the exact
    # cache key for the directory-wide stats memo below -- see _stats_key().
    # --- layered review -----------------------------------------------------
    # `layers` is always populated in a live session -- an empty one means
    # the directory did not load at all, since the run's dets/ files are the
    # only detection source. The `not self.layers` short-circuits below are
    # deliberate: the pure helpers are constructed with a bare
    # DirectoryReview by smoke_test_gui, and a zero-layer guard is cheaper
    # than a precondition every caller has to remember.
    #
    # Clustering runs over ALL layers, never over the visible subset. That is
    # the invariant the whole design rests on: toggling a layer off is a
    # draw-time filter, so it can never move a centroid and therefore can never
    # re-key or orphan a judgment.
    layers: Tuple["LayerSpec", ...] = ()
    # consensus-cluster key -> per-layer raw-detection count / mean confidence,
    # as DENSE tuples aligned to `layers` (index == layer index).
    cluster_layer_counts: Dict[Tuple[int, float, float], Tuple[int, ...]] = field(default_factory=dict)
    cluster_layer_conf: Dict[Tuple[int, float, float], Tuple[float, ...]] = field(default_factory=dict)
    # Directory-wide naive counters, one entry per layer, accumulated in the
    # same pass as the union matching (see recompute_layered_matching_steps --
    # they cost no extra distance computations).
    layer_matched: Tuple[int, ...] = ()             # naive TP per layer
    layer_unmatched: Tuple[int, ...] = ()           # naive FP per layer
    layer_unmatched_labels: Tuple[int, ...] = ()    # naive FN per layer
    matching_version: int = 0
    # Memo for the three directory-wide stats. Rebuilding these on every
    # render tick (three full passes over every cluster in the directory,
    # ~0.2s/tick measured on the full 2,120-frame sample flight) would starve
    # cv2.waitKey() and make the window unresponsive. They are provably
    # invariant between mutations, so this memo is exact: it is keyed on the
    # two version counters, not on a time budget. Deliberately stored on the
    # instance rather than keyed by id(dr) -- id() is reused once
    # App._arm_recompute evicts a DirectoryReview.
    _stats_memo: Dict[str, Tuple] = field(default_factory=dict)

    def _stats_key(self) -> Tuple[int, int]:
        return ((self.store.version if self.store is not None else 0), self.matching_version)

    def _memo(self, name: str, compute):
        key = self._stats_key()
        hit = self._stats_memo.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        value = compute()
        self._stats_memo[name] = (key, value)
        return value

    def unmatched_total(self) -> int:
        """RAW detection count -- feeds naive_counts()'s FP. NOT the review
        unit (clusters are); see progress()."""
        return self._memo("unmatched_total",
                           lambda: sum(len(v) for v in self.unmatched_by_frame.values()))

    def cluster_total(self) -> int:
        """Review-unit count (centroids), as opposed to unmatched_total()'s
        raw-detection count. No call site in this module -- the [?] HUD uses
        progress()'s total instead -- but the test suite asserts on it as the
        clustering-compression measure, so it stays."""
        return self._memo("cluster_total",
                           lambda: sum(len(v) for v in self.unmatched_clusters_by_frame.values()))

    def spawned_total(self) -> int:
        return self._memo("spawned_total",
                           lambda: sum(len(v) for v in self.spawned_by_frame.values()))

    def naive_counts(self) -> Tuple[int, int, int]:
        """(tp, fp, fn), scoped to this directory's review-eligible frames
        (frames with a detection -- label-only frames contribute
        nothing here, same scope this tool reviews). Deliberately RAW-
        detection-based regardless of clustering -- clustering is a review-
        efficiency device, not a redefinition of what "naive" means.

        NOT directly comparable to an exclusive one-to-one (Hungarian)
        matcher's naive counts: this tool's live matching
        (match_spatial_many_to_one) deliberately lets several nearby
        detections of one real target all count as matched, where a
        one-to-one assignment would let only the single nearest one win and
        count the rest as naive false positives -- so this tool's tp will
        tend to run higher and fp lower for the same directory/threshold."""
        def compute():
            tp = sum(len(v) for v in self.matched_by_frame.values())
            fp = self.unmatched_total()
            fn = sum(len(v) for v in self.unmatched_labels_by_frame.values())
            return tp, fp, fn
        return self._memo("naive_counts", compute)

    def corrected_counts(self) -> Tuple[int, int, int, int, int]:
        """(tp, fp, fn, n_confirmed_real, n_hard_negative) after folding in
        judgments made so far. Judgments are made once per CLUSTER CENTROID
        but counted here at full raw-detection resolution: a confirmed-real
        cluster moves ALL of its member detections from FP to TP (not just
        1) -- matching what export() actually writes out. Uncertain/still-
        unreviewed clusters stay counted as FP -- a running, monotonically
        improving lower-bound estimate, never an overclaim. FN is untouched
        HERE -- this function's contract really is FP->TP only. That is not the
        same as saying recall cannot move: `n_confirmed` is also consumed by
        corrected_prf() as the `c` term, because a confirmed-real unmatched
        detection is a real target the annotator omitted, and it therefore
        joins the positive class that recall's denominator counts."""
        def compute():
            tp, fp, fn = self.naive_counts()
            n_confirmed = n_hard = 0
            for frame_idx, centroids in self.unmatched_clusters_by_frame.items():
                for c in centroids:
                    j = self.store.get(c.frame_idx, c.x, c.y)
                    if j not in ("confirmed_real", "hard_negative"):
                        continue
                    n_members = len(self.cluster_members.get((c.frame_idx, c.x, c.y), [c]))
                    if j == "confirmed_real":
                        n_confirmed += n_members
                    else:
                        n_hard += n_members
            return tp + n_confirmed, fp - n_confirmed, fn, n_confirmed, n_hard
        return self._memo("corrected_counts", compute)

    def member_map(self) -> Dict[Tuple[int, float, float], List[Detection]]:
        """Every key export() is allowed to consider 'current' (not
        orphaned), mapped to the raw Detection(s) a judgment on that key
        should be written out for: a cluster centroid's key maps to all its
        raw member detections (full per-detection export resolution from a
        single review action); a spawned point's key maps to a singleton
        list containing itself (nothing to aggregate -- it IS the point).
        A spawned point can never be orphaned by a modified dataset the way a
        stale detection judgment could -- it lives entirely in
        review_log.jsonl / spawned_by_frame, not in any detection file."""
        out: Dict[Tuple[int, float, float], List[Detection]] = dict(self.cluster_members)
        for frame_idx, pts in self.spawned_by_frame.items():
            for p in pts:
                out[(frame_idx, p.x, p.y)] = [p]
        return out

    def progress(self) -> Tuple[int, int]:
        """(n_reviewed, n_total) in CLUSTERS -- the actual remaining-click
        count and the whole point of clustering. See unmatched_total() for
        the separate raw-detection count naive_counts() uses."""
        def compute():
            total = 0
            reviewed = 0
            for frame_idx, centroids in self.unmatched_clusters_by_frame.items():
                for c in centroids:
                    total += 1
                    if self.store.get(c.frame_idx, c.x, c.y) != "unreviewed":
                        reviewed += 1
            return reviewed, total
        return self._memo("progress", compute)

    def frame_progress(self, frame_idx: int) -> Tuple[int, int]:
        centroids = self.unmatched_clusters_by_frame.get(frame_idx, [])
        reviewed = sum(1 for c in centroids if self.store.get(c.frame_idx, c.x, c.y) != "unreviewed")
        return reviewed, len(centroids)


def cluster_frame_points(frame_idx: int, pts: List[Detection], image_path: str,
                          cluster_radius: float
                          ) -> Tuple[List[Detection], Dict[Tuple[int, float, float], List[Detection]]]:
    """Complete-linkage clustering of `pts` -- every pairwise distance
    *within* a cluster is guaranteed <= cluster_radius. (Deliberately NOT
    single-linkage/DBSCAN(min_samples=1): that chains unboundedly through a
    dense field of points -- verified on this dataset's densest real frame,
    it produced a 1633-member, 528x356px "cluster" at radius=40, spanning
    nearly half the frame. Complete-linkage bounds cluster diameter instead.)

    cluster_radius <= 0 disables clustering: every point becomes its own
    singleton cluster (today's per-point-review behavior), with the
    centroid exactly equal to the raw coordinate (mean of one float is
    exact, no rounding drift).

    Deterministic given a fixed `pts` order: members are bucketed by
    iterating `pts` in its OWN order (never scipy's internal linkage/label
    order) before averaging with plain left-to-right sum()/len(), so
    repeated calls on the same input reproduce bit-identical centroid
    floats -- the property that keeps JudgmentStore's exact-key JSONL
    lookup resumable session to session, same as it already is for raw
    points today.

    Returns (centroids, members) where `members` maps each centroid's
    (frame_idx, x, y) key to its raw member Detections.
    """
    if not pts:
        return [], {}
    if cluster_radius <= 0 or len(pts) == 1:
        labels = list(range(len(pts)))
    else:
        coords = np.array([[p.x, p.y] for p in pts], dtype=np.float64)
        labels = fcluster(linkage(coords, method="complete"), t=cluster_radius, criterion="distance")

    groups: Dict[int, List[Detection]] = {}
    for p, lbl in zip(pts, labels):
        groups.setdefault(int(lbl), []).append(p)

    centroids: List[Detection] = []
    members: Dict[Tuple[int, float, float], List[Detection]] = {}
    for group in groups.values():
        n = len(group)
        cx = sum(p.x for p in group) / n
        cy = sum(p.y for p in group) / n
        cconf = sum(p.confidence for p in group) / n
        centroid = Detection(frame_idx, image_path, cx, cy, cconf)
        centroids.append(centroid)
        members[(frame_idx, cx, cy)] = group
    return centroids, members


def migrate_legacy_judgments(dr: DirectoryReview) -> None:
    """One-time, idempotent promotion of judgments recorded against RAW
    detection coordinates (from a review session before clustering existed)
    to their new cluster centroid's key, so old review work isn't silently
    orphaned by export()'s centroid-only lookup -- which would otherwise
    truncate hard_negatives.txt/confirmed_detections.txt on the very next
    commit/save/quit. Only ever APPENDS new JSONL records (marked
    "auto_migrated": true for transparency) -- never rewrites or deletes
    anything. Idempotent: a cluster already carrying a centroid-level
    judgment (from a prior migration run, or a fresh human decision) is
    skipped -- including the common case where a cluster happens to be a
    singleton, whose centroid key is byte-identical to its one member's raw
    key, so no migration is even needed there.

    When a cluster's members carry disagreeing prior judgments (rare --
    ~1 in ~800 on this project's real review history), the cluster is left
    unreviewed rather than guessed at; a one-line note is printed."""
    n_migrated = n_conflicts = 0
    for frame_idx, centroids in dr.unmatched_clusters_by_frame.items():
        for c in centroids:
            ckey = (c.frame_idx, c.x, c.y)
            if ckey in dr.store.judgments:
                continue
            members = dr.cluster_members.get(ckey, [])
            prior = {dr.store.get(frame_idx, m.x, m.y) for m in members}
            prior.discard("unreviewed")
            if not prior:
                continue
            if len(prior) > 1:
                n_conflicts += 1
                continue
            # Carry the source of whichever member records the judgment
            # forward rather than stamping SOURCE_HUMAN: a migration must not
            # silently launder a classifier-accepted call into a human one.
            verdict = next(iter(prior))
            sources = {judgment_source(dr.store.judgments.get((frame_idx, m.x, m.y)))
                       for m in members
                       if dr.store.get(frame_idx, m.x, m.y) == verdict}
            dr.store.commit(c.frame_idx, c.x, c.y, c.confidence, verdict,
                             dr.image_path_by_frame.get(frame_idx, ""),
                             extra={"auto_migrated": True,
                                    "source": sources.pop() if len(sources) == 1 else SOURCE_HUMAN})
            n_migrated += 1
    if n_migrated or n_conflicts:
        note = f", {n_conflicts} cluster(s) had disagreeing prior judgments -- left unreviewed" if n_conflicts else ""
        print(f"  {dr.rel_name}: migrated {n_migrated} pre-clustering judgment(s) to cluster centroids{note}")


# ---------------------------------------------------------------------------
# Run resolution: deriving a whole review session from one run directory.
#
# Nothing this GUI reads from the run is expected to change while it is open,
# except new dets/ files (see LAYER_POLL_SECONDS).
# ---------------------------------------------------------------------------






def fmt_num(value, spec: str = ".4f", missing: str = "n/a") -> str:
    """Format a number that a future/foreign schema may legitimately hand us
    as None or as a non-numeric string, without raising.

    `dict.get(key, 0)` returns None for a key that is PRESENT with a null
    value -- the default only covers a MISSING key -- so a format spec applied
    straight to it raises TypeError, and (main() has no except around
    render()) that would kill the whole GUI the moment [p] is pressed."""
    if value is None:
        return missing
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


STATS_REQUIRED_KEYS = ("per_directory", "overall")






@dataclass
class RunContext:
    """Everything a `results/<run>/` directory implies for a review session.

    WHY: a run directory already records how its detections were produced;
    reading it back means separate path/parameter flags cannot disagree with
    it. Using the parser default spatial_threshold=30 on detections that were
    produced and selected at 80 would silently review them under different
    matching. What is derived is spatial_threshold -- read from
    run_manifest.json's winning_hyperparameters -- plus coverage, the test
    flight and the layer metadata.

    Pure data: no GUI state, no side effects, so it is testable without a window.
    `problems` are fatal (refuse to start); `notes` are informational.
    """
    run: Optional[Path] = None
    # The run's gui/ artifacts are not read by this GUI.
    spatial_threshold: Optional[float] = None
    matching: Optional[str] = None
    coverage: Tuple[str, ...] = ()
    test_dir: Optional[str] = None
    scores_honest_for_test_dir: bool = False
    # Model layers (see the LayerSpec block below). `layers` is filled per
    # FLIGHT, in App._dr, because which files exist depends on rel_name;
    # resolve_run_artifacts only records the run-level metadata and the shape.
    layer_source: Optional[str] = None
    preregistration: Optional[dict] = None
    selection_pooled: Optional[dict] = None
    layer_notes: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def looks_like_run(p) -> bool:
    """A results/<run>/ directory. Identified by CONTENT, not by position on the
    command line, so the two positionals can be given in either order and a swap
    is an error rather than a silent misload."""
    p = Path(p)
    return p.is_dir() and ((p / "run_manifest.json").exists()
                           or (p / "gui").is_dir()
                           # A LAYERED run: per-checkpoint detection files and
                           # (optionally) a sealed preregistration. A run can
                           # have none of the three markers above, and without
                           # these it would be misclassified as a flight
                           # directory.
                           or (p / "PREREGISTRATION.json").exists()
                           or (p / "selection_pooled.json").exists()
                           or (p / "dets").is_dir()
                           or any(p.glob("*results*.json")))


def looks_like_dataset(p) -> bool:
    """A flight directory, identified by its GROUND TRUTH.

    Detections come from a run's per-checkpoint `dets/` files and never from
    the flight, so the flight's defining artifact is `labels.txt` -- the
    projected RTK survey, which is the one thing a run cannot supply (ground
    truth does not belong to a run).

    Unambiguous against looks_like_run(): no run directory carries a
    labels.txt, and classify_paths tests looks_like_run FIRST anyway.
    REVIEWABILITY is not implied by this predicate. It is decided by whether
    the run has detection files for the flight; see discover_directories and
    App._layers_for."""
    p = Path(p)
    return p.is_dir() and (p / "labels.txt").exists()




def resolve_run_artifacts(run) -> RunContext:
    """Derive a whole review session from one run directory, tolerating every
    absence.
    """
    run = Path(run)
    ctx = RunContext(run=run)
    if not run.is_dir():
        ctx.problems.append(f"run directory does not exist: {run}")
        return ctx

    manifest = {}
    mpath = run / "run_manifest.json"
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text())
        except (OSError, json.JSONDecodeError) as e:
            ctx.notes.append(f"run_manifest.json unreadable ({e})")

    # COVERAGE: which flights this run holds detections for, read from the
    # detection files themselves.
    #
    # A flat dets/ run holds one flight's checkpoints with the flight named
    # nowhere in the filenames, so there is nothing to enumerate and coverage
    # stays empty -- filled in below from PREREGISTRATION.json when the run
    # sealed one. Coverage only gates an advisory "this run does not cover
    # that flight" warning, so an empty tuple suppresses a warning rather than
    # blocking a review, which is the right way round.
    ctx.coverage = tuple(sorted(
        d.name.replace("__", "/") for d in run.iterdir()
        if d.is_dir() and d.name not in ("dets", "gui", "logs", "review")
        and any(d.glob("val_*.txt"))))

    # `matching` is NOT taken from the manifest: the manifest's
    # params.matching is the REPORTING convention ("both"), not what the
    # detections were matched with. There is no honest source for it, so it
    # stays None rather than being filled from a field that means something
    # else -- [p] simply omits the line.
    ctx.test_dir = manifest.get("test_dir")
    ctx.scores_honest_for_test_dir = False

    # Model layers: run-level metadata only. The per-flight file list needs a
    # rel_name and is resolved in App._dr.
    ctx.preregistration, ctx.selection_pooled, ctx.layer_notes = load_layer_metadata(run)
    if (run / "dets").is_dir():
        ctx.layer_source = "dets"
    elif any(d.is_dir() and any(d.glob("val_*.txt")) for d in run.iterdir()):
        ctx.layer_source = "flight_subdir"
    prereg = ctx.preregistration or {}
    if prereg:
        # A sealed preregistration names its own held-out flight.
        ctx.test_dir = ctx.test_dir or prereg.get("test_dir")
        if not ctx.coverage and prereg.get("test_dir"):
            ctx.coverage = (prereg["test_dir"],)
        # A holdout run's whole point is that nothing touched the test flight,
        # and PREREGISTRATION.json records the scan that proves it. Without
        # this, a holdout run with a test_dir would print the "CONTAMINATED"
        # banner, which is exactly backwards for a sealed holdout.
        assertion = prereg.get("test_set_untouched_assertion") or {}
        if assertion.get("n_referencing_test_dir") == 0:
            ctx.scores_honest_for_test_dir = True
            ctx.notes.append(
                f"PREREGISTRATION.json asserts the test set was untouched "
                f"({assertion.get('n_detection_files_scanned')} detection files scanned, "
                f"0 referencing {prereg.get('test_dir')})")
    for n in ctx.layer_notes:
        ctx.notes.append(n)

    # spatial_threshold: what the detections were actually produced at,
    # from run_manifest.json's winning_hyperparameters.
    st = (manifest.get("winning_hyperparameters") or {}).get("spatial_threshold")
    if st is None:
        ctx.notes.append("run records no spatial_threshold; using the GUI default")
    else:
        try:
            ctx.spatial_threshold = float(st)
        except (TypeError, ValueError):
            ctx.notes.append(f"run records a non-numeric spatial_threshold ({st!r})")
    return ctx


@dataclass(frozen=True)
class ModelMetricRow:
    """One model layer's scorecard, derived from the single shared
    adjudication pass. Pure data, so the smoke test can assert on the bound
    glyph and the column alignment without scraping pixels."""
    layer_id: str
    label: str
    verdict: str                       # healthy | collapsed | sparse | unknown
    arms: Tuple[str, ...]
    visible: bool
    n_dets: int                        # this layer's raw detections in the directory
    n_clusters_fired: int              # consensus clusters this layer contributed to
    pre: Tuple[float, float, float]    # point estimates
    post: Tuple[float, float, float]   # post[1], post[2] are UPPER BOUNDS
    n_confirmed: int                   # this layer's own confirmed-real detections (the `c` term)
    n_hard: int
    # Consensus clusters confirmed real that this layer did NOT fire on. See
    # layer_metric_rows() for why this is reported beside the headline bound
    # rather than folded into it.
    n_confirmed_elsewhere: int


def layer_naive_counts(dr: DirectoryReview) -> List[Tuple[int, int, int]]:
    """(tp, fp, fn) per layer, straight off the counters the layered recompute
    accumulated. RAW-detection based, the same deliberate choice naive_counts()
    documents -- clustering is a review-efficiency device, not a redefinition
    of what "naive" means."""
    n = len(dr.layers)
    if not n or len(dr.layer_matched) != n:
        return []
    return [(dr.layer_matched[i], dr.layer_unmatched[i], dr.layer_unmatched_labels[i])
            for i in range(n)]


def layer_corrected_counts(dr: DirectoryReview) -> List[Tuple[int, int, int, int, int]]:
    """Per-layer (tp, fp, fn, n_confirmed, n_hard) -- corrected_counts()'s
    contract, restricted to each layer's own member detections.

    One pass over the consensus clusters, reading the dense per-layer count
    tuple the recompute already built, so this is O(clusters x layers) of
    integer adds rather than a rematch."""
    n = len(dr.layers)
    naive = layer_naive_counts(dr)
    if not naive:
        return []
    conf = [0] * n
    hard = [0] * n
    for frame_idx, centroids in dr.unmatched_clusters_by_frame.items():
        for c in centroids:
            key = (c.frame_idx, c.x, c.y)
            j = dr.store.get(*key) if dr.store is not None else "unreviewed"
            if j not in ("confirmed_real", "hard_negative"):
                continue
            counts = dr.cluster_layer_counts.get(key)
            if counts is None:
                continue
            tgt = conf if j == "confirmed_real" else hard
            for k in range(n):
                tgt[k] += counts[k]
    return [(naive[k][0] + conf[k], naive[k][1] - conf[k], naive[k][2], conf[k], hard[k])
            for k in range(n)]


def layer_metric_rows(dr: DirectoryReview, visible: Optional[Set[str]] = None
                      ) -> List[ModelMetricRow]:
    """Every layer's PRE/POST scorecard, from ONE adjudication pass.

    This is the point of the whole feature: a reviewer judges each physical
    spot once, and eleven checkpoints get scored by it.

    CONVENTIONS, which are deliberate and not style:
      - Precision AND recall are always reported as a naive/adjudicated PAIR.
      - POST recall (and the F1 built from it) is an UPPER BOUND, never a point
        estimate. It comes from corrected_prf() -- the same function the
        single-source HUD uses, called once per layer, so the bound semantics
        cannot drift between the two.
      - `n_added` is SHARED across every layer. A human-placed target is a miss
        for all of them identically, which is exactly what the `a` term means.

    ONE THING THIS FEATURE CREATES THAT DID NOT EXIST BEFORE, and that is
    deliberately NOT folded into the headline number. When a consensus cluster
    is confirmed real and layer L never fired there, L has a demonstrably
    discoverable miss -- impossible in single-source review, where
    corrected_prf reasons that "an unlabelled positive is discoverable only
    where the detector fired" and `c` can therefore only raise the ratio. That
    reasoning is now false across layers. Rather than silently redefine R<=,
    this reports the project-conventional bound and carries
    `n_confirmed_elsewhere` alongside, so the tighter cross-layer bound

        (tp0 + c) / (tp0 + fn0 + c + a + n_confirmed_elsewhere)

    can be shown as its own labelled quantity. Two numbers, both honest;
    neither quietly replaces the other.
    """
    n = len(dr.layers)
    cc = layer_corrected_counts(dr)
    if not cc:
        return []
    naive = layer_naive_counts(dr)
    n_added = dr.spawned_total()
    vis = visible if visible is not None else {s.layer_id for s in dr.layers}

    n_fired = [0] * n
    elsewhere = [0] * n
    for frame_idx, centroids in dr.unmatched_clusters_by_frame.items():
        for c in centroids:
            key = (c.frame_idx, c.x, c.y)
            counts = dr.cluster_layer_counts.get(key)
            if counts is None:
                continue
            confirmed = (dr.store.get(*key) == "confirmed_real") if dr.store is not None else False
            for k in range(n):
                if counts[k]:
                    n_fired[k] += 1
                elif confirmed:
                    elsewhere[k] += 1

    rows: List[ModelMetricRow] = []
    for k, spec in enumerate(dr.layers):
        tp0, fp0, fn0 = naive[k]
        tp1, fp1, _fn1, n_conf, n_hard = cc[k]
        rows.append(ModelMetricRow(
            layer_id=spec.layer_id, label=spec.label, verdict=spec.verdict, arms=spec.arms,
            visible=spec.layer_id in vis,
            n_dets=tp0 + fp0, n_clusters_fired=n_fired[k],
            pre=prf(tp0, fp0, fn0),
            post=corrected_prf(tp0, fp0, fn0, tp1, fp1, n_conf, n_added),
            n_confirmed=n_conf, n_hard=n_hard, n_confirmed_elsewhere=elsewhere[k]))
    return rows


def layer_cross_bound_recall(row: ModelMetricRow, dr: DirectoryReview) -> float:
    """The tighter cross-layer recall bound described in layer_metric_rows().
    Separate function, separate label on screen: it is a DIFFERENT estimand
    from the R<= the rest of this project reports, and must never be
    substituted for it silently."""
    naive = layer_naive_counts(dr)
    idx = next((i for i, s in enumerate(dr.layers) if s.layer_id == row.layer_id), None)
    if idx is None:
        return 0.0
    tp0, _fp0, fn0 = naive[idx]
    den = tp0 + fn0 + row.n_confirmed + dr.spawned_total() + row.n_confirmed_elsewhere
    return (tp0 + row.n_confirmed) / den if den else 0.0


# ---------------------------------------------------------------------------
# Model layers: reviewing every detection file of a run at once.
#
# A "layer" is one detection file produced by run_detector.py for one CNN
# checkpoint (and, where a single forward pass was harvested at several
# thresholds, one threshold). Two on-disk shapes exist and both are read:
#
#   results/17_holdout_pieranch/dets/val_<ckpt>[_t<thresh>].txt      (holdout)
#   results/<run>/<flight_slug>__f500_t0.03/val_*.txt                (screen)
#
# The format is identical to labels.txt -- `frame_idx image_path x,y,conf`,
# read by the same parse_annotations -- with one consequential difference:
# these carry REAL confidences, while labels.txt hardcodes 1.0.
#
# WHY THIS EXISTS: screening 30 checkpoints finds that 19 are degenerate and
# that no single number picks a winner -- the flights disagree, F1 hides
# collapse, and a dead model ranks as the most precise one. Comparing
# checkpoints one GUI session at a time cannot be reduced to a scalar. So:
# all layers at once, one adjudication per physical spot, per-model scores
# derived from that one pass.
# ---------------------------------------------------------------------------

# Detection files are named val_<ckpt>.txt or val_<ckpt>_t<thresh>.txt by
# run_detector.py (`{t:g}` formatting, hence t0.03 for 0.030 and
# t0.035 for 0.035).
LAYER_FILE_RE = re.compile(r"^val_(?P<ckpt>[0-9]+)(?:_t(?P<thresh>[0-9.]+))?$")


@dataclass(frozen=True)
class LayerSpec:
    """One model layer: a detection file plus everything known about the
    checkpoint that produced it.

    `verdict` is the degeneracy screen's health call (healthy / collapsed /
    sparse). It is DISPLAYED AND NEVER FILTERED ON: the degenerate checkpoints
    are the single most important structural finding of a model screen. A
    collapsed model must be visible and labelled as collapsed, never quietly
    dropped.

    Frozen and pure so the layer set can be a cache key and the smoke test can
    build one without touching disk.
    """
    layer_id: str                      # file stem: "val_013", "val_030_t0.06"
    path: Path
    ckpt: str                          # "030"
    thresh: Optional[float]            # 0.06, or None when the name carries none
    label: str                         # "030 t.06" -- short, sized for the HUD
    verdict: str = "unknown"           # healthy | collapsed | sparse | unknown
    arms: Tuple[str, ...] = ()         # confirmatory arms this checkpoint serves
    config: Optional[dict] = None      # eps / min_samples / eps_predict
    selection: Optional[dict] = None   # pooled recall / precision_like / f1 / ...
    source: str = ""                   # "dets" | "flight_subdir"

    @property
    def degenerate(self) -> bool:
        return self.verdict in ("collapsed", "sparse")


@dataclass
class LayerDetection(Detection):
    """A Detection that remembers which model layer emitted it.

    THE LAYER CANNOT RIDE ON A COORDINATE KEY, which is the idiomatic-looking
    mistake here: point_to_cluster_key and the propagator's member_label are
    both keyed on (frame_idx, x, y), so a Dict[key -> layer] would sit
    perfectly naturally beside them. Measured on the full sample flight's
    holdout run, it would also be wrong: 7,738 of 224,758 detections (3.4%)
    are EXACT coordinate coincidences between layers, up to six deep, because
    several checkpoints legitimately fire on the same pixel. A coordinate dict
    silently keeps the last writer and loses the rest of the attribution.

    `layer` needs a default because detections.Detection already defaults
    `confidence`, and a dataclass may not follow a defaulted field with a
    bare one.
    """
    layer: int = -1


def flight_slug(rel_name: str) -> str:
    """"2025_04_16/pieranch_rect" -> "2025_04_16__pieranch_rect".

    The naming used for a run's per-flight detection directories, and the
    inverse of the one resolve_run_artifacts applies when reading a run's
    coverage back."""
    return rel_name.replace("/", "__")


def _layer_label(ckpt: str, thresh: Optional[float]) -> str:
    if thresh is None:
        return ckpt
    # ".06" not "0.06": the label column is 150px of Hershey at scale 0.34 and
    # the leading zero buys nothing.
    return f"{ckpt} t{('%g' % thresh).lstrip('0')}"


def _dets_dir_covers(dets: Path, rel_name: str) -> bool:
    """Does this `dets/` directory hold detections for THIS flight?

    `dets/` is not named after a flight -- one run's dets/ belongs to whichever
    flight run_detector.py was pointed at, and the flight only shows up in the
    relative image_path on each row. So peek one row.

    A run whose dets/ covers a different flight would otherwise be silent:
    every layer agrees on the wrong flight, so no cross-layer disagreement is
    raised, and the result is one flight's detections drawn over another
    flight's image with a full HUD of numbers beside them. With dets/ the
    only detection source, that has to be caught up front.

    Cheap: reads at most a few lines of one file, never the 0.4-3.4 MB body.
    Permissive on anything it cannot determine -- an unreadable or headerless
    file is discover_layers' business to report, not this predicate's."""
    for f in sorted(dets.glob("val_*.txt")):
        try:
            with f.open() as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    # `<frame_idx> <relative_image_path> <x>,<y>,<conf>`
                    return parts[1].startswith(rel_name.rstrip("/") + "/")
        except OSError:
            continue
    return True


def layer_source_dir(run: Path, rel_name: str) -> Optional[Path]:
    """Where this run keeps its per-checkpoint detection files for this flight.

    Two shapes, checked in order. The second is matched with startswith, not
    equality: the per-flight directory name appends a `__f<frames>_t<thresh>`
    suffix to the slug, e.g. `2025_04_16__pieranch_rect__f500_t0.03`.

    The first shape (`dets/`) is CONTENT-CHECKED against the flight -- see
    _dets_dir_covers. The second needs no check: its name carries the slug."""
    dets = run / "dets"
    if dets.is_dir() and _dets_dir_covers(dets, rel_name):
        return dets
    slug = flight_slug(rel_name)
    for d in sorted(run.iterdir()) if run.is_dir() else []:
        if d.is_dir() and d.name.startswith(slug) and any(d.glob("val_*.txt")):
            return d
    return None


def layer_dir_signature(layer_dir: Optional[Path]) -> Tuple[Tuple[str, int, int], ...]:
    """Sorted (name, mtime_ns, size) over the layer files, for change detection.

    The same (mtime_ns, size) pair a per-file signature would use, lifted to a
    directory so
    that an ADDED file and a REWRITTEN one are one comparison.

    *.partial is EXCLUDED, and that is load-bearing rather than tidy:
    run_detector.py streams into `<name>.partial` and only Path.replace()s it
    into position on success, so a partial file's
    size changes on every poll. Including it would mean the signature never
    settles and the settle check in _poll_layer_dir would flap forever."""
    if layer_dir is None or not layer_dir.is_dir():
        return ()
    out = []
    for p in sorted(layer_dir.glob("val_*.txt")):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append((p.name, st.st_mtime_ns, st.st_size))
    return tuple(out)


def load_layer_metadata(run: Path) -> Tuple[Optional[dict], Optional[dict], List[str]]:
    """(preregistration, selection_pooled, notes) for a run directory.

    selection_pooled is resolved through PREREGISTRATION.provenance.selection
    when that is present, so a run points at the screen it was actually built
    from rather than at whatever happens to sit beside it. A sha256 mismatch is
    a NOTE, never a refusal: this is a viewer, and it shows data rather than
    gating on seals -- but it must say so, because a mismatched screen means
    the health verdicts on screen are not the ones the run was sealed against.
    """
    notes: List[str] = []
    prereg = sel = None
    pp = run / "PREREGISTRATION.json"
    if pp.exists():
        try:
            prereg = json.loads(pp.read_text())
        except (OSError, json.JSONDecodeError) as e:
            notes.append(f"PREREGISTRATION.json unreadable ({e})")

    cands: List[Path] = []
    if prereg:
        rel = (prereg.get("provenance", {}).get("selection", {}) or {}).get("path")
        if rel:
            cands.append(run.parent.parent / rel if not Path(rel).is_absolute() else Path(rel))
    cands.append(run / "selection_pooled.json")
    for c in cands:
        if not c.exists():
            continue
        try:
            sel = json.loads(c.read_text())
        except (OSError, json.JSONDecodeError) as e:
            notes.append(f"{c.name} unreadable ({e})")
            continue
        want = (prereg or {}).get("provenance", {}).get("selection", {}).get("sha256")
        if want:
            got = hashlib.sha256(c.read_bytes()).hexdigest()
            if got != want:
                notes.append(
                    f"{c.name} sha256 differs from the one PREREGISTRATION.json was sealed "
                    f"against -- health verdicts shown are from the CURRENT file, not the sealed one")
        break
    return prereg, sel, notes


def discover_layers(run: Path, rel_name: str,
                    prereg: Optional[dict] = None,
                    selection: Optional[dict] = None) -> Tuple[List[LayerSpec], List[str]]:
    """Every model layer this run holds for this flight, plus notes on what was
    refused and why.

    Deterministic order -- (ckpt, thresh, layer_id) -- because the palette
    indexes into this list, and a layer appearing mid-session must not
    recolour the ones already on screen.

    REFUSALS, each noted rather than raised:
      *.partial      still being written by run_detector.py:
                     parse_annotations would read a truncated file happily and
                     score it as a low-detection model.
      empty / unparseable
      an unrecognised val_* name
    A run root's stray val_*.txt is never reached, because layer_source_dir
    only ever returns a subdirectory -- those files are unattributable to a
    flight and scoring them here would silently mix another flight's
    detections into this review.
    """
    notes: List[str] = []
    src = layer_source_dir(run, rel_name)
    if src is None:
        return [], notes
    source = "dets" if src.name == "dets" else "flight_subdir"

    for p in sorted(src.glob("val_*.partial")):
        notes.append(f"{p.name}: still being written by run_detector.py -- refused")

    specs: List[LayerSpec] = []
    pooled = (selection or {}).get("pooled", {}) or {}
    ck_cfg = (prereg or {}).get("checkpoints", {}) or {}
    # Keyed by (ckpt, thresh), not by ckpt: PREREGISTRATION pins each
    # confirmatory arm to one checkpoint AT ONE THRESHOLD. 030 serves
    # max_recall at t=0.04 only, so tagging its t=0.06 layer as an arm too
    # would misreport the sealed preregistration on screen.
    arms_by_cfg: Dict[Tuple[str, Optional[float]], List[str]] = {}
    for arm, spec in ((prereg or {}).get("confirmatory_arms", {}) or {}).items():
        at = (spec.get("config") or {}).get("thresh")
        arms_by_cfg.setdefault((str(spec.get("ckpt")), None if at is None else float(at)),
                               []).append(arm)

    for p in sorted(src.glob("val_*.txt")):
        m = LAYER_FILE_RE.match(p.stem)
        if m is None:
            notes.append(f"{p.name}: name does not parse as val_<ckpt>[_t<thresh>] -- skipped")
            continue
        try:
            if p.stat().st_size == 0:
                notes.append(f"{p.name}: empty -- skipped")
                continue
        except OSError as e:
            notes.append(f"{p.name}: unreadable ({e}) -- skipped")
            continue
        ckpt = m.group("ckpt")
        thresh = float(m.group("thresh")) if m.group("thresh") else None
        # Health is keyed by the BARE checkpoint tag, so val_030_t0.04 and
        # val_030_t0.06 share one verdict. They are two layers but ONE
        # checkpoint, and counting them as two degenerate checkpoints would
        # double-count the report's headline "19 of 30" finding.
        rec = pooled.get(ckpt) or {}
        cfgs = (ck_cfg.get(ckpt) or {}).get("configs") or []
        cfg = None
        for c in cfgs:
            if thresh is None or abs(float(c.get("thresh", -1)) - thresh) < 1e-9:
                cfg = c
                break
        specs.append(LayerSpec(
            layer_id=p.stem, path=p, ckpt=ckpt, thresh=thresh,
            label=_layer_label(ckpt, thresh),
            verdict=str(rec.get("verdict", "unknown")),
            arms=tuple(sorted(set((cfg or {}).get("arms", []) or [])
                              | set(arms_by_cfg.get((ckpt, thresh), []))
                              | (set(arms_by_cfg.get((ckpt, None), [])) if thresh is None else set()))),
            config=cfg, selection=rec or None, source=source))

    specs.sort(key=lambda s: (s.ckpt, -1.0 if s.thresh is None else s.thresh, s.layer_id))
    return specs, notes


def draw_badge(img, text: str, org: Tuple[int, int], color, scale: float = 0.5) -> None:
    """Draw a short text badge that stays readable on ANY background.

    Plain putText at thickness 1 was measured illegible on this dataset: a
    1px glyph over sunlit dirt gave a mean |delta luma| of only 27-34. The fix
    is the standard halo -- black at thickness 3 first, colour at thickness 2
    on top -- so the glyph carries its own contrast regardless of what is
    underneath.

    The "xN" cluster-size badge uses it -- LIVE CLUSTER DATA computed from
    the sliders.
    """
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# [m] propagation overlay geometry.
#
# The RING communicates "here is a candidate label for this cluster, before
# you commit it"; its source is a human verdict propagated across frames.
# See propagated_label().
# ---------------------------------------------------------------------------

# A ring drawn around (not touching) each cluster centroid. Radius and
# thickness sit clearly outside the centroid's own filled dot (radius 6) with
# a visible gap, and clearly inside the yellow selection ring (radius
# MODEL_HALO_RADIUS_PX + a few px -- see the selection-ring draw call), so the
# two never merge into one indistinguishable ring on a selected cluster.
MODEL_HALO_RADIUS_PX = 11
MODEL_HALO_THICKNESS_PX = 3








# Rows parsed per generator yield. Sized so one chunk lands near
# CHUNK_TIME_BUDGET_SECONDS rather than far over it: at 20k rows a single
# chunk measured ~0.8s on the real 212MB CSV, which is one 0.8s freeze per
# reload -- the budget loop can only preempt BETWEEN yields, so the chunk
# size is the real latency floor.
CSV_PARSE_CHUNK_ROWS = 4000
















def cluster_geo_clusters(dr: DirectoryReview, ckey: Tuple[int, float, float],
                         member_label: Dict[Tuple[int, float, float], int]) -> List[int]:
    """The geo cluster ids a GUI cluster's raw members land in, deduplicated.

    Normally one: a GUI cluster is a tight complete-linkage blob within a
    single frame, so its members share a 3 cm cell neighbourhood and DBSCAN
    puts them together. It can be more than one when the blob straddles two
    geo clusters, and it can be EMPTY when every member is DBSCAN noise or
    has no resolvable pose -- geo_metrics.locate() conflates those two and so
    does this, because the caller cannot act on the difference.

    Returns them in first-seen member order, which is stable across renders
    because cluster_members preserves parse order."""
    out: List[int] = []
    for m in dr.cluster_members.get(ckey, []):
        lab = member_label.get((m.frame_idx, m.x, m.y), -1)
        if lab >= 0 and lab not in out:
            out.append(lab)
    return out


def propagated_label(dr: DirectoryReview, ckey: Tuple[int, float, float],
                     verdicts, member_label: Dict[Tuple[int, float, float], int],
                     rule: Optional[str]) -> Optional[dict]:
    """What a human adjudication elsewhere implies about THIS cluster, or None.

    This is the label propagator. It is entirely user-driven: every input is a
    human verdict, and the only machine involvement is the geometry that
    decides which verdicts are about the same physical spot.

    HOW IT PROPAGATES ACROSS FRAMES. A geo cluster is a DBSCAN group in UTM
    (geo_metrics.build_scaffold, eps=0.20 m) built from every frame's
    detections, so it spans frames BY CONSTRUCTION -- one burrow seen in
    frames 700 and 830 is one cluster. Confirming it in frame 700 therefore
    says something about the detection in frame 830, and that is the whole
    mechanism. Nothing here re-clusters or re-projects; the join was already
    computed once into `member_label`.

    IT RETURNS None RATHER THAN GUESSING, in four distinct situations, and the
    caller must treat all four the same way -- leave the cluster unreviewed,
    orange, with no ring:

      * `rule` is None -- no tie-break convention has been chosen ([o] is
        off). There is deliberately no default; see VERDICT_RULES.
      * the cluster reaches no geo cluster (`cluster_geo_clusters` empty).
      * no human has judged anything in its geo cluster, or only abstentions
        have landed there (GeoVerdicts.verdict -> None).
      * the rule itself abstains -- `majority` on a tie, where "geometry
        stands" and there is no majority to propagate.

    A cluster straddling two geo clusters propagates only if they AGREE.
    Disagreement is not a tie to be broken: the two verdicts are about
    different physical spots that this GUI cluster happens to bridge, and
    picking one would be inventing a claim no human made.

    Returns {"judgment", "rule", "geo_clusters", "votes", "n_members"} where
    `votes` is the summed per-judgment tally over the contributing geo
    clusters -- carried so [a] can record WHY it wrote what it wrote, and so
    the selected-point panel can show the evidence rather than just the
    conclusion."""
    if rule is None or verdicts is None or not member_label:
        return None
    labs = cluster_geo_clusters(dr, ckey, member_label)
    if not labs:
        return None
    calls = {lab: verdicts.verdict(lab, rule) for lab in labs}
    decided = {v for v in calls.values() if v is not None}
    if len(decided) != 1:
        # Empty: nobody judged this spot, or the rule abstained.
        # More than one: two geo clusters disagree -- see the docstring.
        return None
    votes: Dict[str, int] = {}
    for lab in labs:
        if calls[lab] is None:
            continue
        for j, n in (verdicts.votes.get(lab) or {}).items():
            votes[j] = votes.get(j, 0) + n
    return {"judgment": next(iter(decided)), "rule": rule,
            "geo_clusters": labs, "votes": votes,
            "n_members": len(dr.cluster_members.get(ckey, []))}


def recompute_matching_steps(dr: DirectoryReview, dets_fd: Dict[int, FrameData], labels_fd: Dict[int, FrameData],
                              spatial_threshold: float, cluster_radius: float
                              ) -> Iterator[Tuple[int, int]]:
    """Generator form of recompute_matching(): identical resets + identical
    per-frame body (matching via match_spatial_many_to_one + clustering via
    cluster_frame_points -- pure coordinates, no image I/O, no dependency
    on any precomputed artifact), but yields
    (frames_done, frames_total) once up front and again after every frame,
    so a caller (the GUI render loop) can drive it in small time-budgeted
    chunks across many render() ticks instead of blocking for the whole
    ~13-16s worst-case cost in one call -- see App._step_chunked_recompute().
    recompute_matching() below is just `for _ in this: pass`, so it's the
    only place this loop body lives.

    Callable twice on the same DirectoryReview (e.g. a live slider change):
    resets every per-frame dict first, since the loop body accumulates via
    .setdefault(...).append(...)/.update(...) and would otherwise double up
    on a second call. Deliberately does NOT touch dr.store or
    dr.spawned_by_frame -- both are independent of matching/clustering
    (spawned points live only in review_log.jsonl, never in a detection file)
    and must survive a recompute untouched."""
    dr.labels_by_frame = {}
    dr.matched_by_frame = {}
    dr.unmatched_by_frame = {}
    dr.unmatched_clusters_by_frame = {}
    dr.cluster_members = {}
    dr.point_to_cluster_key = {}
    dr.unmatched_labels_by_frame = {}
    dr.image_path_by_frame = {}
    # Every dict the directory-wide stats memo depends on was just reset and
    # is about to be repopulated frame by frame -- invalidate the memo now,
    # so a stats read at any point during or after this generator recomputes
    # rather than returning pre-recompute numbers.
    dr.matching_version += 1

    total = len(dr.frames_all)
    yield 0, total
    for i, frame_idx in enumerate(dr.frames_all):
        det_points = dets_fd[frame_idx].points
        dr.image_path_by_frame[frame_idx] = dets_fd[frame_idx].image_path
        gt_fd = labels_fd.get(frame_idx)
        gt_points = gt_fd.points if gt_fd else []
        matches, unmatched_idx, unmatched_gt_idx = match_spatial_many_to_one(det_points, gt_points, spatial_threshold)
        matched_idx = {di for di, gi, score in matches}
        if gt_points:
            dr.labels_by_frame[frame_idx] = gt_points
        if matched_idx:
            dr.matched_by_frame[frame_idx] = [det_points[i2] for i2 in sorted(matched_idx)]
        if unmatched_idx:
            unmatched_pts = [det_points[i2] for i2 in unmatched_idx]
            dr.unmatched_by_frame[frame_idx] = unmatched_pts
            centroids, members = cluster_frame_points(frame_idx, unmatched_pts, dets_fd[frame_idx].image_path,
                                                        cluster_radius)
            dr.unmatched_clusters_by_frame[frame_idx] = centroids
            dr.cluster_members.update(members)
            for ckey, group in members.items():
                for p in group:
                    dr.point_to_cluster_key[(p.frame_idx, p.x, p.y)] = ckey
        if unmatched_gt_idx:
            dr.unmatched_labels_by_frame[frame_idx] = [gt_points[i2] for i2 in unmatched_gt_idx]
        yield i + 1, total
    # Bump again on completion, not just at the reset above: the dicts are
    # populated incrementally as this generator is driven, so anything that
    # read a stat MID-recompute would otherwise have memoized a
    # partially-populated result under a version that never changes again.
    # Nothing reads stats mid-flight today (render() short-circuits to the
    # progress banner while _recompute_gen is live), but this file has been
    # bitten twice by exactly that kind of "nothing does that today"
    # assumption in its render caches -- so make it unconditional.
    dr.matching_version += 1


def recompute_matching(dr: DirectoryReview, dets_fd: Dict[int, FrameData], labels_fd: Dict[int, FrameData],
                        spatial_threshold: float, cluster_radius: float, verbose: bool = False) -> None:
    """Blocking, complete-in-one-call convenience wrapper around
    recompute_matching_steps() -- the contract every existing caller (incl.
    load_layered_directory() and the test suite) relies on. See
    recompute_matching_steps() for the resumable/chunked form the live GUI
    uses instead."""
    if verbose:
        print(f"  matching {dr.rel_name}: {len(dr.frames_all)} frames ...")
    t0 = time.monotonic()
    for i, total in recompute_matching_steps(dr, dets_fd, labels_fd, spatial_threshold, cluster_radius):
        if verbose and i and i % 200 == 0:
            print(f"    ...{i}/{total} frames matched ({time.monotonic()-t0:.1f}s)")


def healthy_union_frames(dr: DirectoryReview) -> Dict[int, FrameData]:
    """dr.dets_fd with every DEGENERATE layer's detections filtered out.

    THE ONE DEFINITION OF "THE UNION" for everything that forms review targets
    or propagates labels. dr.dets_fd itself stays complete -- Tier 2 rendering,
    the per-layer scorecards, the per-layer geo scaffolds and the 3D cloud all
    need every layer, including the broken ones, because a collapsed
    checkpoint held the highest precision-like score in the whole screen and
    must stay visible and labelled.

    But a degenerate checkpoint must not manufacture work: it creates no
    cluster to click (recompute_layered_matching_steps) and no link in a
    DBSCAN chain (the union geo scaffold). Those two callers share this
    function so they cannot disagree about which detections "the union" holds
    -- if they did, a cluster could exist to be judged with no geo cluster to
    propagate through, or the reverse.

    Returns dr.dets_fd itself when nothing is degenerate, which is the common
    case on a holdout run and avoids copying 234,646 points for nothing.
    """
    if not dr.layers or not any(sp.degenerate for sp in dr.layers):
        return dr.dets_fd
    healthy = {li for li, sp in enumerate(dr.layers) if not sp.degenerate}
    out: Dict[int, FrameData] = {}
    for fi, fd in dr.dets_fd.items():
        pts = [p for p in fd.points if getattr(p, "layer", -1) in healthy]
        # The frame is KEPT even with no points: image_path_by_frame and the
        # scaffold's frame walk both key off this dict, and dropping a frame
        # would silently shorten the flight.
        out[fi] = FrameData(fi, fd.image_path, pts)
    return out


def recompute_layered_matching_steps(dr: DirectoryReview, dets_fd: Dict[int, FrameData],
                                      labels_fd: Dict[int, FrameData],
                                      spatial_threshold: float, cluster_radius: float
                                      ) -> Iterator[Tuple[int, int]]:
    """recompute_matching_steps() for a LAYERED review: the union of every
    model layer, matched and clustered once, with the per-layer split derived
    from that single pass.

    Deliberately mirrors recompute_matching_steps() step for step -- same
    resets, same yield contract, same double version bump, same refusal to
    touch dr.store or dr.spawned_by_frame. The union semantics of every
    per-frame dict it fills are IDENTICAL, so naive_counts(), corrected_counts(),
    member_map(), the geo scaffold, the [d] cloud and the whole judgment path
    mean the same thing with or without layers; the layer axis is strictly
    additional.

    WHY ONE UNION PASS IS EXACT, NOT AN APPROXIMATION. In
    match_spatial_many_to_one the detection side is computed independently per
    detection, so a detection's matched/unmatched status does not depend on
    which other detections are in the list -- matching layer L alone and
    matching it inside the union give the same answer. The ground-truth side
    DOES accumulate across detections, which is exactly why
    match_spatial_many_to_one_layered collects a per-layer bitmask inside the
    existing distance test rather than leaving it to be reconstructed. The
    smoke test asserts the equivalence against per-layer matching directly.

    Measured on the full sample flight's holdout run (11 layers, 224,758
    detections): 1.06 s for all 2,092 frames, yielding 74,141 consensus
    clusters.

    CLUSTERS OVER dr.layers, NEVER OVER A VISIBILITY MASK. Toggling a layer is
    a draw-time filter; if it reached this function it would move centroids and
    silently re-key every judgment in every frame that layer touches.

    DEGENERATE LAYERS ARE EXCLUDED FROM THE UNION, AND ONLY FROM THE UNION.
    A collapsed or sparse checkpoint (see LayerSpec.degenerate and the
    degeneracy screen) contributes no points to the review pools -- so it
    cannot create a cluster you have to click, cannot move a centroid, and
    cannot pull a geo cluster into a longer DBSCAN chain. On a 30-checkpoint
    screen 19 of 30 were degenerate, and a collapsed model firing 34,817
    times on one flight would manufacture review burden out of a model
    already known to be broken.

    What it does NOT do is drop them from anything else, because a collapsed
    model can hold the highest precision-like score in a whole screen -- it
    must stay visible and labelled, never quietly disappear:
      * their PER-LAYER counters (lm/lu/lf below) are still accumulated over
        EVERY layer, so each degenerate checkpoint keeps its own PRE
        scorecard, its row in the HUD, its point in the health-coded PR
        figure, and its legend entry;
      * dr.dets_fd still holds their detections, so Tier 2 still draws them,
        the 3D cloud still shows their shape, and their own geo scaffold is
        still built;
      * their POST arms will simply equal their PRE, because none of their
        detections is in a cluster anyone can adjudicate. That is the honest
        answer -- nobody reviewed them -- and `n_confirmed_elsewhere` already
        reports the confirmed clusters a layer never fired on.
    """
    # Layer indices whose detections may form review targets. A property of
    # the run's sealed degeneracy screen, not of anything the reviewer can
    # toggle mid-session. healthy_union_frames() applies the same rule to the
    # geo scaffold; keep the two in step.
    healthy = {li for li, sp in enumerate(dr.layers) if not sp.degenerate}
    dr.labels_by_frame = {}
    dr.matched_by_frame = {}
    dr.unmatched_by_frame = {}
    dr.unmatched_clusters_by_frame = {}
    dr.cluster_members = {}
    dr.point_to_cluster_key = {}
    dr.unmatched_labels_by_frame = {}
    dr.image_path_by_frame = {}
    dr.cluster_layer_counts = {}
    dr.cluster_layer_conf = {}
    n_layers = len(dr.layers)
    lm = [0] * n_layers
    lu = [0] * n_layers
    lf = [0] * n_layers
    dr.matching_version += 1

    total = len(dr.frames_all)
    yield 0, total
    for i, frame_idx in enumerate(dr.frames_all):
        det_points = dets_fd[frame_idx].points
        dr.image_path_by_frame[frame_idx] = dets_fd[frame_idx].image_path
        gt_fd = labels_fd.get(frame_idx)
        gt_points = gt_fd.points if gt_fd else []
        det_layer = [getattr(p, "layer", -1) for p in det_points]
        matches, unmatched_idx, unmatched_gt_idx, gt_layer_mask = \
            match_spatial_many_to_one_layered(det_points, gt_points, spatial_threshold,
                                              det_layer, n_layers)
        matched_idx = {di for di, gi, score in matches}
        if gt_points:
            dr.labels_by_frame[frame_idx] = gt_points
        # THE PER-LAYER COUNTERS COUNT EVERY LAYER. Only the union POOLS below
        # are restricted to healthy layers -- see the docstring. Keeping these
        # two apart is the whole point: a degenerate checkpoint keeps its own
        # scorecard while creating no review burden.
        for i2 in matched_idx:
            if 0 <= det_layer[i2] < n_layers:
                lm[det_layer[i2]] += 1
        for i2 in unmatched_idx:
            if 0 <= det_layer[i2] < n_layers:
                lu[det_layer[i2]] += 1
        matched_keep = [i2 for i2 in sorted(matched_idx) if det_layer[i2] in healthy]
        if matched_keep:
            dr.matched_by_frame[frame_idx] = [det_points[i2] for i2 in matched_keep]
        unmatched_keep = [i2 for i2 in unmatched_idx if det_layer[i2] in healthy]
        if unmatched_keep:
            unmatched_pts = [det_points[i2] for i2 in unmatched_keep]
            dr.unmatched_by_frame[frame_idx] = unmatched_pts
            centroids, members = cluster_frame_points(frame_idx, unmatched_pts,
                                                       dets_fd[frame_idx].image_path, cluster_radius)
            dr.unmatched_clusters_by_frame[frame_idx] = centroids
            dr.cluster_members.update(members)
            for ckey, group in members.items():
                counts = [0] * n_layers
                csum = [0.0] * n_layers
                for p in group:
                    dr.point_to_cluster_key[(p.frame_idx, p.x, p.y)] = ckey
                    li = getattr(p, "layer", -1)
                    if 0 <= li < n_layers:
                        counts[li] += 1
                        csum[li] += p.confidence
                dr.cluster_layer_counts[ckey] = tuple(counts)
                dr.cluster_layer_conf[ckey] = tuple(
                    (csum[k] / counts[k]) if counts[k] else 0.0 for k in range(n_layers))
        if unmatched_gt_idx:
            dr.unmatched_labels_by_frame[frame_idx] = [gt_points[i2] for i2 in unmatched_gt_idx]
        # A ground-truth point is a FALSE NEGATIVE FOR LAYER L exactly when no
        # detection of layer L came within threshold of it -- which is bit L of
        # the mask, and is NOT the same as the union's unmatched_gt_idx: the
        # union finds a label that any layer found, and per-layer recall must
        # not inherit another checkpoint's success.
        for j in range(len(gt_points)):
            mask = gt_layer_mask[j]
            for k in range(n_layers):
                if not (mask >> k) & 1:
                    lf[k] += 1
        yield i + 1, total
    if dr.layers and not healthy:
        # EVERY checkpoint in this run is degenerate, so the union is empty and
        # there is nothing to click. Silence here would look identical to a
        # fully-reviewed flight (0 clusters, 0 remaining), which is the one
        # reading that must never happen by accident.
        print(f"  {dr.rel_name}: ALL {len(dr.layers)} layer(s) are degenerate "
              f"(collapsed/sparse) -- the union is EMPTY and there is nothing "
              f"to review. Their per-model scorecards are still computed.")
    dr.layer_matched = tuple(lm)
    dr.layer_unmatched = tuple(lu)
    dr.layer_unmatched_labels = tuple(lf)
    dr.matching_version += 1


def recompute_layered_matching(dr: DirectoryReview, dets_fd: Dict[int, FrameData],
                                labels_fd: Dict[int, FrameData], spatial_threshold: float,
                                cluster_radius: float, verbose: bool = False) -> None:
    """Blocking wrapper, mirroring recompute_matching()."""
    if verbose:
        print(f"  matching {dr.rel_name}: {len(dr.frames_all)} frames x {len(dr.layers)} layers ...")
    t0 = time.monotonic()
    for i, total in recompute_layered_matching_steps(dr, dets_fd, labels_fd,
                                                      spatial_threshold, cluster_radius):
        if verbose and i and i % 400 == 0:
            print(f"    ...{i}/{total} frames matched ({time.monotonic()-t0:.1f}s)")


def remap_cluster_judgments(dr: DirectoryReview,
                            prior_members: Dict[Tuple[int, float, float], List[Detection]]
                            ) -> Tuple[int, int, int]:
    """Carry judgments from a PREVIOUS clustering's centroids onto the current
    one. Returns (carried, split, conflict).

    THIS FIXES A PRE-EXISTING DATA-LOSS PATH, not just a layered-mode concern.
    A cluster judgment is keyed on the centroid's exact float coordinates, and
    a --cluster-radius or --spatial-threshold slider move recomputes every
    centroid. Today those judgments simply stop resolving: export() counts them
    as "orphaned", _finish_recompute prints the count, and nothing recovers
    them. Folding a new model layer in has the same effect (its points can
    merge two previously-distinct clusters), which is what made this
    unavoidable to fix here.

    EXACT, NOT SPATIAL. For each judged old centroid, its former raw members
    are looked up in the NEW point_to_cluster_key. The judgment is carried only
    when all of them landed in exactly one new cluster and that cluster carries
    no judgment of its own. A split, a merge collision, or a member that
    vanished leaves the judgment where it was and increments a counter --
    the same refuse-to-guess rule migrate_legacy_judgments already applies,
    because silently re-attaching a human's "confirmed real" to a cluster they
    never looked at is worse than making them look again.
    """
    store = dr.store
    if store is None or not prior_members:
        return 0, 0, 0
    carried = split = conflict = 0
    for key, rec in list(store.judgments.items()):
        judgment = rec.get("judgment")
        if judgment not in EXPORTABLE or judgment == "added":
            continue
        if key in dr.cluster_members:
            continue                       # still resolves; nothing to do
        old = prior_members.get(key)
        if not old:
            continue
        targets = {dr.point_to_cluster_key.get((m.frame_idx, m.x, m.y)) for m in old}
        targets.discard(None)
        if len(targets) != 1:
            split += 1
            continue
        new_key = targets.pop()
        if store.get(*new_key) != "unreviewed":
            conflict += 1
            continue
        det = next((c for c in dr.unmatched_clusters_by_frame.get(new_key[0], [])
                    if (c.frame_idx, c.x, c.y) == new_key), None)
        store.commit(new_key[0], new_key[1], new_key[2],
                     det.confidence if det is not None else rec.get("original_confidence", 1.0),
                     judgment, rec.get("image_path", dr.image_path_by_frame.get(new_key[0], "")),
                     extra={"source": judgment_source(rec), "remapped_from": [key[0], key[1], key[2]]})
        # Retire the old key. Without this it stays in the log as a
        # confirmed_real that resolves to nothing -- counted as "orphaned"
        # forever, and re-offered to the NEXT remap, where it would collide
        # with the judgment it just created and be reported as a conflict.
        store.commit(key[0], key[1], key[2], rec.get("original_confidence", 1.0),
                     "unreviewed", rec.get("image_path", ""),
                     extra={"source": judgment_source(rec),
                            "remapped_to": [new_key[0], new_key[1], new_key[2]]})
        carried += 1
    return carried, split, conflict


def parse_layer_union(layers: Sequence[LayerSpec]) -> Tuple[Dict[int, FrameData], List[str]]:
    """Every layer's detections, merged into ONE FrameData per frame.

    Each point is a LayerDetection carrying its layer index, so attribution
    survives the merge even where two checkpoints fire on the identical pixel
    (7,738 such coincidences on the full sample flight -- see LayerDetection).

    A frame's image_path must agree across layers. It always does here -- every
    layer of a run comes from one run_detector.py pass over one flight, and the
    frame index is the pose-table row -- but a disagreement would mean the
    layers are of DIFFERENT FLIGHTS, and silently keeping the last writer would
    then draw one flight's detections over another flight's image. So it is
    checked and reported rather than assumed.
    """
    union: Dict[int, FrameData] = {}
    notes: List[str] = []
    mismatched = 0
    for li, spec in enumerate(layers):
        for frame_idx, fd in parse_annotations(spec.path).items():
            slot = union.get(frame_idx)
            if slot is None:
                slot = union[frame_idx] = FrameData(frame_idx, fd.image_path, [])
            elif slot.image_path != fd.image_path:
                mismatched += 1
            for p in fd.points:
                slot.points.append(LayerDetection(p.frame_idx, p.image_path, p.x, p.y,
                                                   p.confidence, layer=li))
    if mismatched:
        notes.append(f"{mismatched} frame(s) disagree on image_path across layers -- "
                     "these layers may not all be of the same flight")
    return union, notes


def load_layered_directory(path: Path, rel_name: str, reviewer: str, spatial_threshold: float,
                            dry_run: bool, layers: Sequence[LayerSpec], log_dir: Path,
                            cluster_radius: float = 30.0,
                            param_overrides: Optional[dict] = None,
                            record_extra: Optional[dict] = None,
                            verbose: bool = False) -> Optional[DirectoryReview]:
    """Load a flight for review against a run's model layers.

    THE ONLY LOADER: every directory is loaded through it.

    labels.txt is read from the FLIGHT directory -- the ground truth does not
    belong to a run. Judgments are written under `log_dir`
    (<run>/review/<flight_slug>/), so any review_log.jsonl and sidecars in
    the flight directory are untouched.
    """
    if not path.is_dir():
        print(f"  {rel_name}: not a directory ({path})")
        return None
    if not layers:
        print(f"  {rel_name}: no model layers found in this run")
        return None
    dets_fd, notes = parse_layer_union(layers)
    for n in notes:
        print(f"  ! {rel_name}: {n}")
    if not dets_fd:
        print(f"  {rel_name}: model layers have no parseable rows")
        return None
    labels_file = path / "labels.txt"
    labels_fd = parse_annotations(labels_file) if labels_file.exists() else {}

    if verbose:
        n_pts = sum(len(fd.points) for fd in dets_fd.values())
        print(f"  {rel_name}: {len(layers)} layers, {n_pts} detections over {len(dets_fd)} frames "
              f"({sum(1 for s in layers if s.degenerate)} of the layers are degenerate and are "
              f"shown, not hidden)")

    dr = DirectoryReview(path=path, rel_name=rel_name, frames_all=sorted(dets_fd.keys()),
                          dets_fd=dets_fd, labels_fd=labels_fd, layers=tuple(layers))
    recompute_layered_matching(dr, dets_fd, labels_fd, spatial_threshold, cluster_radius,
                                verbose=verbose)

    dr.store = LayeredJudgmentStore(
        path, rel_name, reviewer, spatial_threshold, dry_run, param_overrides,
        log_dir=log_dir, schema_version=2, record_extra=record_extra, layers=layers)
    migrate_legacy_judgments(dr)

    for (frame_idx, x, y), rec in dr.store.judgments.items():
        if rec["judgment"] == "added":
            dr.spawned_by_frame.setdefault(frame_idx, []).append(
                Detection(frame_idx, rec["image_path"], x, y,
                          confidence=rec.get("original_confidence", 1.0)))
    return dr


def discover_directories(base: Path, run: Optional[Path] = None) -> List[Path]:
    """Flight directories worth opening: a labels.txt, plus detections for it.

    A flight is reviewable when the RUN has per-checkpoint detection files
    covering it.

    THIS FILTER IS LOAD-BEARING, not tidiness: many directories can carry a
    labels.txt without the run covering them, and without it auto-discovery
    would open flights with nothing to review -- the GUI would open, find no
    detections and skip, once per directory.

    `run=None` keeps the bare labels.txt list (the caller has no run, so it
    cannot know what is covered and load_layered_directory will say so per
    directory)."""
    out = []
    for labels_file in sorted(base.rglob("labels.txt")):
        d = labels_file.parent
        if run is None:
            out.append(d)
            continue
        try:
            rel = str(d.relative_to(base))
        except ValueError:
            continue
        if layer_source_dir(run, rel) is not None:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Display geometry: fit-to-window scaling + screen<->image coordinate maps.
# ---------------------------------------------------------------------------

class DisplayGeometry:
    def __init__(self, native_w: int, native_h: int, max_w: int, max_h: int):
        self.scale = min(max_w / native_w, max_h / native_h, 1.0)
        self.disp_w = max(1, int(round(native_w * self.scale)))
        self.disp_h = max(1, int(round(native_h * self.scale)))

    def to_screen(self, x: float, y: float) -> Tuple[int, int]:
        return int(round(x * self.scale)), int(round(y * self.scale))

    def to_image(self, sx: int, sy: int) -> Tuple[float, float]:
        return sx / self.scale, sy / self.scale


# ---------------------------------------------------------------------------
# Top-level app / state machine
# ---------------------------------------------------------------------------

class App:
    def __init__(self, args: argparse.Namespace, base: Path, dir_specs: List[Tuple[Path, str]]):
        # The run-derived params carry None as their "not given" sentinel so
        # apply_run_context can tell an override from an absent flag. Callers
        # that build args directly (the smoke tests) never pass through that,
        # so resolve the sentinel here -- every construction path ends up here.
        for _k, _v in PARAM_FALLBACKS.items():
            if getattr(args, _k, None) is None:
                setattr(args, _k, _v)
        self.args = args
        self.base = base
        self.dir_specs = dir_specs
        self.dir_cache: Dict[int, Optional[DirectoryReview]] = {}
        # Model layers. `layers` is per-flight display state, rebuilt whenever
        # the directory changes; `_layer_sig` is the on-disk signature the
        # poller compares against, so a checkpoint that finishes mid-session is
        # noticed but never folded in without the reviewer asking.
        self.layers = LayerState(())
        self.layers_sort = "f1"
        self._layer_sig: Tuple = ()
        # Rate-limit clock for _poll_layer_dir.
        self._layer_last_poll = 0.0
        self._layers_stale: List[str] = []
        self._layer_notes: List[str] = []
        self.dir_idx = 0
        self.frame_pos = 0          # index into current dr.frames_all
        self.selected_key: Optional[Tuple[int, float, float]] = None
        self.mode = "overview"      # "overview" | "zoom"
        self.show_help = False
        self.help_scroll = 0  # in lines; see _draw_paginated -- reset to 0 whenever the overlay closes
        # Video play/rewind ([5]/[6] toggle direction, [f] cycles speed) --
        # see _play_tick(). 0 = stopped, +1 = playing forward, -1 = rewinding.
        self._play_direction = 0
        self._play_speed_idx = 0
        self._last_play_advance_time = 0.0
        self.status_msg = ""
        self.status_until = 0.0
        self.image_cache_key: Optional[Tuple[int, int]] = None
        self.image_cache_img: Optional[np.ndarray] = None
        # Static per-frame composite cache (labels + matched + translucent
        # raw layer -- everything that doesn't change between mainloop
        # ticks). render() already redraws every ~20ms with no dirty-
        # checking, and the worst real frame has ~4500 raw points to blend;
        # only the small, frequently-changing centroid/selection layer gets
        # drawn fresh on top each call.
        self.static_cache_key: Optional[Tuple[int, int, int, int]] = None
        self.static_cache_img: Optional[np.ndarray] = None
        # Main-frame display budget in screen pixels (excludes the side
        # panel). Starts at whatever main() resolved --display-width/height
        # to (auto-detected from the monitor if not explicitly passed);
        # _sync_window_size() keeps it live-updated to the actual window
        # size once the cv2 window exists and the user drags to resize it.
        self.window_w = args.display_width
        self.window_h = args.display_height
        # Live spatial-threshold/cluster-radius sliders: _last_applied_* is
        # what's currently in effect; _slider_pending*/_debounce_slider_values()
        # track a value that hasn't settled yet; _recompute_target_to_show
        # arms the chunked recompute sequence in render() (see
        # _start_chunked_recompute()/_step_chunked_recompute(), below).
        # [v] 3D flight-pose window. None whenever the window is closed, which
        # is also the state after any failure -- pose3d is a strictly optional
        # companion view and must never be able to take the review session
        # down with it. See cycle_pose_view()/render_pose_view().
        self.pose_view = None
        self.pose_view_dir: Optional[str] = None
        # "off" | "inline" (in the dead band under the frame) | "window"
        # (its own resizable cv2 window). [v] cycles forward through these.
        self.pose_mode = "off"
        # True while a drag that STARTED in the inline strip is in progress,
        # so an orbit that wanders out over the image keeps going instead of
        # turning into a detection click.
        self._pose_drag_capture = False
        # Follow state to restore when playback stops; None when playback is
        # not the thing that turned following on. See _sync_pose_follow().
        self._pose_follow_restore: Optional[bool] = None
        # Per-point palette colour for the directory-wide 3D detection cloud.
        # There is no per-point IDENTITY array and no JudgmentStore.version
        # stamp: every point is a model layer's detection carrying a fixed
        # colour, and nothing in the cloud recolours as you review.
        self._pose_cloud_fixed: List[Optional[Tuple[int, int, int]]] = []
        # Geospatial metrics (geo_metrics.py). Built eagerly by a worker --
        # measured 2.6s / 1.2GB on the full 2,120-frame sample flight, which is fine once behind a
        # keypress but must never sit on the eager directory-open path. The
        # scaffold is geometry only, so it survives every judgment; only the
        # cheap verdict layer is rebuilt when JudgmentStore.version ticks.
        self._geo_scaffold = None
        self._geo_scaffold_key: Optional[Tuple] = None
        self._geo_summary: Optional[dict] = None
        self._geo_summary_key: Optional[Tuple] = None
        self._geo_proj = None
        # The 3D geo-cluster centroid population, keyed on (propagation rule,
        # judgment version, scaffold) -- see _sync_pose_centroids. Invalidated
        # in _finish_recompute too, because re-clustering re-keys the cluster
        # members that the reach rings are computed from.
        self._pose_centroid_key: Optional[Tuple] = None
        self._geo_verdicts = None
        # The propagator's join cache: raw (frame_idx, x, y) -> geo cluster id,
        # -1 for DBSCAN noise or unresolvable. Keyed on the SCAFFOLD key, not
        # on the judgment key, because it is pure geometry -- a commit must not
        # invalidate 0.2s of work that no judgment can change.
        self._geo_member_label: Optional[Dict[Tuple[int, float, float], int]] = None
        self._geo_member_label_key: Optional[Tuple] = None
        # The per-layer geo arm's worker. `_geo_pub` is the only object shared
        # with it, and only ever whole-entry replaced under the lock.
        self._geo_lock = threading.Lock()
        self._geo_pub: dict = {"gen": 0, "layers": {}, "done": False, "error": None}
        self._geo_gen = 0
        self._geo_worker_key: Optional[Tuple] = None
        self._geo_thread: Optional[threading.Thread] = None
        self._layer_geo_key: Optional[Tuple] = None
        self._layer_geo_cache: Dict[str, dict] = {}
        self._pose_errors: set = set()   # _report_loop_error dedup, same idiom as the mainloop's
        # Directories whose scene failed to build, so the per-tick rebuild in
        # render_pose_view() doesn't retry them forever. Pressing [v] clears
        # the entry -- an explicit request deserves a fresh attempt.
        self._pose_unavailable_dirs: set = set()
        # [a] bulk-accept-classifier-labels state. _accept_armed holds the
        # (frame_idx, reference_op, deadline) of a first [a] press; the
        # commit only happens if a second [a] arrives while all three still
        # match, so the confirmation can't leak across a frame change, an
        # [o] operating-point cycle, or a coffee break. _last_bulk_accept is
        # the key list [z] needs to undo the whole batch in one action.
        self._accept_armed: Optional[Tuple[int, str, float]] = None
        self._last_bulk_accept: List[Tuple[int, float, float]] = []
        self._last_applied_spatial = args.spatial_threshold
        self._last_applied_cluster = args.cluster_radius
        self._slider_pending: Optional[Tuple[float, float]] = None
        self._slider_pending_since = 0.0
        self._recompute_target_to_show: Optional[Tuple[float, float]] = None
        # Chunked-recompute state (the live-progress "waiting screen"): a
        # slider settling arms a resumable generator here instead of
        # blocking for the whole ~13-16s worst case in one call. See
        # _start_chunked_recompute()/_step_chunked_recompute()/
        # _finish_chunked_recompute().
        self._recompute_gen: Optional[Iterator[Tuple[int, int]]] = None
        self._recompute_dr: Optional[DirectoryReview] = None
        self._recompute_target: Optional[Tuple[float, float]] = None
        self._recompute_started_at = 0.0
        self._recompute_frames_done = 0
        self._recompute_frames_total = 0
        # A chunked recompute runs in ONE budgeted phase, "match" (matching +
        # clustering). The phase dispatch exists because it is what stops a
        # job re-arming itself from its own completion path.
        self._recompute_phase = "match"


        # THE [p] OVERLAY reports THIS session's own numbers, at per-model
        # granularity, and needs no file. See _stats_overlay_lines().
        self.show_stats = False
        self.stats_scroll = 0  # in lines; see _draw_paginated -- reset to 0 whenever the overlay closes

        # PROPAGATION OVERLAY ([m]/[o]). What [m] paints is a candidate label
        # PROPAGATED FROM HUMAN ADJUDICATIONS through the geo clusters. See
        # propagated_label().
        self._last_ring_count = 0
        self.model_overlay_mode = False
        # The [o] tie-break rule, or None for "no convention chosen, propose
        # nothing". None is the START state on purpose and cycles back around
        # to None -- see the [o] branch of on_key, and GEO_METRICS.md's
        # "there is no default rule" section, which this preserves.
        self.propagation_rule: Optional[str] = None
        # The model window ([e]). Its own OpenCV window; all layer state lives
        # in LayerState, so this is only "is a window on screen".
        self.model_win = False
        self.model_win_h = MODEL_WIN_MAX_H
        # The HUD body's scroll offset, in whole model groups. Clamped by
        # hud_scroll_view, which hands back what it actually applied.
        self._hud_scroll = 0
        self._last_hud_view: Optional[HudScroll] = None
        # The only inputs are the run's dets/, labels.txt, flight_data.db,
        # review_log.jsonl and PREREGISTRATION.json.

        print(f"Indexing images under {base} (for stale-path fallback) ...")
        self.file_index = build_file_index(base)
        print(f"Indexed {len(self.file_index)} image files.")
        self._advance_to_first_reviewable_dir()

    def panel_width(self) -> int:
        return compute_panel_width(self.args.zoom_crop_px, self.args.zoom_scale)

    def _sync_window_size(self) -> None:
        """Pull the cv2 window's current on-screen size (updated live when
        the user drags to resize it, since the window is WINDOW_NORMAL) and
        use it as the main-frame display budget. No-ops safely if no window
        exists yet (e.g. render() called directly, as in tests) or the
        window was closed."""
        try:
            _, _, w, h = cv2.getWindowImageRect(WINDOW)
        except cv2.error:
            return
        panel_w = self.panel_width()
        if w > panel_w + 200 and h > 150:  # sane lower bound, else keep last-known size
            self.window_w = w - panel_w
            self.window_h = h

    # -- live spatial-threshold / cluster-radius sliders ---------------------

    def _debounce_slider_values(self, spatial: float, cluster: float) -> None:
        """Pure state-machine step (no cv2 calls -- directly unit-testable):
        given the slider's current (spatial, cluster) reading, decide
        whether to arm a recompute. A value only triggers one once it has
        been stable (unchanged) for SLIDER_DEBOUNCE_SECONDS -- a single drag
        gesture changes the trackbar on nearly every ~20ms mainloop tick,
        and the worst real directory takes ~13-16s to recompute, so without
        this a drag would queue a pileup of blocking recomputes rather than
        one, at the end, once the user actually stops."""
        current = (spatial, cluster)
        if current == (self._last_applied_spatial, self._last_applied_cluster):
            self._slider_pending = None  # back to the already-applied value -- nothing to do
            return
        if current != self._slider_pending:
            self._slider_pending, self._slider_pending_since = current, time.monotonic()
            return
        if time.monotonic() - self._slider_pending_since >= SLIDER_DEBOUNCE_SECONDS:
            self._recompute_target_to_show = current
            self._slider_pending = None

    def _poll_sliders(self) -> None:
        """Read the live cv2 trackbar positions (no-ops safely if no window/
        trackbars exist yet, e.g. render() called directly as in tests --
        same guard pattern as _sync_window_size()) and feed them to the
        debounce state machine. No-ops entirely while a chunked recompute
        is already in flight, so a slider nudge mid-recompute can't arm a
        second, overlapping generator."""
        if self._recompute_gen is not None:
            return
        try:
            spatial = float(cv2.getTrackbarPos(TRACKBAR_SPATIAL, WINDOW))
            cluster = float(cv2.getTrackbarPos(TRACKBAR_CLUSTER, WINDOW))
        except cv2.error:
            return
        self._debounce_slider_values(spatial, cluster)



    def _arm_recompute(self, spatial_threshold: float, cluster_radius: float) -> Optional[DirectoryReview]:
        """Shared upfront bookkeeping for both the blocking and chunked
        recompute paths: stamps the new session defaults, evicts every
        OTHER cached directory (App._dr() reloads a cache miss from disk
        using self.args, so anything visited after this is guaranteed to
        pick up the new values -- never a stale mix of old/new parameters
        across directories), and returns the current directory to
        recompute (None if nothing is loaded here yet)."""
        self._play_direction = 0  # a recompute and video play/rewind never run "at once"
        self.args.spatial_threshold = spatial_threshold
        self.args.cluster_radius = cluster_radius
        self._last_applied_spatial, self._last_applied_cluster = spatial_threshold, cluster_radius
        dr = self.dir_cache.get(self.dir_idx)
        self.dir_cache = {self.dir_idx: dr} if dr is not None else {}
        # Snapshot the CURRENT clustering before the recompute resets it, so
        # _finish_recompute can carry judgments onto the new centroids. ~74k
        # shallow dict entries, ~10 ms. Without it every judgment keyed on a
        # centroid the new clustering does not reproduce is simply orphaned --
        # counted, printed, and lost.
        self._prior_members = dict(dr.cluster_members) if dr is not None else {}
        if dr is None:
            self.set_status(f"spatial={spatial_threshold:.0f}px, cluster={cluster_radius:.0f}px "
                             f"(new default -- nothing loaded here yet to recompute)")
        return dr

    def _finish_recompute(self, dr: DirectoryReview, spatial_threshold: float, cluster_radius: float,
                           started_at: float) -> None:
        """Shared post-processing once the (blocking or chunked) matching+
        clustering pass has finished."""
        # JudgmentStore stamps this into every future commit's JSONL record
        # for audit/provenance -- without updating it here, every judgment
        # made for the rest of the session would misreport the stale
        # CLI-launch threshold even though a different one is now in effect.
        dr.store.spatial_threshold = spatial_threshold
        carried, split, conflict = remap_cluster_judgments(dr, getattr(self, "_prior_members", {}))
        self._prior_members = {}
        migrate_legacy_judgments(dr)
        counts = dr.store.export(dr.member_map())
        self._last_remap = (carried, split, conflict)

        self.selected_key, self.mode = None, "overview"  # the selected centroid may not exist any more
        self.static_cache_key = None  # depends on point_to_cluster_key, which just changed
        self._layer_cloud_key = None  # the world cloud is keyed on matching_version
        self._pose_centroid_key = None  # (rule, store version, scaffold) the centroids belong to
        self.frame_pos = min(self.frame_pos, max(0, len(dr.frames_all) - 1))
        # The propagator's join cache is keyed on the SCAFFOLD, not on
        # matching -- but re-clustering re-keys the cluster members, so the
        # cached verdict layer is dropped and rebuilt on next use.
        self._geo_summary_key = None
        rev, tot = dr.progress()
        self.set_status(f"recomputed (spatial={spatial_threshold:.0f}px, cluster={cluster_radius:.0f}px, "
                         f"{time.monotonic()-started_at:.1f}s): {tot} clusters ({dr.unmatched_total()} raw), "
                         f"{rev} already reviewed, {counts['orphaned']} orphaned"
                         + (f", {carried} carried" if carried else "")
                         + (f", {split} split" if split else "")
                         + (f", {conflict} collided" if conflict else ""), seconds=6.0)





    def _apply_slider_recompute(self, spatial_threshold: float, cluster_radius: float) -> None:
        """Blocking, complete-in-one-call convenience path (used by tests
        and any other non-interactive caller). The live GUI uses the
        chunked path instead -- see _start_chunked_recompute()/
        _step_chunked_recompute() -- so the "Recomputing..." banner can
        show real progress instead of blocking for the whole ~13-16s
        worst case with the window apparently frozen."""
        t0 = time.monotonic()
        dr = self._arm_recompute(spatial_threshold, cluster_radius)
        if dr is None:
            return
        if dr.layers:
            recompute_layered_matching(dr, dr.dets_fd, dr.labels_fd,
                                        spatial_threshold, cluster_radius)
        else:
            recompute_matching(dr, dr.dets_fd, dr.labels_fd, spatial_threshold, cluster_radius)
        self._finish_recompute(dr, spatial_threshold, cluster_radius, t0)

    # -- chunked (resumable) recompute: the live "waiting screen" ------------

    def _start_chunked_recompute(self, spatial_threshold: float, cluster_radius: float) -> None:
        t0 = time.monotonic()
        dr = self._arm_recompute(spatial_threshold, cluster_radius)
        if dr is None:
            return
        self._recompute_dr = dr
        self._recompute_target = (spatial_threshold, cluster_radius)
        self._recompute_started_at = t0
        self._recompute_frames_done, self._recompute_frames_total = 0, len(dr.frames_all)
        self._recompute_phase = "match"
        # The layered generator when there are layers: recompute_matching_steps
        # would rebuild the union correctly but leave cluster_layer_counts and
        # the per-layer counters stale, so every per-model number on screen
        # would silently describe the PREVIOUS clustering.
        steps = recompute_layered_matching_steps if dr.layers else recompute_matching_steps
        self._recompute_gen = steps(dr, dr.dets_fd, dr.labels_fd,
                                     spatial_threshold, cluster_radius)

    def _step_chunked_recompute(self, budget_seconds: float = CHUNK_TIME_BUDGET_SECONDS) -> bool:
        """Advance self._recompute_gen for up to budget_seconds of actual
        wall-clock work (not a fixed frame count -- per-frame cost varies a
        lot with point density), called once per render() tick while a
        chunked recompute is in flight. Returns True while still running;
        False once drained and finalized (this same call)."""
        t_start = time.monotonic()
        try:
            while time.monotonic() - t_start < budget_seconds:
                self._recompute_frames_done, self._recompute_frames_total = next(self._recompute_gen)
        except StopIteration:
            return self._advance_phase()
        return True

    def _advance_phase(self) -> bool:
        """One generator drained -- arm the next phase of this job, or
        finalize it. Returns True if work remains (keep showing the banner).

        Dispatching explicitly on the CURRENT phase is load-bearing, not
        style: re-arming a phase from inside its own completion path loops
        forever with the banner stuck on screen. Every branch below either
        advances to a strictly later phase or finalizes.

        Jobs and their phases:
          slider recompute:  match -> done

        The multi-phase machinery is kept rather than collapsed into a single
        generator because a chunked job has to be able to hand off to a
        successor without re-arming itself from inside its own completion
        path.
        """
        phase = self._recompute_phase
        if phase == "match":
            self._finish_chunked_recompute()
            return False
        # Unknown phase: fail safe by ending the job rather than spinning.
        self._recompute_gen = None
        self._recompute_phase = "match"
        return False




    def _finish_chunked_recompute(self) -> None:
        dr, (spatial_threshold, cluster_radius) = self._recompute_dr, self._recompute_target
        self._finish_recompute(dr, spatial_threshold, cluster_radius, self._recompute_started_at)
        self._recompute_gen = self._recompute_dr = self._recompute_target = None
        self._recompute_phase = "match"
        # A slider change re-clusters the whole directory, so every key in the
        # 3D cloud is now stale -- the judgment-version check cannot see this
        # (no judgment changed), so rebuild explicitly.
        if self.pose_view is not None:
            self._build_pose_cloud()

    # -- directory / frame access -----------------------------------------

    def _dr(self, idx: int) -> Optional[DirectoryReview]:
        if idx < 0 or idx >= len(self.dir_specs):  # bounds-safe: never IndexError
            return None
        if idx not in self.dir_cache:
            path, rel_name = self.dir_specs[idx]
            layers, notes = self._layers_for(rel_name)
            ctx = getattr(self.args, "run_context", None)
            if not layers:
                # THE ONLY DETECTION SOURCE IS THE RUN: a flight the run has
                # no detection files for has nothing to review, and this says
                # so by name. The caller skips a None directory.
                where = f" in {ctx.run.name}" if (ctx is not None and ctx.run is not None) \
                        else " (no run directory given)"
                print(f"  {rel_name}: no model layers{where} -- nothing to review")
                dr = None
            else:
                # Detections come from the run's per-checkpoint files and
                # judgments go under the run, so the flight's own review_log
                # and sidecars are neither read nor written.
                run = ctx.run
                dr = load_layered_directory(
                    path, rel_name, self.args.reviewer, self.args.spatial_threshold,
                    self.args.dry_run, layers, run / "review" / flight_slug(rel_name),
                    self.args.cluster_radius, getattr(self.args, "param_overrides", None),
                    record_extra={"review_unit": "consensus_cluster", "run": run.name,
                                  "layer_set": [s.layer_id for s in layers],
                                  "cluster_radius": self.args.cluster_radius},
                    verbose=True)
            self.dir_cache[idx] = dr
            if idx == self.dir_idx:
                self._sync_layer_state(dr, layers, notes)
        return self.dir_cache[idx]

    def _layers_for(self, rel_name: str) -> Tuple[List[LayerSpec], List[str]]:
        """This flight's model layers, from the run context.

        Empty means NOTHING TO REVIEW: there is no second detection source
        to fall back on."""
        ctx = getattr(self.args, "run_context", None)
        if ctx is None or ctx.run is None:
            return [], []
        return discover_layers(ctx.run, rel_name, ctx.preregistration, ctx.selection_pooled)

    def _sync_layer_state(self, dr: Optional[DirectoryReview],
                          layers: Sequence[LayerSpec], notes: Sequence[str]) -> None:
        self.layers.set_layers(layers if dr is not None else ())
        self._layer_notes = list(notes)
        self._layers_stale = []
        ctx = getattr(self.args, "run_context", None)
        if ctx is not None and ctx.run is not None and layers:
            self._layer_sig = layer_dir_signature(layer_source_dir(ctx.run, dr.rel_name))
        else:
            self._layer_sig = ()

    @property
    def dr(self) -> Optional[DirectoryReview]:
        return self._dr(self.dir_idx)

    def _advance_to_first_reviewable_dir(self) -> None:
        """On startup, skip forward past directories with nothing to
        review -- but ONLY when directories were auto-discovered via
        --base-dir. Explicitly-named directories (positional `dirs` args)
        are never skipped, even if unloadable or empty: the tool opens on
        the first one and shows why (the loader already prints the
        reason), rather than silently walking past what the user asked
        for -- or worse, walking off the end of dir_specs entirely."""
        if not self.args.auto_discovered:
            self.frame_pos = 0
            return
        while self.dir_idx < len(self.dir_specs):
            dr = self.dr
            if dr is None:
                pass  # the loader already printed why
            elif dr.unmatched_total() == 0:
                print(f"  skipping {dr.rel_name}: 0 unmatched detections, nothing to review")
            else:
                break
            self.dir_idx += 1
        self.frame_pos = 0

    def current_frame_idx(self) -> Optional[int]:
        dr = self.dr
        if dr is None or not dr.frames_all:
            return None
        self.frame_pos = max(0, min(self.frame_pos, len(dr.frames_all) - 1))
        return dr.frames_all[self.frame_pos]

    # -- navigation ----------------------------------------------------------

    def goto_dir(self, delta: int) -> None:
        new_idx = self.dir_idx + delta
        if 0 <= new_idx < len(self.dir_specs):
            self.dir_idx = new_idx
            self.frame_pos = 0
            self.selected_key = None
            self.mode = "overview"
            self._reset_hud_scroll()
            if self.dr is None:
                self.set_status(f"{self.dir_specs[self.dir_idx][1]}: no model layers in this run")
        else:
            self.set_status("no more directories" if delta > 0 else "already at first directory")

    def goto_frame(self, delta: int) -> None:
        dr = self.dr
        if dr is None or not dr.frames_all:
            return
        new_pos = self.frame_pos + delta
        if 0 <= new_pos < len(dr.frames_all):
            self.frame_pos = new_pos
            self.selected_key = None
            self.mode = "overview"
        else:
            self.set_status("no more frames in this directory" if delta > 0 else "already at first frame")

    def _play_tick(self, now: float) -> None:
        """Video play/rewind's per-render()-tick advance check -- same
        time.monotonic()-driven idiom as _debounce_slider_values()/
        _step_chunked_recompute(), and deliberately just as pure (an
        injected `now` makes it directly unit-testable with no real
        sleeping needed). Called from render() only in the normal-
        rendering branch (after the recompute/show_stats/show_help early-
        returns), so play structurally cannot advance during any of those
        states without needing its own guard here."""
        if self._play_direction == 0:
            return
        if now - self._last_play_advance_time < PLAY_INTERVALS_SECONDS[self._play_speed_idx]:
            return
        self._last_play_advance_time = now
        old_pos = self.frame_pos
        self.goto_frame(self._play_direction)
        if self.frame_pos == old_pos:
            # Hit a directory boundary -- goto_frame already clamps and
            # sets a status message; stop rather than wrap or cross into
            # the next/previous directory (n/b already does that, deliberately
            # differently -- see REVIEW_GUI.md's "Video play/rewind" section).
            self._play_direction = 0

    def jump_to_unreviewed(self, forward: bool) -> None:
        """Crosses frame and directory boundaries to find the next/previous
        frame containing an unreviewed unmatched point."""
        n_dirs = len(self.dir_specs)
        d_idx = self.dir_idx
        f_pos = self.frame_pos
        for _ in range(n_dirs * 2):  # bounded: at most a full sweep, twice (covers wrap from start)
            dr = self._dr(d_idx)
            if dr is not None and dr.frames_all:
                positions = range(f_pos + 1, len(dr.frames_all)) if forward else range(f_pos - 1, -1, -1)
                for pos in positions:
                    frame_idx = dr.frames_all[pos]
                    centroids = dr.unmatched_clusters_by_frame.get(frame_idx, [])
                    if any(dr.store.get(frame_idx, c.x, c.y) == "unreviewed" for c in centroids):
                        self.dir_idx, self.frame_pos = d_idx, pos
                        self.selected_key, self.mode = None, "overview"
                        return
            d_idx += 1 if forward else -1
            if d_idx < 0 or d_idx >= n_dirs:
                break
            f_pos = -1 if forward else (len(self._dr(d_idx).frames_all) if self._dr(d_idx) else 0)
        self.set_status("no more unreviewed points found")

    # -- selection / judgment ------------------------------------------------

    def _frame_points(self) -> List[Detection]:
        """Review-navigable points only (space/n/b traverse these): cluster
        CENTROIDS, not raw detections -- spawned points are already
        'resolved' the moment they're created, there's nothing to
        review-advance through for them either."""
        dr, frame_idx = self.dr, self.current_frame_idx()
        if dr is None or frame_idx is None:
            return []
        return dr.unmatched_clusters_by_frame.get(frame_idx, [])

    def _spawned_frame_points(self) -> List[Detection]:
        dr, frame_idx = self.dr, self.current_frame_idx()
        if dr is None or frame_idx is None:
            return []
        return dr.spawned_by_frame.get(frame_idx, [])

    def _nearest_point(self, pts: List[Detection], sx: int, sy: int, geom: DisplayGeometry) -> Optional[Detection]:
        best, best_d2 = None, self.args.click_tolerance_px ** 2
        for p in pts:
            psx, psy = geom.to_screen(p.x, p.y)
            d2 = (psx - sx) ** 2 + (psy - sy) ** 2
            if d2 <= best_d2:
                best, best_d2 = p, d2
        return best

    def select_nearest(self, sx: int, sy: int, geom: DisplayGeometry) -> None:
        """Left-click: select the nearest point of *either* kind (unmatched
        detection or already-spawned), whichever is closer -- just for
        inspection/judging, no side effect."""
        best = self._nearest_point(self._frame_points() + self._spawned_frame_points(), sx, sy, geom)
        if best is not None:
            self.selected_key = (best.frame_idx, best.x, best.y)
            self.mode = "zoom"

    def right_click(self, sx: int, sy: int, geom: DisplayGeometry) -> None:
        """Right-click: three-tier behavior --
        1. Near an existing unmatched detection -> fast-confirm it real
           (equivalent to selecting + pressing [c]), no new point created.
        2. Near an already-spawned point -> just select it (avoids
           accidentally spawning a duplicate on top of one).
        3. Nowhere near anything -> spawn a brand-new point exactly at the
           click location: a target the model never detected at all."""
        dr, frame_idx = self.dr, self.current_frame_idx()
        if dr is None or frame_idx is None:
            return
        nearest_unmatched = self._nearest_point(self._frame_points(), sx, sy, geom)
        if nearest_unmatched is not None:
            self.selected_key = (nearest_unmatched.frame_idx, nearest_unmatched.x, nearest_unmatched.y)
            self.mode = "zoom"
            self.commit("confirmed_real")
            return
        nearest_spawned = self._nearest_point(self._spawned_frame_points(), sx, sy, geom)
        if nearest_spawned is not None:
            self.selected_key = (nearest_spawned.frame_idx, nearest_spawned.x, nearest_spawned.y)
            self.mode = "zoom"
            return
        ix, iy = geom.to_image(sx, sy)
        self.spawn_point(frame_idx, dr.image_path_by_frame[frame_idx], ix, iy)

    def advance_selection(self) -> bool:
        """Select the next unreviewed point in the current frame, after the
        currently-selected one (or from the start if none selected). Returns
        False (and clears selection) if none remain."""
        dr, frame_idx = self.dr, self.current_frame_idx()
        pts = self._frame_points()
        start = 0
        if self.selected_key is not None:
            keys = [(p.frame_idx, p.x, p.y) for p in pts]
            if self.selected_key in keys:
                start = keys.index(self.selected_key) + 1
        for p in pts[start:]:
            if dr.store.get(p.frame_idx, p.x, p.y) == "unreviewed":
                self.selected_key = (p.frame_idx, p.x, p.y)
                self.mode = "zoom"
                return True
        self.selected_key, self.mode = None, "overview"
        self.set_status("frame complete -- no unreviewed points left")
        return False

    @staticmethod
    def _layer_commit_extra(dr: DirectoryReview, key: Tuple[int, float, float]) -> dict:
        """Which model layers fired on this consensus cluster, for the record.

        LAYER IDS, NEVER BIT INDICES. A layer's index is its position in the
        sorted file list for THIS session; a checkpoint that finishes later
        shifts every index after it, so an index written today would decode to
        a different checkpoint tomorrow.

        This is what makes one adjudication scoreable per model after the fact,
        outside this GUI -- the by_layer/ sidecars are derived from the same
        information, and a downstream reader can recompute either from the log
        alone."""
        if not dr.layers:
            return {}
        counts = dr.cluster_layer_counts.get(key)
        if counts is None:
            return {}
        conf = dr.cluster_layer_conf.get(key, (0.0,) * len(dr.layers))
        fired = [i for i, n in enumerate(counts) if n]
        return {
            "layers_fired": [dr.layers[i].layer_id for i in fired],
            "layer_counts": {dr.layers[i].layer_id: counts[i] for i in fired},
            "layer_conf": {dr.layers[i].layer_id: round(conf[i], 6) for i in fired},
        }

    def commit(self, judgment: str) -> None:
        """Judge the selected cluster CENTROID -- the judgment is recorded
        once here but export() fans it back out to every raw detection in
        the cluster (see JudgmentStore.export()/DirectoryReview.member_map()).
        Not valid for a spawned point (see spawn_point()/remove_spawned()
        below) -- a spawned point is a human assertion, not a model
        detection to judge."""
        dr = self.dr
        if dr is None or self.selected_key is None:
            self.set_status("select a point first (click one, or press space)")
            return
        frame_idx, x, y = self.selected_key
        centroids = dr.unmatched_clusters_by_frame.get(frame_idx, [])
        det = next((c for c in centroids if c.x == x and c.y == y), None)
        if det is None:
            if self._is_spawned(frame_idx, x, y):
                self.set_status("spawned points can't be judged -- press [r] to remove instead")
            return
        n_members = len(dr.cluster_members.get((frame_idx, x, y), [det]))
        dr.store.commit(frame_idx, x, y, det.confidence, judgment, dr.image_path_by_frame[frame_idx],
                        extra={"source": SOURCE_HUMAN, **self._layer_commit_extra(dr, (frame_idx, x, y))})
        self.static_cache_key = None  # this cluster's raw members' tint in the translucent layer just changed
        counts = dr.store.export(dr.member_map())
        cluster_note = f"  ({n_members} raw detections)" if n_members > 1 else ""
        self.set_status(f"{judgment}: {det.x:.0f},{det.y:.0f}{cluster_note}  (dir totals: "
                         f"{counts['hard_negative']} hard-neg, {counts['confirmed_real']} confirmed-real, "
                         f"{counts['added']} added, {counts['orphaned']} orphaned)")
        self.advance_selection()

    # -- [a] bulk-accept the classifier's calls for one frame ---------------

    def propagate_frame_plan(self, frame_idx: int) -> Tuple[List[Tuple[Detection, str, dict]], int, int]:
        """What [a] *would* do to `frame_idx`, computed without writing
        anything -- the preview and the commit both read this, so the count
        shown in the confirmation prompt is by construction the count that
        gets written.

        Returns (accepts, n_already_judged, n_no_label):
          accepts          -- (centroid, judgment, propagation) per cluster
                              that would be newly judged
          n_already_judged -- clusters skipped because a judgment already
                              exists. Never overwritten: a human decision
                              outranks a propagated one, and silently
                              clobbering one is the failure that would make
                              review_log.jsonl untrustworthy. Note this now
                              protects a human verdict from being overwritten
                              by an inference DRAWN FROM ANOTHER human verdict,
                              which is a weaker conflict than the classifier
                              case but resolved the same way: the judgment
                              made on this cluster wins over one deduced about
                              it.
          n_no_label       -- clusters where nothing propagates. Reported
                              rather than guessed at, matching
                              propagated_label()'s own refusal to invent a
                              call -- and it is the majority case early in a
                              review, when few geo clusters carry a verdict
                              yet.
        """
        dr = self.dr
        accepts: List[Tuple[Detection, str, dict]] = []
        n_already = n_no_label = 0
        if dr is None or self.propagation_rule is None:
            return accepts, n_already, n_no_label
        for c in dr.unmatched_clusters_by_frame.get(frame_idx, []):
            ckey = (c.frame_idx, c.x, c.y)
            if dr.store.get(*ckey) != "unreviewed":
                n_already += 1
                continue
            prop = self.propagation_for(ckey)
            if prop is None:
                n_no_label += 1
                continue
            accepts.append((c, prop["judgment"], prop))
        return accepts, n_already, n_no_label

    def _accept_blocked_reason(self) -> Optional[str]:
        """Why [a] cannot run right now, or None if it can."""
        if self.dr is None or self.current_frame_idx() is None:
            return "no frame loaded"
        if self.propagation_rule is None:
            # No default rule, deliberately -- so there is a state in which [a]
            # has nothing it is entitled to do. See the [o] branch of on_key.
            return "no propagation rule chosen -- press [o] to pick one"
        g = self._ensure_geo_verdicts()
        if g is None:
            return "geospatial clustering still building -- [a] needs it to propagate"
        if not g.get("available"):
            return f"no geospatial clustering: {g.get('reason')}"
        return None

    def accept_propagated_frame(self) -> None:
        """[a]: adopt the PROPAGATED label -- carried from a human verdict on
        the same physical spot in another frame, under the current [o]
        tie-break rule -- as the adjudication for every still-unreviewed
        cluster on this frame.

        Two-step: the first press arms and previews, a second press within
        ACCEPT_ARM_SECONDS commits.

        WHAT IS WRITTEN, AND WHY IT IS source=human. Every input to a
        propagated label is a human verdict; the only machine involvement is
        the DBSCAN geometry deciding which verdicts are about the same spot.
        So these rows are human judgments, and a SOURCE_CLASSIFIER stamp
        (with its "the model grading its own output" caveat) would be a false
        provenance claim -- there is no model in this loop at all.

        The `propagated` block is recorded anyway, and it is not decoration:
        WHICH TIE-BREAK CONVENTION produced the label is the ~15-precision-
        point choice GEO_METRICS.md refuses to let anyone quote blind, and a
        row that did not carry it could not be re-derived or re-scored under a
        different rule later. `votes` carries the evidence, not just the
        conclusion."""
        reason = self._accept_blocked_reason()
        if reason:
            self._accept_armed = None
            self.set_status(reason, seconds=6.0)
            return
        dr, rule = self.dr, self.propagation_rule
        frame_idx = self.current_frame_idx()
        accepts, n_already, n_no_label = self.propagate_frame_plan(frame_idx)
        if not accepts:
            self._accept_armed = None
            self.set_status(f"nothing to propagate on frame {frame_idx} "
                             f"({n_already} already judged, {n_no_label} with no propagated label)",
                             seconds=6.0)
            return

        armed = self._accept_armed
        now = time.monotonic()
        # The arm is keyed on the RULE as well as the frame, so cycling [o]
        # invalidates it -- committing under a convention the reviewer just
        # moved away from is exactly what the two-press arm prevents.
        if not (armed and armed[0] == frame_idx and armed[1] == rule and now < armed[2]):
            self._accept_armed = (frame_idx, rule, now + ACCEPT_ARM_SECONDS)
            n_real = sum(1 for _, j, _ in accepts if j == "confirmed_real")
            self.set_status(f"propagate {len(accepts)} labels on frame {frame_idx} @ {rule} "
                             f"({n_real} real, {len(accepts) - n_real} hard-neg; {n_already} already judged, "
                             f"{n_no_label} no label)?  press [a] again to confirm",
                             seconds=ACCEPT_ARM_SECONDS)
            return

        self._accept_armed = None
        keys: List[Tuple[int, float, float]] = []
        for c, judgment, prop in accepts:
            dr.store.commit(c.frame_idx, c.x, c.y, c.confidence, judgment,
                             dr.image_path_by_frame[frame_idx],
                             extra={"source": SOURCE_HUMAN,
                                    "propagated": {"rule": prop["rule"],
                                                   "geo_clusters": prop["geo_clusters"],
                                                   "votes": prop["votes"],
                                                   "n_members": prop["n_members"]}})
            keys.append((c.frame_idx, c.x, c.y))
        self._last_bulk_accept = keys
        self.static_cache_key = None  # every accepted cluster's raw members just took a new tint
        counts = dr.store.export(dr.member_map())
        n_real = sum(1 for _, j, _ in accepts if j == "confirmed_real")
        self.set_status(f"propagated {len(keys)} labels @ {rule} on frame {frame_idx} "
                         f"({n_real} real, {len(keys) - n_real} hard-neg)  [z] to undo  (dir totals: "
                         f"{counts['hard_negative']} hard-neg, {counts['confirmed_real']} confirmed-real)",
                         seconds=8.0)

    def undo_bulk_accept(self) -> None:
        """[z]: reset exactly the clusters written by the last [a] back to
        unreviewed. Scoped to that one batch rather than "all propagated
        judgments" -- undo should reverse the last action, not sweep the
        directory. The reset is itself an ordinary appended commit (the log
        is append-only), stamped human because pressing [z] is a human
        decision to reject the propagation."""
        if not self._last_bulk_accept:
            self.set_status("nothing to undo -- [z] reverses the last [a] bulk accept")
            return
        dr = self.dr
        if dr is None:
            return
        n = 0
        for frame_idx, x, y in self._last_bulk_accept:
            det = next((c for c in dr.unmatched_clusters_by_frame.get(frame_idx, [])
                        if c.x == x and c.y == y), None)
            if det is None:
                continue  # cluster does not exist any more (a slider recompute re-clustered the frame)
            dr.store.commit(frame_idx, x, y, det.confidence, "unreviewed",
                             dr.image_path_by_frame.get(frame_idx, ""),
                             extra={"source": SOURCE_HUMAN, "undo_of": "propagated"})
            n += 1
        stale = len(self._last_bulk_accept) - n
        self._last_bulk_accept = []
        self.static_cache_key = None
        dr.store.export(dr.member_map())
        note = f" ({stale} cluster(s) do not exist any more -- re-clustered since)" if stale else ""
        self.set_status(f"undid {n} propagated judgment(s){note}")

    # -- [v] 3D flight-pose companion window --------------------------------

    def _ensure_pose_scene(self) -> bool:
        """Build (or reuse) the PoseScene for the current directory. Returns
        True when self.pose_view is usable. Never raises.

        pose3d is imported HERE, not at module scope: it pulls in geo_eval,
        which pulls in scipy/sklearn and opens a SQLite file. None of that
        should be able to stop the GUI from starting, and the GUI has to keep
        working as-is on directories that ship no flight_data.db at all."""
        if self.dr is None:
            self.set_status("no directory loaded")
            return False
        if self.pose_view is not None and self.pose_view_dir == self.dr.rel_name:
            return True
        self.pose_view = None
        try:
            from . import pose3d
        except Exception as e:  # noqa: BLE001
            self.set_status(f"3D view unavailable: {type(e).__name__}: {e}", seconds=8.0)
            return False
        try:
            # max frame INDEX, not the frame count: a detection file often covers a
            # subset of the flight, so the sample flight's 1,355 detection
            # frames carry indices up to 2,119.
            scene = pose3d.PoseScene(self.dr.path, self.dr.rel_name,
                                      max(self.dr.frames_all) if self.dr.frames_all else 0)
        except pose3d.PoseUnavailable as e:
            self.set_status(f"3D view: {e}", seconds=8.0)
            return False
        except Exception as e:  # noqa: BLE001
            _report_loop_error(e, self._pose_errors, "pose3d-open")
            self.set_status(f"3D view failed to open: {type(e).__name__}: {e}", seconds=8.0)
            return False
        self.pose_view = pose3d.PoseView(scene)
        self.pose_view_dir = self.dr.rel_name
        if scene.warning:
            print(f"  3D pose view ({self.dr.rel_name}): {scene.warning}")
        self._build_pose_cloud()
        return True

    def _build_pose_cloud(self) -> None:
        """Back-project the WHOLE directory's detections into the 3D scene,
        one colour-keyed population PER MODEL LAYER, so the shape of each
        checkpoint's output over the whole flight is visible at once rather
        than just the frame on screen.

        pose3d additionally marks whichever points land within 0.35 m of a
        survey geotag (geo_eval's own EPS_PREDICT_DETS).

        Every population is ONE MODEL LAYER in its palette colour. There is
        no union population (cluster centroids coloured by judgment, or
        label-matched detections in grey): that would superimpose all thirty
        checkpoints into one undifferentiated cloud, which is the reading
        failure the per-layer palette exists to prevent.

        WHAT THAT COSTS, stated because it is a real trade-off: [d] does not
        show review progress in 3D -- it shows detector output, and every
        point carries a fixed palette colour. Review progress stays legible
        in 2D, on the centroids themselves and on the HUD's coverage bar.

        Model layers occupy slots 0..n-1. Measured 0.18 s for the full
        sample flight's 191,202 points, so this runs inline on open rather
        than needing the chunked-recompute treatment."""
        dr, view = self.dr, self.pose_view
        if dr is None or view is None:
            return
        frames: List[int] = []
        xy: List[Tuple[float, float]] = []
        layer_of: List[int] = []
        fixed: List[Optional[Tuple[int, int, int]]] = []
        # Subsampled PER LAYER: CLOUD_MAX_POINTS is a global cap, so a
        # 34,817-detection collapsed checkpoint would otherwise decimate the
        # healthy ones on its behalf, and each layer's SHAPE is what the cloud
        # is being read for.
        if dr.layers:
            ids = [sp.layer_id for sp in dr.layers]
            per_layer_cap = max(2000, 200_000 // max(1, len(dr.layers)))
            buckets: Dict[int, List[Tuple[int, float, float]]] = {}
            for fi, fd in dr.dets_fd.items():
                for pt in fd.points:
                    li = getattr(pt, "layer", -1)
                    if li >= 0:
                        buckets.setdefault(li, []).append((fi, pt.x, pt.y))
            for li, pts in buckets.items():
                stride = max(1, math.ceil(len(pts) / per_layer_cap))
                st = layer_style(ids[li], ids, dr.layers[li].verdict)
                for fi, px, py in pts[::stride]:
                    frames.append(fi); xy.append((px, py))
                    layer_of.append(li); fixed.append(st.color)
        if not frames:
            self._pose_cloud_fixed = []
            return
        try:
            valid = view.scene.build_cloud(np.asarray(frames, dtype=np.int64),
                                            np.asarray(xy, dtype=float),
                                            dr.image_path_by_frame,
                                            layer=np.asarray(layer_of, dtype=np.int16))
        except Exception as e:  # noqa: BLE001 -- an extra layer must never break the view
            _report_loop_error(e, self._pose_errors, "pose3d-cloud")
            self._pose_cloud_fixed = []
            return
        # Every point carries a FIXED palette colour now, so this is written
        # once here and never recoloured. Visibility does not go through the
        # colour path at all -- it is a bool mask on the scene -- which is
        # what keeps a layer toggle from triggering a 400k-element rebuild.
        self._pose_cloud_fixed = [f for f, v in zip(fixed, valid) if v]
        try:
            view.scene.set_cloud_colors(
                np.asarray([f or COLOR["text"] for f in self._pose_cloud_fixed],
                           dtype=np.uint8))
        except Exception as e:  # noqa: BLE001 -- never break the view
            _report_loop_error(e, self._pose_errors, "pose3d-colors")

    # -- geospatial metrics (the paper's geospatial precision/recall) -------

    def _ensure_geo_scaffold(self) -> None:
        """Build (or reuse) the geospatial scaffold for the current directory.

        geo_metrics is imported HERE, not at module scope, for the same reason
        pose3d is: it pulls in sklearn and geo_eval, and nothing about an
        optional metric may affect this tool's startup.

        Keyed on the DIRECTORY only. The scaffold clusters raw detections in
        UTM, which neither --cluster-radius nor --spatial-threshold can touch,
        so a slider drag must not rebuild it -- only the verdict layer depends
        on the sliders, and that is a dict lookup.
        """
        dr = self.dr
        if dr is None:
            return
        key = (dr.rel_name, self.args.geo_eps, self.args.geo_min_samples,
               self.args.geo_match_radius, self.args.geo_dedup_cm)
        if self._geo_scaffold is not None and self._geo_scaffold_key == key:
            return
        try:
            from . import geo_metrics
            from .geo_eval import Projector
        except Exception as e:  # noqa: BLE001 -- an optional metric, never fatal
            _report_loop_error(e, self._pose_errors, "geo-import")
            # geo_metrics itself is what failed to import, so its own
            # _empty_scaffold is out of reach -- a duck-typed stand-in keeps
            # every caller on the `.ok` / `.unavailable` contract.
            self._geo_scaffold = _GeoImportFailed(f"geo_metrics unimportable: {e}")
            self._geo_scaffold_key = key
            return
        db = dr.path / "flight_data.db"
        self._geo_proj = None
        try:
            if db.exists():
                self._geo_proj = Projector(db)
            self._geo_scaffold = geo_metrics.build_scaffold(
                dr.path, healthy_union_frames(dr), eps=self.args.geo_eps,
                min_samples=self.args.geo_min_samples,
                eps_predict=self.args.geo_match_radius,
                dedup_cm=self.args.geo_dedup_cm, proj=self._geo_proj)
        except Exception as e:  # noqa: BLE001
            _report_loop_error(e, self._pose_errors, "geo-scaffold")
            self._geo_scaffold = geo_metrics._empty_scaffold(f"scaffold failed: {e}")
        self._geo_scaffold_key = key
        self._geo_summary_key = None   # force a verdict rebuild

    def _ensure_geo_verdicts(self) -> Optional[dict]:
        """The label propagator's substrate: per-geo-cluster human verdict
        tallies plus the pixel->cluster join, or None until the worker has
        published the union scaffold.

        WHY THE UNION SCAFFOLD. It is exactly the right substrate for
        propagation and there is no other full-fidelity one. Per-layer
        scaffolds are deliberately REDUCED to a LayerGeoSummary (30 live ones
        do not fit in memory) and each carries 11-22% of its own points as
        DBSCAN noise, so a per-layer join would propagate to fewer detections
        for a reason that has nothing to do with the reviewer's intent.
        Propagation is model-agnostic anyway: one adjudication scores every
        checkpoint, which is the whole premise of layered review. No union
        geo_prf() is computed or shown.

        Rebuilt only when JudgmentStore.version or the matching version
        actually changes. Everything expensive is already in the scaffold.

        Returns {"verdicts", "member_label", "params", "n_clusters"} or
        {"available": False, "reason": ...}.
        """
        dr = self.dr
        if dr is None:
            return None
        if self._geo_scaffold is None:
            # Published by the worker, not built here. None means it has not
            # landed yet -- the caller renders that as pending, never as zero.
            pub = self._geo_published()
            self._geo_scaffold = pub.get("union")
            if self._geo_scaffold is None:
                return None
            self._geo_scaffold_key = self._geo_worker_key
            self._geo_proj = pub.get("proj")
            self._geo_summary_key = None
        scaffold = self._geo_scaffold
        if not scaffold.ok:
            return {"available": False, "reason": scaffold.unavailable}
        key = (dr._stats_key(), self._geo_scaffold_key)
        if self._geo_summary is not None and self._geo_summary_key == key:
            return self._geo_summary
        from . import geo_metrics
        by_index = {i: row for i, row in enumerate(self._geo_proj.poses)} \
            if self._geo_proj is not None else {}

        # THE JOIN CACHE. Computed for EVERY union detection, not just the
        # judged ones, and keyed on the raw (frame_idx, x, y) -- because which
        # points are judged changes on every commit while this mapping does
        # not. It is the same trick, and very nearly the same code, as
        # _reduce_layer_scaffold(): geo_metrics.locate() is expensive PER
        # FRAME (it recomputes the pose and to_3d for every frame handed to
        # it), so propagating by calling it per keystroke would cost ~1 s a
        # press. Cached, [m] and [a] are dict lookups.
        #
        # Rebuilt on `key` rather than on the scaffold alone because a
        # matching change re-clusters and re-keys the members.
        member_label: Dict[Tuple[int, float, float], int] = {}
        if self._geo_member_label_key == self._geo_scaffold_key \
                and self._geo_member_label is not None:
            member_label = self._geo_member_label
        else:
            for fi, fd in dr.dets_fd.items():
                row = by_index.get(int(fi))
                if row is None or not fd.points:
                    continue
                xy = np.array([[pt.x, pt.y] for pt in fd.points], dtype=float)
                try:
                    labs = geo_metrics.locate(scaffold, self._geo_proj, row, int(fi), xy)
                except Exception:  # noqa: BLE001 -- one bad frame must not lose the join
                    continue
                for pt, lab in zip(fd.points, labs):
                    member_label[(int(fi), pt.x, pt.y)] = int(lab)
            self._geo_member_label = member_label
            self._geo_member_label_key = self._geo_scaffold_key

        # Fan a cluster-level judgment out to its raw members, exactly as
        # JudgmentStore.export() does, so "what a judgment covers" means the
        # same thing here as in the exported sidecars.
        members = dr.member_map()
        judged = []
        for ckey, dets in members.items():
            judgment = dr.store.get(*ckey)
            if judgment in geo_metrics.ABSTAIN_JUDGMENTS:
                continue
            for d in dets:
                judged.append((d.frame_idx, d.x, d.y, judgment))
        verdicts = geo_metrics.collect_verdicts(
            scaffold, self._geo_proj, judged, lambda fi: by_index.get(fi))

        # REACH: how many GUI clusters, over how many frames, each geo cluster
        # would propagate a label to.
        #
        # THIS IS NOT DECORATION, AND IT IS THE MOST IMPORTANT NUMBER HERE.
        # Measured on the full 2,120-frame sample flight (run
        # 17_holdout_pieranch): the median geo cluster covers 16 GUI
        # clusters -- one target seen in sixteen frames, exactly what
        # propagation is for -- but the distribution has a brutal tail. The
        # largest single geo cluster covers 18,146 of the 62,867 clickable
        # clusters (29%), and the five largest cover 46% of the entire review
        # backlog. DBSCAN at eps=0.20 m chains, and on this flight it has
        # chained large stretches of ground into one component.
        #
        # So a single verdict can carry to a third of a flight. That is a
        # property of the clustering, not of this code, and it cannot be
        # silently mitigated here -- narrowing eps is a decision with its own
        # costs (see GEO_METRICS.md). What CAN be done is refuse to hide it:
        # the reach is shown on the selected point, in [p], and in the PDF, so
        # "this verdict propagates to 18,146 clusters in 316 frames" is on
        # screen BEFORE [a] is pressed rather than discovered afterwards.
        reach: Dict[int, List[int]] = {}
        for ckey in dr.cluster_members:
            for lab in cluster_geo_clusters(dr, ckey, member_label):
                r = reach.setdefault(lab, [0, 0, set()])
                r[0] += 1
                r[2].add(ckey[0])
        reach_out = {lab: (r[0], len(r[2])) for lab, r in reach.items()}

        summary = {"available": True, "verdicts": verdicts,
                   "member_label": member_label, "params": scaffold.params,
                   "reach": reach_out,
                   "n_clusters": len({l for l in member_label.values() if l >= 0})}
        self._geo_verdicts = verdicts
        self._geo_summary, self._geo_summary_key = summary, key
        return summary

    def propagation_for(self, ckey: Tuple[int, float, float]) -> Optional[dict]:
        """This cluster's propagated label under the current [o] rule, or None.

        The single entry point every consumer uses -- [m]'s ring, [a]'s plan,
        the selected-point panel and [p]'s coverage counts -- so there is one
        answer to "what would be propagated here" and no second implementation
        to drift from it."""
        if self.propagation_rule is None or self.dr is None:
            return None
        g = self._ensure_geo_verdicts()
        if not g or not g.get("available"):
            return None
        return propagated_label(self.dr, ckey, g["verdicts"], g["member_label"],
                                self.propagation_rule)

    # -- the per-layer geo arm, built off-thread ---------------------------
    #
    # The geospatial block is always on. Layer files carry no accumulated
    # re-emissions, so 3cm dedup removes ~6% of points. Measured on the full
    # 2,120-frame sample flight's 12-layer holdout run: union scaffold 1.74s,
    # twelve layer scaffolds 5.05s, twelve join caches 1.99s -- about 8.9s
    # total, PEAK RSS 0.78 GB set entirely by the union build and flat across
    # every layer build. Retained: ~25 MB of union scaffold plus ~55 MB of
    # layer summaries.
    #
    # So it is computed eagerly, in a worker, publishing INCREMENTALLY: the
    # union lands at ~1.8s and each layer as it finishes, so the HUD fills in
    # top-down instead of showing nothing for nine seconds.

    def _geo_snapshot(self, dr: DirectoryReview) -> Optional[dict]:
        """Everything the worker needs, captured on THIS thread.

        The worker must never touch self.dr, dr.store or any App attribute --
        _reload_layers replaces the whole DirectoryReview, and a worker walking
        dets_fd lazily would read half of one directory and half of another.
        Splitting the union into per-layer frame dicts costs 0.09s measured, so
        it is done here rather than deferred.
        """
        try:
            from . import geo_metrics  # noqa: F401  -- fail here, on the main thread
            from .geo_eval import Projector
        except Exception as e:  # noqa: BLE001
            _report_loop_error(e, self._pose_errors, "geo-import")
            return None
        db = dr.path / "flight_data.db"
        proj = None
        if db.exists():
            try:
                # Built HERE and shared with the worker. Projector closes its
                # sqlite connection in __init__ and is immutable afterwards
                # (K, D, T_IC, poses, clicks), and frame_pose/to_3d/to_2d are
                # pure -- so sharing it is safe and saves reloading the pose
                # table once per layer.
                proj = Projector(db)
            except Exception as e:  # noqa: BLE001
                _report_loop_error(e, self._pose_errors, "geo-projector")
                proj = None
        per_layer: Dict[str, Dict[int, FrameData]] = {}
        for fi, fd in dr.dets_fd.items():
            for pt in fd.points:
                li = getattr(pt, "layer", -1)
                if not (0 <= li < len(dr.layers)):
                    continue
                lid = dr.layers[li].layer_id
                slot = per_layer.setdefault(lid, {})
                got = slot.get(fi)
                if got is None:
                    got = slot[fi] = FrameData(fi, fd.image_path, [])
                got.points.append(pt)
        return {"gen": self._geo_gen, "flight_dir": dr.path, "proj": proj,
                # HEALTHY LAYERS ONLY -- the same union the review pools are
                # built from (healthy_union_frames). `per_layer` above keeps
                # EVERY layer, degenerate included, so each still gets its own
                # scaffold and its own scorecard.
                "union_frames": healthy_union_frames(dr), "per_layer": per_layer,
                "order": [sp.layer_id for sp in dr.layers],
                "eps": self.args.geo_eps, "min_samples": self.args.geo_min_samples,
                "eps_predict": self.args.geo_match_radius,
                "dedup_cm": self.args.geo_dedup_cm}

    def _geo_key(self, dr: DirectoryReview) -> Tuple:
        """What invalidates the SCAFFOLDS (not the scoring).

        THE LAYER SET IS PART OF THIS: folding in a new checkpoint with [l]
        must rebuild the union scaffold, or the verdict layer would refresh
        off a scaffold built over the old layer set.
        """
        return (dr.rel_name, self.args.geo_eps, self.args.geo_min_samples,
                self.args.geo_match_radius, self.args.geo_dedup_cm,
                tuple(sp.layer_id for sp in dr.layers))

    def _ensure_geo_worker(self) -> None:
        """Start (or restart) the geo worker when the scaffold key changes.

        Judgment commits and slider drags do NOT come through here: those
        change the SCORING, which is a dict lookup over the join cache, not the
        geometry. Layer visibility toggles do not either -- geo is computed for
        every layer, like the pixel scorecards are, so pressing a layer off
        must not cost 9 seconds.
        """
        dr = self.dr
        if dr is None:
            return
        key = self._geo_key(dr)
        if key == self._geo_worker_key:
            return
        self._geo_worker_key = key
        self._geo_gen += 1
        gen = self._geo_gen
        with self._geo_lock:
            self._geo_pub = {"gen": gen, "layers": {}, "done": False, "error": None}
        self._geo_scaffold = None
        self._geo_scaffold_key = None
        self._geo_summary_key = None
        snap = self._geo_snapshot(dr)
        if snap is None:
            with self._geo_lock:
                self._geo_pub = {"gen": gen, "layers": {}, "done": True,
                                 "error": "geo_metrics unimportable"}
            return
        t = threading.Thread(target=self._geo_worker, args=(snap,),
                             name=f"geo-{gen}", daemon=True)
        # daemon: [q] must never wait on a scaffold build.
        self._geo_thread = t
        t.start()

    def _geo_worker(self, snap: dict) -> None:
        """Build the union scaffold, then one per layer, publishing as it goes.

        Off-thread and cv2-FREE by construction -- everything here is numpy,
        sklearn and geo_metrics; HighGUI is not thread-safe and all drawing
        stays on the main thread. Errors are captured per unit and published,
        never raised: an optional metric may cost one row, never the session.
        """
        gen = snap["gen"]

        def stale() -> bool:
            return gen != self._geo_gen

        try:
            from . import geo_metrics
        except Exception as e:  # noqa: BLE001
            with self._geo_lock:
                self._geo_pub = {"gen": gen, "layers": {}, "done": True, "error": str(e)}
            return
        kw = dict(eps=snap["eps"], min_samples=snap["min_samples"],
                  eps_predict=snap["eps_predict"], dedup_cm=snap["dedup_cm"],
                  proj=snap["proj"])
        try:
            union = geo_metrics.build_scaffold(snap["flight_dir"], snap["union_frames"], **kw)
        except Exception as e:  # noqa: BLE001
            union = geo_metrics._empty_scaffold(f"scaffold failed: {e}")
        if stale():
            return
        with self._geo_lock:
            self._geo_pub["union"] = union
            self._geo_pub["proj"] = snap["proj"]
        for lid in snap["order"]:
            if stale():
                return
            frames = snap["per_layer"].get(lid) or {}
            try:
                sc = geo_metrics.build_scaffold(snap["flight_dir"], frames, **kw)
                summary = _reduce_layer_scaffold(geo_metrics, lid, sc, frames, snap["proj"])
                # Drop the scaffold before the next build so peak memory is
                # ONE scaffold rather than N -- which is what keeps the
                # 30-layer model-selection screen viable.
                del sc
            except Exception as e:  # noqa: BLE001
                summary = LayerGeoSummary(layer_id=lid, ok=False,
                                          unavailable=f"layer scaffold failed: {e}")
            if stale():
                return
            with self._geo_lock:
                self._geo_pub["layers"][lid] = summary
        with self._geo_lock:
            if not stale():
                self._geo_pub["done"] = True

    def _geo_published(self) -> dict:
        """A snapshot of what the worker has published. The lock is held for
        microseconds: the worker only ever REPLACES these entries, never
        mutates them in place, so a shallow copy is a consistent view."""
        with self._geo_lock:
            pub = dict(self._geo_pub)
            pub["layers"] = dict(pub.get("layers") or {})
        return pub

    def write_geo_metrics(self) -> Optional[Path]:
        """Persist the geo metrics beside review_log.jsonl.

        Called from the explicit [s]/[q] save, NOT from JudgmentStore.export()
        -- that runs on essentially every mutation and would rewrite this file
        on every keystroke. What the HUD shows has to land on disk so the
        numbers are kept; once per save is enough.
        """
        dr = self.dr
        if dr is None:
            return None
        g = self._ensure_geo_verdicts()
        if not g or not g.get("available"):
            return None
        per_layer = self._layer_geo_summaries()
        if not per_layer:
            # Nothing to write: the per-layer arm IS the content, and a file
            # with no results in it would be worse than none.
            return None
        if getattr(dr.store, "dry_run", False) or self.args.dry_run:
            return None
        from . import geo_metrics
        # BESIDE review_log.jsonl, NOT in the flight directory: a
        # geo_metrics.json already in the flight directory may record a
        # different detection set, and a layered session must never overwrite
        # it. JudgmentStore.log_dir is the same separation the judgments get
        # structurally.
        log_dir = Path(getattr(dr.store, "log_dir", dr.path))
        out = log_dir / "geo_metrics.json"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            # PER-MODEL ONLY. There is no top-level union precision/recall:
            # the union is not scored on any surface (HUD, [p], PDF), and
            # this file must not be the one place such a number could be
            # quoted from.
            #
            # `verdicts` and `scaffold` are passed because write_json's
            # signature takes them; it does not currently read either.
            verdicts = g["verdicts"]
            summary = {
                "schema": "per_layer_only",
                "schema_note": (
                    "There is NO top-level union arm. Every "
                    "precision/recall figure lives under per_layer[<layer_id>], "
                    "each entry a geo_prf dict. The union of all checkpoints' "
                    "detections is the review substrate but is not "
                    "scored -- it is a review-burden measure, not any model's "
                    "result."),
                "params": dict(g["params"]),
                "propagation": {
                    "engine": "geo_metrics.GeoVerdicts over the union scaffold",
                    "rule_selected": self.propagation_rule,
                    "rules_available": list(_GEO_VERDICT_RULES),
                    "no_default_rule": True,
                    "n_geo_clusters": g["n_clusters"],
                    "n_clusters_judged": len(verdicts.votes),
                    "n_clusters_conflicted": verdicts.n_conflicted,
                    "n_judged_keys": verdicts.n_judged_keys,
                    "n_unresolved": verdicts.n_unresolved,
                },
            }
            return geo_metrics.write_json(
                out, summary, verdicts, self._geo_scaffold,
                extra={"directory": dr.rel_name,
                       "reviewer": self.args.reviewer,
                       "detection_source": self.geo_source_note(),
                       "gui_params": {"spatial_threshold": self.args.spatial_threshold,
                                      "cluster_radius": self.args.cluster_radius},
                       # Keyed by layer_id, each entry a geo_prf dict.
                       "per_layer": per_layer,
                       "per_layer_note": (
                           "One scaffold per checkpoint over that layer's own detections. "
                           "The recall denominator is shared (it is a property of the pose "
                           "data); `a` is PER LAYER because collect_added's in-cluster test "
                           "is per model; and `c`'s censoring argument does not hold across "
                           "layers -- see n_confirmed_elsewhere in the pixel arm.")})
        except Exception as e:  # noqa: BLE001
            _report_loop_error(e, self._pose_errors, "geo-json")
            return None

    def export_report(self) -> Optional[Path]:
        """[x]: write the [p] panel as a PDF report beside review_log.jsonl.

        Same contract as write_geo_metrics() above and for the same stated
        reason -- anything the GUI puts on screen must also land on disk -- but
        covering what geo_metrics.json does not: the run parameters, the input
        digests, the cross-space comparison, and the caveats. Those existed only
        as pixels in a scrollback nobody kept.

        Lazy import (reportlab, and matplotlib deeper in) so the review loop
        never carries a reporting dependency, and never fatal: an export that
        raised would kill a session mid-review over an optional artifact.
        """
        dr = self.dr
        if dr is None:
            return None
        if getattr(dr.store, "dry_run", False) or self.args.dry_run:
            self.set_status("--dry-run: report NOT written", seconds=6.0)
            return None
        # BESIDE review_log.jsonl, which is what the docs have always said and
        # what write_geo_metrics() already does. This wrote to `dr.path` -- the
        # FLIGHT directory -- and for a layered review the log lives under
        # <run>/review/<flight>/ instead, so the report landed away from the
        # judgments it describes and two runs reviewing the same flight
        # overwrote each other's PDF. Same separation the judgments already get
        # structurally; this artifact had escaped it, exactly as
        # geo_metrics.json had.
        log_dir = Path(getattr(dr.store, "log_dir", dr.path))
        out = log_dir / "review_report.pdf"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            from . import review_report
            self.set_status("building PDF report ...")
            self.render()
            review_report.build_report(out, review_report.collect_context(self))
            self.set_status(f"report written: {out}", seconds=8.0)
            return out
        except Exception as e:  # noqa: BLE001 -- optional artifact, never fatal
            _report_loop_error(e, self._pose_errors, "pdf-report")
            self.set_status(f"report FAILED: {type(e).__name__}: {e}", seconds=8.0)
            return None

    @staticmethod
    def geo_source_note() -> str:
        """What the geo numbers were computed over. Loud on purpose.

        The source is the run's raw per-frame `dets/` files -- one row per
        detection on the frame that produced it, carrying a real confidence.
        It is NOT the birdseye onboard pipeline's results.txt, which this tool
        does not read: that file is an accumulated reprojection log
        (harelab-ucsc/birdseye: birdseye/birdseye.py reprojects every
        accumulated cluster into every later frame), so geo metrics over it
        would count re-emissions rather than detections."""
        return ("run dets/ per-frame detections (raw, real confidences; NOT an "
                "accumulated reprojection log) -- see GEO_METRICS.md")


    def _sync_pose_layer_visibility(self) -> None:
        """Mirror the legend's visibility into the 3D cloud.

        A bool write the scene applies AFTER projection, never a colour
        rebuild -- which is what makes a layer toggle free in the 3D view as
        well as the 2D one.

        The mask is exactly one entry per layer (slot 0 is a model layer), so
        an off-by-one here would silently show the wrong checkpoint."""
        view = self.pose_view
        if view is None or not self.layers.ids:
            return
        mask = [(lid in self.layers.visible and self.layers.enabled)
                for lid in self.layers.ids]
        if getattr(self, "_pose_layer_mask", None) == mask:
            return
        self._pose_layer_mask = mask
        try:
            view.scene.set_layer_visible(mask)
        except Exception as e:  # noqa: BLE001
            _report_loop_error(e, self._pose_errors, "pose-layer-visibility")

    def inline_pose_rect(self, disp_w: int, disp_h: int, canvas_h: int
                          ) -> Optional[Tuple[int, int, int, int]]:
        """(x0, y0, x1, y1) of the band under the frame image that the 3D view
        can use, or None if it would be too short to be worth anything.

        The band exists because `canvas_h = max(window_h, panel_w)` while the
        image is letterboxed to its 1920x1200 aspect, so the 820px side panel
        leaves a permanently black strip below the image. Using it costs the
        review view nothing: the image is not resized, only space that is
        already dead gets filled.

        One predicate, used by BOTH the renderer and the mouse hit-test, so
        the two can never disagree about where the 3D view is."""
        y0 = disp_h + POSE_INLINE_DIVIDER_H
        y1 = canvas_h - POSE_STATUS_RESERVE_H
        if y1 - y0 < MIN_INLINE_POSE_H or disp_w < 320:
            return None
        return (0, y0, disp_w, y1)

    def _inline_rect_now(self) -> Optional[Tuple[int, int, int, int]]:
        """inline_pose_rect() for the geometry of the CURRENT tick, recomputed
        from the same inputs render() uses. Cheap (no image I/O): the native
        frame size is fixed per directory, so it comes from the cached image
        when there is one and the dataset default otherwise."""
        img = self.image_cache_img
        native_h, native_w = img.shape[:2] if img is not None else (1200, 1920)
        geom = DisplayGeometry(native_w, native_h, self.window_w, self.window_h)
        canvas_h = max(self.window_h, self.panel_width())
        return self.inline_pose_rect(geom.disp_w, geom.disp_h, canvas_h)

    def cycle_pose_view(self) -> None:
        """[v]: off -> inline -> separate window -> off.

        Inline is skipped when the window's aspect leaves no usable band --
        presenting a 100px-tall squashed strip would be worse than the window,
        and silently doing nothing would look like [v] was broken."""
        if self.pose_mode == "off":
            if not self._ensure_pose_scene():
                return
            if self._inline_rect_now() is None:
                self._open_pose_window()
                self.set_status("3D view: no room under the image at this window size "
                                 "-- opened in its own window instead ([v] closes)", seconds=8.0)
                return
            self.pose_mode = "inline"
            self.set_status("3D view inline, under the frame -- drag to orbit, right-drag to pan, "
                             "ctrl-drag or +/- zoom, [g] survey/follow, [v] for its own window", seconds=8.0)
        elif self.pose_mode == "inline":
            self.pose_mode = "off"          # so the reopen below is unambiguous
            if not self._ensure_pose_scene():
                return
            self._open_pose_window()
            self.set_status("3D view in its own window -- [v] closes it", seconds=6.0)
        else:
            self.close_pose_view()
            self.set_status("3D view closed")

    def _open_pose_window(self) -> None:
        from . import pose3d
        self.pose_mode = "window"
        # The scene is reused across modes; only the viewport differs. Refit
        # on arrival, since the window's shape has nothing to do with the
        # inline strip's.
        self.pose_view.scene._fitted = False
        cv2.namedWindow(pose3d.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(pose3d.WINDOW, 900, 700)
        cv2.setMouseCallback(pose3d.WINDOW, self.pose_view.on_mouse)

    def close_pose_view(self) -> None:
        """Tear everything down and forget it. Safe to call repeatedly and
        from inside an exception handler -- destroyWindow on an
        already-destroyed window raises on some highgui backends."""
        was_window = self.pose_mode == "window"
        self.pose_mode = "off"
        self.pose_view = None
        self.pose_view_dir = None
        self._pose_drag_capture = False
        self._pose_cloud_fixed = []
        if was_window:
            try:
                from . import pose3d
                cv2.destroyWindow(pose3d.WINDOW)
            except Exception:  # noqa: BLE001
                pass

    def _sync_pose_follow(self) -> None:
        """Ride along with the aircraft while [5]/[6] playback is running.

        At whole-survey zoom one frame moves the aircraft well under a pixel,
        so playback looks frozen unless the camera follows. The pre-playback
        framing is remembered and restored when playback stops -- and a manual
        [g] mid-playback clears that intent, so the automatic behaviour can
        always be overridden."""
        scene = self.pose_view.scene
        playing = self._play_direction != 0
        if playing and self._pose_follow_restore is None:
            self._pose_follow_restore = scene.follow
            if not scene.follow:
                scene.set_follow(True)
        elif not playing and self._pose_follow_restore is not None:
            want, self._pose_follow_restore = self._pose_follow_restore, None
            if scene.follow != want:
                scene.set_follow(want)

    def _feed_pose_scene(self) -> bool:
        """Push this tick's frame into the scene. Returns False when there is
        nothing to render. Shared by the inline and windowed paths so the two
        can never show different frames."""
        if self.pose_view is None or self.dr is None:
            return False
        # A [3]/[4] directory change invalidates the scene: it is built from
        # one flight's db. Rebuild lazily rather than closing the view -- the
        # user asked for a 3D view, not a 3D view of one directory.
        if self.pose_view_dir != self.dr.rel_name:
            target = self.dr.rel_name
            if target in self._pose_unavailable_dirs:
                return False  # already tried and failed; don't re-report every tick
            if not self._ensure_pose_scene():
                # render() runs ~50x/sec, so a directory with no pose data
                # must be remembered, not retried: otherwise every tick
                # re-imports, re-stats the db path, and overwrites the status
                # line, drowning out whatever else it was saying.
                self._pose_unavailable_dirs.add(target)
                return False
        frame_idx = self.current_frame_idx()
        if frame_idx is None:
            return False
        self._sync_pose_follow()
        self._sync_pose_layer_visibility()
        self._sync_pose_centroids()
        dets: List[Tuple[float, float, Tuple[int, int, int]]] = []
        for c in self.dr.unmatched_clusters_by_frame.get(frame_idx, []):
            dets.append((c.x, c.y, judgment_color(self.dr.store.get(c.frame_idx, c.x, c.y))))
        for pt in self.dr.spawned_by_frame.get(frame_idx, []):
            dets.append((pt.x, pt.y, COLOR["spawned"]))
        if self.dr.layers and self.layers.enabled:
            ids = self.layers.ids
            vis = set(self.layers.visible_indices())
            fd = self.dr.dets_fd.get(frame_idx)
            for pt in (fd.points if fd is not None else ()):
                li = getattr(pt, "layer", -1)
                if li in vis:
                    dets.append((pt.x, pt.y,
                                 layer_style(ids[li], ids, self.dr.layers[li].verdict).accent))
        selected = None
        if self.selected_key is not None and self.selected_key[0] == frame_idx:
            selected = (self.selected_key[1], self.selected_key[2])
        self.pose_view.scene.set_frame(frame_idx, self.dr.image_path_by_frame.get(frame_idx),
                                        dets, selected)
        return True

    def _sync_pose_centroids(self) -> None:
        """Push the geo-cluster centroids and their review status into the 3D
        scene, colour-coded by the CURRENT [o] rule.

        WHY THIS EXISTS. The [d] detection cloud is per-layer detector output
        in fixed palette colours, so it does not show review progress (see
        _build_pose_cloud). Geo clusters are not a superimposition of every
        checkpoint: there is ONE per physical spot, whatever the checkpoint
        count, so they show review progress in 3D without the undifferentiated
        smear the per-layer palette exists to prevent.

        STATUS FOLLOWS [o], AND IS ABSENT WHEN [o] IS OFF. A geo cluster's
        colour here is GeoVerdicts.verdict(cluster, rule), which REQUIRES a
        rule -- there is deliberately no default (see geo_metrics.VERDICT_RULES
        and the [o] handler). With [o] off the centroids draw in
        COLOR["centroid_none"] grey: the geometry is shown, no verdict is
        claimed. That is the same refusal the 2D ring, [a]'s plan and [p]'s
        counts all make, and it matters because which verdict wins in a
        conflicted cluster is a convention worth ~14 precision points across
        24 clusters on the full sample flight. Grey is not "unreviewed" -- it is "you have
        not told me which convention to read this under".

        Conflicted clusters (holding BOTH a positive and a negative human
        verdict) additionally get a ring, because those are exactly the ones
        whose fill colour is a function of the convention rather than of
        agreement, and a flat fill would hide that the rule, not the evidence,
        decided them.

        Wide-reaching clusters get an outer ring at CENTROID_REACH_RING_MIN.
        Reach spans 16 (median) to 18,146 (max) review clusters on the full sample flight
        because DBSCAN chains, and drawing an 18,146-cluster blob the same
        size as a 1-cluster spot would hide the single most important fact
        about the substrate being reviewed.

        Gated on (rule, judgment version, scaffold identity) so it is an
        array write on a normal tick, not a recompute 50x a second."""
        dr, view = self.dr, self.pose_view
        if dr is None or view is None:
            return
        # PIGGYBACKS ON THE CACHE, NEVER FORCES IT. Calling
        # _ensure_geo_verdicts() from here instead was wrong twice over: it
        # latches the published scaffold and builds the whole-union join cache
        # from the RENDER path, which runs before union matching has finished,
        # so the cache was keyed and cached against an incomplete
        # dr.cluster_members and never rebuilt -- the smoke test caught it as
        # "[w] 12 of 18 judged points failed to resolve to a geo cluster" plus
        # nine [layers] failures. It also made opening the 3D view silently pay
        # for a 2.3 s scaffold build. The HUD's geo block already calls
        # _ensure_geo_verdicts() every render, so reading its result here is
        # free; until it has run, there are simply no centroids, which is the
        # same laziness every other geo consumer has.
        g = self._geo_summary
        if not g or not g.get("available"):
            return
        scaffold = self._geo_scaffold
        if scaffold is None or not getattr(scaffold, "ok", False):
            return
        key = (self.propagation_rule, self._geo_summary_key)
        if self._pose_centroid_key == key:
            return
        self._pose_centroid_key = key
        try:
            # Both imported locally, matching every other pose3d/geo_metrics
            # call site in this class -- neither is a module-scope dependency
            # (see _ensure_pose_scene and _ensure_geo_scaffold for why).
            from . import geo_metrics
            from . import pose3d
            labels = scaffold.labels
            pts = scaffold.points
            verdicts, rule = g["verdicts"], self.propagation_rule
            reach = g.get("reach") or {}
            ids = np.unique(labels[labels >= 0])
            if not len(ids):
                view.scene.set_centroids(np.zeros((0, 2)), np.zeros((0, 3), np.uint8))
                return
            # One centroid per cluster, in the ground plane -- geo_eval's
            # DBSCAN and its geotag matching both work on UTM x/y and never z,
            # so a z-mean here would be the only thing in the pipeline
            # claiming a height.
            order = np.argsort(labels, kind="stable")
            lab_s = labels[order]
            start = np.searchsorted(lab_s, ids, side="left")
            end = np.searchsorted(lab_s, ids, side="right")
            xy = np.array([pts[order[a:b], :2].mean(axis=0)
                           for a, b in zip(start, end)], dtype=float)
            colors = np.zeros((len(ids), 3), np.uint8)
            conflict = np.zeros(len(ids), dtype=bool)
            wide = np.zeros(len(ids), dtype=bool)
            tally: Dict[str, int] = {}
            for i, lab in enumerate(ids):
                lab = int(lab)
                r = reach.get(lab)
                wide[i] = bool(r and r[0] >= pose3d.CENTROID_REACH_RING_MIN)
                v = verdicts.votes.get(lab) or {}
                conflict[i] = bool(v.get(geo_metrics.POSITIVE_JUDGMENT, 0)
                                   and v.get(geo_metrics.NEGATIVE_JUDGMENT, 0))
                if rule is None:
                    colors[i] = pose3d.COLOR["centroid_none"]
                    continue
                j = verdicts.verdict(lab, rule)
                # None = nobody judged in here, or only abstentions landed:
                # the same four refusal paths propagated_label() has, and the
                # same rendering -- unreviewed orange, no claim made.
                name = j or "unreviewed"
                colors[i] = judgment_color(name)
                tally[name] = tally.get(name, 0) + 1
            note = ""
            n_conf = int(conflict.sum())
            if rule is not None and n_conf:
                note = f"{n_conf} conflicted cluster(s) resolved by [{rule}]"
            view.scene.set_centroids(xy, colors, conflict, wide, tally, rule, note)
        except Exception as e:  # noqa: BLE001 -- an extra layer must never break the view
            _report_loop_error(e, self._pose_errors, "pose3d-centroids")

    # -- model layers: the two drawing tiers ------------------------------

    def _ensure_projector(self):
        """The flight's Projector, built once and shared.

        Hoisted out of _ensure_geo_scaffold so the accumulation tier does not
        require any keypress. Returns None when this flight has no
        flight_data.db (7 of the 30 labelled directories have none)."""
        dr = self.dr
        if dr is None:
            return None
        if getattr(self, "_proj_dir", None) == dr.rel_name:
            return self._proj
        self._proj_dir, self._proj = dr.rel_name, None
        db = dr.path / "flight_data.db"
        if not db.exists():
            self._proj_reason = (f"no flight_data.db in {dr.rel_name} -- the reprojected "
                                 f"accumulation tier needs pose data")
            return None
        try:
            from .geo_eval import Projector
            self._proj = Projector(db)
            self._proj_reason = ""
        except Exception as e:  # noqa: BLE001 -- an extra layer must never break review
            _report_loop_error(e, self._pose_errors, "layer-projector")
            self._proj_reason = f"pose data unusable ({e})"
        return self._proj

    def _ensure_layer_clouds(self):
        """Per-layer world clouds: every detection of every layer, back-
        projected through ITS OWN frame's pose.

        Built once per (directory, layer set) and then reused -- the geometry
        is fixed for the flight. Measured basis: pose3d.build_cloud does
        191,202 points in 0.18 s by grouping into ONE Projector.to_3d call per
        frame rather than per point, and this uses the same shape.

        A uniform grid index is stored alongside so the per-frame reprojection
        can cull in WORLD space before projecting anything -- see
        _draw_layer_accum for why that is what makes this affordable.
        """
        dr = self.dr
        if dr is None or not dr.layers:
            return None
        key = (dr.rel_name, tuple(s.layer_id for s in dr.layers), dr.matching_version)
        if getattr(self, "_layer_cloud_key", None) == key:
            return self._layer_clouds
        proj = self._ensure_projector()
        self._layer_cloud_key, self._layer_clouds = key, None
        if proj is None:
            return None
        try:
            from . import pose3d
            t0 = time.monotonic()
            max_fi = max(dr.frames_all) if dr.frames_all else 0
            positional, warn = pose3d.build_frame_index(proj, max_fi)
            bn = pose3d.basename_index(proj)

            # Group every layer's points by frame, so to_3d is called once per
            # frame rather than once per point.
            by_frame: Dict[int, List[Tuple[float, float, int]]] = {}
            for fi, fd in dr.dets_fd.items():
                slot = by_frame.setdefault(fi, [])
                for p in fd.points:
                    slot.append((p.x, p.y, getattr(p, "layer", -1)))

            xyz_parts, lay_parts = [], []
            for fi in sorted(by_frame):
                row_i = bn.get(Path(dr.image_path_by_frame.get(fi, "")).name)
                if row_i is None:
                    row_i = positional.get(fi)
                if row_i is None:
                    continue
                T_WC, radalt = proj.frame_pose(proj.poses[row_i])
                pts = by_frame[fi]
                arr = np.array([[x, y] for x, y, _ in pts], dtype=float)
                p3 = proj.to_3d(arr, T_WC, radalt)
                # to_3d silently drops non-finite rows, which would desync the
                # parallel layer array -- pose3d.build_cloud drops the whole
                # frame for the same reason, so do the same rather than
                # mis-attributing points to the wrong layer.
                if len(p3) != len(pts):
                    continue
                xyz_parts.append(p3)
                lay_parts.append(np.array([l for _, _, l in pts], dtype=np.int16))
            if not xyz_parts:
                return None
            xyz = np.vstack(xyz_parts).astype(np.float64)
            lay = np.concatenate(lay_parts)
            origin = xyz[:, :2].mean(axis=0)

            # Uniform grid over world XY. Points sorted by cell so a cell is a
            # contiguous slice and the per-frame gather is pure slicing.
            cell = np.floor((xyz[:, :2] - origin) / LAYER_GRID_M).astype(np.int64)
            ckey = cell[:, 0] * 1000003 + cell[:, 1]
            order = np.argsort(ckey, kind="stable")
            xyz, lay, ckey, cell = xyz[order], lay[order], ckey[order], cell[order]
            starts = {}
            uniq, first = np.unique(ckey, return_index=True)
            for i, u in enumerate(uniq):
                stop = first[i + 1] if i + 1 < len(first) else len(ckey)
                starts[int(u)] = (int(first[i]), int(stop))

            self._layer_clouds = {
                "xyz": xyz, "layer": lay, "origin": origin, "cells": starts,
                "basenames": bn,
                "note": pose3d.boresight_note_for(dr.rel_name),
                "built_s": time.monotonic() - t0, "n": len(xyz),
            }
            self.set_status(f"layer clouds: {len(xyz)} points in "
                            f"{self._layer_clouds['built_s']:.2f}s", 3.0)
        except Exception as e:  # noqa: BLE001
            _report_loop_error(e, self._pose_errors, "layer-cloud")
            self._layer_clouds = None
        return self._layer_clouds

    def accum_available(self) -> bool:
        return self._ensure_layer_clouds() is not None

    def accum_note(self) -> str:
        """Why the accumulation tier is unavailable, or what caveat applies."""
        c = getattr(self, "_layer_clouds", None)
        if c is None:
            return getattr(self, "_proj_reason", "") or "no pose data for this flight"
        return c.get("note", "")

    def _project_layer_cloud(self, dr, frame_idx: int, w: int, h: int):
        """(pixels, layer_indices) for the accumulation cloud in THIS frame.

        Culls in world space first. The camera footprint is obtained by
        back-projecting the four image corners, and only the grid cells that
        bbox touches are projected. Measured: 11.4 ms to brute-force the whole
        225k-point corpus per tick, against well under 1 ms this way -- the
        difference between "costs half a frame budget" and "free".

        DELIBERATELY DOES NOT CALL Projector.to_2d. That function divides by
        the depth with errstate suppressed, so a point BEHIND the camera comes
        back as a plausible in-frame pixel; pose3d guards this everywhere with
        an explicit `depth > NEAR_PLANE_M` mask and this must too. The three
        lines are inlined here precisely so the mask can be kept.
        """
        clouds = self._ensure_layer_clouds()
        proj = self._ensure_projector()
        if clouds is None or proj is None:
            return None, None
        try:
            from . import pose3d
            from .geo_eval import AXIS_SWAP
            # basename_index walks every pose row (2,120 on the sample flight), so it is
            # built once per flight rather than once per frame -- doing it per
            # tick cost more than the projection it was serving.
            bn = clouds["basenames"]
            row_i = bn.get(Path(dr.image_path_by_frame.get(frame_idx, "")).name)
            if row_i is None:
                row_i = frame_idx if frame_idx < len(proj.poses) else None
            if row_i is None:
                return None, None
            T_WC, radalt = proj.frame_pose(proj.poses[row_i])

            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
            fp = proj.to_3d(corners, T_WC, radalt)
            if len(fp) != 4:
                return None, None
            origin = clouds["origin"]
            lo = (fp[:, :2].min(axis=0) - origin) / LAYER_GRID_M
            hi = (fp[:, :2].max(axis=0) - origin) / LAYER_GRID_M
            idx = []
            for cx in range(int(np.floor(lo[0])) - 1, int(np.ceil(hi[0])) + 2):
                for cy in range(int(np.floor(lo[1])) - 1, int(np.ceil(hi[1])) + 2):
                    sl = clouds["cells"].get(cx * 1000003 + cy)
                    if sl is not None:
                        idx.append(np.arange(sl[0], sl[1]))
            if not idx:
                return None, None
            sel = np.concatenate(idx)
            pts = clouds["xyz"][sel]

            P = np.hstack([pts[:, :3], np.ones((len(pts), 1))])
            cam = AXIS_SWAP @ (np.linalg.inv(T_WC) @ P.T)
            # DEPTH IS NEGATIVE-FORWARD in this convention. to_3d builds its
            # ground intersection as `T_WC[:,3] - scale * unit` -- note the
            # minus -- so a point the camera can actually see comes back with
            # cam[2] < 0. Verified by round-tripping to_3d -> this projection,
            # which reproduces the source pixels to 1.8e-6 px while cam[2]
            # reads about -3.5 m at a 3.5 m radalt. Testing `cam[2] >
            # NEAR_PLANE_M` (the sign pose3d's own forward projector uses, in
            # ITS camera frame) silently discards every point and looks exactly
            # like "the accumulation tier draws nothing".
            depth = -cam[2, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                pr = proj.K @ cam[:3, :]
                pr = pr / pr[2]
            px = pr[:2].T
            ok = (depth > pose3d.NEAR_PLANE_M) & np.isfinite(px).all(axis=1)
            ok &= (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
            return px[ok], clouds["layer"][sel][ok]
        except Exception as e:  # noqa: BLE001
            _report_loop_error(e, self._pose_errors, "layer-reproject")
            return None, None

    def _draw_layer_accum(self, main, geom, dr, frame_idx: int) -> None:
        """Tier 1: the reprojected accumulation cloud, one colour per layer.

        This shows the whole flight's findings in place on any frame. The
        per-checkpoint dets/ files are raw per-frame detections with no
        accumulation, so the view is built here: back-project through each
        detection's own frame pose, forward-project the cloud into this one.

        Read it knowing what it is. A target seen in 200 frames contributes 200
        points, each landing on that frame's own ground plane, so the cloud
        SMEARS rather than converging to one dot -- a property of per-frame
        ground-plane back-projection rather than a defect here.
        """
        if not dr.layers or not self.layers.enabled:
            return
        mode = self.layers.effective_accum(self.accum_available())
        if mode == "off":
            return
        # NATIVE image dimensions, not the display size: the cloud is projected
        # with the flight's own intrinsics, which belong to proj.res. Passing
        # the letterboxed display size computes a footprint for a camera that
        # does not exist.
        proj = self._ensure_projector()
        nat_w, nat_h = proj.res if proj is not None else (geom.disp_w, geom.disp_h)
        px, lay = self._project_layer_cloud(dr, frame_idx, nat_w, nat_h)
        if px is None or not len(px):
            return
        vis = set(self.layers.visible_indices())
        keep = np.isin(lay, list(vis)) if vis else np.zeros(len(lay), bool)
        px, lay = px[keep], lay[keep]
        if not len(px):
            return
        xs = np.clip((px[:, 0] * geom.scale).astype(np.int32), 0, geom.disp_w - 1)
        ys = np.clip((px[:, 1] * geom.scale).astype(np.int32), 0, geom.disp_h - 1)

        buf = np.zeros_like(main)
        mono = np.zeros(main.shape[:2], np.uint8)
        # PER LAYER, in the palette colours -- the only way this tier is
        # drawn. A "how many models agree" luminance vote would be a union
        # reprojection by another name.
        ids = self.layers.ids
        # Back to front = reverse legend order, so the topmost legend row
        # is the topmost pixel. Deterministic, and explainable when someone
        # asks why one layer is hidden under another.
        for li in sorted(vis, reverse=True):
            st = layer_style(ids[li], ids, dr.layers[li].verdict)
            m = lay == li
            buf[ys[m], xs[m]] = st.color
            mono[ys[m], xs[m]] = 255
        if LAYER_ACCUM_PX > 1:
            k = np.ones((LAYER_ACCUM_PX, LAYER_ACCUM_PX), np.uint8)
            buf, mono = cv2.dilate(buf, k), cv2.dilate(mono, k)
        a = self.args.layer_accum_alpha
        blended = cv2.addWeighted(buf, a, main, 1 - a, 0)
        cv2.copyTo(blended, mono, main)

    def _draw_layer_frame(self, main, geom, dr, frame_idx: int) -> None:
        """Tier 2: THIS frame's own detections, per layer -- larger, nearly
        opaque, shaped by layer and hollowed/struck by health."""
        if not dr.layers or not self.layers.enabled:
            return
        fd = dr.dets_fd.get(frame_idx)
        if fd is None:
            return
        vis = set(self.layers.visible_indices())
        if not vis:
            return
        ids = self.layers.ids
        n = max(1, len(ids))
        overlay = main.copy()
        drew = 0
        for p in fd.points:
            li = getattr(p, "layer", -1)
            if li not in vis:
                continue
            st = layer_style(ids[li], ids, dr.layers[li].verdict)
            sx, sy = geom.to_screen(p.x, p.y)
            ang = 2.0 * math.pi * (li / n) - math.pi / 2.0
            fx = int(round(sx + LAYER_FAN_R * math.cos(ang)))
            fy = int(round(sy + LAYER_FAN_R * math.sin(ang)))
            draw_layer_glyph(overlay, (fx, fy), st, LAYER_FRAME_GLYPH_PX, st.accent)
            drew += 1
        if drew:
            a = self.args.layer_frame_alpha
            cv2.addWeighted(overlay, a, main, 1 - a, 0, main)
        self._last_layer_frame_count = drew


    def _draw_inline_pose(self, canvas, rect: Tuple[int, int, int, int]) -> None:
        """Composite the 3D view into the dead band, inside render().

        Being part of the same pass is what makes the inline view structurally
        frame-accurate: _play_tick has already advanced frame_pos earlier in
        this same render(), so the strip cannot lag the image it sits under."""
        x0, y0, x1, y1 = rect
        cv2.line(canvas, (x0, y0 - 1), (x1 - 1, y0 - 1), COLOR["hud_border"], 1)
        canvas[y0:y1, x0:x1] = self.pose_view.render(x1 - x0, y1 - y0)

    def render_pose_view(self) -> Optional[np.ndarray]:
        """Render for the SEPARATE WINDOW. Returns None when there is nothing
        to show, which the mainloop treats as "skip", not as an error. The
        inline path does not come through here -- see _draw_inline_pose."""
        if self.pose_mode != "window" or not self._feed_pose_scene():
            return None
        # Live-resize, same idea as _sync_window_size() for the main window.
        w, h = 900, 700
        try:
            from . import pose3d
            _, _, rw, rh = cv2.getWindowImageRect(pose3d.WINDOW)
            if rw > 0 and rh > 0:
                w, h = rw, rh
        except Exception:  # noqa: BLE001
            pass
        return self.pose_view.render(w, h)


    def _is_spawned(self, frame_idx: int, x: float, y: float) -> bool:
        dr = self.dr
        if dr is None:
            return False
        return any(p.x == x and p.y == y for p in dr.spawned_by_frame.get(frame_idx, []))

    def spawn_point(self, frame_idx: int, image_path: str, x: float, y: float) -> None:
        """Add a brand-new point the model never detected at all -- pure
        human-added recall signal, distinct from confirmed_real (which
        recovers a detection the model already made)."""
        dr = self.dr
        if dr is None:
            return
        det = Detection(frame_idx, image_path, x, y, confidence=1.0)
        dr.spawned_by_frame.setdefault(frame_idx, []).append(det)
        dr.store.commit(frame_idx, x, y, 1.0, "added", image_path, extra={"source": SOURCE_HUMAN})
        counts = dr.store.export(dr.member_map())
        self.selected_key = (frame_idx, x, y)
        self.mode = "zoom"
        self.set_status(f"added new target: {x:.0f},{y:.0f}  (dir total: {counts['added']} added)")

    def remove_spawned(self, frame_idx: int, x: float, y: float) -> None:
        dr = self.dr
        if dr is None:
            return
        pts = dr.spawned_by_frame.get(frame_idx, [])
        dr.spawned_by_frame[frame_idx] = [p for p in pts if not (p.x == x and p.y == y)]
        dr.store.commit(frame_idx, x, y, 1.0, "removed", dr.image_path_by_frame.get(frame_idx, ""),
                        extra={"source": SOURCE_HUMAN})
        counts = dr.store.export(dr.member_map())
        self.selected_key, self.mode = None, "overview"
        self.set_status(f"removed added target  (dir total: {counts['added']} added)")

    def set_status(self, msg: str, seconds: float = 4.0) -> None:
        self.status_msg = msg
        self.status_until = time.monotonic() + seconds

    # -- rendering --------------------------------------------------------

    def _load_frame_image(self, dr: DirectoryReview, frame_idx: int) -> Optional[np.ndarray]:
        key = (id(dr), frame_idx)
        if self.image_cache_key == key:
            return self.image_cache_img
        img = load_image(dr.image_path_by_frame[frame_idx], self.file_index)
        self.image_cache_key, self.image_cache_img = key, img
        return img

    def render(self) -> np.ndarray:
        a = self.args
        self._sync_window_size()  # live-fit to whatever size the user has dragged the window to
        self._poll_sliders()
        # THE RUN'S dets/ IS WATCHED. A holdout run writes one detection file
        # per checkpoint over several hours, so layers appear while a review
        # is open -- this is what makes [l] able to fold one in. Its
        # signature is a DIRECTORY LISTING rather than a file. Its rate limit
        # and mid-recompute guard live in _poll_layer_dir -- see there.
        self._poll_layer_dir()
        # The presentation sequence is decided ONCE, here, before anything
        # draws or hit-tests -- so the model window, the HUD's per-model
        # groups, [ and ], the scroll clamp and every hit box address the same
        # list. See _sync_layer_order.
        self._sync_layer_order(self.dr)
        # Start (or restart) the geo worker if the SCAFFOLD key changed. Cheap
        # and idempotent when it has not: one tuple compare.
        self._ensure_geo_worker()
        panel_w = self.panel_width()
        canvas_h = max(self.window_h, panel_w)  # never let the panel get clipped

        # Chunked recompute sequence: a settled slider arms a resumable
        # generator (_start_chunked_recompute) instead of blocking for the
        # whole ~13-16s worst case in one call; every subsequent tick
        # advances one time-budgeted chunk (_step_chunked_recompute) and
        # re-renders the banner with LIVE progress, until drained. This
        # works because main()'s loop unconditionally does
        # cv2.imshow(render())+waitKey() every tick, so each partial-
        # progress banner actually reaches the screen. render() stays a
        # pure function throughout (no internal imshow), so direct
        # app.render() calls (as in tests) are unaffected; _apply_slider_
        # recompute() remains available as a separate, fully-blocking,
        # complete-in-one-call path for tests/non-interactive callers.
        if self._recompute_gen is not None:
            still_running = self._step_chunked_recompute()
            if still_running:
                return self._render_recomputing_banner(self.window_w + panel_w, canvas_h)
            # drained + finalized this tick -- fall through to normal rendering below
        elif self._recompute_target_to_show is not None:
            target, self._recompute_target_to_show = self._recompute_target_to_show, None
            self._start_chunked_recompute(*target)
            return self._render_recomputing_banner(self.window_w + panel_w, canvas_h)

        if self.show_stats:
            return self._render_stats_overlay(self.window_w + panel_w, canvas_h)

        if self.show_help:
            return self._render_help(self.window_w + panel_w, canvas_h)

        self._play_tick(time.monotonic())

        dr, frame_idx = self.dr, self.current_frame_idx()
        if dr is None:
            canvas = np.zeros((canvas_h, self.window_w + panel_w, 3), np.uint8)
            cv2.putText(canvas, f"{self.dir_specs[self.dir_idx][1]}: nothing to review "
                                f"(this run has no dets/ for this flight)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR["error"], 2)
            self._draw_status(canvas, self.window_w)
            return canvas

        img = self._load_frame_image(dr, frame_idx) if frame_idx is not None else None
        if img is None:
            canvas = np.zeros((canvas_h, self.window_w + panel_w, 3), np.uint8)
            cv2.putText(canvas, f"frame {frame_idx}: image could not be loaded", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR["error"], 2)
            cv2.putText(canvas, dr.image_path_by_frame.get(frame_idx, "?"), (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR["text"], 1)
            self._draw_status(canvas, self.window_w)
            return canvas

        h, w = img.shape[:2]
        geom = DisplayGeometry(w, h, self.window_w, self.window_h)

        # Static composite (labels + matched context + translucent raw
        # layer) doesn't change between mainloop ticks -- cache it and only
        # redraw the small, frequently-changing centroid/selection layer on
        # top each call. Invalidated explicitly whenever a judgment changes
        # (see commit()/spawn_point()/remove_spawned()/the 'r' handler --
        # NOT part of this cache key, since the raw layer's colors depend on
        # live judgment state that a (dr, frame, size) key alone can't see).
        # dr.matching_version is in the key as well as id(dr): a slider
        # recompute mutates a DirectoryReview IN PLACE, so id(dr) does not
        # change and the cache was relying on the scattered manual
        # `static_cache_key = None` resets to notice. The layer render key
        # covers visibility/solo/accum -- without it, toggling a layer with a
        # warm cache shows the previous composite and reads as "the toggle is
        # broken".
        static_key = (id(dr), dr.matching_version, frame_idx, geom.disp_w, geom.disp_h,
                      self.layers.render_key())
        if self.static_cache_key == static_key:
            main = self.static_cache_img.copy()
        else:
            main = cv2.resize(img, (geom.disp_w, geom.disp_h), interpolation=cv2.INTER_AREA)
            # Tier 1: the reprojected accumulation cloud, UNDER everything --
            # it is context for where each model fires across the whole flight,
            # never a review target.
            self._draw_layer_accum(main, geom, dr, frame_idx)
            for p in dr.labels_by_frame.get(frame_idx, []):
                cv2.circle(main, geom.to_screen(p.x, p.y), 6, COLOR["label"], 1)
            for p in dr.matched_by_frame.get(frame_idx, []):
                # Radius/thickness bumped (was 5px/1px, easy to miss against
                # grass) per user feedback -- these are context-only, never
                # interactive, but should still actually be visible.
                cv2.circle(main, geom.to_screen(p.x, p.y), 8, COLOR["matched"], 2, cv2.LINE_AA)
            # No translucent raw-detection layer is drawn here: tinting every
            # raw union point by its cluster's judgment would be a UNION
            # REPROJECTION -- all thirty checkpoints' detections in one
            # undifferentiated wash, with no way to tell which model put a
            # point there. Tier 2 (_draw_layer_frame, below the centroids)
            # draws the same underlying detections PER LAYER, in the palette
            # colours, which is the same information differentiated.
            self.static_cache_key, self.static_cache_img = static_key, main.copy()

        selected_det, selected_is_spawned = None, False
        # So "[m] on, nothing propagates here" can be distinguished from
        # "[m] off".
        n_rings = 0
        for c in dr.unmatched_clusters_by_frame.get(frame_idx, []):
            j = dr.store.get(c.frame_idx, c.x, c.y)
            sx, sy = geom.to_screen(c.x, c.y)
            cv2.circle(main, (sx, sy), 6, judgment_color(j), -1)
            # [m] PROPAGATION RING. Drawn on `main` unconditionally every
            # tick and never cached in static_cache_img, so toggling the mode
            # can never leave a stale ring behind. A ring around the centroid
            # rather than a text badge, so the call reads at a glance even at
            # a zoomed-out whole-frame view.
            #
            # A candidate label carried here from a
            # human verdict on the same physical spot in another frame.
            #
            # NO RING WHEN NOTHING PROPAGATES, which is the whole reading
            # discipline of this overlay: an unringed orange centroid means
            # "no human has judged this spot and [a] will not touch it", and
            # that has to be visually distinct from "judged negative". A ring
            # in the hard-negative colour is a claim; absence of a ring is
            # not. propagated_label() returns None for all four reasons
            # (no rule, no geo cluster, no verdict, rule abstained) and they
            # are all drawn the same, because the reviewer's next action is
            # the same in every one of them: look at it yourself.
            #
            # Coloured by judgment_color() so a candidate speaks the same
            # colour language as a committed point -- the ring is a preview of
            # exactly what [a] would write.
            if self.model_overlay_mode:
                prop = self.propagation_for((c.frame_idx, c.x, c.y))
                if prop is not None:
                    cv2.circle(main, (sx, sy), MODEL_HALO_RADIUS_PX,
                               judgment_color(prop["judgment"]),
                               MODEL_HALO_THICKNESS_PX, cv2.LINE_AA)
                    n_rings += 1
            n_members = len(dr.cluster_members.get((c.frame_idx, c.x, c.y), []))
            if n_members > 1:
                draw_badge(main, f"x{n_members}", (sx + 14, sy - 14), COLOR["text"], scale=0.42)
            if self.selected_key == (c.frame_idx, c.x, c.y):
                selected_det = c
        self._last_ring_count = n_rings
        # Tier 2: THIS frame's own detections, per layer, fanned around the spot
        # they share. Drawn in the UNCACHED pass, AFTER the consensus centroids:
        # inside the static cache it was painted before the centroid and the
        # translucent raw layer, both of which sit on the same pixel, so the
        # petals were buried and a layered frame looked exactly like an
        # unlayered one. LAYER_FAN_R puts them outside the centroid's 6px disc.
        self._draw_layer_frame(main, geom, dr, frame_idx)
        for p in dr.spawned_by_frame.get(frame_idx, []):
            cv2.drawMarker(main, geom.to_screen(p.x, p.y), COLOR["spawned"], cv2.MARKER_STAR, 14, 2)
            if self.selected_key == (p.frame_idx, p.x, p.y):
                selected_det, selected_is_spawned = p, True
        if selected_det is not None:
            # Radius kept clearly outside the model-overlay halo's outer
            # edge (MODEL_HALO_RADIUS_PX + MODEL_HALO_THICKNESS_PX/2 ~= 12.5)
            # so both rings stay visually distinct when a model-scored
            # cluster is selected, instead of merging into one ring.
            cv2.circle(main, geom.to_screen(selected_det.x, selected_det.y), 16, COLOR["selected"], 2)

        canvas = np.zeros((canvas_h, geom.disp_w + panel_w, 3), np.uint8)
        canvas[:geom.disp_h, :geom.disp_w] = main
        canvas[:, geom.disp_w:] = COLOR["panel_bg"]
        self._draw_panel(canvas, geom.disp_w, panel_w, dr, frame_idx, img, selected_det, selected_is_spawned)
        # Inline 3D BEFORE the status bar: the status text is bottom-anchored
        # to the whole canvas and must stay on top of anything sharing the
        # band. inline_pose_rect() already reserves its rows, so they should
        # not collide -- this is belt and braces.
        if self.pose_mode == "inline":
            rect = self.inline_pose_rect(geom.disp_w, geom.disp_h, canvas_h)
            try:
                if rect is None:
                    # The user resized the window until the band vanished.
                    # Move to the separate window rather than silently
                    # dropping the view they asked for.
                    self._open_pose_window()
                    self.set_status("3D view: no room under the image at this window size "
                                     "-- moved to its own window", seconds=6.0)
                elif self._feed_pose_scene():
                    self._draw_inline_pose(canvas, rect)
            except Exception as e:  # noqa: BLE001
                _report_loop_error(e, self._pose_errors, "pose3d-inline")
                self.close_pose_view()
        self._draw_status(canvas, geom.disp_w, dr, frame_idx)
        return canvas

    @staticmethod
    def _draw_bar(canvas, x: int, y: int, w: int, h: int, frac: float, color) -> None:
        frac = max(0.0, min(1.0, frac))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (70, 70, 70), -1)
        if frac > 0:
            cv2.rectangle(canvas, (x, y), (x + int(w * frac), y + h), color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR["hud_border"], 1)

    def _draw_hud(self, canvas, x0: int, panel_w: int, band: "PanelBand",
                  view: "HudScroll") -> None:
        """Live progress bars + pre-vs-post-adjudication stats, painted into
        the band the layout gave it.

        Directory-scoped, not per-frame: a single frame's handful of judgments
        barely moves precision, the running directory total is what is actually
        informative as you work.

        THE BOX IS PAINTED AT THE HEIGHT IT ACTUALLY DRAWS, so the rows never
        spill out of their own background. There is no "am I about to overlap
        something" early return here, which would silently blank the entire
        HUD on a short panel: non-overlap is panel_layout's guarantee, and
        running out of room truncates the last rows rather than losing the
        whole block."""
        if not band:
            return
        y0 = band.y0
        cv2.rectangle(canvas, (x0 + 6, y0), (x0 + panel_w - 6, band.y1), COLOR["hud_bg"], -1)
        cv2.rectangle(canvas, (x0 + 6, y0), (x0 + panel_w - 6, band.y1), COLOR["hud_border"], 1)

        pad = x0 + HUD_PAD_X
        bar_w = panel_w - 2 * HUD_PAD_X
        limit = band.y1 - HUD_ADV["descent"]

        # The FOOTER is bottom-anchored, not appended after the body. It
        # carries the caveat row, and a caveat that can scroll out of view is
        # a caveat that will be missed -- the same reasoning that makes it
        # priority 0 in status_segments.
        foot_h = sum(r.advance for r in view.footer)
        rows = list(view.pinned) + list(view.body)
        if view.more_above or view.more_below:
            # Say so, in the one place a reader is already looking. Without
            # this, a scrolled HUD is indistinguishable from a short one.
            lo = view.scroll + 1
            rows.append(HudRow("text", HUD_ADV["note"], scale=0.34, cells=[
                (0, f"{'^' if view.more_above else ' '} models "
                    f"{lo}-{view.scroll + view.groups_shown}/{view.n_groups} "
                    f"{'v' if view.more_below else ' '}   [,][.] scroll",
                 (150, 150, 150))]))

        y = y0 + HUD_ADV["top"]
        for row in rows + list(view.footer):
            if row.zone == "footer" and foot_h:
                # Jump to the bottom band once, when the first footer row is
                # reached.
                y = max(y, limit - foot_h + HUD_ADV["warn"])
                foot_h = 0
            if y > limit:
                break
            if row.rule:
                # Separator sits in the gap ABOVE the row's baseline, so a
                # section break costs no vertical budget of its own.
                cv2.line(canvas, (pad, y - 11), (pad + bar_w, y - 11), COLOR["hud_border"], 1)
            if row.kind == "bar":
                self._draw_bar(canvas, pad, y, bar_w, 8, row.frac, row.color)
            else:
                # Each cell is fitted to the gap before the NEXT cell, not to
                # the end of the bar. The old code measured against `bar_w - dx`
                # and the comment claimed that stopped a long section label
                # running under the P column -- it did not, because the label
                # sits at dx=0 and `bar_w - 0` is the whole width. It went
                # unnoticed while every label happened to be short enough;
                # "PIXEL UNION(12) labels.txt m2o 30/30px" is not.
                stops = sorted(dx for dx, _, _ in row.cells)
                for dx, text, colour in row.cells:
                    nxt = next((o for o in stops if o > dx), None)
                    avail = (nxt - dx - 6) if nxt is not None else (bar_w - dx)
                    cv2.putText(canvas, fit_text(text, max(8, avail), row.scale),
                                (pad + dx, y), cv2.FONT_HERSHEY_SIMPLEX,
                                row.scale, colour, 1)
            y += row.advance

    def _layer_geo_summaries(self) -> Dict[str, dict]:
        """Per-layer geo PRE + POST-under-every-rule, keyed by layer_id.

        Only layers the worker has finished appear; a missing key is PENDING
        and the HUD renders it as such. Scored through geo_metrics.geo_prf --
        the SAME function the union arm uses -- so the PRE/POST definitions
        cannot drift between the arms. Memoised on (store.version,
        matching_version, generation, how many layers have landed), so it
        re-scores when a judgment commits or a slider moves and re-keys as
        each layer arrives.

        DEFINITIONS, spelled out because two of them differ from the pixel arm
        deliberately:

        * `n_den` (the recall denominator) is SHARED. Geotags come from
          proj.clicks and in-view from geotags_in_view(proj.poses), both
          functions of pose data alone -- so every layer scores against the
          same 161 geotags and 151 in-view, and only n_hit moves. That is
          exactly the comparison reviewing every checkpoint at once exists to
          make, and it is exact rather than approximate.
        * `a` (added markers) is PER-LAYER, and this diverges from
          layer_metric_rows, where pixel `a` is shared. collect_added drops a
          marker that falls inside a CORE POINT of the scaffold it is given
          ("the detector did fire there"), and that test is per-model by
          construction: a marker inside 014's cluster but outside every
          cluster of 030 is an undiscovered miss for 030 and not for 014.
          Pixel space has no such spatial test, so it cannot express the
          distinction and keeps `a` shared. Both numbers are carried.
        * `c`'s censoring argument does NOT survive across layers, exactly as
          layer_metric_rows already documents for pixel space: a consensus
          cluster a human confirmed that layer L never fired on is a
          DEMONSTRABLE miss for L, so R<= is not an upper bound in the sense
          post_recall derives. The conventional per-layer R<= stays the
          headline and `n_confirmed_elsewhere` is carried beside it, never
          folded in.
        """
        dr = self.dr
        if dr is None or not dr.layers:
            return {}
        pub = self._geo_published()
        landed = pub.get("layers") or {}
        if not landed:
            return {}
        key = (dr._stats_key(), pub.get("gen"), len(landed))
        cached = getattr(self, "_layer_geo_key", None)
        if cached == key:
            return self._layer_geo_cache
        from . import geo_metrics
        proj = pub.get("proj")
        by_index = {i: row for i, row in enumerate(proj.poses)} if proj is not None else {}

        # Fan cluster judgments out to raw members exactly as export() does,
        # then split by layer. One pass, not one per layer.
        judged_by_layer: Dict[str, List[Tuple[Tuple[int, float, float], str]]] = {}
        members = dr.member_map()
        for ckey, dets in members.items():
            judgment = dr.store.get(*ckey)
            if judgment in geo_metrics.ABSTAIN_JUDGMENTS:
                continue
            for d in dets:
                li = getattr(d, "layer", -1)
                if not (0 <= li < len(dr.layers)):
                    continue
                judged_by_layer.setdefault(dr.layers[li].layer_id, []).append(
                    ((int(d.frame_idx), float(d.x), float(d.y)), judgment))
        added_pts = [(fi, pt.x, pt.y)
                     for fi, pts in dr.spawned_by_frame.items() for pt in pts]

        out: Dict[str, dict] = {}
        for lid, sm in landed.items():
            if not sm.ok:
                out[lid] = {"available": False, "reason": sm.unavailable or "unavailable"}
                continue
            view = _ScaffoldView(sm)
            v = geo_metrics.GeoVerdicts()
            for mkey, judgment in judged_by_layer.get(lid, ()):
                # THE JOIN CACHE. A dict lookup where collect_verdicts would
                # recompute a frame pose -- see LayerGeoSummary.
                lab = sm.member_label.get(mkey, -1)
                v.n_judged_keys += 1
                if lab < 0:
                    v.n_unresolved += 1
                else:
                    v.tally(int(lab), judgment)
            added = None
            if added_pts and sm.core_xy is not None:
                try:
                    added = geo_metrics.collect_added(
                        view, proj, added_pts, lambda fi: by_index.get(fi))
                except Exception as e:  # noqa: BLE001 -- one layer, never the arm
                    _report_loop_error(e, self._pose_errors, "geo-layer-added")
            try:
                summary = geo_metrics.geo_prf(view, v, added)
            except Exception as e:  # noqa: BLE001
                _report_loop_error(e, self._pose_errors, "geo-layer-prf")
                summary = {"available": False, "reason": f"scoring failed: {e}"}
            if summary.get("available"):
                # Carried, never folded in -- see the docstring.
                summary["n_noise_points"] = sm.n_noise_points
            out[lid] = summary
        self._layer_geo_key, self._layer_geo_cache = key, out
        return out

    def hud_scroll_by(self, groups: int) -> None:
        """Scroll the HUD body by whole model groups.

        Clamped against the LIVE band, via the same pure function that slices
        it -- so the scroll can never point past the last screenful and the
        next scroll the other way takes effect immediately."""
        view = self._last_hud_view
        n = view.n_groups if view else 0
        if n <= 0:
            return
        self._hud_scroll = max(0, min(self._hud_scroll + groups, max(0, n - 1)))
        shown = view.groups_shown if view else 0
        self.set_status(f"models {self._hud_scroll + 1}-"
                        f"{min(self._hud_scroll + max(1, shown), n)}/{n}")

    def _hud_model_stats(self, dr: DirectoryReview) -> List[HudModelStat]:
        """Per-model HUD rows, in the display order, for the VISIBLE layers.

        Memoised through dr._memo on (store.version, matching_version) like
        every other per-layer aggregate: this is called once per render tick
        and formatting 30 x 7 rows of f-strings per tick is exactly the kind
        of per-frame work that once starved cv2.waitKey.

        Only the visible layers get a group -- the HUD answers "how are the
        models I am looking at doing", and the toggle is the control for that.
        Clustering still runs over ALL layers regardless, so this is a display
        filter and cannot move a centroid.
        """
        met = self._layer_metrics(dr) or {}
        ids = self.layers.ids
        geo_all = self._layer_geo_summaries()
        out: List[HudModelStat] = []
        for lid in self.layers.sequence():
            if lid not in self.layers.visible:
                continue
            m = met.get(lid)
            spec = next((x for x in self.layers.layers if x.layer_id == lid), None)
            if spec is None:
                continue
            st = layer_style(lid, ids, spec.verdict)
            g = geo_all.get(lid)
            geo_pre = geo_post = None
            note = ""
            if g is None:
                note = HUD_PENDING
                geo_post = (None,) * len(_GEO_RULE_ABBRS)
            elif not g.get("available", False):
                note = f"geo n/a: {g.get('reason', '')}"[:40]
                geo_post = (None,) * len(_GEO_RULE_ABBRS)
            else:
                rec = g["recall"]
                geo_pre = (g["pre"]["precision"], rec["recall_all"])
                geo_post = tuple(
                    (g["post"][r]["precision"], rec["post"][r]["recall_all_bound"])
                    for r in g["verdict_rules"])
            out.append(HudModelStat(
                layer_id=lid, label=spec.label, verdict=spec.verdict, color=st.color,
                n_dets=(m.n_dets if m else 0),
                pix_pre=(m.pre if m else None), pix_post=(m.post if m else None),
                geo_pre=geo_pre, geo_post=geo_post, geo_note=note))
        return out

    def hud_model(self, dr: DirectoryReview, frame_idx: Optional[int]) -> List[HudRow]:
        """This directory's HUD as rows -- everything _draw_hud paints, with no
        cv2 in it, so the smoke test can assert column alignment and the height
        budget without scraping pixels."""
        rev, tot = dr.progress()  # clusters -- the real remaining-click count

        # The HUD keeps the two progress bars and one pinned params row; every
        # metric row is per model. No union PIXEL or GEO rows are computed --
        # see hud_rows()'s docstring for why, and _ensure_geo_verdicts() for
        # what the union scaffold is used for.
        #
        # dr.corrected_counts() has no caller here and is kept deliberately:
        # it is the documented POST half of a PRE/POST pair -- the arms are
        # only meaningful together, layer_corrected_counts() is defined as
        # "corrected_counts()'s contract, per layer", and REVIEW_GUI.md
        # defines POST in terms of it.
        layer_note = ""
        if dr.layers and self._layers_stale:
            layer_note = f"{len(self._layers_stale)} NEW LAYER(S) on disk -- [l] to fold in"

        return hud_rows(
            frame_idx=frame_idx, frame_pos=self.frame_pos, n_frames=len(dr.frames_all),
            reviewed=rev, n_clusters=tot, n_raw=dr.unmatched_total(),
            spatial_threshold=self.args.spatial_threshold,
            cluster_radius=self.args.cluster_radius,
            geo_eps=self.args.geo_eps, geo_match=self.args.geo_match_radius,
            prop_note=self.propagation_rule,
            overrides=getattr(self.args, "param_overrides", None),
            n_layers=len(dr.layers),
            layer_note=layer_note,
            model_stats=self._hud_model_stats(dr),
            play_line=play_status_line(self._play_direction, self._play_speed_idx))

    def _selected_point_info_lines(self, dr: DirectoryReview, selected_det: Detection,
                                    selected_is_spawned: bool) -> List[str]:
        """Pure line-builder for the selected-point info block in the side
        panel -- separated from _draw_panel()'s cv2 drawing specifically so
        tests can assert on real string content, mirroring how
        _stats_overlay_lines() was split from _render_stats_overlay()."""
        if selected_is_spawned:
            return [
                f"frame_idx = {selected_det.frame_idx}",
                f"x, y = {selected_det.x:.1f}, {selected_det.y:.1f}",
                "judgment = added (human-placed target)",
                "",
                "[r] remove this point",
            ]  # point-specific, unlike the static hints in the keybind strip
        n_members = len(dr.cluster_members.get(
            (selected_det.frame_idx, selected_det.x, selected_det.y), [selected_det]))
        lines = [
            f"frame_idx = {selected_det.frame_idx}",
            f"x, y = {selected_det.x:.1f}, {selected_det.y:.1f}  (cluster centroid)" if n_members > 1
            else f"x, y = {selected_det.x:.1f}, {selected_det.y:.1f}",
            f"confidence = {selected_det.confidence:.3f} (mean)" if n_members > 1
            else f"confidence = {selected_det.confidence:.3f}",
            f"cluster size = {n_members} raw detection(s)",
            f"judgment = {self._judgment_label(dr, selected_det)}",
        ]
        if dr.layers:
            counts = dr.cluster_layer_counts.get(
                (selected_det.frame_idx, selected_det.x, selected_det.y))
            if counts is not None:
                healthy = [i for i, sp in enumerate(dr.layers) if not sp.degenerate]
                fired_h = sum(1 for i in healthy if counts[i])
                fired_d = sum(1 for i, sp in enumerate(dr.layers)
                              if sp.degenerate and counts[i])
                # THE DENOMINATOR IS HEALTHY LAYERS ONLY. A collapsed model's
                # silence is not evidence and its firing is not agreement; both
                # are still listed and counted, just never folded into the
                # headline number.
                lines.append(f"agreement = {fired_h}/{len(healthy)} healthy models fired"
                             + (f"  (+{fired_d} degenerate)" if fired_d else ""))
        # PROPAGATION EVIDENCE for the selected cluster. This block reports
        # ALL THREE tie-break rules, not just the active one, and that is the
        # point of putting it here: the per-point view is the only place the
        # ~15-precision-point spread between the conventions is actionable. On
        # the HUD three rows per model is a bracket you read; here it tells you
        # whether THIS cluster is one of the conflicted ones whose label the
        # convention actually decides. A cluster where all three agree needs no
        # thought; one where they disagree is asking you to look at it.
        g = self._ensure_geo_verdicts()
        if g is not None and g.get("available"):
            ckey = (selected_det.frame_idx, selected_det.x, selected_det.y)
            labs = cluster_geo_clusters(dr, ckey, g["member_label"])
            if not labs:
                # DBSCAN noise or no resolvable pose -- geo_metrics.locate()
                # conflates those and so does this, because the reviewer
                # cannot act on the difference.
                lines.append("geo cluster: none (DBSCAN noise or no pose) -- nothing can propagate")
            else:
                verdicts = g["verdicts"]
                tally: Dict[str, int] = {}
                for lab in labs:
                    for j, n in (verdicts.votes.get(lab) or {}).items():
                        tally[j] = tally.get(j, 0) + n
                n_cl = sum(g["reach"].get(l, (0, 0))[0] for l in labs)
                n_fr = max((g["reach"].get(l, (0, 0))[1] for l in labs), default=0)
                lines.append("geo cluster " + ",".join(str(l) for l in labs)
                             + ("  votes: " + " ".join(f"{k}={v}" for k, v in sorted(tally.items()))
                                if tally else "  no human verdict yet"))
                # The blast radius, before [a] is pressed rather than after.
                lines.append(f"  reach: {n_cl} cluster(s) in {n_fr} frame(s)")
                calls = []
                for r in _GEO_VERDICT_RULES:
                    prop = propagated_label(dr, ckey, verdicts, g["member_label"], r)
                    mark = "*" if r == self.propagation_rule else " "
                    calls.append(f"{mark}{_GEO_RULE_ABBRS[_GEO_VERDICT_RULES.index(r)]}="
                                 + (prop["judgment"].replace("confirmed_real", "real")
                                    .replace("hard_negative", "hard-neg")
                                    if prop else "-"))
                lines.append("  propagates: " + "  ".join(calls)
                             + ("" if self.propagation_rule else "   ([o] to choose)"))
        # The [c]/[h]/[u]/[r] hints sit in the panel's bottom keybind strip
        # (_draw_panel), not here: they are static text, and at 3 lines they
        # would cost the zoom inset ~66px of the height it competes for --
        # see PANEL_INFO_LINE_H.
        return lines

    @staticmethod
    def _judgment_label(dr: DirectoryReview, det: Detection) -> str:
        """The selected point's judgment, annotated when it was PROPAGATED
        rather than made on this cluster directly.

        A propagated label has no model in it -- it is a human verdict about
        the same physical spot, carried here by geometry -- so the note says
        WHICH CONVENTION carried it, which is the one thing about it a reader
        might want to re-examine.

        Rows with source=classifier carry reference_op and are labelled as
        machine calls: they were written by an automated classifier, not a
        human."""
        key = (det.frame_idx, det.x, det.y)
        verdict = dr.store.get(*key)
        rec = dr.store.judgments.get(key) or {}
        prop = rec.get("propagated")
        if prop:
            return f"{verdict}  (propagated @ {prop.get('rule', '?')})"
        if rec.get("reference_op"):
            # A classifier-written row (source=classifier).
            return f"{verdict}  (CLASSIFIER @ {rec.get('reference_op')}, not human)"
        return verdict

    def _draw_panel(self, canvas, x0: int, panel_w: int, dr: DirectoryReview, frame_idx: int,
                     native_img: np.ndarray, selected_det: Optional[Detection], selected_is_spawned: bool = False) -> None:
        a = self.args
        # ONE layout, computed once and stored: hud_scroll_view slices the
        # rows against the very band that was drawn, and the [,]/[.] clamp
        # reads the view that resulted. Nothing recomputes panel arithmetic in
        # a second place.
        # THE NO-SELECTION HINT IS INFO CONTENT, and it has to be declared as
        # such: routing it through the same list the selected-point info uses
        # means one bound, one band, and no overlap with the HUD's background
        # rect.
        info_lines: List[str] = []
        if selected_det is not None:
            info_lines = self._selected_point_info_lines(dr, selected_det, selected_is_spawned)
        else:
            info_lines = list(PANEL_IDLE_HINT)
        hud_row_list = self.hud_model(dr, frame_idx)
        inset_want = (a.zoom_crop_px * a.zoom_scale) if selected_det is not None else 0
        layout = panel_layout(
            canvas_h=canvas.shape[0], hud_h=hud_rows_height(hud_row_list),
            inset_want=inset_want, info_lines=len(info_lines),
            has_selection=selected_det is not None)
        self._last_panel_layout = layout
        y = layout.inset.y0

        if selected_det is not None:
            crop = a.zoom_crop_px
            h, w = native_img.shape[:2]
            cx, cy = int(round(selected_det.x)), int(round(selected_det.y))
            x1, y1 = max(0, cx - crop // 2), max(0, cy - crop // 2)
            x2, y2 = min(w, x1 + crop), min(h, y1 + crop)
            x1, y1 = max(0, x2 - crop), max(0, y2 - crop)
            patch = native_img[y1:y2, x1:x2]
            if patch.size:
                inset = cv2.resize(patch, (patch.shape[1] * a.zoom_scale, patch.shape[0] * a.zoom_scale),
                                    interpolation=cv2.INTER_NEAREST)
                ih, iw = inset.shape[:2]
                # The layout already decided how tall the inset may be, against
                # a real floor. The old arithmetic here computed a budget that
                # could go NEGATIVE and then forced it to 60px with a
                # `max(60, ...)`, which is how the info text ended up painted
                # over the HUD's top border.
                ih = min(ih, layout.inset.h)
                iw = min(iw, panel_w - 20)
                # Hard clamp to the *actual* canvas bounds too (not just the
                # budgeted layout above) -- guards a pathologically tiny
                # window from a shape-mismatch crash on the paste below.
                ih = max(1, min(ih, canvas.shape[0] - y - 1))
                iw = max(1, min(iw, canvas.shape[1] - (x0 + 10) - 1))
                scale_applied = ih / inset.shape[0]
                if scale_applied < 1.0:
                    iw = min(iw, int(inset.shape[1] * scale_applied))
                canvas[y:y + ih, x0 + 10:x0 + 10 + iw] = cv2.resize(inset, (iw, ih))
                # marker crosshair at the point's location within the (possibly re-shrunk) crop
                mx = x0 + 10 + int((cx - x1) * a.zoom_scale * scale_applied)
                my = y + int((cy - y1) * a.zoom_scale * scale_applied)
                cv2.drawMarker(canvas, (mx, my), COLOR["selected"], cv2.MARKER_CROSS, 16, 2)
        # BOUNDED by the layout, and drawn for BOTH cases. This loop had no
        # bound at all, which is the defect that painted selected-point text
        # over the HUD box. The baseline starts one line INSIDE the band --
        # putText draws upward from the baseline, so starting at info.y0 put
        # the ascenders above the band and clipped the first line against the
        # top of the canvas.
        y = layout.info.y0 + PANEL_INFO_LINE_H - 4
        for line in info_lines[:layout.info_rows]:
            cv2.putText(canvas, fit_text(line, panel_w - 20, 0.48), (x0 + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR["text"], 1)
            y += PANEL_INFO_LINE_H

        # Slice the rows to the band the layout gave the HUD, and STORE the
        # clamped scroll back -- the clamp lives in hud_scroll_view and the
        # stored value has to follow it, or scrolling past the end leaves the
        # attribute out of range and the next scroll the other way looks dead
        # (exactly the defect LayerState.scroll_by documents).
        view = hud_scroll_view(hud_row_list, band_h=layout.hud.h, scroll=self._hud_scroll)
        self._hud_scroll = view.scroll
        self._last_hud_view = view
        self._draw_hud(canvas, x0, panel_w, layout.hud, view)
        self._draw_hint(canvas, x0, panel_w, layout.hint)

    # -- the layer legend widget ------------------------------------------

    def _draw_hint(self, canvas, x0: int, panel_w: int, band: "PanelBand") -> None:
        """The one always-visible help affordance.

        Drawn as a bordered control in COLOR["selected"] at scale 0.5 rather
        than as grey prose at 0.42, because it is the single entry point to
        every key. [q] rides along deliberately: "how do I get out of this"
        must not itself require pressing an unknown key."""
        if not band:
            return
        cv2.rectangle(canvas, (x0 + 6, band.y0 + 2), (x0 + panel_w - 6, band.y1 - 4),
                      COLOR["hud_border"], 1)
        cv2.putText(canvas, fit_text(PANEL_HINT_LINE, panel_w - 28, 0.5),
                    (x0 + 14, band.y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOR["selected"], 1)

    def _layer_metrics(self, dr: Optional[DirectoryReview]
                       ) -> Optional[Dict[str, "ModelMetricRow"]]:
        """Per-model PIXEL scorecards, memoised on (store.version,
        matching_version): they change only when a judgment lands or the
        matching is rebuilt, so the existing key is exact. Recomputing them
        per render tick is the mistake that once starved cv2.waitKey and froze
        the window."""
        if dr is None or not dr.layers:
            return None
        return {m.layer_id: m for m in
                dr._memo("layer_metric_rows",
                         lambda: layer_metric_rows(dr, self.layers.visible))}

    def _layer_sel_counts(self, dr: Optional[DirectoryReview]
                          ) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, float]]]:
        """(counts, mean confidence) per layer for the SELECTED cluster, or
        (None, None) with nothing selected. None is meaningful: it is what
        tells the display order not to partition and the model window to show
        frame counts instead of per-cluster ones."""
        key = self.selected_key
        if dr is None or key is None or key not in dr.cluster_layer_counts:
            return None, None
        cnt = dr.cluster_layer_counts[key]
        cf = dr.cluster_layer_conf.get(key, (0.0,) * len(dr.layers))
        return ({sp.layer_id: cnt[i] for i, sp in enumerate(dr.layers)},
                {sp.layer_id: cf[i] for i, sp in enumerate(dr.layers)})

    def _sync_layer_order(self, dr: Optional[DirectoryReview]) -> None:
        """THE ONE PLACE the presentation sequence is decided. Called once per
        render, before anything draws or hit-tests.

        Nothing else may sort the layers. The defect this replaces was two
        orderings with no owner: _layer_display_order sorted a local copy on
        every render for the table, while LayerState.move_focus walked
        self.ids, so [ and ] and the follow-scroll addressed a different
        sequence than the screen."""
        if not self.layers.ids:
            return
        sel_counts, _ = self._layer_sel_counts(dr)
        self.layers.set_order(model_display_order(
            self.layers.layers, metrics=self._layer_metrics(dr),
            sort=self.layers_sort, sel_counts=sel_counts))

    def model_rows_shown(self) -> int:
        """How many model rows the window shows right now -- the number the
        scroll step and the focus-follow must both use.

        The window owns its own height, so the answer is arithmetic on one
        constant instead of a lookup into whatever the panel last laid out --
        and there is no panel band that a click could be mistakenly routed
        into."""
        if not self.layers.ids:
            return 0
        return layer_rows_shown(len(self.layers.ids),
                                self.model_win_h - 2 * MODEL_WIN_PAD)

    def _model_rows(self, dr: Optional[DirectoryReview],
                    frame_idx: Optional[int]) -> List[LegendRow]:
        """The model window's row model for the CURRENT state -- ONE builder,
        called by both the painter and the hit test, so a click always lands on
        the row the reviewer can see."""
        counts: Dict[str, int] = {}
        if dr is not None and frame_idx is not None:
            fd = dr.dets_fd.get(frame_idx)
            if fd is not None:
                for pt in fd.points:
                    li = getattr(pt, "layer", -1)
                    if 0 <= li < len(self.layers.ids):
                        lid = self.layers.ids[li]
                        counts[lid] = counts.get(lid, 0) + 1
        note = self.accum_note()
        if self._layers_stale:
            note = f"+{len(self._layers_stale)} new layer(s) on disk -- [l] to fold in"
        sel_counts, sel_conf = self._layer_sel_counts(dr)
        # Per-layer detection totals, from the memoised scorecards -- the old
        # adapter never passed `totals` at all, so with no cluster selected
        # and no per-frame count the column fell back to a hardcoded 0.
        met = self._layer_metrics(dr) or {}
        totals = {lid: m.n_dets for lid, m in met.items()}
        return model_window_rows(
            layers=self.layers.layers, order=self.layers.sequence(),
            visible=self.layers.visible, solo=self.layers.solo,
            focus=self.layers.focus, scroll=self.layers.scroll,
            width=MODEL_WIN_W - 2 * MODEL_WIN_PAD,
            max_h=self.model_win_h - 2 * MODEL_WIN_PAD,
            accum_mode=self.layers.accum_mode, accum_available=self.accum_available(),
            accum_effective=self.layers.effective_accum(self.accum_available()),
            note=note, sort=self.layers_sort,
            counts_this_frame=counts, totals=totals,
            sel_counts=sel_counts, sel_conf=sel_conf)

    def model_window_open(self) -> bool:
        return bool(self.model_win) and bool(self.layers.ids)

    def toggle_model_window(self) -> None:
        """[e]: open or close the model window.

        Every layer key ([l] [i] [;] ['] [[] []] [0]) works whether the window
        is open or shut, because cv2.waitKey() reports the keystroke but not
        which window had focus.
        """
        if not self.layers.ids:
            self.set_status("no model layers -- point this at a run with dets/ for this flight")
            return
        if self.model_win:
            self.close_model_window()
            self.set_status("model layers window closed  [e] to reopen")
            return
        self.open_model_window()
        self.set_status("model layers: click a row to toggle, right-click to solo, "
                        "[ALL]/[NONE]/[ACC]/[SORT]  [e] closes", seconds=8.0)

    def open_model_window(self) -> None:
        w, h = model_window_size(len(self.layers.ids))
        self.model_win = True
        self.model_win_h = h
        # AUTOSIZE, NOT WINDOW_NORMAL -- and this is the one load-bearing
        # divergence from the 3D pose window's pattern. That window is
        # resizable because orbiting wants a viewport and a drag has no hit
        # boxes. This window is ENTIRELY hit boxes, and AUTOSIZE pins it to the
        # canvas so the mouse coordinates Qt reports ARE the canvas
        # coordinates the rows were laid out in. No getWindowImageRect
        # rescale, and therefore no second source of truth about where a row
        # is -- which is how a click lands on the row above the one clicked.
        cv2.namedWindow(MODEL_WINDOW, cv2.WINDOW_AUTOSIZE)
        # Re-registered on every open, never hoisted out of the reopen path:
        # Qt drops the callback along with the window. _open_pose_window does
        # the same, for the same reason.
        cv2.setMouseCallback(MODEL_WINDOW, self.on_model_mouse)

    def close_model_window(self) -> None:
        """Safe to call repeatedly and from inside an exception handler --
        destroyWindow on an already-destroyed window raises on some highgui
        backends. Every piece of layer state lives in LayerState and is
        entirely indifferent to whether a window exists, so closing loses
        nothing."""
        was_open, self.model_win = self.model_win, False
        if was_open:
            try:
                cv2.destroyWindow(MODEL_WINDOW)
            except Exception:  # noqa: BLE001
                pass

    def render_model_window(self) -> Optional[np.ndarray]:
        if not self.model_window_open():
            return None
        dr = self.dr
        rows = self._model_rows(dr, self.current_frame_idx())
        w, h = MODEL_WIN_W, self.model_win_h
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = COLOR["hud_bg"]
        cv2.rectangle(canvas, (1, 1), (w - 2, h - 2), COLOR["hud_border"], 1)
        self._paint_model_rows(canvas, MODEL_WIN_PAD, MODEL_WIN_PAD,
                               w - 2 * MODEL_WIN_PAD, rows)
        return canvas

    def _paint_model_rows(self, canvas, wx0: int, wy0: int, ww: int,
                          rows: Sequence[LegendRow]) -> None:
        """Paint a model-row list at an origin -- the one painter for these
        rows."""
        ids = self.layers.ids
        for r in rows:
            base_y = wy0 + r.y + (LAYER_ROW_H - 5 if r.kind in ("layer", "metric")
                                  else r.h - 6)
            if r.kind == "layer" and r.swatch is not None:
                li = ids.index(r.tag) if r.tag in ids else 0
                st = layer_style(r.tag, ids, self.layers.layers[li].verdict)
                draw_layer_glyph(canvas, (wx0 + LAYER_X_SWATCH + 5, base_y - 4), st, 9, st.color)
                draw_layer_glyph(canvas, (wx0 + LAYER_X_GLYPH + 5, base_y - 4), st, 9, st.accent)
                if r.emphasis:
                    cv2.putText(canvas, ">", (wx0 - 1, base_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.40, COLOR["selected"], 1)
                elif r.fired:
                    # This layer fired on the selected cluster. A filled dot in
                    # the gutter, so "which models found this spot" reads down
                    # the left edge without parsing the count column.
                    cv2.circle(canvas, (wx0 + 2, base_y - 4), 2, st.accent, -1)
            # Fit each cell to the gap before the NEXT one, not to the end of
            # the widget. Fitting to the full width is what let the count
            # column ("1d .09") run straight through the P column -- the same
            # defect _draw_hud had with its section labels.
            stops = sorted((dx if dx >= 0 else ww + dx) for dx, _, _ in r.cells)
            for dx, text, colour in r.cells:
                x = wx0 + (dx if dx >= 0 else ww + dx)
                here = dx if dx >= 0 else ww + dx
                nxt = next((o for o in stops if o > here), None)
                avail = (nxt - here - 4) if nxt is not None else (ww - here - 4)
                cv2.putText(canvas, fit_text(text, max(8, avail), 0.36), (x, base_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, colour, 1)
                if r.strike and dx == LAYER_X_LABEL:
                    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
                    cv2.line(canvas, (x, base_y - 4), (x + tw, base_y - 4), colour, 1)
            if r.kind == "btn":
                bx0, by0, bx1, by1 = r.hit
                cv2.rectangle(canvas, (wx0 + bx0, wy0 + by0), (wx0 + bx1, wy0 + by1),
                              COLOR["hud_border"] if r.enabled else (60, 60, 60), 1)

    def on_model_mouse(self, event, x, y, flags, userdata) -> None:
        """Clicks in the model window.

        Deliberately does NOT stop playback, unlike on_mouse: that one resets
        _play_direction on any click because it is the review surface, whereas
        toggling a layer while watching the flight play back is a legitimate
        gesture and ending playback would punish it."""
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        if not self.model_window_open():
            return
        rows = self._model_rows(self.dr, self.current_frame_idx())
        hit = layer_hit_test(rows, x - MODEL_WIN_PAD, y - MODEL_WIN_PAD)
        if hit is not None:
            self._apply_layer_action(hit, event == cv2.EVENT_RBUTTONDOWN)

    def _poll_layer_dir(self) -> None:
        """Notice a checkpoint that finished while the review was open.

        REPORTS, NEVER RELOADS. Folding a new layer in adds points that can
        merge two previously-distinct consensus clusters, which moves every
        centroid in every frame that layer touches -- re-keying judgments under
        the reviewer's cursor mid-click. So this sets a flag, says so on the
        HUD's existing caveat row and in the status bar, and waits for [l].

        A signature must be STABLE across one poll before it is acted on,
        because run_detector.py is writing into the same directory. *.partial
        is excluded from the signature entirely (see layer_dir_signature) --
        it changes on every poll by construction and would make this flap
        forever.

        THE RATE LIMIT AND THE MID-RECOMPUTE GUARD LIVE HERE. Without them a
        directory listing would run on every render tick, and a settle window
        would be measured in ticks rather than seconds -- so a file still
        being written could satisfy "stable across one poll" in 50 ms.
        """
        ctx = getattr(self.args, "run_context", None)
        dr = self.dir_cache.get(self.dir_idx)
        if ctx is None or ctx.run is None or dr is None or not dr.layers:
            return
        # Never mid-recompute: the caller may be about to re-cluster, and a
        # layer notice arriving in the middle of that reads as being about the
        # wrong clustering.
        if self._recompute_gen is not None:
            return
        now = time.monotonic()
        if now - self._layer_last_poll < LAYER_POLL_SECONDS:
            return
        self._layer_last_poll = now
        sig = layer_dir_signature(layer_source_dir(ctx.run, dr.rel_name))
        if sig == self._layer_sig:
            self._layer_sig_pending = None
            return
        if getattr(self, "_layer_sig_pending", None) != sig:
            self._layer_sig_pending = sig      # settle: seen once, act next time
            return
        known = {sp.layer_id for sp in dr.layers}
        new = [Path(n).stem for n, _, _ in sig if Path(n).stem not in known]
        self._layer_sig, self._layer_sig_pending = sig, None
        if new:
            self._layers_stale = new
            self.set_status(f"+{len(new)} layer(s) finished on disk ({', '.join(new)}) "
                            f"-- press [l] to fold them in", 10.0)

    def _layer_key(self, ch: str) -> None:
        """Keyboard mirror of the legend's mouse verbs. Mouse is primary; these
        exist so the feature is usable without leaving the keyboard mid-review.

        Every binding is UNSHIFTED. cv2.waitKey does not reliably deliver Shift
        as a case change on every highgui backend (Qt in particular), so a
        shifted letter arrives indistinguishable from its lowercase form and
        silently fires the wrong action."""
        ls = self.layers
        if not ls.ids:
            if ch in ("l", "e"):
                self.set_status("no model layers -- pass a results/<run> directory with dets/")
            return
        if ch == "l":
            if self._layers_stale:
                self._reload_layers()
                return
            ls.enabled = not ls.enabled
            self.set_status(f"model layers {'on' if ls.enabled else 'off'}")
        elif ch == "e":
            self.toggle_model_window()
        elif ch == "i":
            m = ls.cycle_accum()
            eff = ls.effective_accum(self.accum_available())
            note = self.accum_note()
            self.set_status(f"accumulation: {m}"
                            + (f" -> {eff} ({len(ls.visible)} layers on)" if eff != m else "")
                            + (f"   {note}" if note else ""))
        elif ch == ";":
            if ls.focus:
                on = ls.toggle(ls.focus)
                self.set_status(f"{ls.focus} {'on' if on else 'off'}  "
                                f"({len(ls.visible)}/{len(ls.ids)} layers)")
        elif ch == "'":
            if ls.focus:
                on = ls.set_solo(ls.focus)
                self.set_status(f"solo {ls.focus}" if on else "solo off")
        elif ch in ("[", "]"):
            tag = ls.move_focus(-1 if ch == "[" else 1, self.model_rows_shown())
            spec = next((x for x in ls.layers if x.layer_id == tag), None)
            self.set_status(f"focus {tag}" + (f"  [{spec.verdict}]" if spec else ""))
        elif ch == "0":
            ls.all_on()
            self.set_status(f"all {len(ls.ids)} layers on")
        self.static_cache_key = None

    def _reload_layers(self) -> None:
        """Fold newly-finished detection files into the review.

        EXPLICIT, never automatic. A new layer adds points that can merge two
        previously-distinct consensus clusters, which moves every centroid in
        every frame that layer touches -- re-keying judgments under the
        reviewer's cursor. So the poller only ever reports; this is the
        reviewer saying "now".
        """
        idx = self.dir_idx
        self.dir_cache.pop(idx, None)
        stale = list(self._layers_stale)
        self._layers_stale = []
        self.static_cache_key = None
        self._layer_cloud_key = None
        # Reset the geo arm too: folding in a checkpoint must not leave the
        # union scaffold built over the OLD layer set while the verdict layer
        # refreshes off it, or the geo numbers would describe a detection set
        # that is not on screen. _ensure_geo_worker will restart on the next
        # render because the key carries the layer ids.
        self._geo_scaffold = None
        self._geo_scaffold_key = None
        self._geo_summary = None
        self._geo_summary_key = None
        self._layer_geo_key = None
        self._layer_geo_cache = {}
        dr = self._dr(idx)
        self.set_status(f"folded in {len(stale)} new layer(s): {', '.join(stale)}"
                        if dr is not None else "layer reload failed -- see console", 6.0)

    def _reset_hud_scroll(self) -> None:
        """The HUD's group scroll is per-directory: model 7 of one flight has
        nothing to do with model 7 of the next."""
        self._hud_scroll = 0
        self._last_hud_view = None

    def _apply_layer_action(self, row: LegendRow, right: bool) -> None:
        """One legend click. Right-click on a layer solos it; left toggles."""
        ls = self.layers
        if row.action == "toggle" and row.tag:
            # A plain toggle does NOT move the focus. The focused model emits
            # an extra PRE row, so focusing on click reflowed every row below
            # the cursor mid-gesture -- the next click then landed on a
            # different model than the one under the pointer. Focus moves with
            # [ and ], and with an explicit solo.
            if right:
                ls.focus = row.tag
                on = ls.set_solo(row.tag)
                self.set_status(f"solo {row.tag}" if on else "solo off")
            else:
                on = ls.toggle(row.tag)
                self.set_status(f"{row.tag} {'on' if on else 'off'}  "
                                f"({len(ls.visible)}/{len(ls.ids)} layers)")
        elif row.action == "all":
            ls.all_on(); self.set_status(f"all {len(ls.ids)} layers on")
        elif row.action == "none":
            ls.none_on(); self.set_status("all layers off")
        elif row.action == "sort":
            # Cycle the sort key the display order reads.
            i = MODEL_SORT_CYCLE.index(self.layers_sort) \
                if self.layers_sort in MODEL_SORT_CYCLE else 0
            self.layers_sort = MODEL_SORT_CYCLE[(i + 1) % len(MODEL_SORT_CYCLE)]
            self._sync_layer_order(self.dr)
            self.set_status(f"models sorted by {self.layers_sort}")
        elif row.action == "accum":
            m = ls.cycle_accum()
            eff = ls.effective_accum(self.accum_available())
            self.set_status(f"accumulation: {m}" + (f" -> {eff} ({len(ls.visible)} layers on)"
                                                     if eff != m else ""))
        elif row.action in ("up", "down"):
            # Against the LIVE height, not a max constant -- and through the
            # state's own clamp, so LayerState.scroll never drifts past the
            # end and makes the next "up" click look dead.
            shown = self.model_rows_shown()
            ls.scroll_by((shown - 1) if row.action == "down" else -(shown - 1),
                         len(ls.ids), shown)
        self.static_cache_key = None      # visibility is part of the composite

    def _draw_status(self, canvas, main_w: int, dr: Optional[DirectoryReview] = None,
                      frame_idx: Optional[int] = None) -> None:
        y = canvas.shape[0] - 12
        if dr is not None and frame_idx is not None:
            dir_rev, dir_tot = dr.progress()
            frame_rev, frame_tot = dr.frame_progress(frame_idx)
            caveat = ""
            if dr.layers:
                caveat = self.accum_note() if self.layers.accum_mode != "off" else ""
                if self._layers_stale:
                    caveat = f"+{len(self._layers_stale)} new layer(s) -- [l]"
            segs = status_segments(
                dir_idx=self.dir_idx, n_dirs=len(self.dir_specs), rel_name=dr.rel_name,
                frame_idx=frame_idx, frame_pos=self.frame_pos, n_frames=len(dr.frames_all),
                frame_rev=frame_rev, frame_tot=frame_tot, dir_rev=dir_rev, dir_tot=dir_tot,
                layers_on=len(self.layers.visible), layers_total=len(self.layers.ids),
                solo=self.layers.solo,
                accum=(self.layers.effective_accum(self.accum_available())
                       if dr.layers and self.layers.accum_mode != "off" else ""),
                caveat=caveat)
            cv2.putText(canvas, fit_status(segs, main_w - 20), (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR["text"], 1)
        if self.status_msg and time.monotonic() < self.status_until:
            # fit_text, because this line had NO clamp at all and a long
            # message ran straight under the side panel. Layer toggles generate
            # a lot of status traffic, so it would have started showing.
            cv2.putText(canvas, fit_text(self.status_msg, main_w - 20, 0.5), (10, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR["selected"], 1)

    def _render_recomputing_banner(self, w: int, h: int) -> np.ndarray:
        """Live progress for a chunked recompute in flight -- reads
        self._recompute_target/_frames_done/_frames_total/_started_at
        directly (set by _start_chunked_recompute(), updated every tick by
        _step_chunked_recompute())."""
        canvas = np.zeros((h, w, 3), np.uint8)
        spatial, cluster = self._recompute_target
        done, total = self._recompute_frames_done, self._recompute_frames_total
        elapsed = time.monotonic() - self._recompute_started_at
        frac = done / total if total else 0.0
        eta = f"{elapsed / done * (total - done):.1f}s" if done else "--"

        y0 = h // 2 - 60
        title = {
            "index": "Re-indexing [m] model overlay ...",
            "reload_index": "Re-indexing [m] model overlay (file modified on disk) ...",
        }.get(self._recompute_phase, "Recomputing matching + clustering ...")
        cv2.putText(canvas, title, (30, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR["selected"], 2)
        cv2.putText(canvas, f"spatial={spatial:.0f}px  cluster={cluster:.0f}px",
                    (30, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR["text"], 1)
        self._draw_bar(canvas, 30, y0 + 50, min(500, w - 60), 14, frac, COLOR["bar_review"])
        cv2.putText(canvas, f"frame {done}/{total} ({frac*100:.0f}%)   elapsed {elapsed:.1f}s   ETA {eta}",
                    (30, y0 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR["text"], 1)
        return canvas





    def _summary_overlay_lines(self) -> List[Tuple[str, Tuple[int, int, int]]]:
        """The [p] overlay's leading SUMMARY block: everything a reviewer
        decides on, above the fold.

        `_draw_paginated` shows ~38 lines at a time, so whatever is first is
        what most readers ever see. This block therefore carries the review's
        own composition and any live caveat -- and every number is read from
        the same objects the detailed sections below re-read, never
        recomputed, so the summary cannot disagree with the detail.

        THERE IS NO UNION SCORECARD HERE, matching the HUD: a pixel PRE->POST
        pair over the union of every active checkpoint is a review-burden
        measure and nobody's precision. The per-model block below is the
        scorecard.
        """
        text, err, head = COLOR["text"], COLOR["error"], COLOR["selected"]
        dr = self.dr
        out: List[Tuple[str, Tuple[int, int, int]]] = [
            ("SUMMARY -- press [x] to export all of this as a PDF report", head),
        ]
        if dr is None:
            return out + [("  no directory loaded", text), ("", text)]

        out.append((f"  directory: {dr.rel_name}    reviewer: {self.args.reviewer}", text))
        ctx = getattr(self.args, "run_context", None)
        if ctx is not None and ctx.run is not None:
            out.append((f"  run:       {ctx.run.name}"
                        + (f"   matching={ctx.matching}" if ctx.matching else "")
                        + (f"   spatial<={ctx.spatial_threshold:g}px"
                           if ctx.spatial_threshold is not None else ""), text))
            for n in ctx.notes:
                out.append((f"    note: {n}", text))
        else:
            out.append(("  run:       (none given -- derived params are off)", text))
        rev, tot = dr.progress()
        out.append((f"  reviewed:  {rev}/{tot} clusters "
                    f"({(rev / tot * 100) if tot else 0:.0f}%)   "
                    f"{dr.unmatched_total()} raw detections   "
                    f"{len(dr.layers)} model layer(s)", text))
        out.append((f"  propagation rule ([o]): "
                    + (self.propagation_rule or "off -- nothing is proposed"),
                    text if self.propagation_rule else head))

        # Caveats above the fold -- a number the reader has already seen must
        # not be un-said three screens later.
        caveats: List[str] = []
        for k, v in (getattr(self.args, "param_overrides", None) or {}).items():
            caveats.append(
                f"{k} OVERRIDDEN: this run's detections were produced at {v['run']:g}, "
                f"you are reviewing at {v['live']:g} -- every number below is measured "
                f"under different matching than the run's own numbers")
        if ctx is not None and ctx.test_dir and not ctx.scores_honest_for_test_dir \
                and dr.rel_name == ctx.test_dir:
            caveats.append(
                f"{dr.rel_name} is this run's HELD-OUT TEST directory and "
                f"{ctx.run.name} does not record scores_are_honest_for_test_dir")
        g = self._ensure_geo_verdicts()
        if g is not None and g.get("available"):
            n_conf = g["verdicts"].n_conflicted
            if n_conf:
                caveats.append(
                    f"{n_conf} geo cluster(s) hold BOTH a confirmed_real and a hard_negative. "
                    "Which one propagates is decided by the [o] convention, not by data -- "
                    "worth ~15 precision points here. There is no primary rule.")
            if g["verdicts"].n_unresolved:
                caveats.append(
                    f"{g['verdicts'].n_unresolved} judged detection(s) reach no geo cluster "
                    "(DBSCAN noise or no pose) -- nothing propagates from or to them")
        if caveats:
            out.append(("", text))
            for c in caveats:
                out.append((f"  ! {c}", err))
        out.append(("", text))
        out.append(("SECTIONS BELOW: per model . adjudication . post-processing config . geospatial",
                    head))
        out.append(("", text))
        return out

    def _per_model_overlay_lines(self) -> List[Tuple[str, Tuple[int, int, int]]]:
        """PER MODEL: the scorecard block, at the granularity that replaced
        the union.

        Carries what the HUD's per-model groups carry PLUS the raw geotag
        counts behind the geo ratios. Those counts already existed -- every
        one of them comes out of geo_prf and is written to geo_metrics.json
        under per_layer -- but they had never been shown anywhere, so a reader
        could see `GEO PRE 0.1888  0.8137` and not know it stood on 131 of 161
        geotags. A ratio without its denominator is not quotable.
        """
        text, head, err = COLOR["text"], COLOR["selected"], COLOR["error"]
        dr = self.dr
        out: List[Tuple[str, Tuple[int, int, int]]] = [
            ("PER MODEL -- one adjudication, every checkpoint scored by it", head)]
        if dr is None or not dr.layers:
            return out + [("  no model layers loaded", text), ("", text)]
        met = self._layer_metrics(dr) or {}
        geo_all = self._layer_geo_summaries()
        try:
            from . import geo_metrics
            abbr = geo_metrics.VERDICT_RULE_ABBR
        except Exception:  # noqa: BLE001 -- an optional metric, never fatal
            abbr = {r: a for r, a in zip(_GEO_VERDICT_RULES, _GEO_RULE_ABBRS)}
        for lid in self.layers.sequence():
            m = met.get(lid)
            spec = next((x for x in self.layers.layers if x.layer_id == lid), None)
            if spec is None:
                continue
            vis = "" if lid in self.layers.visible else "   [hidden]"
            tag = {"collapsed": "  COLLAPSED", "sparse": "  SPARSE"}.get(spec.verdict, "")
            out.append((f"  {spec.label}{tag}{vis}", head if not tag else err))
            if m is None:
                out.append(("    computing...", text))
                out.append(("", text))
                continue
            out.append((f"    {'':<14} {'P':>9} {'R':>10} {'F1':>9}", text))
            out.append((f"    {'PIXEL PRE':<14} {m.pre[0]:>9.4f} {m.pre[1]:>10.4f} "
                        f"{m.pre[2]:>9.4f}", COLOR["pre"]))
            out.append((f"    {'PIXEL POST':<14} {m.post[0]:>9.4f} {m.post[1]:>9.4f}<= "
                        f"{m.post[2]:>8.4f}<=", COLOR["post"]))
            out.append((f"    {'':<14} n_dets={m.n_dets}  fired in {m.n_clusters_fired} clusters  "
                        f"c={m.n_confirmed} hard-neg={m.n_hard}", text))
            if m.n_confirmed_elsewhere:
                # Never folded into R<=: a confirmed cluster this layer did not
                # fire on is a DEMONSTRABLE miss for it, so the bound argument
                # does not hold. See layer_metric_rows().
                out.append((f"    {'':<14} {m.n_confirmed_elsewhere} cluster(s) confirmed real that "
                            f"this layer never fired on (NOT in R<=)", text))
            gm = geo_all.get(lid)
            if gm is None:
                out.append((f"    {'GEO':<14} computing...", text))
            elif not gm.get("available"):
                out.append((f"    {'GEO':<14} unavailable: {gm.get('reason','')}", err))
            else:
                rec = gm["recall"]
                out.append((f"    {'GEO PRE':<14} {gm['pre']['precision']:>9.4f} "
                            f"{rec['recall_all']:>10.4f}", COLOR["pre"]))
                # ALL THREE RULES, ALWAYS. The spread between them IS the error
                # bar on POST; a single row would be an editorial choice about
                # the result. See geo_metrics.VERDICT_RULES.
                for rule in gm["verdict_rules"]:
                    gp, gr = gm["post"][rule], rec["post"][rule]
                    out.append((f"    {'GEO POST ' + abbr[rule]:<14} {gp['precision']:>9.4f} "
                                f"{gr['recall_all_bound']:>9.4f}<=", COLOR["post"]))
                # THE COUNTS BEHIND THE RATIOS -- the enrichment this block
                # exists for.
                out.append((f"    {'':<14} geotags hit {rec['tp_geotags']}/{rec['n_geotags']} "
                            f"(in view {rec['tp_geotags_in_view']}/{rec['n_in_view']}), "
                            f"missed {rec['fn_geotags']}", COLOR["geo"]))
                pc = rec.get("post_common", {})
                out.append((f"    {'':<14} clusters {gm.get('n_clusters','?')}  "
                            f"a={pc.get('a_added_units','?')} added marker unit(s)  "
                            f"cov {gm['adjudication']['coverage'] * 100:.0f}%", COLOR["geo"]))
            out.append(("", text))
        out.append(("  pixel and geo are NOT comparable -- different units and ground truth. "
                    "Geo has no F1: precision is per cluster, recall per geotag.", text))
        out.append(("", text))
        return out

    def _adjudication_provenance_lines(self) -> List[Tuple[str, Tuple[int, int, int]]]:
        """ADJUDICATION: what is in this directory's review_log.jsonl, and what
        the propagator would add to it.

        Every judgment this tool writes is a human one, and a propagated label
        is a human verdict about the same physical spot carried across frames
        by geometry. So the split that matters is BY CONVENTION, not by author
        -- which tie-break rule produced a label, and how many labels each
        rule WOULD produce if applied. That last number is given for all three
        rules rather than only the active one, for the same reason the metric
        shows all three POST rows: the choice is worth ~15 precision points
        and a single count would hide that it was a choice at all. Rows with
        source=classifier (reference_op set) are counted separately and
        flagged.
        """
        text, err, head = COLOR["text"], COLOR["error"], COLOR["selected"]
        dr = self.dr
        if dr is None:
            return []
        out = [("ADJUDICATION (this directory's review_log.jsonl):", head)]
        by: Dict[str, Dict[str, int]] = {}
        total = 0
        for rec in dr.store.judgments.values():
            verdict = rec.get("judgment")
            if verdict not in EXPORTABLE:
                continue
            total += 1
            prop = (rec.get("propagated") or {}).get("rule")
            key = f"propagated:{prop}" if prop else (
                "legacy classifier" if rec.get("reference_op") else "judged directly")
            by.setdefault(verdict, {}).setdefault(key, 0)
            by[verdict][key] += 1
        if total == 0:
            out += [("  no adjudications yet", text), ("", text)]
        else:
            out.append((f"  total {total}, by verdict and by how it was arrived at:", text))
            for verdict in sorted(by):
                for key in sorted(by[verdict]):
                    out.append((f"    {verdict:<18} {key:<26} {by[verdict][key]}", text))
                    if key == "legacy classifier":
                        out.append(("      ! rows with source=classifier were written by an automated "
                                    "classifier, not a human; precision over them is not an "
                                    "independent check.", err))
            out.append(("", text))

        # What the propagator can see, and what each convention would do.
        g = self._ensure_geo_verdicts()
        out.append(("  PROPAGATION COVERAGE (geo clusters carrying a human verdict):", head))
        if g is None:
            out += [("    computing... (clustering this directory's detections in UTM)", text),
                    ("", text)]
            return out
        if not g.get("available"):
            return out + [(f"    unavailable: {g.get('reason','')}", err), ("", text)]
        v = g["verdicts"]
        out.append((f"    {len(v.votes)} of {g['n_clusters']} geo clusters judged   "
                    f"{v.n_conflicted} conflicted (both verdicts)   "
                    f"{v.n_unresolved} judged detection(s) unresolved", text))
        # THE REACH DISTRIBUTION. A geo cluster is meant to be one target seen
        # in many frames, and mostly it is -- but DBSCAN chaining puts a heavy
        # tail on that, and the tail decides how much of a flight one verdict
        # can label. Reported as the top few rather than a mean, because a mean
        # over this distribution says nothing.
        reach = g.get("reach") or {}
        if reach:
            top = sorted(reach.items(), key=lambda kv: -kv[1][0])[:5]
            n_gui = sum(r[0] for r in reach.values())
            out.append((f"    reach -- how many clusters one verdict would label:", head))
            for lab, (n_cl, n_fr) in top:
                judged = "judged" if lab in v.votes else "unjudged"
                out.append((f"      geo cluster {lab:<6} {n_cl:>6} cluster(s) in {n_fr:>4} "
                            f"frame(s)   {n_cl / max(1, n_gui) * 100:>5.1f}% of the backlog   "
                            f"[{judged}]", err if n_cl / max(1, n_gui) > 0.05 else text))
            out.append((f"      ! DBSCAN chains at eps={g['params']['eps']:.2f}m. A verdict on a "
                        f"chained cluster propagates to ALL of it -- check the reach on the "
                        f"selected point before pressing [a].", err))
        frame_idx = self.current_frame_idx()
        if frame_idx is not None:
            saved = self.propagation_rule
            try:
                for rule in _GEO_VERDICT_RULES:
                    self.propagation_rule = rule
                    acc, already, none = self.propagate_frame_plan(frame_idx)
                    n_real = sum(1 for _, j, _ in acc if j == "confirmed_real")
                    mark = "*" if rule == saved else " "
                    out.append((f"   {mark}{rule:<20} would propagate {len(acc):>4} on this frame "
                                f"({n_real} real, {len(acc) - n_real} hard-neg); "
                                f"{already} already judged, {none} no label",
                                COLOR["post"] if rule == saved else text))
            finally:
                self.propagation_rule = saved
            out.append(("    the spread between the rules IS the uncertainty -- quote it as a range",
                        text))
        out.append(("", text))
        return out

    def _postproc_overlay_lines(self) -> List[Tuple[str, Tuple[int, int, int]]]:
        """POST-PROCESSING CONFIG: exactly how the detector outputs on screen
        were turned into the points being reviewed.

        Three layers of it, and the block exists because none of them was
        visible anywhere in this GUI before:

          1. the DETECTOR's own heatmap post-processing, fixed in
             run_detector.py and identical for every checkpoint;
          2. each CHECKPOINT's own parameters, read from the run's
             PREREGISTRATION.json into LayerSpec.config -- which was already
             being parsed and then used only for `arms`;
          3. this SESSION's live matching and clustering parameters, which the
             reviewer can move with the sliders.

        A PDF that reports precision without saying which of these produced
        the detections is not reproducible, which is the whole point of [x].
        """
        text, head = COLOR["text"], COLOR["selected"]
        dr = self.dr
        out = [("POST-PROCESSING CONFIG -- how detector output became these points", head)]
        out.append(("  detector (run_detector.py, same for every checkpoint):", text))
        out.append(("    GaussianBlur 5x5 sigma=1 -> binary threshold -> findContours", text))
        out.append((f"    -> contourArea > MIN_AREA={_DETECTOR_MIN_AREA} -> moment centroid, "
                    f"peak confidence", text))
        out.append((f"    default threshold {_DETECTOR_DEFAULT_THRESH} (each layer's own "
                    f"threshold is below)", text))
        out.append(("", text))
        out.append(("  this session, live (sliders / flags):", text))
        out.append((f"    spatial_threshold {self.args.spatial_threshold:g}px   "
                    f"cluster_radius {self.args.cluster_radius:g}px   many-to-one", text))
        out.append((f"    geo: DBSCAN eps {self.args.geo_eps:g}m  min_samples "
                    f"{self.args.geo_min_samples}  match {self.args.geo_match_radius:g}m  "
                    f"dedup {self.args.geo_dedup_cm:g}cm", text))
        out.append(("", text))
        if dr is None or not dr.layers:
            return out + [("  no model layers loaded", text), ("", text)]
        out.append(("  per checkpoint (from the run's PREREGISTRATION.json):", text))
        out.append((f"    {'layer':<22} {'thresh':>8} {'eps':>7} {'min_s':>7} {'eps_pred':>9}", text))
        for spec in self.layers.layers:
            cfg = spec.config or {}
            if not cfg:
                out.append((f"    {spec.label:<22} {'(run records no config for this checkpoint)'}",
                            text))
                continue
            out.append((f"    {spec.label:<22} {fmt_num(cfg.get('thresh')):>8} "
                        f"{fmt_num(cfg.get('eps')):>7} {str(cfg.get('min_samples','?')):>7} "
                        f"{fmt_num(cfg.get('eps_predict')):>9}", text))
        out.append(("", text))
        return out

    def _geo_overlay_lines(self) -> List[Tuple[str, Tuple[int, int, int]]]:
        """GEOSPATIAL: the shared definitions and provenance behind every
        per-model geo number above.

        The per-model VALUES live in _per_model_overlay_lines; what is here is
        what they all share -- the geospatial precision/recall definitions,
        the recall-bound derivation, the boresight and detection-source
        provenance. Said once, because it is identical for every layer and
        repeating it per model would bury the numbers.
        """
        text, head, err = COLOR["text"], COLOR["selected"], COLOR["error"]
        out: List[Tuple[str, Tuple[int, int, int]]] = [
            ("GEOSPATIAL -- the paper's geospatial precision/recall definitions (geo_metrics.py)", head)]
        g = self._ensure_geo_verdicts()
        if g is None:
            pub = self._geo_published()
            why = pub.get("error") or "clustering this directory's detections in UTM"
            return out + [(f"  computing... ({why})", text), ("", text)]
        if not g.get("available"):
            return out + [(f"  unavailable: {g.get('reason','')}", err), ("", text)]
        try:
            from . import geo_metrics
        except Exception as e:  # noqa: BLE001
            return out + [(f"  unavailable: {e}", err), ("", text)]
        p = g["params"]
        out.append((f"  DBSCAN eps<={p['eps']:.2f}m  min_samples={p['min_samples']}  "
                    f"match<={p['eps_predict']:.2f}m  dedup={p['dedup_cm']:g}cm", COLOR["geo"]))
        out.append((f"  {self.geo_source_note()}", text))
        # DEFINITIONS is nested one level (pre/post each hold a dict), so it
        # is walked rather than printed -- flattening it in a comprehension
        # produced "{'tp': ...}" as a single unreadable cell.
        for k, v in geo_metrics.DEFINITIONS.items():
            if isinstance(v, dict):
                out.append((f"    {k}:", text))
                for k2, v2 in v.items():
                    out.append((f"      {k2:<10} {v2}", text))
            else:
                out.append((f"    {k:<14} {v}", text))
        out.append(("  R<= is an UPPER BOUND, not an estimate: an unlabelled positive is only", err))
        out.append(("    discoverable where the detector fired, so c can never be a false", err))
        out.append(("    negative. Added markers (a) are the only term pulling it back down.", err))
        out.append(("", text))
        return out

    def _stats_overlay_lines(self) -> List[Tuple[str, Tuple[int, int, int]]]:
        """Pure line-builder for the [p] overlay (text + colour per line) --
        separated from _render_stats_overlay()'s cv2 drawing loop specifically
        so tests can assert on real string content instead of scraping pixels.

        SECTION ORDER IS A DESIGN DECISION, not an accident of which append
        runs first, which is why it is one declarative expression. Results
        lead; provenance and configuration follow.

        NOTHING HERE READS A FILE. Every number below is this session's own,
        computed live, at per-model granularity.
        """
        return (self._summary_overlay_lines()
                + self._per_model_overlay_lines()
                + self._adjudication_provenance_lines()
                + self._postproc_overlay_lines()
                + self._geo_overlay_lines())

    def _render_stats_overlay(self, w: int, h: int) -> np.ndarray:
        canvas = np.zeros((h, w, 3), np.uint8)
        self.stats_scroll = self._draw_paginated(canvas, self._stats_overlay_lines(), self.stats_scroll,
                                                 extra_hint="[x] export PDF",
                                                  top=30, line_h=22, font_scale=0.48)
        return canvas

    def _help_lines(self) -> List[str]:
        """Pure line-builder for the [?] help overlay -- separated from
        _render_help()'s cv2 drawing loop for the same reason
        _stats_overlay_lines() was: so tests can assert on real string
        content instead of scraping pixels."""
        return [
            "DETECTION REVIEW -- KEYBINDINGS", "",
            # QUICK KEYS lives at the very top on purpose. This overlay is
            # ~160 lines, i.e. five pages at a 1000px window, and these are
            # the keys used hourly, so they must be readable WITHOUT scrolling
            # -- smoke_test_gui.check_keybinding_docs asserts they fall inside
            # the first PANEL_HELP_QUICK_LINES.
            "QUICK KEYS",
            "  [c] confirmed   [h] hard-neg   [u] uncertain   [r] reset",
            "  [n]/[b] next/prev unreviewed    [1][2] frame   [3][4] dir   [5][6] rewind/play",
            "  [p] stats  [x] PDF  [m] propagation  [o] rule  [v] 3D  [d] cloud",
            "  [a] accept  [z] undo  [s] save  [q] save + quit",
            "  [l] layers [e] models [i] accum [;] toggle ['] solo [[]/[]] focus [0] all",
            "  [,]/[.] scroll the HUD's per-model PRE/POST table",
            "  [e] opens the model window: left-click a row toggles that model,",
            "  right-click solos it, [ALL]/[NONE]/[ACC]/[SORT] along the bottom",
            "",
            "Nearby unmatched detections are pre-clustered (--cluster-radius);",
            "you judge each cluster's CENTROID once and it applies to every raw",
            "point in that cluster on export. Raw points show as a translucent",
            "layer under the solid centroid marker; a 'xN' label marks clusters",
            "with more than one member.", "",
            "left-click     select nearest point (cluster centroid or added target)",
            "right-click    near a detection: fast-confirm its cluster real",
            "               near an added target: just select it",
            "               on empty ground: add a NEW target the model missed entirely",
            "space          select next unreviewed point in this frame",
            "c              commit: confirmed real",
            "h              commit: hard negative (tricky-texture false positive)",
            "u              commit: uncertain (logged, not exported)",
            "r              reset selected point to unreviewed / remove an added point",
            "ESC            deselect / back to overview",
            "1 / 2          previous / next frame",
            "3 / 4          previous / next directory",
            "5 / 6          toggle rewind / play (continuous, stops at the directory's first/last frame)",
            "f              cycle play/rewind speed",
            "n / b          jump forward/back to next frame with an unreviewed point",
            "s              save (re-export sidecar files now)",
            "q              save and quit",
            "",
            "MODEL LAYERS -- reviewing every checkpoint at once",
            "  Given a results/<run> with per-checkpoint detection files, every",
            "  model is drawn at once as a colour-keyed layer, and you adjudicate",
            "  each physical spot ONCE: the review target is the CONSENSUS cluster",
            "  over all layers, and every model's precision/recall is derived from",
            "  that single pass. Adjudicate once, score eleven.",
            "",
            "  Two tiers are drawn per layer:",
            "    accumulation  every detection of that layer, back-projected through",
            "                  its own frame's pose and reprojected into this frame.",
            "                  Faint, 1px. It shows every cluster found so far in",
            "                  every later frame. It SMEARS: a target seen in 200",
            "                  frames contributes 200 points, each on its own frame's",
            "                  ground plane. Needs flight_data.db.",
            "    this frame    that layer's own detections here. Larger, opaque, shaped.",
            "",
            f"  Identity is HUE x SHAPE, in a pale band no judgment colour uses. "
            f"{layer_cvd_safe_count()} of",
            f"  {len(LAYER_HUES_CVD)} Okabe-Ito colourblind-safe anchors survive the collision",
            "  filter against COLOR, so beyond ~12 layers hue GROUPS rather than",
            "  identifies and THE LEGEND IS THE AUTHORITY. A layer marker must",
            "  never read as green/red/orange. Health is never a hue:",
            "    filled glyph   healthy",
            "    hollow glyph   SPARSE     -- too few points to cluster",
            "    struck glyph   COLLAPSED  -- a near-constant output pixel",
            "  19 of the 30 screened checkpoints are degenerate. They are always shown",
            "  and always labelled; a collapsed model held the HIGHEST precision-like",
            "  score in the whole screen, which is exactly why it must not be hidden.",
            "",
            "  Above 6 visible layers the accumulation tier turns OFF -- more than a",
            "  handful of translucent hues over grass is not a readable picture. Said so",
            "  in the status bar; solo a layer, or hide some, to get it back. There is",
            "  no combined agreement-density view: that would be a union reprojection.",
            "",
            "  l              model layers on/off  (also: fold in newly-finished layers)",
            "  e              open / close the model window",
            "  i              accumulation tier: off -> per-layer",
            "  ;              toggle the focused layer          '   solo the focused layer",
            "  [ / ]          move focus up / down              0   all layers on",
            "  left-click a legend row toggles it; right-click solos it; the header",
            "  carries [ALL] [NONE] [ACC].",
            "",
            "  Judgments go to <run>/review/<flight>/review_log.jsonl, plus a per-layer",
            "  breakdown under by_layer/. The flight's own review_log.jsonl is never",
            "  read and never written.",
            "?              toggle this help",
            "p              toggle the stats overlay. SUMMARY, then PER MODEL (pixel + geo",
            "                 PRE/POST and the geotag counts behind them), the adjudication",
            "                 breakdown, the post-processing config, and the shared geospatial",
            "                 definitions. Needs no files. j/k scroll.",
            "x              export the whole [p] panel as a PDF beside review_log.jsonl",
            "                 (review_report.pdf): title/provenance with input digests, data &",
            "                 methods incl. the R<= derivation, every run parameter, per-model",
            "                 results, figures incl. the PR-plane PRE/POST plot, and the",
            "                 caveats. Honours --dry-run. Same principle as geo_metrics.json --",
            "                 anything the GUI shows must also land on disk, or it cannot be",
            "                 cited later.",
            "",
            "o              cycle the LABEL PROPAGATION RULE:",
            "                 off -> positive_wins -> majority -> hard_negative_wins -> off",
            "                 A geo cluster (DBSCAN eps 0.20m in UTM) spans frames, so a verdict",
            "                 you give in one frame says something about the same spot in every",
            "                 other frame that saw it. When a cluster holds BOTH verdicts, which",
            "                 one wins is a convention worth ~15 precision points -- so it starts",
            "                 OFF and nothing is proposed until you choose one. The reported",
            "                 metric is unaffected: all three POST rows are always computed.",
            "m              toggle the propagation overlay: a ring on each cluster showing the",
            "                 candidate label under the current [o] rule, in that judgment's own",
            "                 colour. NO RING means nothing propagates -- no rule chosen, no geo",
            "                 cluster, no verdict there yet, or the rule abstained. Select a",
            "                 point to see all three rules' calls and the vote tally.",
            "a              APPLY THE PROPAGATED LABELS for this frame as adjudications:",
            "                 every still-unreviewed cluster with a propagated label is judged",
            "                 to match the [m] ring. Press [a] once to preview, again to commit.",
            "                 - a judgment you already made is NEVER overwritten",
            "                 - a cluster with no propagated label is SKIPPED, never guessed",
            "                 - cycling [o] cancels a pending confirmation",
            "                 Rows are stamped source=\"human\" -- every input is a human verdict",
            "                 and no model is involved -- with a \"propagated\" block recording",
            "                 the rule, the geo cluster and the votes, so which convention",
            "                 produced a label is always recoverable.",
            "z              undo the last [a] -- resets exactly that batch to unreviewed",
            "v              cycle the 3D flight-pose view: off -> inline -> own window -> off",
            "                 (needs flight_data.db in the flight directory). Shows the camera",
            "                 track, the current frame's pose and ground footprint, this frame's",
            "                 detections back-projected to the ground plane, and the geotag",
            "                 survey. Drag to orbit, right-drag to pan, ctrl-drag (or the",
            "                 middle button, or +/-) to zoom. NOT the mouse wheel -- see below.",
            "                 INLINE fills the dead band under the frame image -- the image is",
            "                 not resized, only space that was already black. If the window's",
            "                 aspect leaves no usable band, [v] goes straight to its own window.",
            "                 During [5]/[6] playback the 3D camera follows the aircraft, then",
            "                 restores your framing when playback stops.",
            "GEO rows       the geospatial metrics block (world-frame, as opposed to the",
            "                 pixel-space PRE/POST above it). Always on: clusters this",
            "                 directory's detections in UTM (DBSCAN eps=0.20m) and scores them",
            "                 against the RTK geotag survey, i.e. the paper's geospatial",
            "                 precision/recall rather than this tool's pixel-space ones.",
            "                 Shows per-model rows in the HUD; the full block, every tie-break",
            "                 rule and the provenance caveat are in [p]. Built by a worker",
            "                 when a directory opens. Needs a flight_data.db; says so and",
            "                 changes nothing when there is none.",
            "                 GEO PRE is geometry only; GEO POST",
            "                 applies your adjudication -- to BOTH precision and recall, since",
            "                 it enlarges the positive class. POST recall shows as R<= because",
            "                 it is an upper bound: unmarked targets are only discoverable where",
            "                 the detector fired. Right-click 'added' markers are the one term",
            "                 that pulls it back down. Report the PRE/POST pair, not one number.",
            "g              3D view: switch between the whole-survey view and following the",
            "                 aircraft (only while the 3D view is open; also cancels playback's",
            "                 temporary auto-follow)",
            "+ / -          3D view: zoom in / out. So does ctrl+left-drag or middle-drag,",
            "                 dragging down to pull back. The mouse WHEEL is deliberately not",
            "                 used: this OpenCV is built against Qt, whose window zooms the",
            "                 whole displayed image on a wheel event and does not let a mouse",
            "                 callback consume it, so a wheel over the 3D view would zoom both",
            "                 at once -- and the window's zoom also throws off where clicks land.",
            "d              3D view: toggle the whole-directory detection cloud -- every",
            "                 frame's cluster centroids back-projected to the ground, in their",
            "                 2D judgment colours (green/red/purple/orange), plus label-matched",
            "                 detections in gray. Points landing within 0.35 m of a survey",
            "                 geotag are drawn slightly heavier in the same colour.",
            "", "colors:",
            "  pale dots     model-layer accumulation (one hue+shape per checkpoint)",
            "  hollow glyph  a SPARSE layer      struck glyph  a COLLAPSED layer",
            "  cyan ring     = label (context)",
            "  gray ring     = matched detection (already-confirmed real, context)",
            "  orange filled = unmatched cluster centroid, unreviewed (interactive)",
            "  green filled  = confirmed real", "  red filled    = hard negative",
            "  purple filled = uncertain", "  white star     = added (human-spawned) target",
            "  faint dots    = raw detections behind each cluster (translucent, tinted by its judgment)",
            "  azure ring    = [m] model overlay: majority of cluster members RECOVERED",
            "  brick ring    = [m] model overlay: majority of cluster members called background",
            "  yellow ring   = currently selected",
        ]

    def _draw_paginated(self, canvas: np.ndarray, lines: List[Tuple[str, Tuple[int, int, int]]],
                         scroll: int, top: int, line_h: int, font_scale: float,
                         extra_hint: str = "") -> int:
        """Shared scrolling engine for the full-window [?] help and [p]
        stats overlays. Without it, content past the bottom edge of a long
        overlay or a short window would be silently invisible. Draws whichever
        slice of `lines` actually fits, clamping `scroll` against the CURRENT canvas height
        and line count on every call (not once in on_key) so it self-heals
        after a window resize between key presses instead of ever indexing
        out of range. A pinned hint/position line is drawn at the very
        bottom OUTSIDE the scrollable slice -- never part of `lines` itself
        -- so the instructions for how to scroll or close can never be the
        exact content that got scrolled out of view. Returns the clamped
        scroll value; callers store it back on self (on_key's j/k handlers
        only ever nudge it un-clamped -- this is the one place that bounds
        it)."""
        h = canvas.shape[0]
        bottom_reserve = 26
        visible = max(1, (h - top - bottom_reserve) // line_h)
        max_scroll = max(0, len(lines) - visible)
        scroll = max(0, min(scroll, max_scroll))
        y = top
        for text, color in lines[scroll:scroll + visible]:
            cv2.putText(canvas, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            y += line_h
        if max_scroll == 0:
            hint = "press any key to close"
        else:
            shown_to = min(len(lines), scroll + visible)
            hint = f"[{scroll + 1}-{shown_to}/{len(lines)}]"
            if scroll > 0:
                hint = "^ " + hint
            if shown_to < len(lines):
                hint = hint + " v"
            hint += "   j/k scroll   any other key closes"
        if extra_hint:
            hint += "   " + extra_hint
        cv2.putText(canvas, hint, (30, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
        return scroll

    def _render_help(self, w: int, h: int) -> np.ndarray:
        canvas = np.zeros((h, w, 3), np.uint8)
        lines = [(line, COLOR["text"]) for line in self._help_lines()]
        self.help_scroll = self._draw_paginated(canvas, lines, self.help_scroll, top=40, line_h=26, font_scale=0.55)
        return canvas

    # -- input --------------------------------------------------------------

    def _canvas_point(self, sx: int, sy: int) -> Tuple[int, int]:
        """Window pixel -> canvas pixel.

        render() returns a canvas of `disp_w + panel_w` by
        `max(window_h, panel_w)`, while the window is `window_w + panel_w` by
        `window_h`; cv2 (WINDOW_NORMAL) stretches one onto the other and
        reports mouse positions in WINDOW space. In the usual width-limited
        fit those are the same size and this is the identity -- which is why
        the existing selection code got away without it. They diverge when the
        window is shorter than the panel, or when the fit is height-limited,
        and there every click currently lands off-target."""
        cw, ch = self.window_w + self.panel_width(), self.window_h
        img = self.image_cache_img
        native_h, native_w = img.shape[:2] if img is not None else (1200, 1920)
        geom = DisplayGeometry(native_w, native_h, self.window_w, self.window_h)
        canvas_w = geom.disp_w + self.panel_width()
        canvas_h = max(self.window_h, self.panel_width())
        if cw <= 0 or ch <= 0:
            return sx, sy
        return int(round(sx * canvas_w / cw)), int(round(sy * canvas_h / ch))

    def on_mouse(self, event: int, sx: int, sy: int, flags: int, userdata) -> None:
        if self._recompute_gen is not None:
            return  # a chunked recompute is in flight -- ignore input entirely until it finishes
        sx, sy = self._canvas_point(sx, sy)
        # Inline 3D routing comes FIRST, before the event-type filter below:
        # orbiting needs EVENT_MOUSEMOVE and the button-up events, which that
        # filter drops. Returning here also keeps the drag clear of the
        # `_play_direction = 0` line further down, so orbiting the 3D view
        # does not stop playback -- watching playback in 3D is a main reason
        # to open it.
        if self.pose_mode == "inline" and self.pose_view is not None:
            rect = self._inline_rect_now()
            inside = rect is not None and rect[0] <= sx < rect[2] and rect[1] <= sy < rect[3]
            if event == cv2.EVENT_LBUTTONDOWN or event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
                self._pose_drag_capture = inside
            if inside or self._pose_drag_capture:
                if rect is not None:
                    self.pose_view.on_mouse(event, sx - rect[0], sy - rect[1], flags, userdata)
                if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
                    self._pose_drag_capture = False
                return
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) or self.show_help or self.show_stats:
            return
        # Placement matters: this must come AFTER the event-type filter
        # above, not before it -- on_mouse fires on every mouse event,
        # including plain EVENT_MOUSEMOVE, so stopping play any earlier
        # would kill it the instant the mouse moves anywhere near the
        # window, making video play/rewind unusable. Only an actual click
        # (already filtered to LBUTTONDOWN/RBUTTONDOWN above) stops it.
        self._play_direction = 0
        dr, frame_idx = self.dr, self.current_frame_idx()
        if dr is None or frame_idx is None:
            return
        # THE SIDE PANEL IS INERT, deliberately. The layer legend lives in its
        # own window with its own mouse callback (on_model_mouse), so there is
        # no panel widget for a click to land in -- and no possibility of a
        # panel click being routed to a row the reviewer cannot see. The
        # `sx > geom.disp_w` early return below is the whole truth.
        img = self._load_frame_image(dr, frame_idx)
        if img is None:
            return
        h, w = img.shape[:2]
        geom = DisplayGeometry(w, h, self.window_w, self.window_h)
        if sx > geom.disp_w:  # elsewhere in the side panel: still inert
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.select_nearest(sx, sy, geom)
        else:
            self.right_click(sx, sy, geom)

    def on_key(self, key: int) -> bool:
        """Returns False to end the main loop."""
        if self._recompute_gen is not None:
            return True  # a chunked recompute is in flight -- no-op everything, including q, until it finishes
        ch = chr(key) if 0 <= key < 256 else ""
        if ch != "a":
            # Any other keystroke cancels a pending [a] confirmation. The
            # (frame, op, deadline) tuple already invalidates itself on a
            # frame change or an [o] cycle; this covers the rest, so a
            # half-finished accept can never be completed by an [a] pressed
            # much later with different intent.
            self._accept_armed = None
        # ",", "." and "e" join the pass-through list: scrolling the metric
        # table or opening the model list while watching playback is a
        # legitimate gesture, and ending playback would punish it. Same
        # reasoning the 3D view's keys already ride on.
        if self._play_direction != 0 and ch not in ("5", "6", "f", "g", "d", "+", "=", "-",
                                                    ",", ".", "e"):
            # Any key other than play's own controls stops play first, then
            # still falls through to its own normal action below (e.g. `c`
            # mid-play stops playback AND commits, rather than swallowing
            # the keypress or leaving playback running underneath). This
            # also covers ESC (already deselects; no need to special-case
            # it again) and, as an accepted side effect, `?`/`p`/`j`/`k`
            # against an overlay that was already open before play started.
            self._play_direction = 0
        if self.show_help or self.show_stats:
            # j/k scroll a long overlay instead of closing it (see
            # _draw_paginated); any OTHER key still closes, and resets scroll
            # so reopening always starts at the top.
            if ch in ("l", "i", ";", "'"):
                # DISPLAY TOGGLES STAY LIVE while an overlay is open. Any other
                # key merely closes it, so without this the natural gesture --
                # open [p] to read the per-model table, then flip a layer --
                # silently ate the first press and looked broken.
                # reading the per-model table in [p] and then flipping a layer
                # is the natural gesture. [ and ] are deliberately NOT here --
                # they move a focus that is invisible under the overlay.
                self._layer_key(ch)
                return True
            if ch == "j":
                if self.show_help:
                    self.help_scroll += SCROLL_STEP_LINES
                else:
                    self.stats_scroll += SCROLL_STEP_LINES
            elif ch == "k":
                if self.show_help:
                    self.help_scroll = max(0, self.help_scroll - SCROLL_STEP_LINES)
                else:
                    self.stats_scroll = max(0, self.stats_scroll - SCROLL_STEP_LINES)
            else:
                self.show_help = False
                self.show_stats = False
                self.help_scroll = 0
                self.stats_scroll = 0
            return True
        if ch == "?":
            self.show_help = True
        elif ch == "p":
            self.show_stats = True
        elif ch in (",", "."):
            # Unshifted and untaken -- verified against every branch of this
            # handler and pose3d.on_key. j/k stay the [p]/[?] overlay scroll.
            self.hud_scroll_by(-1 if ch == "," else 1)
        elif ch == "m":
            self.model_overlay_mode = not self.model_overlay_mode
            if not self.model_overlay_mode:
                self.set_status("propagation overlay off")
            elif self.propagation_rule is None:
                self.set_status("propagation overlay on -- no rule chosen, press [o]")
            else:
                self.set_status(f"propagation overlay on (rule: {self.propagation_rule})")
        elif ch == "o":
            # UNSHIFTED, and there is no shifted alias. A shifted letter does
            # not survive some highgui backends -- cv2.waitKey() delivers the
            # plain lowercase char, so Shift+M would arrive indistinguishable
            # from 'm' and silently re-trigger the overlay toggle.
            #
            # CYCLES THROUGH None FIRST, and that is the argument of
            # GEO_METRICS.md's "there is no default rule" section, preserved
            # rather than contradicted. Which verdict wins in a conflicted
            # cluster is a CONVENTION worth ~15 precision points, so a
            # reviewer who never presses [o] must never be shown a convention
            # somebody else picked for them. The reported METRIC is unaffected
            # either way -- geo_prf() still computes all three rules and the
            # HUD still draws all three POST rows. This selects a rule for a
            # different consumer: what to propagate.
            rules = (None,) + _GEO_VERDICT_RULES
            idx = (rules.index(self.propagation_rule) + 1) % len(rules) \
                if self.propagation_rule in rules else 0
            self.propagation_rule = rules[idx]
            # An armed [a] was armed against the OLD rule, so a cycle cancels
            # it -- committing a batch under a convention the reviewer just
            # switched away from is exactly the surprise the two-press arm
            # exists to prevent.
            self._accept_armed = None
            if self.propagation_rule is None:
                self.set_status("propagation rule: off -- nothing will be proposed")
            else:
                self.set_status(f"propagation rule: {self.propagation_rule}")
        elif ch == "1":
            self.goto_frame(-1)
        elif ch == "2":
            self.goto_frame(1)
        elif ch == "3":
            self.goto_dir(-1)
        elif ch == "4":
            self.goto_dir(1)
        elif ch == "5":
            # Toggle rewind (mirrors 1=prev-frame, but continuous). Pressing
            # while already rewinding stops it; pressing while playing
            # forward switches direction immediately, no need to stop first.
            self._play_direction = 0 if self._play_direction == -1 else -1
            self._last_play_advance_time = 0.0  # advance on the very next tick, don't wait a full interval
            self.set_status("rewinding" if self._play_direction == -1 else "stopped")
        elif ch == "6":
            # Toggle play forward (mirrors 2=next-frame, but continuous).
            self._play_direction = 0 if self._play_direction == 1 else 1
            self._last_play_advance_time = 0.0
            self.set_status("playing" if self._play_direction == 1 else "stopped")
        elif ch == "f":
            self._play_speed_idx = (self._play_speed_idx + 1) % len(PLAY_INTERVALS_SECONDS)
            self.set_status(f"play speed: {1.0 / PLAY_INTERVALS_SECONDS[self._play_speed_idx]:.1f} frames/sec")
        elif ch == "n":
            self.jump_to_unreviewed(forward=True)
        elif ch == "b":
            self.jump_to_unreviewed(forward=False)
        elif key == 32:  # space
            self.advance_selection()
        elif ch in ("l", "e", "i", ";", "'", "[", "]", "0"):
            self._layer_key(ch)
        elif ch == "x":
            # Unshifted, like every binding added here: some OpenCV highgui
            # backends do not deliver Shift as a case change to cv2.waitKey().
            self.export_report()
        elif ch == "v":
            self.cycle_pose_view()
        elif self.pose_view is not None and self.pose_view.on_key(ch):
            # [g]/[+]/[-] belong to the 3D view while it is open. A manual [g]
            # also cancels playback's temporary auto-follow, so an explicit
            # choice outranks the automatic one for the rest of the run.
            if ch == "g":
                self._pose_follow_restore = None
        elif ch == "a":
            self.accept_propagated_frame()
        elif ch == "z":
            self.undo_bulk_accept()
        elif ch == "c":
            self.commit("confirmed_real")
        elif ch == "h":
            self.commit("hard_negative")
        elif ch == "u":
            self.commit("uncertain")
        elif ch == "r":
            if self.selected_key is not None and self.dr is not None:
                frame_idx, x, y = self.selected_key
                if self._is_spawned(frame_idx, x, y):
                    self.remove_spawned(frame_idx, x, y)  # a spawn has no "unreviewed" state -- undo means delete
                else:
                    det = next((c for c in self.dr.unmatched_clusters_by_frame.get(frame_idx, []) if c.x == x and c.y == y), None)
                    if det is not None:
                        self.dr.store.commit(frame_idx, x, y, det.confidence, "unreviewed",
                                              self.dr.image_path_by_frame[frame_idx],
                                              extra={"source": SOURCE_HUMAN})
                        self.static_cache_key = None
                        self.dr.store.export(self.dr.member_map())
                        self.set_status("reset to unreviewed")
        elif key == 27:  # ESC
            self.selected_key, self.mode = None, "overview"
        elif ch == "s":
            if self.dr is not None:
                counts = self.dr.store.export(self.dr.member_map())
                geo = self.write_geo_metrics()
                self.set_status(f"saved: {counts['hard_negative']} hard-neg, {counts['confirmed_real']} confirmed-real, "
                                 f"{counts['added']} added, {counts['orphaned']} orphaned"
                                 + (f"  + {geo.name}" if geo else ""))
        elif ch == "q":
            if self.dr is not None:
                self.dr.store.export(self.dr.member_map())
                self.write_geo_metrics()
            return False
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive human review of ambiguous model detections.",
        epilog="Usage:\n"
               "  birdseye-review <flight-dir> <results/run>   review one flight against a run\n"
               "  birdseye-review <results/run>                review every flight the run covers\n"
               "  birdseye-review <flight-dir>                 review only; no artifacts, no derived params\n"
               "\nExample, from the root of the extracted sample:\n"
               "  birdseye-review 2025_04_16/pieranch_rect results/17_holdout_pieranch\n"
               "\nThe two positionals are identified by CONTENT (a run has run_manifest.json or\n"
               "gui/ or dets/; a flight has labels.txt), so order does not matter. A RUN IS\n"
               "REQUIRED -- detections come from its per-checkpoint dets/ files. The matching\n"
               "threshold is derived from the run too, so it cannot drift from the detections.\n"
               "The flags below still override it; overrides are recorded in review_log.jsonl\n"
               "and shown on every surface.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.set_defaults(dirs=[], run_context=None, param_overrides={}, auto_discovered=False)
    p.add_argument("paths", nargs="*", metavar="PATH",
                   help="A flight directory and/or a results/<run> directory, in either order.")
    p.add_argument("--base-dir", default=None,
                   help="Root for resolving relative paths and for auto-discovery (default: the current directory).")
    p.add_argument("--spatial-threshold", type=float, default=None,
                    help="Matching gate (px) for the live matched/unmatched split (match_spatial_many_to_one): "
                         "a detection counts as 'matched' (gray ring, context only) if ANY label lies within "
                         "this many pixels, and vice versa for a label -- it's a hard cutoff, not a soft "
                         "distance penalty, so a real detection sitting just outside this radius of its label "
                         "still lands in the orange unmatched/review pool, not a near-miss 'almost matched'. "
                         "LOOSENING this (raising it) can only ever move points unmatched->matched, shrinking "
                         "the review pool and raising this tool's naive precision/recall; it never creates new "
                         "unmatched points. Bigger is not simply 'better' though -- push it too high and "
                         "genuinely distinct nearby detections start being treated as already-handled context "
                         "instead of ambiguous cases needing your judgment. Empirically the single largest "
                         "lever on naive precision/recall/F1, but with diminishing returns, not an "
                         "ever-larger effect the more you loosen it: +57%% naive F1 from 20px->40px, then "
                         "only +5%% more all the way out to 60px. NOT the same algorithm as a one-to-one "
                         "(Hungarian) matcher's spatial threshold even when set to the same number -- see "
                         "REVIEW_GUI.md's 'Matching' section. "
                         "DEFAULT: taken from the run's run_manifest.json, so the review matches what the "
                         "detections were selected at. Setting it OVERRIDES the run and is recorded as such in "
                         "review_log.jsonl and shown on the HUD, [p] and the PDF report.")
    p.add_argument("--cluster-radius", type=float, default=30.0,
                    help="Complete-linkage clustering radius (px) for the unmatched-detection review pool -- "
                         "every pairwise distance within a cluster is guaranteed <= this. Judging a cluster's "
                         "centroid once propagates to every raw detection in it on export. <=0 disables "
                         "clustering (one point per review).")
    p.add_argument("--layer-accum-alpha", type=float, default=LAYER_ACCUM_ALPHA,
                    help="Opacity of the reprojected accumulation tier (every detection of a layer, "
                         "back-projected through its own frame's pose and forward-projected into this "
                         "frame).")
    p.add_argument("--layer-frame-alpha", type=float, default=LAYER_FRAME_ALPHA,
                    help="Opacity of the current frame's own per-layer detections, drawn over the "
                         "accumulation tier.")
    p.add_argument("--display-width", type=int, default=None,
                    help="Initial main-frame width in px (default: auto-fit to your monitor). "
                         "The window is resizable afterward regardless -- this only sets the starting size.")
    p.add_argument("--display-height", type=int, default=None,
                    help="Initial main-frame height in px (default: auto-fit to your monitor).")
    p.add_argument("--zoom-scale", type=int, default=DEFAULT_ZOOM_SCALE,
                    help="Nearest-neighbour magnification of the selected-point zoom inset. Also sets the "
                         "side-panel width (see compute_panel_width); auto-reduced by fit_zoom_scale() if "
                         "that would eat too much of the monitor.")
    p.add_argument("--zoom-crop-px", type=int, default=200,
                    help="Side length, in native frame pixels, of the crop shown in the zoom inset.")
    p.add_argument("--click-tolerance-px", type=int, default=15)
    # Geospatial metrics. Defaults are geo_eval.py's, which follow the
    # birdseye onboard pipeline (harelab-ucsc/birdseye: birdseye/birdseye.py)
    # and the values used in the paper -- except --geo-dedup-cm, see below.
    p.add_argument("--geo-eps", type=float, default=_GEO_EPS,
                   help="DBSCAN clustering distance for detections, metres (value used in the paper: 0.20).")
    p.add_argument("--geo-min-samples", type=int, default=_GEO_MIN_SAMPLES,
                   help="DBSCAN min_samples (value used in the paper: 10). Counts POINTS, not frames.")
    p.add_argument("--geo-match-radius", type=float, default=_GEO_EPS_PREDICT,
                   help="Geotag -> cluster core match radius, metres (value used in the paper: 0.35).")
    p.add_argument("--geo-dedup-cm", type=float, default=_GEO_GUI_DEDUP_CM,
                   help="Grid dedup before clustering, cm. 1.0 reproduces geo_eval.py "
                        "exactly but costs 21s/24GB; the 3.0 default costs 2.6s/1.2GB "
                        "for identical geotag recall and 1.2%% fewer clusters.")
    p.add_argument("--reviewer", default="unspecified",
                   help="Identifier stamped into every review_log.jsonl record and the PDF report; "
                        "set it to your name or email.")
    p.add_argument("--dry-run", action="store_true", help="Render/navigate normally but write nothing to disk.")
    return p


def detect_screen_size() -> Optional[Tuple[int, int]]:
    """Best-effort screen resolution via tkinter (stdlib). Returns None if
    tkinter or a display is unavailable -- caller falls
    back to a fixed default rather than crashing on a headless machine."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return None


def resolve_display_size(args: argparse.Namespace, panel_w: int,
                          screen: Optional[Tuple[int, int]] = None) -> None:
    """Fill in --display-width/--display-height when not explicitly given,
    fitting the *initial* window to the actual monitor (minus the side
    panel and some margin for window chrome/taskbars). The window stays
    resizable afterward either way -- this only picks a sane starting
    point instead of a one-size-fits-all guess that may not match this
    monitor's resolution or aspect ratio.

    `screen` is injectable so main() can query detect_screen_size() ONCE and
    share the answer with fit_zoom_scale() -- two tkinter round-trips for the
    same number, with the panel width decided between them, would be both
    wasteful and racy. Omitted (None) it queries for itself, preserving the
    original call signature."""
    if args.display_width is not None and args.display_height is not None:
        return
    if screen is None:
        screen = detect_screen_size()
    if screen is None:
        if args.display_width is None:
            args.display_width = 1600
        if args.display_height is None:
            args.display_height = 1000
        return
    screen_w, screen_h = screen
    # Cap at the dataset's native frame size (1920x1200): DisplayGeometry
    # never upscales past native resolution anyway (scale is capped at
    # 1.0), so requesting more than that just leaves blank window chrome --
    # and it guards against screen-size APIs that report a multi-monitor
    # virtual desktop's *combined* width rather than one physical monitor
    # (a known quirk on X11/Wayland; tkinter has no reliable per-monitor
    # query without extra dependencies this environment doesn't have).
    if args.display_width is None:
        args.display_width = min(max(400, int(screen_w * 0.9) - panel_w), 1920)
    if args.display_height is None:
        args.display_height = min(max(300, int(screen_h * 0.85)), 1200)


def pose_window_name() -> str:
    """pose3d.WINDOW, resolved lazily. The mainloop only ever asks once a
    PoseView exists, so the import has already succeeded by then; the fallback
    string just keeps this total rather than raising inside an except block."""
    try:
        from . import pose3d
        return pose3d.WINDOW
    except Exception:  # noqa: BLE001
        return "detection review -- 3D flight pose"


def _report_loop_error(exc: Exception, seen: set, where: str) -> None:
    """Print a full traceback the first time each distinct error is seen,
    then a one-liner on repeats -- render() runs ~50x/sec, so an unguarded
    print would bury the console in identical tracebacks within a second."""
    key = (where, type(exc).__name__, str(exc))
    if key in seen:
        return
    seen.add(key)
    print(f"\n*** {where}() raised {type(exc).__name__}: {exc}")
    traceback.print_exc()
    print(f"*** The GUI is still running. This is usually a schema mismatch in a file it "
          f"reads (run dets/, labels.txt, flight_data.db). Further identical errors "
          f"will be silenced. ***\n")


def _render_error_frame(app: "App", exc: Exception) -> np.ndarray:
    """Minimal, dependency-free error canvas -- deliberately does NOT call
    back into app.render(), since that is what just failed."""
    w = max(800, getattr(app, "window_w", 1200))
    h = max(400, getattr(app, "window_h", 800))
    canvas = np.zeros((h, w, 3), np.uint8)
    for i, line in enumerate([
            "RENDER ERROR -- the GUI is still running",
            f"{type(exc).__name__}: {exc}",
            "",
            "Full traceback printed to the console.",
            "This usually means a file the GUI reads has an unexpected shape",
            "(run dets/, labels.txt, flight_data.db).",
            "",
            "[q] quit   [?] help   any other key to continue",
    ]):
        cv2.putText(canvas, line, (30, 60 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 if i < 2 else 0.55, COLOR["error"] if i < 2 else COLOR["text"], 1, cv2.LINE_AA)
    return canvas


# Parameters whose value changes what the metrics MEAN, and which a run
# records. Overriding one of these is a deviation from the run and is stamped
# everywhere. Everything else (cluster_radius, zoom, click tolerance, reviewer,
# dry-run, display size) is a GUI-only preference: it changes what you see or
# how you work, not what a number means, so it is a plain flag.
RUN_DERIVED_PARAMS = ("spatial_threshold",)

# Used when neither the command line nor the run supplies a value. The parser
# default for these is None -- the sentinel meaning "not given" -- so that
# apply_run_context can tell an explicit --spatial-threshold 80 (an override to
# be recorded) from an absent flag (adopt the run's value silently).
PARAM_FALLBACKS = {"spatial_threshold": 30.0}


def classify_paths(paths, base) -> Tuple[List[str], Optional[Path]]:
    """Split the positionals into (flight dirs, run dir) BY CONTENT.

    Order-independent on purpose. The old CLI had the run's artifacts arriving
    through five separate flags, and the observed real-world failure was one of
    them pointing at the wrong kind of file; making position irrelevant and
    content decisive removes a whole class of that mistake. A path that is
    neither is left in `dirs` so the loader's own error names it.
    """
    dirs, run = [], None
    for p in paths:
        cand = Path(p)
        if not cand.is_absolute():
            for c in ((Path.cwd() / cand), (base / cand)):
                if c.is_dir():
                    cand = c
                    break
        if looks_like_run(cand) and not looks_like_dataset(cand):
            if run is not None:
                print(f"Note: more than one run directory given; using {run}, ignoring {p}")
                continue
            run = cand
        else:
            dirs.append(p)
    return dirs, run


def apply_run_context(args, run_arg, base) -> Optional[RunContext]:
    """Resolve the run, adopt its parameters where the user did not override,
    and run the three cross-pollination checks. Returns None when no run was
    given (review-only mode)."""
    explicit = {k: getattr(args, k) for k in RUN_DERIVED_PARAMS
                if getattr(args, k, None) is not None}
    if run_arg is None:
        for k in RUN_DERIVED_PARAMS:
            if getattr(args, k, None) is None:
                setattr(args, k, PARAM_FALLBACKS[k])
        args.param_overrides = {}
        if args.dirs:
            print("No run directory given: [p]/[m] will be empty and no parameters "
                  "were derived. Pass a results/<run> directory to populate them.")
        return None

    ctx = resolve_run_artifacts(run_arg)
    if not ctx.ok:
        return ctx
    print(f"Run: {ctx.run}")
    for n in ctx.notes:
        print(f"  note: {n}")

    # Only the run's PARAMETERS are derived.

    args.param_overrides = {}
    for k in RUN_DERIVED_PARAMS:
        run_value = getattr(ctx, k, None)
        if k in explicit:
            if run_value is not None and float(explicit[k]) != float(run_value):
                args.param_overrides[k] = {"run": run_value, "live": explicit[k]}
        elif run_value is not None:
            setattr(args, k, run_value)
        else:
            setattr(args, k, PARAM_FALLBACKS[k])
    if args.param_overrides:
        for k, v in args.param_overrides.items():
            print(f"  ! OVERRIDE {k}: run built at {v['run']}, reviewing at {v['live']}")
        print("    Recorded in review_log.jsonl and shown on the HUD, [p] and the report.")

    # Coverage: refuse a flight this run cannot populate, rather than rendering
    # empty overlays that look like a result.
    if ctx.coverage:
        for d in args.dirs:
            rel = _rel_for_coverage(d, base)
            if rel is not None and rel not in ctx.coverage:
                ctx.problems.append(
                    f"run {ctx.run.name} does not cover {rel}.\n"
                    f"          It covers {len(ctx.coverage)}: "
                    + ", ".join(ctx.coverage[:6])
                    + (" ..." if len(ctx.coverage) > 6 else ""))

    # Held-out contamination: a run that does not record honest scores for
    # its test directory must not be the basis for reviewing that directory --
    # exactly the mistake the held-out set exists to prevent.
    if ctx.test_dir and not ctx.scores_honest_for_test_dir:
        for d in args.dirs:
            if _rel_for_coverage(d, base) == ctx.test_dir:
                print(f"\n  *** {ctx.test_dir} is this run's HELD-OUT TEST directory, and "
                      f"{ctx.run.name} does not\n"
                      f"      record scores_are_honest_for_test_dir. Its models were fitted with\n"
                      f"      this flight included -- results here are CONTAMINATED. ***\n")
    return ctx


def _rel_for_coverage(d, base) -> Optional[str]:
    """The `dir` key a run's coverage list uses for this flight path."""
    p = Path(d)
    if not p.is_absolute():
        for c in ((Path.cwd() / p), (base / p)):
            if c.is_dir():
                p = c
                break
    try:
        return str(Path(p).resolve().relative_to(base))
    except ValueError:
        return None


def main() -> None:
    args = build_arg_parser().parse_args()
    # Screen queried once and threaded through both calls: the panel width
    # depends on the (possibly reduced) zoom scale, and the frame-view width
    # depends on the panel width, so the order here matters.
    screen = detect_screen_size()
    # Only auto-reduce the DEFAULT. An explicitly-passed --zoom-scale is the
    # user overriding this exact heuristic, so honour it even if the panel
    # ends up wide -- silently shrinking it would make the flag a no-op with
    # no way to get the behaviour back.
    if args.zoom_scale == DEFAULT_ZOOM_SCALE:
        fitted = fit_zoom_scale(args.zoom_crop_px, args.zoom_scale, screen[0] if screen else None)
        if fitted != args.zoom_scale:
            print(f"default --zoom-scale {args.zoom_scale} would give a "
                  f"{compute_panel_width(args.zoom_crop_px, args.zoom_scale)}px side panel on a "
                  f"{screen[0]}px-wide screen; reduced to {fitted} "
                  f"({compute_panel_width(args.zoom_crop_px, fitted)}px). Pass --zoom-scale to override.")
            args.zoom_scale = fitted
    resolve_display_size(args, compute_panel_width(args.zoom_crop_px, args.zoom_scale), screen)
    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()

    args.dirs, run_arg = classify_paths(args.paths, base)
    args.run_context = ctx = apply_run_context(args, run_arg, base)
    if ctx is not None and not ctx.ok:
        for pr in ctx.problems:
            print(f"REFUSING: {pr}")
        return

    if args.dirs:
        args.auto_discovered = False
        dir_specs = []
        for d in args.dirs:
            candidate = Path(d)
            if not candidate.is_absolute():
                # A bare directory name is ambiguous between "relative to
                # the current directory" and "relative to --base-dir" --
                # prefer whichever actually exists, so a naive CWD-relative
                # resolve() does not silently miss a directory under base/.
                cwd_candidate = (Path.cwd() / candidate).resolve()
                base_candidate = (base / candidate).resolve()
                if cwd_candidate.is_dir():
                    candidate = cwd_candidate
                elif base_candidate.is_dir():
                    candidate = base_candidate
                else:
                    candidate = cwd_candidate  # neither exists; keep this so the loader's error names it
            path = Path(candidate).resolve()
            try:
                rel_name = str(path.relative_to(base))
            except ValueError:
                rel_name = str(path)
            dir_specs.append((path, rel_name))
    elif ctx is not None and ctx.coverage:
        # No flight given, but a run was: the run's own coverage defines the
        # scope. Better than rglob-ing the whole dataset root, because these are
        # exactly the flights whose artifacts this run can actually populate.
        args.auto_discovered = True
        dir_specs = []
        for rel in ctx.coverage:
            d = (base / rel)
            if d.is_dir():
                dir_specs.append((d.resolve(), rel))
        print(f"Reviewing the {len(dir_specs)} flight(s) this run covers "
              f"(of {len(ctx.coverage)} listed; [3]/[4] to move between them).")
    else:
        args.auto_discovered = True
        print(f"Discovering reviewable directories under {base} ...")
        run_for_discovery = ctx.run if (ctx is not None and ctx.run is not None) else None
        dir_specs = [(d, str(d.relative_to(base)))
                     for d in discover_directories(base, run_for_discovery)]
        if run_for_discovery is None:
            print(f"Found {len(dir_specs)} directory(ies) with labels.txt, but NO RUN was "
                  f"given -- detections come from a run's dets/, so there is nothing to "
                  f"review. Pass a results/<run> directory.")
        else:
            print(f"Found {len(dir_specs)} directory(ies) with labels.txt that "
                  f"{run_for_discovery.name} has detections for.")

    if not dir_specs:
        print("No directories to review.")
        return

    app = App(args, base, dir_specs)
    if app.dr is None:
        if args.auto_discovered:
            print("Nothing left to review (every discovered directory was skipped -- see above).")
        else:
            print("None of the given directories could be loaded (see the reason(s) printed above).")
        return

    # WINDOW_NORMAL (not AUTOSIZE): user-resizable, and render()/on_mouse
    # both live-sync to whatever size the window actually is via
    # _sync_window_size(), so dragging the window auto-fits its contents.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, app.window_w + app.panel_width(), app.window_h)
    # Live sliders. Sized dynamically so a launch value above the 150px
    # baseline (unlikely, but possible via --spatial-threshold/--cluster-radius)
    # doesn't get silently clamped by OpenCV -- that would desync
    # self.args from what the trackbar actually reports and fire a phantom
    # recompute back down to the clamp on the very first tick. The onChange
    # callback is required by the cv2 API but unused -- App._poll_sliders()
    # reads positions by polling once per render() tick instead, debounced
    # (see SLIDER_DEBOUNCE_SECONDS) so a drag doesn't queue a pileup of the
    # ~15s-worst-case blocking recompute.
    slider_max = max(150, int(round(args.spatial_threshold)) + 20, int(round(args.cluster_radius)) + 20)
    cv2.createTrackbar(TRACKBAR_SPATIAL, WINDOW, max(0, int(round(args.spatial_threshold))), slider_max,
                        lambda v: None)
    cv2.createTrackbar(TRACKBAR_CLUSTER, WINDOW, max(0, int(round(args.cluster_radius))), slider_max,
                        lambda v: None)
    cv2.setMouseCallback(WINDOW, app.on_mouse)
    # Safety net around render()/on_key(). Previously this loop had a
    # `finally` but no `except`, so ANY exception from either -- including
    # ones this tool cannot control, because it renders numbers parsed out
    # of files that a separate, independently-evolving pipeline writes --
    # terminated the process outright. A KeyError from a renamed JSON field
    # should degrade to a visible error, not kill an in-progress review
    # session. The traceback is printed once per distinct error so the
    # console still tells the whole story.
    seen_errors = set()
    try:
        while True:
            try:
                cv2.imshow(WINDOW, app.render())
            except Exception as e:
                _report_loop_error(e, seen_errors, "render")
                cv2.imshow(WINDOW, _render_error_frame(app, e))
            # The optional [v] companion window, rendered right after the
            # main one so a single waitKey() below services both. Every
            # failure path here ends in close_pose_view(), never in `break`:
            # the 3D view is a convenience, and losing it must never cost an
            # in-progress review session.
            # Only the SEPARATE-WINDOW mode needs anything here; the inline
            # strip is composed inside app.render() above, which is what makes
            # it structurally frame-accurate.
            if app.pose_mode == "window":
                try:
                    frame3d = app.render_pose_view()
                    if frame3d is not None:
                        cv2.imshow(pose_window_name(), frame3d)
                    if cv2.getWindowProperty(pose_window_name(), cv2.WND_PROP_VISIBLE) < 1:
                        app.close_pose_view()  # user closed IT -- not a reason to exit
                except Exception as e:
                    _report_loop_error(e, seen_errors, "pose3d")
                    app.close_pose_view()
            # The model window ([e]), beside the 3D one and on the same rule:
            # the user closing IT is not a reason to end the session. One
            # waitKey below services every window.
            if app.model_win:
                try:
                    frame_models = app.render_model_window()
                    if frame_models is not None:
                        cv2.imshow(MODEL_WINDOW, frame_models)
                    if cv2.getWindowProperty(MODEL_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        app.close_model_window()
                except Exception as e:
                    _report_loop_error(e, seen_errors, "model-window")
                    app.close_model_window()
            key = cv2.waitKey(20) & 0xFF
            if key != 255:
                try:
                    if not app.on_key(key):
                        break
                except Exception as e:
                    _report_loop_error(e, seen_errors, "on_key")
            try:
                # Deliberately keyed to the MAIN window only: the session ends
                # when the review window closes, not when the 3D one does.
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
