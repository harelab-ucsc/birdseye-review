#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 harelab-ucsc
"""
Headless smoke test for the birdseye_review GUI's read-only overlays.

WHY THIS EXISTS
---------------
The review GUI must not break. Import checks and `--help` are not enough:
overlay bugs that draw nothing pass both cleanly and still produce a GUI that
silently renders nothing. Two examples of the class:

  1. Float32 vs float64 point coordinates make an exact-float lookup match
     zero rows.
  2. An __init__ block that resets loaded state after it was loaded clobbers it.

Neither raises. Both just draw nothing. So this test exercises the real
render() path and asserts the pixels actually change.

App.__init__ creates no cv2 window (only main() does) and cv2 draws into numpy
arrays, so all of this runs without a display.

    python3 tests/smoke_test_gui.py --data-root /path/to/birdseye-review-sample
"""
import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from birdseye_review import review_gui as rg

# The main app's subject flight. It MUST be one the layer run has dets/ for:
# a flight with no model layers has nothing to review at all and App.dr is
# None. This is the published sample flight.
DEFAULT_DIR = "2025_04_16/pieranch_rect"
# The same flight. Kept as a separate name because the two roles are
# distinct -- POSE_DIR must have a flight_data.db -- and collapsing them would
# hide which property a future failure is about.
POSE_DIR = "2025_04_16/pieranch_rect"
# The run whose dets/ supply detections. Set by main(); the standalone checks
# below need it because a flight is always loaded through a run.
LAYER_RUN: str = "results/17_holdout_pieranch"


def single_dir_app(base: Path, flight: str, extra=()):
    """An App on one flight, against LAYER_RUN, with the run context resolved.

    One builder means one place to keep correct.
    """
    argv = [flight, str(LAYER_RUN),
            "--spatial-threshold", "80", "--dry-run", *extra]
    args = rg.build_arg_parser().parse_args(argv)
    args.auto_discovered = False
    args.dirs, run_arg = rg.classify_paths(args.paths, base)
    if run_arg is not None:
        args.run_context = rg.apply_run_context(args, run_arg, base)
    app = rg.App(args, base, [(base / flight, flight)])
    app.window_w, app.window_h = 1400, 900
    return app


# Fixed panel overhead of the compact layout: y=20 start + 3x22px of
# judgment-key hints inside the info block + a 20px gap + 270px HUD
# reservation + 90px keybind strip + 10px. The inset must stay larger than
# what that leaves -- a future line added to the HUD or the info block
# silently eats it back.
OLD_PANEL_OVERHEAD_PX = 608

# Mean |delta luma| below which a colour reads as "nothing is displaying".
# Shared by the [t] badge check and the pose3d palette check -- one bar, so
# the two cannot drift apart.
MIN_CONTRAST = 40.0

# The divider line sits just above the inline band, so the "did it draw above
# its rectangle" comparison starts a couple of rows higher up.
POSE_DIVIDER_SLACK = 3

# Pixels the detection cloud must actually paint in a 900x700 view before it
# counts as visible. Floor for the published sample (frames 600-641 of
# 2025_04_16/pieranch_rect), which paints ~1,600.
MIN_CLOUD_PIXELS = 1200


def check_inset(app, fails: list) -> None:
    """The zoom inset must actually be larger than the old layout allowed,
    and must still shrink rather than overflow on a short window."""
    dr = app.dr
    sel = None
    for f in dr.frames_all:
        cs = dr.unmatched_clusters_by_frame.get(f, [])
        if cs:
            app.frame_pos = dr.frames_all.index(f)
            sel = (f, cs[0].x, cs[0].y)
            break
    if sel is None:
        fails.append("inset: no cluster to select in this directory")
        return
    app.selected_key = sel
    panel_w = app.panel_width()
    if panel_w < 620:
        fails.append(f"inset: panel width regressed to {panel_w} (was 620 at the old default)")

    app.window_w, app.window_h = 1400, 1000
    canvas = app.render()
    # The inset is the only thing drawn at the very top of the panel, so its
    # height is the run of non-background rows starting at y=20.
    x0 = canvas.shape[1] - panel_w
    strip = canvas[:, x0 + 10:x0 + panel_w - 10]
    bg = np.array(rg.COLOR["panel_bg"])
    rows = np.nonzero((np.abs(strip.astype(int) - bg).sum(axis=2) > 30).any(axis=1))[0]
    run = 20
    while run + 1 in rows and run < canvas.shape[0] - 1:
        run += 1
    inset_h = run - 20
    old_budget = canvas.shape[0] - OLD_PANEL_OVERHEAD_PX
    if inset_h <= old_budget:
        fails.append(f"inset: {inset_h}px tall, no bigger than the old {old_budget}px budget")
    else:
        print(f"  inset OK  {inset_h}px tall in a {panel_w}px panel "
              f"(old layout allowed {old_budget}px)")

    # A window too short for the reserved blocks must still render, with the
    # inset shrinking first rather than anything overflowing the canvas.
    for h in (500, 320):
        app.window_h = h
        try:
            app.render()
        except Exception as e:  # noqa: BLE001
            fails.append(f"inset: render() raised at window_h={h}: {type(e).__name__}: {e}")
    app.window_h = 900


def check_propagated_accept(app, fails: list) -> None:
    """[a] must apply PROPAGATED labels, never clobber a judgment already made,
    stamp the convention that produced each one, and be fully undoable by [z].

    [a] propagates HUMAN verdicts through the geo clusters, so the provenance
    being asserted is source=human PLUS a `propagated` block naming the rule.

    It needs the geo worker, since propagation IS the geo clustering -- so it
    waits for the union scaffold rather than assuming a synthetic fixture.
    """
    dr = app.dr
    if not await_geo(app, union_only=True):
        fails.append("[a] the geo worker never published a union scaffold, so propagation "
                     "could not be exercised")
        return
    g = app._ensure_geo_verdicts()
    if not g or not g.get("available"):
        fails.append(f"[a] no geo clustering available: {(g or {}).get('reason')}")
        return
    ml = g["member_label"]

    # WITH NO RULE CHOSEN, [a] MUST REFUSE AND SAY WHY. This is the state a
    # session starts in, and an [a] that armed here would be offering to
    # commit under a convention nobody picked.
    app.propagation_rule = None
    app._accept_armed = None
    app.status_msg = ""
    app.on_key(ord("a"))
    if app._accept_armed is not None:
        fails.append("[a] armed with no propagation rule chosen -- it must refuse")
    elif "rule" not in (app.status_msg or ""):
        fails.append(f"[a] refused with no rule but did not say so: {app.status_msg!r}")

    # Find a frame holding >=2 unreviewed clusters that reach a geo cluster,
    # and seed a verdict on a DIFFERENT frame of the same geo cluster -- so
    # what [a] commits is genuinely propagated, not locally judged.
    target = seed = None
    for f in dr.frames_all:
        cs = [c for c in dr.unmatched_clusters_by_frame.get(f, [])
              if dr.store.get(c.frame_idx, c.x, c.y) == "unreviewed"
              and rg.cluster_geo_clusters(dr, (c.frame_idx, c.x, c.y), ml)]
        if len(cs) < 2:
            continue
        lab = rg.cluster_geo_clusters(dr, (cs[0].frame_idx, cs[0].x, cs[0].y), ml)[0]
        for f2 in dr.frames_all:
            if f2 == f:
                continue
            for c2 in dr.unmatched_clusters_by_frame.get(f2, []):
                k2 = (c2.frame_idx, c2.x, c2.y)
                if lab in rg.cluster_geo_clusters(dr, k2, ml):
                    target, seed = f, c2
                    break
            if seed is not None:
                break
        if seed is not None:
            break
    if target is None:
        fails.append("[a] found no frame whose clusters share a geo cluster with another "
                     "frame -- propagation cannot be exercised")
        return

    dr.store.commit(seed.frame_idx, seed.x, seed.y, seed.confidence, "confirmed_real",
                    dr.image_path_by_frame[seed.frame_idx],
                    extra={"source": rg.SOURCE_HUMAN})
    app._geo_summary_key = None
    app.frame_pos = dr.frames_all.index(target)
    app.propagation_rule = "positive_wins"

    # Pin a judgment on THIS frame; [a] must leave it alone.
    human = next(c for c in dr.unmatched_clusters_by_frame[target]
                 if dr.store.get(c.frame_idx, c.x, c.y) == "unreviewed")
    app.selected_key = (target, human.x, human.y)
    app.commit("uncertain")

    planned, _, _ = app.propagate_frame_plan(target)
    if not planned:
        fails.append(f"[a] nothing propagated to frame {target} despite a seeded verdict "
                     f"in frame {seed.frame_idx} of the same geo cluster")
        return
    app.on_key(ord("a"))                       # arms only
    if app._accept_armed is None:
        fails.append("[a] first press did not arm")
    if len(app.propagate_frame_plan(target)[0]) != len(planned):
        fails.append("[a] first press committed something -- it must only preview")

    # A RULE CHANGE MUST CANCEL THE ARM. Committing under a convention the
    # reviewer just moved away from is the surprise the two-press arm exists
    # to prevent.
    app.on_key(ord("o"))
    if app._accept_armed is not None:
        fails.append("[o] did not cancel an armed [a]")
    app.propagation_rule = "positive_wins"

    app.on_key(ord("a")); app.on_key(ord("a"))  # arm, then confirm
    keys = list(app._last_bulk_accept)
    if len(keys) != len(planned):
        fails.append(f"[a] committed {len(keys)} of {len(planned)} planned")
    bad = []
    for k in keys:
        rec = dr.store.judgments[k]
        prop = rec.get("propagated") or {}
        if rg.judgment_source(rec) != rg.SOURCE_HUMAN:
            bad.append((k, "source is not human"))
        elif prop.get("rule") != "positive_wins":
            bad.append((k, f"propagated.rule={prop.get('rule')!r}"))
        elif not prop.get("geo_clusters") or not prop.get("votes"):
            bad.append((k, "propagated block carries no geo_clusters/votes"))
    if bad:
        fails.append(f"[a] {len(bad)} committed row(s) lack propagation provenance: {bad[:3]}")
    if dr.store.get(target, human.x, human.y) != "uncertain":
        fails.append("[a] OVERWROTE a judgment already made")
    if not any("propagated" in t for t, _ in app._adjudication_provenance_lines()):
        fails.append("[p] no propagated/direct split shown after a bulk apply")

    app.on_key(ord("z"))
    still = [k for k in keys if dr.store.get(*k) != "unreviewed"]
    if still:
        fails.append(f"[z] left {len(still)} propagated judgment(s) in place")
    if dr.store.get(target, human.x, human.y) != "uncertain":
        fails.append("[z] disturbed a judgment made directly")
    print(f"  [a] OK  {len(keys)} propagated to frame {target} from frame "
          f"{seed.frame_idx}, provenance stamped, [o] cancels the arm, [z] reverted all")


def check_propagation_rules(app, fails: list) -> None:
    """The three tie-break rules must actually differ on a CONFLICTED cluster,
    in the documented direction, and `majority` must abstain on a tie.

    Built on a synthetic GeoVerdicts rather than on real review state: a
    conflicted cluster needs both verdicts in one geo cluster, which a fresh
    review log does not have, and waiting for one to occur naturally would
    make the test depend on the order somebody reviewed a flight in.
    """
    from birdseye_review import geo_metrics as gm

    # The mirrored constants must not drift from geo_metrics'.
    if tuple(rg._GEO_VERDICT_RULES) != tuple(gm.VERDICT_RULES):
        fails.append(f"review_gui._GEO_VERDICT_RULES {rg._GEO_VERDICT_RULES} != "
                     f"geo_metrics.VERDICT_RULES {gm.VERDICT_RULES}")
    if tuple(rg._GEO_RULE_ABBRS) != tuple(gm.VERDICT_RULE_ABBR[r] for r in gm.VERDICT_RULES):
        fails.append("review_gui._GEO_RULE_ABBRS drifted from geo_metrics.VERDICT_RULE_ABBR")

    # [o] selects a rule for the PROPAGATOR; the reported METRIC has no default
    # and no way to select one. If geo_prf grows a rule argument, the argument
    # in GEO_METRICS.md has quietly been lost.
    import inspect
    if "rule" in inspect.signature(gm.geo_prf).parameters:
        fails.append("geo_prf() grew a `rule` parameter -- it must always compute all three")
    if inspect.signature(gm.GeoVerdicts.verdict).parameters["rule"].default \
            is not inspect.Parameter.empty:
        fails.append("GeoVerdicts.verdict() grew a default rule -- a caller must name one")

    # A conflicted cluster: 3 positive, 1 negative.
    v = gm.GeoVerdicts()
    for _ in range(3):
        v.tally(7, "confirmed_real")
    v.tally(7, "hard_negative")
    want = {"positive_wins": "confirmed_real",
            "majority": "confirmed_real",
            "hard_negative_wins": "hard_negative"}
    for rule, expect in want.items():
        got = v.verdict(7, rule)
        if got != expect:
            fails.append(f"conflicted cluster under {rule}: got {got!r}, expected {expect!r}")
    # A TIE MUST ABSTAIN under majority -- "geometry stands", and for the
    # propagator that means nothing is proposed and the point stays orange.
    v2 = gm.GeoVerdicts()
    v2.tally(9, "confirmed_real"); v2.tally(9, "hard_negative")
    if v2.verdict(9, "majority") is not None:
        fails.append("majority did not abstain on a tie")
    if v2.verdict(9, "positive_wins") != "confirmed_real":
        fails.append("positive_wins did not resolve a tie to confirmed_real")
    if v2.verdict(9, "hard_negative_wins") != "hard_negative":
        fails.append("hard_negative_wins did not resolve a tie to hard_negative")
    # An abstention-only cluster propagates nothing at all.
    v3 = gm.GeoVerdicts()
    v3.tally(11, "uncertain")
    if any(v3.verdict(11, r) is not None for r in gm.VERDICT_RULES):
        fails.append("an abstention-only cluster produced a verdict under some rule")

    # And propagated_label() must return None for every one of its four
    # documented reasons -- the caller draws all four identically, so a bug in
    # any of them looks like a missing ring rather than an error.
    dr = app.dr
    ckey = next(iter(dr.cluster_members))
    if rg.propagated_label(dr, ckey, v, {}, "positive_wins") is not None:
        fails.append("propagated_label returned a label with an empty member_label join")
    if rg.propagated_label(dr, ckey, v, {(0, 0.0, 0.0): 7}, None) is not None:
        fails.append("propagated_label returned a label with rule=None")
    print("  [o] OK  three rules bracket a conflicted cluster, majority abstains on a tie, "
          "propagated_label refuses on all four no-label paths")


def check_degenerate_excluded_from_union(app, fails: list) -> None:
    """A degenerate checkpoint must contribute NO review targets, while keeping
    its own scorecard.

    Both halves matter and they pull in opposite directions: a collapsed model
    firing 34,817 times manufactures review burden out of a model already known
    to be broken, but "show all data across all models" is a standing
    requirement and a collapsed checkpoint held the HIGHEST precision-like
    score in the whole 30-checkpoint screen. So the union drops it and the
    per-model arm does not.

    Forced synthetically: the runs with dets/ on disk are the 12-layer holdout,
    where every layer is healthy, so waiting for real degenerate input would
    mean never running this.
    """
    import dataclasses
    dr = app.dr
    if not dr.layers or len(dr.layers) < 3:
        fails.append("[union] too few layers to exercise degenerate exclusion")
        return
    before_clusters = sum(len(v) for v in dr.unmatched_clusters_by_frame.values())
    before = {m.layer_id: (m.n_dets, m.pre) for m in rg.layer_metric_rows(dr)}
    saved_layers = list(dr.layers)
    try:
        big = sorted(range(len(dr.layers)),
                     key=lambda i: -before[dr.layers[i].layer_id][0])[:2]
        dr.layers = [dataclasses.replace(sp, verdict="collapsed") if i in big else sp
                     for i, sp in enumerate(dr.layers)]
        if not all(dr.layers[i].degenerate for i in big):
            fails.append("[union] LayerSpec.degenerate did not follow verdict='collapsed'")
            return
        full = sum(len(f.points) for f in dr.dets_fd.values())
        healthy = sum(len(f.points) for f in rg.healthy_union_frames(dr).values())
        if healthy >= full:
            fails.append(f"[union] healthy_union_frames kept {healthy} of {full} points -- "
                         f"the degenerate layers were not excluded")
        if set(rg.healthy_union_frames(dr)) != set(dr.dets_fd):
            fails.append("[union] healthy_union_frames dropped whole FRAMES -- it must keep "
                         "every frame, even an empty one, or the flight silently shortens")
        # THE APP'S OWN PARAMETERS, not literals. Recomputing at a different
        # spatial_threshold than the baseline was captured under moves every
        # per-layer counter, which made this test report the exact regression
        # it exists to catch -- from its own fixture rather than from the code.
        st, cr = app.args.spatial_threshold, app.args.cluster_radius
        rg.recompute_layered_matching(dr, dr.dets_fd, dr.labels_fd, st, cr)
        after_clusters = sum(len(v) for v in dr.unmatched_clusters_by_frame.values())
        if after_clusters >= before_clusters:
            fails.append(f"[union] review targets did not shrink: {before_clusters} -> "
                         f"{after_clusters}")
        # No cluster may contain a degenerate layer's detection.
        leaked = 0
        for group in dr.cluster_members.values():
            for p in group:
                li = getattr(p, "layer", -1)
                if 0 <= li < len(dr.layers) and dr.layers[li].degenerate:
                    leaked += 1
        if leaked:
            fails.append(f"[union] {leaked} degenerate detection(s) are still in review clusters")
        # THE OTHER HALF: every per-model scorecard must be untouched.
        after = {m.layer_id: (m.n_dets, m.pre) for m in rg.layer_metric_rows(dr)}
        changed = [lid for lid in before
                   if before[lid][0] != after[lid][0]
                   or max(abs(a - b) for a, b in zip(before[lid][1], after[lid][1])) > 1e-12]
        if changed:
            fails.append(f"[union] excluding degenerate layers CHANGED the per-model PRE "
                         f"scorecard of {changed} -- degenerate models must keep their own "
                         f"numbers, they are only removed from the union")
        print(f"  [union] OK  2 degenerate layer(s) removed {before_clusters - after_clusters} "
              f"of {before_clusters} review clusters; all {len(before)} per-model scorecards "
              f"unchanged")
    finally:
        dr.layers = saved_layers
        rg.recompute_layered_matching(dr, dr.dets_fd, dr.labels_fd,
                                      app.args.spatial_threshold, app.args.cluster_radius)



def layer_summary(app, lid: str = None) -> dict:
    """One layer's geo_prf dict.

    THE CHOICE OF LAYER IS PINNED on first use, and that is load-bearing rather
    than tidy. The invariants are all of the form "commit a judgment, re-read,
    assert the PRE arm did not move" -- and the geo worker may still be landing
    layers between those two reads. Picking "the first landed layer" fresh each
    call could return a DIFFERENT model's numbers before and after, which the
    tests would correctly report as the PRE arm moving.

    Returns {} when nothing has landed yet, which the callers already handle as
    "not computed yet".
    """
    all_ = app._layer_geo_summaries()
    if lid is not None:
        got = all_.get(lid) or {}
        return got if got.get("available") else {}
    pinned = getattr(app, "_smoke_geo_layer", None)
    if pinned is not None:
        got = all_.get(pinned) or {}
        return got if got.get("available") else {}
    for key in (app.layers.sequence() if getattr(app, "layers", None) else list(all_)):
        got = all_.get(key)
        if got and got.get("available"):
            app._smoke_geo_layer = key
            return got
    return {}


def await_geo(app, union_only: bool = False, timeout: float = 180.0,
              want_layers: bool = False) -> bool:
    """Block until the geo worker has published what we are about to assert on.

    Every geo assertion downstream would otherwise race a background thread.
    Deterministic by construction: it polls the SAME published dict the render
    loop reads, so a test that passes here is testing what the GUI shows.

    `union_only` returns as soon as the union scaffold lands (~1.8 s), which is
    all the PROPAGATOR needs -- it runs on that scaffold. `want_layers` also
    requires at least one per-layer summary, which is what the METRIC
    assertions need. Without it those tests race the worker and see {}.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        pub = app._geo_published()
        if pub.get("error"):
            return False
        if want_layers:
            if pub.get("layers"):
                app._ensure_geo_verdicts()
                if layer_summary(app):
                    return True
            time.sleep(0.2)
            continue
        if union_only and pub.get("union") is not None:
            # Pull it through the accessor so app._geo_scaffold is populated
            # exactly as a render tick would have done.
            app._ensure_geo_verdicts()
            return True
        if pub.get("done"):
            app._ensure_geo_verdicts()
            return True
        time.sleep(0.2)
    return False


def check_pose3d(base: Path, pose_dir: str, fails: list) -> None:
    """The 3D pose window must build a real scene where pose data exists, and
    fail cleanly (never raise past the caller) where it does not."""
    try:
        from birdseye_review import pose3d
    except Exception as e:  # noqa: BLE001
        fails.append(f"pose3d: import failed: {type(e).__name__}: {e}")
        return
    # No flight_data.db -> a typed, message-carrying refusal, not a crash.
    with tempfile.TemporaryDirectory() as tmp:
        try:
            pose3d.PoseScene(Path(tmp), "no_pose_flight", 10)
            fails.append("pose3d: a directory with no flight_data.db did not raise PoseUnavailable")
        except pose3d.PoseUnavailable:
            print("  [v] OK  no-pose-data directory refuses cleanly")
        except Exception as e:  # noqa: BLE001
            fails.append(f"pose3d: wrong exception for a missing db: {type(e).__name__}: {e}")

    d = base / pose_dir
    if not (d / "flight_data.db").exists():
        print(f"  [v] SKIP  {pose_dir} has no flight_data.db")
        return
    try:
        frames = scene_frames(d)
        scene = pose3d.PoseScene(d, pose_dir, max(frames) if frames else 0)
    except Exception as e:  # noqa: BLE001
        fails.append(f"pose3d: {pose_dir} failed to load: {type(e).__name__}: {e}")
        return
    scene.set_frame(len(scene.proj.poses) // 2, None,
                    detections=[(500.0, 600.0, (0, 165, 255))], selected=(500.0, 600.0))
    if scene.row_idx is None or not scene.pose_text:
        fails.append("pose3d: mid-flight frame resolved to no pose row")
    img = scene.render(900, 700)
    if img.shape != (700, 900, 3):
        fails.append(f"pose3d: render returned {img.shape}")
    if len(np.unique(img.reshape(-1, 3), axis=0)) < 20:
        fails.append("pose3d: rendered canvas is effectively blank")
    before = img.copy()
    scene.camera.orbit(60, 15)
    if np.array_equal(before, scene.render(900, 700)):
        fails.append("pose3d: orbiting changed nothing -- the view is not interactive")
    if not np.isfinite(scene.camera.project(scene.track, 900, 700)[1]).all():
        fails.append("pose3d: track depths are not finite")

    # Legibility: every scene colour must clear the MIN_CONTRAST
    # mean-|delta-luma| bar, or the far grid and the unflown track read as
    # empty black.
    bg_luma = sum(pose3d.COLOR["bg"]) / 3.0
    dim = [(k, round(abs(sum(v) / 3.0 - bg_luma), 1)) for k, v in pose3d.COLOR.items()
           if k != "bg" and abs(sum(v) / 3.0 - bg_luma) < MIN_CONTRAST]
    if dim:
        fails.append(f"pose3d: colours below the {MIN_CONTRAST} dluma bar vs the background: {dim}")
    else:
        print(f"  [v] OK  all {len(pose3d.COLOR) - 1} scene colours clear dluma {MIN_CONTRAST}")

    # Fit + HUD bounds across the shapes the view actually gets: its own
    # window, the wide-short inline strip, and a narrow column.
    for w, h in ((900, 700), (1100, 250), (400, 700)):
        scene._fitted = False
        scene.set_frame(len(scene.proj.poses) // 2, None,
                        detections=[(500.0, 600.0, (0, 165, 255))], selected=(500.0, 600.0))
        out = scene.render(w, h)
        x0, y0, x1, y1 = scene.scene_rect(w, h)
        sub = out[y0:y1, x0:x1].astype(int)
        nz = np.nonzero((np.abs(sub - np.array(pose3d.COLOR["bg"])).sum(axis=2) > 20))
        if not len(nz[0]):
            fails.append(f"pose3d: nothing drawn in the scene area at {w}x{h}")
            continue
        fill = (nz[0].max() - nz[0].min() + 1) / max(1, y1 - y0)
        if fill < 0.5:
            fails.append(f"pose3d: at {w}x{h} the scene fills only {fill:.0%} of its area "
                         "-- the fit is not using the viewport")
        scale = scene._hud_metrics(w, h)[0]
        t1, t2, t3 = scene.hud_lines()
        cols = 3 if scene._hud_metrics(w, h)[2] else 1
        budget = (w - 32) // cols
        over = [t for t, _ in t1 + t2 + t3
                if cv2.getTextSize(pose3d._fit_text(t, budget, scale),
                                   cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > budget]
        if over:
            fails.append(f"pose3d: {len(over)} HUD line(s) overflow at {w}x{h}: {over[:1]}")
        print(f"  [v] OK  {w}x{h}: scene fills {fill:.0%}, HUD within bounds")
    if not any("pose3d" in f for f in fails):
        print(f"  [v] OK  {pose_dir}: {len(scene.track)} poses, {len(scene.geotags)} geotags, "
              f"orbit responsive")


def check_inline_pose(base: Path, pose_dir: str, fails: list) -> None:
    """The 3D view in the dead band under the frame: it must fill that
    rectangle and nothing else, cycle off -> inline -> window -> off, fall back
    to the window when the band is too short, stay frame-accurate during
    playback, and not stop playback when you orbit it."""
    import time
    try:
        from birdseye_review import pose3d
    except Exception:  # noqa: BLE001
        return  # already reported by check_pose3d
    if not (base / pose_dir / "flight_data.db").exists():
        print(f"  [v] SKIP  inline checks need pose data ({pose_dir} has none)")
        return
    app = single_dir_app(base, pose_dir)
    if app.dr is None:
        fails.append(f"[v] inline: {pose_dir} failed to load")
        return
    app.window_w, app.window_h = 1100, 1000
    app.render()   # populate the frame-image cache the rect helper reads

    rect = app._inline_rect_now()
    if rect is None:
        fails.append("[v] inline: no band at 1100x1000, where there should be ~260px")
        return
    app.on_key(ord("v"))
    if app.pose_mode != "inline":
        fails.append(f"[v] inline: first press gave {app.pose_mode!r}, expected 'inline'")
        return
    with_3d = app.render()
    x0, y0, x1, y1 = rect
    band = with_3d[y0:y1, x0:x1].astype(int)
    lit = int((np.abs(band - np.array(pose3d.COLOR["bg"])).sum(axis=2) > 20).sum())
    if lit < 2000:
        fails.append(f"[v] inline: only {lit} lit px in the band -- it looks empty")
    app.pose_mode = "off"
    plain = app.render()
    app.pose_mode = "inline"
    if not np.array_equal(plain[:y0 - POSE_DIVIDER_SLACK, :x1], with_3d[:y0 - POSE_DIVIDER_SLACK, :x1]):
        fails.append("[v] inline: the 3D view drew above its rectangle, over the frame image")
    if not np.array_equal(plain[y1:, :], with_3d[y1:, :]):
        fails.append("[v] inline: the 3D view drew into the reserved status-bar band")

    app.on_key(ord("v"))
    if app.pose_mode != "window":
        fails.append(f"[v] cycle: second press gave {app.pose_mode!r}, expected 'window'")
    app.on_key(ord("v"))
    if app.pose_mode != "off":
        fails.append(f"[v] cycle: third press gave {app.pose_mode!r}, expected 'off'")

    # A window shape with no usable band must skip inline, not squash it.
    app.window_w, app.window_h = 1100, 700
    app.render()
    if app._inline_rect_now() is not None:
        fails.append("[v] inline: a 1100x700 window should leave no usable band")
    app.on_key(ord("v"))
    if app.pose_mode != "window":
        fails.append(f"[v] inline: with no band, [v] gave {app.pose_mode!r}, expected 'window'")
    app.close_pose_view()

    # Playback: same frame in both halves, follow engages, orbit doesn't stop it.
    app.window_w, app.window_h = 1100, 1000
    app.render()
    app.on_key(ord("v"))
    scene = app.pose_view.scene
    app.on_key(ord("6"))
    app._play_speed_idx = len(rg.PLAY_INTERVALS_SECONDS) - 1
    pairs = []
    for _ in range(12):
        app.render()
        pairs.append((app.current_frame_idx(), scene.frame_idx))
        time.sleep(rg.PLAY_INTERVALS_SECONDS[-1] + 0.01)
    if any(a != b for a, b in pairs):
        fails.append(f"[v] playback: 3D frame lagged the GUI: {pairs}")
    elif len({a for a, _ in pairs}) < 2:
        fails.append("[v] playback: the frame never advanced -- play is not running")
    else:
        print(f"  [v] OK  playback in sync over {len(pairs)} ticks "
              f"(frames {pairs[0][0]}..{pairs[-1][0]})")
    if not scene.follow:
        fails.append("[v] playback: the 3D camera did not auto-follow the aircraft")

    az = scene.camera.azimuth
    mx, my = (x0 + x1) // 2, (y0 + y1) // 2
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, mx, my, 0, None)
    app.on_mouse(cv2.EVENT_MOUSEMOVE, mx + 60, my + 10, 0, None)
    app.on_mouse(cv2.EVENT_LBUTTONUP, mx + 60, my + 10, 0, None)
    if scene.camera.azimuth == az:
        fails.append("[v] inline: dragging in the band did not orbit the 3D view")
    if app._play_direction == 0:
        fails.append("[v] inline: orbiting the 3D view stopped playback")
    else:
        print("  [v] OK  orbit in the inline band works and leaves playback running")

    app.on_key(ord("6"))
    app.render()
    if scene.follow:
        fails.append("[v] playback: follow was not restored after playback stopped")

    # And a click on the image selects a detection.
    app.selected_key = None
    now = app.current_frame_idx()
    here = app.dr.unmatched_clusters_by_frame.get(now, [])
    if here:
        geom = rg.DisplayGeometry(1920, 1200, app.window_w, app.window_h)
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, *geom.to_screen(here[0].x, here[0].y), 0, None)
        if app.selected_key is None:
            fails.append("[v] inline: a click on the frame image does not select a detection")
        else:
            print("  [v] OK  frame-image clicks select normally")
    app.close_pose_view()


def check_pose_centroids(base: Path, pose_dir: str, fails: list) -> None:
    """The geo-cluster centroid layer: it must follow [o],
    dim the cloud, and toggle on an UNSHIFTED key.

    Driven through a real PoseScene rather than an App: review_gui's
    _sync_pose_centroids is the only caller and all it does is fill these
    arrays, so what needs asserting is the scene contract.

    THE KEY MUST BE UNSHIFTED AND MUST GO THROUGH on_key(). cv2.waitKey()
    reports a lowercase char and drops Shift entirely on this build, so a
    shifted binding is silently dead. Asserting through on_key() rather than
    by reading the source is what makes that testable.
    """
    try:
        from birdseye_review import pose3d
    except Exception:  # noqa: BLE001
        return
    import numpy as _np
    d = base / pose_dir
    if not (d / "flight_data.db").exists():
        print(f"  [y] SKIP  centroid checks need pose data ({pose_dir} has none)")
        return
    try:
        sc = pose3d.PoseScene(d, pose_dir, 2200)
    except Exception as e:  # noqa: BLE001
        fails.append(f"[y] centroids: scene build failed: {type(e).__name__}: {e}")
        return
    view = pose3d.PoseView(sc)

    # 1. no centroids -> the cloud must NOT be dimmed for nothing on screen
    if sc.centroids_primary():
        fails.append("[y] centroids: empty population reported as primary")

    ox, oy = float(sc.origin[0]), float(sc.origin[1])
    xy = _np.array([[ox + 0.5, oy + 0.5], [ox + 1.0, oy + 1.0], [ox + 2.0, oy + 2.0]])
    cols = _np.array([[0, 200, 0], [0, 0, 255], [0, 140, 255]], _np.uint8)
    conf = _np.array([True, False, False])
    wide = _np.array([False, True, False])

    # 2. absolute UTM in, origin-relative out -- the caller holds
    #    GeoScaffold.points, which are absolute, and must not have to know.
    sc.set_centroids(xy, cols, conf, wide, {"confirmed_real": 2}, "majority", "n")
    if not sc.centroids_primary():
        fails.append("[y] centroids: populated but not primary")
    if abs(float(sc.centroids[0][0]) - 0.5) > 1e-6:
        fails.append(f"[y] centroids: not recentred on origin "
                     f"({sc.centroids[0][0]} should be 0.5)")
    if abs(float(sc.centroids[0][2]) - sc.ground_z) > 1e-9:
        fails.append("[y] centroids: not on the ground plane -- a geo cluster "
                     "is a claim about a spot on the ground")

    # 3. a length mismatch must be refused, not half-applied: colours are
    #    index-aligned and a partial write would mis-colour every verdict.
    before = len(sc.centroids)
    sc.set_centroids(xy, cols[:2])
    if len(sc.centroids) != before:
        fails.append("[y] centroids: mismatched colours were applied anyway")

    # 4. [o] OFF means NO STATUS CLAIMED, and the HUD must say so rather than
    #    letting grey read as 'unreviewed'. There is deliberately no default
    #    rule; see geo_metrics.VERDICT_RULES.
    sc.set_centroids(xy, cols, conf, wide, {}, None, "")
    line = " ".join(t for t, _ in sc.hud_lines()[2])
    if "no status" not in line or "[o]" not in line:
        fails.append(f"[y] centroids: [o]-off HUD does not say no status is "
                     f"claimed: {line!r}")

    # 5. [o] ON names the rule on screen -- the colours mean nothing without it
    sc.set_centroids(xy, cols, conf, wide,
                     {"confirmed_real": 1, "hard_negative": 1}, "positive_wins", "")
    line = " ".join(t for t, _ in sc.hud_lines()[2])
    for want in ("positive_wins", "confirmed_real"):
        if want not in line:
            fails.append(f"[y] centroids: HUD omits {want!r}: {line!r}")

    # 6. they must actually be drawn, and drawn ON TOP of the dimmed cloud
    n = 3000
    rng = _np.random.default_rng(0)
    sc.cloud_xyz = _np.column_stack([rng.normal(0, 10, n), rng.normal(0, 10, n),
                                     _np.full(n, sc.ground_z)])
    sc.cloud_frame = _np.zeros(n, _np.int64)
    sc.cloud_colors = _np.full((n, 3), 90, _np.uint8)
    sc.cloud_geotag_hit = _np.zeros(n, dtype=bool)
    sc.cloud_layer = _np.zeros(n, _np.int16)
    sc.cloud_layer_visible = _np.ones(1, dtype=bool)
    sc.frame_idx = None
    with_c = sc.render(900, 700)
    sc.set_centroids(_np.zeros((0, 2)), _np.zeros((0, 3), _np.uint8))
    without = sc.render(900, 700)
    if _np.array_equal(with_c, without):
        fails.append("[y] centroids: rendering is identical with and without them")

    # 7. the toggle, through on_key, unshifted
    sc.set_centroids(xy, cols, conf, wide, {"confirmed_real": 3}, "majority", "")
    if not view.on_key("y"):
        fails.append("[y] centroids: on_key('y') not consumed")
    if sc.show_centroids or sc.centroids_primary():
        fails.append("[y] centroids: toggle did not turn them off")
    line = " ".join(t for t, _ in sc.hud_lines()[2])
    if "OFF" not in line or "[y]" not in line:
        fails.append(f"[y] centroids: HUD does not name the key when off: {line!r}")
    view.on_key("y")
    if not sc.show_centroids:
        fails.append("[y] centroids: toggle is not reversible")
    if view.on_key("Y"):
        fails.append("[y] centroids: a SHIFTED alias was consumed -- cv2.waitKey "
                     "drops Shift on this build, so it could never fire")

    # 8. the cloud must actually step back, in weight AND in size
    if not (pose3d.CLOUD_ALPHA_DIMMED < pose3d.CLOUD_ALPHA):
        fails.append("[y] centroids: dimmed cloud alpha is not lower")
    if not (pose3d.CLOUD_MARKER_PX_DIMMED < pose3d.CLOUD_MARKER_PX):
        fails.append("[y] centroids: dimmed cloud marker is not smaller")

    print("  [y] OK  centroids: recentred onto the ground plane, status follows "
          "[o] (grey + 'no status' when off), drawn over a dimmed cloud, "
          "unshifted toggle reversible")


def check_pose_cloud(base: Path, pose_dir: str, fails: list) -> None:
    """The whole-directory detection cloud: every MODEL LAYER's detections
    back-projected through their own frame's pose, in the layer palette, with
    geotag hits marked.

    Model layers occupy cloud_layer slots 0..n-1. The load-bearing part is
    that the pose index must be keyed off frame INDICES, not the frame COUNT.
    """
    try:
        from birdseye_review import pose3d
    except Exception:  # noqa: BLE001
        return
    if not (base / pose_dir / "flight_data.db").exists():
        print(f"  [d] SKIP  cloud checks need pose data ({pose_dir} has none)")
        return
    app = single_dir_app(base, pose_dir)
    if app.dr is None:
        fails.append(f"[d] cloud: {pose_dir} failed to load")
        return
    app.window_w, app.window_h = 1100, 1000
    app.render()
    app.on_key(ord("v"))
    if app.pose_view is None:
        fails.append("[d] cloud: the 3D view did not open")
        return
    scene = app.pose_view.scene
    dr = app.dr

    # Regression: frame INDICES can run well past the frame COUNT (a flight
    # subset can start hundreds of frames in). Building the pose index off the
    # count silently drops every frame above it.
    n_fixed = len(app._pose_cloud_fixed)
    if len(scene.cloud_xyz) != n_fixed:
        fails.append(f"[d] cloud: {len(scene.cloud_xyz)} points for {n_fixed} colours -- "
                     "geometry and colours are misaligned")
    if dr.frames_all and max(dr.frames_all) >= len(dr.frames_all):
        hi = max(dr.frames_all)
        if not (scene.cloud_frame > len(dr.frames_all)).any():
            fails.append(f"[d] cloud: no points from frames above index {len(dr.frames_all)} "
                         f"(indices run to {hi}) -- the pose index is keyed off the frame COUNT")
    if not scene.cloud_geotag_hit.any():
        fails.append("[d] cloud: no point was matched to a geotag (expected some at 0.35 m)")

    # EVERY POINT IS A MODEL LAYER'S, and every point carries a fixed palette
    # colour. A None here would mean a population slipped in with no layer
    # identity.
    if any(f is None for f in app._pose_cloud_fixed):
        fails.append(f"[d] cloud: {sum(1 for f in app._pose_cloud_fixed if f is None)} "
                     "point(s) carry no palette colour -- a non-layer population is present")
    if dr.layers:
        slots = set(int(v) for v in np.unique(scene.cloud_layer))
        if slots and min(slots) != 0:
            fails.append(f"[d] cloud: layer slots start at {min(slots)}, expected 0")
        if max(slots) >= len(dr.layers):
            fails.append(f"[d] cloud: layer slot {max(slots)} exceeds {len(dr.layers)} layers")

    # [d] toggles it, and the toggle actually changes the pixels. The pixel
    # count is also the visibility floor: single-pixel markers at a low alpha
    # are hard to pick out.
    on = scene.render(900, 700)
    app.on_key(ord("d"))
    if scene.show_cloud:
        fails.append("[d] did not toggle the cloud off")
    off = scene.render(900, 700)
    if np.array_equal(on, off):
        fails.append("[d] cloud: toggling changed nothing on screen")
    lit = int((np.abs(on.astype(int) - off.astype(int)).sum(axis=2) > 12).sum())
    if lit < MIN_CLOUD_PIXELS:
        fails.append(f"[d] cloud: only {lit} pixels differ with it on "
                     f"(want >= {MIN_CLOUD_PIXELS}) -- the markers are too faint")
    else:
        print(f"  [d] OK  cloud paints {lit} pixels at alpha {pose3d.CLOUD_ALPHA}")
    app.on_key(ord("d"))

    # Zoom must be drag/key driven, never the wheel: this OpenCV is built
    # against Qt, whose window zooms the whole displayed image on a wheel
    # event and will not let a callback consume it.
    view = app.pose_view
    cam = scene.camera
    d0 = cam.distance
    view.on_mouse(cv2.EVENT_MOUSEWHEEL, 400, 300, 7864320, None)
    view.on_mouse(cv2.EVENT_MOUSEWHEEL, 400, 300, -7864320, None)
    if cam.distance != d0:
        fails.append("[v] the 3D view still zooms on the mouse wheel -- Qt owns that gesture")
    for label, down, flags in (("ctrl+left-drag", cv2.EVENT_LBUTTONDOWN, cv2.EVENT_FLAG_CTRLKEY),
                               ("middle-drag", cv2.EVENT_MBUTTONDOWN, 0)):
        before, az = cam.distance, cam.azimuth
        up = cv2.EVENT_LBUTTONUP if down == cv2.EVENT_LBUTTONDOWN else cv2.EVENT_MBUTTONUP
        view.on_mouse(down, 400, 300, flags, None)
        view.on_mouse(cv2.EVENT_MOUSEMOVE, 400, 360, flags, None)
        view.on_mouse(up, 400, 360, 0, None)
        if cam.distance <= before:
            fails.append(f"[v] {label} down did not pull the camera back")
        if cam.azimuth != az:
            fails.append(f"[v] {label} also orbited -- zoom and orbit are crossed")
        cam.distance = before
    if not any("wheel" in f or "drag" in f for f in fails):
        print("  [v] OK  zoom is ctrl-drag / middle-drag / +- only; wheel left to Qt")
    if not fails:
        print(f"  [d] OK  cloud {len(scene.cloud_xyz)} pts over "
              f"{len(dr.layers)} layer slot(s), all palette-coloured, "
              f"{int(scene.cloud_geotag_hit.sum())} on geotags")
    app.close_pose_view()


def _pre_recall_only(recall: dict) -> dict:
    """The frozen PRE recall block: everything except the POST arm and its
    shared metadata. `post` is keyed by tie-break rule and `post_common` carries
    the added-marker counts, so both move legitimately -- only what is left is
    the arm that no keystroke may touch."""
    return {k: v for k, v in recall.items() if k not in ("post", "post_common")}


def check_geo_metrics(base: Path, pose_dir: str, fails: list) -> None:
    """Geospatial metrics: the paper's geospatial precision/recall definitions,
    computed live.

    The valuable assertions here are the invariants, not the absolute numbers:
    PRE must not move when a judgment is committed -- in EITHER precision or
    recall. POST recall, by contrast,
    MUST move: POST precision enlarges the positive class, and recall's
    denominator is the positive class. So assert it moves in the right
    direction, only via `c` and `a`, and stays a bound. See
    geo_metrics.post_recall for the derivation.
    """
    try:
        from birdseye_review import geo_metrics as gm
        from birdseye_review import geo_eval as ge
        from birdseye_review import run_detector as rd
    except Exception as e:  # noqa: BLE001
        print(f"  [geo] SKIP  geo_metrics unimportable ({e})")
        return

    # The GUI mirrors these at module scope so build_arg_parser() need not pull
    # in sklearn. If they ever drift, every default silently changes meaning.
    for gui_val, ref_val, name in ((rg._GEO_EPS, ge.DBSCAN_EPS, "eps"),
                                   (rg._GEO_MIN_SAMPLES, ge.DBSCAN_MIN_SAMPLES, "min_samples"),
                                   (rg._GEO_EPS_PREDICT, ge.EPS_PREDICT_DETS, "eps_predict"),
                                   (rg._GEO_GUI_DEDUP_CM, gm.GUI_DEDUP_CM, "dedup_cm")):
        if gui_val != ref_val:
            fails.append(f"[geo] constant drift: review_gui {name}={gui_val} != {ref_val}")

    # The in-view polygon test (image bounds inset by the 81.5 px
    # projection-noise buffer): the vectorised twin must agree with
    # run_detector.inside exactly,
    # including the on-edge (cross product == 0) case it deliberately skips.
    poly = rd.inner_bound((1920, 1200))
    pts = np.array([[float(x), float(y)] for x in (-5, 0, 81.5, 82, 960, 1838.5, 1839, 1925)
                    for y in (-5, 0, 81.5, 82, 600, 1118.5, 1119, 1205)])
    mine = gm.inside_vec(pts, poly)
    theirs = np.array([rd.inside(pt, poly) for pt in pts])
    if not (mine == theirs).all():
        bad = pts[mine != theirs]
        fails.append(f"[geo] inside_vec disagrees with run_detector.inside at {len(bad)} pts, e.g. {bad[:3].tolist()}")

    # Verdict precedence, on a hand-built tally -- no I/O, no clustering.
    v = gm.GeoVerdicts()
    v.tally(0, gm.POSITIVE_JUDGMENT)
    v.tally(1, gm.NEGATIVE_JUDGMENT)
    v.tally(2, gm.POSITIVE_JUDGMENT); v.tally(2, gm.POSITIVE_JUDGMENT); v.tally(2, gm.NEGATIVE_JUDGMENT)
    v.tally(3, gm.POSITIVE_JUDGMENT); v.tally(3, gm.NEGATIVE_JUDGMENT)
    cases = [
        # unanimous clusters: every rule must agree
        (0, "positive_wins", gm.POSITIVE_JUDGMENT),
        (0, "hard_negative_wins", gm.POSITIVE_JUDGMENT), (0, "majority", gm.POSITIVE_JUDGMENT),
        (1, "positive_wins", gm.NEGATIVE_JUDGMENT),
        (1, "hard_negative_wins", gm.NEGATIVE_JUDGMENT), (1, "majority", gm.NEGATIVE_JUDGMENT),
        # conflicted (2 real, 1 hard-neg): the rules must actually diverge here,
        # which is the whole reason all three are reported
        (2, "positive_wins", gm.POSITIVE_JUDGMENT),        # default: the human real is decisive
        (2, "hard_negative_wins", gm.NEGATIVE_JUDGMENT),   # one hard-neg outranks two reals
        (2, "majority", gm.POSITIVE_JUDGMENT),             # ... and loses the vote
        # conflicted 1-1: positive wins; majority abstains
        (3, "positive_wins", gm.POSITIVE_JUDGMENT),
        (3, "hard_negative_wins", gm.NEGATIVE_JUDGMENT),
        (3, "majority", None),                             # tie -> geometry stands
        (9, "positive_wins", None),                        # unjudged -> geometry stands
        (9, "majority", None),
    ]
    # There must be NO default and no way to select a primary rule: every display
    # shows all three, and `verdict()` requires the caller to name one.
    import inspect
    if inspect.signature(gm.GeoVerdicts.verdict).parameters["rule"].default is not inspect.Parameter.empty:
        fails.append("[geo] GeoVerdicts.verdict() has a default rule again")
    if tuple(gm.VERDICT_RULES) != ("positive_wins", "majority", "hard_negative_wins"):
        fails.append(f"[geo] VERDICT_RULES must be ordered most-positive first, got "
                     f"{gm.VERDICT_RULES} -- displays rely on that order reading as a bracket")
    if set(gm.VERDICT_RULE_ABBR) != set(gm.VERDICT_RULES):
        fails.append("[geo] VERDICT_RULE_ABBR does not cover every rule")
    if tuple(gm.VERDICT_RULE_ABBR[r] for r in gm.VERDICT_RULES) != rg._GEO_RULE_ABBRS:
        fails.append(f"[geo] review_gui._GEO_RULE_ABBRS {rg._GEO_RULE_ABBRS} has drifted from "
                     f"geo_metrics.VERDICT_RULE_ABBR -- the HUD budget depends on the count")
    for cid, rule, want in cases:
        got = v.verdict(cid, rule)
        if got != want:
            fails.append(f"[geo] verdict({cid}, {rule}) = {got!r}, expected {want!r}")
    if v.n_conflicted != 2:
        fails.append(f"[geo] n_conflicted = {v.n_conflicted}, expected 2")

    # ---- the POST recall arm, as closed-form arithmetic. No I/O, no clustering.
    # R_POST = (n_hit + c) / (n_den + c + a).  Note the FOURTH case: with added
    # markers R_POST drops BELOW R_PRE, which is why the R_POST >= R_PRE
    # invariant below is guarded on a == 0. An unguarded assertion would be
    # wrong the first time a reviewer right-clicks.
    for n_hit, n_den, c, a, want in (
            (10, 20, 0, 0, 0.50),   # c == 0 and a == 0 -> exactly R_PRE
            (10, 20, 5, 0, 0.60),   # c raises it, and it stays >= R_PRE
            (10, 20, 5, 5, 0.50),   # a pulls it back down
            (10, 20, 0, 5, 0.40),   # a alone puts it BELOW R_PRE = 0.50
            (20, 20, 7, 0, 1.00),   # saturates at 1, never above
            (0, 0, 0, 0, 0.00)):    # zero-division safe
        got = gm.post_recall(n_hit, n_den, c, a)[2]
        if abs(got - want) > 1e-12:
            fails.append(f"[geo] post_recall({n_hit},{n_den},{c},{a}) = {got:.4f}, expected {want}")

    # The same formula lives twice -- review_gui cannot import geo_metrics at
    # module scope (sklearn), so corrected_prf reimplements it for pixel space.
    # Assert the two copies agree, the way the _GEO_* constants above are
    # asserted not to drift.
    for tp0, fn0, c, a in ((10, 10, 0, 0), (10, 10, 5, 0), (10, 10, 5, 3),
                           (0, 7, 2, 1), (5, 0, 0, 4), (0, 0, 0, 0)):
        pix = rg.corrected_prf(tp0, 1, fn0, tp0 + c, 1, c, a)[1]
        geo = gm.post_recall(tp0, tp0 + fn0, c, a)[2]
        if abs(pix - geo) > 1e-12:
            fails.append(f"[geo] corrected_prf and post_recall disagree at "
                         f"(tp0={tp0}, fn0={fn0}, c={c}, a={a}): {pix} != {geo}")

    # post_spread over a hand-built summary: no I/O, and it pins the ordering
    # invariant the whole display rests on.
    synth = {"post": {"positive_wins": {"precision": 0.48},
                      "majority": {"precision": 0.41},
                      "hard_negative_wins": {"precision": 0.32}},
             "recall": {"post": {"positive_wins": {"recall_all_bound": 0.68},
                                 "majority": {"recall_all_bound": 0.67},
                                 "hard_negative_wins": {"recall_all_bound": 0.66}}},
             "verdict_rules": list(gm.VERDICT_RULES)}
    if gm.post_spread(synth, "precision") != (0.32, 0.48):
        fails.append(f"[geo] post_spread precision = {gm.post_spread(synth, 'precision')}")
    if gm.post_spread(synth, "recall_all_bound") != (0.66, 0.68):
        fails.append("[geo] post_spread does not read the recall block")

    if not (base / pose_dir / "flight_data.db").exists():
        print(f"  [geo] SKIP  live checks need pose data ({pose_dir} has none)")
        return

    app = single_dir_app(base, pose_dir)
    if app.dr is None:
        fails.append(f"[geo] {pose_dir} failed to load")
        return
    app.window_w, app.window_h = 1400, 1000
    app.render()

    # THE ARM IS UNCONDITIONAL: there is no key to press. Wait for the worker
    # instead of toggling a flag.
    app.cycle_pose_view()
    t0 = time.time()
    if not await_geo(app, want_layers=True):
        fails.append("[geo] the geo worker never published a per-layer summary")
        return
    build_s = time.time() - t0

    s0 = layer_summary(app)
    if s0 is None or not s0.get("available"):
        fails.append(f"[geo] metrics unavailable on {pose_dir}: {s0}")
        return
    print(f"  [geo] {s0['n_clusters']} clusters, {s0['recall']['tp_geotags']}/"
          f"{s0['recall']['n_geotags']} geotags hit, {s0['recall']['n_in_view']} in view, "
          f"built in {build_s:.1f}s")

    # The index join must resolve judged points, not silently drop them -- that
    # is the whole reason it is a grid-cell lookup rather than a radius search.
    adj = s0["adjudication"]
    if adj["n_judged_points"] and adj["n_unresolved_points"] > 0.02 * adj["n_judged_points"]:
        fails.append(f"[geo] {adj['n_unresolved_points']} of {adj['n_judged_points']} judged points "
                     "failed to resolve to a geo cluster -- the cell join is not exact")

    # In-view matches are a SUBSET of all matches -- that much is structural.
    #
    # Equality does not hold: matching counts a geotag as hit when it lies
    # within eps_predict (0.35 m) of a CORE POINT, while geotags_in_view() is
    # exact polygon containment under the in-view polygon test (image bounds
    # inset by the 81.5 px projection-noise buffer). A geotag sitting a few
    # centimetres outside every frame's footprint can therefore legitimately
    # match a detection at the frame edge, and the two sets are not nested.
    # Measured on the full 2,120-frame sample flight: 4 such geotags, all
    # 0.18-0.34 m from a core point, i.e. every one inside the match tolerance.
    #
    # So assert the property that actually has teeth: the discrepancy must be
    # EXPLAINED BY THE TOLERANCE. A matched geotag that is neither in view nor
    # within eps_predict of a core point would mean the denominator really is
    # wrong.
    r = s0["recall"]
    outside = r["tp_geotags"] - r["tp_geotags_in_view"]
    if outside < 0:
        fails.append(f"[geo] {r['tp_geotags_in_view']} in-view matches exceed {r['tp_geotags']} "
                     "total matches -- in-view hits must be a subset")
    if outside:
        sc = app._geo_scaffold
        eps = float(sc.params["eps_predict"])
        core = sc.points[sc.core_mask][:, :2]
        hit = sc.geotag_label != -1
        unexplained = []
        for i in np.where(hit & ~sc.in_view)[0]:
            d = float(np.linalg.norm(core - sc.geotags[i], axis=1).min()) if len(core) else 9e9
            if d > eps + 1e-9:
                unexplained.append((int(i), round(d, 3)))
        if unexplained:
            fails.append(f"[geo] matched geotag(s) neither in view nor within eps_predict={eps} "
                         f"of a core point: {unexplained} -- the in-view denominator is wrong")
        else:
            print(f"  [geo] note: {outside} matched geotag(s) sit outside every frame polygon but "
                  f"within eps_predict={eps} m of a core detection -- the match tolerance "
                  f"reaches past the imaged footprint; in-view recall is the tighter arm")
    if r["n_in_view"] > r["n_geotags"]:
        fails.append("[geo] more geotags in view than exist")

    # The POST arm's own attainability. c_in_view == c by construction (a
    # cluster exists only because the detector fired on it in a processed
    # frame), so the two POST numerators differ by exactly what the PRE ones do.
    rp = r["post"][gm.VERDICT_RULES[0]]
    common = r["post_common"]
    if set(r["post"]) != set(gm.VERDICT_RULES):
        fails.append(f"[geo] recall.post must carry every tie-break rule, has {sorted(r['post'])}")
    if set(s0["post"]) != set(gm.VERDICT_RULES):
        fails.append(f"[geo] post must carry every tie-break rule, has {sorted(s0['post'])}")
    for rule in gm.VERDICT_RULES:
        q = r["post"][rule]
        if q["tp_geotags"] - q["tp_geotags_in_view"] != r["tp_geotags"] - r["tp_geotags_in_view"]:
            fails.append(f"[geo] POST[{rule}] recall numerators disagree with PRE by more than c "
                         "-- c_in_view should equal c")
        if not q["is_bound"]:
            fails.append(f"[geo] recall.post[{rule}] must be flagged is_bound")
    rp = dict(rp, **common)
    # Every raw added click lands in exactly one bucket: dropped as unresolved,
    # dropped near a geotag, dropped inside a cluster, or folded into a group.
    # The added-marker twin of the POST tp+fp == n_clusters check below.
    if rp["n_added_raw"] < (rp["n_added_unresolved"] + rp["n_added_near_geotag"]
                            + rp["n_added_in_cluster"]):
        fails.append(f"[geo] added-marker buckets over-count: {rp['n_added_raw']} raw clicks "
                     f"but {rp['n_added_unresolved']}+{rp['n_added_near_geotag']}"
                     f"+{rp['n_added_in_cluster']} dropped")
    folded = (rp["n_added_raw"] - rp["n_added_unresolved"] - rp["n_added_near_geotag"]
              - rp["n_added_in_cluster"])
    if rp["a_added_units"] > folded:
        fails.append(f"[geo] {rp['a_added_units']} added units from only {folded} surviving "
                     "clicks -- dedup cannot invent targets")
    if rp["a_added_units_in_view"] > rp["a_added_units"]:
        fails.append("[geo] more added markers in view than exist")

    # PRE is geometry only. Committing a judgment must not touch it,
    # and must not touch recall in either arm.
    target = next((c for f, cs in app.dr.unmatched_clusters_by_frame.items() for c in cs
                   if app.dr.store.get(c.frame_idx, c.x, c.y) == "unreviewed"), None)
    if target is None:
        print("  [geo] note: no unreviewed cluster left to test the override with")
        check_added_markers(app, gm, fails)
        return
    app.selected_key = (target.frame_idx, target.x, target.y)
    app.frame_pos = app.dr.frames_all.index(target.frame_idx)
    # TWO BASELINES, deliberately. `s0` is the pre-commit state and stays put:
    # the PRE arm must be byte-identical after ANY number of commits. The
    # c/recall assertions below are DELTAS, so they need the state immediately
    # before this particular commit -- comparing the second commit against `s0`
    # would measure the combined effect of both and report "hard_negative moved
    # c 0 -> 1 the wrong way" when confirmed_real raised c to 1 and
    # hard_negative left it there.
    prev = s0
    for judgment in ("confirmed_real", "hard_negative"):
        app.commit(judgment)
        s1 = layer_summary(app)
        if s1["pre"] != s0["pre"]:
            fails.append(f"[geo] committing {judgment} moved the PRE arm "
                         f"{s0['pre']} -> {s1['pre']} -- PRE must be geometry only")
        # PRE recall is geometry only: byte-identical across any commit.
        pre0, pre1 = _pre_recall_only(s0["recall"]), _pre_recall_only(s1["recall"])
        if pre1 != pre0:
            fails.append(f"[geo] committing {judgment} moved the PRE recall block -- "
                         "it is geometry only and must not carry an adjudication clause")
        # POST recall, by contrast, must track the verdict. confirmed_real can
        # only add to c; hard_negative can only take away.
        a_held = (s1["recall"]["post_common"]["a_added_units"]
                  == prev["recall"]["post_common"]["a_added_units"])
        a_now = s1["recall"]["post_common"]["a_added_units"]
        # Every rule must move the right way -- not just one privileged arm.
        for rule in gm.VERDICT_RULES:
            p0, p1 = prev["recall"]["post"][rule], s1["recall"]["post"][rule]
            c0, c1 = p0["c_confirmed_unmatched_clusters"], p1["c_confirmed_unmatched_clusters"]
            moved_ok = (c1 >= c0) if judgment == "confirmed_real" else (c1 <= c0)
            if not moved_ok:
                fails.append(f"[geo] committing {judgment} moved c[{rule}] {c0} -> {c1}, "
                             "the wrong way")
            if a_held and c1 != c0:
                want_up = judgment == "confirmed_real"
                got_up = p1["recall_all_bound"] >= p0["recall_all_bound"] - 1e-12
                if want_up != got_up:
                    fails.append(f"[geo] committing {judgment} moved the POST[{rule}] recall bound "
                                 f"{p0['recall_all_bound']:.4f} -> {p1['recall_all_bound']:.4f}, "
                                 "the wrong way for that verdict")
            for key, pre_key in (("recall_all_bound", "recall_all"),
                                 ("recall_in_view_bound", "recall_in_view")):
                if p1[key] > 1.0 + 1e-12:
                    fails.append(f"[geo] POST[{rule}] {key} = {p1[key]} exceeds 1")
                # R_POST == R_PRE exactly when nothing has been discovered...
                if c1 == 0 and a_now == 0 and abs(p1[key] - s1["recall"][pre_key]) > 1e-12:
                    fails.append(f"[geo] c == 0 and a == 0 but POST[{rule}] {key} != PRE {pre_key}")
                # ... and >= R_PRE ONLY while a == 0. With added markers it may
                # and should fall below: those are misses the survey never recorded.
                if a_now == 0 and p1[key] < s1["recall"][pre_key] - 1e-12:
                    fails.append(f"[geo] a == 0 but POST[{rule}] {key} {p1[key]:.4f} < PRE "
                                 f"{pre_key} {s1['recall'][pre_key]:.4f}")
            q = s1["post"][rule]
            if q["tp"] + q["fp"] != s1["n_clusters"]:
                fails.append(f"[geo] POST[{rule}] tp+fp != n_clusters -- every cluster must land "
                             "in exactly one bucket")
        prev = s1
    # Ordering the three rules by how positive they are must order their
    # precisions AND their recall bounds the same way. That monotonicity is what
    # lets a reader read the three rows as a bracket rather than as three
    # unrelated estimates, so the displays depend on it.
    s2 = layer_summary(app)
    for key, src in (("precision", s2["post"]),
                     ("recall_all_bound", s2["recall"]["post"]),
                     ("recall_in_view_bound", s2["recall"]["post"])):
        vals = [src[r][key] for r in gm.VERDICT_RULES]
        if not all(a >= b - 1e-12 for a, b in zip(vals, vals[1:])):
            fails.append(f"[geo] POST {key} is not monotone across VERDICT_RULES: "
                         f"{dict(zip(gm.VERDICT_RULES, vals))}")
    lo, hi = gm.post_spread(s2, "precision")
    print(f"  [geo] PRE arm held; all {len(gm.VERDICT_RULES)} tie-break rules moved with c and "
          f"stayed bounds (POST precision spans {lo:.4f}-{hi:.4f})")

    # The per-model geo VALUES are in [p]'s PER MODEL block; the geo block
    # carries only what every layer shares -- the definitions, the parameters
    # and the bound caveat. Both halves are asserted, because a value that
    # appears in neither would otherwise be caught by neither.
    geo_panel = app._geo_overlay_lines()
    geo_text = "\n".join(t for t, _ in geo_panel)
    if "R<= is" not in geo_text:
        fails.append("[p] the geo block shows no bound caveat -- the POST recall must never "
                     "read as a point estimate")
    if not any(t.lstrip().startswith("R<= is") and c == rg.COLOR["error"] for t, c in geo_panel):
        fails.append("[p] the bound caveat must render in the error colour, as a warning "
                     "rather than as a result")
    if "DBSCAN eps" not in geo_text:
        fails.append("[p] the geo block does not state the clustering parameters")

    panel = app._per_model_overlay_lines()
    text = "\n".join(t for t, _ in panel)
    if "GEO PRE" not in text or "GEO POST" not in text:
        fails.append("[p] the per-model block is missing a geo arm")
    if "<=" not in text:
        fails.append("[p] the per-model block shows no bound glyph on POST recall")
    # ALL THREE rules, for EVERY model, every time -- not "when they disagree".
    n_models = sum(1 for t, _ in panel if t.lstrip().startswith("GEO PRE"))
    n_post = sum(1 for t, _ in panel if t.lstrip().startswith("GEO POST"))
    if n_models and n_post != n_models * len(gm.VERDICT_RULES):
        fails.append(f"[p] the per-model block shows {n_post} GEO POST lines for {n_models} "
                     f"model(s), expected {n_models * len(gm.VERDICT_RULES)} -- one per "
                     f"tie-break rule per model")
    for abbr in rg._GEO_RULE_ABBRS:
        if f"GEO POST {abbr}" not in text:
            fails.append(f"[p] the per-model block never shows the {abbr!r} tie-break row")
    # THE ENRICHMENT THIS BLOCK EXISTS FOR: the raw geotag counts behind the
    # ratios. A ratio without its denominator is not quotable.
    if "geotags hit" not in text:
        fails.append("[p] the per-model block omits the raw geotag counts behind the ratios")
    print(f"  [p] OK  per-model block: {n_models} model(s) x {len(gm.VERDICT_RULES)} POST "
          f"rules + geotag counts; geo block carries the shared definitions and the bound")

    # The HUD must do the same, ON THE LIVE APP. check_hud_layout asserts the
    # row model from synthetic stats; this exercises it against a real scaffold.
    #
    # There is no height assertion here. The HUD carries a per-model block --
    # checkpoints x 7 rows -- so its total height exceeds PANEL_HUD_H by
    # design. The band-fitting property is asserted where it belongs, by
    # sweeping hud_scroll_view in check_hud_layout.
    rows = app.hud_model(app.dr, app.dr.frames_all[0])
    pinned = [r.cells[0][1] for r in rows if r.kind == "cells" and r.zone == "pinned"]
    if pinned:
        fails.append(f"HUD pinned zone carries metric rows {pinned} -- every metric is "
                     f"per model now, and the pinned zone is bars plus the params rule")
    # The pinned PARAMS rule row carries the live values that decide every
    # number below it.
    prule = [r for r in rows if r.zone == "pinned" and r.kind == "rule"]
    if len(prule) != 1:
        fails.append(f"expected exactly 1 pinned params rule row, got {len(prule)}")
    else:
        txt = prule[0].cells[0][1]
        for want in ("m2o ", "geo eps ", "match ", "prop:"):
            if want not in txt:
                fails.append(f"the pinned params row {txt!r} omits {want!r}")
        if len(prule[0].cells) != 1:
            fails.append("the pinned params row carries extra cells -- the P/R/F1 heads "
                         "would cap it at HUD_COL_X[0] and silently truncate it")
    # EVERY body group is a per-model group, and each repeats the full set.
    body = [r for r in rows if r.zone == "body"]
    per_model = sorted({r.group for r in body})
    for gi in per_model:
        labs = [r.cells[0][1] for r in body if r.group == gi and r.kind == "cells"]
        exp = ["PIXEL PRE", "PIXEL POST", "GEO PRE"] + \
              [f"GEO POST {gm.VERDICT_RULE_ABBR[r]}" for r in gm.VERDICT_RULES]
        if labs != exp:
            fails.append(f"HUD model group {gi} rows are {labs}, expected {exp}")
            break
    if len(per_model) != len(app.layers.visible):
        fails.append(f"HUD has {len(per_model)} model groups for "
                     f"{len(app.layers.visible)} visible layers")
    print(f"  HUD OK  geo on: {len(rows)} rows, {len(per_model)} per-model "
          f"groups, one POST row per tie-break rule in every one")

    check_added_markers(app, gm, fails)


def check_added_markers(app, gm, fails: list) -> None:
    """The `a` term end to end: an added marker is a target the detector never
    fired on, so it is a false negative the geotag survey also missed and it
    belongs in the POST recall DENOMINATOR -- the only term that constrains that
    arm's upper bound from above.

    The assertions that matter here are the two that stop `a` being wrong in
    opposite directions: the SAME burrow clicked in several frames must count
    ONCE (otherwise a is inflated by however many frames saw it, and the bound
    collapses), and a click on a target the survey ALREADY recorded must count
    NOT AT ALL (otherwise one miss is charged twice).
    """
    proj, scaffold = app._geo_proj, app._geo_scaffold
    if proj is None or scaffold is None or not scaffold.ok:
        print("  [geo] SKIP  added-marker checks need a live scaffold")
        return
    by_index = {i: row for i, row in enumerate(proj.poses)}
    poly = gm.inner_bound(proj.res)

    def fold(pts):
        return gm.collect_added(scaffold, proj, pts, lambda fi: by_index.get(fi))

    def to_pixel(frame_idx, utm_xy):
        """UTM -> pixel in one frame, on that frame's own ground plane (the
        inverse of _frame_to_utm, matching geotags_in_view's construction)."""
        row = by_index.get(frame_idx)
        if row is None:
            return None
        T_WC, radalt = proj.frame_pose(row)
        if radalt <= gm.MIN_RADALT_M:
            return None
        p3 = np.array([[float(utm_xy[0]), float(utm_xy[1]), T_WC[2, 3] - radalt]])
        px = proj.to_2d(p3, T_WC)
        if not np.isfinite(px).all() or not gm.inside_vec(px, poly)[0]:
            return None
        return float(px[0, 0]), float(px[0, 1])

    # Find one pixel that survives both drop rules -- i.e. genuinely unsurveyed
    # ground the detector never fired on. Scanning beats hand-picking: which
    # pixels qualify depends on this flight's geotags and clusters.
    frames = [fi for fi in app.dr.frames_all if fi in by_index]
    w, h = proj.res
    spot = None
    for fi in frames[::37]:
        for fx in (0.25, 0.5, 0.75):
            for fy in (0.25, 0.5, 0.75):
                x, y = w * fx, h * fy
                if fold([(fi, x, y)]).n_units == 1:
                    spot = (fi, x, y)
                    break
            if spot:
                break
        if spot:
            break
    if spot is None:
        print("  [geo] SKIP  no unsurveyed, undetected pixel found to place a marker on")
        return
    fi, x, y = spot
    utm = gm._frame_to_utm(proj, by_index[fi], np.array([[x, y]]))[0, :2]

    # The dedup rule. The same burrow clicked from a DIFFERENT frame is one
    # target, not two -- and the two projections disagree by decimetres, which
    # is exactly why this cannot use build_scaffold's grid dedup.
    second = next((p for fi2 in frames if fi2 != fi
                   for p in [to_pixel(fi2, utm)] if p is not None), None)
    if second is not None:
        fi2, (x2, y2) = next((f, to_pixel(f, utm)) for f in frames
                             if f != fi and to_pixel(f, utm) is not None)
        both = fold([(fi, x, y), (fi2, x2, y2)])
        if both.n_units != 1:
            fails.append(f"[geo] the same target clicked in frames {fi} and {fi2} counted as "
                         f"{both.n_units} added markers -- cross-frame dedup is broken, and "
                         "a is inflated by however many frames saw the burrow")
    else:
        print("  [geo] note: target visible in only one frame; cross-frame dedup untested")

    # The double-count rule. A click on a geotagged target is already in the
    # recall denominator, and already an FN if no cluster reached it.
    gt = next(((f, to_pixel(f, g)) for g in scaffold.geotags for f in frames
               if to_pixel(f, g) is not None), None)
    if gt is not None:
        fi3, (gx, gy) = gt
        near = fold([(fi3, gx, gy)])
        if near.n_near_geotag != 1 or near.n_units != 0:
            fails.append(f"[geo] a marker on an existing geotag counted as {near.n_units} added "
                         f"unit(s) (near_geotag={near.n_near_geotag}) -- that miss would be "
                         "charged twice")

    # Live: spawning must lower the bound, and removing must restore it exactly.
    s0 = layer_summary(app)
    top = gm.VERDICT_RULES[0]
    a0 = s0["recall"]["post_common"]["a_added_units"]
    r0 = s0["recall"]["post"][top]["recall_all_bound"]
    pre0 = _pre_recall_only(s0["recall"])
    app.spawn_point(fi, app.dr.image_path_by_frame.get(fi, ""), x, y)
    s1 = layer_summary(app)
    a1 = s1["recall"]["post_common"]["a_added_units"]
    r1 = s1["recall"]["post"][top]["recall_all_bound"]
    if a1 != a0 + 1:
        fails.append(f"[geo] spawning one added marker moved a {a0} -> {a1}, expected +1")
    if not r1 < r0:
        fails.append(f"[geo] an added marker must LOWER the POST recall bound "
                     f"({r0:.6f} -> {r1:.6f}) -- it is a miss nobody recorded")
    if _pre_recall_only(s1["recall"]) != pre0:
        fails.append("[geo] an added marker moved the PRE recall arm -- it must not")
    app.remove_spawned(fi, x, y)
    s2 = layer_summary(app)
    if (s2["recall"]["post_common"]["a_added_units"] != a0
            or abs(s2["recall"]["post"][top]["recall_all_bound"] - r0) > 1e-12):
        fails.append("[geo] removing the added marker did not restore a and the bound")
    print(f"  [geo] added markers: dedup across frames, no double-count on geotags, "
          f"and one marker moved the bound {r0:.4f} -> {r1:.4f}")


def scene_frames(d: Path) -> list:
    """Frame indices in this directory's labels.txt -- just enough for
    PoseScene's pose-coverage sanity check, without loading the whole GUI.
    Note PoseScene wants the max INDEX, not the count: they differ whenever
    the labels cover a subset of the flight."""
    seen = set()
    f = d / "labels.txt"
    if f.exists():
        for line in f.read_text().splitlines():
            parts = line.split()
            if parts:
                try:
                    seen.add(int(parts[0]))
                except ValueError:
                    pass
    return sorted(seen)


def check_run_resolution(rg, base, fails: list) -> None:
    """The CLI derives the session from a run directory instead of six flags.

    Uses one synthetic run (a run_manifest.json carrying
    winning_hyperparameters.spatial_threshold = 80 and no per-flight
    coverage) plus the sample's sealed run 17.
    """
    with tempfile.TemporaryDirectory() as tmp:
        synth = Path(tmp) / "synthetic_run"
        synth.mkdir()
        (synth / "run_manifest.json").write_text(
            json.dumps({"winning_hyperparameters": {"spatial_threshold": 80}}))
        _check_run_resolution(rg, base, synth, fails)


def _check_run_resolution(rg, base, synth: Path, fails: list) -> None:
    # 1. Content-based classification, order-independent.
    flight = "2025_04_16/pieranch_rect"
    if not rg.looks_like_run(synth) or rg.looks_like_dataset(synth):
        fails.append("[cli] run directory misclassified")
    if not rg.looks_like_dataset(base / flight) or rg.looks_like_run(base / flight):
        fails.append("[cli] flight directory misclassified")

    def resolve(argv):
        a = rg.build_arg_parser().parse_args(argv)
        a.dirs, run_arg = rg.classify_paths(a.paths, base)
        return a, rg.apply_run_context(a, run_arg, base)

    a1, _ = resolve([flight, str(synth)])
    a2, _ = resolve([str(synth), flight])
    if (a1.dirs, a1.spatial_threshold) != (a2.dirs, a2.spatial_threshold):
        fails.append("[cli] positional order changed the result -- classification is by "
                     "content, so it must not")

    # 2. Derivation: the matching threshold comes from run_manifest.json's
    #    winning_hyperparameters.
    ctx = rg.resolve_run_artifacts(synth)
    if not ctx.ok:
        fails.append(f"[cli] {synth.name} did not resolve: {ctx.problems}")
    elif ctx.spatial_threshold != 80.0:
        fails.append(f"[cli] {synth.name} spatial_threshold={ctx.spatial_threshold}, "
                     f"expected 80 (from run_manifest winning_hyperparameters)")

    # 3. The run's value is adopted with no flag; an explicit flag overrides and
    #    is recorded.
    a, _ = resolve([flight, str(synth)])
    if a.spatial_threshold != 80.0 or a.param_overrides:
        fails.append(f"[cli] derived spatial={a.spatial_threshold} overrides={a.param_overrides}, "
                     "expected 80.0 and none")
    a, _ = resolve([flight, str(synth), "--spatial-threshold", "30"])
    ov = a.param_overrides.get("spatial_threshold")
    if a.spatial_threshold != 30.0 or not ov or ov["run"] != 80.0 or ov["live"] != 30.0:
        fails.append(f"[cli] override not recorded correctly: {a.param_overrides}")

    # 4. COVERAGE AND THE HONESTY FLAG. Coverage is read from the run's own
    #    detection subdirectories and the honesty flag only from a sealed
    #    PREREGISTRATION.json. A run that keeps its detections nowhere this
    #    can see has empty coverage -- and empty coverage SUPPRESSES the "does
    #    not cover that flight" warning instead of firing it. Suppressing is
    #    the right way round: refusing a flight because no file listed it
    #    would be worse than not warning. Run 17, which has a sealed
    #    preregistration, gets both.
    a, ctx = resolve(["2099_01_01/uncovered_rect", str(synth)])
    if ctx.coverage:
        # If a run does expose its coverage, the refusal must work.
        if ctx.ok or not any("does not cover" in p for p in ctx.problems):
            fails.append("[cli] a run WITH coverage did not refuse an uncovered flight")
    elif not ctx.ok:
        fails.append(f"[cli] the synthetic run has no derivable coverage, so no flight may be "
                     f"refused on coverage grounds -- got problems {ctx.problems}")
    run17 = base / "results/17_holdout_pieranch"
    if run17.is_dir():
        c17 = rg.resolve_run_artifacts(run17)
        if not c17.scores_honest_for_test_dir:
            fails.append("[cli] run 17's sealed PREREGISTRATION.json does not establish "
                         "test-set honesty")
        if c17.test_dir != "2025_04_16/pieranch_rect":
            fails.append(f"[cli] run 17 test_dir={c17.test_dir!r}")

    # 5. Review-only mode: no run, no artifacts, GUI default parameter.
    a, ctx = resolve([flight])
    if ctx is not None or a.spatial_threshold != rg.PARAM_FALLBACKS["spatial_threshold"]:
        fails.append(f"[cli] review-only mode wrong: ctx={ctx} spatial={a.spatial_threshold}")
    print("  [cli] OK  spatial_threshold 80 derived from run_manifest; overrides "
          "recorded; run 17 honesty from PREREGISTRATION.json")


def check_stats_overlay_order(app, rg, txt, fails: list) -> None:
    """[p] must LEAD with the numbers a reviewer decides on.

    _draw_paginated shows ~38 of ~95 lines, so whatever is first is what most
    readers ever see. Provenance and calibration ahead of the metrics would push
    the metrics below the fold -- a real usability defect, not a matter of
    taste, and so it gets an assertion.
    """
    if not txt or not txt[0].startswith("SUMMARY"):
        fails.append(f"[p] must open with the SUMMARY block, got {txt[0]!r}" if txt
                     else "[p] produced no lines")
        return
    for needle in ("directory:", "reviewed:", "propagation rule", "SECTIONS BELOW"):
        if not any(needle in t for t in txt[:40]):
            fails.append(f"[p] {needle!r} is missing from the first screen")
    # Results before provenance and configuration.
    def first(pred):
        return next((i for i, t in enumerate(txt) if pred(t)), 10 ** 6)
    i_summary = first(lambda t: t.startswith("SUMMARY"))
    i_model = first(lambda t: t.startswith("PER MODEL"))
    i_adj = first(lambda t: t.startswith("ADJUDICATION"))
    i_cfg = first(lambda t: t.startswith("POST-PROCESSING CONFIG"))
    i_geo = first(lambda t: t.startswith("GEOSPATIAL"))
    if not (i_summary < i_model < i_adj < i_cfg < i_geo):
        fails.append(f"[p] section order regressed: summary={i_summary} per-model={i_model} "
                     f"adjudication={i_adj} config={i_cfg} geospatial={i_geo}")
    # cv2's Hershey fonts have no glyph for U+26A0 and draw a literal "?",
    # which reads as uncertainty about the claim rather than as a warning.
    bad = [t for t in txt if "⚠" in t]
    if bad:
        fails.append(f"[p] {len(bad)} line(s) use the unrenderable warning glyph, e.g. {bad[0]!r}")
    # [p] MUST NEED NO FILES AT ALL: every section is present from live state
    # alone.
    bare = [t for t, _ in app._stats_overlay_lines()]
    if not bare or not bare[0].startswith("SUMMARY"):
        fails.append("[p] does not open with SUMMARY")
    missing = [h for h in ("PER MODEL", "ADJUDICATION", "POST-PROCESSING CONFIG",
                           "GEOSPATIAL")
               if not any(t.startswith(h) for t in bare)]
    if missing:
        fails.append(f"[p] sections missing with no files loaded: {missing}")
    else:
        print(f"  [p] OK  opens with SUMMARY, sections ordered, {len(bare)} lines, no files needed")


def check_hud_layout(app, rg, fails: list) -> None:
    """The HUD is a SCROLLING TABLE of per-model PRE/POST rows.

    Asserted on the row model, never on pixels -- that is what hud_rows()
    exists for. There is no total height to bound (thirty checkpoints are
    30 x 118px of rows and the body scrolls). Asserted: the pinned/footer
    zones are exactly what they claim to be, the columns are readable at the
    narrowest panel, every group carries all three tie-break rules, a pending
    cell never reads as a measurement, and scrolling reaches every model.
    """
    rows = app.hud_model(app.dr, app.dr.frames_all[0])

    # -- zones ------------------------------------------------------------
    pinned = [r for r in rows if r.zone == "pinned"]
    footer = [r for r in rows if r.zone == "footer"]
    kinds = [r.kind for r in pinned]
    if kinds[:4] != ["text", "bar", "text", "bar"]:
        fails.append(f"HUD pinned block starts {kinds[:4]}, expected the two labelled bars")
    labels = [r.cells[0][1] for r in pinned if r.kind == "cells"]
    if labels:
        fails.append(f"HUD pinned metric rows are {labels} -- every metric is per model, "
                     f"and the pinned zone is bars plus the params rule")
    if kinds.count("rule") != 1:
        fails.append(f"HUD pinned zone has {kinds.count('rule')} rule rows, expected exactly "
                     f"the one params row")
    # The caveat and play rows are the ONLY things allowed in the footer: it is
    # bottom-anchored, and anything else put there would be silently pinned to
    # the bottom of the panel.
    for r in footer:
        if r.kind != "text":
            fails.append(f"HUD footer carries a {r.kind} row -- only the caveat and play "
                         "lines belong in the bottom-anchored zone")

    # -- the column grid, which is the whole readability property ---------
    metric = [r for r in rows if r.kind == "cells"]
    allowed = {0} | set(rg.HUD_COL_X)
    for r in metric:
        offs = {dx for dx, _, _ in r.cells}
        if not offs <= allowed:
            fails.append(f"HUD metric row {r.cells[0][1]!r} draws at {sorted(offs - allowed)}, "
                         f"outside HUD_COL_X {rg.HUD_COL_X}")
    for r in metric:
        texts = [t for _, t, _ in r.cells[1:]]
        # A PENDING row carries no value and therefore no glyph -- exempting it
        # is not a loosening: check_hud_layout's pending block below asserts
        # separately that such a row says "computing" and never shows a 0.0000.
        pending = any(rg.HUD_PENDING in t for t in texts)
        label = r.cells[0][1]
        is_post = label.endswith("POST") or " POST " in f" {label} "
        if is_post and not pending:
            if not any("<=" in t for t in texts):
                fails.append(f"HUD row {label!r} is a POST row with no bound glyph")
        elif not is_post and any("<=" in t for t in texts):
            fails.append(f"HUD row {label!r} is not a POST row but carries '<=' -- "
                         "the bound glyph must mark POST and only POST")

    # -- readability at every supported panel width, with numbers ---------
    #
    # _draw_hud fits each cell to the gap before the next stop, so an
    # over-long cell is TRUNCATED, not overlapped -- "0.7911<=" arriving as
    # "0.79.." is invisible to any check on heights.
    for panel_w in (rg.PANEL_FLOOR_W, 620, 820, 1024):
        why = rg.hud_columns_clear(panel_w)
        if why:
            fails.append(f"HUD columns not readable: {why}")

    # -- per-model groups -------------------------------------------------
    specs = _legend_specs(rg, 30)
    stats = [rg.HudModelStat(layer_id=sp.layer_id, label=sp.label, verdict=sp.verdict,
                             n_dets=1000 + i, pix_pre=(0.4, 0.7, 0.5),
                             pix_post=(0.45, 0.79, 0.58), geo_pre=(0.40, 0.63),
                             geo_post=tuple((0.47, 0.67) for _ in rg._GEO_RULE_ABBRS))
             for i, sp in enumerate(specs)]
    many = rg.hud_rows(
        frame_idx=0, frame_pos=0, n_frames=1, reviewed=0, n_clusters=1, n_raw=0,
        spatial_threshold=30.0, cluster_radius=30.0, n_layers=30, model_stats=stats)
    body = [r for r in many if r.zone == "body"]
    groups = sorted({r.group for r in body})
    if len(groups) != 30:
        fails.append(f"30 model stats produced {len(groups)} HUD groups")
    for gi in groups:
        grp = [r for r in body if r.group == gi]
        if len(grp) != rg.HUD_GROUP_ROWS:
            fails.append(f"HUD group {gi} has {len(grp)} rows, expected {rg.HUD_GROUP_ROWS}")
        if not grp[0].rule:
            fails.append(f"HUD group {gi} does not start with a section rule")
        # The P/R/F1 heads REPEAT on every group header: heads printed once at
        # the top are heads you have scrolled past.
        if not {t for _, t, _ in grp[0].cells} >= set(rg.HUD_COL_HEADS):
            fails.append(f"HUD group {gi}'s header does not repeat the P/R/F1 heads")
        posts = [r.cells[0][1] for r in grp if r.cells[0][1].startswith("GEO POST")]
        want = [f"GEO POST {a}" for a in rg._GEO_RULE_ABBRS]
        if posts != want:
            fails.append(f"HUD group {gi} geo POST rows are {posts}, expected {want} -- all "
                         "three tie-break rules, most-positive first, always")

    # -- a pending cell must never read as a measurement ------------------
    pend = rg.hud_rows(
        frame_idx=0, frame_pos=0, n_frames=1, reviewed=0, n_clusters=1, n_raw=0,
        spatial_threshold=30.0, cluster_radius=30.0, n_layers=1,
        model_stats=[rg.HudModelStat(layer_id="val_013", label="013", verdict="healthy",
                                     pix_pre=(0.4, 0.7, 0.5), pix_post=(0.45, 0.79, 0.58))])
    pend_body = [r for r in pend if r.zone == "body"]
    if len(pend_body) != rg.HUD_GROUP_ROWS:
        fails.append(f"a pending group draws {len(pend_body)} rows, not {rg.HUD_GROUP_ROWS} -- "
                     "the row count must not change as results land, or the table reflows "
                     "under the cursor")
    geo_rows = [r for r in pend_body if r.cells and r.cells[0][1].startswith("GEO")]
    for r in geo_rows:
        texts = [t for _, t, _ in r.cells[1:]]
        if not any(rg.HUD_PENDING in t for t in texts):
            fails.append(f"pending geo row {r.cells[0][1]!r} does not say it is computing")
        if any("0.0000" in t for t in texts):
            fails.append(f"pending geo row {r.cells[0][1]!r} rendered a 0.0000 -- a value that "
                         "has not been computed must never look like a measurement")

    # -- the scroll view: swept ------------------------------------------
    seen_groups = set()
    for band_h in range(80, 1400, 13):
        for scroll in range(-2, 40):
            v = rg.hud_scroll_view(many, band_h=band_h, scroll=scroll)
            if len(v.pinned) != len(pinned_of(many)) or len(v.footer) != len(footer_of(many)):
                fails.append(f"hud_scroll_view dropped a pinned/footer row at band_h={band_h}")
                return
            if v.scroll < 0 or (v.n_groups and v.scroll >= v.n_groups):
                fails.append(f"hud_scroll_view returned scroll={v.scroll} for "
                             f"{v.n_groups} groups")
                return
            # Whole groups only -- half a group is unreadable.
            for gi in {r.group for r in v.body}:
                if sum(1 for r in v.body if r.group == gi) != \
                        sum(1 for r in many if r.zone == "body" and r.group == gi):
                    fails.append(f"hud_scroll_view sliced group {gi} in half at "
                                 f"band_h={band_h} scroll={scroll}")
                    return
            drawn = (rg.HUD_ADV["top"] + sum(r.advance for r in v.pinned)
                     + sum(r.advance for r in v.body) + sum(r.advance for r in v.footer)
                     + rg.HUD_ADV["descent"])
            if v.body and drawn > band_h:
                fails.append(f"hud_scroll_view drew {drawn}px into a {band_h}px band")
                return
            # Clamping must be idempotent.
            again = rg.hud_scroll_view(many, band_h=band_h, scroll=v.scroll)
            if again.scroll != v.scroll:
                fails.append(f"hud_scroll_view is not idempotent: {v.scroll} -> {again.scroll}")
                return
            if band_h >= 1000:
                seen_groups |= {r.group for r in v.body}
    # Every model must be reachable. Degenerate checkpoints sort LAST, so an
    # unreachable tail would hide exactly the models the 30-checkpoint screen
    # exists to expose.
    if len(seen_groups) != 30:
        fails.append(f"only {len(seen_groups)}/30 model groups are reachable by scrolling")

    print(f"  HUD OK  {len(rows)} rows ({len(pinned)} pinned, {len(footer)} footer), "
          f"30 groups x {rg.HUD_GROUP_ROWS} rows all reachable, columns clear at "
          f"{rg.PANEL_FLOOR_W}px, 3 tie-break rules per model")


def pinned_of(rows):
    return [r for r in rows if r.zone == "pinned"]


def footer_of(rows):
    return [r for r in rows if r.zone == "footer"]


def check_layer_palette(rg, fails: list) -> None:
    """The per-layer palette must not speak the judgment colour language.

    review_gui's COLOR dict is entirely semantic -- green IS confirmed_real --
    so a layer marker that lands near one of those hues turns every screenshot
    of this tool into a misreading. The module-scope assertions already gate
    this at import; this re-checks them with the numbers printed, and adds the
    property no assertion can express: STABILITY. A checkpoint that finishes
    mid-session and appears as an extra layer must not recolour the layers
    already on screen, or the reviewer's mental map silently rewires.
    """
    import itertools
    pal30 = rg.layer_palette(30)
    if len(set(pal30)) != 30:
        fails.append(f"layer_palette(30) has {len(set(pal30))} distinct colours, expected 30")

    de_sem = rg._min_delta_e_vs_COLOR(pal30)
    if de_sem < rg.LAYER_PALETTE_MIN_DE:
        fails.append(f"layer colour within {de_sem:.1f} deltaE of a semantic COLOR entry "
                     f"(gate {rg.LAYER_PALETTE_MIN_DE})")
    de_shape = rg._min_same_shape_delta_e(30)
    if de_shape < rg.LAYER_PAIR_MIN_DE:
        fails.append(f"two same-shape layers only {de_shape:.1f} deltaE apart "
                     f"(gate {rg.LAYER_PAIR_MIN_DE})")

    # Prefix stability, which is what makes layer_style() stable under growth.
    for n in (8, 11, 12, 20):
        if rg.layer_palette(n) != pal30[:n]:
            fails.append(f"layer_palette({n}) is not the prefix of layer_palette(30) -- "
                         "adding a layer would recolour the existing ones")
            break

    ids11 = [f"val_{i:03d}" for i in (13, 14, 15, 17, 21, 28, 30, 31, 42, 44, 45)]
    before = {i: rg.layer_style(i, ids11).color for i in ids11}
    after = {i: rg.layer_style(i, ids11 + ["val_099"]).color for i in ids11}
    moved = [k for k in before if before[k] != after[k]]
    if moved:
        fails.append(f"adding a 12th layer recoloured {moved}")

    # Health is a modifier, never a hue: same layer, three verdicts, one colour.
    cols = {rg.layer_style("val_014", ids11, v).color for v in ("healthy", "sparse", "collapsed")}
    if len(cols) != 1:
        fails.append("health changed a layer's hue -- it must only change hollow/strike")
    if not rg.layer_style("val_014", ids11, "collapsed").strike:
        fails.append("a collapsed layer must draw struck through")
    if not rg.layer_style("val_014", ids11, "sparse").hollow:
        fails.append("a sparse layer must draw hollow")
    if rg.layer_style("val_014", ids11, "healthy").hollow:
        fails.append("a healthy layer must draw filled")

    cvd = rg.layer_cvd_safe_count()
    print(f"  layers OK  palette 30: deltaE>={de_sem:.1f} vs semantic, >={de_shape:.1f} same-shape; "
          f"{cvd}/{len(rg.LAYER_HUES_CVD)} Okabe-Ito anchors survived; prefix-stable")


def _legend_specs(rg, n: int, degenerate_every: int = 3):
    """n synthetic LayerSpecs, deliberately mixing health verdicts."""
    out = []
    for i in range(n):
        v = "healthy"
        if i % degenerate_every == 1:
            v = "collapsed"
        elif i % degenerate_every == 2:
            v = "sparse"
        out.append(rg.LayerSpec(layer_id=f"val_{i:03d}", path=Path(f"val_{i:03d}.txt"),
                                ckpt=f"{i:03d}", thresh=None, label=f"{i:03d}", verdict=v))
    return out


def check_model_window(rg, fails: list) -> None:
    """The model window's geometry and hit boxes.

    These are the two load-bearing properties of a clickable widget nobody can
    test by clicking: NO TWO HIT BOXES OVERLAP, and every row round-trips
    through layer_hit_test back to the tag it was drawn for. Plus the one the
    30-checkpoint screen needs -- EVERY LAYER IS REACHABLE, degenerate ones
    included, since those sort last.

    Replaces check_layer_widget_geometry AND check_model_table. What is no
    longer asserted here: the metric columns and the focused model's extra PRE
    row. Those left this widget for the HUD, where check_hud_layout tests them
    on the shared HUD_COL_X grid. This is a toggle control now.
    """
    specs = _legend_specs(rg, 30)
    ids = {sp.layer_id for sp in specs}
    order = rg.model_display_order(specs)

    # The window is sized to its content, and the point of the separate window
    # is that this is enough for every checkpoint at once.
    w, h = rg.model_window_size(30)
    if w != rg.MODEL_WIN_W:
        fails.append(f"model_window_size width {w} != MODEL_WIN_W {rg.MODEL_WIN_W}")
    rows = rg.model_window_rows(layers=specs, order=order, visible=set(ids),
                                width=w - 2 * rg.MODEL_WIN_PAD,
                                max_h=h - 2 * rg.MODEL_WIN_PAD)
    shown = {r.tag for r in rows if r.kind == "layer"}
    if shown != ids:
        fails.append(f"the model window sized for 30 layers shows {len(shown)} of {len(ids)} "
                     "without scrolling -- that is the whole reason it is its own window")

    checked = 0
    seen = set()
    for max_h in (120, 200, 400, 592, 900):
        for sc in range(0, 32, 3):
            for foc in (None, sorted(ids)[0], sorted(ids)[-1]):
                rr = rg.model_window_rows(layers=specs, order=order, visible=set(ids),
                                          focus=foc, scroll=sc,
                                          width=rg.MODEL_WIN_W - 2 * rg.MODEL_WIN_PAD,
                                          max_h=max_h)
                checked += 1
                boxes = [(r.kind, r.hit) for r in rr if r.action and r.enabled]
                for i, (k1, b1) in enumerate(boxes):
                    for k2, b2 in boxes[i + 1:]:
                        if k1 != k2:
                            continue
                        if not (b1[2] <= b2[0] or b2[2] <= b1[0]
                                or b1[3] <= b2[1] or b2[3] <= b1[1]):
                            fails.append(f"model window: two {k1} hit boxes overlap at "
                                         f"max_h={max_h} scroll={sc}: {b1} {b2}")
                            return
                for r in [x for x in rr if x.kind == "layer"]:
                    cx = (r.hit[0] + r.hit[2]) // 2
                    cy = (r.hit[1] + r.hit[3]) // 2
                    got = rg.layer_hit_test(rr, cx, cy)
                    if got is None or got.tag != r.tag:
                        fails.append(f"model window: the centre of {r.tag}'s row resolves to "
                                     f"{got.tag if got else None}")
                        return
                    if r.y + r.h > max_h:
                        fails.append(f"model window: {r.tag} draws past max_h={max_h}")
                        return
                seen |= {r.tag for r in rr if r.kind == "layer"}
    if seen != ids:
        fails.append(f"only {len(seen)}/{len(ids)} layers are reachable by scrolling -- "
                     "degenerate checkpoints sort last and must not be stranded")

    # An OFF layer must still be clickable, or it can never be turned back on.
    off = rg.model_window_rows(layers=specs, order=order, visible=set(),
                               width=rg.MODEL_WIN_W - 2 * rg.MODEL_WIN_PAD, max_h=900)
    for r in [x for x in off if x.kind == "layer"]:
        if not r.action or not r.enabled:
            fails.append(f"model window: {r.tag} is off and therefore unclickable")
            break

    # Degenerate checkpoints are shown, and SPELLED OUT rather than badged with
    # a single glyph -- 19 of 30 checkpoints being degenerate is the most
    # important structural finding in the screen and a tilde was carrying it.
    deg = [sp.layer_id for sp in specs if sp.degenerate]
    all_rows = rg.model_window_rows(layers=specs, order=order, visible=set(ids),
                                    width=rg.MODEL_WIN_W - 2 * rg.MODEL_WIN_PAD, max_h=900)
    for lid in deg:
        row = next((r for r in all_rows if r.tag == lid), None)
        if row is None:
            fails.append(f"degenerate checkpoint {lid} is not shown at all")
            continue
        text = " ".join(t for _, t, _ in row.cells)
        if "COLLAPSED" not in text and "SPARSE" not in text:
            fails.append(f"degenerate checkpoint {lid} is not labelled: {text!r}")

    # THE ORDERING REGRESSION TEST. With a non-identity display order the
    # on-screen sequence must BE that order -- this is the defect where the
    # table drew a sorted list while [ and ] walked raw disk order.
    rev = list(reversed(order))
    rr = rg.model_window_rows(layers=specs, order=rev, visible=set(ids),
                              width=rg.MODEL_WIN_W - 2 * rg.MODEL_WIN_PAD, max_h=900)
    drawn = [r.tag for r in rr if r.kind == "layer"]
    if drawn != rev:
        fails.append(f"the model window drew {drawn[:4]}... for order {rev[:4]}... -- the "
                     "screen order must BE the presentation sequence")

    # ...and LayerState must walk that same sequence, not self.ids.
    ls = rg.LayerState(specs)
    ls.set_order(rev)
    ls.focus = rev[0]
    walked = [ls.focus]
    for _ in range(4):
        walked.append(ls.move_focus(1, shown=10))
    if walked != rev[:5]:
        fails.append(f"[/] walked {walked} but the screen shows {rev[:5]} -- focus and "
                     "display order have come apart again")

    # set_order must REFUSE anything that is not a permutation: `ids` is the
    # index space for render_key()'s bitmask and the per-layer count tuples.
    try:
        ls.set_order(rev[:-1])
        fails.append("LayerState.set_order accepted a short order -- ids is an index space "
                     "and dropping one silently re-keys the composite cache")
    except AssertionError:
        pass

    print(f"  models OK  window {w}x{h} fits all 30; {checked} layouts, hit boxes disjoint "
          f"and round-tripping, every layer reachable, focus follows the display order")


# Hand-written on purpose, and NOT derived from review_gui: a list generated
# from the source would agree with a typo. `e` is the model window now (it was
# the in-panel legend's expand/collapse, a verb that ceased to exist when the
# table left the panel), and `,`/`.` scroll the HUD's per-model table.
LAYER_KEYS = ("l", "e", "i", ";", "'", "[", "]", "0", ",", ".")
# Keys checked for reachability with their own observable state, because the
# generic layer-state tuple cannot see them. [o]/[m] are the propagator's, and
# [p]/[x] were previously exercised by setting a flag or calling the handler --
# which passes even while the KEY is unreachable, the exact failure the
# no-shifted-keys rule exists to catch. Each entry is (key, reader).
EXTRA_KEYS = (
    ("o", lambda app: app.propagation_rule),
    ("m", lambda app: app.model_overlay_mode),
    ("p", lambda app: app.show_stats),
)


def check_keybinding_docs(rg, fails: list) -> None:
    """Every layer key must appear in all three places keys are documented:
    the always-visible panel strip, the [?] overlay, and REVIEW_GUI.md.

    Three hand-maintained copies is a standing hazard in this file; this makes
    drift a test failure instead of a thing someone notices six months later.
    """
    app = rg.App.__new__(rg.App)
    a = rg.build_arg_parser().parse_args(["--dry-run"])
    for k, v in rg.PARAM_FALLBACKS.items():
        setattr(a, k, v)
    a.run_context = None
    app.args = a
    all_lines = rg.App._help_lines(app)
    help_text = "\n".join(all_lines)
    quick = "\n".join(all_lines[:rg.PANEL_HELP_QUICK_LINES])
    md_path = Path(rg.__file__).resolve().parent / "REVIEW_GUI.md"
    md = md_path.read_text() if md_path.exists() else ""

    for key in LAYER_KEYS:
        # The key must be reachable in [?] WITHOUT SCROLLING: a key buried at
        # line 140 of 160 is documented in name only.
        if f"[{key}]" not in quick:
            fails.append(f"[{key}] is not on the [?] overlay's first page "
                         f"(first {rg.PANEL_HELP_QUICK_LINES} lines) -- it is unreachable "
                         f"without scrolling")
        if key not in help_text:
            fails.append(f"the [?] help overlay does not mention {key!r}")
        # The table pairs related keys on one row ("| `[` / `]` |", as it
        # already does for "| `1` / `2` |"), so require the key as an inline
        # code span in a table row rather than at the start of one.
        if md and not re.search(r"^\|[^\n]*`" + re.escape(key) + r"`", md, re.M):
            fails.append(f"REVIEW_GUI.md's keybindings table has no row mentioning `{key}`")

    # The hint is the only always-visible help affordance, so it must actually
    # be one line, actually point at [?], and actually fit the narrowest panel
    # unfitted -- a truncated promise is worse than no promise.
    if "\n" in rg.PANEL_HINT_LINE:
        fails.append("PANEL_HINT_LINE is more than one line")
    if "[?]" not in rg.PANEL_HINT_LINE:
        fails.append("PANEL_HINT_LINE does not mention [?]")
    w = cv2.getTextSize(rg.PANEL_HINT_LINE, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    if w > 420 - 28:
        fails.append(f"PANEL_HINT_LINE is {w}px, too wide for the 420px floor panel")
    print(f"  layers OK  {len(LAYER_KEYS)} keys on [?]'s first page and in REVIEW_GUI.md; "
          f"hint {w}px")


def check_keys_reachable(app, rg, fails: list) -> None:
    """A documented key that on_key does not claim is worse than no key.

    Asserted through app.on_key(ord(k)) rather than by calling the handler
    directly: a test that calls _layer_key("e") passes while [e] is
    unreachable, for example when a shifted binding falls through to the 3D
    view's lowercase key. See the no-shifted-keys rule at the top of on_key.
    """
    for k in LAYER_KEYS:
        before = (app._hud_scroll, app.model_win, app.layers.accum_mode,
                  frozenset(app.layers.visible), app.layers.focus,
                  app.layers.enabled, app.status_msg)
        try:
            alive = app.on_key(ord(k))
        except Exception as e:  # noqa: BLE001
            fails.append(f"on_key({k!r}) raised {type(e).__name__}: {e}")
            continue
        if not alive:
            fails.append(f"on_key({k!r}) asked the session to quit")
        after = (app._hud_scroll, app.model_win, app.layers.accum_mode,
                 frozenset(app.layers.visible), app.layers.focus,
                 app.layers.enabled, app.status_msg)
        if after == before:
            fails.append(f"[{k}] changed nothing observable -- the binding is documented "
                         f"but on_key does not claim it")
    # The keys whose effect the tuple above cannot see, each read through its
    # own state. [x] is deliberately NOT pressed here: under --dry-run it is a
    # no-op by design, so it has no observable state to assert -- its
    # reachability is covered by check_pdf_report driving on_key directly.
    for k, read in EXTRA_KEYS:
        before = read(app)
        try:
            alive = app.on_key(ord(k))
        except Exception as e:  # noqa: BLE001
            fails.append(f"on_key({k!r}) raised {type(e).__name__}: {e}")
            continue
        if not alive:
            fails.append(f"on_key({k!r}) asked the session to quit")
        if read(app) == before:
            fails.append(f"[{k}] changed nothing observable -- the binding is documented "
                         f"but on_key does not claim it")
    if app.show_stats:          # [p] left the overlay open; close it again
        app.on_key(27)

    print(f"  keys OK  {len(LAYER_KEYS) + len(EXTRA_KEYS)} bindings reachable through "
          f"on_key")



def check_info_line_cap(app, rg, fails: list) -> None:
    """The selected-point info block must fit its layout band.

    panel_layout caps the info band at PANEL_INFO_MAX_LINES and _draw_panel
    slices to layout.info_rows, so an extra line added later is SILENTLY
    TRUNCATED rather than drawn over the HUD -- which is the failure this whole
    rewrite exists to remove. Silent truncation is still a loss, so the cap is
    asserted here: a ninth line has to fail loudly and be dealt with, not
    vanish."""
    dr = app.dr
    worst = 0
    seen = 0
    for frame_idx, centroids in list(dr.unmatched_clusters_by_frame.items())[:40]:
        for c in centroids[:4]:
            lines = app._selected_point_info_lines(dr, c, False)
            worst = max(worst, len(lines))
            seen += 1
    spawned = next((p for v in dr.spawned_by_frame.values() for p in v), None)
    if spawned is not None:
        worst = max(worst, len(app._selected_point_info_lines(dr, spawned, True)))
        seen += 1
    if worst > rg.PANEL_INFO_MAX_LINES:
        fails.append(f"_selected_point_info_lines returns up to {worst} lines but the panel "
                     f"band caps at PANEL_INFO_MAX_LINES={rg.PANEL_INFO_MAX_LINES} -- the extra "
                     f"lines are silently truncated")
    print(f"  panel OK  selected-point info peaks at {worst} lines over {seen} points "
          f"(cap {rg.PANEL_INFO_MAX_LINES})")


def check_layer_status_line(rg, fails: list) -> None:
    """The status bar must keep working, and must not lose the caveat first."""
    segs = rg.status_segments(
        dir_idx=0, n_dirs=3, rel_name="2025_04_16/pieranch_rect", frame_idx=301,
        frame_pos=300, n_frames=2092, frame_rev=2, frame_tot=40, dir_rev=17, dir_tot=71753,
        layers_on=7, layers_total=12, solo=None, accum="per_layer",
        caveat="! no boresight calibration covers 2025-05-10")
    full = rg.fit_status(segs, 1600)
    for must in ("pieranch_rect", "frame 301", "layers 7/12", "accum per_layer",
                 "dir clusters", "boresight"):
        if must not in full:
            fails.append(f"status line lost {must!r} at full width")
    # The narrow case: compute_panel_width floors at 420, so this is the real floor.
    narrow = rg.fit_status(segs, 400)
    if "boresight" not in narrow:
        fails.append("the status line drops the boresight caveat before the counts -- "
                     "a correctness statement outranks a count")
    if narrow.rstrip().endswith(("|", "  ")):
        fails.append(f"status line ends on a dangling separator: {narrow!r}")
    for w in (1600, 900, 600, 420, 300, 120):
        t = rg.fit_status(segs, w)
        got = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        if got > w:
            fails.append(f"status line is {got}px at a {w}px budget")
    print(f"  layers OK  status line degrades by whole segments, caveat retained to {300}px")


def layered_app(rg, base, flight: str, run: str):
    """An App in layered mode, or None when that run has no layers here."""
    args = rg.build_arg_parser().parse_args([flight, run, "--dry-run"])
    args.auto_discovered = False
    args.dirs, run_arg = rg.classify_paths(args.paths, base)
    if run_arg is None:
        return None
    args.run_context = rg.apply_run_context(args, run_arg, base)
    # EXISTENCE IS NOT CREATION. A REAL review session may already have
    # written the layered log dir, so the property the test wants is that the
    # dry run did not CREATE it: snapshot that before the App gets a chance to.
    review = base / run / "review"
    pre_existing = {
        d.resolve() for d in review.glob("*") if d.is_dir()
    } if review.is_dir() else set()
    app = rg.App(args, base, [(base / flight, flight)])
    app._smoke_pre_existing_log_dirs = pre_existing
    app.window_w, app.window_h = 1400, 900
    return app if (app.dr is not None and app.dr.layers) else None


def check_layer_reprojection(rg, app, fails: list) -> None:
    """The accumulation tier's geometry.

    THE ROUND TRIP IS THE REAL TEST. An axis-swap or a pose mix-up would put
    the whole tier in the wrong place, and because the cloud legitimately
    SMEARS, a wrong placement looks exactly like the smear the docs describe
    and gets dismissed. Reprojecting a known pixel and demanding it come back
    is the only check that can tell those apart.
    """
    import numpy as np
    from birdseye_review import pose3d
    from birdseye_review.geo_eval import AXIS_SWAP
    dr = app.dr
    proj = app._ensure_projector()
    if proj is None:
        fails.append("[layers] no projector for a flight that has flight_data.db")
        return
    app.frame_pos = len(app.dr.frames_all) // 2
    fi = app.current_frame_idx()
    row = proj.poses[pose3d.basename_index(proj).get(
        Path(dr.image_path_by_frame[fi]).name, fi)]
    T_WC, radalt = proj.frame_pose(row)
    W, H = proj.res

    src = np.array([[100., 200.], [960., 600.], [1800., 1100.]])
    back = proj.to_2d(proj.to_3d(src, T_WC, radalt), T_WC)
    err = float(np.abs(back - src).max())
    if err > 1e-3:
        fails.append(f"[layers] to_2d(to_3d(x)) is off by {err:.4g}px -- the accumulation "
                     "tier is projecting through the wrong transform")

    clouds = app._ensure_layer_clouds()
    if clouds is None:
        fails.append("[layers] no world cloud built")
        return

    # The cull must be CONSERVATIVE: exactly the brute-force set, not a subset.
    # A cull that quietly drops visible points is silent data loss and would
    # read as "that model fires less than it does".
    xyz = clouds["xyz"]
    P = np.hstack([xyz[:, :3], np.ones((len(xyz), 1))])
    cam = AXIS_SWAP @ (np.linalg.inv(T_WC) @ P.T)
    depth = -cam[2, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = proj.K @ cam[:3, :]
        pr = pr / pr[2]
    pxa = pr[:2].T
    ok = ((depth > pose3d.NEAR_PLANE_M) & np.isfinite(pxa).all(axis=1)
          & (pxa[:, 0] >= 0) & (pxa[:, 0] < W) & (pxa[:, 1] >= 0) & (pxa[:, 1] < H))
    px, lay = app._project_layer_cloud(dr, fi, W, H)
    n_brute, n_cull = int(ok.sum()), (0 if px is None else len(px))
    if n_cull != n_brute:
        fails.append(f"[layers] the footprint cull kept {n_cull} points where brute force "
                     f"keeps {n_brute} -- it must be conservative, not merely fast")

    # Behind the camera must be dropped. geo_eval.to_2d would keep it: it
    # divides by the depth with errstate suppressed and returns a plausible
    # in-frame pixel, which is why this code inlines the projection instead.
    cam_pos = np.array([float(row[0]), float(row[1]), float(row[2])])
    up = cam_pos + np.array([0.0, 0.0, 50.0])       # 50 m ABOVE the camera
    Pb = np.array([[up[0], up[1], up[2], 1.0]])
    cb = AXIS_SWAP @ (np.linalg.inv(T_WC) @ Pb.T)
    if -cb[2, 0] > pose3d.NEAR_PLANE_M:
        fails.append("[layers] the behind-camera probe is not actually behind the camera; "
                     "the depth-guard check proves nothing")
    naive = proj.to_2d(up.reshape(1, 3), T_WC)[0]
    if 0 <= naive[0] < W and 0 <= naive[1] < H:
        print(f"  [layers] note: a behind-camera point projects to an in-frame "
              f"({naive[0]:.0f},{naive[1]:.0f}) under bare to_2d -- the depth guard is "
              f"load-bearing, not defensive")

    print(f"  layers OK  reprojection: round trip {err:.2e}px, cull conservative "
          f"({n_cull} == {n_brute}), depth guard active, cloud {clouds['n']} pts in "
          f"{clouds['built_s']:.2f}s")


def check_layer_hit_boxes(rg, app, fails: list) -> None:
    """The model window's clicks, through the REAL on_model_mouse.

    Routed through the model window's own callback, not on_mouse: the widget
    has its own OpenCV window, whose coordinates ARE its canvas coordinates
    because the window is AUTOSIZE. That removes the rect predicate and the
    canvas rescale from the click path entirely, so a click cannot land on a
    row the reviewer cannot see.

    Asserted: toggle, toggle back, solo, un-solo, NONE, ALL.
    """
    import cv2
    app.frame_pos = len(app.dr.frames_all) // 2
    app.render()
    app.open_model_window()
    rows = app._model_rows(app.dr, app.current_frame_idx())
    lrows = [r for r in rows if r.kind == "layer"]
    if not lrows:
        fails.append("[layers] the model window drew no layer rows")
        return

    def click(row, right=False):
        # +MODEL_WIN_PAD: the callback subtracts the pad, so a test that
        # forgot it would silently probe one row up.
        cx = rg.MODEL_WIN_PAD + (row.hit[0] + row.hit[2]) // 2
        cy = rg.MODEL_WIN_PAD + (row.hit[1] + row.hit[3]) // 2
        app.on_model_mouse(cv2.EVENT_RBUTTONDOWN if right else cv2.EVENT_LBUTTONDOWN,
                           cx, cy, 0, None)

    r = lrows[min(2, len(lrows) - 1)]
    was = r.tag in app.layers.visible
    click(r)
    if (r.tag in app.layers.visible) == was:
        fails.append(f"[layers] left-click on {r.tag} did not toggle it")
    click(r)
    if (r.tag in app.layers.visible) != was:
        fails.append(f"[layers] {r.tag} did not toggle back -- an OFF layer that cannot be "
                     "re-enabled is unusable")
    click(r, right=True)
    if app.layers.solo != r.tag or len(app.layers.visible) != 1:
        fails.append(f"[layers] right-click did not solo {r.tag}")
    click(r, right=True)
    if app.layers.solo is not None or len(app.layers.visible) != len(app.layers.ids):
        fails.append("[layers] un-solo did not restore the previous visibility set")

    for act, check in (("none", lambda: len(app.layers.visible) == 0),
                       ("all", lambda: len(app.layers.visible) == len(app.layers.ids))):
        # Recompute: the rows reflow as state changes, exactly as they do for a
        # real user between clicks, so stale coordinates would test nothing.
        rows = app._model_rows(app.dr, app.current_frame_idx())
        b = next((x for x in rows if x.action == act), None)
        if b is None:
            fails.append(f"[layers] no [{act.upper()}] button in the model window")
            continue
        click(b)
        if not check():
            fails.append(f"[layers] the [{act.upper()}] button did nothing")

    # [SORT] must actually sort -- a sort key read by the display order but
    # written by nothing would leave the footer printing the arm as though it
    # were selectable.
    rows = app._model_rows(app.dr, app.current_frame_idx())
    b = next((x for x in rows if x.action == "sort"), None)
    if b is None:
        fails.append("[layers] no [SORT] button in the model window")
    else:
        before = app.layers_sort
        click(b)
        if app.layers_sort == before:
            fails.append("[layers] [SORT] did not change the sort key")
        if app.layers_sort not in rg.MODEL_SORT_CYCLE:
            fails.append(f"[layers] [SORT] produced {app.layers_sort!r}, not in "
                         f"MODEL_SORT_CYCLE")

    # A click delivered while the window is shut must be harmless, and must
    # not act on a stale row list.
    app.close_model_window()
    snapshot = frozenset(app.layers.visible)
    click(r)
    if frozenset(app.layers.visible) != snapshot:
        fails.append("[layers] a click was applied while the model window was closed")

    # The side panel must be inert everywhere, while the frame view selects.
    # A stale branch swallowing panel clicks is the risk.
    img = app._load_frame_image(app.dr, app.current_frame_idx())
    geom = rg.DisplayGeometry(img.shape[1], img.shape[0], app.window_w, app.window_h)
    app.selected_key = None
    for sy in (40, 300, 600):
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, geom.disp_w + 20, sy, 0, None)
        if app.selected_key is not None:
            fails.append(f"[layers] a side-panel click at y={sy} selected something -- the "
                         "panel is inert now that the legend has its own window")
            break
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, geom.disp_w - 2, geom.disp_h // 2, 0, None)
    print(f"  layers OK  hit boxes: toggle/solo/ALL/NONE/SORT routed through on_model_mouse, "
          f"closed-window clicks inert, frame view intact, side panel inert")


def check_layer_3d(rg, app, fails: list) -> None:
    """Per-model layers in the [d] cloud, and the cache property that makes
    toggling them cheap."""
    import numpy as np
    if not app._ensure_pose_scene():
        fails.append("[layers] 3D scene would not open for a flight with pose data")
        return
    app._build_pose_cloud()
    sc = app.pose_view.scene
    if not len(getattr(sc, "cloud_layer", [])):
        fails.append("[layers] the 3D cloud carries no layer ids")
        return
    # SLOTS ARE 0-BASED: max() is the LAST layer index and the count is max()+1.
    n_slots = int(sc.cloud_layer.max()) + 1
    if int(sc.cloud_layer.min()) != 0:
        fails.append(f"[layers] the 3D cloud's lowest layer slot is "
                     f"{int(sc.cloud_layer.min())}, expected 0")
    if n_slots != len(app.dr.layers):
        fails.append(f"[layers] the 3D cloud has {n_slots} layer slot(s) for "
                     f"{len(app.dr.layers)} layers")
    sc.show_cloud = True
    before_key = getattr(sc, "_cloud_proj_key", None)
    a = sc.render(900, 700)
    app.layers.visible = set(app.layers.ids[:1])
    app._pose_layer_mask = None
    app._sync_pose_layer_visibility()
    b = sc.render(900, 700)
    after_key = getattr(sc, "_cloud_proj_key", None)
    if int((a != b).any(axis=2).sum()) == 0:
        fails.append("[layers] hiding 11 of 12 layers changed nothing in the 3D cloud")
    if before_key is not None and before_key != after_key:
        fails.append("[layers] a visibility toggle invalidated the cloud projection cache -- "
                     "the mask must be applied after projection, not by slicing the points")
    # Every point carries a fixed palette colour, so there is no colour
    # rebuild to trigger. What matters is that the visibility mask is exactly
    # one entry per layer: an off-by-one here would silently show the wrong
    # checkpoint.
    if len(app._pose_layer_mask or ()) != len(app.layers.ids):
        fails.append(f"[layers] the 3D visibility mask has "
                     f"{len(app._pose_layer_mask or ())} entries for "
                     f"{len(app.layers.ids)} layers")
    app.layers.all_on()
    app._pose_layer_mask = None
    app._sync_pose_layer_visibility()
    print(f"  layers OK  3D: {n_slots} layer slots from 0, toggle changes pixels, "
          f"projection cache untouched, mask is one entry per layer")


def check_layer_poll_wired(rg, app, fails: list) -> None:
    """render() must notice a checkpoint that finishes mid-session.

    If `_poll_layer_dir` is not called from the render loop, [l]'s "N NEW
    LAYER(S) on disk" notice is permanently dead and nothing fails; the
    feature just stops. So the wiring is asserted, not assumed, along with
    the rate-limit and mid-recompute guards.
    """
    app.render()
    if not app._layer_sig:
        fails.append("[l] render() left _layer_sig empty -- _poll_layer_dir is not wired "
                     "into the render loop, so a checkpoint finishing mid-session will "
                     "never be noticed")
        return
    # Rate limited: a directory listing on every ~25ms tick is real syscall
    # traffic for a file that appears at most once every few minutes.
    app._layer_last_poll = 0.0
    app._poll_layer_dir()
    first = app._layer_last_poll
    app._poll_layer_dir()
    if first <= 0.0:
        fails.append("[l] _poll_layer_dir did not stamp its rate-limit clock")
    elif app._layer_last_poll != first:
        fails.append(f"[l] _poll_layer_dir is not rate limited "
                     f"(LAYER_POLL_SECONDS={rg.LAYER_POLL_SECONDS})")
    # And never mid-recompute: a layer notice arriving mid-re-cluster reads as
    # being about the wrong clustering.
    app._layer_last_poll = 0.0
    saved, app._recompute_gen = app._recompute_gen, iter(())
    try:
        app._poll_layer_dir()
        if app._layer_last_poll != 0.0:
            fails.append("[l] _poll_layer_dir ran during a recompute")
    finally:
        app._recompute_gen = saved
    print(f"  [l] OK  dets/ polled from render ({len(app._layer_sig)} file(s) in the "
          f"signature), rate limited, inert mid-recompute")


def check_layer_union_equivalence(rg, app, fails: list) -> None:
    """THE correctness claim behind the one-pass design.

    Every per-layer number on screen is split out of a SINGLE match of the
    union against labels.txt. That is only legitimate because a detection's
    matched/unmatched status does not depend on the other detections in the
    list, and because the per-layer ground-truth mask is collected inside the
    same distance test. If either stops being true, every per-model precision
    and recall this tool reports is quietly wrong -- so it is checked against
    matching each layer on its own, on the densest frames available.
    """
    from birdseye_review.detections import match_spatial_many_to_one as m1
    dr = app.dr
    n = len(dr.layers)
    T = app.args.spatial_threshold
    labels = dr.labels_fd
    cand = sorted(dr.dets_fd, key=lambda f: -len(dr.dets_fd[f].points))[:5]
    cand += sorted((f for f in dr.dets_fd if labels.get(f) and labels[f].points),
                   key=lambda f: -len(labels[f].points))[:5]
    cand = sorted(set(cand))

    um, uu, uf = [0] * n, [0] * n, [0] * n
    sm, su, sf = [0] * n, [0] * n, [0] * n
    for f in cand:
        pts = dr.dets_fd[f].points
        gt = labels[f].points if f in labels else []
        dl = [getattr(p, "layer", -1) for p in pts]
        ma, ui, _ug, mask = rg.match_spatial_many_to_one_layered(pts, gt, T, dl, n)
        for i in {d for d, _, _ in ma}:
            um[dl[i]] += 1
        for i in ui:
            uu[dl[i]] += 1
        for j in range(len(gt)):
            for k in range(n):
                if not (mask[j] >> k) & 1:
                    uf[k] += 1
        for k in range(n):
            sub = [p for p in pts if getattr(p, "layer", -1) == k]
            a, b, c = m1(sub, gt, T)
            sm[k] += len({d for d, _, _ in a}); su[k] += len(b); sf[k] += len(c)

    for name, u, so in (("matched", um, sm), ("unmatched", uu, su), ("false-neg", uf, sf)):
        if u != so:
            bad = [(i, u[i], so[i]) for i in range(n) if u[i] != so[i]][:3]
            fails.append(f"[layers] union-derived {name} != per-layer matching {bad} -- "
                         "the one-pass split is not exact")
    print(f"  layers OK  union split == per-layer matching over {len(cand)} frames "
          f"(max {max(len(dr.dets_fd[f].points) for f in cand)} pts/frame), {n} layers")


def check_layer_metrics(rg, app, fails: list) -> None:
    """Adjudicate once, score N -- and the conventions that must survive it."""
    import copy
    dr = app.dr
    rows = rg.layer_metric_rows(dr, app.layers.visible)
    if len(rows) != len(dr.layers):
        fails.append(f"[layers] {len(rows)} metric rows for {len(dr.layers)} layers")
        return
    # With nothing adjudicated, POST is PRE. If these differ, corrected_prf is
    # being fed something it should not be.
    for r in rows:
        if any(abs(r.pre[i] - r.post[i]) > 1e-9 for i in range(3)):
            fails.append(f"[layers] {r.layer_id}: POST != PRE with no adjudications "
                         f"({r.pre} vs {r.post})")
            break
    if any(r.n_confirmed or r.n_confirmed_elsewhere for r in rows):
        fails.append("[layers] non-zero confirmed counts before any adjudication")

    # One judgment must move exactly the layers that fired, and book a
    # discoverable miss against the ones that did not.
    target = None
    for f, cs in dr.unmatched_clusters_by_frame.items():
        for c in cs:
            k = (c.frame_idx, c.x, c.y)
            cnt = dr.cluster_layer_counts.get(k)
            if cnt and 2 <= sum(1 for v in cnt if v) < len(dr.layers):
                target = (k, cnt)
                break
        if target:
            break
    if target is None:
        print("  layers -- no mixed-agreement cluster to test scoring with")
        return
    k, cnt = target
    before = {r.layer_id: r for r in rows}
    dr.store.commit(k[0], k[1], k[2], 0.5, "confirmed_real", "x.png", extra={"source": "human"})
    after = {r.layer_id: r for r in rg.layer_metric_rows(dr, app.layers.visible)}
    fired = [dr.layers[i].layer_id for i, v in enumerate(cnt) if v]
    silent = [dr.layers[i].layer_id for i, v in enumerate(cnt) if not v]
    for lid in fired:
        if after[lid].post[0] <= before[lid].post[0]:
            fails.append(f"[layers] {lid} fired on a confirmed cluster but its POST precision "
                         "did not rise")
        if after[lid].n_confirmed_elsewhere:
            fails.append(f"[layers] {lid} fired there but was charged a discoverable miss")
    for lid in silent:
        if after[lid].n_confirmed_elsewhere != 1:
            fails.append(f"[layers] {lid} was silent on a confirmed cluster but "
                         f"n_confirmed_elsewhere is {after[lid].n_confirmed_elsewhere}")
        if after[lid].post != before[lid].post:
            fails.append(f"[layers] {lid} was silent yet its score moved -- the conventional "
                         "bound must not absorb another layer's confirmation")
    # POST recall must remain a BOUND, never a point estimate.
    if any(r.post[1] < 0 or r.post[1] > 1 for r in after.values()):
        fails.append("[layers] a POST recall bound left [0,1]")
    dr.store.commit(k[0], k[1], k[2], 0.5, "unreviewed", "x.png", extra={"source": "human"})
    print(f"  layers OK  one adjudication scored {len(fired)}/{len(dr.layers)} models and "
          f"booked {len(silent)} discoverable miss(es)")


def check_layer_geo(rg, app, fails: list) -> None:
    """The PER-LAYER geospatial arm: one scaffold per checkpoint, scored
    through the same geo_prf as the union.

    The valuable assertions are the invariants that follow from the geometry,
    not the absolute numbers.
    """
    import numpy as np
    try:
        from birdseye_review import geo_metrics as gm
    except Exception as e:  # noqa: BLE001
        print(f"  [geo/layers] SKIP  geo_metrics unimportable ({e})")
        return
    if not await_geo(app):
        pub = app._geo_published()
        print(f"  [geo/layers] SKIP  worker did not finish ({pub.get('error')})")
        return
    per = app._layer_geo_summaries()
    live = {lid: d for lid, d in per.items() if d.get("available")}
    if len(live) < 2:
        fails.append(f"[geo/layers] only {len(live)} layers scored -- expected one per "
                     f"checkpoint")
        return
    union = layer_summary(app)

    # 1. THE RECALL DENOMINATOR IS SHARED. Geotags come from proj.clicks and
    #    in-view from geotags_in_view(proj.poses) -- both functions of pose
    #    data alone, so every layer scores against the same denominator and
    #    only n_hit moves. That is exactly the comparison this feature exists
    #    to make, and it is exact rather than approximate.
    dens = {(d["recall"]["n_geotags"], d["recall"]["n_in_view"]) for d in live.values()}
    if len(dens) != 1:
        fails.append(f"[geo/layers] layers disagree on the recall denominator: {dens} -- it is "
                     "a property of the pose data and cannot vary by model")
    if union and union.get("available"):
        u = (union["recall"]["n_geotags"], union["recall"]["n_in_view"])
        if u not in dens:
            fails.append(f"[geo/layers] the union's denominator {u} differs from the "
                         f"layers' {dens}")
        # 2. A layer's matched geotags are a SUBSET of the union's: adding
        #    points can only grow the core set, so no model can hit a geotag
        #    the union misses. Do NOT assert the same for cluster COUNTS --
        #    chaining at eps=0.20m legitimately merges (measured on the full
        #    2,120-frame sample flight: 602 union clusters against 2,890
        #    summed across layers).
        umax = max(d["recall"]["tp_geotags"] for d in live.values())
        if union["recall"]["tp_geotags"] < umax:
            fails.append(f"[geo/layers] a layer hit {umax} geotags but the union only "
                         f"{union['recall']['tp_geotags']} -- impossible")

    # 3. Per rule, per layer: the buckets partition and the bound stays a bound.
    for lid, d in live.items():
        for rule in d["verdict_rules"]:
            q = d["post"][rule]
            if q["tp"] + q["fp"] != d["n_clusters"]:
                fails.append(f"[geo/layers] {lid} POST[{rule}] tp+fp != n_clusters")
            for k in ("recall_all_bound", "recall_in_view_bound"):
                if d["recall"]["post"][rule][k] > 1.0 + 1e-12:
                    fails.append(f"[geo/layers] {lid} {k} = "
                                 f"{d['recall']['post'][rule][k]} exceeds 1")
        # All three rules, always -- never one, never a subset.
        if list(d["verdict_rules"]) != list(gm.VERDICT_RULES):
            fails.append(f"[geo/layers] {lid} reports rules {d['verdict_rules']}, expected "
                         f"all of {gm.VERDICT_RULES} in most-positive-first order")

    # 4. THE MOST VALUABLE ONE: reduce-from-summary == reduce-from-scaffold.
    #    The compact LayerGeoSummary drops points/core_mask/cell_to_point and
    #    replaces locate() with a precomputed join. This pins that it lost
    #    nothing geo_prf needs and that the join reproduces locate() exactly.
    dr = app.dr
    pub = app._geo_published()
    proj = pub.get("proj")
    lid = min(live, key=lambda k: per[k]["n_clusters"])
    sm = pub["layers"][lid]
    li = next(i for i, sp in enumerate(dr.layers) if sp.layer_id == lid)
    frames = {}
    for fi, fd in dr.dets_fd.items():
        pts = [pt for pt in fd.points if getattr(pt, "layer", -1) == li]
        if pts:
            frames[fi] = rg.FrameData(fi, fd.image_path, pts)
    sc = gm.build_scaffold(dr.path, frames, eps=app.args.geo_eps,
                           min_samples=app.args.geo_min_samples,
                           eps_predict=app.args.geo_match_radius,
                           dedup_cm=app.args.geo_dedup_cm, proj=proj)
    if not sc.ok:
        print(f"  [geo/layers] SKIP  reference scaffold for {lid} unavailable")
    else:
        by_index = {i: row for i, row in enumerate(proj.poses)} if proj is not None else {}
        judged = []
        for ckey, dets in dr.member_map().items():
            j = dr.store.get(*ckey)
            if j in gm.ABSTAIN_JUDGMENTS:
                continue
            for d in dets:
                if getattr(d, "layer", -1) == li:
                    judged.append((d.frame_idx, d.x, d.y, j))
        classic = gm.geo_prf(sc, gm.collect_verdicts(sc, proj, judged,
                                                     lambda fi: by_index.get(fi)))
        # Same path the app takes, minus `added` on both sides so the two are
        # comparable term for term.
        v = gm.GeoVerdicts()
        for fi, x, y, j in judged:
            lab = sm.member_label.get((int(fi), float(x), float(y)), -1)
            v.n_judged_keys += 1
            if lab < 0:
                v.n_unresolved += 1
            else:
                v.tally(int(lab), j)
        reduced = gm.geo_prf(rg._ScaffoldView(sm), v)
        for field in ("n_clusters", "n_noise"):
            if classic.get(field) != reduced.get(field):
                fails.append(f"[geo/layers] {lid} {field}: scaffold {classic.get(field)} vs "
                             f"summary {reduced.get(field)}")
        if classic["pre"] != reduced["pre"]:
            fails.append(f"[geo/layers] {lid} PRE differs: {classic['pre']} vs "
                         f"{reduced['pre']} -- the reduced summary lost something geo_prf "
                         f"reads, or the join does not reproduce locate()")
        for rule in gm.VERDICT_RULES:
            if classic["post"][rule] != reduced["post"][rule]:
                fails.append(f"[geo/layers] {lid} POST[{rule}] differs between the live "
                             f"scaffold and the reduced summary")
                break
        if classic["recall"] != reduced["recall"]:
            fails.append(f"[geo/layers] {lid} recall block differs between the live scaffold "
                         f"and the reduced summary")

    hits = {lid: d["recall"]["tp_geotags"] for lid, d in live.items()}
    print(f"  [geo/layers] OK  {len(live)} per-layer arms, shared denominator {dens.pop()}, "
          f"hits {min(hits.values())}-{max(hits.values())}, all 3 rules each; "
          f"{lid} scores identically from the reduced summary and a live scaffold")


def check_layer_store_isolation(rg, app, fails: list) -> None:
    """A layered review must be incapable of touching the flight's own files."""
    dr = app.dr
    st = dr.store
    if not isinstance(st, rg.LayeredJudgmentStore):
        fails.append(f"[layers] store is {type(st).__name__}, not LayeredJudgmentStore")
        return
    flight = dr.path.resolve()
    for pth in [st.log_path] + list(st.export_paths.values()):
        if pth.resolve().parent == flight:
            fails.append(f"[layers] a layered store writes into the FLIGHT directory: {pth}")
    if "review" not in st.log_path.parts:
        fails.append(f"[layers] the layered log is not under <run>/review/: {st.log_path}")
    # geo_metrics.json must obey the SAME separation: writing it to
    # `dr.path / "geo_metrics.json"` would let a layered session overwrite the
    # flight's own file with numbers over a different detection set.
    import inspect
    # Comments stripped: the function's own docstring/comments may QUOTE the
    # flight path to explain the hazard, and matching those would make
    # this assertion fire on its own explanation.
    src = "\n".join(ln.split("#", 1)[0] for ln in
                    inspect.getsource(type(app).write_geo_metrics).splitlines())
    if 'dr.path / "geo_metrics.json"' in src:
        fails.append("[layers] write_geo_metrics still targets the FLIGHT directory -- a "
                     "layered review would clobber the flight's own geo_metrics.json")
    if "log_dir" not in src:
        fails.append("[layers] write_geo_metrics does not route through the store's log_dir")
    if st.schema_version != 2:
        fails.append(f"[layers] layered records carry schema_version {st.schema_version}, want 2")
    if not st.record_extra.get("layer_set"):
        fails.append("[layers] records do not carry the ordered layer_set")
    # --dry-run must leave no trace whatsoever -- meaning it must not CREATE
    # the log dir. A dir a real session wrote earlier is not a dry-run leak.
    pre = getattr(app, "_smoke_pre_existing_log_dirs", set())
    if app.args.dry_run and st.log_dir.exists() and st.log_dir.resolve() not in pre:
        fails.append(f"[layers] --dry-run created {st.log_dir}")
    print(f"  layers OK  judgments isolated to {st.log_path.parent}, schema v{st.schema_version}, "
          f"flight dir untouched")


def check_panel_layout(rg, fails: list) -> None:
    """The side panel's geometry, swept.

    NON-OVERLAP IS THE WHOLE POINT. Computing bands in several places from
    worst-case reserves fails badly: at 1200x700 with a cluster selected the
    zoom-inset budget comes out negative, gets force-clamped, and the
    selected-point text is painted over the HUD box's top border. One pure
    function owns every y, and this sweeps it.

    The panel's geometry does not depend on the checkpoint count. `hud_h` is
    swept rather than pinned at a constant -- the HUD wants to grow, so a
    fixed `hud_h=PANEL_HUD_H` would exercise none of the growth order.
    """
    checked = 0
    hud_wants = (rg.PANEL_HUD_MIN_H, rg.PANEL_HUD_H,
                 rg.hud_chrome_height() + 30 * rg.hud_group_height(), 10_000)
    for canvas_h in list(range(320, 2001, 7)):
        for hud_h in hud_wants:
            for info_lines in (0, 2, 8):
                for sel in (False, True):
                    L = rg.panel_layout(canvas_h=canvas_h, hud_h=hud_h,
                                        inset_want=420, info_lines=info_lines,
                                        has_selection=sel)
                    checked += 1
                    bands = [(n, b) for n, b in L.ordered() if b]
                    for (na, a), (nb, b) in zip(bands, bands[1:]):
                        if a.y1 > b.y0:
                            fails.append(f"panel h={canvas_h} hud={hud_h} info={info_lines} "
                                         f"sel={sel}: {na} ends {a.y1} but {nb} starts {b.y0}")
                            return
                    if bands and bands[0][1].y0 < 0:
                        fails.append(f"panel h={canvas_h}: first band starts above 0")
                        return
                    if L.hint.y1 != canvas_h:
                        fails.append(f"panel h={canvas_h}: hint does not reach the bottom "
                                     f"({L.hint.y1})")
                        return
                    # The bound the info loop never had.
                    if L.info_rows * rg.PANEL_INFO_LINE_H > L.info.h:
                        fails.append(f"panel h={canvas_h}: info_rows={L.info_rows} does not fit "
                                     f"in {L.info.h}px")
                        return
                    if L.info_rows > min(info_lines, rg.PANEL_INFO_MAX_LINES):
                        fails.append(f"panel h={canvas_h}: info_rows={L.info_rows} exceeds the "
                                     f"{info_lines} lines it was given")
                        return
                    # The HUD band is chrome plus a WHOLE number of model
                    # groups, or nothing. A band 10px taller than an exact
                    # number of groups is 10px of nothing.
                    if L.hud.h:
                        extra = L.hud.h - rg.hud_chrome_height()
                        if extra >= 0 and extra % rg.hud_group_height() != 0 \
                                and L.hud.h < min(hud_h, rg.PANEL_HUD_H):
                            fails.append(f"panel h={canvas_h} hud={hud_h}: HUD band "
                                         f"{L.hud.h}px is not chrome + whole groups")
                            return
                        if L.hud_groups != rg.hud_groups_fitting(L.hud.h):
                            fails.append(f"panel h={canvas_h}: hud_groups={L.hud_groups} "
                                         f"disagrees with hud_groups_fitting({L.hud.h})")
                            return
                    # Degradation must follow the declared order, never an ad-hoc one.
                    order = [d for d in rg.PANEL_SHRINK_ORDER if d in L.dropped]
                    if list(L.dropped) != order:
                        fails.append(f"panel h={canvas_h}: dropped {L.dropped}, not in "
                                     f"PANEL_SHRINK_ORDER {rg.PANEL_SHRINK_ORDER}")
                        return
                    if sel and "inset" not in L.dropped and L.inset.h < rg.INSET_FLOOR_H:
                        fails.append(f"panel h={canvas_h} hud={hud_h}: inset {L.inset.h}px is "
                                     f"under INSET_FLOOR_H without being declared dropped")
                        return

    # The guarantees the import-time probes pin, restated where a reader of the
    # test suite will see them.
    ref = rg.panel_layout(canvas_h=1000, hud_h=10_000, inset_want=10_000,
                          info_lines=rg.PANEL_INFO_MAX_LINES, has_selection=True)
    if ref.inset.h < rg.INSET_TARGET_H:
        fails.append(f"reference window: inset {ref.inset.h} < INSET_TARGET_H")
    if ref.hud_groups < 1:
        fails.append("reference window: the HUD cannot show one whole model group")
    # "A tall window shows what it can": with a realistic inset ceiling the
    # HUD takes the tail of the growth order, so a 1440px panel shows several
    # models AND keeps the inset at its target.
    big = rg.panel_layout(canvas_h=1440, hud_h=10_000, inset_want=420,
                          info_lines=6, has_selection=True)
    if big.hud_groups < 4:
        fails.append(f"1440px panel shows only {big.hud_groups} whole model groups -- the "
                     f"point of the layout is that a tall window shows everything it can")
    if big.inset.h < rg.INSET_TARGET_H:
        fails.append(f"1440px panel starves the inset to {big.inset.h}px while growing the HUD")
    print(f"  panel OK  {checked} layouts swept 320-2000px: bands ordered and disjoint, "
          f"HUD snaps to whole groups, degradation follows PANEL_SHRINK_ORDER; "
          f"1440px shows {big.hud_groups} model groups with a {big.inset.h}px inset")


def check_pdf_report(app, rg, fails: list) -> None:
    """[x]: the [p] panel as a document.

    Two things are worth asserting beyond "it built": that the caveats actually
    reach the page (a report whose numbers travel without their caveats is worse
    than no report), and that losing matplotlib costs the reader nothing but the
    figures.
    """
    import tempfile
    try:
        from birdseye_review import review_report as rr
    except Exception as e:  # noqa: BLE001
        fails.append(f"[x] review_report unimportable: {e}")
        return
    ctx = rr.collect_context(app)
    if ctx.directory != app.dr.rel_name:
        fails.append("[x] context did not pick up the directory")
    if not ctx.caveats:
        fails.append("[x] report carries no caveats -- numbers must not travel without them")
    if not any("not comparable" in c.lower() for c in ctx.caveats):
        fails.append("[x] the metric-space-comparability caveat is missing")
    if not any("union" in c.lower() for c in ctx.caveats):
        fails.append("[x] the report does not say it carries NO union scorecard -- a reader "
                     "who expects one would otherwise assume the numbers are the union's")
    # PER-MODEL RESULTS ARE THE REPORT.
    if not ctx.models:
        fails.append("[x] the report context carries no per-model results")
    else:
        m = ctx.models[0]
        for key in ("layer_id", "label", "verdict", "pixel_pre", "pixel_post",
                    "n_dets", "config"):
            if key not in m:
                fails.append(f"[x] per-model entry is missing {key!r}")
    if not any("reprojection log" in c.lower() for c in ctx.caveats):
        fails.append("[x] the reprojection-log provenance caveat is missing")
    if not ctx.params or "geo_eps" not in ctx.params:
        fails.append("[x] run parameters missing from the report context")

    with tempfile.TemporaryDirectory() as td:
        out = rr.build_report(Path(td) / "r.pdf", ctx)
        blob = out.read_bytes()
        if not blob.startswith(b"%PDF"):
            fails.append("[x] output is not a PDF")
        pages = blob.count(b"/Type /Page") - 1
        if pages < 5 or len(blob) < 20_000:
            fails.append(f"[x] report looks truncated: {pages} pages, {len(blob)} bytes")

        # Force the matplotlib import to fail and confirm graceful degradation.
        import builtins
        real_import = builtins.__import__

        def blocked(name, *a, **k):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("blocked by smoke test")
            return real_import(name, *a, **k)

        builtins.__import__ = blocked
        try:
            out2 = rr.build_report(Path(td) / "r2.pdf", ctx)
            blob2 = out2.read_bytes()
        finally:
            builtins.__import__ = real_import
        if not blob2.startswith(b"%PDF"):
            fails.append("[x] report does not build without matplotlib")
        elif len(blob2) >= len(blob):
            fails.append("[x] figure-free report is not smaller -- figures may not have been drawn")
        else:
            print(f"  [x] OK  {pages}p / {len(blob):,}B with figures, "
                  f"{len(blob2):,}B without, {len(ctx.caveats)} caveats")

    # --dry-run must write nothing, and [x] must be REACHABLE -- asserted
    # through on_key, not by calling export_report, because a test that calls
    # the handler passes while the key itself is unclaimed.
    from pathlib import Path as _P
    log_dir = _P(getattr(app.dr.store, "log_dir", app.dr.path))
    # BESIDE review_log.jsonl, not in the flight directory: for a layered
    # review the log lives under <run>/review/<flight>/, and the report used
    # to land in the flight dir where two runs would overwrite each other's.
    for where in (log_dir / "review_report.pdf", app.dr.path / "review_report.pdf"):
        before = where.exists()
        app.status_msg = ""
        app.on_key(ord("x"))
        if where.exists() != before:
            fails.append(f"[x] wrote {where} under --dry-run")
    if "dry-run" not in (app.status_msg or ""):
        fails.append(f"[x] did not say it was suppressed by --dry-run: {app.status_msg!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True,
                    help="the extracted birdseye-review sample (the directory holding "
                         "2025_04_16/ and results/)")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="an adjudicated flight directory")
    ap.add_argument("--pose-dir", default=POSE_DIR,
                    help="a flight directory WITH a flight_data.db, for the [v] 3D-view checks")
    ap.add_argument("--layer-run", type=Path,
                    default=Path("results/17_holdout_pieranch"),
                    help="a run directory with per-checkpoint detection files, for the "
                         "model-layer checks")
    a = ap.parse_args()
    base = a.data_root.resolve()
    for need in (base / a.dir / "labels.txt", base / a.layer_run / "dets"):
        if not need.exists():
            print(f"missing {need}: --data-root must be the extracted birdseye-review sample")
            sys.exit(2)

    # POSITIONAL RUN = THE LAYER RUN. Detections come only from a run's dets/,
    # so the positional pair has to be (flight, run-with-detections).
    argv = [a.dir, str(a.layer_run), "--spatial-threshold", "80", "--dry-run"]
    args = rg.build_arg_parser().parse_args(argv)
    args.auto_discovered = False
    # RESOLVE THE POSITIONALS. Detections come from the run's dets/, so
    # without a run_context every directory reports "no model layers" and
    # App.dr is None.
    args.dirs, run_arg = rg.classify_paths(args.paths, base)
    assert run_arg is not None, f"{a.layer_run} was not recognised as a run directory"
    args.run_context = rg.apply_run_context(args, run_arg, base)
    # apply_run_context may derive a spatial_threshold from the run; the
    # explicit --spatial-threshold 80 above must still win.
    assert args.spatial_threshold == 80, (
        f"an explicit --spatial-threshold lost to the run's own value "
        f"({args.spatial_threshold}) -- the override path is broken")

    app = rg.App(args, base, [(base / a.dir, a.dir)])
    assert app.dr is not None, f"{a.dir} failed to load"
    assert app.dr.layers, f"{a.dir} loaded with no model layers"
    # Normally set from the live cv2 window by _sync_window_size().
    app.window_w, app.window_h = 1400, 900
    fails = []

    # The mirrored detector constants must not drift from run_detector.py's.
    # [p] and the PDF both state how the points on screen were produced, and a
    # stale copy here would make that statement quietly wrong.
    try:
        from birdseye_review import run_detector
        if rg._DETECTOR_DEFAULT_THRESH != run_detector.DEFAULT_THRESH:
            fails.append(f"_DETECTOR_DEFAULT_THRESH {rg._DETECTOR_DEFAULT_THRESH} != "
                         f"run_detector.DEFAULT_THRESH {run_detector.DEFAULT_THRESH}")
        if rg._DETECTOR_MIN_AREA != run_detector.MIN_AREA:
            fails.append(f"_DETECTOR_MIN_AREA {rg._DETECTOR_MIN_AREA} != "
                         f"run_detector.MIN_AREA {run_detector.MIN_AREA}")
    except Exception as e:  # noqa: BLE001 -- run_detector pulls in torch
        print(f"  note: could not cross-check the detector constants ({e})")

    # [m] must draw NOTHING with no rule chosen, and something with one. This
    # is the whole reading discipline of the overlay: an unringed centroid
    # means "nothing propagates here and [a] will not touch it".
    app.propagation_rule = None
    app.model_overlay_mode = True
    app.render()
    if app._last_ring_count:
        fails.append(f"[m] drew {app._last_ring_count} ring(s) with no propagation rule "
                     f"chosen -- nothing may be proposed until [o] picks a convention")
    else:
        print("  [m] OK  draws nothing with no rule chosen")

    txt = [t for t, _ in app._stats_overlay_lines()]
    print(f"  [p] OK  {len(txt)} lines, no files needed")

    check_run_resolution(rg, base, fails)
    check_stats_overlay_order(app, rg, txt, fails)
    app._render_stats_overlay(1400, 900)
    check_hud_layout(app, rg, fails)
    check_panel_layout(rg, fails)
    check_layer_palette(rg, fails)
    check_model_window(rg, fails)
    check_layer_status_line(rg, fails)
    check_info_line_cap(app, rg, fails)
    check_keybinding_docs(rg, fails)
    check_keys_reachable(app, rg, fails)
    check_pdf_report(app, rg, fails)

    # -- model layers. These need a run with per-checkpoint detection files.
    lay = layered_app(rg, base, a.pose_dir, str(a.layer_run))
    if lay is None:
        fails.append(f"layers: {a.layer_run} has no detection files for {a.pose_dir}")
    else:
        print(f"  layers: {len(lay.dr.layers)} layers, {sum(len(f.points) for f in lay.dr.dets_fd.values())} "
              f"detections, {lay.dr.cluster_total()} consensus clusters")
        check_layer_poll_wired(rg, lay, fails)
        check_layer_union_equivalence(rg, lay, fails)
        check_layer_metrics(rg, lay, fails)
        check_layer_reprojection(rg, lay, fails)
        check_layer_hit_boxes(rg, lay, fails)
        check_layer_3d(rg, lay, fails)
        check_layer_geo(rg, lay, fails)
        check_layer_store_isolation(rg, lay, fails)

    # [?] must document the propagator.
    help_lines = app._help_lines()
    for key, what in (("o ", "the propagation rule"), ("m ", "the propagation overlay"),
                      ("a ", "applying propagated labels")):
        if not any(t.strip().startswith(key) for t in help_lines):
            fails.append(f"[?] help has no row for [{key.strip()}] ({what})")
    if not any("propagat" in t for t in help_lines):
        fails.append("[?] help never mentions propagation")
    else:
        print("  [?] OK  help documents [o]/[m]/[a] propagation")
    app._render_help(1400, 900)

    check_inset(app, fails)
    check_propagation_rules(app, fails)
    check_pose3d(base, a.pose_dir, fails)
    check_inline_pose(base, a.pose_dir, fails)
    check_pose_cloud(base, a.pose_dir, fails)
    check_pose_centroids(base, a.pose_dir, fails)
    check_geo_metrics(base, a.pose_dir, fails)
    # These two need the geo worker, so they run on the layered app that has
    # one, after the geo checks have already waited for it.
    if lay is not None:
        check_degenerate_excluded_from_union(lay, fails)
        check_propagated_accept(lay, fails)

    print()
    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        return 1
    print("ALL GUI SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
