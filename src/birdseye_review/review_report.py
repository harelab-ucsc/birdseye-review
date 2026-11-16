# SPDX-License-Identifier: MIT
# Copyright (c) 2026 harelab-ucsc
"""
The [p] panel as a citable document: `review_gui.py`'s [x] export.

WHY THIS EXISTS
---------------
Every number the GUI puts on screen must also land on disk, or it cannot be
cited or regenerated later. `geo_metrics.json` honours that for one metric
space. The rest of what [p] shows -- the run parameters, the input digests, the
cross-space comparison, and above all the CAVEATS -- would otherwise exist only
as pixels in a scrollback nobody keeps.

That is the gap this fills. The output is a self-contained PDF: someone reading
it a year from now, with no access to this machine, can tell what was measured,
against what ground truth, under what parameters, and what not to conclude.

WHAT IT IS NOT
--------------
Not a recomputation. `collect_context()` snapshots values the GUI has already
computed and hands the builder a plain dataclass; the builder does no metric
arithmetic of its own. A report that recomputed its numbers could disagree with
the screen it claims to transcribe, which would make it worse than useless.

DEPENDENCIES
------------
`reportlab` for the document. `matplotlib` is imported LAZILY and its failure is
never fatal -- see `_figure_flowables` -- so the report degrades to tables when
it is unavailable. `review_gui.py` must not gain a module-scope import of either
library.
"""
import hashlib
import io
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PAGE_TITLE = "BirdsEye detection review"

# Colours matching the GUI's own PRE/POST language, so a reader moving between
# the screen and the document is not relearning which arm is which. The GUI is
# BGR (cv2); these are RGB.
PRE_RGB = (0.42, 0.42, 0.78)
POST_RGB = (0.36, 0.72, 0.36)


def _digest(p: Path) -> Optional[str]:
    """sha256 prefix of a file, or None (same algorithm, 16-char truncation as
    the digests in a run_manifest.json, so the two compare directly).
    Input provenance is recorded as SHA-256 file digests."""
    try:
        if not p.exists():
            return None
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


@dataclass
class ReportContext:
    """Everything the report needs, already computed, with no GUI in it.

    Deliberately a plain dataclass rather than a handle on App: the builder must
    be runnable (and testable) without a window, a directory, or a loaded
    dataset, and it must not be able to trigger a recompute as a side effect of
    rendering a page.
    """
    directory: str = "(none)"
    reviewer: str = "(unknown)"
    generated: str = ""
    command: str = ""
    detection_source: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    non_default: Sequence[str] = ()
    run_name: Optional[str] = None
    run_notes: Sequence[str] = ()
    param_overrides: Dict[str, Any] = field(default_factory=dict)
    digests: List[Tuple[str, Optional[str], Optional[int]]] = field(default_factory=list)

    reviewed: int = 0
    n_clusters: int = 0
    n_raw: int = 0

    # No union arms here. The union of every active checkpoint's detections is
    # a review-burden measure, not a result, so this dataclass has no field
    # for a union scorecard. The underlying counts exist on DirectoryReview and
    # drive the sidecar export -- they just have no scorecard.

    geo_cluster_sizes: List[int] = field(default_factory=list)
    geo_params: Dict[str, Any] = field(default_factory=dict)
    # The detector's own heatmap post-processing, carried rather than
    # re-declared here: review_gui already mirrors run_detector.py's values
    # (which are asserted equal in the smoke test), and a third copy in this
    # module would be a third thing to drift.
    detector_thresh: Any = None
    detector_min_area: Any = None

    # PER-MODEL RESULTS. One entry per checkpoint -- which is what one
    # adjudication actually scores. The union is a review-burden measure and
    # nobody's precision, so it is not reported.
    #
    # `models` is ordered for presentation; each entry:
    #   {layer_id, label, verdict, n_dets, n_clusters_fired,
    #    pixel_pre, pixel_post, n_confirmed, n_hard, n_confirmed_elsewhere,
    #    config, geo}   with geo the layer's own geo_prf dict or None.
    models: List[Dict[str, Any]] = field(default_factory=list)

    # The propagator's state and coverage -- see review_gui.propagated_label.
    propagation: Dict[str, Any] = field(default_factory=dict)

    # verdict -> {"judged directly" | "propagated:<rule>" | "legacy classifier": n}
    adjudication: Dict[str, Dict[str, int]] = field(default_factory=dict)
    n_adjudicated: int = 0
    caveats: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Snapshotting a live App
# ---------------------------------------------------------------------------

def collect_context(app) -> ReportContext:
    """Live `review_gui.App` -> ReportContext. Reads, never recomputes.

    Every value here is one the HUD or the [p] overlay is already showing, taken
    from the same memoised accessors, so the document and the screen cannot
    disagree. Kept in this module rather than on App so the GUI file does not
    grow a reporting concern.
    """
    from . import review_gui as rg

    dr = app.dr
    ctx = ReportContext(
        generated=datetime.now().astimezone().isoformat(timespec="seconds"),
        command=" ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        reviewer=getattr(app.args, "reviewer", "(unknown)"),
        detection_source=app.geo_source_note(),
        params={k: _param_repr(v) for k, v in sorted(vars(app.args).items())},
    )
    ctx_run = getattr(app.args, "run_context", None)
    if ctx_run is not None and getattr(ctx_run, "run", None) is not None:
        ctx.run_name = str(ctx_run.run)
        ctx.run_notes = tuple(ctx_run.notes)
    ctx.param_overrides = dict(getattr(app.args, "param_overrides", None) or {})
    ctx.detector_thresh = rg._DETECTOR_DEFAULT_THRESH
    ctx.detector_min_area = rg._DETECTOR_MIN_AREA
    parser = rg.build_arg_parser()
    defaults = {a.dest: a.default for a in parser._actions}
    # Compared against the RAW argparse values, never the rendered ones:
    # _param_repr turns a default of {} into "{}" and [] into "[]", so
    # comparing rendered values would mark those flags non-default on a run
    # that never touched them.
    raw = vars(app.args)
    ctx.non_default = tuple(k for k in ctx.params
                            if k in defaults and k in raw and defaults[k] != raw[k])
    if dr is None:
        return ctx

    ctx.directory = dr.rel_name
    ctx.reviewed, ctx.n_clusters = dr.progress()
    ctx.n_raw = dr.unmatched_total()
    # Detections come from the run's dets/ files, so those are what a reader
    # needs to compare against a run_manifest.
    for name in ("flight_data.db", "labels.txt", "review_log.jsonl"):
        p = dr.path / name
        ctx.digests.append((name, _digest(p), p.stat().st_size if p.exists() else None))
    for spec in getattr(dr, "layers", ()) or ():
        ctx.digests.append((f"dets/{spec.path.name}", _digest(spec.path),
                            spec.path.stat().st_size if spec.path.exists() else None))

    # The geospatial arm is built by a background worker, so "not available
    # yet" replaces "not requested". Nothing here recomputes: an unfinished
    # arm simply leaves the section out rather than filling it with zeros.
    g = app._ensure_geo_verdicts()
    if g and g.get("available"):
        ctx.geo_params = dict(g["params"])
        v = g["verdicts"]
        ctx.propagation = {
            "rule_selected": app.propagation_rule,
            "rules_available": list(rg._GEO_VERDICT_RULES),
            "n_geo_clusters": g["n_clusters"],
            "n_clusters_judged": len(v.votes),
            "n_clusters_conflicted": v.n_conflicted,
            "n_judged_keys": v.n_judged_keys,
            "n_unresolved": v.n_unresolved,
        }
        # THE REACH TAIL, carried because it is the number a reader most needs
        # and the one a summary statistic destroys. See
        # review_gui._ensure_geo_verdicts.
        reach = g.get("reach") or {}
        if reach:
            n_gui = sum(r[0] for r in reach.values())
            top = sorted(reach.items(), key=lambda kv: -kv[1][0])[:5]
            ctx.propagation["reach_top"] = [
                {"geo_cluster": lab, "n_clusters": n_cl, "n_frames": n_fr,
                 "share_of_backlog": (n_cl / n_gui) if n_gui else 0.0,
                 "judged": lab in v.votes}
                for lab, (n_cl, n_fr) in top]
            ctx.propagation["reach_total_clusters"] = n_gui
        scaffold = app._geo_scaffold
        if scaffold is not None and getattr(scaffold, "ok", False):
            import numpy as np
            lab = scaffold.labels
            ctx.geo_cluster_sizes = [int((lab == c).sum())
                                     for c in np.unique(lab) if c != -1]

    # PER MODEL. Read from the same memoised accessors the HUD and [p] use, so
    # the document and the screen cannot disagree.
    met = app._layer_metrics(dr) or {}
    geo_all = app._layer_geo_summaries()
    specs = {sp.layer_id: sp for sp in (getattr(dr, "layers", ()) or ())}
    order = app.layers.sequence() if getattr(app, "layers", None) else list(met)
    for lid in order:
        m, sp = met.get(lid), specs.get(lid)
        if m is None or sp is None:
            continue
        gm = geo_all.get(lid)
        ctx.models.append({
            "layer_id": lid, "label": m.label, "verdict": m.verdict,
            "visible": lid in app.layers.visible,
            "n_dets": m.n_dets, "n_clusters_fired": m.n_clusters_fired,
            "pixel_pre": tuple(m.pre), "pixel_post": tuple(m.post),
            "n_confirmed": m.n_confirmed, "n_hard": m.n_hard,
            "n_confirmed_elsewhere": m.n_confirmed_elsewhere,
            "config": dict(getattr(sp, "config", None) or {}),
            "geo": gm if (gm and gm.get("available")) else None,
        })

    for rec in dr.store.judgments.values():
        verdict = rec.get("judgment")
        if verdict not in rg.EXPORTABLE:
            continue
        ctx.n_adjudicated += 1
        prop = (rec.get("propagated") or {}).get("rule")
        # The split is by convention, not by author. A propagated label is a
        # human verdict about the same physical spot, carried across frames by
        # geometry, so the question a reader needs answered is which tie-break
        # rule produced it. Rows carrying "reference_op" were written by an
        # automated classifier and need a separate warning.
        key = (f"propagated:{prop}" if prop
               else "legacy classifier" if rec.get("reference_op")
               else "judged directly")
        b = ctx.adjudication.setdefault(verdict, {})
        b[key] = b.get(key, 0) + 1

    ctx.caveats = _caveats(ctx)
    return ctx


def _caveats(ctx: ReportContext) -> List[str]:
    """The things a reader must not conclude. Assembled in one place so the
    page-6 list, the [p] SUMMARY block and the HUD warning cannot drift apart
    -- and so that adding a caveat anywhere means adding it everywhere."""
    out: List[str] = []
    out.append(
        "Detection source: " + ctx.detection_source + ". These are per-frame detections "
        "from the run's dets/. They are not the birdseye onboard pipeline's results.txt, "
        "which is an accumulated reprojection log (every cluster point accumulated so far "
        "is reprojected into every later frame, and every confidence is 1.0) and which "
        "this tool does not read.")
    n_legacy = sum(b.get("legacy classifier", 0) for b in ctx.adjudication.values())
    if n_legacy:
        out.append(
            f"{n_legacy} of {ctx.n_adjudicated} adjudications: rows with source=classifier "
            "were written by an automated classifier, not a human; precision over them is "
            "not an independent check. They carry a \"reference_op\" field in "
            "review_log.jsonl; filter on it to separate them.")
    n_prop = sum(n for b in ctx.adjudication.values()
                 for k, n in b.items() if k.startswith("propagated:"))
    if n_prop:
        rules = sorted({k.split(":", 1)[1] for b in ctx.adjudication.values()
                        for k in b if k.startswith("propagated:")})
        out.append(
            f"{n_prop} of {ctx.n_adjudicated} adjudications were PROPAGATED rather than made "
            f"on the cluster directly, under {', '.join(rules)}. Every input is a human "
            "verdict and no model is involved, so these are human judgments -- but the "
            "tie-break rule that carried them is a CONVENTION worth ~15 precision points, so "
            "a result computed over them is conditional on it. Each row records the rule, the "
            "geo cluster and the votes in its \"propagated\" block.")
    if ctx.propagation:
        p = ctx.propagation
        if p.get("n_clusters_conflicted"):
            out.append(
                f"{p['n_clusters_conflicted']} of {p.get('n_clusters_judged','?')} judged geo "
                "clusters hold BOTH a confirmed_real and a hard_negative. DBSCAN at eps=0.20 m "
                "chains, so a real burrow and a rejected rock 30 cm apart are one cluster. "
                "Which verdict wins is a tie-break CONVENTION, not data, and is worth ~15 "
                "precision points on this dataset. There is no primary rule and no default: "
                "all three are always reported in the per-model tables, and the spread is the "
                "result.")
        top = (p.get("reach_top") or [None])[0]
        if top and top["share_of_backlog"] > 0.05:
            out.append(
                f"ONE GEO CLUSTER COVERS {top['share_of_backlog'] * 100:.1f}% OF THE REVIEW "
                f"BACKLOG ({top['n_clusters']:,} clusters across {top['n_frames']} frames). "
                "DBSCAN chaining at eps=0.20 m has merged a stretch of ground into a single "
                "component, so a single human verdict anywhere in it propagates a label to all "
                "of it. Any figure computed over propagated labels on this flight rests on very "
                "few independent decisions - count the verdicts, not the labels.")
        if p.get("n_unresolved"):
            out.append(
                f"{p['n_unresolved']} judged detection(s) reach no geo cluster -- DBSCAN noise "
                "or no resolvable pose. Nothing propagates from or to them, and they take no "
                "part in any geo figure.")
    if ctx.models:
        no_a = [m["label"] for m in ctx.models
                if m["geo"] and not (((m["geo"].get("recall") or {}).get("post_common")
                                      or {}).get("a_added_units"))]
        if no_a:
            out.append(
                f"a = 0 added markers for {len(no_a)} model(s). POST recall is therefore an "
                "UNCONSTRAINED upper bound there: unmarked targets are only discoverable where "
                "the detector fired, so the adjudication terms can only raise it. Added markers "
                "are the one observable slice of the misses nobody recorded.")
        out.append(
            "THERE IS NO UNION SCORECARD IN THIS REPORT. Every precision and recall figure is "
            "per checkpoint. The union of all checkpoints' detections is the review substrate "
            "-- it is what you clicked -- but it is a review-burden measure and not any "
            "model's result, so it is not scored here or on screen.")
        out.append(
            "min_samples thresholds POINTS, not distinct frames. After grid dedup the two have "
            "come apart, so the parameter is very nearly inert on this input -- see "
            "diagnostics.n_clusters_under_min_frames.")
        out.append(
            "The in-view recall denominator uses the in-view polygon test (image bounds inset "
            "by the 81.5 px projection-noise buffer).")
    out.append(
        "The two metric spaces in this report are NOT comparable. They use different units "
        "(pixel detection / UTM cluster and geotag), different ground truth (labels.txt / RTK "
        "field survey) and different matchers (live many-to-one / DBSCAN). Reporting one as "
        "though it were another is the single easiest mistake to make here.")
    return out


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

def _styles():
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("Body2", parent=ss["BodyText"], fontSize=9, leading=12.5,
                          alignment=TA_LEFT, spaceAfter=6))
    ss.add(ParagraphStyle("Mono", parent=ss["BodyText"], fontName="Courier",
                          fontSize=7.5, leading=9.5, spaceAfter=4))
    ss.add(ParagraphStyle("Caveat", parent=ss["BodyText"], fontSize=9, leading=12.5,
                          leftIndent=12, spaceAfter=8, textColor="#8a1f11"))
    return ss


_CELL_STYLE = None


_PARAM_MAX_CHARS = 300
_MISSING = object()


def _param_repr(v):
    """One argparse value, rendered SHORT enough to be a table cell.

    `vars(args)` is not all scalars. `run_context` is a whole RunContext --
    which carries the parsed PREREGISTRATION.json and selection_pooled.json --
    so stringifying it verbatim produced a single table cell 12,902 points
    tall (~180 inches) and reportlab raised LayoutError, killing the export on
    page 5. That was not a test artifact: it would fire on [x] in any layered
    session, i.e. every session now that the run directory is mandatory.

    So: name the structured things instead of dumping them, and hard-cap
    everything else. A parameter table exists to answer "what was this run
    configured with", and a 4,000-character JSON blob answers that worse than
    a one-line summary does.
    """
    if v is None or isinstance(v, (bool, int, float)):
        return v
    run = getattr(v, "run", _MISSING)
    if run is not _MISSING:            # a RunContext -- name it, don't dump it
        return f"<run context: {run}>"
    if isinstance(v, (list, tuple)):
        if not v:
            return "[]"
        body = ", ".join(str(x) for x in v)
        return body if len(body) <= _PARAM_MAX_CHARS else \
            f"{len(v)} item(s): {body[:_PARAM_MAX_CHARS]}..."
    if isinstance(v, dict):
        if not v:
            return "{}"
        body = ", ".join(f"{k}={x}" for k, x in sorted(v.items()))
        return body if len(body) <= _PARAM_MAX_CHARS else \
            f"{len(v)} key(s): {body[:_PARAM_MAX_CHARS]}..."
    text = str(v)
    return text if len(text) <= _PARAM_MAX_CHARS else text[:_PARAM_MAX_CHARS] + "..."


def _cell(text: str):
    """A table cell that WRAPS. A bare string in a reportlab Table is laid out
    on one line and silently runs off the page edge -- which is exactly what the
    definitions table did, since geo_metrics.DEFINITIONS values are sentences.
    Anything that can be long has to come through here."""
    global _CELL_STYLE
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    if _CELL_STYLE is None:
        _CELL_STYLE = ParagraphStyle("Cell", fontName="Helvetica", fontSize=8, leading=9.8)
    return Paragraph(str(text), _CELL_STYLE)


def _table(rows, col_widths=None, header=True, highlight=None):
    """A table in the house style: hairline grid, shaded header, small mono-ish
    body. `highlight` maps a row index to a colour, which carries the GUI's
    PRE/POST colour language onto the page."""
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    for idx, hexcol in (highlight or {}).items():
        style.append(("TEXTCOLOR", (0, idx), (-1, idx), colors.HexColor(hexcol)))
        style.append(("FONTNAME", (0, idx), (0, idx), "Helvetica-Bold"))
    # repeatRows: a table that splits across a page break re-draws its header,
    # so the continuation is still readable on its own.
    return Table(rows, colWidths=col_widths, style=TableStyle(style), hAlign="LEFT",
                 repeatRows=1 if header else 0)


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v if v is not None else "-")


_RULE_ABBR = {"positive_wins": "pos", "majority": "maj", "hard_negative_wins": "neg"}


def _abbr(rule: str) -> str:
    """Tie-break rule -> short label, mirroring geo_metrics.VERDICT_RULE_ABBR.
    Local so a page builder never has to import geo_metrics (and so a report
    still renders if that import fails)."""
    return _RULE_ABBR.get(rule, rule)


def _adjudication_summary(ctx: ReportContext) -> str:
    """One-line composition of the adjudication log, by HOW each judgment was
    arrived at rather than by author.

    What a reader needs is which judgments were propagated (and under which
    convention), because that is the part a result is conditional on -- see
    _caveats."""
    if not ctx.n_adjudicated:
        return "0"
    parts = {}
    for b in ctx.adjudication.values():
        for k, n in b.items():
            parts[k] = parts.get(k, 0) + n
    return f"{ctx.n_adjudicated} (" + ", ".join(
        f"{n} {k}" for k, n in sorted(parts.items())) + ")"


def _page_title(ctx, ss):
    from reportlab.platypus import Paragraph, Spacer
    P = Paragraph
    out = [P(PAGE_TITLE, ss["Title"]),
           P(f"{ctx.directory}", ss["Heading2"]),
           Spacer(1, 6),
           _table([["field", "value"],
                   ["reviewer", _cell(ctx.reviewer)],
                   ["generated", ctx.generated],
                   ["reviewed", f"{ctx.reviewed} / {ctx.n_clusters} clusters "
                                f"({(ctx.reviewed / ctx.n_clusters * 100) if ctx.n_clusters else 0:.1f}%)"],
                   ["raw detections", str(ctx.n_raw)],
                   ["adjudications", _cell(_adjudication_summary(ctx))],
                   ["models", str(len(ctx.models))],
                   ["propagation rule",
                    (ctx.propagation.get("rule_selected") or "off (nothing propagated)")]],
                  col_widths=[110, 360]),
           Spacer(1, 12),
           P("Command", ss["Heading3"]),
           P(ctx.command or "(unknown)", ss["Mono"]),
           Spacer(1, 8),
           P("Inputs", ss["Heading3"]),
           P("Input provenance is recorded as SHA-256 file digests. They are sha256 "
             "prefixes, computed the same way as those in a run_manifest.json, and can be "
             "compared directly against one.",
             ss["Body2"]),
           _table([["file", "sha256[:16]", "bytes"]]
                  + [[_cell(n), d or "(absent)", f"{b:,}" if b is not None else "-"]
                     for n, d, b in ctx.digests],
                  col_widths=[130, 180, 90]),
           Spacer(1, 10),
           P("Detection source", ss["Heading3"]),
           P(ctx.detection_source or "(unknown)", ss["Body2"])]
    return out


def _page_methods(ctx, ss):
    from reportlab.platypus import Paragraph, Spacer, PageBreak
    P = Paragraph
    out = [PageBreak(), P("Data and methods", ss["Heading1"]),
           P("Two metric spaces are reported, both PER CHECKPOINT. They share one ground truth "
             "(the RTK geotag survey; labels.txt is its per-frame projection) but compare "
             "before and after projection, in different counting units, and are not "
             "interchangeable.", ss["Body2"]),
           _table([["", "pixel", "geospatial"],
                   ["unit", "detection, pixels", "DBSCAN cluster / geotag"],
                   ["ground truth", _cell("RTK geotag survey (labels.txt is its per-frame projection)"),
                    _cell("RTK geotag survey (labels.txt is its per-frame projection)")],
                   ["matcher", "many-to-one", "DBSCAN 0.20 m"],
                   ["computed by", "review_gui.py", "geo_metrics.py"],
                   ["scope", "one checkpoint", "one checkpoint"]],
                  col_widths=[90, 170, 200]),
           Spacer(1, 8),
           P("Label propagation", ss["Heading3"]),
           P("A geospatial cluster is a DBSCAN group in UTM built from every frame's "
             "detections, so it spans frames by construction: one target seen in frames 700 "
             "and 830 is one cluster. A human verdict on it in either frame is therefore "
             "evidence about the same physical spot in the other, which is what lets one "
             "adjudication pass label the whole flight. When a cluster holds both a "
             "confirmed_real and a hard_negative - chaining at 0.20 m merges physically "
             "distinct spots - the reviewer chooses which wins; there is no default, and "
             "nothing is propagated until a convention is named.", ss["Body2"]),
           Spacer(1, 12),
           P("The two arms", ss["Heading3"]),
           P("<b>PRE</b> is geometry only, with every adjudication clause stripped out - the "
             "published arm. <b>POST</b> applies the human verdicts. Both are always reported: "
             "the gap between them reflects that field geotags do not exhaustively enumerate "
             "the visually detectable targets, so the naive statistic understates precision.",
             ss["Body2"]),
           Spacer(1, 6),
           P("Why POST recall is a bound, not an estimate", ss["Heading3"]),
           P("POST precision defines a true positive as <i>(human-adjudicated positive) OR "
             "(geometrically matched)</i>. That enlarges the positive class, and recall's "
             "denominator <i>is</i> the positive class - so the two arms cannot share one recall "
             "without scoring under two different definitions of \"true positive\".", ss["Body2"]),
           P("R_PRE  = n_hit / n_den<br/>"
             "R_POST = (n_hit + c) / (n_den + c + a)<br/>"
             "R_true = (n_hit + c) / (n_den + c + a + m)&nbsp;&nbsp;&nbsp;m &gt;= 0, unobservable",
             ss["Mono"]),
           P("<b>c</b> is adjudicated positives matching no survey unit; <b>a</b> is deduplicated "
             "human-added markers; <b>m</b> is the targets neither surveyed nor detected. An "
             "unmarked target is only <i>discoverable where the detector fired</i>, so every unit "
             "of c is by construction a true positive and never a false negative: c is a censored "
             "sample and can only raise the ratio. R_POST is exactly the m = 0 case, and is "
             "therefore an upper bound, written <b>&lt;=</b> throughout this report and in the GUI.",
             ss["Body2"]),
           P("R_PRE remains the unbiased point estimate under the assumption that survey omission "
             "is independent of detectability. That assumption is defensible here specifically: "
             "the geotags are a walking RTK field survey while the detector reads aerial imagery, "
             "so the two miss-mechanisms are near-independent. Report the pair; neither number "
             "alone is \"the\" recall.", ss["Body2"])]
    try:
        from . import geo_metrics
        defs = geo_metrics.DEFINITIONS
        out += [Spacer(1, 8), P("Geospatial definitions (verbatim from geo_metrics.DEFINITIONS)",
                                ss["Heading3"]),
                _table([["term", "definition"]]
                       + [[_cell(f"PRE {k}"), _cell(v)] for k, v in defs["pre"].items()]
                       + [[_cell(f"POST {k}"), _cell(v)] for k, v in defs["post"].items()]
                       + [[_cell(f"recall {k}"), _cell(v)] for k, v in defs["recall"].items()],
                       col_widths=[95, 375])]
    except Exception as e:  # noqa: BLE001 -- an optional page section, never fatal
        out.append(P(f"(geo definitions unavailable: {e})", ss["Body2"]))
    return out


def _page_params(ctx, ss):
    from reportlab.platypus import Paragraph, PageBreak
    P = Paragraph
    rows = [["parameter", "value", ""]]
    for k, v in ctx.params.items():
        note = "non-default" if k in ctx.non_default else ""
        if k in ctx.param_overrides:
            note = f"OVERRIDES RUN ({ctx.param_overrides[k]['run']:g})"
        rows.append([_cell(k), _cell(v), _cell(note)])
    out = [PageBreak(), P("Run parameters", ss["Heading1"])]
    if ctx.run_name:
        out += [P(f"Derived from run <b>{ctx.run_name}</b>. The detections and the matching "
                  "threshold both come from that directory, so they cannot have drifted "
                  "apart.", ss["Body2"])]
        for n in ctx.run_notes:
            out.append(P(f"note: {n}", ss["Body2"]))
    else:
        out.append(P("No run directory was given, so no parameters were derived.", ss["Body2"]))
    if ctx.param_overrides:
        out.append(P("<b>Parameters were overridden relative to the run.</b> The measurements "
                     "below were therefore taken under different settings than the run's own "
                     "artifacts were built with:", ss["Caveat"]))
        out.append(_table([["parameter", "run built at", "reviewed at"]]
                          + [[k, f"{v['run']:g}", f"{v['live']:g}"]
                             for k, v in ctx.param_overrides.items()],
                          col_widths=[170, 110, 110]))
    # ---- POST-PROCESSING CONFIG -------------------------------------------
    #
    # How the detector's raw heatmap output became the points being reviewed.
    # None of this reached the report before: LayerSpec.config was parsed out
    # of the run's PREREGISTRATION.json and then read only for `arms`, so a
    # PDF could report a checkpoint's precision without saying at what
    # threshold it had fired. That is not reproducible, which is the whole
    # purpose of this document.
    out += [P("Post-processing configuration", ss["Heading2"]),
            P("How detector output became the points reviewed here. The heatmap stage is "
              "fixed in run_detector.py and identical for every checkpoint; the per-checkpoint "
              "parameters come from the run's PREREGISTRATION.json; the matching and clustering "
              "parameters are this session's, listed in full in the table below.", ss["Body2"]),
            _table([["stage", "operation"],
                    ["heatmap", _cell("GaussianBlur 5x5 sigma=1 -> binary threshold -> "
                                      "findContours")],
                    ["detections", _cell(f"contourArea > MIN_AREA={ctx.detector_min_area} -> "
                                         f"moment centroid, peak confidence")],
                    ["default threshold", str(ctx.detector_thresh)]],
                   col_widths=[110, 360])]
    if any(m["config"] for m in ctx.models):
        out.append(_table(
            [["checkpoint", "thresh", "eps", "min_samples", "eps_predict"]]
            + [[_cell(m["label"]),
                _fmt(m["config"].get("thresh"), 3),
                _fmt(m["config"].get("eps"), 2),
                str(m["config"].get("min_samples", "-")),
                _fmt(m["config"].get("eps_predict"), 2)]
               for m in ctx.models],
            col_widths=[150, 80, 70, 90, 90]))
    elif ctx.models:
        out.append(P("The run records no per-checkpoint post-processing config "
                     "(no PREREGISTRATION.json, or no configs in it).", ss["Body2"]))
    if ctx.geo_params:
        g = ctx.geo_params
        out.append(_table([["geospatial parameter", "value"],
                           ["DBSCAN eps (m)", _fmt(g.get("eps"), 2)],
                           ["DBSCAN min_samples", str(g.get("min_samples", "-"))],
                           ["geotag match radius (m)", _fmt(g.get("eps_predict"), 2)],
                           ["grid dedup (cm)", _fmt(g.get("dedup_cm"), 1)]],
                          col_widths=[200, 240]))

    out += [P("Every command-line parameter of the session that produced this report. Values "
              "differing from the parser default are marked, since those are the ones that "
              "change what the numbers mean.", ss["Body2"]),
            _table(rows, col_widths=[170, 200, 100])]
    return out


def _page_results(ctx, ss):
    """Results, PER CHECKPOINT.

    The union of every active checkpoint's detections is what the reviewer
    clicked, which makes it a review-burden measure, not any model's result, so
    this page has no union table. One adjudication scores every checkpoint, and
    this is the page where that pays off.
    """
    from reportlab.platypus import Paragraph, Spacer, PageBreak
    P = Paragraph
    pre_hex, post_hex = "#5555aa", "#2f8f2f"
    out = [PageBreak(), P("Results", ss["Heading1"])]
    if not ctx.models:
        out += [P("No model layers were loaded, so there is nothing to score.", ss["Body2"])]
        return out
    out += [P("One adjudication, every checkpoint scored by it", ss["Heading3"]),
            P("Each block below is one checkpoint. <b>Pixel</b> is live many-to-one matching "
              "against labels.txt; <b>geo</b> is that checkpoint's own DBSCAN clusters against "
              "the RTK geotag survey (the paper's geospatial precision/recall). The two are "
              "not comparable - see "
              "the caveats. Geo has no F1 column on purpose: precision is per cluster and "
              "recall per geotag, so their harmonic mean is a category error.", ss["Body2"])]
    for m in ctx.models:
        tag = {"collapsed": " - COLLAPSED (degenerate)",
               "sparse": " - SPARSE (degenerate)"}.get(m["verdict"], "")
        out += [Spacer(1, 8),
                P(f"{m['label']}{tag}" + ("" if m["visible"] else " - hidden on screen"),
                  ss["Heading3"])]
        rows = [["arm", "precision", "recall", "F1"],
                ["PIXEL PRE", _fmt(m["pixel_pre"][0]), _fmt(m["pixel_pre"][1]),
                 _fmt(m["pixel_pre"][2])],
                ["PIXEL POST", _fmt(m["pixel_post"][0]), _fmt(m["pixel_post"][1]) + " <=",
                 _fmt(m["pixel_post"][2]) + " <="]]
        hl = {1: pre_hex, 2: post_hex}
        g = m["geo"]
        if g:
            rec = g["recall"]
            rows.append(["GEO PRE", _fmt(g["pre"]["precision"]), _fmt(rec["recall_all"]), ""])
            hl[len(rows) - 1] = pre_hex
            for rule in g["verdict_rules"]:
                q, r = g["post"][rule], rec["post"][rule]
                rows.append([f"GEO POST {_abbr(rule)}", _fmt(q["precision"]),
                             _fmt(r["recall_all_bound"]) + " <=", ""])
                hl[len(rows) - 1] = post_hex
        out.append(_table(rows, col_widths=[90, 100, 110, 100], highlight=hl))
        line = (f"n_dets={m['n_dets']} &nbsp; fired in {m['n_clusters_fired']} clusters "
                f"&nbsp;|&nbsp; c={m['n_confirmed']} hard_negative={m['n_hard']}")
        if m["n_confirmed_elsewhere"]:
            # Never folded into R<=: a cluster confirmed real that this layer
            # did not fire on is a DEMONSTRABLE miss for it, so the bound
            # argument post_recall derives does not hold across layers.
            line += (f" &nbsp;|&nbsp; {m['n_confirmed_elsewhere']} confirmed cluster(s) this "
                     f"layer never fired on (NOT in R&lt;=)")
        if g:
            rec = g["recall"]
            common = rec["post_common"]
            line += (f"<br/>geotags hit {rec['tp_geotags']}/{rec['n_geotags']} "
                     f"(in view: {rec['tp_geotags_in_view']}/{rec['n_in_view']}) "
                     f"&nbsp;|&nbsp; missed {rec['fn_geotags']} "
                     f"&nbsp;|&nbsp; clusters {g['n_clusters']} "
                     f"&nbsp;|&nbsp; a={common['a_added_units']} "
                     f"&nbsp;|&nbsp; coverage {g['adjudication']['coverage'] * 100:.1f}% "
                     f"({g['adjudication']['n_clusters_judged']}/{g['n_clusters']}, "
                     f"{g['adjudication']['n_conflicted']} conflicted)")
            line += (" &nbsp;|&nbsp; c per rule: "
                     + ", ".join(f"{_abbr(r)}="
                                 f"{rec['post'][r]['c_confirmed_unmatched_clusters']}"
                                 for r in g["verdict_rules"]))
        else:
            line += "<br/>geo: not computed yet for this layer"
        out.append(P(line, ss["Mono"]))
    out += [Spacer(1, 10),
            P("<b>Three GEO POST rows per model, one per tie-break rule, and there is no "
              "primary one.</b> When a DBSCAN cluster holds both a <i>confirmed_real</i> and a "
              "<i>hard_negative</i> - which happens because chaining at 0.20 m merges "
              "physically distinct spots - which verdict wins is decided by a convention, not "
              "by the data. Quote the spread; a single POST number is not a measurement.",
              ss["Body2"])]

    if ctx.propagation:
        p = ctx.propagation
        out += [Spacer(1, 10), P("Label propagation", ss["Heading3"]),
                P("Human adjudications spread across frames through the geospatial clusters: a "
                  "DBSCAN group in UTM is built from every frame's detections, so it spans "
                  "frames by construction and a verdict given in one frame is evidence about "
                  "the same physical spot in every other frame that saw it. The tie-break rule "
                  "is chosen by the reviewer and starts unset; nothing is proposed until one "
                  "is picked.", ss["Body2"]),
                _table([["field", "value"],
                        ["rule selected", str(p.get("rule_selected") or "(none - off)")],
                        ["rules available", ", ".join(p.get("rules_available", ()))],
                        ["geo clusters", str(p.get("n_geo_clusters"))],
                        ["clusters carrying a verdict", str(p.get("n_clusters_judged"))],
                        ["clusters holding BOTH verdicts", str(p.get("n_clusters_conflicted"))],
                        ["judged detections unresolved", str(p.get("n_unresolved"))]],
                       col_widths=[200, 240])]
        if p.get("reach_top"):
            out += [P("How far one verdict reaches", ss["Heading3"]),
                    P("A geospatial cluster is meant to be one target seen in many frames, and "
                      "mostly it is. But DBSCAN chains, and on this data the distribution has a "
                      "heavy tail: the largest clusters below are stretches of ground merged "
                      "into a single component, and a verdict on any part of one propagates to "
                      "all of it. This table is the blast radius, and it is why the reach is "
                      "also shown on the selected point in the GUI - before [a] is pressed, "
                      "not after.", ss["Body2"]),
                    _table([["geo cluster", "clusters", "frames", "share of backlog", "judged?"]]
                           + [[str(r["geo_cluster"]), f"{r['n_clusters']:,}",
                               str(r["n_frames"]), f"{r['share_of_backlog'] * 100:.1f}%",
                               "yes" if r["judged"] else "no"]
                              for r in p["reach_top"]],
                           col_widths=[90, 90, 70, 120, 70])]

    if ctx.adjudication:
        keys = sorted({k for v in ctx.adjudication.values() for k in v})
        out += [Spacer(1, 10), P("Adjudication composition", ss["Heading3"]),
                P("By how each judgment was arrived at, not by author: every row here is a "
                  "human judgment. A <i>propagated</i> one was carried from a verdict on the "
                  "same physical spot in another frame under the named convention. A "
                  "<i>legacy classifier</i> row: rows with source=classifier were written by "
                  "an automated classifier, not a human; precision over them is not an "
                  "independent check.", ss["Body2"]),
                _table([["verdict"] + keys]
                       + [[k] + [str(v.get(kk, 0)) for kk in keys]
                          for k, v in sorted(ctx.adjudication.items())],
                       col_widths=[130] + [max(60, 340 // max(1, len(keys)))] * len(keys))]
    return out


def _figure_flowables(ctx, ss):
    """Page 5. Every plot here restates something already tabulated on page 4 or
    listed on page 6 -- redundant ON PURPOSE, so that losing matplotlib costs
    the reader nothing but convenience."""
    from reportlab.platypus import Paragraph, Spacer, PageBreak, Image
    P = Paragraph
    out = [PageBreak(), P("Figures", ss["Heading1"])]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001 -- see the module docstring
        out.append(P(f"Figures unavailable: {type(e).__name__}: {e}. Every number they would "
                     "show is tabulated above; nothing is lost but convenience.", ss["Body2"]))
        return out

    def _png(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    figs: List[Tuple[str, Any]] = []

    # ---- FIGURE 1: the PR plane, PRE -> POST, one marker per checkpoint ----
    #
    # The conventions follow the model-selection report's PR-plane figure:
    # the same axis labels, the same per-checkpoint annotation, and the same
    # health coding, matched by hand here, with STYLE below mirroring it.
    #
    # WHY HEALTH CODING IS NOT OPTIONAL. On an unlabelled scatter the
    # degenerate checkpoints land at the top-left -- near-zero recall, the
    # HIGHEST precision in the field -- which is exactly the position a reader
    # interprets as "conservative and precise". Colour and marker are what
    # stop that misreading, and LayerSpec.verdict already carries the screen's
    # health call, so no new data source is needed.
    #
    # WHY POST IS A SEGMENT AND NOT A POINT. Each model's PRE is one position,
    # but its POST is THREE -- one per tie-break rule -- and GEO_METRICS.md
    # forbids collapsing them, because the choice between them is a convention
    # worth ~15 precision points. So PRE is the filled marker, an arrow runs to
    # the most-positive POST, and the remaining rules are drawn as the rest of
    # the segment. A single POST dot would be a claim the data does not make.
    STYLE = {"healthy": ("#2f8f2f", "o", 34),
             "sparse": ("#c26a1c", "^", 26),
             "collapsed": ("#b5232a", "x", 34)}
    DEFAULT = ("#8a8a8a", "s", 22)
    geo_models = [m for m in ctx.models if m["geo"]]
    if geo_models:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        seen = set()
        for m in geo_models:
            colour, marker, size = STYLE.get(m["verdict"], DEFAULT)
            g = m["geo"]
            rec = g["recall"]
            x0, y0 = rec["recall_all"], g["pre"]["precision"]
            pts = [(rec["post"][r]["recall_all_bound"], g["post"][r]["precision"])
                   for r in g["verdict_rules"]]
            lab = m["verdict"] if m["verdict"] in STYLE else "unclassified"
            ax.scatter([x0], [y0], s=size, color=colour, marker=marker, zorder=3,
                       label=(lab if lab not in seen else None))
            seen.add(lab)
            # The POST span, drawn most-positive first (verdict_rules order).
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=colour, lw=1.2, alpha=0.85, zorder=2)
            ax.scatter(xs, ys, s=10, color=colour, marker="|", zorder=2)
            ax.annotate("", xy=pts[0], xytext=(x0, y0), zorder=2,
                        arrowprops=dict(arrowstyle="->", color=colour, lw=0.9, alpha=0.7))
            ax.annotate(m["label"], (x0, y0), fontsize=5.5, xytext=(3, 3),
                        textcoords="offset points", color=colour)
        ax.set_xlabel("recall (geotags in view found)")
        ax.set_ylabel("prec_like (clusters containing a geotag)")
        ax.set_title("Geospatial PRE -> POST by checkpoint, health-coded", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        figs.append(
            ("Each filled marker is one checkpoint's PRE position; the arrow runs to its "
             "most-positive POST and the short segment covers all three tie-break rules - that "
             "segment is a CONVENTION, not measurement error. Health coding is load-bearing: "
             "the degenerate checkpoints cluster at the top-left, at near-zero recall with the "
             "highest precision in the field, which is exactly the position an unlabelled "
             "scatter invites a reader to read as 'conservative and precise'. Only the healthy "
             "points form a genuine frontier, and even among those precision is biased low at "
             "the high-recall end because the geotag survey is not exhaustive - so the real "
             "frontier is flatter than the one drawn.", _png(fig)))

    # ---- FIGURE 2: PRE vs POST per model, pixel space ----------------------
    if ctx.models:
        labels = [m["label"] for m in ctx.models]
        pres = [m["pixel_pre"][0] for m in ctx.models]
        posts = [m["pixel_post"][0] for m in ctx.models]
        fig, ax = plt.subplots(figsize=(6.2, 2.8))
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], pres, width=0.4, label="PIXEL PRE", color=PRE_RGB)
        ax.bar([i + 0.2 for i in x], posts, width=0.4, label="PIXEL POST", color=POST_RGB)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylim(0, 1); ax.set_ylabel("precision")
        ax.legend(fontsize=7, frameon=False)
        ax.set_title("Pixel-space precision, PRE vs POST, per checkpoint", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        figs.append(("Adjudication moves every checkpoint's precision, from the same single "
                     "review pass. Pixel and geo space are not comparable - see the caveats.",
                     _png(fig)))

    if ctx.geo_cluster_sizes:
        fig, ax = plt.subplots(figsize=(6.2, 2.4))
        ax.hist(ctx.geo_cluster_sizes, bins=40, color="#4a6fa5")
        ax.set_xlabel("points per DBSCAN cluster"); ax.set_ylabel("clusters")
        ax.set_yscale("log")
        ax.set_title(f"Geo cluster sizes (n={len(ctx.geo_cluster_sizes)})", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        figs.append(("Cluster size is what min_samples thresholds - points, not distinct frames. "
                     "See the caveat on the next page.", _png(fig)))

    if ctx.adjudication:
        verdicts = sorted(ctx.adjudication)
        keys = sorted({k for v in ctx.adjudication.values() for k in v})
        # One stacked band per verdict, split by HOW the judgment was arrived at.
        palette = {"judged directly": "#2f8f2f", "legacy classifier": "#b5232a"}
        fig, ax = plt.subplots(figsize=(6.2, 2.2))
        left = [0.0] * len(verdicts)
        for i, k in enumerate(keys):
            vals = [ctx.adjudication[v].get(k, 0) for v in verdicts]
            ax.barh(verdicts, vals, left=left,
                    color=palette.get(k, "#c26a1c" if k.startswith("propagated:") else "#8a8a8a"),
                    label=k)
            left = [a + b for a, b in zip(left, vals)]
        ax.set_xlabel("adjudications"); ax.legend(fontsize=7, frameon=False)
        ax.set_title("How each call was arrived at", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        figs.append(("Every band is a human judgment. A propagated one was carried from a "
                     "verdict on the same physical spot in another frame, so a result over it "
                     "is conditional on the tie-break convention named in the legend. In a "
                     "'legacy classifier' band, rows with source=classifier were written by an "
                     "automated classifier, not a human; precision over them is not an "
                     "independent check.",
                     _png(fig)))

    for i, (caption, buf) in enumerate(figs):
        out += [Image(buf, width=440, height=440 * 0.42, kind="proportional"),
                P(caption, ss["Body2"])]
        # No trailing spacer after the last figure: a Spacer that does not fit
        # on the page starts a new one, and the following PageBreak then leaves
        # that page blank.
        if i != len(figs) - 1:
            out.append(Spacer(1, 8))
    return out


def _page_caveats(ctx, ss):
    from reportlab.platypus import Paragraph, PageBreak
    P = Paragraph
    out = [PageBreak(), P("Caveats", ss["Heading1"]),
           P("Read these before quoting any number above.", ss["Body2"])]
    for c in ctx.caveats:
        out.append(P(c, ss["Caveat"]))
    return out


def build_report(path: Path, ctx: ReportContext) -> Path:
    """Render `ctx` to a PDF at `path`. Returns the path written."""
    from reportlab.platypus import SimpleDocTemplate
    from reportlab.lib.pagesizes import LETTER

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ss = _styles()
    story = (_page_title(ctx, ss) + _page_methods(ctx, ss) + _page_params(ctx, ss)
             + _page_results(ctx, ss) + _figure_flowables(ctx, ss) + _page_caveats(ctx, ss))
    doc = SimpleDocTemplate(str(path), pagesize=LETTER,
                            leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=48,
                            title=f"{PAGE_TITLE} - {ctx.directory}", author=ctx.reviewer)

    def _footer(canv, _doc):
        canv.saveState()
        canv.setFont("Helvetica", 7)
        canv.setFillGray(0.45)
        canv.drawString(54, 30, f"{ctx.directory}   {ctx.generated}   review_gui.py [x]")
        canv.drawRightString(LETTER[0] - 54, 30, f"page {canv.getPageNumber()}")
        canv.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path
