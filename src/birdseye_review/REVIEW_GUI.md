# Detection Review GUI

`birdseye-review` is a human-in-the-loop tool for adjudicating the one thing
an automated comparison against labels cannot verify: the "naive false
positive" pool — detections that didn't match any human label. Each one is
either a valid target the annotator didn't bother marking, or the model
hallucinating on a tricky texture. You look at each one and decide, and your
decisions become a correction signal a future training run can use to
specifically penalize the model where it fell for a tricky texture.

You can also **add points of your own** — burrow damage you can see but the
model never detected at all (not even a naive, unmatched detection). That's a
complementary signal, distinct from judging what the model already flagged: an
added marker is the only *observable* evidence of the targets that neither the
annotator recorded nor the detector found, which makes it the one thing that
constrains the adjudicated recall bound from above. See the `POST` HUD line.

## What it shows you

For each flight directory (the flight's `labels.txt` + frame images +
optional `flight_data.db`, and the run's `dets/`), the matched vs. unmatched
split is computed **live**, per frame, straight from those files — there is
no precomputed evaluation output to go stale.

- **Labels** (cyan ring) — human ground truth, context only, never
  interactive.
- **Matched detections** (light gray ring) — already matched to a label,
  already confirmed real, context only.
- **Unmatched detections** (orange, filled) — the ambiguous pool. These are
  what you review — but see clustering, below: you don't review them one at
  a time.
- **Added targets** (white star) — points you've spawned yourself; see
  below.

Click an orange point (or press `space` to jump to the next unreviewed one)
to select it. A magnified inset of the native-resolution crop around the
point opens in the side panel, along with its frame index, coordinates, and
current judgment.

The inset shows a `--zoom-crop-px` (200 px) square of the **native**-resolution
frame at `--zoom-scale` (4×) nearest-neighbour magnification. The panel is
sized to fit it (`zoom_crop_px × zoom_scale + 20`), and on a monitor too narrow
to spare that much width the default scale is automatically reduced — never
below 3 — with a line printed saying so. Pass `--zoom-scale` explicitly to
override the fit
heuristic in either direction. The inset still shrinks first if the window is
too short to fit it above the live-stats HUD, so the HUD is never pushed off.

### Matching: many-to-one

Whether a detection counts as "matched" (gray ring, context only) or falls
into the unmatched review pool (orange) is decided per-point, independently
on each side: a detection is matched if **any** label lies within
`--spatial-threshold`, and a label is matched (found) if **any** detection
lies within that threshold. Several overlapping/duplicate detections of one
real target can all legitimately count as matched at once.

### Clustering: you review groups, not individual points

Some frames have hundreds to thousands of unmatched detections, usually
tightly bunched around a single physical mound. Reviewing each one
individually would be tedious and mostly redundant, so nearby detections
are pre-clustered per frame (complete-linkage, `--cluster-radius`, default
30px — every pairwise distance *within* a cluster is guaranteed to stay
under this, unlike naive chained clustering, which can silently sprawl a
"cluster" across most of a frame) and **you review the cluster's centroid,
once** — that one judgment fans back out to every raw detection in the
cluster when it's exported, so training-relevant output still lands at full
per-detection resolution.

- The raw detections themselves still render, as a **translucent layer**
  underneath the solid centroid marker, tinted by their cluster's current
  judgment — so the whole physical blob visibly turns green or red together
  the moment you judge its centroid.
- A cluster with more than one member shows a small `x{N}` label next to
  its centroid (e.g. `x7`) so you know how many raw detections one click
  represents; the selected-point panel repeats this as "cluster size".
- `--cluster-radius 0` disables clustering entirely — every detection
  becomes its own single-point cluster, exactly reproducing one-point-per-
  review behavior, if you want that for a specific directory.
- `--raw-layer-alpha` (default `0.35`) controls how visible the translucent
  raw layer is.

### Live sliders: adjust either threshold without restarting

Two sliders sit at the top of the window — **Spatial Thresh (px)** and
**Cluster Radius (px)** — so you can tune either while looking at the actual
effect on the actual data, instead of guessing, quitting, and relaunching
with a different flag. Also echoed as always-visible text in the HUD (see
"Statistics reference" below) — the slider widget's own label is easy to
miss, the HUD line isn't.

**What Spatial Thresh actually controls** (see "Matching" above for the
full mechanism): it's the many-to-one matching gate — a detection is
"matched" (gray ring, taken out of your review pool) if *any* label is
within this many pixels, and a label counts as "found" if *any* detection
is. It's a hard cutoff, not a soft distance penalty, so a real detection
sitting just past this radius from its label doesn't get any partial credit
— it lands squarely in the orange unmatched/review pool. **Raising it can
only ever move points unmatched → matched** (shrinking your review pool and
raising this tool's naive precision/recall), never the reverse — but bigger
isn't simply "better": push it far enough and genuinely distinct nearby
detections start reading as already-handled context instead of the
ambiguous cases you actually need to judge. Empirically this is the single
largest lever on naive precision/recall/F1, but with sharply diminishing
returns above ~40px.

- Drag either one and let go; after it settles (~0.6s of no further
  movement) the tool re-matches and re-clusters the directory you're
  currently viewing, in place, and picks up the new value as the default
  for anything you open afterward.
- **This can take up to ~20 seconds on the largest directories** (clustering
  is the expensive part) — a "Recomputing..." banner shows **live progress**
  while it works (frame count, elapsed time, an ETA, and a progress bar),
  not just a static message, so you can see it's actually moving rather
  than wondering if the window froze. Small/medium directories update in
  well under a second. All input (including `q`) is ignored until it
  finishes.
- Any prior judgment whose point lands in a different cluster afterward is
  reconciled the same way reopening an old directory is (see "Reopening a
  directory with per-point judgments", below) — nothing is
  ever lost; a judgment that momentarily has nowhere to attach (e.g. its
  point got matched away at a looser threshold) just shows as "orphaned"
  in the status line and reappears if you slide back.

**Adding a missed target**: right-click anywhere the model didn't already
place a marker to drop a new point exactly there — no separate confirm step,
it's immediately recorded and selected so you can see the zoom inset confirm
the placement. Right-clicking near an *existing* unmatched detection instead
fast-confirms that one as real (equivalent to selecting it and pressing
`c`, without creating a duplicate point); right-clicking near a point you've
already added just selects it rather than spawning another one on top.

### Video play/rewind (`[5]`/`[6]`/`[f]`): scrub through a frame sequence

`1`/`2` step one frame at a time; `5`/`6` do the same thing continuously,
like a video player, so you can watch a directory's frames go by instead of
clicking through them one at a time.

- `[6]` toggles playing forward, `[5]` toggles rewinding (playing
  backward); pressing the same key again stops it, and pressing the
  *other* one switches direction immediately, no need to stop first.
- `[f]` cycles the playback speed through a short set of presets
  (0.5 / 0.25 / 0.12 / 0.06 seconds per frame), wrapping back to the
  slowest. A persistent `> playing (N fps)` / `< rewinding (N fps)` line
  shows in the HUD (unlike the status line, this doesn't expire — it stays
  visible for as long as playback is actually running).
- Playback goes through every frame in the directory in order — it does
  **not** skip to the next unreviewed point the way `n`/`b` do; that's a
  different, existing feature.
- **Stops at the directory's first/last frame** (mirrors `1`/`2`'s own
  clamp-and-stop behavior) — it never wraps around or crosses into the
  next/previous directory the way `n`/`b` do.
- **Any other key or an actual mouse click stops playback first**, then
  still does its own thing normally — e.g. pressing `c` mid-playback stops
  it and commits, rather than the keypress being swallowed or playback
  continuing underneath. Plain mouse *movement* never stops it (only a
  real click does) — otherwise merely having the cursor over the window
  would make the feature unusable.
- Moving a slider also stops playback — a recompute and playback never run
  at the same time.
- Every play tick pays a full frame reload and re-render (there's no
  frame-ahead prefetching), so on a large flight directory the fastest
  speed preset may visibly run slower than its nominal rate — an expected
  cost of scrubbing through full-resolution imagery, not a bug.

### `[p]` the stats overlay: this session's own numbers, per model

`[p]` reports what **this session** has measured, at per-checkpoint
granularity, and needs no files at all. Five blocks, in this order:

1. **`SUMMARY`** — directory, run, reviewer, review progress, the active
   propagation rule, and every live caveat. `_draw_paginated` shows ~38 lines
   at a time, so what is first is what most readers ever see.
2. **`PER MODEL`** — one block per checkpoint: pixel `PRE`/`POST`, geo `PRE`
   plus one `POST` row per tie-break rule, and **the raw geotag counts behind
   those ratios** (`geotags hit 131/161`, in-view, missed, cluster count, `a`,
   coverage), also written to `geo_metrics.json` — a ratio without its
   denominator is not quotable.
3. **`ADJUDICATION`** — the log's composition by **how each judgment was
   arrived at**: judged directly, or propagated under a named convention. Then
   the propagator's coverage, and *how many labels each of the three rules
   would produce on the current frame* — all three, because the choice between
   them is worth ~15 precision points and a single count would hide that it
   was a choice.
4. **`POST-PROCESSING CONFIG`** — how detector output became these points:
   `run_detector`'s fixed heatmap stage, each checkpoint's own
   `thresh`/`eps`/`min_samples`/`eps_predict` from the run's
   `PREREGISTRATION.json`, and this session's live matching parameters.
5. **`GEOSPATIAL`** — what every layer shares: the geospatial precision/recall definitions, the
   clustering parameters, the recall-bound argument and the provenance.

`j`/`k` scroll. `[x]` exports the whole thing as a PDF.

### `[o]`/`[m]`/`[a]`: the label propagator

One adjudication should not stop at one frame. A **geospatial cluster** is a
DBSCAN group in UTM (`eps` 0.20 m) built from every frame's detections, so it
**spans frames by construction** — one target seen in frames 700 and 830 is
one cluster. A verdict you give in either frame is therefore evidence about
the same physical spot in the other, and that is the whole mechanism. Nothing
here re-clusters or re-projects; the pixel→cluster join is computed once
(`geo_metrics.locate` into a cached `member_label`) so each keystroke is dict
lookups.

**`[o]` chooses the tie-break convention**, cycling

```
off  →  positive_wins  →  majority  →  hard_negative_wins  →  off
```

and **starts off**. When a cluster holds both a `confirmed_real` and a
`hard_negative` — chaining at 0.20 m merges physically distinct spots — which
verdict wins is a *convention, not data*, and on this dataset it is worth ~15
precision points. So a reviewer who never presses `[o]` is never shown a
convention somebody else picked for them. The **reported metric is unaffected
either way**: `geo_prf()` still computes all three rules and every per-model
block still shows all three `POST` rows. `[o]` selects a rule for a different
consumer — what to *propagate*. See `GEO_METRICS.md`'s "there is no default
rule", which this preserves rather than contradicts.

`[o]` is unshifted: some OpenCV backends deliver the plain lowercase char to
`cv2.waitKey()`, so a `Shift`+letter binding would arrive indistinguishable
from the lowercase key.

**`[m]` rings each cluster** with the candidate label, in that judgment's own
colour — green for `confirmed_real`, red for `hard_negative` — so a ring is a
preview of exactly what `[a]` would write. The ring sits outside the centroid
dot and inside the yellow selection ring, so all three stay distinct.

**No ring means nothing propagates**, and that is the reading discipline of
the whole overlay: an unringed orange centroid means "no human has judged
this spot and `[a]` will not touch it", which has to be visually distinct
from "judged negative". A ring in the hard-negative colour is a claim;
absence of a ring is not. There are four reasons nothing propagates, and they
all draw identically because the reviewer's next action is the same in every
one — look at it yourself:

- no rule chosen (`[o]` is off);
- the cluster reaches no geo cluster (DBSCAN noise, or no resolvable pose);
- no human has judged anything in its geo cluster, or only abstentions have;
- the rule itself abstains — `majority` on a tie, where geometry stands.

A cluster straddling two geo clusters propagates only if they **agree**.
Disagreement is not a tie to be broken: the two verdicts are about different
physical spots that this cluster happens to bridge.

**Select a point** and the panel shows its geo cluster id, the vote tally, the
call under **each** of the three rules (the active one starred), and — most
importantly — the **reach**:

```
geo cluster 454  votes: confirmed_real=4
  reach: 18146 cluster(s) in 316 frame(s)
  propagates: *pos=real  maj=real  neg=real
```

### ⚠ Reach: how far one verdict actually carries

Read this before pressing `[a]`. Measured on the full 2,120-frame sample flight `2025_04_16/pieranch_rect`
against `results/17_holdout_pieranch`:

| | GUI clusters per geo cluster |
|---|---|
| median | 16 |
| p90 | 107 |
| **largest** | **18,146 — 29% of the whole review backlog** |
| 5 largest combined | **46% of the backlog** |

The median is the intended behaviour: one target seen in sixteen frames. The
tail is not. DBSCAN chaining at 0.20 m has merged large stretches of ground
into single components, so **a single verdict can propagate a label to a third
of a flight**. That is a property of the clustering, not of the propagator,
and it cannot be quietly mitigated — narrowing `eps` is a decision with its
own costs (see `GEO_METRICS.md`). What the GUI does instead is refuse to hide
it: the reach is on the selected point, in `[p]`, in the PDF and in a caveat,
so it is on screen *before* `[a]` is pressed rather than discovered after.

If you propagate on this flight, **count the verdicts, not the labels** — the
labels rest on very few independent decisions.

## Statistics reference: what every number means

There are **two** sources, and neither reads a precomputed file. Everything on
screen is computed live by this session over the run's `dets/`, `labels.txt`
and `flight_data.db`.

| Source | Computed by | Requires | Scope |
|---|---|---|---|
| Live HUD (bottom of side panel) | `review_gui` itself, live, every render | nothing — always on | the open directory, per checkpoint |
| `[p]` overlay / `[x]` PDF | `review_gui` + `geo_metrics`, live | nothing (geo needs `flight_data.db`) | the open directory, per checkpoint |

The geospatial arm is **always on** and built by a background worker. A
directory with no `flight_data.db` says so and changes nothing.

### The live HUD (always on, no flag needed)

Directory-scoped (it sums over every frame in the currently open directory,
not just the one on screen) and recomputed on every render, so it always
reflects your judgments up to the second.

- **Frame-position bar** (amber) — `(frame_pos + 1) / n_frames`. Pure
  navigation: how far through this directory's frame list you are. Says
  nothing about how much you've actually reviewed.
- **Review-coverage bar** (green) — `reviewed / total` **clusters**
  (`DirectoryReview.progress()`), i.e. the real remaining-click count, not
  raw detections. The `[N raw]` bracket next to it is `unmatched_total()`,
  the raw-detection count clustering compresses down from — shown for scale,
  not as a second progress metric.

### The metric table

`PRE` and `POST` are **rows of one table**, and the pixel and geo blocks share
the same column positions:

```
frame 812 (813/1355)                          <- pinned, never scrolls
[====================]
reviewed 90/168 clusters (54%)   [12345 raw]
[===============     ]
-- m2o 80/30px  geo eps 0.20m  match 0.35m  prop:positive_wins --  <- pinned
-- 030 t.04   healthy   44012d --        P        R        F1   <- body, scrolls
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
   ! OVERRIDE spatial 80->30                  <- footer, bottom-anchored
```

**Every metric row is per model.** There is no union scorecard: the union of
every active checkpoint's detections is what you *clicked*, which makes it a
**review-burden measure and nobody's precision**, and a union row would invite
exactly the comparison it cannot support. The union *matching* is still the
detection substrate and the judgment path; it simply has no scorecard.

The comparisons a reviewer actually makes — `PRE` vs `POST` within a space, one
space against another, and **one model against another** — are reading, not
arithmetic. That last one is the point of the whole layered feature: one
adjudication scores every checkpoint, and the scorecards have to sit in one grid
for that to be legible.

**Three zones.** The frame/review bars and **one params rule row** are
**pinned** and never scroll. That row is not a metric: it carries the live
values that decide every number below it — both spaces' matching parameters
plus `prop:`, the `[o]` propagation rule. It deliberately carries **no `P R F1` heads**:
`_draw_hud` fits each cell to the gap before the next cell *in its own row*,
so heads would cap the label at `HUD_COL_X[0] - 6` = 144px against a line that
is 290px at the reference setting, and it would be silently truncated. Nothing
is lost — the heads repeat on every model group's own rule row, which is
already the documented reason they repeat.

The per-model groups are the **body** and scroll with `[,]`/`[.]`, one model at a
time, sliced on group boundaries so a header is on screen if and only if all its
rows are — half a group is unreadable. The caveat row and the play line are a
bottom-anchored **footer**: a caveat you can scroll out of view is a caveat that
will be missed, which is the same reasoning that gives it priority 0 in
`status_segments`.

**The `P`/`R`/`F1` heads repeat on every group header.** Heads printed once at
the top are heads you have scrolled past, and riding them on the group's own
section rule costs no vertical budget (the separator is drawn in the gap *above*
the baseline). The header carries the layer's own palette colour so the HUD and
the model window's swatch speak one identity language; the metric rows keep the
`PRE`/`POST` arm colours, because arm and identity are different axes and
overloading one colour with both makes neither readable.

**A cell that has not been computed says so.** The geo arm is built by a
background worker, so a group can legitimately have its pixel numbers before its
geo ones; those cells read `computing...`. They never render `0.0000` — a zero
would be read as a measurement — and never render blank, because blank already
means "not applicable" (which is what geo's empty `F1` column means). The row
count and the labels are identical in both states, so nothing reflows under the
cursor when a result lands.

**Geo gets three `POST` rows, one per tie-break rule, and there is no primary
one.** When a DBSCAN cluster holds both a `confirmed_real` and a
`hard_negative`, which verdict wins is a convention rather than data — worth ~15
precision points here. Reading the three rows top to bottom *is* reading the
error bar; they are ordered most-positive first and that ordering is asserted.
See `GEO_METRICS.md`. Geo's `F1` column
is deliberately **blank** rather than collapsed: precision is per cluster and
recall per geotag, so their harmonic mean is a category error, and closing the
gap would put geo's `R` under pixel's `F1`.

Built by `hud_rows()` — a pure function returning `HudRow` records, no `cv2` in
it, so the smoke test asserts column alignment and the height budget on the row
model instead of scraping pixels. Cells are drawn at fixed offsets from
`HUD_COL_X`; Hershey is proportional, so space-padding one string does *not*
align columns.

**The budget is enforced by asserting what it protects, not by bounding a
total.** The HUD carries a **per-model** block — PIXEL `PRE`/`POST` plus GEO
`PRE` and one `POST` row per tie-break rule, for every active layer — so thirty
checkpoints are 30 × 118px of rows and there is no worst case to bound. The
body scrolls (`[,]`/`[.]`).

Two properties are asserted directly instead:

- **the zoom inset stays usable** — `panel_layout()` is run at import on the
  shortest supported panel (820px) and the inset must clear `INSET_FLOOR_H`;
- **the columns stay readable** — `hud_columns_clear(PANEL_FLOOR_W)` builds a
  worst-case group and checks with `getTextSize` that no cell runs into the next
  column. This is the property that actually matters and the one the height
  check could not see: `_draw_hud` fits each cell to the gap before the next
  stop, so an over-long cell is silently **truncated**, and `0.7911<=` arriving
  as `0.79..` is invisible to any assertion about heights.

`PANEL_HUD_H` is a *derived target* (`hud_chrome_height() + 2 ×
hud_group_height()`), with `PANEL_HUD_MIN_H` the floor of chrome plus **one
complete model group** — a HUD that cannot show one whole model is not worth the
pixels.

The matching parameters and the `P R F1` heads ride on the **section rule**
rather than taking lines of their own, for the same reason. Lines describing
parameters that cannot change while you review live in `[p]`'s `SUMMARY` block
instead. Only `op:` stays, because it changes what `[m]` paints.

- **`-- PIXEL  labels.txt m2o N/Mpx --`** — the section rule carries the two live
  values (`--spatial-threshold`/`--cluster-radius`, whatever the sliders are
  currently set to) that decide the matched/unmatched split and the review unit
  for every number below it. Same values the `[p]` overlay's `PARAMS` section
  echoes as its "live:" line.
- **`PRE`** — `naive_counts()` (raw tp/fp/fn from this
  directory's live matching) run through `prf()`. This is "if you stopped
  reviewing right now and called every still-unjudged unmatched detection a
  false positive" — the pessimistic baseline every judgment improves on.
- **`POST`** — `corrected_counts()` run through
  `corrected_prf()`. Precision uses `confirmed_real` judgments to move their
  *entire raw cluster* from FP to TP (matching what export actually writes).
  **Recall moves too**, and the `<=` is not decoration:

      R_POST = (tp0 + c) / (tp0 + fn0 + c + a)

  where `c` = `confirmed_real` raw detections (unmatched by construction) and
  `a` = `spawned_total()`. Recall is not pinned to the naive value because
  `POST` precision defines a TP as *(adjudicated positive) OR (matched to a
  label)*, which enlarges the positive class, and recall's denominator **is**
  the positive class — pinning it would score the two arms under two different
  definitions of "true positive".

  **`R<=` is an upper bound, not an estimate.** An unlabelled positive is only
  discoverable where the detector fired, so `c` can never correspond to a false
  negative and can only raise the ratio. Added markers (`a`) are the one
  observable slice of the misses nobody recorded, and the only term that pulls
  it back down — so with enough of them `R<=` can legitimately fall *below* the
  `PRE` recall. `PRE` stays the unbiased point estimate; report the pair. F1
  inherits the bound for the same reason recall does. Full derivation, symbol
  table and worked example in **`GEO_METRICS.md`**; the same treatment applies
  in geo space and the two formulas are asserted equal in the smoke test.
- **`+N TP reclassified, M hard-neg, K added`** — `N` and `M` are raw-
  detection counts (cluster-fanned-out, not click counts) of `confirmed_real`
  / `hard_negative` judgments so far; `hard_negative` does **not** change
  P/R/F1 at all (it was already counted as a naive FP and stays one — the
  judgment only confirms it's a true hallucination worth exporting to
  `hard_negatives.txt`, it doesn't reclassify anything). `K` is
  `spawned_total()`. Added targets **do** enter the accounting, in exactly one
  place: the `POST` recall **denominator**, one per click. They are targets the
  model never fired on, so they are false negatives the annotator missed too —
  and they are the only term that constrains `R<=` from above. (One per click,
  no dedup: `labels.txt` is per-frame and `naive_counts()` sums per-frame, so a
  burrow legitimately contributes one label per frame and a spawned point is
  exactly one per-frame missing label. Geo space *does* dedup, because there a
  burrow is one UTM target however many frames saw it.) The trailer spells this
  out as `-> R<= (+N c, +K a)`.
- **`> playing (N fps)` / `< rewinding (N fps)`** — only shown while video
  play/rewind (`[5]`/`[6]`/`[f]`, see above) is actually running; blank
  otherwise so the HUD's height never jumps between the two states.

### The geospatial block — per model

**This is a second metric space, not a variant of the pixel one.** The
`PIXEL PRE`/`PIXEL POST` rows score *pixel* detections against `labels.txt`.
The `GEO` rows score *UTM DBSCAN clusters* against the RTK geotag survey in
`flight_data.db` — the paper's geospatial precision/recall. The numbers are not
comparable and are not meant to be. Full detail in **`GEO_METRICS.md`**;
the short version:

- **`GEO PRE`** — geometry only, no adjudication. A cluster is a TP if a geotag
  lies within 0.35 m of one of its core points.
- **`GEO POST`** — your verdicts override geometry both ways: `hard_negative`
  forces FP, `confirmed_real` forces TP. `uncertain`/`removed` are abstentions
  and fall back to geometry; a cluster holding *both* verdicts resolves to the
  positive by default. This is the row the adjudication effort exists to fill.
- **Recall differs between the arms**, and `POST` recall is a **bound**:
  `R_POST = (n_hit + c) / (n_den + c + a)`, with `c` = confirmed clusters no
  geotag reached and `a` = spatially deduplicated added markers. `PRE` recall is
  geometry only; `POST` shows as `R<=` because unmarked
  targets are only discoverable where the detector fired. Same treatment
  `corrected_prf` gives in pixel space, and for the same reason.
- **Each arm gets two recall denominators**: all geotags, and geotags in view
  under the in-view polygon test (image bounds inset by the 81.5 px
  projection-noise buffer). The gap is ~4 points on the sample flight.
- **No F1.** Precision is per cluster, recall per geotag, and one cluster can
  absorb five geotags — their harmonic mean would be a category error.

The HUD carries the aligned table; the full block with per-rule counts, the
`c`/`a` breakdown and the provenance caveat lives in **`[p]`**. That split is
not cosmetic: the panel lays out bottom-up, so HUD height comes straight off the
zoom inset — paying for the three `POST` rows is what moved the `op:` line and
the `+c +a` trailer into `[p]`.

Three things to know before quoting a number:

1. **Conflicted clusters dominate the uncertainty in `POST`.** DBSCAN at 0.20 m
   merges spots 30 cm apart, so ~26 of 95 judged clusters here hold *both* a
   `confirmed_real` and a `hard_negative` — an artefact of the chaining, not a
   real disagreement. **There is no primary tie-break rule and no flag to pick
   one**: all three are computed and displayed everywhere, and the spread (~15
   precision points, and it reaches recall too, since `c` moves with the rule)
   *is* the result. Quote it as a range.
2. **`cov` moves with the sliders.** Judgment keys are float-exact centroids
   that shift with `--cluster-radius`, so read coverage next to the `match:`
   line, never on its own.

**It is always on.** Measured on the full 2,120-frame sample flight with 12
layers: union scaffold **1.74 s**, twelve
per-layer scaffolds **5.05 s**, twelve join caches **1.99 s** — about **8.9 s**
total, with **peak RSS 0.78 GB set entirely by the union build and flat across
every layer build**. So it is computed eagerly in a background worker that
publishes **incrementally**: the union lands at ~1.8 s and each layer as it
finishes, so the HUD fills in top-down instead of showing nothing for nine
seconds. Cells that have not landed read `computing...`.

**Each model gets its own scaffold, and it has to.** A union cluster is not any
single model's cluster: at eps = 0.20 m the union chains across all twelve
layers into 602 clusters where the layers separately produce 113–445 (2,890
summed). Precision is *per cluster*, so the clustering **is** the metric —
restricting the union's labels to one layer's members would score that layer
against a granularity it did not produce. The union's grid dedup also destroys
the attribution outright: 234,646 back-projections collapse to 125,239 points
and the survivor belongs to whichever layer sorted first.

**The union scaffold is still built, and it is what the propagator runs on.**
It has no *metric rows*, but the scaffold
itself is the only full-fidelity clustering there is — per-layer scaffolds are
deliberately reduced to a `LayerGeoSummary` (thirty live ones do not fit in
memory) and each carries 11–22% of its own points as DBSCAN noise. It is also
the *correct* substrate: one adjudication scores every checkpoint, so
propagation is model-agnostic. Degenerate checkpoints are excluded from it —
see "Degenerate models are excluded from the union" below.

What is shared and what is not:

- **The recall denominator is shared, exactly.** Geotags come from
  `proj.clicks` and in-view from `geotags_in_view(proj.poses)`, both functions of
  pose data alone — so all twelve layers score against the same 161 geotags and
  151 in view, and only `n_hit` moves (measured 74–131, against 143 for the
  union). That is the comparison this feature exists to make.
- **`a` is per model**, and this diverges from the pixel arm deliberately.
  `collect_added` drops a marker that falls inside a *core point of the scaffold
  it is given* ("the detector did fire there"), and that test is per-model by
  construction: a marker inside `014`'s cluster but outside every cluster of
  `030` is an undiscovered miss for `030` and not for `014`. Pixel space has no
  such spatial test, so it keeps `a` shared. Both numbers are carried.
- **`c`'s censoring argument does not survive across layers.** `post_recall`
  derives `R<=` from "an unlabelled positive is only discoverable where the
  detector fired" — but with twelve detectors, a consensus cluster a human
  confirmed that layer L never fired on is a *demonstrable* miss for L. This is
  the same defect `layer_metric_rows` already documents in pixel space, and it
  gets the same treatment: the conventional per-layer `R<=` stays the headline
  and `n_confirmed_elsewhere` is carried beside it, **never folded in**.

**Re-scoring is a dict lookup, not a re-clustering.** `geo_metrics.locate()` is
expensive per *frame* (it recomputes the frame pose and `to_3d` for each one), so
folding judgments in the obvious way costs ~0.075 s per layer — about a second on
every keypress across twelve. But the pixel → 3 cm cell → cluster mapping is pure
geometry and independent of every judgment, so it is computed once per layer and
retained; a commit then re-scores all twelve layers in **20 ms**. Each scaffold
is reduced to that join plus the ten fields `geo_prf` actually reads and dropped
before the next is built, so peak memory is one scaffold rather than N — which is
what keeps the 30-checkpoint screen viable. The smoke test asserts that scoring
from the reduced summary is **identical** to scoring from a live scaffold, which
is what pins that the reduction lost nothing and the join reproduces `locate()`.

A directory without a `flight_data.db` says so and
changes nothing. `[s]`/`[q]` writes `geo_metrics.json` beside
`review_log.jsonl`.

### `[x]` the PDF report

`[x]` writes **`review_report.pdf`** next to `review_log.jsonl` — which for a
layered review means `<run>/review/<flight_slug>/`, not the flight directory,
so two runs reviewing the same flight never overwrite each other's report.
Built by `review_report`. It exists for the reason `geo_metrics.write_json`
states outright: *anything the GUI puts on screen must also land on disk*.
`geo_metrics.json` honours that for one metric space; this covers what it does
not — the run parameters, the input digests, the cross-space comparison, and
above all the caveats.

| page | contents |
|---|---|
| 1 | Title, directory, reviewer, timestamp, the exact command, input digests, detection-source note, the adjudication composition and the active propagation rule |
| 2 | **Data & methods** — the two-metric-space table, how label propagation works, the two arms, the `R_POST` bound derivation, and `geo_metrics.DEFINITIONS` verbatim |
| 3 | **Run parameters** — the post-processing config (detector heatmap stage, each checkpoint's own `thresh`/`eps`/`min_samples`/`eps_predict`, the geospatial parameters), then every argparse flag with non-defaults marked |
| 4 | **Results** — one block per checkpoint: pixel `PRE`/`POST`, geo `PRE` + one `POST` row per tie-break rule, the raw geotag counts, then the propagation state, its reach table and the adjudication composition |
| 5 | **Figures** — the PR-plane `PRE`→`POST` scatter (health-coded, one marker per checkpoint), per-checkpoint pixel precision bars, geo cluster-size histogram, and how each call was arrived at |
| 6 | **Caveats** — read before quoting anything above |

**The PR-plane figure** uses a health coding that is load-bearing, not
decoration — healthy green circle, sparse orange triangle, collapsed red
cross: on an unlabelled scatter
the degenerate checkpoints land at the **top-left**, at near-zero recall with
the *highest* precision in the field, which is exactly the position a reader
reads as "conservative and precise". `LayerSpec.verdict` already carries the
screen's health call, so no new data source is needed.

Each model is a filled `PRE` marker with an arrow to its most-positive POST.
POST is **not a point**: it is three, one per
tie-break rule, which `GEO_METRICS.md` forbids collapsing — so the arrow ends
in a short segment spanning all three. That segment is a *convention*, not
measurement error.

Design points worth knowing:

- **It never recomputes.** `collect_context()` snapshots values the HUD and
  `[p]` already computed, from the same memoised accessors. A report that
  recomputed could disagree with the screen it claims to transcribe.
- **There is no union scorecard in it**, matching the screen. Every precision
  and recall figure is per checkpoint, and a caveat says so — a reader who
  expected a union number would otherwise assume these were it.
- **Input provenance is recorded as SHA-256 file digests.** They are sha256
  prefixes, so they compare directly against a `run_manifest.json`.
- **Figures are optional and degrade.** `matplotlib` is imported lazily inside
  the figure builder; if it fails, the report still builds with every table and
  a note in place of page 5. The smoke test asserts this by blocking the
  import outright. Every figure restates something already tabulated, so
  losing them costs the reader nothing but convenience. `review_gui` has no
  module-scope import of `reportlab` or `matplotlib`.
- **Never fatal.** An export that raised would kill a session mid-review over an
  optional artifact, so failures route through `_report_loop_error` and surface
  in the status line.

### `[p]` overlay fields

Section order is one declarative expression at the bottom of
`_stats_overlay_lines()` rather than an emergent property of which `append`
runs first, and the smoke test asserts it: **results first, then provenance,
then configuration.** `_draw_paginated` shows ~38 lines at a time, so whatever
is first is what most readers ever see.

Nothing here is read from a file, and nothing is recomputed for the overlay:
every value comes from the same memoised accessors the HUD reads, so the two
cannot disagree.

- **`SUMMARY`** — directory, run, reviewer, review coverage in clusters, layer
  count, the active propagation rule, and the live caveats (parameter
  overrides, held-out contamination, conflicted geo clusters, judged
  detections that reach no cluster).
- **`PER MODEL`** — per checkpoint: `PIXEL PRE`/`PIXEL POST`, `GEO PRE` and
  one `GEO POST` row per tie-break rule, `n_dets`, how many clusters the layer
  fired in, `c`, `hard_negative`, and `n_confirmed_elsewhere` when non-zero.
  Then **the raw counts behind the geo ratios** — `geotags hit 131/161`,
  in-view under the in-view polygon test, missed, cluster count, `a`,
  coverage: a ratio without its denominator is not quotable.
- **`ADJUDICATION`** — the log's composition by verdict × *how the judgment
  was arrived at*: `judged directly`, `propagated:<rule>`, or `legacy
  classifier` (rows with source=classifier were written by an automated
  classifier, not a human; precision over them is not an independent check).
  Then **`PROPAGATION COVERAGE`**: geo clusters judged, conflicted, unresolved; the
  reach distribution; and how many labels **each of the three rules** would
  propagate on this frame, because the spread between them is the uncertainty.
- **`POST-PROCESSING CONFIG`** — the detector's fixed heatmap stage from
  `run_detector`, this session's live matching and geospatial parameters,
  and each checkpoint's own `thresh`/`eps`/`min_samples`/`eps_predict` read
  from the run's `PREREGISTRATION.json`, so a report never quotes a
  checkpoint's precision without saying at what threshold it fired.
- **`GEOSPATIAL`** — what every layer shares: the clustering parameters, the
  detection-source note, `geo_metrics.DEFINITIONS` walked in full, and the
  recall-bound argument.

### Cheat sheet: which number answers which question

| Question | Look at |
|---|---|
| How far through this directory am I? | HUD frame-position bar (amber) |
| How much of this directory have I actually reviewed? | HUD review-coverage bar/%, in clusters (green) |
| What does my review imply about **this checkpoint's** precision/recall in pixel space? | its HUD group's `PIXEL PRE` (point estimate) / `PIXEL POST` (adjudicated; recall as `R<=`, an upper bound) |
| ...and in geospatial metric space, against the RTK survey? | its `GEO PRE` and the three `GEO POST` rows — read all three as a bracket |
| How many geotags did this checkpoint actually find? | `[p]`'s `PER MODEL` block: `geotags hit N/161`, in-view under the in-view polygon test |
| Which checkpoint should I prefer? | the `[x]` PDF's PR-plane figure — and check the health coding before reading the top-left as "precise" |
| At what threshold did these detections get produced? | `[p]`'s `POST-PROCESSING CONFIG`, or page 3 of the PDF |
| Would a verdict here carry to other frames, and how far? | select the point: its geo cluster, vote tally, all three rules' calls, and the **reach** |
| How many of my adjudications were propagated rather than made directly? | `[p]`'s `ADJUDICATION` block, split by convention |
| Which tie-break convention am I proposing under? | the HUD's pinned params row, `prop:` |

Every number above is this session's own.

## Using this before a full-dataset run or hyperparameter sweep

This tool's actual value ahead of a big automated run is as a **visual
sanity check on cheap, targeted samples**, not a replacement for the run's own
evaluation:

- Open a handful of representative directories with the sliders at your
  candidate `--spatial-threshold`/`--cluster-radius` and confirm the
  matched/unmatched split and clustering *look* right on real frames before
  running either against the full dataset or a wide hyperparameter grid.
- Don't use the HUD's `PRE`/`POST` numbers as a stand-in for a sweep result —
  they use this tool's own many-to-one matching (above), which a sweep using
  one-to-one assignment would not reproduce. Read that run's own artifacts
  with its own reporting script.
- Read the **per-model** groups against each other, which is what one
  adjudication pass buys you, and read the `[x]` PDF's PR-plane figure before
  preferring a checkpoint — with attention to the health coding, because the
  degenerate ones occupy the corner that looks best.

## What is deliberately *not* adaptive

A review session outlives the files it reads. Worth knowing, because the
failure is silent:

- **`dets/` / `labels.txt` for a directory already loaded.** These are
  parsed once per directory load and cached (`dets_fd`/`labels_fd`); even a
  slider recompute re-feeds the cached parse rather than re-reading disk. If
  a model run regenerates detections under you, the frames you are looking
  at keep showing the old ones. Restart to pick them up. Note also that
  judgments are keyed on cluster centroids, which are a function of the
  member coordinates — so regenerated detections shift centroids and orphan
  prior judgments on those clusters.
- **Newly created flight directories.** Discovery runs once, before the app
  starts; a directory that appears mid-session will not show up.
- **The startup image-filename index**, used as the stale-path fallback.
  Images added mid-session resolve only if their recorded path exists
  verbatim. This one at least fails visibly ("image could not be loaded").

### Concurrent writers

`review_log.jsonl` is read once, at directory load. The three derived
sidecars are regenerated from *that in-memory snapshot* with a truncating
write. So if another process appends judgments to the same log — a second
review session or a scripted reviewer — **its
judgments are erased from `hard_negatives.txt` /
`confirmed_detections.txt` / `added_targets.txt` on this session's next
keystroke**, with no warning and without being counted as orphaned.

The JSONL itself is append-only and survives, so nothing is permanently
lost and a restart recovers it. But the `.txt` files are what a training
run consumes, so they can be wrong in the meantime. **Avoid reviewing the
same directory from two places at once**, and prefer a restart over a long
session if another tool has been writing.

## Running it

**Two positionals: a flight directory and a `results/<run>/` directory.**
The detections come from the run's `dets/` and the matching threshold is
derived from its `run_manifest.json`, so the two cannot drift apart. The
examples below are run from the root of the extracted sample.

```bash
# One flight against one run. Matching uses the run's own spatial_threshold
# when its run_manifest.json carries one, not the GUI default.
birdseye-review 2025_04_16/pieranch_rect results/17_holdout_pieranch

# Run only: review every flight that run covers, [3]/[4] to move between them
birdseye-review results/17_holdout_pieranch

# Order does not matter -- the two are told apart by CONTENT (a run has
# run_manifest.json, gui/, PREREGISTRATION.json or dets/; a flight has labels.txt)
birdseye-review results/17_holdout_pieranch 2025_04_16/pieranch_rect

# Write nothing (renders/navigates normally, all writes suppressed)
birdseye-review --dry-run 2025_04_16/pieranch_rect results/17_holdout_pieranch

# Override a derived parameter. Works, but is recorded as an override in
# review_log.jsonl and shown on the HUD, in [p] and in the PDF report.
birdseye-review 2025_04_16/pieranch_rect results/17_holdout_pieranch \
    --spatial-threshold 30
```

Deriving parameters from the run directory rather than from hand-passed flags
means a flag that does not belong with the run cannot silently change what a
number means. A flight the run does not cover is **refused** (with the covered
list printed) rather than rendering empty overlays, and reviewing the held-out
test directory against a run that lacks `scores_are_honest_for_test_dir`
prints a contamination warning at startup and in `[p]`.

Derived parameters can still be overridden by flag. Overriding a
metric-bearing parameter is stamped on the HUD, in `[p]`, in every
`review_log.jsonl` record and on the report's parameters page. GUI-only
preferences (`--cluster-radius`, zoom, click tolerance, `--reviewer`,
`--dry-run`, display size) are plain flags — they change what you see, not what
a number means.

Dependencies: a desktop `opencv-python` build, `numpy`, `scipy` and
`scikit-learn`; `reportlab` (and optionally `matplotlib`) for the `[x]` PDF.
The window auto-fit described below additionally uses `tkinter` (stdlib) if
available, falling back to a fixed size if it isn't.

**Window sizing**: the window opens auto-fit to your monitor and is fully
resizable afterward (drag an edge/corner — the content re-fits live, so
there's no "wrong aspect ratio, stuck that way" problem). Pass
`--display-width`/`--display-height` only if you want a specific *starting*
size instead of the auto-detected one; the window stays resizable
regardless.

## Keybindings

| Key | Action |
|---|---|
| left-click | on the frame: select the nearest point (cluster centroid or added target). **On a layer-legend row: toggle that layer.** |
| right-click | on the frame — near a detection: fast-confirm its cluster real · near an added target: select it · on empty ground: add a new one. **On a layer-legend row: solo that layer.** |
| `space` | select the next unreviewed cluster in this frame |
| `c` | commit **confirmed real** on the selected cluster (all its raw detections), auto-advance |
| `h` | commit **hard negative** on the selected cluster (all its raw detections), auto-advance |
| `u` | commit **uncertain** on the selected cluster, auto-advance |
| `r` | reset the selected cluster to unreviewed, **or** remove a selected added target |
| `ESC` | deselect / back to overview |
| `1` / `2` | previous / next frame |
| `3` / `4` | previous / next directory |
| `5` / `6` | toggle rewind / play (continuous, stops at the directory's first/last frame — see below) |
| `f` | cycle play/rewind speed |
| `n` / `b` | jump forward/back to the next frame (crossing directories) with an unreviewed point |
| `s` | save now (writes are already immediate on every commit; this just re-exports on demand) |
| `?` | toggle this keybinding list on-screen |
| `p` | toggle the stats overlay: `SUMMARY`, `PER MODEL`, `ADJUDICATION`, `POST-PROCESSING CONFIG`, `GEOSPATIAL`. Needs no files. `j`/`k` scroll |
| `x` | **export the whole `[p]` panel as a PDF** to `review_report.pdf` beside `review_log.jsonl` (see below). Honours `--dry-run`. Unshifted, like every binding here |
| `o` | cycle the **label-propagation tie-break rule**: off → `positive_wins` → `majority` → `hard_negative_wins` → off. **Starts off** — nothing is proposed until you choose a convention (see below) |
| `m` | toggle the **propagation overlay**: a ring on each cluster showing the candidate label under the current `[o]` rule, in that judgment's own colour. **No ring means nothing propagates** |
| `a` | **apply this frame's propagated labels** as adjudications (see below). Press once to preview, again to commit; `[o]` cancels a pending confirmation |
| `z` | undo the last `[a]` — resets exactly that batch to unreviewed |
| `v` | cycle the **3D flight-pose view**: off → inline (under the frame) → its own window → off (see below; needs `flight_data.db` in the flight directory) |
| `g` | *3D view only:* switch between the whole-survey view and following the aircraft |
| `+` / `-` | *3D view only:* zoom in / out. `ctrl`+left-drag or middle-drag does the same, dragging down to pull back. **Not** the mouse wheel — see below |
| `d` | *3D view only:* toggle the whole-directory detection cloud (see below) |
| `l` | model layers on/off — and, when a checkpoint has finished on disk mid-session, fold it in |
| `e` | open / close the **model window** — one click-toggle row per checkpoint in the run's `dets/`. Left-click a row toggles that model, right-click solos it, and `[ALL]`/`[NONE]`/`[ACC]`/`[SORT]` sit along the bottom. Its own OpenCV window, sized to its content (420x592 fits all 30 checkpoints with no scrolling); closing it with the window manager is not a reason to end the session |
| `,` / `.` | scroll the HUD's per-model `PRE`/`POST` table up / down, one model at a time |
| `i` | cycle the accumulation tier: off → per-layer. (Above 6 visible layers it turns itself off — more than a handful of translucent hues over grass is not a readable picture) |
| `;` | toggle the focused layer |
| `'` | solo the focused layer (again restores the previous set) |
| `[` / `]` | move layer focus up / down |
| `0` | all layers on |
| `q` | save and quit |

`g`, `d`, `+` and `-` do nothing while the 3D view is closed. They are global rather
than focus-scoped because `cv2.waitKey()` reports the keystroke but not which
window had focus — which is also why the 3D view's mode switch is `g` and not
the more obvious `f` (already the play-speed cycle). Unlike every other key,
they do **not** stop playback: watching the flight play back in 3D is a main
reason to open the view, so adjusting it mid-playback must not end it.

Auto-advance stops at the end of the frame ("frame complete") rather than
silently crossing into the next one — use `n`/`1`/`2` to move on
deliberately.

## Model layers — reviewing every checkpoint at once

Screening 30 checkpoints found two things that make one-model-at-a-time review
the wrong instrument: **19 of the 30 are degenerate**, and **no single number
picks a winner** — validation flights disagree, F1 hides collapse, and a dead
model ranks as the most precise one in the whole screen.

So: point the GUI at a run directory that holds per-checkpoint detection files and
every model is drawn at once.

```bash
# the held-out confirmation run: 12 checkpoint/threshold layers over the sample flight
birdseye-review 2025_04_16/pieranch_rect results/17_holdout_pieranch

# the run's dets/ are the ONLY detection source, so a run directory is
# required -- a flight the run has no dets/ for has nothing to review and
# says so by name.
```

Two on-disk shapes are read, `<run>/dets/val_<ckpt>[_t<thresh>].txt` and
`<run>/<flight_slug>…/val_*.txt`. A `.partial` file is refused — `run_detector`
streams into it and only renames on success, so reading one would score a truncated
file as a low-detection model.

### Adjudicate once, score eleven

**The review unit is the CONSENSUS CLUSTER across all layers**, not one cluster per
model. You judge each physical spot once; the record stores which checkpoints fired
there, and every model's PRE/POST precision and recall is derived from that single
pass. On the full sample flight one click updates eight of twelve scorecards.

Clustering runs over **all** layers regardless of what is switched on, so toggling a
layer is purely a display change — it can never move a centroid and therefore can
never re-key or orphan a judgment.

There is no union PRE/POST row on the HUD. Union precision pooled over twelve
checkpoints is a **review-burden** measure, not any model's score, and a reader
would take it for a result. The union counts still exist and drive the sidecar
export; they have no scorecard.

### Degenerate models are excluded from the union

A **collapsed** or **sparse** checkpoint contributes **no review targets**.
It creates no cluster you have to click, moves no centroid, and
adds no link to a DBSCAN chain. On a 30-checkpoint screen 19 of 30 were
degenerate, and a collapsed model firing 34,817 times on one flight would
manufacture review burden out of a model already known to be broken.
Measured on the full 2,120-frame sample flight, forcing four of the twelve
layers collapsed eliminated **42,102 of 62,867 review clusters — 67%**.

**It is excluded from the union and from nothing else.** "Show all data across
all models" is a standing requirement here, and a collapsed checkpoint held
the *highest* precision-like score in the whole screen — it must stay visible
and labelled, never quietly disappear. So:

- its **per-model scorecard is untouched** — its own PRE row, its HUD group,
  its point in the health-coded PR figure, its legend entry. Verified: forcing
  layers collapsed leaves every layer's PRE precision and detection count
  bit-identical.
- `dr.dets_fd` still holds its detections, so Tier 2 still draws them, the 3D
  cloud still shows its shape, and its own geo scaffold is still built.
- its POST arms will simply **equal its PRE**, because none of its detections
  is in a cluster anyone can adjudicate. That is the honest answer — nobody
  reviewed them — and `n_confirmed_elsewhere` already reports the confirmed
  clusters a layer never fired on.

One definition, `healthy_union_frames()`, is shared by the review-pool pass
and the geo scaffold, so the two cannot disagree about what "the union" holds.
If every checkpoint in a run is degenerate the union is empty and the GUI
**says so by name** — silence there would look identical to a fully-reviewed
flight.

### Where judgments are written

`<run>/review/<flight_slug>/review_log.jsonl`, plus the usual `.txt` sidecars and a
per-checkpoint breakdown under `by_layer/<layer_id>/`. **The flight directory's own
`review_log.jsonl` and sidecars are never read and never written** — a run's
adjudications cannot corrupt another review of the same flight, and reviewing the
same flight under a different run gives an independent set.

Records carry `schema_version: 2` and a `layers_fired` / `layer_counts` /
`layer_conf` triple. Layer **ids** are persisted, never bit positions — those are
session-local and shift when a file appears. The union sidecars are deduped on
`(frame_idx, x, y)`: on the full 2,120-frame sample flight, 7,738 of the 224,758
detections across eleven layers are exact coordinate coincidences between
checkpoints, so without dedup a training pool would carry 3.4% phantom
duplicates — and precisely of the easy, unanimous detections.

### What is drawn

Two tiers per layer:

| tier | what | why |
|---|---|---|
| **accumulation** | every detection of that layer, back-projected through *its own* frame's pose and reprojected into the current frame, clipped. Faint 1px. | Any frame shows the whole flight's findings. The `dets/` files are raw per-frame detections, so the view has to be rebuilt. |
| **this frame** | that layer's own detections on the frame you are looking at. Larger, opaque, shaped. | The thing you are actually reviewing sits in front of the accumulated context. |

**The accumulation tier smears, by construction.** A target seen in 200 frames
contributes 200 points, each landing on that frame's own ground plane. It is not
a substitute for `geo_eval`'s 3D clustering.

It needs a `flight_data.db`; without one the legend says so and only the
current-frame tier draws. If no boresight campaign covers the flight date the tier
still draws but is stamped approximate on the legend, the HUD and the status bar.

### Colour, shape, and health

Identity is **hue × shape**, in a pale low-chroma band that no judgment colour
occupies — a layer marker must never read as green (`confirmed_real`) or red
(`hard_negative`). The separation is *enforced at import*, not commented: a module
assertion requires every layer colour to clear ΔE 22 against every semantic `COLOR`
entry, and same-shape layers to clear ΔE 8 against each other.

Be honest about what that buys. Five of the eight Okabe-Ito colourblind-safe anchors
survive the collision filter; up to ~12 layers hue and shape distinguish them on
sight, and **at 30 layers they group rather than identify — the legend is the
authority.**

Health is a modifier, never a hue:

| glyph | verdict |
|---|---|
| filled | healthy |
| hollow | `SPARSE` — genuine positional variety, too few points to cluster |
| hollow + struck through | `COLLAPSED` — a near-constant output pixel |

**Degenerate checkpoints are always drawn, always labelled, never filtered.** The
collapsed group holds the highest `prec_like` anywhere in the 30-checkpoint screen,
so sorting on a metric is not enough to keep them out of the way either — they sort
last by an explicit health key, and the smoke test asserts every layer stays
reachable by scrolling.

Above **6** visible layers the accumulation tier turns itself off — eleven
translucent hues over grass is not a readable picture. Soloing a layer restores it.

### The panel, and the model table

The side panel is laid out by **one pure function**, `panel_layout()`, which owns
every y-coordinate in it and hands each block a band. Nothing else does panel
arithmetic.

The model table is not in the panel: the panel's geometry does not depend on
how many checkpoints exist, so a 30-layer run and a 1-layer run lay it out
identically. The list of checkpoints is its own window (`[e]`, see below) and
their numbers are in the HUD.

Blocks get floors first, then surplus in a fixed order — info, the HUD to **one
complete model group**, the zoom inset to its target, the HUD to `PANEL_HUD_H`,
the inset to what it can actually use, then the HUD takes whatever is left. The
HUD is guaranteed one whole model before the inset's target and a second one
after it, and that ordering *is* the argument: **model groups three onward are
one keypress away at zero pixel
cost, whereas inset pixels are recoverable by no keystroke at all.** So the inset
wins the middle of the order and the HUD takes the tail — which is why a tall
window shows many models without ever starving the inset. Measured with a
realistic inset ceiling: 2 groups at 1000px, 6 at 1440px, 10 at 2000px, with the
inset holding ~400–425px throughout.

When even the floors do not fit, blocks are given up along `PANEL_SHRINK_ORDER`
(`("info", "hud", "inset")`) and the layout records which in `dropped`, so a
test can assert *which* degradation happened rather than merely that nothing
overlapped. One more degradation is recorded: when the band cannot hold a single
whole group, `hud_groups` is 0 and the HUD says
`models: window too short -- [e] for the model list` instead of drawing half a
group.

`smoke_test_gui.check_panel_layout` sweeps 320–2000px × four HUD wants × 0/2/8
info lines × selection — 5,784 layouts — asserting the bands are ordered and
disjoint, that the HUD band is always chrome plus a *whole* number of groups, and
that degradation follows the declared order.

### `[e]` the model window

One click-toggle row per checkpoint in the run's `dets/`, in its own OpenCV
window: swatch, shape glyph, full `layer_id`, health spelled out
(`healthy`/`SPARSE`/`COLLAPSED`), detection count, and `[x]`/`[ ]`. Left-click
toggles, right-click solos, and `[ALL]` `[NONE]` `[ACC]` `[SORT]` sit along the
bottom. `model_window_size(30)` is **420×592**, so thirty checkpoints are on
screen at once with no scrolling — an in-panel table could not show that many
without taking the pixels off the zoom inset.

It is `cv2.WINDOW_AUTOSIZE`, not `WINDOW_NORMAL`, and that is the one deliberate
divergence from the 3D view's pattern. The 3D window is resizable because
orbiting wants a viewport and a drag has no hit boxes; this window is *entirely*
hit boxes, and AUTOSIZE pins it to its canvas so the mouse coordinates Qt reports
**are** the coordinates the rows were laid out in. No `getWindowImageRect`
rescale means no second source of truth about where a row is — which is how a
click lands on the row above the one clicked. Closing it with the window manager
is not a reason to end the session, and every piece of layer state lives in
`LayerState`, which is entirely indifferent to whether a window exists.

The widget carries no metric columns: the per-model numbers are in the HUD, on
the HUD's own grid, so there is no second grid to keep aligned with the first.

**One ordering, one owner.** `model_display_order()` is a pure function returning
layer ids, `App._sync_layer_order()` calls it once per render, and
`LayerState.set_order()` stores the result — so the window, the HUD's per-model
groups, `[`/`]`, the scroll clamp and every hit box all address the same list.
A second ordering would make `[`/`]` step through a different sequence than the
screen and the follow-scroll jump to an unrelated row.

**`LayerState.ids` stays in disk order for ever.** It is the *index space*:
`render_key()` builds the static composite's bitmask from `enumerate(self.ids)`
and `cluster_layer_counts` tuples are indexed by layer position, so re-sorting it
would re-key the composite cache every tick and mis-attribute per-layer counts.
`set_order()` asserts its argument is a permutation of `ids` and nothing more.

`[SORT]` cycles `f1 → recall → precision → dets`. There is no arm selector: the
HUD shows *both* arms for every model.

**A plain left-click toggles a layer and does not move the focus.** Focus moves
with `[` / `]` and with an explicit solo.

### Which models fired here

Select a cluster and the same table answers it: layers that fired get a bullet in the
gutter and their detection count and confidence in place of the frame count, silent
layers dim to `--`, and fired layers **partition stably to the top**. On 30 layers the
answer would otherwise sit below the fold.

The one-line `agreement = n/m healthy models fired` is in the selected-point
info. **The denominator is healthy layers only** — a collapsed model's silence
is not evidence and its firing is not agreement — while degenerate layers are
still listed, tagged and counted separately.

### The legend, and the panel budget

Left-click a legend row to toggle, right-click to solo, and the header
carries `[ALL] [NONE] [ACC]`. One predicate (`App.layer_widget_rect`) owns where the
widget is and both the renderer and the hit test call it, so a click cannot land on
the wrong row; the smoke test round-trips every control and asserts no two hit boxes
overlap.

The HUD is measured with `hud_rows_height()` and painted at exactly that height.

### One number this feature creates, deliberately not folded in

When a consensus cluster is confirmed real and a layer never fired there, that layer
has a **demonstrably discoverable miss** — something impossible in single-source
review, where `corrected_prf` reasons that "an unlabelled positive is discoverable
only where the detector fired" and the `c` term can therefore only raise the ratio.
That reasoning is false across layers.

Rather than silently redefine `R<=`, the displayed bound stays the
project-conventional one and the count of clusters this layer missed is carried
beside it as `n_confirmed_elsewhere`, from which the tighter cross-layer bound

    (tp0 + c) / (tp0 + fn0 + c + a + n_confirmed_elsewhere)

is available as its own labelled quantity. Two numbers, both honest; neither quietly
replaces the other.

### Layers that finish while you are reviewing

A holdout run writes one detection file per checkpoint over several hours, so layers
appear mid-session. The poller **reports and never reloads**: a new layer's points
can merge two previously-distinct consensus clusters, which would move every centroid
in every frame it touches and re-key judgments under your cursor. It says so on the
HUD's existing caveat row and in the status bar; `[l]` folds them in when you are
ready.

## `[a]` apply this frame's propagated labels

With a rule chosen via `[o]`, the `[m]` overlay already rings each cluster
with the label a human verdict elsewhere implies for it. `[a]` adopts those
rings as adjudications for the **current frame**, so a frame of thirty obvious
clusters is two keystrokes instead of thirty.

What it does, precisely:

- every still-**unreviewed** cluster on the frame that has a propagated label
  is committed to that label;
- a cluster you already judged is **never** overwritten — the judgment made
  *on* a cluster outranks one deduced *about* it, and clobbering one is the
  single thing that would make `review_log.jsonl` untrustworthy;
- a cluster with **no** propagated label is **skipped, not guessed at**
  (`propagated_label()` refuses to invent a call, and this respects that).
  Early in a review that is the majority case, because few geo clusters carry
  a verdict yet;
- the first press previews the counts, a second within ~6 s commits. Any other
  key, a frame change, or an **`[o]` rule cycle** cancels it — committing under
  a convention you just moved away from is exactly the surprise the two-press
  arm exists to prevent;
- `[z]` resets exactly that batch back to unreviewed.

It refuses, with a reason, when no rule has been chosen or the geospatial
clustering has not finished.

### Provenance — read this before quoting a precision number

Every row `[a]` writes carries, in `review_log.jsonl`:

```json
{"judgment": "confirmed_real", "source": "human",
 "propagated": {"rule": "positive_wins", "geo_clusters": [454],
                "votes": {"confirmed_real": 4}, "n_members": 7}}
```

**`source` is `"human"`, and that is not a convenience.** Every input to a
propagated label is a human verdict; the only machine involvement is the
DBSCAN geometry deciding which verdicts are about the same physical spot.
There is no model in this loop at all, so a `"source": "classifier"` stamp
would be a false provenance claim here.

**The `propagated` block is recorded anyway, and it is the part that matters.**
Which convention produced a label is the ~15-precision-point choice
`GEO_METRICS.md` refuses to let anyone quote blind, and a row without it could
not be re-derived or re-scored under a different rule later. `votes` carries
the evidence, not just the conclusion.

**When quoting a number over propagated labels, count the verdicts, not the
labels.** See the reach table above: on this dataset one verdict can carry to
29% of a flight, so a large propagated population may rest on very few
independent decisions. `[p]`, the PDF and a caveat all say so.

Judgments you make yourself carry `"source": "human"` and no `propagated`
block. Records with no `source` field at all are human by construction — so
the correct filter for "made by a person, on this cluster" is
*absent-or-`"human"` **and** no `propagated` key*. Rows with
`"source": "classifier"` were written by an automated classifier, not a human;
precision over them is not an independent check, and the selected-point panel,
`[p]` and the PDF label them as such.

## `[v]` the 3D flight-pose view

`pose3d`, showing where the aircraft actually was, streamed as you scrub
frames or play the flight back:

| element | source |
|---|---|
| ground grid | 10 m, on the current frame's ground plane (`camera_z − radalt`) |
| camera track | the whole flight; bright behind the current frame, dim ahead of it |
| current pose | a frustum: four rays from the projection centre to the footprint corners |
| footprint | the frame's image corners ray-cast to the ground — what the camera sees now |
| detections | this frame's cluster centroids back-projected, in their judgment colours |
| **detection cloud** | **every *other* frame's detections too — see below** |
| geotags | the `clicks_flight_data` survey as ground markers |

Drag to orbit, right-drag to pan, `ctrl`+left-drag (or middle-drag, or
`+`/`-`) to zoom, `[g]` to switch between the whole-survey view and following
the aircraft.

**The mouse wheel is deliberately not used.** This OpenCV is built against Qt5,
and its window's own viewport (`DefaultViewPort::wheelEvent`) zooms the whole
displayed image on a wheel event. It forwards the event to our callback *and*
scales the view, and a `setMouseCallback` handler receives events without
consuming them — so there is no way to claim the wheel for the 3D camera.
Handling it anyway zoomed two things at once. Worse, the Qt viewport's zoom
silently changes the window-to-image mapping that every click on the frame
picture depends on, so a stray scroll would put your detection clicks off
target. Zooming the 3D view by drag or key removes any reason to scroll over
the window at all.

### `[d]` the whole-directory detection cloud

By default the view also shows **every detection in the directory**, not just
the frame on screen — each one back-projected through *its own* frame's pose,
drawn translucent, **one colour-keyed population per model layer** in the same
palette the 2D per-layer tiers use.

Points landing within **0.35 m of a survey geotag** — `geo_eval`'s own
`EPS_PREDICT_DETS`, so "hit a geotag" means here what it means in the
geospatial metrics — are drawn slightly heavier, in the same colour. The HUD reports the
totals.

Layer visibility is a bool mask the scene applies **after** projection, not a
slice of the points, so toggling a layer never invalidates the projection
cache — which is what makes a toggle free in the 3D view as well as the 2D one.

`[d]` toggles the layer off; it costs about 7 ms of the ~25 ms render tick.

**Read the cloud knowing what it is.** A target seen in 200 frames contributes
200 points, each landing on that frame's own ground plane, so the cloud
**smears** rather than converging to one dot per target. That is a property of
per-frame ground-plane back-projection, not a defect in the view. It is
emphatically not a substitute for `geo_eval`'s 3D clustering.

### Where it draws: inline, then its own window

`[v]` cycles **off → inline → its own window → off**.

**Inline** puts the view in the band under the frame image. That band is dead
space, not something taken from the review view: `canvas_h` is
`max(window_h, panel_width)` while the image is letterboxed to its 1920×1200
aspect, so widening the side panel left a permanently black strip below it.
The image is never resized to make room. Typical size is ~900–1100 × 225–350,
with the bottom 48 px still reserved for the status bar.

If the window's aspect leaves no usable band — under 180 px, where the 3D HUD
alone needs ~66 px — `[v]` **skips inline and opens the separate window**,
saying so in the status line, rather than presenting a squashed strip. If you
resize the window until the band disappears while inline, the view moves to its
own window on the next frame instead of vanishing.

Being drawn inline also makes the view structurally frame-accurate: it is
composited inside `App.render()`, in the same pass that advances playback, so
it cannot lag the image above it.

### Playback

During `[5]`/`[6]` playback the 3D camera **follows the aircraft**, and your
previous framing is restored when playback stops. Without this the motion is
invisible in the whole-survey view — one frame moves the aircraft about 0.4 m
across a ~400 m survey, well under a pixel. Pressing `[g]` during playback
cancels the automatic follow for the rest of the run, so an explicit choice
always wins.

Two other things make the advance readable: the current pose is drawn at a
fixed **screen** size (with a drop line to the ground plane, so its position on
the grid is unambiguous at an oblique angle), and the last ~30 frames of track
are drawn as a brighter, thicker **comet tail**.

### View scaling

Focal length is derived from the viewport's **width** (a fixed horizontal
field), so the wide, short inline strip gets no wide-angle edge distortion.
The standoff is then fitted separately, by projecting the content and pulling
the camera in until its actual bounding box fills the available area — a plain
bounding-sphere fit is far too pessimistic for a near-flat scene viewed
obliquely, and left the survey at a fifth of the strip's width. The opening
elevation also tracks the viewport aspect: a wide band gets a low, oblique
angle that foreshortens the depth axis to suit it. Resizing rescales the
framing rather than refitting, so a manual zoom or orbit survives a window
drag.

All of the projection math is reused verbatim from `geo_eval`
(`Projector.to_3d` / `.frame_pose`, the scalar-last quaternion convention, the
tilt-corrected radalt, and the **flight-date-specific** boresight). When the
flight falls outside the boresight's validity window
(2025/04/03–2025/05/02) the window says so on screen rather than presenting an
approximate footprint as exact.

**Requirements and failure modes.** It needs a `flight_data.db` in the flight
directory; without one `[v]` prints
a status line and changes nothing. `pose3d` is imported *lazily*, on the first
`[v]`, so nothing about it — a missing module, a broken scipy, an unreadable
db — can affect this tool's startup. If rendering ever raises mid-session the
view is closed and logged; the review session continues, and `App.render()`
still returns its canvas. Closing the separate 3D window does not end the
session. Frame→pose rows are matched by image
**basename** first (falling back to the positional row index every other
consumer in this project assumes), and the window says so on screen if it ever
has to fall back.

**Why not matplotlib.** The 3D view is drawn with numpy+cv2 so it needs no
matplotlib 3D backend. The mainloop is `cv2.imshow` + `cv2.waitKey(20)`;
dropping a Qt/Tk event loop into it is the most reliable way to break an
interactive tool. The renderer is serviced by the *same* `waitKey`.

## What gets written, and where

Everything lands in `<run>/review/<flight_slug>/` (see "Where judgments are
written" above). The flight's `labels.txt` and the run's `dets/` are never
modified.

- **`review_log.jsonl`** — the source of truth. Append-only; one JSON line
  per commit, never rewritten or deleted (a reset appends a new
  `"unreviewed"` record rather than erasing history):
  ```json
  {"schema_version": 1, "timestamp": "2026-11-01T08:12:03-08:00",
   "reviewer": "you@example.org", "directory": "2025_04_16/pieranch_rect",
   "frame_idx": 1234, "image_path": "2025_04_16/pieranch_rect/cam0_1744817888.804665000.png",
   "x": 842.01, "y": 1055.65, "original_confidence": 1.0,
   "judgment": "hard_negative", "spatial_threshold": 30.0, "source": "human"}
  ```
  `source` distinguishes a judgment you made from one written by an automated
  tool — see "Provenance" above. Records without the field are human.
- **`hard_negatives.txt`**, **`confirmed_detections.txt`**, and
  **`added_targets.txt`** — pure derived exports, fully regenerated on every
  commit and on quit. Precisely: regenerated from this session's **in-memory
  view** of the judgments — the snapshot read from `review_log.jsonl` when
  the directory was loaded, plus everything committed since. That
  distinction is invisible in single-session use and is the whole basis of
  the hazard described under "Concurrent writers" above. Same 3-column
  format as `labels.txt` and the run's `dets/` files
  (`<frame_idx> <image_path> <x>,<y>,<value>`), so a future training loader
  can read any of them with the same parsing already used for `labels.txt`.
  `"uncertain"` judgments are logged but never exported to any file — not
  yet a validated signal.

**Resuming**: reopen the same directory later (even a different session)
and every prior judgment *and every added target* reloads automatically
from `review_log.jsonl` — already-reviewed points show in their judged
color, added targets reappear as white stars, and both are skipped by the
`space`/`n`/`b` unreviewed-navigation, so you never lose progress or
re-review the same point twice. Removing an added target (`r`) is itself
just another `review_log.jsonl` record (a `"removed"` entry superseding the
`"added"` one) — the log is never rewritten, only appended to.

**Reopening a directory with per-point judgments**: per-point judgments are
keyed by raw coordinates, but export only looks
up cluster-centroid keys. On load, any per-point judgment whose point falls
in a cluster is automatically re-committed at that cluster's centroid (a
new, clearly-marked `"auto_migrated": true` JSONL record — nothing is
rewritten or deleted) as long as every raw member with a prior judgment
agrees; a cluster whose old judgments disagree is left unreviewed for a
quick manual re-click instead of guessed at. This runs once per cluster
(idempotent on later reloads) and only ever appends, so it can't lose or
overwrite prior work.

## How this is meant to plug into future training (not implemented yet)

All three exports use the same 3-column format `detections.parse_annotations`
already reads, so a future loader can pick up any of them alongside
`labels.txt` with no format translation. The natural landing spot is the
`weighted_heatmap_loss` of the onboard trainer in `harelab-ucsc/birdseye`,
which is currently unused and
whose `y_true[..., 1]` weight channel already
multiplies the loss term: `hard_negatives.txt` points would rasterize into
that channel at elevated weight to specifically penalize the model where it
hallucinated. `confirmed_detections.txt` and `added_targets.txt` points
would both rasterize into the target channel alongside `labels.txt`,
promoting them to first-class positives — the two files differ only in
provenance (a detection the model already made vs. one it never made at
all), not in how a future loss would treat them. That wiring itself is a
separate follow-up, not part of this tool.
