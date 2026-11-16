# Geospatial precision and recall

`geo_metrics.py` computes the paper's geospatial precision and recall (see the
README's Citation section) against the adjudication state. The review GUI shows
them **always, per model**: the arm is built eagerly by a background worker, and
every row on screen is one checkpoint's.

It is also the engine behind the GUI's **label propagator** (`[o]`/`[m]`/`[a]`):
a DBSCAN cluster spans frames by construction, so a human verdict on it in one
frame is evidence about the same physical spot in every other frame that saw
it. See "The LABEL PROPAGATOR does name a rule" below.

This is a **second metric space**, alongside the pixel one `REVIEW_GUI.md`
describes. It is not a variant of it:

| | HUD `PIXEL PRE`/`POST` | **`GEO` rows** |
|---|---|---|
| unit | pixel, per detection | **UTM cluster / geotag** |
| ground truth | RTK geotag survey (`labels.txt` is its per-frame projection) | **RTK geotag survey (`labels.txt` is its per-frame projection)** |
| clustering | complete-linkage, px | **DBSCAN, 0.20 m** |
| computed by | `review_gui.py` | **`geo_metrics.py`** |
| scope | one checkpoint | **one checkpoint** |

The numbers are not comparable across columns and are not meant to be.

`labels.txt` is not an independent annotation: it is the RTK geotag survey
projected into each frame. Back-projecting all 6,354 label rows of the sample
flight lands them on geotags with 0.000 m error, and reprojecting the 161
geotags reproduces every label point. The pixel and geo spaces therefore share
one ground truth and differ in whether it is compared before or after
projection and in the counting unit.

---

## The definitions

Precision is over detection **clusters**, recall over **geotags**. That
asymmetry is inherent to the definition — one cluster can absorb five geotags on this data
— so **no F1 is reported**: the harmonic mean of two ratios with different
denominators is a category error, not a summary. The pixel HUD reports F1
because there both ratios share a per-detection denominator.

Two arms, always both, because the gap between geometry-only and adjudicated
precision is the point of human review:

### `GEO PRE` — geometry only

The unadjudicated definition. No adjudication clause anywhere.

- **TP** — cluster with ≥1 geotag within `eps_predict` of a cluster **core** point
- **FP** — cluster with none
- **FN** — geotag assigned to no cluster

### `GEO POST` — adjudicated

`PRE`, then human verdicts override geometry **in both directions**:
`hard_negative` → FP, `confirmed_real` → TP.

`uncertain` and `removed` are **excluded from the vote entirely** — an
abstention, not a negative. A cluster whose only verdicts are those falls back
to its `PRE` geometry.

Where a cluster holds **both** verdicts there is **no single answer**: the
result is reported under all three tie-break rules, always. See the
conflicted-cluster section below.

### Recall differs between the arms — and `POST` is a bound

#### The estimand

Recall is trying to measure **the fraction of real burrows in view that the
detector found**. That is a statement about *real burrows*, not about geotags.
The geotag survey is an *estimator* of that population — a good one, but a
demonstrably incomplete one, which is why adjudication can find positives the
survey missed.

#### Why one recall for both arms would be incoherent

`POST` precision defines a true positive as

> (human-adjudicated positive) **OR** (geometrically matched to a geotag)

That definition **enlarges the positive class** beyond the geotag survey.
Recall's denominator **is** the positive class — `TP + FN` is, by definition,
the number of real positives. Holding recall at the `PRE` value while precision
uses the enlarged definition would score the two arms under **two different
definitions of "true positive"**, which is not a confusion matrix. Whatever the
answer is, the arms cannot share one recall.

#### Symbols

| sym | meaning | unit | produced by | observed? |
|---|---|---|---|---|
| `n_hit` | geotags a cluster reached | geotag | `scaffold.geotag_label != -1` | yes |
| `n_den` | geotags in the denominator | geotag | `n_geotags` (161) or `n_in_view` (151) | yes |
| `c` | clusters a human confirmed that **no geotag reached** | cluster | `geo_prf`, `recall.post.c_confirmed_unmatched_clusters` | yes |
| `a` | deduplicated human **added** markers | marker group | `geo_metrics.collect_added` | yes |
| `m` | real burrows neither surveyed **nor** detected | burrow | — | **never** |

#### The formulas

```
R_PRE  = n_hit / n_den                       geo_prf: recall_all, recall_in_view
R_POST = (n_hit + c) / (n_den + c + a)       geo_prf: recall.post.*_bound
R_true = (n_hit + c) / (n_den + c + a + m)   m >= 0, unobservable
```

`c` enters **both** numerator and denominator: a confirmed cluster with no
geotag is a real burrow the survey missed *and* one the detector found. `a`
enters the **denominator only**: an added marker is a real burrow the survey
missed *and* the detector missed too — a false negative on both counts.

In code: `geo_metrics.post_recall(n_hit, n_den, c, a)`, called twice by
`geo_prf` (all / in-view). `review_gui.corrected_prf` reimplements the identical
expression for pixel space, because `review_gui` cannot import `geo_metrics` at
module scope (sklearn); the smoke test asserts the two copies agree.

#### What each arm assumes, and which way it is wrong

| arm | assumes | bias |
|---|---|---|
| `R_PRE` | survey omission is **independent of detectability** (MCAR) | unbiased *if* that holds |
| `R_POST` | `m = 0` — the detector missed **no** unsurveyed burrow | **upward**; a strict upper bound |

`R_POST` is biased upward because an unmarked burrow is only *discoverable*
where the detector fired. Every unit of `c` is therefore, by construction, a
true positive and never a false negative: `c` is a **censored** sample of the
omitted-positive population and can only raise the ratio. Since `m >= 0`,
`R_true <= R_POST` always — `R_POST` is exactly the `m = 0` case.

`R_PRE`'s MCAR assumption is defensible **here specifically**: the geotags are
a **walking RTK field survey** (recorded with the clicker rig of
harelab-ucsc/birdseye) while the detector reads **aerial imagery**. A burrow is
missed on foot for reasons largely unrelated to whether it is visible from
above, so the omission and the miss are near-independent — which is exactly
MCAR.

**Neither number is "the" recall.** `R_PRE` is the point estimate, `R_POST` the
upper bound; the pair brackets the truth from one side each, and that is why
both are always displayed. The HUD writes the bound as `R<=`, never `R=`.

#### Worked example — the sample flight

Measured on the full 2,120-frame sample flight: 161 geotags, 151 in view under
the in-view polygon test, 103 hit. Mid-review, `c = 15`:

```
R_PRE  (all)     = 103 / 161            = 0.6398
R_PRE  (in view) = 103 / 151            = 0.6821
R_POST (all)     = (103+15) / (161+15)  = 118/176 = 0.6705   <= bound
R_POST (in view) = (103+15) / (151+15)  = 118/166 = 0.7108   <= bound
```

`a = 0` here — nobody has right-clicked yet — so nothing constrains the bound
from above, and the HUD says so on its own line. One added marker:

```
R_POST (all) = 118 / 177 = 0.6667
```

i.e. **down** 0.4 points. Enough added markers and `R_POST` drops *below*
`R_PRE`, which is correct behaviour, not a bug: those markers are misses the
survey never recorded. Any check asserting `R_POST >= R_PRE` unconditionally is
wrong the first time a reviewer right-clicks; the smoke test guards that
invariant on `a == 0`.

#### Unit caveats — they do **not** cancel

- `c` counts **clusters**; recall counts **geotags**, and one cluster holds up
  to five geotags on this data. So `c` **under-counts** discovered burrows.
  Since `c` enters numerator and denominator together, under-counting pushes
  `R_POST` *down*, toward `R_PRE`: **conservative**.
- `a` counts **DBSCAN groups of clicks at 0.20 m**, which can swallow two
  adjacent added burrows. So `a` under-counts too — but `a` is in the
  **denominator only**, so that pushes `R_POST` *up*: **anti-conservative**.

Opposite directions, different magnitudes. Report them as two caveats; do not
net them off.

#### Why `hard_negative` does not cancel a geotag hit

A `hard_negative` forces its cluster to `FP` in `POST` precision, but it does
**not** subtract from `n_hit` even when that cluster contains a geotag. This is
a deliberate choice, not an oversight. A human rejecting a cluster is a claim
about the **cluster** — that this detection is not a target — not a claim that
the RTK survey mis-recorded a burrow. The geotag survey is the reference
standard on the recall side; nothing a reviewer presses re-surveys the field.

#### What would falsify this

- **Adjudication accepting non-targets.** `c` counts confirmed clusters at face
  value. If reviewers confirm things that are not burrows, `c` is inflated and
  it inflates the numerator *and* `POST` precision together — the bracket does
  not protect against a bad reference, only against an incomplete one.
- **Correlated omission.** If field workers systematically skipped burrows that
  are also hard to see from the air (say, both are obscured by the same
  vegetation), MCAR fails, `R_PRE` is biased, and it becomes an
  overestimate rather than a neutral point estimate.
- **`a` measured on a different population.** Added markers are only evidence
  about `m` if reviewers place them wherever they see an unmarked burrow, not
  only near detections they happened to be inspecting. Review order is not
  spatially random, so `a` is a floor on the correction, not a sample of it.

#### Denominators

Each arm is reported against **two denominators**:

- `recall_all` — over every geotag
- `recall_in_view` — over geotags passing the in-view polygon test (image
  bounds inset by the 81.5 px projection-noise buffer): the cross-product
  signed-area test on a clockwise walk of the frame rectangle, implemented as
  `run_detector.inside` / `inner_bound` and vectorised as
  `geo_metrics.inside_vec` (asserted point-for-point equivalent in the smoke
  test, on-edge cases included).

The in-view polygon test is a visibility test. The onboard pipeline
(harelab-ucsc/birdseye: birdseye/birdseye.py) instead counts geotags "viewed"
with a 0.05 m projection *round-trip* test (raw clicks matched against
reprojected clicks), which is not a visibility test. Measured on the full
2,120-frame sample flight, the polygon test puts **151** of 161 geotags in view.
Both denominators are shown so the gap between all and in-view stays on screen.

The recall numerator matches the onboard pipeline's: DBSCAN over detections,
`dbscan_predict` at `eps_predict=0.35`, hits = geotags whose predicted label is
not `-1`. That is exactly `geo_prf`'s `PRE` numerator.

---

## Added markers — the `a` term

An **added** marker (right-click) is a target **the detector never fired on**.
That makes it categorically different from `confirmed_real`, which recovers a
detection the model already made:

| | discovered by | detector fired? | enters |
|---|---|---|---|
| `c` (`confirmed_real`, unmatched) | adjudicating a detection | **yes** | numerator **and** denominator |
| `a` (`added`) | a human seeing an unmarked target | **no** | **denominator only** |

`a` is therefore the **only observable part of `m`** — the misses nobody
recorded — and the only term that constrains `R_POST`'s upper bound from above.
It stays in `ABSTAIN_JUDGMENTS` (it must never vote on cluster *precision*,
having nothing to say about whether a detection is real) and is consumed
separately by `geo_metrics.collect_added`.

### The four filters, applied in this order

The order matters: the buckets are disjoint, so every raw click is accounted for
exactly once and the smoke test can assert the arithmetic closes.

1. **Unresolved** — no pose row, below the `MIN_RADALT_M` gate, or `to_3d`
   dropped the row. A marker that cannot be placed on the ground is not counted.
2. **Near a geotag** (within `eps_predict`) — **dropped**. That target is
   *already* in the recall denominator, and already a false negative if no
   cluster reached it. Counting the click too charges one miss twice. This is
   the single most important filter here.
3. **Inside a cluster** (within `eps_predict` of a DBSCAN **core** point, the
   same radius rule `dbscan_predict` implements) — **dropped**. The detector
   *did* fire there, so it is not an undiscovered miss; if the human also
   confirmed that cluster it is already counted in `c`.
4. **Dedup** — survivors are grouped with `DBSCAN(eps=0.20 m, min_samples=1)`.

`min_samples=1` because a human click is authoritative evidence, unlike a
detector point that needs density to be believed — and with it there is no noise
label, so every survivor lands in exactly one group.

### Why DBSCAN and not the 3 cm grid dedup

The grid dedup in `build_scaffold` works because it only merges back-projected
3D points that land in the same few-centimetre cell — which is precisely why recall
survives it untouched. Two human clicks on the same burrow in frames 700 and 830
are two **independent** projections whose disagreement is decimetres. A grid
would count one burrow dozens of times and inflate `a` catastrophically,
collapsing the bound. `eps` at 0.20 m is the chaining unit the detection
clusters already use, so `a` and `c` share a spatial unit.

### The in-view test applies to added markers too

Tested on the **pixel** coordinates directly (no UTM round-trip, so no
projection error). A click inside the raw frame but in `inner_bound`'s margin
sits where the detector was never allowed to fire; charging it as a miss would
penalise the model for exactly the region `recall_in_view` exists to exclude. A
group counts as in view if **any** member click is — the same "seen in at least
one frame" semantics `geotags_in_view` uses.

---

## Provenance

The geo arm computes over the detections the GUI itself loads — the run's
`dets/` per-checkpoint files, **excluding any degenerate (collapsed/sparse)
checkpoint** — back-projected through each frame's own pose. The engine takes
its detections as an argument, and the HUD and the JSON both state the source.

---

## Cost, and why the grid dedup exists

Measured on a dense accumulated 3D point set from the full 2,120-frame sample
flight (690,162 raw back-projections, 161 geotags):

| dedup | points | DBSCAN | peak RSS | clusters | geotags hit | clusters hit |
|---|---|---|---|---|---|---|
| **1 cm** (`REF_DEDUP_CM`) | 332,330 | 19.9 s | **24.2 GB** | **170** | **103** | **67** |
| 2 cm | 178,538 | 4.2 s | 3.8 GB | 169 | 103 | 68 |
| **3 cm** (`GUI_DEDUP_CM`) | 113,169 | 1.6 s | 1.2 GB | 168 | 103 | 68 |
| 5 cm | 57,077 | 0.45 s | 0.6 GB | 166 | 103 | 68 |

1 cm is the reference resolution. It is also unusable interactively — 24 GB on
a 31 GB machine.

**Geotags hit is 103 at every resolution.** Recall does not move at all; only
the cluster count drifts, by 1.2% at 3 cm. So the GUI uses 3 cm by default (2.6 s
end to end including the in-view test and the verdict join); `--geo-dedup-cm`
overrides it, and the value used is recorded as `dedup_cm` in every JSON
`geo_metrics.py` writes.

**Grid dedup is not subsampling.** It removes redundant copies of the *same* 3D
point, which is why recall survives it untouched. Do not reach for a stride
instead: striding divides points-per-burrow and `min_samples=10` then rejects
real burrows as noise.

---

## ⚠ Conflicted clusters — the largest uncertainty in `POST`

DBSCAN at `eps=0.20 m` chains. Median cluster extent on the sample flight is
0.44 m and one cluster holds five geotags, so a real burrow and a hard-negative
rock 30 cm apart are **one cluster**.

Measured mid-review: **26 of 95 judged clusters contain both a `confirmed_real`
and a `hard_negative`**, and which verdict wins is worth **~15 precision
points**:

| rule | POST precision | POST recall (all) | `c` |
|---|---|---|---|
| `positive_wins` | 0.4762 | 0.6760 | 18 |
| `majority` | 0.4107 | 0.6686 | 14 |
| `hard_negative_wins` | 0.3214 | 0.6628 | 11 |

(Those figures move as review continues. The spread is the point, not the exact
numbers.) Note the spread reaches **recall** as well: `c` counts confirmed
clusters, so it moves with the tie-break too.

### There is no default rule, and no way to select one — for the METRIC

**All three are always computed and always displayed** — the HUD draws one row
each per model, `[p]` one line each, the PDF one table row each, and
`geo_metrics.json` one object each under `post` and `recall.post`. `geo_prf()`
takes no `rule` argument.

That is not fence-sitting. Three things make a single POST number indefensible
here:

1. **The choice is a convention, not a measurement.** A conflicted cluster is an
   artefact of DBSCAN chaining — two physically distinct spots 30 cm apart got
   merged, so the two verdicts are about *different objects*. No amount of
   further review resolves that; only a smaller `eps` would, and that has its
   own costs.
2. **It is worth more than most effects this tool reports.** 0.476 vs 0.321
   is a 15-point swing decided by which convention you picked.
3. **An appendix is where the inconvenient number goes to die.** A spread shown
   only when the rules happen to disagree leaves one headline number to be
   quoted. The surest way to stop one number being quoted alone is to never
   compute it alone.

So the spread **is** the result. Quote it as a range.

### The LABEL PROPAGATOR does name a rule, and that is not a contradiction

`review_gui.py`'s `[o]` cycles these three rules. That is a different consumer
from the metric above, and the argument in this section survives it intact.

The metric asks *"what is this checkpoint's precision?"* — a question whose
answer must not be quietly conditioned on a convention, which is why all three
rows are always shown and `geo_prf()` refuses a `rule` argument.

The propagator asks *"should I carry this verdict to that detection?"* — an
**action**, and an action has to pick one. `GeoVerdicts.verdict(cluster, rule)`
demands that a caller name its convention explicitly, and the propagator is
exactly such a caller.

Three things keep the two axes from collapsing into one:

1. **`[o]` starts OFF and cycles back through off**
   (`off → positive_wins → majority → hard_negative_wins → off`). A reviewer
   who never presses it is never shown a convention somebody else picked. The
   smoke test asserts that with no rule chosen `[m]` draws nothing and `[a]`
   refuses with a reason.
2. **Choosing a propagation rule changes no reported number.** `geo_prf()` is
   untouched, and every per-model block carries all three `POST` rows. The
   smoke test asserts that `geo_prf()` has no `rule` parameter and that
   `GeoVerdicts.verdict()` has no default.
3. **Every propagated label records the rule that produced it**, in a
   `"propagated"` block in `review_log.jsonl` alongside the geo cluster and
   the votes. A label whose convention could not be recovered would be exactly
   the "single POST number" this section forbids, one adjudication further
   downstream.

⚠ **What propagation adds that scoring does not: reach.** A conflicted cluster
costs ~15 precision points *as a number*. As an **action** it costs a label on
every detection in that cluster — and DBSCAN chaining makes some clusters
enormous. Measured on the full `2025_04_16/pieranch_rect` flight against
`results/17_holdout_pieranch`: the median geo cluster covers 16 review
clusters, but the largest covers **18,146 of 62,867 (29% of the whole
backlog)** and the five largest cover **46%**. So one verdict can label a third
of a flight. `review_gui.py` reports the reach on the selected point, in
`[p]`, in the PDF and in a caveat rather than mitigating it silently, because
narrowing `eps` is a decision with its own costs. **When quoting a figure
computed over propagated labels, count the verdicts, not the labels.**

Reading the three, ordered most-positive first (`VERDICT_RULES`), the ordering
`positive_wins >= majority >= hard_negative_wins` holds by construction and is
asserted in the smoke test — for precision *and* both recall denominators — so
the three rows read as a **bracket** rather than three unrelated estimates.

What each rule means, and why none of them is obviously right:

- **`positive_wins`** — a human `confirmed_real` is direct positive evidence
  that the cluster contains a real target, which is exactly the claim precision
  makes; a `hard_negative` alongside it only says the merged blob *also* caught
  something spurious. Under this rule `POST` (0.476) sits **above** `PRE`
  (0.405), the direction expected when the survey is incomplete.
- **`hard_negative_wins`** — the literal reading of the FP clause. But it lets
  a single rejected rock discard a cluster a human explicitly confirmed holds a
  burrow. `POST` falls to 0.321, **below** `PRE`.
- **`majority`** — lets the weight of evidence decide, ties falling back to
  geometry. Sounds neutral, but the vote is over *raw member detections*, so it
  is really weighted by how many frames saw each spot, which is a property of
  the flight path rather than of the targets.

---

## ⚠ `min_samples` counts points, not frames

`min_samples=10` reads naturally as "a burrow seen in fewer frames is dropped as
noise". **It counts points, not distinct frames**, and after dedup the two come
apart: on the full sample flight only 34 of 113,169 points are noise, while
**29 of 168 clusters are supported by fewer than 10 distinct source frames**
(minimum 2). The parameter is very nearly inert on this input.

`diagnostics.n_clusters_under_min_frames` reports this. No UI string describes
the filter as frame-based.

---

## Reading the display

There is no union row: the arm is always on, and the HUD draws it inside
**each model's group**, sharing the pixel table's columns — `GEO PRE`, then
**one row per tie-break rule**, ordered most-positive first:

    -- 030 t.04   healthy   44012d --        P        R        F1
       PIXEL PRE                            0.1888   0.8137   0.3064
       PIXEL POST                           0.2011   0.8203<= 0.3232<=
       GEO PRE                              0.1888   0.8137
       GEO POST pos                         0.2011   0.8156<=
       GEO POST maj                         0.1954   0.8149<=
       GEO POST neg                         0.1888   0.8137<=

The union of every active checkpoint is not scored: it is the review substrate,
which makes it a review-burden measure and nobody's precision. The union
*scaffold* is still built — it is what the propagator runs on.

The `F1` column stays blank: precision is per cluster, recall per geotag, so
their harmonic mean is a category error, and collapsing the column would put
geo's `R` under pixel's `F1`.

The full block — per-rule `c`, the `a` breakdown, the spread and the provenance
caveat — is in the **`[p]`** overlay. That split is deliberate: the panel lays
out bottom-up, so every pixel the HUD takes comes straight off the zoom inset.
The budget is enforced by asserting what it protects: at import, the zoom inset
must clear its floor on the shortest supported panel, and
`review_gui.hud_columns_clear()` must find no cell running into the next column.
The `+c +a` trailer lives in `[p]` because `c` differs per rule and one number
would be wrong for two of the three.

A directory with no `flight_data.db` says so and changes nothing.

**`cov`** is the fraction of geo clusters carrying at least one human verdict.
It is a function of the sliders — judgment keys are float-exact centroids that
move with `--cluster-radius` — so read it next to the HUD's `match:` line, never
alone.

---

## Persistence

`[s]` / `[q]` writes **`geo_metrics.json`** beside `review_log.jsonl`.

The file is **per model only**. The top level carries `schema`, `schema_note`,
`params`, `definitions`, `propagation` and `per_layer`, and **no** top-level
`pre`/`post`/`recall` keys. Every precision and recall figure lives under
`per_layer[<layer_id>]`, each entry a `geo_prf` dict. The union is not scored
anywhere — not in the HUD, `[p]` or the PDF — so it is not in this file either.

The `propagation` block records the propagator's state and coverage:
`rule_selected`, `rules_available`, `no_default_rule: true`,
`n_geo_clusters`, `n_clusters_judged`, `n_clusters_conflicted`,
`n_judged_keys`, `n_unresolved`.

Within `per_layer`, the `recall.*` keys (`tp_geotags`, `recall_all`,
`recall_in_view`, …) are the `PRE` arm, and the adjudicated numbers live in a
nested `recall.post` sub-dict. Inside `recall.post` the ratio keys are named
`recall_all_bound` / `recall_in_view_bound` on purpose — you cannot read the
adjudicated recall without typing the word "bound" — and `is_bound: true` sits
alongside them.

Anything the HUD shows also lands on disk; the onboard pipeline only prints its
geospatial counts, so nothing there can be regenerated.

`[x]` carries the same block into **`review_report.pdf`** (see
`REVIEW_GUI.md`), together with the run parameters, the input digests and the
caveats that `geo_metrics.json` has no place for. Same principle, one step
further: the JSON is for a tool to re-read, the PDF is for a person to cite.

It is written from the explicit save handler, **not** from
`JudgmentStore.export()` — that runs on essentially every mutation and would
rewrite the file on every keystroke. `--dry-run` writes nothing.

---

## Verification

```bash
# the GUI's own checks, including the geo invariants
python tests/smoke_test_gui.py --data-root /path/to/birdseye-review-sample
```

The geo smoke checks assert, per model:

- `inside_vec` agrees with `run_detector.inside` point-for-point, on-edge included
- the verdict precedence table across all three rules, ties and abstentions
- that `GeoVerdicts.verdict()` has no default — a caller must name a rule, and
  `geo_prf` answers by computing all of them
- that `post` and `recall.post` carry **every** rule, and that the HUD and `[p]`
  each draw one row per rule **per model** rather than one row
- that the propagator refuses on all four of its no-label paths (no rule, no
  geo cluster, no verdict, rule abstained), that `majority` abstains on a tie,
  and that `[o]` cancels an armed `[a]`
- that a **degenerate checkpoint contributes no review targets while its own
  PRE scorecard stays bit-identical** — both halves, because they pull in
  opposite directions
- that POST precision **and** both POST recall denominators come out ordered by
  how positive the rule is, so the three rows read as a bracket
- `review_gui`'s mirrored constants have not drifted from `geo_eval`'s
- **`PRE` does not move when a judgment is committed** — it is the
  unadjudicated arm and a refactor must not be able to disturb it
- **`PRE` recall does not move in either arm**, for either judgment type — the
  whole `recall` block minus `post` must come back byte-identical after a commit
- **`POST` recall does move**, and only the right way: `confirmed_real` can only
  raise `c`, `hard_negative` can only lower it, and the bound follows
- **the bound degenerates correctly** — `R_POST == R_PRE` exactly when
  `c == 0 and a == 0`, and `R_POST >= R_PRE` **only while `a == 0`** (the
  guarded form; unguarded it is wrong the first time someone right-clicks)
- **`post_recall` and `review_gui.corrected_prf` agree** on a table of
  `(tp0, fn0, c, a)` — the formula lives in two files and must not drift
- **added markers dedup across frames** — the same target clicked in two frames
  counts once — and **a marker on an existing geotag is dropped**, so one miss
  is never charged twice
- **an added marker lowers the bound and leaves `PRE` untouched**, and removing
  it restores both exactly
- every judged point resolves to a geo cluster (the grid-cell join is exact;
  `n_unresolved_points` is 0)
- every matched geotag is in view, or the in-view denominator is too strict
- `POST` tp + fp == n_clusters — every cluster lands in exactly one bucket
