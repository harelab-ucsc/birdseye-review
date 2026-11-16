# birdseye-review

A human-in-the-loop review GUI for UAV detection output. It steps through a
flight frame by frame with each checkpoint's detections overlaid, lets you judge
clusters of detections as confirmed targets or hard negatives (or add targets
the detector missed), propagates verdicts across frames through geospatial
clustering of the back-projected detections, and scores geospatial precision and
recall against an RTK geotag survey.

## Install

```sh
pip install "birdseye-review[report] @ git+https://github.com/harelab-ucsc/birdseye-review"
```

Requires Python ≥ 3.10 and a desktop OpenCV build (`opencv-python`, not
`opencv-python-headless`): the interface is OpenCV highgui. The `report` extra
adds `reportlab` and `matplotlib` for the PDF report.

## Try it on the sample

Download the sample flight
([birdseye-review-sample-v1.zip](https://github.com/harelab-ucsc/birdseye-review/releases/download/sample-v1/birdseye-review-sample-v1.zip),
~200 MB: frames 600–641 of one flight, its RTK survey and the detections of 12
checkpoints), then:

```sh
unzip birdseye-review-sample-v1.zip
cd birdseye-review-sample
birdseye-review --reviewer you@example.org 2025_04_16/pieranch_rect results/17_holdout_pieranch
```

Press `?` for every key and mouse gesture. Verdicts are written under
`results/17_holdout_pieranch/review/`.

## Input layout

Paths are resolved against the current directory (or `--base-dir`).

- **Flight directory** `<root>/<YYYY_MM_DD>/<name>/`
  - `labels.txt`: one point per line, `<frame_idx> <image_path> <x>,<y>,<conf>`
  - the frame images
  - optional `flight_data.db` (SQLite) with the tables
    `cam0_images_flight_data` (camera poses), `clicks_flight_data` (RTK
    geotags) and `parameters_flight_data` (camera parameters). It is needed for
    the 3D view and the geospatial metrics; `frame_idx` is the positional row
    of `cam0_images_flight_data`.
- **Run directory**
  - `dets/val_<ckpt>[_t<thresh>].txt`: one file per checkpoint, same line
    format as `labels.txt`, image paths relative to `<root>`
  - optional `PREREGISTRATION.json` and `run_manifest.json`

## Outputs

All outputs go under `<run>/review/<flight>/`; the flight directory, `labels.txt`
and the run's `dets/` are never modified.

- `review_log.jsonl`: append-only record of every verdict (fields in
  [REVIEW_GUI.md](src/birdseye_review/REVIEW_GUI.md))
- `confirmed_detections.txt`, `hard_negatives.txt`, `added_targets.txt`: the
  verdicts exported in the `labels.txt` line format
- `by_layer/`: the same exports per checkpoint
- `geo_metrics.json`: geospatial precision/recall per checkpoint
- `review_report.pdf`: the `[x]` report (needs the `report` extra)

`--dry-run` renders and navigates normally but writes nothing.

## Documentation

- [REVIEW_GUI.md](src/birdseye_review/REVIEW_GUI.md): keys, workflow, file formats
- [GEO_METRICS.md](src/birdseye_review/GEO_METRICS.md): geospatial precision and recall definitions

## Producing detections

`python -m birdseye_review.run_detector` runs the birdseye EfficientNetV2-B3
heatmap detector over a flight and writes a `dets/` file. It needs the
`detector` extra (`tensorflow`) and birdseye checkpoint weights, which are not
distributed here.

## Environment

- `BIRDSEYE_NO_BORESIGHT_SELFCHECK`: any non-empty value disables the
  labels-vs-geotag boresight self-check run when a flight's pose database is
  loaded.

## Testing

```sh
python tests/smoke_test_gui.py --data-root /path/to/birdseye-review-sample
```

The smoke test is headless and needs the extracted sample.

## Citation

"Human-in-the-Loop Geospatial Annotation for Rapid Dataset Construction in
Field-Deployed UAV Systems", https://arxiv.org/abs/2609.28767

## License

Code: MIT ([LICENSE](LICENSE)). Sample data: CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/), harelab-ucsc.
