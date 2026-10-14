# Sample data

A small slice of real footage so the pipeline can be tried without the full match set
(the full videos are kept out of git because of their size).

| File | What it is |
|---|---|
| `brynnsarah_sample.mp4` | 100 s excerpt (0:25–2:05) of one match, re-encoded at 12 fps, 480×270, no audio (~1.3 MB) |
| `brynnsarah_sample.labels.csv` | Hand-labeled highlight ranges for the excerpt (`start_sec,end_sec,label`), in clip time |

The labels come from the original Groot labeling pass on the full match, shifted by −25 s:

| Clip time | Notes |
|---|---|
| 0:06–0:11 | sprawl and circle |
| 0:44–0:46 | sprawl |
| 1:28–1:35 | single leg |

`wrestleai extract samples` registers the clip and imports the sidecar label file
automatically, producing 39 windows (5 highlight, 34 non-highlight) with the default
5 s / 2.5 s windowing.

One clip is enough to exercise every stage, not to train a useful model. Cross-validation
needs at least two labeled videos per fold, so use the synthetic generator
(`scripts/make_synthetic_data.py`) or the full match set for training runs.
