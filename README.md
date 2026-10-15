# Wrestling Highlight Detector - Groot Pipeline

A machine learning project to automatically detect highlights in wrestling matches using a sophisticated 4-stage deep learning pipeline.

## Project Evolution

### Phase 1: Live Filter (Legacy - Not Used)
- Basic live vs non-live detection using manual labeling
- Terminal-based range labeling tools
- Database management system
- **Note**: This implementation is preserved in `live_filter/` directory but is not part of the current Groot pipeline

### Phase 2: Groot LSTM Pipeline (Current Focus)
Advanced LSTM-based highlight detection with modular, testable components, implemented as the
`wrestleai` Python package in `src/wrestleai/`. The original Groot data-processing scripts in
`groot/` are kept for reference; their logic has been ported into the package.

## Groot Pipeline Architecture

The Groot pipeline is divided into modular, independently testable components:

1. **Data Processing** - Video preprocessing and dataset preparation
2. **Segmentation Model** (DeepLabV3+) - Extract wrestler silhouettes
3. **2D CNN Feature Extractor** (EfficientNet-B3) - Extract per-frame features
4. **Bi-LSTM Temporal Modeling** - Capture temporal patterns
5. **Dense Classifier** - Final highlight/non-highlight classification
6. **Connector** - `HighlightModel` chains stages 2–5 into one model
7. **Training Phase** - k-fold cross-validated training with early stopping
8. **Testing Phase** - Out-of-fold evaluation and video-level inference

```
video ─► 12 fps frames ─► 5 s windows (2.5 s stride) ─► 16 frames/window
      ─► [DeepLabV3+ silhouette mask] ─► EfficientNet-B3 (1536-d per frame)
      ─► Bi-LSTM (2 layers, 384 hidden, bidirectional) ─► Dense 512 ─► 128 ─► classes
```

## Development Philosophy

### Modular Development Approach
- **Independent Testing**: Each component must work standalone before integration
- **Incremental Progress**: No jumping to new parts until previous ones are guaranteed to work
- **Sub-pipeline Validation**: Complex sub-components are tested at smaller scale first
- **Elegant Problem Solving**: Address issues while working on smaller, modular components
- **Error Prevention**: Avoid massive debugging sessions by catching issues early

### Development Habits
- Test each component thoroughly before moving to the next
- Create comprehensive test cases for each module
- Document any deviations or sub-pipeline complexities
- Maintain clear separation of concerns between components
- Use configuration files for easy parameter tuning
- Implement proper logging and error handling at each stage

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest                             # 66 tests, ~20 s on CPU, fully offline
```

Requires Python 3.9+ (tested on 3.11). `requirements.txt` pins torch 2.2.x, the last release with Intel-macOS
wheels; on other platforms torch/torchvision can be upgraded together. The older scripts in
`live_filter/`, `highlights/` and `database/` use `requirements-legacy.txt`.

### Try it on synthetic data (about 1–2 minutes on CPU)

```bash
python scripts/make_synthetic_data.py --out data/synthetic/videos
wrestleai extract data/synthetic/videos --config configs/test.yaml
wrestleai train    --config configs/test.yaml --run-name demo
wrestleai evaluate --config configs/test.yaml
wrestleai infer data/synthetic/videos/synthetic_match_03.mp4 --config configs/test.yaml
```

The synthetic videos are toy data (fast-moving blobs during "highlights"). They show that
every stage runs and that the model can learn, not how it performs on real wrestling.

### Real sample clip

`samples/` holds a 100 s, 1.3 MB excerpt of a real match with hand-labeled highlight ranges
(see [`samples/README.md`](samples/README.md)):

```bash
wrestleai extract samples                       # 39 windows, 5 labeled highlight
wrestleai label brynnsarah_sample.mp4 --list    # show the labeled ranges
```

## CLI

All commands accept `--config <yaml>` (default `configs/default.yaml`), repeatable
`--set section.key=value` overrides, `--log-level` and `--log-file`.

| Command | What it does |
|---|---|
| `wrestleai extract <videos or dirs> [--convert]` | Probe videos with OpenCV, store them in SQLite, import `<name>.labels.csv` files found next to them, and slice each video into labeled windows. `--convert` writes 12 fps, 224×224 copies for faster training. With no inputs, re-windows every video already in the database. |
| `wrestleai label <video>` | Interactive terminal labeler (`1:30 1:42 highlight # double leg`, `list`, `delete 2`, `play`, `done`). Non-interactive: `--add START END [LABEL]`, `--import file.csv`, `--list`. |
| `wrestleai export-labels [--out labels]` | Write each video's labeled ranges (with notes) to `<name>.labels.csv`, e.g. to keep them in git. |
| `wrestleai train [--run-name NAME]` | k-fold cross-validation over all labeled windows. Windows from the same video stay in one fold. Writes `fold_<k>.pt`, `best.pt`, `config.yaml` and `cv_summary.json` to `checkpoints/<run>/`. |
| `wrestleai evaluate [--run NAME]` | Out-of-fold accuracy, precision, recall, F1, ROC-AUC, PR-AUC and confusion matrix: each window is scored by the fold model that never trained on it. `--checkpoint X --video-ids ...` scores one checkpoint on held-out videos. |
| `wrestleai infer <video> [--threshold T]` | Per-window highlight probabilities (`outputs/<name>.windows.csv`) and merged highlight time ranges (`outputs/<name>.highlights.json`). |

Labels are `start_sec,end_sec,label` ranges. A window takes a range's label when that range
covers at least `labels.overlap_threshold` (default 50%) of it; everything else is
`non_highlight`. Set `labels.classes` to more names (e.g. `[none, takedown, pin]`) for
multi-class technique labels.

Labels for the 16 labeled matches are kept in [`labels/`](labels/) (one
`<match>.labels.csv` per video, with notes). After labeling, refresh them with
`wrestleai export-labels`. To rebuild the database from them on a new machine (with the
videos in `match_directory/`):

```bash
wrestleai extract match_directory --labels-dir labels
```

To bring over labels made with the original Groot terminal labeler (read-only; the old
database is not modified):

```bash
python scripts/import_legacy_groot_db.py --legacy-db database/wrestling_pipeline.db
wrestleai extract --labeled-only
```

## Configuration

Every hyperparameter lives in [`configs/default.yaml`](configs/default.yaml), backed by
typed dataclasses in `src/wrestleai/config.py` (unknown keys are rejected).
[`configs/test.yaml`](configs/test.yaml) is a tiny CPU-only variant used by the tests and
the synthetic demo. Key settings:

- `data`: 12 fps, 224×224 frames, 5 s windows, 2.5 s stride, 16 frames per window
- `segmentation.enabled` (default `false`) and `segmentation.checkpoint`
- `features.backbone` (any timm model, default `efficientnet_b3`) and `features.freeze`
- `temporal`, `classifier`: Bi-LSTM and dense head sizes
- `training`: `k_folds`, epochs, learning rate, class weighting, early stopping on `monitor`
- `inference`: threshold, merge gap and minimum range length

## Project Structure

```
wrestling-highlight-detector/
├── src/wrestleai/             # Groot pipeline package
│   ├── config.py              # Typed YAML config
│   ├── cli.py                 # `wrestleai` command
│   ├── db/                    # SQLite: videos, label_ranges, clip_windows
│   ├── data/                  # Stage 1: video I/O, windowing, label mapping, Dataset, dedup, synthetic data
│   ├── labeling/              # Terminal range labeler
│   ├── segmentation/          # Stage 2: DeepLabV3+ silhouette masking
│   ├── features/              # Stage 3: EfficientNet-B3 per-frame features
│   ├── temporal/              # Stage 4: Bi-LSTM
│   ├── models/                # Stage 5: classifier head + HighlightModel
│   ├── training/              # k-fold trainer, metrics, checkpoints, evaluation
│   ├── inference/             # Window scoring and highlight-range merging
│   └── utils/                 # Logging, seeding, device, time parsing
├── configs/                   # default.yaml, test.yaml
├── scripts/                   # Synthetic data generator, legacy DB import
├── labels/                    # Labeled ranges for every labeled match (CSV)
├── samples/                   # Small real sample clip + labels
├── tests/                     # pytest suite (one file per stage + end-to-end smoke test)
├── groot/                     # Original Groot data-processing scripts (reference)
├── live_filter/               # Phase 1: Live vs Non-Live Detection (Legacy - Not Used)
├── highlights/                # Earlier highlight detection (Legacy)
├── database/                  # Legacy database management
├── archives/                  # Old/utility scripts
└── match_directory/           # Full match videos (not in git)
```

Generated files (`data/`, `checkpoints/`, `outputs/`, `*.db`) are git-ignored.

## Current Status

- **Working end to end:** data processing, labeling, all five model stages, k-fold
  training, evaluation and inference, covered by 66 offline tests including an
  end-to-end smoke test through the CLI.
- **Labeled data:** 8 highlight ranges across 2 matches, far too few to train a useful
  model. Labeling more matches is the main next step.
- **Segmentation:** DeepLabV3+ only has an ImageNet-pretrained encoder. Its masks are not
  meaningful until it is fine-tuned on person masks (e.g. the Kaggle person-segmentation
  dataset), so it is disabled by default and warns if enabled without a checkpoint.
- **Short labels:** with 5 s windows and a 50% overlap rule, labeled ranges of 2 s or less
  never produce a highlight window. Shorter windows or a lower `labels.overlap_threshold`
  would keep them.

## Next Steps

1. Label more matches (`wrestleai label`) and re-run `wrestleai extract --labeled-only`
2. Fine-tune DeepLabV3+ on person-segmentation data and enable masking
3. Cache per-frame CNN features so training with a frozen EfficientNet-B3 is fast on CPU
4. Train on the full labeled set and evaluate on held-out matches
5. Multi-class technique labels (takedown, sprawl, single leg, ...)

## Database Schema

### Groot Tables (`data/wrestleai.db`)
- `videos` - Video path, FPS, frame count, resolution, duration, optional resampled copy
- `label_ranges` - Labeled time ranges (`start_sec`, `end_sec`, `label`, labeler, notes)
- `clip_windows` - Fixed windows per video and the label mapped onto each

### Legacy Tables (`database/wrestling_pipeline.db`, not used by the package)
- `original_matches`, `match_ranges`, `labels`: Phase 1 live filter
- `groot_videos`, `groot_highlight_clips`: original Groot scripts (importable with
  `scripts/import_legacy_groot_db.py`)

## Development Guidelines

- **Test First**: Always test components independently
- **Document Changes**: Update this README with any pipeline modifications
- **Modular Design**: Keep components loosely coupled
- **Error Handling**: Implement robust error handling at each stage
- **Configuration**: Use config files for easy parameter management
- **Logging**: Maintain comprehensive logs for debugging 