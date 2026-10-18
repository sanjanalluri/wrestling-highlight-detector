# Wrestling Highlight Detector

This project watches a full wrestling match and points out the moments worth putting in a highlight reel. You give it a match video, and it gives back a list of time ranges where something exciting probably happened, like a takedown, a reversal, a big throw, or a near fall.

Highlight reels are still mostly cut by hand. Someone has to scrub through a six or seven minute match to find the twenty seconds that matter. The goal here is to do that first pass automatically, so a person only has to review a short list of candidates instead of the whole match.

## How it works

The model looks at a match in short overlapping windows and decides, one window at a time, whether it is a highlight.

1. **Break the video into windows.** Each match is resampled to 12 frames per second and split into five second windows that overlap by half. Sixteen frames are taken from each window.
2. **Describe what each frame looks like.** Every frame goes through EfficientNet-B3, an image network pretrained on ImageNet, which turns the picture into a list of numbers describing what is in it.
3. **Describe how things are moving.** Optical flow measures how pixels move from one frame to the next. Broadcast cameras pan and zoom constantly, so the overall camera movement is removed first, leaving the motion that comes from the wrestlers.
4. **Read the window as a sequence.** A bidirectional LSTM reads the sixteen frames in order, so it can follow how an exchange builds and finishes instead of judging a single still image.
5. **Make the call.** A small classifier turns that into a highlight probability, and neighboring windows that score high are merged into time ranges.

There is also an optional DeepLabV3+ stage that masks out everything except the wrestlers' silhouettes. It is built and tested but switched off for now, because it still needs to be trained on person segmentation data before its masks are useful.

## My process

**Early experiments.** My first version, in the `live_filter/` folder, tried to separate live wrestling from stoppages and downtime using a 3D CNN on both video and audio. After that I tried motion detection and a 3D CNN aimed at highlights directly, in `highlights/`. Those versions are kept in the repository for reference, and the current pipeline grew out of what I learned from them.

**Defining a highlight.** Before labeling anything I wrote down what counts. A clean takedown, an escape or reversal, a sustained pinning combination, a match ending move, or a failed attempt that leads to a real counter all count. Handfighting, resets, out of bounds, and downtime do not, even when they look busy. Keeping that definition fixed is what keeps the labels consistent from one match to the next.

**Labeling.** I built a small terminal tool for marking time ranges while watching a match, then labeled 16 matches by hand, with a short note on most highlights (double leg, gutwrench, arm throw, and so on). I learned to keep each range tight around the actual action. Long ranges that included the setup and handfighting taught the model the wrong thing, and very short ones were too brief to count once the video was cut into windows.

**Building in stages.** Each part of the pipeline, from video processing to the classifier, was written and tested on its own before being connected to the next. The test suite runs without internet access or a GPU, and synthetic videos with known highlights let me check the whole pipeline end to end in seconds.

**Making it practical on a laptop.** Running EfficientNet-B3 over every frame on every training pass would take hours on a CPU. Because that network is frozen, its output for a given frame never changes, so the features are computed once and cached. That brought a full training run down to about an hour, which made real experimentation possible.

**Evaluating honestly.** Neighboring windows from the same match look almost identical, so testing on a match the model has already seen would make it look far better than it is. Every result comes from cross validation where each match stays entirely on the training side or entirely on the testing side. Plain accuracy is also misleading here, because most of a match is not a highlight and a model that always says "no" scores well. Instead I looked at how well the model ranks real highlights above everything else, how many labeled highlights it finds, and how many false alarms it raises per hour of footage.

**Adding motion.** Highlights are mostly bursts of movement, and image features alone do not capture motion directly. Adding camera compensated optical flow helped the model find more of the labeled highlights without raising more false alarms, so it is now part of the default setup.

## Where it stands

The model has clearly learned something real. It ranks true highlights well above ordinary wrestling and catches a good share of them, but it still raises enough false alarms that its output works best as a shortlist for a person to review rather than a finished reel. Going from 10 to 16 labeled matches helped less than I expected, which suggests the approach itself, more than the amount of data, is the main limit right now. Next steps would include fine tuning the image network on wrestling footage, training the silhouette segmentation stage so the model focuses on the wrestlers, and labeling specific techniques like takedowns and pins instead of a single highlight label.

## To Run

Requires Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

Try the full pipeline on small synthetic videos.

```bash
python scripts/make_synthetic_data.py --out data/synthetic/videos
wrestleai extract data/synthetic/videos --config configs/test.yaml
wrestleai train --config configs/test.yaml --run-name demo
wrestleai evaluate --config configs/test.yaml
wrestleai infer data/synthetic/videos/synthetic_match_03.mp4 --config configs/test.yaml
```

Run it on your own matches.

```bash
wrestleai label match_directory/my_match.mp4
wrestleai extract --labeled-only
wrestleai train --run-name my-run
wrestleai evaluate
python scripts/analyze_run.py
wrestleai infer match_directory/new_match.mp4
```

Settings live in `configs/default.yaml` and can be changed with `--set`, for example `--set training.epochs=20`.

## Repository layout

| Folder | What is in it |
|---|---|
| `src/wrestleai/` | The pipeline, organized by stage (data, segmentation, features, temporal, models, training, inference) |
| `configs/` | Default settings and a small configuration used for tests |
| `scripts/` | Synthetic data, run analysis, and importing older labels |
| `labels/` | Hand labeled highlight ranges for 16 matches |
| `samples/` | A short clip from a real match with its labels |
| `tests/` | Tests for every stage plus full end to end runs |
| `live_filter/`, `highlights/`, `groot/`, `database/`, `archives/` | Earlier experiments, kept for reference |
