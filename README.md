# Wrestling Highlight Detector

A web-based labeling tool for wrestling match highlights using computer vision and machine learning.

## Features

- **Web-based labeling interface** - Shareable via Gradio link
- **Three labeling options**: Highlight, Not Highlight, Unsure
- **Progress tracking** - Real-time statistics and progress monitoring
- **Guidance system** - Clear criteria for what constitutes a highlight
- **Persistent storage** - Labels saved to CSV with timestamps and labeler info
- **Unsure handling** - Clips marked as "Unsure" remain in the queue for later review

## Quick Start

### 1. Setup Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Add Wrestling Clips
Place your wrestling highlight clips (MP4 format) in the `unlabeled_clips/` directory. The app will automatically detect and serve these clips to labelers.

### 3. Launch the Labeling App
```bash
python labeling_app.py
```

This will start a web server and provide you with:
- A local URL (e.g., `http://127.0.0.1:7860`)
- A public Gradio URL that you can share with labelers

### 4. Share with Labelers
Send the Gradio URL to your labelers. They can:
- Enter their name
- View clips one at a time
- Label each clip as Highlight/Not Highlight/Unsure
- See real-time progress

### 5. Monitor Progress
Check labeling progress from your end:
```bash
python check_progress.py
```

This shows:
- Total clips vs labeled clips
- Highlight ratio
- Recent activity
- Labeler statistics

## File Structure

```
wrestling-highlight-detector/
├── unlabeled_clips/           # Wrestling clips (MP4 files)
├── labeling_app.py       # Main labeling application
├── check_progress.py     # Progress monitoring script
├── requirements.txt      # Python dependencies
├── labels.csv           # Label storage (created automatically)
└── README.md
```

## Labeling Guidelines

### What is a HIGHLIGHT?
- **Clean takedown**
- **Escape or reversal** with clear motion and point-scoring intent
- **Sustained pinning combination** (not just a quick tilt, should show control and move the match forward)
- **Match-ending move** (pin, tech, etc.)
- **Opening whistle** if it starts with aggressive action right away
- **Failed attempt** can be a highlight if it leads to an opponent's meaningful response, like a counter, reversal, or clear shift in control

### What is NOT a HIGHLIGHT?
- **Failed attempts** that don't lead to anything (short scramble, no score, reset)
- **Breaks, out-of-bounds resets**, or walking back to center
- **Ref motion or downtime**
- **Post-match running around** or crowd movement
- **Stuff that feels like filler**, even if it looks chaotic (ex. handfighting)

### If the clip is unclear or you don't know what to label it as, click **UNSURE**

## Data Storage

Labels are stored in `labels.csv` with the following columns:
- `filename`: Name of the clip file
- `label`: Highlight, Not Highlight, or Unsure
- `timestamp`: When the label was applied
- `labeler`: Name of the person who labeled it

## Workflow

1. **You**: Add wrestling clips to `unlabeled_clips/` directory
2. **You**: Launch the app and share the URL
3. **Labelers**: Label clips through the web interface
4. **You**: Monitor progress with `check_progress.py`
5. **You**: Review "Unsure" clips manually if needed

## Technical Details

- Built with Gradio for the web interface
- Uses pandas for data management
- Random clip selection to avoid bias
- Automatic progress tracking
- Responsive design for mobile/desktop use

## Troubleshooting

- **No clips found**: Ensure MP4 files are in the `unlabeled_clips/` directory
- **App won't start**: Check that all dependencies are installed with `pip install -r requirements.txt`
- **Progress not updating**: Verify the virtual environment is activated
- **Missing packages**: Run `pip install -r requirements.txt` to install all dependencies