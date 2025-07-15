# Wrestling Highlight Detector

A machine learning project to automatically detect highlights in wrestling matches.

## Project Structure

```
wrestling-highlight-detector/
├── live_filter/           # Phase 1: Live vs Non-Live Detection
│   ├── terminal_range_labeler.py    # Main tool for labeling LIVE ranges
│   ├── live_filter_utils.py         # Utilities for live filter phase
│   ├── add_matches_to_db.py         # Add match videos to database
│   └── clips/                       # Generated clips (if any)
├── highlights/            # Phase 2: Highlight Detection (Future)
│   ├── gradio_app.py                # Highlight labeling interface
│   ├── motion_detection.py          # Motion detection algorithms
│   ├── motion_plotter.py            # Motion visualization
│   └── wrestling_model.py           # ML model for highlights
├── database/              # Database management
│   ├── database_manager.py          # Main database operations
│   ├── database_setup.py            # Database initialization
│   ├── database_utils.py            # Database utilities
│   └── view_database.py             # Database inspection tools
├── archives/              # Old/utility scripts
│   ├── add_gender_field.py          # Database migration script
│   ├── cleanup_originals.py         # Cleanup utility
│   ├── label_stats.py               # Labeling statistics
│   └── quick_start.py               # Initial setup script
├── match_directory/       # Original match videos
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Current Workflow

### Phase 1: Live Filter (Current Focus)
The project is currently in Phase 1, focusing on distinguishing between live match time and non-live time (breaks, downtime, etc.).

#### Getting Started
1. **Setup Database**: Run `python database/database_setup.py` to initialize the database
2. **Add Matches**: Use `python live_filter/add_matches_to_db.py` to add match videos to the database
3. **Label LIVE Ranges**: Use `python live_filter/terminal_range_labeler.py` to label LIVE sections of matches

#### Key Tools
- **Terminal Range Labeler**: The main tool for labeling LIVE ranges in matches
- **Live Filter Utils**: Utilities for checking progress, exporting data, and validating database integrity

#### Database Schema
- `original_matches`: Stores match video information
- `match_ranges`: Stores labeled time ranges (LIVE/NOT LIVE)
- `labels`: Stores individual clip labels (for future highlight detection)

### Phase 2: Highlight Detection (Future)
Once Phase 1 is complete, the project will move to Phase 2 for detecting highlights within the LIVE sections.

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
python database/database_setup.py
```

## Usage

### Labeling LIVE Ranges
```bash
python live_filter/terminal_range_labeler.py
```

### Check Progress
```bash
python live_filter/live_filter_utils.py
```

### View Database
```bash
python database/view_database.py
```

## Database Tables

- **original_matches**: Match video metadata
- **match_ranges**: Time ranges labeled as LIVE or NOT LIVE
- **labels**: Individual clip labels (for future use)

## Next Steps

1. Complete labeling LIVE ranges for all matches
2. Generate training clips from LIVE ranges
3. Train model to automatically detect live match time
4. Move to Phase 2: Highlight detection within LIVE sections