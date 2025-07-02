# Wrestling Highlight Detector

A comprehensive wrestling highlight detection and labeling system using computer vision, motion analysis, and machine learning. The system automatically extracts potential highlights from full wrestling videos and provides a web-based interface for human labeling.

## 🚀 Features

- **🎬 Automatic Highlight Detection**: Motion analysis to extract potential highlights from full wrestling videos
- **🗄️ Database-Driven Pipeline**: SQLite database for tracking videos, clips, labels, and processing history
- **🌐 Web-based Labeling Interface**: Gradio-powered UI for collaborative labeling
- **📊 Progress Tracking**: Real-time statistics and progress monitoring
- **🔄 Incremental Processing**: Only process new videos, preserve existing work
- **👥 Multi-Labeler Support**: Track multiple labelers with their decisions
- **💾 Persistent Storage**: All data stored in SQLite database with backup capabilities
- **🎯 Three Labeling Options**: Highlight, Not Highlight, Unsure
- **🔍 Database Visualization**: Built-in tools to view and analyze your data

## 📁 Project Structure

```
wrestling-highlight-detector/
├── match_directory/           # Full wrestling videos (MP4, AVI, MOV, etc.)
├── unlabeled_clips/          # Generated highlight clips (auto-created)
├── gradio_app.py             # Main labeling interface
├── motion_detection.py       # Video processing and highlight extraction
├── database_manager.py       # Database operations and management
├── database_setup.py         # Database initialization
├── view_database.py          # Database visualization tool
├── cleanup_originals.py      # Storage management utility
├── quick_start.py            # Automated setup and verification
├── requirements.txt          # Python dependencies
├── wrestling_pipeline.db     # SQLite database (auto-created)
└── README.md
```

## 🛠️ Installation & Setup

### Quick Start (Recommended)
```bash
# Run the automated setup guide
python3 quick_start.py
```

This will check all prerequisites and guide you through the setup process.

### Manual Setup

#### 1. Prerequisites
```bash
# Install FFmpeg (required for video processing)
# On macOS:
brew install ffmpeg

# On Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# On Windows: Download from https://ffmpeg.org/download.html
```

#### 2. Python Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

#### 3. Database Setup
```bash
# Initialize the database (runs automatically, but you can run manually)
python3 database_setup.py
```

## 🎬 Usage Workflow

### Step 1: Add Wrestling Videos
Place your full wrestling match videos in the `match_directory/` folder:
```bash
# Supported formats: MP4, AVI, MOV, MKV, WMV
cp your_wrestling_video.mp4 match_directory/
```

### Step 2: Extract Highlight Clips
```bash
# Process all videos and extract potential highlights
python3 motion_detection.py
```

This will:
- Analyze motion patterns in your videos
- Extract 5-second clips around high-motion moments
- Store metadata in the database
- Create web-compatible MP4 clips in `unlabeled_clips/`

### Step 3: Launch Labeling Interface
```bash
# Start the web-based labeling app
python3 gradio_app.py
```

This provides:
- Local URL (e.g., `http://127.0.0.1:7860`)
- Public Gradio URL for sharing with labelers
- Real-time progress tracking

### Step 4: Label Clips
Labelers can:
- Enter their name
- View clips one at a time
- Label as Highlight/Not Highlight/Unsure
- See source video information
- Track their progress

### Step 5: Monitor Progress
```bash
# View database statistics
python3 view_database.py

# Check labeling progress
python3 -c "from database_manager import DatabaseManager; db = DatabaseManager(); p = db.get_labeling_progress(); print(f'Progress: {p[\"progress_percent\"]:.1f}% ({p[\"labeled_clips\"]}/{p[\"total_clips\"]} clips)')"
```

## 🗄️ Database Management

### View Database Contents
```bash
# Text-based database viewer
python3 view_database.py

# Visual database browser (opens GUI)
open -a "DB Browser for SQLite" wrestling_pipeline.db
```

### Storage Management
```bash
# Clean up original videos after processing (optional)
python3 cleanup_originals.py
```

### Database Tables
- **videos**: Source video metadata and processing status
- **clips**: Generated highlight clips with timing and motion scores
- **labels**: Human labels with labeler information
- **processing_log**: Processing history and error tracking

## 🎯 Labeling Guidelines

### What is a HIGHLIGHT?
- **Clean takedown** with clear execution
- **Escape or reversal** with point-scoring intent
- **Sustained pinning combination** showing control
- **Match-ending move** (pin, tech, etc.)
- **Opening whistle** with aggressive action
- **Failed attempt** that leads to meaningful counter/reversal

### What is NOT a HIGHLIGHT?
- **Failed attempts** without follow-up action
- **Breaks, out-of-bounds resets**
- **Referee motion or downtime**
- **Post-match activity**
- **Filler content** (handfighting, etc.)

### Use UNSURE when:
- The clip is unclear or ambiguous
- You're not confident in your decision
- The action is borderline

## 🔧 Advanced Usage

### Add New Videos Incrementally
```bash
# Add new videos to match_directory/
cp new_video.mp4 match_directory/

# Process only new videos
python3 motion_detection.py
```

### Export Data for Analysis
```bash
# Export labels to CSV
python3 -c "
from database_manager import DatabaseManager
import pandas as pd
db = DatabaseManager()
conn = db.get_connection()
df = pd.read_sql_query('SELECT * FROM labels', conn)
df.to_csv('exported_labels.csv', index=False)
conn.close()
print('Labels exported to exported_labels.csv')
"
```

### Database Backup
```bash
# Create backup
cp wrestling_pipeline.db wrestling_pipeline_backup.db

# Restore from backup
cp wrestling_pipeline_backup.db wrestling_pipeline.db
```

## 📊 System Statistics

The system tracks:
- **Total videos processed**
- **Clips generated per video**
- **Labeling progress percentage**
- **Highlight vs Not Highlight ratio**
- **Labeler activity and decisions**
- **Processing time and errors**

## 🛠️ Troubleshooting

### Common Issues

**"Video not playable" error:**
- Clips are automatically encoded with web-compatible settings
- If issues persist, regenerate clips: `python3 motion_detection.py`

**"No clips found":**
- Ensure videos are in `match_directory/`
- Run `python3 motion_detection.py` to generate clips
- Check `unlabeled_clips/` directory exists

**Database errors:**
- Verify database exists: `ls wrestling_pipeline.db`
- Reinitialize if needed: `python3 database_setup.py`

**FFmpeg not found:**
- Install FFmpeg: `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Ubuntu)

### Performance Tips

- **Large videos**: Process in batches to avoid memory issues
- **Storage**: Use `cleanup_originals.py` to free space after processing
- **Labeling**: Multiple labelers can work simultaneously
- **Backup**: Regularly backup `wrestling_pipeline.db`

## 🔄 Workflow Summary

1. **Add videos** → `match_directory/`
2. **Extract clips** → `python3 motion_detection.py`
3. **Launch UI** → `python3 gradio_app.py`
4. **Label clips** → Web interface
5. **Monitor progress** → `python3 view_database.py`
6. **Add more videos** → Repeat from step 1

## 📈 Future Enhancements

- Machine learning model training on labeled data
- Automatic highlight classification
- Advanced motion analysis algorithms
- Export to various formats (JSON, Excel)
- Integration with video editing software

## 🤝 Contributing

This system is designed for wrestling analysis and can be extended for other sports or video analysis tasks. The modular design allows for easy customization and enhancement.

---

**Built with**: Python, OpenCV, Gradio, SQLite, FFmpeg
**License**: MIT