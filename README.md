# WrestleAI Highlight Detector

An AI-powered tool for detecting and labeling wrestling match highlights using motion detection and machine learning.

## Features

- **Motion Detection**: Automatically extracts potential highlight moments from wrestling videos
- **Web-based Labeling**: Easy-to-use interface for labeling clips as highlights or not
- **Database Management**: Persistent storage of video metadata and labels
- **Progress Tracking**: Real-time statistics on labeling progress

## Quick Start

### Local Development

1. **Setup Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add Videos**:
   - Place wrestling videos in the `match_directory/` folder
   - Supported formats: MP4, AVI, MOV, MKV, WMV

3. **Process Videos**:
   ```bash
   python motion_detection.py
   ```

4. **Start Labeling**:
   ```bash
   python gradio_app.py
   ```
   - Open your browser to the provided URL
   - Enter your name and start labeling clips

## Deployment Options

### Option 1: Hugging Face Spaces (Recommended)

**Free, permanent hosting with automatic deployment**

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co)

2. **Create a new Space**:
   - Go to your profile → "New Space"
   - Choose "Gradio" as the SDK
   - Name it something like `wrestling-highlight-detector`

3. **Upload your files**:
   - `app.py` (use the deployment version I created)
   - `requirements.txt`
   - `database/` folder with all database files
   - Your `unlabeled_clips/` folder with processed videos
   - Your `database/wrestling_pipeline.db` database file

4. **Your app will be live** at: `https://yourusername-wrestling-highlight-detector.hf.space`

### Option 2: Railway (Easy Cloud Deployment)

1. **Sign up** at [railway.app](https://railway.app)
2. **Connect your GitHub repository**
3. **Deploy automatically** - Railway will detect your Python app
4. **Get a permanent URL** like: `https://your-app-name.railway.app`

### Option 3: Render (Free Tier Available)

1. **Sign up** at [render.com](https://render.com)
2. **Create a new Web Service**
3. **Connect your repository**
4. **Deploy** - Render handles the rest

### Option 4: Keep Local (Temporary Sharing)

If you want to share temporarily while your laptop is on:

```bash
python gradio_app.py
# This creates a temporary public URL that others can access
# Note: Your laptop must stay on and connected to the internet
```

## File Structure

```
wrestling-highlight-detector/
├── match_directory/          # Source wrestling videos
├── unlabeled_clips/         # Generated highlight clips
├── database/                # Database files and utilities
│   ├── wrestling_pipeline.db   # SQLite database
│   ├── database_manager.py     # Database operations
│   ├── database_setup.py       # Database initialization
│   ├── database_utils.py       # Database utilities
│   └── view_database.py        # Database viewer
├── motion_detection.py      # Video processing script
├── gradio_app.py           # Local web interface
├── wrestling_model.py      # 3D CNN model training
├── label_stats.py          # Labeling statistics
├── add_gender_field.py     # Gender categorization tool
├── requirements.txt        # Python dependencies
└── README.md
```

## Database Management

- **View Progress**: `python label_stats.py`
- **View Database**: `python database/view_database.py`
- **Database Utils**: `python database/database_utils.py status`
- **Add Gender Tags**: `python add_gender_field.py`

## Troubleshooting

### Common Issues

1. **"No video files found"**: Make sure videos are in `match_directory/` folder
2. **"FFmpeg not found"**: Install FFmpeg on your system
3. **Database errors**: Run `python database/database_setup.py` to reset the database

### Deployment Issues

1. **Missing dependencies**: Check `requirements.txt` is complete
2. **File paths**: Make sure all files are uploaded to the cloud platform
3. **Database access**: Ensure the database file is included in deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Feel free to use and modify for your own wrestling analysis needs!