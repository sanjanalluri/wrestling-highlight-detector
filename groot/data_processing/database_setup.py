"""
Database setup for Groot data processing pipeline.
Creates new tables separate from the legacy live filter implementation.
"""

import sqlite3
import os
from pathlib import Path
import sys

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DATABASE_PATH, TABLE_NAMES

def create_groot_tables():
    """Create new database tables for the Groot pipeline"""
    
    # Ensure database directory exists
    db_dir = Path(DATABASE_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    print("🔧 Creating Groot data processing tables...")
    
    # Table 1: groot_videos - Video metadata and frame rate info
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS groot_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_path TEXT UNIQUE NOT NULL,
            video_name TEXT NOT NULL,
            duration_seconds REAL,
            original_fps REAL,
            frame_count INTEGER,
            width INTEGER,
            height INTEGER,
            file_size_bytes INTEGER,
            processing_status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print(f"✅ Created table: {TABLE_NAMES['groot_videos']}")
    
    # Table 2: groot_highlight_clips - Manually labeled highlight clips
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS groot_highlight_clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            start_time_seconds REAL NOT NULL,
            end_time_seconds REAL NOT NULL,
            duration_seconds REAL NOT NULL,
            is_highlight BOOLEAN NOT NULL,
            confidence REAL DEFAULT 1.0,
            notes TEXT,
            labeled_by TEXT DEFAULT 'manual',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES groot_videos (id)
        )
    ''')
    print(f"✅ Created table: {TABLE_NAMES['groot_highlight_clips']}")
    
    # Table 3: groot_processed_frames - Final 16-frame samples
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS groot_processed_frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clip_id INTEGER NOT NULL,
            frame_index INTEGER NOT NULL,  -- 0 to 15
            frame_path TEXT NOT NULL,
            original_timestamp REAL,  -- Original timestamp in video
            processed_timestamp REAL,  -- Timestamp after processing
            ssim_score REAL,  -- SSIM score with previous frame
            optical_flow_magnitude REAL,  -- Average optical flow magnitude
            is_duplicate BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (clip_id) REFERENCES groot_highlight_clips (id)
        )
    ''')
    print(f"✅ Created table: {TABLE_NAMES['groot_processed_frames']}")
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_groot_videos_path ON groot_videos(video_path)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_groot_clips_video_id ON groot_highlight_clips(video_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_groot_clips_highlight ON groot_highlight_clips(is_highlight)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_groot_frames_clip_id ON groot_processed_frames(clip_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_groot_frames_index ON groot_processed_frames(frame_index)')
    
    conn.commit()
    conn.close()
    
    print("✅ All Groot data processing tables created successfully!")

def verify_tables():
    """Verify that all required tables exist"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    required_tables = [
        TABLE_NAMES['groot_videos'],
        TABLE_NAMES['groot_highlight_clips'], 
        TABLE_NAMES['groot_processed_frames']
    ]
    
    print("🔍 Verifying Groot tables...")
    for table_name in required_tables:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if cursor.fetchone():
            print(f"✅ Table exists: {table_name}")
        else:
            print(f"❌ Table missing: {table_name}")
    
    conn.close()

if __name__ == "__main__":
    create_groot_tables()
    verify_tables()
