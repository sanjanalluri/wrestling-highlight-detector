import sqlite3
import os
from datetime import datetime

def create_database():
    """Create the database and tables for the wrestling highlight pipeline"""
    
    # Create database file
    db_path = "database/wrestling_pipeline.db"
    
    # Connect to database (creates it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create videos table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            filepath TEXT NOT NULL,
            file_size INTEGER,
            duration REAL,
            fps REAL,
            resolution TEXT,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_date TIMESTAMP,
            status TEXT DEFAULT 'pending',
            hash TEXT,
            notes TEXT
        )
    ''')
    
    # Create clips table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            filename TEXT UNIQUE NOT NULL,
            filepath TEXT NOT NULL,
            start_time REAL,
            end_time REAL,
            duration REAL,
            motion_score REAL,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos (id)
        )
    ''')
    
    # Create labels table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clip_id INTEGER,
            labeler_name TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL,
            labeled_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            FOREIGN KEY (clip_id) REFERENCES clips (id)
        )
    ''')
    
    # Create processing_log table for tracking pipeline runs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            error_message TEXT,
            FOREIGN KEY (video_id) REFERENCES videos (id)
        )
    ''')

    # Create original_matches table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS original_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            filepath TEXT NOT NULL,
            preview_filepath TEXT,
            duration REAL,
            file_size INTEGER,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create inclusive_match_clips table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inclusive_match_clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            filename TEXT UNIQUE NOT NULL,
            filepath TEXT NOT NULL,
            clip_number INTEGER,
            start_time REAL,
            end_time REAL,
            duration REAL,
            live_label TEXT,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (match_id) REFERENCES original_matches (id)
        )
    ''')

    # Create match_ranges table for manually selected and labeled ranges
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS match_ranges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            range_number INTEGER,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            duration REAL NOT NULL,
            label TEXT NOT NULL CHECK (label IN ('LIVE', 'NOT LIVE')),
            labeler_name TEXT NOT NULL,
            filename TEXT UNIQUE,
            filepath TEXT,
            notes TEXT,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (match_id) REFERENCES original_matches (id)
        )
    ''')

    # Create training_chunks table for processed training data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE NOT NULL,
            label TEXT NOT NULL,
            virtual_start REAL,
            virtual_end REAL,
            duration REAL,
            visual_path TEXT,
            audio_path TEXT,
            metadata_path TEXT,
            range_ids TEXT,  -- JSON array of range IDs this chunk spans
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Add index for better performance when querying by match_id
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_inclusive_clips_match_id 
        ON inclusive_match_clips (match_id)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_match_ranges_match_id 
        ON match_ranges (match_id)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_training_chunks_label 
        ON training_chunks (label)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_training_chunks_chunk_id 
        ON training_chunks (chunk_id)
    ''')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"✅ Database created: {db_path}")
    print("📊 Tables created:")
    print("   - videos: Track source videos and their processing status")
    print("   - clips: Track generated highlight clips")
    print("   - labels: Track clip labels from human annotators")
    print("   - processing_log: Track pipeline execution history")
    print("   - original_matches: Track source match videos")
    print("   - inclusive_match_clips: Track 5-second clips from matches")
    print("   - match_ranges: Track manually selected and labeled time ranges from matches")
    print("   - training_chunks: Track processed training data")

def add_sample_data():
    """Add some sample data to demonstrate the database structure"""
    conn = sqlite3.connect("database/wrestling_pipeline.db")
    cursor = conn.cursor()
    
    # Add a sample video
    cursor.execute('''
        INSERT OR IGNORE INTO videos (filename, filepath, status)
        VALUES (?, ?, ?)
    ''', ("sample_match.mp4", "match_directory/sample_match.mp4", "pending"))
    
    # Get the video ID
    video_id = cursor.lastrowid
    
    # Add a sample clip
    cursor.execute('''
        INSERT OR IGNORE INTO clips (video_id, filename, filepath, start_time, end_time, duration)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (video_id, "sample_match_highlight_1.mp4", "unlabeled_clips/sample_match_highlight_1.mp4", 10.5, 15.0, 4.5))
    
    conn.commit()
    conn.close()
    
    print("📝 Sample data added to demonstrate database structure")

if __name__ == "__main__":
    create_database()
    add_sample_data()
    print("\n🎯 Database setup complete! You can now use the pipeline with database tracking.") 