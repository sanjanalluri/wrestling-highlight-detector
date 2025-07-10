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
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"✅ Database created: {db_path}")
    print("📊 Tables created:")
    print("   - videos: Track source videos and their processing status")
    print("   - clips: Track generated highlight clips")
    print("   - labels: Track clip labels from human annotators")
    print("   - processing_log: Track pipeline execution history")

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