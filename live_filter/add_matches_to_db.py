#!/usr/bin/env python3
"""
Add all videos from match_directory to the original_matches table
"""

import os
import glob
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database_manager import DatabaseManager

def add_matches_to_database():
    """Add all videos from match_directory to the database"""
    db_manager = DatabaseManager()
    
    # Check if match directory exists
    match_dir = "match_directory"
    if not os.path.exists(match_dir):
        print(f"❌ Error: {match_dir} directory not found!")
        return
    
    # Find all video files in match directory
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(match_dir, ext)))
        video_files.extend(glob.glob(os.path.join(match_dir, ext.upper())))
    
    if not video_files:
        print(f"❌ No video files found in {match_dir} directory!")
        return
    
    print(f"🎬 Found {len(video_files)} video(s) in {match_dir}:")
    for video_file in video_files:
        print(f"   - {os.path.basename(video_file)}")
    
    # Add each video to the database
    added_count = 0
    for video_file in video_files:
        try:
            filename = os.path.basename(video_file)
            file_size = os.path.getsize(video_file)
            
            # Get video duration using OpenCV
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"❌ Could not open video: {filename}")
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            # Check if match already exists
            conn = db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM original_matches WHERE filename = ?", (filename,))
            existing = cursor.fetchone()
            
            if existing:
                print(f"⏭️  Skipped {filename} (already exists)")
                conn.close()
                continue
            
            # Add new match
            cursor.execute('''
                INSERT INTO original_matches (filename, filepath, duration, file_size)
                VALUES (?, ?, ?, ?)
            ''', (filename, video_file, duration, file_size))
            
            match_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"✅ Added {filename} (ID: {match_id}, Duration: {duration:.1f}s)")
            added_count += 1
            
        except Exception as e:
            print(f"❌ Error adding {os.path.basename(video_file)}: {str(e)}")
    
    print(f"\n🎯 Added {added_count} new matches to database")
    
    # Show final count
    matches = db_manager.get_all_matches()
    print(f"📊 Total matches in database: {len(matches)}")

if __name__ == "__main__":
    add_matches_to_database() 