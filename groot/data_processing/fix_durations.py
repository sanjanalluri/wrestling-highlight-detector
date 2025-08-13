#!/usr/bin/env python3
"""
Fix duration_seconds for 12fps videos in the database
"""

import sqlite3
import cv2
from pathlib import Path
import sys

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DATABASE_PATH

# Override database path for this script
DATABASE_PATH = "../database/wrestling_pipeline.db"

def fix_12fps_durations():
    """Fix duration_seconds for all 12fps videos"""
    print("🔧 Fixing duration_seconds for 12fps videos...")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get all 12fps videos with NULL duration
    cursor.execute('''
        SELECT id, video_path, video_name
        FROM groot_videos 
        WHERE original_fps = 12.0 AND duration_seconds IS NULL
    ''')
    
    videos = cursor.fetchall()
    conn.close()
    
    if not videos:
        print("✅ All 12fps videos already have duration set")
        return
    
    print(f"📹 Found {len(videos)} 12fps videos with missing duration")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    fixed_count = 0
    
    for video_id, video_path, video_name in videos:
        print(f"\n🔍 Processing: {video_name}")
        
        # Check if video file exists
        if not Path(video_path).exists():
            print(f"   ❌ Video file not found: {video_path}")
            continue
        
        try:
            # Open video and get duration
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"   ❌ Could not open video: {video_path}")
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Update database
            cursor.execute('''
                UPDATE groot_videos 
                SET duration_seconds = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (duration, video_id))
            
            print(f"   ✅ Updated duration: {duration:.1f}s ({frame_count} frames at {fps:.1f} fps)")
            fixed_count += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {video_name}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Fixed duration for {fixed_count}/{len(videos)} videos")

def verify_fix():
    """Verify that all 12fps videos now have duration"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT video_name, duration_seconds, original_fps
        FROM groot_videos 
        WHERE original_fps = 12.0
        ORDER BY video_name
    ''')
    
    videos = cursor.fetchall()
    conn.close()
    
    print(f"\n📊 Verification - 12fps videos:")
    for video_name, duration, fps in videos:
        if duration is None:
            print(f"   ❌ {video_name}: Missing duration")
        else:
            print(f"   ✅ {video_name}: {duration:.1f}s")

if __name__ == "__main__":
    fix_12fps_durations()
    verify_fix()
