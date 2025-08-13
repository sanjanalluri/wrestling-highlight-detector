#!/usr/bin/env python3
"""
Reconvert existing 12fps videos with H.264 codec for better compatibility
"""

import cv2
import os
from pathlib import Path
import sqlite3
from tqdm import tqdm
import sys

# Database path
DATABASE_PATH = "../database/wrestling_pipeline.db"

def reconvert_12fps_videos():
    """Reconvert existing 12fps videos with H.264 codec"""
    print("🔄 Reconverting 12fps videos with H.264 codec...")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get all 12fps videos
    cursor.execute('''
        SELECT id, video_path, video_name, duration_seconds
        FROM groot_videos 
        WHERE original_fps = 12.0 AND processing_status = 'converted'
        ORDER BY video_name
    ''')
    
    videos = cursor.fetchall()
    conn.close()
    
    if not videos:
        print("❌ No 12fps videos found")
        return
    
    print(f"📹 Found {len(videos)} 12fps videos to reconvert")
    
    for video_id, video_path, video_name, duration in videos:
        print(f"\n🎬 Reconverting: {video_name}")
        
        if not Path(video_path).exists():
            print(f"   ❌ Video file not found: {video_path}")
            continue
        
        try:
            # Open original video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"   ❌ Could not open video")
                continue
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create new video writer with H.264 codec
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                print(f"   ✅ Using H.264 codec")
            except:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Fallback
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                print(f"   ⚠️  Using mp4v codec (fallback)")
            
            if not out.isOpened():
                print(f"   ❌ Could not create output video")
                cap.release()
                continue
            
            # Copy all frames
            print(f"   🔄 Copying {frame_count} frames...")
            with tqdm(total=frame_count, desc="Copying frames", leave=False) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    out.write(frame)
                    pbar.update(1)
            
            # Clean up
            cap.release()
            out.release()
            
            # Verify the new video
            verify_cap = cv2.VideoCapture(video_path)
            if verify_cap.isOpened():
                actual_fps = verify_cap.get(cv2.CAP_PROP_FPS)
                actual_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                verify_cap.release()
                
                print(f"   ✅ Reconversion successful: {actual_fps:.1f} fps, {actual_frames} frames")
            else:
                print(f"   ❌ Verification failed")
            
        except Exception as e:
            print(f"   ❌ Error reconverting {video_name}: {e}")
    
    print(f"\n✅ Reconversion complete!")

if __name__ == "__main__":
    reconvert_12fps_videos()
