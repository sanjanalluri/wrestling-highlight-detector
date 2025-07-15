#!/usr/bin/env python3
"""
Clean up original videos after processing
This script helps manage storage by removing original videos after clips have been generated
"""

import os
from database.database_manager import DatabaseManager

def cleanup_original_videos():
    """Remove original videos that have been processed and have clips"""
    db_manager = DatabaseManager()
    
    # Get all processed videos
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT v.id, v.filename, v.filepath, 
               COUNT(c.id) as clip_count
        FROM videos v
        LEFT JOIN clips c ON v.id = c.video_id
        WHERE v.status = 'processed'
        GROUP BY v.id, v.filename, v.filepath
    ''')
    
    processed_videos = cursor.fetchall()
    conn.close()
    
    print("🔍 Checking processed videos...")
    
    for video_id, filename, filepath, clip_count in processed_videos:
        if not os.path.exists(filepath):
            print(f"⚠️  {filename} - File already deleted")
            continue
            
        if clip_count == 0:
            print(f"⚠️  {filename} - No clips generated, skipping deletion")
            continue
            
        print(f"🗑️  {filename} - {clip_count} clips generated")
        
        # Ask for confirmation
        response = input(f"Delete {filename}? (y/N): ").strip().lower()
        if response == 'y':
            try:
                os.remove(filepath)
                print(f"✅ Deleted {filename}")
            except Exception as e:
                print(f"❌ Error deleting {filename}: {e}")
        else:
            print(f"⏭️  Skipped {filename}")
    
    print("\n💡 Storage saved! Original videos removed.")
    print("📊 Your clips and database contain all the data you need.")

def show_storage_info():
    """Show current storage usage"""
    db_manager = DatabaseManager()
    
    # Get video info
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT COUNT(*) as total_videos,
               SUM(file_size) as total_size,
               COUNT(CASE WHEN status = 'processed' THEN 1 END) as processed_videos
        FROM videos
    ''')
    
    video_stats = cursor.fetchone()
    
    cursor.execute('SELECT COUNT(*) FROM clips')
    clip_count = cursor.fetchone()[0]
    
    conn.close()
    
    total_size_mb = video_stats[1] / (1024 * 1024) if video_stats[1] else 0
    
    print("📊 Storage Information:")
    print(f"   Total videos: {video_stats[0]}")
    print(f"   Processed videos: {video_stats[2]}")
    print(f"   Total video size: {total_size_mb:.1f} MB")
    print(f"   Generated clips: {clip_count}")
    
    # Check unlabeled_clips directory size
    if os.path.exists('unlabeled_clips'):
        clip_files = os.listdir('unlabeled_clips')
        clip_size = sum(os.path.getsize(os.path.join('unlabeled_clips', f)) for f in clip_files)
        clip_size_mb = clip_size / (1024 * 1024)
        print(f"   Clips directory size: {clip_size_mb:.1f} MB")

if __name__ == "__main__":
    print("🧹 Wrestling Video Cleanup Utility")
    print("=" * 40)
    
    show_storage_info()
    print()
    
    response = input("Clean up original videos? (y/N): ").strip().lower()
    if response == 'y':
        cleanup_original_videos()
    else:
        print("⏭️  No cleanup performed") 