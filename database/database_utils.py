#!/usr/bin/env python3
"""
Database Utilities for Wrestling Highlight Pipeline
Provides useful commands for monitoring and managing the database
"""

import sqlite3
import argparse
from database_manager import DatabaseManager

def show_status():
    """Show current pipeline status"""
    db_manager = DatabaseManager()
    
    print("🏆 WRESTLING HIGHLIGHT PIPELINE STATUS")
    print("=" * 50)
    
    # Get statistics
    stats = db_manager.get_labeling_progress()
    
    # Video statistics
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM videos")
    total_videos = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM videos WHERE status = 'processed'")
    processed_videos = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM videos WHERE status = 'pending'")
    pending_videos = cursor.fetchone()[0]
    
    print(f"📹 VIDEOS:")
    print(f"   Total: {total_videos}")
    print(f"   Processed: {processed_videos}")
    print(f"   Pending: {pending_videos}")
    
    print(f"\n🎬 CLIPS:")
    print(f"   Total: {stats['total_clips']}")
    print(f"   Labeled: {stats['labeled_clips']}")
    print(f"   Unlabeled: {stats['total_clips'] - stats['labeled_clips']}")
    
    print(f"\n🏷️  LABELS:")
    print(f"   Highlights: {stats['highlight_count']}")
    print(f"   Not Highlights: {stats['not_highlight_count']}")
    print(f"   Progress: {stats['progress_percent']:.1f}%")
    
    # Show recent processing history
    print(f"\n📋 RECENT PROCESSING HISTORY:")
    history = db_manager.get_processing_history()
    for entry in history[:5]:  # Show last 5 entries
        status_emoji = "✅" if entry['status'] == 'completed' else "❌" if entry['status'] == 'failed' else "🔄"
        print(f"   {status_emoji} {entry['filename']} - {entry['stage']} ({entry['status']})")
    
    conn.close()

def list_videos():
    """List all videos in the database"""
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT filename, status, added_date, processed_date, duration, fps
        FROM videos
        ORDER BY added_date DESC
    ''')
    
    print("📹 VIDEOS IN DATABASE:")
    print("=" * 80)
    print(f"{'Filename':<30} {'Status':<12} {'Added':<20} {'Processed':<20} {'Duration':<10}")
    print("-" * 80)
    
    for row in cursor.fetchall():
        filename, status, added_date, processed_date, duration, fps = row
        status_emoji = "✅" if status == 'processed' else "⏳" if status == 'pending' else "❓"
        duration_str = f"{duration:.1f}s" if duration else "N/A"
        processed_str = processed_date[:19] if processed_date else "Not processed"
        added_str = added_date[:19] if added_date else "Unknown"
        
        print(f"{filename:<30} {status_emoji} {status:<10} {added_str:<20} {processed_str:<20} {duration_str:<10}")
    
    conn.close()

def list_clips():
    """List all clips in the database"""
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT c.filename, v.filename as source_video, c.start_time, c.end_time, c.duration,
               CASE WHEN l.label IS NOT NULL THEN l.label ELSE 'Unlabeled' END as label
        FROM clips c
        JOIN videos v ON c.video_id = v.id
        LEFT JOIN (
            SELECT clip_id, label 
            FROM labels 
            WHERE label IN ('Highlight', 'Not Highlight')
            GROUP BY clip_id
        ) l ON c.id = l.clip_id
        ORDER BY c.created_date DESC
    ''')
    
    print("🎬 CLIPS IN DATABASE:")
    print("=" * 100)
    print(f"{'Clip Filename':<35} {'Source Video':<25} {'Time Range':<15} {'Duration':<10} {'Label':<12}")
    print("-" * 100)
    
    for row in cursor.fetchall():
        clip_filename, source_video, start_time, end_time, duration, label = row
        time_range = f"{start_time:.1f}s-{end_time:.1f}s"
        duration_str = f"{duration:.1f}s"
        
        label_emoji = "🎯" if label == "Highlight" else "❌" if label == "Not Highlight" else "🤔" if label == "Unsure" else "📝"
        
        print(f"{clip_filename:<35} {source_video:<25} {time_range:<15} {duration_str:<10} {label_emoji} {label:<10}")
    
    conn.close()

def show_labels():
    """Show labeling statistics and recent labels"""
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    # Get label statistics
    cursor.execute("SELECT label, COUNT(*) FROM labels WHERE label IN ('Highlight', 'Not Highlight') GROUP BY label")
    label_stats = cursor.fetchall()
    
    print("🏷️  LABELING STATISTICS:")
    print("=" * 40)
    for label, count in label_stats:
        emoji = "🎯" if label == "Highlight" else "❌"
        print(f"   {emoji} {label}: {count}")
    
    # Show recent labels
    print(f"\n📝 RECENT LABELS:")
    print("=" * 80)
    cursor.execute('''
        SELECT l.label, l.labeler_name, l.labeled_date, c.filename
        FROM labels l
        JOIN clips c ON l.clip_id = c.id
        WHERE l.label IN ('Highlight', 'Not Highlight')
        ORDER BY l.labeled_date DESC
        LIMIT 10
    ''')
    
    print(f"{'Label':<12} {'Labeler':<15} {'Date':<20} {'Clip':<30}")
    print("-" * 80)
    
    for row in cursor.fetchall():
        label, labeler, date, clip_filename = row
        label_emoji = "🎯" if label == "Highlight" else "❌"
        date_str = date[:19] if date else "Unknown"
        
        print(f"{label_emoji} {label:<10} {labeler:<15} {date_str:<20} {clip_filename:<30}")
    
    conn.close()

def reset_video_status(video_filename):
    """Reset a video's status to pending (useful for reprocessing)"""
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE videos 
        SET status = 'pending', processed_date = NULL 
        WHERE filename = ?
    ''', (video_filename,))
    
    if cursor.rowcount > 0:
        print(f"✅ Reset status for video: {video_filename}")
    else:
        print(f"❌ Video not found: {video_filename}")
    
    conn.commit()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Database utilities for wrestling highlight pipeline")
    parser.add_argument('command', choices=['status', 'videos', 'clips', 'labels', 'reset'],
                       help='Command to run')
    parser.add_argument('--video', help='Video filename for reset command')
    
    args = parser.parse_args()
    
    if args.command == 'status':
        show_status()
    elif args.command == 'videos':
        list_videos()
    elif args.command == 'clips':
        list_clips()
    elif args.command == 'labels':
        show_labels()
    elif args.command == 'reset':
        if not args.video:
            print("❌ Please provide a video filename with --video")
            return
        reset_video_status(args.video)

if __name__ == "__main__":
    main() 