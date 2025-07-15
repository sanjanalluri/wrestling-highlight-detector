#!/usr/bin/env python3
"""
Check labeling progress and statistics
Shows current progress and statistics for the labeling process
"""

import pandas as pd
import os
from database.database_manager import DatabaseManager

def check_progress():
    """Check current labeling progress"""
    print("📊 Wrestling Highlight Labeling Progress")
    print("=" * 50)
    
    # Get progress from database
    db_manager = DatabaseManager()
    progress = db_manager.get_labeling_progress()
    
    if progress['total_clips'] == 0:
        print("❌ No clips found in database")
        return
    
    print(f"🎬 Total clips: {progress['total_clips']}")
    print(f"🏷️  Labeled clips: {progress['labeled_clips']}")
    print(f"📈 Progress: {progress['progress_percent']:.1f}%")
    print(f"🎯 Highlights: {progress['highlight_count']}")
    print(f"❌ Not highlights: {progress['not_highlight_count']}")
    
    # Calculate remaining work
    remaining = progress['total_clips'] - progress['labeled_clips']
    if remaining > 0:
        print(f"\n⏳ Remaining clips to label: {remaining}")
        
        # Estimate time (assuming 30 seconds per clip)
        estimated_minutes = remaining * 0.5
        print(f"⏰ Estimated time remaining: {estimated_minutes:.1f} minutes")
    
    # Show recent activity
    print(f"\n📋 Recent Activity:")
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT l.label, l.labeler_name, l.labeled_date, c.filename
        FROM labels l
        JOIN clips c ON l.clip_id = c.id
        WHERE l.label IN ('Highlight', 'Not Highlight')
        ORDER BY l.labeled_date DESC
        LIMIT 5
    ''')
    
    recent_labels = cursor.fetchall()
    if recent_labels:
        for label, labeler, date, filename in recent_labels:
            emoji = "🎯" if label == "Highlight" else "❌"
            print(f"   {emoji} {label} by {labeler} - {filename}")
    else:
        print("   No recent labels found")
    
    conn.close()

if __name__ == "__main__":
    check_progress() 