#!/usr/bin/env python3
"""
Progress checker for wrestling highlight labeling
Run this script to see current labeling progress and statistics
"""

import pandas as pd
import os
from datetime import datetime

CLIP_DIR = "unlabeled_clips"
LABEL_FILE = "labels.csv"

def check_progress():
    print("=== WrestleAI Labeling Progress Report ===\n")
    
    # Check if clips directory exists
    if not os.path.exists(CLIP_DIR):
        print(f"❌ Clips directory '{CLIP_DIR}' not found!")
        return
    
    # Count total clips
    total_clips = len([f for f in os.listdir(CLIP_DIR) if f.endswith(".mp4")])
    print(f"📁 Total clips in directory: {total_clips}")
    
    if total_clips == 0:
        print("❌ No clips found in directory!")
        return
    
    # Load labels
    if os.path.exists(LABEL_FILE):
        df = pd.read_csv(LABEL_FILE)
        print(f"📊 Labels file found: {LABEL_FILE}")
        
        # Calculate statistics
        labeled_clips = len(df[df.label != "Unsure"]) if not df.empty else 0
        highlight_count = len(df[df.label == "Highlight"]) if not df.empty else 0
        not_highlight_count = len(df[df.label == "Not Highlight"]) if not df.empty else 0
        unsure_count = len(df[df.label == "Unsure"]) if not df.empty else 0
        
        # Progress percentage
        progress_pct = (labeled_clips / total_clips) * 100 if total_clips > 0 else 0
        
        print(f"\n📈 Progress: {labeled_clips}/{total_clips} clips labeled ({progress_pct:.1f}%)")
        print(f"🎯 Highlights: {highlight_count}")
        print(f"❌ Not Highlights: {not_highlight_count}")
        print(f"🤔 Unsure (saved): {unsure_count}")
        print(f"⏳ Remaining: {total_clips - labeled_clips}")
        
        # Highlight ratio
        if labeled_clips > 0:
            highlight_ratio = (highlight_count / labeled_clips) * 100
            print(f"\n📊 Highlight Ratio: {highlight_ratio:.1f}%")
        
        # Recent activity
        if not df.empty:
            print(f"\n🕒 Recent Activity:")
            recent = df.tail(5)
            for _, row in recent.iterrows():
                timestamp = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')
                print(f"   {timestamp} - {row['labeler']} labeled '{row['filename']}' as {row['label']}")
        
        # Labeler statistics
        if not df.empty:
            print(f"\n👥 Labeler Activity:")
            labeler_stats = df.groupby('labeler').size().sort_values(ascending=False)
            for labeler, count in labeler_stats.items():
                print(f"   {labeler}: {count} labels")
                
    else:
        print(f"❌ No labels file found: {LABEL_FILE}")
        print(f"📊 Progress: 0/{total_clips} clips labeled (0.0%)")
        print(f"⏳ Remaining: {total_clips}")

if __name__ == "__main__":
    check_progress() 