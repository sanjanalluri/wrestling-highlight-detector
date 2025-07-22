#!/usr/bin/env python3
"""
Utilities for the Sequential Chunk Processor
Check status, statistics, and manage processed chunks
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database_manager import DatabaseManager

def show_processing_status():
    """Show current processing status and statistics"""
    db_manager = DatabaseManager()
    
    print("🏆 Sequential Chunk Processing Status")
    print("=" * 50)
    
    # Get range statistics
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    # Total ranges
    cursor.execute("SELECT COUNT(*) FROM match_ranges")
    total_ranges = cursor.fetchone()[0]
    
    # Ranges by label
    cursor.execute("SELECT label, COUNT(*) FROM match_ranges GROUP BY label")
    ranges_by_label = dict(cursor.fetchall())
    
    # Total chunks
    cursor.execute("SELECT COUNT(*) FROM training_chunks")
    total_chunks = cursor.fetchone()[0]
    
    # Chunks by label
    cursor.execute("SELECT label, COUNT(*) FROM training_chunks GROUP BY label")
    chunks_by_label = dict(cursor.fetchall())
    
    # Unprocessed ranges - check all elements in range_ids arrays
    cursor.execute('SELECT range_ids FROM training_chunks WHERE range_ids IS NOT NULL')
    processed_range_ids = set()
    for row in cursor.fetchall():
        if row[0]:
            range_ids = json.loads(row[0])
            processed_range_ids.update(range_ids)
    
    # Count unprocessed ranges
    cursor.execute('SELECT COUNT(*) FROM match_ranges WHERE id NOT IN ({})'.format(
        ','.join('?' * len(processed_range_ids)) if processed_range_ids else 'NULL'
    ), list(processed_range_ids) if processed_range_ids else [])
    unprocessed_ranges = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"📊 RANGE STATISTICS:")
    print(f"  Total ranges: {total_ranges}")
    print(f"  LIVE ranges: {ranges_by_label.get('LIVE', 0)}")
    print(f"  NOT LIVE ranges: {ranges_by_label.get('NOT LIVE', 0)}")
    print(f"  Unprocessed ranges: {unprocessed_ranges}")
    
    print(f"\n🎯 CHUNK STATISTICS:")
    print(f"  Total chunks: {total_chunks}")
    print(f"  LIVE chunks: {chunks_by_label.get('LIVE', 0)}")
    print(f"  NOT LIVE chunks: {chunks_by_label.get('NOT LIVE', 0)}")
    
    if total_ranges > 0:
        progress_percent = ((total_ranges - unprocessed_ranges) / total_ranges) * 100
        print(f"  Processing progress: {progress_percent:.1f}%")
    
    # Check output directories
    output_dir = "live_filter/training_chunks"
    if os.path.exists(output_dir):
        live_dir = os.path.join(output_dir, "live")
        not_live_dir = os.path.join(output_dir, "not_live")
        
        live_files = len([f for f in os.listdir(live_dir) if f.endswith('.npy')]) if os.path.exists(live_dir) else 0
        not_live_files = len([f for f in os.listdir(not_live_dir) if f.endswith('.npy')]) if os.path.exists(not_live_dir) else 0
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"  LIVE feature files: {live_files}")
        print(f"  NOT LIVE feature files: {not_live_files}")
        print(f"  Total feature files: {live_files + not_live_files}")

def show_chunk_details():
    """Show detailed information about processed chunks"""
    db_manager = DatabaseManager()
    
    print("🔍 CHUNK DETAILS")
    print("=" * 50)
    
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    # Get chunk statistics
    cursor.execute('''
        SELECT label, 
               COUNT(*) as count,
               AVG(duration) as avg_duration,
               MIN(duration) as min_duration,
               MAX(duration) as max_duration
        FROM training_chunks 
        GROUP BY label
    ''')
    
    for row in cursor.fetchall():
        label, count, avg_dur, min_dur, max_dur = row
        print(f"\n🏷️  {label} CHUNKS:")
        print(f"  Count: {count}")
        print(f"  Average duration: {avg_dur:.1f}s")
        print(f"  Duration range: {min_dur:.1f}s - {max_dur:.1f}s")
    
    # Show some example chunks
    cursor.execute('''
        SELECT chunk_id, label, virtual_start, virtual_end, duration, range_ids
        FROM training_chunks 
        ORDER BY created_date DESC 
        LIMIT 5
    ''')
    
    print(f"\n📋 RECENT CHUNKS:")
    for row in cursor.fetchall():
        chunk_id, label, v_start, v_end, duration, range_ids = row
        range_count = len(json.loads(range_ids)) if range_ids else 0
        print(f"  {chunk_id}: {label} ({duration:.1f}s, spans {range_count} ranges)")
    
    conn.close()

def cleanup_temp_files():
    """Clean up temporary files from chunk processing"""
    temp_dir = "live_filter/temp_chunks"
    
    if not os.path.exists(temp_dir):
        print("✅ No temp directory found")
        return
    
    temp_files = [f for f in os.listdir(temp_dir) if f.endswith('.mp4')]
    
    if not temp_files:
        print("✅ No temp files to clean up")
        return
    
    print(f"🧹 Cleaning up {len(temp_files)} temp files...")
    
    for temp_file in temp_files:
        file_path = os.path.join(temp_dir, temp_file)
        try:
            os.remove(file_path)
            print(f"  ✅ Removed: {temp_file}")
        except Exception as e:
            print(f"  ❌ Failed to remove {temp_file}: {e}")
    
    print("✅ Cleanup complete!")

def show_feature_statistics():
    """Show statistics about extracted features"""
    output_dir = "live_filter/training_chunks"
    
    if not os.path.exists(output_dir):
        print("❌ No training chunks directory found")
        return
    
    print("📊 FEATURE STATISTICS")
    print("=" * 50)
    
    for label in ['live', 'not_live']:
        label_dir = os.path.join(output_dir, label)
        if not os.path.exists(label_dir):
            continue
        
        visual_files = [f for f in os.listdir(label_dir) if f.endswith('_visual.npy')]
        audio_files = [f for f in os.listdir(label_dir) if f.endswith('_audio.npy')]
        metadata_files = [f for f in os.listdir(label_dir) if f.endswith('_metadata.json')]
        
        print(f"\n🏷️  {label.upper()} FEATURES:")
        print(f"  Visual feature files: {len(visual_files)}")
        print(f"  Audio feature files: {len(audio_files)}")
        print(f"  Metadata files: {len(metadata_files)}")
        
        # Check file sizes
        if visual_files:
            sample_visual = os.path.join(label_dir, visual_files[0])
            if os.path.exists(sample_visual):
                size_mb = os.path.getsize(sample_visual) / (1024 * 1024)
                print(f"  Visual feature size: {size_mb:.2f} MB per file")
        
        if audio_files:
            sample_audio = os.path.join(label_dir, audio_files[0])
            if os.path.exists(sample_audio):
                size_mb = os.path.getsize(sample_audio) / (1024 * 1024)
                print(f"  Audio feature size: {size_mb:.2f} MB per file")

def main():
    """Main utility interface"""
    print("🔧 Sequential Chunk Processor Utilities")
    print("=" * 40)
    print("1. Show processing status")
    print("2. Show chunk details")
    print("3. Show feature statistics")
    print("4. Clean up temp files")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        show_processing_status()
    elif choice == "2":
        show_chunk_details()
    elif choice == "3":
        show_feature_statistics()
    elif choice == "4":
        cleanup_temp_files()
    elif choice == "5":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 