#!/usr/bin/env python3
"""
Groot Terminal Highlight Labeler
Label highlight clips from 12 fps match videos for the Groot pipeline
"""

import os
import sys
import sqlite3
from pathlib import Path
import subprocess
import platform

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DATABASE_PATH, TABLE_NAMES

def format_time(seconds):
    """Convert seconds to MM:SS format"""
    if seconds is None:
        return "Unknown"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

def parse_time(time_str):
    """Convert MM:SS format to seconds"""
    if not time_str or ':' not in time_str:
        return 0
    
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        return 0
    except:
        return 0

def get_12fps_videos():
    """Get list of all 12 fps videos from database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, video_path, video_name, duration_seconds
        FROM groot_videos 
        WHERE processing_status = 'converted' AND original_fps = 12.0
        ORDER BY video_name
    ''')
    
    videos = cursor.fetchall()
    conn.close()
    
    return [{'id': v[0], 'video_path': v[1], 'video_name': v[2], 'duration': v[3]} for v in videos]

def get_existing_clips(video_id):
    """Get existing highlight clips for a video"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, start_time_seconds, end_time_seconds, is_highlight, notes
        FROM groot_highlight_clips 
        WHERE video_id = ?
        ORDER BY start_time_seconds
    ''', (video_id,))
    
    clips = cursor.fetchall()
    conn.close()
    
    return [{'id': c[0], 'start_time': c[1], 'end_time': c[2], 'is_highlight': c[3], 'notes': c[4]} for c in clips]

def add_highlight_clip(video_id, start_time, end_time, is_highlight, labeler_name, notes=""):
    """Add a highlight clip to the database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        duration = end_time - start_time
        
        cursor.execute('''
            INSERT INTO groot_highlight_clips 
            (video_id, start_time_seconds, end_time_seconds, duration_seconds, 
             is_highlight, confidence, notes, labeled_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (video_id, start_time, end_time, duration, is_highlight, 1.0, notes, labeler_name))
        
        clip_id = cursor.lastrowid
        conn.commit()
        
        highlight_status = "HIGHLIGHT" if is_highlight else "NOT HIGHLIGHT"
        print(f"✅ Added {highlight_status} clip: {format_time(start_time)} → {format_time(end_time)} (ID: {clip_id})")
        return True
        
    except Exception as e:
        print(f"❌ Error adding clip: {str(e)}")
        return False
    finally:
        conn.close()

def play_video_segment(video_path, start_time, end_time):
    """Play a video segment using system video player"""
    print(f"🎬 Playing segment: {format_time(start_time)} → {format_time(end_time)}")
    
    # Determine video player based on platform
    if platform.system() == "Darwin":  # macOS
        player = "open"
        # Use QuickTime with time offset
        cmd = [player, "-a", "QuickTime Player", video_path]
    elif platform.system() == "Linux":
        player = "vlc"
        cmd = [player, video_path, "--start-time", str(start_time), "--stop-time", str(end_time)]
    elif platform.system() == "Windows":
        player = "start"
        cmd = [player, video_path]
    else:
        print("⚠️  Unknown platform, trying default video player")
        cmd = [video_path]
    
    try:
        subprocess.Popen(cmd)
        print(f"🎥 Video player launched")
    except Exception as e:
        print(f"❌ Could not launch video player: {e}")
        print(f"   Please open {video_path} manually and navigate to {format_time(start_time)}")

def show_video_info(video):
    """Display video information"""
    print(f"\n🎬 Video: {video['video_name']}")
    print(f"📊 Duration: {format_time(video['duration'])}")
    print(f"📁 Path: {video['video_path']}")

def label_video_highlights(video, labeler_name):
    """Label highlight clips for a specific video"""
    show_video_info(video)
    
    # Show existing clips
    existing_clips = get_existing_clips(video['id'])
    if existing_clips:
        print(f"\n📋 Existing clips ({len(existing_clips)}):")
        for i, clip in enumerate(existing_clips, 1):
            start_str = format_time(clip['start_time'])
            end_str = format_time(clip['end_time'])
            highlight_status = "HIGHLIGHT" if clip['is_highlight'] else "NOT HIGHLIGHT"
            print(f"  {i:2d}. {start_str} → {end_str} ({highlight_status})")
            if clip['notes']:
                print(f"      Notes: {clip['notes']}")
    
    print(f"\n🎯 Labeling highlights for: {video['video_name']}")
    print("Commands:")
    print("  <start> <end> highlight    - Add highlight clip (e.g., '1:30 2:45 highlight')")
    print("  <start> <end> not          - Add non-highlight clip (e.g., '1:30 2:45 not')")
    print("  play <start> <end>         - Play video segment")
    print("  list                       - Show existing clips")
    print("  done                       - Finish labeling this video")
    print("  help                       - Show this help")
    
    while True:
        user_input = input(f"\n[{video['video_name']}] Enter command: ").strip().lower()
        
        if user_input == 'done':
            print(f"✅ Finished labeling {video['video_name']}")
            break
        
        elif user_input == 'help':
            print("Commands:")
            print("  <start> <end> highlight    - Add highlight clip (e.g., '1:30 2:45 highlight')")
            print("  <start> <end> not          - Add non-highlight clip (e.g., '1:30 2:45 not')")
            print("  play <start> <end>         - Play video segment")
            print("  list                       - Show existing clips")
            print("  done                       - Finish labeling this video")
            print("  help                       - Show this help")
        
        elif user_input == 'list':
            existing_clips = get_existing_clips(video['id'])
            if existing_clips:
                print(f"\n📋 Existing clips ({len(existing_clips)}):")
                for i, clip in enumerate(existing_clips, 1):
                    start_str = format_time(clip['start_time'])
                    end_str = format_time(clip['end_time'])
                    highlight_status = "HIGHLIGHT" if clip['is_highlight'] else "NOT HIGHLIGHT"
                    print(f"  {i:2d}. {start_str} → {end_str} ({highlight_status})")
                    if clip['notes']:
                        print(f"      Notes: {clip['notes']}")
            else:
                print("📋 No clips labeled yet")
        
        elif user_input.startswith('play '):
            # Parse play command: play <start> <end>
            parts = user_input.split()
            if len(parts) != 3:
                print("❌ Usage: play <start> <end> (e.g., 'play 1:30 2:45')")
                continue
            
            start_str, end_str = parts[1], parts[2]
            start_time = parse_time(start_str)
            end_time = parse_time(end_str)
            
            if start_time == 0 or end_time == 0:
                print("❌ Invalid time format. Use MM:SS (e.g., '1:30')")
                continue
            
            if start_time >= end_time:
                print("❌ Start time must be before end time")
                continue
            
            if end_time > video['duration']:
                print(f"❌ End time exceeds video duration ({format_time(video['duration'])})")
                continue
            
            play_video_segment(video['video_path'], start_time, end_time)
        
        else:
            # Parse clip command: <start> <end> <highlight/not>
            parts = user_input.split()
            if len(parts) != 3:
                print("❌ Usage: <start> <end> <highlight/not> (e.g., '1:30 2:45 highlight')")
                continue
            
            start_str, end_str, highlight_type = parts[0], parts[1], parts[2]
            
            if highlight_type not in ['highlight', 'not']:
                print("❌ Third parameter must be 'highlight' or 'not'")
                continue
            
            start_time = parse_time(start_str)
            end_time = parse_time(end_str)
            
            if start_time == 0 or end_time == 0:
                print("❌ Invalid time format. Use MM:SS (e.g., '1:30')")
                continue
            
            if start_time >= end_time:
                print("❌ Start time must be before end time")
                continue
            
            if end_time > video['duration']:
                print(f"❌ End time exceeds video duration ({format_time(video['duration'])})")
                continue
            
            # Check for overlapping clips
            existing_clips = get_existing_clips(video['id'])
            overlap_found = False
            for clip in existing_clips:
                if (start_time < clip['end_time'] and end_time > clip['start_time']):
                    print(f"⚠️  Warning: Overlaps with existing clip {format_time(clip['start_time'])} → {format_time(clip['end_time'])}")
                    overlap_found = True
            
            if overlap_found:
                proceed = input("Continue anyway? (y/n): ").strip().lower()
                if proceed != 'y':
                    continue
            
            # Add the clip
            is_highlight = (highlight_type == 'highlight')
            notes = input("Notes (optional): ").strip()
            
            if add_highlight_clip(video['id'], start_time, end_time, is_highlight, labeler_name, notes):
                print(f"✅ Added {'HIGHLIGHT' if is_highlight else 'NOT HIGHLIGHT'} clip: {start_str} → {end_str}")

def show_labeling_stats():
    """Show statistics about labeled clips"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get total clips
    cursor.execute('SELECT COUNT(*) FROM groot_highlight_clips')
    total_clips = cursor.fetchone()[0]
    
    # Get highlight clips
    cursor.execute('SELECT COUNT(*) FROM groot_highlight_clips WHERE is_highlight = 1')
    highlight_clips = cursor.fetchone()[0]
    
    # Get non-highlight clips
    cursor.execute('SELECT COUNT(*) FROM groot_highlight_clips WHERE is_highlight = 0')
    non_highlight_clips = cursor.fetchone()[0]
    
    # Get videos with clips
    cursor.execute('SELECT COUNT(DISTINCT video_id) FROM groot_highlight_clips')
    videos_with_clips = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\n📊 Labeling Statistics:")
    print(f"   Total clips: {total_clips}")
    print(f"   Highlight clips: {highlight_clips}")
    print(f"   Non-highlight clips: {non_highlight_clips}")
    print(f"   Videos with clips: {videos_with_clips}")

def analyze_highlight_patterns():
    """Analyze highlight clip patterns to determine optimal clip length"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get all highlight clips
    cursor.execute('''
        SELECT duration_seconds 
        FROM groot_highlight_clips 
        WHERE is_highlight = 1
        ORDER BY duration_seconds
    ''')
    
    highlight_durations = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    if not highlight_durations:
        print("❌ No highlight clips found. Please label some highlights first.")
        return None
    
    # Calculate statistics
    avg_duration = sum(highlight_durations) / len(highlight_durations)
    min_duration = min(highlight_durations)
    max_duration = max(highlight_durations)
    
    # Calculate percentiles
    sorted_durations = sorted(highlight_durations)
    median = sorted_durations[len(sorted_durations) // 2]
    p75 = sorted_durations[int(len(sorted_durations) * 0.75)]
    p90 = sorted_durations[int(len(sorted_durations) * 0.90)]
    
    print(f"\n📊 Highlight Clip Analysis ({len(highlight_durations)} clips):")
    print(f"   Average duration: {avg_duration:.1f}s")
    print(f"   Median duration: {median:.1f}s")
    print(f"   75th percentile: {p75:.1f}s")
    print(f"   90th percentile: {p90:.1f}s")
    print(f"   Min duration: {min_duration:.1f}s")
    print(f"   Max duration: {max_duration:.1f}s")
    
    # Recommend optimal clip length
    # Use 75th percentile as target, but ensure it's reasonable
    recommended_length = min(max(p75, 3.0), 10.0)  # Between 3-10 seconds
    
    print(f"\n🎯 Recommended clip length: {recommended_length:.1f}s")
    print(f"   Based on 75th percentile of highlight durations")
    
    return {
        'avg_duration': avg_duration,
        'median_duration': median,
        'p75_duration': p75,
        'recommended_length': recommended_length,
        'total_clips': len(highlight_durations)
    }

def get_unsplit_non_highlight_clips():
    """Get non-highlight clips that haven't been split yet"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, video_id, start_time_seconds, end_time_seconds, duration_seconds, notes
        FROM groot_highlight_clips 
        WHERE is_highlight = 0 AND duration_seconds > 10
        ORDER BY video_id, start_time_seconds
    ''')
    
    clips = cursor.fetchall()
    conn.close()
    
    return [{'id': c[0], 'video_id': c[1], 'start_time': c[2], 'end_time': c[3], 'duration': c[4], 'notes': c[5]} for c in clips]

def split_large_non_highlight_clips(target_length):
    """Split large non-highlight clips into smaller chunks"""
    print(f"🔪 Splitting large non-highlight clips into {target_length:.1f}s chunks...")
    
    unsplit_clips = get_unsplit_non_highlight_clips()
    if not unsplit_clips:
        print("✅ No large non-highlight clips to split")
        return
    
    print(f"📦 Found {len(unsplit_clips)} large non-highlight clips to split")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    total_new_clips = 0
    
    for clip in unsplit_clips:
        print(f"\n🔪 Splitting clip {format_time(clip['start_time'])} → {format_time(clip['end_time'])} ({clip['duration']:.1f}s)")
        
        # Calculate number of chunks
        num_chunks = int(clip['duration'] // target_length)
        if num_chunks < 1:
            print(f"   ⏭️  Too short to split (duration: {clip['duration']:.1f}s)")
            continue
        
        # Create chunks
        chunks_created = 0
        current_start = clip['start_time']
        
        for i in range(num_chunks):
            chunk_end = min(current_start + target_length, clip['end_time'])
            chunk_duration = chunk_end - current_start
            
            if chunk_duration < 1.0:  # Skip very short chunks
                break
            
            try:
                cursor.execute('''
                    INSERT INTO groot_highlight_clips 
                    (video_id, start_time_seconds, end_time_seconds, duration_seconds, 
                     is_highlight, confidence, notes, labeled_by, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    clip['video_id'], 
                    current_start, 
                    chunk_end, 
                    chunk_duration, 
                    0,  # is_highlight = False
                    1.0,  # confidence
                    f"Auto-split from large clip (ID: {clip['id']})", 
                    "auto_split"
                ))
                
                chunks_created += 1
                print(f"   ✅ Created chunk {i+1}: {format_time(current_start)} → {format_time(chunk_end)} ({chunk_duration:.1f}s)")
                
            except Exception as e:
                print(f"   ❌ Error creating chunk: {e}")
            
            current_start = chunk_end
        
        # Delete the original large clip
        try:
            cursor.execute('DELETE FROM groot_highlight_clips WHERE id = ?', (clip['id'],))
            print(f"   🗑️  Deleted original large clip (ID: {clip['id']})")
        except Exception as e:
            print(f"   ❌ Error deleting original clip: {e}")
        
        total_new_clips += chunks_created
        print(f"   📊 Created {chunks_created} chunks from this clip")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Split complete: {total_new_clips} new clips created")

def show_clip_statistics():
    """Show detailed statistics about all clips"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get all clips
    cursor.execute('''
        SELECT is_highlight, duration_seconds, COUNT(*) as count
        FROM groot_highlight_clips 
        GROUP BY is_highlight
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    highlight_clips = []
    non_highlight_clips = []
    
    for is_highlight, duration, count in results:
        if is_highlight:
            highlight_clips.append((duration, count))
        else:
            non_highlight_clips.append((duration, count))
    
    print(f"\n📊 Detailed Clip Statistics:")
    
    if highlight_clips:
        print(f"\n🎯 Highlight Clips:")
        for duration, count in highlight_clips:
            print(f"   {duration:.1f}s: {count} clips")
    
    if non_highlight_clips:
        print(f"\n❌ Non-Highlight Clips:")
        for duration, count in non_highlight_clips:
            print(f"   {duration:.1f}s: {count} clips")
    
    # Show large clips that might need splitting
    large_clips = get_unsplit_non_highlight_clips()
    if large_clips:
        print(f"\n⚠️  Large Non-Highlight Clips (may need splitting):")
        for clip in large_clips:
            print(f"   {format_time(clip['start_time'])} → {format_time(clip['end_time'])} ({clip['duration']:.1f}s)")

def main():
    """Main terminal interface"""
    print("🌱 Groot Terminal Highlight Labeler")
    print("=" * 50)
    print("Label highlight clips from 12 fps match videos")
    print("=" * 50)
    
    # Get labeler name
    labeler_name = input("Enter your name: ").strip()
    if not labeler_name:
        print("❌ Name is required")
        return
    
    while True:
        print("\n" + "=" * 50)
        
        # Show available videos
        videos = get_12fps_videos()
        if not videos:
            print("❌ No 12 fps videos found in database")
            print("   Make sure to run the frame rate converter first")
            return
        
        print("Available 12 fps videos:")
        for i, video in enumerate(videos, 1):
            duration_str = format_time(video['duration'])
            print(f"{i:2d}. {video['video_name']} ({duration_str})")
        
        print("\nCommands:")
        print("  <number> - Select video to label")
        print("  stats    - Show labeling statistics")
        print("  analyze  - Analyze highlight patterns")
        print("  split    - Split large non-highlight clips")
        print("  quit     - Exit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == 'quit':
            show_labeling_stats()
            print("👋 Goodbye!")
            break
        
        elif choice == 'stats':
            show_labeling_stats()
            show_clip_statistics()
        
        elif choice == 'analyze':
            patterns = analyze_highlight_patterns()
            if patterns:
                print(f"\n💡 Based on {patterns['total_clips']} highlight clips")
                print(f"   Recommended clip length: {patterns['recommended_length']:.1f}s")
        
        elif choice == 'split':
            # First analyze patterns
            patterns = analyze_highlight_patterns()
            if not patterns:
                continue
            
            # Ask for confirmation
            print(f"\n🔪 This will split all large non-highlight clips into {patterns['recommended_length']:.1f}s chunks")
            confirm = input("Continue? (y/n): ").strip().lower()
            
            if confirm == 'y':
                split_large_non_highlight_clips(patterns['recommended_length'])
                print("\n✅ Split complete! Run 'stats' to see the results.")
        
        else:
            # Select video to label
            try:
                video_num = int(choice) - 1
                if 0 <= video_num < len(videos):
                    video = videos[video_num]
                    label_video_highlights(video, labeler_name)
                else:
                    print("❌ Invalid video number")
            except ValueError:
                print("❌ Please enter a valid number")

if __name__ == "__main__":
    main()
