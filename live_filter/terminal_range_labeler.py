#!/usr/bin/env python3
"""
Terminal-based range labeling tool
Watch videos locally and enter timestamps for live sections
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database_manager import DatabaseManager

def format_time(seconds):
    """Convert seconds to MM:SS format"""
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

def get_matches_list():
    """Get list of all matches"""
    db_manager = DatabaseManager()
    matches = db_manager.get_all_matches()
    return matches

def get_match_ranges(match_id):
    """Get existing ranges for a match"""
    db_manager = DatabaseManager()
    ranges = db_manager.get_ranges_for_match(match_id)
    return ranges

def add_live_range(match_id, start_time, end_time, labeler_name, notes=""):
    """Add a live range to the database"""
    db_manager = DatabaseManager()
    try:
        range_id = db_manager.add_match_range(match_id, start_time, end_time, "LIVE", labeler_name, notes)
        print(f"✅ Added LIVE range: {format_time(start_time)} → {format_time(end_time)} (ID: {range_id})")
        return True
    except Exception as e:
        print(f"❌ Error adding range: {str(e)}")
        return False

def generate_not_live_ranges(match_id, labeler_name):
    """Generate NOT LIVE ranges for all remaining time periods"""
    db_manager = DatabaseManager()
    match = db_manager.get_match_by_id(match_id)
    if not match:
        print("❌ Match not found")
        return
    
    # Get existing ranges
    existing_ranges = db_manager.get_ranges_for_match(match_id)
    
    # Sort existing ranges by start time
    existing_ranges.sort(key=lambda x: x['start_time'])
    
    # Find gaps and create NOT LIVE ranges
    video_duration = match['duration']
    not_live_ranges = []
    
    current_time = 0.0
    
    for range_info in existing_ranges:
        # If there's a gap before this range, it's NOT LIVE
        if range_info['start_time'] > current_time:
            gap_start = current_time
            gap_end = range_info['start_time']
            not_live_ranges.append((gap_start, gap_end))
        
        # Move to the end of this range
        current_time = range_info['end_time']
    
    # If there's remaining time after the last range, it's NOT LIVE
    if current_time < video_duration:
        not_live_ranges.append((current_time, video_duration))
    
    # Add all NOT LIVE ranges to database
    added_count = 0
    for start_time, end_time in not_live_ranges:
        if end_time - start_time > 1.0:  # Only add ranges longer than 1 second
            try:
                range_id = db_manager.add_match_range(match_id, start_time, end_time, "NOT LIVE", labeler_name, "Auto-generated")
                print(f"✅ Added NOT LIVE range: {format_time(start_time)} → {format_time(end_time)} (ID: {range_id})")
                added_count += 1
            except Exception as e:
                print(f"❌ Error adding NOT LIVE range: {str(e)}")
    
    print(f"🎯 Generated {added_count} NOT LIVE ranges")

def main():
    """Main terminal interface"""
    print("🏆 WrestleAI Terminal Range Labeler")
    print("=" * 50)
    
    # Get labeler name
    labeler_name = input("Enter your name: ").strip()
    if not labeler_name:
        print("❌ Name is required")
        return
    
    while True:
        print("\n" + "=" * 50)
        
        # Show available matches
        matches = get_matches_list()
        if not matches:
            print("❌ No matches found in database")
            return
        
        print("Available matches:")
        for i, match in enumerate(matches, 1):
            duration_str = format_time(match['duration'])
            print(f"{i:2d}. {match['filename']} ({duration_str})")
        
        print("\nCommands:")
        print("  <number> - Select match to label")
        print("  'done'   - Generate NOT LIVE ranges for a match")
        print("  'quit'   - Exit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == 'quit':
            print("👋 Goodbye!")
            break
        
        elif choice == 'done':
            # Generate NOT LIVE ranges
            print("\nGenerate NOT LIVE ranges for which match?")
            for i, match in enumerate(matches, 1):
                print(f"{i:2d}. {match['filename']}")
            
            try:
                match_num = int(input("Enter match number: ")) - 1
                if 0 <= match_num < len(matches):
                    match_id = matches[match_num]['id']
                    generate_not_live_ranges(match_id, labeler_name)
                else:
                    print("❌ Invalid match number")
            except ValueError:
                print("❌ Please enter a valid number")
        
        else:
            # Select match to label
            try:
                match_num = int(choice) - 1
                if 0 <= match_num < len(matches):
                    match = matches[match_num]
                    label_match(match, labeler_name)
                else:
                    print("❌ Invalid match number")
            except ValueError:
                print("❌ Please enter a valid number")

def label_match(match, labeler_name):
    """Label live ranges for a specific match"""
    print(f"\n🎬 Labeling: {match['filename']}")
    print(f"📊 Duration: {format_time(match['duration'])}")
    
    # Show existing ranges
    ranges = get_match_ranges(match['id'])
    if ranges:
        print("\nExisting ranges:")
        for r in ranges:
            start_str = format_time(r['start_time'])
            end_str = format_time(r['end_time'])
            print(f"  {start_str} → {end_str} ({r['label']})")
    
    print("\nEnter LIVE ranges (format: MM:SS MM:SS)")
    print("Examples:")
    print("  1:30 2:45    - Live from 1:30 to 2:45")
    print("  5:00 8:30    - Live from 5:00 to 8:30")
    print("  done         - Finish labeling this match")
    
    while True:
        user_input = input("\nEnter range (or 'done'): ").strip().lower()
        
        if user_input == 'done':
            print(f"✅ Finished labeling {match['filename']}")
            break
        
        # Parse range input
        parts = user_input.split()
        if len(parts) != 2:
            print("❌ Please enter two times separated by space (e.g., '1:30 2:45')")
            continue
        
        start_str, end_str = parts
        start_time = parse_time(start_str)
        end_time = parse_time(end_str)
        
        if start_time == 0 or end_time == 0:
            print("❌ Invalid time format. Use MM:SS (e.g., '1:30')")
            continue
        
        if start_time >= end_time:
            print("❌ Start time must be before end time")
            continue
        
        if end_time > match['duration']:
            print(f"❌ End time exceeds video duration ({format_time(match['duration'])})")
            continue
        
        # Add the range
        notes = input("Notes (optional): ").strip()
        if add_live_range(match['id'], start_time, end_time, labeler_name, notes):
            print(f"✅ Added LIVE range: {start_str} → {end_str}")

if __name__ == "__main__":
    main() 