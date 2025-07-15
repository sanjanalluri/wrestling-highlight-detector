#!/usr/bin/env python3
"""
Live Filter Utilities
Combined functionality for the live filter phase
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database_manager import DatabaseManager

def show_live_filter_stats():
    """Show statistics for the live filter phase"""
    db_manager = DatabaseManager()
    
    # Get progress statistics
    progress = db_manager.get_range_labeling_progress()
    
    print("🏆 Live Filter Phase Statistics")
    print("=" * 40)
    print(f"Total matches: {progress['total_matches']}")
    print(f"Matches with ranges: {progress['matches_with_ranges']}")
    print(f"Progress: {progress['progress_percent']:.1f}%")
    print(f"Total ranges: {progress['total_ranges']}")
    print(f"  - LIVE: {progress['live_count']}")
    print(f"  - NOT LIVE: {progress['not_live_count']}")
    
    # Show matches and their status
    matches = db_manager.get_all_matches()
    print(f"\nMatch Status:")
    for match in matches:
        ranges = db_manager.get_ranges_for_match(match['id'])
        live_ranges = [r for r in ranges if r['label'] == 'LIVE']
        not_live_ranges = [r for r in ranges if r['label'] == 'NOT LIVE']
        
        status = "✅ Complete" if len(ranges) > 0 else "⏳ Not started"
        print(f"  {match['filename']}: {status} ({len(live_ranges)} LIVE, {len(not_live_ranges)} NOT LIVE)")

def export_live_ranges_to_csv():
    """Export all LIVE ranges to a CSV file for analysis"""
    import csv
    from datetime import datetime
    
    db_manager = DatabaseManager()
    matches = db_manager.get_all_matches()
    
    # Get all LIVE ranges
    all_live_ranges = []
    for match in matches:
        ranges = db_manager.get_ranges_for_match(match['id'])
        live_ranges = [r for r in ranges if r['label'] == 'LIVE']
        
        for r in live_ranges:
            all_live_ranges.append({
                'match_filename': match['filename'],
                'match_duration': match['duration'],
                'range_start': r['start_time'],
                'range_end': r['end_time'],
                'range_duration': r['duration'],
                'labeler': r['labeler_name'],
                'notes': r['notes'] or '',
                'created_date': r.get('created_date', '')
            })
    
    # Export to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"live_filter_exports/live_ranges_{timestamp}.csv"
    
    os.makedirs("live_filter_exports", exist_ok=True)
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['match_filename', 'match_duration', 'range_start', 'range_end', 
                     'range_duration', 'labeler', 'notes', 'created_date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in all_live_ranges:
            writer.writerow(row)
    
    print(f"✅ Exported {len(all_live_ranges)} LIVE ranges to {filename}")

def validate_database_integrity():
    """Check database integrity and report any issues"""
    db_manager = DatabaseManager()
    
    print("🔍 Database Integrity Check")
    print("=" * 30)
    
    # Check matches
    matches = db_manager.get_all_matches()
    print(f"✅ Found {len(matches)} matches in database")
    
    # Check file existence
    missing_files = []
    for match in matches:
        if not os.path.exists(match['filepath']):
            missing_files.append(match['filename'])
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
    else:
        print("✅ All match files exist")
    
    # Check ranges
    total_ranges = 0
    for match in matches:
        ranges = db_manager.get_ranges_for_match(match['id'])
        total_ranges += len(ranges)
        
        # Check for overlapping ranges
        ranges.sort(key=lambda x: x['start_time'])
        for i in range(len(ranges) - 1):
            if ranges[i]['end_time'] > ranges[i+1]['start_time']:
                print(f"⚠️  Overlapping ranges in {match['filename']}: {ranges[i]['start_time']}-{ranges[i]['end_time']} and {ranges[i+1]['start_time']}-{ranges[i+1]['end_time']}")
    
    print(f"✅ Found {total_ranges} total ranges")

def main():
    """Main utility interface"""
    print("🔧 Live Filter Utilities")
    print("=" * 30)
    print("1. Show statistics")
    print("2. Export LIVE ranges to CSV")
    print("3. Validate database integrity")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        show_live_filter_stats()
    elif choice == "2":
        export_live_ranges_to_csv()
    elif choice == "3":
        validate_database_integrity()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 