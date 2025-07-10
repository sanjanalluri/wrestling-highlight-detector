"""
Add gender/category field to existing database
This script safely adds a new column to track men's vs women's matches
"""

import sqlite3
import os
from database.database_manager import DatabaseManager

def add_gender_field():
    """Add gender field to videos table"""
    print("🔧 Adding gender field to database...")
    
    # Connect to database
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    try:
        # Check if gender column already exists
        cursor.execute("PRAGMA table_info(videos)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'gender' not in columns:
            # Add gender column
            cursor.execute("ALTER TABLE videos ADD COLUMN gender TEXT DEFAULT 'unknown'")
            print("✅ Added 'gender' column to videos table")
        else:
            print("✅ Gender column already exists")
        
        # Commit changes
        conn.commit()
        
        # Show current videos for manual categorization
        cursor.execute("SELECT id, filename, gender FROM videos ORDER BY created_at DESC")
        videos = cursor.fetchall()
        
        print(f"\n📊 Current videos in database:")
        print("-" * 60)
        for video_id, filename, gender in videos:
            print(f"ID: {video_id:2d} | Gender: {gender:8s} | File: {filename}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error adding gender field: {e}")
        conn.close()

def update_video_gender(video_id, gender):
    """Update gender for a specific video"""
    if gender.lower() not in ['men', 'women', 'mens', 'womens', 'male', 'female']:
        print(f"❌ Invalid gender: {gender}. Use 'men', 'women', 'mens', 'womens', 'male', or 'female'")
        return
    
    # Normalize gender values
    gender_normalized = 'mens' if gender.lower() in ['men', 'mens', 'male'] else 'womens'
    
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("UPDATE videos SET gender = ? WHERE id = ?", (gender_normalized, video_id))
        
        if cursor.rowcount > 0:
            print(f"✅ Updated video ID {video_id} to '{gender_normalized}'")
        else:
            print(f"❌ No video found with ID {video_id}")
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error updating video gender: {e}")
        conn.close()

def bulk_update_by_filename():
    """Interactive tool to update multiple videos by filename pattern"""
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    # Check if gender column exists, add it if not
    cursor.execute("PRAGMA table_info(videos)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'gender' not in columns:
        print("⚠️  Gender column doesn't exist. Adding it now...")
        cursor.execute("ALTER TABLE videos ADD COLUMN gender TEXT DEFAULT 'unknown'")
        conn.commit()
        print("✅ Gender column added!")
    
    cursor.execute("SELECT id, filename, gender FROM videos ORDER BY filename")
    videos = cursor.fetchall()
    
    print("\n🔄 Bulk Gender Update Tool")
    print("=" * 50)
    
    for video_id, filename, current_gender in videos:
        print(f"\nVideo: {filename}")
        print(f"Current gender: {current_gender}")
        
        new_gender = input("Enter gender (mens/womens/skip): ").strip().lower()
        
        if new_gender in ['mens', 'womens']:
            cursor.execute("UPDATE videos SET gender = ? WHERE id = ?", (new_gender, video_id))
            print(f"✅ Updated to '{new_gender}'")
        elif new_gender == 'skip':
            print("⏭️  Skipped")
        else:
            print("❌ Invalid input, skipped")
    
    conn.commit()
    conn.close()
    print("\n✅ Bulk update complete!")

def show_gender_statistics():
    """Show statistics by gender"""
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    # Video statistics by gender
    cursor.execute("""
        SELECT gender, COUNT(*) as video_count, 
               SUM(CASE WHEN status = 'processed' THEN 1 ELSE 0 END) as processed_count
        FROM videos 
        GROUP BY gender
    """)
    
    video_stats = cursor.fetchall()
    
    # Clip statistics by gender
    cursor.execute("""
        SELECT v.gender, COUNT(c.id) as clip_count,
               COUNT(l.id) as labeled_count,
               SUM(CASE WHEN l.label = 'Highlight' THEN 1 ELSE 0 END) as highlight_count
        FROM videos v
        LEFT JOIN clips c ON v.id = c.video_id
        LEFT JOIN labels l ON c.id = l.clip_id AND l.label IN ('Highlight', 'Not Highlight')
        GROUP BY v.gender
    """)
    
    clip_stats = cursor.fetchall()
    
    print("\n📊 Gender Statistics")
    print("=" * 50)
    
    print("Videos by Gender:")
    for gender, video_count, processed_count in video_stats:
        print(f"  {gender:8s}: {video_count:2d} videos ({processed_count:2d} processed)")
    
    print("\nClips by Gender:")
    for gender, clip_count, labeled_count, highlight_count in clip_stats:
        if clip_count > 0:
            highlight_rate = (highlight_count / labeled_count * 100) if labeled_count > 0 else 0
            print(f"  {gender:8s}: {clip_count:3d} clips, {labeled_count:3d} labeled, {highlight_count:2d} highlights ({highlight_rate:.1f}%)")
    
    conn.close()

if __name__ == "__main__":
    print("🏆 Wrestling Database Gender Field Manager")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Add gender field to database")
        print("2. Update single video gender")
        print("3. Bulk update by filename")
        print("4. Show gender statistics")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            add_gender_field()
        elif choice == '2':
            try:
                video_id = int(input("Enter video ID: "))
                gender = input("Enter gender (mens/womens): ").strip()
                update_video_gender(video_id, gender)
            except ValueError:
                print("❌ Invalid video ID")
        elif choice == '3':
            bulk_update_by_filename()
        elif choice == '4':
            show_gender_statistics()
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice") 