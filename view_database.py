#!/usr/bin/env python3
"""
Simple database viewer for the wrestling highlight detector database.
Shows all tables and their contents in a readable format.
"""

from database_manager import DatabaseManager
import os

def view_database():
    """Display all database contents in a readable format"""
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    print("🗄️  Wrestling Highlight Detector Database")
    print("=" * 50)
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    for table in tables:
        print(f"\n📋 Table: {table}")
        print("-" * 30)
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        # Get all data
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        
        if not rows:
            print("   (empty)")
            continue
        
        # Print column headers
        header = " | ".join(f"{name:15}" for name in column_names)
        print(f"   {header}")
        print("   " + "-" * len(header))
        
        # Print data rows
        for row in rows:
            formatted_row = []
            for i, value in enumerate(row):
                if value is None:
                    formatted_row.append("NULL".ljust(15))
                elif isinstance(value, str) and len(value) > 12:
                    formatted_row.append(f"{value[:10]}...".ljust(15))
                else:
                    formatted_row.append(str(value).ljust(15))
            print(f"   {' | '.join(formatted_row)}")
    
    conn.close()

def show_summary():
    """Show a summary of the database contents"""
    db_manager = DatabaseManager()
    progress = db_manager.get_labeling_progress()
    
    print("\n📊 Database Summary")
    print("=" * 30)
    print(f"   Total videos: {progress.get('total_videos', 'N/A')}")
    print(f"   Total clips: {progress['total_clips']}")
    print(f"   Labeled clips: {progress['labeled_clips']}")
    print(f"   Progress: {progress['progress_percent']:.1f}%")
    print(f"   Highlights: {progress['highlight_count']}")
    print(f"   Not highlights: {progress['not_highlight_count']}")

if __name__ == "__main__":
    view_database()
    show_summary()
    
    print(f"\n💡 To open in DB Browser for SQLite:")
    print(f"   open -a 'DB Browser for SQLite' wrestling_pipeline.db") 