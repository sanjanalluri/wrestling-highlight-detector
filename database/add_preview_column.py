#!/usr/bin/env python3
"""
Add preview_filepath column to existing original_matches table
"""

import sqlite3
import os

def add_preview_column():
    """Add preview_filepath column to original_matches table"""
    db_path = "database/wrestling_pipeline.db"
    
    if not os.path.exists(db_path):
        print("❌ Database not found!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(original_matches)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'preview_filepath' in columns:
            print("✅ preview_filepath column already exists")
            return
        
        # Add the column
        cursor.execute('''
            ALTER TABLE original_matches 
            ADD COLUMN preview_filepath TEXT
        ''')
        
        conn.commit()
        print("✅ Added preview_filepath column to original_matches table")
        
    except Exception as e:
        print(f"❌ Error adding column: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    add_preview_column() 