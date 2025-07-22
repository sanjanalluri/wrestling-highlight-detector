#!/usr/bin/env python3
"""
Clean up partial chunk data from failed processing run
"""

import os
import sys
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database_manager import DatabaseManager

def cleanup_partial_data():
    """Remove all partial chunk data from database and file system"""
    print("🧹 Cleaning up partial chunk data...")
    
    # Clean up database
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    # Count chunks before deletion
    cursor.execute("SELECT COUNT(*) FROM training_chunks")
    total_chunks = cursor.fetchone()[0]
    print(f"  Found {total_chunks} chunks in database")
    
    # Delete all chunks from database
    cursor.execute("DELETE FROM training_chunks")
    deleted_count = cursor.rowcount
    conn.commit()
    print(f"  Deleted {deleted_count} chunks from database")
    
    conn.close()
    
    # Clean up file system
    features_dir = "live_filter/features"
    if os.path.exists(features_dir):
        # Count files before deletion
        feature_files = [f for f in os.listdir(features_dir) if f.endswith('.npz')]
        print(f"  Found {len(feature_files)} feature files in file system")
        
        # Remove all feature files
        for file in feature_files:
            file_path = os.path.join(features_dir, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"  Warning: Could not remove {file}: {e}")
        
        print(f"  Removed all feature files from: {features_dir}")
    else:
        print("  No features directory found")
    
    # Clean up temp files
    temp_dir = "live_filter/temp_chunks"
    if os.path.exists(temp_dir):
        temp_files = [f for f in os.listdir(temp_dir) if f.endswith('.mp4')]
        print(f"  Found {len(temp_files)} temp files")
        
        for temp_file in temp_files:
            file_path = os.path.join(temp_dir, temp_file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"  Warning: Could not remove {temp_file}: {e}")
        
        print(f"  Cleaned up temp files")
    
    print("✅ Partial data cleanup complete!")

if __name__ == "__main__":
    cleanup_partial_data() 