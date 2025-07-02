import sqlite3
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class DatabaseManager:
    def __init__(self, db_path: str = "wrestling_pipeline.db"):
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Ensure the database and tables exist"""
        if not os.path.exists(self.db_path):
            from database_setup import create_database
            create_database()
    
    def get_connection(self):
        """Get a database connection"""
        return sqlite3.connect(self.db_path)
    
    def calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA-256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_video_info(self, filepath: str) -> Dict:
        """Get video metadata using OpenCV"""
        import cv2
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'duration': duration,
            'resolution': f"{width}x{height}",
            'frame_count': frame_count
        }
    
    def register_video(self, filepath: str) -> int:
        """Register a new video in the database"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Video file not found: {filepath}")
        
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)
        file_hash = self.calculate_file_hash(filepath)
        video_info = self.get_video_info(filepath)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if video already exists
        cursor.execute("SELECT id FROM videos WHERE filename = ?", (filename,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing video if hash changed
            cursor.execute("SELECT hash FROM videos WHERE id = ?", (existing[0],))
            old_hash = cursor.fetchone()[0]
            
            if old_hash != file_hash:
                # File changed, update metadata
                cursor.execute('''
                    UPDATE videos SET 
                        file_size = ?, hash = ?, duration = ?, fps = ?, resolution = ?, 
                        processed_date = NULL, status = 'pending'
                    WHERE id = ?
                ''', (file_size, file_hash, video_info.get('duration'), 
                     video_info.get('fps'), video_info.get('resolution'), existing[0]))
                conn.commit()
                print(f"🔄 Updated video: {filename} (hash changed)")
            else:
                print(f"⏭️  Skipped video: {filename} (no changes)")
            
            conn.close()
            return existing[0]
        else:
            # Insert new video
            cursor.execute('''
                INSERT INTO videos (filename, filepath, file_size, duration, fps, resolution, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (filename, filepath, file_size, video_info.get('duration'),
                 video_info.get('fps'), video_info.get('resolution'), file_hash))
            
            video_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"📹 Registered new video: {filename}")
            return video_id
    
    def get_unprocessed_videos(self) -> List[Dict]:
        """Get all videos that haven't been processed yet"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, filepath, status FROM videos 
            WHERE status = 'pending' OR status IS NULL
        ''')
        
        videos = []
        for row in cursor.fetchall():
            videos.append({
                'id': row[0],
                'filename': row[1],
                'filepath': row[2],
                'status': row[3]
            })
        
        conn.close()
        return videos
    
    def mark_video_processed(self, video_id: int):
        """Mark a video as processed"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE videos SET status = 'processed', processed_date = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (video_id,))
        
        conn.commit()
        conn.close()
    
    def add_clip(self, video_id: int, filename: str, filepath: str, 
                 start_time: float, end_time: float, motion_score: float = None) -> int:
        """Add a generated clip to the database"""
        duration = end_time - start_time
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO clips (video_id, filename, filepath, start_time, end_time, duration, motion_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (video_id, filename, filepath, start_time, end_time, duration, motion_score))
        
        clip_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return clip_id
    
    def get_unlabeled_clips(self) -> List[Dict]:
        """Get all clips that haven't been labeled with final decisions yet"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.id, c.filename, c.filepath, c.start_time, c.end_time, c.duration,
                   v.filename as source_video
            FROM clips c
            JOIN videos v ON c.video_id = v.id
            WHERE c.id NOT IN (
                SELECT DISTINCT clip_id FROM labels WHERE label IN ('Highlight', 'Not Highlight')
            )
        ''')
        
        clips = []
        for row in cursor.fetchall():
            clips.append({
                'id': row[0],
                'filename': row[1],
                'filepath': row[2],
                'start_time': row[3],
                'end_time': row[4],
                'duration': row[5],
                'source_video': row[6]
            })
        
        conn.close()
        return clips
    
    def add_label(self, clip_id: int, labeler_name: str, label: str, 
                  confidence: float = None, notes: str = None):
        """Add a label for a clip"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO labels (clip_id, labeler_name, label, confidence, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (clip_id, labeler_name, label, confidence, notes))
        
        conn.commit()
        conn.close()
    
    def get_labeling_progress(self) -> Dict:
        """Get labeling progress statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total clips
        cursor.execute("SELECT COUNT(*) FROM clips")
        total_clips = cursor.fetchone()[0]
        
        # Labeled clips (only final decisions)
        cursor.execute("SELECT COUNT(DISTINCT clip_id) FROM labels WHERE label IN ('Highlight', 'Not Highlight')")
        labeled_clips = cursor.fetchone()[0]
        
        # Highlight counts
        cursor.execute("SELECT COUNT(*) FROM labels WHERE label = 'Highlight'")
        highlight_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM labels WHERE label = 'Not Highlight'")
        not_highlight_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_clips': total_clips,
            'labeled_clips': labeled_clips,
            'highlight_count': highlight_count,
            'not_highlight_count': not_highlight_count,
            'progress_percent': (labeled_clips / total_clips * 100) if total_clips > 0 else 0
        }
    
    def log_processing(self, video_id: int, stage: str, status: str, error_message: str = None):
        """Log a processing step"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO processing_log (video_id, stage, status, error_message)
            VALUES (?, ?, ?, ?)
        ''', (video_id, stage, status, error_message))
        
        conn.commit()
        conn.close()
    
    def get_processing_history(self, video_id: int = None) -> List[Dict]:
        """Get processing history for a video or all videos"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if video_id:
            cursor.execute('''
                SELECT pl.stage, pl.status, pl.start_time, pl.end_time, pl.error_message,
                       v.filename
                FROM processing_log pl
                JOIN videos v ON pl.video_id = v.id
                WHERE pl.video_id = ?
                ORDER BY pl.start_time DESC
            ''', (video_id,))
        else:
            cursor.execute('''
                SELECT pl.stage, pl.status, pl.start_time, pl.end_time, pl.error_message,
                       v.filename
                FROM processing_log pl
                JOIN videos v ON pl.video_id = v.id
                ORDER BY pl.start_time DESC
                LIMIT 50
            ''')
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'stage': row[0],
                'status': row[1],
                'start_time': row[2],
                'end_time': row[3],
                'error_message': row[4],
                'filename': row[5]
            })
        
        conn.close()
        return history 