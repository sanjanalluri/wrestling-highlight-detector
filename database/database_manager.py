import sqlite3
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class DatabaseManager:
    def __init__(self, db_path: str = "database/wrestling_pipeline.db"):
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Ensure the database and tables exist"""
        if not os.path.exists(self.db_path):
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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

    # Live labeling methods for inclusive_match_clips
    def get_unlabeled_live_clips(self) -> List[Dict]:
        """Get all inclusive match clips that haven't been labeled yet"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.id, c.filename, c.filepath, c.clip_number, c.start_time, c.end_time, c.duration,
                   m.filename as source_match
            FROM inclusive_match_clips c
            JOIN original_matches m ON c.match_id = m.id
            WHERE c.live_label IS NULL OR c.live_label = ''
        ''')
        
        clips = []
        for row in cursor.fetchall():
            clips.append({
                'id': row[0],
                'filename': row[1],
                'filepath': row[2],
                'clip_number': row[3],
                'start_time': row[4],
                'end_time': row[5],
                'duration': row[6],
                'source_match': row[7]
            })
        
        conn.close()
        return clips

    def add_live_label(self, clip_id: int, labeler_name: str, label: str, notes: str = None):
        """Add a live label for an inclusive match clip"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Update the live_label field in inclusive_match_clips table
        cursor.execute('''
            UPDATE inclusive_match_clips 
            SET live_label = ?
            WHERE id = ?
        ''', (label, clip_id))
        
        # Also add to labels table for consistency
        cursor.execute('''
            INSERT INTO labels (clip_id, labeler_name, label, notes)
            VALUES (?, ?, ?, ?)
        ''', (clip_id, labeler_name, label, notes))
        
        conn.commit()
        conn.close()

    def get_live_labeling_progress(self) -> Dict:
        """Get live labeling progress statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total inclusive clips
        cursor.execute("SELECT COUNT(*) FROM inclusive_match_clips")
        total_clips = cursor.fetchone()[0]
        
        # Labeled clips
        cursor.execute("SELECT COUNT(*) FROM inclusive_match_clips WHERE live_label IS NOT NULL AND live_label != ''")
        labeled_clips = cursor.fetchone()[0]
        
        # Label counts
        cursor.execute("SELECT COUNT(*) FROM inclusive_match_clips WHERE live_label = 'LIVE'")
        live_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM inclusive_match_clips WHERE live_label = 'NOT LIVE'")
        not_live_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM inclusive_match_clips WHERE live_label = 'UNSURE'")
        unsure_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_clips': total_clips,
            'labeled_clips': labeled_clips,
            'live_count': live_count,
            'not_live_count': not_live_count,
            'unsure_count': unsure_count,
            'progress_percent': (labeled_clips / total_clips * 100) if total_clips > 0 else 0
        }

    def get_live_clip_by_filename(self, filename: str) -> Optional[Dict]:
        """Get a live clip by filename"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.id, c.filename, c.filepath, c.clip_number, c.start_time, c.end_time, c.duration,
                   m.filename as source_match
            FROM inclusive_match_clips c
            JOIN original_matches m ON c.match_id = m.id
            WHERE c.filename = ?
        ''', (filename,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'filename': row[1],
                'filepath': row[2],
                'clip_number': row[3],
                'start_time': row[4],
                'end_time': row[5],
                'duration': row[6],
                'source_match': row[7]
            }
        return None

    # Match range methods for manual selection and labeling
    def get_all_matches(self) -> List[Dict]:
        """Get all original matches"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, filepath, preview_filepath, duration, file_size
            FROM original_matches
            ORDER BY filename
        ''')
        
        matches = []
        for row in cursor.fetchall():
            matches.append({
                'id': row[0],
                'filename': row[1],
                'filepath': row[2],
                'preview_filepath': row[3],
                'duration': row[4],
                'file_size': row[5]
            })
        
        conn.close()
        return matches

    def get_match_by_id(self, match_id: int) -> Optional[Dict]:
        """Get a specific match by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, filepath, preview_filepath, duration, file_size
            FROM original_matches
            WHERE id = ?
        ''', (match_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'filename': row[1],
                'filepath': row[2],
                'preview_filepath': row[3],
                'duration': row[4],
                'file_size': row[5]
            }
        return None

    def get_ranges_for_match(self, match_id: int) -> List[Dict]:
        """Get all ranges for a specific match"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, range_number, start_time, end_time, duration, label, labeler_name, filename, notes
            FROM match_ranges
            WHERE match_id = ?
            ORDER BY start_time
        ''', (match_id,))
        
        ranges = []
        for row in cursor.fetchall():
            ranges.append({
                'id': row[0],
                'range_number': row[1],
                'start_time': row[2],
                'end_time': row[3],
                'duration': row[4],
                'label': row[5],
                'labeler_name': row[6],
                'filename': row[7],
                'notes': row[8]
            })
        
        conn.close()
        return ranges

    def add_match_range(self, match_id: int, start_time: float, end_time: float, 
                       label: str, labeler_name: str, notes: str = None) -> int:
        """Add a new match range"""
        duration = end_time - start_time
        
        # Get the next range number for this match
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COALESCE(MAX(range_number), 0) + 1
            FROM match_ranges
            WHERE match_id = ?
        ''', (match_id,))
        
        range_number = cursor.fetchone()[0]
        
        # Generate filename
        match_info = self.get_match_by_id(match_id)
        if match_info:
            base_name = os.path.splitext(match_info['filename'])[0]
            filename = f"{base_name}_range_{range_number:03d}.mp4"
        else:
            filename = f"match_{match_id}_range_{range_number:03d}.mp4"
        
        # Insert the range
        cursor.execute('''
            INSERT INTO match_ranges 
            (match_id, range_number, start_time, end_time, duration, label, labeler_name, filename, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (match_id, range_number, start_time, end_time, duration, label, labeler_name, filename, notes))
        
        range_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return range_id

    def get_range_labeling_progress(self) -> Dict:
        """Get progress statistics for range labeling"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total matches
        cursor.execute("SELECT COUNT(*) FROM original_matches")
        total_matches = cursor.fetchone()[0]
        
        # Matches with ranges
        cursor.execute("SELECT COUNT(DISTINCT match_id) FROM match_ranges")
        matches_with_ranges = cursor.fetchone()[0]
        
        # Total ranges
        cursor.execute("SELECT COUNT(*) FROM match_ranges")
        total_ranges = cursor.fetchone()[0]
        
        # Label counts
        cursor.execute("SELECT COUNT(*) FROM match_ranges WHERE label = 'LIVE'")
        live_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM match_ranges WHERE label = 'NOT LIVE'")
        not_live_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_matches': total_matches,
            'matches_with_ranges': matches_with_ranges,
            'total_ranges': total_ranges,
            'live_count': live_count,
            'not_live_count': not_live_count,
            'progress_percent': (matches_with_ranges / total_matches * 100) if total_matches > 0 else 0
        }

    def update_preview_filepath(self, match_id: int, preview_filepath: str):
        """Update the preview filepath for a match"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE original_matches 
            SET preview_filepath = ?
            WHERE id = ?
        ''', (preview_filepath, match_id))
        
        conn.commit()
        conn.close() 