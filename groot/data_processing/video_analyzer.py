"""
Video analyzer for Groot data processing pipeline.
Analyzes video metadata including frame rates, dimensions, and duration.
"""

import cv2
import os
from pathlib import Path
import sqlite3
from tqdm import tqdm
import sys

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DATABASE_PATH, TABLE_NAMES, SUPPORTED_VIDEO_FORMATS, MAX_VIDEO_SIZE_GB

class VideoAnalyzer:
    """Analyzes video files and stores metadata in the database"""
    
    def __init__(self, match_directory="../match_directory"):
        self.match_directory = Path(match_directory)
        self.db_path = DATABASE_PATH
        
    def get_video_files(self):
        """Get all video files from the match directory"""
        video_files = []
        
        if not self.match_directory.exists():
            print(f"❌ Match directory not found: {self.match_directory}")
            return video_files
            
        for video_format in SUPPORTED_VIDEO_FORMATS:
            video_files.extend(self.match_directory.rglob(f"*{video_format}"))
            video_files.extend(self.match_directory.rglob(f"*{video_format.upper()}"))
        
        return sorted(video_files)
    
    def analyze_video(self, video_path):
        """Analyze a single video file and return metadata"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f"❌ Could not open video: {video_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_seconds = frame_count / fps if fps > 0 else 0
            
            # Get file size
            file_size_bytes = video_path.stat().st_size
            file_size_gb = file_size_bytes / (1024**3)
            
            cap.release()
            
            # Check if file is too large
            if file_size_gb > MAX_VIDEO_SIZE_GB:
                print(f"⚠️  Video too large ({file_size_gb:.2f}GB): {video_path.name}")
                return None
            
            return {
                'video_path': str(video_path),
                'video_name': video_path.name,
                'duration_seconds': duration_seconds,
                'original_fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'file_size_bytes': file_size_bytes,
                'file_size_gb': file_size_gb
            }
            
        except Exception as e:
            print(f"❌ Error analyzing video {video_path}: {e}")
            return None
    
    def store_video_metadata(self, metadata):
        """Store video metadata in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO groot_videos 
                (video_path, video_name, duration_seconds, original_fps, frame_count, 
                 width, height, file_size_bytes, processing_status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                metadata['video_path'],
                metadata['video_name'],
                metadata['duration_seconds'],
                metadata['original_fps'],
                metadata['frame_count'],
                metadata['width'],
                metadata['height'],
                metadata['file_size_bytes'],
                'analyzed'
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error storing metadata: {e}")
            return False
        finally:
            conn.close()
    
    def analyze_all_videos(self):
        """Analyze all videos in the match directory"""
        print("🔍 Analyzing all videos in match directory...")
        
        video_files = self.get_video_files()
        if not video_files:
            print("❌ No video files found!")
            return
        
        print(f"📹 Found {len(video_files)} video files")
        
        # Analyze each video
        successful_analyses = 0
        for video_path in tqdm(video_files, desc="Analyzing videos"):
            metadata = self.analyze_video(video_path)
            if metadata:
                if self.store_video_metadata(metadata):
                    successful_analyses += 1
                    print(f"✅ Analyzed: {metadata['video_name']} ({metadata['original_fps']:.1f} fps, {metadata['duration_seconds']:.1f}s)")
        
        print(f"\n📊 Analysis complete: {successful_analyses}/{len(video_files)} videos analyzed")
    
    def get_frame_rate_summary(self):
        """Get a summary of frame rates across all videos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT original_fps, COUNT(*) as count
            FROM groot_videos 
            WHERE processing_status = 'analyzed'
            GROUP BY original_fps
            ORDER BY original_fps
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print("❌ No analyzed videos found in database")
            return
        
        print("\n📊 Frame Rate Summary:")
        print("FPS\t| Count\t| Videos")
        print("-" * 30)
        
        for fps, count in results:
            print(f"{fps:.1f}\t| {count}\t| ", end="")
            
            # Get sample video names for this FPS
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT video_name FROM groot_videos 
                WHERE original_fps = ? AND processing_status = 'analyzed'
                LIMIT 3
            ''', (fps,))
            
            sample_names = [row[0] for row in cursor.fetchall()]
            print(", ".join(sample_names))
            conn.close()
    
    def check_target_fps_compatibility(self, target_fps=12):
        """Check if videos are compatible with target FPS"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT video_name, original_fps, duration_seconds
            FROM groot_videos 
            WHERE processing_status = 'analyzed'
            ORDER BY original_fps
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print("❌ No analyzed videos found")
            return
        
        print(f"\n🎯 Target FPS Compatibility Check ({target_fps} fps):")
        print("Video\t\t| Original FPS\t| Compatible\t| Notes")
        print("-" * 70)
        
        for video_name, original_fps, duration in results:
            # Check if target FPS is a factor of original FPS
            if original_fps % target_fps == 0:
                compatible = "✅ Yes"
                notes = f"Can sample every {int(original_fps/target_fps)} frames"
            elif target_fps % original_fps == 0:
                compatible = "⚠️  Upsample"
                notes = f"Need to interpolate {int(target_fps/original_fps)}x"
            else:
                compatible = "❌ No"
                notes = "Will need frame interpolation"
            
            print(f"{video_name[:15]:<15}\t| {original_fps:<11.1f}\t| {compatible:<11}\t| {notes}")

def main():
    """Main function to run video analysis"""
    analyzer = VideoAnalyzer()
    
    # Analyze all videos
    analyzer.analyze_all_videos()
    
    # Show frame rate summary
    analyzer.get_frame_rate_summary()
    
    # Check target FPS compatibility
    analyzer.check_target_fps_compatibility(target_fps=12)

if __name__ == "__main__":
    main()
