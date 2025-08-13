"""
Frame rate converter for Groot data processing pipeline.
Converts all videos to target frame rate (12 fps) for consistent processing.
"""

import cv2
import os
from pathlib import Path
import sqlite3
from tqdm import tqdm
import sys
import shutil

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DATABASE_PATH, TABLE_NAMES, TARGET_FPS

# Override database path for this script
DATABASE_PATH = "../database/wrestling_pipeline.db"

class FrameRateConverter:
    """Converts videos to target frame rate"""
    
    def __init__(self, target_fps=TARGET_FPS, output_dir="processed_videos"):
        self.target_fps = target_fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.db_path = DATABASE_PATH
        
    def convert_video_fps(self, input_path, output_path, target_fps):
        """Convert a single video to target frame rate"""
        try:
            # Open input video
            cap = cv2.VideoCapture(str(input_path))
            
            if not cap.isOpened():
                print(f"❌ Could not open video: {input_path}")
                return False
            
            # Get original video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer with more compatible codec
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec, more compatible
                out = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))
            except:
                # Fallback to mp4v if H.264 not available
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))
            
            if not out.isOpened():
                print(f"❌ Could not create output video: {output_path}")
                cap.release()
                return False
            
            # Calculate frame interval for target FPS
            frame_interval = original_fps / target_fps
            
            print(f"   Converting {original_fps:.1f} fps → {target_fps} fps")
            print(f"   Frame interval: {frame_interval:.2f}")
            
            # Process frames
            frame_idx = 0
            written_frames = 0
            
            with tqdm(total=frame_count, desc="Converting frames", leave=False) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Write frame if it matches the target interval
                    if frame_idx % frame_interval < 1:
                        out.write(frame)
                        written_frames += 1
                    
                    frame_idx += 1
                    pbar.update(1)
            
            # Clean up
            cap.release()
            out.release()
            
            # Verify output video
            verify_cap = cv2.VideoCapture(str(output_path))
            if verify_cap.isOpened():
                actual_fps = verify_cap.get(cv2.CAP_PROP_FPS)
                actual_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                verify_cap.release()
                
                print(f"   ✅ Output: {actual_fps:.1f} fps, {actual_frames} frames")
                return True
            else:
                print(f"   ❌ Failed to verify output video")
                return False
                
        except Exception as e:
            print(f"❌ Error converting video: {e}")
            return False
    
    def convert_all_videos(self, force_reconvert=False):
        """Convert all videos in the database to target frame rate"""
        print(f"🔄 Converting all videos to {self.target_fps} fps...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all analyzed videos (including original videos that need conversion)
        cursor.execute('''
            SELECT id, video_path, video_name, original_fps, processing_status
            FROM groot_videos 
            WHERE processing_status = 'analyzed' OR (processing_status = 'converted' AND original_fps != 12.0)
            ORDER BY video_name
        ''')
        
        videos = cursor.fetchall()
        conn.close()
        
        if not videos:
            print("❌ No videos found in database")
            return
        
        print(f"📹 Found {len(videos)} videos to process")
        
        successful_conversions = 0
        skipped_conversions = 0
        
        for video_id, video_path, video_name, original_fps, status in videos:
            print(f"\n🎬 Processing: {video_name}")
            
            # Check if conversion is needed
            if abs(original_fps - self.target_fps) < 0.1:
                print(f"   ⏭️  Already at target FPS ({original_fps:.1f} fps)")
                skipped_conversions += 1
                continue
            
            # Create output path
            output_path = self.output_dir / f"{Path(video_name).stem}_12fps.mp4"
            
            # Check if already converted
            if output_path.exists() and not force_reconvert:
                print(f"   ⏭️  Already converted: {output_path.name}")
                skipped_conversions += 1
                continue
            
            # Convert video
            print(f"   🔄 Converting {original_fps:.1f} fps → {self.target_fps} fps")
            
            if self.convert_video_fps(video_path, output_path, self.target_fps):
                # Update database
                self.update_video_status(video_id, output_path)
                successful_conversions += 1
                print(f"   ✅ Conversion successful: {output_path.name}")
            else:
                print(f"   ❌ Conversion failed: {video_name}")
        
        print(f"\n📊 Conversion Summary:")
        print(f"   ✅ Successful: {successful_conversions}")
        print(f"   ⏭️  Skipped: {skipped_conversions}")
        print(f"   📁 Output directory: {self.output_dir}")
    
    def update_video_status(self, video_id, output_path):
        """Update video status in database after conversion"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Update the original video record with converted path
            cursor.execute('''
                UPDATE groot_videos 
                SET processing_status = 'converted', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (video_id,))
            
            # Add converted video as new record
            cursor.execute('''
                INSERT INTO groot_videos 
                (video_path, video_name, original_fps, processing_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (
                str(output_path),
                output_path.name,
                self.target_fps,
                'converted'
            ))
            
            conn.commit()
            
        except Exception as e:
            print(f"❌ Error updating database: {e}")
        finally:
            conn.close()
    
    def verify_conversions(self):
        """Verify that all converted videos have correct frame rate"""
        print("🔍 Verifying converted videos...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT video_path, video_name, original_fps
            FROM groot_videos 
            WHERE processing_status = 'converted' AND original_fps = ?
        ''', (self.target_fps,))
        
        converted_videos = cursor.fetchall()
        conn.close()
        
        if not converted_videos:
            print("❌ No converted videos found")
            return
        
        print(f"📹 Found {len(converted_videos)} converted videos")
        
        verification_results = []
        
        for video_path, video_name, expected_fps in converted_videos:
            print(f"\n🔍 Verifying: {video_name}")
            
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / actual_fps if actual_fps > 0 else 0
                
                cap.release()
                
                fps_match = abs(actual_fps - expected_fps) < 0.1
                status = "✅" if fps_match else "❌"
                
                print(f"   {status} FPS: {actual_fps:.1f} (expected: {expected_fps:.1f})")
                print(f"   📊 Frames: {frame_count}, Duration: {duration:.1f}s")
                
                verification_results.append({
                    'video_name': video_name,
                    'expected_fps': expected_fps,
                    'actual_fps': actual_fps,
                    'frame_count': frame_count,
                    'duration': duration,
                    'fps_match': fps_match
                })
            else:
                print(f"   ❌ Could not open video: {video_path}")
        
        # Summary
        successful_verifications = sum(1 for r in verification_results if r['fps_match'])
        print(f"\n📊 Verification Summary:")
        print(f"   ✅ Correct FPS: {successful_verifications}/{len(verification_results)}")
        
        if successful_verifications < len(verification_results):
            print("   ⚠️  Some videos may need re-conversion")
    
    def cleanup_original_videos(self, backup_dir="original_videos_backup"):
        """Move original videos to backup directory (optional)"""
        print("🗂️  Creating backup of original videos...")
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT video_path, video_name
            FROM groot_videos 
            WHERE processing_status = 'analyzed'
        ''')
        
        original_videos = cursor.fetchall()
        conn.close()
        
        moved_count = 0
        for video_path, video_name in original_videos:
            source_path = Path(video_path)
            if source_path.exists():
                dest_path = backup_path / video_name
                try:
                    shutil.move(str(source_path), str(dest_path))
                    moved_count += 1
                    print(f"   📦 Moved: {video_name}")
                except Exception as e:
                    print(f"   ❌ Failed to move {video_name}: {e}")
        
        print(f"📊 Backup Summary: {moved_count} videos moved to {backup_path}")

def main():
    """Main function to run frame rate conversion"""
    converter = FrameRateConverter(target_fps=12)
    
    # Convert all videos with force reconversion
    converter.convert_all_videos(force_reconvert=True)
    
    # Verify conversions
    converter.verify_conversions()
    
    # Ask about backup
    print("\n🤔 Do you want to create a backup of original videos? (y/n): ", end="")
    # For now, we'll skip the backup - you can uncomment the line below if needed
    # if input().lower() == 'y':
    #     converter.cleanup_original_videos()

if __name__ == "__main__":
    main()
