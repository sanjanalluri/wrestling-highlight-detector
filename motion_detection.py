import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import find_peaks
import subprocess
import os
import glob
from database.database_manager import DatabaseManager

# Configuration
MATCH_DIR = 'match_directory'  # Directory containing full wrestling videos
OUTPUT_DIR = 'unlabeled_clips'  # Directory for generated highlight clips

def process_video(video_path: str, output_dir: str, db_manager: DatabaseManager, video_id: int):
    """Process a single video file and extract highlight clips"""
    print(f"\n🎬 Processing video: {os.path.basename(video_path)}")
    
    try:
        # Log processing start
        db_manager.log_processing(video_id, "motion_detection", "started")

        # setup
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if not cap.isOpened():
            error_msg = f"Could not open video {video_path}"
            db_manager.log_processing(video_id, "motion_detection", "failed", error_msg)
            print(f"❌ Error: {error_msg}")
            return
        
        print(f"📊 Video FPS: {fps}")

        # MOTION SCORE CALCULATION **********
        motion_scores = []
        prev_gray = None
        frame_count = 0

        print("🔄 Analyzing motion...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_score = np.sum(diff)
                motion_scores.append(motion_score)

            prev_gray = gray
            frame_count += 1
            
            # Progress indicator
            if frame_count % 1000 == 0:
                print(f"   Processed {frame_count} frames...")

        cap.release()
        
        if len(motion_scores) == 0:
            error_msg = f"No motion data extracted from {video_path}"
            db_manager.log_processing(video_id, "motion_detection", "failed", error_msg)
            print(f"❌ {error_msg}")
            return
        
        # smooth the motion scores
        smoothed = scipy.ndimage.gaussian_filter1d(motion_scores, sigma=5)

        # DETECT PEAKS **********
        peaks, _ = find_peaks(smoothed, distance=fps*2, height=(np.mean(smoothed)))  # tune this if too many/too few

        # create 5-sec windows around each peak
        clip_ranges = []
        clip_duration = 2.5  # seconds before and after

        for peak in peaks:
            peak_time = peak / fps
            start = max(0, peak_time - clip_duration)
            end = peak_time + clip_duration
            clip_ranges.append((start, end))

        # printing clip time ranges
        print(f"🎯 Found {len(clip_ranges)} potential highlights:")
        for i, (start, end) in enumerate(clip_ranges):
            print(f"   Clip {i+1}: {start:.2f}s → {end:.2f}s")

        # CREATE HIGHLIGHT CLIPS **********
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        print(f"\n🎬 Creating {len(clip_ranges)} highlight clips...")
        clips_created = 0

        for i, (start, end) in enumerate(clip_ranges):
            output_filename = f'{video_name}_highlight_{i+1}.mp4'
            output_path = os.path.join(output_dir, output_filename)
            print(f"   Creating clip {i+1}/{len(clip_ranges)}: {start:.2f}s → {end:.2f}s")
    
            # Use proper encoding instead of fast copy
            cmd = [
                'ffmpeg',
                '-y',  # overwrite output
                '-i', video_path,
                '-ss', f'{start:.2f}',
                '-to', f'{end:.2f}',
                '-c:v', 'libx264',           # H.264 video codec
                '-profile:v', 'baseline',     # Most compatible profile
                '-level', '3.0',              # Standard level
                '-preset', 'fast',            # Fast encoding
                '-crf', '23',                 # Good quality
                '-c:a', 'aac',                # AAC audio codec
                '-b:a', '128k',               # Audio bitrate
                '-movflags', '+faststart',    # Optimize for web streaming
                '-pix_fmt', 'yuv420p',        # Standard pixel format
                output_path
            ]
    
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ✓ Created {output_filename}")
                
                # Add clip to database
                motion_score = smoothed[peaks[i]] if i < len(peaks) else None
                db_manager.add_clip(video_id, output_filename, output_path, start, end, motion_score)
                clips_created += 1
            else:
                print(f"   ✗ Failed to create {output_filename}")
                print(f"   Error: {result.stderr}")

        # Mark video as processed
        db_manager.mark_video_processed(video_id)
        db_manager.log_processing(video_id, "motion_detection", "completed")
        
        print(f"✅ Successfully processed video: {os.path.basename(video_path)}")
        print(f"📁 Created {clips_created} clips")
        
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        db_manager.log_processing(video_id, "motion_detection", "failed", error_msg)
        print(f"❌ {error_msg}")

def main():
    """Main function to process all videos in match_directory"""
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if match directory exists
    if not os.path.exists(MATCH_DIR):
        print(f"❌ Error: {MATCH_DIR} directory not found!")
        print(f"Please create the {MATCH_DIR} directory and add your wrestling videos.")
        return
    
    # Find all video files in match directory
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(MATCH_DIR, ext)))
        video_files.extend(glob.glob(os.path.join(MATCH_DIR, ext.upper())))
    
    if not video_files:
        print(f"❌ No video files found in {MATCH_DIR} directory!")
        print(f"Supported formats: {', '.join(video_extensions)}")
        return
    
    print(f"🎬 Found {len(video_files)} video(s) in {MATCH_DIR}:")
    for video_file in video_files:
        print(f"   - {os.path.basename(video_file)}")
    
    # Register all videos in database
    print(f"\n📝 Registering videos in database...")
    video_ids = []
    for video_file in video_files:
        try:
            video_id = db_manager.register_video(video_file)
            video_ids.append(video_id)
        except Exception as e:
            print(f"❌ Error registering {os.path.basename(video_file)}: {e}")
    
    # Get unprocessed videos
    unprocessed_videos = db_manager.get_unprocessed_videos()
    
    if not unprocessed_videos:
        print(f"\n✅ All videos have been processed!")
        return
    
    print(f"\n🔄 Processing {len(unprocessed_videos)} unprocessed video(s)...")
    
    # Process each unprocessed video
    total_clips_created = 0
    for video_info in unprocessed_videos:
        process_video(video_info['filepath'], OUTPUT_DIR, db_manager, video_info['id'])
    
    # Get final statistics
    progress = db_manager.get_labeling_progress()
    print(f"\n✅ Processing complete!")
    print(f"📊 Database Statistics:")
    print(f"   - Total clips: {progress['total_clips']}")
    print(f"   - Labeled clips: {progress['labeled_clips']}")
    print(f"   - Progress: {progress['progress_percent']:.1f}%")
    print(f"   - Highlights: {progress['highlight_count']}")
    print(f"   - Not highlights: {progress['not_highlight_count']}")
    print(f"\n🎯 Clips are ready for labeling in the web app.")

if __name__ == "__main__":
    main() 