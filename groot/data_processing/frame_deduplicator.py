"""
Frame deduplication for Groot data processing pipeline.
Tests SSIM + optical flow approach on sample clips before main deployment.
"""

import cv2
import numpy as np
from pathlib import Path
import sqlite3
from tqdm import tqdm
import sys
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATABASE_PATH, TABLE_NAMES, SSIM_THRESHOLD, 
    OPTICAL_FLOW_THRESHOLD, NUM_FRAMES
)

class FrameDeduplicator:
    """Handles frame deduplication using SSIM + optical flow"""
    
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.ssim_threshold = SSIM_THRESHOLD
        self.optical_flow_threshold = OPTICAL_FLOW_THRESHOLD
        
    def calculate_ssim(self, frame1, frame2):
        """Calculate SSIM between two frames"""
        # Convert to grayscale for SSIM calculation
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        score = ssim(gray1, gray2)
        return score
    
    def calculate_optical_flow(self, frame1, frame2):
        """Calculate average optical flow magnitude between two frames"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow using Lucas-Kanade method
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate magnitude of flow vectors
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Return average magnitude
        return np.mean(magnitude)
    
    def is_duplicate_frame(self, frame1, frame2):
        """Check if two frames are duplicates based on SSIM + optical flow"""
        ssim_score = self.calculate_ssim(frame1, frame2)
        optical_flow_mag = self.calculate_optical_flow(frame1, frame2)
        
        is_duplicate = (ssim_score > self.ssim_threshold and 
                       optical_flow_mag < self.optical_flow_threshold)
        
        return is_duplicate, ssim_score, optical_flow_mag
    
    def extract_frames_from_video(self, video_path, start_time, end_time, target_fps=12):
        """Extract frames from a video segment at target FPS"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return []
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = original_fps / target_fps
        
        # Calculate start and end frame numbers
        start_frame = int(start_time * original_fps)
        end_frame = int(end_time * original_fps)
        
        frames = []
        frame_numbers = []
        timestamps = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame_num = start_frame
        while current_frame_num <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sample frames at target FPS
            if len(frames) == 0 or (current_frame_num - start_frame) % frame_interval < 1:
                frames.append(frame.copy())
                frame_numbers.append(current_frame_num)
                timestamps.append(current_frame_num / original_fps)
            
            current_frame_num += 1
        
        cap.release()
        return frames, frame_numbers, timestamps
    
    def deduplicate_frames(self, frames, frame_numbers, timestamps):
        """Remove duplicate frames from a sequence"""
        if len(frames) <= 1:
            return frames, frame_numbers, timestamps, [], []
        
        unique_frames = [frames[0]]
        unique_frame_numbers = [frame_numbers[0]]
        unique_timestamps = [timestamps[0]]
        
        removed_indices = []
        ssim_scores = []
        optical_flow_scores = []
        
        for i in range(1, len(frames)):
            is_duplicate, ssim_score, optical_flow_mag = self.is_duplicate_frame(
                unique_frames[-1], frames[i]
            )
            
            ssim_scores.append(ssim_score)
            optical_flow_scores.append(optical_flow_mag)
            
            if is_duplicate:
                removed_indices.append(i)
            else:
                unique_frames.append(frames[i])
                unique_frame_numbers.append(frame_numbers[i])
                unique_timestamps.append(timestamps[i])
        
        return unique_frames, unique_frame_numbers, unique_timestamps, ssim_scores, optical_flow_scores
    
    def test_on_sample_clips(self, num_samples=10):
        """Test deduplication on a small sample of clips"""
        print(f"🧪 Testing frame deduplication on {num_samples} sample clips...")
        
        # Get some sample clips from the database (we'll use existing labeled clips if available)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # For now, let's test on the first few videos
        cursor.execute('''
            SELECT id, video_path, video_name, duration_seconds
            FROM groot_videos 
            WHERE processing_status = 'analyzed'
            LIMIT ?
        ''', (num_samples,))
        
        videos = cursor.fetchall()
        conn.close()
        
        if not videos:
            print("❌ No videos found in database")
            return
        
        results = []
        
        for video_id, video_path, video_name, duration in videos:
            print(f"\n📹 Testing: {video_name}")
            
            # Test on a 10-second segment from the middle of the video
            start_time = max(0, duration / 2 - 5)
            end_time = min(duration, duration / 2 + 5)
            
            print(f"   Segment: {start_time:.1f}s - {end_time:.1f}s")
            
            # Extract frames
            frames, frame_numbers, timestamps = self.extract_frames_from_video(
                video_path, start_time, end_time, target_fps=12
            )
            
            print(f"   Extracted {len(frames)} frames at 12 fps")
            
            # Deduplicate frames
            unique_frames, unique_frame_numbers, unique_timestamps, ssim_scores, optical_flow_scores = \
                self.deduplicate_frames(frames, frame_numbers, timestamps)
            
            removed_count = len(frames) - len(unique_frames)
            removal_rate = removed_count / len(frames) if frames else 0
            
            print(f"   Removed {removed_count}/{len(frames)} frames ({removal_rate:.1%})")
            
            if ssim_scores:
                avg_ssim = np.mean(ssim_scores)
                avg_flow = np.mean(optical_flow_scores)
                print(f"   Avg SSIM: {avg_ssim:.3f}, Avg Flow: {avg_flow:.3f}")
            
            results.append({
                'video_name': video_name,
                'original_frames': len(frames),
                'unique_frames': len(unique_frames),
                'removed_frames': removed_count,
                'removal_rate': removal_rate,
                'avg_ssim': np.mean(ssim_scores) if ssim_scores else 0,
                'avg_flow': np.mean(optical_flow_scores) if optical_flow_scores else 0
            })
        
        # Print summary
        print(f"\n📊 Deduplication Test Summary:")
        print("Video\t\t\t| Original | Unique | Removed | Rate\t| Avg SSIM | Avg Flow")
        print("-" * 80)
        
        for result in results:
            print(f"{result['video_name'][:20]:<20}\t| {result['original_frames']:<8} | {result['unique_frames']:<6} | {result['removed_frames']:<7} | {result['removal_rate']:.1%}\t| {result['avg_ssim']:.3f}\t| {result['avg_flow']:.3f}")
        
        return results
    
    def visualize_deduplication(self, video_path, start_time, end_time, save_path=None):
        """Visualize the deduplication process for a specific video segment"""
        print(f"🎨 Visualizing deduplication for {Path(video_path).name}")
        
        # Extract frames
        frames, frame_numbers, timestamps = self.extract_frames_from_video(
            video_path, start_time, end_time, target_fps=12
        )
        
        # Deduplicate
        unique_frames, unique_frame_numbers, unique_timestamps, ssim_scores, optical_flow_scores = \
            self.deduplicate_frames(frames, frame_numbers, timestamps)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Frame Deduplication Analysis: {Path(video_path).name}')
        
        # Plot 1: SSIM scores over time
        if ssim_scores:
            axes[0, 0].plot(ssim_scores, 'b-', label='SSIM Score')
            axes[0, 0].axhline(y=self.ssim_threshold, color='r', linestyle='--', label=f'Threshold ({self.ssim_threshold})')
            axes[0, 0].set_title('SSIM Scores Over Time')
            axes[0, 0].set_xlabel('Frame Pair Index')
            axes[0, 0].set_ylabel('SSIM Score')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot 2: Optical flow magnitudes
        if optical_flow_scores:
            axes[0, 1].plot(optical_flow_scores, 'g-', label='Optical Flow Magnitude')
            axes[0, 1].axhline(y=self.optical_flow_threshold, color='r', linestyle='--', label=f'Threshold ({self.optical_flow_threshold})')
            axes[0, 1].set_title('Optical Flow Magnitudes Over Time')
            axes[0, 1].set_xlabel('Frame Pair Index')
            axes[0, 1].set_ylabel('Average Flow Magnitude (pixels)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot 3: Frame removal visualization
        axes[1, 0].bar(['Original', 'After Deduplication'], [len(frames), len(unique_frames)], 
                      color=['lightblue', 'lightgreen'])
        axes[1, 0].set_title('Frame Count Comparison')
        axes[1, 0].set_ylabel('Number of Frames')
        
        # Add text annotations
        for i, v in enumerate([len(frames), len(unique_frames)]):
            axes[1, 0].text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        # Plot 4: Sample frame comparison
        if len(frames) >= 2 and len(unique_frames) >= 1:
            # Show first two original frames
            axes[1, 1].imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title('Sample: First Frame')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved to: {save_path}")
        
        plt.show()

def main():
    """Main function to test frame deduplication"""
    deduplicator = FrameDeduplicator()
    
    # Test on sample clips
    results = deduplicator.test_on_sample_clips(num_samples=5)
    
    # Visualize one example
    if results:
        # Get the first video for visualization
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT video_path FROM groot_videos LIMIT 1')
        video_path = cursor.fetchone()[0]
        conn.close()
        
        # Create plots directory
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        deduplicator.visualize_deduplication(
            video_path, 10, 20, 
            save_path=plots_dir / "frame_deduplication_test.png"
        )

if __name__ == "__main__":
    main()
