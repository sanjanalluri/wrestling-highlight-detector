#!/usr/bin/env python3
"""
Motion Plotter for Wrestling Videos
Interactive tool to visualize motion scores and manually annotate highlights
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import find_peaks
import subprocess
import os
import glob

def get_video_files():
    """Get list of video files in match_directory"""
    match_dir = 'match_directory'
    if not os.path.exists(match_dir):
        print(f"❌ {match_dir} directory not found!")
        return []
    
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(match_dir, ext)))
        video_files.extend(glob.glob(os.path.join(match_dir, ext.upper())))
    
    return video_files

def select_video():
    """Let user select a video file"""
    video_files = get_video_files()
    
    if not video_files:
        print("❌ No video files found in match_directory!")
        return None
    
    print("📹 Available videos:")
    for i, video_file in enumerate(video_files, 1):
        filename = os.path.basename(video_file)
        print(f"   {i}. {filename}")
    
    while True:
        try:
            choice = input(f"\nSelect video (1-{len(video_files)}): ").strip()
            video_index = int(choice) - 1
            
            if 0 <= video_index < len(video_files):
                selected_video = video_files[video_index]
                print(f"✅ Selected: {os.path.basename(selected_video)}")
                return selected_video
            else:
                print(f"❌ Please enter a number between 1 and {len(video_files)}")
        except ValueError:
            print("❌ Please enter a valid number")

def get_manual_highlights():
    """Get manual highlight timestamps from user"""
    print("\n🎯 Manual Highlight Annotation")
    print("Enter timestamps (in seconds) where you think highlights occur")
    print("Enter 'done' when finished, or 'skip' to skip manual annotation")
    
    highlights = []
    
    while True:
        timestamp_input = input("Highlight timestamp (seconds): ").strip().lower()
        
        if timestamp_input == 'done':
            break
        elif timestamp_input == 'skip':
            return []
        
        try:
            timestamp = float(timestamp_input)
            if timestamp < 0:
                print("❌ Timestamp must be positive")
                continue
            highlights.append(timestamp)
            print(f"✅ Added highlight at {timestamp:.2f}s")
        except ValueError:
            print("❌ Please enter a valid number or 'done'/'skip'")
    
    return sorted(highlights)

def analyze_video(video_path, manual_highlight_seconds=None):
    """Analyze video and create motion plot"""
    print(f"🔄 Analyzing motion in {os.path.basename(video_path)}...")
    
    # setup
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f} seconds")
    
    # MOTION SCORE CALCULATION **********
    motion_scores = []
    prev_gray = None
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = np.sum(diff)
            motion_scores.append(motion_score)
        else:
            # For the first frame, use a motion score of 0
            motion_scores.append(0)
        
        prev_gray = gray
        frame_count += 1
        
        # Progress indicator
        if frame_count % 1000 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Progress: {progress:.1f}%")
    
    cap.release()
    
    if len(motion_scores) == 0:
        print("❌ No motion data extracted!")
        return
    
    print(f"✅ Extracted {len(motion_scores)} motion scores")
    
    # smooth the motion scores
    smoothed = scipy.ndimage.gaussian_filter1d(motion_scores, sigma=5)
    
    # DETECT PEAKS **********
    peaks, _ = find_peaks(smoothed, distance=fps*2, height=(np.mean(smoothed)))  # tune this if too many/too few
    
    # create 5-sec windows around each peak
    clip_ranges = []
    clip_duration = 2.5  # seconds before and after
    
    for peak in peaks:
        peak_time = peak / fps
        start = peak_time - clip_duration
        end = peak_time + clip_duration
        clip_ranges.append((start, end))
    
    # printing clip time ranges
    print(f"\n📊 Auto-detected {len(clip_ranges)} highlight ranges:")
    for i, (start, end) in enumerate(clip_ranges):
        print(f"   Clip {i+1}: {start:.2f}s → {end:.2f}s")
    
    # PLOTTING **********
    # manual marked regions
    if manual_highlight_seconds is None:
        manual_highlight_seconds = []
    
    manual_highlight_frames = [int(s * fps) for s in manual_highlight_seconds]
    buffer = int(2.5 * fps)  # 2.5 second buffer around each highlight
    
    # Plot motion curve
    plt.figure(figsize=(15, 8))
    plt.plot(smoothed, label='Motion Score', linewidth=1)
    
    # Plot detected peaks as dots
    plt.plot(peaks, smoothed[peaks], "o", color='blue', label=f'Detected Peaks ({len(peaks)})', markersize=8)
    
    # Add shaded highlight regions
    for f in manual_highlight_frames:
        plt.axvspan(f - buffer, f + buffer, color='red', alpha=0.3)
    
    plt.xlabel("Frame Index")
    plt.ylabel("Motion Score")
    plt.title(f"Motion Analysis: {os.path.basename(video_path)}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    plot_filename = f"motion_analysis_{video_name}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as: {plot_filename}")
    
    plt.show()
    
    print(f"\n✅ Motion analysis complete for {os.path.basename(video_path)}!")

def main():
    """Main function"""
    print("🏆 Wrestling Motion Analysis Tool")
    print("=" * 40)
    
    # Select video
    video_path = select_video()
    if not video_path:
        return
    
    # Get manual highlights
    manual_highlights = get_manual_highlights()
    
    # Analyze video and create plot
    analyze_video(video_path, manual_highlights)

if __name__ == "__main__":
    main()