import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import find_peaks
import subprocess
import os

# setup
cap = cv2.VideoCapture('wrestlemotion.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# MOTION SCORE CALCULATION **********
motion_scores = []
prev_gray = None

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

cap.release()
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
print("Highlight Ranges (in seconds):")
for i, (start, end) in enumerate(clip_ranges):
    print(f"Clip {i+1}: {start:.2f}s → {end:.2f}s")

# PLOTTING **********
# Comment out plotting to focus on clip generation
# manual marked regions
# manual_highlight_seconds = [13, 22, 31, 57, 71, 105, 124, 143]
# manual_highlight_frames = [int(s * fps) for s in manual_highlight_seconds]
# buffer = int(fps * 0.5)  # half-second buffer on each side

# # Plot motion curve
# plt.plot(smoothed, label='Motion Score')

# # Plot detected peaks as dots
# plt.plot(peaks, smoothed[peaks], "o", color='blue', label='Detected Peaks')

# # Add shaded highlight regions
# for f in manual_highlight_frames:
#     plt.axvspan(f - buffer, f + buffer, color='red', alpha=0.3)

# plt.title("Motion Score with Manual Highlight Regions")
# plt.xlabel("Frame Index")
# plt.ylabel("Motion Score")
# plt.legend()
# plt.show()

# CREATE HIGHLIGHT CLIPS **********
input_video = 'wrestlemotion.mp4'
output_dir = 'unlabeled_clips'
os.makedirs(output_dir, exist_ok=True)

print(f"\nCreating {len(clip_ranges)} highlight clips with proper encoding...")

for i, (start, end) in enumerate(clip_ranges):
    output_path = os.path.join(output_dir, f'highlight_{i+1}.mp4')
    print(f"Creating clip {i+1}/{len(clip_ranges)}: {start:.2f}s → {end:.2f}s")
    
    # Use proper encoding instead of fast copy
    cmd = [
        'ffmpeg',
        '-y',  # overwrite output
        '-i', input_video,
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
        print(f"✓ Created {output_path}")
    else:
        print(f"✗ Failed to create {output_path}")
        print(f"Error: {result.stderr}")

print(f"\n✅ All clips created in '{output_dir}' directory!")
print("These clips should play smoothly in web browsers without lag.") 