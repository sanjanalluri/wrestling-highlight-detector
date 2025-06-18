import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import find_peaks

# setup
cap = cv2.VideoCapture('wrestlemotion.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# motion score calculation
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
smoothed = scipy.ndimage.gaussian_filter1d(motion_scores, sigma=5)

# Detect peaks in smoothed signal
peaks, _ = find_peaks(smoothed, distance=fps*2, height=np.mean(smoothed))  # tune this if too many/too few

# Create 5-sec windows around each peak
clip_ranges = []
clip_duration = 2.5  # seconds before and after

for peak in peaks:
    peak_time = peak / fps
    start = max(0, peak_time - clip_duration)
    end = peak_time + clip_duration
    clip_ranges.append((start, end))

# Print the clip time ranges
print("Highlight Ranges (in seconds):")
for i, (start, end) in enumerate(clip_ranges):
    print(f"Clip {i+1}: {start:.2f}s → {end:.2f}s")

# manual marked regions
manual_highlight_seconds = [13, 22, 31, 57, 71, 105, 124, 143]
manual_highlight_frames = [int(s * fps) for s in manual_highlight_seconds]
buffer = int(fps * 0.5)  # half-second buffer on each side

# Plot motion curve
plt.plot(smoothed, label='Motion Score')

# Add shaded highlight regions
for f in manual_highlight_frames:
    plt.axvspan(f - buffer, f + buffer, color='red', alpha=0.3)

plt.title("Motion Score with Manual Highlight Regions")
plt.xlabel("Frame Index")
plt.ylabel("Motion Score")
plt.legend()
plt.show()