# Optimized Web-based Clip Labeling App (Gradio)
# Features: Single video loading to prevent progressive lag

import gradio as gr
import os
import random
import pandas as pd
from datetime import datetime

# Config
CLIP_DIR = "unlabeled_clips"  # Use new properly encoded clips
LABEL_FILE = "labels.csv"
GUIDANCE_TEXT = """
## What is a HIGHLIGHT?

- **Clean takedown**
- **Escape or reversal** with clear motion and point-scoring intent
- **Sustained pinning combination** (not just a quick tilt, should show control and move the match forward)
- **Match-ending move** (pin, tech, etc.)
- **Opening whistle** if it starts with aggressive action right away
- **Failed attempt** can be a highlight if it leads to an opponent's meaningful response, like a counter, reversal, or clear shift in control

## What is NOT a HIGHLIGHT?

- **Failed attempts** that don't lead to anything (short scramble, no score, reset)
- **Breaks, out-of-bounds resets**, or walking back to center
- **Ref motion or downtime**
- **Post-match running around** or crowd movement
- **Stuff that feels like filler**, even if it looks chaotic (ex. handfighting)

## If the clip is unclear or you don't know what to label it as, click **UNSURE**
"""

# Initialize label store
def load_labels():
    if os.path.exists(LABEL_FILE):
        return pd.read_csv(LABEL_FILE)
    return pd.DataFrame(columns=["filename", "label", "timestamp", "labeler"])

def save_label(filename, label, labeler):
    df = load_labels()
    # Check if already labeled with final decision
    if filename in df[df.label != "Unsure"].filename.values:
        return  # Already labeled with final decision
    
    if label == "Unsure":
        # For unsure, we don't save to CSV but also don't remove from queue
        return
    
    # Save the label
    new_entry = {"filename": filename, "label": label, "timestamp": datetime.now(), "labeler": labeler}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(LABEL_FILE, index=False)

def get_labeling_progress():
    """Get current labeling progress statistics"""
    if not os.path.exists(CLIP_DIR):
        return "No clips directory found"
    
    total_clips = len([f for f in os.listdir(CLIP_DIR) if f.endswith(".mp4")])
    if total_clips == 0:
        return "No clips found in directory"
    
    df = load_labels()
    labeled_clips = len(df[df.label != "Unsure"]) if not df.empty else 0
    highlight_count = len(df[df.label == "Highlight"]) if not df.empty else 0
    not_highlight_count = len(df[df.label == "Not Highlight"]) if not df.empty else 0
    
    progress = f"Progress: {labeled_clips}/{total_clips} clips labeled ({labeled_clips/total_clips*100:.1f}%)"
    ratio = f"Ratio: {highlight_count} highlights, {not_highlight_count} not highlights"
    return f"{progress}\n{ratio}"

# Load clips
def get_unlabeled_clip():
    if not os.path.exists(CLIP_DIR):
        return None
    
    df = load_labels()
    # Get clips that have been labeled with final decisions
    labeled_final = df[df.label != "Unsure"]["filename"].unique() if not df.empty else []
    
    # Get all clips in directory
    all_clips = [f for f in os.listdir(CLIP_DIR) if f.endswith(".mp4")]
    
    # Find unlabeled clips
    unlabeled = [clip for clip in all_clips if clip not in labeled_final]
    
    if unlabeled:
        return random.choice(unlabeled)
    return None

def label_clip(label, video_path, labeler):
    if not video_path:
        return "No video loaded", None
    
    filename = os.path.basename(video_path)
    save_label(filename, label, labeler)
    return run_labeling(labeler)

def run_labeling(labeler):
    next_clip = get_unlabeled_clip()
    if not next_clip:
        return "No more unlabeled clips!", None
    return get_labeling_progress(), os.path.join(CLIP_DIR, next_clip)

def initialize_app():
    """Initialize the app with the first video"""
    first_clip = get_unlabeled_clip()
    if first_clip:
        video_path = os.path.join(CLIP_DIR, first_clip)
        return get_labeling_progress(), video_path
    else:
        return "No clips available for labeling!", None

# Gradio UI
def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown("## 🏆 WrestleAI Highlight Labeling Tool")
        gr.Markdown(GUIDANCE_TEXT)
        
        # Add notice about video source
        gr.Markdown("### 📹 **Note: Videos are pre-loaded from the highlights folder. You cannot upload your own videos.**")

        labeler_name = gr.Textbox(label="Your Name", placeholder="Enter your name to start")
        
        # Video component - optimized for single video loading
        video = gr.Video(label="Current Clip to Label", autoplay=False)
        
        # Progress display
        progress_display = gr.Textbox(label="Labeling Progress", interactive=False, value=get_labeling_progress())

        with gr.Row():
            highlight_btn = gr.Button("🎯 Highlight", variant="primary")
            not_highlight_btn = gr.Button("❌ Not a Highlight", variant="secondary")
            unsure_btn = gr.Button("🤔 Unsure", variant="stop")

        def label_highlight(video_path, labeler):
            return label_clip("Highlight", video_path, labeler)

        def label_not_highlight(video_path, labeler):
            return label_clip("Not Highlight", video_path, labeler)

        def label_unsure(video_path, labeler):
            return run_labeling(labeler)

        def update_progress():
            return get_labeling_progress()

        # Button click handlers
        highlight_btn.click(label_highlight, inputs=[video, labeler_name], outputs=[progress_display, video])
        not_highlight_btn.click(label_not_highlight, inputs=[video, labeler_name], outputs=[progress_display, video])
        unsure_btn.click(label_unsure, inputs=[video, labeler_name], outputs=[progress_display, video])

        # Initialize with first video when app loads
        demo.load(initialize_app, outputs=[progress_display, video])

    demo.launch(share=True)  # Make sharable via Gradio link

if __name__ == "__main__":
    launch_app() 