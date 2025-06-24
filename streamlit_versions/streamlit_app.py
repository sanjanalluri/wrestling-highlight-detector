# Optimized Streamlit-based Clip Labeling App
# Features: Single video loading to prevent progressive lag

import streamlit as st
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

def label_clip(filename, label, labeler):
    save_label(filename, label, labeler)
    return get_unlabeled_clip()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="WrestleAI Highlight Labeling Tool",
        page_icon="🏆",
        layout="wide"
    )
    
    st.title("🏆 WrestleAI Highlight Labeling Tool")
    
    # Display guidance
    st.markdown(GUIDANCE_TEXT)
    
    # Notice about video source
    st.info("📹 **Note: Videos are pre-loaded from the highlights folder. You cannot upload your own videos.**")
    
    # Sidebar for user info and progress
    with st.sidebar:
        st.header("Labeler Information")
        labeler_name = st.text_input("Your Name", placeholder="Enter your name to start")
        
        st.header("Progress")
        progress_text = get_labeling_progress()
        st.text(progress_text)
        
        # Show recent activity
        df = load_labels()
        if not df.empty:
            st.header("Recent Activity")
            recent = df.tail(5)
            for _, row in recent.iterrows():
                timestamp = pd.to_datetime(row['timestamp']).strftime('%H:%M')
                st.text(f"{timestamp} - {row['labeler']}: {row['label']}")
    
    # Main content area
    if not labeler_name:
        st.warning("Please enter your name to start labeling.")
        return
    
    # Get current clip
    current_clip = st.session_state.get('current_clip', None)
    if current_clip is None:
        current_clip = get_unlabeled_clip()
        if current_clip:
            st.session_state['current_clip'] = current_clip
        else:
            st.success("🎉 No more unlabeled clips!")
            return
    
    # Display current clip
    st.header(f"Current Clip: {current_clip}")
    
    video_path = os.path.join(CLIP_DIR, current_clip)
    if os.path.exists(video_path):
        try:
            # Load video with minimal caching to prevent memory buildup
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()
            
            # Display video with optimized settings
            st.video(video_bytes, start_time=0)
            
        except Exception as e:
            st.error(f"Error loading video: {e}")
            return
    else:
        st.error(f"Video file not found: {video_path}")
        return
    
    # Labeling buttons
    st.header("Label this clip:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 HIGHLIGHT", type="primary", use_container_width=True):
            next_clip = label_clip(current_clip, "Highlight", labeler_name)
            if next_clip:
                st.session_state['current_clip'] = next_clip
            else:
                st.session_state['current_clip'] = None
            st.rerun()
    
    with col2:
        if st.button("❌ NOT A HIGHLIGHT", type="secondary", use_container_width=True):
            next_clip = label_clip(current_clip, "Not Highlight", labeler_name)
            if next_clip:
                st.session_state['current_clip'] = next_clip
            else:
                st.session_state['current_clip'] = None
            st.rerun()
    
    with col3:
        if st.button("🤔 UNSURE", type="secondary", use_container_width=True):
            # For unsure, just move to next clip without saving
            next_clip = get_unlabeled_clip()
            if next_clip:
                st.session_state['current_clip'] = next_clip
            else:
                st.session_state['current_clip'] = None
            st.rerun()

if __name__ == "__main__":
    main() 