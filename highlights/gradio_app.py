# Optimized Web-based Clip Labeling App (Gradio)
# Features: Database-driven clip management and labeling

import gradio as gr
import os
import random
import pandas as pd
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database_manager import DatabaseManager

# Config
CLIP_DIR = "../live_filter/clips"  # Use new properly encoded clips
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

# Initialize database manager
db_manager = DatabaseManager()

def get_labeling_progress():
    """Get current labeling progress statistics from database"""
    progress = db_manager.get_labeling_progress()
    
    if progress['total_clips'] == 0:
        return "No clips found in database"
    
    progress_text = f"Progress: {progress['labeled_clips']}/{progress['total_clips']} clips labeled ({progress['progress_percent']:.1f}%)"
    ratio = f"Ratio: {progress['highlight_count']} highlights, {progress['not_highlight_count']} not highlights"
    return f"{progress_text}\n{ratio}"

def get_source_video_info():
    """Get information about source videos in match_directory"""
    match_dir = "../match_directory"
    if not os.path.exists(match_dir):
        return "No match_directory found"
    
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']:
        video_files.extend([f for f in os.listdir(match_dir) if f.lower().endswith(ext.replace('*', ''))])
    
    if not video_files:
        return "No source videos found in match_directory"
    
    return f"Source videos: {', '.join(video_files)}"

def get_database_stats():
    """Get comprehensive database statistics"""
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    # Video statistics
    cursor.execute("SELECT COUNT(*) FROM videos")
    total_videos = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM videos WHERE status = 'processed'")
    processed_videos = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM videos WHERE status = 'pending'")
    pending_videos = cursor.fetchone()[0]
    
    # Clip statistics
    cursor.execute("SELECT COUNT(*) FROM clips")
    total_clips = cursor.fetchone()[0]
    
    # Label statistics
    cursor.execute("SELECT COUNT(DISTINCT clip_id) FROM labels WHERE label IN ('Highlight', 'Not Highlight')")
    labeled_clips = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM labels WHERE label = 'Highlight'")
    highlights = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM labels WHERE label = 'Not Highlight'")
    not_highlights = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total_videos': total_videos,
        'processed_videos': processed_videos,
        'pending_videos': pending_videos,
        'total_clips': total_clips,
        'labeled_clips': labeled_clips,
        'highlights': highlights,
        'not_highlights': not_highlights
    }

# Load clips
def get_unlabeled_clip():
    """Get a random unlabeled clip from the database"""
    unlabeled_clips = db_manager.get_unlabeled_clips()
    
    if not unlabeled_clips:
        print("❌ No unlabeled clips found in database")
        return None
    
    print(f"📊 Found {len(unlabeled_clips)} unlabeled clips")
    if len(unlabeled_clips) <= 5:
        for clip in unlabeled_clips:
            print(f"   - {clip['filename']}")
    
    # Return a random unlabeled clip
    selected_clip = random.choice(unlabeled_clips)
    print(f"🎯 Selected clip: {selected_clip['filename']}")
    return selected_clip

def label_clip(clip_info, label, labeler):
    """Label a clip and save to database"""
    if not clip_info or not labeler:
        return "Missing clip or labeler information", None
    
    try:
        print(f"🎯 Saving label: {label} for clip {clip_info['filename']} by {labeler}")
        # Add label to database
        db_manager.add_label(clip_info['id'], labeler, label)
        print(f"✅ Successfully saved label to database")
        
        # Get next clip
        return run_labeling(labeler)
    except Exception as e:
        print(f"❌ Error saving label: {str(e)}")
        return f"Error saving label: {str(e)}", None

def run_labeling(labeler):
    """Get next clip for labeling"""
    next_clip = get_unlabeled_clip()
    if not next_clip:
        return "No more unlabeled clips!", None
    
    # Return progress and video path
    return get_labeling_progress(), next_clip['filepath']

def initialize_app():
    """Initialize the app with the first video"""
    first_clip = get_unlabeled_clip()
    if first_clip:
        return get_labeling_progress(), first_clip['filepath']
    else:
        return "No clips available for labeling!", None

# Gradio UI
def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown("## 🏆 WrestleAI Highlight Labeling Tool")
        gr.Markdown(GUIDANCE_TEXT)
        
        # Add notice about video source and directory structure
        gr.Markdown("### 📹 **Video Processing Pipeline**")
        gr.Markdown("1. **Source videos** are placed in `match_directory/` folder")
        gr.Markdown("2. **Motion detection** analyzes videos and extracts potential highlights")
        gr.Markdown("3. **Highlight clips** are generated in `unlabeled_clips/` folder")
        gr.Markdown("4. **Labeling** happens here in the web app")
        
        # Display source video information
        source_info = get_source_video_info()
        gr.Markdown(f"**Current source videos:** {source_info}")
        
        # Display database statistics
        stats = get_database_stats()
        gr.Markdown(f"**Database Stats:** {stats['total_videos']} videos ({stats['processed_videos']} processed, {stats['pending_videos']} pending), {stats['total_clips']} clips")

        labeler_name = gr.Textbox(label="Your Name", placeholder="Enter your name to start")
        
        # Video component - optimized for single video loading
        video = gr.Video(label="Current Clip to Label", autoplay=False)
        
        # Progress display
        progress_display = gr.Textbox(label="Labeling Progress", interactive=False, value=get_labeling_progress())

        with gr.Row():
            highlight_btn = gr.Button("🎯 Highlight", variant="primary")
            not_highlight_btn = gr.Button("❌ Not a Highlight", variant="secondary")
            unsure_btn = gr.Button("🤔 Unsure", variant="stop")

        def get_clip_info_from_filename(filename):
            """Get clip info from database, handling special characters"""
            if not filename:
                return None
            
            # Get all unlabeled clips
            unlabeled_clips = db_manager.get_unlabeled_clips()
            
            # Try exact match first
            clip_info = next((clip for clip in unlabeled_clips if clip['filename'] == filename), None)
            if clip_info:
                return clip_info
            
            # Try normalized comparison (handle special characters)
            normalized_filename = filename.replace('：', ':').replace('：', ':')
            for clip in unlabeled_clips:
                normalized_clip_name = clip['filename'].replace('：', ':').replace('：', ':')
                if normalized_clip_name == normalized_filename:
                    return clip
            
            # Try partial match as fallback
            for clip in unlabeled_clips:
                if filename in clip['filename'] or clip['filename'] in filename:
                    return clip
            
            return None

        def label_highlight(video_path, labeler):
            # Get clip info from database based on filename
            if not video_path or not labeler:
                return "Missing video path or labeler name", None
            
            # Extract filename from video_path (Gradio provides full path)
            filename = os.path.basename(video_path)
            print(f"🔍 Looking for clip: {filename}")
            
            # Get clip info from database based on filename
            clip_info = get_clip_info_from_filename(filename)
            
            if not clip_info:
                print(f"❌ Could not find clip info for filename: {filename}")
                return f"Error: Could not find clip {filename} in database", None
            
            return label_clip(clip_info, "Highlight", labeler)

        def label_not_highlight(video_path, labeler):
            # Get clip info from database based on filename
            if not video_path or not labeler:
                return "Missing video path or labeler name", None
            
            # Extract filename from video_path (Gradio provides full path)
            filename = os.path.basename(video_path)
            print(f"🔍 Looking for clip: {filename}")
            
            # Get clip info from database based on filename
            clip_info = get_clip_info_from_filename(filename)
            
            if not clip_info:
                print(f"❌ Could not find clip info for filename: {filename}")
                return f"Error: Could not find clip {filename} in database", None
            
            return label_clip(clip_info, "Not Highlight", labeler)

        def label_unsure(video_path, labeler):
            # Get clip info from database based on filename
            if not video_path or not labeler:
                return "Missing video path or labeler name", None
            
            # Extract filename from video_path (Gradio provides full path)
            filename = os.path.basename(video_path)
            print(f"🔍 Looking for clip: {filename}")
            
            # Get clip info from database based on filename
            clip_info = get_clip_info_from_filename(filename)
            
            if clip_info and labeler:
                try:
                    # Save "Unsure" label to database
                    db_manager.add_label(clip_info['id'], labeler, "Unsure")
                    print(f"🤔 Marked clip as Unsure: {clip_info['filename']}")
                except Exception as e:
                    print(f"Error saving Unsure label: {str(e)}")
                    return f"Error saving Unsure label: {str(e)}", None
            else:
                print(f"❌ Could not find clip info for filename: {filename}")
                return f"Error: Could not find clip {filename} in database", None
            
            # Move to next clip
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