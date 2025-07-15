# Manual Range Selection and Labeling App (Gradio)
# Features: Watch matches and manually select time ranges to label as LIVE/NOT LIVE

import gradio as gr
import os
import random
import pandas as pd
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database_manager import DatabaseManager

# Config
GUIDANCE_TEXT = """
## Manual Range Selection and Labeling

**How to use:**
1. Select a match from the dropdown
2. Watch the video and identify time ranges
3. Use the time inputs to set start/end times for each range
4. Label each range as LIVE or NOT LIVE
5. Add any notes if needed
6. Click "Add Range" to save the range
7. Continue until you've covered the entire match

**Labeling Guidelines:**
- **LIVE**: Any time the ref is actively watching and the match is on the clock
- **NOT LIVE**: Breaks, walking back to center, injury time, before/after match, etc.

**Tips:**
- You can add multiple ranges for the same match
- Ranges can be any length (30 seconds to several minutes)
- Focus on representative periods rather than every second
- You can skip obvious non-live sections (long breaks, etc.)
"""

# Initialize database manager
db_manager = DatabaseManager()

def get_range_labeling_progress():
    """Get current range labeling progress statistics from database"""
    progress = db_manager.get_range_labeling_progress()
    
    if progress['total_matches'] == 0:
        return "No matches found in database"
    
    progress_text = f"Progress: {progress['matches_with_ranges']}/{progress['total_matches']} matches processed ({progress['progress_percent']:.1f}%)"
    ranges_text = f"Total ranges: {progress['total_ranges']} ({progress['live_count']} LIVE, {progress['not_live_count']} NOT LIVE)"
    return f"{progress_text}\n{ranges_text}"

def get_matches_dropdown():
    """Get list of matches for dropdown"""
    matches = db_manager.get_all_matches()
    if not matches:
        return [], "No matches found"
    
    # Create dropdown options: "filename (duration)" format
    options = []
    for match in matches:
        duration_min = int(match['duration'] // 60)
        duration_sec = int(match['duration'] % 60)
        option_text = f"{match['filename']} ({duration_min}:{duration_sec:02d})"
        options.append((option_text, match['id']))
    
    return options, f"Found {len(matches)} matches"

def get_match_info(match_id):
    """Get information about a specific match"""
    if not match_id:
        return "No match selected", None, []
    
    match = db_manager.get_match_by_id(match_id)
    if not match:
        return "Match not found", None, []
    
    # Format duration
    duration_min = int(match['duration'] // 60)
    duration_sec = int(match['duration'] % 60)
    duration_text = f"{duration_min}:{duration_sec:02d}"
    
    # Get existing ranges for this match
    ranges = db_manager.get_ranges_for_match(match_id)
    ranges_text = ""
    if ranges:
        ranges_text = f"\n\n**Existing ranges ({len(ranges)}):**\n"
        for r in ranges:
            start_min = int(r['start_time'] // 60)
            start_sec = int(r['start_time'] % 60)
            end_min = int(r['end_time'] // 60)
            end_sec = int(r['end_time'] % 60)
            ranges_text += f"- {start_min}:{start_sec:02d} → {end_min}:{end_sec:02d} ({r['label']})\n"
    
    info_text = f"**Match:** {match['filename']}\n**Duration:** {duration_text}\n**File size:** {match['file_size'] / (1024*1024):.1f} MB{ranges_text}"
    
    # Use preview video if available, otherwise use original
    if match.get('preview_filepath') and os.path.exists(match['preview_filepath']):
        filepath_to_use = match['preview_filepath']
        info_text += "\n**Using preview video for faster loading**"
    else:
        filepath_to_use = match['filepath']
        info_text += "\n**Using original video (preview not available)**"
    
    # Use absolute path for Gradio
    absolute_filepath = os.path.abspath(filepath_to_use)
    
    return info_text, absolute_filepath, ranges

def add_range(match_id, start_time, end_time, label, labeler_name, notes):
    """Add a new range to the database"""
    if not match_id or not labeler_name:
        return "Missing match or labeler information", None, None, None, None, None
    
    if not start_time or not end_time:
        return "Please set both start and end times", None, None, None, None, None
    
    if start_time >= end_time:
        return "Start time must be before end time", None, None, None, None, None
    
    try:
        # Add range to database
        range_id = db_manager.add_match_range(match_id, start_time, end_time, label, labeler_name, notes)
        print(f"✅ Added range: {start_time:.1f}s → {end_time:.1f}s ({label})")
        
        # Get updated match info
        info_text, filepath, ranges = get_match_info(match_id)
        
        # Clear the form
        return f"✅ Range added successfully! (ID: {range_id})", info_text, filepath, ranges, "", ""
        
    except Exception as e:
        print(f"❌ Error adding range: {str(e)}")
        return f"Error adding range: {str(e)}", None, None, None, None, None

def format_time_input(time_seconds):
    """Convert seconds to MM:SS format for display"""
    if not time_seconds:
        return "0:00"
    minutes = int(time_seconds // 60)
    seconds = int(time_seconds % 60)
    return f"{minutes}:{seconds:02d}"

def parse_time_input(time_str):
    """Convert MM:SS format to seconds"""
    if not time_str or ':' not in time_str:
        return 0
    
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        return 0
    except:
        return 0

# Gradio UI
def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown("## 🏆 WrestleAI Manual Range Selection Tool")
        gr.Markdown(GUIDANCE_TEXT)
        
        # Progress display
        progress_display = gr.Textbox(label="Range Labeling Progress", interactive=False, value=get_range_labeling_progress())

        with gr.Row():
            # Left column: Match selection and video
            with gr.Column(scale=2):
                gr.Markdown("### 📹 Match Selection")
                
                # Match dropdown
                matches_options, matches_status = get_matches_dropdown()
                match_dropdown = gr.Dropdown(
                    choices=matches_options,
                    label="Select Match",
                    type="value"
                )
                
                # Match info display
                match_info = gr.Markdown("Select a match to begin")
                
                # Video player
                video = gr.Video(label="Match Video", autoplay=False)
                
                # Existing ranges display
                ranges_display = gr.Markdown("")
            
            # Right column: Range input form
            with gr.Column(scale=1):
                gr.Markdown("### ✏️ Range Input")
                
                labeler_name = gr.Textbox(label="Your Name", placeholder="Enter your name")
                
                # Time inputs
                start_time_input = gr.Textbox(
                    label="Start Time (MM:SS)", 
                    placeholder="0:30",
                    info="Format: MM:SS (e.g., 1:30 for 1 minute 30 seconds)"
                )
                
                end_time_input = gr.Textbox(
                    label="End Time (MM:SS)", 
                    placeholder="2:15",
                    info="Format: MM:SS (e.g., 2:15 for 2 minutes 15 seconds)"
                )
                
                # Label selection
                label_radio = gr.Radio(
                    choices=["LIVE", "NOT LIVE"],
                    label="Range Label",
                    value="LIVE"
                )
                
                notes_input = gr.Textbox(
                    label="Notes (Optional)", 
                    placeholder="Any additional notes about this range"
                )
                
                # Add range button
                add_range_btn = gr.Button("➕ Add Range", variant="primary")
                
                # Status message
                status_msg = gr.Textbox(label="Status", interactive=False)

        # Event handlers
        def on_match_select(match_id):
            """Handle match selection"""
            if not match_id:
                return "No match selected", None, ""
            
            info_text, filepath, ranges = get_match_info(match_id)
            
            # Debug: print filepath
            print(f"🎬 Selected match filepath: {filepath}")
            if filepath:
                import os
                print(f"📁 File exists: {os.path.exists(filepath)}")
                print(f"📊 File size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
            
            # Format ranges for display
            ranges_text = ""
            if ranges:
                ranges_text = "**Existing ranges:**\n"
                for r in ranges:
                    start_str = format_time_input(r['start_time'])
                    end_str = format_time_input(r['end_time'])
                    ranges_text += f"- {start_str} → {end_str} ({r['label']})\n"
            
            return info_text, filepath, ranges_text

        def on_add_range(match_id, start_str, end_str, label, labeler, notes):
            """Handle adding a range"""
            if not match_id:
                return "No match selected", None, None, None, None, None
            
            # Parse time inputs
            start_time = parse_time_input(start_str)
            end_time = parse_time_input(end_str)
            
            # Add the range
            result = add_range(match_id, start_time, end_time, label, labeler, notes)
            status = result[0]
            info_text = result[1] if result[1] else ""
            ranges = result[3] if result[3] else []
            
            # Format ranges for display
            ranges_text = ""
            if ranges:
                ranges_text = "**Existing ranges:**\n"
                for r in ranges:
                    start_str = format_time_input(r['start_time'])
                    end_str = format_time_input(r['end_time'])
                    ranges_text += f"- {start_str} → {end_str} ({r['label']})\n"
            
            # Keep the current video loaded by getting the match info again
            current_match_info = get_match_info(match_id)
            current_filepath = current_match_info[1]  # Get the filepath from match info
            
            return status, info_text, current_filepath, ranges_text, "", ""

        # Connect event handlers
        match_dropdown.change(
            on_match_select,
            inputs=[match_dropdown],
            outputs=[match_info, video, ranges_display]
        )
        
        add_range_btn.click(
            on_add_range,
            inputs=[match_dropdown, start_time_input, end_time_input, label_radio, labeler_name, notes_input],
            outputs=[status_msg, match_info, video, ranges_display, start_time_input, end_time_input]
        )

    demo.launch(share=True)  # Make sharable via Gradio link

if __name__ == "__main__":
    launch_app() 