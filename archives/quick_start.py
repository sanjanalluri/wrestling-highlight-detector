#!/usr/bin/env python3
"""
Quick Start Guide for Wrestling Highlight Detector
This script guides you through the complete setup and usage process.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = ['gradio', 'opencv-python', 'pandas', 'numpy', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def create_directories():
    """Create necessary directories"""
    directories = ['match_directory', 'unlabeled_clips']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    print("🎬 Wrestling Highlight Detector - Quick Start")
    print("=" * 50)
    
    # Check FFmpeg
    print("\n🔍 Checking FFmpeg installation...")
    if check_ffmpeg():
        print("✅ FFmpeg is installed")
    else:
        print("❌ FFmpeg not found!")
        print("Please install FFmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        return
    
    # Check Python packages
    print("\n🔍 Checking Python packages...")
    missing = check_python_packages()
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Please install with: pip install -r requirements.txt")
        return
    else:
        print("✅ All required packages are installed")
    
    # Create directories
    print("\n📁 Setting up directories...")
    create_directories()
    
    # Check for videos
    print("\n🎥 Checking for videos...")
    video_files = list(Path('match_directory').glob('*.mp4')) + \
                  list(Path('match_directory').glob('*.avi')) + \
                  list(Path('match_directory').glob('*.mov'))
    
    if video_files:
        print(f"✅ Found {len(video_files)} video(s) in match_directory/")
        for video in video_files:
            print(f"   - {video.name}")
    else:
        print("⚠️  No videos found in match_directory/")
        print("Please add your wrestling videos to the match_directory/ folder")
        print("Supported formats: MP4, AVI, MOV, MKV, WMV")
    
    # Setup database
    print("\n🗄️  Setting up database...")
    try:
        from database_setup import create_database
        create_database()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return
    
    # Check if clips exist
    print("\n🎬 Checking for existing clips...")
    clip_files = list(Path('unlabeled_clips').glob('*.mp4'))
    if clip_files:
        print(f"✅ Found {len(clip_files)} existing clips")
    else:
        print("⚠️  No clips found - you'll need to run motion detection")
    
    print("\n🚀 Setup Complete!")
    print("\nNext steps:")
    print("1. Add videos to match_directory/ (if not already done)")
    print("2. Run: python3 motion_detection.py")
    print("3. Run: python3 gradio_app.py")
    print("4. Open the provided URL in your browser")
    print("\nFor help, see README.md or run: python3 view_database.py")

if __name__ == "__main__":
    main() 