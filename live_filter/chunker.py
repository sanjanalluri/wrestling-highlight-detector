#!/usr/bin/env python3
"""
Sequential Chunk Processor for Live Filter Training
Simple approach: extract 10-second chunks by taking what's needed from each range sequentially
Now with audio and sequential processing
"""

import os
import sys
import cv2
import numpy as np
import librosa
import subprocess
import tempfile
import json
from datetime import datetime
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database_manager import DatabaseManager

# Configuration
CHUNK_SIZE = 10.0  # seconds
CHUNK_OVERLAP = 2.0  # seconds overlap between chunks
FEATURES_DIR = "live_filter/features"
TEMP_DIR = "live_filter/temp_chunks"

class SequentialChunkProcessor:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.chunk_counter = 0  # Global counter for unique chunk IDs
        os.makedirs(FEATURES_DIR, exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
        
    def get_unprocessed_ranges(self) -> List[Dict]:
        """Get ranges that haven't been processed into chunks yet"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        # Get all ranges
        cursor.execute('''
            SELECT r.id, r.match_id, r.start_time, r.end_time, r.duration, r.label, r.labeler_name,
                   m.filename as match_filename, m.filepath as match_filepath
            FROM match_ranges r
            JOIN original_matches m ON r.match_id = m.id
            ORDER BY r.label, r.match_id, r.start_time
        ''')
        
        ranges = []
        for row in cursor.fetchall():
            ranges.append({
                'id': row[0],
                'match_id': row[1],
                'start_time': row[2],
                'end_time': row[3],
                'duration': row[4],
                'label': row[5],
                'labeler_name': row[6],
                'match_filename': row[7],
                'match_filepath': row[8]
            })
        
        # Get processed range IDs
        cursor.execute('SELECT range_ids FROM training_chunks')
        processed_range_ids = set()
        for row in cursor.fetchall():
            if row[0]:
                range_ids = json.loads(row[0])
                processed_range_ids.update(range_ids)
        
        # Filter out processed ranges
        unprocessed = [r for r in ranges if r['id'] not in processed_range_ids]
        conn.close()
        
        return unprocessed
    
    def create_chunks_from_ranges(self, ranges: List[Dict]) -> List[Dict]:
        """
        Create 10-second chunks using stacking window logic
        Handles all edge cases:
        - Ranges bigger than 10 seconds
        - Ranges exactly 10 seconds  
        - Ranges smaller than 10 seconds
        - Transitions between ranges within same match
        - Transitions between different matches
        - Only ONE overlap chunk at the very end (if needed)
        """
        if not ranges:
            return []
        
        # Calculate total duration
        total_duration = sum(r['duration'] for r in ranges)
        chunks = []
        current_range_idx = 0
        current_range_offset = 0.0  # How much time we've used from current range
        
        print(f"  Creating chunks from {len(ranges)} ranges (total duration: {total_duration:.1f}s)...")
        
        # Calculate how many full 10-second chunks we can create
        num_full_chunks = int(total_duration // CHUNK_SIZE)
        remaining_time = total_duration - (num_full_chunks * CHUNK_SIZE)
        
        # Create full 10-second chunks
        for chunk_idx in range(num_full_chunks):
            chunk_ranges = []
            remaining_chunk_time = CHUNK_SIZE
            
            # Fill this chunk
            while remaining_chunk_time > 0 and current_range_idx < len(ranges):
                current_range = ranges[current_range_idx]
                available_in_range = current_range['duration'] - current_range_offset
                
                # How much we can take from this range
                time_to_take = min(remaining_chunk_time, available_in_range)
                
                # Calculate actual start/end times within this range
                range_start = current_range['start_time'] + current_range_offset
                range_end = range_start + time_to_take
                
                chunk_ranges.append({
                    'range_id': current_range['id'],
                    'match_id': current_range['match_id'],
                    'match_filepath': current_range['match_filepath'],
                    'start_time': range_start,
                    'end_time': range_end,
                    'duration': time_to_take
                })
                
                remaining_chunk_time -= time_to_take
                current_range_offset += time_to_take
                
                # If we used all of this range, move to next
                if current_range_offset >= current_range['duration']:
                    current_range_idx += 1
                    current_range_offset = 0.0
                # Otherwise, we still have time left in this range for next chunk
                else:
                    break
            
            # Create chunk info with unique ID
            chunk_info = {
                'chunk_id': self.chunk_counter,
                'label': ranges[0]['label'],
                'duration': CHUNK_SIZE - remaining_chunk_time,
                'ranges': chunk_ranges,
                'range_ids': [r['range_id'] for r in chunk_ranges]
            }
            chunks.append(chunk_info)
            self.chunk_counter += 1
        
        # Now check if we need to create an overlap chunk
        # Only create overlap if we have more than 1 chunk and there's remaining time less than 10 seconds
        if len(chunks) > 1 and 0 < remaining_time < CHUNK_SIZE:
            # Create overlap chunk that goes back 10s from the end
            overlap_chunk_ranges = []
            overlap_remaining = CHUNK_SIZE
            
            # Find starting position for overlap (10s back from end)
            overlap_start_time = total_duration - CHUNK_SIZE
            
            # Find which range and offset this corresponds to
            temp_duration = 0.0
            overlap_range_idx = 0
            overlap_offset = 0.0
            
            for i, range_data in enumerate(ranges):
                if temp_duration + range_data['duration'] > overlap_start_time:
                    overlap_range_idx = i
                    overlap_offset = overlap_start_time - temp_duration
                    break
                temp_duration += range_data['duration']
            
            # Build overlap chunk
            while overlap_remaining > 0 and overlap_range_idx < len(ranges):
                current_range = ranges[overlap_range_idx]
                available_in_range = current_range['duration'] - overlap_offset
                
                time_to_take = min(overlap_remaining, available_in_range)
                range_start = current_range['start_time'] + overlap_offset
                range_end = range_start + time_to_take
                
                overlap_chunk_ranges.append({
                    'range_id': current_range['id'],
                    'match_id': current_range['match_id'],
                    'match_filepath': current_range['match_filepath'],
                    'start_time': range_start,
                    'end_time': range_end,
                    'duration': time_to_take
                })
                
                overlap_remaining -= time_to_take
                overlap_offset += time_to_take
                
                if overlap_offset >= current_range['duration']:
                    overlap_range_idx += 1
                    overlap_offset = 0.0
                else:
                    break
            
            # Create overlap chunk with unique ID
            overlap_chunk_info = {
                'chunk_id': self.chunk_counter,
                'label': ranges[0]['label'],
                'duration': CHUNK_SIZE - overlap_remaining,
                'ranges': overlap_chunk_ranges,
                'range_ids': [r['range_id'] for r in overlap_chunk_ranges]
            }
            
            chunks.append(overlap_chunk_info)
            self.chunk_counter += 1
        
        print(f"  Created {len(chunks)} chunks")
        return chunks
    
    def extract_chunk_features(self, chunk_info: Dict) -> Dict:
        """Extract visual and audio features from a chunk"""
        try:
            frames = []
            audio_segments = []
            
            # Extract frames and audio from all ranges in this chunk
            for range_info in chunk_info['ranges']:
                # Extract video frames
                range_frames = self.extract_frames_from_range(
                    range_info['match_filepath'],
                    range_info['start_time'],
                    range_info['end_time']
                )
                frames.extend(range_frames)
                
                # Extract audio
                range_audio = self.extract_audio_from_range(
                    range_info['match_filepath'],
                    range_info['start_time'],
                    range_info['end_time']
                )
                if len(range_audio) > 0:
                    audio_segments.append(range_audio)
            
            # Sample frames evenly across the chunk
            target_frames = 8
            if len(frames) >= target_frames:
                indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
                sampled_frames = [frames[i] for i in indices]
            else:
                # Pad with last frame if not enough frames
                sampled_frames = frames + [frames[-1]] * (target_frames - len(frames))
            
            # Combine audio segments (fix the concatenation issue)
            if audio_segments:
                # Filter out empty segments and ensure all have same dimensions
                valid_audio_segments = [seg for seg in audio_segments if len(seg) > 0 and seg.ndim == 1]
                if valid_audio_segments:
                    combined_audio = np.concatenate(valid_audio_segments)
                else:
                    # Return default audio array (1 second of silence at 22050 Hz)
                    combined_audio = np.zeros(22050, dtype=np.float32)
            else:
                # Return default audio array (1 second of silence at 22050 Hz)
                combined_audio = np.zeros(22050, dtype=np.float32)
            
            # Extract features
            visual_features = self.extract_visual_features(sampled_frames)
            audio_features = self.extract_audio_features(combined_audio)
            
            # Save features
            chunk_id = chunk_info['chunk_id']
            features_path = os.path.join(FEATURES_DIR, f"chunk_{chunk_id}.npz")
            
            np.savez_compressed(
                features_path,
                visual_features=visual_features,
                audio_features=audio_features,
                label=chunk_info['label'],
                chunk_info=chunk_info
            )
            
            return {
                'chunk_id': chunk_id,
                'metadata_path': features_path,
                'label': chunk_info['label'],
                'duration': chunk_info['duration'],
                'range_ids': chunk_info['range_ids']
            }
            
        except Exception as e:
            print(f"Error processing chunk {chunk_info['chunk_id']}: {str(e)}")
            return None
    
    def extract_frames_from_range(self, video_path: str, start_time: float, end_time: float) -> List[np.ndarray]:
        """Extract frames from a specific time range in a video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                break
        
        cap.release()
        return frames
    
    def extract_audio_from_range(self, video_path: str, start_time: float, end_time: float) -> np.ndarray:
        """Extract audio from a specific time range in a video"""
        try:
            # Create unique temp file name
            temp_audio = os.path.join(TEMP_DIR, f"temp_audio_{start_time}_{end_time}_{os.getpid()}.wav")
            
            # Use ffmpeg to extract audio segment
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '22050', '-ac', '1',  # Mono audio
                '-y', temp_audio
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFmpeg failed for {video_path} {start_time}-{end_time}")
                return np.array([], dtype=np.float32)
            
            # Load audio with librosa
            audio, sr = librosa.load(temp_audio, sr=22050)
            
            # Clean up temp file
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return audio
            
        except Exception as e:
            print(f"Error extracting audio from {video_path}: {str(e)}")
            return np.array([], dtype=np.float32)
    
    def extract_visual_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract visual features from frames"""
        features = []
        
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Basic features: mean RGB, std RGB, histogram
            mean_rgb = np.mean(frame_rgb, axis=(0, 1))
            std_rgb = np.std(frame_rgb, axis=(0, 1))
            
            # Simple histogram features
            hist_r = cv2.calcHist([frame_rgb], [0], None, [16], [0, 256]).flatten()
            hist_g = cv2.calcHist([frame_rgb], [1], None, [16], [0, 256]).flatten()
            hist_b = cv2.calcHist([frame_rgb], [2], None, [16], [0, 256]).flatten()
            
            # Normalize histograms
            hist_r = hist_r / (hist_r.sum() + 1e-8)
            hist_g = hist_g / (hist_g.sum() + 1e-8)
            hist_b = hist_b / (hist_b.sum() + 1e-8)
            
            frame_features = np.concatenate([
                mean_rgb, std_rgb, hist_r, hist_g, hist_b
            ])
            
            features.append(frame_features)
        
        return np.array(features)
    
    def extract_audio_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract audio features"""
        if len(audio) == 0:
            return np.zeros(29)  # Default feature size (13+13+1+1+1 = 29)
        
        try:
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=22050)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=22050)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            
            # Normalize MFCC features to -1 to 1 range
            mfcc_means = np.mean(mfcc, axis=1)
            mfcc_stds = np.std(mfcc, axis=1)
            
            # Normalize MFCC means and stds using min-max scaling
            mfcc_means_norm = np.clip((mfcc_means - np.mean(mfcc_means)) / (np.std(mfcc_means) + 1e-8), -1, 1)
            mfcc_stds_norm = np.clip((mfcc_stds - np.mean(mfcc_stds)) / (np.std(mfcc_stds) + 1e-8), -1, 1)
            
            # Normalize spectral features to 0-1 range (percentages)
            max_freq = 22050 / 2  # Nyquist frequency
            spectral_centroid_norm = np.mean(spectral_centroid) / max_freq
            spectral_rolloff_norm = np.mean(spectral_rolloff) / max_freq
            
            # Zero crossing rate is already normalized (0-1)
            zcr_norm = np.mean(zcr)
            
            # Combine features - ensure all are 1D arrays
            features = np.concatenate([
                mfcc_means_norm,
                mfcc_stds_norm,
                np.array([spectral_centroid_norm]),  # Convert 0D to 1D
                np.array([spectral_rolloff_norm]),   # Convert 0D to 1D
                np.array([zcr_norm])                 # Convert 0D to 1D
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {str(e)}")
            return np.zeros(29)
    
    def save_chunk_metadata(self, chunk_metadata: Dict):
        """Save chunk metadata to database"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_chunks 
            (chunk_id, label, metadata_path, duration, range_ids, created_date)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            chunk_metadata['chunk_id'],
            chunk_metadata['label'],
            chunk_metadata['metadata_path'],
            chunk_metadata['duration'],
            json.dumps(chunk_metadata['range_ids']),
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
    
    def process_all_ranges(self):
        """Process all unprocessed ranges into chunks"""
        print("🔍 Finding unprocessed ranges...")
        unprocessed_ranges = self.get_unprocessed_ranges()
        
        if not unprocessed_ranges:
            print("✅ All ranges already processed!")
            return
        
        print(f"📊 Found {len(unprocessed_ranges)} unprocessed ranges")
        
        # Group by label
        live_ranges = [r for r in unprocessed_ranges if r['label'] == 'LIVE']
        not_live_ranges = [r for r in unprocessed_ranges if r['label'] == 'NOT LIVE']
        
        print(f"  LIVE ranges: {len(live_ranges)}")
        print(f"  NOT LIVE ranges: {len(not_live_ranges)}")
        
        all_chunks = []
        
        # Process LIVE ranges
        if live_ranges:
            print(f"\n🎬 Processing {len(live_ranges)} LIVE ranges...")
            live_chunks = self.create_chunks_from_ranges(live_ranges)
            all_chunks.extend(live_chunks)
            print(f"  Created {len(live_chunks)} LIVE chunks")
        
        # Process NOT LIVE ranges
        if not_live_ranges:
            print(f"\n⏸️  Processing {len(not_live_ranges)} NOT LIVE ranges...")
            not_live_chunks = self.create_chunks_from_ranges(not_live_ranges)
            all_chunks.extend(not_live_chunks)
            print(f"  Created {len(not_live_chunks)} NOT LIVE chunks")
        
        print(f"\n🚀 Processing {len(all_chunks)} total chunks...")
        
        # Process chunks sequentially
        successful_chunks = 0
        for chunk_info in all_chunks:
            try:
                metadata = self.extract_chunk_features(chunk_info)
                if metadata:
                    self.save_chunk_metadata(metadata)
                    successful_chunks += 1
                    print(f"✅ Processed chunk {chunk_info['chunk_id']} ({successful_chunks}/{len(all_chunks)})")
                else:
                    print(f"❌ Failed to process chunk {chunk_info['chunk_id']}")
            except Exception as e:
                print(f"❌ Failed to process chunk {chunk_info['chunk_id']}: {str(e)}")
        
        print(f"\n🎉 Processing complete! {successful_chunks}/{len(all_chunks)} chunks processed successfully")
        
        # Clean up temp directory
        self.cleanup_temp_files()
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for file in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("🧹 Cleaned up temporary files")
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {str(e)}")

if __name__ == "__main__":
    processor = SequentialChunkProcessor()
    processor.process_all_ranges() 