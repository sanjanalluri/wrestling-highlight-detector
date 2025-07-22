#!/usr/bin/env python3
"""
Simple test to show basic info about live ranges from first 2 matches
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database_manager import DatabaseManager
from live_filter.chunker import SequentialChunkProcessor

def test_first_two_matches():
    #  Get live ranges from first two matches
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT r.id, r.match_id, r.start_time, r.end_time, r.duration, r.label, r.labeler_name,
                m.filename as match_filename, m.filepath as match_filepath
        FROM match_ranges r
        JOIN original_matches m ON r.match_id = m.id
        WHERE r.label = 'LIVE'
        AND r.match_id IN (SELECT id FROM original_matches ORDER BY id LIMIT 2)
        ORDER BY r.match_id, r.start_time
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
    
    conn.close()
    
    if not ranges:
        print("No live ranges found in database.")
        return
    
    # Group by match
    matches = {}
    for r in ranges:
        match_id = r['match_id']
        if match_id not in matches:
            matches[match_id] = []
        matches[match_id].append(r)
    
    print(f"Total live ranges: {len(ranges)}")
    print(f"Number of matches: {len(matches)}")
    print()
    
    for match_id, match_ranges in matches.items():
        print(f"Match {match_id}:")
        print(f"  Number of live ranges: {len(match_ranges)}")
        print(f"  Total duration: {sum(r['duration'] for r in match_ranges):.1f}s")
        print(f"  Individual range lengths:")
        for i, r in enumerate(match_ranges):
            print(f"    Range {i+1}: {r['duration']:.1f}s")
        print()

def test_chunking_first_num_ranges(num_ranges):

    #  Get live ranges from first two matches
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT r.id, r.match_id, r.start_time, r.end_time, r.duration, r.label, r.labeler_name,
                m.filename as match_filename, m.filepath as match_filepath
        FROM match_ranges r
        JOIN original_matches m ON r.match_id = m.id
        WHERE r.label = 'LIVE'
        AND r.match_id IN (SELECT id FROM original_matches ORDER BY id LIMIT 2)
        ORDER BY r.match_id, r.start_time
        LIMIT ?
    ''', (num_ranges,))

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
    
    conn.close()
    
    if len(ranges) < 2:
        print("Need at least 2 live ranges to test chunking.")
        return
    
    print(f"Testing chunking on first {num_ranges} live ranges:")
    for i in range(min(num_ranges, len(ranges))):
        print(f"Range {i+1}: {ranges[i]['duration']:.1f}s (Match {ranges[i]['match_id']})")
    print()
    # Debug: show all ranges that were selected
    print("All ranges selected:")
    for i, r in enumerate(ranges):
        print(f"  Range {i+1}: {r['duration']:.1f}s from Match {r['match_id']} (ID: {r['id']})")
    print()
    
    processor = SequentialChunkProcessor()
    chunks = processor.create_chunks_from_ranges(ranges)
    
    print(f"Created {len(chunks)} chunks:")
    
    # Calculate total duration for overlap calculation
    total_duration = sum(r['duration'] for r in ranges)
    
    for i, chunk in enumerate(chunks):
        # Check if this is the overlap chunk (last chunk with duration < 10s or different logic)
        if i == len(chunks) - 1 and len(chunks) > 1:
            # This is the overlap chunk - it goes back 10s from the end
            virtual_start = total_duration - 10.0
            virtual_end = total_duration
        else:
            # Regular consecutive chunks
            virtual_start = i * 10.0
            virtual_end = virtual_start + chunk['duration']
        
        print(f"Chunk {i+1}: Virtual timeline {virtual_start:.1f}s - {virtual_end:.1f}s")
        print(f"  Covers {len(chunk['ranges'])} range(s):")
        for j, r in enumerate(chunk['ranges']):
            # Find the original range index in the input ranges list
            original_range_idx = None
            for idx, orig_range in enumerate(ranges):
                if orig_range['id'] == r['range_id']:
                    original_range_idx = idx + 1
                    break
            print(f"    Range {original_range_idx}: {r['duration']:.1f}s from Match {r['match_id']}")
        
        # Show unique match IDs this chunk spans
        unique_matches = set(r['match_id'] for r in chunk['ranges'])
        if len(unique_matches) > 1:
            print(f"  Spans matches: {sorted(unique_matches)}")
        print()

def test_visual_data_integrity(num_ranges):
    """Test that visual data is correctly extracted from chunks"""
    
    # Get first two ranges
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT r.id, r.match_id, r.start_time, r.end_time, r.duration, r.label, r.labeler_name,
                m.filename as match_filename, m.filepath as match_filepath
        FROM match_ranges r
        JOIN original_matches m ON r.match_id = m.id
        WHERE r.label = 'LIVE'
        AND r.match_id IN (SELECT id FROM original_matches ORDER BY id LIMIT 2)
        ORDER BY r.match_id, r.start_time
        LIMIT ?
    ''', (num_ranges,))

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
    
    conn.close()
    
    if len(ranges) < 2:
        print("Need at least 2 live ranges to test visual integrity.")
        return
    
    print("Testing visual data integrity on first two ranges:")
    print(f"Range 1: {ranges[0]['duration']:.1f}s from Match {ranges[0]['match_id']}")
    print(f"Range 2: {ranges[1]['duration']:.1f}s from Match {ranges[1]['match_id']}")
    print()
    
    # Run the actual chunking script to process these ranges
    processor = SequentialChunkProcessor()
    chunks = processor.create_chunks_from_ranges(ranges)
    
    # Process chunks to extract features (this is what the real script does)
    for chunk in chunks:
        processor.extract_chunk_features(chunk)
    
    print(f"Created {len(chunks)} chunks, checking visual data integrity:")
    print()
    
    # Check the saved feature files
    for i, chunk in enumerate(chunks):
        chunk_id = chunk['chunk_id']
        feature_file = f"live_filter/features/chunk_{chunk_id}.npz"
        
        print(f"Chunk {i+1}:")
        
        if os.path.exists(feature_file):
            # Load the saved features
            data = np.load(feature_file)
            
            if 'visual_features' in data:
                visual_features = data['visual_features']
                
                # Calculate virtual timeline position for this chunk
                total_duration = sum(r['duration'] for r in ranges)
                if i == len(chunks) - 1 and len(chunks) > 1:
                    # This is the overlap chunk - it goes back 10s from the end
                    virtual_start = total_duration - 10.0
                    virtual_end = total_duration
                else:
                    # Regular consecutive chunks
                    virtual_start = i * 10.0
                    virtual_end = virtual_start + chunk['duration']
                
                print(f"  Virtual timeline: {virtual_start:.1f}s - {virtual_end:.1f}s")
                print(f"  Visual features shape: {visual_features.shape}")
                
                # Check frame dimensions and data integrity
                if len(visual_features) > 0:
                    print(f"  Number of frames: {visual_features.shape[0]}")
                    print(f"  Features per frame: {visual_features.shape[1]}")
                    
                    # Check if features have reasonable values
                    min_val = np.min(visual_features)
                    max_val = np.max(visual_features)
                    print(f"  Feature value range: {min_val:.4f} to {max_val:.4f}")
                    
                    # Check if features are different between frames
                    if visual_features.shape[0] > 1:
                        first_frame_mean = np.mean(visual_features[0])
                        last_frame_mean = np.mean(visual_features[-1])
                        frames_different = abs(first_frame_mean - last_frame_mean) > 1.0
                        print(f"  First frame mean: {first_frame_mean:.1f}")
                        print(f"  Last frame mean: {last_frame_mean:.1f}")
                        print(f"  Frames are different: {frames_different}")
                else:
                    print("  No visual features extracted!")
            else:
                print("  No visual features found in saved file!")
        else:
            print(f"  Feature file not found: {feature_file}")
        
        print()

def test_audio_data_integrity():
    """Test that audio data is correctly extracted from chunks"""
    
    # Get first two ranges
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT r.id, r.match_id, r.start_time, r.end_time, r.duration, r.label, r.labeler_name,
                m.filename as match_filename, m.filepath as match_filepath
        FROM match_ranges r
        JOIN original_matches m ON r.match_id = m.id
        WHERE r.label = 'LIVE'
        AND r.match_id IN (SELECT id FROM original_matches ORDER BY id LIMIT 2)
        ORDER BY r.match_id, r.start_time
        LIMIT 2
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
    
    conn.close()
    
    if len(ranges) < 2:
        print("Need at least 2 live ranges to test audio integrity.")
        return
    
    print("Testing audio data integrity on first two ranges:")
    print(f"Range 1: {ranges[0]['duration']:.1f}s from Match {ranges[0]['match_id']}")
    print(f"Range 2: {ranges[1]['duration']:.1f}s from Match {ranges[1]['match_id']}")
    print()
    
    # Run the actual chunking script to process these ranges
    processor = SequentialChunkProcessor()
    chunks = processor.create_chunks_from_ranges(ranges)
    
    # Process chunks to extract features (this is what the real script does)
    for chunk in chunks:
        processor.extract_chunk_features(chunk)
    
    print(f"Created {len(chunks)} chunks, checking saved audio features:")
    print()
    
    # Check the saved feature files
    for i, chunk in enumerate(chunks):
        chunk_id = chunk['chunk_id']
        feature_file = f"live_filter/features/chunk_{chunk_id}.npz"
        
        print(f"Chunk {i+1} (ID: {chunk_id}):")
        
        if os.path.exists(feature_file):
            # Load the saved features
            data = np.load(feature_file)
            
            if 'audio_features' in data:
                audio_features = data['audio_features']
                print(f"  Audio features array: {audio_features}")
                print(f"  Audio features mean: {np.mean(audio_features):.4f}")
                print(f"  Audio features std: {np.std(audio_features):.4f}")
                print(f"  Audio features range: {np.min(audio_features):.4f} to {np.max(audio_features):.4f}")
                
                # Check if features are reasonable
                has_features = len(audio_features) > 0
                reasonable_range = np.all((audio_features >= -1) & (audio_features <= 1))
                print(f"  Has audio features: {has_features}")
                print(f"  Reasonable range: {reasonable_range}")
            else:
                print("  No audio features found in saved file!")
        else:
            print(f"  Feature file not found: {feature_file}")
        
        print()

def test_visual_comparison():
    """Compare local visual processing vs chunker script processing"""
    
    # Get first two ranges
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT r.id, r.match_id, r.start_time, r.end_time, r.duration, r.label, r.labeler_name,
                m.filename as match_filename, m.filepath as match_filepath
        FROM match_ranges r
        JOIN original_matches m ON r.match_id = m.id
        WHERE r.label = 'LIVE'
        AND r.match_id IN (SELECT id FROM original_matches ORDER BY id LIMIT 2)
        ORDER BY r.match_id, r.start_time
        LIMIT 2
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
    
    conn.close()
    
    if len(ranges) < 2:
        print("Need at least 2 live ranges to test visual comparison.")
        return
    
    print("Comparing local vs chunker visual processing:")
    print(f"Range 1: {ranges[0]['duration']:.1f}s from Match {ranges[0]['match_id']}")
    print(f"Range 2: {ranges[1]['duration']:.1f}s from Match {ranges[1]['match_id']}")
    print()
    
    processor = SequentialChunkProcessor()
    chunks = processor.create_chunks_from_ranges(ranges)
    
    print(f"Created {len(chunks)} chunks, comparing processing methods:")
    print()
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        
        # Calculate virtual timeline position
        total_duration = sum(r['duration'] for r in ranges)
        if i == len(chunks) - 1 and len(chunks) > 1:
            virtual_start = total_duration - 10.0
            virtual_end = total_duration
        else:
            virtual_start = i * 10.0
            virtual_end = virtual_start + chunk['duration']
        
        print(f"  Virtual timeline: {virtual_start:.1f}s - {virtual_end:.1f}s")
        
        # METHOD 1: Local processing (like original test)
        print("  LOCAL PROCESSING:")
        frames = []
        for range_info in chunk['ranges']:
            range_frames = processor.extract_frames_from_range(
                range_info['match_filepath'],
                range_info['start_time'],
                range_info['end_time']
            )
            frames.extend(range_frames)
        
        print(f"    Total frames extracted: {len(frames)}")
        
        if frames:
            # Check frame differences using all frames
            if len(frames) > 1:
                first_frame_mean = np.mean(frames[0])
                last_frame_mean = np.mean(frames[-1])
                frames_different_local = abs(first_frame_mean - last_frame_mean) > 1.0
                print(f"    First frame mean: {first_frame_mean:.1f}")
                print(f"    Last frame mean: {last_frame_mean:.1f}")
                print(f"    Frames are different: {frames_different_local}")
            
            # Show frame sampling that chunker would use
            target_frames = 8
            if len(frames) >= target_frames:
                indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
                sampled_frames = [frames[i] for i in indices]
                print(f"    Chunker would sample frames: {indices}")
                
                if len(sampled_frames) > 1:
                    first_sampled_mean = np.mean(sampled_frames[0])
                    last_sampled_mean = np.mean(sampled_frames[-1])
                    sampled_different = abs(first_sampled_mean - last_sampled_mean) > 1.0
                    print(f"    First sampled frame mean: {first_sampled_mean:.1f}")
                    print(f"    Last sampled frame mean: {last_sampled_mean:.1f}")
                    print(f"    Sampled frames are different: {sampled_different}")
        
        # METHOD 2: Chunker script processing
        print("  CHUNKER SCRIPT:")
        chunk_id = chunk['chunk_id']
        feature_file = f"live_filter/features/chunk_{chunk_id}.npz"
        
        if os.path.exists(feature_file):
            data = np.load(feature_file)
            if 'visual_features' in data:
                visual_features = data['visual_features']
                print(f"    Visual features shape: {visual_features.shape}")
                
                if visual_features.shape[0] > 1:
                    first_frame_mean = np.mean(visual_features[0])
                    last_frame_mean = np.mean(visual_features[-1])
                    frames_different_chunker = abs(first_frame_mean - last_frame_mean) > 0.5
                    print(f"    First frame mean: {first_frame_mean:.1f}")
                    print(f"    Last frame mean: {last_frame_mean:.1f}")
                    print(f"    Frames are different: {frames_different_chunker}")
        else:
            print(f"    Feature file not found: {feature_file}")
        
        print()

if __name__ == "__main__":
    # test_first_two_matches()
    print("=" * 50)
    # test_chunking_first_num_ranges(22)
    print("=" * 50)
    # test_visual_comparison()
    test_audio_data_integrity()