"""
Shared Utilities for Pitcher Labeling (Manual and Automatic)

Provides common functions used by both label_pitchers.py and auto_label_pitchers.py:
- Saving pitcher/no-pitcher labels to pitcher_labels directory
- Extracting pitcher data for JSON output
- Batch labeling frames with same track_id
- Ground truth arm side determination

Usage:
    from utils.label_utils import save_pitcher_label, save_no_pitcher_label
    
    # Save pitcher
    save_pitcher_label(video_dir, frame_name, pitcher_data, arm_side, track_id)
    
    # Save no pitcher
    save_no_pitcher_label(video_dir, frame_name)
    
    # Get frames with track_id
    frames = get_frames_with_track_id(video_dir, track_id)
"""

import cv2
import numpy as np

from . import pose_utils, crop_utils


def save_no_pitcher_label(video_dir, frame_name, output_subdir='pitcher_labels'):
    """
    Save 'no pitcher detected' label for a frame.
    
    Args:
        video_dir: Path to video directory
        frame_name: Name of the frame file (e.g., 'frame_0509.jpg')
        output_subdir: Subdirectory to save to (default: 'pitcher_labels')
    """
    pitcher_frame_name = pose_utils.format_frame_name(frame_name, 'pitcher')
    output_dir = video_dir / output_subdir / pitcher_frame_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        'frame': frame_name,
        'pitcher_detected': False,
        'pitcher_person_id': None,
        'pitcher_track_id': None
    }
    pose_utils.save_json(output_data, output_dir / 'data.json')


def extract_pitcher_output(person_data, frame_name, arm_side, track_id, clip_selection=None):
    """
    Build pitcher_labels JSON payload compatible with downstream scripts.
    
    Args:
        person_data: Person data dict from poses with keypoints, bbox, etc.
        frame_name: Name of the frame file
        arm_side: 'left' or 'right'
        track_id: YOLO track ID of the pitcher
        clip_selection: Optional dict with CLIP aggregation metrics
    
    Returns:
        Dictionary with pitcher data ready for JSON output
    """
    keypoints = np.array(person_data['keypoints'])

    if arm_side == 'right':
        shoulder_idx = pose_utils.KEYPOINT_NAMES['right_shoulder']
        elbow_idx = pose_utils.KEYPOINT_NAMES['right_elbow']
        wrist_idx = pose_utils.KEYPOINT_NAMES['right_wrist']
    else:
        shoulder_idx = pose_utils.KEYPOINT_NAMES['left_shoulder']
        elbow_idx = pose_utils.KEYPOINT_NAMES['left_elbow']
        wrist_idx = pose_utils.KEYPOINT_NAMES['left_wrist']

    shoulder = keypoints[shoulder_idx]
    elbow = keypoints[elbow_idx]
    wrist = keypoints[wrist_idx]

    shoulder_key = f'{arm_side}_shoulder'
    elbow_key = f'{arm_side}_elbow'
    wrist_key = f'{arm_side}_wrist'

    output_data = {
        'frame': frame_name,
        'pitcher_detected': True,
        'pitcher_person_id': person_data.get('person_id'),
        'pitcher_track_id': track_id,
        'bbox': person_data['bbox'],
        'keypoints': person_data['keypoints'],
        'arm_side': arm_side,
        shoulder_key: {
            'x': float(shoulder[0]),
            'y': float(shoulder[1]),
            'confidence': float(shoulder[2])
        },
        elbow_key: {
            'x': float(elbow[0]),
            'y': float(elbow[1]),
            'confidence': float(elbow[2])
        },
        wrist_key: {
            'x': float(wrist[0]),
            'y': float(wrist[1]),
            'confidence': float(wrist[2])
        }
    }
    
    # Add CLIP aggregation data if provided
    if clip_selection:
        output_data['clip_selection'] = clip_selection
    
    return output_data


def save_pitcher_label(video_dir, frame_name, pitcher_data, arm_side, track_id, clip_selection=None, output_subdir='pitcher_labels'):
    """
    Save pitcher label data.
    
    Saves both:
    - JSON file with pitcher data and keypoints
    - Cropped image of the pitcher
    
    Args:
        video_dir: Path to video directory
        frame_name: Name of the frame file
        pitcher_data: Person data dict from poses
        arm_side: 'left' or 'right' (from ground truth)
        track_id: YOLO track ID of the pitcher
        clip_selection: Optional dict with CLIP aggregation metrics (for auto-labeling)
        output_subdir: Subdirectory to save to (default: 'pitcher_labels')
    """
    pitcher_frame_name = pose_utils.format_frame_name(frame_name, 'pitcher')
    output_dir = video_dir / output_subdir / pitcher_frame_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON data
    output_data = extract_pitcher_output(pitcher_data, frame_name, arm_side, track_id, clip_selection)
    pose_utils.save_json(output_data, output_dir / 'data.json')

    release_frame_path = video_dir / 'release_frames' / frame_name
    if release_frame_path.exists():
        image = cv2.imread(str(release_frame_path))
        if image is not None:
            try:
                cropped_pitcher = crop_utils.extract_person_crop(
                    image,
                    pitcher_data,
                    padding_percent=15,
                    draw_overlay=False
                )
                output_img = output_dir / f"{pitcher_frame_name}.jpg"
                cv2.imwrite(str(output_img), cropped_pitcher)
            except Exception as e:
                print(f"Warning: Failed to save pitcher crop for {frame_name}: {e}")


def get_frames_with_track_id(video_dir, track_id):
    """
    Get all frames in video that contain a specific track_id.
    
    Args:
        video_dir: Path to video directory
        track_id: YOLO track ID to search for
    
    Returns:
        List of (frame_name, frame_path) tuples
    """
    frames_with_track = []
    release_frames = pose_utils.get_release_frames(video_dir)

    for frame_path in release_frames:
        poses_frame_name = pose_utils.format_frame_name(frame_path.name, 'yolo_poses')
        poses_json = video_dir / '_yolo_poses' / poses_frame_name / 'data.json'

        if poses_json.exists():
            poses_data = pose_utils.load_json(poses_json)
            track_ids = [p.get('track_id') for p in poses_data.get('persons', [])]

            if track_id in track_ids:
                frames_with_track.append((frame_path.name, frame_path))

    return frames_with_track


def get_arm_side(video_id, ground_truth_data, default='right'):
    """
    Determine pitcher arm side from ground truth data.
    
    Args:
        video_id: Video ID to look up
        ground_truth_data: Dict of ground truth data loaded from CSV
        default: Default arm side if not found ('right' or 'left')
    
    Returns:
        'left' or 'right'
    """
    if video_id in ground_truth_data:
        pitcher_hand = ground_truth_data[video_id].get('PitcherHand', default)
        return 'right' if str(pitcher_hand).upper() == 'R' else 'left'
    return default


def batch_label_frames_with_track_id(video_dir, track_id, pitcher_data_map, 
                                     arm_side, processed_frames, force=False, output_subdir='pitcher_labels'):
    """
    Batch label all frames with a specific track_id.
    
    Args:
        video_dir: Path to video directory
        track_id: YOLO track ID to label
        pitcher_data_map: Dict mapping frame_name -> pitcher_person_data from poses
        arm_side: 'left' or 'right' for pitcher's throwing arm
        processed_frames: Set of frames already labeled in this session (to skip)
        force: Whether to force reprocessing
        output_subdir: Subdirectory to save to (default: 'pitcher_labels')
    
    Returns:
        Tuple of (labeled_count, no_pitcher_count) - only for newly labeled frames
    """
    release_frames = pose_utils.get_release_frames(video_dir)
    labeled_count = 0
    no_pitcher_count = 0

    for frame_path in release_frames:
        pitcher_frame_name = pose_utils.format_frame_name(frame_path.name, 'pitcher')

        if pitcher_frame_name in processed_frames:
            continue

        if not force and pose_utils.check_output_exists(video_dir, pitcher_frame_name, output_subdir):
            continue

        # Check if we have pitcher data for this frame
        if frame_path.name in pitcher_data_map:
            pitcher_data = pitcher_data_map[frame_path.name]
            save_pitcher_label(video_dir, frame_path.name, pitcher_data, arm_side, track_id, output_subdir)
            labeled_count += 1
        else:
            save_no_pitcher_label(video_dir, frame_path.name, output_subdir)
            no_pitcher_count += 1

    return labeled_count, no_pitcher_count