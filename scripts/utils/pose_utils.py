"""
Shared utility functions for baseball pitcher pose analysis pipeline.

This module provides:
- Model initialization helpers for MMDetection and MMPose
- Path and directory management utilities
- JSON I/O functions
- Keypoint extraction and manipulation
- Angle calculation functions
- CSV loading for ground truth data
"""

import os
import json
import math
import csv
from pathlib import Path
import warnings

# try:
#     from mmpose.apis import init_pose_model
#     from mmpose.datasets import DatasetInfo
#     from mmdet.apis import init_detector
#
#     MMPOSE_AVAILABLE = True
# except ImportError:
#     MMPOSE_AVAILABLE = False
#     warnings.warn("MMPose/MMDet not available. Some functions will not work.")

# COCO-WholeBody Keypoint Indices (133 total keypoints)
KEYPOINT_NAMES = {
    # Body (0-16)
    'nose': 0,
    'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
    # Feet (17-22) - simplified
    # Face (23-90) - 68 facial landmarks
    # Left Hand (91-111) - 21 keypoints
    # Right Hand (112-132) - 21 keypoints
}


def get_desktop_path():
    """Get path to user's Desktop directory."""
    home = Path.home()
    desktop = home / "Desktop"

    if not desktop.exists():
        raise FileNotFoundError(f"Desktop directory not found at: {desktop}")

    return desktop


def get_baseball_vids_dir(custom_path=None):
    """
    Get path to baseball_vids directory.

    Args:
        custom_path: Optional custom path to baseball_vids directory

    Returns:
        Path object pointing to baseball_vids directory
    """
    if custom_path:
        return Path(custom_path)

    desktop = get_desktop_path()
    return desktop / "baseball_vids"


def get_video_dirs(baseball_vids_dir):
    """
    Get all video directories (subdirectories with all_frames and release_frames).

    Args:
        baseball_vids_dir: Path to baseball_vids directory

    Returns:
        List of (video_id, video_dir_path) tuples
    """
    baseball_vids_dir = Path(baseball_vids_dir)
    video_dirs = []

    for item in baseball_vids_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it has release_frames subdirectory
            release_frames = item / "release_frames"
            if release_frames.exists() and release_frames.is_dir():
                video_dirs.append((item.name, item))

    return sorted(video_dirs)


def get_release_frames(video_dir):
    """
    Get all frame files from release_frames directory.

    Args:
        video_dir: Path to video directory

    Returns:
        List of frame file paths, sorted
    """
    release_frames_dir = Path(video_dir) / "release_frames"

    if not release_frames_dir.exists():
        return []

    # Get all jpg/png files
    frames = list(release_frames_dir.glob("frame_*.jpg"))
    frames.extend(list(release_frames_dir.glob("frame_*.png")))

    return sorted(frames)


def check_output_exists(video_dir, frame_name, stage):
    """
    Check if output already exists for a given frame and processing stage.

    Args:
        video_dir: Path to video directory
        frame_name: Name of the frame file (without extension)
        stage: Processing stage - 'poses', 'pitcher_labels', or 'pitcher_calculations'

    Returns:
        Boolean indicating if output exists
    """
    stage_dir = Path(video_dir) / stage / frame_name

    if not stage_dir.exists():
        return False

    # Check if data.json exists
    json_file = stage_dir / "data.json"
    return json_file.exists()


def save_json(data, output_path):
    """
    Save data to JSON file.

    Args:
        data: Dictionary to save
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(json_path):
    """
    Load data from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary loaded from JSON
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def load_ground_truth_csv(csv_path):
    """
    Load ground truth data from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        Dictionary mapping PitchId to {FileName, PitcherHand, ArmAngle}
    """
    ground_truth = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pitch_id = row['PitchId']
            ground_truth[pitch_id] = {
                'FileName': row['FileName'],
                'PitcherHand': row['PitcherHand'],
                'ArmAngle': float(row['ArmAngle'])
            }

    return ground_truth


def calculate_angle(point1, point2, x_multiplier=1):
    """
    Calculate angle between two points with respect to x-axis.

    Args:
        point1: Starting point (shoulder or elbow) as (x, y) tuple
        point2: End point (wrist) as (x, y) tuple
        x_multiplier: 1 for right arm, -1 for left arm

    Returns:
        Tuple of (angle_degrees, magnitude)
    """
    # Vector from point1 to point2
    vector = (point2[0] - point1[0], point2[1] - point1[1])

    # Calculate angle using dot product with respect to x-axis
    dot_product = vector[0] * x_multiplier + vector[1] * 0
    magnitude = math.sqrt(vector[0] ** 2 + vector[1] ** 2)

    if magnitude == 0:
        return None, 0

    cos_angle = dot_product / magnitude
    cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)

    return angle_deg, magnitude


def extract_keypoints_for_person(pose_result, arm_side, start_joint='shoulder'):
    """
    Extract relevant keypoints for a detected person.

    Args:
        pose_result: Pose result dictionary from inference
        arm_side: 'left' or 'right'
        start_joint: 'shoulder' or 'elbow'

    Returns:
        Dictionary with extracted keypoint data
    """
    keypoints = pose_result['keypoints']

    # Determine keypoint indices based on arm side
    if arm_side == 'right':
        shoulder_idx = KEYPOINT_NAMES['right_shoulder']
        elbow_idx = KEYPOINT_NAMES['right_elbow']
        wrist_idx = KEYPOINT_NAMES['right_wrist']
        hip_idx = KEYPOINT_NAMES['right_hip']
        x_multiplier = 1
    else:  # left
        shoulder_idx = KEYPOINT_NAMES['left_shoulder']
        elbow_idx = KEYPOINT_NAMES['left_elbow']
        wrist_idx = KEYPOINT_NAMES['left_wrist']
        hip_idx = KEYPOINT_NAMES['left_hip']
        x_multiplier = -1

    # Extract keypoint data
    shoulder = keypoints[shoulder_idx]
    elbow = keypoints[elbow_idx]
    wrist = keypoints[wrist_idx]
    hip = keypoints[hip_idx]

    # Determine start point based on joint selection
    if start_joint == 'shoulder':
        start_point = shoulder
        start_name = f'{arm_side}_shoulder'
    else:  # elbow
        start_point = elbow
        start_name = f'{arm_side}_elbow'

    # Calculate angle
    angle, magnitude = calculate_angle(
        start_point[:2],
        wrist[:2],
        x_multiplier
    )

    return {
        f'{arm_side}_shoulder': {
            'x': float(shoulder[0]),
            'y': float(shoulder[1]),
            'confidence': float(shoulder[2])
        },
        f'{arm_side}_elbow': {
            'x': float(elbow[0]),
            'y': float(elbow[1]),
            'confidence': float(elbow[2])
        },
        f'{arm_side}_wrist': {
            'x': float(wrist[0]),
            'y': float(wrist[1]),
            'confidence': float(wrist[2])
        },
        f'{arm_side}_hip': {
            'x': float(hip[0]),
            'y': float(hip[1]),
            'confidence': float(hip[2])
        },
        'start_joint': start_joint,
        'start_point': start_name,
        'arm_angle_degrees': float(angle) if angle is not None else None,
        'arm_magnitude': float(magnitude),
        'arm_side': arm_side,
        'x_multiplier': x_multiplier
    }


def init_models(det_config, det_checkpoint, pose_config, pose_checkpoint, device='cpu'):
    """
    Initialize MMDetection and MMPose models.

    Note: MMPose/MMDet modules are imported here (lazy loading) to avoid
    triggering apex warnings when pose_utils is imported by scripts that
    don't need model inference.

    Args:
        det_config: Path to detection config file
        det_checkpoint: Path to detection checkpoint
        pose_config: Path to pose config file
        pose_checkpoint: Path or URL to pose checkpoint
        device: Device to use ('cpu' or 'cuda:0')

    Returns:
        Tuple of (det_model, pose_model, dataset_info)
    """
    # Lazy import - only load mmpose/mmdet when actually needed
    try:
        from mmpose.apis import init_pose_model
        from mmpose.datasets import DatasetInfo
        from mmdet.apis import init_detector
    except ImportError:
        raise ImportError(
            "MMPose and MMDet must be installed to use model initialization. "
            "Install with: pip install mmpose mmdet"
        )

    # Initialize detection model
    det_model = init_detector(det_config, det_checkpoint, device=device.lower())

    # Initialize pose model
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device.lower())

    # Get dataset info
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config. '
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning
        )
    else:
        dataset_info = DatasetInfo(dataset_info)

    return det_model, pose_model, dataset_info


def format_frame_name(original_name, suffix):
    """
    Format frame name with appropriate suffix.

    Args:
        original_name: Original frame filename (e.g., 'frame_0001.jpg')
        suffix: Suffix to add ('poses', 'pitcher', 'angle')

    Returns:
        Formatted name (e.g., 'frame_0001_poses')
    """
    # Remove extension
    name_without_ext = Path(original_name).stem

    # Add suffix
    return f"{name_without_ext}_{suffix}"