"""
Generate Results CSV Script for Baseball Pitcher Pose Analysis

This script collects all calculated pitcher angle data from pitcher_calculations/
and generates a comprehensive CSV file with one row per frame.

Output CSV columns:
- video_id: Video identifier (directory name)
- frame_name: Frame filename
- pitcher_angle_shoulder_wrist: Calculated angle from shoulder to wrist (or N/A if not calculated)
- pitcher_angle_elbow_wrist: Calculated angle from elbow to wrist (or N/A if not calculated)
- ground_truth_angle: Ground truth angle from CSV

Usage:
    From Pirates_Arm_Angle directory:
    python scripts/generate_results_csv.py [OPTIONS]

Arguments:
    --videos-dir PATH: Optional path to baseball_vids directory (default: ~/Desktop/baseball_vids)
    --output PATH: Output CSV path (default: baseball_vids/data_analysis/results.csv)
    --force: Force regeneration even if output exists

Examples:
    python scripts/generate_results_csv.py
    python scripts/generate_results_csv.py --output /path/to/output.csv
"""

import sys
import csv
from pathlib import Path
from argparse import ArgumentParser

from utils import pose_utils


def collect_frame_data(baseball_vids_dir):
    """
    Collect all frame data from pitcher_calculations directories.

    Args:
        baseball_vids_dir: Path to baseball_vids directory

    Returns:
        List of dictionaries with frame data
    """
    frame_data_list = []

    # Get all video directories
    video_dirs = pose_utils.get_video_dirs(baseball_vids_dir)

    for video_id, video_dir in video_dirs:
        # Check for pitcher_calculations directory
        calc_dir = video_dir / "pitcher_calculations"
        if not calc_dir.exists():
            continue

        # Get all frame calculation directories
        frame_calc_dirs = sorted(calc_dir.glob("frame_*_angle*"))

        for frame_calc_dir in frame_calc_dirs:
            json_path = frame_calc_dir / "data.json"

            if not json_path.exists():
                continue

            try:
                # Load JSON data
                data = pose_utils.load_json(json_path)
                pitcher_data = data.get('pitcher_data', {})

                # Extract frame name (remove _angle and everything after it)
                frame_name = frame_calc_dir.name.split('_angle')[0]

                # Get angles based on start_joint used
                start_joint = pitcher_data.get('start_joint', 'shoulder')
                angle = pitcher_data.get('arm_angle_degrees')
                ground_truth = pitcher_data.get('ground_truth_angle_degrees')

                # Create frame data entry
                frame_entry = {
                    'video_id': video_id,
                    'frame_name': frame_name,
                    'ground_truth_angle': ground_truth if ground_truth is not None else 'N/A',
                    'pitcher_angle_shoulder_wrist': 'N/A',
                    'pitcher_angle_elbow_wrist': 'N/A'
                }

                # Set the appropriate angle based on start_joint
                if angle is not None:
                    if start_joint == 'shoulder':
                        frame_entry['pitcher_angle_shoulder_wrist'] = round(angle, 3)
                    elif start_joint == 'elbow':
                        frame_entry['pitcher_angle_elbow_wrist'] = round(angle, 3)

                frame_data_list.append(frame_entry)

            except Exception as e:
                print(f"Warning: Failed to process {json_path}: {e}")
                continue

    return frame_data_list


def merge_shoulder_elbow_data(frame_data_list):
    """
    Merge shoulder and elbow calculations for the same frames.

    Some frames may have been calculated with both shoulder and elbow joints.
    This function combines them into single rows.

    Args:
        frame_data_list: List of frame data dictionaries

    Returns:
        List of merged frame data dictionaries
    """
    # Group by (video_id, frame_name)
    frame_map = {}

    for entry in frame_data_list:
        key = (entry['video_id'], entry['frame_name'])

        if key not in frame_map:
            frame_map[key] = entry.copy()
        else:
            # Merge angles if not N/A
            existing = frame_map[key]
            if entry['pitcher_angle_shoulder_wrist'] != 'N/A':
                existing['pitcher_angle_shoulder_wrist'] = entry['pitcher_angle_shoulder_wrist']
            if entry['pitcher_angle_elbow_wrist'] != 'N/A':
                existing['pitcher_angle_elbow_wrist'] = entry['pitcher_angle_elbow_wrist']

    # Convert back to list and sort
    merged_list = list(frame_map.values())
    merged_list.sort(key=lambda x: (x['video_id'], x['frame_name']))

    return merged_list


def write_results_csv(frame_data_list, output_path):
    """
    Write frame data to CSV file.

    Args:
        frame_data_list: List of frame data dictionaries
        output_path: Path to output CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define CSV columns
    fieldnames = [
        'video_id',
        'frame_name',
        'ground_truth_angle',
        'pitcher_angle_shoulder_wrist',
        'pitcher_angle_elbow_wrist'
    ]

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for frame_data in frame_data_list:
            writer.writerow(frame_data)

    print(f"✓ Results CSV saved to: {output_path}")


def main():
    parser = ArgumentParser(
        description="Generate results CSV from pitcher calculations"
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=None,
        help="Path to baseball_vids directory (default: ~/Desktop/baseball_vids)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: baseball_vids/data_analysis/results.csv)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if output exists"
    )

    args = parser.parse_args()

    # Get baseball_vids directory
    try:
        baseball_vids_dir = pose_utils.get_baseball_vids_dir(args.videos_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not baseball_vids_dir.exists():
        print(f"Error: Directory not found: {baseball_vids_dir}")
        sys.exit(1)

    # Get output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = baseball_vids_dir / "data_analysis" / "results.csv"

    # Check if output already exists
    if output_path.exists() and not args.force:
        print(f"Output file already exists: {output_path}")
        print("Use --force to regenerate")
        sys.exit(0)

    print(f"\nBaseball videos directory: {baseball_vids_dir}")
    print(f"Output CSV: {output_path}\n")

    print("Collecting frame data from pitcher_calculations...")
    frame_data_list = collect_frame_data(baseball_vids_dir)

    if not frame_data_list:
        print("✗ No calculated frame data found!")
        print("  Make sure you've run calculate_pitcher_angles.py first.")
        sys.exit(1)

    print(f"✓ Found {len(frame_data_list)} calculated frame(s)")

    # Merge shoulder and elbow data for same frames
    print("Merging shoulder and elbow calculations...")
    merged_data = merge_shoulder_elbow_data(frame_data_list)
    print(f"✓ Merged into {len(merged_data)} unique frame(s)")

    # Write CSV
    print("Writing CSV file...")
    write_results_csv(merged_data, output_path)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total frames: {len(merged_data)}")

    # Count videos
    unique_videos = len(set(entry['video_id'] for entry in merged_data))
    print(f"Videos: {unique_videos}")

    # Count angle types
    shoulder_count = sum(1 for entry in merged_data if entry['pitcher_angle_shoulder_wrist'] != 'N/A')
    elbow_count = sum(1 for entry in merged_data if entry['pitcher_angle_elbow_wrist'] != 'N/A')

    print(f"Shoulder-wrist angles: {shoulder_count}")
    print(f"Elbow-wrist angles: {elbow_count}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()