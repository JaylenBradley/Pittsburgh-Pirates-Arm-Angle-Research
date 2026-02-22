"""
Calculate Pitcher Angles Script for Baseball Pitcher Pose Analysis

This script calculates arm angles from labeled pitcher data and compares with ground truth.
For each frame in pitcher_labels/, it:
1. Loads the pitcher-specific data with keypoints
2. Calculates the arm angle using specified joints (shoulder/elbow to wrist)
3. Retrieves ground truth angle from CSV using video ID
4. Computes error metrics
5. Generates visualization image with angle overlays
6. Saves results to pitcher_calculations/FRAME_NAME/

Usage:
    From Pirates_Arm_Angle directory:
    python scripts/calculate_pitcher_angles.py [OPTIONS]

Arguments:
    --videos-dir PATH: Optional path to baseball_vids directory (default: ~/Desktop/baseball_vids)
    --csv PATH: Path to ground truth CSV file (default: baseball_vids/arm_angles_high_speed.csv)
    --start-joint shoulder|elbow: Joint to start angle measurement from (default: shoulder)
    --force: Force reprocessing of already-calculated frames

Examples:
    # Calculate shoulder-to-wrist angles (default)
    python scripts/calculate_pitcher_angles.py

    # Calculate elbow-to-wrist angles
    python scripts/calculate_pitcher_angles.py --start-joint elbow

    # Force recalculation
    python scripts/calculate_pitcher_angles.py --force
"""

import sys
import os
from pathlib import Path
from argparse import ArgumentParser
import cv2
import numpy as np
import math

# Import our utilities
import pose_utils


def draw_angle_on_image(image, pitcher_data, ground_truth_angle, start_joint='shoulder'):
    """
    Draw arm angle visualization on image.

    Args:
        image: Image array
        pitcher_data: Dictionary with pitcher keypoints and angle data
        ground_truth_angle: Ground truth angle in degrees
        start_joint: 'shoulder' or 'elbow' - which joint to start from

    Returns:
        Image with angle visualization
    """
    img = image.copy()
    arm_side = pitcher_data['arm_side']

    # Get keypoint data
    shoulder_key = f'{arm_side}_shoulder'
    elbow_key = f'{arm_side}_elbow'
    wrist_key = f'{arm_side}_wrist'

    shoulder = pitcher_data[shoulder_key]
    elbow = pitcher_data[elbow_key]
    wrist = pitcher_data[wrist_key]

    # Determine start and end points based on joint selection
    if start_joint == 'shoulder':
        start_pt = (int(shoulder['x']), int(shoulder['y']))
        end_pt = (int(wrist['x']), int(wrist['y']))
        joint_label = "Shoulder"
    else:  # elbow
        start_pt = (int(elbow['x']), int(elbow['y']))
        end_pt = (int(wrist['x']), int(wrist['y']))
        joint_label = "Elbow"

    # Draw skeleton points
    cv2.circle(img, (int(shoulder['x']), int(shoulder['y'])), 8, (0, 255, 0), -1)  # Green
    cv2.circle(img, (int(elbow['x']), int(elbow['y'])), 8, (255, 255, 0), -1)  # Cyan
    cv2.circle(img, (int(wrist['x']), int(wrist['y'])), 8, (0, 0, 255), -1)  # Red

    # Draw arm line
    cv2.line(img, start_pt, end_pt, (255, 0, 255), 3)  # Magenta line

    # Draw reference horizontal line from start point
    ref_length = 150
    if arm_side == 'right':
        ref_end = (start_pt[0] + ref_length, start_pt[1])
    else:  # left
        ref_end = (start_pt[0] - ref_length, start_pt[1])

    cv2.line(img, start_pt, ref_end, (255, 255, 255), 2)  # White reference line

    # Add angle arc
    angle_rad = math.radians(pitcher_data['arm_angle_degrees'])
    arc_radius = 80

    if arm_side == 'right':
        start_angle_deg = 0
        end_angle_deg = -pitcher_data['arm_angle_degrees']
    else:  # left
        start_angle_deg = 180
        end_angle_deg = 180 - pitcher_data['arm_angle_degrees']

    cv2.ellipse(img, start_pt, (arc_radius, arc_radius), 0,
                min(start_angle_deg, end_angle_deg),
                max(start_angle_deg, end_angle_deg),
                (0, 255, 255), 2)

    # Add text annotations
    predicted_angle = pitcher_data['arm_angle_degrees']
    error = abs(predicted_angle - ground_truth_angle)

    # Position text in upper portion of image
    y_offset = 40
    cv2.putText(img, f"Ground Truth: {ground_truth_angle:.2f}",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.putText(img, f"Predicted ({joint_label}-Wrist): {predicted_angle:.2f}",
                (20, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

    cv2.putText(img, f"Error: {error:.2f}",
                (20, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.putText(img, f"Arm: {arm_side.upper()}",
                (20, y_offset + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return img


def calculate_frame_angle(frame_path, video_dir, video_id, ground_truth_data,
                          start_joint='shoulder', force=False):
    """
    Calculate angle for a single frame using pitcher label data.

    Args:
        frame_path: Path to the original frame in release_frames
        video_dir: Path to video directory
        video_id: Video ID (used to look up ground truth)
        ground_truth_data: Dictionary with ground truth data from CSV
        start_joint: 'shoulder' or 'elbow' - which joint to calculate from
        force: If True, reprocess even if output exists

    Returns:
        Tuple of (success: bool, message: str)
    """
    frame_name = frame_path.stem  # e.g., 'frame_0001'

    # Check if already processed
    if not force and pose_utils.check_output_exists(video_dir, frame_name, 'pitcher_calculations'):
        return True, "Already calculated"

    # Build paths
    pitcher_label_dir = video_dir / "pitcher_labels" / f"{frame_name}_pitcher"
    pitcher_json = pitcher_label_dir / "data.json"

    if not pitcher_json.exists():
        return False, "Pitcher label data not found (run label_pitchers.py first)"

    # Load pitcher data
    try:
        pitcher_data = pose_utils.load_json(pitcher_json)
    except Exception as e:
        return False, f"Failed to load pitcher data: {str(e)}"

    # Check if pitcher was detected
    if pitcher_data.get('pitcher_detected') == False:
        return False, "No pitcher detected in frame (skipping)"

    # Get ground truth for this video
    if video_id not in ground_truth_data:
        return False, f"No ground truth found for video {video_id}"

    gt_info = ground_truth_data[video_id]
    ground_truth_angle = gt_info['ArmAngle']
    pitcher_hand = gt_info['PitcherHand']

    # Determine arm side from pitcher hand
    arm_side = 'right' if pitcher_hand.upper() == 'R' else 'left'

    # Verify arm side matches
    if pitcher_data.get('arm_side') != arm_side:
        # Recalculate with correct arm side if needed
        pass

    # Recalculate angle with specified start joint
    arm_side_actual = pitcher_data['arm_side']
    x_multiplier = 1 if arm_side_actual == 'right' else -1

    if start_joint == 'shoulder':
        shoulder_key = f'{arm_side_actual}_shoulder'
        start_point = (pitcher_data[shoulder_key]['x'], pitcher_data[shoulder_key]['y'])
    else:  # elbow
        elbow_key = f'{arm_side_actual}_elbow'
        start_point = (pitcher_data[elbow_key]['x'], pitcher_data[elbow_key]['y'])

    wrist_key = f'{arm_side_actual}_wrist'
    end_point = (pitcher_data[wrist_key]['x'], pitcher_data[wrist_key]['y'])

    # Calculate angle
    angle, magnitude = pose_utils.calculate_angle(start_point, end_point, x_multiplier)

    if angle is None:
        return False, "Could not calculate angle"

    # Calculate error
    error = abs(angle - ground_truth_angle)

    # Update pitcher data with new calculations
    pitcher_data['arm_angle_degrees'] = angle
    pitcher_data['arm_magnitude'] = magnitude
    pitcher_data['start_joint'] = start_joint
    pitcher_data['ground_truth_angle_degrees'] = ground_truth_angle
    pitcher_data['error_degrees'] = error
    pitcher_data['pitcher_hand'] = pitcher_hand

    # Create output directory
    output_dir = video_dir / "pitcher_calculations" / f"{frame_name}_angle"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON data
    output_json = output_dir / "data.json"
    result_data = {
        'image': frame_path.name,
        'video_id': video_id,
        'frame_name': frame_name,
        'pitcher_data': pitcher_data,
        'processing': {
            'start_joint': start_joint,
            'arm_side': arm_side_actual
        }
    }
    pose_utils.save_json(result_data, output_json)

    # Load original image and draw angle visualization
    try:
        image = cv2.imread(str(frame_path))
        if image is None:
            return False, "Could not load original image"

        # Draw angle visualization
        angle_image = draw_angle_on_image(image, pitcher_data, ground_truth_angle, start_joint)

        # Save visualization image
        output_image = output_dir / f"{frame_name}_angle.jpg"
        cv2.imwrite(str(output_image), angle_image)

    except Exception as e:
        return False, f"Failed to create visualization: {str(e)}"

    return True, f"Calculated (angle={angle:.2f}, GT={ground_truth_angle:.2f}, error={error:.2f})"


def process_all_videos(baseball_vids_dir, ground_truth_data, start_joint='shoulder', force=False):
    """
    Process all videos in the baseball_vids directory.

    Args:
        baseball_vids_dir: Path to baseball_vids directory
        ground_truth_data: Dictionary with ground truth angles from CSV
        start_joint: 'shoulder' or 'elbow'
        force: If True, reprocess all frames
    """
    # Get all video directories
    video_dirs = pose_utils.get_video_dirs(baseball_vids_dir)

    if not video_dirs:
        print("No video directories found!")
        return

    print(f"Found {len(video_dirs)} video directories\n")
    print(f"{'=' * 50}")
    print(f"CALCULATING PITCHER ANGLES")
    print(f"Start joint: {start_joint}")
    print(f"{'=' * 50}\n")

    # Statistics
    total_frames = 0
    total_processed = 0
    total_skipped = 0
    total_failed = 0

    # Process each video
    for video_id, video_dir in video_dirs:
        print(f"[Video: {video_id}]")

        # Check if this video has pitcher_labels directory
        pitcher_labels_dir = video_dir / "pitcher_labels"
        if not pitcher_labels_dir.exists():
            print(f"  Skipping (no pitcher_labels directory found)")
            print()
            continue

        # Get all labeled frames
        labeled_frames = sorted(pitcher_labels_dir.glob("frame_*_pitcher"))

        if not labeled_frames:
            print(f"  Skipping (no labeled frames found)")
            print()
            continue

        print(f"  Found {len(labeled_frames)} labeled frame(s)")

        video_processed = 0
        video_skipped = 0
        video_failed = 0

        # Process each frame
        for labeled_frame_dir in labeled_frames:
            frame_name_base = labeled_frame_dir.name.replace('_pitcher', '')

            # Find original frame in release_frames
            release_frames_dir = video_dir / "release_frames"
            frame_path = None
            for ext in ['.jpg', '.png']:
                potential_path = release_frames_dir / f"{frame_name_base}{ext}"
                if potential_path.exists():
                    frame_path = potential_path
                    break

            if frame_path is None:
                print(f"  [{frame_name_base}] ✗ Original frame not found")
                video_failed += 1
                total_failed += 1
                continue

            total_frames += 1

            print(f"  [{frame_path.name}] ", end='', flush=True)

            try:
                success, message = calculate_frame_angle(
                    frame_path, video_dir, video_id, ground_truth_data,
                    start_joint=start_joint, force=force
                )

                if success:
                    if message == "Already calculated":
                        video_skipped += 1
                        total_skipped += 1
                        print("SKIPPED")
                    else:
                        video_processed += 1
                        total_processed += 1
                        print(f"✓ {message}")
                else:
                    video_failed += 1
                    total_failed += 1
                    print(f"✗ {message}")
            except Exception as e:
                video_failed += 1
                total_failed += 1
                print(f"✗ Error: {str(e)}")

        print(f"  Video summary: {video_processed} processed, {video_skipped} skipped, {video_failed} failed")
        print()

    # Print overall summary
    print(f"{'=' * 50}")
    print(f"OVERALL SUMMARY")
    print(f"{'=' * 50}")
    print(f"Videos:     {len(video_dirs)}")
    print(f"Frames:     {total_frames}")
    print(f"Calculated: {total_processed}")
    print(f"Skipped:    {total_skipped}")
    print(f"Failed:     {total_failed}")
    print(f"{'=' * 50}\n")

    # Calculate average error if any frames were processed
    if total_processed > 0:
        print("Collecting error statistics...")
        errors = []

        for video_id, video_dir in video_dirs:
            calc_dir = video_dir / "pitcher_calculations"
            if not calc_dir.exists():
                continue

            for frame_calc_dir in calc_dir.glob("frame_*_angle"):
                json_path = frame_calc_dir / "data.json"
                if json_path.exists():
                    try:
                        data = pose_utils.load_json(json_path)
                        error = data['pitcher_data'].get('error_degrees')
                        if error is not None:
                            errors.append(error)
                    except:
                        pass

        if errors:
            avg_error = sum(errors) / len(errors)
            min_error = min(errors)
            max_error = max(errors)
            print(f"\nError Statistics ({len(errors)} frames):")
            print(f"  Average error: {avg_error:.2f}")
            print(f"  Min error:     {min_error:.2f}")
            print(f"  Max error:     {max_error:.2f}")
            print()


def main():
    parser = ArgumentParser(
        description="Calculate pitcher arm angles and compare with ground truth"
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=None,
        help="Path to baseball_vids directory (default: ~/Desktop/baseball_vids)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to ground truth CSV file (default: baseball_vids/arm_angles_high_speed.csv)"
    )
    parser.add_argument(
        "--start-joint",
        type=str,
        default="shoulder",
        choices=["shoulder", "elbow"],
        help="Joint to start angle measurement from (default: shoulder)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already-calculated frames"
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

    # Get CSV path
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = baseball_vids_dir / "arm_angles_high_speed.csv"

    if not csv_path.exists():
        print(f"Error: Ground truth CSV not found: {csv_path}")
        print("\nPlease ensure the CSV file exists with columns: PitchId, FileName, PitcherHand, ArmAngle")
        sys.exit(1)

    print(f"\nBaseball videos directory: {baseball_vids_dir}")
    print(f"Ground truth CSV: {csv_path}")
    print(f"Start joint: {args.start_joint}")
    print(f"Force reprocessing: {args.force}\n")

    # Load ground truth data
    print("Loading ground truth data...")
    try:
        ground_truth_data = pose_utils.load_ground_truth_csv(csv_path)
        print(f"✓ Loaded ground truth for {len(ground_truth_data)} videos\n")
    except Exception as e:
        print(f"✗ Failed to load ground truth CSV: {e}")
        sys.exit(1)

    # Process all videos
    process_all_videos(baseball_vids_dir, ground_truth_data,
                       start_joint=args.start_joint, force=args.force)


if __name__ == "__main__":
    main()