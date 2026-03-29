"""
Label Pitchers Script for Baseball Pitcher Pose Analysis

Interactive tool to manually select the pitcher from detected persons.
For each frame in release_frames/, it:
1. Loads detected persons from poses/
2. Filters out detections without body and hand keypoints
3. Displays filtered candidates as cropped tiles
4. Allows user to select the pitcher by clicking or pressing number key
5. Saves pitcher-specific data to pitcher_labels/

Usage:
    python scripts/label_pitchers.py [--videos-dir PATH] [--force]

Arguments:
    --videos-dir: Path to baseball_vids directory (default: ~/Desktop/baseball_vids)
    --force: Force reprocessing of already-labeled frames

Controls:
    Click on a crop or press number (1-9) to select pitcher
    'n' = no pitcher detected
    's' = skip frame
    'q' = quit

Example:
    python scripts/label_pitchers.py
    python scripts/label_pitchers.py --force
"""

import sys
from pathlib import Path
from argparse import ArgumentParser
import cv2
import numpy as np

from utils import pose_utils, crop_utils


def select_pitcher_interactive(image, filtered_persons):
    """
    Display interactive UI to select pitcher from filtered candidates.
    
    Args:
        image: Full frame image (clean from release_frames)
        filtered_persons: List of person dicts with 'original_index' field
    
    Returns:
        Original index of selected person, -1 for no pitcher, -2 for skip, None for quit
    """
    if len(filtered_persons) == 0:
        return -1
    
    # Build candidate list for display
    candidates = []
    for post_filter_idx, person_data in enumerate(filtered_persons):
        original_idx = person_data.get('original_index', post_filter_idx)
        crop = crop_utils.extract_person_crop(
            image,
            person_data,
            padding_percent=20,
            draw_overlay=True  # Show red overlay for manual inspection
        )
        candidates.append({
            'post_filter_idx': post_filter_idx,
            'original_idx': original_idx,
            'crop': crop
        })
    
    # Create tiled display
    display, cols = crop_utils.create_tiled_display_from_crops(candidates)
    
    if display is None:
        return -1
    
    # Setup window and interaction
    window_name = "Select Pitcher - Click or Press Number (n = No Pitcher, s = Skip, q = Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    selected_idx = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            tile_size = 300
            padding = 10
            tile_with_padding = tile_size + padding
            col = x // tile_with_padding
            row = y // tile_with_padding
            idx = row * cols + col
            if idx < len(candidates):
                selected_idx = candidates[idx]['original_idx']
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        
        # Number key (1-9)
        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(candidates):
                selected_idx = candidates[idx]['original_idx']
                break
        
        # 'n' = no pitcher
        elif key == ord('n'):
            selected_idx = -1
            break
        
        # 's' = skip
        elif key == ord('s'):
            selected_idx = -2
            break
        
        # 'q' = quit
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None
        
        # Mouse click
        if selected_idx is not None:
            break
    
    cv2.destroyAllWindows()
    return selected_idx


def process_frame(frame_path, video_dir, video_id, ground_truth_data, force=False):
    """
    Process a single frame to label the pitcher.
    
    Args:
        frame_path: Path to release frame image
        video_dir: Path to video directory
        video_id: Video ID (for ground truth lookup)
        ground_truth_data: Dict of ground truth pitcher hands
        force: Force reprocessing if already labeled
    
    Returns:
        Tuple of (success, message, should_quit)
    """
    # Get the poses frame name
    poses_frame_name = pose_utils.format_frame_name(frame_path.name, 'yolo_poses')
    pitcher_frame_name = pose_utils.format_frame_name(frame_path.name, 'pitcher')

    # Check if already labeled
    if not force and pose_utils.check_output_exists(video_dir, pitcher_frame_name, 'pitcher_labels'):
        return True, "Already labeled", False

    # Load poses data
    poses_dir = video_dir / '_yolo_poses' / poses_frame_name
    poses_json = poses_dir / 'data.json'

    if not poses_json.exists():
        return False, "Poses data not found - run process_release_frames.py first", False

    poses_data = pose_utils.load_json(poses_json)
    persons_data = poses_data.get('persons', [])

    # Filter persons (body + hand keypoints required)
    filtered_persons, _ = crop_utils.filter_persons_by_keypoints(
        persons_data,
        confidence_threshold=0.2,
        require_hands=False  # YOLO only provides 17 body keypoints, no hands
    )

    # Load clean image from release_frames
    release_frame_path = video_dir / 'release_frames' / frame_path.name
    if release_frame_path.exists():
        image = cv2.imread(str(release_frame_path))
    else:
        # Fallback
        poses_vis_img = poses_dir / f"{poses_frame_name}.jpg"
        image = cv2.imread(str(poses_vis_img)) if poses_vis_img.exists() else None

    if image is None:
        return False, "Failed to load image", False

    # Select pitcher
    print(f"    Select pitcher ({len(filtered_persons)} candidate(s))...")
    selected_original_idx = select_pitcher_interactive(image, filtered_persons)

    # Check responses
    if selected_original_idx is None:
        return False, "Quit by user", True
    if selected_original_idx == -2:
        return "skip", "Skipped current frame", False

    # Create output directory
    output_dir = video_dir / 'pitcher_labels' / pitcher_frame_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle no pitcher
    if selected_original_idx == -1:
        output_data = {
            'frame': frame_path.name,
            'pitcher_detected': False,
            'pitcher_person_id': None
        }
        pose_utils.save_json(output_data, output_dir / 'data.json')
        return "skip", "No pitcher detected", False

    # Get pitcher data
    pitcher_data = persons_data[selected_original_idx]

    # Determine arm side from ground truth
    arm_side = ''
    if video_id in ground_truth_data:
        pitcher_hand = ground_truth_data[video_id]['PitcherHand']
        arm_side = 'right' if pitcher_hand.upper() == 'R' else 'left'

    # Extract keypoints for arm
    keypoints_array = np.array(pitcher_data['keypoints'])
    if arm_side == 'right':
        shoulder_idx = pose_utils.KEYPOINT_NAMES['right_shoulder']
        elbow_idx = pose_utils.KEYPOINT_NAMES['right_elbow']
        wrist_idx = pose_utils.KEYPOINT_NAMES['right_wrist']
    else:
        shoulder_idx = pose_utils.KEYPOINT_NAMES['left_shoulder']
        elbow_idx = pose_utils.KEYPOINT_NAMES['left_elbow']
        wrist_idx = pose_utils.KEYPOINT_NAMES['left_wrist']

    shoulder = keypoints_array[shoulder_idx]
    elbow = keypoints_array[elbow_idx]
    wrist = keypoints_array[wrist_idx]

    # Create output data
    shoulder_key = f'{arm_side}_shoulder'
    elbow_key = f'{arm_side}_elbow'
    wrist_key = f'{arm_side}_wrist'

    output_data = {
        'frame': frame_path.name,
        'pitcher_detected': True,
        'pitcher_person_id': pitcher_data['person_id'],
        'bbox': pitcher_data['bbox'],
        'keypoints': pitcher_data['keypoints'],
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

    pose_utils.save_json(output_data, output_dir / 'data.json')

    # Create and save cropped pitcher image WITH red overlay and keypoints (as shown during labeling)
    cropped_pitcher_with_overlay = crop_utils.extract_person_crop(
        image,
        pitcher_data,
        padding_percent=10,
        draw_overlay=True  # Red box outline & keypoint markers
    )

    output_img = output_dir / f"{pitcher_frame_name}.jpg"
    cv2.imwrite(str(output_img), cropped_pitcher_with_overlay)

    return True, f"Labeled pitcher (person {selected_original_idx + 1}, {arm_side} arm)", False


def process_all_videos(baseball_vids_dir, ground_truth_data, force=False):
    """
    Process all videos in baseball_vids directory.
    """
    video_dirs = pose_utils.get_video_dirs(baseball_vids_dir)

    if not video_dirs:
        print(f"✗ No video directories found in: {baseball_vids_dir}")
        return

    print(f"\n{'=' * 50}")
    print(f"Found {len(video_dirs)} video(s) to process")
    print(f"{'=' * 50}\n")

    total_processed = 0
    total_skipped = 0
    total_failed = 0
    total_frames = 0

    for video_idx, (video_id, video_dir) in enumerate(video_dirs, 1):
        print(f"[{video_idx}/{len(video_dirs)}] Processing video: {video_id}")

        release_frames = pose_utils.get_release_frames(video_dir)

        if not release_frames:
            print(f"  No frames in release_frames/\n")
            continue

        total_frames += len(release_frames)
        print(f"  Found {len(release_frames)} frame(s)\n")

        video_processed = 0
        video_skipped = 0
        video_failed = 0
        should_quit = False

        for frame_idx, frame_path in enumerate(release_frames, 1):
            print(f"  [{frame_idx}/{len(release_frames)}] {frame_path.name}", end=" ... ")

            try:
                success, message, quit_flag = process_frame(
                    frame_path, video_dir, video_id, ground_truth_data, force=force
                )

                if quit_flag:
                    print(f"\nQuitting...")
                    should_quit = True
                    break

                if success == "skip" or (success and message == "Already labeled"):
                    video_skipped += 1
                    total_skipped += 1
                    print(f"SKIPPED")
                elif success:
                    video_processed += 1
                    total_processed += 1
                    print(f"OK")
                else:
                    video_failed += 1
                    total_failed += 1
                    print(f"FAILED")
            except Exception as e:
                video_failed += 1
                total_failed += 1
                print(f"ERROR: {str(e)}")

        print(f"Video summary: {video_processed} labeled, {video_skipped} skipped, {video_failed} failed\n")

        if should_quit:
            break

    # Print summary
    print(f"{'=' * 50}")
    print(f"OVERALL SUMMARY")
    print(f"{'=' * 50}")
    print(f"Videos:  {len(video_dirs)}")
    print(f"Frames:  {total_frames}")
    print(f"Labeled: {total_processed}")
    print(f"Skipped: {total_skipped}")
    print(f"Failed:  {total_failed}")
    print(f"{'=' * 50}\n")


def main():
    parser = ArgumentParser(description="Label pitchers in frames")
    parser.add_argument("--videos-dir", type=str, default=None,
                        help="Path to baseball_vids directory")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to ground truth CSV file")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing of already-labeled frames")

    args = parser.parse_args()

    try:
        baseball_vids_dir = pose_utils.get_baseball_vids_dir(args.videos_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not baseball_vids_dir.exists():
        print(f"Error: Directory not found: {baseball_vids_dir}")
        sys.exit(1)

    # Load ground truth
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = baseball_vids_dir / "arm_angles_high_speed.csv"

    if not csv_path.exists():
        print(f"Warning: Ground truth CSV not found: {csv_path}")
        ground_truth_data = {}
    else:
        print("Loading ground truth data...")
        try:
            ground_truth_data = pose_utils.load_ground_truth_csv(csv_path)
            print(f"✓ Loaded ground truth for {len(ground_truth_data)} videos\n")
        except Exception as e:
            print(f"Warning: Failed to load ground truth: {e}\n")
            ground_truth_data = {}

    print(f"Baseball videos directory: {baseball_vids_dir}")
    print(f"Force reprocessing: {args.force}\n")

    process_all_videos(baseball_vids_dir, ground_truth_data, force=args.force)

if __name__ == "__main__":
    main()

