"""
Label Pitchers Script for Baseball Pitcher Pose Analysis

Interactive tool to manually select the pitcher from detected persons.
For each frame in release_frames/, it:
1. Loads detected persons from _yolo_poses/
2. Displays all candidates as cropped tiles
3. Allows user to select the pitcher by clicking or pressing number key
4. Batch labels all frames sharing same track_id

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

from utils import pose_utils, crop_utils, label_utils


def select_pitcher_interactive(image, filtered_persons):
    """
    Display interactive UI to select pitcher from candidates.
    
    Args:
        image: Full frame image from release_frames
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
            padding_percent=15,
            draw_overlay=True  # Show red overlay around bbox
        )
        candidates.append({
            'post_filter_idx': post_filter_idx,
            'original_idx': original_idx,
            'crop': crop
        })

    display, cols = crop_utils.create_tiled_display_from_crops(candidates)
    
    if display is None:
        return -1

    window_name = "Select Pitcher - Click or Press Number (n = No Pitcher, s = Skip, q = Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    selected_idx = None
    
    def mouse_callback(event, x, y, _flags, _param):
        nonlocal selected_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            tile_size = 300
            padding = 15
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


def process_frame(frame_path, video_dir, video_id, ground_truth_data, processed_frames, force=False):
    """
    Process a single frame to label the pitcher.
    
    Returns:
        Tuple of (success, message, should_quit)
    """
    poses_frame_name = pose_utils.format_frame_name(frame_path.name, 'yolo_poses')
    pitcher_frame_name = pose_utils.format_frame_name(frame_path.name, 'pitcher')

    # Check if already labeled
    if pitcher_frame_name in processed_frames:
        return True, "Already labeled", False
    
    if not force and pose_utils.check_output_exists(video_dir, pitcher_frame_name, 'pitcher_labels'):
        processed_frames.add(pitcher_frame_name)
        return True, "Already labeled", False

    # Load poses data
    poses_dir = video_dir / '_yolo_poses' / poses_frame_name
    poses_json = poses_dir / 'data.json'

    if not poses_json.exists():
        return False, "Poses data not found - run process_release_frames_yolo.py first", False

    poses_data = pose_utils.load_json(poses_json)
    persons_data = poses_data.get('persons', [])

    # Create filtered_persons list with original_index (no actual filtering for YOLO)
    filtered_persons = []
    for idx, person in enumerate(persons_data):
        person_copy = person.copy()
        person_copy['original_index'] = idx
        filtered_persons.append(person_copy)

    release_frame_path = video_dir / 'release_frames' / frame_path.name
    if release_frame_path.exists():
        image = cv2.imread(str(release_frame_path))
    else:
        poses_vis_img = poses_dir / f"{poses_frame_name}.jpg"
        image = cv2.imread(str(poses_vis_img)) if poses_vis_img.exists() else None

    if image is None:
        return False, "Failed to load image", False

    print(f"    Select pitcher ({len(filtered_persons)} candidate(s))...")
    selected_original_idx = select_pitcher_interactive(image, filtered_persons)

    if selected_original_idx is None:
        return False, "Quit by user", True
    if selected_original_idx == -2:
        return "skip", "Skipped", False

    # Handle no pitcher (only for current frame)
    if selected_original_idx == -1:
        label_utils.save_no_pitcher_label(video_dir, frame_path.name)
        processed_frames.add(pitcher_frame_name)
        return True, "No pitcher detected", False

    # Pitcher selected - get track_id for batch processing
    pitcher_data = persons_data[selected_original_idx]
    pitcher_track_id = pitcher_data.get('track_id')

    if pitcher_track_id is None:
        return False, "Track ID not found for selected pitcher", False

    # Get arm side from ground truth
    arm_side = label_utils.get_arm_side(video_id, ground_truth_data)

    # Batch process: find all frames with this track_id and label them
    print(f"    Batch processing frames with track_id {pitcher_track_id}...")
    
    # Build pitcher_data_map (frame_name -> pitcher_person_data)
    pitcher_data_map = {}
    all_release_frames = pose_utils.get_release_frames(video_dir)

    for batch_frame_path in all_release_frames:
        batch_poses_frame_name = pose_utils.format_frame_name(batch_frame_path.name, 'yolo_poses')
        batch_poses_json = video_dir / '_yolo_poses' / batch_poses_frame_name / 'data.json'
        
        if batch_poses_json.exists():
            batch_poses_data = pose_utils.load_json(batch_poses_json)
            batch_persons = batch_poses_data.get('persons', [])

            # Find pitcher track_id in this frame
            for person in batch_persons:
                if person.get('track_id') == pitcher_track_id and pitcher_track_id is not None:
                    pitcher_data_map[batch_frame_path.name] = person
                    break

    # Batch label all frames
    labeled_count, no_pitcher_count = label_utils.batch_label_frames_with_track_id(
        video_dir, pitcher_track_id, pitcher_data_map, arm_side, processed_frames, force=force
    )
    
    # Mark processed frames
    for frame_path_item in all_release_frames:
        frame_pitcher_name = pose_utils.format_frame_name(frame_path_item.name, 'pitcher')
        processed_frames.add(frame_pitcher_name)

    message = f"Labeled pitcher (person {selected_original_idx} with track_id {pitcher_track_id}): {labeled_count} pitcher(s), {no_pitcher_count} no-pitcher"
    return True, message, False


def process_all_videos(baseball_vids_dir, ground_truth_data, force=False):
    """Process all videos in baseball_vids directory."""
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
    total_pitcher_labels = 0
    total_no_pitcher_labels = 0

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
        video_pitcher_labels = 0
        video_no_pitcher_labels = 0
        should_quit = False
        processed_frames = set()

        for frame_idx, frame_path in enumerate(release_frames, 1):
            pitcher_frame_name = pose_utils.format_frame_name(frame_path.name, 'pitcher')
            
            # Skip if already processed in batch
            if pitcher_frame_name in processed_frames:
                print(f"  [{frame_idx}/{len(release_frames)}] {frame_path.name}... BATCH PROCESSED")
                video_processed += 1
                total_processed += 1
                continue

            print(f"  [{frame_idx}/{len(release_frames)}] {frame_path.name}", end=" ... ")

            try:
                success, message, quit_flag = process_frame(
                    frame_path, video_dir, video_id, ground_truth_data, processed_frames, force=force
                )

                if quit_flag:
                    print(f"\nQuitting...")
                    should_quit = True
                    break

                if success == "skip" or (success and message == "Already labeled"):
                    video_skipped += 1
                    total_skipped += 1
                    print(f"SKIPPED ({message})")
                elif success:
                    video_processed += 1
                    total_processed += 1
                    print(f"OK")
                    
                    # Extract pitcher/no_pitcher counts from message
                    if "pitcher(s)" in message and "no-pitcher" in message:
                        try:
                            parts = message.split(":")
                            if len(parts) >= 2:
                                counts_str = parts[1].strip()
                                pitcher_match = counts_str.split(",")[0].strip()
                                no_pitcher_match = counts_str.split(",")[1].strip()
                                
                                pitcher_count = int(pitcher_match.split()[0])
                                no_pitcher_count = int(no_pitcher_match.split()[0])
                                
                                video_pitcher_labels += pitcher_count
                                video_no_pitcher_labels += no_pitcher_count
                                total_pitcher_labels += pitcher_count
                                total_no_pitcher_labels += no_pitcher_count
                                
                                print(f"    Batch result: {pitcher_count} pitcher, {no_pitcher_count} no-pitcher")
                        except Exception as e:
                            print(f"    (Could not parse counts: {e})")
                    elif message == "No pitcher detected":
                        video_no_pitcher_labels += 1
                        total_no_pitcher_labels += 1
                        print(f"    No pitcher frame labeled")
                else:
                    video_failed += 1
                    total_failed += 1
                    print(f"FAILED")
            except Exception as e:
                video_failed += 1
                total_failed += 1
                print(f"ERROR: {str(e)}")

        print(f"\n  Video summary: {video_processed} labeled ({video_pitcher_labels} pitcher, {video_no_pitcher_labels} no-pitcher), {video_skipped} skipped, {video_failed} failed\n")

        if should_quit:
            break

    # Print summary
    print(f"{'=' * 50}")
    print(f"OVERALL SUMMARY")
    print(f"{'=' * 50}")
    print(f"Videos:  {len(video_dirs)}")
    print(f"Frames:  {total_frames}")
    print(f"Labeled: {total_processed}")
    print(f"  Pitcher:    {total_pitcher_labels}")
    print(f"  No-Pitcher: {total_no_pitcher_labels}")
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