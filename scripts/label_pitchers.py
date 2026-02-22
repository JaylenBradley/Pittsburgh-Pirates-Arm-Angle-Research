"""
Label Pitchers Script for Baseball Pitcher Pose Analysis

This script provides an interactive tool to select the pitcher from detected persons.
For each frame in poses/, it:
1. Loads all detected persons from the poses data
2. Creates cropped images of each person's bounding box region
3. Displays all crops in a tiled window
4. Allows user to select the pitcher by clicking or pressing number key
5. Saves pitcher-specific data to pitcher_labels/FRAME_NAME/

Usage:
    From Pirates_Arm_Angle directory:
    python scripts/label_pitchers.py [--videos-dir PATH] [--force]

Arguments:
    --videos-dir: Optional path to baseball_vids directory (default: ~/Desktop/baseball_vids)
    --force: Force reprocessing of already-labeled frames

Controls:
    - Click on a person's crop to select them as the pitcher
    - Or press the number key (1-9) corresponding to the person
    - Press '0' or 'n' to mark frame as "no pitcher detected"
    - Press 'q' to quit
    - Press 's' to skip current frame

Example:
    python scripts/label_pitchers.py
    python scripts/label_pitchers.py --force
"""

import sys
from pathlib import Path
from argparse import ArgumentParser
import cv2
import numpy as np

import pose_utils


class PitcherLabeler:
    """Interactive pitcher labeling tool."""

    def __init__(self, tile_size=300, padding=10):
        """
        Initialize the labeler.

        Args:
            tile_size: Size of each person crop tile (pixels)
            padding: Padding between tiles (pixels)
        """
        self.tile_size = tile_size
        self.padding = padding
        self.selected_person = None
        self.window_name = "Select Pitcher - Click or Press Number (0/n=No Pitcher, s=Skip, q=Quit)"

    def create_person_crop(self, image, bbox, person_id):
        """
        Create a crop of a person from the image.

        Args:
            image: Full image array
            bbox: Bounding box dictionary with x1, y1, x2, y2
            person_id: ID of the person

        Returns:
            Cropped and resized image with person ID label
        """
        x1 = int(bbox['x1'])
        y1 = int(bbox['y1'])
        x2 = int(bbox['x2'])
        y2 = int(bbox['y2'])

        # Add some padding to bounding box
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)

        # Crop
        crop = image[y1:y2, x1:x2].copy()

        # Resize to tile size (maintain aspect ratio)
        h, w = crop.shape[:2]
        if h > w:
            new_h = self.tile_size
            new_w = int(w * (self.tile_size / h))
        else:
            new_w = self.tile_size
            new_h = int(h * (self.tile_size / w))

        crop_resized = cv2.resize(crop, (new_w, new_h))

        # Create square tile with padding
        tile = np.ones((self.tile_size, self.tile_size, 3), dtype=np.uint8) * 50

        # Center the crop in the tile
        y_offset = (self.tile_size - new_h) // 2
        x_offset = (self.tile_size - new_w) // 2
        tile[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = crop_resized

        # Add person ID label
        label = f"Person {person_id + 1}"
        cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)

        return tile

    def create_tiled_display(self, image, persons_data):
        """
        Create a tiled display of all detected persons.

        Args:
            image: Full image array
            persons_data: List of person data dictionaries

        Returns:
            Tiled display image
        """
        num_persons = len(persons_data)

        # Calculate grid dimensions
        cols = min(3, num_persons)  # Max 3 columns
        rows = (num_persons + cols - 1) // cols

        # Create tiles for each person
        tiles = []
        for person_data in persons_data:
            tile = self.create_person_crop(image, person_data['bbox'], person_data['person_id'])
            tiles.append(tile)

        # Pad with empty tiles if needed
        while len(tiles) < rows * cols:
            empty_tile = np.ones((self.tile_size, self.tile_size, 3), dtype=np.uint8) * 50
            tiles.append(empty_tile)

        # Create tiled display
        tile_rows = []
        for r in range(rows):
            row_tiles = tiles[r * cols:(r + 1) * cols]
            # Add padding between tiles
            row_with_padding = [row_tiles[0]]
            for tile in row_tiles[1:]:
                padding = np.ones((self.tile_size, self.padding, 3), dtype=np.uint8) * 100
                row_with_padding.append(padding)
                row_with_padding.append(tile)
            row = np.hstack(row_with_padding)
            tile_rows.append(row)

        # Stack rows vertically with padding
        display_rows = [tile_rows[0]]
        for row in tile_rows[1:]:
            padding = np.ones((self.padding, row.shape[1], 3), dtype=np.uint8) * 100
            display_rows.append(padding)
            display_rows.append(row)

        display = np.vstack(display_rows)

        return display, cols

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            cols, num_persons = param

            # Calculate which tile was clicked
            tile_with_padding = self.tile_size + self.padding
            col = x // tile_with_padding
            row = y // (self.tile_size + self.padding)

            person_idx = row * cols + col

            if person_idx < num_persons:
                self.selected_person = person_idx
                cv2.destroyAllWindows()

    def select_pitcher(self, image, persons_data):
        """
        Display interactive UI to select pitcher.

        Args:
            image: Full frame image
            persons_data: List of person data dictionaries

        Returns:
            Selected person ID (int), or -1 for no pitcher, or None for skip
        """
        if len(persons_data) == 0:
            print("    No persons detected in frame")
            return -1

        if len(persons_data) == 1:
            print(f"    Only 1 person detected - auto-selecting")
            return 0

        # Create tiled display
        display, cols = self.create_tiled_display(image, persons_data)

        # Setup window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback,
                             param=(cols, len(persons_data)))

        self.selected_person = None

        while True:
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF

            # Number key pressed (1-9)
            if ord('1') <= key <= ord('9'):
                person_idx = key - ord('1')
                if person_idx < len(persons_data):
                    self.selected_person = person_idx
                    break

            # 0 or 'n' pressed - no pitcher
            elif key == ord('0') or key == ord('n'):
                self.selected_person = -1
                break

            # 's' pressed - skip
            elif key == ord('s'):
                self.selected_person = None
                break

            # 'q' pressed - quit
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None

            # Mouse click processed
            if self.selected_person is not None:
                break

        cv2.destroyAllWindows()
        return self.selected_person


def process_frame(frame_path, video_dir, video_id, ground_truth_data, labeler, force=False):
    """
    Process a single frame to label the pitcher.

    Args:
        frame_path: Path to the frame image
        video_dir: Path to the video directory
        labeler: PitcherLabeler instance
        force: Force reprocessing

    Returns:
        Tuple of (success, message, should_quit)
    """
    frame_name = frame_path.stem
    # Get the poses frame name
    poses_frame_name = pose_utils.format_frame_name(frame_path.name, 'poses')
    pitcher_frame_name = pose_utils.format_frame_name(frame_path.name, 'pitcher')

    # Check if already labeled
    if not force and pose_utils.check_output_exists(video_dir, pitcher_frame_name, 'pitcher_labels'):
        return True, "Already labeled", False

    # Load poses data
    poses_dir = video_dir / 'poses' / poses_frame_name
    poses_json = poses_dir / 'data.json'

    if not poses_json.exists():
        return False, "Poses data not found - run process_release_frames.py first", False

    poses_data = pose_utils.load_json(poses_json)
    persons_data = poses_data.get('persons', [])

    # Load the visualization image from poses directory (has keypoints already drawn)
    poses_vis_img = poses_dir / f"{poses_frame_name}.jpg"
    if poses_vis_img.exists():
        image = cv2.imread(str(poses_vis_img))
    else:
        # Fallback to original image
        image = cv2.imread(str(frame_path))

    if image is None:
        return False, "Failed to load image", False

    # Select pitcher
    print(f"    Select pitcher ({len(persons_data)} person(s) detected)...")
    selected_person_idx = labeler.select_pitcher(image, persons_data)

    # Check for quit
    if selected_person_idx is None:
        return False, "Skipped by user", True

    # Create output directory
    output_dir = video_dir / 'pitcher_labels' / pitcher_frame_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle no pitcher case
    if selected_person_idx == -1:
        output_data = {
            'frame': frame_path.name,
            'pitcher_detected': False,
            'pitcher_person_id': None
        }
        pose_utils.save_json(output_data, output_dir / 'data.json')
        return "skip", "No pitcher detected", False

    # Get pitcher data
    pitcher_data = persons_data[selected_person_idx]

    # Get ground truth data to determine arm side
    arm_side = 'right'  # default
    if video_id in ground_truth_data:
        pitcher_hand = ground_truth_data[video_id]['PitcherHand']
        arm_side = 'right' if pitcher_hand.upper() == 'R' else 'left'

    # Extract keypoints for the specific arm
    keypoints_array = np.array(pitcher_data['keypoints'])

    # Get keypoint indices
    if arm_side == 'right':
        shoulder_idx = pose_utils.KEYPOINT_NAMES['right_shoulder']
        elbow_idx = pose_utils.KEYPOINT_NAMES['right_elbow']
        wrist_idx = pose_utils.KEYPOINT_NAMES['right_wrist']
    else:
        shoulder_idx = pose_utils.KEYPOINT_NAMES['left_shoulder']
        elbow_idx = pose_utils.KEYPOINT_NAMES['left_elbow']
        wrist_idx = pose_utils.KEYPOINT_NAMES['left_wrist']

    # Extract specific keypoints
    shoulder = keypoints_array[shoulder_idx]
    elbow = keypoints_array[elbow_idx]
    wrist = keypoints_array[wrist_idx]

    # Create keypoint dictionaries
    shoulder_key = f'{arm_side}_shoulder'
    elbow_key = f'{arm_side}_elbow'
    wrist_key = f'{arm_side}_wrist'

    # Create output data with extracted keypoints
    output_data = {
        'frame': frame_path.name,
        'pitcher_detected': True,
        'pitcher_person_id': pitcher_data['person_id'],
        'bbox': pitcher_data['bbox'],
        'keypoints': pitcher_data['keypoints'],  # Store all 133 keypoints for flexibility
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

    # Save JSON
    pose_utils.save_json(output_data, output_dir / 'data.json')

    # Load the visualization image that already has pose keypoints drawn
    poses_vis_img = poses_dir / f"{poses_frame_name}.jpg"
    if poses_vis_img.exists():
        vis_image = cv2.imread(str(poses_vis_img))
    else:
        # Fallback - use the image we already loaded
        vis_image = image.copy()

    # Create and save cropped pitcher image with keypoints
    bbox = pitcher_data['bbox']
    x1, y1 = int(bbox['x1']), int(bbox['y1'])
    x2, y2 = int(bbox['x2']), int(bbox['y2'])

    # Add padding
    pad = 50
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(vis_image.shape[1], x2 + pad)
    y2 = min(vis_image.shape[0], y2 + pad)

    # Crop the visualization image (which already has keypoints drawn)
    crop = vis_image[y1:y2, x1:x2].copy()

    # Draw bounding box around the pitcher
    cv2.rectangle(crop, (5, 5), (crop.shape[1] - 5, crop.shape[0] - 5), (0, 255, 0), 2)

    # Save image
    output_img = output_dir / f"{pitcher_frame_name}.jpg"
    cv2.imwrite(str(output_img), crop)

    return True, f"Labeled pitcher (person {selected_person_idx + 1}, {arm_side} arm)", False


def process_all_videos(baseball_vids_dir, ground_truth_data, force=False):
    """
    Process all videos in the baseball_vids directory.

    Args:
        baseball_vids_dir: Path to baseball_vids directory
        force: Force reprocessing
    """
    video_dirs = pose_utils.get_video_dirs(baseball_vids_dir)

    if not video_dirs:
        print(f"✗ No video directories found in: {baseball_vids_dir}")
        return

    print(f"\n{'=' * 50}")
    print(f"Found {len(video_dirs)} video(s) to process")
    print(f"{'=' * 50}\n")

    labeler = PitcherLabeler()

    total_processed = 0
    total_skipped = 0
    total_failed = 0
    total_frames = 0

    for video_idx, (video_id, video_dir) in enumerate(video_dirs, 1):
        print(f"[{video_idx}/{len(video_dirs)}] Processing video: {video_id}")

        # Get all release frames
        release_frames = pose_utils.get_release_frames(video_dir)

        if not release_frames:
            print(f"  No frames in release_frames/")
            print()
            continue

        total_frames += len(release_frames)
        print(f"  Found {len(release_frames)} frame(s) in release_frames/")

        video_processed = 0
        video_skipped = 0
        video_failed = 0
        should_quit = False

        for frame_idx, frame_path in enumerate(release_frames, 1):
            frame_name = frame_path.name
            print(f"  [{frame_idx}/{len(release_frames)}] {frame_name}")

            try:
                success, message, quit_flag = process_frame(
                    frame_path, video_dir, video_id, ground_truth_data, labeler, force=force
                )

                if quit_flag:
                    print(f"Quitting...")
                    should_quit = True
                    break

                if success == "skip" or (success and message == "Already labeled"):
                    video_skipped += 1
                    total_skipped += 1
                    print(f"SKIPPED: {message}")
                elif success:
                    video_processed += 1
                    total_processed += 1
                    print(f"{message}")
                else:
                    video_failed += 1
                    total_failed += 1
                    print(f"{message}")
            except Exception as e:
                video_failed += 1
                total_failed += 1
                print(f"Error: {str(e)}")

        print(f"Video summary: {video_processed} processed, {video_skipped} skipped, {video_failed} failed")
        print()

        if should_quit:
            break

    # Print overall summary
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
    parser = ArgumentParser(
        description="Interactive tool to label pitchers in frames"
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
        "--force",
        action="store_true",
        help="Force reprocessing of already-labeled frames"
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
        print(f"Warning: Ground truth CSV not found: {csv_path}")
        print("Will use default 'right' arm side for all pitchers\n")
        ground_truth_data = {}
    else:
        # Load ground truth data
        print("Loading ground truth data...")
        try:
            ground_truth_data = pose_utils.load_ground_truth_csv(csv_path)
            print(f"✓ Loaded ground truth for {len(ground_truth_data)} videos\n")
        except Exception as e:
            print(f"Warning: Failed to load ground truth CSV: {e}")
            print("Will use default 'right' arm side for all pitchers\n")
            ground_truth_data = {}

    print(f"\nBaseball videos directory: {baseball_vids_dir}")
    print(f"Force reprocessing: {args.force}\n")

    # Process all videos
    process_all_videos(baseball_vids_dir, ground_truth_data, force=args.force)

if __name__ == "__main__":
    main()