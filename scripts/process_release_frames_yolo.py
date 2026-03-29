"""
Process Release Frames with YOLO for Baseball Pitcher Pose Analysis

This script processes all frames in release_frames/ directories using YOLO for multi-frame
person tracking. Instead of detecting each frame independently, it:
1. Groups frames in batches of 5 (or remaining frames for last batch)
2. Runs YOLO on all frames in the batch with tracking enabled
3. Extracts 17 body keypoints per detected person for consistency across frames
4. Saves raw pose data to _yolo_poses/FRAME_NAME/ in same JSON format as VitPose
5. Saves visualization image with all detected poses and track IDs

Usage:
    From Pirates_Arm_Angle directory:
    python scripts/process_release_frames_yolo.py [--videos-dir PATH] [--force] [--device DEVICE]

Arguments:
    --videos-dir: Optional path to baseball_vids directory (default: ~/Desktop/baseball_vids)
    --force: Force reprocessing of already-processed frames
    --device: Device to use for inference (default: cpu)

Example:
    python scripts/process_release_frames_yolo.py
    python scripts/process_release_frames_yolo.py --videos-dir /path/to/baseball_vids
    python scripts/process_release_frames_yolo.py --force --device cuda:0
"""

import sys
from argparse import ArgumentParser
import cv2
from ultralytics import YOLO
from utils import pose_utils


# 17 body keypoint indices from YOLO (OpenPose format)
BODY_KEYPOINT_INDICES = {
    'nose': 0,
    'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
}


def process_frame_batch(frame_paths, video_dir, model, force=False):
    """
    Process a batch of frames (up to 5) with YOLO tracking enabled.
    
    Tracking across the batch helps maintain consistent person IDs across frames.

    Args:
        frame_paths: List of Path objects for frames in this batch
        video_dir: Path to the video directory
        model: YOLO model instance
        force: Force reprocessing

    Returns:
        Dictionary with frame_name -> (success, message) mappings
    """
    results_dict = {}

    # Process each frame with tracking enabled
    for frame_idx, frame_path in enumerate(frame_paths):
        frame_name = frame_path.stem
        output_frame_name = pose_utils.format_frame_name(frame_name, 'yolo_poses')

        # Check if already processed
        if not force and pose_utils.check_output_exists(video_dir, output_frame_name, '_yolo_poses'):
            results_dict[frame_name] = (True, "Already processed")
            continue

        # Validate image file
        if not frame_path.exists():
            results_dict[frame_name] = (False, "Image file does not exist")
            continue

        file_size = frame_path.stat().st_size
        if file_size == 0:
            results_dict[frame_name] = (False, "Image file is empty (0 bytes)")
            continue

        if file_size < 100:
            results_dict[frame_name] = (False, f"Image file is too small ({file_size} bytes)")
            continue

        # Try to read the image
        try:
            image = cv2.imread(str(frame_path))
            if image is None:
                results_dict[frame_name] = (False, "Image file is corrupted or iCloud placeholder")
                continue
        except Exception as e:
            results_dict[frame_name] = (False, f"Failed to read image: {str(e)}")
            continue

        # Create output directory
        output_dir = video_dir / '_yolo_poses' / output_frame_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run YOLO inference with tracking
        # Use .track() method for multi-frame tracking (persist=True maintains tracking across frames)
        try:
            results = model.track(str(frame_path), persist=True, conf=0.5, verbose=False)
        except Exception as e:
            results_dict[frame_name] = (False, f"YOLO tracking failed: {str(e)}")
            continue

        # Extract persons data from YOLO results
        if len(results) == 0 or results[0].keypoints is None:
            # No persons detected
            output_data = {
                'frame': frame_path.name,
                'persons_detected': 0,
                'persons': []
            }
            pose_utils.save_json(output_data, output_dir / 'data.json')
            results_dict[frame_name] = (True, "No persons detected")
            continue

        result = results[0]
        persons_data = []

        # Extract data from YOLO detections
        if result.boxes is not None and result.keypoints is not None:
            for detection_idx in range(len(result.boxes)):
                try:
                    # Get bounding box (x1, y1, x2, y2)
                    box = result.boxes.xyxy[detection_idx].cpu().numpy()
                    box_conf = float(result.boxes.conf[detection_idx].cpu().numpy())

                    # Get keypoints - shape is (17, 3) for x, y, confidence
                    keypoints = result.keypoints.xy[detection_idx].cpu().numpy()
                    keypoint_conf = result.keypoints.conf[detection_idx].cpu().numpy()

                    # Get track ID if available
                    track_id = None
                    if result.boxes.id is not None:
                        track_id = int(result.boxes.id[detection_idx].cpu().numpy())

                    # Build keypoints array in expected format: [[x, y, conf], ...]
                    # YOLO provides 17 body keypoints
                    keypoints_with_conf = []
                    for kpt_idx in range(len(keypoints)):
                        x, y = keypoints[kpt_idx]
                        conf = float(keypoint_conf[kpt_idx])
                        keypoints_with_conf.append([float(x), float(y), conf])

                    person_data = {
                        'person_id': detection_idx,
                        'track_id': track_id,
                        'bbox': {
                            'x1': float(box[0]),
                            'y1': float(box[1]),
                            'x2': float(box[2]),
                            'y2': float(box[3]),
                            'score': box_conf
                        },
                        'keypoints': keypoints_with_conf  # 17 body keypoints
                    }
                    persons_data.append(person_data)

                except Exception as e:
                    # Skip this detection if there's an error
                    continue

        # Save JSON data
        output_data = {
            'frame': frame_path.name,
            'persons_detected': len(persons_data),
            'persons': persons_data
        }
        pose_utils.save_json(output_data, output_dir / 'data.json')

        # Save visualization
        try:
            output_img_path = output_dir / f"{output_frame_name}.jpg"
            image_with_poses = result.plot()  # YOLO built-in visualization
            cv2.imwrite(str(output_img_path), image_with_poses)
            results_dict[frame_name] = (True, f"Processed {len(persons_data)} person(s)")
        except Exception as e:
            results_dict[frame_name] = (False, f"Failed to save visualization: {str(e)}")

    return results_dict


def process_all_videos(baseball_vids_dir, model, force=False):
    """
    Process all videos in the baseball_vids directory.

    Args:
        baseball_vids_dir: Path to baseball_vids directory
        model: YOLO model instance
        force: Force reprocessing
    """
    video_dirs = pose_utils.get_video_dirs(baseball_vids_dir)

    if not video_dirs:
        print(f"✗ No video directories found in: {baseball_vids_dir}")
        print(f"  Make sure videos have been extracted using extract_video_frames.py")
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

        # Process frames in batches of 5
        batch_size = 5
        frame_counter = 0
        for batch_idx in range(0, len(release_frames), batch_size):
            batch_frames = release_frames[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            print(f"  Processing batch {batch_num} ({len(batch_frames)} frame(s))...")

            # Process this batch
            batch_results = process_frame_batch(batch_frames, video_dir, model, force=force)

            # Track results
            for frame_path in batch_frames:
                frame_name = frame_path.stem
                frame_counter += 1
                
                if frame_name in batch_results:
                    success, message = batch_results[frame_name]
                    print(f"    [{frame_counter}/{len(release_frames)}] {frame_name}...", end=" ")

                    if success:
                        if message == "Already processed":
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

        print(f"  Video summary: {video_processed} processed, {video_skipped} skipped, {video_failed} failed")
        print()

    # Print overall summary
    print(f"{'=' * 50}")
    print(f"OVERALL SUMMARY")
    print(f"{'=' * 50}")
    print(f"Videos:    {len(video_dirs)}")
    print(f"Frames:    {total_frames}")
    print(f"Processed: {total_processed}")
    print(f"Skipped:   {total_skipped}")
    print(f"Failed:    {total_failed}")
    print(f"{'=' * 50}\n")


def main():
    parser = ArgumentParser(
        description="Process release frames using YOLO for pose detection with multi-frame tracking"
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=None,
        help="Path to baseball_vids directory (default: ~/Desktop/baseball_vids)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already-processed frames"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for inference (default: cpu). Use 'cuda:0' for GPU."
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

    print(f"\nBaseball videos directory: {baseball_vids_dir}")
    print(f"Device: {args.device}")
    print(f"Force reprocessing: {args.force}\n")

    # Load YOLO model
    print("Loading YOLO model (yolov8l-pose)...")
    try:
        model = YOLO("yolov8l-pose.pt")
        model.to(args.device)
        print("✓ YOLO model loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        print("  Make sure ultralytics is installed: pip install ultralytics")
        sys.exit(1)

    # Process all videos
    process_all_videos(baseball_vids_dir, model, force=args.force)


if __name__ == "__main__":
    main()