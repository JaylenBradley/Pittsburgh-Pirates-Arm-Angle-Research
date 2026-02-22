"""
Process Release Frames Script for Baseball Pitcher Pose Analysis

This script processes all frames in release_frames/ directories across all videos.
For each frame, it:
1. Runs MMDetection to detect all persons in the frame
2. Runs MMPose to extract 133 keypoints for each detected person
3. Saves raw pose data (all persons with full keypoint data) to poses/FRAME_NAME/
4. Saves visualization image with all detected poses

Usage:
    From Pirates_Arm_Angle directory:
    python scripts/process_release_frames.py [--videos-dir PATH] [--force] [--device DEVICE]

Arguments:
    --videos-dir: Optional path to baseball_vids directory (default: ~/Desktop/baseball_vids)
    --force: Force reprocessing of already-processed frames
    --device: Device to use for inference (default: cpu)

Example:
    python scripts/process_release_frames.py
    python scripts/process_release_frames.py --videos-dir /path/to/baseball_vids
    python scripts/process_release_frames.py --force --device cpu
"""

import sys
from pathlib import Path
from argparse import ArgumentParser
import cv2

# Add ViTPose to path
vitpose_path = Path(__file__).parent.parent / "ViTPose"
sys.path.insert(0, str(vitpose_path))

from mmpose.apis import inference_top_down_pose_model, vis_pose_result, process_mmdet_results
from mmdet.apis import inference_detector

# Import our utilities
import pose_utils


def process_frame(frame_path, video_dir, det_model, pose_model, dataset, dataset_info,
                  bbox_thr=0.3, force=False):
    """
    Process a single frame to extract pose data for all detected persons.

    Args:
        frame_path: Path to the frame image
        video_dir: Path to the video directory
        det_model: MMDetection model
        pose_model: MMPose model
        dataset: Dataset name
        dataset_info: Dataset info object
        bbox_thr: Bounding box threshold
        force: Force reprocessing

    Returns:
        Tuple of (success, message)
    """
    frame_name = frame_path.stem  # e.g., 'frame_0001'
    output_frame_name = pose_utils.format_frame_name(frame_path.name, 'poses')

    # Check if already processed
    if not force and pose_utils.check_output_exists(video_dir, output_frame_name, 'poses'):
        return True, "Already processed"

    # Validate image file before processing
    if not frame_path.exists():
        return False, "Image file does not exist"

    file_size = frame_path.stat().st_size
    if file_size == 0:
        return False, "Image file is empty (0 bytes) - likely iCloud placeholder or corrupted. Run: python scripts/check_icloud_files.py --download"

    if file_size < 100:
        return False, f"Image file is too small ({file_size} bytes) - likely iCloud placeholder or corrupted"

    # Try to read the image with OpenCV
    try:
        test_image = cv2.imread(str(frame_path))
        if test_image is None:
            return False, "Image file is corrupted or iCloud placeholder (OpenCV cannot read it). Run: python scripts/check_icloud_files.py --download"
    except Exception as e:
        return False, f"Failed to read image: {str(e)}"

    # Create output directory
    output_dir = video_dir / 'poses' / output_frame_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run detection
    mmdet_results = inference_detector(det_model, str(frame_path))

    # Get person bounding boxes (category 1 = person in COCO)
    person_results = process_mmdet_results(mmdet_results, cat_id=1)

    if len(person_results) == 0:
        # No persons detected - save minimal JSON
        output_data = {
            'frame': frame_path.name,
            'persons_detected': 0,
            'persons': []
        }
        pose_utils.save_json(output_data, output_dir / 'data.json')
        return True, "No persons detected"

    # Run pose estimation
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        str(frame_path),
        person_results,
        bbox_thr=bbox_thr,
        format='xyxy',
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=False
    )

    # Extract and save pose data for all detected persons
    persons_data = []
    for person_id, pose_result in enumerate(pose_results):
        # Get bounding box
        bbox = person_results[person_id]['bbox']  # [x1, y1, x2, y2, score]

        # Get all 133 keypoints
        keypoints = pose_result['keypoints']  # Shape: (133, 3) - [x, y, confidence]

        person_data = {
            'person_id': person_id,
            'bbox': {
                'x1': float(bbox[0]),
                'y1': float(bbox[1]),
                'x2': float(bbox[2]),
                'y2': float(bbox[3]),
                'score': float(bbox[4]) if len(bbox) > 4 else 1.0
            },
            'keypoints': keypoints.tolist()  # Convert numpy array to list
        }
        persons_data.append(person_data)

    # Save JSON data
    output_data = {
        'frame': frame_path.name,
        'persons_detected': len(persons_data),
        'persons': persons_data
    }
    pose_utils.save_json(output_data, output_dir / 'data.json')

    # Save visualization
    output_img_path = output_dir / f"{output_frame_name}.jpg"
    vis_pose_result(
        pose_model,
        str(frame_path),
        pose_results,
        dataset=dataset,
        dataset_info=dataset_info,
        kpt_score_thr=0.3,
        radius=4,
        thickness=1,
        show=False,
        out_file=str(output_img_path)
    )

    return True, f"Processed {len(persons_data)} person(s)"


def process_all_videos(baseball_vids_dir, det_model, pose_model, dataset, dataset_info, force=False):
    """
    Process all videos in the baseball_vids directory.

    Args:
        baseball_vids_dir: Path to baseball_vids directory
        det_model: MMDetection model
        pose_model: MMPose model
        dataset: Dataset name
        dataset_info: Dataset info object
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

        for frame_idx, frame_path in enumerate(release_frames, 1):
            frame_name = frame_path.name
            print(f"    [{frame_idx}/{len(release_frames)}] {frame_name}...", end=" ")

            try:
                success, message = process_frame(
                    frame_path, video_dir, det_model, pose_model,
                    dataset, dataset_info, force=force
                )

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
    print(f"Videos:    {len(video_dirs)}")
    print(f"Frames:    {total_frames}")
    print(f"Processed: {total_processed}")
    print(f"Skipped:   {total_skipped}")
    print(f"Failed:    {total_failed}")
    print(f"{'=' * 50}\n")


def main():
    parser = ArgumentParser(
        description="Process release frames to extract pose data for all detected persons"
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
        help="Device to use for inference (default: cpu)"
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

    # Model paths (from your existing setup)
    det_config = vitpose_path / "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
    det_checkpoint = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
    pose_config = vitpose_path / "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
    pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"

    print("Loading models...")
    try:
        det_model, pose_model, dataset_info = pose_utils.init_models(
            str(det_config),
            det_checkpoint,
            str(pose_config),
            pose_checkpoint,
            device=args.device
        )
        dataset = pose_model.cfg.data['test']['type']
        print("✓ Models loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        sys.exit(1)

    # Process all videos
    process_all_videos(baseball_vids_dir, det_model, pose_model, dataset, dataset_info, force=args.force)


if __name__ == "__main__":
    main()