"""
Auto Label Pitchers Script (CLIP-based) for Baseball Pitcher Pose Analysis with YOLO

This script automatically labels the pitcher for each video using YOLO pose detections.

Per video, it:
1. Finds the best frame with 4+ detected persons
2. Scores candidates using CLIP with pitcher prompts
3. Batch labels all frames with same track_id
4. Marks remaining frames as "no pitcher"

Usage:
    python scripts/auto_label_pitchers.py [OPTIONS]

Examples:
    python scripts/auto_label_pitchers.py
    python scripts/auto_label_pitchers.py --force
    python scripts/auto_label_pitchers.py --min-pitcher-score 0.25
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2

from utils import pose_utils, label_utils
from utils.auto_label_utils import CLIPPitcherSelector, build_candidates


def find_best_frame(video_dir):
    """
    Find the best frame for CLIP scoring (with 4+ detected persons, or most detections).
    
    Returns:
        (frame_path, poses_data, num_persons) or (None, None, 0) if no valid frame
    """
    release_frames = pose_utils.get_release_frames(video_dir)
    best_frame = None
    best_poses_data = None
    best_num_persons = 0

    for frame_path in release_frames:
        poses_frame_name = pose_utils.format_frame_name(frame_path.name, 'yolo_poses')
        poses_json = video_dir / '_yolo_poses' / poses_frame_name / 'data.json'

        if not poses_json.exists():
            continue

        poses_data = pose_utils.load_json(poses_json)
        num_persons = len(poses_data.get('persons', []))

        # Prefer frames with 4+ persons, otherwise take the one with most
        if num_persons >= 4:
            if best_num_persons < 4 or num_persons > best_num_persons:
                best_frame = frame_path
                best_poses_data = poses_data
                best_num_persons = num_persons
        elif best_num_persons < 4 and num_persons > best_num_persons:
            best_frame = frame_path
            best_poses_data = poses_data
            best_num_persons = num_persons

    return best_frame, best_poses_data, best_num_persons


def process_video(video_id, video_dir, ground_truth_data, selector, min_pitcher_score, force=False):
    """
    Process a single video: find best frame with CLIP, batch label all frames with same track_id.
    
    Returns:
        Tuple of (success, message, num_labeled, num_no_pitcher)
    """
    release_frames = pose_utils.get_release_frames(video_dir)

    if not release_frames:
        return False, "No frames in release_frames/", 0, 0

    # Find best frame for CLIP scoring
    best_frame, best_poses_data, num_persons = find_best_frame(video_dir)

    if best_frame is None or best_poses_data is None:
        return False, "No valid poses found for any frame", 0, 0

    if num_persons == 0:
        return False, "No persons detected in best frame", 0, 0

    # Load clean image for CLIP scoring
    score_image = cv2.imread(str(best_frame))
    if score_image is None:
        return False, f"Failed to load best frame: {best_frame.name}", 0, 0

    # Get arm side from ground truth
    arm_side = label_utils.get_arm_side(video_id, ground_truth_data)

    # Build filtered candidates from best frame, sorted by YOLO bbox confidence
    persons_data = best_poses_data.get('persons', [])
    candidates = build_candidates(
        persons_data,
        score_image,
        min_bbox_score=0.6
    )

    if not candidates:
        return False, "No valid pitcher candidates after filtering", 0, 0

    # Only show CLIP the top 4 highest-confidence detections
    top_candidates = candidates[:4]
    
    # Score only top candidates with CLIP
    for candidate in top_candidates:
        candidate['pitcher_score'] = selector.score_pitcherness(candidate['crop'])
        candidate['track_id'] = candidate['person'].get('track_id')

    # Sort by CLIP score and take the best
    top_candidates.sort(key=lambda c: c['pitcher_score'], reverse=True)
    best_candidate = top_candidates[0]
    best_score = best_candidate['pitcher_score']
    pitcher_track_id = best_candidate['track_id']

    # Check minimum score threshold
    if min_pitcher_score is not None and best_score < min_pitcher_score:
        # Mark entire video as no pitcher detected
        num_no_pitcher = 0
        for frame_path in release_frames:
            pitcher_frame_name = pose_utils.format_frame_name(frame_path.name, 'pitcher')
            if not force and pose_utils.check_output_exists(video_dir, pitcher_frame_name, 'pitcher_labels_auto'):
                continue
            label_utils.save_no_pitcher_label(video_dir, frame_path.name, output_subdir='pitcher_labels_auto')
            num_no_pitcher += 1

        return True, f"All frames marked no pitcher (best_score={best_score:.4f} < {min_pitcher_score})", 0, num_no_pitcher

    # Build pitcher_data_map for batch labeling
    pitcher_data_map = {}
    for frame_path in release_frames:
        poses_frame_name = pose_utils.format_frame_name(frame_path.name, 'yolo_poses')
        poses_json = video_dir / '_yolo_poses' / poses_frame_name / 'data.json'

        if poses_json.exists():
            poses_data = pose_utils.load_json(poses_json)
            frame_persons = poses_data.get('persons', [])

            # Find pitcher track_id in this frame
            for person in frame_persons:
                if person.get('track_id') == pitcher_track_id and pitcher_track_id is not None:
                    pitcher_data_map[frame_path.name] = person
                    break

    # Batch label all frames
    labeled_count, no_pitcher_count = label_utils.batch_label_frames_with_track_id(
        video_dir, pitcher_track_id, pitcher_data_map, arm_side, set(), force=force, output_subdir='pitcher_labels_auto'
    )

    return True, f"Labeled pitcher (track_id {pitcher_track_id}, score={best_score:.4f}), {labeled_count} frame(s)", labeled_count, no_pitcher_count


def process_all_videos(baseball_vids_dir, ground_truth_data, selector, min_pitcher_score, force=False):
    """Run CLIP auto-labeling over all videos."""
    video_dirs = pose_utils.get_video_dirs(baseball_vids_dir)

    if not video_dirs:
        print(f'✗ No video directories found in: {baseball_vids_dir}')
        return

    print(f"\n{'=' * 60}")
    print('AUTO LABELING PITCHERS (CLIP + YOLO)')
    print(f"{'=' * 60}")
    print(f'Found {len(video_dirs)} video(s)')
    print()

    total_videos = len(video_dirs)
    total_labeled = 0
    total_no_pitcher = 0
    total_failed = 0

    for video_idx, (video_id, video_dir) in enumerate(video_dirs, 1):
        release_frames = pose_utils.get_release_frames(video_dir)

        if not release_frames:
            print(f'[{video_idx}/{total_videos}] {video_id}... No frames in release_frames/')
            total_failed += 1
            continue

        print(f'[{video_idx}/{total_videos}] {video_id}... ', end='')

        try:
            success, message, num_labeled, num_no_pitcher = process_video(
                video_id=video_id,
                video_dir=video_dir,
                ground_truth_data=ground_truth_data,
                selector=selector,
                min_pitcher_score=min_pitcher_score,
                force=force
            )

            if success:
                total_labeled += num_labeled
                total_no_pitcher += num_no_pitcher
                print(f'✓ {message}')
            else:
                total_failed += 1
                print(f'✗ {message}')

        except Exception as exc:
            total_failed += 1
            print(f'✗ Error: {exc}')

    print(f"\n{'=' * 60}")
    print('OVERALL SUMMARY')
    print(f"{'=' * 60}")
    print(f'Videos:      {total_videos}')
    print(f'Labeled:     {total_labeled}')
    print(f'No Pitcher:  {total_no_pitcher}')
    print(f'Failed:      {total_failed}')
    print(f"{'=' * 60}\n")


def main():
    parser = ArgumentParser(description='Automatically label pitchers using CLIP on YOLO poses')
    parser.add_argument(
        '--videos-dir',
        type=str,
        default=None,
        help='Path to baseball_vids directory (default: ~/Desktop/baseball_vids)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Path to ground truth CSV file (default: baseball_vids/arm_angles_high_speed.csv)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing of already-labeled frames'
    )
    parser.add_argument(
        '--clip-model',
        type=str,
        default='openai/clip-vit-base-patch32',
        help='Hugging Face CLIP model name'
    )
    parser.add_argument(
        '--clip-device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device for CLIP inference'
    )
    parser.add_argument(
        '--min-pitcher-score',
        type=float,
        default=0.15,
        help='Minimum CLIP pitcherness score threshold (default: 0.15)'
    )

    args = parser.parse_args()

    try:
        baseball_vids_dir = pose_utils.get_baseball_vids_dir(args.videos_dir)
    except FileNotFoundError as exc:
        print(f'Error: {exc}')
        sys.exit(1)

    if not baseball_vids_dir.exists():
        print(f'Error: Directory not found: {baseball_vids_dir}')
        sys.exit(1)

    # Load ground truth
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = baseball_vids_dir / 'arm_angles_high_speed.csv'

    if not csv_path.exists():
        print(f'Warning: Ground truth CSV not found: {csv_path}')
        ground_truth_data = {}
    else:
        print('Loading ground truth data...')
        try:
            ground_truth_data = pose_utils.load_ground_truth_csv(csv_path)
            print(f'✓ Loaded ground truth for {len(ground_truth_data)} videos\n')
        except Exception as exc:
            print(f'Warning: Failed to load ground truth CSV: {exc}\n')
            ground_truth_data = {}

    print(f'Baseball videos directory: {baseball_vids_dir}')
    print(f'CLIP model: {args.clip_model}')
    print(f'CLIP device: {args.clip_device}')
    print(f'Min pitcher score: {args.min_pitcher_score}')
    print(f'Force reprocessing: {args.force}\n')

    print('Loading CLIP model...')
    try:
        selector = CLIPPitcherSelector(
            model_name=args.clip_model,
            device=args.clip_device
        )
        print(f'✓ CLIP loaded on device: {selector.device}\n')
    except Exception as exc:
        print(f'✗ Failed to load CLIP model: {exc}')
        sys.exit(1)

    process_all_videos(
        baseball_vids_dir=baseball_vids_dir,
        ground_truth_data=ground_truth_data,
        selector=selector,
        min_pitcher_score=args.min_pitcher_score,
        force=args.force
    )


if __name__ == '__main__':
    main()