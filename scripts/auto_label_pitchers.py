"""
Auto Label Pitchers Script (CLIP-based) for Baseball Pitcher Pose Analysis with YOLO

This script automatically labels the pitcher for each video using YOLO pose detections.

Per video, it:
1. Scores ALL frames with CLIP (candidate persons detected by YOLO)
2. Aggregates CLIP scores across frames by track_id using spatial priors
3. Selects best track (highest aggregated score)
4. Batch labels all frames with that track_id
5. Marks frames missing the track as "no pitcher"

Usage:
    python scripts/auto_label_pitchers.py [OPTIONS]

Examples:
    python scripts/auto_label_pitchers.py
    python scripts/auto_label_pitchers.py --force
    python scripts/auto_label_pitchers.py --min-pitcher-score 0.20
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2

from utils import pose_utils, label_utils
from utils.auto_label_utils import (
    CLIPPitcherSelector, 
    build_candidates, 
    score_candidates_batch,
    apply_spatial_prior,
    aggregate_track_scores,
    is_video_fully_processed
)


def process_all_frames(video_dir, selector):
    """
    Score all release frames with CLIP and return aggregated track scores.
    
    Returns:
        (frames_data, track_aggregates) where:
        - frames_data: List of frame processing results
        - track_aggregates: Dict of track_id -> aggregation info
    """
    release_frames = pose_utils.get_release_frames(video_dir)
    frames_data = []

    for frame_path in release_frames:
        frame_name = frame_path.name
        poses_frame_name = pose_utils.format_frame_name(frame_name, 'yolo_poses')
        poses_json = video_dir / '_yolo_poses' / poses_frame_name / 'data.json'

        if not poses_json.exists():
            continue

        # Load YOLO poses data
        poses_data = pose_utils.load_json(poses_json)
        persons_data = poses_data.get('persons', [])

        if not persons_data:
            frames_data.append({
                'frame_name': frame_name,
                'frame_path': frame_path,
                'num_persons': 0,
                'candidates_scored': []
            })
            continue

        # Load frame image
        frame_image = cv2.imread(str(frame_path))
        if frame_image is None:
            continue

        # Build and score candidates
        candidates = build_candidates(persons_data, frame_image, min_bbox_score=0.6)
        if candidates:
            candidates = score_candidates_batch(candidates, selector, frame_image.shape)
            
            # Compute bbox centers for spatial prior
            bbox_centers = []
            for candidate in candidates:
                person = candidate['person']
                bbox = person.get('bbox', {})
                cx = (bbox.get('x1', 0) + bbox.get('x2', 0)) / 2
                cy = (bbox.get('y1', 0) + bbox.get('y2', 0)) / 2
                bbox_centers.append((cx, cy))
            
            # Apply spatial prior
            scored_with_prior = apply_spatial_prior(candidates, bbox_centers, frame_image.shape)
        else:
            scored_with_prior = []

        frames_data.append({
            'frame_name': frame_name,
            'frame_path': frame_path,
            'num_persons': len(persons_data),
            'candidates_scored': scored_with_prior
        })

    # Aggregate scores across all frames by track_id
    track_aggregates = aggregate_track_scores(frames_data)

    return frames_data, track_aggregates


def process_video(video_id, video_dir, ground_truth_data, selector, min_pitcher_score, force=False):
    """
    Process a single video: score all frames, aggregate tracks, batch label.
    
    Returns:
        Tuple of (success, message, num_labeled, num_no_pitcher)
    """
    release_frames = pose_utils.get_release_frames(video_dir)

    if not release_frames:
        return False, "No frames in release_frames/", 0, 0

    if not force and is_video_fully_processed(video_dir):
        return True, "Video already processed", 0, 0

    # Score all frames and aggregate
    frames_data, track_aggregates = process_all_frames(video_dir, selector)

    if not track_aggregates:
        return False, "No tracks detected in any frame", 0, 0

    # Find best track (highest aggregated adjusted score)
    best_track_id = max(
        track_aggregates.keys(),
        key=lambda tid: track_aggregates[tid]['mean_adjusted_score']
    )
    best_track_info = track_aggregates[best_track_id]
    best_score = best_track_info['mean_adjusted_score']

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

    # Get arm side from ground truth
    arm_side = label_utils.get_arm_side(video_id, ground_truth_data)

    # Build clip_selection data for all frames
    # Reference spatial prior config from selector to avoid duplication
    clip_selection = {
        'aggregation_method': 'mean_across_frames',
        'spatial_prior_enabled': True,
        'spatial_prior_center_pct': [
            selector.SPATIAL_PRIOR_CENTER_X,
            selector.SPATIAL_PRIOR_CENTER_Y
        ],
        'spatial_prior_sigma': [
            selector.SPATIAL_PRIOR_SIGMA_X,
            selector.SPATIAL_PRIOR_SIGMA_Y
        ],
        'mean_base_score': float(best_track_info['mean_base_score']),
        'mean_adjusted_score': float(best_track_info['mean_adjusted_score']),
        'num_appearances': int(best_track_info['num_appearances'])
    }

    # Build pitcher_data_map: frame_name -> pitcher_person_data for frames containing best_track_id
    pitcher_data_map = {}
    frames_with_track = set()
    frames_without_track = set()

    for frame_path in release_frames:
        frame_name = frame_path.name
        poses_frame_name = pose_utils.format_frame_name(frame_name, 'yolo_poses')
        poses_json = video_dir / '_yolo_poses' / poses_frame_name / 'data.json'

        if poses_json.exists():
            poses_data = pose_utils.load_json(poses_json)
            persons_data = poses_data.get('persons', [])

            # Find best_track_id in this frame
            found = False
            for person in persons_data:
                if person.get('track_id') == best_track_id:
                    pitcher_data_map[frame_name] = person
                    frames_with_track.add(frame_name)
                    found = True
                    break

            if not found:
                frames_without_track.add(frame_name)
        else:
            frames_without_track.add(frame_name)

    # Label frames with pitcher (has track_id)
    labeled_count = 0
    for frame_name in frames_with_track:
        pitcher_frame_name = pose_utils.format_frame_name(frame_name, 'pitcher')
        if force or not pose_utils.check_output_exists(video_dir, pitcher_frame_name, 'pitcher_labels_auto'):
            person_data = pitcher_data_map[frame_name]
            label_utils.save_pitcher_label(
                video_dir, frame_name, person_data, arm_side, best_track_id,
                clip_selection=clip_selection,
                output_subdir='pitcher_labels_auto'
            )
            labeled_count += 1

    # Label frames without pitcher (no track_id)
    no_pitcher_count = 0
    for frame_name in frames_without_track:
        pitcher_frame_name = pose_utils.format_frame_name(frame_name, 'pitcher')
        if force or not pose_utils.check_output_exists(video_dir, pitcher_frame_name, 'pitcher_labels_auto'):
            label_utils.save_no_pitcher_label(video_dir, frame_name, output_subdir='pitcher_labels_auto')
            no_pitcher_count += 1

    return True, f"Labeled pitcher (track_id {best_track_id}, score={best_score:.4f}), {labeled_count} pitcher, {no_pitcher_count} no-pitcher", labeled_count, no_pitcher_count


def process_all_videos(baseball_vids_dir, ground_truth_data, selector, min_pitcher_score, force=False):
    """Run CLIP auto-labeling over all videos."""
    video_dirs = pose_utils.get_video_dirs(baseball_vids_dir)

    if not video_dirs:
        print(f'✗ No video directories found in: {baseball_vids_dir}')
        return

    print(f"\n{'=' * 60}")
    print('AUTO LABELING PITCHERS (CLIP + YOLO)')
    print(f"{'=' * 60}")
    print(f'Found {len(video_dirs)} video(s)\n')

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

        print(f'[{video_idx}/{total_videos}] {video_id}... ', end='', flush=True)

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
