"""
Auto Label Pitchers Script (CLIP-based) for Baseball Pitcher Pose Analysis.

This script labels the pitcher for each frame using precomputed pose detections
from process_release_frames.py.

Per frame, it:
1. Loads detected persons from poses/FRAME_NAME_poses/data.json
2. Applies keypoint/bbox filtering and builds candidate crops
3. Scores every candidate crop using contrastive CLIP prompts:
   - Positive prompts: pitcher poses (mid-delivery, throwing, winding up, etc.)
   - Negative prompts: non-pitcher people (umpire, catcher, spectator, batter, legs)
   - Score = max(pitcher_scores) - max(negative_scores)
4. Selects the highest-scoring candidate
5. Saves output JSON/image to {output_subdir}/FRAME_NAME_pitcher/

Uses contrastive scoring to improve discrimination against false positives
(e.g., frames with umpires, catchers, or spectators but no pitcher).

Usage:
    python scripts/auto_label_pitchers.py [OPTIONS]

Examples:
    python scripts/auto_label_pitchers.py
    python scripts/auto_label_pitchers.py --output-subdir pitcher_labels_auto
    python scripts/auto_label_pitchers.py --min-pitcher-score 0.05
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from utils import pose_utils
from utils.auto_label_utils import CLIPPitcherSelector, build_candidates


def save_no_pitcher_label(output_dir, frame_name, reason, metadata=None):
    """Save a compatible no-pitcher JSON output."""
    payload = {
        'frame': frame_name,
        'pitcher_detected': False,
        'pitcher_person_id': None,
        'reason': reason,
        'label_source': 'clip_auto'
    }
    if metadata:
        payload['clip_selection'] = metadata

    pose_utils.save_json(payload, output_dir / 'data.json')


def extract_pitcher_output(person_data, frame_name, arm_side):
    """Build pitcher_labels JSON payload compatible with downstream scripts."""
    keypoints = np.array(person_data['keypoints'])

    if arm_side == 'right':
        shoulder_idx = pose_utils.KEYPOINT_NAMES['right_shoulder']
        elbow_idx = pose_utils.KEYPOINT_NAMES['right_elbow']
        wrist_idx = pose_utils.KEYPOINT_NAMES['right_wrist']
    else:
        shoulder_idx = pose_utils.KEYPOINT_NAMES['left_shoulder']
        elbow_idx = pose_utils.KEYPOINT_NAMES['left_elbow']
        wrist_idx = pose_utils.KEYPOINT_NAMES['left_wrist']

    shoulder = keypoints[shoulder_idx]
    elbow = keypoints[elbow_idx]
    wrist = keypoints[wrist_idx]

    shoulder_key = f'{arm_side}_shoulder'
    elbow_key = f'{arm_side}_elbow'
    wrist_key = f'{arm_side}_wrist'

    return {
        'frame': frame_name,
        'pitcher_detected': True,
        'pitcher_person_id': person_data['person_id'],
        'bbox': person_data['bbox'],
        'keypoints': person_data['keypoints'],
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
        },
        'label_source': 'clip_auto'
    }


def process_frame(frame_path, video_dir, video_id, ground_truth_data, selector,
                  min_keypoint_conf, min_upper_body_points,
                  min_hand_points, crop_padding, min_bbox_score,
                  output_subdir='pitcher_labels', min_pitcher_score=None,
                  force=False):
    """Process one frame and auto-select pitcher using CLIP pitcherness score."""
    poses_frame_name = pose_utils.format_frame_name(frame_path.name, 'poses')
    pitcher_frame_name = pose_utils.format_frame_name(frame_path.name, 'pitcher')

    if not force and pose_utils.check_output_exists(video_dir, pitcher_frame_name, output_subdir):
        return True, 'Already labeled'

    poses_dir = video_dir / 'poses' / poses_frame_name
    poses_json = poses_dir / 'data.json'
    if not poses_json.exists():
        return False, 'Poses data not found - run process_release_frames.py first'

    poses_data = pose_utils.load_json(poses_json)
    persons_data = poses_data.get('persons', [])

    output_dir = video_dir / output_subdir / pitcher_frame_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # CLIP scoring must use clean release_frame image only.
    score_image = cv2.imread(str(frame_path))
    if score_image is None:
        return False, 'Failed to load release frame image for CLIP scoring'

    if len(persons_data) == 0:
        save_no_pitcher_label(output_dir, frame_path.name, 'No persons detected in pose output')
        return 'skip', 'No persons detected'

    arm_side = ''
    if video_id in ground_truth_data:
        pitcher_hand = ground_truth_data[video_id].get('PitcherHand', 'R')
        arm_side = 'right' if str(pitcher_hand).upper() == 'R' else 'left'

    candidates = build_candidates(
        persons_data=persons_data,
        score_image=score_image,
        arm_side=arm_side,
        min_keypoint_conf=min_keypoint_conf,
        min_upper_body_points=min_upper_body_points,
        min_hand_points=min_hand_points,
        crop_padding=crop_padding,
        min_bbox_score=min_bbox_score
    )

    if not candidates:
        save_no_pitcher_label(output_dir, frame_path.name,
                              'No valid detections with required arm/hand keypoints')
        return 'skip', 'No valid candidates after skeleton filtering'

    # Score each crop by pitcherness and pick max.
    for candidate in candidates:
        candidate['pitcher_score'] = selector.score_pitcherness(candidate['crop'])

    candidates.sort(key=lambda c: c['pitcher_score'], reverse=True)
    best = candidates[0]
    best_score = best['pitcher_score']

    selection_meta = {
        'selection_mode': 'max_inner_product',
        'best_pitcher_score': best_score,
        'best_bbox_score': best['bbox_score'],
        'num_candidates': len(candidates),
        'thresholds': {
            'min_bbox_score': min_bbox_score,
            'min_pitcher_score': min_pitcher_score
        }
    }

    # Optional absolute floor. If unset, always choose top candidate.
    if min_pitcher_score is not None and best_score < min_pitcher_score:
        save_no_pitcher_label(
            output_dir,
            frame_path.name,
            'Best pitcherness score below threshold',
            metadata=selection_meta
        )
        return 'skip', f'No pitcher detected (best={best_score:.4f})'

    pitcher_person = best['person']
    output_data = extract_pitcher_output(pitcher_person, frame_path.name, arm_side)
    output_data['clip_selection'] = selection_meta
    pose_utils.save_json(output_data, output_dir / 'data.json')

    # Save selected clean crop.
    px1, py1, px2, py2 = best['crop_xyxy']
    output_crop = score_image[py1:py2, px1:px2].copy()

    out_img_path = output_dir / f'{pitcher_frame_name}.jpg'
    cv2.imwrite(str(out_img_path), output_crop)

    return True, (
        f"Labeled pitcher (person {pitcher_person['person_id'] + 1}, "
        f"score={best_score:.4f})"
    )


def process_all_videos(baseball_vids_dir, ground_truth_data, selector,
                       min_keypoint_conf, min_upper_body_points,
                       min_hand_points, crop_padding, min_bbox_score,
                       output_subdir='pitcher_labels', min_pitcher_score=None,
                       force=False):
    """Run CLIP auto-labeling over all videos and release frames."""
    video_dirs = pose_utils.get_video_dirs(baseball_vids_dir)
    if not video_dirs:
        print(f'✗ No video directories found in: {baseball_vids_dir}')
        return

    print(f"\n{'=' * 60}")
    print('AUTO LABELING PITCHERS (CLIP)')
    print(f"{'=' * 60}")
    print(f'Found {len(video_dirs)} video(s)')
    print(f'Output subdir: {output_subdir}')
    print()

    total_frames = 0
    total_processed = 0
    total_skipped = 0
    total_failed = 0

    for video_idx, (video_id, video_dir) in enumerate(video_dirs, 1):
        print(f'[{video_idx}/{len(video_dirs)}] Processing video: {video_id}')
        release_frames = pose_utils.get_release_frames(video_dir)

        if not release_frames:
            print('  No frames in release_frames/')
            print()
            continue

        print(f'  Found {len(release_frames)} frame(s) in release_frames/')
        total_frames += len(release_frames)

        video_processed = 0
        video_skipped = 0
        video_failed = 0

        for frame_idx, frame_path in enumerate(release_frames, 1):
            print(f'  [{frame_idx}/{len(release_frames)}] {frame_path.name}... ', end='')
            try:
                success, message = process_frame(
                    frame_path=frame_path,
                    video_dir=video_dir,
                    video_id=video_id,
                    ground_truth_data=ground_truth_data,
                    selector=selector,
                    min_keypoint_conf=min_keypoint_conf,
                    min_upper_body_points=min_upper_body_points,
                    min_hand_points=min_hand_points,
                    crop_padding=crop_padding,
                    min_bbox_score=min_bbox_score,
                    output_subdir=output_subdir,
                    min_pitcher_score=min_pitcher_score,
                    force=force
                )

                if success == 'skip' or (success and message == 'Already labeled'):
                    video_skipped += 1
                    total_skipped += 1
                    print(f'SKIPPED: {message}')
                elif success:
                    video_processed += 1
                    total_processed += 1
                    print(f'✓ {message}')
                else:
                    video_failed += 1
                    total_failed += 1
                    print(f'✗ {message}')
            except Exception as exc:
                video_failed += 1
                total_failed += 1
                print(f'✗ Error: {exc}')

        print(f'  Video summary: {video_processed} processed, {video_skipped} skipped, {video_failed} failed')
        print()

    print(f"{'=' * 60}")
    print('OVERALL SUMMARY')
    print(f"{'=' * 60}")
    print(f'Videos:  {len(video_dirs)}')
    print(f'Frames:  {total_frames}')
    print(f'Labeled: {total_processed}')
    print(f'Skipped: {total_skipped}')
    print(f'Failed:  {total_failed}')
    print(f"{'=' * 60}\n")


def main():
    parser = ArgumentParser(description='Automatically label pitchers using CLIP on precomputed poses')
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
        '--output-subdir',
        type=str,
        default='pitcher_labels',
        help='Output label subdirectory under each video (e.g. pitcher_labels_auto)'
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
        '--min-keypoint-confidence',
        type=float,
        default=0.20,
        help='Min confidence for keypoint validity checks'
    )
    parser.add_argument(
        '--min-upper-body-points',
        type=int,
        default=4,
        help='Min number of upper-body keypoints above confidence threshold'
    )
    parser.add_argument(
        '--min-hand-points',
        type=int,
        default=1,
        help='Min detected keypoints for the throwing hand (left/right hand set)'
    )
    parser.add_argument(
        '--crop-padding',
        type=float,
        default=0.25,
        help='Relative padding for keypoint-guided crops (default: 0.25)'
    )
    parser.add_argument(
        '--min-bbox-score',
        type=float,
        default=0.60,
        help='Minimum detector bbox confidence for candidate filtering'
    )
    parser.add_argument(
        '--min-pitcher-score',
        type=float,
        default=None,
        help='Optional absolute floor for pitcherness score (default: disabled)'
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

    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = baseball_vids_dir / 'arm_angles_high_speed.csv'

    if not csv_path.exists():
        print(f'Warning: Ground truth CSV not found: {csv_path}')
        print("Will use default 'right' arm side for all pitchers\n")
        ground_truth_data = {}
    else:
        print('Loading ground truth data...')
        try:
            ground_truth_data = pose_utils.load_ground_truth_csv(csv_path)
            print(f'✓ Loaded ground truth for {len(ground_truth_data)} videos\n')
        except Exception as exc:
            print(f'Warning: Failed to load ground truth CSV: {exc}')
            print("Will use default 'right' arm side for all pitchers\n")
            ground_truth_data = {}

    print(f'Baseball videos directory: {baseball_vids_dir}')
    print(f'Output subdir: {args.output_subdir}')
    print(f'CLIP model: {args.clip_model}')
    print(f'CLIP device: {args.clip_device}')
    print(f'Crop padding: {args.crop_padding}')
    print(f'Min hand points: {args.min_hand_points}')
    print(f'Min bbox score: {args.min_bbox_score}')
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
        min_keypoint_conf=args.min_keypoint_confidence,
        min_upper_body_points=args.min_upper_body_points,
        min_hand_points=args.min_hand_points,
        crop_padding=args.crop_padding,
        min_bbox_score=args.min_bbox_score,
        output_subdir=args.output_subdir,
        min_pitcher_score=args.min_pitcher_score,
        force=args.force
    )


if __name__ == '__main__':
    main()
