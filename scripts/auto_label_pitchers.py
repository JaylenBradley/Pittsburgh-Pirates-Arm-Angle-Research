"""
Auto Label Pitchers Script (CLIP-based) for Baseball Pitcher Pose Analysis

This script automatically labels the pitcher for each frame using precomputed
pose detections from process_release_frames.py.

For each frame in release_frames/, it:
1. Loads all detected persons from poses/FRAME_NAME_poses/data.json
2. Filters out low-quality detections (bad bbox or insufficient upper-body keypoints)
3. Crops each valid person with 10% bbox padding
4. Scores each crop with CLIP using pitcher-focused text prompts
5. Selects the top pitcher candidate or marks frame as no pitcher detected
6. Saves output to pitcher_labels/FRAME_NAME_pitcher/data.json

The output JSON structure is kept compatible with calculate_pitcher_angles.py.

Usage:
    From Pirates_Arm_Angle directory:
    python scripts/auto_label_pitchers.py [OPTIONS]

Arguments:
    --videos-dir PATH: Optional path to baseball_vids directory
                       (default: ~/Desktop/baseball_vids)
    --csv PATH: Path to ground truth CSV file
                (default: baseball_vids/arm_angles_high_speed.csv)
    --force: Force reprocessing of already-labeled frames
    --clip-model NAME: Hugging Face CLIP model name
                       (default: openai/clip-vit-base-patch32)
    --clip-device DEVICE: Device for CLIP inference (cpu, cuda, mps, auto)
                          (default: auto)
    --min-keypoint-confidence FLOAT: Minimum confidence for upper-body keypoint validity
                                     (default: 0.20)
    --min-upper-body-points INT: Min number of upper-body keypoints required
                                 (default: 4)
    --min-pitcher-score FLOAT: Min CLIP pitcher probability threshold
                               (default: 0.55)

Examples:
    python scripts/auto_label_pitchers.py
    python scripts/auto_label_pitchers.py --force
    python scripts/auto_label_pitchers.py --clip-device cpu
"""

import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from scripts.utils import pose_utils

# Upper-body keypoints used to validate whether a bbox likely contains a person
# with usable skeleton data.
UPPER_BODY_KEYPOINT_NAMES = [
    'nose',
    'left_eye', 'right_eye',
    'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip'
]
UPPER_BODY_KEYPOINT_INDICES = [pose_utils.KEYPOINT_NAMES.get(name) for name in UPPER_BODY_KEYPOINT_NAMES
                               if pose_utils.KEYPOINT_NAMES.get(name) is not None]

# COCO-WholeBody hand ranges in the 133-keypoint format.
LEFT_HAND_RANGE = range(91, 112)
RIGHT_HAND_RANGE = range(112, 133)


class CLIPPitcherSelector:
    """Selects the most likely pitcher candidate from person crops using CLIP."""

    PITCHER_PROMPTS = [
        "an image of a baseball pitcher winding up to throw",
        "an image of a baseball player pitching a baseball",
        "an image of a pitcher in a baseball game"
    ]

    NON_PITCHER_PROMPTS = [
        "an image of a baseball umpire",
        "an image of a baseball batter",
        "an image of a baseball catcher",
        "an image of a baseball player's legs only",
        "an image of a cropped lower body and legs"
    ]

    def __init__(self, model_name='openai/clip-vit-base-patch32', device='auto'):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers/torch are required for CLIP auto-labeling. "
                "Install with: pip install transformers torch"
            ) from exc

        self.torch = torch
        self.device = self._resolve_device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self._pitcher_text_features = self._encode_texts(self.PITCHER_PROMPTS)
        self._non_pitcher_text_features = self._encode_texts(self.NON_PITCHER_PROMPTS)

    def _resolve_device(self, device):
        device = device.lower()
        if device != 'auto':
            return self.torch.device(device)

        if self.torch.cuda.is_available():
            return self.torch.device('cuda')
        if hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
            return self.torch.device('mps')
        return self.torch.device('cpu')

    def _encode_texts(self, prompts):
        with self.torch.no_grad():
            text_inputs = self.processor(text=prompts, return_tensors='pt', padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def score_crop(self, crop_bgr):
        """Score a crop as pitcher vs non-pitcher with one absolute threshold."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        with self.torch.no_grad():
            inputs = self.processor(images=crop_rgb, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            pitcher_sims = image_features @ self._pitcher_text_features.T
            pitcher_sim = pitcher_sims.mean().item()
            non_pitcher_sims = image_features @ self._non_pitcher_text_features.T
            non_pitcher_sim = non_pitcher_sims.mean().item()

            logits = self.torch.tensor([pitcher_sim, non_pitcher_sim], device=self.device) * 10.0
            probs = self.torch.softmax(logits, dim=0)
            pitcher_prob = float(probs[0].item())

        return {
            'pitcher_prob': pitcher_prob,
            'pitcher_sim': float(pitcher_sim),
            'non_pitcher_sim': float(non_pitcher_sim)
        }


def clamp_bbox_to_image(bbox, image_shape):
    """Clamp bbox coordinates to image bounds and return integer xyxy."""
    height, width = image_shape[:2]

    x1 = int(max(0, min(width - 1, math.floor(bbox['x1']))))
    y1 = int(max(0, min(height - 1, math.floor(bbox['y1']))))
    x2 = int(max(0, min(width, math.ceil(bbox['x2']))))
    y2 = int(max(0, min(height, math.ceil(bbox['y2']))))

    return x1, y1, x2, y2


def add_relative_padding(x1, y1, x2, y2, image_shape, pad_ratio=0.10):
    """Expand bbox by a relative padding ratio and clamp to image bounds."""
    height, width = image_shape[:2]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    pad_x = int(round(box_w * pad_ratio))
    pad_y = int(round(box_h * pad_ratio))

    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(width, x2 + pad_x)
    py2 = min(height, y2 + pad_y)

    return px1, py1, px2, py2


def has_valid_upper_body_keypoints(person_keypoints, min_conf=0.25, min_points=4):
    """Return True if enough upper-body keypoints are present above confidence threshold."""
    if not isinstance(person_keypoints, list) or len(person_keypoints) < 17:
        return False

    valid_count = 0
    for idx in UPPER_BODY_KEYPOINT_INDICES:
        if idx >= len(person_keypoints):
            continue

        kp = person_keypoints[idx]
        if not isinstance(kp, (list, tuple)) or len(kp) < 3:
            continue

        x, y, conf = kp[0], kp[1], kp[2]
        if conf is None:
            continue
        if conf >= min_conf and x is not None and y is not None:
            valid_count += 1

    return valid_count >= min_points


def has_any_valid_keypoints(person_keypoints, min_conf=0.05):
    """Return True if at least one keypoint has usable x/y/conf values."""
    if not isinstance(person_keypoints, list):
        return False
    for kp in person_keypoints:
        if not isinstance(kp, (list, tuple)) or len(kp) < 3:
            continue
        x, y, conf = kp[0], kp[1], kp[2]
        if x is not None and y is not None and conf is not None and conf >= min_conf:
            return True
    return False


def keypoint_is_valid(person_keypoints, idx, min_conf):
    """Return True when keypoint index exists and has valid x/y/conf."""
    if idx >= len(person_keypoints):
        return False
    kp = person_keypoints[idx]
    if not isinstance(kp, (list, tuple)) or len(kp) < 3:
        return False
    x, y, conf = kp[0], kp[1], kp[2]
    return x is not None and y is not None and conf is not None and conf >= min_conf


def has_valid_throwing_arm_and_hand(person_keypoints, arm_side, min_conf=0.20,
                                    min_hand_points=1):
    """Require shoulder-elbow-wrist chain and at least N hand keypoints for arm side."""
    if not isinstance(person_keypoints, list) or len(person_keypoints) < 17:
        return False

    if arm_side == 'right':
        shoulder_idx = pose_utils.KEYPOINT_NAMES['right_shoulder']
        elbow_idx = pose_utils.KEYPOINT_NAMES['right_elbow']
        wrist_idx = pose_utils.KEYPOINT_NAMES['right_wrist']
        hand_range = RIGHT_HAND_RANGE
    else:
        shoulder_idx = pose_utils.KEYPOINT_NAMES['left_shoulder']
        elbow_idx = pose_utils.KEYPOINT_NAMES['left_elbow']
        wrist_idx = pose_utils.KEYPOINT_NAMES['left_wrist']
        hand_range = LEFT_HAND_RANGE

    if not keypoint_is_valid(person_keypoints, shoulder_idx, min_conf):
        return False
    if not keypoint_is_valid(person_keypoints, elbow_idx, min_conf):
        return False
    if not keypoint_is_valid(person_keypoints, wrist_idx, min_conf):
        return False

    valid_hand = 0
    for idx in hand_range:
        if keypoint_is_valid(person_keypoints, idx, min_conf):
            valid_hand += 1
            if valid_hand >= min_hand_points:
                return True

    return False


def has_valid_torso_geometry(person_keypoints, min_conf=0.20):
    """Require shoulder-above-hip torso geometry to reject leg-only boxes."""
    ls = pose_utils.KEYPOINT_NAMES['left_shoulder']
    rs = pose_utils.KEYPOINT_NAMES['right_shoulder']
    lh = pose_utils.KEYPOINT_NAMES['left_hip']
    rh = pose_utils.KEYPOINT_NAMES['right_hip']

    shoulders = [idx for idx in (ls, rs) if keypoint_is_valid(person_keypoints, idx, min_conf)]
    hips = [idx for idx in (lh, rh) if keypoint_is_valid(person_keypoints, idx, min_conf)]
    if len(shoulders) == 0 or len(hips) == 0:
        return False

    shoulder_y = float(np.mean([person_keypoints[idx][1] for idx in shoulders]))
    hip_y = float(np.mean([person_keypoints[idx][1] for idx in hips]))
    return shoulder_y < hip_y


def build_keypoint_guided_crop(person_keypoints, arm_side, image_shape, min_conf,
                               pad_ratio, fallback_bbox):
    """Build tighter crop from torso/arm/hand keypoints; fall back to detector bbox."""
    if arm_side == 'right':
        arm_ids = [
            pose_utils.KEYPOINT_NAMES['right_shoulder'],
            pose_utils.KEYPOINT_NAMES['right_elbow'],
            pose_utils.KEYPOINT_NAMES['right_wrist']
        ]
        hand_ids = list(RIGHT_HAND_RANGE)
    else:
        arm_ids = [
            pose_utils.KEYPOINT_NAMES['left_shoulder'],
            pose_utils.KEYPOINT_NAMES['left_elbow'],
            pose_utils.KEYPOINT_NAMES['left_wrist']
        ]
        hand_ids = list(LEFT_HAND_RANGE)

    torso_ids = [
        pose_utils.KEYPOINT_NAMES['left_shoulder'],
        pose_utils.KEYPOINT_NAMES['right_shoulder'],
        pose_utils.KEYPOINT_NAMES['left_hip'],
        pose_utils.KEYPOINT_NAMES['right_hip']
    ]

    points = []
    for idx in arm_ids + torso_ids + hand_ids:
        if keypoint_is_valid(person_keypoints, idx, min_conf):
            points.append((float(person_keypoints[idx][0]), float(person_keypoints[idx][1])))

    # Fallback when there are not enough reliable points.
    if len(points) < 4:
        x1, y1, x2, y2 = clamp_bbox_to_image(fallback_bbox, image_shape)
        return add_relative_padding(x1, y1, x2, y2, image_shape, pad_ratio=max(0.05, pad_ratio * 0.5))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = int(math.floor(min(xs)))
    y1 = int(math.floor(min(ys)))
    x2 = int(math.ceil(max(xs)))
    y2 = int(math.ceil(max(ys)))

    # Ensure non-zero region then pad and clamp.
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1

    bbox_for_crop = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    cx1, cy1, cx2, cy2 = clamp_bbox_to_image(bbox_for_crop, image_shape)
    return add_relative_padding(cx1, cy1, cx2, cy2, image_shape, pad_ratio=pad_ratio)


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
                  min_pitcher_score, min_hand_points, crop_padding, min_bbox_score,
                  force=False):
    """Process one frame and auto-select pitcher using CLIP."""
    poses_frame_name = pose_utils.format_frame_name(frame_path.name, 'poses')
    pitcher_frame_name = pose_utils.format_frame_name(frame_path.name, 'pitcher')

    if not force and pose_utils.check_output_exists(video_dir, pitcher_frame_name, 'pitcher_labels'):
        return True, 'Already labeled'

    poses_dir = video_dir / 'poses' / poses_frame_name
    poses_json = poses_dir / 'data.json'
    if not poses_json.exists():
        return False, 'Poses data not found - run process_release_frames.py first'

    poses_data = pose_utils.load_json(poses_json)
    persons_data = poses_data.get('persons', [])

    output_dir = video_dir / 'pitcher_labels' / pitcher_frame_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image used for CLIP scoring. Fall back to pose visualization if needed.
    score_image = cv2.imread(str(frame_path))
    if score_image is None:
        poses_vis_path = poses_dir / f'{poses_frame_name}.jpg'
        score_image = cv2.imread(str(poses_vis_path))
    if score_image is None:
        return False, 'Failed to load frame image for CLIP scoring'

    # Save the clean crop so CLIP sees and we store the same image content.
    output_vis = score_image.copy()

    if len(persons_data) == 0:
        save_no_pitcher_label(output_dir, frame_path.name, 'No persons detected in pose output')
        return 'skip', 'No persons detected'

    # Determine arm side from CSV pitcher hand when available.
    arm_side = 'right'
    if video_id in ground_truth_data:
        pitcher_hand = ground_truth_data[video_id].get('PitcherHand', 'R')
        arm_side = 'right' if str(pitcher_hand).upper() == 'R' else 'left'

    candidates = []
    for person in persons_data:
        bbox = person.get('bbox')
        keypoints = person.get('keypoints')

        if not isinstance(bbox, dict):
            continue
        if not has_any_valid_keypoints(keypoints, min_conf=min_keypoint_conf):
            continue

        bbox_score = float(bbox.get('score', 1.0))
        if bbox_score < min_bbox_score:
            continue

        if not has_valid_upper_body_keypoints(keypoints, min_conf=min_keypoint_conf,
                                              min_points=min_upper_body_points):
            continue
        if not has_valid_torso_geometry(keypoints, min_conf=min_keypoint_conf):
            continue
        if not has_valid_throwing_arm_and_hand(
                keypoints,
                arm_side=arm_side,
                min_conf=min_keypoint_conf,
                min_hand_points=min_hand_points):
            continue

        px1, py1, px2, py2 = build_keypoint_guided_crop(
            keypoints,
            arm_side=arm_side,
            image_shape=score_image.shape,
            min_conf=min_keypoint_conf,
            pad_ratio=crop_padding,
            fallback_bbox=bbox,
        )
        if px2 <= px1 or py2 <= py1:
            continue
        crop = score_image[py1:py2, px1:px2]
        if crop.size == 0:
            continue

        scores = selector.score_crop(crop)
        candidates.append({
            'person': person,
            'bbox_score': bbox_score,
            'crop_xyxy': (px1, py1, px2, py2),
            'scores': scores
        })

    if not candidates:
        save_no_pitcher_label(output_dir, frame_path.name,
                              'No valid detections with required arm/hand keypoints')
        return 'skip', 'No valid candidates after skeleton filtering'

    candidates.sort(key=lambda c: c['scores']['pitcher_prob'], reverse=True)
    best = candidates[0]
    best_prob = best['scores']['pitcher_prob']
    best_sim = best['scores']['pitcher_sim']
    best_non_sim = best['scores']['non_pitcher_sim']

    selection_meta = {
        'best_pitcher_prob': best_prob,
        'best_pitcher_similarity': best_sim,
        'best_non_pitcher_similarity': best_non_sim,
        'best_bbox_score': best['bbox_score'],
        'num_candidates': len(candidates),
        'thresholds': {
            'min_pitcher_score': min_pitcher_score,
            'min_bbox_score': min_bbox_score
        }
    }

    if best_prob < min_pitcher_score:
        save_no_pitcher_label(
            output_dir,
            frame_path.name,
            'CLIP confidence below pitcher threshold',
            metadata=selection_meta
        )
        return 'skip', f"No pitcher detected (best={best_prob:.3f})"

    pitcher_person = best['person']

    output_data = extract_pitcher_output(pitcher_person, frame_path.name, arm_side)
    output_data['clip_selection'] = selection_meta
    pose_utils.save_json(output_data, output_dir / 'data.json')

    # Save cropped pitcher image from the clean frame.
    px1, py1, px2, py2 = best['crop_xyxy']
    output_crop = output_vis[py1:py2, px1:px2].copy()

    out_img_path = output_dir / f'{pitcher_frame_name}.jpg'
    cv2.imwrite(str(out_img_path), output_crop)

    return True, (
        f"Labeled pitcher (person {pitcher_person['person_id'] + 1}, "
        f"score={best_prob:.3f})"
    )


def process_all_videos(baseball_vids_dir, ground_truth_data, selector,
                       min_keypoint_conf, min_upper_body_points,
                       min_pitcher_score, min_hand_points, crop_padding, min_bbox_score,
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
                    min_pitcher_score=min_pitcher_score,
                    min_hand_points=min_hand_points,
                    crop_padding=crop_padding,
                    min_bbox_score=min_bbox_score,
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
        help='Min confidence for upper-body keypoint validity'
    )
    parser.add_argument(
        '--min-upper-body-points',
        type=int,
        default=4,
        help='Min number of upper-body keypoints above confidence threshold'
    )
    parser.add_argument(
        '--min-pitcher-score',
        type=float,
        default=0.45,
        help='Min CLIP pitcher probability to select a candidate'
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
        default=0.10,
        help='Relative padding for keypoint-guided crops (default: 0.10)'
    )
    parser.add_argument(
        '--min-bbox-score',
        type=float,
        default=0.60,
        help='Minimum detector bbox confidence for candidate filtering'
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
    print(f'CLIP model: {args.clip_model}')
    print(f'CLIP device: {args.clip_device}')
    print(f'Crop padding: {args.crop_padding}')
    print(f'Min hand points: {args.min_hand_points}')
    print(f'Min bbox score: {args.min_bbox_score}')
    print(f'Force reprocessing: {args.force}\n')

    print('Loading CLIP model...')
    try:
        selector = CLIPPitcherSelector(model_name=args.clip_model, device=args.clip_device)
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
        min_pitcher_score=args.min_pitcher_score,
        min_hand_points=args.min_hand_points,
        crop_padding=args.crop_padding,
        min_bbox_score=args.min_bbox_score,
        force=args.force
    )


if __name__ == '__main__':
    main()
