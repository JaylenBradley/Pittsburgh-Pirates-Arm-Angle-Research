"""
Utilities for CLIP-based automatic pitcher labeling.

This module centralizes:
- CLIP image-text pitcherness scoring
- Candidate filtering on keypoint/bbox quality
- Keypoint-guided crop generation

It is intentionally model-agnostic and reused by auto_label_pitchers.py.
"""

import math

import cv2
import numpy as np

from . import crop_utils, pose_utils

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
UPPER_BODY_KEYPOINT_INDICES = [
    pose_utils.KEYPOINT_NAMES.get(name)
    for name in UPPER_BODY_KEYPOINT_NAMES
    if pose_utils.KEYPOINT_NAMES.get(name) is not None
]

# COCO-WholeBody hand ranges in the 133-keypoint format.
LEFT_HAND_RANGE = range(91, 112)
RIGHT_HAND_RANGE = range(112, 133)


class CLIPPitcherSelector:
    """Scores crops by CLIP image-text inner product for pitcher prompts."""

    # Hardcoded multiple pitcher prompts to improve detection accuracy.
    DEFAULT_PITCHER_PROMPTS = [
        "baseball pitcher's baseball glove",
        "a baseball pitcher throwing a baseball",
        "arm extended throwing a baseball"
    ]

    def __init__(self, model_name='openai/clip-vit-base-patch32', device='auto',
                 pitcher_prompts=None):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                'transformers/torch are required for CLIP auto-labeling. '
                'Install with: pip install transformers torch'
            ) from exc

        self.torch = torch
        self.device = self._resolve_device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Use provided prompts or default set.
        if pitcher_prompts is None:
            pitcher_prompts = self.DEFAULT_PITCHER_PROMPTS
        self.pitcher_prompts = pitcher_prompts

        # Encode all pitcher prompts.
        self._pitcher_text_features = [
            self._encode_text(prompt) for prompt in pitcher_prompts
        ]

    def _resolve_device(self, device):
        device = device.lower()
        if device != 'auto':
            return self.torch.device(device)

        if self.torch.cuda.is_available():
            return self.torch.device('cuda')
        if hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
            return self.torch.device('mps')
        return self.torch.device('cpu')

    def _encode_text(self, prompt):
        with self.torch.no_grad():
            text_inputs = self.processor(text=[prompt], return_tensors='pt', padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def score_pitcherness(self, crop_bgr):
        """Return max inner product across all pitcher text embeddings."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        with self.torch.no_grad():
            inputs = self.processor(images=crop_rgb, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute score with each prompt and take max.
            scores = []
            for text_feature in self._pitcher_text_features:
                score = float((image_features @ text_feature.T).item())
                scores.append(score)

            # Return max score across all prompts.
            score = max(scores)
        return score


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


def keypoint_is_valid(person_keypoints, idx, min_conf):
    """Return True when keypoint index exists and has valid x/y/conf."""
    if idx >= len(person_keypoints):
        return False
    kp = person_keypoints[idx]
    if not isinstance(kp, (list, tuple)) or len(kp) < 3:
        return False
    x, y, conf = kp[0], kp[1], kp[2]
    return x is not None and y is not None and conf is not None and conf >= min_conf


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


def has_valid_upper_body_keypoints(person_keypoints, min_conf=0.25, min_points=4):
    """Return True if enough upper-body keypoints are present above threshold."""
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


def has_valid_throwing_arm_and_hand(person_keypoints, arm_side, min_conf=0.20,
                                    min_hand_points=1):
    """Require shoulder-elbow-wrist chain and at least N hand keypoints."""
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
    """Build tighter crop from torso/arm/hand keypoints; fall back to bbox."""
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

    if len(points) < 4:
        x1, y1, x2, y2 = clamp_bbox_to_image(fallback_bbox, image_shape)
        return add_relative_padding(
            x1, y1, x2, y2, image_shape,
            pad_ratio=max(0.05, pad_ratio * 0.5)
        )

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = int(math.floor(min(xs)))
    y1 = int(math.floor(min(ys)))
    x2 = int(math.ceil(max(xs)))
    y2 = int(math.ceil(max(ys)))

    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1

    bbox_for_crop = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    cx1, cy1, cx2, cy2 = clamp_bbox_to_image(bbox_for_crop, image_shape)
    return add_relative_padding(cx1, cy1, cx2, cy2, image_shape, pad_ratio=pad_ratio)


def build_candidates(persons_data, score_image, arm_side,
                     min_keypoint_conf, min_upper_body_points,
                     min_hand_points, crop_padding, min_bbox_score):
    """Build filtered candidate list with crops ready for CLIP scoring."""
    filtered_persons, _ = crop_utils.filter_persons_by_keypoints(
        persons_data,
        confidence_threshold=min_keypoint_conf
    )

    candidates = []
    for person in filtered_persons:
        bbox = person.get('bbox')
        keypoints = person.get('keypoints')

        if not isinstance(bbox, dict):
            continue
        if not has_any_valid_keypoints(keypoints, min_conf=min_keypoint_conf):
            continue

        bbox_score = float(bbox.get('score', 1.0))
        if bbox_score < min_bbox_score:
            continue

        if not has_valid_upper_body_keypoints(
                keypoints,
                min_conf=min_keypoint_conf,
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

        candidates.append({
            'person': person,
            'bbox_score': bbox_score,
            'crop_xyxy': (px1, py1, px2, py2),
            'crop': crop
        })

    return candidates

