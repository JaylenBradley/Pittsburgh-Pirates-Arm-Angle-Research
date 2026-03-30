"""
Utilities for CLIP-based automatic pitcher labeling.

This module provides:
- CLIP image-text pitcherness scoring
- Candidate filtering on bbox quality
- Bbox extraction for CLIP inference

It is model-agnostic and reused by auto_label_pitchers.py.
"""

import math
import cv2


class CLIPPitcherSelector:
    """Scores crops by CLIP image-text inner product for pitcher prompts."""

    DEFAULT_PITCHER_PROMPTS = [
        "baseball pitcher's baseball glove",
        "a baseball pitcher throwing a baseball",
        "arm extended throwing a baseball"
        "baseball pitcher after release on the mound",
        "baseball pitcher throwing",
        "pitcher releasing the ball",
        "pitcher mid-delivery",
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


def build_candidates(persons_data, score_image, min_bbox_score):
    """
    Build filtered candidate list with exact bbox crops for CLIP scoring.
    
    YOLO bounding boxes are already high quality, so we apply minimal filtering:
    - Must meet minimum bbox score threshold
    
    CLIP sees the exact bounding box with NO padding.
    """
    candidates = []
    for person in persons_data:
        bbox = person.get('bbox')
        keypoints = person.get('keypoints')

        if not isinstance(bbox, dict):
            continue

        # Check bbox score threshold - use 0.0 as default (reject boxes without explicit scores)
        bbox_score = float(bbox.get('score', 0.0))
        if bbox_score < min_bbox_score:
            continue

        # Extract exact bbox crop (no padding) for CLIP
        x1, y1, x2, y2 = clamp_bbox_to_image(bbox, score_image.shape)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = score_image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        candidates.append({
            'person': person,
            'bbox_score': bbox_score,
            'crop_xyxy': (x1, y1, x2, y2),
            'crop': crop
        })

    # Sort by YOLO bbox confidence (highest first)
    # This ensures CLIP only sees the most confident detections
    candidates.sort(key=lambda c: c['bbox_score'], reverse=True)
    
    return candidates