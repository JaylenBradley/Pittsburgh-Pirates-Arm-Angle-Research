"""
Utilities for CLIP-based automatic pitcher labeling with track aggregation.

This module provides:
- CLIP image-text pitcherness scoring with cached embeddings
- Track probability aggregation across multiple frames
- Spatial prior (bottom-center bias) for pitcher detection
- Candidate filtering on bbox quality
- Multi-frame batch scoring and labeling

It is model-agnostic and reused by auto_label_pitchers.py.
"""

import math
import cv2
import numpy as np


class CLIPPitcherSelector:
    """Scores crops by CLIP image-text inner product for pitcher prompts with spatial priors."""

    DEFAULT_PITCHER_PROMPTS = [
        "baseball pitcher's baseball glove",
        "a baseball pitcher throwing a baseball",
        "arm extended throwing a baseball"
        "baseball pitcher after release on the mound",
        "baseball pitcher throwing",
        "pitcher releasing the ball",
        "pitcher mid-delivery",
    ]
    
    # Spatial prior configuration for pitcher (bottom-center bias)
    SPATIAL_PRIOR_CENTER_X = 0.5      # Center horizontally
    SPATIAL_PRIOR_CENTER_Y = 0.70     # Lower vertically (70% down)
    SPATIAL_PRIOR_SIGMA_X = 0.33      # 1/3 width
    SPATIAL_PRIOR_SIGMA_Y = 0.33      # 1/3 height
    SPATIAL_PRIOR_WEIGHT = 1.0        # Weight relative to CLIP scores

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

        # Use provided prompts or default set
        if pitcher_prompts is None:
            pitcher_prompts = self.DEFAULT_PITCHER_PROMPTS
        self.pitcher_prompts = pitcher_prompts

        # Cache text embeddings (computed once)
        self._pitcher_text_embedding = self._encode_pitcher_texts()

    def _resolve_device(self, device):
        device = device.lower()
        if device != 'auto':
            return self.torch.device(device)

        if self.torch.cuda.is_available():
            return self.torch.device('cuda')
        if hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
            return self.torch.device('mps')
        return self.torch.device('cpu')

    def _encode_pitcher_texts(self):
        """Encode all pitcher prompts once and average them, then normalize."""
        with self.torch.no_grad():
            embeddings = []
            for prompt in self.pitcher_prompts:
                text_inputs = self.processor(text=[prompt], return_tensors='pt', padding=True)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                text_features = self.model.get_text_features(**text_inputs)
                embeddings.append(text_features)
            
            # Average all prompt embeddings
            combined = self.torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
            combined = combined / combined.norm(dim=-1, keepdim=True)
        
        return combined

    def score_pitcherness(self, crop_bgr):
        """
        Return CLIP pitcherness score for a single crop (0-1 range).
        Score is inner product between image and averaged pitcher text embeddings.
        """
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        with self.torch.no_grad():
            inputs = self.processor(images=crop_rgb, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Inner product gives similarity in [-1, 1], map to [0, 1]
            score = float((image_features @ self._pitcher_text_embedding.T).item())
            score = (score + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        
        return score


def clamp_bbox_to_image(bbox, image_shape):
    """Clamp bbox coordinates to image bounds and return integer xyxy."""
    height, width = image_shape[:2]

    x1 = int(max(0, min(width - 1, math.floor(bbox['x1']))))
    y1 = int(max(0, min(height - 1, math.floor(bbox['y1']))))
    x2 = int(max(0, min(width, math.ceil(bbox['x2']))))
    y2 = int(max(0, min(height, math.ceil(bbox['y2']))))

    return x1, y1, x2, y2


def add_relative_padding(x1, y1, x2, y2, image_shape, pad_ratio=0.15):
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
    Build filtered candidate list with crops for CLIP scoring.
    
    CLIP sees the exact bounding box with NO padding.
    """
    candidates = []
    for person in persons_data:
        bbox = person.get('bbox')

        if not isinstance(bbox, dict):
            continue

        # Check bbox score threshold
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
    candidates.sort(key=lambda c: c['bbox_score'], reverse=True)
    
    return candidates


def score_candidates_batch(candidates, selector, image_shape):
    """
    Score all candidates and return them with pitcherness scores.
    
    Returns:
        List of candidates with 'pitcher_score' added
    """
    for candidate in candidates:
        crop = candidate['crop']
        candidate['pitcher_score'] = selector.score_pitcherness(crop)
        candidate['track_id'] = candidate['person'].get('track_id')
    
    return candidates


def apply_spatial_prior(candidates, bbox_centers, image_shape):
    """
    Apply spatial prior (bottom-center bias) to CLIP scores.
    
    Returns:
        Adjusted scores with spatial prior applied
    """
    height, width = image_shape[:2]
    center_x = width * CLIPPitcherSelector.SPATIAL_PRIOR_CENTER_X
    center_y = height * CLIPPitcherSelector.SPATIAL_PRIOR_CENTER_Y
    sigma_x = width * CLIPPitcherSelector.SPATIAL_PRIOR_SIGMA_X
    sigma_y = height * CLIPPitcherSelector.SPATIAL_PRIOR_SIGMA_Y
    
    adjusted_scores = []
    for candidate, (bbox_cx, bbox_cy) in zip(candidates, bbox_centers):
        base_score = candidate['pitcher_score']
        
        # Gaussian spatial prior (higher weight for bottom-center)
        if sigma_x > 0 and sigma_y > 0:
            spatial_weight = math.exp(
                -0.5 * (((bbox_cx - center_x) / sigma_x) ** 2 + 
                         ((bbox_cy - center_y) / sigma_y) ** 2)
            )
        else:
            spatial_weight = 1.0
        
        spatial_boost = CLIPPitcherSelector.SPATIAL_PRIOR_WEIGHT * spatial_weight
        adjusted_score = base_score + spatial_boost
        adjusted_scores.append({
            'candidate': candidate,
            'base_score': base_score,
            'spatial_weight': spatial_weight,
            'spatial_boost': spatial_boost,
            'adjusted_score': adjusted_score
        })
    
    return adjusted_scores


def aggregate_track_scores(frames_data):
    """
    Aggregate CLIP scores across all frames by track_id.
    
    Frames are scored independently, then we aggregate by track.
    For each track, we calculate: sum(scores) / num_frames_where_track_appears
    
    Args:
        frames_data: List of dicts, each with 'frame_name', 'candidates_scored'
    
    Returns:
        Dict mapping track_id -> aggregated score info
    """
    track_stats = {}
    
    for frame_info in frames_data:
        candidates = frame_info.get('candidates_scored', [])
        
        for scored in candidates:
            candidate = scored['candidate']
            track_id = candidate.get('track_id')
            
            if track_id is None:
                continue
            
            track_id = int(track_id)
            
            if track_id not in track_stats:
                track_stats[track_id] = {
                    'sum_base': 0.0,
                    'sum_adjusted': 0.0,
                    'count': 0,
                    'appearances': []
                }
            
            track_stats[track_id]['sum_base'] += scored['base_score']
            track_stats[track_id]['sum_adjusted'] += scored['adjusted_score']
            track_stats[track_id]['count'] += 1
            track_stats[track_id]['appearances'].append({
                'frame': frame_info['frame_name'],
                'base_score': scored['base_score'],
                'spatial_weight': scored['spatial_weight'],
                'spatial_boost': scored['spatial_boost'],
                'adjusted_score': scored['adjusted_score']
            })
    
    # Compute aggregated scores
    aggregated = {}
    for track_id, stats in track_stats.items():
        count = stats['count']
        aggregated[track_id] = {
            'mean_base_score': stats['sum_base'] / count if count > 0 else 0.0,
            'mean_adjusted_score': stats['sum_adjusted'] / count if count > 0 else 0.0,
            'num_appearances': count,
            'appearances_per_frame': stats['appearances']
        }
    
    return aggregated
