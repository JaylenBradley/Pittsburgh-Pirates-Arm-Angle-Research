"""
Crop and Filter Utility for Baseball Pitcher Labeling

Provides shared utilities for:
1. Filtering detections based on valid skeleton keypoints
2. Extracting and padding crops of detected persons
3. Optional visualization overlays for manual labeling
4. Creating tiled displays of candidates

These utilities are used by both manual (label_pitchers.py) and
automatic (auto_label_pitchers.py) labeling pipelines.

Usage:
    from utils import crop_utils
    
    # Filter detections
    filtered_persons = crop_utils.filter_persons_by_keypoints(persons_data)
    
    # Extract crop (without overlay)
    crop_img = crop_utils.extract_person_crop(
        image, 
        person_data, 
        padding_percent=10,
        draw_overlay=False
    )
    
    # Create tiled display
    display, cols = crop_utils.create_tiled_display_from_crops(
        candidates,
        tile_size=300
    )
"""

import cv2

# Body keypoint indices (COCO-WholeBody format, indices 0-16)
BODY_KEYPOINTS = {
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

# Hand keypoint indices (COCO-WholeBody format)
LEFT_HAND_KEYPOINTS = list(range(91, 112))  # 91-111
RIGHT_HAND_KEYPOINTS = list(range(112, 133))  # 112-132

# Upper body and arm keypoints for visualization
VISUALIZATION_KEYPOINTS = {
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'nose': 0,
}


def has_valid_body_keypoints(keypoints, confidence_threshold=0.2):
    """
    Check if a person has at least one valid body keypoint.
    
    Body keypoints are indices 0-16 (COCO-WholeBody format).
    A keypoint is valid if its confidence score exceeds the threshold.
    
    Args:
        keypoints: List of [x, y, confidence] for each keypoint (133 total)
        confidence_threshold: Minimum confidence score (default: 0.2)
    
    Returns:
        Boolean indicating if person has at least one valid body keypoint
    """
    # Check only body keypoints (0-16)
    for idx in range(17):
        if idx < len(keypoints):
            kpt = keypoints[idx]
            confidence = kpt[2] if len(kpt) > 2 else 0.0
            if confidence > confidence_threshold:
                return True
    return False


def has_valid_hand_keypoints(keypoints, confidence_threshold=0.2):
    """
    Check if a person has at least one valid hand keypoint.
    
    Hand keypoints are indices 91-132 (COCO-WholeBody format).
    Checks both left (91-111) and right (112-132) hands.
    
    Args:
        keypoints: List of [x, y, confidence] for each keypoint (133 total)
        confidence_threshold: Minimum confidence score (default: 0.2)
    
    Returns:
        Boolean indicating if person has at least one valid hand keypoint
    """
    hand_indices = LEFT_HAND_KEYPOINTS + RIGHT_HAND_KEYPOINTS
    
    for idx in hand_indices:
        if idx < len(keypoints):
            kpt = keypoints[idx]
            confidence = kpt[2] if len(kpt) > 2 else 0.0
            if confidence > confidence_threshold:
                return True
    return False


def filter_persons_by_keypoints(persons_data, confidence_threshold=0.2, require_hands=True):
    """
    Filter persons list to keep only those with:
    - At least one valid body keypoint (0-16)
    - At least one valid hand keypoint (91-132) if require_hands=True
    
    Preserves original person_id and array position for index mapping.
    
    Args:
        persons_data: List of person dictionaries from poses/data.json
        confidence_threshold: Minimum keypoint confidence (default: 0.2)
        require_hands: Whether to require hand keypoints (default: True for VitPose, 
                      set to False for YOLO which only has 17 body keypoints)
    
    Returns:
        Tuple of:
        - filtered_persons: List of valid person dicts with original_index field added
        - original_to_filtered_map: Dict mapping original index -> filtered index
    """
    filtered_persons = []
    original_to_filtered_map = {}
    
    for original_idx, person in enumerate(persons_data):
        keypoints = person.get('keypoints', [])
        
        # Check if we have body keypoints
        has_body = has_valid_body_keypoints(keypoints, confidence_threshold)
        if not has_body:
            continue
        
        # If require_hands=True, also check for hand keypoints
        if require_hands:
            has_hands = has_valid_hand_keypoints(keypoints, confidence_threshold)
            if not has_hands:
                continue
        
        # This person passed all checks
        person_copy = person.copy()
        person_copy['original_index'] = original_idx
        filtered_persons.append(person_copy)
        original_to_filtered_map[original_idx] = len(filtered_persons) - 1
    
    return filtered_persons, original_to_filtered_map


def extract_person_crop(image, person_data, padding_percent=25, draw_overlay=False):
    """
    Extract a cropped region around a detected person.
    
    Args:
        image: Full frame image (BGR numpy array)
        person_data: Person dict with bbox and keypoints from poses/data.json
        padding_percent: Padding percentage around bbox (default: 25%)
        draw_overlay: If True, draw red bbox outline and keypoint markers (for manual inspection only)
    
    Returns:
        Cropped image region (BGR numpy array)
    """
    bbox = person_data['bbox']
    x1 = int(bbox['x1'])
    y1 = int(bbox['y1'])
    x2 = int(bbox['x2'])
    y2 = int(bbox['y2'])
    
    # Store original bbox for overlay (before padding)
    orig_x1, orig_y1, orig_x2, orig_y2 = x1, y1, x2, y2
    
    # Calculate padding
    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * padding_percent / 100)
    pad_y = int(height * padding_percent / 100)
    
    # Apply padding and clamp to image bounds
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image.shape[1], x2 + pad_x)
    y2 = min(image.shape[0], y2 + pad_y)
    
    # Extract crop
    crop = image[y1:y2, x1:x2].copy()
    
    # Draw overlay if requested (only for manual inspection)
    if draw_overlay:
        # Draw red bounding box outline (original bbox before padding)
        # Adjust to crop coordinate space
        bbox_x1 = max(0, orig_x1 - x1)
        bbox_y1 = max(0, orig_y1 - y1)
        bbox_x2 = min(crop.shape[1], orig_x2 - x1)
        bbox_y2 = min(crop.shape[0], orig_y2 - y1)
        
        cv2.rectangle(crop, (int(bbox_x1), int(bbox_y1)), (int(bbox_x2), int(bbox_y2)),
                      (0, 0, 255), 2)  # Red in BGR
        
        # Draw keypoint markers for upper body and arms
        keypoints = person_data.get('keypoints', [])
        keypoint_radius = 4
        keypoint_color = (0, 0, 255)  # Red
        
        for kpt_name, kpt_idx in VISUALIZATION_KEYPOINTS.items():
            if kpt_idx < len(keypoints):
                kpt = keypoints[kpt_idx]
                x, y, conf = kpt[0], kpt[1], kpt[2]
                
                # Only draw if confidence is above threshold
                if conf > 0.2:
                    # Adjust to crop coordinate space
                    crop_x = int(x - x1)
                    crop_y = int(y - y1)
                    
                    # Check if within crop bounds
                    if 0 <= crop_x < crop.shape[1] and 0 <= crop_y < crop.shape[0]:
                        cv2.circle(crop, (crop_x, crop_y), keypoint_radius, keypoint_color, -1)
    
    return crop


def resize_crop_to_tile(crop, tile_size=300):
    """
    Resize a crop to fit in a tile while maintaining aspect ratio.
    
    Args:
        crop: Crop image (BGR numpy array)
        tile_size: Target tile size in pixels (square)
    
    Returns:
        Resized crop image
    """
    h, w = crop.shape[:2]
    if h > w:
        new_h = tile_size
        new_w = int(w * (tile_size / h))
    else:
        new_w = tile_size
        new_h = int(h * (tile_size / w))
    
    return cv2.resize(crop, (new_w, new_h))


def create_tile_with_label(crop, label_pos, label_orig, tile_size=300):
    """
    Create a square tile from a crop with labels.
    
    Args:
        crop: Resized crop image
        label_pos: Display position label (e.g., "#1")
        label_orig: Original index label (e.g., "original index: 5")
        tile_size: Size of the square tile
    
    Returns:
        Square tile image with crop centered and labels added
    """
    import numpy as np
    
    # Create square tile with background
    tile = np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 50
    
    # Center the crop in the tile
    h, w = crop.shape[:2]
    y_offset = (tile_size - h) // 2
    x_offset = (tile_size - w) // 2
    tile[y_offset:y_offset + h, x_offset:x_offset + w] = crop
    
    # Add position label
    cv2.putText(tile, label_pos, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)
    
    # Add original index label
    cv2.putText(tile, label_orig, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)
    
    return tile


def create_tiled_display_from_crops(candidates, tile_size=300, padding=10):
    """
    Create a tiled grid display of candidate crops with labels.
    
    Args:
        candidates: List of candidate dicts with:
            - 'crop': crop image
            - 'original_idx': original detection index
            - 'post_filter_idx': index after filtering
        tile_size: Size of each tile (default: 300)
        padding: Padding between tiles (default: 10)
    
    Returns:
        Tuple of (display_image, num_cols)
    """
    import numpy as np
    
    num_candidates = len(candidates)
    if num_candidates == 0:
        return None, 0
    
    # Calculate grid dimensions
    cols = min(3, num_candidates)
    rows = (num_candidates + cols - 1) // cols
    
    # Prepare tiles
    tiles = []
    for candidate in candidates:
        crop = candidate['crop']
        original_idx = candidate['original_idx']
        post_filter_idx = candidate['post_filter_idx']
        
        # Resize crop to tile size
        resized_crop = resize_crop_to_tile(crop, tile_size)
        
        # Create labels
        label_pos = f"#{post_filter_idx + 1}"
        label_orig = f"original index: {original_idx}"
        
        # Create tile with labels
        tile = create_tile_with_label(resized_crop, label_pos, label_orig, tile_size)
        tiles.append(tile)
    
    # Pad with empty tiles if needed
    while len(tiles) < rows * cols:
        empty_tile = np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 50
        tiles.append(empty_tile)
    
    # Create tiled display
    tile_rows = []
    for r in range(rows):
        row_tiles = tiles[r * cols:(r + 1) * cols]
        # Add padding between tiles
        row_with_padding = [row_tiles[0]]
        for tile in row_tiles[1:]:
            pad = np.ones((tile_size, padding, 3), dtype=np.uint8) * 100
            row_with_padding.append(pad)
            row_with_padding.append(tile)
        row = np.hstack(row_with_padding)
        tile_rows.append(row)
    
    # Stack rows vertically with padding
    display_rows = [tile_rows[0]]
    for row in tile_rows[1:]:
        pad = np.ones((padding, row.shape[1], 3), dtype=np.uint8) * 100
        display_rows.append(pad)
        display_rows.append(row)
    
    display = np.vstack(display_rows)
    
    return display, cols

