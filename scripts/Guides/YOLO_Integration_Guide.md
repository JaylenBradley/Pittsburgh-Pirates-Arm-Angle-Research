# YOLO Integration Guide

## Overview

This guide explains the new `process_release_frames_yolo.py` script that replaces VitPose/MMPose with YOLO for pose detection while maintaining compatibility with existing downstream scripts.

## Key Differences from VitPose

### Model & Detection
- **VitPose**: Uses MMDetection (Faster R-CNN) for person detection + MMPose for pose estimation (133 whole-body keypoints)
- **YOLO**: Uses YOLOv8-Pose which combines detection and pose estimation (17 body keypoints)

### Keypoints
- **VitPose Output**: 133 keypoints per person (body + face + hands)
- **YOLO Output**: 17 keypoints per person (standard OpenPose format - body only)

### Tracking
- **VitPose**: Frame-by-frame independent detection (no tracking across frames)
- **YOLO**: Processes frames in batches of 5 with multi-frame tracking enabled (`persist=True`)
  - Helps maintain consistent person IDs across consecutive frames
  - Reduces duplicate detections of the same person in adjacent frames

### Output Directory Structure
- **VitPose**: `poses/FRAME_NAME_poses/`
- **YOLO**: `_yolo_poses/FRAME_NAME_yolo_poses/`

The underscore prefix makes it easy to distinguish YOLO outputs and keep both methods side-by-side for comparison.

## JSON Output Format

Both methods produce the same JSON structure for downstream compatibility:

```json
{
  "frame": "frame_0001.jpg",
  "persons_detected": 2,
  "persons": [
    {
      "person_id": 0,
      "track_id": 1,
      "bbox": {
        "x1": 100.5,
        "y1": 150.3,
        "x2": 300.7,
        "y2": 450.2,
        "score": 0.95
      },
      "keypoints": [
        [x, y, confidence],  # 17 entries for YOLO (17 body keypoints)
        ...
      ]
    }
  ]
}
```

**Note**: YOLO adds `track_id` to maintain person identity across frames in a batch.

## Usage

```bash
# Process all videos (default: CPU)
python scripts/process_release_frames_yolo.py

# Specify custom videos directory
python scripts/process_release_frames_yolo.py --videos-dir /path/to/baseball_vids

# Force reprocessing (skip cached results)
python scripts/process_release_frames_yolo.py --force

# Use GPU
python scripts/process_release_frames_yolo.py --device cuda:0
```

## Processing Details

- **Batch Size**: 5 frames per batch
- **Tracking**: Enabled within each batch to maintain consistent person IDs
- **Confidence Threshold**: 0.5 (YOLO detection confidence)
- **Model**: YOLOv8l-Pose (Large variant, good balance of speed/accuracy)

## Integration with Downstream Scripts

The following scripts can use YOLO outputs by changing the directory name:

1. **label_pitchers.py**: Change `poses/` to `_yolo_poses/`
2. **auto_label_pitchers.py**: Change `poses/` to `_yolo_poses/`
3. **calculate_pitcher_angles.py**: Change `poses/` to `_yolo_poses/`

Since YOLO provides 17 body keypoints instead of 133, the keypoint indices remain the same:
- Shoulder = indices 5 (left) and 6 (right)
- Elbow = indices 7 (left) and 8 (right)
- Wrist = indices 9 (left) and 10 (right)

## Next Steps

1. Test YOLO output format with downstream scripts
2. Verify that tracking IDs help maintain consistent pitcher detection
3. Compare tracking consistency between VitPose and YOLO approaches
4. Once validated, integrate YOLO into main pipeline (rename `_yolo_poses` to `poses`, update downstream scripts)

## Troubleshooting

- **Model Download Error**: YOLO will download the model on first run (~200MB)
- **GPU Memory Error**: Try `--device cpu` or reduce batch size
- **Missing Frames**: Check that frames exist in `release_frames/` directory
- **Tracking Not Working**: Ensure frames are sequential and batch size is at least 2

