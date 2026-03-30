"""
Evaluate Auto Labels vs Manual Labels.

Compares auto-labeled pitcher selections against manual labels for each frame and
reports agreement metrics. Also estimates duplicate-detection impact by checking
how often multiple pose detections overlap the manual pitcher bbox.

Usage:
    python scripts/evaluate_auto_labels.py --videos-dir ~/Desktop/baseball_vids \
        --manual-subdir pitcher_labels --auto-subdir pitcher_labels_auto
"""

import csv
import sys
from argparse import ArgumentParser
from pathlib import Path

from utils import pose_utils


def bbox_iou(a, b):
    """Compute IoU for bbox dicts with x1,y1,x2,y2."""
    ax1, ay1, ax2, ay2 = float(a['x1']), float(a['y1']), float(a['x2']), float(a['y2'])
    bx1, by1, bx2, by2 = float(b['x1']), float(b['y1']), float(b['x2']), float(b['y2'])

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def pose_overlap_count(video_dir, pitcher_frame_dir_name, manual_bbox, iou_thr):
    """Count how many pose detections overlap manual bbox above IoU threshold."""
    poses_frame_name = pitcher_frame_dir_name.replace('_pitcher', '_poses')
    poses_json = video_dir / 'poses' / poses_frame_name / 'data.json'
    if not poses_json.exists():
        return 0

    poses_data = pose_utils.load_json(poses_json)
    persons = poses_data.get('persons', [])
    count = 0
    for person in persons:
        bbox = person.get('bbox')
        if not isinstance(bbox, dict):
            continue
        if bbox_iou(manual_bbox, bbox) >= iou_thr:
            count += 1
    return count


def safe_load_label(label_json_path):
    """Load a label JSON if it exists, else None."""
    if not label_json_path.exists():
        return None
    return pose_utils.load_json(label_json_path)


def evaluate_video(video_id, video_dir, manual_subdir, auto_subdir, iou_thr):
    """Evaluate one video and return metrics dict."""
    manual_root = video_dir / manual_subdir
    auto_root = video_dir / auto_subdir
    if not manual_root.exists() or not auto_root.exists():
        return None

    manual_frames = sorted([p for p in manual_root.iterdir() if p.is_dir()])
    auto_frames = sorted([p for p in auto_root.iterdir() if p.is_dir()])

    manual_names = {p.name for p in manual_frames}
    auto_names = {p.name for p in auto_frames}
    shared = sorted(manual_names.intersection(auto_names))

    if not shared:
        return None

    metrics = {
        'video_id': video_id,
        'shared_frames': len(shared),
        'both_detected': 0,
        'id_matches': 0,
        'iou_matches': 0,
        'id_mismatch_iou_match': 0,
        'manual_no_pitcher': 0,
        'auto_no_pitcher': 0,
        'both_no_pitcher': 0,
        'frames_with_duplicate_overlap': 0,
        'avg_overlap_count': 0.0,
    }

    overlap_counts = []

    for frame_name in shared:
        manual_json = safe_load_label(manual_root / frame_name / 'data.json')
        auto_json = safe_load_label(auto_root / frame_name / 'data.json')
        if manual_json is None or auto_json is None:
            continue

        manual_detected = bool(manual_json.get('pitcher_detected', False))
        auto_detected = bool(auto_json.get('pitcher_detected', False))

        if not manual_detected:
            metrics['manual_no_pitcher'] += 1
        if not auto_detected:
            metrics['auto_no_pitcher'] += 1
        if not manual_detected and not auto_detected:
            metrics['both_no_pitcher'] += 1

        if not (manual_detected and auto_detected):
            continue

        metrics['both_detected'] += 1

        manual_id = manual_json.get('pitcher_track_id')
        auto_id = auto_json.get('pitcher_track_id')
        if manual_id == auto_id:
            metrics['id_matches'] += 1

        manual_bbox = manual_json.get('bbox')
        auto_bbox = auto_json.get('bbox')
        if isinstance(manual_bbox, dict) and isinstance(auto_bbox, dict):
            iou = bbox_iou(manual_bbox, auto_bbox)
            if iou >= iou_thr:
                metrics['iou_matches'] += 1
                if manual_id != auto_id:
                    metrics['id_mismatch_iou_match'] += 1

            overlaps = pose_overlap_count(video_dir, frame_name, manual_bbox, iou_thr)
            overlap_counts.append(overlaps)
            if overlaps > 1:
                metrics['frames_with_duplicate_overlap'] += 1

    if overlap_counts:
        metrics['avg_overlap_count'] = sum(overlap_counts) / len(overlap_counts)

    return metrics


def overall_from_rows(rows):
    """Aggregate per-video rows into one overall metrics row."""
    if not rows:
        return None

    keys_sum = [
        'shared_frames', 'both_detected', 'id_matches', 'iou_matches',
        'id_mismatch_iou_match', 'manual_no_pitcher', 'auto_no_pitcher',
        'both_no_pitcher', 'frames_with_duplicate_overlap'
    ]
    out = {'video_id': 'OVERALL'}
    for key in keys_sum:
        out[key] = sum(r[key] for r in rows)

    denom = len(rows)
    out['avg_overlap_count'] = (
        sum(r['avg_overlap_count'] for r in rows) / denom if denom > 0 else 0.0
    )
    return out


def print_summary(rows, iou_thr):
    """Print concise summary to terminal."""
    overall = overall_from_rows(rows)
    if overall is None:
        print('No comparable frames found.')
        return

    both_detected = max(1, overall['both_detected'])
    id_acc = overall['id_matches'] / both_detected
    iou_acc = overall['iou_matches'] / both_detected

    print('\n' + '=' * 64)
    print('AUTO VS MANUAL EVALUATION')
    print('=' * 64)
    print(f"Videos compared: {len(rows)}")
    print(f"Shared frames: {overall['shared_frames']}")
    print(f"Both detected: {overall['both_detected']}")
    print(f"ID match rate: {id_acc:.3f}")
    print(f"BBox IoU>={iou_thr:.2f} match rate: {iou_acc:.3f}")
    print(f"ID mismatch but IoU match: {overall['id_mismatch_iou_match']}")
    print(f"Frames with duplicate overlaps: {overall['frames_with_duplicate_overlap']}")
    print(f"Avg overlap count near manual box: {overall['avg_overlap_count']:.2f}")
    print('=' * 64 + '\n')


def write_csv(rows, output_csv):
    """Write per-video and overall metrics to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'video_id', 'shared_frames', 'both_detected', 'id_matches', 'iou_matches',
        'id_mismatch_iou_match', 'manual_no_pitcher', 'auto_no_pitcher',
        'both_no_pitcher', 'frames_with_duplicate_overlap', 'avg_overlap_count'
    ]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

        overall = overall_from_rows(rows)
        if overall is not None:
            writer.writerow(overall)


def main():
    parser = ArgumentParser(description='Evaluate auto labels against manual labels')
    parser.add_argument('--videos-dir', type=str, default=None,
                        help='Path to baseball_vids directory (default: ~/Desktop/baseball_vids)')
    parser.add_argument('--manual-subdir', type=str, default='pitcher_labels',
                        help='Manual label subdir under each video')
    parser.add_argument('--auto-subdir', type=str, default='pitcher_labels_auto',
                        help='Auto label subdir under each video')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for relaxed matching and duplicate overlap checks')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Output CSV path (default: baseball_vids/data_analysis/auto_vs_manual_eval.csv)')

    args = parser.parse_args()

    try:
        baseball_vids_dir = pose_utils.get_baseball_vids_dir(args.videos_dir)
    except FileNotFoundError as exc:
        print(f'Error: {exc}')
        sys.exit(1)

    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        output_csv = baseball_vids_dir / 'data_analysis' / 'auto_vs_manual_eval.csv'

    video_dirs = pose_utils.get_video_dirs(baseball_vids_dir)
    rows = []

    for video_id, video_dir in video_dirs:
        row = evaluate_video(
            video_id=video_id,
            video_dir=video_dir,
            manual_subdir=args.manual_subdir,
            auto_subdir=args.auto_subdir,
            iou_thr=args.iou_threshold
        )
        if row is not None:
            rows.append(row)

    print_summary(rows, args.iou_threshold)
    write_csv(rows, output_csv)
    print(f'Wrote evaluation CSV: {output_csv}')


if __name__ == '__main__':
    main()