"""
Evaluate Auto Labels vs Manual Labels.

Compares auto-labeled pitcher track IDs against manual labels.


Usage:
    python scripts/evaluate_auto_labels.py --videos-dir ~/Desktop/baseball_vids
"""

import csv
import sys
from argparse import ArgumentParser
from pathlib import Path

from utils import pose_utils


def safe_load_label(label_json_path):
    """Load a label JSON if it exists, else None."""
    if not label_json_path.exists():
        return None
    return pose_utils.load_json(label_json_path)


def evaluate_video(video_id, video_dir, manual_subdir, auto_subdir):
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
        'track_id_matches': 0,
        'manual_no_pitcher': 0,
        'auto_no_pitcher': 0,
        'both_no_pitcher': 0,
    }

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

        manual_track_id = manual_json.get('pitcher_track_id')
        auto_track_id = auto_json.get('pitcher_track_id')
        if manual_track_id == auto_track_id and manual_track_id is not None:
            metrics['track_id_matches'] += 1

    return metrics


def overall_from_rows(rows):
    """Aggregate per-video rows into one overall metrics row."""
    if not rows:
        return None

    keys_sum = [
        'shared_frames', 'both_detected', 'track_id_matches',
        'manual_no_pitcher', 'auto_no_pitcher', 'both_no_pitcher'
    ]
    out = {'video_id': 'OVERALL'}
    for key in keys_sum:
        out[key] = sum(r[key] for r in rows)

    return out


def print_summary(rows):
    """Print concise summary to terminal."""
    overall = overall_from_rows(rows)
    if overall is None:
        print('No comparable frames found.')
        return

    both_detected = max(1, overall['both_detected'])
    match_rate = overall['track_id_matches'] / both_detected

    print('\n' + '=' * 64)
    print('AUTO VS MANUAL EVALUATION')
    print('=' * 64)
    print(f"Videos compared: {len(rows)}")
    print(f"Shared frames: {overall['shared_frames']}")
    print(f"Both detected: {overall['both_detected']}")
    print(f"Track ID match rate: {match_rate:.3f}")
    print(f"Manual no-pitcher: {overall['manual_no_pitcher']}")
    print(f"Auto no-pitcher: {overall['auto_no_pitcher']}")
    print(f"Both no-pitcher: {overall['both_no_pitcher']}")
    print('=' * 64 + '\n')


def write_csv(rows, output_csv):
    """Write per-video and overall metrics to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'video_id', 'shared_frames', 'both_detected', 'track_id_matches',
        'manual_no_pitcher', 'auto_no_pitcher', 'both_no_pitcher'
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
            auto_subdir=args.auto_subdir
        )
        if row is not None:
            rows.append(row)

    print_summary(rows)
    write_csv(rows, output_csv)
    print(f'Wrote evaluation CSV: {output_csv}')


if __name__ == '__main__':
    main()