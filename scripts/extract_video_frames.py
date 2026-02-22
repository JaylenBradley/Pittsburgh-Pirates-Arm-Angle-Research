"""
Video Frame Extraction Script for Baseball Pitching Analysis

This script processes all MP4 videos in a baseball_vids directory on the Desktop.
For each video, it:
1. Creates a subdirectory named after the video (without extension)
2. Extracts all frames to a 'all_frames' subdirectory using ffmpeg
3. Creates an empty 'release_frames' subdirectory for manual frame selection
4. AUTOMATICALLY DELETES the original video file after successful frame extraction

Usage:
    python scripts/extract_video_frames.py [--videos-dir PATH] [OPTIONS]

Arguments:
    --videos-dir: Optional path to the videos directory (default: ~/Desktop/baseball_vids)
    --force: Force reprocessing of already-processed videos
    --keep-videos: Keep original video files instead of deleting them after extraction

Example:
    python scripts/extract_video_frames.py
    python scripts/extract_video_frames.py --videos-dir /path/to/videos
    python scripts/extract_video_frames.py --force
    python scripts/extract_video_frames.py --keep-videos  # Preserve original videos
"""

import os
import subprocess
import sys
from pathlib import Path
from argparse import ArgumentParser


def check_ffmpeg_installed():
    """Check if ffmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"FFmpeg found: {result.stdout.split()[2]}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found. Please install ffmpeg:")
        print("brew install ffmpeg")
        return False


def get_desktop_path():
    home = Path.home()
    desktop = home / "Desktop"

    if not desktop.exists():
        raise FileNotFoundError(f"Desktop directory not found at: {desktop}")

    return desktop


def get_video_files(videos_dir):
    """Get all MP4 files in the videos directory."""
    video_files = list(Path(videos_dir).glob("*.mp4"))
    video_files.extend(list(Path(videos_dir).glob("*.MP4")))  # Handle uppercase

    return sorted(video_files)



def is_already_processed(video_path, videos_dir):
    """Check if a video has already been processed."""
    video_name = video_path.stem
    all_frames_dir = Path(videos_dir) / video_name / "all_frames"

    # Check if all_frames directory exists and has files
    if all_frames_dir.exists():
        frame_files = list(all_frames_dir.glob("frame_*.jpg"))
        if frame_files:
            return True, len(frame_files)

    return False, 0


def extract_frames(video_path, output_dir):
    """
    Extract all frames from a video using ffmpeg.

    Args:
        video_path: Path to the input video file
        output_dir: Directory where frames will be saved

    Returns:
        tuple: (success: bool, frame_count: int, error_message: str)
    """

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # FFmpeg command to extract frames
    # Using the same format as in notes.txt: frame_%04d.jpg
    output_pattern = str(output_dir / "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        output_pattern
    ]

    try:
        print(f"Running: ffmpeg -i {video_path.name} {output_pattern}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout = 300
        )

        # Count extracted frames
        frame_files = list(output_dir.glob("frame_*.jpg"))
        frame_count = len(frame_files)

        return True, frame_count, None

    except subprocess.TimeoutExpired:
        error_msg = "FFmpeg timed out (>5 minutes) - video file may be corrupted or too large"
        return False, 0, error_msg
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg error: {e.stderr}"
        return False, 0, error_msg

def delete_video(video_path):
    """
    Delete a video file after successful frame extraction.

    Args:
        video_path: Path to the video file to delete

    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    try:
        os.remove(video_path)
        return True, None
    except Exception as e:
        return False, str(e)

def process_videos(videos_dir, force=False, keep_videos=False):
    """
    Process all videos in the videos directory.

    Args:
        videos_dir: Path to directory containing video files
        force: If True, reprocess already-processed videos
        keep_videos: If True, keep original video files after extraction
    """

    videos_dir = Path(videos_dir)

    if not videos_dir.exists():
        print(f"Videos directory not found: {videos_dir}")
        print(f"Please create the directory and add your video files.")
        return

    # Get all video files
    video_files = get_video_files(videos_dir)

    if not video_files:
        print(f"No mp4 video files found in: {videos_dir}")
        return

    print(f"\n{'=' * 50}")
    print(f"Found {len(video_files)} video(s) to process")
    print(f"{'=' * 50}\n")

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    deleted_count = 0

    for i, video_path in enumerate(video_files, 1):
        video_name = video_path.stem
        print(f"[{i}/{len(video_files)}] Processing: {video_path.name}")

        # Check if already processed
        already_processed, existing_frames = is_already_processed(video_path, videos_dir)

        if already_processed and not force:
            print(f"Skipping (already processed: {existing_frames} frames)")
            skipped_count += 1
            print()
            continue

        # Create directory structure
        video_dir = videos_dir / video_name
        all_frames_dir = video_dir / "all_frames"
        release_frames_dir = video_dir / "release_frames"

        # Create directories
        all_frames_dir.mkdir(parents=True, exist_ok=True)
        release_frames_dir.mkdir(parents=True, exist_ok=True)

        print(f"Created: {video_name}/all_frames/")
        print(f"Created: {video_name}/release_frames/")

        # Extract frames
        print(f"Extracting frames...")
        success, frame_count, error_msg = extract_frames(video_path, all_frames_dir)

        if success:
            print(f"Extracted {frame_count} frames")
            processed_count += 1

            # Delete the original video file if requested (default behavior)
            if not keep_videos:
                print(f"Deleting original video file...")
                delete_success, delete_error = delete_video(video_path)
                if delete_success:
                    print(f"Deleted: {video_path.name}")
                    deleted_count += 1
                else:
                    print(f"Failed to delete video: {delete_error}")

        else:
            print(f"Failed to extract frames")
            print(f"Error: {error_msg}")
            failed_count += 1

        print()

    # Print summary
    print(f"{'=' * 50}")
    print(f"SUMMARY")
    print(f"{'=' * 50}")
    print(f"Processed: {processed_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Failed:    {failed_count}")
    if not keep_videos:
        print(f"Deleted:   {deleted_count}")
    print(f"Total:     {len(video_files)}")
    print(f"{'=' * 50}\n")


def main():
    parser = ArgumentParser(
        description="Extract frames from baseball videos for pitching analysis"
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=None,
        help="Path to directory containing video files (default: ~/Desktop/baseball_vids)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already-processed videos"
    )
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Keep original video files after extraction (default: delete them)"
    )

    args = parser.parse_args()

    # Determine videos directory
    if args.videos_dir:
        videos_dir = Path(args.videos_dir)
    else:
        try:
            desktop = get_desktop_path()
            videos_dir = desktop / "baseball_vids"
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    print(f"\nVideos directory: {videos_dir}\n")

    # Check ffmpeg installation
    if not check_ffmpeg_installed():
        sys.exit(1)

    # Process videos
    process_videos(videos_dir, force=args.force, keep_videos=args.keep_videos)


if __name__ == "__main__":
    main()