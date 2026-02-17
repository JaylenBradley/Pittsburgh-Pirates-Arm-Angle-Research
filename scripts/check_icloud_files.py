#!/usr/bin/env python3
"""
iCloud File Checker and Downloader

This script checks for iCloud-offloaded files in release_frames directories
and forces them to download locally.

Usage:
    python scripts/check_icloud_files.py
    python scripts/check_icloud_files.py --download
"""

import sys
import os
from pathlib import Path
import subprocess
from argparse import ArgumentParser

# Import utilities
sys.path.insert(0, str(Path(__file__).parent))
import pose_utils


def is_icloud_placeholder(file_path):
    """
    Check if a file is an iCloud placeholder (not fully downloaded).

    Returns:
        Tuple of (is_placeholder: bool, details: str)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False, "File does not exist"

    # Check file size
    file_size = file_path.stat().st_size

    # Very small files (< 1KB) are suspicious for images
    if file_size < 1024:
        return True, f"File size: {file_size} bytes (suspiciously small)"

    # Check for iCloud metadata using xattr (macOS extended attributes)
    try:
        result = subprocess.run(
            ['xattr', '-l', str(file_path)],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Look for iCloud-related attributes
        if 'com.apple.LaunchServices.quarantine' in result.stdout or \
                'com.apple.metadata' in result.stdout:
            # Check if file has the "download pending" attribute
            attrs = result.stdout.lower()
            if 'downloading' in attrs or 'pending' in attrs:
                return True, "Has iCloud download pending attributes"
    except Exception as e:
        pass

    # Try to read first few bytes to see if it's actually accessible
    try:
        with open(file_path, 'rb') as f:
            header = f.read(100)
            if len(header) < 100 and file_size > 1000:
                return True, f"Cannot read full file content (read {len(header)} bytes, expected {file_size})"
    except Exception as e:
        return True, f"Cannot read file: {str(e)}"

    return False, "File appears to be downloaded"


def force_download_icloud_file(file_path):
    """
    Force download an iCloud file by reading it completely.

    Returns:
        Tuple of (success: bool, message: str)
    """
    file_path = Path(file_path)

    try:
        # Method 1: Simply reading the file forces iCloud to download it
        with open(file_path, 'rb') as f:
            data = f.read()

        # Verify it's actually downloaded now
        new_size = file_path.stat().st_size

        if new_size > 1024:
            return True, f"Downloaded successfully ({new_size:,} bytes)"
        else:
            return False, f"File still too small ({new_size} bytes)"

    except Exception as e:
        # Method 2: Use macOS 'brctl download' command if available
        try:
            result = subprocess.run(
                ['brctl', 'download', str(file_path)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                new_size = file_path.stat().st_size
                return True, f"Downloaded via brctl ({new_size:,} bytes)"
            else:
                return False, f"brctl failed: {result.stderr}"
        except Exception as e2:
            return False, f"Failed to download: {str(e)}, {str(e2)}"


def check_video_frames(video_dir, download=False):
    """
    Check all frames in a video's release_frames directory.

    Returns:
        Tuple of (total, placeholders, downloaded, failed)
    """
    release_frames_dir = video_dir / "release_frames"

    if not release_frames_dir.exists():
        return 0, 0, 0, 0

    frames = list(release_frames_dir.glob("frame_*.jpg"))
    frames.extend(list(release_frames_dir.glob("frame_*.png")))
    frames = sorted(frames)

    total = len(frames)
    placeholders = 0
    downloaded = 0
    failed = 0

    for frame_path in frames:
        is_placeholder, details = is_icloud_placeholder(frame_path)

        if is_placeholder:
            placeholders += 1
            print(f"      ⚠️  {frame_path.name}: {details}")

            if download:
                print(f"         Downloading...", end=" ", flush=True)
                success, message = force_download_icloud_file(frame_path)

                if success:
                    downloaded += 1
                    print(f"✓ {message}")
                else:
                    failed += 1
                    print(f"✗ {message}")

    return total, placeholders, downloaded, failed


def main():
    parser = ArgumentParser(
        description="Check for and download iCloud-offloaded files"
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=None,
        help="Path to baseball_vids directory (default: ~/Desktop/baseball_vids)"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Automatically download iCloud placeholders"
    )

    args = parser.parse_args()

    # Get baseball_vids directory
    try:
        baseball_vids_dir = pose_utils.get_baseball_vids_dir(args.videos_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not baseball_vids_dir.exists():
        print(f"Error: Directory not found: {baseball_vids_dir}")
        sys.exit(1)

    print(f"\nChecking for iCloud-offloaded files...")
    print(f"Directory: {baseball_vids_dir}")
    print(f"Auto-download: {args.download}\n")
    print("=" * 60)

    # Get all video directories
    video_dirs = pose_utils.get_video_dirs(baseball_vids_dir)

    if not video_dirs:
        print("No video directories found!")
        return

    total_files = 0
    total_placeholders = 0
    total_downloaded = 0
    total_failed = 0

    for video_id, video_dir in video_dirs:
        print(f"\n[{video_id}]")

        count, placeholders, downloaded, failed = check_video_frames(
            video_dir, download=args.download
        )

        total_files += count
        total_placeholders += placeholders
        total_downloaded += downloaded
        total_failed += failed

        if placeholders == 0:
            print(f"   ✓ All {count} frames are fully downloaded")
        else:
            print(f"   Found {placeholders} iCloud placeholder(s) out of {count} total")

            if args.download:
                if downloaded > 0:
                    print(f"   ✓ Successfully downloaded: {downloaded}")
                if failed > 0:
                    print(f"   ✗ Failed to download: {failed}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files checked:       {total_files}")
    print(f"iCloud placeholders found: {total_placeholders}")

    if args.download:
        print(f"Successfully downloaded:   {total_downloaded}")
        print(f"Failed to download:        {total_failed}")

        if total_downloaded > 0:
            print("\n✓ Files have been downloaded. You can now run process_release_frames.py")
    else:
        if total_placeholders > 0:
            print(f"\n⚠️  Found {total_placeholders} iCloud placeholder(s)!")
            print("Run with --download flag to download them:")
            print("    python scripts/check_icloud_files.py --download")

    print("=" * 60 + "\n")

    # Additional tips
    if total_placeholders > 0 and not args.download:
        print("💡 TIP: To prevent iCloud optimization in the future:")
        print("   1. Open System Settings → Apple ID → iCloud → iCloud Drive → Options")
        print("   2. Ensure 'Desktop & Documents Folders' is checked")
        print("   3. Or right-click the baseball_vids folder → 'Download Now'\n")


if __name__ == "__main__":
    main()
