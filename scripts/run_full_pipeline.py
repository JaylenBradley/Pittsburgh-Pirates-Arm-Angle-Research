"""
Run Full Pipeline Script for Baseball Pitcher Pose Analysis

This master script runs the complete pitcher analysis pipeline in sequence:
1. process_release_frames.py - Extract pose data from release frames
2. label_pitchers.py - Interactive pitcher labeling
3. calculate_pitcher_angles.py - Calculate angles and errors
4. generate_results_csv.py - Generate per-frame results CSV

Usage:
    From Pirates_Arm_Angle directory:
    python scripts/run_full_pipeline.py [OPTIONS]

Arguments:
    --videos-dir PATH: Path to baseball_vids directory (default: ~/Desktop/baseball_vids)
    --csv PATH: Path to ground truth CSV (default: baseball_vids/arm_angles_high_speed.csv)
    --start-joint JOINT: Joint to start angle measurement (default: shoulder, options: shoulder, elbow)
    --device DEVICE: Device for inference (default: cpu)
    --force: Force reprocessing of all stages
    --skip-poses: Skip pose extraction stage (if already done)
    --skip-labeling: Skip pitcher labeling stage (if already done)
    --skip-angles: Skip angle calculation stage (if already done)
    --skip-csv: Skip CSV generation stage (if already done)

Examples:
    # Run full pipeline with defaults (shoulder-to-wrist)
    python scripts/run_full_pipeline.py

    # Run with elbow-to-wrist angles
    python scripts/run_full_pipeline.py --start-joint elbow

    # Skip already-completed stages
    python scripts/run_full_pipeline.py --skip-poses --skip-labeling

    # Force reprocess everything
    python scripts/run_full_pipeline.py --force
"""

import sys
import subprocess
from pathlib import Path
from argparse import ArgumentParser

# Import our utilities
import pose_utils


def run_command(cmd, description):
    """
    Run a command and handle errors.

    Args:
        cmd: Command list to run
        description: Description of the command for logging

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'=' * 50}")
    print(f"{description}")
    print(f"{'=' * 50}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {description} interrupted by user")
        return False


def main():
    parser = ArgumentParser(
        description="Run the complete baseball pitcher analysis pipeline"
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=None,
        help="Path to baseball_vids directory (default: ~/Desktop/baseball_vids)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to ground truth CSV file (default: baseball_vids/arm_angles_high_speed.csv)"
    )
    parser.add_argument(
        "--start-joint",
        type=str,
        default="shoulder",
        choices=["shoulder", "elbow", "both"],
        help="Joint to start angle measurement from (default: shoulder). Use 'both' to calculate both joints."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (default: cpu)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all stages"
    )
    parser.add_argument(
        "--skip-poses",
        action="store_true",
        help="Skip pose extraction stage"
    )
    parser.add_argument(
        "--skip-labeling",
        action="store_true",
        help="Skip pitcher labeling stage"
    )
    parser.add_argument(
        "--skip-angles",
        action="store_true",
        help="Skip angle calculation stage"
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="Skip CSV generation stage"
    )

    args = parser.parse_args()

    # Get baseball_vids directory
    try:
        if args.videos_dir:
            baseball_vids_dir = Path(args.videos_dir)
        else:
            baseball_vids_dir = pose_utils.get_baseball_vids_dir()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not baseball_vids_dir.exists():
        print(f"Error: Directory not found: {baseball_vids_dir}")
        sys.exit(1)

    # Get CSV path
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = baseball_vids_dir / "arm_angles_high_speed.csv"

    if not csv_path.exists():
        print(f"Error: Ground truth CSV not found: {csv_path}")
        sys.exit(1)

    # Print configuration
    print(f"\n{'=' * 50}")
    print(f"FULL PIPELINE CONFIGURATION")
    print(f"{'=' * 50}")
    print(f"Baseball videos directory: {baseball_vids_dir}")
    print(f"Ground truth CSV: {csv_path}")
    print(f"Start joint: {args.start_joint}")
    print(f"Device: {args.device}")
    print(f"Force reprocess: {args.force}")
    print(f"\nStages to run:")
    print(f"  1. Pose extraction: {'SKIP' if args.skip_poses else 'RUN'}")
    print(f"  2. Pitcher labeling: {'SKIP' if args.skip_labeling else 'RUN'}")
    print(f"  3. Angle calculation: {'SKIP' if args.skip_angles else 'RUN'}")
    print(f"  4. CSV generation: {'SKIP' if args.skip_csv else 'RUN'}")
    print(f"{'=' * 50}\n")

    # Confirm before starting
    try:
        response = input("Continue with pipeline? [Y/N]: ")
        if response.lower() != 'y':
            print("Pipeline cancelled.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nPipeline cancelled.")
        sys.exit(0)

    scripts_dir = Path(__file__).parent
    success_count = 0
    total_stages = 4

    # Stage 1: Process release frames
    if not args.skip_poses:
        cmd = [
            sys.executable,
            str(scripts_dir / "process_release_frames.py"),
            "--videos-dir", str(baseball_vids_dir),
        ]
        if args.force:
            cmd.append("--force")

        if run_command(cmd, "STAGE 1: Pose Extraction"):
            success_count += 1
        else:
            print("\n⚠ Pipeline halted due to error in Stage 1")
            sys.exit(1)
    else:
        print("\n⊘ Skipping Stage 1: Pose Extraction")
        total_stages -= 1

    # Stage 2: Label pitchers
    if not args.skip_labeling:
        cmd = [
            sys.executable,
            str(scripts_dir / "label_pitchers.py"),
            "--videos-dir", str(baseball_vids_dir),
        ]
        if args.force:
            cmd.append("--force")

        if run_command(cmd, "STAGE 2: Pitcher Labeling"):
            success_count += 1
        else:
            print("\n⚠ Pipeline halted due to error in Stage 2")
            sys.exit(1)
    else:
        print("\n⊘ Skipping Stage 2: Pitcher Labeling")
        total_stages -= 1

    # Stage 3: Calculate angles
    if not args.skip_angles:
        cmd = [
            sys.executable,
            str(scripts_dir / "calculate_pitcher_angles.py"),
            "--videos-dir", str(baseball_vids_dir),
            "--csv", str(csv_path),
            "--start-joint", args.start_joint
        ]
        if args.force:
            cmd.append("--force")

        if run_command(cmd, "STAGE 3: Angle Calculation"):
            success_count += 1
        else:
            print("\n⚠ Pipeline halted due to error in Stage 3")
            sys.exit(1)
    else:
        print("\n⊘ Skipping Stage 3: Angle Calculation")
        total_stages -= 1

    # Stage 4: Generate results CSV
    if not args.skip_csv:
        cmd = [
            sys.executable,
            str(scripts_dir / "generate_results_csv.py"),
            "--videos-dir", str(baseball_vids_dir)
        ]
        if args.force:
            cmd.append("--force")

        if run_command(cmd, "STAGE 4: Results CSV Generation"):
            success_count += 1
        else:
            print("\n⚠ Pipeline halted due to error in Stage 4")
            sys.exit(1)
    else:
        print("\n⊘ Skipping Stage 4: Results CSV Generation")
        total_stages -= 1

    # Final summary
    print(f"\n{'=' * 50}")
    print(f"PIPELINE COMPLETE")
    print(f"{'=' * 50}")
    print(f"Stages completed: {success_count}/{total_stages}")

    if success_count == total_stages:
        print(f"\n✓ All stages completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. Review results CSV: {baseball_vids_dir}/data_analysis/results.csv")
        print(f"  2. Generate summary statistics:")
        print(f"     python scripts/generate_summary_statistics.py --plot")
    else:
        print(f"\n⚠ Some stages were skipped or failed")

    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()