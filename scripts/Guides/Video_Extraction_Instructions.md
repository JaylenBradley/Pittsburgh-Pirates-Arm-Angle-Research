# Video Frame Extraction Script Setup

## Usage

```bash
# Make sure you're in the project directory, ex:
cd /Users/jaylen/Developer/Research/Pirates_Arm_Angle

# Process all videos in ~/Desktop/baseball_vids
python scripts/extract_video_frames.py

# Force reprocess already-processed videos
python scripts/extract_video_frames.py --force

# Use a custom videos directory
python scripts/extract_video_frames.py --videos-dir /path/to/your/videos
```

## What the Script Does

1. **Finds all MP4 videos** in the `baseball_vids` folder on your Desktop
2. **For each video** (e.g., `12650F12-7313-4C8D-A62B-032F5D5C1276.mp4`):
   - Creates a subdirectory: `baseball_vids/12650F12-7313-4C8D-A62B-032F5D5C1276/`
   - Extracts all frames to: `baseball_vids/12650F12-7313-4C8D-A62B-032F5D5C1276/all_frames/`
   - Creates empty directory: `baseball_vids/12650F12-7313-4C8D-A62B-032F5D5C1276/release_frames/`
3. **Skips videos** that have already been processed (unless `--force` is used)
4. **Uses JPG format** with naming `frame_0001.jpg`, `frame_0002.jpg`, etc.
5. **Works portably** - detects the Desktop automatically so multiple people can use it

## Directory Structure After Running

```
~/Desktop/baseball_vids/
├── 12650F12-7313-4C8D-A62B-032F5D5C1276.mp4
├── 12650F12-7313-4C8D-A62B-032F5D5C1276/
│   ├── all_frames/
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   │   └── ...
│   └── release_frames/  (empty - for manual selection)
├── another-video.mp4
├── another-video/
│   ├── all_frames/
│   └── release_frames/
└── ...
```