## Quick Fix for iCloud-Offloaded Files

### What happened:
Your frames were likely offloaded to iCloud to save local storage. When this happens:
- Files appear in Finder but are actually "placeholders" 
- They show a small file size (often 0-100 bytes)
- OpenCV can't read them because they're not actually downloaded

### Immediate Solution:

**Option 1: Use the iCloud checker script I just created**

Save the `check_icloud_files.py` script to your scripts folder, then:

```bash
cd /Users/jaylen/Developer/Research/Pirates_Arm_Angle

# Check which files are iCloud placeholders
python scripts/check_icloud_files.py

# Auto-download all iCloud placeholders
python scripts/check_icloud_files.py --download
```

**Option 2: Manual download via Finder**

Right-click on the `baseball_vids` folder → **"Download Now"**

Or for just that one video:
```bash
# Force download via reading the files
cd /Users/jaylen/Desktop/baseball_vids/0C7A7E1B-487D-4150-9510-04AF3DE9DFF5/release_frames
cat frame_0509.jpg frame_0510.jpg frame_0511.jpg > /dev/null
```

**Option 3: Disable iCloud optimization for this folder**

1. Move `baseball_vids` out of iCloud-synced directories (like Desktop/Documents)
2. Or disable optimization: System Settings → Apple ID → iCloud → iCloud Drive → Options → Uncheck "Optimize Mac Storage"

### Updated Error Messages:

I also recommend updating your validation messages in `process_release_frames.py` to mention iCloud:

**Line 74** - Change:
```python
return False, "Image file is empty (0 bytes) - corrupted file"
```
To:
```python
return False, "Image file is empty (0 bytes) - likely iCloud placeholder (run: python scripts/check_icloud_files.py --download)"
```

**Line 77** - Change:
```python
return False, f"Image file is too small ({file_size} bytes) - likely corrupted"
```
To:
```python
return False, f"Image file is too small ({file_size} bytes) - likely iCloud placeholder or corrupted"
```

**Line 83** - Change:
```python
return False, "Image file is corrupted (OpenCV cannot read it)"
```
To:
```python
return False, "Image file is corrupted or iCloud placeholder (OpenCV cannot read it)"
```

### Prevent Future Issues:

Add this to your pipeline documentation - before running scripts, ensure files are downloaded:
```bash
# Always check for iCloud issues first
python scripts/check_icloud_files.py --download
```
