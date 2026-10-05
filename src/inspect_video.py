"""
RoadWatch AI - Milestone 1: Video Input & Frame Inspection
---------------------------------------------------------
This module inspects a dashcam video using OpenCV:
- Opens the video file via cv2.VideoCapture
- Reads video metadata (resolution, FPS, total frames, duration)
- Verifies that frames can be decoded into NumPy arrays
- Extracts sample frames and saves them to outputs/ for inspection
"""

import os
import sys
from typing import Dict, Any
import cv2
import numpy as np


def inspect_video(video_path: str, output_dir: str = "outputs") -> Dict[str, Any]:
    """
    Inspects a video file, prints diagnostic metadata, and extracts sample frames.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where sample extracted frames will be saved.

    Returns:
        Dictionary containing metadata about the video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    # Initialize OpenCV VideoCapture object
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video stream: {video_path}")

    # Read video properties using OpenCV property identifiers
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = (total_frames / fps) if fps > 0 else 0.0

    print("=" * 60)
    print(" RoadWatch AI - Video Stream Inspection Report")
    print("=" * 60)
    print(f"File Path        : {video_path}")
    print(f"Resolution       : {width} x {height} (Width x Height)")
    print(f"Frame Rate (FPS) : {fps:.2f} frames/sec")
    print(f"Total Frames     : {total_frames}")
    print(f"Duration         : {duration_seconds:.2f} seconds ({duration_seconds / 60:.2f} minutes)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Determine sample frame positions to extract (start, 25%, 50%, 75%)
    sample_indices = [
        0,
        int(total_frames * 0.25),
        int(total_frames * 0.50),
        int(total_frames * 0.75),
    ]

    saved_samples = []
    read_success_count = 0

    for idx in sample_indices:
        # Seek directly to the specified frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()

        if success and frame is not None:
            read_success_count += 1
            filename = f"sample_frame_{idx:05d}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, frame)
            saved_samples.append(save_path)
            print(f"✓ Successfully extracted frame #{idx:05d} -> Saved to: {save_path}")
            print(f"  Frame array shape: {frame.shape} (H, W, Channels), Data type: {frame.dtype}")
        else:
            print(f"✗ Failed to extract frame #{idx:05d}")

    # Release the video capture resource
    cap.release()

    metadata = {
        "video_path": video_path,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration_seconds,
        "sample_frames_extracted": saved_samples,
        "all_samples_read": (read_success_count == len(sample_indices)),
    }

    return metadata


if __name__ == "__main__":
    target_video = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_dashcam.mp4"
    try:
        report = inspect_video(target_video)
        print("\nInspection successfully completed.")
    except Exception as e:
        print(f"\n[ERROR] Video inspection failed: {e}", file=sys.stderr)
        sys.exit(1)
