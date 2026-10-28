#!/usr/bin/env python3
"""Sample video frames at a fixed rate into a VGGSfM/2DGS scene."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2


def sample_video(video: Path, output_dir: Path, fps: float = 2.0) -> list[Path]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Video contains no readable frames: {video}")
    if fps <= 0:
        capture.release()
        raise ValueError("fps must be > 0")

    video_fps = capture.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        capture.release()
        raise RuntimeError(f"Could not determine video frame rate: {video}")

    # Sample at fixed time intervals and avoid duplicate source frames when the
    # requested sampling rate exceeds the video's native frame rate.
    duration = frame_count / video_fps
    sample_count = max(1, math.ceil(duration * fps - 1e-9))
    indices = sorted(set(
        min(frame_count - 1, int(round(i * video_fps / fps)))
        for i in range(sample_count)
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for output_index, frame_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not read frame {frame_index} from {video}")
        path = output_dir / f"frame_{output_index:04d}.jpg"
        if not cv2.imwrite(str(path), frame):
            capture.release()
            raise RuntimeError(f"Could not write image: {path}")
        written.append(path)
    capture.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("scene_dir", type=Path,
                        help="Scene directory; frames are saved to scene_dir/images/")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frames to sample per second (default: 2)")
    args = parser.parse_args()

    frames = sample_video(args.video.resolve(), args.scene_dir.resolve() / "images", args.fps)
    print(f"Saved {len(frames)} frames:")
    for frame in frames:
        print(f"  {frame}")


if __name__ == "__main__":
    main()
