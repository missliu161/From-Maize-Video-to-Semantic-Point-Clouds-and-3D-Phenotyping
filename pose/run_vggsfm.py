#!/usr/bin/env python3
"""Run the upstream VGGSfM demo and arrange its COLMAP output for this 2DGS repo."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path,
                        help="Scene folder containing images/ (created by extract_video_frames.py)")
    parser.add_argument("--vggsfm-dir", type=Path,
                        default=os.environ.get("VGGSFM_DIR"),
                        help="Local clone of https://github.com/facebookresearch/vggsfm")
    parser.add_argument("--query-frame-num", type=int, default=2)
    parser.add_argument("--max-query-pts", type=int, default=4096)
    parser.add_argument("--camera-type", default="SIMPLE_PINHOLE")
    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument("--shared-camera", dest="shared_camera", action="store_true")
    camera_group.add_argument("--no-shared-camera", dest="shared_camera", action="store_false")
    parser.set_defaults(shared_camera=True)
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER,
                        help="Extra Hydra overrides passed through to the upstream demo")
    args = parser.parse_args()

    if args.vggsfm_dir is None:
        parser.error("provide --vggsfm-dir or set VGGSFM_DIR to a VGGSfM source checkout")
    vggsfm_dir = args.vggsfm_dir.resolve()
    scene_dir = args.scene_dir.resolve()
    images_dir = scene_dir / "images"
    demo = vggsfm_dir / "demo.py"
    if not demo.is_file():
        parser.error(f"VGGSfM demo.py not found: {demo}")
    if not images_dir.is_dir() or not list(images_dir.iterdir()):
        parser.error(f"scene has no input images: {images_dir}")

    sparse_dir = scene_dir / "sparse"
    sparse_zero = sparse_dir / "0"
    if sparse_zero.exists() and any(sparse_zero.iterdir()):
        parser.error(f"refusing to overwrite existing COLMAP model: {sparse_zero}")
    if sparse_dir.is_dir():
        unexpected = [item for item in sparse_dir.iterdir()
                      if item.name != "0" or (item.is_dir() and any(item.iterdir()))]
        if unexpected:
            parser.error(
                "refusing to overwrite existing sparse reconstruction artifacts: "
                + ", ".join(str(path) for path in unexpected)
            )

    command = [
        sys.executable, str(demo), f"SCENE_DIR={scene_dir}",
        f"query_frame_num={args.query_frame_num}",
        f"max_query_pts={args.max_query_pts}",
        f"camera_type={args.camera_type}",
        f"shared_camera={str(args.shared_camera).lower()}",
    ]
    if args.extra_args:
        command.extend(args.extra_args)
    subprocess.run(command, cwd=vggsfm_dir, check=True)

    # VGGSfM writes COLMAP files directly into sparse/; the 2DGS loader expects
    # the standard COLMAP sparse/0 layout. Move only files and directories that
    # are direct children of sparse/, retaining any pre-existing content.
    if not sparse_dir.is_dir():
        raise FileNotFoundError(f"VGGSfM did not create {sparse_dir}")
    required = ("cameras.bin", "images.bin", "points3D.bin")
    if not all((sparse_zero / name).is_file() for name in required):
        sparse_zero.mkdir(parents=True, exist_ok=True)
        for item in list(sparse_dir.iterdir()):
            if item.name == "0":
                continue
            destination = sparse_zero / item.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            item.rename(destination)
    missing = [name for name in required if not (sparse_zero / name).is_file()]
    if missing:
        raise RuntimeError(f"VGGSfM output is missing COLMAP files in {sparse_zero}: {missing}")
    print(f"VGGSfM COLMAP model ready for 2DGS: {sparse_zero}")


if __name__ == "__main__":
    main()
