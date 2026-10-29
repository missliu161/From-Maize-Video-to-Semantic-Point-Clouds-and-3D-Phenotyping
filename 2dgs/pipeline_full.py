#!/usr/bin/env python3
"""
pipeline_full.py - End-to-end pipeline: training → YOLO segmentation → cross-frame matching → semantic point cloud

Workflow:
  1. Train the 2DGS model (calls train.py)
  2. Segment, match, vote on GS IDs, and export a semantic point cloud (calls pipeline_seg.py)

Usage:
  Single folder: python pipeline_full.py -s /path/to/dataset
  Batch mode:    python pipeline_full.py -s /path/to/parent --batch
  Inference only: python pipeline_full.py -s /path/to/dataset --skip_train

Notes:
  - Training outputs are written to <dataset_path>/output/, which pipeline_seg.py finds automatically.
  - All parameters have defaults; providing -s is sufficient to run the pipeline.
  - In batch mode, existing models in child folders are detected and training is skipped.
  - --skip_train means "force skipping training even when no model exists".
"""

import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: training → YOLO segmentation → cross-frame matching → semantic point cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python pipeline_full.py -s /path/to/B73-1_frames                     # Full training and segmentation
  python pipeline_full.py -s /path/to/B73-1_frames --skip_train        # Force skipping training
  python pipeline_full.py -s /path/to/parent --batch                    # Batch mode (skip trained scenes)
  python pipeline_full.py -s /path/to/data --iterations 30000 --quiet  # Quiet mode
  python pipeline_full.py -s /path/to/data --yolo_model /path/best.pt  # Specify the YOLO model
""")

    parser.add_argument("-s", "--source_path", required=True,
                        help="Data folder containing a COLMAP scene with images/ and sparse/")
    parser.add_argument("--batch", action="store_true",
                        help="Process all child folders under source_path and skip trained scenes")

    parser.add_argument("--iterations", type=int, default=30000,
                        help="Number of training iterations (default: 30000)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Force skipping training even when the model does not exist")
    parser.add_argument("--lambda_normal", type=float, default=0.05,
                        help="Normal-consistency loss weight")
    parser.add_argument("--lambda_dist", type=float, default=0.0,
                        help="Distortion loss weight")

    parser.add_argument("--yolo_model", type=str, default="",
                        help="YOLO model path (found automatically by default)")
    parser.add_argument("--skip_yolo", action="store_true",
                        help="Skip YOLO when masks already exist")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="YOLO confidence threshold (default: 0.15)")
    parser.add_argument("--spatial_threshold", type=float, default=1.5,
                        help="Spatial matching threshold in meters (default: 1.5)")
    parser.add_argument("--class2_start", type=int, default=222,
                        help="Starting global ID for class 2")
    parser.add_argument("--class3_start", type=int, default=300,
                        help="Starting global ID for class 3")
    parser.add_argument("--dbscan_eps", type=float, default=0.1,
                        help="DBSCAN neighborhood radius")
    parser.add_argument("--dbscan_min_samples", type=int, default=10,
                        help="Minimum number of DBSCAN samples")
    parser.add_argument("--skip_voting", action="store_true",
                        help="Skip GS-ID voting and semantic point-cloud export")
    parser.add_argument("--output_subdir", type=str, default=None,
                         help="pipeline_seg output subdirectory (default: semantic_matchin)")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output")

    args = parser.parse_args()
    script_dir = Path(__file__).parent.resolve()
    src = Path(args.source_path).resolve()

    if args.batch:
        folders = sorted([d for d in src.iterdir() if d.is_dir()])
        if not folders:
            print(f"❌ No child folders found under {src}")
            sys.exit(1)
        for f in folders:
            _run_single(f, args, script_dir)
    else:
        _run_single(src, args, script_dir)


def _check_ply(folder: Path, iterations: int):
    """Return the trained point_cloud.ply path, or None if it does not exist."""
    candidate = folder / "output" / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    return candidate if candidate.exists() else None


def _looks_like_scene(folder: Path) -> bool:
    """Return whether a folder is a COLMAP/Blender scene root for 2DGS training.
    A sparse/ directory or transforms_train.json marks a valid scene.
    """
    return (folder / "sparse").exists() or (folder / "transforms_train.json").exists()


def _run_single(folder: Path, args, script_dir: Path):
    name = folder.name

    if not _looks_like_scene(folder):
        print(f"\n{'='*60}")
        print(f"⏭ Skipping {name}: not a COLMAP/Blender scene (missing sparse/ or transforms_train.json)")
        print(f"{'='*60}")
        return

    final_ply = _check_ply(folder, args.iterations)

    if final_ply is not None:
        need_train = False
        skip_reason = f"existing model {final_ply.name} detected; skipping training"
    elif args.skip_train:
        need_train = False
        skip_reason = "--skip_train forced skipping because no model exists"
    else:
        need_train = True
        skip_reason = None

    if need_train:
        print(f"\n{'='*60}")
        print(f"Step 1: Train 2DGS — {name}")
        print(f"{'='*60}")
        model_dir = folder / "output"
        model_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable, str(script_dir / "train.py"),
            "-s", str(folder),
            "-m", str(model_dir),
            "--iterations", str(args.iterations),
            "--lambda_normal", str(args.lambda_normal),
            "--lambda_dist", str(args.lambda_dist),
            "--test_iterations", str(args.iterations),
            "--save_iterations", str(args.iterations),
            "--quiet",
        ]
        subprocess.run(train_cmd, check=True)
    else:
        print(f"\n{'='*60}")
        print(f"Step 1: Skip training — {name} ({skip_reason})")
        print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"Step 2: Segmentation + matching + semantic point cloud — {name}")
    print(f"{'='*60}")

    pipeline_cmd = [
        sys.executable, str(script_dir / "pipeline_seg.py"),
        str(folder),
        "--spatial_threshold", str(args.spatial_threshold),
        "--class2_start", str(args.class2_start),
        "--class3_start", str(args.class3_start),
        "--dbscan_eps", str(args.dbscan_eps),
        "--dbscan_min_samples", str(args.dbscan_min_samples),
        "--conf", str(args.conf),
    ]
    if args.yolo_model:
        pipeline_cmd += ["--yolo_model", args.yolo_model]
    if args.skip_yolo:
        pipeline_cmd.append("--skip_yolo")
    if args.skip_voting:
        pipeline_cmd.append("--skip_voting")
    if args.output_subdir:
        pipeline_cmd += ["--output_subdir", str(args.output_subdir)]
    if args.quiet:
        pipeline_cmd.append("--quiet")

    subprocess.run(pipeline_cmd, check=True)

    model_dir = folder / "output"
    out_subdir = args.output_subdir if args.output_subdir else "semantic_matchin"
    print(f"\n{'='*60}")
    print(f"✅ Complete: {name}")
    print(f"   Model: {model_dir}")
    if not args.skip_voting:
        ply_files = list((folder / out_subdir).glob("*point_cloud*.ply"))
        for pf in ply_files:
            print(f"   Point cloud: {pf}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
