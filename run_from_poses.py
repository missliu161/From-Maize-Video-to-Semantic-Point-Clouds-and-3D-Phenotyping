#!/usr/bin/env python3
"""Run the workflow from an existing COLMAP scene through phenotype output.

The scene must already contain images/ and a sparse/0 COLMAP reconstruction.
All processing uses the current Python environment; no VGGSfM checkout is used.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PACKAGE_ROOT, check=True)


def validate_colmap_scene(scene: Path) -> None:
    images = scene / "images"
    sparse = scene / "sparse" / "0"
    if not images.is_dir() or not any(p.is_file() for p in images.iterdir()):
        raise FileNotFoundError(f"No input images found in {images}")
    if not sparse.is_dir():
        raise FileNotFoundError(f"COLMAP model directory not found: {sparse}")

    for stem in ("cameras", "images", "points3D"):
        if not any((sparse / f"{stem}.{suffix}").is_file()
                   for suffix in ("bin", "txt")):
            raise FileNotFoundError(
                f"COLMAP model is missing {stem}.bin or {stem}.txt in {sparse}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path,
                        help="Existing scene containing images/ and sparse/0/")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Workflow Python; defaults to the current environment")
    parser.add_argument("--yolo-model", type=Path,
                        default=PACKAGE_ROOT / "2dgs" / "models" / "yolo_seg.pt")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--skip-train", action="store_true",
                        help="Require and reuse an existing matching 2DGS model")
    parser.add_argument("--h-mode", choices=("balanced", "precision", "recall"),
                        default="balanced")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing sphere-crop/denoising outputs")
    args = parser.parse_args()

    scene = args.scene_dir.resolve()
    python = args.python.resolve()
    yolo = args.yolo_model.resolve()
    if not scene.is_dir():
        parser.error(f"scene directory not found: {scene}")
    if not python.is_file():
        parser.error(f"Python executable not found: {python}")
    if not yolo.is_file():
        parser.error(f"YOLO checkpoint not found: {yolo}")
    validate_colmap_scene(scene)

    gs = PACKAGE_ROOT / "2dgs"
    train_command = [
        str(python), str(gs / "pipeline_full.py"), "-s", str(scene),
        "--iterations", str(args.iterations), "--yolo_model", str(yolo),
        "--conf", str(args.conf),
    ]
    if args.skip_train:
        train_command.append("--skip_train")
    run(train_command)

    run([
        str(python), str(gs / "batch_postprocess.py"), str(scene.parent),
        "--only-scene", scene.name, "--enable-local-relabel",
    ])
    semantic = scene / "semantic_matchin" / "point_cloud_object_post_split.ply"
    if not semantic.is_file():
        raise FileNotFoundError(f"Postprocessed semantic cloud not found: {semantic}")

    remove_command = [str(python), str(gs / "remove_background.py"), str(scene)]
    if args.overwrite:
        remove_command.append("--overwrite")
    run(remove_command)

    full = scene / "point_cloud_sphere_denoised.ply"
    if not full.is_file():
        raise FileNotFoundError(f"Denoised full point cloud not found: {full}")
    run([
        str(python), str(gs / "phenotype_from_semantic_b.py"),
        "--semantic", str(semantic), "--full", str(full),
        "--output-dir", str(scene / "phenotype_results"),
        "--h-python", str(python), "--h-mode", args.h_mode,
    ])

    print("\nPost-pose workflow completed.")
    print(f"Semantic PLY: {semantic}")
    print(f"Denoised full PLY: {full}")
    print(f"Phenotype results: {scene / 'phenotype_results'}")


if __name__ == "__main__":
    main()
