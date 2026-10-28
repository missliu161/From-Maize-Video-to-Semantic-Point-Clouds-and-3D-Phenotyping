#!/usr/bin/env python3
"""Run the complete video-to-phenotype reproduction workflow for one video.

The bundled YOLO and H-v3 trunk models are used by default.  VGGSfM remains
an external checkout because its upstream source and checkpoint are not copied
into this package.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def run(command: list[str], cwd: Path | None = None) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video")
    parser.add_argument("scene_dir", type=Path,
                        help="Output scene directory (images/sparse/output/...)" )
    parser.add_argument("--vggsfm-dir", type=Path, required=True,
                        help="Local official VGGSfM checkout")
    parser.add_argument("--vggsfm-python", type=Path, default=None,
                        help="Python executable for VGGSfM (default: current Python)")
    parser.add_argument("--gs-python", type=Path, default=None,
                        help="Python executable for 2DGS/postprocess/phenotype (default: current Python)")
    parser.add_argument("--h-python", type=Path, default=None,
                        help="Python executable for the bundled H model")
    parser.add_argument("--yolo-model", type=Path,
                        default=PACKAGE_ROOT / "2dgs" / "models" / "yolo_seg.pt",
                        help="YOLO segmentation checkpoint")
    parser.add_argument("--sampling-fps", type=float, default=2.0,
                        help="Video frames to sample per second (default: 2)")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse an existing 2DGS point cloud")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite sphere-crop outputs")
    args = parser.parse_args()

    video = args.video.resolve()
    scene = args.scene_dir.resolve()
    vggsfm = args.vggsfm_dir.resolve()
    yolo = args.yolo_model.resolve()
    vggsfm_python = (args.vggsfm_python or Path(sys.executable)).resolve()
    gs_python = (args.gs_python or Path(sys.executable)).resolve()
    h_python = (args.h_python or gs_python).resolve()
    if not video.is_file():
        parser.error(f"video not found: {video}")
    if not vggsfm.is_dir():
        parser.error(f"VGGSfM checkout not found: {vggsfm}")
    if not yolo.is_file():
        parser.error(f"YOLO checkpoint not found: {yolo}")
    if not vggsfm_python.is_file():
        parser.error(f"VGGSfM Python executable not found: {vggsfm_python}")
    if not gs_python.is_file():
        parser.error(f"2DGS Python executable not found: {gs_python}")

    if scene.exists() and any(scene.iterdir()):
        parser.error(f"scene directory must be new or empty: {scene}")

    pose = PACKAGE_ROOT / "pose"
    gs = PACKAGE_ROOT / "2dgs"

    run([str(gs_python), str(pose / "extract_video_frames.py"),
         str(video), str(scene), "--fps", str(args.sampling_fps)])
    images_dir = scene / "images"
    image_count = sum(path.is_file() for path in images_dir.iterdir())
    if image_count == 0:
        raise RuntimeError(f"No frames were extracted from video: {video}")
    run([str(vggsfm_python), str(pose / "run_vggsfm.py"), str(scene),
         "--vggsfm-dir", str(vggsfm), "--query-frame-num", str(image_count)])
    train_command = [str(gs_python), str(gs / "pipeline_full.py"),
                     "-s", str(scene), "--iterations", str(args.iterations),
                     "--yolo_model", str(yolo), "--conf", str(args.conf)]
    if args.skip_train:
        train_command.append("--skip_train")
    run(train_command)

    run([str(gs_python), str(gs / "batch_postprocess.py"), str(scene.parent),
         "--only-scene", scene.name, "--enable-local-relabel"])
    split_semantic = scene / "semantic_matchin" / "point_cloud_object_post_split.ply"
    if not split_semantic.exists():
        raise FileNotFoundError(f"Semantic ID-split output not found: {split_semantic}")
    remove_command = [str(gs_python), str(gs / "remove_background.py"), str(scene)]
    if args.overwrite:
        remove_command.append("--overwrite")
    run(remove_command)

    semantic = scene / "semantic_matchin" / "point_cloud_object_post_split.ply"
    full = scene / "point_cloud_sphere_denoised.ply"
    phenotype = [str(gs_python), str(gs / "phenotype_from_semantic_b.py"),
                 "--semantic", str(semantic), "--full", str(full),
                 "--output-dir", str(scene / "phenotype_results"),
                 "--h-python", str(h_python)]
    run(phenotype)

    print("\nComplete reproduction finished.")
    print(f"Semantic PLY: {semantic}")
    print(f"Full PLY: {full}")
    print(f"Phenotype results: {scene / 'phenotype_results'}")


if __name__ == "__main__":
    main()
