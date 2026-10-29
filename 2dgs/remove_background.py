"""
Filter the point cloud with a centroid-centered sphere and use DBSCAN to keep
the largest connected cluster.
For each scene subdirectory:
  1. Read semantic_matchin/point_cloud_object_post.ply and compute its centroid.
   2. Read output/point_cloud/iteration_30000/point_cloud.ply.
   3. Keep only points within radius 2 of the centroid and hold the crop in memory.
   4. Optionally (enabled by default), run DBSCAN on the crop, keep the largest
      connected cluster, and write point_cloud_sphere_denoised.ply.

Supports batch processing (a parent directory) and single-scene processing.

Example:
    python sphere_crop_by_centroid.py \\
        /path/to/scenes

    python sphere_crop_by_centroid.py \\
        /path/to/scenes/scene1

    python sphere_crop_by_centroid.py \\
        /path/to/scenes \\
        --no-denoise
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN


CENTROID_PLY_RELPATH = Path("semantic_matchin/point_cloud_object_post.ply")
SOURCE_PLY_RELPATH = Path("output/point_cloud/iteration_30000/point_cloud.ply")
DENOISED_PLY_NAME = "point_cloud_sphere_denoised.ply"


def compute_centroid(ply_path: Path) -> np.ndarray:
    """Read a PLY file and return the xyz centroid."""
    ply = PlyData.read(str(ply_path))
    vertex_data = ply["vertex"].data
    xyz = np.stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1
    ).astype(np.float64)
    if xyz.shape[0] == 0:
        raise ValueError(f"{ply_path} contains no vertices")
    return xyz.mean(axis=0)


def filter_ply_vertices(ply: PlyData, keep_mask: np.ndarray) -> PlyData:
    """Filter vertices by a mask while preserving PLY metadata and other elements."""
    pv = ply["vertex"]
    new_v = pv[keep_mask]
    new_el = PlyElement.describe(new_v, "vertex")
    other_elements = [e for e in ply.elements if e.name != "vertex"]
    return PlyData(
        [new_el] + other_elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    )


def sphere_crop_ply(src_ply: Path, center: np.ndarray, radius: float) -> tuple:
    """Keep points in src_ply with distance <= radius from center in memory.

    Return (cropped PlyData, n_before, n_after).
    """
    ply = PlyData.read(str(src_ply))
    vertex_data = ply["vertex"].data
    xyz = np.stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1
    ).astype(np.float64)
    n_before = xyz.shape[0]

    if n_before == 0:
        raise ValueError(f"{src_ply} contains no vertices")

    dist = np.linalg.norm(xyz - center, axis=1)
    keep_mask = dist <= radius
    n_after = int(keep_mask.sum())

    return filter_ply_vertices(ply, keep_mask), n_before, n_after


def normalize_xyz(xyz: np.ndarray) -> tuple:
    """Normalize isotropically by the bounding-box diagonal and return
    (xyz_norm, scale, center).
    """
    center = xyz.mean(axis=0)
    centered = xyz - center
    diag = np.linalg.norm(centered.max(axis=0) - centered.min(axis=0))
    scale = diag if diag > 0 else 1.0
    return centered / scale, scale, center


def denoise_dbscan(ply: PlyData, eps: float, min_samples: int,
                   normalize_bbox: bool, dst_ply: Path) -> dict:
    """Run DBSCAN, keep the largest cluster, and return summary statistics."""
    vertex_data = ply["vertex"].data
    xyz = np.stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1
    ).astype(np.float64)
    n_before = xyz.shape[0]
    if n_before == 0:
        raise ValueError("The sphere crop contains no vertices")

    if normalize_bbox:
        xyz_for_cluster, _scale, _center = normalize_xyz(xyz)
    else:
        xyz_for_cluster = xyz

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz_for_cluster)

    keep_mask = np.zeros(n_before, dtype=bool)
    biggest_label = -1
    biggest_size = 0
    n_noise = int((labels == -1).sum())
    valid_labels = labels[labels >= 0]
    if valid_labels.size > 0:
        counts = np.bincount(valid_labels)
        biggest_label = int(np.argmax(counts))
        biggest_size = int(counts[biggest_label])
        keep_mask = labels == biggest_label

    n_after = int(keep_mask.sum())

    new_ply = filter_ply_vertices(ply, keep_mask)
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    new_ply.write(str(dst_ply))

    return {
        "n_before": n_before,
        "n_after": n_after,
        "n_noise": n_noise,
        "biggest_label": biggest_label,
        "biggest_size": biggest_size,
    }


def process_scene(scene_dir: Path, radius: float, overwrite: bool = False,
                  do_denoise: bool = True,
                  dbscan_eps: float = 0.03,
                  dbscan_min_samples: int = 20,
                  dbscan_normalize_bbox: bool = False,
                  denoised_name: str = DENOISED_PLY_NAME) -> bool:
    """Process one scene directory and return True on success."""
    centroid_ply = scene_dir / CENTROID_PLY_RELPATH
    source_ply = scene_dir / SOURCE_PLY_RELPATH
    denoised_ply = scene_dir / denoised_name

    if not centroid_ply.exists():
        print(f"  ⚠️  Missing centroid PLY: {centroid_ply}")
        return False
    if not source_ply.exists():
        print(f"  ⚠️  Missing source PLY: {source_ply}")
        return False

    if do_denoise and denoised_ply.exists() and not overwrite:
        print(f"  ⏭️  DBSCAN output already exists; skipping: {denoised_ply} (use --overwrite to replace)")
        return True

    try:
        center = compute_centroid(centroid_ply)
        cropped_ply, n_before, n_cropped = sphere_crop_ply(source_ply, center, radius)
        print(
            f"  ✅ {scene_dir.name}: centroid=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f}) "
            f"r={radius} → {n_before} -> {n_cropped} points (crop kept in memory only)"
        )
        if not do_denoise:
            print("  ℹ️  DBSCAN skipped; the intermediate crop was not saved")
            return True

        info = denoise_dbscan(cropped_ply, dbscan_eps, dbscan_min_samples,
                              dbscan_normalize_bbox, denoised_ply)
        print(
            f"  🧹 DBSCAN: eps={dbscan_eps} min_samples={dbscan_min_samples} "
            f"normalize_bbox={dbscan_normalize_bbox} "
            f"biggest_cluster=#{info['biggest_label']}({info['biggest_size']} pts) "
            f"noise={info['n_noise']} → {info['n_before']} -> {info['n_after']} points"
        )
        print(f"     Saved to: {denoised_ply}")
        return True
    except Exception as exc:
        print(f"  ❌ DBSCAN failed for {scene_dir.name}: {exc}")
        return False


def collect_scenes(path: Path) -> list:
    """Return child scenes for a parent path, or [path] when path is a scene."""
    if (path / CENTROID_PLY_RELPATH).exists() and (path / SOURCE_PLY_RELPATH).exists():
        return [path]
    scenes = []
    for entry in sorted(path.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / CENTROID_PLY_RELPATH).exists() and (entry / SOURCE_PLY_RELPATH).exists():
            scenes.append(entry)
    return scenes


def main():
    parser = argparse.ArgumentParser(
        description="Use the centroid of semantic_matchin/point_cloud_object_post.ply "
                    "as the center for radius-based cropping of "
                    "output/point_cloud/iteration_30000/point_cloud.ply; "
                    "optionally keep the largest DBSCAN cluster."
    )
    parser.add_argument("path", type=str,
                        help="Parent directory for batch processing or one scene directory")
    parser.add_argument("--radius", type=float, default=2.0,
                        help="Sphere radius (default: 2.0)")
    parser.add_argument("--overwrite", action="store_true",
                         help="Overwrite existing output files")
    parser.add_argument("--no-denoise", action="store_true",
                         help="Skip DBSCAN; the in-memory cropped cloud is not saved")
    parser.add_argument("--denoise-eps", type=float, default=0.03,
                         help="DBSCAN neighborhood radius (default: 0.03)")
    parser.add_argument("--denoise-min-samples", type=int, default=20,
                         help="Minimum DBSCAN neighbors (default: 20)")
    parser.add_argument("--denoise-normalize-bbox", action="store_true",
                         help="Normalize coordinates by the bounding-box diagonal before DBSCAN")
    parser.add_argument("--denoised-name", type=str, default=DENOISED_PLY_NAME,
                         help=f"DBSCAN output PLY filename (default: {DENOISED_PLY_NAME})")
    args = parser.parse_args()

    target = Path(args.path).resolve()
    if not target.exists():
        print(f"❌ Path does not exist: {target}")
        sys.exit(1)

    scenes = collect_scenes(target)
    if not scenes:
        print(f"❌ No child scene containing "
              f"{CENTROID_PLY_RELPATH} and {SOURCE_PLY_RELPATH} was found under {target}")
        sys.exit(1)

    mode = "single-scene" if len(scenes) == 1 and scenes[0] == target else "batch"
    print(f"📂 {mode} processing: {len(scenes)} scene(s), r={args.radius}")
    if not args.no_denoise:
        print(f"   DBSCAN: eps={args.denoise_eps} min_samples={args.denoise_min_samples} "
              f"normalize_bbox={args.denoise_normalize_bbox}")
    else:
        print("   DBSCAN: skipped (--no-denoise)")
    for s in scenes:
        print(f"  - {s}")

    ok = 0
    fail = 0
    for scene in scenes:
        print(f"\n{'='*60}\n  {scene.name}\n{'='*60}")
        if process_scene(
            scene,
            args.radius,
            args.overwrite,
            do_denoise=not args.no_denoise,
            dbscan_eps=args.denoise_eps,
            dbscan_min_samples=args.denoise_min_samples,
            dbscan_normalize_bbox=args.denoise_normalize_bbox,
            denoised_name=args.denoised_name,
        ):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*60}")
    print(f"🎉 Complete: {ok} succeeded / {fail} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
