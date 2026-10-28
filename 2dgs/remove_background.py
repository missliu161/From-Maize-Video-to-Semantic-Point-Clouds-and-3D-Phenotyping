"""
按质心球过滤点云 + DBSCAN 保留最大连通簇去离群点。

对每个数据子文件夹：
  1. Read semantic_matchin/point_cloud_object_post.ply and compute its centroid.
  2. 读取 output/point_cloud/iteration_30000/point_cloud.ply。
  3. 仅保留以 A 为球心、半径 2 范围内的点，在内存中暂存裁剪结果。
  4. (可选, 默认开启) 对上一步结果做 DBSCAN，仅保留最大连通簇，
     写为 point_cloud_sphere_denoised.ply。

支持批量处理（指定父目录）和单文件夹处理（直接指定子文件夹）。

Example:
    # 批量处理 (球裁剪 + DBSCAN)
    python sphere_crop_by_centroid.py \\
        /path/to/scenes

    # 单文件夹处理
    python sphere_crop_by_centroid.py \\
        /path/to/scenes/scene1

    # 不执行 DBSCAN（裁剪结果也不会写入文件）
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
    """读取 PLY，返回 xyz 质心。"""
    ply = PlyData.read(str(ply_path))
    vertex_data = ply["vertex"].data
    xyz = np.stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1
    ).astype(np.float64)
    if xyz.shape[0] == 0:
        raise ValueError(f"{ply_path} 中没有顶点")
    return xyz.mean(axis=0)


def filter_ply_vertices(ply: PlyData, keep_mask: np.ndarray) -> PlyData:
    """按掩码筛选顶点，并保留 PLY 元数据及其他元素。"""
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
    """保留 src_ply 中距离 center <= radius 的点，在内存中返回裁剪结果。

    返回 (裁剪后的 PlyData, n_before, n_after)。
    """
    ply = PlyData.read(str(src_ply))
    vertex_data = ply["vertex"].data
    xyz = np.stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1
    ).astype(np.float64)
    n_before = xyz.shape[0]

    if n_before == 0:
        raise ValueError(f"{src_ply} 中没有顶点")

    dist = np.linalg.norm(xyz - center, axis=1)
    keep_mask = dist <= radius
    n_after = int(keep_mask.sum())

    return filter_ply_vertices(ply, keep_mask), n_before, n_after


def normalize_xyz(xyz: np.ndarray) -> tuple:
    """按 bbox 对角线长度做各向同性缩放到 [0,1] 量级。返回 (xyz_norm, scale, center)。"""
    center = xyz.mean(axis=0)
    centered = xyz - center
    diag = np.linalg.norm(centered.max(axis=0) - centered.min(axis=0))
    scale = diag if diag > 0 else 1.0
    return centered / scale, scale, center


def denoise_dbscan(ply: PlyData, eps: float, min_samples: int,
                   normalize_bbox: bool, dst_ply: Path) -> dict:
    """DBSCAN 聚类，保留最大簇；返回统计信息 dict。"""
    vertex_data = ply["vertex"].data
    xyz = np.stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1
    ).astype(np.float64)
    n_before = xyz.shape[0]
    if n_before == 0:
        raise ValueError("球裁剪结果中没有顶点")

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
    """处理单个场景目录。返回 True 表示成功。"""
    centroid_ply = scene_dir / CENTROID_PLY_RELPATH
    source_ply = scene_dir / SOURCE_PLY_RELPATH
    denoised_ply = scene_dir / denoised_name

    if not centroid_ply.exists():
        print(f"  ⚠️  缺少质心 PLY: {centroid_ply}")
        return False
    if not source_ply.exists():
        print(f"  ⚠️  缺少源 PLY: {source_ply}")
        return False

    if do_denoise and denoised_ply.exists() and not overwrite:
        print(f"  ⏭️  DBSCAN 输出已存在，跳过: {denoised_ply}（用 --overwrite 覆盖）")
        return True

    try:
        center = compute_centroid(centroid_ply)
        cropped_ply, n_before, n_cropped = sphere_crop_ply(source_ply, center, radius)
        print(
            f"  ✅ {scene_dir.name}: 质心=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f}) "
            f"r={radius} → {n_before} -> {n_cropped} 点（裁剪结果仅保留在内存中）"
        )
        if not do_denoise:
            print("  ℹ️  DBSCAN 已跳过；未保存裁剪中间点云")
            return True

        info = denoise_dbscan(cropped_ply, dbscan_eps, dbscan_min_samples,
                              dbscan_normalize_bbox, denoised_ply)
        print(
            f"  🧹 DBSCAN: eps={dbscan_eps} min_samples={dbscan_min_samples} "
            f"normalize_bbox={dbscan_normalize_bbox} "
            f"biggest_cluster=#{info['biggest_label']}({info['biggest_size']} pts) "
            f"noise={info['n_noise']} → {info['n_before']} -> {info['n_after']} 点"
        )
        print(f"     保存到: {denoised_ply}")
        return True
    except Exception as exc:
        print(f"  ❌ {scene_dir.name} DBSCAN 失败: {exc}")
        return False


def collect_scenes(path: Path) -> list:
    """path 是父目录时返回其下所有场景；path 本身是场景目录时返回 [path]。"""
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
                        help="父目录（批量处理）或单个场景子目录")
    parser.add_argument("--radius", type=float, default=2.0,
                        help="球半径（默认 2.0）")
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
        print(f"❌ 路径不存在: {target}")
        sys.exit(1)

    scenes = collect_scenes(target)
    if not scenes:
        print(f"❌ 在 {target} 下未找到任何含 "
              f"{CENTROID_PLY_RELPATH} 和 {SOURCE_PLY_RELPATH} 的子目录")
        sys.exit(1)

    mode = "单文件夹" if len(scenes) == 1 and scenes[0] == target else "批量"
    print(f"📂 {mode}处理 {len(scenes)} 个场景，r={args.radius}")
    if not args.no_denoise:
        print(f"   DBSCAN: eps={args.denoise_eps} min_samples={args.denoise_min_samples} "
              f"normalize_bbox={args.denoise_normalize_bbox}")
    else:
        print("   DBSCAN: 跳过（--no-denoise）")
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
    print(f"🎉 完成: {ok} 成功 / {fail} 失败")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


# python remove_background.py /path/to/scenes
