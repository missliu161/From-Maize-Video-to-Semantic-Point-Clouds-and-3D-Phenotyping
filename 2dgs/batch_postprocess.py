"""
Batch post-process multiple scenes under a parent directory.

For each subfolder containing semantic_matchin/point_cloud_object.ply,
calls postprocess_object_ply.py with the same arguments.

Example:
    python batch_postprocess.py \\
        /path/to/scenes \\
        --enable-local-relabel
"""

import argparse
import subprocess
import sys
import os
from collections import Counter
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

SCRIPT_DIR = Path(__file__).parent
POSTPROCESS_SCRIPT = SCRIPT_DIR / "postprocess_object_ply.py"
TARGET_RELPATH = "semantic_matchin/point_cloud_object.ply"
OUTPUT_RELPATH = "semantic_matchin/point_cloud_object_post.ply"
SPLIT_OUTPUT_RELPATH = "semantic_matchin/point_cloud_object_post_split.ply"


def cluster_objects_post_ply(
    post_ply_path,
    min_points=100,
    voxel_size=0.05,
):
    """For each object: drop if point count < `min_points`, else keep only
    the largest voxel-connected cluster (26-connectivity on a
    `voxel_size`-sized grid).

    Returns (n_before, n_after, n_objects_kept_after_clustering,
             n_objects_dropped_small, n_objects_total).
    """
    ply = PlyData.read(str(post_ply_path))
    v = ply["vertex"].data
    xyz = np.stack(
        [v["x"], v["y"], v["z"]], axis=1
    ).astype(np.float64)
    obj_ids = np.rint(np.asarray(v["object"], dtype=np.float64)).astype(np.int64)
    n_before = xyz.shape[0]

    if n_before == 0:
        return n_before, n_before, 0, 0, 0

    keep_mask = np.zeros(n_before, dtype=bool)
    n_objects_total = 0
    n_objects_dropped_small = 0
    n_objects_kept_after_clustering = 0

    for oid in np.unique(obj_ids):
        if oid <= 0:
            mask = obj_ids == oid
            keep_mask |= mask
            continue

        if oid == 221:
            keep_mask |= obj_ids == oid
            continue

        n_objects_total += 1
        obj_mask = obj_ids == oid
        obj_xyz = xyz[obj_mask]
        obj_indices = np.where(obj_mask)[0]

        if len(obj_xyz) < min_points:
            n_objects_dropped_small += 1
            continue

        voxel_coords = np.floor(obj_xyz / voxel_size).astype(np.int64)
        unique_voxels, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
        n_voxels = len(unique_voxels)

        if n_voxels == 0:
            n_objects_dropped_small += 1
            continue

        voxel_to_idx = {tuple(vc): i for i, vc in enumerate(unique_voxels)}
        rows, cols = [], []
        for vc, idx in voxel_to_idx.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor = (vc[0] + dx, vc[1] + dy, vc[2] + dz)
                        nb_idx = voxel_to_idx.get(neighbor)
                        if nb_idx is not None:
                            rows.append(idx)
                            cols.append(nb_idx)

        if rows:
            graph = csr_matrix(
                (np.ones(len(rows), dtype=np.int8), (rows, cols)),
                shape=(n_voxels, n_voxels),
            )
            n_comp, comp_labels = connected_components(graph, directed=False)
        else:
            comp_labels = np.zeros(n_voxels, dtype=np.int64)

        comp_counts = Counter(comp_labels[inverse].tolist())
        largest_comp = max(comp_counts, key=comp_counts.get)
        keep_local = comp_labels[inverse] == largest_comp
        keep_mask[obj_indices[keep_local]] = True
        n_objects_kept_after_clustering += 1

    n_after = int(keep_mask.sum())
    if n_after == n_before:
        return n_before, n_after, n_objects_kept_after_clustering, n_objects_dropped_small, n_objects_total

    pv = ply["vertex"]
    new_v = pv[keep_mask]
    new_el = PlyElement.describe(new_v, "vertex")
    other_elements = [e for e in ply.elements if e.name != "vertex"]
    new_ply = PlyData(
        [new_el] + other_elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    )
    post_ply_path.parent.mkdir(parents=True, exist_ok=True)
    new_ply.write(str(post_ply_path))
    return n_before, n_after, n_objects_kept_after_clustering, n_objects_dropped_small, n_objects_total


def sphere_filter_post_ply(post_ply_path, radius):
    """Keep only points within `radius` of the point cloud centroid.

    Removes scattered outlier fragments that are far from the main body.
    Returns (n_before, n_after).
    """
    ply = PlyData.read(str(post_ply_path))
    vertex_data = ply["vertex"].data
    xyz = np.stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1
    ).astype(np.float64)
    n_before = xyz.shape[0]

    if n_before == 0 or radius <= 0:
        return n_before, n_before

    centroid = xyz.mean(axis=0)
    dist = np.linalg.norm(xyz - centroid, axis=1)
    keep_mask = dist <= radius
    n_after = int(keep_mask.sum())

    if n_after == n_before:
        return n_before, n_after

    pv = ply["vertex"]
    new_v = pv[keep_mask]
    new_el = PlyElement.describe(new_v, "vertex")
    other_elements = [e for e in ply.elements if e.name != "vertex"]
    new_ply = PlyData(
        [new_el] + other_elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    )
    post_ply_path.parent.mkdir(parents=True, exist_ok=True)
    new_ply.write(str(post_ply_path))
    return n_before, n_after


def _adaptive_split_eps(points, knn, eps_mult, min_eps, max_eps):
    """Estimate a scale-aware DBSCAN radius from local point spacing."""
    query_k = min(max(2, knn + 1), len(points))
    distances, _ = cKDTree(points).query(points, k=query_k)
    kth = np.asarray(distances[:, -1], dtype=np.float64)
    kth = kth[np.isfinite(kth) & (kth > 0)]
    base = float(np.median(kth)) if kth.size else min_eps
    eps = max(min_eps, eps_mult * base)
    return min(eps, max_eps) if max_eps > 0 else eps


def split_duplicate_ids_post_ply(
    post_ply_path,
    output_ply_path,
    start_object_id=300,
    new_object_start_id=400,
    min_cluster_points=300,
    min_cluster_ratio=0.25,
    knn=8,
    eps_mult=2.0,
    min_eps=0.02,
    max_eps=0.20,
    min_centroid_eps=3.0,
):
    """Split object IDs (>= `start_object_id`) that contain two spatially
    disconnected instances into two IDs.

    Runs label-free: it uses only XYZ + object and an adaptive-radius DBSCAN.
    For each candidate ID the two largest density clusters are compared; a
    split is accepted only when the smaller cluster is large enough, its share
    of the two main clusters is high enough, and their centroids are far apart
    relative to the adaptive radius. Every point of the ID is then assigned to
    the nearest of the two main clusters (so no geometry is discarded), and the
    second cluster receives a fresh ID starting from `new_object_start_id`
    (400, 401, 402, ... skipping IDs already present in the file).

    Writes to `output_ply_path` (which may be the same file as the input when
    in-place splitting is requested) and returns
    (n_candidates, n_splits, split_report).
    """
    ply = PlyData.read(str(post_ply_path))
    vertex_element = ply["vertex"]
    vertices = vertex_element.data.copy()
    names = set(vertices.dtype.names or ())
    xyz = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]], axis=1
    ).astype(np.float64)
    labels = np.rint(np.asarray(vertices["object"], dtype=np.float64)).astype(np.int64)

    candidate_ids = [
        int(oid) for oid in np.unique(labels) if int(oid) >= start_object_id
    ]
    used_object_ids = {int(oid) for oid in np.unique(labels)}
    next_object_id = new_object_start_id
    split_report = []

    for oid in candidate_ids:
        indices = np.where(labels == oid)[0]
        points = xyz[indices]
        if len(points) < 2 * min_cluster_points:
            continue

        eps = _adaptive_split_eps(points, knn, eps_mult, min_eps, max_eps)
        cluster_labels = DBSCAN(eps=eps, min_samples=knn, n_jobs=-1).fit_predict(points)
        valid_ids, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
        if len(valid_ids) < 2:
            continue

        order = np.argsort(counts)[::-1]
        first_id = int(valid_ids[order[0]])
        second_id = int(valid_ids[order[1]])
        first_count = int(counts[order[0]])
        second_count = int(counts[order[1]])
        clustered_count = first_count + second_count
        if second_count < min_cluster_points:
            continue
        if second_count / max(clustered_count, 1) < min_cluster_ratio:
            continue

        first_points = points[cluster_labels == first_id]
        second_points = points[cluster_labels == second_id]
        centroid_distance = float(np.linalg.norm(
            first_points.mean(axis=0) - second_points.mean(axis=0)
        ))
        if centroid_distance < min_centroid_eps * eps:
            continue

        # Assign all minor-cluster / DBSCAN-noise points to the nearer main
        # cluster so splitting never discards geometry.
        tree_first = cKDTree(first_points)
        tree_second = cKDTree(second_points)
        dist_first, _ = tree_first.query(points, k=1)
        dist_second, _ = tree_second.query(points, k=1)
        second_mask = dist_second < dist_first

        while next_object_id in used_object_ids:
            next_object_id += 1
        new_oid = next_object_id
        used_object_ids.add(new_oid)
        next_object_id += 1
        labels[indices[second_mask]] = new_oid
        split_report.append({
            "object": oid,
            "new_object": new_oid,
            "points_original": int(len(points) - np.count_nonzero(second_mask)),
            "points_new": int(np.count_nonzero(second_mask)),
            "eps": eps,
            "centroid_distance": centroid_distance,
            "dbscan_clusters": int(len(valid_ids)),
        })

    vertices["object"] = labels.astype(vertices["object"].dtype, copy=False)

    if all(name in names for name in ("r", "g", "b")):
        rgb_names = ("r", "g", "b")
    elif all(name in names for name in ("red", "green", "blue")):
        rgb_names = ("red", "green", "blue")
    else:
        rgb_names = None
    if rgb_names and split_report:
        current_max = max(float(np.max(vertices[name])) for name in rgb_names)
        scale = 255.0 if current_max > 1.5 else 1.0
        for item in split_report:
            rng = np.random.RandomState(int(item["new_object"]) % 1000000)
            color = rng.uniform(0.3, 1.0, 3) * scale
            mask = labels == item["new_object"]
            for name, value in zip(rgb_names, color):
                vertices[name][mask] = value

    new_element = PlyElement.describe(vertices, "vertex")
    other_elements = [e for e in ply.elements if e.name != "vertex"]
    new_ply = PlyData(
        [new_element] + other_elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    )
    output_ply_path.parent.mkdir(parents=True, exist_ok=True)
    new_ply.write(str(output_ply_path))

    report_path = output_ply_path.with_suffix(output_ply_path.suffix + ".split_report.json")
    try:
        import json
        with open(report_path, "w") as report_file:
            json.dump(
                {
                    "input": str(post_ply_path),
                    "output": str(output_ply_path),
                    "start_object_id": start_object_id,
                    "new_object_start_id": new_object_start_id,
                    "candidate_objects": candidate_ids,
                    "split_count": len(split_report),
                    "splits": split_report,
                },
                report_file,
                indent=2,
            )
    except Exception:
        pass
    return len(candidate_ids), len(split_report), split_report


def collect_scenes(parent_dir: Path):
    scenes = []
    for entry in sorted(parent_dir.iterdir()):
        if not entry.is_dir():
            continue
        ply_path = entry / TARGET_RELPATH
        if ply_path.exists():
            scenes.append(entry)
    return scenes


def main():
    parser = argparse.ArgumentParser(
        description="Batch post-process multiple scene folders",
    )
    parser.add_argument("parent_dir", type=str,
                        help="Parent directory containing scene subfolders")
    parser.add_argument("--only-scene", type=str, default=None,
                        help="Process only this child scene directory name")
    parser.add_argument("--resume", action="store_true",
                        help="Skip scenes that already have an output PLY")
    parser.add_argument("--dry-run", action="store_true",
                        help="List scenes and args without processing")
    parser.add_argument("--max-failures", type=int, default=5,
                        help="Stop after this many consecutive failures")
    parser.add_argument("--sphere-radius", type=float, default=2.0,     #1.8
                        help="Radius of the centroid sphere used to keep points "
                             "in the post-processed PLY. Points farther than this "
                             "from the cloud centroid are dropped. Set to 0 to disable.")
    parser.add_argument("--disable-sphere-filter", action="store_true",
                        help="Skip the post-processing centroid-sphere filter.")
    parser.add_argument("--min-object-points", type=int, default=200,
                        help="Drop objects whose point count is below this value.")
    parser.add_argument("--cluster-voxel-size", type=float, default=0.05,
                        help="Voxel edge length (meters) used for per-object clustering.")
    parser.add_argument("--disable-cluster-filter", action="store_true",
                        help="Skip the per-object cluster filter step.")
    parser.add_argument("--disable-duplicate-id-split", action="store_true",
                        help="Skip the duplicate-instance ID split step for object >= "
                             "--split-start-object-id.")
    parser.add_argument("--split-in-place", action="store_true",
                        help="Overwrite point_cloud_object_post.ply with the split result "
                             "instead of writing point_cloud_object_post_split.ply.")
    parser.add_argument("--split-start-object-id", type=int, default=300,
                        help="Only consider object IDs at or above this value for splitting.")
    parser.add_argument("--split-new-object-start-id", type=int, default=400,
                        help="Assign newly split objects from this ID upward (400, 401, ...).")
    parser.add_argument("--split-min-cluster-points", type=int, default=300,
                        help="Minimum points required in the smaller accepted split cluster.")
    parser.add_argument("--split-min-cluster-ratio", type=float, default=0.25,
                        help="Minimum fraction the smaller cluster must occupy of the two main clusters.")
    parser.add_argument("--split-knn", type=int, default=8,
                        help="Neighbor rank used to estimate adaptive clustering radius.")
    parser.add_argument("--split-eps-mult", type=float, default=2.0,
                        help="Multiplier applied to median k-nearest-neighbor distance.")
    parser.add_argument("--split-min-eps", type=float, default=0.02,
                        help="Minimum adaptive DBSCAN radius in meters.")
    parser.add_argument("--split-max-eps", type=float, default=0.20,
                        help="Maximum adaptive DBSCAN radius in meters; 0 disables the cap.")
    parser.add_argument("--split-min-centroid-eps", type=float, default=3.0,
                        help="Required centroid distance as a multiple of adaptive radius.")

    remaining_args, passthrough_args = parser.parse_known_args()

    parent_dir = Path(remaining_args.parent_dir).resolve()
    if not parent_dir.is_dir():
        print(f"❌ Directory not found: {parent_dir}")
        sys.exit(1)

    if not POSTPROCESS_SCRIPT.exists():
        print(f"❌ postprocess_object_ply.py not found at {POSTPROCESS_SCRIPT}")
        sys.exit(1)

    scenes = collect_scenes(parent_dir)
    if remaining_args.only_scene:
        scenes = [scene for scene in scenes if scene.name == remaining_args.only_scene]
    if not scenes:
        suffix = (f" named {remaining_args.only_scene}" if remaining_args.only_scene else "")
        print(f"❌ No scenes found{suffix} (no subfolder with {TARGET_RELPATH})")
        sys.exit(1)

    print(f"📂 Found {len(scenes)} scenes under {parent_dir}")
    print(f"   Args: {passthrough_args}")

    if remaining_args.dry_run:
        print("\nScenes to process:")
        for scene in scenes:
            print(f"  {scene.name}")
        return

    ckpt_path = parent_dir / ".batch_postprocess_ckpt.json"
    skip_set = set()
    if remaining_args.resume and ckpt_path.exists():
        import json
        try:
            with open(ckpt_path) as f:
                skip_data = json.load(f)
            skip_set = set(skip_data.get("completed", []))
            print(f"  ⏭️  Resuming: {len(skip_set)} already completed")
        except Exception:
            pass

    ok = 0
    fail = 0
    consecutive_fail = 0

    for scene in scenes:
        scene_name = scene.name
        if scene_name in skip_set:
            print(f"  ⏭️  {scene_name} (skipped)")
            ok += 1
            continue

        print(f"\n{'='*60}")
        print(f"  {scene_name}")
        print(f"{'='*60}")

        ply_path = scene / TARGET_RELPATH
        cmd = [sys.executable, str(POSTPROCESS_SCRIPT), str(ply_path)] + passthrough_args

        result = subprocess.run(cmd, capture_output=False)

        if result.returncode == 0:
            ok += 1
            consecutive_fail = 0

            if not remaining_args.disable_sphere_filter and remaining_args.sphere_radius > 0:
                post_ply_path = scene / OUTPUT_RELPATH
                if post_ply_path.exists():
                    try:
                        n_before, n_after = sphere_filter_post_ply(
                            post_ply_path, remaining_args.sphere_radius
                        )
                        if n_before != n_after:
                            print(
                                f"  🪐 sphere filter (r={remaining_args.sphere_radius}): "
                                f"{n_before} -> {n_after} pts"
                            )
                    except Exception as exc:
                        print(f"  ⚠️  sphere filter failed: {exc}")

            if not remaining_args.disable_cluster_filter:
                post_ply_path = scene / OUTPUT_RELPATH
                if post_ply_path.exists():
                    try:
                        cb, ca, kept, dropped_small, total = cluster_objects_post_ply(
                            post_ply_path,
                            min_points=remaining_args.min_object_points,
                            voxel_size=remaining_args.cluster_voxel_size,
                        )
                        if cb != ca or dropped_small > 0:
                            print(
                                f"  🔍 voxel cluster filter "
                                f"(min_pts={remaining_args.min_object_points}, "
                                f"voxel={remaining_args.cluster_voxel_size}): "
                                f"{cb} -> {ca} pts "
                                f"({kept} obj kept, {dropped_small}/{total} obj dropped < min)"
                            )
                    except Exception as exc:
                        print(f"  ⚠️  cluster filter failed: {exc}")

            # Split duplicate-instance object IDs (>= start id) AFTER all
            # original post-processing. By default write a separate PLY so the
            # original point_cloud_object_post.ply is preserved; use
            # --split-in-place to overwrite it instead.
            if not remaining_args.disable_duplicate_id_split:
                post_ply_path = scene / OUTPUT_RELPATH
                if post_ply_path.exists():
                    try:
                        if remaining_args.split_in_place:
                            split_output_path = post_ply_path
                        else:
                            split_output_path = scene / SPLIT_OUTPUT_RELPATH
                        candidates, split_count, split_report = split_duplicate_ids_post_ply(
                            post_ply_path,
                            split_output_path,
                            start_object_id=remaining_args.split_start_object_id,
                            new_object_start_id=remaining_args.split_new_object_start_id,
                            min_cluster_points=remaining_args.split_min_cluster_points,
                            min_cluster_ratio=remaining_args.split_min_cluster_ratio,
                            knn=remaining_args.split_knn,
                            eps_mult=remaining_args.split_eps_mult,
                            min_eps=remaining_args.split_min_eps,
                            max_eps=remaining_args.split_max_eps,
                            min_centroid_eps=remaining_args.split_min_centroid_eps,
                        )
                        print(
                            f"  ✂️ duplicate-ID split: {split_count}/{candidates} objects split; "
                            f"output={split_output_path}"
                        )
                        for item in split_report:
                            print(
                                f"    object {item['object']} -> {item['object']} + "
                                f"{item['new_object']} "
                                f"({item['points_original']} + {item['points_new']} pts, "
                                f"eps={item['eps']:.4f}, "
                                f"centroid_dist={item['centroid_distance']:.4f})"
                            )
                    except Exception as exc:
                        print(f"  ⚠️  duplicate-ID split failed: {exc}")

            if remaining_args.resume:
                import json
                try:
                    completed = set(skip_set)
                    completed.add(scene_name)
                    with open(ckpt_path, "w") as f:
                        json.dump({"completed": sorted(completed)}, f)
                except Exception:
                    pass
        else:
            fail += 1
            consecutive_fail += 1
            print(f"  ❌ {scene_name} failed (returncode={result.returncode})")
            if consecutive_fail >= remaining_args.max_failures:
                print(f"  ⛔ {consecutive_fail} consecutive failures, stopping")
                break

    print(f"\n{'='*60}")
    print(f"🎉 Done: {ok}/{ok + fail} successful, {fail} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()



# Example: python batch_postprocess.py /path/to/scenes --enable-local-relabel
