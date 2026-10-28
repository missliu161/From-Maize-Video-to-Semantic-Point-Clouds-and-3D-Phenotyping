"""
Post-process semantic Gaussian PLY files exported by pipeline_seg.py.

The script keeps the original PLY vertex attributes, updates the `object` field,
and writes a new PLY. By default it only performs outlier cleanup:

1. Per-object outlier cleanup with adaptive radius clustering.
   Small detached clusters are removed, while multiple large clusters are kept.
2. Optional same-class object merging for likely split instances.
   Enable it with --enable-merge. The safe path merges only same-class objects
   with frame metadata and no co-visible frames.
 3. Optional voxel-occupancy IoU merging.
    Enable it with --enable-local-relabel. Objects sharing many voxels are
    merged (handles front/back same-surface ID splits correctly).

Example:
    python postprocess_object_ply.py \
        /path/to/semantic_matchin/point_cloud_object.ply
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


FIXED_OBJECT_IDS = {220, 221}
CLASS3_START_ID = 300


@dataclass
class ObjectStats:
    obj_id: int
    indices: np.ndarray
    xyz: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    class_id: int
    frames: set[int]

    @property
    def count(self) -> int:
        return int(self.indices.size)


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        return True


def parse_id_set(value: str) -> set[int]:
    if not value:
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def resolve_input_path(input_path: Path) -> Path:
    if input_path.is_dir():
        candidate = input_path / "point_cloud_object.ply"
        if candidate.exists():
            return candidate
    return input_path


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_post.ply")


def auto_mapping_path(input_path: Path) -> Path | None:
    candidate = input_path.parent / "id_mapping.json"
    return candidate if candidate.exists() else None


def load_mapping_info(mapping_path: Path | None):
    class_by_object = {}
    frames_by_object = defaultdict(set)
    if mapping_path is None or not mapping_path.exists():
        return class_by_object, frames_by_object

    with open(mapping_path) as f:
        data = json.load(f)

    for obj_str, info in data.get("global_instances", {}).items():
        obj_id = int(obj_str)
        if "class_id" in info:
            class_by_object[obj_id] = int(info["class_id"])
        for frame in info.get("frames", []):
            frames_by_object[obj_id].add(int(frame))

    for frame_str, local_map in data.get("frame_mappings", {}).items():
        frame = int(frame_str)
        for obj_id in local_map.values():
            frames_by_object[int(obj_id)].add(frame)

    return class_by_object, frames_by_object


def infer_class_id(obj_id: int, class_by_object: dict[int, int]) -> int:
    if obj_id in class_by_object:
        return class_by_object[obj_id]
    if obj_id >= CLASS3_START_ID:
        return 3
    if obj_id >= 222:
        return 2
    return obj_id


def get_vertex_element(plydata: PlyData):
    for element in plydata.elements:
        if element.name == "vertex":
            return element
    raise ValueError("PLY does not contain a vertex element")


def read_vertices(input_path: Path):
    plydata = PlyData.read(str(input_path))
    vertex_element = get_vertex_element(plydata)
    vertices = vertex_element.data
    names = vertices.dtype.names or ()
    required = {"x", "y", "z", "object"}
    missing = sorted(required - set(names))
    if missing:
        raise ValueError(f"PLY missing required vertex fields: {missing}")

    xyz = np.stack(
        [np.asarray(vertices["x"]), np.asarray(vertices["y"]), np.asarray(vertices["z"])],
        axis=1,
    ).astype(np.float64)
    labels = np.rint(np.asarray(vertices["object"])).astype(np.int64)
    return plydata, vertices, xyz, labels


def adaptive_eps(points: np.ndarray, k: int, eps_mult: float, min_eps: float, max_eps: float):
    if len(points) <= 1:
        return min_eps
    query_k = min(max(2, k + 1), len(points))
    dists, _ = cKDTree(points).query(points, k=query_k)
    kth = np.asarray(dists[:, -1], dtype=np.float64)
    kth = kth[np.isfinite(kth)]
    positive = kth[kth > 0]
    if positive.size > 0:
        base = float(np.median(positive))
    elif kth.size > 0:
        base = float(np.median(kth))
    else:
        base = min_eps
    eps = max(min_eps, eps_mult * base)
    if max_eps > 0:
        eps = min(max_eps, eps)
    return eps


def radius_connected_components(points: np.ndarray, eps: float, min_samples: int):
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=eps)
    is_core = np.array([len(local_neighbors) >= min_samples for local_neighbors in neighbors])
    labels = np.full(len(points), -1, dtype=np.int64)
    cluster_id = 0

    for start in range(len(points)):
        if labels[start] != -1 or not is_core[start]:
            continue

        stack = [start]
        labels[start] = cluster_id
        while stack:
            current = stack.pop()
            for neighbor in neighbors[current]:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                    if is_core[neighbor]:
                        stack.append(neighbor)
        cluster_id += 1

    return labels


def cleanup_outliers(xyz: np.ndarray, labels: np.ndarray, args, ignore_ids: set[int]):
    keep_mask = np.ones(labels.shape[0], dtype=bool)
    cleaned_labels = labels.copy()
    report = {}

    for obj_id in sorted(np.unique(labels)):
        obj_id = int(obj_id)
        if obj_id <= 0 or obj_id in ignore_ids:
            continue

        indices = np.where(labels == obj_id)[0]
        points = xyz[indices]
        obj_report = {
            "points_before": int(len(indices)),
            "removed": 0,
            "eps": None,
            "kept_clusters": [],
            "dropped_clusters": [],
        }

        if len(indices) < args.min_object_points:
            keep_mask[indices] = False
            cleaned_labels[indices] = 0
            obj_report["removed"] = int(len(indices))
            obj_report["reason"] = "small_object"
            report[str(obj_id)] = obj_report
            continue

        if args.disable_outlier_filter or len(indices) < args.dbscan_min_samples:
            obj_report["reason"] = "skipped"
            report[str(obj_id)] = obj_report
            continue

        eps = adaptive_eps(points, args.knn, args.eps_mult, args.min_eps, args.max_eps)
        obj_report["eps"] = float(eps)
        cluster_labels = radius_connected_components(points, eps, args.dbscan_min_samples)

        cluster_ids, cluster_counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
        if cluster_ids.size == 0:
            keep_mask[indices] = False
            cleaned_labels[indices] = 0
            obj_report["removed"] = int(len(indices))
            obj_report["reason"] = "all_noise"
            report[str(obj_id)] = obj_report
            continue

        min_keep = max(args.min_cluster_points, int(np.ceil(args.min_cluster_ratio * len(indices))))
        keep_clusters = {
            int(cluster_id)
            for cluster_id, count in zip(cluster_ids, cluster_counts)
            if int(count) >= min_keep
        }
        if not keep_clusters:
            keep_clusters = {int(cluster_ids[int(np.argmax(cluster_counts))])}

        local_keep = np.isin(cluster_labels, list(keep_clusters))
        drop_local = ~local_keep
        keep_mask[indices[drop_local]] = False
        cleaned_labels[indices[drop_local]] = 0

        obj_report["removed"] = int(np.count_nonzero(drop_local))
        for cluster_id, count in zip(cluster_ids, cluster_counts):
            item = {"cluster": int(cluster_id), "points": int(count)}
            if int(cluster_id) in keep_clusters:
                obj_report["kept_clusters"].append(item)
            else:
                obj_report["dropped_clusters"].append(item)
        noise_count = int(np.count_nonzero(cluster_labels == -1))
        if noise_count:
            obj_report["dropped_clusters"].append({"cluster": -1, "points": noise_count})
        report[str(obj_id)] = obj_report

    return cleaned_labels, keep_mask, report


def build_object_stats(
    xyz: np.ndarray,
    labels: np.ndarray,
    class_by_object: dict[int, int],
    frames_by_object: dict[int, set[int]],
    ignore_ids: set[int],
):
    stats = {}
    for obj_id in sorted(np.unique(labels)):
        obj_id = int(obj_id)
        if obj_id <= 0 or obj_id in ignore_ids:
            continue
        indices = np.where(labels == obj_id)[0]
        if indices.size == 0:
            continue
        points = xyz[indices]
        stats[obj_id] = ObjectStats(
            obj_id=obj_id,
            indices=indices,
            xyz=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            class_id=infer_class_id(obj_id, class_by_object),
            frames=set(frames_by_object.get(obj_id, set())),
        )
    return stats


def bbox_gap(a: ObjectStats, b: ObjectStats) -> float:
    gap = np.maximum(0.0, np.maximum(a.bbox_min - b.bbox_max, b.bbox_min - a.bbox_max))
    return float(np.linalg.norm(gap))


def expanded_bbox_overlap(a: ObjectStats, b: ObjectStats, pad: float) -> float:
    a_min = a.bbox_min - pad
    a_max = a.bbox_max + pad
    b_min = b.bbox_min - pad
    b_max = b.bbox_max + pad
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_extent = np.maximum(0.0, inter_max - inter_min)
    inter_volume = float(np.prod(inter_extent))
    vol_a = float(np.prod(np.maximum(a_max - a_min, 1e-9)))
    vol_b = float(np.prod(np.maximum(b_max - b_min, 1e-9)))
    return inter_volume / max(min(vol_a, vol_b), 1e-9)


def sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator):
    if len(points) <= max_points:
        return points
    sample_idx = rng.choice(len(points), size=max_points, replace=False)
    return points[sample_idx]


def nearest_relation(a: ObjectStats, b: ObjectStats, max_points: int, rng: np.random.Generator):
    points_a = sample_points(a.xyz, max_points, rng)
    points_b = sample_points(b.xyz, max_points, rng)
    if len(points_a) > len(points_b):
        points_a, points_b = points_b, points_a
    dists, _ = cKDTree(points_b).query(points_a, k=1)
    dists = np.asarray(dists, dtype=np.float64)
    return float(np.min(dists)), float(np.quantile(dists, 0.25)), dists


def find_merge_edges(stats: dict[int, ObjectStats], args, rng: np.random.Generator):
    edges = []
    objects = sorted(stats)
    for i, obj_a in enumerate(objects):
        stat_a = stats[obj_a]
        for obj_b in objects[i + 1:]:
            stat_b = stats[obj_b]
            if stat_a.class_id != stat_b.class_id and not args.allow_cross_class_merge:
                continue
            if args.require_frame_metadata and (not stat_a.frames or not stat_b.frames):
                continue

            shared_frames = stat_a.frames & stat_b.frames
            covisible = bool(shared_frames)
            if covisible and not args.allow_covisible_merge:
                continue

            gap = bbox_gap(stat_a, stat_b)
            if gap > args.merge_bbox_gap:
                continue

            nearest_dist, qdist, dists = nearest_relation(
                stat_a, stat_b, args.merge_sample_points, rng
            )
            close_ratio = float(np.mean(dists <= args.merge_distance))
            centroid_dist = float(np.linalg.norm(stat_a.centroid - stat_b.centroid))
            bbox_overlap = expanded_bbox_overlap(stat_a, stat_b, args.merge_distance)

            if covisible:
                close_surface = (
                    nearest_dist <= args.covisible_merge_distance
                    and close_ratio >= args.covisible_merge_close_ratio
                )
                overlapping_bbox = (
                    bbox_overlap >= args.covisible_merge_bbox_overlap
                    and centroid_dist <= args.merge_centroid_distance
                )
            else:
                close_surface = (
                    nearest_dist <= args.merge_distance
                    and close_ratio >= args.merge_close_ratio
                    and qdist <= args.merge_q25_distance
                )
                overlapping_bbox = (
                    bbox_overlap >= args.merge_bbox_overlap
                    and centroid_dist <= args.merge_centroid_distance
                )
            if not close_surface and not overlapping_bbox:
                continue

            score = min(nearest_dist, qdist) + 0.1 * centroid_dist - bbox_overlap
            edges.append(
                {
                    "a": int(obj_a),
                    "b": int(obj_b),
                    "score": float(score),
                    "nearest_distance": float(nearest_dist),
                    "q25_distance": float(qdist),
                    "close_ratio": float(close_ratio),
                    "centroid_distance": float(centroid_dist),
                    "bbox_gap": float(gap),
                    "bbox_overlap": float(bbox_overlap),
                    "covisible": covisible,
                    "shared_frames": sorted(int(frame) for frame in shared_frames),
                }
            )
    return sorted(edges, key=lambda item: item["score"])


def merge_objects(labels: np.ndarray, stats: dict[int, ObjectStats], edges, args):
    merged_labels = labels.copy()
    union_find = UnionFind(stats.keys())
    component_frames = {obj_id: set(stat.frames) for obj_id, stat in stats.items()}
    component_classes = {obj_id: {stat.class_id} for obj_id, stat in stats.items()}
    for edge in edges:
        root_a = union_find.find(edge["a"])
        root_b = union_find.find(edge["b"])
        if root_a == root_b:
            continue

        classes_a = component_classes.get(root_a, set())
        classes_b = component_classes.get(root_b, set())
        if not args.allow_cross_class_merge and classes_a != classes_b:
            continue

        frames_a = component_frames.get(root_a, set())
        frames_b = component_frames.get(root_b, set())
        if args.require_frame_metadata and (not frames_a or not frames_b):
            continue
        if frames_a & frames_b and not args.allow_covisible_merge:
            continue

        if union_find.union(root_a, root_b):
            new_root = union_find.find(root_a)
            merged_frames = frames_a | frames_b
            merged_classes = classes_a | classes_b
            component_frames.pop(root_a, None)
            component_frames.pop(root_b, None)
            component_classes.pop(root_a, None)
            component_classes.pop(root_b, None)
            component_frames[new_root] = merged_frames
            component_classes[new_root] = merged_classes

    components = defaultdict(list)
    for obj_id in stats:
        components[union_find.find(obj_id)].append(obj_id)

    remap = {}
    merged_groups = []
    for members in components.values():
        members = sorted(members)
        if len(members) == 1:
            continue
        if args.canonical_id == "min":
            target = min(members)
        else:
            target = max(members, key=lambda obj_id: (stats[obj_id].count, -obj_id))
        for obj_id in members:
            remap[obj_id] = target
        merged_groups.append(
            {
                "target": int(target),
                "members": [int(obj_id) for obj_id in members],
                "points": {str(obj_id): int(stats[obj_id].count) for obj_id in members},
            }
        )

    if remap:
        for obj_id, target in remap.items():
            merged_labels[labels == obj_id] = target

    return merged_labels, remap, merged_groups


def local_majority_relabel(
    xyz: np.ndarray,
    labels: np.ndarray,
    class_by_object: dict[int, int],
    ignore_ids: set[int],
    args,
):
    """
    Voxel occupancy IoU merging. For each pair of same-class objects, compute
    overlapping voxel IoU. If >= threshold, merge the smaller into the larger.
    This correctly handles front/back leaf surfaces (same 3D → same voxels)
    while avoiding false merges for nearby different leaves.
    """
    relabeled = labels.copy()
    report = []
    if not args.enable_local_relabel or args.disable_local_relabel:
        return relabeled, report

    unique_ids = sorted([
        int(v) for v in np.unique(labels)
        if int(v) > 0 and int(v) not in ignore_ids
    ])
    if not unique_ids:
        return relabeled, report

    if args.bbox_class_aware:
        id_groups = defaultdict(list)
        for oid in unique_ids:
            id_groups[infer_class_id(oid, class_by_object)].append(oid)
    else:
        id_groups = {"all": unique_ids}

    voxel_size = args.voxel_size
    if voxel_size <= 0:
        n_small = min(5000, len(xyz))
        rng = np.random.RandomState(0)
        sample_idx = rng.choice(len(xyz), size=n_small, replace=False)
        k = min(10, len(xyz))
        dists, _ = cKDTree(xyz[sample_idx]).query(xyz[sample_idx], k=k)
        kth = np.asarray(dists[:, -1], dtype=np.float64)
        kth = kth[np.isfinite(kth) & (kth > 0)]
        median_dist = float(np.median(kth)) if kth.size > 0 else 0.02
        voxel_size = median_dist * args.voxel_mult

    ijk = np.floor(xyz / voxel_size).astype(np.int64)
    total_changed = 0

    for class_key, ids_in_class in id_groups.items():
        if len(ids_in_class) <= 1:
            continue

        obj_voxels = {}
        for oid in ids_in_class:
            mask = (relabeled == oid)
            if mask.sum() < args.bbox_min_points:
                continue
            keys = set()
            for pt in ijk[mask]:
                keys.add(tuple(pt.tolist()))
            obj_voxels[oid] = keys

        if len(obj_voxels) <= 1:
            continue

        uf = UnionFind(list(obj_voxels.keys()))
        sorted_items = sorted(obj_voxels.items(), key=lambda x: -len(x[1]))

        for i, (oid_a, vox_a) in enumerate(sorted_items):
            for oid_b, vox_b in sorted_items[i + 1:]:
                if uf.find(oid_a) == uf.find(oid_b):
                    continue
                inter = len(vox_a & vox_b)
                union = len(vox_a | vox_b)
                iou = inter / max(union, 1)
                if iou >= args.voxel_iou_threshold:
                    uf.union(oid_a, oid_b)

        components = defaultdict(list)
        for oid in obj_voxels:
            root = uf.find(oid)
            components[root].append(oid)

        for members in components.values():
            if len(members) <= 1:
                continue
            pts = {oid: int(np.count_nonzero(relabeled == oid)) for oid in members}
            target = max(pts, key=pts.get)
            mask = np.isin(relabeled, list(members))
            changed = int(np.count_nonzero(relabeled[mask] != target))
            relabeled[mask] = target
            total_changed += changed

            report.append({
                "class": class_key if isinstance(class_key, str) else int(class_key),
                "merged_ids": sorted(pts.keys()),
                "point_counts": pts,
                "target": target,
                "points_changed": changed,
            })

    return relabeled, report


def object_palette_color(obj_id: int) -> tuple[float, float, float]:
    if obj_id == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(int(obj_id) % 1000000)
    return tuple(float(v) for v in rng.uniform(0.3, 1.0, 3))


def recolor_vertices(vertices: np.ndarray, labels: np.ndarray):
    names = vertices.dtype.names or ()
    if all(name in names for name in ("r", "g", "b")):
        rgb_names = ("r", "g", "b")
    elif all(name in names for name in ("red", "green", "blue")):
        rgb_names = ("red", "green", "blue")
    else:
        return False

    current_max = max(float(np.max(vertices[name])) for name in rgb_names)
    scale = 255.0 if current_max > 1.5 else 1.0
    for obj_id in np.unique(labels):
        color = np.asarray(object_palette_color(int(obj_id))) * scale
        mask = labels == obj_id
        for name, value in zip(rgb_names, color):
            vertices[name][mask] = value
    return True


def write_vertices(
    plydata: PlyData,
    vertices: np.ndarray,
    labels: np.ndarray,
    keep_mask: np.ndarray,
    output_path: Path,
    args,
):
    other_elements = [element for element in plydata.elements if element.name != "vertex"]
    if other_elements and args.noise_action == "drop" and np.count_nonzero(~keep_mask) > 0:
        raise ValueError(
            "Cannot drop vertices from PLY files with non-vertex elements. "
            "Use --noise-action zero instead."
        )

    if args.noise_action == "drop":
        out_vertices = vertices[keep_mask].copy()
        out_labels = labels[keep_mask]
    else:
        out_vertices = vertices.copy()
        out_labels = labels

    out_vertices["object"] = out_labels.astype(out_vertices["object"].dtype, copy=False)
    recolored = False if args.no_recolor else recolor_vertices(out_vertices, out_labels)

    vertex_element = PlyElement.describe(out_vertices, "vertex")
    out_ply = PlyData(
        [vertex_element] + other_elements,
        text=plydata.text,
        byte_order=plydata.byte_order,
        comments=plydata.comments,
        obj_info=plydata.obj_info,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_ply.write(str(output_path))
    return int(len(out_vertices)), recolored


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Post-process point_cloud_object.ply object labels."
    )
    parser.add_argument("input", type=Path, help="PLY file or output directory")
    parser.add_argument("--output", type=Path, default=None, help="Output PLY path")
    parser.add_argument("--id_mapping", type=Path, default=None, help="Optional id_mapping.json")
    parser.add_argument("--report", type=Path, default=None, help="Output report JSON path")

    parser.add_argument("--ignore-ids", default="220,221", help="Comma-separated object IDs to skip")
    parser.add_argument("--noise-action", choices=("drop", "zero"), default="drop")
    parser.add_argument("--disable-outlier-filter", action="store_true")
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--eps-mult", type=float, default=2.5)
    parser.add_argument("--min-eps", type=float, default=0.02)
    parser.add_argument("--max-eps", type=float, default=0.5)
    parser.add_argument("--dbscan-min-samples", type=int, default=10)
    parser.add_argument("--min-object-points", type=int, default=10)
    parser.add_argument("--min-cluster-points", type=int, default=30)
    parser.add_argument("--min-cluster-ratio", type=float, default=0.02)

    parser.add_argument("--enable-merge", action="store_true")
    parser.add_argument("--disable-merge", action="store_true")
    parser.add_argument("--merge-distance", type=float, default=0.15)
    parser.add_argument("--merge-q25-distance", type=float, default=0.20)
    parser.add_argument("--merge-close-ratio", type=float, default=0.03)
    parser.add_argument("--merge-bbox-gap", type=float, default=0.20)
    parser.add_argument("--merge-bbox-overlap", type=float, default=0.15)
    parser.add_argument("--merge-centroid-distance", type=float, default=1.0)
    parser.add_argument("--merge-sample-points", type=int, default=2000)
    parser.add_argument("--allow-missing-frame-metadata", dest="require_frame_metadata", action="store_false")
    parser.set_defaults(require_frame_metadata=True)
    parser.add_argument("--allow-covisible-merge", action="store_true")
    parser.add_argument("--covisible-merge-distance", type=float, default=0.06)
    parser.add_argument("--covisible-merge-close-ratio", type=float, default=0.08)
    parser.add_argument("--covisible-merge-bbox-overlap", type=float, default=0.35)
    parser.add_argument("--allow-cross-class-merge", action="store_true")
    parser.add_argument("--canonical-id", choices=("largest", "min"), default="largest")

    parser.add_argument("--enable-local-relabel", action="store_true",
                        help="Voxel occupancy IoU merging - unify IDs occupying same spatial voxels")
    parser.add_argument("--disable-local-relabel", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=0.0,
                        help="Voxel edge length (0=adaptive from point density)")
    parser.add_argument("--voxel-mult", type=float, default=3.0,
                        help="Multiplier for adaptive voxel size")
    parser.add_argument("--voxel-iou-threshold", type=float, default=0.3,
                        help="IoU threshold to merge two objects")
    parser.add_argument("--bbox-class-aware", action="store_true", default=True,
                        help="Only merge objects within the same class")
    parser.add_argument("--no-bbox-class-aware", dest="bbox_class_aware", action="store_false")
    parser.add_argument("--bbox-min-points", type=int, default=10,
                        help="Minimum points per object to process")

    parser.add_argument("--no-recolor", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def json_safe_params(args):
    params = vars(args).copy()
    for key, value in list(params.items()):
        if isinstance(value, Path):
            params[key] = str(value)
    return params


def main():
    args = build_arg_parser().parse_args()
    input_path = resolve_input_path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path = args.output.resolve() if args.output else default_output_path(input_path)
    report_path = args.report.resolve() if args.report else output_path.with_suffix(".report.json")
    mapping_path = args.id_mapping.resolve() if args.id_mapping else auto_mapping_path(input_path)
    ignore_ids = parse_id_set(args.ignore_ids) | FIXED_OBJECT_IDS
    rng = np.random.default_rng(args.seed)

    plydata, vertices, xyz, labels = read_vertices(input_path)
    class_by_object, frames_by_object = load_mapping_info(mapping_path)

    labels_after_clean, keep_mask, outlier_report = cleanup_outliers(
        xyz, labels, args, ignore_ids
    )
    if args.noise_action == "drop":
        work_xyz = xyz[keep_mask]
        work_labels = labels_after_clean[keep_mask]
    else:
        work_xyz = xyz
        work_labels = labels_after_clean

    merge_edges = []
    merged_groups = []
    local_relabel_report = []
    remap = {}
    if args.enable_merge and not args.disable_merge:
        stats = build_object_stats(
            work_xyz, work_labels, class_by_object, frames_by_object, ignore_ids
        )
        merge_edges = find_merge_edges(stats, args, rng)
        work_labels, remap, merged_groups = merge_objects(work_labels, stats, merge_edges, args)

    work_labels, local_relabel_report = local_majority_relabel(
        work_xyz, work_labels, class_by_object, ignore_ids, args
    )

    if args.noise_action == "drop":
        final_labels = labels_after_clean.copy()
        final_labels[keep_mask] = work_labels
    else:
        final_labels = work_labels

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "id_mapping": str(mapping_path) if mapping_path else None,
        "points_before": int(len(vertices)),
        "points_after": int(np.count_nonzero(keep_mask)) if args.noise_action == "drop" else int(len(vertices)),
        "noise_action": args.noise_action,
        "objects_before": sorted(int(v) for v in np.unique(labels) if int(v) > 0),
        "objects_after": sorted(
            int(v)
            for v in np.unique(final_labels[keep_mask] if args.noise_action == "drop" else final_labels)
            if int(v) > 0
        ),
        "removed_points": int(np.count_nonzero(~keep_mask)),
        "outliers": outlier_report,
        "merge_edges": merge_edges,
        "merged_groups": merged_groups,
        "local_relabel": local_relabel_report,
        "remap": {str(k): int(v) for k, v in sorted(remap.items())},
        "params": json_safe_params(args),
    }

    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    points_written, recolored = write_vertices(
        plydata, vertices, final_labels, keep_mask, output_path, args
    )
    report["points_written"] = points_written
    report["recolored"] = recolored
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"report: {report_path}")
    print(f"removed_points: {report['removed_points']}")
    print(f"merged_groups: {len(merged_groups)}")
    print(f"local_relabel_components: {len(local_relabel_report)}")


if __name__ == "__main__":
    main()
