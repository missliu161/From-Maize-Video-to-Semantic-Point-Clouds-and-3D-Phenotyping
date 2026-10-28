"""Semantic-instance surface distances used to construct H-model features."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


GROUPS = ("id220", "reference", "fruit", "leaf")


def masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "id220": labels == 220,
        "reference": labels == 221,
        "fruit": (labels >= 222) & (labels < 300),
        "leaf": labels >= 300,
    }


def instance_ids(labels: np.ndarray, group: str) -> np.ndarray:
    return np.unique(labels[masks(labels)[group]])


def point_distance(full: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    if not len(semantic):
        return np.full(len(full), np.inf, dtype=np.float32)
    return cKDTree(semantic).query(full, k=1, workers=-1)[0].astype(np.float32)


def spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.005
    sample = points if len(points) <= 5000 else points[np.linspace(0, len(points) - 1, 5000).astype(int)]
    distances = cKDTree(sample).query(sample, k=2, workers=-1)[0][:, 1]
    positive = distances[distances > 1e-8]
    return float(np.median(positive)) if len(positive) else 0.005


def instance_mesh(points: np.ndarray, factor: float) -> o3d.geometry.TriangleMesh | None:
    if len(points) < 4:
        return None
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    alpha = max(spacing(points), 0.0005) * factor
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        return mesh if len(mesh.triangles) else None
    except RuntimeError:
        return None


def mesh_distance(full: np.ndarray, semantic: np.ndarray, labels: np.ndarray,
                  group: str, method: str, factor: float) -> tuple[np.ndarray, dict]:
    """Distance to the closest semantic point or its per-instance alpha mesh."""
    del method  # The published feature configuration uses alpha meshes.
    nearest = point_distance(full, semantic[masks(labels)[group]])
    scene = o3d.t.geometry.RaycastingScene()
    triangles = 0
    successful = 0
    failed = 0
    for object_id in instance_ids(labels, group):
        mesh = instance_mesh(semantic[labels == object_id], factor)
        if mesh is None:
            failed += 1
            continue
        triangles += len(mesh.triangles)
        successful += 1
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    if not triangles:
        return nearest, {"instances": successful + failed, "meshed": 0, "triangles": 0}
    surface = scene.compute_distance(o3d.core.Tensor(full.astype(np.float32))).numpy()
    return np.minimum(nearest, surface).astype(np.float32), {
        "instances": successful + failed, "meshed": successful, "triangles": int(triangles)
    }
