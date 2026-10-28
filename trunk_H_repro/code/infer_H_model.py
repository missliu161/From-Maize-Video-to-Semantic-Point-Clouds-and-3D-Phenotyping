#!/usr/bin/env python3
"""Apply the trained H-v3 ensemble to one new full/semantic point-cloud pair.

The published ``model_H_final.pkl`` contains the three ExtraTrees estimators and
the learned thresholds.  This script recreates the 21 H-v3 features for the
current scene, so inference does not require the authors' multi-gigabyte
training caches.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
import common as C
import methods as M
import methods_ml as ML
from semantic_volume_trunk import GROUPS, masks, mesh_distance


FEATURE_NAMES3 = ML.FEATURE_NAMES2 + [
    "pca_linearity16", "pca_planarity16", "pca_scattering16", "pca_vertical16",
]


def read_scene(full_path: Path, semantic_path: Path):
    full_ply = PlyData.read(str(full_path))
    semantic_ply = PlyData.read(str(semantic_path))
    full_vertex = full_ply["vertex"].data
    semantic_vertex = semantic_ply["vertex"].data
    full_xyz, _ = C.read_points(full_path)
    semantic_xyz = np.column_stack([
        semantic_vertex[name] for name in ("x", "y", "z")
    ]).astype(np.float64)
    if "object" not in (semantic_vertex.dtype.names or ()):
        raise ValueError("semantic PLY must contain an object field")
    labels = np.rint(semantic_vertex["object"]).astype(np.int64)
    return full_ply, full_vertex, full_xyz, semantic_xyz, labels


def alpha_distances(full_xyz: np.ndarray, semantic_xyz: np.ndarray,
                    labels: np.ndarray) -> dict[str, np.ndarray]:
    distances = {}
    for group in GROUPS:
        factor = float(C.ALPHA_FACTORS[group])
        distances[group], _ = mesh_distance(
            full_xyz, semantic_xyz, labels, group, "alpha", factor,
        )
    return distances


def exact_v3_features(coords: np.ndarray, normals: np.ndarray,
                       old_x: np.ndarray, geo: dict) -> np.ndarray:
    k = min(33, len(coords))
    if k <= 1:
        knn8 = knn32 = np.zeros(len(coords), dtype=np.float32)
        shape = np.zeros((len(coords), 4), dtype=np.float32)
    else:
        distances, indices = cKDTree(coords).query(coords, k=k, workers=1)
        d = max(float(geo["d"]), 1e-3)
        knn8 = distances[:, 1:min(9, k)].mean(axis=1).astype(np.float32) / d
        knn32 = distances[:, 1:k].mean(axis=1).astype(np.float32) / d

        kpca = min(17, k)
        neighbours = coords[indices[:, 1:kpca]]
        centred = neighbours - neighbours.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centred, centred, optimize=True)
        covariance /= max(kpca - 2, 1)
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, 0.0)
        scale = np.maximum(values[:, 2], 1e-12)
        shape = np.column_stack([
            (values[:, 2] - values[:, 1]) / scale,
            (values[:, 1] - values[:, 0]) / scale,
            values[:, 0] / scale,
            np.abs(vectors[:, 1, 2]),
        ]).astype(np.float32)

    norm = np.linalg.norm(normals, axis=1)
    verticality = (1.0 - np.abs(normals[:, 1]) /
                   np.maximum(norm, 1e-6)).astype(np.float32)
    verticality[norm <= 1e-6] = 0.0
    x = old_x.astype(np.float32, copy=True)
    x[:, 10] = knn8
    x[:, 11] = knn32
    x[:, 12] = verticality
    return np.nan_to_num(
        np.column_stack([x, shape]).astype(np.float16),
        nan=0.0, posinf=6.0, neginf=0.0,
    )


def write_subset(source: PlyData, vertices: np.ndarray, keep: np.ndarray,
                 output: Path) -> None:
    selected = vertices[keep].copy()
    if "object" not in (selected.dtype.names or ()):
        dtype = np.dtype((selected.dtype.descr or []) + [("object", "i4")])
        extended = np.empty(selected.shape, dtype=dtype)
        for name in selected.dtype.names or ():
            extended[name] = selected[name]
        selected = extended
    selected["object"] = 1
    elements = [PlyElement.describe(selected, "vertex")]
    elements.extend(element for element in source.elements if element.name != "vertex")
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=source.text, byte_order=source.byte_order,
            comments=source.comments, obj_info=source.obj_info).write(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full", type=Path, help="point_cloud_sphere_denoised.ply")
    parser.add_argument("semantic", type=Path,
                        help="point_cloud_object_post.ply used for H features")
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" /
                        "model_H_final.pkl")
    parser.add_argument("--report-b", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" /
                        "report_B.json")
    parser.add_argument("--mode", choices=("balanced", "precision", "recall"),
                        default="balanced")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=None,
                        help="Optional .npz cache for this scene's H-v3 features")
    args = parser.parse_args()

    for path in (args.full, args.semantic, args.model, args.report_b):
        if not path.exists():
            raise FileNotFoundError(path)

    artifact = pickle.load(args.model.open("rb"))
    if not isinstance(artifact, dict) or not artifact.get("models"):
        raise ValueError(f"Invalid H model artifact: {args.model}")
    if list(artifact.get("feature_names", ())) != FEATURE_NAMES3:
        raise ValueError("model_H_final.pkl feature names do not match bundled H code")
    models = artifact["models"]
    mode_info = artifact.get("modes", {}).get(args.mode)
    if mode_info is None:
        raise ValueError(f"H model has no threshold for mode {args.mode!r}")
    threshold = float(mode_info["threshold"])

    source, vertices, coords, semantic_xyz, labels = read_scene(args.full, args.semantic)
    _, normals = C.read_points(args.full)
    distances = alpha_distances(coords, semantic_xyz, labels)
    report_b = json.loads(args.report_b.read_text(encoding="utf-8"))
    rec = {
        "name": (args.full.parent.name[:-len("_frames")]
                 if args.full.parent.name.endswith("_frames")
                 else args.full.parent.name),
        "coords": coords.astype(np.float32),
        "dist": distances,
        "knn8": np.zeros(len(coords), dtype=np.float32),
        "knn32": np.zeros(len(coords), dtype=np.float32),
        "verticality": np.zeros(len(coords), dtype=np.float32),
    }
    geo = M.centerline_geometry(rec, report_b["tuned_on_train"])
    old_x = ML.features_v2(rec, geo)
    features = exact_v3_features(coords, normals, old_x, geo)
    if features.shape[1] != len(FEATURE_NAMES3):
        raise ValueError(f"Expected {len(FEATURE_NAMES3)} H features, got {features.shape[1]}")

    probability = np.zeros(len(features), dtype=np.float32)
    for model in models:
        probability += model.predict_proba(features.astype(np.float32))[:, 1] / len(models)
    keep = probability >= threshold
    write_subset(source, vertices, keep, args.output)
    if args.feature_cache:
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.feature_cache, features=features,
                            probability=probability, threshold=threshold)
    args.output.with_suffix(".json").write_text(json.dumps({
        "model": str(args.model), "mode": args.mode, "threshold": threshold,
        "full_points": int(len(coords)), "trunk_points": int(keep.sum()),
    }, indent=2), encoding="utf-8")
    print(f"H model: {args.model}")
    print(f"H mode/threshold: {args.mode}/{threshold:.6f}")
    print(f"wrote {int(keep.sum())}/{len(keep)} trunk points to {args.output}")


if __name__ == "__main__":
    main()
