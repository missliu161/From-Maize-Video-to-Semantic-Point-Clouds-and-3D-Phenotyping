#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从语义点云和 H 方法主干点云计算单株玉米表型。

这个脚本是独立脚本，不修改项目中的任何已有文件。

输入
----
1. ``--semantic``：最终语义点云，例如
   ``point_cloud_object_post_split.ply``。其中约定：

   * 220：雄穗
   * 221：参照物
   * 222--299：果穗
   * >=300：叶片

2. ``--full``：该植株对应的完整原始点云。它用于确认场景名称。

主干点云默认由发布包内的训练好的 H-v3 模型
``trunk_H_repro/results/model_H_final.pkl`` 直接生成。它读取当前场景的完整点云和
最终语义点云，现场重建 H-v3 特征后推理，不需要重新训练 H 模型或复制训练缓存。
也可以使用 ``--trunk`` 明确指定已有主干 PLY。

输出
----
* ``*_B.ply``：语义点云 + 主干点云。语义点云原有点和 object 值原样保留，
  追加的主干点 object 统一设置为 100。
* ``*_phenotype.xlsx``: ``Plant_Summary``, ``Leaf_Phenotype``,
  ``Ear_Analysis``, and ``ID_Statistics`` sheets.

叶位推断
--------
不使用人工叶片文件名。先在果穗 object 中找到质心 y 最小的果穗 A，取
``B=A 的 y_min``。对每个叶片 ID，取距离主干最近的 30% 点，并将其中的
最大 y 坐标作为叶片基部 y。只在叶片基部 y 大于 B 的叶片中，选择与 B
距离最近的叶片作为穗位叶。由于本数据的 y 轴正方向向下，叶片基部 y 小于
穗位叶基部 y 的叶片属于穗上，按从近到远标为穗上一叶、穗上二叶……；叶片
基部 y 大于穗位叶基部 y 的叶片属于穗下，按从近到远标为穗下一叶、穗下二叶……。

株高按用户定义计算为 ``主干 y_max - ID220 的 y_min``；穗位参考和穗位高
使用同一个果穗 A：先在 222--299 中选择质心 y 最小的 object A，再用
``主干 y_max - A 的 y_max`` 计算穗位高。原始量纲是输入点云坐标量纲；
如果参照物存在，脚本按照表型提取原代码的 PCA/alpha-shape 标定方法换算
为 cm 和 cm²。
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull, KDTree
from sklearn.decomposition import PCA

try:
    import open3d as o3d
except ImportError:  # 允许 --help 和静态检查在没有 open3d 时执行
    o3d = None


# --------------------------------------------------------------------------- #
# ID 约定
# --------------------------------------------------------------------------- #
TASSEL_ID = 220
REFERENCE_ID = 221
EAR_MIN_ID = 222
EAR_MAX_ID = 299
TRUNK_ID = 100
LEAF_MIN_ID = 300

REF_REAL_LENGTH_CM = 9.5
REF_REAL_AREA_CM2 = 47.5
LEAF_AREA_ALPHA = 0.015

LEAF_NEAREST_PERCENT = 30
STEM_HEIGHT_PERCENT = 18
PCA_COMPONENT_SCORE_MARGIN = 0.15

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H_MODEL = PACKAGE_ROOT / "trunk_H_repro" / "results" / "model_H_final.pkl"
DEFAULT_H_INFER = PACKAGE_ROOT / "trunk_H_repro" / "code" / "infer_H_model.py"
DEFAULT_H_REPORT_B = PACKAGE_ROOT / "trunk_H_repro" / "results" / "report_B.json"


# --------------------------------------------------------------------------- #
# PLY IO 和场景名
# --------------------------------------------------------------------------- #
def read_vertex(path: Path):
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY 不包含 vertex element: {path}")
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    missing = {"x", "y", "z"} - names
    if missing:
        raise ValueError(f"PLY 缺少坐标字段 {sorted(missing)}: {path}")
    xyz = np.column_stack(
        [np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])]
    ).astype(np.float64)
    return ply, vertex, xyz


def normalize_scene_name(text: str) -> str:
    """把场景目录/文件名归一化为 KA103-2 这类名称。"""
    value = Path(text).stem
    for suffix in (
        "_frames",
        " - point_cloud_sphere_denoised",
        " - point_cloud_object_post_split",
        " - point_cloud_object_post",
        "_point_cloud_object_post_split",
        "_point_cloud_object_post",
    ):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value.rstrip("-_ ")


def scene_name_candidates(semantic: Path, full: Path) -> list[str]:
    candidates = []
    for path in (full, semantic):
        candidates.append(normalize_scene_name(path.name))
        candidates.append(normalize_scene_name(path.parent.name))
        candidates.append(normalize_scene_name(path.parent.parent.name))
        candidates.append(normalize_scene_name(path.parent.parent.parent.name))
    out = []
    for item in candidates:
        if item and item not in out and item not in {
            "point_cloud_object_post_split",
            "point_cloud_object_post",
            "point_cloud_sphere_denoised",
             "semantic_matchin",
        }:
            out.append(item)
    return out


def choose_scene_name(semantic: Path, full: Path) -> str:
    """优先使用完整点云所在场景目录名，而不是通用 PLY 文件名。"""
    candidates = scene_name_candidates(semantic, full)
    generic = {
        "point_cloud_sphere_denoised",
        "point_cloud_object_post_split",
        "point_cloud_object_post",
    }
    for candidate in candidates:
        if candidate not in generic:
            return candidate
    return normalize_scene_name(full.name)


def generate_h_trunk(infer_script: Path, model: Path, report_b: Path,
                     full: Path, semantic: Path, output: Path,
                     mode: str = "balanced",
                     python_executable: Path | None = None) -> Path:
    """Use the bundled trained H-v3 ensemble to produce this scene's trunk PLY."""
    if not infer_script.exists():
        raise FileNotFoundError(f"主干推理脚本不存在: {infer_script}")
    if not model.exists():
        raise FileNotFoundError(f"训练好的主干提取模型不存在: {model}")
    if not report_b.exists():
        raise FileNotFoundError(f"H 特征配置不存在: {report_b}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_executable or sys.executable), str(infer_script.resolve()),
        str(full.resolve()), str(semantic.resolve()), "--model", str(model.resolve()),
        "--report-b", str(report_b.resolve()), "--mode", mode,
        "--output", str(output.resolve()),
    ]
    subprocess.run(command, check=True)
    if not output.exists():
        raise FileNotFoundError(f"主干提取脚本未生成输出: {output}")
    return output


def make_b_cloud(semantic_ply: PlyData, semantic_vertex: np.ndarray,
                 semantic_xyz: np.ndarray, trunk_vertex: np.ndarray,
                 trunk_xyz: np.ndarray, output: Path) -> tuple[int, int]:
    """追加主干点，不重写语义点；追加点的 object=100。"""
    semantic_names = semantic_vertex.dtype.names or ()
    if "object" not in semantic_names:
        raise ValueError("语义点云必须包含 object 字段")

    # 可选字段全部沿用语义点云的 dtype；主干中没有的字段填 0。
    merged = np.empty(len(semantic_vertex) + len(trunk_vertex), dtype=semantic_vertex.dtype)
    merged[: len(semantic_vertex)] = semantic_vertex
    trunk_names = set(trunk_vertex.dtype.names or ())
    for field in semantic_names:
        if field in trunk_names:
            merged[field][len(semantic_vertex) :] = trunk_vertex[field]
        else:
            merged[field][len(semantic_vertex) :] = 0
    trunk_slice = slice(len(semantic_vertex), None)
    merged["object"][trunk_slice] = TRUNK_ID

    # B 点云中追加的主干统一显示为黑色；语义点云原有颜色保持不变。
    # 兼容常见的两种 PLY RGB 字段命名，并沿用原字段的数据类型/量纲。
    if all(name in semantic_names for name in ("r", "g", "b")):
        rgb_names = ("r", "g", "b")
    elif all(name in semantic_names for name in ("red", "green", "blue")):
        rgb_names = ("red", "green", "blue")
    else:
        rgb_names = ()
    for name in rgb_names:
        merged[name][trunk_slice] = 0

    element = PlyElement.describe(merged, "vertex")
    others = [e for e in semantic_ply.elements if e.name != "vertex"]
    out_ply = PlyData(
        [element] + others,
        text=semantic_ply.text,
        byte_order=semantic_ply.byte_order,
        comments=semantic_ply.comments,
        obj_info=semantic_ply.obj_info,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    out_ply.write(str(output))
    return len(semantic_vertex), len(trunk_vertex)


# --------------------------------------------------------------------------- #
# 参照物标定和叶片表型算法（参照表型提取_合并_参照物.py）
# --------------------------------------------------------------------------- #
def compute_pca_direction(points: np.ndarray):
    if len(points) < 3:
        return None
    pca = PCA(n_components=3).fit(points - points.mean(axis=0))
    return pca.components_[0]


def fit_reference(points: np.ndarray):
    if len(points) < 4:
        return None
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    u, v = vt[0], vt[1]
    x = centered @ u
    y = centered @ v
    length = float(np.ptp(x))
    width = float(np.ptp(y))
    if width > length:
        length, width = width, length
    return {
        "actual_length_3d": length,
        "rectangle_area_2d": length * width,
        "point_count": int(len(points)),
    }


def calculate_leaf_area_alpha_shape(points: np.ndarray, alpha=LEAF_AREA_ALPHA) -> float:
    if len(points) < 30 or o3d is None:
        return 0.0
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if len(points) > 2000:
            pcd = pcd.voxel_down_sample(voxel_size=0.003)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha=alpha
        )
        return float(mesh.get_surface_area()) if len(mesh.triangles) else 0.0
    except Exception:
        return 0.0


def calibrate_reference(points: np.ndarray):
    result = fit_reference(points)
    if result is None:
        return None
    ref_area = calculate_leaf_area_alpha_shape(points)
    if ref_area <= 0:
        ref_area = result["rectangle_area_2d"]
    result["ref_area_raw"] = ref_area
    result["length_scale_cm_per_unit"] = REF_REAL_LENGTH_CM / max(
        result["actual_length_3d"], 1e-12
    )
    result["area_scale_cm2_per_unit2"] = REF_REAL_AREA_CM2 / max(ref_area, 1e-12)
    return result


def calculate_leaf_length(points: np.ndarray, n_sections=20) -> float:
    if len(points) < 30:
        return 0.0
    try:
        centered = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projections = centered @ vt[0]
        n_sections = min(n_sections, len(points) // 10)
        if n_sections < 2:
            return float(np.ptp(projections))
        bins = np.linspace(projections.min(), projections.max(), n_sections + 1)
        axis_points = []
        for i in range(n_sections):
            mask = ((projections >= bins[i]) & (projections <= bins[i + 1])) if i == n_sections - 1 else ((projections >= bins[i]) & (projections < bins[i + 1]))
            if np.count_nonzero(mask) >= 3:
                axis_points.append(points[mask].mean(axis=0))
        if len(axis_points) < 2:
            return float(np.ptp(projections))
        return float(np.linalg.norm(np.diff(np.asarray(axis_points), axis=0), axis=1).sum())
    except Exception:
        return 0.0


def calculate_leaf_width(points: np.ndarray, n_sections=15) -> tuple[float, float]:
    if len(points) < 30:
        return 0.0, 0.0
    try:
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        lp = centered @ vt[0]
        n_sections = min(n_sections, len(points) // 10)
        if n_sections < 2:
            wp = centered @ vt[1]
            return float(np.ptp(wp)), float(np.ptp(wp))
        bins = np.linspace(lp.min(), lp.max(), n_sections + 1)
        widths = []
        for i in range(n_sections):
            mask = ((lp >= bins[i]) & (lp <= bins[i + 1])) if i == n_sections - 1 else ((lp >= bins[i]) & (lp < bins[i + 1]))
            if np.count_nonzero(mask) >= 3:
                widths.append(float(np.ptp((points[mask] - centroid) @ vt[1])))
        if not widths:
            width = float(np.ptp(centered @ vt[1]))
            return width, width
        return float(np.mean(widths)), float(np.max(widths))
    except Exception:
        return 0.0, 0.0


def find_leaf_base_and_extension_points(leaf_points, stem_points, top_percent=30):
    if len(leaf_points) == 0 or len(stem_points) == 0:
        return leaf_points, leaf_points[:0]
    distances, _ = KDTree(stem_points).query(leaf_points)
    k = min(len(leaf_points), max(3, int(len(leaf_points) * top_percent / 100)))
    order = np.argsort(distances)
    return leaf_points[order[:k]], leaf_points[order[k : min(2 * k, len(leaf_points))]]


def leaf_base_statistics(leaf_points, stem_points,
                         top_percent=LEAF_NEAREST_PERCENT) -> tuple[float, float]:
    """返回叶片基部候选点的 (y最大值, y质心)。

    基部候选点定义为距离主干最近的 ``top_percent``% 叶片点。y轴正方向
    向下，因此 y 最大值可用于穗位叶的边界候选判定；候选点的 y 质心用于
    确定穗位叶之后其它叶片的穗上/穗下位置。
    """
    used, _ = find_leaf_base_and_extension_points(
        leaf_points, stem_points, top_percent
    )
    source = used if len(used) else leaf_points
    return float(np.max(source[:, 1])), float(np.mean(source[:, 1]))


def compute_leaf_length_direction(leaf_used, extension_used):
    if len(leaf_used) < 3:
        return None, None, None
    pca = PCA(n_components=3).fit(leaf_used - leaf_used.mean(axis=0))
    if len(extension_used) == 0:
        return pca.components_[0], 1, np.array([np.nan, np.nan])
    direction_to_extension = extension_used.mean(axis=0) - leaf_used.mean(axis=0)
    norm = np.linalg.norm(direction_to_extension)
    if norm <= 1e-12:
        return pca.components_[0], 1, np.array([np.nan, np.nan])
    direction_to_extension /= norm
    scores = np.abs(pca.components_[:2] @ direction_to_extension)
    component = 1 if scores[1] > scores[0] + PCA_COMPONENT_SCORE_MARGIN else 0
    direction = pca.components_[component]
    if direction @ direction_to_extension < 0:
        direction = -direction
    return direction, component + 1, scores


def calculate_leaf_angle(leaf_points, stem_points):
    leaf_used, extension = find_leaf_base_and_extension_points(
        leaf_points, stem_points, LEAF_NEAREST_PERCENT
    )
    if len(leaf_used) == 0 or len(stem_points) < 3:
        return None, None, None, len(leaf_used), 0
    stem_height_tree = KDTree(leaf_used[:, 1].reshape(-1, 1))
    height_distances, _ = stem_height_tree.query(
        stem_points[:, 1].reshape(-1, 1)
    )
    stem_used = stem_points[
        np.argsort(height_distances)[: max(3, int(len(stem_points) * STEM_HEIGHT_PERCENT / 100))]
    ]
    leaf_direction, component, scores = compute_leaf_length_direction(
        leaf_used, extension
    )
    stem_direction = compute_pca_direction(stem_used)
    if leaf_direction is None or stem_direction is None:
        return None, component, scores, len(leaf_used), len(stem_used)
    if stem_direction[1] > 0:
        stem_direction = -stem_direction
    leaf_direction /= max(np.linalg.norm(leaf_direction), 1e-12)
    stem_direction /= max(np.linalg.norm(stem_direction), 1e-12)
    angle = float(np.degrees(np.arccos(np.clip(leaf_direction @ stem_direction, -1, 1))))
    return angle, component, scores, len(leaf_used), len(stem_used)


# --------------------------------------------------------------------------- #
# 叶位、果穗和侧视面积
# --------------------------------------------------------------------------- #
def format_index(number: int) -> str:
    """Return an Arabic index for generated result tables."""
    return str(number)


def infer_leaf_positions(leaf_records, ear_y_min: float):
    """按 y 轴向下的约定推断穗上/穗位/穗下叶位。

    ``ear_y_min`` 是选定果穗 A 的 B 值。穗位叶只能从
    ``base_y_max > ear_y_min`` 的叶片中选择，并取距离 B 最近者；确定穗位叶
    后，其它叶片改用 ``base_y_centroid`` 与穗位叶的 ``base_y_centroid`` 比较。
    返回
    ``(穗位叶基部y, 判定模式)``，判定模式用于记录是否触发无候选回退。
    """
    if not leaf_records:
        return None, "no_leaf"

    candidates = [r for r in leaf_records if r["base_y_max"] > ear_y_min]
    if candidates:
        nearest = min(candidates, key=lambda r: r["base_y_max"] - ear_y_min)
        mode = "base_y_max_gt_A_y_min_then_nearest"
    else:
        # 正常数据应当存在候选。为了让单株处理仍能输出完整结果，异常时
        # 退化为所有叶片中最接近 B 的一片，并在 Excel 中明确记录回退。
        nearest = min(leaf_records, key=lambda r: abs(r["base_y_max"] - ear_y_min))
        mode = "fallback_no_base_y_max_gt_A_y_min"

    nearest["position"] = "Ear-level leaf"
    nearest["position_group"] = "ear_level"
    anchor_y = nearest["base_y_centroid"]

    for record in leaf_records:
        if record is nearest:
            continue
        if record["base_y_centroid"] < anchor_y:
            record["position_group"] = "above"
        elif record["base_y_centroid"] > anchor_y:
            record["position_group"] = "below"
        else:
            record["position_group"] = "same_level"
            record["position"] = "Same-level leaf"

    above = sorted(
        [r for r in leaf_records if r["position_group"] == "above"],
        # y 轴向下：越接近穗位叶，基部质心 y 越大。
        key=lambda r: r["base_y_centroid"], reverse=True,
    )
    below = sorted(
        [r for r in leaf_records if r["position_group"] == "below"],
        # y 轴向下：越接近穗位叶，基部质心 y 越小。
        key=lambda r: r["base_y_centroid"],
    )
    for index, record in enumerate(above, 1):
        record["position"] = f"Above-ear leaf {format_index(index)}"
    for index, record in enumerate(below, 1):
        record["position"] = f"Below-ear leaf {format_index(index)}"

    # 叶片序号按穗上远端到近端、穗位、穗下近端到远端排序，和原表型脚本
    # 的输出习惯一致；每行同时保留 base_y 和 object_id，避免丢失真实 ID。
    order = sorted(
        leaf_records,
        key=lambda r: (
            0 if r["position_group"] == "above" else
            1 if r["position_group"] in {"ear_level", "near_ear"} else 2,
            r["base_y_centroid"],
        ),
    )
    for index, record in enumerate(order, 1):
        record["leaf_index"] = index
    for record in leaf_records:
        record["ear_y_min_B"] = ear_y_min
        record["ear_level_leaf_base_y_centroid"] = anchor_y
        record["leaf_position_mode"] = mode
    return anchor_y, mode


def sil_area(points, theta):
    direction = np.array([math.cos(theta), 0.0, math.sin(theta)])
    projected = np.column_stack([points @ direction, points[:, 1]])
    try:
        return float(ConvexHull(projected).volume)
    except Exception:
        return 0.0


def max_side_area(points):
    if len(points) < 4:
        return 0.0, 0.0
    try:
        points = points[ConvexHull(points).vertices]
    except Exception:
        pass
    thetas = np.linspace(0.0, np.pi, 181, endpoint=False)
    values = np.array([sil_area(points, theta) for theta in thetas])
    best_index = int(np.argmax(values))
    best_theta = float(thetas[best_index])
    best_value = float(values[best_index])
    for theta in np.linspace(best_theta - 0.02, best_theta + 0.02, 21):
        value = sil_area(points, theta)
        if value > best_value:
            best_value, best_theta = value, float(theta)
    return best_value, best_theta


def ref_inlier_hull_area(points, tolerance=0.01):
    if len(points) < 10:
        return None
    center = points.mean(axis=0)
    centered = points - center
    keep = np.ones(len(points), dtype=bool)
    vt = None
    for _ in range(3):
        if keep.sum() < 10:
            return None
        _, _, vt = np.linalg.svd(centered[keep], full_matrices=False)
        distance = centered @ vt[2]
        keep = np.abs(distance) < tolerance
    if keep.sum() < 10:
        return None
    q = centered[keep] @ vt[:2].T
    try:
        area = float(ConvexHull(np.unique(np.round(q, 7), axis=0)).volume)
    except Exception:
        return None
    return area, float(keep.mean()), float(np.ptp(q, axis=0).max())


# --------------------------------------------------------------------------- #
# 单株处理
# --------------------------------------------------------------------------- #
def process_one(semantic_path: Path, full_path: Path, trunk_path: Path,
                output_b: Path, output_xlsx: Path, leaf_base_percent: float,
                ear_level_tolerance: float):
    semantic_ply, semantic_vertex, semantic_xyz = read_vertex(semantic_path)
    _, trunk_vertex, trunk_xyz = read_vertex(trunk_path)
    if "object" not in (semantic_vertex.dtype.names or ()):
        raise ValueError("语义点云必须包含 object 字段")

    semantic_labels = np.rint(np.asarray(semantic_vertex["object"], dtype=np.float64)).astype(np.int64)
    make_b_cloud(semantic_ply, semantic_vertex, semantic_xyz, trunk_vertex, trunk_xyz, output_b)

    # B 的标签：语义点保持原标签，追加主干固定为 100。
    xyz = np.vstack([semantic_xyz, trunk_xyz])
    labels = np.concatenate([semantic_labels, np.full(len(trunk_xyz), TRUNK_ID, dtype=np.int64)])
    stem_points = trunk_xyz

    length_scale = None
    area_scale = None
    reference_points = semantic_xyz[semantic_labels == REFERENCE_ID]
    reference_result = calibrate_reference(reference_points) if len(reference_points) else None
    if reference_result:
        length_scale = reference_result["length_scale_cm_per_unit"]
        area_scale = reference_result["area_scale_cm2_per_unit2"]

    # 果穗：逐 object 统计，并严格按“质心 y 最小”选择 A。
    ear_records = []
    for object_id in sorted(set(semantic_labels.tolist())):
        if not EAR_MIN_ID <= object_id <= EAR_MAX_ID:
            continue
        points = semantic_xyz[semantic_labels == object_id]
        if not len(points):
            continue
        centroid = points.mean(axis=0)
        ear_records.append({
            "object_id": int(object_id),
            "point_count": int(len(points)),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "y_min": float(points[:, 1].min()),
            "y_max": float(points[:, 1].max()),
            "y_median": float(np.median(points[:, 1])),
        })
    selected_ear = min(ear_records, key=lambda r: r["centroid_y"]) if ear_records else None
    stem_y_max = float(stem_points[:, 1].max()) if len(stem_points) else None
    stem_y_min = float(stem_points[:, 1].min()) if len(stem_points) else None

    # 叶片表型。
    leaf_records = []
    for object_id in sorted(set(semantic_labels.tolist())):
        if object_id < LEAF_MIN_ID:
            continue
        points = semantic_xyz[semantic_labels == object_id]
        if not len(points):
            continue
        centroid = points.mean(axis=0)
        if len(stem_points):
            base_y_max, base_y_centroid = leaf_base_statistics(
                points, stem_points, leaf_base_percent
            )
        else:
            base_y_max = float(np.max(points[:, 1]))
            base_y_centroid = float(np.mean(points[:, 1]))
        length_raw = calculate_leaf_length(points)
        avg_width_raw, max_width_raw = calculate_leaf_width(points)
        area_raw = calculate_leaf_area_alpha_shape(points)
        angle, component, scores, n_a, n_b = (
            calculate_leaf_angle(points, stem_points)
            if len(stem_points)
            else (None, None, None, 0, 0)
        )
        leaf_records.append({
            "plant": choose_scene_name(semantic_path, full_path),
            "object_id": int(object_id),
            "point_count": int(len(points)),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "base_y_max": base_y_max,
            "base_y_centroid": base_y_centroid,
            "length_raw": length_raw,
            "avg_width_raw": avg_width_raw,
            "max_width_raw": max_width_raw,
            "area_raw": area_raw,
            "angle": angle,
            "pca_component": f"PC{component}" if component else None,
            "pc1_score": float(scores[0]) if scores is not None and np.isfinite(scores[0]) else None,
            "pc2_score": float(scores[1]) if scores is not None and np.isfinite(scores[1]) else None,
            "a_points": n_a,
            "b_points": n_b,
        })

    # B：选定果穗 A 的 y 最小值。穗位叶只从 base_y > B 的叶片中选择。
    ear_y_min_B = (
        selected_ear["y_min"]
        if selected_ear is not None
        else (float(np.median(stem_points[:, 1])) if len(stem_points) else 0.0)
    )
    anchor_y, leaf_position_mode = infer_leaf_positions(leaf_records, ear_y_min_B)
    if anchor_y is None:
        anchor_y = ear_y_min_B

    # 参照物存在时使用真实单位；否则保留原始值并把换算列留空。
    for record in leaf_records:
        if length_scale is not None:
            record["length_cm"] = record["length_raw"] * length_scale
            record["avg_width_cm"] = record["avg_width_raw"] * length_scale
            record["max_width_cm"] = record["max_width_raw"] * length_scale
        else:
            record["length_cm"] = None
            record["avg_width_cm"] = None
            record["max_width_cm"] = None
        record["area_cm2"] = record["area_raw"] * area_scale if area_scale is not None else None
        record["anchor_y"] = anchor_y

    tassel_points = semantic_xyz[semantic_labels == TASSEL_ID]
    tassel_y_min = float(tassel_points[:, 1].min()) if len(tassel_points) else None
    # 按用户给出的顺序计算：主干 y_max - 目标对象 y_min。
    # 同时保留绝对值，便于结果核对。
    ear_height_signed = (
        stem_y_max - selected_ear["y_max"]
        if stem_y_max is not None and selected_ear is not None else None
    )
    plant_height_signed = (
        stem_y_max - tassel_y_min
        if tassel_y_min is not None and stem_y_max is not None else None
    )
    ear_height = ear_height_signed
    plant_height = plant_height_signed

    # 侧视轮廓面积沿用原算法：排除参照物，保留雄穗、果穗、叶片和主干。
    plant_mask = labels != REFERENCE_ID
    side_raw, side_theta = max_side_area(xyz[plant_mask])
    side_area_cm2 = None
    side_ref = ref_inlier_hull_area(reference_points) if len(reference_points) >= 50 else None
    side_scale = None
    if side_ref and side_ref[2] > 0:
        side_scale = (REF_REAL_LENGTH_CM / side_ref[2]) ** 2
        side_area_cm2 = side_raw * side_scale

    plant_row = {
        "plant": choose_scene_name(semantic_path, full_path),
        "semantic_point_cloud": str(semantic_path),
        "full_point_cloud": str(full_path),
        "trunk_point_cloud_H": str(trunk_path),
        "total_points_B": int(len(xyz)),
        "semantic_points": int(len(semantic_vertex)),
        "appended_trunk_points": int(len(trunk_vertex)),
        "trunk_object_id": TRUNK_ID,
        "leaf_count": len(leaf_records),
        "ear_count": len(ear_records),
        "tassel_points": int(len(tassel_points)),
        "reference_points": int(len(reference_points)),
        "reference_length_measured": reference_result["actual_length_3d"] if reference_result else None,
        "length_scale_cm_per_unit": length_scale,
        "area_scale_cm2_per_unit2": area_scale,
        "trunk_y_min": stem_y_min,
        "trunk_y_max": stem_y_max,
        "selected_ear_A_object": selected_ear["object_id"] if selected_ear else None,
        "selected_ear_A_centroid_y": selected_ear["centroid_y"] if selected_ear else None,
        "selected_ear_A_y_min_B": selected_ear["y_min"] if selected_ear else None,
        "selected_ear_A_y_max": selected_ear["y_max"] if selected_ear else None,
        "ear_level_leaf_base_y_centroid": anchor_y,
        "leaf_position_inference_mode": leaf_position_mode,
        "ear_height_signed": ear_height_signed,
        "ear_height_measurement": ear_height,
        "ear_height_absolute": abs(ear_height) if ear_height is not None else None,
        "ear_height_cm": ear_height * length_scale if ear_height is not None and length_scale else None,
        "id220_y_min": tassel_y_min,
        "plant_height_signed": plant_height_signed,
        "plant_height_measurement": plant_height,
        "plant_height_absolute": abs(plant_height) if plant_height is not None else None,
        "plant_height_cm": plant_height * length_scale if plant_height is not None and length_scale else None,
        "best_side_view_theta_deg": math.degrees(side_theta),
        "side_view_area_raw": side_raw,
        "side_view_area_cm2": side_area_cm2,
    }

    rows_ear = []
    for record in ear_records:
        row = {"plant": plant_row["plant"], **record}
        row["is_selected_ear_A"] = record["object_id"] == plant_row["selected_ear_A_object"]
        rows_ear.append(row)

    rows_id = []
    for object_id in sorted(set(labels.tolist())):
        mask = labels == object_id
        points = xyz[mask]
        if not len(points):
            continue
        rows_id.append({
            "plant": plant_row["plant"],
            "object": int(object_id),
            "class": "trunk" if object_id == TRUNK_ID else
            "tassel" if object_id == TASSEL_ID else
            "reference_object" if object_id == REFERENCE_ID else
            "ear" if EAR_MIN_ID <= object_id <= EAR_MAX_ID else
            "leaf" if object_id >= LEAF_MIN_ID else "other",
            "point_count": int(len(points)),
            "centroid_x": float(points[:, 0].mean()),
            "centroid_y": float(points[:, 1].mean()),
            "centroid_z": float(points[:, 2].mean()),
            "y_min": float(points[:, 1].min()),
            "y_max": float(points[:, 1].max()),
        })

    rows_leaf = []
    for record in leaf_records:
        rows_leaf.append({
            "plant": record["plant"],
            "leaf_index": record["leaf_index"],
            "semantic_object_id": record["object_id"],
            "leaf_name_inferred": record["position"],
            "leaf_position_group": record["position_group"],
            "leaf_point_count": record["point_count"],
            "leaf_base_y_max": record["base_y_max"],
            "leaf_base_y_centroid": record["base_y_centroid"],
            "ear_A_y_min_B": record["ear_y_min_B"],
            "ear_level_leaf_base_y_centroid": record["ear_level_leaf_base_y_centroid"],
            "ear_level_reference_y": record["anchor_y"],
            "leaf_position_inference_mode": record["leaf_position_mode"],
            "length_3d": record["length_raw"],
            "mean_width_3d": record["avg_width_raw"],
            "max_width_3d": record["max_width_raw"],
            "area_3d": record["area_raw"],
            "length_cm": record["length_cm"],
            "mean_width_cm": record["avg_width_cm"],
            "max_width_cm": record["max_width_cm"],
            "area_cm2": record["area_cm2"],
            "leaf_angle_deg": record["angle"],
            "selected_pca_component": record["pca_component"],
            "pc1_score": record["pc1_score"],
            "pc2_score": record["pc2_score"],
            "a_point_count": record["a_points"],
            "b_point_count": record["b_points"],
        })

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        pd.DataFrame([plant_row]).to_excel(writer, sheet_name="Plant_Summary", index=False)
        pd.DataFrame(rows_leaf).to_excel(writer, sheet_name="Leaf_Phenotype", index=False)
        pd.DataFrame(rows_ear).to_excel(writer, sheet_name="Ear_Analysis", index=False)
        pd.DataFrame(rows_id).to_excel(writer, sheet_name="ID_Statistics", index=False)

    return plant_row, rows_leaf, rows_ear, rows_id


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", type=Path, required=True, help="Final semantic point-cloud PLY")
    parser.add_argument("--full", type=Path, required=True, help="Corresponding full point-cloud PLY")
    parser.add_argument("--trunk", type=Path, default=None, help="Optional explicit trunk PLY")
    parser.add_argument("--h-model", type=Path, default=DEFAULT_H_MODEL,
                        help="Bundled trained H model (model_H_final.pkl)")
    parser.add_argument("--h-infer-script", type=Path, default=DEFAULT_H_INFER,
                        help="H-v3 direct inference script")
    parser.add_argument("--h-report-b", type=Path, default=DEFAULT_H_REPORT_B,
                        help="H-v3 B geometry configuration")
    parser.add_argument("--h-python", type=Path,
                        default=Path(os.environ["H_MODEL_PYTHON"])
                        if os.environ.get("H_MODEL_PYTHON") else None,
                        help="Python executable for H-v3 model inference")
    parser.add_argument("--h-mode", choices=("balanced", "precision", "recall"),
                        default="balanced", help="H model delivery threshold mode")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: semantic PLY directory)")
    parser.add_argument("--output-b", type=Path, default=None, help="B point-cloud output path")
    parser.add_argument("--output-xlsx", type=Path, default=None, help="Excel output path")
    parser.add_argument("--leaf-base-percent", type=float, default=30.0,
                        help="Percentage of leaf points nearest to the trunk used for base height (default: 30)")
    return parser


def main():
    args = build_parser().parse_args()
    semantic = args.semantic.resolve()
    full = args.full.resolve()
    if not semantic.exists():
        raise FileNotFoundError(semantic)
    if not full.exists():
        raise FileNotFoundError(full)

    explicit_trunk = args.trunk.resolve() if args.trunk else None
    if explicit_trunk is None:
        learned_output = full.parent / "point_cloud_trunk.ply"
        print(f"使用训练好的 H 模型生成: {args.h_model}")
        # H-v3 features were trained from the unsplit, postprocessed semantic
        # cloud; keep that input for prediction while the split cloud is still
        # used below for per-instance phenotype measurements.
        h_semantic = semantic.with_name("point_cloud_object_post.ply")
        if not h_semantic.exists():
            h_semantic = semantic
        explicit_trunk = generate_h_trunk(
            args.h_infer_script.resolve(), args.h_model.resolve(), args.h_report_b.resolve(),
            full, h_semantic, learned_output, args.h_mode,
            args.h_python.resolve() if args.h_python else None,
        )
    trunk = explicit_trunk
    if trunk is None or not trunk.is_file():
        raise FileNotFoundError(f"主干点云不存在: {trunk}")
    output_dir = args.output_dir.resolve() if args.output_dir else semantic.parent
    scene = choose_scene_name(semantic, full)
    output_b = args.output_b.resolve() if args.output_b else output_dir / f"{scene}_B.ply"
    output_xlsx = args.output_xlsx.resolve() if args.output_xlsx else output_dir / f"{scene}_phenotype.xlsx"

    plant, leaves, ears, ids = process_one(
        semantic, full, trunk, output_b, output_xlsx,
        args.leaf_base_percent, 0.0,
    )
    print(f"语义点云: {semantic}")
    print(f"完整点云: {full}")
    print(f"主干点云: {trunk}")
    print(f"B point cloud: {output_b}")
    print(f"Excel: {output_xlsx}")
    print(f"Leaves: {len(leaves)}, ears: {len(ears)}, IDs: {len(ids)}")
    print(f"Selected ear A object: {plant['selected_ear_A_object']}")
    print(f"Ear height: {plant['ear_height_measurement']}, "
          f"plant height: {plant['plant_height_measurement']}")


if __name__ == "__main__":
    main()
