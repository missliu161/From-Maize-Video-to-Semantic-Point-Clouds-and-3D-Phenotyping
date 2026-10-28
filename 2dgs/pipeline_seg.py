"""
pipeline_seg.py - 一体化语义分割重建流水线

YOLO分割 → 跨帧实例匹配(3D高斯质心) → GS-ID投票赋语义 → 导出语义点云

支持:
  单文件夹: python pipeline_seg.py /path/to/folder [options]
  批量处理: python pipeline_seg.py /path/to/parent_dir --batch [options]

流程:
  1. YOLO分割 → masks_results/integer_masks/ + class_info
  2. 渲染GS-ID图（仅一次，后续匹配+voting复用）
  3. 计算实例3D质心 + 跨帧3D/GS联合匹配
  4. 生成unified masks（全局一致实例ID）
  5. GS-ID多数投票 → 每个高斯获得语义标签
  6. 导出 point_cloud_object.ply（含object属性，自动过滤背景）
"""

import os
import sys
import re
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from PIL import Image as PILImage
from sklearn.cluster import DBSCAN

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams

try:
    from ultralytics import YOLO as YOLOModel
except ImportError:
    YOLOModel = None


# ───────────────────────── 默认常量 ─────────────────────────
# YOLO分割参数 (方案A调优)
YOLO_CONF = 0.15          # 原 0.25，更激进召回
YOLO_IOU = 0.45
YOLO_MAX_DET = 30
YOLO_IMGSZ = 1280         # 原 640，提升小目标检测

MIN_PIXELS = 10           # 原 5，过滤过小mask
CENTROID_OUTLIER_MULT = 2.0
SPATIAL_THRESHOLD = 1.5   # 原 2.0，紧缩距离阈值（PlanA最优）

# ──────────── YOLO 类别约定与 global ID 分配 ────────────
# YOLO 模型实际类别顺序 (model.names):
#   class 0 = ear       (固定 ID 220)
#   class 1 = fruit     (内部类别 2, global ID 222+)
#   class 2 = leaf      (内部类别 3, global ID 300+)
#   class 3 = obj       (固定 ID 221)
# 因此: leaf → global ID 300+, fruit → global ID 222+
CLASS2_START_ID = 222   # fruit 起点
CLASS3_START_ID = 300   # leaf 起点

GS_OVERLAP_THRESHOLD = 0.05
GS_IOU_THRESHOLD = 0.01
GS_OVERLAP_DISTANCE_MULT = 2.0
GS_OVERLAP_WEIGHT = 0.9    # 原 0.7，增强重叠权重
GS_IOU_WEIGHT = 0.1       # 原 0.3，降低IoU权重
GS_NO_OVERLAP_DISTANCE_MULT = 0.5
GS_NO_OVERLAP_FRAME_GAP = 12
GS_NO_OVERLAP_PENALTY = 0.8

DBSCAN_EPS = 0.1
DBSCAN_MIN_SAMPLES = 10

# ── EMA质心更新参数 ──
# 使用指数滑动平均稳定参考质心，防止单帧抖动带偏
EMA_ALPHA = 0.2                 # EMA更新系数（越小越稳定，0.1-0.3推荐）

# 固定 ID 分配 (YOLO class 0/3 不参与匹配, 写入 unified_masks 用固定 global ID)
FIXED_CLASS_MAP = {0: 220, 3: 221}   # YOLO cls 0 (ear) → 220, cls 3 (obj) → 221
FIXED_PALETTE = {220: (0, 0, 255), 221: (0, 255, 0)}

# Association configuration used by the published workflow.
MATCHING_CONFIG = {
    # 维度 1：核心组件（核心算法：3D 距离 + GS 高斯重叠）
    "use_3d_distance": True,          # False → 不用 3D 距离
    "use_gs_overlap": True,           # False → 不用高斯重叠
    "use_overlap": True,              # False → 不用 overlap
    "use_iou": True,                  # False → 不用 IoU
    "use_fallback": True,             # False → 无重叠时不 fallback 到距离匹配

    # 维度 2：距离度量
    "dist_metric": "euclidean",       # "euclidean"|"manhattan"|"xy_only"

    # 维度 3：GS 重叠权重
    "gs_overlap_weight": 0.9,
    "gs_iou_weight": 0.1,

    # 维度 4：空间距离阈值
    "spatial_threshold": 1.5,

    # 维度 5：EMA 系数（0 = 禁用 EMA，直接覆盖）
    "ema_alpha": 0.2,

    # 维度 6：过滤没有有效高斯贡献的背景像素
    "alpha_threshold": 1e-3,
}
FIXED_PALETTE = {220: (0, 0, 255), 221: (0, 255, 0)}

_DISTINCT_COLORS_RGB = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 0, 128), (0, 128, 0), (128, 0, 0), (0, 128, 128),
    (128, 0, 128), (128, 128, 0), (0, 165, 255), (203, 192, 255), (42, 42, 165),
    (127, 255, 0), (147, 20, 255), (211, 0, 148), (0, 215, 255), (255, 191, 0),
    (140, 230, 240), (47, 255, 173), (193, 182, 255), (96, 164, 244), (30, 105, 210),
    (71, 99, 255), (0, 140, 255), (32, 165, 218), (50, 205, 154), (204, 209, 72),
]

# Set YOLO_MODEL_PATH to a local checkpoint, or place it at models/yolo_seg.pt.
DEFAULT_YOLO_PATH = os.environ.get(
    "YOLO_MODEL_PATH", str(Path(__file__).resolve().parent / "models" / "yolo_seg.pt")
)
# ───────────────────── 工具函数 ─────────────────────

def extract_frame_number(filename: str) -> int:
    matches = re.findall(r'(\d+)', Path(filename).stem)
    return int(matches[-1]) if matches else -1


def resolve_camera_frame(cam, camera_index, available_frames):
    """Resolve a rendered camera to an existing mask frame."""
    frame_name = str(cam.image_name)
    fn = extract_frame_number(frame_name)
    if fn >= 0:
        return fn
    fallback = camera_index + 1
    return fallback if fallback in available_frames else -1


def find_images_dir(folder: Path):
    for candidate in ["images_real", "images"]:
        p = folder / candidate
        if p.is_dir():
            return p
    return None


def find_model_dir(folder: Path):
    for pat in ["output_*", "output"]:
        hits = sorted(folder.glob(pat))
        if hits:
            return Path(hits[0])
    return None


def _gid_to_color(gid):
    """基于gid生成确定性颜色（同样的gid永远得到同样的颜色）。

    用于跨实验/跨序列比对：第一帧的同gid实例在不同实验里颜色一致，
    便于直观看出匹配差异。
    """
    seed = (int(gid) * 2654435761) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    return tuple(int(x) for x in rng.randint(50, 256, size=3))


# ─────────── Step 1: YOLO 分割 ───────────

def step_yolo_segmentation(model, images_dir: Path, output_dir: Path, quiet=False):
    int_mask_dir = output_dir / "integer_masks"
    color_mask_dir = output_dir / "colored_masks"
    overlay_dir = output_dir / "overlay_masks"
    for d in [int_mask_dir, color_mask_dir, overlay_dir]:
        d.mkdir(parents=True, exist_ok=True)

    exts = ('.jpg', '.jpeg', '.png')
    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.lower().endswith(exts)])
    if not image_files:
        return 0, 0

    success = 0
    for img_name in tqdm(image_files, desc="  YOLO", disable=quiet):
        img_path = images_dir / img_name
        base_name = Path(img_name).stem
        try:
            original_img = cv2.imread(str(img_path))
            if original_img is None:
                continue

            results = model.predict(str(img_path), conf=YOLO_CONF, iou=YOLO_IOU,
                                     verbose=False, retina_masks=True,
                                     max_det=YOLO_MAX_DET, imgsz=YOLO_IMGSZ)
            res = results[0]
            H_ori, W_ori = res.orig_img.shape[:2]

            integer_mask = np.zeros((H_ori, W_ori), dtype=np.uint16)
            color_mask = np.zeros((H_ori, W_ori, 3), dtype=np.uint8)
            overlay_img = original_img.copy()
            class_info = []

            if res.masks is not None and len(res.masks) > 0:
                masks = res.masks.data.cpu().numpy()
                classes = (res.boxes.cls.cpu().numpy().astype(int)
                           if res.boxes is not None
                           else np.zeros(len(masks), dtype=int))
                for idx, mask in enumerate(masks):
                    m = cv2.resize(mask.astype(np.float32), (W_ori, H_ori),
                                   interpolation=cv2.INTER_NEAREST)
                    binary = (m > 0.01).astype(np.uint8)
                    if binary.sum() < 10:
                        continue
                    inst_id = idx + 1
                    integer_mask[binary > 0] = inst_id
                    r = (idx * 67 + 100) % 256
                    g = (idx * 137 + 50) % 256
                    b = (idx * 193) % 256
                    color = [b, g, r]
                    color_mask[binary > 0] = color
                    class_info.append({
                        "instance_id": int(inst_id),
                        "class_id": int(classes[idx]),
                        "color": [int(b), int(g), int(r)]
                    })
                    mc = np.zeros_like(overlay_img, dtype=np.float32)
                    mc[:] = color
                    overlay_img = np.where(
                        binary[:, :, np.newaxis] > 0,
                        (overlay_img * 0.7 + mc * 0.3).astype(np.uint8),
                        overlay_img
                    )
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay_img, contours, -1, color, 1)

            cv2.imwrite(str(int_mask_dir / f"mask_{base_name}.png"), integer_mask)
            cv2.imwrite(str(color_mask_dir / f"colored_mask_{base_name}.png"), color_mask)
            cv2.imwrite(str(overlay_dir / f"overlay_{base_name}.png"), overlay_img)
            with open(int_mask_dir / f"class_info_{base_name}.json", 'w') as f:
                json.dump({"image": img_name, "instances": class_info}, f, indent=2)
            success += 1
        except Exception as e:
            if not quiet:
                print(f"  ❌ {img_name}: {e}")
    return success, len(image_files)


# ─────────── Step 2: 加载模型 + 渲染 GS-ID 图 ───────────

def step_load_and_render(folder: Path, model_dir: Path, quiet=False):
    folder, model_dir = Path(folder).resolve(), Path(model_dir).resolve()

    args = Namespace(
        source_path=str(folder),
        model_path=str(model_dir),
        images="images",
        resolution=-1,
        white_background=False,
        data_device="cuda",
        eval=False,
        sh_degree=3,
        depth_ratio=0.0,
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    dataset = ModelParams(ArgumentParser(), sentinel=False).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    bg_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not quiet:
        print(f"  加载模型: {model_dir}")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    train_cameras = scene.getTrainCameras()

    if not quiet:
        n_gs = gaussians._xyz.shape[0]
        n_cam = len(train_cameras)
        print(f"  高斯点数: {n_gs}, 训练相机: {n_cam}")

    if gaussians._object.numel() == 0:
        gaussians._object = torch.zeros(gaussians._xyz.shape[0],
                                         dtype=torch.long, device="cuda")

    if not quiet:
        print(f"  渲染 GS-ID 图...")
    gs_id_maps = []
    alpha_threshold = MATCHING_CONFIG.get("alpha_threshold", 1e-3)
    for cam in tqdm(train_cameras, desc="  render", disable=quiet):
        render_pkg = render(cam, gaussians, pipe, bg_tensor)
        gs_id_map = render_pkg["gs_id"]
        alpha_map = render_pkg["rend_alpha"].squeeze(0)
        gs_id_map = torch.where(
            alpha_map > alpha_threshold,
            gs_id_map,
            torch.full_like(gs_id_map, -1),
        )
        gs_id_maps.append(gs_id_map)

    return gaussians, train_cameras, gs_id_maps, bg_tensor


# ─────────── Step 3: 计算实例 3D 质心 ───────────

def compute_instance_geometry(mask_region_mask, gs_id_map, pc):
    ys, xs = torch.where(mask_region_mask)
    if len(ys) < MIN_PIXELS:
        return None
    H, W = gs_id_map.shape[:2]
    ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    ys, xs = ys[ok], xs[ok]
    if len(ys) < MIN_PIXELS:
        return None

    local_gs_ids = gs_id_map[ys, xs].long()
    valid = (local_gs_ids >= 0) & (local_gs_ids < pc._xyz.shape[0])
    local_gs_ids = local_gs_ids[valid]
    if local_gs_ids.numel() < MIN_PIXELS:
        return None

    local_xyz = pc._xyz[local_gs_ids]
    centroid = local_xyz.mean(dim=0)
    dists = torch.norm(local_xyz - centroid, dim=1)
    inlier = dists < (dists.median() * CENTROID_OUTLIER_MULT + 0.01)
    if inlier.sum() < MIN_PIXELS:
        return None
    visible_ids = local_gs_ids[inlier].unique().detach().cpu().numpy()
    visible_ids = {int(v) for v in visible_ids.tolist()}
    centroid = local_xyz[inlier].mean(dim=0).detach().cpu().numpy()
    return centroid, visible_ids


def _resize_mask_to_gs(mask, gs_shape):
    """E1 修复: 把 YOLO mask resize 到 GS-ID 图尺寸，避免像素错位。"""
    if mask.shape[:2] == gs_shape:
        return mask
    return cv2.resize(mask.astype(np.int32), (gs_shape[1], gs_shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(mask.dtype)


def step_compute_centroids(gaussians, train_cameras, gs_id_maps,
                            yolo_masks, class_info_map, quiet=False):
    all_instances = {}
    available_frames = set(yolo_masks)

    for i, cam in enumerate(tqdm(train_cameras, desc="  质心", disable=quiet)):
        fn = resolve_camera_frame(cam, i, available_frames)
        if fn not in yolo_masks:
            continue
        mask = yolo_masks[fn]
        gid_map = gs_id_maps[i]
        # E1: 把 mask 缩放到 GS-ID 图尺寸
        if mask.shape[:2] != gid_map.shape[:2]:
            mask = _resize_mask_to_gs(mask, gid_map.shape)
        ci = class_info_map.get(fn)
        inst_to_class = {}
        if ci:
            for inst in ci["instances"]:
                inst_to_class[inst["instance_id"]] = inst["class_id"]

        for local_id in np.unique(mask):
            if local_id == 0:
                continue
            region = torch.from_numpy(mask == local_id).cuda()
            geometry = compute_instance_geometry(region, gid_map, gaussians)
            if geometry is not None:
                centroid, visible_ids = geometry
                class_id = inst_to_class.get(int(local_id), 0)
                all_instances[(fn, int(local_id))] = {
                    "frame": fn, "local_id": int(local_id),
                    "class_id": class_id, "centroid": centroid,
                    "gaussian_ids": visible_ids,
                }
    if not quiet:
        print(f"  总实例-帧对: {len(all_instances)}")
    return all_instances


# ─────────── Step 4: 跨帧匹配 ───────────

def _weighted_dist(c1, c2):
    """
    根据 MATCHING_CONFIG["dist_metric"] 选择距离度量。

    Args:
        c1, c2: 3D坐标 (x, y, z) numpy数组

    Returns:
        float: 距离
    """
    metric = MATCHING_CONFIG.get("dist_metric", "euclidean")
    if metric == "manhattan":
        return float(np.abs(c1 - c2).sum())
    if metric == "xy_only":
        return float(np.linalg.norm(c1[:2] - c2[:2]))
    return float(np.linalg.norm(c1 - c2))


def _fallback_match_score(dist, ref, fn):
    last_frame = ref.get("last_frame", -1)
    frame_gap = fn - last_frame if last_frame >= 0 else 0
    if last_frame >= 0 and (frame_gap <= 0 or frame_gap > GS_NO_OVERLAP_FRAME_GAP):
        return None
    sp_t = MATCHING_CONFIG.get("spatial_threshold", SPATIAL_THRESHOLD)
    if dist >= sp_t * GS_NO_OVERLAP_DISTANCE_MULT:
        return None
    dist_score = dist / max(sp_t, 1e-6)
    return dist_score + GS_NO_OVERLAP_PENALTY


def _match_score(inst, ref, fn):
    dist = _weighted_dist(inst["centroid"], ref["centroid"])
    use_3d = MATCHING_CONFIG.get("use_3d_distance", True)
    use_gs = MATCHING_CONFIG.get("use_gs_overlap", True)
    use_overlap = use_gs and MATCHING_CONFIG.get("use_overlap", True)
    use_iou = use_gs and MATCHING_CONFIG.get("use_iou", True)
    use_fallback = MATCHING_CONFIG.get("use_fallback", True)
    gs_w = MATCHING_CONFIG.get("gs_overlap_weight", GS_OVERLAP_WEIGHT)
    gs_iou_w = MATCHING_CONFIG.get("gs_iou_weight", GS_IOU_WEIGHT)
    sp_t = MATCHING_CONFIG.get("spatial_threshold", SPATIAL_THRESHOLD)

    inst_ids = inst.get("gaussian_ids", set())
    ref_ids = ref.get("gaussian_ids", set())
    overlap = 0.0
    iou = 0.0
    gs_enabled = use_overlap or use_iou

    if gs_enabled:
        # GS-only 模式必须有真实交集；零证据候选不能以 0 分进入分配。
        if inst_ids and ref_ids:
            inter = len(inst_ids & ref_ids)
        else:
            inter = 0

        if inter > 0:
            overlap = inter / max(1, min(len(inst_ids), len(ref_ids)))
            iou = inter / max(1, len(inst_ids | ref_ids))
            gs_ok = ((use_overlap and overlap >= GS_OVERLAP_THRESHOLD)
                     or (use_iou and iou >= GS_IOU_THRESHOLD))
            if gs_ok:
                if use_3d and dist >= sp_t * GS_OVERLAP_DISTANCE_MULT:
                    return None
                score = 0.0
                if use_3d:
                    score += dist / max(sp_t, 1e-6)
                if use_overlap:
                    score -= gs_w * overlap
                if use_iou:
                    score -= gs_iou_w * iou
                return score, dist, overlap, iou

        # GS 证据缺失或不足时，只能通过受限 fallback 匹配。
        if use_fallback and use_3d:
            fallback_score = _fallback_match_score(dist, ref, fn)
            if fallback_score is not None:
                return fallback_score, dist, overlap, iou
        return None

    # 只有明确关闭 GS 信号时，才执行纯距离匹配。
    if use_3d:
        if dist >= sp_t:
            return None
        return dist / max(sp_t, 1e-6), dist, overlap, iou

    return None


def _update_reference(ref, inst, fn):
    """
    使用EMA更新参考质心，保留历史质心用于长期重识别。

    改进点：
      - 防止单帧YOLO分割抖动把参考质心带偏
      - 保留短期适应能力，适合旋转视角变化
      - 保留历史质心用于长期重识别
    """
    inst_centroid = inst["centroid"]
    ema_alpha = MATCHING_CONFIG.get("ema_alpha", EMA_ALPHA)

    if "centroid" in ref and ref["centroid"] is not None:
        if ema_alpha > 0:
            # EMA更新: ref = (1-alpha) * ref + alpha * inst
            ref["centroid"] = ((1 - ema_alpha) * ref["centroid"]
                              + ema_alpha * inst_centroid)
        else:
            # EMA 禁用，直接覆盖
            ref["centroid"] = inst_centroid
    else:
        # 第一次出现，直接赋值
        ref["centroid"] = inst_centroid

    ref["last_frame"] = fn
    ref.setdefault("gaussian_ids", set()).update(inst.get("gaussian_ids", set()))


def step_matching(all_instances, ref_centroids,
                  class2_counter, class3_counter, quiet=False):
    ref_by_class = {2: {}, 3: {}}
    for gid, ref in ref_centroids.items():
        cid = ref["class_id"]
        if cid in ref_by_class:
            ref_by_class[cid][gid] = ref

    new_global_mapping = {}
    matched = 0
    new_ids_count = 0
    instances_by_frame_class = defaultdict(list)

    for key in sorted(all_instances, key=lambda k: (k[0], k[1])):
        fn, local_id = key
        inst = all_instances[key]
        class_id = inst["class_id"]
        # YOLO 类别过滤: 跳过 YOLO cls 0 (ear) 和 cls 3 (obj), 不参与跨帧匹配
        if class_id in (0, 3):
            continue
        # YOLO 类别 → 内部匹配通道:
        #   YOLO cls 1 (fruit) → cid 2 → 写入 gid 222+ (CLASS2_START_ID)
        #   YOLO cls 2 (leaf)  → cid 3 → 写入 gid 300+ (CLASS3_START_ID)
        cid = 2 if class_id == 1 else (3 if class_id == 2 else 2)
        instances_by_frame_class[(fn, cid)].append(key)

    for (fn, cid), keys in sorted(instances_by_frame_class.items()):
        candidates = []
        for key in keys:
            inst = all_instances[key]
            for gid, ref in ref_by_class[cid].items():
                if ref.get("last_frame") == fn:
                    continue
                score = _match_score(inst, ref, fn)
                if score is not None:
                    final_score = score[0]
                    candidates.append((final_score, score[1], -score[2], key, gid))

        final_candidates = []
        for item in candidates:
            final_score, _, _, key, gid = item
            final_candidates.append((final_score, key, gid))

        used_keys = set()
        used_gids = set()
        for _, key, gid in sorted(final_candidates):
            if key in used_keys or gid in used_gids:
                continue
            new_global_mapping[key] = gid
            _update_reference(ref_by_class[cid][gid], all_instances[key], fn)
            used_keys.add(key)
            used_gids.add(gid)
            matched += 1

        for key in keys:
            if key in used_keys:
                continue
            inst = all_instances[key]
            class_id = inst["class_id"]

            # 新 ID 分配: YOLO cls 1 (fruit) → class2_counter (gid 222+)
            #            YOLO cls 2 (leaf)  → class3_counter (gid 300+)
            if class_id == 1:
                global_id = class2_counter
                class2_counter += 1
            elif class_id == 2:
                global_id = class3_counter
                class3_counter += 1
            else:
                global_id = class2_counter
                class2_counter += 1
            new_global_mapping[key] = global_id
            new_ids_count += 1
            ref_by_class[cid][global_id] = {
                "centroid": inst["centroid"],
                "class_id": cid,
                "gaussian_ids": set(inst.get("gaussian_ids", set())),
                "last_frame": fn,
            }

    if not quiet:
        print(f"  ✅ 匹配: {matched}, 新ID: {new_ids_count}")
    return new_global_mapping, class2_counter, class3_counter


# ─────────── Step 5: 生成 Unified Masks ───────────

def step_generate_unified_masks(yolo_masks, class_info_map, new_global_mapping,
                                 output_dir, quiet=False):
    unified_dir = output_dir / "unified_masks"
    color_dir = output_dir / "color_masks"
    unified_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)

    all_ids_in_masks = set()
    for fn in yolo_masks:
        ym = yolo_masks[fn]
        ci = class_info_map.get(fn)
        l2c = {}
        if ci:
            for inst in ci["instances"]:
                l2c[inst["instance_id"]] = inst["class_id"]
        for local_id in np.unique(ym):
            if local_id == 0:
                continue
            lid = int(local_id)
            cid = l2c.get(lid)
            if cid in FIXED_CLASS_MAP:
                all_ids_in_masks.add(FIXED_CLASS_MAP[cid])
            elif (fn, lid) in new_global_mapping:
                all_ids_in_masks.add(new_global_mapping[(fn, lid)])

    np.random.seed(42)
    id_colors = {0: (0, 0, 0)}
    for gid in sorted(all_ids_in_masks):
        if gid in FIXED_PALETTE:
            id_colors[gid] = FIXED_PALETTE[gid]
        else:
            # 确定性颜色：同gid在不同实验里颜色一致
            id_colors[gid] = _gid_to_color(gid)

    first_frame_lid_colors = {}
    if yolo_masks:
        first_fn = min(yolo_masks.keys())
        ym = yolo_masks[first_fn]
        first_frame_lids = []
        for local_id in np.unique(ym):
            if local_id == 0:
                continue
            lid = int(local_id)
            mask = ym == local_id
            ys, xs = np.where(mask)
            if len(ys) > 0:
                cy = (int(ys.min()) + int(ys.max())) / 2
                first_frame_lids.append((lid, cy))

        first_frame_lids.sort(key=lambda x: x[1])
        for i, (lid, cy) in enumerate(first_frame_lids):
            color = _DISTINCT_COLORS_RGB[i % len(_DISTINCT_COLORS_RGB)]
            first_frame_lid_colors[lid] = color
            key = (first_fn, lid)
            if key in new_global_mapping:
                id_colors[new_global_mapping[key]] = color

    for fn in tqdm(sorted(yolo_masks), desc="  unified", disable=quiet):
        ym = yolo_masks[fn]
        unified = np.zeros_like(ym, dtype=np.int32)
        colored = np.zeros((*ym.shape, 3), dtype=np.uint8)
        ci = class_info_map.get(fn)
        l2c = {}
        if ci:
            for inst in ci["instances"]:
                l2c[inst["instance_id"]] = inst["class_id"]
        for local_id in np.unique(ym):
            if local_id == 0:
                continue
            lid = int(local_id)
            cid = l2c.get(lid)
            if cid in FIXED_CLASS_MAP:
                gid = FIXED_CLASS_MAP[cid]
                mask = ym == local_id
                unified[mask] = gid
                colored[mask] = first_frame_lid_colors.get(lid, id_colors[gid]) if fn == first_fn else id_colors[gid]
                continue
            key = (fn, lid)
            if key in new_global_mapping:
                gid = new_global_mapping[key]
                mask = ym == local_id
                unified[mask] = gid
                colored[mask] = first_frame_lid_colors.get(lid, id_colors[gid]) if fn == first_fn else id_colors[gid]
            elif fn == min(yolo_masks.keys()) and lid in first_frame_lid_colors:
                mask = ym == local_id
                colored[mask] = first_frame_lid_colors[lid]

        max_val = unified.max()
        if max_val < 65536:
            PILImage.fromarray(unified.astype(np.uint16), mode="I;16").save(
                str(unified_dir / f"unified_mask_{fn:04d}.png"))
        else:
            PILImage.fromarray(unified.astype(np.uint32), mode="I").save(
                str(unified_dir / f"unified_mask_{fn:04d}.png"))
        cv2.imwrite(str(color_dir / f"color_mask_{fn:04d}.png"),
                    colored[:, :, ::-1])

    return unified_dir, id_colors


# ─────────── Step 6: GS-ID 投票 ───────────

def accumulate_counts(id_tensor, sem_mask, counts_total):
    if sem_mask.ndim == 3:
        if sem_mask.shape[0] == 3:
            sem_mask = sem_mask.permute(1, 2, 0)
        sem_mask = (0.2989 * sem_mask[..., 0] +
                    0.5870 * sem_mask[..., 1] +
                    0.1140 * sem_mask[..., 2]).round().long()
    elif sem_mask.shape[0] == 1:
        sem_mask = sem_mask.squeeze(0).long()

    flat_ids = id_tensor.view(-1).long()
    flat_sem = sem_mask.view(-1).long()
    valid = flat_ids >= 0
    flat_ids = flat_ids[valid]
    flat_sem = flat_sem[valid]
    nc = counts_total.shape[1]
    indices = flat_ids * nc + flat_sem
    bc = torch.bincount(indices, minlength=counts_total.shape[0] * nc)
    counts = bc.view(-1, nc)
    counts_total[:counts.shape[0]] += counts
    return counts_total


def remove_outlier_semantics(gaussians, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    xyz = gaussians._xyz.detach().cpu().numpy()
    labels = gaussians._object.detach().cpu().numpy()
    new_labels = labels.copy()
    for sem_id in np.unique(labels):
        if sem_id == 0:
            continue
        sem_mask = (labels == sem_id)
        sem_xyz = xyz[sem_mask]
        sem_idx = np.where(sem_mask)[0]
        if len(sem_idx) < min_samples:
            new_labels[sem_idx] = 0
            continue
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(sem_xyz)
        cl = clustering.labels_
        if np.all(cl == -1):
            new_labels[sem_idx] = 0
            continue
        major = np.argmax(np.bincount(cl[cl >= 0]))
        new_labels[sem_idx[cl != major]] = 0
    gaussians._object = torch.from_numpy(new_labels).to(
        gaussians._object.device).long()
    return gaussians


def step_gsid_voting(gaussians, train_cameras, gs_id_maps,
                      unified_dir, yolo_masks, quiet=False):
    # 先扫描所有 unified masks 找到最大标签值，确定 num_classes
    max_label = 0
    for fn in yolo_masks:
        p = unified_dir / f"unified_mask_{fn:04d}.png"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if m is not None:
                max_label = max(max_label, m.max())
    nc = int(max_label) + 100
    nc = min(max(nc, 100), 10000)
    if not quiet:
        print(f"  max_label={max_label}, num_classes={nc}")

    n_gs = gaussians._xyz.shape[0]
    counts_total = torch.zeros((n_gs, nc), device='cuda')

    available_frames = set(yolo_masks)
    for i, cam in enumerate(tqdm(train_cameras, desc="  voting", disable=quiet)):
        fn = resolve_camera_frame(cam, i, available_frames)
        if fn not in yolo_masks:
            continue
        gid_map = gs_id_maps[i]
        p = unified_dir / f"unified_mask_{fn:04d}.png"
        if not p.exists():
            continue
        sm = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if sm is None:
            continue
        gh, gw = gid_map.shape[:2]
        if sm.shape[:2] != (gh, gw):
            sm = cv2.resize(sm, (gw, gh), interpolation=cv2.INTER_NEAREST)
        sm_t = torch.from_numpy(sm.astype(np.int64)).cuda()
        counts_total = accumulate_counts(gid_map, sm_t, counts_total)

    _object = counts_total.argmax(dim=1)
    gaussians._object = _object.clone()

    if not quiet:
        print(f"  投票后标签数: {torch.unique(_object).numel()}")
    remove_outlier_semantics(gaussians, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    if not quiet:
        print(f"  去离群后标签数: {torch.unique(gaussians._object).numel()}")
    return gaussians


# ─────────── Step 7: 导出 ───────────

def _object_palette_color(obj_id):
    if obj_id == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.RandomState(int(obj_id) % 1000000)
    return tuple(rng.uniform(0.3, 1.0, 3))


def step_export_ply(gaussians, output_dir, quiet=False):
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(gaussians, '_object') and gaussians._object.numel() > 0:
        obj_ids = gaussians._object.long()
        unique_ids = torch.unique(obj_ids)
        max_id = int(unique_ids.max().item())
        palette = np.zeros((max_id + 1, 3), dtype=np.float32)
        for oid in range(1, max_id + 1):
            palette[oid] = _object_palette_color(oid)
        palette_t = torch.tensor(palette, device=gaussians._rgb.device, dtype=torch.float)
        gaussians._rgb.data = palette_t[obj_ids]

    path = output_dir / "point_cloud_object.ply"
    gaussians.save_ply(str(path))
    if not quiet:
        print(f"  ✅ 语义点云: {path}")


def save_id_mapping(new_global_mapping, all_instances, yolo_masks,
                     output_dir, quiet=False):
    frame_mappings = defaultdict(dict)
    for (fn, lid), gid in new_global_mapping.items():
        frame_mappings[str(fn)][str(lid)] = gid

    all_gids = sorted(set(new_global_mapping.values()))
    class_votes = defaultdict(list)
    for k, gid in new_global_mapping.items():
        inst = all_instances.get(k)
        if inst is not None:
            class_votes[gid].append(inst["class_id"])
    cid_for_gid = {}
    for gid, votes in class_votes.items():
        cid_for_gid[gid] = Counter(votes).most_common(1)[0][0]

    target_ids = {"2": [], "3": []}
    export = {}
    for gid in all_gids:
        if gid in (220, 221):
            continue
        frames, cents = [], []
        for k, mgid in new_global_mapping.items():
            if mgid == gid:
                frames.append(k[0])
                if k in all_instances:
                    cents.append(all_instances[k]["centroid"])
        if not cents:
            continue
        cid = cid_for_gid.get(gid, 2)
        avg = np.mean(cents, axis=0).tolist()
        dists = [np.linalg.norm(c - np.array(avg)) for c in cents]
        avg_d = float(np.mean(dists)) if dists else 0.0
        if cid == 1:
            target_ids["2"].append(gid)
        elif cid == 2:
            target_ids["3"].append(gid)
        export[str(gid)] = {
            "point_3d": avg,
            "frames": frames,
            "num_views": len(set(frames)),
            "confidence": min(1.0, len(frames) / 10),
            "avg_distance": avg_d,
            "class_id": cid,
            "is_fixed": False,
        }

    data = {
        "method": "pipeline_seg",
        "matching_config": MATCHING_CONFIG.copy(),
        "spatial_threshold": MATCHING_CONFIG["spatial_threshold"],
        "source_path": str(output_dir),
        "fixed_class_ids": FIXED_CLASS_MAP,
        "frame_mappings": dict(frame_mappings),
        "global_instances": export,
        "summary": {
            "total_frames": len(yolo_masks),
            "total_unique_instances": len(all_gids),
            "target_global_ids": target_ids,
        },
    }
    with open(output_dir / "id_mapping.json", 'w') as f:
        json.dump(data, f, indent=2)
    with open(output_dir / "matching_config.json", 'w') as f:
        json.dump(MATCHING_CONFIG, f, indent=2)


# ─────────── 处理单个文件夹 ───────────

def process_single_folder(folder, yolo_model, skip_yolo=False, skip_voting=False, quiet=False, output_subdir=None):
    folder = Path(folder).resolve()
    name = folder.name
    if not quiet:
        print(f"\n{'='*60}")
        print(f"📁 {name}")
        print(f"{'='*60}")

    images_dir = find_images_dir(folder)
    if images_dir is None:
        print(f"  ❌ 未找到 images_real 或 images")
        return False
    model_dir = find_model_dir(folder)
    if model_dir is None:
        print(f"  ❌ 未找到 output_* 模型目录")
        return False

    # ── Step 1: YOLO ──
    mask_dir = folder / "masks_results"
    if skip_yolo:
        if not (mask_dir / "integer_masks").is_dir():
            print(f"  ❌ --skip_yolo 需要已有 masks_results/integer_masks")
            return False
        if not quiet:
            print(f"  ⏭️  跳过 YOLO")
    else:
        if not quiet:
            print(f"  🔄 YOLO 分割...")
        ok, total = step_yolo_segmentation(yolo_model, images_dir, mask_dir, quiet)
        if ok == 0:
            print(f"  ❌ YOLO 无结果")
            return False
        if not quiet:
            print(f"  ✅ YOLO: {ok}/{total}")

    int_mask_dir = mask_dir / "integer_masks"
    mask_files = sorted(int_mask_dir.glob("mask_*.png"))
    if not mask_files:
        print(f"  ❌ 无掩码文件")
        return False

    # ── 加载 YOLO 数据 ──
    class_info_map = {}
    for cp in int_mask_dir.glob("class_info_*.json"):
        fn = extract_frame_number(cp.name)
        if fn >= 0:
            with open(cp) as f:
                class_info_map[fn] = json.load(f)

    yolo_masks = {}
    for mf in tqdm(mask_files, desc="  加载掩码", disable=quiet):
        fn = extract_frame_number(mf.name)
        mask = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        yolo_masks[fn] = mask.astype(np.int32)
    if not quiet:
        print(f"  帧数: {len(yolo_masks)}")

    # ── Step 2: 渲染 GS-ID ──
    gaussians, train_cameras, gs_id_maps, bg_tensor = \
        step_load_and_render(folder, model_dir, quiet)

    # ── Step 3: 计算质心 ──
    all_instances = step_compute_centroids(
        gaussians, train_cameras, gs_id_maps, yolo_masks, class_info_map, quiet
    )
    if not all_instances:
        print(f"  ❌ 无有效实例")
        return False

    # ── Step 4: 匹配 ──
    c2_ct = CLASS2_START_ID
    c3_ct = CLASS3_START_ID
    ref_centroids = {}  # 冷启动：每个序列独立匹配，不跨序列加载历史
    mapping, c2_ct, c3_ct = step_matching(
        all_instances, ref_centroids, c2_ct, c3_ct, quiet
    )

    # ── Step 5: 生成 unified ──
    if output_subdir:
        out_dir = Path(output_subdir) if Path(output_subdir).is_absolute() else folder / output_subdir
    else:
        out_dir = folder / "semantic_matchin"
    unified_dir, id_colors = step_generate_unified_masks(
        yolo_masks, class_info_map, mapping, out_dir, quiet
    )

    # ── Step 6: GS-ID 投票 ──
    if not skip_voting:
        gaussians = step_gsid_voting(
            gaussians, train_cameras, gs_id_maps, unified_dir, yolo_masks, quiet
        )

        # ── Step 7: 导出 ──
        step_export_ply(gaussians, out_dir, quiet)
    else:
        gaussians._object = torch.zeros(gaussians._xyz.shape[0],
                                         dtype=torch.long, device="cuda")

    save_id_mapping(mapping, all_instances, yolo_masks, out_dir, quiet)

    if not quiet:
        print(f"  ✅ {name} 完成")
        print(f"  📂 {out_dir}")
    return True


# ───────────────────── 主入口 ─────────────────────

def main():
    global YOLO_CONF, SPATIAL_THRESHOLD, CLASS2_START_ID, CLASS3_START_ID
    global DBSCAN_EPS, DBSCAN_MIN_SAMPLES
    global MATCHING_CONFIG

    parser = ArgumentParser(
        description="一体化流水线: YOLO分割 → 匹配 → GS-ID投票 → 语义点云")
    parser.add_argument("folder", type=str, help="数据文件夹路径")
    parser.add_argument("--batch", action="store_true",
                        help="批量处理子文件夹")
    parser.add_argument("--yolo_model", type=str, default=None,
                        help="YOLO权重路径")
    parser.add_argument("--skip_yolo", action="store_true",
                        help="跳过YOLO，使用已有掩码")
    parser.add_argument("--skip_voting", action="store_true",
                        help="跳过GS-ID投票和语义点云导出（只做分割+匹配）")
    parser.add_argument("--output_subdir", type=str, default=None,
                        help="Output subdirectory (default: semantic_matchin)")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--conf", type=float, default=YOLO_CONF)
    parser.add_argument("--spatial_threshold", type=float, default=None,
                        help="覆盖跨帧匹配的空间距离阈值")
    parser.add_argument("--alpha_threshold", type=float, default=None,
                        help="显式覆盖GS渲染alpha有效性阈值")
    parser.add_argument("--class2_start", type=int, default=CLASS2_START_ID)
    parser.add_argument("--class3_start", type=int, default=CLASS3_START_ID)
    parser.add_argument("--dbscan_eps", type=float, default=DBSCAN_EPS)
    parser.add_argument("--dbscan_min_samples", type=int, default=DBSCAN_MIN_SAMPLES)

    args = parser.parse_args()
    YOLO_CONF = args.conf
    CLASS2_START_ID = args.class2_start
    CLASS3_START_ID = args.class3_start
    DBSCAN_EPS = args.dbscan_eps
    DBSCAN_MIN_SAMPLES = args.dbscan_min_samples

    if args.spatial_threshold is not None:
        MATCHING_CONFIG["spatial_threshold"] = args.spatial_threshold
    if args.alpha_threshold is not None:
        MATCHING_CONFIG["alpha_threshold"] = args.alpha_threshold
    SPATIAL_THRESHOLD = MATCHING_CONFIG["spatial_threshold"]
    if not args.quiet:
        print(f"Association config: {MATCHING_CONFIG}")

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"❌ 不存在: {folder}")
        sys.exit(1)

    # ── YOLO 模型；复用已有 masks 时无需加载 ──
    yolo_model = None
    if not args.skip_yolo:
        if YOLOModel is None:
            print("❌ 请安装 ultralytics: pip install ultralytics")
            sys.exit(1)

        yolo_path = args.yolo_model
        if yolo_path is None:
            yolo_path = DEFAULT_YOLO_PATH if os.path.exists(DEFAULT_YOLO_PATH) else None
        if yolo_path is None or not os.path.exists(yolo_path):
            print(f"❌ YOLO模型不存在，用 --yolo_model 指定")
            sys.exit(1)

        if not args.quiet:
            print(f"🚀 YOLO: {yolo_path}")
        yolo_model = YOLOModel(yolo_path)
        device = 0 if torch.cuda.is_available() else "cpu"
        yolo_model.to(device)

    skip_voting = args.skip_voting

    # ── 处理 ──
    if args.batch:
        subs = sorted([
            d for d in folder.iterdir()
            if d.is_dir() and find_images_dir(d) and find_model_dir(d)
        ])
        if not subs:
            print(f"❌ 无有效子文件夹")
            sys.exit(1)
        if not args.quiet:
            print(f"📂 批量: {len(subs)} 个")
        ok = 0
        for sf in subs:
            try:
                if process_single_folder(sf, yolo_model, args.skip_yolo, args.skip_voting,
                                         args.quiet, args.output_subdir):
                    ok += 1
            except Exception as e:
                print(f"❌ {sf.name}: {e}")
                import traceback
                traceback.print_exc()
        if not args.quiet:
            print(f"\n{'='*60}")
            print(f"🎉 {ok}/{len(subs)} 完成")
            print(f"{'='*60}")
    else:
        process_single_folder(folder, yolo_model, args.skip_yolo, args.skip_voting,
                              args.quiet, args.output_subdir)


if __name__ == "__main__":
    main()


# python pipeline_seg.py /path/to/scenes --batch
