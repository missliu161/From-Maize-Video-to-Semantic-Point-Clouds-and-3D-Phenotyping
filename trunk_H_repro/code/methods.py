#!/usr/bin/env python3
"""Geometric trunk-extraction methods (all-new, vectorised numpy).

    scale                  : robust per-plant stem diameter from the candidate
                             cloud (used to normalise every length below)
    B  global_centerline   : semantic Alpha subtraction + a globally optimal
                             (dynamic-programming) curved trunk axis + per-height
                             variable elliptical cross-section, all length
                             thresholds expressed in stem-diameter units.

The B centreline geometry is shared with the published H-v3 inference model.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter

BASE_RADII = {"id220": 0.0505, "reference": 0.096, "fruit": 0.01, "leaf": 0.02}
GROUPS = ("id220", "reference", "fruit", "leaf")

def base_mask(rec: dict, scale: float) -> np.ndarray:
    keep = np.ones(len(rec["coords"]), dtype=bool)
    for g in GROUPS:
        keep &= rec["dist"][g] > BASE_RADII[g] * scale
    return keep


# --------------------------------------------------------------------------- #
# robust per-plant scale + density cores                                      #
# --------------------------------------------------------------------------- #

def height_bins(coords_y, ylo, yhi, bins):
    return np.clip(((coords_y - ylo) / max(yhi - ylo, 1e-9) * bins).astype(np.int64), 0, bins - 1)


def _density_core(sel, grid, core_frac, min_bin, min_core, smooth):
    """Boolean mask over sel: True where a point lies in the smoothed dense core.

    Returns None if the bin has too few points / too few dense cells.
    """
    if len(sel) < min_bin:
        return None
    gx = max(1, int(np.ceil((sel[:, 0].max() - sel[:, 0].min()) / grid)) + 1)
    gz = max(1, int(np.ceil((sel[:, 2].max() - sel[:, 2].min()) / grid)) + 1)
    if gx * gz > 60000:
        return None
    ox, oz = sel[:, 0].min(), sel[:, 2].min()
    cx = np.clip(((sel[:, 0] - ox) / grid).astype(np.int64), 0, gx - 1)
    cz = np.clip(((sel[:, 2] - oz) / grid).astype(np.int64), 0, gz - 1)
    hist = np.bincount(cz * gx + cx, minlength=gx * gz).reshape(gz, gx).astype(np.float64)
    w = gaussian_filter(hist, sigma=smooth)
    m = w.max()
    if m <= 0:
        return None
    core_grid = w >= core_frac * m
    if int(core_grid.sum()) < min_core:
        return None
    return core_grid[cz, cx]


def bin_cores(sel, grid, core_frac=0.30, min_bin=25, min_core=15, smooth=1.2):
    """Return (center[2], extent[2]) of the dense trunk core in one height bin."""
    core = _density_core(sel, grid, core_frac, min_bin, min_core, smooth)
    if core is None:
        return None
    cx = sel[core, 0]
    cz = sel[core, 2]
    c = np.array([cx.mean(), cz.mean()])
    e = np.array([max((cx - c[0]).max(), (c[0] - cx).max()),
                  max((cz - c[1]).max(), (c[1] - cz).max())])
    return c, e


def estimate_stem_diameter(sel, ybin, bins, grid=0.02):
    """Median 2*half-extent over reliable bins (robust stem diameter estimate)."""
    exts = []
    for k in range(bins):
        s = sel[ybin == k]
        bc = bin_cores(s, grid)
        if bc is not None:
            exts.append(0.5 * (bc[1][0] + bc[1][1]))
    if not exts:
        return 0.05
    return float(2.0 * np.median(exts))


# --------------------------------------------------------------------------- #
# B  global DP centreline + variable elliptical cross-section                  #
# --------------------------------------------------------------------------- #

B_DEFAULT = {
    "radii_scale": 0.7, "bins": 24, "grid": 0.02, "smooth": 1.2, "core_frac": 0.30,
    "top_k": 24, "w_leaf": 0.6, "lambda_shift": 1.1, "dens_floor": 0.15,
    "rad_q": 0.86, "rad_scale": 1.15, "margin_d": 0.35, "rmin_d": 0.5, "rmax_d": 4.0,
    "rad_smooth": 1.4, "w_branch": 1.0,
}


def global_centerline(rec, p=B_DEFAULT):
    """Return (keep, centers, rxz, d) for one scene under B."""
    coords = rec["coords"]
    cand = base_mask(rec, p["radii_scale"])
    keep = cand.copy()
    if not cand.any():
        return keep, None, None, 0.05
    sel = coords[cand]
    ylo, yhi = float(coords[:, 1].min()), float(coords[:, 1].max())
    bins = p["bins"]
    ybin = height_bins(sel[:, 1], ylo, yhi, bins)

    # stem scale: coarse grid first
    d = estimate_stem_diameter(sel, ybin, bins, p["grid"])
    d = max(d, 0.02)
    grid = max(0.008, min(0.05, p["grid"]))

    # candidate nodes per bin: top-K dense cells + a leaf-distance bonus
    leafd = rec["dist"]["leaf"][cand]
    nodes = []  # list per bin of (x, z, score, rx, rz)
    bin_has = np.zeros(bins, dtype=bool)
    for k in range(bins):
        idx = np.nonzero(ybin == k)[0]
        if len(idx) < 20:
            continue
        s = sel[idx]
        gx = max(1, int(np.ceil((s[:, 0].max() - s[:, 0].min()) / grid)) + 1)
        gz = max(1, int(np.ceil((s[:, 2].max() - s[:, 2].min()) / grid)) + 1)
        if gx * gz > 60000:
            continue
        ox, oz = s[:, 0].min(), s[:, 2].min()
        cx = np.clip(((s[:, 0] - ox) / grid).astype(np.int64), 0, gx - 1)
        cz = np.clip(((s[:, 2] - oz) / grid).astype(np.int64), 0, gz - 1)
        hist = np.bincount(cz * gx + cx, minlength=gx * gz).reshape(gz, gx).astype(np.float64)
        if hist.max() < 8:
            continue
        w = gaussian_filter(hist, sigma=p["smooth"])
        mx = w.max()
        if mx <= 0:
            continue
        cy, cxz = np.nonzero(w >= max(p["dens_floor"] * mx, 1e-6))
        # world coords of selected cell centroids
        wx = ox + (cxz + 0.5) * grid
        wz = oz + (cy + 0.5) * grid
        dens = w[cy, cxz] / mx
        # leaf bonus: mean leaf distance of candidate pts near each node
        ld = leafd[idx]
        leaf_norm = np.clip(ld / max(2 * d, 1e-3), 0, 1)
        order = np.argsort(-dens)[:p["top_k"]]
        for j in order:
            dist2 = (s[:, 0] - wx[j]) ** 2 + (s[:, 2] - wz[j]) ** 2
            near = dist2 <= (max(grid * 1.5, d) ** 2)
            if near.sum() < 3:
                continue
            lm = leaf_norm[near].mean()
            rx = np.percentile(s[near, 0], 90) - np.percentile(s[near, 0], 10)
            rz = np.percentile(s[near, 2], 90) - np.percentile(s[near, 2], 10)
            score = dens[j] + p["w_leaf"] * lm
            nodes.append((k, wx[j], wz[j], score, max(rx, 0.01), max(rz, 0.01)))
            bin_has[k] = True
    rel_bins = np.nonzero(bin_has)[0]
    if len(rel_bins) < 2:
        return keep, None, None, d

    # DP over reliable bins
    by_bin = {k: [n for n in nodes if n[0] == k] for k in rel_bins}
    dp = {}
    bp = {}
    for bi, k in enumerate(rel_bins):
        cur = by_bin[k]
        if bi == 0:
            for i, n in enumerate(cur):
                dp[(bi, i)] = n[3]
                bp[(bi, i)] = None
            continue
        kprev = rel_bins[bi - 1]
        gap = max(k - kprev, 1)
        prev = by_bin[kprev]
        for i, n in enumerate(cur):
            best, arg = -np.inf, None
            for j, pn in enumerate(prev):
                horiz = np.hypot(n[1] - pn[1], n[2] - pn[2])
                radc = abs(n[4] - pn[4]) + abs(n[5] - pn[5])
                cost = (p["lambda_shift"] * horiz / max(2 * d, 1e-3)
                        + p["w_branch"] * radc / max(2 * d, 1e-3)) / gap
                val = dp[(bi - 1, j)] - cost
                if val > best:
                    best, arg = val, j
            dp[(bi, i)] = best + n[3]
            bp[(bi, i)] = arg
    last = len(rel_bins) - 1
    end_i = int(np.argmax([dp[(last, i)] for i in range(len(by_bin[rel_bins[last]]))]))
    path = []
    bi, i = last, end_i
    while bi >= 0:
        path.append(by_bin[rel_bins[bi]][i])
        i = bp[(bi, i)]
        bi -= 1
    path = path[::-1]

    # interpolate centres & radii to all bins
    pb = np.array([n[0] for n in path], dtype=float)
    centers = np.full((bins, 2), np.nan)
    rxz = np.full((bins, 2), np.nan)
    for ax in range(2):
        centers[:, ax] = np.interp(np.arange(bins), pb, [n[1 + ax] for n in path])
    for ax in range(2):
        rxz[:, ax] = np.interp(np.arange(bins), pb, [n[4 + ax] for n in path])
    centers = gaussian_filter1d(centers, p["rad_smooth"], axis=0, mode="nearest")
    for ax in range(2):
        rxz[:, ax] = gaussian_filter1d(rxz[:, ax], p["rad_smooth"], mode="nearest")
        rxz[:, ax] = np.clip(rxz[:, ax], p["rmin_d"] * d, p["rmax_d"] * d)

    # variable ellipse corridor decision
    b_all = height_bins(coords[:, 1], ylo, yhi, bins)
    cx = centers[b_all, 0]
    cz = centers[b_all, 1]
    rx = rxz[b_all, 0] * p["rad_scale"] + p["margin_d"] * d
    rz = rxz[b_all, 1] * p["rad_scale"] + p["margin_d"] * d
    inside = (np.abs(coords[:, 0] - cx) <= rx) & (np.abs(coords[:, 2] - cz) <= rz)
    keep = cand & inside
    return keep, centers, rxz, d


def centerline_geometry(rec, p=B_DEFAULT):
    """Compute the B centreline once and return reusable per-point geometry.

    Returns dict with: cand, centers, rxz, d, ylo, yhi, bins, and per-point
    arrays nx (normalised height), d2line (xz dist to axis / d), rx_d, rz_d.
    """
    coords = rec["coords"]
    keep, centers, rxz, d = global_centerline(rec, p)
    out = {"keep_B": keep, "cand": base_mask(rec, p["radii_scale"]),
           "centers": centers, "rxz": rxz, "d": d}
    ylo, yhi = float(coords[:, 1].min()), float(coords[:, 1].max())
    bins = p["bins"]
    if centers is None:
        n = len(coords)
        out.update(nx=np.zeros(n, np.float32), d2line=np.ones(n, np.float32) * 9,
                   rx_d=np.full(n, 1.0, np.float32), rz_d=np.full(n, 1.0, np.float32))
        return out
    b = height_bins(coords[:, 1], ylo, yhi, bins)
    cx, cz = centers[b, 0], centers[b, 1]
    rx, rz = rxz[b, 0], rxz[b, 1]
    out["nx"] = ((coords[:, 1] - ylo) / max(yhi - ylo, 1e-9)).astype(np.float32)
    out["d2line"] = (np.hypot(coords[:, 0] - cx, coords[:, 2] - cz) / max(d, 1e-3)).astype(np.float32)
    out["rx_d"] = (rx / max(d, 1e-3)).astype(np.float32)
    out["rz_d"] = (rz / max(d, 1e-3)).astype(np.float32)
    return out
