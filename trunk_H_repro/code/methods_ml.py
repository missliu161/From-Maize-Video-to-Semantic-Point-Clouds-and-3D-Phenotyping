"""H-v3 geometric feature construction used by the published classifier."""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = [
    "nx", "nx2", "d2line", "rx_d", "rz_d", "radial_frac",
    "lid_log", "fid_log", "rid_log", "ref_log",
    "knn8_d", "knn32_d", "verticality", "rcent_d",
]
FEATURE_NAMES2 = FEATURE_NAMES + ["sem_margin", "dmin_d", "out_of_radius"]


def _caplog(dist, d, cap=6.0):
    x = np.minimum(dist, 1.0) / d
    return np.clip(np.log1p(x), 0.0, cap).astype(np.float32)


def features(rec, geo):
    d = max(geo["d"], 1e-3)
    coords = rec["coords"]
    nx = geo["nx"]
    d2line = geo["d2line"]
    rx = np.maximum(geo["rx_d"], 0.05)
    lid = _caplog(rec["dist"]["leaf"], d)
    fid = _caplog(rec["dist"]["fruit"], d)
    rid = _caplog(rec["dist"]["id220"], d)
    ref = _caplog(rec["dist"]["reference"], d)
    cx = coords[:, 0] - np.median(coords[:, 0])
    cz = coords[:, 2] - np.median(coords[:, 2])
    rcent = np.hypot(cx, cz) / d
    return np.column_stack([
        nx, nx * nx, d2line, rx, np.maximum(geo["rz_d"], 0.05), d2line / rx,
        lid, fid, rid, ref,
        rec["knn8"] / d, rec["knn32"] / d, rec["verticality"], rcent,
    ]).astype(np.float32)


def features_v2(rec, geo):
    """Return the 17 features used by the published H-v3 model."""
    d = max(geo["d"], 1e-3)
    x = features(rec, geo)
    stack = np.column_stack([rec["dist"][group]
                             for group in ("id220", "reference", "fruit", "leaf")])
    stack = np.minimum(stack, 1.0)
    sorted_distances = np.sort(stack, axis=1)
    margin = np.clip((sorted_distances[:, 1] - sorted_distances[:, 0]) / d,
                     0, 6).astype(np.float32)
    dmin = np.clip(sorted_distances[:, 0] / d, 0, 6).astype(np.float32)
    out = np.clip(geo["d2line"] - np.maximum(geo["rx_d"], geo["rz_d"]),
                  0, 6).astype(np.float32)
    return np.column_stack([x, margin, dmin, out]).astype(np.float32)
