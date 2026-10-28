"""Shared point-cloud constants and readers for H-model inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData


ALPHA_FACTORS = {"id220": 2.0, "reference": 3.0, "fruit": 2.5, "leaf": 3.0}


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read xyz coordinates and normals from a PLY vertex element."""
    data = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([data[name] for name in ("x", "y", "z")]).astype(np.float64)
    names = data.dtype.names or ()
    normals = (
        np.column_stack([data[name] for name in ("nx", "ny", "nz")]).astype(np.float32)
        if all(name in names for name in ("nx", "ny", "nz"))
        else np.zeros((len(xyz), 3), dtype=np.float32)
    )
    return xyz, normals
