#!/usr/bin/env python3
"""Check the single environment used by every post-pose workflow stage."""

from __future__ import annotations

import importlib.metadata as metadata
import sys


EXPECTED = {
    "torch": "2.0.0+cu118",
    "torchvision": "0.15.0+cu118",
    "numpy": "1.24.4",
    "scipy": "1.10.1",
    "scikit-learn": "1.3.2",
    "open3d": "0.18.0",
    "ultralytics": "8.4.36",
    "opencv-python": "4.12.0.88",
    "pandas": "2.0.3",
    "openpyxl": "3.1.5",
    "plyfile": "1.0.3",
}


def main() -> int:
    failures = []
    for distribution, expected in EXPECTED.items():
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            failures.append(f"{distribution}: not installed (expected {expected})")
            continue
        if actual != expected:
            failures.append(f"{distribution}: {actual} (expected {expected})")

    try:
        import cv2
        import numpy as np
        import open3d as o3d
        import sklearn
        import torch
        import torchvision
        import ultralytics
        import diff_surfel_rasterization
        import simple_knn._C

        if cv2.__version__ != "4.12.0":
            failures.append(f"cv2 runtime: {cv2.__version__} (expected 4.12.0)")
        if torch.version.cuda != "11.8":
            failures.append(f"torch CUDA runtime: {torch.version.cuda} (expected 11.8)")
        if not torch.cuda.is_available():
            failures.append("torch.cuda.is_available() is False")
        print(f"Python: {sys.version.split()[0]} ({sys.executable})")
        print(f"Torch/CUDA: {torch.__version__} / {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"cv2/Open3D/sklearn: {cv2.__version__} / {o3d.__version__} / {sklearn.__version__}")
        print(f"Ultralytics: {ultralytics.__version__}")
        print(f"Rasterizer: {diff_surfel_rasterization.__file__}")
        print(f"simple-knn: {simple_knn._C.__file__}")
    except Exception as exc:
        failures.append(f"workflow import check failed: {exc}")

    if failures:
        print("Environment check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Environment check passed: all post-pose workflow dependencies are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
