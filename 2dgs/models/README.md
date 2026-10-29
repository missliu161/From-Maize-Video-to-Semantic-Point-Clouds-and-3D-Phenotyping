# Bundled YOLO model

The reproduction pipeline expects the trained YOLO segmentation checkpoint at
`yolo_seg.pt`. `pipeline_seg.py` automatically uses this file when no
`--yolo_model` argument and no `YOLO_MODEL_PATH` environment variable are
provided. The binary checkpoint is excluded from ordinary Git commits by
`.gitignore`; download it from the [download link in the root README](../../README.md#models)
and place it at this path before running the pipeline.

The checkpoint is a binary model artifact. Keep its version and SHA-256
documented when publishing or redistributing the bundle.

SHA-256 of the supplied checkpoint:

```text
efdf681c6539ca9378cb5d7fa6ef3f1a8b83a9f80bbe70c372e6a8db6cfdad7e  yolo_seg.pt
```
