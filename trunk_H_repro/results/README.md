# Trunk model files and per-scene trunk PLYs

`model_H_final.pkl` is the supplied trained H-v3 ensemble. The direct inference
entry point reconstructs its 21 features for the current scene from the full
and postprocessed (not ID-split) semantic PLYs, so `features_v3.pkl` and
`bases_v3.pkl` are not required for ordinary inference. `report_B.json`
contains the tuned B geometry configuration used by the feature builder and is
tracked in the code repository.

Existing scene-specific H trunk PLYs can also be passed directly to the
phenotype script with `--trunk` to skip model inference.

The pickle is excluded from ordinary Git commits because of its size. Obtain it
from the project's versioned release and verify its SHA-256 before inference:

```text
6dbde0654b8b38e7b183025a3149310943e35a60aab5cd4142d751ab50c72756  model_H_final.pkl
```

Direct per-scene H inference writes `point_cloud_trunk.ply` next to the
scene's full point cloud.
The model file is a Python pickle and must only be loaded from a trusted source.
Do not commit private/raw input data or large model files to the code-only
repository; publish approved artifacts as a versioned release/download with
checksums.
