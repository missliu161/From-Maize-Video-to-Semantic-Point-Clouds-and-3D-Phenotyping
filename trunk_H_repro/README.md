# Trunk Model Inference (H-v3)

This directory provides the pretrained H-v3 trunk-extraction model and the code required to compute features and run inference on new scenes. H-v3 identifies the model and method version used here. The release package does not contain training data or training caches and does not provide a model-retraining workflow.

The main workflow automatically calls `code/infer_H_model.py` from `2dgs/phenotype_from_semantic_b.py`. It requires:

- `results/model_H_final.pkl`: the pretrained H model.
- `results/report_B.json`: the B geometry configuration required for feature computation.
- The current scene's full denoised point cloud and post-processed semantic point cloud.

The inference script normally does not need to be run directly. To call it separately, run the following command from the package root:

```bash
python trunk_H_repro/code/infer_H_model.py \
  /path/to/point_cloud_sphere_denoised.ply \
  /path/to/point_cloud_object_post.ply \
  --output /path/to/point_cloud_trunk.ply
```

The model file is a Python pickle and should only be loaded after it has been obtained from an approved project release channel and its checksum has been verified. Model download and SHA-256 information are documented in the root [README](../README.md).
