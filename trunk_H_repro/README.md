# 主干提取模型推理（H-v3）

此目录提供完整流程所需的预训练 H-v3 主干提取模型，以及对新场景执行特征计算和推理的代码。这里的 H-v3 是当前模型/方法版本标识。发布包不包含训练数据和训练缓存，也不提供模型重训流程。

主流程由 `2dgs/phenotype_from_semantic_b.py` 自动调用 `code/infer_H_model.py`。需要：

- `results/model_H_final.pkl`：预训练 H 模型。
- `results/report_B.json`：特征计算所需的 B 几何配置。
- 当前场景的完整去噪点云和后处理语义点云。

通常无需直接运行推理脚本；如需单独调用，可在发布包根目录执行：

```bash
python trunk_H_repro/code/infer_H_model.py \
  /path/to/point_cloud_sphere_denoised.ply \
  /path/to/point_cloud_object_post.ply \
  --output /path/to/point_cloud_trunk.ply
```

模型文件是 pickle 格式，只应加载从项目发布渠道取得且校验通过的文件。模型下载与 SHA-256 信息将在发布模型资产时补充。
