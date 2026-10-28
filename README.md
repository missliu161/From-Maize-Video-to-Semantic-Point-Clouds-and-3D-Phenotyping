# 玉米视频到语义点云与三维表型

本仓库提供从视频或已有 COLMAP 位姿开始，运行 2D Gaussian Splatting（2DGS）、语义点云处理、主干提取（预训练 H-v3 模型）和三维表型计算的代码。

## 流程

```text
视频 → 抽帧 → VGGSfM 位姿估计 → 2DGS 与语义分割 → 点云后处理
     → 主干提取（H-v3 模型） → 表型结果（PLY、Excel）
```

如果已有 COLMAP 位姿和图像，可跳过抽帧与 VGGSfM，直接从 2DGS 开始。

## 环境

完整流程需要 Linux x86_64、Conda、NVIDIA GPU/驱动、CUDA Toolkit 11.8（含 `nvcc`）和 GCC/G++ 11 或更低版本。`setup_environment.sh` 会新建默认名为 `gsid-seg` 的 Conda 环境，安装 2DGS 和后处理依赖、编译 CUDA 扩展并检查环境；不会更新或删除已有环境。

在仓库根目录执行：

```bash
bash setup_environment.sh
conda activate gsid-seg
python check_environment.py
```

若 `gsid-seg` 已存在，可指定新的环境名：

```bash
GSID_ENV_NAME=gsid-seg-new bash setup_environment.sh
conda activate gsid-seg-new
```

脚本安装 Python 3.8.18 和固定依赖，详见 [`2dgs/environment.yml`](2dgs/environment.yml) 与 [`2dgs/requirements.txt`](2dgs/requirements.txt)。VGGSfM 使用独立环境，不由本安装脚本安装。

## 准备模型

运行前需取得与本项目类别映射匹配的模型文件，并放到以下路径：

| 文件 | 用途 |
|---|---|
| `2dgs/models/yolo_seg.pt` | YOLO 实例分割 |
| `trunk_H_repro/results/model_H_final.pkl` | 预训练主干提取模型（H-v3） |

模型二进制不包含在代码仓库中，本仓库目前也未提供下载链接；请向项目维护者获取。放置方式和模型说明见 [`2dgs/models/README.md`](2dgs/models/README.md) 与 [`trunk_H_repro/results/README.md`](trunk_H_repro/results/README.md)。下载后可用以下 SHA-256 校验：

```text
yolo_seg.pt        efdf681c6539ca9378cb5d7fa6ef3f1a8b83a9f80bbe70c372e6a8db6cfdad7e
model_H_final.pkl  6dbde0654b8b38e7b183025a3149310943e35a60aab5cd4142d751ab50c72756
```

在仓库根目录执行校验：

```bash
sha256sum 2dgs/models/yolo_seg.pt trunk_H_repro/results/model_H_final.pkl
```

`trunk_H_repro/results/report_B.json` 是 H 特征配置，已包含在仓库中，需与对应 H 模型匹配。
这里的 H/H-v3 是所用模型与方法版本的标识；流程输出的主干点云统一命名为 `point_cloud_trunk.ply`。

## 从视频运行完整流程

以下命令从仓库根目录执行，并使用新的场景目录。输入视频可放在任意位置，无需复制到仓库；场景结果建议放入已被 `.gitignore` 忽略的 `data/` 目录。完整流程还需要从 [VGGSfM 官方仓库](https://github.com/facebookresearch/vggsfm) 下载源码，并按其官方说明安装独立环境；运行时提供 VGGSfM 源码目录和该环境的 Python 可执行文件。

```bash
conda activate gsid-seg
python run_end_to_end.py \
  /path/to/input_video.mp4 \
  data/scenes/my_scene \
  --vggsfm-dir /path/to/vggsfm \
  --vggsfm-python /path/to/vggsfm-env/bin/python \
  --gs-python "$CONDA_PREFIX/bin/python"
```

默认按每秒 2 帧从视频抽帧，VGGSfM 使用实际抽取的图像数量；训练 30,000 步。场景目录可放在仓库外。可用 `--sampling-fps` 调整抽帧频率。若已有对应训练结果，可添加 `--skip-train`；需要重算去噪点云时添加 `--overwrite`。其他参数可通过 `python run_end_to_end.py --help` 查看。

## 从已有位姿运行

场景目录需包含 `images/` 和 COLMAP 格式的 `sparse/0/`（`cameras`、`images`、`points3D` 的 `.bin` 或 `.txt` 文件）。

```bash
conda activate gsid-seg
python run_from_poses.py /path/to/scene
```

若该场景已有 `output/point_cloud/iteration_30000/point_cloud.ply`，可添加 `--skip-train`。此入口不需要 VGGSfM，也不负责视频抽帧。

## 主要输出

结果写入场景目录：

```text
semantic_matchin/point_cloud_object_post.ply
semantic_matchin/point_cloud_object_post_split.ply
point_cloud_sphere_denoised.ply
point_cloud_trunk.ply
phenotype_results/<scene>_B.ply
phenotype_results/<scene>_phenotype.xlsx
```

球裁剪点云只在内存中传给 DBSCAN，不会生成中间 PLY；场景根目录仅保存去噪后的 `point_cloud_sphere_denoised.ply`。表型长度/面积标定使用语义点云中的 `object=221` 作为参照物，运行前请确认类别 ID 与模型映射相符。

## 数据与许可

原始视频、输入图像、模型权重和生成结果不应提交到代码仓库。`.gitignore` 已忽略常见数据、模型和输出文件。请确认本项目及所用 2DGS、CUDA 扩展、VGGSfM 和模型权重各自的许可与引用要求；相关上游许可见各自仓库及 [`2dgs/LICENSE.md`](2dgs/LICENSE.md)。
