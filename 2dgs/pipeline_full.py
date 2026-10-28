#!/usr/bin/env python3
"""
pipeline_full.py - 全自动流水线: 训练 → YOLO分割 → 跨帧匹配 → 语义点云

流程:
  1. 训练2DGS模型（调用 train.py）
  2. 分割 + 匹配 + GS-ID投票 + 导出语义点云（调用 pipeline_seg.py）

用法:
  单文件夹:   python pipeline_full.py -s /path/to/dataset
  批量模式:   python pipeline_full.py -s /path/to/parent --batch
  仅推理:     python pipeline_full.py -s /path/to/dataset --skip_train

说明:
  - 模型训练输出到 <dataset_path>/output/，pipeline_seg.py 自动发现
  - 所有参数都有默认值，直接 -s 即可运行
  - 批量模式下自动检测每个子文件夹的模型，存在则跳过训练
  - --skip_train 表示"强制跳过训练（即使无模型）"
"""

import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="全自动流水线: 训练 → YOLO分割 → 跨帧匹配 → 语义点云",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python pipeline_full.py -s /path/to/B73-1_frames                     # 完整训练+分割
  python pipeline_full.py -s /path/to/B73-1_frames --skip_train        # 强制跳过训练
  python pipeline_full.py -s /path/to/parent --batch                    # 批量（自动跳过已训练）
  python pipeline_full.py -s /path/to/data --iterations 30000 --quiet  # 静默模式
  python pipeline_full.py -s /path/to/data --yolo_model /path/best.pt  # 指定YOLO
""")

    # ── 核心路径 ──
    parser.add_argument("-s", "--source_path", required=True,
                        help="数据文件夹路径（含images/和sparse/的COLMAP目录）")
    parser.add_argument("--batch", action="store_true",
                        help="批量处理source_path下的所有子文件夹（自动跳过已训练的）")

    # ── 训练参数 ──
    parser.add_argument("--iterations", type=int, default=30000,
                        help="训练迭代数（默认 30000）")
    parser.add_argument("--skip_train", action="store_true",
                        help="强制跳过训练（即使模型不存在也不训练）")
    parser.add_argument("--lambda_normal", type=float, default=0.05,
                        help="法向一致性损失权重")
    parser.add_argument("--lambda_dist", type=float, default=0.0,
                        help="畸变损失权重")

    # ── 分割/匹配参数 ──
    parser.add_argument("--yolo_model", type=str, default="",
                        help="YOLO模型路径（默认自动查找）")
    parser.add_argument("--skip_yolo", action="store_true",
                        help="跳过YOLO（掩码已存在时使用）")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="YOLO置信度阈值（默认 0.15）")
    parser.add_argument("--spatial_threshold", type=float, default=1.5,
                        help="空间匹配阈值（米，默认 1.5）")
    parser.add_argument("--class2_start", type=int, default=222,
                        help="class 2 起始全局ID")
    parser.add_argument("--class3_start", type=int, default=300,
                        help="class 3 起始全局ID")
    parser.add_argument("--dbscan_eps", type=float, default=0.1,
                        help="DBSCAN 邻域半径")
    parser.add_argument("--dbscan_min_samples", type=int, default=10,
                        help="DBSCAN 最小样本数")
    parser.add_argument("--skip_voting", action="store_true",
                        help="跳过GS-ID投票和语义点云导出")
    parser.add_argument("--output_subdir", type=str, default=None,
                         help="pipeline_seg output subdirectory (default: semantic_matchin)")
    parser.add_argument("--quiet", action="store_true",
                        help="减少输出")

    args = parser.parse_args()
    script_dir = Path(__file__).parent.resolve()
    src = Path(args.source_path).resolve()

    if args.batch:
        folders = sorted([d for d in src.iterdir() if d.is_dir()])
        if not folders:
            print(f"❌ {src} 下没有子文件夹")
            sys.exit(1)
        for f in folders:
            _run_single(f, args, script_dir)
    else:
        _run_single(src, args, script_dir)


def _check_ply(folder: Path, iterations: int):
    """返回已训练模型的 point_cloud.ply 路径，若不存在返回 None。"""
    candidate = folder / "output" / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    return candidate if candidate.exists() else None


def _looks_like_scene(folder: Path) -> bool:
    """判定目录是否是 2DGS 训练所需的 COLMAP/Blender 场景根目录。
    含 sparse/ 子目录或 transforms_train.json 视为有效场景。
    """
    return (folder / "sparse").exists() or (folder / "transforms_train.json").exists()


def _run_single(folder: Path, args, script_dir: Path):
    name = folder.name

    if not _looks_like_scene(folder):
        print(f"\n{'='*60}")
        print(f"⏭ 跳过 {name}：不是 COLMAP/Blender 场景目录（缺少 sparse/ 或 transforms_train.json）")
        print(f"{'='*60}")
        return

    # ─── Step 1: 训练 ───
    final_ply = _check_ply(folder, args.iterations)

    if final_ply is not None:
        need_train = False
        skip_reason = f"检测到已有模型 {final_ply.name}，自动跳过训练"
    elif args.skip_train:
        need_train = False
        skip_reason = "--skip_train 强制跳过（无模型）"
    else:
        need_train = True
        skip_reason = None

    if need_train:
        print(f"\n{'='*60}")
        print(f"Step 1: 训练 2DGS — {name}")
        print(f"{'='*60}")
        model_dir = folder / "output"
        model_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable, str(script_dir / "train.py"),
            "-s", str(folder),
            "-m", str(model_dir),
            "--iterations", str(args.iterations),
            "--lambda_normal", str(args.lambda_normal),
            "--lambda_dist", str(args.lambda_dist),
            "--test_iterations", str(args.iterations),
            "--save_iterations", str(args.iterations),
            "--quiet",
        ]
        subprocess.run(train_cmd, check=True)
    else:
        print(f"\n{'='*60}")
        print(f"Step 1: 跳过训练 — {name}（{skip_reason}）")
        print(f"{'='*60}")

    # ─── Step 2: 分割 + 匹配 + 投票 + 导出 ───
    print(f"\n{'='*60}")
    print(f"Step 2: 分割 + 匹配 + 语义点云 — {name}")
    print(f"{'='*60}")

    pipeline_cmd = [
        sys.executable, str(script_dir / "pipeline_seg.py"),
        str(folder),
        "--spatial_threshold", str(args.spatial_threshold),
        "--class2_start", str(args.class2_start),
        "--class3_start", str(args.class3_start),
        "--dbscan_eps", str(args.dbscan_eps),
        "--dbscan_min_samples", str(args.dbscan_min_samples),
        "--conf", str(args.conf),
    ]
    if args.yolo_model:
        pipeline_cmd += ["--yolo_model", args.yolo_model]
    if args.skip_yolo:
        pipeline_cmd.append("--skip_yolo")
    if args.skip_voting:
        pipeline_cmd.append("--skip_voting")
    if args.output_subdir:
        pipeline_cmd += ["--output_subdir", str(args.output_subdir)]
    if args.quiet:
        pipeline_cmd.append("--quiet")

    subprocess.run(pipeline_cmd, check=True)

    # ─── 完成 ───
    model_dir = folder / "output"
    out_subdir = args.output_subdir if args.output_subdir else "semantic_matchin"
    print(f"\n{'='*60}")
    print(f"✅ 完成: {name}")
    print(f"   模型: {model_dir}")
    if not args.skip_voting:
        ply_files = list((folder / out_subdir).glob("*point_cloud*.ply"))
        for pf in ply_files:
            print(f"   点云: {pf}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

# # 完整训练+分割（单个数据集）
# python pipeline_full.py -s /path/to/B73-1_frames

# # 批量处理（自动跳过已训练）
# python pipeline_full.py -s /path/to/parent --batch

# # 只跑分割匹配（模型已存在）
# python pipeline_full.py -s /path/to/B73-1_frames --skip_train

# # 指定YOLO权重
# python pipeline_full.py -s /path/to/data --yolo_model /path/best.pt

# # 自定义匹配参数
# python pipeline_full.py -s /path/to/data --skip_train \
#     --spatial_threshold 1.0 \
#     --output_subdir matching_variant
