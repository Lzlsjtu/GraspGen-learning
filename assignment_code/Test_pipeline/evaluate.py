#!/usr/bin/env python3
"""
[作业新增] graspnet_pipeline/evaluate.py
非 GraspGen 原始项目文件。

GraspNet-1Billion 评估管线：将 GraspGen 输出对接 graspnetAPI 的 GraspNetEval。

支持两种模式:
  1. --test-real-scenes: 使用已有的 18 个真实场景 JSON（无需下载测试集）
  2. --dataset_root <path>: 使用完整 GraspNet-1Billion 测试集

用法:
  # 用现有真实场景测试（无需下载）
  python evaluate.py --test-real-scenes

  # 用 GraspNet 测试集评估（需先下载数据集）
  python evaluate.py --dataset_root ../GraspNet_datasets --camera realsense

  # 对比我们自己训练的模型
  python evaluate.py --test-real-scenes --our_model
"""

import sys
sys.path.insert(0, "/home/lzl/Projects/6Dpose/GraspGen")

import os
import json
import argparse
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── GraspNet API ────────────────────────────────────────────────────────────
from graspnetAPI import GraspGroup, GraspNetEval

# ─── GraspGen 核心 ───────────────────────────────────────────────────────────
from grasp_gen.grasp_server import GraspGenSampler

# ─── 管线模块 ────────────────────────────────────────────────────────────────
from config import (
    OUTPUT_DIR, NUM_POINTS, SOR_K, SOR_THRESHOLD, KAPPA,
    NUM_GRASPS, COLLISION_THRESHOLD, NUM_COLLISION_SAMPLES,
)
from model_loader import load_pretrained_model, load_our_model
from preprocess import preprocess_point_cloud, pc_stats
from collision import check_grasp_collisions
from visualize import visualize_results


# ─── Gripper 参数 (franka_panda) ──────────────────────────────────────────────
GRIPPER_WIDTH  = 0.08    # 夹爪开口宽度 (m)
GRIPPER_HEIGHT = 0.02    # 夹爪手指高度 (m)
GRIPPER_DEPTH  = 0.105   # 夹爪手指深度 (m)


def matrix_to_graspgroup(matrices_4x4, scores=None, width=GRIPPER_WIDTH,
                         height=GRIPPER_HEIGHT, depth=GRIPPER_DEPTH,
                         obj_id=-1):
    """将 GraspGen 输出的 4x4 齐次矩阵转换为 graspnetAPI 的 GraspGroup 格式。

    GraspGroup 期望的 17 列格式:
      [score, width, height, depth, R11..R33, tx, ty, tz, obj_id]

    Args:
        matrices_4x4: (N, 4, 4) numpy array, grasp poses in world/camera frame
        scores:        (N,) numpy array, confidence scores
        width/height/depth: gripper physical dimensions
        obj_id:        object instance ID (-1 for unknown)

    Returns:
        GraspGroup instance
    """
    N = len(matrices_4x4)
    if scores is None:
        scores = np.ones(N)

    grasps_17 = np.zeros((N, 17), dtype=np.float32)

    for i in range(N):
        T = matrices_4x4[i]
        R = T[:3, :3]          # 3x3 rotation matrix
        t = T[:3, 3]           # translation vector

        grasps_17[i, 0]  = scores[i]          # score
        grasps_17[i, 1]  = width               # width
        grasps_17[i, 2]  = height              # height
        grasps_17[i, 3]  = depth               # depth
        grasps_17[i, 4:13] = R.flatten()       # rotation matrix (row-major, 9 values)
        grasps_17[i, 13:16] = t                # translation (3 values)
        grasps_17[i, 16] = obj_id              # object ID

    return GraspGroup(grasps_17)


def evaluate_on_real_scenes(model_info, scene_dir=None, output_dir=None):
    """在已有的 18 个真实场景 JSON 上评估。

    这些场景来自 GraspGenModels/sample_data/real_scene_pc/，包含:
      - scene_info: img_color, img_depth, full_pc, obj_mask
      - object_info: pc, pc_color
      - grasp_info: grasp_poses, grasp_conf (用作参考，不是 GT)

    评估指标: 置信度分布、碰撞通过率、推理速度

    Args:
        model_info: (sampler, model, cfg, gripper_info, use_pretrained) tuple
        scene_dir:  JSON 场景目录
        output_dir: 输出目录

    Returns:
        dict: 汇总统计
    """
    sampler, model, cfg, gripper_info, use_pretrained = model_info

    if scene_dir is None:
        scene_dir = Path("/home/lzl/Projects/6Dpose/GraspGen/GraspGenModels/sample_data/real_scene_pc")
    if output_dir is None:
        output_dir = Path(OUTPUT_DIR) / "eval_real_scenes"

    scene_dir = Path(scene_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(scene_dir.glob("*.json"))
    # 过滤掉没有 full_pc 的场景
    valid_files = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        if 'full_pc' in data.get('scene_info', {}):
            valid_files.append(jf)

    print(f"\n{'='*60}")
    print(f"  Evaluating on {len(valid_files)}/{len(json_files)} real scenes")
    print(f"  Model: {'Official Pretrained' if use_pretrained else 'Our Trained'}")
    print(f"{'='*60}\n")

    all_stats = []
    dump_dir = output_dir / "dump"

    for idx, json_path in enumerate(valid_files):
        scene_name = json_path.stem
        print(f"[{idx+1:2d}/{len(valid_files)}] {scene_name}")

        with open(json_path) as f:
            data = json.load(f)

        scene_pc = np.array(data['scene_info']['full_pc'])[0]   # (N, 3)
        obj_mask = np.array(data['scene_info']['obj_mask'])      # (H, W) bool
        object_pc = scene_pc[obj_mask.flatten()]

        # ── 预处理 ──
        object_pc_processed = preprocess_point_cloud(
            object_pc, num_points=NUM_POINTS,
            sor_k=SOR_K, sor_threshold=SOR_THRESHOLD
        )

        # ── 推理 ──
        t0 = time.time()
        if use_pretrained:
            grasps, confidence = GraspGenSampler.run_inference(
                object_pc=object_pc_processed,
                grasp_sampler=sampler,
                num_grasps=NUM_GRASPS,
                remove_outliers=False,
            )
        else:
            from inference import run_graspgen_inference
            grasps, confidence = run_graspgen_inference(
                model, object_pc_processed,
                num_grasps=NUM_GRASPS, kappa=KAPPA
            )
        elapsed = time.time() - t0

        # 还原中心化
        pc_center = object_pc.mean(axis=0)
        if hasattr(grasps, 'cpu'):
            grasps_np = grasps.cpu().numpy().copy()
        else:
            grasps_np = grasps.copy()
        grasps_np[:, :3, 3] += pc_center

        if hasattr(confidence, 'cpu'):
            conf_np = confidence.detach().cpu().numpy()
        else:
            conf_np = np.array(confidence)

        # ── 碰撞检测 ──
        scene_without_obj = scene_pc[~obj_mask.flatten()]
        if len(scene_without_obj) > 10000:
            idx_sample = np.random.choice(len(scene_without_obj), 10000, replace=False)
            scene_for_collision = scene_without_obj[idx_sample]
        else:
            scene_for_collision = scene_without_obj

        collision_mask = check_grasp_collisions(
            scene_pc=scene_for_collision,
            grasp_poses=grasps_np,
            gripper_collision_mesh=gripper_info.collision_mesh,
            threshold=COLLISION_THRESHOLD,
            num_samples=NUM_COLLISION_SAMPLES,
        )

        num_free = collision_mask.sum()
        conf_free = conf_np[collision_mask] if num_free > 0 else np.array([])
        conf_colliding = conf_np[~collision_mask] if num_free < len(conf_np) else np.array([])

        # ── 记录统计 ──
        stats = {
            "scene": scene_name,
            "obj_points": len(object_pc),
            "num_grasps": len(grasps_np),
            "time_sec": elapsed,
            "conf_mean": float(conf_np.mean()),
            "conf_max": float(conf_np.max()),
            "conf_min": float(conf_np.min()),
            "collision_free": int(num_free),
            "collision_rate": float(num_free / max(len(grasps_np), 1)),
            "conf_free_mean": float(conf_free.mean()) if len(conf_free) > 0 else 0.0,
            "conf_colliding_mean": float(conf_colliding.mean()) if len(conf_colliding) > 0 else 0.0,
        }
        all_stats.append(stats)

        print(f"  → {len(grasps_np)} grasps in {elapsed:.2f}s, "
              f"conf={conf_np.mean():.3f}, "
              f"collision-free={num_free}/{len(grasps_np)} ({100*num_free/max(len(grasps_np),1):.0f}%)")

        # ── 保存 GraspGroup 格式的 .npy ──
        gg = matrix_to_graspgroup(grasps_np, scores=conf_np)
        scene_dump_dir = dump_dir / scene_name / "realsense"
        scene_dump_dir.mkdir(parents=True, exist_ok=True)
        gg.save_npy(str(scene_dump_dir / "0000.npy"))

        # ── 可视化 (前 3 个场景) ──
        if idx < 3:
            visualize_results(
                object_pc=object_pc,
                scene_pc=scene_for_collision,
                grasp_poses=grasps_np,
                confidence=confidence,
                collision_mask=collision_mask,
                output_path=str(output_dir / f"eval_{scene_name}.png"),
                gripper_info=gripper_info,
            )

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  Evaluation Summary ({len(all_stats)} scenes)")
    print(f"{'='*60}")

    conf_means = [s["conf_mean"] for s in all_stats]
    collision_rates = [s["collision_rate"] for s in all_stats]
    times = [s["time_sec"] for s in all_stats]
    total_grasps = sum(s["num_grasps"] for s in all_stats)
    total_free = sum(s["collision_free"] for s in all_stats)

    print(f"  Total grasps:       {total_grasps}")
    print(f"  Total collision-free: {total_free}/{total_grasps} ({100*total_free/max(total_grasps,1):.1f}%)")
    print(f"  Avg confidence:     {np.mean(conf_means):.4f} ± {np.std(conf_means):.4f}")
    print(f"  Avg collision rate: {np.mean(collision_rates):.4f} ± {np.std(collision_rates):.4f}")
    print(f"  Avg inference time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
    print(f"  Per-scene range:    conf=[{min(conf_means):.3f}, {max(conf_means):.3f}], "
          f"collision=[{min(collision_rates):.1%}, {max(collision_rates):.1%}]")

    # 保存 JSON 汇总
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "model": "pretrained" if use_pretrained else "our_trained",
            "num_scenes": len(all_stats),
            "per_scene": all_stats,
            "aggregate": {
                "total_grasps": total_grasps,
                "total_collision_free": total_free,
                "overall_collision_rate": float(total_free / max(total_grasps, 1)),
                "mean_confidence": float(np.mean(conf_means)),
                "std_confidence": float(np.std(conf_means)),
                "mean_inference_time": float(np.mean(times)),
            }
        }, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")

    return all_stats


def evaluate_on_graspnet_dataset(model_info, dataset_root, camera="realsense",
                                 dump_dir=None, num_workers=4):
    """在完整 GraspNet-1Billion 测试集上评估 (需先下载数据集)。

    Args:
        model_info:     (sampler, model, cfg, gripper_info, use_pretrained)
        dataset_root:   GraspNet 数据集根目录 (包含 scenes/, models/, grasp_label/)
        camera:         'realsense' 或 'kinect'
        dump_dir:       .npy dump 目录
        num_workers:    并行评估 worker 数
    """
    sampler, model, cfg, gripper_info, use_pretrained = model_info

    if dump_dir is None:
        dump_dir = os.path.join(OUTPUT_DIR, "graspnet_eval_dump")

    print(f"\n{'='*60}")
    print(f"  Full GraspNet Evaluation")
    print(f"  Dataset: {dataset_root}")
    print(f"  Camera: {camera}")
    print(f"  Model: {'Official Pretrained' if use_pretrained else 'Our Trained'}")
    print(f"{'='*60}\n")

    # TODO: 遍历 GraspNet 测试集场景，对每帧运行推理，保存 .npy，然后调用 GraspNetEval
    print("[WARN] Full GraspNet evaluation requires the test dataset.")
    print("       Please download test_novel.zip, grasp_label.zip, models.zip")
    print("       and place them in the GraspNet_datasets/ directory.")
    print()
    print("       Manual download links:")
    print("       test_novel:    https://drive.google.com/file/d/1xixvgY0yK7TEALq3k7JcJk2_SP_6r8nk")
    print("       grasp_label:   https://drive.google.com/file/d/1FCV6j2J2eQpVk_ddJXljJvjRT1KU3sJ6")
    print("       models:        https://drive.google.com/file/d/1Gxwu2C5wRQ0QwjdA8CbMXx-bYf_wwPT5")
    print()

    return None


def main():
    parser = argparse.ArgumentParser(description="GraspNet-1Billion Evaluation Pipeline")
    parser.add_argument("--test-real-scenes", action="store_true",
                        help="Evaluate on existing 18 real scenes (no download needed)")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Path to GraspNet-1Billion dataset root")
    parser.add_argument("--camera", type=str, default="realsense",
                        help="Camera split: realsense or kinect")
    parser.add_argument("--our_model", action="store_true",
                        help="Use our trained model (default: official pretrained)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    # ── 加载模型 ──
    if args.our_model:
        print("Loading OUR trained model...")
        model, cfg, gripper_info = load_our_model()
        model_info = (None, model, cfg, gripper_info, False)
    else:
        print("Loading OFFICIAL pretrained model...")
        sampler, cfg, gripper_info = load_pretrained_model()
        model_info = (sampler, None, cfg, gripper_info, True)

    # ── 运行评估 ──
    if args.test_real_scenes or (not args.dataset_root):
        evaluate_on_real_scenes(model_info, output_dir=args.output_dir)
    elif args.dataset_root:
        evaluate_on_graspnet_dataset(
            model_info, args.dataset_root, camera=args.camera
        )


if __name__ == "__main__":
    main()
