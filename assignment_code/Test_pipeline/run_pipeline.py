#!/usr/bin/env python3
"""
[作业新增文件] graspnet_pipeline/run_pipeline.py
非 GraspGen 原始项目文件。

GraspNet-1Billion RGB-D → 点云 → GraspGen 推理 → 碰撞检测 → Open3D 可视化
的完整管线入口。

用法:
  # 对 GraspNet 场景进行推理:
  python run_pipeline.py --scene_dir <path_to_graspnet_scene>

  # 使用合成测试数据测试管线:
  python run_pipeline.py --test

  # 对单个 RGB-D 文件推理:
  python run_pipeline.py --rgb <rgb.png> --depth <depth.png> --intrinsics <fx,fy,cx,cy>
"""

import sys
sys.path.insert(0, "/home/lzl/Projects/6Dpose/GraspGen")

import argparse
import os
import numpy as np
from pathlib import Path

# ─── 复用 GraspGen 核心工具 ──────────────────────────────────────────────────
from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
from grasp_gen.grasp_server import GraspGenSampler

# ─── 管线模块 ────────────────────────────────────────────────────────────────
from config import (
    GRASPGEN_DIR, CHECKPOINT_DIR, OUTPUT_DIR, PLOTS_DIR,
    NUM_POINTS, SOR_K, SOR_THRESHOLD, KAPPA,
    NUM_GRASPS, COLLISION_THRESHOLD, NUM_COLLISION_SAMPLES,
    REALSENSE_INTRINSICS, PRETRAINED_CONFIG, PRETRAINED_GEN, PRETRAINED_DIS,
)
from model_loader import load_pretrained_model, load_our_model
from preprocess import (
    depth_image_to_point_cloud,
    preprocess_point_cloud,
    pc_stats,
)
from inference import run_graspgen_inference
from collision import check_grasp_collisions
from visualize import visualize_results


def pipeline_from_rgbd_pretrained(
    sampler,
    cfg,
    gripper_info,
    rgb: np.ndarray,
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    mask: np.ndarray = None,
    output_prefix: str = "result",
):
    """完整管线 (官方预训练): RGB-D → 推理 → 碰撞检测 → 可视化。

    Args:
        sampler:        GraspGenSampler 实例 (已加载权重)
        cfg:            OmegaConf 训练配置
        gripper_info:   GripperInfo
        rgb:            (H, W, 3) RGB 图像
        depth:          (H, W) 深度图 (米)
        fx, fy, cx, cy: 相机内参
        mask:           实例分割掩膜 (可选, True=目标物体)
        output_prefix:  输出文件名前缀
    """
    print("=" * 60)
    print("  GraspGen RGB-D Pipeline (Official Pretrained)")
    print("=" * 60)

    # ── Step 1: RGB-D → 点云 ──
    print("\n[Step 1] RGB-D → Point Cloud")
    scene_pc = depth_image_to_point_cloud(depth, fx, fy, cx, cy, rgb=rgb)
    pc_stats(scene_pc, "scene_raw")

    # 物体区域提取
    if mask is not None:
        object_pc = scene_pc[mask.flatten()[:len(scene_pc)]]
    else:
        # 无 mask: 随机降采样以避免内存溢出
        if len(scene_pc) > 8192:
            print(f"  Downsampling scene: {len(scene_pc)} → 8192 (random)")
            idx = np.random.choice(len(scene_pc), 8192, replace=False)
            object_pc = scene_pc[idx]
        else:
            object_pc = scene_pc
    pc_stats(object_pc, "object_raw")

    # ── Step 2: 点云预处理 ──
    print("\n[Step 2] Preprocessing (SOR + FPS)")
    object_pc_centered = preprocess_point_cloud(
        object_pc,
        num_points=NUM_POINTS,
        sor_k=SOR_K,
        sor_threshold=SOR_THRESHOLD,
    )
    pc_stats(object_pc_centered, "object_processed")

    # ── Step 3: GraspGenSampler 推理 ──
    print(f"\n[Step 3] Inference ({NUM_GRASPS} grasps, pretrained PTV3)")
    grasps, confidence = GraspGenSampler.run_inference(
        object_pc=object_pc_centered,
        grasp_sampler=sampler,
        num_grasps=NUM_GRASPS,
        remove_outliers=False,
    )

    if len(grasps) == 0:
        print("[ERROR] No grasps generated!")
        return None

    # 还原中心化
    pc_center = object_pc.mean(axis=0)
    grasps_np = grasps.cpu().numpy().copy()
    grasps_np[:, :3, 3] += pc_center
    conf_np = confidence.cpu().numpy()

    print(f"  Generated {len(grasps_np)} grasps")
    print(f"  Confidence: min={conf_np.min():.4f}, max={conf_np.max():.4f}, "
          f"mean={conf_np.mean():.4f}")

    # ── Step 4: 碰撞检测 ──
    print(f"\n[Step 4] Collision Detection (threshold={COLLISION_THRESHOLD*1000:.0f}mm)")
    collision_mask = check_grasp_collisions(
        scene_pc=object_pc,
        grasp_poses=grasps_np,
        gripper_collision_mesh=gripper_info.collision_mesh,
        threshold=COLLISION_THRESHOLD,
        num_samples=NUM_COLLISION_SAMPLES,
    )
    num_free = collision_mask.sum()
    print(f"  Collision-free: {num_free}/{len(grasps_np)} ({100*num_free/max(len(grasps_np),1):.1f}%)")

    # ── Step 5: Open3D 可视化 ──
    print(f"\n[Step 5] Open3D Visualization")
    visualize_results(
        object_pc=object_pc,
        scene_pc=None,
        grasp_poses=grasps_np,
        confidence=confidence,
        collision_mask=collision_mask,
        output_path=os.path.join(OUTPUT_DIR, output_prefix + ".png"),
        gripper_info=gripper_info,
    )

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)

    return {
        "scene_pc": scene_pc,
        "object_pc": object_pc,
        "object_pc_processed": object_pc_centered,
        "grasps": grasps_np,
        "confidence": conf_np,
        "collision_mask": collision_mask,
        "num_collision_free": int(num_free),
        "total_grasps": len(grasps_np),
    }


def pipeline_from_rgbd(
    model,
    cfg,
    gripper_info,
    rgb: np.ndarray,
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    mask: np.ndarray = None,
    output_prefix: str = "result",
):
    """完整管线：RGB-D → 推理 → 碰撞检测 → 可视化。

    Args:
        model:          GraspGenGenerator (已加载权重)
        cfg:            OmegaConf 训练配置
        gripper_info:   GripperInfo
        rgb:            (H, W, 3) RGB 图像
        depth:          (H, W) 深度图 (米)
        fx, fy, cx, cy: 相机内参
        mask:           实例分割掩膜 (可选, True=目标物体)
        output_prefix:  输出文件名前缀

    Returns:
        dict: 包含所有中间结果和指标的字典
    """
    print("=" * 60)
    print("  GraspGen Inference Pipeline")
    print("=" * 60)

    # ── Step 1: RGB-D → 点云 ──
    print("\n[Step 1] RGB-D → Point Cloud")
    scene_pc = depth_image_to_point_cloud(depth, fx, fy, cx, cy, rgb=rgb)
    pc_stats(scene_pc, "scene_raw")

    # 物体区域提取 (如果有 mask)
    if mask is not None:
        mask_flat = mask.flatten()
        # depth2points 输出的点数可能与 mask 不同（因为 depth>0 过滤）
        # 取交集
        if len(mask_flat) == depth.size:
            valid_mask = depth.flatten() > 0
            object_mask = mask_flat & valid_mask
            object_pc = scene_pc[object_mask[depth.flatten() > 0]]
        else:
            object_pc = scene_pc
    else:
        # 无 mask: 随机降采样以避免 SOR 内存溢出
        if len(scene_pc) > 8192:
            print(f"  Downsampling scene: {len(scene_pc)} → 8192 (random)")
            idx = np.random.choice(len(scene_pc), 8192, replace=False)
            object_pc = scene_pc[idx]
        else:
            object_pc = scene_pc

    pc_stats(object_pc, "object_raw")

    # ── Step 2: 点云预处理 ──
    print("\n[Step 2] Preprocessing (SOR + FPS)")
    object_pc_processed = preprocess_point_cloud(
        object_pc,
        num_points=NUM_POINTS,
        sor_k=SOR_K,
        sor_threshold=SOR_THRESHOLD,
    )
    pc_stats(object_pc_processed, "object_processed")

    # ── Step 3: GraspGen 推理 ──
    print(f"\n[Step 3] GraspGen Inference ({NUM_GRASPS} grasps, {cfg.diffusion.num_diffusion_iters} diffusion steps)")
    grasps, confidence = run_graspgen_inference(
        model,
        object_pc_processed,
        num_grasps=NUM_GRASPS,
        kappa=KAPPA,
    )

    if len(grasps) == 0:
        print("[ERROR] No grasps generated!")
        return None

    # 恢复中心化 (推理时做了中心化)
    pc_center = object_pc.mean(axis=0)
    grasps_np = grasps.numpy().copy()
    grasps_np[:, :3, 3] += pc_center  # 平移还原

    print(f"  Generated {len(grasps_np)} grasps")
    print(f"  Confidence: min={confidence.min():.4f}, max={confidence.max():.4f}, "
          f"mean={confidence.mean():.4f}")

    # ── Step 4: 碰撞检测 ──
    print(f"\n[Step 4] Collision Detection (threshold={COLLISION_THRESHOLD*1000:.0f}mm)")
    collision_mask = check_grasp_collisions(
        scene_pc=scene_pc if mask is not None else object_pc,
        grasp_poses=grasps_np,
        gripper_collision_mesh=gripper_info.collision_mesh,
        threshold=COLLISION_THRESHOLD,
        num_samples=NUM_COLLISION_SAMPLES,
    )
    num_free = collision_mask.sum()
    print(f"  Collision-free: {num_free}/{len(grasps_np)} ({100*num_free/max(len(grasps_np),1):.1f}%)")

    # ── Step 5: Open3D 可视化 ──
    print(f"\n[Step 5] Open3D Visualization")
    output_png = os.path.join(OUTPUT_DIR, f"{output_prefix}.png")
    visualize_results(
        object_pc=object_pc,
        scene_pc=scene_pc if mask is not None else None,
        grasp_poses=grasps_np,
        confidence=confidence,
        collision_mask=collision_mask,
        output_path=os.path.join(OUTPUT_DIR, output_prefix + ".png"),
        gripper_info=gripper_info,
    )

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)

    return {
        "scene_pc": scene_pc,
        "object_pc": object_pc,
        "object_pc_processed": object_pc_processed,
        "grasps": grasps_np,
        "confidence": confidence.numpy(),
        "collision_mask": collision_mask,
        "num_collision_free": int(num_free),
        "total_grasps": len(grasps_np),
    }


def run_on_test_data(model, cfg, gripper_info):
    """使用合成测试数据运行管线（验证代码可用性）。

    在没有 GraspNet 数据时，用 GraspGen 自带的 box.obj 生成合成深度图测试管线。
    """
    import trimesh
    from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal
    import torch

    print("=" * 60)
    print("  TEST MODE: Synthetic Data Pipeline Verification")
    print("=" * 60)

    # 加载测试物体
    mesh_path = GRASPGEN_DIR / "assets" / "objects" / "box.obj"
    if not mesh_path.exists():
        print(f"[ERROR] Test mesh not found: {mesh_path}")
        print("  assets/ directory excluded from git. Using random point cloud instead.")
        pc = np.random.randn(1024, 3) * 0.1
        pc = pc - pc.mean(axis=0)
    else:
        mesh = trimesh.load(str(mesh_path))
        pc, _ = trimesh.sample.sample_surface(mesh, 2000)
        pc = np.array(pc)
        pc = pc - pc.mean(axis=0)  # center

    # 模拟深度图 (用点云反投影)
    fx = fy = 500.0
    cx = cy = 256.0
    depth = np.zeros((512, 512), dtype=np.float32)
    rgb = np.zeros((512, 512, 3), dtype=np.uint8)

    # 简单模拟: 直接传入点云
    print(f"\n[Test] Using {len(pc)} synthetic points")

    # 预处理
    pc_processed = preprocess_point_cloud(pc, num_points=NUM_POINTS)
    pc_stats(pc_processed, "test_processed")

    # 推理
    print(f"\n[Test] Running inference...")
    grasps, confidence = run_graspgen_inference(
        model, pc_processed, num_grasps=min(NUM_GRASPS, 50), kappa=KAPPA
    )

    if len(grasps) == 0:
        print("[ERROR] No grasps generated!")
        return None

    grasps_np = grasps.numpy().copy()
    grasps_np[:, :3, 3] += pc.mean(axis=0)  # 还原中心化

    print(f"  Generated {len(grasps_np)} grasps")
    print(f"  Confidence: min={confidence.min():.4f}, max={confidence.max():.4f}")

    # 碰撞检测
    print(f"\n[Test] Collision detection...")
    collision_mask = check_grasp_collisions(
        scene_pc=pc,
        grasp_poses=grasps_np,
        gripper_collision_mesh=gripper_info.collision_mesh,
        threshold=COLLISION_THRESHOLD,
        num_samples=500,
    )
    print(f"  Collision-free: {collision_mask.sum()}/{len(grasps_np)}")

    # 可视化
    print(f"\n[Test] Open3D Visualization...")
    visualize_results(
        object_pc=pc,
        scene_pc=None,
        grasp_poses=grasps_np,
        confidence=confidence,
        collision_mask=collision_mask,
        output_path=os.path.join(OUTPUT_DIR, "test_result.png"),
        gripper_info=gripper_info,
    )

    return {"grasps": grasps_np, "confidence": confidence.numpy(),
            "collision_mask": collision_mask}


def run_test_with_pretrained(sampler, cfg, gripper_info):
    """使用官方预训练模型 (GraspGenSampler) 的测试管线。"""
    import trimesh

    print("=" * 60)
    print("  TEST MODE: Using OFFICIAL Pretrained Model")
    print("=" * 60)

    # 加载测试物体
    mesh_path = GRASPGEN_DIR / "assets" / "objects" / "box.obj"
    if mesh_path.exists():
        mesh = trimesh.load(str(mesh_path))
        pc, _ = trimesh.sample.sample_surface(mesh, 2000)
        pc = np.array(pc)
        pc = pc - pc.mean(axis=0)
    else:
        pc = np.random.randn(1024, 3) * 0.1
        pc = pc - pc.mean(axis=0)

    pc_center = pc.mean(axis=0)
    pc_centered = pc - pc_center

    # 推理 (使用 GraspGenSampler.run_inference)
    print(f"\n[Test] Running inference with official model...")
    grasps, confidence = GraspGenSampler.run_inference(
        object_pc=pc_centered,
        grasp_sampler=sampler,
        num_grasps=min(NUM_GRASPS, 50),
        remove_outliers=False,
    )

    if len(grasps) == 0:
        print("[ERROR] No grasps generated!")
        return None

    grasps_np = grasps.cpu().numpy()
    grasps_np[:, :3, 3] += pc_center  # 还原中心化

    print(f"  Generated {len(grasps_np)} grasps")
    conf_np = confidence.cpu().numpy()
    print(f"  Confidence: min={conf_np.min():.4f}, max={conf_np.max():.4f}")

    # 碰撞检测
    print(f"\n[Test] Collision detection...")
    from collision import check_grasp_collisions
    collision_mask = check_grasp_collisions(
        scene_pc=pc,
        grasp_poses=grasps_np,
        gripper_collision_mesh=gripper_info.collision_mesh,
        threshold=COLLISION_THRESHOLD,
        num_samples=500,
    )
    print(f"  Collision-free: {collision_mask.sum()}/{len(grasps_np)}")

    # 可视化
    print(f"\n[Test] Open3D Visualization...")
    visualize_results(
        object_pc=pc,
        scene_pc=None,
        grasp_poses=grasps_np,
        confidence=confidence,
        collision_mask=collision_mask,
        output_path=os.path.join(OUTPUT_DIR, "test_pretrained_result.png"),
        gripper_info=gripper_info,
    )

    return {"grasps": grasps_np, "confidence": conf_np, "collision_mask": collision_mask}


def main():
    parser = argparse.ArgumentParser(description="GraspNet-1Billion GraspGen Inference Pipeline")
    parser.add_argument("--test", action="store_true",
                        help="Run pipeline on synthetic test data")
    parser.add_argument("--our_model", action="store_true",
                        help="Use our trained model (160 epochs) instead of official pretrained")
    parser.add_argument("--rgb", type=str, help="Path to RGB image")
    parser.add_argument("--depth", type=str, help="Path to depth image (16-bit PNG, mm)")
    parser.add_argument("--mask", type=str, help="Path to instance mask (optional)")
    parser.add_argument("--fx", type=float, default=591.0, help="Camera focal length x")
    parser.add_argument("--fy", type=float, default=590.6, help="Camera focal length y")
    parser.add_argument("--cx", type=float, default=322.5, help="Camera principal point x")
    parser.add_argument("--cy", type=float, default=238.3, help="Camera principal point y")
    parser.add_argument("--config", type=str,
                        default=str(GRASPGEN_DIR / "runs" / "results_assignment" / "logs" / "config.yaml"),
                        help="Path to training config.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default=str(CHECKPOINT_DIR / "last.pth"),
                        help="Path to generator checkpoint")
    args = parser.parse_args()

    # ── 加载模型 ──
    if args.our_model:
        print("Loading OUR trained model (160 epochs)...")
        model, cfg, gripper_info = load_our_model(
            checkpoint_path=args.checkpoint,
            config_yaml=args.config,
        )
        use_pretrained = False
    else:
        print("Loading OFFICIAL pretrained model...")
        sampler, cfg, gripper_info = load_pretrained_model()
        use_pretrained = True

    # ── 运行管线 ──
    if args.test:
        if use_pretrained:
            run_test_with_pretrained(sampler, cfg, gripper_info)
        else:
            run_on_test_data(model, cfg, gripper_info)
    elif args.rgb and args.depth:
        import imageio
        rgb = imageio.imread(args.rgb)
        depth_raw = imageio.imread(args.depth)
        # GraspNet depth 通常是 16-bit PNG，单位 mm → m
        if depth_raw.dtype == np.uint16:
            depth = depth_raw.astype(np.float32) / 1000.0
        else:
            depth = depth_raw.astype(np.float32)
        mask = None
        if args.mask:
            mask = imageio.imread(args.mask) > 0

        if use_pretrained:
            pipeline_from_rgbd_pretrained(
                sampler, cfg, gripper_info,
                rgb=rgb, depth=depth,
                fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                mask=mask,
                output_prefix="graspnet_result",
            )
        else:
            pipeline_from_rgbd(
                model, cfg, gripper_info,
                rgb=rgb, depth=depth,
                fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                mask=mask,
                output_prefix="graspnet_result",
            )
    else:
        print("Usage:")
        print("  --test              Run pipeline on synthetic test data")
        print("  --rgb + --depth     Run pipeline on RGB-D images")
        print("\nNo valid arguments provided. Running --test by default.")
        if use_pretrained:
            run_test_with_pretrained(sampler, cfg, gripper_info)
        else:
            run_on_test_data(model, cfg, gripper_info)


if __name__ == "__main__":
    main()
