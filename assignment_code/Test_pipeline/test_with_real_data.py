#!/usr/bin/env python3
"""
[作业新增] 使用 GraspGenModels 自带的真实场景 JSON 数据测试管线。
绕过深度图→点云步骤，直接从 JSON 中提取物体点云 + 场景点云。

用法:
  python test_with_real_data.py                    # 官方预训练模型
  python test_with_real_data.py --our_model        # 我们自己训练的模型
"""
import sys
sys.path.insert(0, "/home/lzl/Projects/6Dpose/GraspGen")

import json, os, time, argparse, numpy as np
from pathlib import Path

from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
from grasp_gen.grasp_server import GraspGenSampler

from config import (
    NUM_POINTS, SOR_K, SOR_THRESHOLD, KAPPA, NUM_GRASPS,
    COLLISION_THRESHOLD, NUM_COLLISION_SAMPLES, OUTPUT_DIR,
)
from model_loader import load_pretrained_model, load_our_model
from preprocess import preprocess_point_cloud, pc_stats
from collision import check_grasp_collisions
from visualize import visualize_results


def run_pipeline_from_json(json_path, model_info, output_prefix="real_scene"):
    """从真实场景 JSON 运行完整管线。"""
    sampler, model, cfg, gripper_info, use_pretrained = model_info

    print("=" * 60)
    print("  Real Scene JSON Pipeline")
    print("=" * 60)

    # 加载 JSON
    print(f"\n[Load] {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    scene_pc = np.array(data['scene_info']['full_pc'])[0]  # (N, 3)
    obj_mask = np.array(data['scene_info']['obj_mask'])     # (H, W) bool
    img_color = np.array(data['scene_info']['img_color'], dtype=np.uint8)
    img_depth = np.array(data['scene_info']['img_depth'], dtype=np.uint16)

    # 提取物体点云
    object_pc = scene_pc[obj_mask.flatten()]
    print(f"  Scene PC: {scene_pc.shape}")
    print(f"  Object PC: {object_pc.shape} (from mask)")
    pc_stats(object_pc, "object_raw")

    # Step 1: 预处理 (SOR + FPS)
    print(f"\n[Step 1] Preprocessing (SOR K={SOR_K} + FPS → {NUM_POINTS})")
    object_pc_processed = preprocess_point_cloud(
        object_pc, num_points=NUM_POINTS, sor_k=SOR_K, sor_threshold=SOR_THRESHOLD
    )
    pc_stats(object_pc_processed, "object_processed")

    # Step 2: 推理
    print(f"\n[Step 2] Inference ({'pretrained' if use_pretrained else 'our model'})")
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
            model, object_pc_processed, num_grasps=NUM_GRASPS, kappa=KAPPA
        )

    elapsed = time.time() - t0
    if isinstance(grasps, np.ndarray):
        grasps_np = grasps.copy()
    else:
        grasps_np = grasps.cpu().numpy()
    if hasattr(confidence, 'cpu'):
        conf_np = confidence.detach().cpu().numpy()
    else:
        conf_np = np.array(confidence)

    # 还原中心化
    pc_center = object_pc.mean(axis=0)
    grasps_np[:, :3, 3] += pc_center

    print(f"  Generated {len(grasps_np)} grasps in {elapsed:.2f}s")
    print(f"  Confidence: min={conf_np.min():.4f}, max={conf_np.max():.4f}, mean={conf_np.mean():.4f}")

    # Step 3: 碰撞检测
    print(f"\n[Step 3] Collision Detection (threshold={COLLISION_THRESHOLD*1000:.0f}mm)")
    # Use scene point cloud (without object) for collision checking
    scene_without_object = scene_pc[~obj_mask.flatten()]
    # Downsample scene for speed
    if len(scene_without_object) > 10000:
        idx = np.random.choice(len(scene_without_object), 10000, replace=False)
        scene_for_collision = scene_without_object[idx]
    else:
        scene_for_collision = scene_without_object

    collision_mask = check_grasp_collisions(
        scene_pc=scene_for_collision,
        grasp_poses=grasps_np,
        gripper_collision_mesh=gripper_info.collision_mesh,
        threshold=COLLISION_THRESHOLD,
        num_samples=NUM_COLLISION_SAMPLES,
    )
    num_free = collision_mask.sum()
    print(f"  Collision-free: {num_free}/{len(grasps_np)} ({100*num_free/max(len(grasps_np),1):.1f}%)")

    # Step 4: 可视化
    print(f"\n[Step 4] Visualization")
    out_path = os.path.join(OUTPUT_DIR, output_prefix + ".png")
    visualize_results(
        object_pc=object_pc,
        scene_pc=scene_for_collision,
        grasp_poses=grasps_np,
        confidence=confidence,
        collision_mask=collision_mask,
        output_path=out_path,
        gripper_info=gripper_info,
    )

    print("\n" + "=" * 60)
    print(f"  Pipeline Complete! Output: {OUTPUT_DIR}/")
    print("=" * 60)

    return {
        "object_pc": object_pc,
        "grasps": grasps_np,
        "confidence": conf_np,
        "collision_mask": collision_mask,
        "num_collision_free": int(num_free),
        "total_grasps": len(grasps_np),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--our_model", action="store_true", help="Use our trained model")
    parser.add_argument("--scene", type=str, default=None, 
                        help="Path to scene JSON (default: first real scene)")
    parser.add_argument("--all", action="store_true", help="Test all available scenes")
    args = parser.parse_args()

    # 加载模型
    if args.our_model:
        print("Loading OUR trained model...")
        model, cfg, gripper_info = load_our_model()
        model_info = (None, model, cfg, gripper_info, False)
    else:
        print("Loading OFFICIAL pretrained model...")
        sampler, cfg, gripper_info = load_pretrained_model()
        model_info = (sampler, None, cfg, gripper_info, True)

    # 选择场景
    scene_dir = Path("/home/lzl/Projects/6Dpose/GraspGen/GraspGenModels/sample_data/real_scene_pc")
    if args.scene:
        scenes = [args.scene]
    elif args.all:
        scenes = sorted(scene_dir.glob("*.json"))
    else:
        # 默认选第一个场景
        scenes = [str(sorted(scene_dir.glob("*.json"))[0])]

    print(f"\nTesting {len(scenes)} scene(s)\n")

    all_results = []
    for i, scene_path in enumerate(scenes):
        scene_name = Path(scene_path).stem
        print(f"\n{'#'*60}")
        print(f"# Scene {i+1}/{len(scenes)}: {scene_name}")
        print(f"{'#'*60}")
        result = run_pipeline_from_json(
            scene_path, model_info,
            output_prefix=f"real_scene_{scene_name}"
        )
        all_results.append(result)

    # 汇总
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  Summary Across All Scenes")
        print(f"{'='*60}")
        total_free = sum(r['num_collision_free'] for r in all_results if r)
        total_grasps = sum(r['total_grasps'] for r in all_results if r)
        print(f"  Total collision-free: {total_free}/{total_grasps} ({100*total_free/total_grasps:.1f}%)")


if __name__ == "__main__":
    main()
