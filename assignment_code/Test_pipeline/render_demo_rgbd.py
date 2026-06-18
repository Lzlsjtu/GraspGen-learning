#!/usr/bin/env python3
"""
[作业新增] render_demo_rgbd.py
从官方 GraspGen demo 场景 JSON 中提取原始 RGB-D 图像并保存；
同时使用针孔相机投影验证映射一致性。

用法:
  python Test_pipeline/render_demo_rgbd.py \
    --scene_json GraspGen/GraspGenModels/sample_data/real_scene_pc/1745766797_642935.json \
    --output_dir Test_pipeline/outputs/demo_rgbd
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_ply_generic(pts, cols, path, label=""):
    """保存通用彩色点云 PLY。"""
    n = len(pts)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    lines = [f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} {int(cols[i,0])} {int(cols[i,1])} {int(cols[i,2])}" for i in range(n)]
    path.write_text(header + "\n".join(lines) + "\n")
    M = path.stat().st_size / 1024**2
    print(f"Saved PLY       → {path} ({n} pts, {M:.1f} MB) {label}")


def save_object_ply(data, out_dir):
    """从 object_info 中提取物体点云及颜色，保存为 PLY。"""
    obj_pc = np.array(data["object_info"]["pc"], dtype=np.float64)
    obj_color = np.array(data["object_info"]["pc_color"], dtype=np.uint8)
    if obj_pc.ndim > 2:
        obj_pc = obj_pc.reshape(-1, 3)
    if obj_color.ndim > 2:
        obj_color = obj_color.reshape(-1, 3)
    save_ply_generic(obj_pc, obj_color, Path(out_dir) / "demo_object.ply", label="object + grasp colors</br>(来自 object_info.pc + pc_color)")


def save_scene_ply(data, img_color, H, W, out_dir):
    """从 full_pc + img_color 提取场景点云，按 mask 分物体/背景保存两个 PLY。"""
    pc = np.array(data["scene_info"]["full_pc"], dtype=np.float64).reshape(-1, 3)
    rgb = img_color.reshape(-1, 3).astype(np.uint8)
    mask = np.array(data["scene_info"]["obj_mask"], dtype=bool).reshape(-1)

    N = len(pc)
    z = pc[:, 2]
    valid = z > 0.01
    valid_indices = np.where(valid)[0]

    # 物体点 (mask valid)
    obj_idx = valid_indices[mask[valid_indices]]
    # 背景点 (non-mask valid) 降采样
    bg_idx_all = valid_indices[~mask[valid_indices]]
    bg_n = min(len(bg_idx_all), 30000)
    if bg_n > 0:
        bg_idx = np.random.choice(bg_idx_all, bg_n, replace=False)

    # 场景 PLY: 物体+背景
    scene_pts = np.vstack([pc[obj_idx], pc[bg_idx]]) if bg_n > 0 else pc[obj_idx]
    scene_cols = np.vstack([rgb[obj_idx], rgb[bg_idx]]) if bg_n > 0 else rgb[obj_idx]
    save_ply_generic(scene_pts, scene_cols, Path(out_dir) / "demo_scene.ply",
                     label=f"obj={len(obj_idx)} bg={bg_n if bg_n>0 else 0}</br>(来自 full_pc + img_color, 像素级对应)")


def extract_original_rgbd(data, out_dir):
    """从 JSON 中提取原始 RGB 和 Depth 图像并保存。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_color = np.asarray(data["scene_info"]["img_color"], dtype=np.uint8)
    img_depth_raw = np.asarray(data["scene_info"]["img_depth"], dtype=np.uint16)

    # 原始分辨率
    if img_color.ndim == 1 and img_depth_raw.ndim == 1:
        # 尝试 reshape 到合理分辨率 (720, 1280)
        H, W = 720, 1280
        img_color = img_color.reshape(H, W, 3)
        img_depth_raw = img_depth_raw.reshape(H, W)
    elif img_color.ndim > 2:
        H, W = img_color.shape[:2]
    else:
        raise RuntimeError(f"Unexpected image shapes: color={img_color.shape} depth={img_depth_raw.shape}")

    # 保存原始深度图 (灰度 PNG)
    depth_vis = np.clip(img_depth_raw.astype(float) / 1000.0, 0, 5)  # meter, clip to 5m for viz
    depth_vis_8 = (depth_vis / depth_vis.max() * 255).astype(np.uint8)
    plt.imsave(out / "demo_depth_raw.png", depth_vis_8, cmap="gray")
    # 保存 16-bit 深度 (PNG 可直接读)
    plt.imsave(out / "demo_depth_mm.png", img_depth_raw, cmap="gray")

    # 保存 RGB
    plt.imsave(out / "demo_rgb_original.png", img_color.astype(np.uint8))

    # 深度叠加 RGB (colored depth overlay)
    depth_f = img_depth_raw.astype(float) / 1000.0
    valid = depth_f > 0
    depth_norm = np.clip(depth_f[valid] / 2.0, 0, 1)
    cmap = plt.get_cmap("turbo")
    overlay = img_color.copy().astype(float)
    colors = (cmap(1.0 - depth_norm)[:, :3] * 255)
    for ch in range(3):
        overlay_flat = overlay[:, :, ch].copy()
        overlay_flat[valid] = (overlay_flat[valid] * 0.4 + colors[:, ch] * 0.6)
        overlay[:, :, ch] = np.clip(overlay_flat, 0, 255)
    plt.imsave(out / "demo_depth_overlay.png", overlay.astype(np.uint8))

    print(f"Extracted RGB    → {out / 'demo_rgb_original.png'} ({W}×{H})")
    print(f"Extracted Depth  → {out / 'demo_depth_mm.png'} ({W}×{H})")
    print(f"Depth range: {img_depth_raw[valid].min()} ~ {img_depth_raw[valid].max()} mm")

    # ---- 保存物体点云 (object_info.pc + pc_color) ----
    save_object_ply(data, out_dir)

    # ---- 保存场景点云 (full_pc + img_color, 像素级对应) ----
    save_scene_ply(data, img_color, H, W, out_dir)

    return H, W, img_depth_raw


def pinhole_verify(data, H, W, fx, fy, cx, cy, out_dir):
    """用针孔投影验证 3D → 2D 映射一致性并输出对比图。"""
    pc = np.array(data["scene_info"]["full_pc"])
    if pc.ndim > 2:
        pc = pc.reshape(-1, 3)
    H_img, W_img = H, W

    # 取部分点验证
    sample = pc[np.random.choice(len(pc), min(5000, len(pc)), replace=False)]
    x, y, z = sample[:, 0], sample[:, 1], sample[:, 2]
    u = (fx * x / z + cx).astype(int)
    v = (fy * y / z + cy).astype(int)
    mask = (z > 0.01) & (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    # Left: RGB
    img_color = np.asarray(data["scene_info"]["img_color"], dtype=np.uint8)
    if img_color.ndim == 1:
        img_color = img_color.reshape(H_img, W_img, 3)
    axes[0].imshow(img_color)
    axes[0].scatter(u[mask], v[mask], c='lime', s=1, alpha=0.6)
    axes[0].set_title(f"RGB + projected 3D points ({mask.sum()} visible)")
    axes[0].axis("off")

    # Right: Depth
    img_depth = np.asarray(data["scene_info"]["img_depth"], dtype=np.uint16)
    if img_depth.ndim == 1:
        img_depth = img_depth.reshape(H_img, W_img)
    axes[1].imshow(img_depth, cmap="gray")
    axes[1].scatter(u[mask], v[mask], c='cyan', s=1, alpha=0.6)
    axes[1].set_title("Depth map + projected 3D points")
    axes[1].axis("off")

    save_path = Path(out_dir) / "demo_projection_verify.png"
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Projection verify → {save_path}  (visible={mask.sum()}/{len(sample)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json", required=True)
    parser.add_argument("--output_dir", default="Test_pipeline/outputs/demo_rgbd")
    parser.add_argument("--fx", type=float, default=591.0)
    parser.add_argument("--fy", type=float, default=590.6)
    parser.add_argument("--cx", type=float, default=639.0)  # 1280/2
    parser.add_argument("--cy", type=float, default=359.0)  # 720/2
    args = parser.parse_args()

    with open(args.scene_json) as f:
        data = json.load(f)

    H, W, depth_mm = extract_original_rgbd(data, args.output_dir)
    pinhole_verify(data, H, W, args.fx, args.fy, args.cx, args.cy, args.output_dir)


if __name__ == "__main__":
    main()
