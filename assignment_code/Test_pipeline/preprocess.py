#!/usr/bin/env python3
"""
[作业新增文件] graspnet_pipeline/preprocess.py
非 GraspGen 原始项目文件。

RGB-D 图像 → 点云的预处理管线。

复用 GraspGen:
  - grasp_gen/dataset/renderer.py:depth2points          (RGB-D → 相机坐标点云)
  - grasp_gen/utils/point_cloud_utils.py                  (SOR 离群点剔除)
  - torch_cluster.fps                                    (FPS 均匀降采样)
"""

import sys
sys.path.insert(0, "/home/lzl/Projects/6Dpose/GraspGen")

import numpy as np
import torch
from torch_cluster import fps

from grasp_gen.dataset.renderer import depth2points
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal


def depth_image_to_point_cloud(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    rgb: np.ndarray = None,
    mask: np.ndarray = None,
) -> np.ndarray:
    """从深度图生成相机坐标系点云。

    复用 GraspGen 的 depth2points，数学原理:
        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d

    Args:
        depth:  (H, W) 深度图 (米)
        fx, fy: 焦距 (像素)
        cx, cy: 光心 (像素)
        rgb:    可选 (H, W, 3) RGB 图像
        mask:   可选 (H, W) 二值掩膜 (True=保留)

    Returns:
        xyz: (N, 3) 有效像素的相机坐标系 3D 坐标
    """
    pts_data = depth2points(
        depth=depth,
        fx=int(fx), fy=int(fy),
        cx=int(cx), cy=int(cy),
        rgb=rgb,
    )

    xyz  = pts_data["xyz"]      # (H, W, 3)
    idx  = pts_data["index"]    # 有效像素 bool 索引 (depth > 0)

    xyz_valid = xyz[idx]        # (N, 3)

    # 可选：仅保留 mask 区域
    if mask is not None:
        mask_valid = mask.flatten()
        if len(mask_valid) == idx.sum():
            xyz_valid = xyz_valid[mask_valid]

    return xyz_valid


def preprocess_point_cloud(
    pc: np.ndarray,
    num_points: int = 1024,
    sor_k: int = 20,
    sor_threshold: float = 0.014,
) -> np.ndarray:
    """点云预处理：离群点剔除 + FPS 降采样 + 中心化。

    Args:
        pc:          (N, 3) 输入点云
        num_points:  目标点数
        sor_k:       SOR KNN 参数
        sor_threshold: SOR 距离阈值

    Returns:
        pc_processed: (num_points, 3) 处理后的点云
    """
    # ① SOR 离群点剔除
    pc_tensor = torch.from_numpy(pc).float()
    pc_filtered, _ = point_cloud_outlier_removal(
        pc_tensor, threshold=sor_threshold, K=sor_k
    )
    pc_filtered = pc_filtered.numpy()

    # ② FPS 均匀降采样
    if len(pc_filtered) > num_points:
        pc_tensor = torch.from_numpy(pc_filtered).float().unsqueeze(0).cuda()
        fps_idx = fps(pc_tensor[0], ratio=num_points / len(pc_filtered))
        # 如果 fps 返回的不是恰好 num_points，再补一次
        if len(fps_idx) < num_points:
            fps_idx = fps(pc_tensor[0], ratio=num_points / len(pc_filtered))
        pc_sampled = pc_filtered[fps_idx.cpu().numpy()[:num_points]]
    elif len(pc_filtered) < num_points:
        # 点数不足，随机重复
        idx = np.random.choice(len(pc_filtered), num_points, replace=True)
        pc_sampled = pc_filtered[idx]
    else:
        pc_sampled = pc_filtered

    # ③ 中心化
    pc_center = pc_sampled.mean(axis=0)
    pc_centered = pc_sampled - pc_center

    return pc_centered.astype(np.float32)


def pc_stats(pc: np.ndarray, name: str = "pc"):
    """打印点云统计信息。"""
    print(f"  [{name}] points={len(pc)}  "
          f"center=({pc[:,0].mean():.3f},{pc[:,1].mean():.3f},{pc[:,2].mean():.3f})  "
          f"extent=[{pc[:,0].min():.3f},{pc[:,0].max():.3f}] "
          f"× [{pc[:,1].min():.3f},{pc[:,1].max():.3f}] "
          f"× [{pc[:,2].min():.3f},{pc[:,2].max():.3f}]")
