#!/usr/bin/env python3
"""
[作业新增文件] graspnet_pipeline/collision.py
非 GraspGen 原始项目文件。

几何碰撞检测：检查预测的夹爪抓取位姿是否与场景点云发生穿模。

复用 GraspGen:
  - grasp_gen/utils/point_cloud_utils.py:filter_colliding_grasps
"""

import sys
sys.path.insert(0, "/home/lzl/Projects/6Dpose/GraspGen")

import numpy as np
import trimesh
from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps


def check_grasp_collisions(
    scene_pc: np.ndarray,
    grasp_poses: np.ndarray,
    gripper_collision_mesh: trimesh.Trimesh,
    threshold: float = 0.003,
    num_samples: int = 1000,
) -> np.ndarray:
    """对一组抓取位姿进行碰撞检测。

    Args:
        scene_pc:               (N, 3) 场景点云 (世界坐标系)
        grasp_poses:            (K, 4, 4) 抓取位姿矩阵
        gripper_collision_mesh: 夹爪碰撞网格 (trimesh.Trimesh)
        threshold:              碰撞判定距离阈值 (m)，默认 3mm
        num_samples:            夹爪表面采样点数

    Returns:
        collision_free_mask:   (K,) bool 数组，True=无碰撞
    """
    if len(grasp_poses) == 0:
        return np.array([], dtype=bool)

    return filter_colliding_grasps(
        scene_pc=scene_pc,
        grasp_poses=grasp_poses,
        gripper_collision_mesh=gripper_collision_mesh,
        collision_threshold=threshold,
        num_collision_samples=num_samples,
    )
