#!/usr/bin/env python3
"""
[作业新增文件] graspnet_pipeline/visualize.py
非 GraspGen 原始项目文件。

Open3D 三维可视化：点云 + 夹爪网格 + 预测抓取位姿 + 碰撞检测结果。
"""

import sys
sys.path.insert(0, "/home/lzl/Projects/6Dpose/GraspGen")

import os
import numpy as np
import trimesh
import trimesh.transformations as tra
import open3d as o3d
from pathlib import Path


# ─── 夹爪网格路径 ────────────────────────────────────────────────────────────
GRIPPER_MESH_DIR = Path("/home/lzl/Projects/6Dpose/GraspGen/assets/franka/meshes/collision")
GRIPPER_VISUAL_DIR = Path("/home/lzl/Projects/6Dpose/GraspGen/assets/franka/meshes/visual")
# 如果没有 assets，尝试从 GraspGenModels 加载
# 夹爪 mesh 也可以通过 gripper_info.collision_mesh 获取


def load_gripper_meshes(gripper_info):
    """从 GripperInfo 加载夹爪可视化网格。"""
    # 使用 trimesh 加载碰撞网格用于显示
    collision_mesh = gripper_info.collision_mesh
    if hasattr(gripper_info, 'visual_mesh'):
        visual_mesh = gripper_info.visual_mesh
    else:
        visual_mesh = collision_mesh
    return collision_mesh, visual_mesh


def create_gripper_lines(grasp_pose: np.ndarray, depth: float = 0.105, width: float = 0.08):
    """创建简化的夹爪线框表示 (二指平行夹爪)。

    画三根线段:
      - 基座横杆 (z=0 平面)
      - 左手指 (z=0 → z=depth)
      - 右手指 (z=0 → z=depth)

    Args:
        grasp_pose: (4, 4) 抓取位姿
        depth:      夹爪手指长度 (默认 0.105m for franka_panda)
        width:      夹爪半宽 (默认 0.04m)

    Returns:
        o3d.geometry.LineSet
    """
    half_w = width / 2

    # 局部坐标系下线框顶点
    points_local = np.array([
        [-half_w, 0, 0],          # 0: 基座左
        [ half_w, 0, 0],          # 1: 基座右
        [-half_w, 0, depth],      # 2: 指尖左
        [ half_w, 0, depth],      # 3: 指尖右
    ])

    # 变换到世界坐标
    points_world = tra.transform_points(points_local, grasp_pose)

    lines = np.array([
        [0, 1],  # 基座横杆
        [0, 2],  # 左手指
        [1, 3],  # 右手指
    ])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def visualize_results(
    object_pc: np.ndarray,
    scene_pc: np.ndarray,
    grasp_poses: np.ndarray,
    confidence: np.ndarray,
    collision_mask: np.ndarray,
    output_path: str = None,
    gripper_info=None,
):
    """使用 Open3D 可视化推理结果。

    Args:
        object_pc:      (N, 3) 物体点云
        scene_pc:       (M, 3) 场景点云 (可选，None 则仅显示物体)
        grasp_poses:    (K, 4, 4) 抓取位姿
        confidence:     (K,) 置信度
        collision_mask: (K,) bool 碰撞检测结果
        output_path:    截图保存路径 (可选)
        gripper_info:   GripperInfo (用于夹爪 mesh 加载)
    """
    geometries = []

    # ── 物体点云 (蓝色) ──
    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(object_pc)
    pcd_obj.paint_uniform_color([0.2, 0.5, 1.0])  # 蓝色
    geometries.append(pcd_obj)

    # ── 场景点云 (灰色，半透明) ──
    if scene_pc is not None and len(scene_pc) > 0:
        pcd_scene = o3d.geometry.PointCloud()
        pcd_scene.points = o3d.utility.Vector3dVector(scene_pc)
        pcd_scene.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色
        geometries.append(pcd_scene)

    # ── 夹爪线框 ──
    depth = 0.105  # franka_panda 手指长度
    if gripper_info is not None:
        depth = gripper_info.depth

    # 只显示 top-20 (避免画面太乱)
    if hasattr(confidence, 'numpy'):
        conf_np = confidence.detach().cpu().numpy()
    else:
        conf_np = np.array(confidence)
    sorted_idx = np.argsort(conf_np)[::-1]
    show_indices = sorted_idx[:20]

    for rank, idx in enumerate(show_indices):
        pose = grasp_poses[idx]
        is_collision_free = collision_mask[idx] if idx < len(collision_mask) else False

        lines = create_gripper_lines(pose, depth=depth)

        # 颜色: 绿色=无碰撞, 红色=碰撞, 透明度按 rank
        alpha = max(0.2, 1.0 - rank * 0.04)
        if is_collision_free:
            color = [0.0, 1.0, 0.0]   # 绿
        else:
            color = [1.0, 0.0, 0.0]   # 红

        lines.paint_uniform_color(color)
        geometries.append(lines)

        # Top-3: 添加坐标系
        if rank < 3:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            coord.transform(pose)
            geometries.append(coord)

    # ── 保存截图 ──
    if output_path is not None:
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=True, width=1280, height=720)
            for g in geometries:
                vis.add_geometry(g)
            vis.poll_events()
            vis.update_renderer()

            # 多视角截图
            ctr = vis.get_view_control()
            view_params = [
                ("front",  (0, 0, 800)),    # 正视图
                ("side",   (800, 0, 100)),  # 侧视图
                ("top",    (0, 800, 50)),   # 俯视图
            ]
            for view_name, cam_pos in view_params:
                ctr.set_front(cam_pos)
                ctr.set_zoom(0.8)
                vis.poll_events()
                vis.update_renderer()
                fname = output_path.replace(".png", f"_{view_name}.png")
                vis.capture_screen_image(fname)
                print(f"  Saved: {fname}")

            vis.destroy_window()
        except Exception as e:
            print(f"  [WARN] Open3D visualization failed (no display?): {e}")
            print(f"  [INFO] Skipping visualization, metrics still valid.")

    # ── 打印统计 ──
    num_collision_free = collision_mask.sum() if len(collision_mask) > 0 else 0
    print(f"\n[Vis] Total grasps: {len(grasp_poses)}")
    print(f"[Vis] Collision-free: {num_collision_free}/{len(grasp_poses)} "
          f"({100*num_collision_free/max(len(grasp_poses),1):.1f}%)")
    if hasattr(confidence, 'numpy'):
        conf_np = confidence.detach().cpu().numpy()
    else:
        conf_np = np.array(confidence)
    print(f"[Vis] Top-5 confidence: {conf_np[sorted_idx[:5]]}")

    return geometries
