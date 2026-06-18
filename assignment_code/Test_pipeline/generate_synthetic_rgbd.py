#!/usr/bin/env python3
"""
[作业新增] 从已有 mesh 生成合成 RGB-D 图像用于管线测试。
使用 Open3D 的离线渲染器，无需显示器。

用法:
  python generate_synthetic_rgbd.py                    # 使用随机 GLB
  python generate_synthetic_rgbd.py --mesh <file.glb>  # 使用指定 mesh
"""
import sys, os, argparse, numpy as np
from pathlib import Path

def generate_synthetic_rgbd(mesh_path=None, output_dir=None):
    """生成合成 RGB-D 图像。"""
    import open3d as o3d
    import trimesh
    import imageio

    if output_dir is None:
        output_dir = Path("/home/lzl/Projects/6Dpose/GraspNet_datasets/synthetic")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找或加载 mesh
    if mesh_path is None:
        # 随机选一个 GLB 文件
        glb_dir = Path("/home/lzl/Projects/6Dpose/GraspGen_datasets/object_dataset")
        glb_files = list(glb_dir.glob("*.glb"))
        if not glb_files:
            print("[ERROR] No GLB files found!")
            return None
        mesh_path = str(np.random.choice(glb_files))

    print(f"[Mesh] {mesh_path}")

    # 加载 mesh
    mesh_trimesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh_trimesh, trimesh.Scene):
        mesh_trimesh = trimesh.util.concatenate(mesh_trimesh.dump())
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(np.array(mesh_trimesh.vertices))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(np.array(mesh_trimesh.faces))
    mesh_o3d.compute_vertex_normals()

    # 归一化到单位球，放在原点
    bbox = mesh_o3d.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    scale = max(bbox.get_extent())
    mesh_o3d.translate(-center)
    mesh_o3d.scale(1.0 / scale, center=[0, 0, 0])
    # 把物体放在相机前方 0.5m 处
    mesh_o3d.translate([0, 0, 0.5])

    print(f"[Mesh] Vertices: {len(mesh_o3d.vertices)}, Triangles: {len(mesh_o3d.triangles)}")

    # 使用 Open3D 离线渲染器 (headless)
    W, H = 640, 480
    fx, fy = 591.0, 590.6   # RealSense D435 intrinsics
    cx, cy = 322.5, 238.3

    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    extrinsic = np.eye(4)  # 相机在原点

    # 创建渲染器
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background(np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32))

    # 材质
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.7, 0.5, 0.3, 1.0]  # 棕橙色

    renderer.scene.add_geometry("object", mesh_o3d, material)

    # 设置相机
    renderer.setup_camera(intrinsic, extrinsic)

    # 添加光照
    renderer.scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
        [0.5, 0.5, -1.0]  # 光源方向
    )

    # 渲染
    img_rgb = np.asarray(renderer.render_to_image())
    img_depth = np.asarray(renderer.render_to_depth_image())

    renderer = None  # release

    print(f"[RGB]   shape={img_rgb.shape}, dtype={img_rgb.dtype}, range=[{img_rgb.min()},{img_rgb.max()}]")
    print(f"[Depth] shape={img_depth.shape}, dtype={img_depth.dtype}, range=[{img_depth.min():.4f},{img_depth.max():.4f}]")

    # 转换为标准格式
    # RGB: uint8 → 保持
    # Depth: float32 (m) → uint16 (mm)
    img_depth_mm = (img_depth * 1000).astype(np.uint16)

    # 保存
    rgb_path = output_dir / "rgb.png"
    depth_path = output_dir / "depth.png"
    imageio.imwrite(str(rgb_path), img_rgb)
    imageio.imwrite(str(depth_path), img_depth_mm)
    print(f"\n[Saved] {rgb_path}")
    print(f"[Saved] {depth_path}")
    print(f"[Intrinsics] fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    info_path = output_dir / "info.txt"
    with open(info_path, 'w') as f:
        f.write(f"mesh: {mesh_path}\n")
        f.write(f"fx={fx} fy={fy} cx={cx} cy={cy}\n")
        f.write(f"width={W} height={H}\n")
    print(f"[Saved] {info_path}")

    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": W, "height": H,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    generate_synthetic_rgbd(args.mesh, args.output)
