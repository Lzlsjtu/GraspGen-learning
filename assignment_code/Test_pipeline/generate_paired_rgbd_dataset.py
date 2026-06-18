#!/usr/bin/env python3
"""
[作业新增] 为所有 31 个点云生成 RGB-D + Mask + 点云 配对数据集。

数据源:
  1. real_scene_pc/  (18个场景) - JSON 中已包含 img_color, img_depth, full_pc, obj_mask
  2. real_object_pc/ (13个物体) - 仅有点云，需通过 Poisson 重建→渲染生成 RGB-D

每个样本输出目录包含:
  - rgb.png          (720×1280 or 480×640, uint8)
  - depth.png        (同上, uint16, 毫米)
  - mask.png         (同上, uint8, 0/255)
  - pc.npy           (N, 3, float32, 点云)
  - intrinsics.txt   (相机内参)
  - info.txt         (统计信息)

用法:
  cd Test_pipeline && python generate_paired_rgbd_dataset.py
"""
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "GraspGen"))

# 尝试导入 Open3D
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    print("[WARN] Open3D not available, object point clouds will be skipped")

import imageio


# 场景点云内参 (720p RealSense)
SCENE_H, SCENE_W = 720, 1280
SCENE_FX, SCENE_FY = 591.0, 590.6
SCENE_CX, SCENE_CY = 639.5, 359.5  # 1280/2, 720/2

# 物体点云渲染内参
OBJ_H, OBJ_W = 480, 640
OBJ_FX, OBJ_FY = 591.0, 590.6
OBJ_CX, OBJ_CY = 319.5, 239.5


def process_scene_pc(json_path, output_root):
    """处理场景点云（JSON 中已有完整 RGB-D + Mask）。"""
    name = Path(json_path).stem
    output_dir = output_root / "scenes" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 提取数据并 reshape
    img_color = np.array(data['scene_info']['img_color'], dtype=np.uint8)
    img_depth_mm = np.array(data['scene_info']['img_depth'], dtype=np.uint16)
    full_pc = np.array(data['scene_info']['full_pc'], dtype=np.float32)
    obj_mask = np.array(data['scene_info']['obj_mask'], dtype=bool)

    # reshape 到 720×1280
    img_color = img_color.reshape(SCENE_H, SCENE_W, 3)
    img_depth_mm = img_depth_mm.reshape(SCENE_H, SCENE_W)
    obj_mask = obj_mask.reshape(SCENE_H, SCENE_W)
    pc = full_pc.reshape(-1, 3)

    # 保存
    imageio.imwrite(str(output_dir / "rgb.png"), img_color)
    imageio.imwrite(str(output_dir / "depth.png"), img_depth_mm)
    imageio.imwrite(str(output_dir / "mask.png"), (obj_mask * 255).astype(np.uint8))
    np.save(str(output_dir / "pc.npy"), pc)

    with open(output_dir / "intrinsics.txt", 'w') as f:
        f.write(f"fx={SCENE_FX}\nfy={SCENE_FY}\ncx={SCENE_CX}\ncy={SCENE_CY}\n")
        f.write(f"width={SCENE_W}\nheight={SCENE_H}\n")

    valid_depth = img_depth_mm > 0
    with open(output_dir / "info.txt", 'w') as f:
        f.write(f"Type: real_scene (extracted from JSON)\n")
        f.write(f"Point cloud: {len(pc)} points\n")
        f.write(f"Depth range: {img_depth_mm[valid_depth].min()} ~ {img_depth_mm[valid_depth].max()} mm\n")
        f.write(f"Object mask pixels: {obj_mask.sum()} / {SCENE_W*SCENE_H} ({obj_mask.sum()/(SCENE_W*SCENE_H)*100:.1f}%)\n")

    return True


def pc_to_mesh_simple(pc_np):
    """简化版点云→网格：Ball Pivoting 比 Poisson 更快更稳定。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    # Ball Pivoting 重建 (更快)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [avg_dist * 2, avg_dist * 4, avg_dist * 8]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    return mesh


def render_object_rgbd(mesh):
    """渲染物体 RGB-D + Mask。"""
    # 归一化
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    mesh.translate(-center)
    scale = max(bbox.get_extent()) * 2.5
    mesh.scale(1.0 / scale, center=[0, 0, 0])
    mesh.translate([0, 0, 0.6])  # 放在相机前方

    intrinsic = o3d.camera.PinholeCameraIntrinsic(OBJ_W, OBJ_H, OBJ_FX, OBJ_FY, OBJ_CX, OBJ_CY)
    extrinsic = np.eye(4)

    renderer = o3d.visualization.rendering.OffscreenRenderer(OBJ_W, OBJ_H)
    renderer.scene.set_background(np.array([0, 0, 0, 1.0], dtype=np.float32))

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.7, 0.5, 0.3, 1.0]
    renderer.scene.add_geometry("object", mesh, material)
    renderer.setup_camera(intrinsic, extrinsic)
    renderer.scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
        [0.5, 0.5, -1.0]
    )

    img_rgb = np.asarray(renderer.render_to_image())
    img_depth = np.asarray(renderer.render_to_depth_image())

    # Mask
    mask = img_depth < 0.99

    renderer = None
    return img_rgb, img_depth, mask


def process_object_pc(json_path, output_root):
    """处理物体点云（生成合成 RGB-D）。"""
    if not O3D_AVAILABLE:
        return False

    name = Path(json_path).stem
    output_dir = output_root / "objects" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 加载点云
    if "pc" in data:
        pc = np.array(data["pc"], dtype=np.float32)
    elif "object_info" in data:
        pc = np.array(data["object_info"]["pc"], dtype=np.float32)
    else:
        # 尝试所有可能的 key
        for k in data.keys():
            if isinstance(data[k], list) and len(data[k]) > 0 and isinstance(data[k][0], list):
                pc = np.array(data[k], dtype=np.float32)
                break
        else:
            return False

    # 点云→网格→渲染
    try:
        mesh = pc_to_mesh_simple(pc)
        img_rgb, img_depth, mask = render_object_rgbd(mesh)
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

    # 保存
    imageio.imwrite(str(output_dir / "rgb.png"), img_rgb)
    img_depth_mm = (np.clip(img_depth, 0, 10.0) * 1000).astype(np.uint16)
    imageio.imwrite(str(output_dir / "depth.png"), img_depth_mm)
    imageio.imwrite(str(output_dir / "mask.png"), (mask * 255).astype(np.uint8))
    np.save(str(output_dir / "pc.npy"), pc)

    with open(output_dir / "intrinsics.txt", 'w') as f:
        f.write(f"fx={OBJ_FX}\nfy={OBJ_FY}\ncx={OBJ_CX}\ncy={OBJ_CY}\n")
        f.write(f"width={OBJ_W}\nheight={OBJ_H}\n")

    valid_depth = img_depth_mm > 0
    with open(output_dir / "info.txt", 'w') as f:
        f.write(f"Type: synthetic_object (rendered from point cloud)\n")
        f.write(f"Original point cloud: {len(pc)} points\n")
        f.write(f"Depth range: {img_depth_mm[valid_depth].min()} ~ {img_depth_mm[valid_depth].max()} mm\n")
        f.write(f"Object pixels: {mask.sum()} / {OBJ_W*OBJ_H} ({mask.sum()/(OBJ_W*OBJ_H)*100:.1f}%)\n")

    return True


def main():
    root = Path("/home/lzl/Projects/6Dpose/GraspGen/GraspGenModels/sample_data")
    output_root = Path("/home/lzl/Projects/6Dpose/GraspGen_datasets/rgbd_paired")
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. 处理场景点云 (18个，直接提取)
    scene_files = sorted(list((root / "real_scene_pc").glob("*.json")))
    print(f"=== Processing {len(scene_files)} real scenes ===")
    scene_success = 0
    for f in tqdm(scene_files, desc="Scenes"):
        try:
            if process_scene_pc(f, output_root):
                scene_success += 1
                tqdm.write(f"  OK: {f.stem}")
        except Exception as e:
            tqdm.write(f"  FAIL: {f.stem}: {e}")

    # 2. 处理物体点云 (13个，合成渲染)
    object_files = sorted(list((root / "real_object_pc").glob("*.json")))
    print(f"\n=== Processing {len(object_files)} objects ===")
    object_success = 0
    for f in tqdm(object_files, desc="Objects"):
        try:
            if process_object_pc(f, output_root):
                object_success += 1
                tqdm.write(f"  OK: {f.stem}")
        except Exception as e:
            tqdm.write(f"  FAIL: {f.stem}: {e}")

    # 总结
    print(f"\n{'='*60}")
    print(f"数据集生成完成!")
    print(f"  场景: {scene_success}/{len(scene_files)}")
    print(f"  物体: {object_success}/{len(object_files)}")
    print(f"  总计: {scene_success + object_success}/31")
    print(f"  保存位置: {output_root}")
    print(f"{'='*60}\n")

    # 列出目录结构
    print("目录结构:")
    print(f"  {output_root}/")
    print(f"  ├── scenes/ (×{scene_success})")
    print(f"  │   ├── <scene_id>/")
    print(f"  │   │   ├── rgb.png       ({SCENE_W}×{SCENE_H})")
    print(f"  │   │   ├── depth.png     (uint16 mm)")
    print(f"  │   │   ├── mask.png      (0/255)")
    print(f"  │   │   ├── pc.npy        (N,3)")
    print(f"  │   │   ├── intrinsics.txt")
    print(f"  │   │   └── info.txt")
    print(f"  └── objects/ (×{object_success})")
    print(f"      └── <object_id>/ (同上)")


if __name__ == "__main__":
    main()
