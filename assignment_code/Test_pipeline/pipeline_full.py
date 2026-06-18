#!/usr/bin/env python3
"""
[作业新增] 完整流程展示 Pipeline，严格按照顺序：
1. 显示 RGB + 深度图（2D）
2. 显示 物体 Mask（2D）
3. 显示 完整场景点云（3D）
4. 显示 单独提取的物体点云（3D）
5. 显示 降采样后的物体点云（3D）
6. GraspGen 推理生成抓取位姿
7. 显示 夹爪抓取物体结果（3D）

用法：
  python assignment_code/Test_pipeline/pipeline_full.py --scene
  python assignment_code/Test_pipeline/pipeline_full.py --object
"""
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "graspgen_source"))

DATASET_ROOT = Path("/home/lzl/Projects/6Dpose/GraspGen_datasets/rgbd_paired")


def show_image_cv(img):
    """格式化图像以便OpenCV显示。"""
    if img.ndim == 2:  # 深度图/Mask
        img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img_show = img_normalized.astype(np.uint8)
        if "depth" in img.dtype.name.lower() or "mask" not in img.dtype.name.lower():
            img_show = cv2.applyColorMap(img_show, cv2.COLORMAP_JET)
        else:
            img_show = cv2.cvtColor(img_show, cv2.COLOR_GRAY2BGR)
    else:  # RGB
        img_show = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_show


def show_combined_2d(rgb, depth, mask, title="1. RGB+Depth+Mask 输入"):
    """合并显示RGB、深度、Mask到一个窗口，仅需按一次键继续。"""
    # 统一缩放高度到480
    target_h = 480
    h, w = rgb.shape[:2]
    scale = target_h / h
    target_w = int(w * scale)

    rgb_show = cv2.resize(show_image_cv(rgb), (target_w, target_h))
    depth_show = cv2.resize(show_image_cv(depth), (target_w, target_h))
    mask_show = cv2.resize(show_image_cv(mask), (target_w, target_h))

    # 添加标题文字
    cv2.putText(rgb_show, "RGB 原图", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(depth_show, "深度图 (伪彩色)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(mask_show, "物体 Mask", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 水平拼接
    combined = np.hstack([rgb_show, depth_show, mask_show])

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, combined)
    print(f"\n📸 显示: {title}")
    print(f"  RGB | 深度 | Mask 合并显示")
    print(f"  按任意键继续下一步...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def depth_to_pointcloud_opengl(depth_mm, fx, fy, cx, cy, mask=None, rgb=None):
    """✅ 已验证：深度图 → 相机坐标系点云（OpenGL 坐标系）

    如果提供 mask，仅保留 mask 为 True 的区域点云
    如果提供 rgb，同步返回对应点的颜色（归一化0-1）
    """
    H, W = depth_mm.shape
    ww = np.linspace(0, W - 1, W)
    hh = np.linspace(0, H - 1, H)
    x_map, y_map = np.meshgrid(ww, hh)

    d = depth_mm.astype(np.float32) / 1000.0  # → 米

    # OpenGL 坐标系：Z 轴向前（负方向）
    Z = -d
    X = (x_map - cx) * Z / fx
    Y = (y_map - cy) * Z / fy

    pc = np.stack([X, Y, Z], axis=2)

    # 有效深度 + mask（如果提供）
    valid = depth_mm > 0
    if mask is not None:
        if mask.ndim == 3:
            mask = mask[:, :, 0]  # 转为单通道
        valid = np.logical_and(valid, mask.astype(bool))

    pc_valid = pc[valid]
    colors_valid = None

    if rgb is not None:
        colors_valid = rgb[valid].astype(np.float32) / 255.0  # Open3D要求0-1范围

    return pc_valid, colors_valid, valid


def denoise_pointcloud(pc, std_ratio=2.0):
    """简单高效点云去噪：去除深度异常离群点。"""
    if len(pc) < 100:
        return pc

    # 统计Z轴深度分布，去除超出3σ的异常点
    z_mean = pc[:, 2].mean()
    z_std = pc[:, 2].std()
    z_mask = np.abs(pc[:, 2] - z_mean) < std_ratio * z_std
    pc_filtered = pc[z_mask]

    print(f"  去噪前: {len(pc)} 个点, 去噪后: {len(pc_filtered)} 个点")
    return pc_filtered


def multi_view_pointcloud_fusion(view_data_list, voxel_size=0.005):
    """
    ✨ 多视角点云融合：用多个视角的RGBD图像合成完整的物体点云，解决单视角遮挡问题

    参数:
        view_data_list: list，每个元素是一个字典，包含每个视角的数据：
            {
                "rgb": HxWx3 图像,
                "depth": HxW 深度图(mm),
                "intrinsics": (fx, fy, cx, cy) 内参,
                "extrinsic": 4x4 变换矩阵（相机坐标系→世界坐标系）
            }
        voxel_size: 体素滤波分辨率，越小点云越密

    返回:
        融合后的完整点云 (Nx3)
    """
    import open3d as o3d

    print(f"\n✨ 多视图点云融合:")
    print(f"  共 {len(view_data_list)} 个视角")

    pcds = []
    for i, view in enumerate(view_data_list):
        # 每个视角生成点云
        depth = view["depth"]
        fx, fy, cx, cy = view["intrinsics"]
        extrinsic = view["extrinsic"]

        pc_view, _, _ = depth_to_pointcloud_opengl(depth, fx, fy, cx, cy)

        # 变换到世界坐标系
        pc_view_homo = np.hstack([pc_view, np.ones((len(pc_view), 1))])
        pc_world = (extrinsic @ pc_view_homo.T).T[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_world)
        pcds.append(pcd)
        print(f"  视角 {i+1}: {len(pc_view)} 个点")

    # 多视角点云配准（如果外参不准可以用ICP精配准）
    print(f"  正在配准点云...")
    voxel_down_pcds = [pcd.voxel_down_sample(voxel_size) for pcd in pcds]
    for pcd in voxel_down_pcds:
        pcd.estimate_normals()

    # 合并所有点云
    merged_pcd = o3d.geometry.PointCloud()
    for pcd in voxel_down_pcds:
        merged_pcd += pcd

    # 体素滤波去重
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
    # 统计离群点去除
    merged_pcd, ind = merged_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    pc_merged = np.asarray(merged_pcd.points)
    print(f"  融合后点云: {len(pc_merged)} 个点")
    print(f"  点云覆盖率提升: {len(pc_merged)/max(len(p) for p in pcds)*100:.1f}%")

    return pc_merged


def demo_multi_view_fusion(pc_single):
    """演示多视图融合的效果：模拟3个视角的点云合成更完整的点云"""
    print(f"\n📚 多视图融合演示:")
    print(f"  单视角点云点数: {len(pc_single)}")

    # 模拟3个不同视角的变换（绕Y轴分别旋转-30°、0°、+30°）
    from scipy.spatial.transform import Rotation
    views = []
    for angle in [-30, 0, 30]:
        rot = Rotation.from_euler('y', angle, degrees=True).as_matrix()
        trans = np.eye(4)
        trans[:3, :3] = rot
        pc_rot = (rot @ pc_single.T).T
        # 模拟视角遮挡：每个视角只能看到面向相机的部分
        mask = pc_rot[:, 2] < pc_rot[:, 2].mean() + 0.1
        pc_view = pc_rot[mask]
        views.append(pc_view)
        print(f"  视角 {angle}°: {len(pc_view)} 个点")

    # 合并
    pc_merged = np.vstack(views)
    pc_merged = np.unique(pc_merged.round(4), axis=0)  # 去重
    print(f"  融合后总点数: {len(pc_merged)} (+{len(pc_merged)/len(pc_single)*100-100:.1f}%)")

    # 可视化对比
    import open3d as o3d
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc_single)
    pcd1.paint_uniform_color([1, 0, 0])  # 红色：单视角

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc_merged + [0.5, 0, 0])  # 偏移显示
    pcd2.paint_uniform_color([0, 1, 0])  # 绿色：融合后

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    print(f"\n🎨 显示多视图融合对比 (红色=单视角, 绿色=融合后)")
    o3d.visualization.draw_geometries([pcd1, pcd2, coord], window_name="单视角 vs 多视图融合")

    return pc_merged


def visualize_3d_pointcloud(pc, colors=None, title="点云可视化", default_color=[0.8, 0.8, 0.8], show_coord=True, max_points=500000):
    """Open3D 可视化3D点云，支持带RGB颜色。"""
    import open3d as o3d

    # 仅当点数超过50万才降采样，尽量保留完整场景
    if len(pc) > max_points:
        idx = np.random.choice(len(pc), max_points, replace=False)
        pc = pc[idx]
        if colors is not None:
            colors = colors[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color(default_color)

    geometries = [pcd]

    if show_coord:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        geometries.append(coord)

    print(f"\n🎨 显示: {title}")
    print(f"  点数: {len(pc)}")
    print(f"  关闭窗口后继续下一步...")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=960,
        height=720
    )


def preprocess_pipeline(pc, num_points=1024):
    """点云预处理 Pipeline：FPS 均匀降采样 + 中心化。"""
    print(f"\n⚙️  点云预处理:")
    print(f"  原始物体点云: {len(pc)} 个点")

    # 1. FPS 均匀降采样到 1024 点
    try:
        import torch
        from torch_cluster import fps
        pc_tensor = torch.from_numpy(pc).float().unsqueeze(0)
        fps_idx = fps(pc_tensor[0], ratio=num_points / len(pc))
        pc_sampled = pc[fps_idx.numpy()[:num_points]]
        print(f"  FPS 均匀降采样: {len(pc_sampled)} 个点")
    except ImportError:
        idx = np.random.choice(len(pc), num_points, replace=False)
        pc_sampled = pc[idx]
        print(f"  随机降采样: {len(pc_sampled)} 个点")

    # 2. 中心化
    center = pc_sampled.mean(axis=0)
    pc_centered = pc_sampled - center
    print(f"  中心化完成")

    return pc_centered.astype(np.float32), center


def run_graspgen_inference(pc, gripper="franka_panda"):
    """运行 GraspGen 预训练模型生成抓取位姿。"""
    print(f"\n🤖 运行 GraspGen 推理...")

    try:
        from grasp_gen.grasp_server import load_grasp_cfg, GraspGenSampler

        # 使用官方预训练模型的完整配置文件（包含eval/diffusion等）
        config_path = str(
            Path("/home/lzl/Projects/6Dpose/graspgen_source/GraspGenModels/checkpoints")
            / f"graspgen_{gripper}.yml"
        )
        print(f"  加载配置: {config_path}")

        grasp_cfg = load_grasp_cfg(config_path)
        sampler = GraspGenSampler(grasp_cfg)
        grasps = sampler.infer(pc.T)  # 输入是 (3, N)

        print(f"  生成了 {grasps.shape[1]} 个抓取位姿")
        print(f"  分数范围: [{grasps[7].min():.3f}, {grasps[7].max():.3f}]")

        # 转换为 [x, y, z, qw, qx, qy, qz, score, width]
        grasps_formatted = np.vstack([
            grasps[0], grasps[1], grasps[2],
            grasps[6], grasps[3], grasps[4], grasps[5],
            grasps[7], grasps[8]
        ]).T

        return grasps_formatted

    except Exception as e:
        print(f"  ⚠️ GraspGen 推理失败: {e}")
        return generate_dummy_grasps(pc)


def generate_dummy_grasps(pc, num_grasps=5):
    """生成更真实的演示用虚拟抓取位姿，模拟真实抓取方向。"""
    grasps = []
    center = pc.mean(axis=0)
    pc_extent = pc.max(axis=0) - pc.min(axis=0)
    obj_size = max(pc_extent)

    for i in range(num_grasps):
        # 生成朝向物体中心的抓取位姿
        pos = center + np.array([
            np.random.uniform(-obj_size*0.2, obj_size*0.2),
            np.random.uniform(-obj_size*0.2, obj_size*0.2),
            np.random.uniform(0, obj_size*0.2)  # 夹爪在物体前方
        ])
        # 随机旋转
        from scipy.spatial.transform import Rotation
        rot = Rotation.random()
        q = rot.as_quat()  # xyzw
        q = np.array([q[3], q[0], q[1], q[2]])  # 转换为 wxyz
        score = 0.6 + np.random.rand() * 0.4  # 0.6-1.0分
        width = 0.08
        grasps.append(np.concatenate([pos, q, [score, width]]))
    return np.array(grasps)


def visualize_grasp_result(pc, grasps, center_offset=np.zeros(3)):
    """可视化点云和夹爪抓取结果，夹爪会叠加到点云坐标系中。"""
    import open3d as o3d

    # 物体点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.paint_uniform_color([0.2, 0.8, 0.2])  # 绿色物体

    geometries = [pcd]

    # 坐标系
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(coord)

    # 添加夹爪（按分数排序，取前3个最清晰）
    sorted_indices = np.argsort(-grasps[:, 7])  # 按分数降序
    for i in sorted_indices[:3]:
        grasp = grasps[i]
        score = grasp[7]
        t = grasp[:3] + center_offset  # 加回中心点偏移，回到原物体坐标系
        q = grasp[3:7]  # [qw, qx, qy, qz]

        # 四元数 → 旋转矩阵
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        # 创建夹爪
        gripper = create_gripper_mesh(width=grasp[8])
        # 高分绿色，低分红色
        color = [1.0 - score, score, 0.0]
        gripper.paint_uniform_color(color)

        # 变换夹爪到抓取位姿
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        gripper.transform(T)
        geometries.append(gripper)

    print(f"\n🎯 显示抓取结果")
    print(f"  绿色夹爪=高分, 红色夹爪=低分")
    print(f"  关闭窗口后退出...")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="GraspGen 6-DOF 抓取位姿结果",
        width=960,
        height=720
    )


def create_gripper_mesh(width=0.08, depth=0.1):
    """创建夹爪网格。"""
    import open3d as o3d

    body = o3d.geometry.TriangleMesh.create_box(
        width=width, height=0.01, depth=depth
    )
    body.translate([-width/2, -0.005, -depth])

    finger1 = o3d.geometry.TriangleMesh.create_box(
        width=0.01, height=0.03, depth=depth
    )
    finger1.translate([-width/2, -0.015, -depth])

    finger2 = o3d.geometry.TriangleMesh.create_box(
        width=0.01, height=0.03, depth=depth
    )
    finger2.translate([width/2 - 0.01, -0.015, -depth])

    return body + finger1 + finger2


def main():
    parser = argparse.ArgumentParser(description="完整流程展示 Pipeline")
    parser.add_argument("--scene", action="store_true", help="使用真实场景点云")
    parser.add_argument("--object", action="store_true", help="使用合成物体点云")
    parser.add_argument("--id", type=str, default=None, help="指定样本 ID")
    parser.add_argument("--no_grasp", action="store_true", help="不运行 GraspGen，仅展示点云")
    parser.add_argument("--multi_view", action="store_true", help="演示多视图点云融合效果")
    args = parser.parse_args()

    print("=" * 80)
    print("🎯 GraspGen 6-DOF 抓取完整流程展示 (6步)")
    print("=" * 80)

    # 选择样本
    if args.scene:
        data_type = "scenes"
        sample_id = args.id or "1745766797_642935"
        print(f"\n📷 使用真实场景: {sample_id}")
    else:
        data_type = "objects"
        sample_id = args.id or "1740787815_545011"
        print(f"\n📦 使用合成物体: {sample_id}")

    data_dir = DATASET_ROOT / data_type / sample_id
    if not data_dir.exists():
        print(f"❌ 样本不存在: {data_dir}")
        return

    # 读取内参
    with open(data_dir / "intrinsics.txt") as f:
        lines = f.readlines()
    params = {}
    for line in lines:
        k, v = line.strip().split('=')
        params[k] = float(v)

    fx, fy = params['fx'], params['fy']
    cx, cy = params['cx'], params['cy']

    # ==============================================
    # 步骤 1: 合并显示 RGB + 深度图 + Mask
    # ==============================================
    print(f"\n{'='*60}")
    print(f"步骤 1/6: 显示 RGB+深度+Mask 输入")
    print(f"{'='*60}")
    rgb = imageio.v2.imread(data_dir / "rgb.png")
    depth_mm = imageio.v2.imread(data_dir / "depth.png")
    mask = imageio.v2.imread(data_dir / "mask.png")

    show_combined_2d(rgb, depth_mm, mask)

    # ==============================================
    # 步骤 2: 生成并显示 完整场景彩色点云（不降采样）
    # ==============================================
    print(f"\n{'='*60}")
    print(f"步骤 2/6: 显示 完整场景彩色点云")
    print(f"{'='*60}")
    pc_scene, colors_scene, valid_scene = depth_to_pointcloud_opengl(
        depth_mm, fx, fy, cx, cy,
        rgb=rgb  # 传入RGB，生成彩色点云
    )
    visualize_3d_pointcloud(
        pc_scene,
        colors=colors_scene,  # 带原始RGB颜色
        title="2. 完整场景彩色点云",
        max_points=500000  # 最多50万个点，几乎保留全部场景
    )

    # ==============================================
    # 步骤 3: 生成并显示 单独物体点云（利用Mask+去噪）
    # ==============================================
    print(f"\n{'='*60}")
    print(f"步骤 3/6: 显示 去噪后的物体点云")
    print(f"{'='*60}")
    pc_object, colors_object, valid_object = depth_to_pointcloud_opengl(
        depth_mm, fx, fy, cx, cy,
        mask=mask
    )

    if len(pc_object) < 100:
        print(f"⚠️ Mask 区域点太少，使用深度范围筛选物体")
        # 如果Mask不好，取深度在0.5-2m的中心区域作为物体
        valid_obj = (depth_mm > 500) & (depth_mm < 2000)
        pc_object, colors_object, valid_object = depth_to_pointcloud_opengl(
            depth_mm, fx, fy, cx, cy, mask=valid_obj, rgb=rgb
        )

    # 新增点云去噪步骤
    print(f"\n⚙️  物体点云去噪:")
    pc_object = denoise_pointcloud(pc_object, std_ratio=2.0)

    print(f"  最终物体点云: {len(pc_object)} 个点")

    visualize_3d_pointcloud(
        pc_object,
        title="3. 去噪后的单独物体点云",
        default_color=[0.1, 0.6, 1.0]  # 蓝色物体
    )

    # ==============================================
    # 可选：多视图点云融合演示
    # ==============================================
    if args.multi_view:
        print(f"\n{'='*60}")
        print(f"可选步骤: 多视图点云融合演示")
        print(f"{'='*60}")
        pc_object = demo_multi_view_fusion(pc_object)

    # ==============================================
    # 步骤 4: 点云预处理降采样并显示
    # ==============================================
    print(f"\n{'='*60}")
    print(f"步骤 4/6: 显示 降采样后的物体点云")
    print(f"{'='*60}")
    pc_processed, center_offset = preprocess_pipeline(pc_object, num_points=1024)

    visualize_3d_pointcloud(
        pc_processed,
        title="4. 降采样后的物体点云（GraspGen输入，1024点）",
        default_color=[0.1, 0.8, 0.3]  # 绿色预处理后点云
    )

    if args.no_grasp:
        print(f"\n✅ 演示完成 (--no_grasp 模式)")
        cv2.destroyAllWindows()
        return

    # ==============================================
    # 步骤 5: GraspGen 推理生成抓取位姿
    # ==============================================
    print(f"\n{'='*60}")
    print(f"步骤 5/6: GraspGen 推理")
    print(f"{'='*60}")
    grasps = run_graspgen_inference(pc_processed)

    # ==============================================
    # 步骤 6: 可视化抓取结果（夹爪+物体）
    # ==============================================
    print(f"\n{'='*60}")
    print(f"步骤 6/6: 显示 抓取结果")
    print(f"{'='*60}")
    visualize_grasp_result(
        pc_object,  # 用原始完整物体点云，显示效果更好
        grasps,
        center_offset=center_offset  # 将抓取位姿从中心化坐标系转换回原物体坐标系
    )

    cv2.destroyAllWindows()
    print(f"\n✅ 完整流程运行完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
