# graspnet_pipeline/ — GraspNet-1Billion RGB-D 推理管线

> [作业新增文件] 非 GraspGen 原始项目文件。

基于 GraspNet-1Billion 数据集的完整推理管线：**RGB-D → 点云 → 推理 → 碰撞检测 → Open3D 可视化**。

## 文件结构

```
graspnet_pipeline/
├── README.md              # 本文件
├── config.py              # 全局配置 (路径、相机内参、预处理参数)
├── model_loader.py        # 模型加载 (官方预训练 / 我们训练的)
├── preprocess.py          # RGB-D → 点云预处理
├── inference.py           # Generator 推理 (已弃用，改用 pretrained + sampler)
├── collision.py           # 碰撞检测
├── visualize.py           # Open3D 三维可视化
├── run_pipeline.py        # 主入口
├── outputs/               # 可视化截图输出
└── plots/                 # 图表输出
```

## 管线概览

```
GraspNet-1Billion RGB-D
       │
       ▼
[preprocess.py]  depth2points() → 相机坐标系点云
                 SOR 离群点剔除 → 去噪
                 FPS 降采样 → 1024 点
       │
       ▼
[model_loader.py]  GraspGenSampler (官方预训练)
                   Generator + Discriminator
       │
       ▼
[collision.py]  filter_colliding_grasps()
                夹爪 mesh → 采样 → 变换 → 最近距离
       │
       ▼
[visualize.py]  Open3D 渲染
                点云 (蓝) + 夹爪线框 (绿=无碰撞 / 红=碰撞)
                多视角截图
```

## 用法

```bash
cd /home/lzl/Projects/6Dpose/GraspGen
source .venv/bin/activate

cd ../graspnet_pipeline

# 使用合成数据测试管线 (验证环境)
python run_pipeline.py --test

# 使用我们训练的模型测试
python run_pipeline.py --test --our_model

# 对 GraspNet RGB-D 图像推理
python run_pipeline.py \
    --rgb GraspNet_datasets/scenes/scene_0000/rgb.png \
    --depth GraspNet_datasets/scenes/scene_0000/depth.png \
    --fx 591.0 --fy 590.6 --cx 322.5 --cy 238.3
```

## 依赖

- GraspGen (导入 grasp_gen 模块)
- open3d (可视化)
- imageio (图像加载)
- torch, numpy, trimesh

## 输出

推理结果截图保存在 `outputs/` 目录:
- `*_front.png` — 正视图
- `*_side.png` — 侧视图
- `*_top.png` — 俯视图
