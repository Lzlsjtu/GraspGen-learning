# 6 自由度抓取位姿估计 — 实现计划

> [作业新增文件] 此文件为 6-DOF 抓取位姿估计课程作业而创建，非 GraspGen 原始项目文件。
> 策略：**不造新轮子，直接跑通 GraspGen 现有代码**。重点放在训练调通、结果分析、原理阐述和失效案例剖析上。

---

## 1. 环境现状

| 项目 | 状态 |
|------|------|
| GPU | RTX 3060 (6 GB VRAM)，CUDA 12.1 |
| Python | 3.10 + PyTorch 2.1.0+cu121（`.venv` 已装好） |
| GraspGen 代码 | `GraspGen/`，可编辑安装（`pip install -e .`） |
| 抓取数据集 | `GraspGen_datasets/grasp_data/`（21 GB，含 franka_panda / robotiq_2f_140 / suction） |
| 物体网格数据集 | `GraspGen_datasets/object_dataset/`（475 个简化 .glb 网格，1.8 GB） |
| 训练切分 | `splits/franka_panda/`：train 7657 / valid 852 个物体 |

数据已经就绪，代码已经就绪，环境已经就绪。**直接开跑即可**。

---

## 2. 总路线图

```
┌─ 已有 GraspGen 代码栈 ────────────────────────────────────────────┐
│                                                                     │
│  ① 数据管线（直接用）                                                │
│     grasp_gen/dataset/dataset.py   → 点云采样 + h5 缓存构建          │
│     grasp_gen/utils/point_cloud_utils.py → 离群点剔除、KNN、碰撞检测  │
│     grasp_gen/dataset/renderer.py  → depth2points（RGB-D→点云）      │
│                                                                     │
│  ② 模型（直接用）                                                    │
│     grasp_gen/models/generator.py  → GraspGenGenerator（扩散模型）    │
│     grasp_gen/models/model_utils.py → PointNetPlusPlus 编码器        │
│     grasp_gen/metrics.py           → GeodesicLoss（旋转测地线损失）   │
│                                                                     │
│  ③ 训练（直接用）                                                    │
│     scripts/train_graspgen.py      → Hydra 训练入口                  │
│     scripts/config.yaml            → 默认配置                        │
│     runs/train_*.sh                → 训练脚本参考                    │
│                                                                     │
│  ④ 推理 + 可视化（直接用）                                           │
│     scripts/inference_graspgen.py  → 核心推理                        │
│     scripts/demo_object_pc.py      → 物体点云可视化                   │
│     scripts/demo_object_mesh.py    → 物体网格可视化                   │
│     scripts/demo_scene_pc.py       → 场景点云 + 碰撞过滤              │
│     grasp_gen/utils/meshcat_utils.py → MeshCat 3D 渲染               │
│                                                                     │
│  ⑤ 评估指标（直接用）                                                │
│     grasp_gen/metrics.py           → translation_L2 + geodesic_error │
│     grasp_gen/utils/point_cloud_utils.py → filter_colliding_grasps   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

需要做的 **唯一新工作**：写一个适配 6GB 显存的训练启动脚本 + 训练完成后跑推理/可视化 + 写报告。

---

## 3. 关键原理（报告核心内容来源）

### 3.1 GraspGen 扩散模型的数学原理

GraspGen 将 6-DOF 抓取位姿生成建模为一个 **去噪扩散过程**（Denoising Diffusion Probabilistic Model, DDPM）。

**前向扩散**：给定一个真实抓取位姿 `g_0`（表示为 9 维向量：3 平移 + 6 旋转 6D 表示），逐步加入高斯噪声：

```
q(g_t | g_{t-1}) = N(g_t; sqrt(1-β_t) * g_{t-1}, β_t * I)
```

经过 T 步后：`g_T ~ N(0, I)`

**逆向去噪**：训练一个神经网络 `ε_θ(g_t, t, pc)` 在给定物体点云 `pc` 的条件下预测噪声 `ε`：

```
L_simple = E_{t, g_0, ε} [||ε - ε_θ(g_t, t, pc)||^2]
```

这是 GraspGen Generator 的核心训练目标。

**推理时**：从 `g_T ~ N(0, I)` 出发，逐步去噪 T 步得到 `g_0`，然后从 6D 旋转表示通过 Gram-Schmidt 正交化恢复旋转矩阵。

### 3.2 点云特征提取原理

GraspGen 支持三种点云编码器（`obs_backbone` 参数）：

| Backbone | 原理 | 内存 |
|----------|------|------|
| `pointnet` | PointNet++ 的 Set Abstraction：在半径邻域内做分组+PointNet，层次化提取局部→全局特征。**对 6GB 显存最友好** | ~2 GB |
| `ptv3` | Point Transformer V3：用稀疏窗口注意力替代 PointNet 的 max pooling，表达能力更强 | ~5 GB |
| `vit` | 将点云投影到多视图深度图后用 ViT 编码 | 较大 |

### 3.3 位姿表示与损失函数

**6D 旋转表示**（`r3_6d`）：取旋转矩阵的前两列 `[R_{:,0}, R_{:,1}]` 展平为 6 维向量。相比四元数不存在 double cover 问题，相比欧拉角不存在万向节锁，比 `so(3)` 轴角更连续。

GraspGen 的联合损失（`generator.py` 的 `compute_grasp_loss`）：

```
L_total = L_diffusion (噪声预测 MSE)
        + L_pointmatching (预测抓取点与 GT 抓取点的 Chamfer 距离)
        + L_l1_pos (平移 L1 损失，可选)
        + L_l1_rot (旋转 L1 损失，可选)
```

### 3.4 碰撞检测原理

`filter_colliding_grasps`（`point_cloud_utils.py:237`）：
1. 在夹爪碰撞网格表面采样 N 个点
2. 将采样点变换到预测抓取位姿下
3. 计算每个采样点到场景点云的最近距离
4. 如果任一采样点与场景点云距离 < 阈值（默认 2mm），判定为碰撞

---

## 4. 执行步骤

### 步骤 1：适配 6GB 显存的训练脚本

GPU 限制：RTX 3060 只有 **6 GB** 显存。必须降低以下参数：

| 参数 | 默认值（需要 A100） | RTX 3060 适配值 | 说明 |
|------|---------------------|-----------------|------|
| `num_points` | 16384 | **1024** | 物体点云采样点数 |
| `batch_size` | 8 | **2** | 每批次物体数 |
| `num_grasps_per_object` | -1 (全部) | **100** | 每个物体的抓取标注数 |
| `redundancy` | 1 | **1** | 视角冗余度（不增加） |
| `obs_backbone` | ptv3 | **pointnet** | 编码器骨干（pointnet 最省显存） |
| `num_workers` | 8 | **1** | 数据加载线程 |
| `num_diffusion_iters` | 100 | **10** | 扩散步数（影响训练速度和精度） |

创建一个最小可跑通的训练脚本：

```bash
#!/usr/bin/env bash
# 文件名: runs/train_assignment_gen.sh

cd /home/lzl/Projects/6Dpose/GraspGen

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

GRASP_DATASET="/home/lzl/Projects/6Dpose/GraspGen_datasets/grasp_data"
OBJECT_DATASET="/home/lzl/Projects/6Dpose/GraspGen_datasets/object_dataset"
RESULTS="/home/lzl/Projects/6Dpose/GraspGen/runs/results_assignment"

mkdir -p "$RESULTS/logs"
mkdir -p "$RESULTS/cache"

cd scripts

python train_graspgen.py \
    data.root_dir="$GRASP_DATASET/splits/franka_panda" \
    data.object_root_dir="$OBJECT_DATASET" \
    data.grasp_root_dir="$GRASP_DATASET/grasp_data/franka_panda" \
    data.gripper_name="franka_panda" \
    data.num_points=1024 \
    data.num_grasps_per_object=100 \
    data.prob_point_cloud=0.5 \
    data.redundancy=1 \
    data.dataset_name="objaverse" \
    data.dataset_version="v2" \
    data.dataset_cls="ObjectPickDataset" \
    data.cache_dir="$RESULTS/cache" \
    data.rotation_augmentation=True \
    data.load_contact=False \
    train.log_dir="$RESULTS/logs" \
    train.batch_size=2 \
    train.num_epochs=200 \
    train.num_workers=1 \
    train.print_freq=5 \
    train.save_freq=20 \
    train.eval_freq=20 \
    train.model_name="diffusion" \
    train.checkpoint="" \
    diffusion.gripper_name="franka_panda" \
    diffusion.num_diffusion_iters=10 \
    diffusion.num_diffusion_iters_eval=10 \
    diffusion.obs_backbone="pointnet" \
    diffusion.grasp_repr="r3_so3" \
    diffusion.compositional_schedular=True \
    diffusion.loss_pointmatching=True \
    diffusion.attention="cat" \
    optimizer.type="ADAMW" \
    optimizer.lr=0.00001 \
    optimizer.grad_clip=-1 \
    | tee "$RESULTS/logs/console_log.txt"
```

**预期时间**：
- 缓存构建（首次运行）：~30 分钟（处理 7657 个物体 × 100 grasps）
- 训练 200 epoch：RTX 3060 上约 **8-12 小时**

### 步骤 2：训练监控与分析

```bash
# 查看 TensorBoard
tensorboard --logdir=/home/lzl/Projects/6Dpose/GraspGen/runs/results_assignment/logs

# 关键指标（生成器，diffusion model）
# ──────────────────────────────────────
# train/loss/all_loss           → 总损失，应持续下降
# train/loss/diffusion_loss     → 扩散噪声预测损失 → 先快后慢收敛
# train/loss/pointmatching_loss → 抓取点匹配 Chamfer 距离
# valid/metric/reconstruction/error_trans_l2  → ★ 平移 L2 误差（目标: < 5cm）
# valid/metric/reconstruction/error_rot_geodesic → ★ 旋转测地线误差（目标: < 0.3 rad）
```

### 步骤 3：推理与可视化

训练完成后，用 checkpoint 跑推理和可视化：

```bash
# 方式 1: 物体网格 → 预测抓取 → MeshCat 3D 可视化
python scripts/demo_object_mesh.py \
    --mesh_file /home/lzl/Projects/6Dpose/GraspGen/assets/objects/box.obj \
    --mesh_scale 1.0 \
    --gripper_config /home/lzl/Projects/6Dpose/GraspGen/runs/results_assignment/logs/config.yaml \
    --num_grasps 50

# 方式 2: 物体点云 → 预测抓取 → MeshCat 3D 可视化
python scripts/demo_object_pc.py \
    --sample_data_dir /home/lzl/Projects/6Dpose/GraspGen/GraspGenModels/sample_data/real_object_pc \
    --gripper_config /home/lzl/Projects/6Dpose/GraspGen/runs/results_assignment/logs/config.yaml

# 方式 3: 场景点云 + 碰撞过滤
python scripts/demo_scene_pc.py \
    --filter_collisions \
    --sample_data_dir /home/lzl/Projects/6Dpose/GraspGen/GraspGenModels/sample_data/real_scene_pc \
    --gripper_config /home/lzl/Projects/6Dpose/GraspGen/runs/results_assignment/logs/config.yaml
```

### 步骤 4：定量评估

写一个简单评估脚本（调用 GraspGen 已有 API）：

```python
# eval/eval_assignment.py
import torch
from grasp_gen.models.grasp_gen import GraspGenGenerator
from grasp_gen.metrics import compute_metrics_given_two_sets_of_poses
from grasp_gen.robot import get_gripper_info

# 1. 加载模型
model = GraspGenGenerator.from_config(cfg)
model.load_state_dict(torch.load(ckpt_path)["model"])
model.eval()

# 2. 逐 batch 评估
for batch in valid_loader:
    with torch.no_grad():
        pred_poses, gt_poses = model(batch, eval=True)

    # 3. 计算指标
    metrics = compute_metrics_given_two_sets_of_poses(
        pred_poses, gt_poses, gripper_info
    )
    # → error_trans_l2 (cm), error_rot_geodesic (rad), recall@2cm

# 4. 碰撞率
from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
collision_free_mask = filter_colliding_grasps(scene_pc, pred_poses, gripper_mesh)
```

---

## 5. 得分点拆解

### 5.1 完整度 30 分 — 对应关系

| 作业要求 | GraspGen 已有实现 | 如何体现 |
|----------|-------------------|----------|
| RGB-D → 点云预处理 | `renderer.py:depth2points`, `point_cloud_utils.py:depth_and_segmentation_to_point_clouds` | 报告中截图 + 代码引用 |
| 点云降采样 | `torch_cluster.fps` 在 `dataset.py` 的 collate 中 | 报告阐述 FPS 原理 |
| 离群点剔除 | `point_cloud_utils.py:point_cloud_outlier_removal` (KNN 距离法) | 报告对比剔除前后的点云 |
| 3D 网络结构（完整可运行） | `generator.py` (PointNet++ 编码器 + Diffusion Head) | 报告中画架构图 |
| 几何干涉检验 | `point_cloud_utils.py:filter_colliding_grasps` | 可视化截图展示碰撞/无碰撞 |

### 5.2 正确性 30 分 — 对应关系

| 作业要求 | 评估方式 |
|----------|----------|
| 正确输出 6-DOF 抓取位姿 | 跑 `demo_object_mesh.py`，截图 3D 可视化 |
| 离线的几何防碰撞评估逻辑严谨准确 | 对比碰撞阈值（1mm/3mm/5mm）下的过滤率 |

### 5.3 深度 30 分 — 关键分析点

| 分析主题 | 要点 | 来源 |
|----------|------|------|
| **三维坐标变换原理** | 相机坐标系 → 物体坐标系 → 夹爪基座标系的变换链；6D 旋转表示 → SO(3) 的 Gram-Schmidt 映射 | `rotation_conversions.py` |
| **点云特征机理** | PointNet++ 的层次化 Set Abstraction：如何从局部几何（半径邻域内 MLP + MaxPool）聚合为全局语义 | `pointnet2_modules.py` 和 Zhou 论文 |
| **深度缺失导致失效** | 局部点云只看到物体正面 → 模型在背面错误生成抓取 → 碰撞或不可达 | 用 `demo_scene_pc.py` 可视化 |
| **硬件噪声导致失效** | 深度传感器噪声 → 点云边缘抖动 → 抓取位姿微小偏差 → 真实环境中夹爪滑动 | 理论分析 + 引用文献 |

### 5.4 创新 — 附加 20 分

可测试以下变体（不需要改代码，只需改 Hydra 参数，然后做对比实验）：

1. **`prob_point_cloud` 的影响**：对比 `prob_point_cloud=0.0`（完全点云）vs `0.5`（50%部分点云）vs `1.0`（始终部分点云）的训练结果 → 证明部分点云训练对 sim2real 的必要性
2. **点云点数的影响**：对比 `num_points=512` vs `1024` vs `2048` → 分析点数对精度和显存的 trade-off
3. **Backbone 对比**：`pointnet` vs `ptv3` → 分析 Transformer 和 PointNet 在抓取任务上的特征提取差异

---

## 6. 报告结构

```
3000 字报告大纲：

1. 引言 (400 字)
   - 6-DOF 抓取估计的问题定义
   - 局部点云 vs 完整点云的挑战
   - 论文背景 (GraspNet / PointNetGPD / Contact-GraspNet)

2. 方法 (800 字)
   2.1 点云预处理
       - 相机模型: (u,v,d) → (X,Y,Z) = ((u-cx)*d/fx, (v-cy)*d/fy, d)
       - SOR 离群点剔除: KNN 距离统计 + 阈值
       - FPS 均匀降采样: 保证空间覆盖均匀性
   2.2 网络结构
       - PointNet++ 特征编码: 多尺度 Set Abstraction 原理
       - 扩散去噪: DDPM 从随机噪声逐步回归抓取位姿
       - 6D 连续旋转表示的优势 (vs 四元数/欧拉角/axis-angle)
   2.3 损失函数
       - 噪声预测 MSE + 点匹配 Chamfer + 平移/旋转 L1
       - 联合训练的权重设计

3. 实验 (800 字)
   3.1 实验设置
       - 数据集: franka_panda, 7657 train / 852 valid
       - 训练参数: 6GB 适配后的配置表格
   3.2 定量结果
       - translation_error_l2 和 rotation_geodesic 表格
       - collision_free_rate
       - 消融实验 (prob_point_cloud / num_points 对比)
   3.3 定性结果
       - 可视化截图 (物体点云 + 预测抓取 + 夹爪模型)
       - 碰撞/无碰撞对比图

4. 失效案例深度分析 (600 字)
   4.1 深度缺失场景
       - 单视角导致物体背面的点云完全缺失
       - 模型在缺失区域生成抓取 → 真实场景中的碰撞
   4.2 边缘噪声
       - 深度不连续处 (物体边缘) 的 flying pixels
       - 对点云法线估计和局部特征的影响
   4.3 对称性歧义
       - 对称物体 (圆柱/球体) 的抓取旋转不唯一
       - 损失函数中对旋转的惩罚导致模型输出模糊均值

5. 结论 (400 字)
   - 总结: 扩散模型 + PointNet++ 可有效处理局部点云的 6-DOF 抓取
   - 局限: 依赖 instance segmentation, 未见物体泛化
   - 展望: on-generator discriminator 可进一步提升抓取质量
```

---

## 7. 提交清单

```
旋转检测-张三李四.zip
├── code/                          # GraspGen 代码的副本（或 git patch）
│   └── runs/train_assignment_gen.sh  # ★ 唯一新增：适配 6GB 的训练脚本
├── README.md                      # 环境配置 + 运行说明
│   ├── 如何安装依赖
│   ├── 如何运行训练
│   └── 如何运行推理/可视化
├── report.pdf                     # ~3000 字报告
├── slides.pptx                   # 答辩 PPT
├── demo_video.mp4                 # 推理 + 可视化录屏
├── results/                       # 训练产物
│   ├── console_log.txt            # 训练日志
│   ├── tensorboard_screenshot.png # 训练曲线截图
│   └── visualization_screenshots/ # 可视化截图
└── division_of_labor.md           # 分工说明
```

---

## 8. 第一步：现在就可以跑

GraspGen 的数据、代码、环境全部就绪。唯一需要的是创建一个适配 6GB 显存的训练启动脚本并执行。建议：

```bash
cd /home/lzl/Projects/6Dpose/GraspGen
source .venv/bin/activate

# 先跑一个快速测试（debug 模式，单 GPU，验证代码能跑通）
cd scripts
python train_graspgen.py \
    train.debug=True \
    data.root_dir="/home/lzl/Projects/6Dpose/GraspGen_datasets/grasp_data/splits/franka_panda" \
    data.object_root_dir="/home/lzl/Projects/6Dpose/GraspGen_datasets/object_dataset" \
    data.grasp_root_dir="/home/lzl/Projects/6Dpose/GraspGen_datasets/grasp_data/grasp_data/franka_panda" \
    data.gripper_name="franka_panda" \
    data.num_points=1024 \
    data.dataset_name="objaverse" \
    data.dataset_version="v2" \
    data.dataset_cls="ObjectPickDataset" \
    train.batch_size=2 \
    train.num_workers=0 \
    train.num_epochs=5 \
    train.checkpoint="" \
    train.model_name="diffusion" \
    diffusion.num_diffusion_iters=10 \
    diffusion.num_diffusion_iters_eval=10 \
    diffusion.obs_backbone="pointnet" \
    diffusion.grasp_repr="r3_so3" \
    diffusion.compositional_schedular=True
```

这会先用 5 个 epoch 验证 pipeline 是否通畅（约 10 分钟），确认 OK 后再跑完整 200 epoch 的训练。

---

## 9. 神经网络学习过程可视化 ★

> 这是作业"深度"得分点的核心支撑：不仅要跑通训练，还要展示网络**学到了什么、如何学到、何时学到**。

### 9.1 可视化总览

```
┌──────────────────────────────────────────────────────────────┐
│                    三层可视化体系                              │
│                                                              │
│  第一层：权重空间演化 (跨 epoch 对比)                          │
│    ├── 各层权重分布直方图 (epoch 0→10→50→100→200)             │
│    ├── 梯度范数变化曲线 (per-layer gradient norm)             │
│    ├── 权重更新余弦相似度 (相邻 epoch 间)                      │
│    └── 权重矩阵有效秩 (SVD singular value decay)              │
│                                                              │
│  第二层：前向激活追踪 (固定测试输入，逐层捕获)                   │
│    ├── SA Module 1/2/3 FPS 关键点 3D 可视化                   │
│    ├── SA Module 特征激活在点云上的热力图                       │
│    ├── Object Embedding 的 PCA/t-SNE 降维                     │
│    └── Diffusion Head 各层特征统计                             │
│                                                              │
│  第三层：输入归因分析 (哪些点对抓取预测最重要)                   │
│    ├── 逐点扰动 → 预测变化量热力图                              │
│    ├── 梯度反向传播 → 输入点显著性                             │
│    └── 局部区域遮挡实验                                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 9.2 模型关键层梳理

理解 GraspGen Generator (`generator.py`) 的内部结构，确定在哪些层插入 Hook：

```
输入点云 (B, 1024, 3)
        │
  ┌─────▼────────────────────────────────────────────┐
  │  PointNet++ Encoder (object_encoder)              │
  │  ─────────────────────────────────────            │
  │  SA Module 1 (npoint=512, radius=0.1)             │  ← Hook ① FPS keypoints + 输出特征
  │    ├── FPS: 选 512 个关键点                       │     3D 散点图: 哪些点被选为关键点
  │    ├── QueryAndGroup (nsample=32)                  │     邻域分组半径可视化
  │    ├── Shared MLP [3→32→32→64]                    │
  │    └── MaxPool → (B, 64, 512)                     │
  │                                                    │
  │  SA Module 2 (npoint=128, radius=0.2)             │  ← Hook ②
  │    ├── FPS: 选 128 个关键点                       │     感受野扩大: 从局部到半局部
  │    ├── Shared MLP [64→64→128]                     │
  │    └── MaxPool → (B, 128, 128)                    │
  │                                                    │
  │  SA Module 3 (radius=0.4, GroupAll)               │  ← Hook ③ 全局聚合
  │    ├── Shared MLP [128→256→512]                   │     最关键: 将 128 个半局部特征
  │    └── MaxPool → (B, 512)                         │     聚合为单一 512 维全局向量
  │                                                    │
  │  输出: object_embedding (B, 512)                   │  ← Hook ④ 全局特征向量
  └────────────────────────────────────────────────────┘
        │
        │  mask_batch 扩展: (B, 512) → (B×K, 512)
        ▼
  ┌─────▼────────────────────────────────────────────┐
  │  Diffusion Head (diffusion_head)                   │
  │  ─────────────────────────────────────             │
  │  Timestep Encoder:                                 │  ← Hook ⑤
  │    SinusoidalPosEmb(t) → MLP → (B×K, 512)         │     时间步的正弦编码
  │                                                    │
  │  Sample Encoder:                                   │  ← Hook ⑥
  │    noisy_grasp → MLP → (B×K, 512)                 │     噪声抓取的编码
  │                                                    │
  │  特征拼接: [sample, timestep, observation]         │  ← Hook ⑦
  │    → (B×K, 512+512+512) = (B×K, 1536)            │     三路信息的融合
  │                                                    │
  │  Prediction Head:                                  │
  │    Linear(1536→768) → ReLU                         │  ← Hook ⑧
  │    Linear(768→384) → ReLU                          │  ← Hook ⑨
  │    Linear(384→6 或 9) → noise_pred                 │  ← Hook ⑩ 输出
  └────────────────────────────────────────────────────┘
```

### 9.3 实现方案：Hook 注册脚本

在 GraspGen 的 `scripts/` 下新建一个分析脚本（不修改 GraspGen 核心代码）：

```python
# scripts/analyze_learning.py
"""
神经网络学习过程可视化工具。

功能:
  1. 加载不同 epoch 的 checkpoint，对比权重分布
  2. 对固定输入注册 forward hook，捕获中间层激活
  3. 生成 3D 可视化 (Open3D) 和统计图表 (matplotlib)

用法:
  python scripts/analyze_learning.py \
      --checkpoint_dir ./runs/results_assignment/logs \
      --input_mesh assets/objects/box.obj \
      --output_dir ./runs/analysis_output
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ============================================================
# Part 1: 权重空间分析
# ============================================================

def analyze_weight_evolution(checkpoint_paths):
    """对比不同 epoch 的模型权重变化。

    对每一层计算:
      - weight_mean, weight_std (权重分布的中心和离散度)
      - gradient_norm (如果有保存优化器状态)
      - weight_update_cosine (相邻 epoch 间权重变化的方向一致性)

    可视化:
      - 权重直方图 (epoch 0, 50, 100, 200 四张子图并列)
      - 每层 L2 范数随 epoch 变化曲线
      - 有效秩 (SVD) 随 epoch 变化 → 展示权重矩阵的低秩结构形成
    """
    snapshots = {}
    for epoch, ckpt_path in checkpoint_paths.items():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        snapshots[epoch] = ckpt["model"]

    # 按层统计
    layer_stats = defaultdict(lambda: {"norm": [], "mean": [], "std": []})

    for epoch, state_dict in snapshots.items():
        for name, param in state_dict.items():
            if "weight" in name and param.dim() >= 2:
                w = param.float()
                layer_stats[name]["norm"].append(torch.norm(w).item())
                layer_stats[name]["mean"].append(w.mean().item())
                layer_stats[name]["std"].append(w.std().item())

    # --- 图 1: 权重范数变化 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = list(checkpoint_paths.keys())
    for name, stats in layer_stats.items():
        ax.plot(epochs, stats["norm"], label=name[:60], alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight L2 Norm")
    ax.set_title("Weight Norm Evolution Across Layers")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # --- 图 2: SVD 有效秩分析 ---
    # 选取关键层 (SA Module MLP, Prediction Head)
    key_layers = [n for n in layer_stats if any(
        k in n for k in ["mlp", "prediction_head", "sample_encoder"])]
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    for ax_i, name in enumerate(key_layers[:6]):
        ax = axes2[ax_i // 3][ax_i % 3]
        for epoch, state_dict in snapshots.items():
            if name in state_dict:
                w = state_dict[name].float()
                if w.dim() == 2:
                    s = torch.linalg.svdvals(w).numpy()
                    s_norm = s / s[0]  # 归一化
                    ax.semilogy(s_norm, alpha=0.5, label=f"ep{epoch}")
        ax.set_title(name.split(".")[-1][:40], fontsize=8)
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Normalized Value (log)")
        ax.grid(True, alpha=0.3)
    fig2.suptitle("SVD Singular Value Decay (Effective Rank Analysis)")
    fig2.tight_layout()

    return fig, fig2, layer_stats


# ============================================================
# Part 2: 前向激活捕获
# ============================================================

class ActivationRecorder:
    """通过 PyTorch forward hook 捕获中间层激活值。

    使用方式:
        recorder = ActivationRecorder(model)
        recorder.register_hooks()      # 注册 Hook
        output = model(data)           # 前向传播
        activations = recorder.saved   # 获取捕获的激活
        recorder.remove_hooks()        # 清理
    """

    def __init__(self, model):
        self.model = model
        self.saved = {}
        self.handles = []

    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.saved[name] = output[0].detach().cpu()
            else:
                self.saved[name] = output.detach().cpu()
        return hook

    def register_hooks(self):
        """在代表性层注册 forward hook。

        选择标准:
          - SA Module 输出: 观察层次化特征聚合
          - Object Embedding: 观察全局特征表示
          - Diffusion Head 中间层: 观察去噪信息流
        """
        # --- PointNet++ Encoder 内部 ---
        # 注意: PointNetPlusPlus 内部是 nn.ModuleList of PointnetSAModule
        obj_enc = self.model.object_encoder
        if hasattr(obj_enc, 'sa_modules'):  # PointnetSAModule 列表
            for i, sa in enumerate(obj_enc.sa_modules):
                # SA Module 的 mlps 和 groupers 已经在 forward 内部
                self.handles.append(
                    sa.register_forward_hook(self._hook_fn(f"SA_Module_{i}"))
                )

        # Object Encoder 整体输出 (global feature)
        self.handles.append(
            obj_enc.register_forward_hook(self._hook_fn("Object_Embedding"))
        )

        # --- Diffusion Head 内部 ---
        dh = self.model.diffusion_head
        # Timestep Encoder 输出
        self.handles.append(
            dh.diffusion_step_encoder.register_forward_hook(
                self._hook_fn("Timestep_Embedding"))
        )
        # Sample Encoder 输出
        if hasattr(dh, 'sample_encoder'):
            self.handles.append(
                dh.sample_encoder.register_forward_hook(
                    self._hook_fn("Sample_Embedding"))
            )
        # Prediction Head 各层
        for i, layer in enumerate(dh.prediction_head):
            if isinstance(layer, nn.Linear):
                self.handles.append(
                    layer.register_forward_hook(
                        self._hook_fn(f"PredHead_L{i}"))
                )

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def analyze_forward_flow(model, fixed_input, checkpoint_paths, output_dir):
    """对固定输入追踪信息流。

    对每个 epoch 的 checkpoint:
      1. 加载模型
      2. 注册 ActivationRecorder
      3. 前向传播一次
      4. 保存各层激活值到 .npy
      5. 生成可视化

    Args:
        model: 未加载权重的模型实例
        fixed_input: dict, 包含 "points" (B, N, 3) 的固定输入
        checkpoint_paths: {epoch: path}
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    all_embeddings = {}  # {epoch: object_embedding}
    sa_activations = defaultdict(list)  # {epoch: [SA0, SA1, SA2]}

    for epoch, ckpt_path in checkpoint_paths.items():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()

        recorder = ActivationRecorder(model)
        recorder.register_hooks()

        with torch.no_grad():
            _ = model(fixed_input, eval=True)

        # 保存激活值
        np.savez(
            os.path.join(output_dir, f"activations_epoch_{epoch}.npz"),
            **{k: v.numpy() for k, v in recorder.saved.items()}
        )

        # 收集 Object Embedding 用于后续 PCA
        if "Object_Embedding" in recorder.saved:
            all_embeddings[epoch] = recorder.saved["Object_Embedding"]

        # 收集 SA Module 输出用于热力图
        for k, v in recorder.saved.items():
            if k.startswith("SA_Module"):
                sa_activations[k].append((epoch, v))

        recorder.remove_hooks()

    # --- 图 3: Object Embedding PCA 演化 ---
    fig3, ax3 = visualize_embedding_evolution(all_embeddings)

    # --- 图 4: 各层激活值统计 (箱线图) ---
    fig4, ax4 = visualize_layer_statistics(output_dir, checkpoint_paths.keys())

    return fig3, fig4


def visualize_embedding_evolution(all_embeddings):
    """对同一物体在不同 epoch 下的全局特征做 PCA 可视化。

    解读:
      - 若点在不同 epoch 间漂移很大 → 特征空间在快速变化 (早期学习)
      - 若点逐渐收敛到一个邻域 → 特征学习趋于稳定 (后期微调)
      - 可同时放入多个不同物体的 embedding → 观察类间分离
    """
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 子图 1: PCA 散点图 (只画第一个物体的 embedding)
    epochs = sorted(all_embeddings.keys())
    all_embs = torch.cat([all_embeddings[e][:1] for e in epochs])  # (N_epochs, 512)
    pca = PCA(n_components=2)
    embs_2d = pca.fit_transform(all_embs.numpy())

    sc = axes[0].scatter(embs_2d[:, 0], embs_2d[:, 1], c=epochs, cmap="viridis", s=60)
    for i, e in enumerate(epochs):
        axes[0].annotate(str(e), (embs_2d[i, 0], embs_2d[i, 1]),
                         fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set_title("Object Embedding PCA Trajectory Across Epochs")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    fig.colorbar(sc, ax=axes[0], label="Epoch")

    # 子图 2: Embedding 各维度方差变化
    all_embs_mat = torch.stack([all_embeddings[e][0] for e in epochs])  # (N_epochs, 512)
    axes[1].plot(epochs, all_embs_mat.std(dim=1).numpy(), "b-", linewidth=2)
    axes[1].fill_between(epochs, 0, all_embs_mat.std(dim=1).numpy(), alpha=0.2)
    axes[1].set_title("Embedding Activation Sparsity Over Time")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Std Dev Across 512 Feature Dimensions")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Global Feature Space Evolution")
    fig.tight_layout()
    return fig, axes


def visualize_layer_statistics(output_dir, epochs):
    """绘制各层激活值的均值和标准差箱线图。

    展示每层的激活分布如何随训练变化:
      - 如果某层激活方差 → 0 → 该层可能已"饱和"不再学习
      - 如果某层激活均值持续偏移 → 该层仍在适应
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    layer_names = None
    means_by_layer = defaultdict(list)  # {layer: [mean_per_epoch]}
    stds_by_layer = defaultdict(list)

    for epoch in epochs:
        data = np.load(os.path.join(output_dir, f"activations_epoch_{epoch}.npz"))
        if layer_names is None:
            layer_names = list(data.keys())
        for name in layer_names:
            arr = data[name]
            means_by_layer[name].append(arr.mean())
            stds_by_layer[name].append(arr.std())

    x = list(epochs)
    for name in layer_names:
        axes[0].plot(x, means_by_layer[name], "o-", label=name[:30], markersize=4)
        axes[1].plot(x, stds_by_layer[name], "o-", label=name[:30], markersize=4)

    axes[0].set_title("Layer Activation Mean vs Epoch")
    axes[0].set_ylabel("Mean Activation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))

    axes[1].set_title("Layer Activation Std Dev vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Std Dev Activation")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Per-Layer Activation Statistics Evolution")
    fig.tight_layout()
    return fig, axes


# ============================================================
# Part 3: SA Module 3D 可视化 (Open3D)
# ============================================================

def visualize_sa_keypoints_3d(model, input_points, epoch, output_dir):
    """
    对 SA Module 的 FPS 关键点选择过程进行 3D 可视化。

    每一层 SA 输出:
      - 该层 FPS 选出的关键点 (红色大球)
      - 被丢弃的点 (蓝色小球)
      - 按激活值大小给关键点着色 (颜色越亮 = 激活值越大)

    这能直观展示 PointNet++ 如何从 1024 点→512→128 逐步聚焦
    到物体的关键几何区域。
    """
    import open3d as o3d

    recorder = ActivationRecorder(model)
    recorder.register_hooks()

    with torch.no_grad():
        model({"points": input_points}, eval=True)

    # 创建 3 个子图 (并排)，每个显示一层 SA 的输出
    geometries = []

    # 原始点云 (蓝色)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(input_points[0].cpu().numpy())
    pcd.paint_uniform_color([0.4, 0.6, 1.0])  # 浅蓝
    geometries.append({"name": "Input Points (1024)", "geometry": pcd})

    # 每层 SA 输出
    colors = [[1, 0.3, 0.3], [1, 0.6, 0.2], [0.2, 1, 0.2]]  # 红→橙→绿
    for i, color in enumerate(colors):
        key = f"SA_Module_{i}"
        if key in recorder.saved:
            feat = recorder.saved[key]  # (B, C, N_kp)
            # 特征的 channel-mean 作为激活强度
            activation = feat[0].mean(dim=0).numpy()  # (N_kp,)
            activation_norm = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
            # 此时没有坐标信息，可以从原始点云中采样 N_kp 个点做近似

    recorder.remove_hooks()

    # 保存 HTML 截图或直接交互显示
    # ...


# ============================================================
# Part 4: 输入归因分析
# ============================================================

def input_attribution_analysis(model, input_points, output_dir):
    """
    分析输入点云中哪些点对抓取预测贡献最大。

    方法 1: 逐点梯度显著性 (Gradient × Input)
      - 计算输出噪声预测对每个输入点坐标的梯度
      - |∂L/∂x_i| 越大 → 该点越重要

    方法 2: 逐点扰动 (Occlusion)
      - 用球体逐区域遮挡点云 → 观察预测变化
      - 预测变化大的区域 → 关键抓取区域

    方法 3: 特征通道归因
      - 对 Object Embedding 的 512 维做聚类
      - 每个簇对应一种"几何语义" (如平面、边缘、曲面)
      - 反向追踪哪些输入点激活了该语义
    """
    model.eval()
    input_points = input_points.clone().detach().requires_grad_(True)

    # 前向传播
    depth = input_points
    object_embedding = model.object_encoder(depth)
    # ... 简化: 直接算 object_embedding 的范数对输入点的梯度

    embedding_norm = object_embedding.norm(p=2, dim=-1).sum()
    embedding_norm.backward()

    # 每个输入点的显著性 = 梯度 L2 范数
    saliency = input_points.grad.norm(dim=-1)  # (B, N)
    saliency = saliency[0].detach().cpu().numpy()

    # --- 可视化: 点云着色 (红=高重要性, 蓝=低重要性) ---
    import open3d as o3d
    from matplotlib import cm

    points_np = input_points[0].detach().cpu().numpy()
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    colors = cm.jet(saliency_norm)[:, :3]  # jet colormap

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(
        os.path.join(output_dir, "saliency_point_cloud.ply"), pcd)

    # --- 生成俯视图/正视图/侧视图 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = [("XY", [0, 1]), ("XZ", [0, 2]), ("YZ", [1, 2])]
    for ax, (title, dims) in zip(axes, views):
        scatter = ax.scatter(
            points_np[:, dims[0]], points_np[:, dims[1]],
            c=saliency_norm, cmap="jet", s=8, alpha=0.7
        )
        ax.set_title(f"Input Saliency - {title} View")
        ax.set_aspect("equal")
        plt.colorbar(scatter, ax=ax, label="Importance")
    fig.suptitle("Per-Point Gradient Saliency Analysis")
    fig.tight_layout()

    return fig, pcd


# ============================================================
# Part 5: 主函数
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--input_mesh", type=str, default="assets/objects/box.obj")
    parser.add_argument("--output_dir", type=str, default="./runs/analysis_output")
    args = parser.parse_args()

    # 1. 找到所有 checkpoint
    ckpt_dir = Path(args.checkpoint_dir)
    checkpoint_paths = {}
    for ckpt_file in sorted(ckpt_dir.glob("epoch_*.pth")):
        epoch = int(ckpt_file.stem.split("_")[1])
        checkpoint_paths[epoch] = ckpt_file

    if not checkpoint_paths:
        print("No checkpoints found!")
        return

    print(f"Found checkpoints at epochs: {list(checkpoint_paths.keys())}")

    # 2. 权重空间分析
    print("\n=== Weight Evolution Analysis ===")
    fig_w1, fig_w2, layer_stats = analyze_weight_evolution(checkpoint_paths)
    fig_w1.savefig(os.path.join(args.output_dir, "weight_norm_evolution.png"), dpi=150)
    fig_w2.savefig(os.path.join(args.output_dir, "svd_effective_rank.png"), dpi=150)

    # 3. 构建固定输入
    from grasp_gen.dataset.dataset import load_object_grasp_data
    from grasp_gen.models.grasp_gen import GraspGenGenerator
    from omegaconf import OmegaConf

    # 加载配置
    cfg_path = os.path.join(args.checkpoint_dir, "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    # 实例化模型
    model = GraspGenGenerator.from_config(cfg.diffusion)

    # 准备一个固定测试输入
    # (从 valid set 选第一个物体)
    # ... 简化: 用随机点云测试
    fixed_pc = torch.randn(1, 1024, 3)  # 占位, 实际应从数据集加载
    fixed_input = {"points": fixed_pc}

    # 4. 前向激活分析
    print("\n=== Forward Activation Analysis ===")
    fig_f1, fig_f2 = analyze_forward_flow(
        model, fixed_input, checkpoint_paths, args.output_dir)
    fig_f1.savefig(os.path.join(args.output_dir, "embedding_pca.png"), dpi=150)
    fig_f2.savefig(os.path.join(args.output_dir, "layer_statistics.png"), dpi=150)

    # 5. 输入归因分析
    print("\n=== Input Attribution Analysis ===")
    fig_att, pcd = input_attribution_analysis(model, fixed_pc, args.output_dir)
    fig_att.savefig(os.path.join(args.output_dir, "input_saliency.png"), dpi=150)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
```

### 9.4 可视化产出物清单

| 编号 | 图表 | 类型 | 论文用途 |
|------|------|------|----------|
| Fig W1 | 各层权重 L2 范数 vs Epoch | 折线图 | 展示不同层的收敛速度差异 |
| Fig W2 | 权重矩阵 SVD 奇异值衰减 | 半对数图 | 证明权重矩阵学习到低秩结构 |
| Fig F1 | Object Embedding PCA 轨迹 | 散点图 | 可视化特征空间随训练的演化 |
| Fig F2 | 各层激活统计 vs Epoch | 折线图 | 展示哪些层先饱和、哪些仍在学习 |
| Fig F3 | SA Module FPS 关键点 3D 渲染 | Open3D 截图 | 展示网络如何聚焦物体关键区域 |
| Fig F4 | 输入点显著性热力图 | 3D 点云着色 | 证明网络关注抓取接触点附近 |
| Fig F5 | 降噪过程逐步可视化 | 动画帧序列 | 展示扩散模型从噪声到抓取的生成过程 |

### 9.5 实际运行步骤

```bash
cd /home/lzl/Projects/6Dpose/GraspGen
source .venv/bin/activate

# 训练期间就会自动保存 checkpoint:
#   epoch_20.pth, epoch_40.pth, ..., epoch_200.pth, last.pth
# 保存路径: runs/results_assignment/logs/

# 训练完成后, 对 checkpoint 做分析:
python scripts/analyze_learning.py \
    --checkpoint_dir ./runs/results_assignment/logs \
    --input_mesh assets/objects/box.obj \
    --output_dir ./runs/analysis_output

# 产出:
#   runs/analysis_output/
#     ├── weight_norm_evolution.png      # 权重演化
#     ├── svd_effective_rank.png          # SVD 分析
#     ├── embedding_pca.png               # 特征嵌入 PCA
#     ├── layer_statistics.png            # 层激活统计
#     ├── saliency_point_cloud.ply        # 输入显著性 3D 模型
#     ├── input_saliency.png              # 输入显著性多视图
#     └── activations_epoch_*.npz         # 各 epoch 中间层激活原始数据
```

### 9.6 报告中的呈现方式

在报告"实验"章节中，用以下逻辑串联这些可视化:

1. **权重演化** → 证明模型确实在学习 (loss 下降的微观解释)
   > "图 X 展示了各层权重范数随训练的变化。可以看到 SA Module 3 (全局聚合层) 的权重范数增长最快，说明高层语义特征的构建是训练早期的重点..."

2. **Embedding PCA** → 证明特征空间从随机到结构化
   > "图 Y 的 PCA 轨迹显示，同一物体的 embedding 在前 50 epoch 快速移动，之后收敛到稳定邻域，与 loss 曲线的拐点一致..."

3. **SA 关键点** → 展示 PointNet++ 的层次化感受野
   > "图 Z 的 3D 可视化直观展示了 FPS 如何在物体表面均匀采样关键点，从 1024→512→128 的层级中，关键点逐渐集中在物体边缘和曲面等高曲率区域..."

4. **输入显著性** → 证明网络关注抓取接触区域
   > "梯度归因分析表明，模型对夹爪接触点附近的点云区域赋予最高的重要性权重，与其说是'看整个物体'，不如说是在寻找可能的抓取接触面..."

5. **去噪过程** → 展示扩散模型的逐步生成
   > "附录中的降噪动画展示了从纯噪声经过 10 步去噪逐渐收敛为合理抓取位姿的过程，t=10 到 t=5 确定了大致位置，t=5 到 t=1 精细调整旋转..."
