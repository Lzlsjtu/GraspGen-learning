# 6-DOF 抓取位姿估计 — 课程作业项目

> 基于 NVIDIA GraspGen 扩散模型框架，实现从局部点云到 6 自由度抓取位姿的端到端训练与评估。

---

## 项目结构

```
/home/lzl/Projects/6Dpose/
│
├── README.md                         # ← 本文件：项目总览
│
├── GraspGen/                         # 原始项目（NVIDIA GraspGen）
│   │                                 #   仅做最小化修改（1 处兼容修复）
│   │                                 #   所有原始代码、模型、训练脚本均在此
│   ├── grasp_gen/                    #   核心 Python 包
│   │   ├── models/                   #     Generator, Discriminator, M2T2, PointNet++
│   │   ├── dataset/                  #     数据加载、点云渲染、碰撞检测
│   │   ├── utils/                    #     旋转数学、可视化、训练工具
│   │   └── serving/                  #     ZMQ 推理服务
│   ├── scripts/                      #   训练/推理/可视化脚本
│   ├── config/                       #   夹爪配置文件
│   ├── runs/                         #   训练启动脚本 + 训练输出
│   │   ├── train_assignment_gen.sh   #     [作业新增] 适配 6GB 显存的训练脚本
│   │   └── results_assignment/       #     训练产物（checkpoint, TB events, 日志）
│   ├── GraspGenModels/               #   预训练模型 checkpoint（HuggingFace）
│   ├── tests/                        #   单元测试
│   └── pointnet2_ops/                #   PointNet++ CUDA 扩展
│
├── GraspGen_datasets/                # 数据集
│   ├── grasp_data/                   #   抓取标注（WebDataset 格式, 21 GB）
│   │   ├── splits/                   #     train/valid/test UUID 列表
│   │   └── grasp_data/               #     per-gripper WebDataset shards
│   └── object_dataset/               #   物体网格（475 个简化 .glb, 1.8 GB）
│
├── analysis/                         # [作业新增] 后处理分析工具
│   │                                 #   独立于 GraspGen，不 import grasp_gen
│   ├── README.md                     #   分析工具说明
│   ├── monitor_training.py           #   训练曲线生成（读 TB events → matplotlib）
│   └── plots/                        #   自动生成的 loss/误差曲线图
│
└── docs/                             # [作业新增] 作业文档
    ├── README.md                     #   文档目录说明
    ├── SETUP_GUIDE.md                #   环境搭建与运行指南（从零复现）
    ├── math_principles.md            #   完整数学原理（RGB-D → 6-DOF 全流程公式推导）
    └── IMPLEMENTATION_PLAN.md        #   实现计划（架构、步骤、得分点拆解）
```

---

## 快速导航

| 我想... | 看这里 |
|---------|--------|
| 了解项目整体结构 | 本文件 |
| 从零搭建环境并运行训练 | [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md) |
| 理解数学原理（相机模型、PointNet++、扩散、旋转表示） | [`docs/math_principles.md`](docs/math_principles.md) |
| 了解实现方案和得分点 | [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) |
| 启动训练 | `bash GraspGen/runs/train_assignment_gen.sh` |
| 查看训练曲线 | `python analysis/monitor_training.py --log_dir ...` |
| 运行推理 | `python GraspGen/scripts/demo_object_mesh.py ...` |

---

## 原始 vs 新增文件区分

所有非 GraspGen 原始项目文件均带有 `[作业新增]` 或 `[作业修改]` 标记：

| 标记 | 含义 |
|------|------|
| `[作业新增]` | 为本次课程作业全新创建的文件 |
| `[作业修改]` | 对 GraspGen 原始代码的修改（仅 1 处：`meshcat_utils.py` 兼容修复） |

**新增文件清单：**

| 文件 | 位置 | 类型 |
|------|------|------|
| `train_assignment_gen.sh` | `GraspGen/runs/` | 耦合型（依赖 GraspGen） |
| `monitor_training.py` | `analysis/` | 分析型（独立） |
| `SETUP_GUIDE.md` | `docs/` | 文档 |
| `math_principles.md` | `docs/` | 文档 |
| `IMPLEMENTATION_PLAN.md` | `docs/` | 文档 |
| `CLAUDE.md` | `GraspGen/` | 项目指南 |
| `meshcat_utils.py` | `GraspGen/grasp_gen/utils/` | 已修改（1 处兼容修复） |

---

## 环境速查

| 项目 | 值 |
|------|-----|
| Python | 3.10 |
| PyTorch | 2.1.0+cu121 |
| CUDA | 12.1 |
| GPU | RTX 3060 Laptop (6 GB) |
| 虚拟环境 | `GraspGen/.venv/` |

```bash
cd GraspGen && source .venv/bin/activate
```

---

## 当前训练状态

| 指标 | Epoch 1 | Epoch 40 | Epoch 80 | 趋势 |
|------|---------|----------|----------|------|
| 训练 Loss | 2.20 | 2.03 | 1.96 | ↓ |
| 平移重建误差 | 0.327 m | 0.197 m | **0.160 m** | ↓ -51% |
| 旋转重建误差 | 1.764 rad | 1.313 rad | **1.147 rad** | ↓ -35% |
| Recall | 0.007 | 0.025 | **0.028** | ↑ 4x |
| Precision | 0.015 | 0.050 | **0.066** | ↑ 4.4x |

训练进行中：Epoch 92 / 200，目标 200 epoch。
