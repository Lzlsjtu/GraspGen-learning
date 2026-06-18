# 6-DOF 抓取位姿估计 — 课程作业

基于 **NVIDIA GraspGen** 扩散模型的 6-DOF 机器人抓取位姿估计作业实现。

---

## 📁 项目目录结构

```
6Dpose/
├── 📂 assignment_code/          # 课程作业核心代码 (176KB)
│   ├── 📂 Test_pipeline/        # 完整推理Pipeline (14个.py)
│   ├── 📂 analysis/             # 训练分析与可视化 (2个脚本)
│   └── 📂 training_scripts/     # 训练脚本 (1个核心脚本)
├── 📂 graspgen_source/          # GraspGen官方源码（只读）
├── 📂 docs/                     # 文档（7个核心文档 + 答辩PPT）
├── 📂 training_data/            # 训练过程数据（不上传GitHub，单独打包）
├── 📂 GraspGen_datasets/        # 数据集（不上传GitHub，需单独下载）
└── 📂 _archive/                 # 归档内容（非交付）
```

---

## 🚀 快速开始

### 环境配置

```bash
cd graspgen_source
source .venv/bin/activate
```

### 完整 Pipeline 演示

**RGB-D → 点云 → 预处理 → GraspGen → 6-DOF 抓取位姿 → Open3D 可视化**

```bash
cd /home/lzl/Projects/6Dpose
python3 assignment_code/Test_pipeline/pipeline_full.py --scene
```

### 推理 Demo 流程（6步）

| 步骤 | 内容 |
|------|------|
| 1 | RGB + Depth + Mask 合并显示 |
| 2 | 完整场景彩色点云 |
| 3 | 去噪后物体点云 |
| 4 | 降采样 1024 点（GraspGen 输入） |
| 5 | GraspGen 推理 |
| 6 | 夹爪抓取结果可视化 |

---

## 📖 文档

[👉 查看完整文档目录](docs/)

---

## ⚠️ 大文件说明（未上传至 GitHub）

| 文件/目录 | 大小 | 获取方式 |
|-----------|------|----------|
| `GraspGen_datasets/` | ~23 GB | HuggingFace 下载 |
| `graspgen_source/GraspGenModels/` | ~1.5 GB | HuggingFace 预训练模型 |
| `training_data/` | ~4.5 GB | 训练自动生成 |
