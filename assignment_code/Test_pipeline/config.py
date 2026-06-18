"""
[作业新增文件] graspnet_pipeline/config.py
非 GraspGen 原始项目文件。

GraspNet-1Billion 推理管线的全局配置。
"""

import os
from pathlib import Path

# ─── 路径配置 ────────────────────────────────────────────────────────────────
PROJECT_ROOT       = Path("/home/lzl/Projects/6Dpose")
GRASPNET_DATASETS  = PROJECT_ROOT / "GraspNet_datasets"       # GraspNet-1Billion 数据集
GRASPNET_BASELINE  = PROJECT_ROOT / "graspnet_baseline"       # graspnet-baseline 代码 + 权重
CHECKPOINT_DIR     = PROJECT_ROOT / "checkpoints"              # 我们训练的权重
PRETRAINED_DIR     = PROJECT_ROOT / "GraspGen" / "GraspGenModels" / "checkpoints"  # GraspGen 官方预训练
GRASPGEN_DIR       = PROJECT_ROOT / "GraspGen"                # GraspGen 代码
OUTPUT_DIR         = PROJECT_ROOT / "graspnet_pipeline" / "outputs"
PLOTS_DIR          = PROJECT_ROOT / "graspnet_pipeline" / "plots"

# 向后兼容 (旧名称)
GRASPNET_DATA = GRASPNET_DATASETS

# ─── 模型配置 ────────────────────────────────────────────────────────────────
# GraspGen 官方预训练权重（效果最好，3000+ epoch × 8×A100）
PRETRAINED_CONFIG    = str(PRETRAINED_DIR / "graspgen_franka_panda.yml")
PRETRAINED_GEN       = str(PRETRAINED_DIR / "graspgen_franka_panda_gen.pth")
PRETRAINED_DIS       = str(PRETRAINED_DIR / "graspgen_franka_panda_dis.pth")

# GraspNet-Baseline 预训练权重 (官方 graspnet-baseline, RealSense camera)
BASELINE_CHECKPOINT  = str(GRASPNET_BASELINE / "checkpoint-rs.tar")
BASELINE_DATASET     = str(GRASPNET_DATASETS)                 # 共用同一数据集根目录

# 我们自己训练的权重（用于对比）
OUR_CHECKPOINT = str(CHECKPOINT_DIR / "last.pth")
OUR_CONFIG     = str(GRASPGEN_DIR / "runs" / "results_assignment" / "logs" / "config.yaml")

GRIPPER_NAME    = "franka_panda"                              # 夹爪类型

# ─── 点云预处理 ──────────────────────────────────────────────────────────────
NUM_POINTS      = 1024                                        # FPS 降采样点数
SOR_K           = 20                                          # SOR 最近邻数
SOR_THRESHOLD   = 0.014                                       # SOR 距离阈值 (m)
KAPPA           = 3.27                                        # 点云缩放因子

# ─── 推理参数 ────────────────────────────────────────────────────────────────
NUM_GRASPS      = 100                                         # 生成的抓取数量
DIFFUSION_STEPS = 10                                          # DDPM 去噪步数

# ─── 碰撞检测 ────────────────────────────────────────────────────────────────
COLLISION_THRESHOLD = 0.003                                   # 碰撞距离阈值 (m)
NUM_COLLISION_SAMPLES = 1000                                  # 夹爪采样点数

# ─── GraspNet 相机内参 (RealSense D435) ─────────────────────────────────────
# 参考: GraspNet-1Billion 官方文档
REALSENSE_INTRINSICS = {
    "fx": 591.0,
    "fy": 590.6,
    "cx": 322.5,
    "cy": 238.3,
    "width":  640,
    "height": 480,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
