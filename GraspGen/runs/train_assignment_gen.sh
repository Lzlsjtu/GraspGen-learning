#!/usr/bin/env bash
# ============================================================================
# [作业新增文件] train_assignment_gen.sh
# 非 GraspGen 原始项目文件，为 6-DOF 抓取位姿估计课程作业而创建。
#
# 用途: GraspGen Generator 训练启动脚本 (RTX 3060 6GB 显存适配)
# 数据集: franka_panda_subset (380 train / 95 valid)
# 用法: bash runs/train_assignment_gen.sh
# 监控: tensorboard --logdir=.../results_assignment/logs
# ============================================================================

set -e

cd /home/lzl/Projects/6Dpose/GraspGen
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

# ─── 路径配置 ────────────────────────────────────────────────────────────────
GRASP_DATASET="/home/lzl/Projects/6Dpose/GraspGen_datasets/grasp_data"
OBJECT_DATASET="/home/lzl/Projects/6Dpose/GraspGen_datasets/object_dataset"
RESULTS="/home/lzl/Projects/6Dpose/GraspGen/runs/results_assignment"
CACHE_DIR="$RESULTS/cache"
LOG_DIR="$RESULTS/logs"

rm -rf "$LOG_DIR" "$CACHE_DIR"
mkdir -p "$LOG_DIR" "$CACHE_DIR"

echo "============================================"
echo "  GraspGen Generator Training (Assignment)"
echo "  GPU: RTX 3060 (6GB VRAM)"
echo "  Dataset: franka_panda_subset (380/95)"
echo "  Backbone: pointnet | Points: 1024 | Batch: 2"
echo "  Epochs: 200 | LR: 1e-5 | Diffusion steps: 10"
echo "  Logs: $LOG_DIR"
echo "============================================"

cd /home/lzl/Projects/6Dpose/GraspGen/scripts

python train_graspgen.py \
    train.debug=True \
    data.root_dir="$GRASP_DATASET/splits/franka_panda_subset" \
    data.object_root_dir="$OBJECT_DATASET" \
    data.grasp_root_dir="$GRASP_DATASET/grasp_data/franka_panda" \
    data.gripper_name="franka_panda" \
    data.num_points=1024 \
    data.num_grasps_per_object=100 \
    data.prob_point_cloud=0.0 \
    data.redundancy=1 \
    data.dataset_name="objaverse" \
    data.dataset_version="v2" \
    data.dataset_cls="ObjectPickDataset" \
    data.cache_dir="$CACHE_DIR" \
    data.rotation_augmentation=True \
    data.load_contact=False \
    data.visualize_batch=False \
    data.preload_dataset=True \
    train.log_dir="$LOG_DIR" \
    train.batch_size=2 \
    train.num_epochs=200 \
    train.num_workers=0 \
    train.print_freq=5 \
    train.save_freq=40 \
    train.eval_freq=40 \
    train.model_name="diffusion" \
    train.checkpoint="" \
    diffusion.gripper_name="franka_panda" \
    diffusion.num_diffusion_iters=10 \
    diffusion.num_diffusion_iters_eval=10 \
    diffusion.obs_backbone="pointnet" \
    diffusion.grasp_repr="r3_so3" \
    diffusion.compositional_schedular=True \
    diffusion.loss_pointmatching=False \
    diffusion.loss_l1_pos=True \
    diffusion.loss_l1_rot=True \
    diffusion.attention="cat" \
    diffusion.kappa=3.27 \
    optimizer.type="ADAMW" \
    optimizer.lr=0.00001 \
    optimizer.grad_clip=-1 \
    optimizer.weight_decay=0.05 \
    | tee "$LOG_DIR/console_log.txt"

echo ""
echo "Training finished!"
echo "Logs saved to: $LOG_DIR"
echo "View with: tensorboard --logdir=$LOG_DIR"
