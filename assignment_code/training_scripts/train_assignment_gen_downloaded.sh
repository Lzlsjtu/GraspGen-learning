#!/usr/bin/env bash
# ============================================================================
# [作业新增文件] train_assignment_gen_downloaded.sh
# 非 GraspGen 原始项目文件，为 6-DOF 抓取位姿估计课程作业而创建。
#
# 用途: 使用当前已下载的 franka_panda 训练数据继续训练 GraspGen Generator。
# 数据集: franka_panda_downloaded (424 train / 51 valid)
# 硬件: RTX 3060 Laptop 6GB，按当前 GPU/CPU 利用率调大 batch。
# 用法: bash runs/train_assignment_gen_downloaded.sh
# ============================================================================

set -e

cd /home/lzl/Projects/6Dpose/GraspGen
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# ─── 路径配置 ────────────────────────────────────────────────────────────────
GRASP_DATASET="/home/lzl/Projects/6Dpose/GraspGen_datasets/grasp_data"
OBJECT_DATASET="/home/lzl/Projects/6Dpose/GraspGen_datasets/object_dataset"
RESULTS="/home/lzl/Projects/6Dpose/GraspGen/runs/results_assignment_downloaded"
CACHE_DIR="$RESULTS/cache"
LOG_DIR="$RESULTS/logs"
SOURCE_CKPT="/home/lzl/Projects/6Dpose/GraspGen/runs/results_assignment/logs/last.pth"

mkdir -p "$LOG_DIR" "$CACHE_DIR"

# 保留一份旧 subset 训练得到的权重作为 downloaded split 的初始化点。
if [ ! -f "$LOG_DIR/last.pth" ] && [ -f "$SOURCE_CKPT" ]; then
    cp "$SOURCE_CKPT" "$LOG_DIR/init_from_subset_last.pth"
fi

CHECKPOINT="$LOG_DIR/last.pth"
if [ ! -f "$CHECKPOINT" ]; then
    CHECKPOINT="$LOG_DIR/init_from_subset_last.pth"
fi

printf '%s\n' "============================================"
printf '%s\n' "  GraspGen Generator Training (Downloaded Full Split)"
printf '%s\n' "  GPU: RTX 3060 Laptop (6GB VRAM)"
printf '%s\n' "  Dataset: franka_panda_downloaded (424 train / 51 valid)"
printf '%s\n' "  Backbone: pointnet | Points: 1024 | Batch: 64"
printf '%s\n' "  Epochs: 10000 | LR: 1e-5 | Diffusion steps: 10"
printf '%s\n' "  Checkpoint: $CHECKPOINT"
printf '%s\n' "  Logs: $LOG_DIR"
printf '%s\n' "============================================"

cd /home/lzl/Projects/6Dpose/GraspGen/scripts

python train_graspgen.py \
    train.debug=True \
    data.root_dir="$GRASP_DATASET/splits/franka_panda_downloaded" \
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
    data.preload_dataset=False \
    train.log_dir="$LOG_DIR" \
    train.batch_size=64 \
    train.num_epochs=10000 \
    train.num_workers=8 \
    train.print_freq=10 \
    train.save_freq=50 \
    train.eval_freq=50 \
    train.model_name="diffusion" \
    train.checkpoint="$CHECKPOINT" \
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
    2>&1 | tee -a "$LOG_DIR/console_log.txt"

printf '\nTraining finished!\n'
printf 'Logs saved to: %s\n' "$LOG_DIR"
printf 'View with: tensorboard --logdir=%s\n' "$LOG_DIR"
