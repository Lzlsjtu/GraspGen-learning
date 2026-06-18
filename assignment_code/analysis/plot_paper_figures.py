#!/usr/bin/env python3
"""
[作业新增文件] plot_paper_figures.py
为答辩 PPT 生成论文风格指标图。

输出目录默认: analysis/plots/paper_figures/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TRAIN_LOSS_TAG = "train/loss/all_loss"
POS_LOSS_TAG = "train/loss/position_loss"
ROT_LOSS_TAG = "train/loss/rotation_loss"
VALID_PREFIX = "valid/metric/reconstruction"

COLORS = {
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "orange": "#EA580C",
    "red": "#DC2626",
    "green": "#16A34A",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "dark": "#0F172A",
}


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 1.0,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "white",
        "legend.edgecolor": "#E2E8F0",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "axes.labelcolor": "#0F172A",
        "text.color": "#0F172A",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load_tb(log_dir):
    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        data[tag] = [(e.step, float(e.value)) for e in ea.Scalars(tag)]
    return data


def steps_to_epoch(items, steps_per_epoch):
    return np.array([s for s, _ in items], dtype=float) / steps_per_epoch


def values(items):
    return np.array([v for _, v in items], dtype=float)


def moving_average(y, window):
    if len(y) < window or window <= 1:
        return y, np.arange(len(y))
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid"), np.arange(window - 1, len(y))


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path}")


def plot_loss(data, steps_per_epoch, out):
    tr = data.get(TRAIN_LOSS_TAG, [])
    if not tr:
        return
    x = steps_to_epoch(tr, steps_per_epoch)
    y = values(tr)
    win = max(10, min(300, len(y) // 40))
    ys, idx = moving_average(y, win)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(x, y, color=COLORS["blue"], alpha=0.16, linewidth=0.7, label="Raw mini-batch loss")
    ax.plot(x[idx], ys, color=COLORS["blue"], linewidth=2.4, label=f"Moving average (window={win})")
    ax.set_title("Training Loss Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Diffusion training loss")
    ax.grid(True, axis="both")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.06, f"Latest loss: {y[-1]:.4f}", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#EFF6FF", edgecolor="#BFDBFE"))
    save(fig, out / "fig01_training_loss.png")


def plot_loss_components(data, steps_per_epoch, out):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    plotted = False
    for tag, color, label in [
        (POS_LOSS_TAG, COLORS["orange"], "Translation component"),
        (ROT_LOSS_TAG, COLORS["green"], "Rotation component"),
    ]:
        if tag in data:
            x = steps_to_epoch(data[tag], steps_per_epoch)
            y = values(data[tag])
            win = max(10, min(300, len(y) // 40))
            ys, idx = moving_average(y, win)
            ax.plot(x[idx], ys, color=color, linewidth=2.3, label=label)
            plotted = True
    if not plotted:
        plt.close(fig); return
    ax.set_title("Translation vs. Rotation Loss Components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 noise prediction loss")
    ax.grid(True)
    ax.legend(loc="upper right")
    save(fig, out / "fig02_loss_components.png")


def plot_pose_metrics(data, steps_per_epoch, out):
    trans_tag = f"{VALID_PREFIX}/error_trans_l2"
    rot_tag = f"{VALID_PREFIX}/error_rot_geodesic"
    if trans_tag not in data and rot_tag not in data:
        return
    fig, ax1 = plt.subplots(figsize=(9.5, 5.4))
    if trans_tag in data:
        x = steps_to_epoch(data[trans_tag], steps_per_epoch)
        y = values(data[trans_tag])
        ax1.plot(x, y * 100, "o-", color=COLORS["blue"], linewidth=2.4, markersize=5.5,
                 label="Translation error")
        for xi, yi in zip(x, y * 100):
            ax1.annotate(f"{yi:.1f}", (xi, yi), xytext=(0, 7), textcoords="offset points",
                         ha="center", fontsize=8, color=COLORS["blue"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Translation error (cm)", color=COLORS["blue"])
    ax1.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax1.grid(True)

    ax2 = ax1.twinx()
    if rot_tag in data:
        x = steps_to_epoch(data[rot_tag], steps_per_epoch)
        y = values(data[rot_tag])
        ax2.plot(x, y, "s--", color=COLORS["red"], linewidth=2.2, markersize=5.2,
                 label="Rotation geodesic error")
        for xi, yi in zip(x, y):
            ax2.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, -14), textcoords="offset points",
                         ha="center", fontsize=8, color=COLORS["red"])
    ax2.set_ylabel("Rotation error (rad)", color=COLORS["red"])
    ax2.tick_params(axis="y", labelcolor=COLORS["red"])
    ax1.set_title("6-DOF Pose Reconstruction Accuracy")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    save(fig, out / "fig03_pose_errors.png")


def plot_recall_precision(data, steps_per_epoch, out):
    recall_tag = f"{VALID_PREFIX}/recall"
    precision_tag = f"{VALID_PREFIX}/precision"
    if recall_tag not in data and precision_tag not in data:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for tag, color, label in [
        (recall_tag, COLORS["green"], "Recall"),
        (precision_tag, COLORS["purple"], "Precision"),
    ]:
        if tag in data:
            x = steps_to_epoch(data[tag], steps_per_epoch)
            y = values(data[tag])
            ax.plot(x, y * 100, "o-", color=color, linewidth=2.4, markersize=5.5, label=label)
            for xi, yi in zip(x, y * 100):
                ax.annotate(f"{yi:.1f}", (xi, yi), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=8, color=color)
    ax.set_title("Grasp Set Matching Quality")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric (%)")
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend(loc="upper left")
    save(fig, out / "fig04_recall_precision.png")


def plot_model_comparison(out):
    labels = ["Official\nPTV3", "Ours\nPointNet"]
    collision_free = [70.18, 71.35]
    confidence = [0.630, 0.010]
    time_ms = [178.3, 48.6]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))
    panels = [
        (collision_free, "Collision-free grasps", "%", COLORS["green"]),
        (confidence, "Mean confidence", "score", COLORS["blue"]),
        (time_ms, "Inference time", "ms", COLORS["orange"]),
    ]
    for ax, (vals, title, ylabel, color) in zip(axes, panels):
        bars = ax.bar(labels, vals, color=[color, COLORS["gray"]], width=0.58, alpha=0.9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}" if v < 10 else f"{v:.1f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10)
    fig.suptitle("Real-scene Evaluation: Official Pretrained vs. Our Lightweight Model", fontweight="bold")
    save(fig, out / "fig05_model_comparison.png")


def count_available(root):
    root = Path(root)
    mesh = {p.stem for p in (root / "GraspGen_datasets/object_dataset").glob("*.glb")}
    result = {}
    for name in ["train", "valid"]:
        ids = [x.strip() for x in (root / "GraspGen_datasets/grasp_data/splits/franka_panda" / f"{name}.txt").read_text().splitlines() if x.strip()]
        result[name] = (sum(x in mesh for x in ids), len(ids))
    result["glb"] = (len(mesh), None)
    return result


def plot_data_coverage(root, out):
    c = count_available(root)
    labels = ["Train", "Validation"]
    avail = [c["train"][0], c["valid"][0]]
    total = [c["train"][1], c["valid"][1]]
    pct = [a / t * 100 for a, t in zip(avail, total)]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, total, color="#E2E8F0", label="Official split size", width=0.58)
    bars = ax.bar(labels, avail, color=[COLORS["blue"], COLORS["green"]], label="Downloaded & usable", width=0.58)
    ax.set_title("Downloaded Objaverse Mesh Coverage")
    ax.set_ylabel("Number of objects")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")
    for b, a, t, p in zip(bars, avail, total, pct):
        ax.annotate(f"{a}/{t}\n({p:.1f}%)", (b.get_x()+b.get_width()/2, a),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=10)
    save(fig, out / "fig06_data_coverage.png")


def plot_pipeline_summary(out):
    stages = ["RGB-D", "Point\nCloud", "SOR\nDenoise", "FPS\n1024", "PointNet\nEncoder", "DDPM\nSampler", "Collision\nFilter"]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(x, np.ones_like(x), color=COLORS["blue"], linewidth=3, zorder=1)
    ax.scatter(x, np.ones_like(x), s=650, color="white", edgecolor=COLORS["blue"], linewidth=2.5, zorder=2)
    for i, s in enumerate(stages):
        ax.text(i, 1, s, ha="center", va="center", fontsize=10, fontweight="bold")
        if i < len(stages) - 1:
            ax.annotate("", xy=(i + 0.74, 1), xytext=(i + 0.26, 1),
                        arrowprops=dict(arrowstyle="->", color=COLORS["dark"], lw=1.4))
    ax.set_xlim(-0.6, len(stages)-0.4)
    ax.set_ylim(0.65, 1.35)
    ax.axis("off")
    ax.set_title("End-to-End Grasp Pose Estimation Pipeline", pad=16)
    save(fig, out / "fig07_pipeline_summary.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--output_dir", default="analysis/plots/paper_figures")
    parser.add_argument("--steps_per_epoch", type=float, default=7)
    parser.add_argument("--project_root", default="/home/lzl/Projects/6Dpose")
    args = parser.parse_args()

    setup_style()
    out = Path(args.output_dir)
    data = load_tb(Path(args.log_dir))

    plot_loss(data, args.steps_per_epoch, out)
    plot_loss_components(data, args.steps_per_epoch, out)
    plot_pose_metrics(data, args.steps_per_epoch, out)
    plot_recall_precision(data, args.steps_per_epoch, out)
    plot_model_comparison(out)
    plot_data_coverage(args.project_root, out)
    plot_pipeline_summary(out)


if __name__ == "__main__":
    main()
