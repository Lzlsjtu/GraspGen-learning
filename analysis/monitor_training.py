#!/usr/bin/env python3
"""
[作业新增文件] monitor_training.py
非 GraspGen 原始项目文件，为 6-DOF 抓取位姿估计课程作业而创建。

训练监控脚本：解析 TensorBoard event 文件，生成按 Epoch 对齐的 loss 和误差曲线。

用法:
  python monitor_training.py --log_dir <tb_logs> --output_dir <plots> [--steps_per_epoch 177] [--watch]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ─── 配置 ────────────────────────────────────────────────────────────────────
TRAIN_LOSS_TAG = 'train/loss/all_loss'
POS_LOSS_TAG   = 'train/loss/position_loss'
ROT_LOSS_TAG   = 'train/loss/rotation_loss'
VALID_PREFIX   = 'valid/metric/reconstruction'

# ─── 数据加载 ────────────────────────────────────────────────────────────────

def load_data(log_dir):
    """加载 TB events，返回 {tag: [(step, value), ...]} 和验证数据。"""
    ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    ea.Reload()
    data = {}
    for tag in ea.Tags()['scalars']:
        data[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return data

# ─── 核心绘图 ────────────────────────────────────────────────────────────────

def step_to_epoch(steps, steps_per_epoch):
    """将 global_step 转换为 epoch 数。"""
    return np.array(steps) / steps_per_epoch


def smooth(vals, window):
    """移动平均平滑。"""
    if len(vals) < window:
        return vals
    kernel = np.ones(window) / window
    return np.convolve(vals, kernel, mode='valid')


def plot_all(data, steps_per_epoch, output_dir):
    """生成 2×2 训练曲线图，x 轴为 Epoch。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    M = steps_per_epoch

    # ── 左上：训练总损失 (Epoch 轴) ──
    ax = axes[0, 0]
    if TRAIN_LOSS_TAG in data:
        steps, vals = zip(*data[TRAIN_LOSS_TAG])
        epochs = step_to_epoch(steps, M)
        # 原始（半透明）
        ax.plot(epochs, vals, 'b-', alpha=0.25, linewidth=0.4)
        # 移动平均（粗线）
        w = max(5, len(vals) // 50)
        s = smooth(vals, w)
        ax.plot(epochs[w-1:], s, 'r-', linewidth=2, label=f'Train Loss (MA{w})')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Total Loss', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 右上：分量损失 ──
    ax = axes[0, 1]
    for tag, color, label in [
        (POS_LOSS_TAG, '#FF9800', 'Position'),
        (ROT_LOSS_TAG, '#4CAF50', 'Rotation'),
    ]:
        if tag in data:
            steps, vals = zip(*data[tag])
            epochs = step_to_epoch(steps, M)
            w = max(5, len(vals) // 20)
            s = smooth(vals, w)
            ax.plot(epochs[w-1:], s, color=color, linewidth=1.5, label=label)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Loss Components (Smoothed)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 左下：平移重建误差（仅验证点） ──
    ax = axes[1, 0]
    trans_tag = f'{VALID_PREFIX}/error_trans_l2'
    if trans_tag in data and len(data[trans_tag]) > 0:
        steps, vals = zip(*data[trans_tag])
        epochs = step_to_epoch(steps, M)
        ax.plot(epochs, vals, 'o-', color='#2196F3', markersize=8,
                linewidth=2, label='Translation Error')
        for ep, v in zip(epochs, vals):
            ax.annotate(f'{v:.3f}', (ep, v), textcoords="offset points",
                        xytext=(0, 10), fontsize=8, ha='center')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Error (m)', fontsize=11)
    ax.set_title('Validation: Translation Reconstruction Error', fontsize=13)
    ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='5 cm')
    ax.axhline(y=0.02, color='orange', linestyle='--', alpha=0.5, label='2 cm')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 右下：旋转重建误差 + Recall/Precision ──
    ax = axes[1, 1]
    rot_tag = f'{VALID_PREFIX}/error_rot_geodesic'
    if rot_tag in data and len(data[rot_tag]) > 0:
        steps, vals = zip(*data[rot_tag])
        epochs = step_to_epoch(steps, M)
        ax.plot(epochs, vals, 'o-', color='#E91E63', markersize=8,
                linewidth=2, label='Rotation Geodesic Error')
        for ep, v in zip(epochs, vals):
            ax.annotate(f'{v:.3f}', (ep, v), textcoords="offset points",
                        xytext=(0, 10), fontsize=8, ha='center')
    # Recall / Precision 双 y 轴
    ax2 = ax.twinx()
    for tag, color, label in [
        (f'{VALID_PREFIX}/recall', '#4CAF50', 'Recall'),
        (f'{VALID_PREFIX}/precision', '#FF9800', 'Precision'),
    ]:
        if tag in data and len(data[tag]) > 0:
            steps, vals = zip(*data[tag])
            epochs = step_to_epoch(steps, M)
            ax2.plot(epochs, vals, 's--', color=color, markersize=6,
                     linewidth=1.2, label=label)
            for ep, v in zip(epochs, vals):
                ax2.annotate(f'{v:.3f}', (ep, v), textcoords="offset points",
                             xytext=(0, -12), fontsize=7, ha='center', color=color)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Rotation Error (rad)', fontsize=11, color='#E91E63')
    ax2.set_ylabel('Recall / Precision', fontsize=11)
    ax.set_title('Validation: Rotation Error & Grasp Quality', fontsize=13)
    ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='0.3 rad')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.suptitle('GraspGen Generator — Training Progress', fontsize=14, fontweight='bold')
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close(fig)


# ─── 终端输出 ────────────────────────────────────────────────────────────────

def print_summary(data, steps_per_epoch):
    """终端打印当前训练摘要。"""
    M = steps_per_epoch
    print(); print('=' * 60)
    print('  GraspGen Generator — 训练状态')
    print('=' * 60)

    # 当前 epoch
    tr = data.get(TRAIN_LOSS_TAG, [])
    if tr:
        s, v = tr[-1]
        ep = s / M
        pct = 100 * ep / 200
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f'  Epoch: {ep:.1f} / 200  [{bar}] {pct:.1f}%')
        print(f'  Train Loss: {v:.4f}  (step {s})')
        # 最近趋势
        if len(tr) > 500:
            old = sum(e[1] for e in tr[-500:-250]) / 250
            new = sum(e[1] for e in tr[-250:]) / 250
            delta = (old - new) / old * 100
            print(f'  Loss 趋势: {old:.4f} → {new:.4f}  ({delta:+.1f}%)')

    # 验证指标
    print()
    print(f'  {"指标":20s} {"Epoch":>6s}  {"值":>10s}')
    print(f'  {"-"*20} {"-"*6}  {"-"*10}')
    for tag, name, unit in [
        (f'{VALID_PREFIX}/error_trans_l2', '平移重建误差', 'm'),
        (f'{VALID_PREFIX}/error_rot_geodesic', '旋转重建误差', 'rad'),
        (f'{VALID_PREFIX}/recall', 'Recall', ''),
        (f'{VALID_PREFIX}/precision', 'Precision', ''),
    ]:
        if tag in data:
            for step, val in data[tag]:
                print(f'  {name:20s} {step/M:5.1f}  {val:10.5f} {unit}')
    print('=' * 60)


# ─── 入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='GraspGen 训练监控')
    parser.add_argument('--log_dir', required=True, help='TensorBoard 日志目录')
    parser.add_argument('--output_dir', default='./plots', help='图片输出目录')
    parser.add_argument('--steps_per_epoch', type=float, default=None,
                        help='每 epoch 的 step 数（默认自动检测）')
    parser.add_argument('--watch', action='store_true', help='持续监控模式')
    parser.add_argument('--interval', type=int, default=300, help='监控间隔（秒）')
    args = parser.parse_args()

    def run_once():
        data = load_data(args.log_dir)
        if not data:
            print('No data yet.')
            return None

        # 自动检测 steps_per_epoch：training_steps / max_epoch_in_data
        if args.steps_per_epoch:
            M = args.steps_per_epoch
        else:
            tr = data.get(TRAIN_LOSS_TAG, [])
            max_step = max(e[0] for e in tr) if tr else 1
            val_steps = []
            for tag in data:
                if 'valid/' in tag and 'reconstruction/' in tag:
                    val_steps.extend([e[0] for e in data[tag]])
            if val_steps:
                # 用验证点的最大 step 和对应的 eval_freq 推断
                max_val_step = max(val_steps)
                # 找最接近的 eval epoch: eval_freq=40 → step_40, step_80, ...
                for eval_ep in [40, 80, 120, 160]:
                    candidates = [s for s in val_steps if abs(s/eval_ep - max_val_step/eval_ep) < 0.1]
                M = max_step / (max(val_steps) / 40 * 40)  # rough
                # 简单法: 用 train loss 的 max_step / 当前 epoch
                # 取验证点中最大的 epoch 来校准
                M = max_step / 200 * 200  # fallback
                M = round(max_step / max(e[1] for e in val_steps) * 40 if val_steps else max_step/200)
            else:
                M = len(tr) / 200 if tr else 177
            if M < 1:
                M = 177
            print(f'Auto-detected steps_per_epoch = {M:.1f}')

        plot_all(data, M, args.output_dir)
        print_summary(data, M)
        return data

    data = run_once()
    if not data:
        return

    if args.watch:
        import time
        print(f'\n持续监控中，每 {args.interval}s 更新... (Ctrl+C 停止)\n')
        while True:
            try:
                time.sleep(args.interval)
                run_once()
            except KeyboardInterrupt:
                print('\n监控已停止。')
                break


if __name__ == '__main__':
    main()
