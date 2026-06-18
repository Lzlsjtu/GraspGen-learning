#!/usr/bin/env python3
"""
[作业新增文件] graspnet_pipeline/inference.py
非 GraspGen 原始项目文件。

加载训练好的 Generator 模型，对预处理后的点云执行抓取推理。
"""

import sys
sys.path.insert(0, "/home/lzl/Projects/6Dpose/GraspGen")

import numpy as np
import torch
from grasp_gen.dataset.dataset import collate
from grasp_gen.utils.point_cloud_utils import knn_points


def run_graspgen_inference(
    model,
    pc_centered: np.ndarray,
    num_grasps: int = 100,
    kappa: float = 3.27,
):
    """使用训练好的 Generator 对点云执行抓取推理。

    直接调用 GraspGenGenerator.infer()，10 步 DDPM 逆向去噪，
    输出 K 个候选抓取位姿 + 扩散似然度 (likelihood) 作为置信度。

    Args:
        model:        GraspGenGenerator (已加载权重)
        pc_centered:  (N, 3) 已中心化的物体点云 (numpy)
        num_grasps:   生成的抓取数量
        kappa:        点云缩放因子 (训练时用的)

    Returns:
        grasps:     (num_grasps, 4, 4) 齐次抓取位姿矩阵
        confidence: (num_grasps,) 扩散似然度分数 (越高越好)
    """
    model.num_grasps_per_object = num_grasps

    # 构造 batch 输入 (必须在 CUDA 上)
    pc_tensor = torch.from_numpy(pc_centered).float().cuda()
    # 应用 kappa 缩放 (与训练时一致)
    if kappa is not None:
        pc_tensor = kappa * pc_tensor

    data = {
        "task": "pick",
        "points": pc_tensor.unsqueeze(0),      # (1, N, 3)
        "inputs": pc_tensor.unsqueeze(0),       # (1, N, 3) 兼容 collate
    }
    data_batch = collate([data])

    # 推理
    with torch.inference_mode():
        outputs, _, _ = model.infer(data_batch)

    grasps    = outputs["grasps_pred"][0]             # (K, 4, 4)
    confidence = outputs["likelihood"][0, :, 0]       # (K,)

    # confidence → [0, 1] softmax 归一化
    if confidence.numel() > 0:
        confidence = torch.softmax(confidence, dim=0)
    else:
        confidence = torch.ones(len(grasps)) / max(len(grasps), 1)

    return grasps.cpu(), confidence.cpu()
