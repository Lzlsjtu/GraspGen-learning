#!/usr/bin/env python3
"""
[作业新增文件] graspnet_pipeline/model_loader.py
非 GraspGen 原始项目文件。

加载 GraspGen 模型用于推理。支持:
  1. 官方预训练权重 (首选, 效果最好)
  2. 我们自己训练的 checkpoint (用于对比)
"""

import sys
sys.path.insert(0, "/home/lzl/Projects/6Dpose/GraspGen")

import os
import torch
import omegaconf
from pathlib import Path
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.robot import get_gripper_info


def load_pretrained_model(gripper_config: str = None):
    """加载官方预训练模型 (Generator + Discriminator)。

    使用 GraspGen 官方的 GraspGenSampler，与 demo 脚本完全一致。

    Args:
        gripper_config: 预训练 config yaml 路径

    Returns:
        sampler: GraspGenSampler 实例 (已加载权重)
        cfg: OmegaConf 配置
        gripper_info: GripperInfo
    """
    if gripper_config is None:
        from config import PRETRAINED_CONFIG
        gripper_config = PRETRAINED_CONFIG

    cfg = load_grasp_cfg(gripper_config)
    sampler = GraspGenSampler(cfg)

    gripper_info = get_gripper_info(cfg.data.gripper_name)

    print(f"[Model] Using pretrained model: {gripper_config}")
    print(f"[Model] Gripper: {cfg.data.gripper_name}")
    print(f"[Model] Backbone: {cfg.diffusion.obs_backbone}")
    print(f"[Model] Diffusion steps: {cfg.diffusion.num_diffusion_iters}")

    return sampler, cfg, gripper_info


def load_our_model(checkpoint_path: str = None, config_yaml: str = None):
    """加载我们自己训练的 Generator (仅用于对比)。

    Args:
        checkpoint_path: 我们训练的 .pth checkpoint
        config_yaml: 训练时的 config.yaml

    Returns:
        model: GraspGenGenerator (eval mode, on CUDA)
        cfg: OmegaConf 配置
        gripper_info: GripperInfo
    """
    if checkpoint_path is None or config_yaml is None:
        from config import OUR_CHECKPOINT, OUR_CONFIG
        checkpoint_path = OUR_CHECKPOINT
        config_yaml = OUR_CONFIG

    from grasp_gen.models.generator import GraspGenGenerator

    cfg = omegaconf.OmegaConf.load(config_yaml)
    model = GraspGenGenerator.from_config(cfg.diffusion)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()

    gripper_info = get_gripper_info(cfg.diffusion.gripper_name)

    print(f"[Model] Using our checkpoint: {checkpoint_path}")
    print(f"[Model] Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"[Model] Gripper: {cfg.diffusion.gripper_name}")
    print(f"[Model] Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model, cfg, gripper_info
