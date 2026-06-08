# CLAUDE.md

> [作业新增文件] 此文件为 6-DOF 抓取位姿估计课程作业而创建，非 GraspGen 原始项目文件。

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install (inference-only, recommended: uv)
uv python install 3.10 && uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e .
./install_uv_pointnet.sh          # pointnet2_ops CUDA extension — install_pointnet.sh for conda/pip

# Verify installation
pytest tests/test_inference_installation.py -v

# Run all tests (requires CUDA)
pytest tests/ -v

# Single test file
pytest tests/test_math_utils.py -v

# Integration tests (require ZMQ server on localhost:5556)
pytest tests/ -v -m integration

# Lint/format (optional dev deps)
black grasp_gen/ scripts/ tests/
isort grasp_gen/ scripts/ tests/
flake8 grasp_gen/ scripts/ tests/
```

## Architecture

GraspGen is a diffusion-based 6-DOF robotic grasp generation framework. The pipeline: **point cloud → encoder → diffusion denoising → grasp poses (N×9)**, with an optional discriminator that scores/ranks generated grasps.

### Three model architectures (`grasp_gen/models/`)

| Model | File | Purpose |
|-------|------|---------|
| `GraspGenGenerator` | `generator.py` | Diffusion model: object point cloud → grasp poses. Supports backbones: `pointnet`, `ptv3`, `vit`. Grasp representations: `r3_6d` (9D), `r3_so3` (6D), `r3_euler` (6D). |
| `GraspGenDiscriminator` | `discriminator.py` | Scores grasp quality given object point cloud + candidate grasps. Uses on-generator training. |
| `M2T2` | `m2t2.py` | End-to-end pick-and-place: scene encoder (PointNet2MSG) + object encoder + contact decoder + action decoder. |
| `GraspGen` | `grasp_gen.py` | Combined wrapper: runs generator then discriminator in a single pipeline. |

### Key source layout

```
grasp_gen/
  models/          # Generator, Discriminator, M2T2, pointnet2/ptv3/vit backbones
  dataset/         # dataset.py (training data loader + h5 caching), dataset_utils.py,
                   #   eval_utils.py (collision checking), webdataset_utils.py,
                   #   suction.py (suction data gen), renderer.py
  serving/         # zmq_server.py, zmq_client.py — ZMQ REP/REQ for remote inference
  grasp_server.py  # GraspGenSampler: loads models, exposes infer() for ZMQ server
  robot.py         # GripperInfo dataclass, URDF loading, get_gripper_info(), control points
  utils/           # math_utils, so3.py, rotation_conversions, train_utils, meshcat_utils,
                   #   viser_utils, plot_utils, point_cloud_utils
  metrics.py       # compute_metrics_given_two_sets_of_poses, compute_recall
scripts/
  train_graspgen.py    # Main training entrypoint (Hydra config, DDP, h5 caching)
  train_m2t2.py        # M2T2 training
  inference_graspgen.py # Core inference (scene/mesh → grasps, collision checking)
  demo_object_pc.py    # Demo: segmented object point cloud → grasps
  demo_object_mesh.py  # Demo: mesh (.obj/.stl/.ply/.usd) → grasps
  demo_scene_pc.py     # Demo: scene point cloud → grasps (with optional collision filtering)
  config.yaml          # Hydra config for data/model/training hyperparameters
client-server/
  graspgen_server.py   # Standalone ZMQ inference server (argparse, no Hydra)
  graspgen_client.py   # Client: mesh file → ZMQ request → grasps (no CUDA needed)
mcp/                   # MCP server for LLM tool-calling GraspGen
```

### Configuration & Grippers

- **Gripper YAML configs** live in `config/grippers/` with companion `.py` files that define per-gripper parameters (e.g., `franka_panda`, `robotiq_2f_140`, `single_suction_cup_30mm`).
- Training uses **Hydra/OmegaConf** (`scripts/config.yaml`). Inference via the ZMQ server uses OmegaConf directly to load the gripper config.
- Checkpoints are downloaded separately from HuggingFace (`git clone https://huggingface.co/adithyamurali/GraspGenModels`). Checkpoint paths in gripper `.yml` files are resolved relative to the config file location.

### Data pipeline

- Dataset: HuggingFace (`nvidia/PhysicalAI-Robotics-GraspGen`) — 57M grasps across 3 grippers and 8515 Objaverse XL objects.
- Training: `train_graspgen.py` first builds an h5 cache from the grasp/object datasets, then trains. Cache is reused across runs.
- Train/val/test splits are in `splits/<gripper_name>/train.txt`.
- `GraspGenDataset` in `dataset/dataset.py` loads object meshes (simplified trimesh), samples point clouds, and applies domain randomization (viewpoint redundancy = `NUM_REDUNDANT_DATAPOINTS`).
- `prob_point_cloud` controls partial vs. complete point cloud sampling for sim2real robustness.

### Key constraints

- **Python 3.10 only** (PyTorch 2.1.0 compatibility).
- **pointnet2_ops** must be installed separately via `install_pointnet.sh` or `install_uv_pointnet.sh` — it is a local CUDA extension in `pointnet2_ops/` that cannot be installed by pip.
- PTV3 backbone does **not** work on CUDA 12.8 / Blackwell GPUs due to an upstream dependency issue. Use `pointnet` backbone instead.
- Training was validated on V100, A100, H100, L40s; recommends 8×A100 for full training runs (3K epochs: ~40h generator, ~90h discriminator).
- CUDA arch `8.6` is the default for `pointnet2_ops` compilation (`TORCH_CUDA_ARCH_LIST="8.6"`). Set env vars `CC`, `CXX`, `CUDAHOSTCXX` to `g++` before building.

### Running inference

```bash
# Object mesh → grasps
python scripts/demo_object_mesh.py --mesh_file <file>.obj --gripper_config <gripper>.yml

# Object point cloud → grasps
python scripts/demo_object_pc.py --sample_data_dir <dir> --gripper_config <gripper>.yml

# Scene point cloud → grasps (with collision filtering)
python scripts/demo_scene_pc.py --filter_collisions --sample_data_dir <dir> --gripper_config <gripper>.yml

# ZMQ server (remote inference, no CUDA on client)
python client-server/graspgen_server.py --gripper_config <gripper>.yml --port 5556
python client-server/graspgen_client.py --mesh_file <file>.obj --host localhost --port 5556
```

### Training

```bash
# Docker required for training
bash docker/build.sh
bash docker/run.sh <path_to_code> --grasp_dataset <path> --object_dataset <path> --results <path>

# Inside container
cd /code && bash runs/train_graspgen_franka_panda_gen.sh    # Generator
cd /code && bash runs/train_graspgen_franka_panda_dis.sh    # Discriminator

# Monitor: reconstruction/error_trans_l2 (gen, → few cm), val AP > 0.8 (disc)
tensorboard --logdir=<LOG_DIR>
```

To train with custom arguments outside Docker, use Hydra CLI overrides:
```bash
python scripts/train_graspgen.py \
    data.root_dir=<splits_dir> data.gripper_name=franka_panda \
    train.model_name=diffusion train.num_gpus=1 train.debug=True
```

### USD / Omniverse workflow

1. `scripts/convert_obj_to_usd.py` — OBJ → USD
2. `scripts/demo_object_mesh.py` — inference on USD mesh, save grasps to YAML
3. `scripts/save_grasps_to_usd.py` — write grasps + wireframe viz into USD
4. `scripts/create_grasp_sim_usd.py` — build multi-env sim USD for Isaac Sim playback
5. `scripts/run_grasp_sim_omniverse.py` — script to run inside Isaac Sim to execute grasps
