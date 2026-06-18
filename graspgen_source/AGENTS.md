# GraspGen Agent 指南

## 项目概述

GraspGen 是 NVIDIA 的 6-DOF 机械臂抓取生成框架，基于扩散模型。核心代码在 `grasp_gen/` 目录。

## 目录结构

| 目录 | 用途 |
|------|------|
| `grasp_gen/` | 核心包：models, dataset, serving, utils |
| `scripts/` | 训练和推理脚本 |
| `client-server/` | ZMQ 服务器实现 |
| `mcp/` | LLM 工具调用 (MCP 协议) |
| `config/grippers/` | 夹爪配置文件 |
| `tests/` | 单元测试和集成测试 |
| `docker/` | 训练用 Docker 配置 |

## 安装

```bash
# 推荐：uv 安装（仅推理）
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e .

# 关键：必须单独安装 pointnet2_ops（CUDA 扩展）
./install_pointnet.sh

# 验证安装
python tests/test_inference_installation.py
```

**注意**：`pointnet2_ops` 无法通过 pip 自动安装，必须运行 `install_pointnet.sh` 脚本。

## 推理方式

### 1. 直接 Python
```bash
python scripts/demo_object_mesh.py --mesh_file <path>.obj --gripper_config <config>.yml
```

### 2. ZMQ 服务器（无 GPU 客户端也能调用）
```bash
# 服务端
python client-server/graspgen_server.py --gripper_config <config>.yml --port 5556

# 客户端
python client-server/graspgen_client.py --mesh_file <path>.obj
```

### 3. MCP（LLM 工具调用）
需要先启动 ZMQ 服务器，然后在 `mcp/` 目录下配置 MCP 客户端。

## 训练（仅 Docker）

```bash
# 构建 Docker 镜像
bash docker/build.sh

# 启动容器（需要挂载数据集和模型路径）
bash docker/run.sh <path_to_code> --grasp_dataset <path> --object_dataset <path> --results <path>

# 训练示例
cd /code && bash runs/train_graspgen_robotiq_2f_140_gen.sh
```

训练需要：
- 抓取数据集（从 HuggingFace 下载）
- 物体网格数据集（通过 `scripts/download_objects.py` 下载）
- 多 GPU（官方使用 8x A100）

## 测试

```bash
# 运行所有测试（需要 CUDA）
pytest tests/ -v

# 运行特定测试
pytest tests/test_inference_installation.py -v

# 集成测试（需要 ZMQ 服务器运行在 localhost:5556）
pytest tests/ -v -m integration
```

## 模型下载

检查点需要单独从 HuggingFace 下载：
```bash
git clone https://huggingface.co/adithyamurali/GraspGenModels
```

支持的夹爪：`robotiq_2f_140`, `franka_panda`, `single_suction_cup_30mm`

## 关键配置

- Python 版本：3.10
- PyTorch：2.1.0（需要 CUDA 12.1 或 12.8）
- CUDA 架构：8.6（如需编译 pointnet2_ops）

## 常见问题

- **训练脚本卡住**：检查 Docker 容器 CPU/内存/GPU 是否充足
- **PTV3 backbone 在 CUDA 12.8 不可用**：使用 PointNet++ backbone
- **pointnet2_ops 安装失败**：确保安装了 g++ 和 CUDA 运行时头文件

## 入口脚本

| 脚本 | 用途 |
|------|------|
| `scripts/inference_graspgen.py` | 核心推理 |
| `scripts/train_graspgen.py` | 模型训练 |
| `scripts/demo_*.py` | 可视化演示 |
| `scripts/download_objects.py` | 下载训练物体 |