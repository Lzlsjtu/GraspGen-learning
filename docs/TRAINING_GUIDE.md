# GraspGen 训练流程指南

## 一、项目概述

GraspGen 是 NVIDIA 的 6-DOF 机械臂抓取生成框架，基于扩散模型（Diffusion Model）。核心功能是从点云数据生成高质量的 6 自由度抓取姿态。

**支持的三种模型架构：**
1. **diffusion** - 扩散模型作为生成器
2. **discriminator** - 判别器用于评分和排序生成的抓取
3. **m2t2** - 端到端的抓取和放置模型

---

## 二、核心计算方法

### 2.1 数据流

```
点云输入 → 编码器 → 特征融合 → 扩散/解码 → 抓取姿态输出
```

### 2.2 模型架构

#### GraspGenGenerator (扩散模型)

- **输入**: 物体点云 (N×3)
- **输出**: 抓取姿态 (N_grasps × 9)

**关键组件:**
1. **Object Encoder** - 编码点云特征
   - 支持: `pointnet`, `ptv3`, `vit`
2. **Diffusion Head** - 噪声预测网络
3. **Grasp Repr** - 抓取表示
   - `r3_6d`: 9维 (3位置 + 6旋转矩阵)
   - `r3_so3`: 6维 (3位置 + 3轴角)
   - `r3_euler`: 6维 (3位置 + 3欧拉角)

#### GraspGenDiscriminator (判别器)

- **输入**: 物体点云 + 候选抓取
- **输出**: 抓取质量分数

#### M2T2 (端到端模型)

- **Scene Encoder**: 场景点云编码
- **Object Encoder**: 物体点云编码
- **Contact Decoder**: 接触区域解码
- **Action Decoder**: 动作解码

---

## 三、环境配置

### 3.1 依赖安装

```bash
# 创建虚拟环境
uv venv --python 3.10 .venv && source .venv/bin/activate

# 安装依赖
uv pip install -e .

# 安装 pointnet2_ops (关键步骤)
./install_pointnet.sh

# 验证安装
python tests/test_inference_installation.py
```

### 3.2 必需的 CUDA 环境

- CUDA 12.1 或 12.8
- PyTorch 2.1.0
- CUDA 架构 8.6 (如需编译 pointnet2_ops)

---

## 四、数据集准备

### 4.1 下载抓取数据集

```bash
git clone https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GraspGen
```

### 4.2 下载物体网格数据集

```bash
python scripts/download_objects.py --uuid_list <path_to_grasp_dataset>/splits/franka_panda/ --output_dir <path_to_object_dataset> --simplify
```

### 4.3 数据集目录结构

```
/path/to/grasp_dataset/
├── grasp_data/
│   ├── franka_panda/
│   ├── robotiq_2f_140/
│   └── single_suction_cup_30mm/
├── splits/
│   ├── franka_panda/
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   └── test.txt
│   └── ...
```

---

## 五、运行训练

### 5.1 直接运行方式

```bash
cd /home/lzl/Projects/6Dpose/GraspGen/scripts

# 训练 GraspGen Generator (扩散模型)
python train_graspgen.py \
    data.root_dir=<path_to_splits> \
    data.object_root_dir=<path_to_objects> \
    data.grasp_root_dir=<path_to_grasp_data> \
    data.gripper_name="franka_panda" \
    train.model_name="diffusion" \
    diffusion.gripper_name="franka_panda" \
    diffusion.num_diffusion_iters=10 \
    train.num_epochs=500 \
    train.num_gpus=1 \
    train.log_dir=/tmp/graspgen_logs

# 训练 GraspGen Discriminator
python train_graspgen.py \
    data.root_dir=<path_to_splits> \
    data.object_root_dir=<path_to_objects> \
    data.grasp_root_dir=<path_to_grasp_data> \
    data.gripper_name="franka_panda" \
    data.load_discriminator_dataset=True \
    train.model_name="discriminator" \
    discriminator.gripper_name="franka_panda" \
    train.num_epochs=500

# 训练 M2T2 模型
python train_m2t2.py \
    data.root_dir=<path_to_splits> \
    data.gripper_name="franka_panda" \
    m2t2.action_decoder.gripper_name="franka_panda" \
    m2t2.grasp_loss.gripper_name="franka_panda" \
    train.model_name="m2t2"
```

### 5.2 使用 Docker

```bash
# 构建 Docker
bash docker/build.sh

# 运行训练容器
bash docker/run.sh <path_to_code> \
    --grasp_dataset <path_to_grasp_dataset> \
    --object_dataset <path_to_object_dataset> \
    --results <path_to_results>

# 在容器内运行训练
cd /code && bash runs/train_graspgen_robotiq_2f_140_gen.sh
```

### 5.3 使用示例脚本 (runs/)

```bash
# 训练 Robotiq 2F-140 生成器
cd GraspGen
bash runs/train_graspgen_robotiq_2f_140_gen.sh

# 训练 Robotiq 2F-140 判别器
bash runs/train_graspgen_robotiq_2f_140_dis.sh
```

---

## 六、关键参数说明

### 训练参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `train.num_gpus` | GPU数量 | 1-8 |
| `train.num_workers` | 数据加载线程数 | CPU核心数/GPU数 |
| `train.batch_size` | 批次大小 | 8-16 |
| `train.num_epochs` | 训练轮数 | 3000+ |
| `train.lr` | 学习率 | 1e-5 |
| `train.grad_clip` | 梯度裁剪 | -1 (禁用) |

### 数据参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `data.num_points` | 点云点数 | 2048-16384 |
| `data.prob_point_cloud` | 部分点云采样概率 | 0.5 |
| `data.redundancy` | 冗余数据点数量 | 7 |
| `data.num_grasps_per_object` | 每个物体的抓取数 | 100-500 |

### 模型参数

| 参数 | 说明 |
|------|------|
| `diffusion.num_diffusion_iters` | 扩散步数 (训练) |
| `diffusion.num_diffusion_iters_eval` | 扩散步数 (评估) |
| `diffusion.obs_backbone` | 骨干网络: `pointnet`, `ptv3`, `vit` |
| `diffusion.grasp_repr` | 抓取表示: `r3_6d`, `r3_so3`, `r3_euler` |

---

## 七、缓存机制

训练脚本会自动构建数据集缓存:

1. **首次运行**: 脚本自动构建 `.h5` 缓存文件
2. **后续运行**: 直接加载缓存开始训练
3. **缓存验证**: 检查 `denylist_*.json` 和 `.h5` 文件完整性

**缓存目录结构:**
```
/cache_dir/
├── denylist_gen.json
├── cache_train_mesh.h5
├── cache_valid_mesh.h5
└── ...
```

---

## 八、训练监控

使用 TensorBoard 监控训练:

```bash
# 启动 TensorBoard
tensorboard --logdir=/tmp/graspgen_logs

# 访问 http://localhost:6006
```

**关键指标:**
- Generator: `reconstruction/error_trans_l2` (验证集，应收敛到几厘米)
- Discriminator: 验证 AP > 0.8

---

## 九、核心代码位置

| 组件 | 文件 |
|------|------|
| 训练脚本 | `scripts/train_graspgen.py`, `scripts/train_m2t2.py` |
| 数据集 | `grasp_gen/dataset/dataset.py` |
| 模型 | `grasp_gen/models/generator.py`, `grasp_gen/models/discriminator.py` |
| 工具 | `grasp_gen/utils/train_utils.py` |
| 配置文件 | `scripts/config.yaml` |

---

## 十、常见问题

### Q1: 训练脚本卡住
- 检查 CPU/内存/GPU 是否充足
- 确认数据集路径正确

### Q2: pointnet2_ops 编译失败
- 确保安装 g++ 和 CUDA 运行时头文件
- 设置 `TORCH_CUDA_ARCH_LIST="8.6"`

### Q3: PTV3 backbone 在 CUDA 12.8 不可用
- 使用 PointNet++ backbone

---

## 十一、运行完整示例

```bash
# 1. 环境准备
cd /home/lzl/Projects/6Dpose/GraspGen
source .venv/bin/activate

# 2. 参数设置
GRASP_DIR="/data/PhysicalAI-Robotics-GraspGen"
OBJECT_DIR="/data/objaverse"
RESULTS_DIR="/tmp/graspgen_train"
SPLIT_DIR="$GRASP_DIR/splits/franka_panda"
GRASP_DATA_DIR="$GRASP_DIR/grasp_data/franka_panda"

# 3. 运行训练
cd scripts
python train_graspgen.py \
    data.root_dir=$SPLIT_DIR \
    data.object_root_dir=$OBJECT_DIR \
    data.grasp_root_dir=$GRASP_DATA_DIR \
    data.gripper_name="franka_panda" \
    data.cache_dir="$RESULTS_DIR/cache" \
    train.log_dir="$RESULTS_DIR/logs" \
    train.model_name="diffusion" \
    train.num_gpus=1 \
    train.num_workers=4 \
    train.batch_size=8 \
    train.num_epochs=100 \
    diffusion.gripper_name="franka_panda" \
    diffusion.num_diffusion_iters=10 \
    diffusion.obs_backbone="pointnet"
```

---

## 十二、配置模板 (config.yaml)

```yaml
data:
  root_dir: '/path/to/splits/franka_panda'
  cache_dir: '/tmp/graspgen_cache'
  object_root_dir: '/path/to/objects'
  grasp_root_dir: '/path/to/grasp_data/franka_panda'
  gripper_name: 'franka_panda'
  num_points: 2048
  prob_point_cloud: 0.5
  redundancy: 7

train:
  model_name: 'diffusion'  # 或 'discriminator', 'm2t2'
  num_gpus: 1
  num_workers: 4
  batch_size: 8
  num_epochs: 500
  log_dir: '/tmp/graspgen_logs'
  lr: 0.00001

diffusion:
  gripper_name: 'franka_panda'
  num_diffusion_iters: 10
  obs_backbone: 'pointnet'
  grasp_repr: 'r3_6d'
```