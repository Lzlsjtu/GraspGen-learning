# 6-DOF 抓取位姿估计 — 完整数学原理与数据流

> [作业新增文件] 此文件为 6-DOF 抓取位姿估计课程作业而创建，非 GraspGen 原始项目文件。
> 本文档从数学原理层面，完整阐述从 RGB-D 图像输入到 6-DOF 抓取位姿输出的全流程。

---

## 1. 总数据流概览

```
RGB 图像 (H×W×3)                      Depth 图像 (H×W)
       │                                     │
       └───────────┬─────────────────────────┘
                   │  相机内参 K(fx, fy, cx, cy)
                   ▼
      ┌───────────────────────┐
      │  §2 像素 → 相机坐标    │  反向投影
      │  (u,v,d) → (Xc,Yc,Zc) │
      └───────────┬───────────┘
                  │
                  ▼  (N, 3) 相机坐标系点云
      ┌───────────────────────┐
      │  §3 点云预处理         │  SOR 离群点剔除
      │  → 去噪              │  FPS 均匀降采样
      │  → 降采样 → 1024 点  │  中心化 & 缩放
      └───────────┬───────────┘
                  │
                  ▼  (B, 1024, 3)
      ┌───────────────────────┐
      │  §4 PointNet++ 编码器 │  SA Module ×3
      │  → 层次化特征聚合     │  FPS → Ball Query
      │  → 全局特征 512 维    │  → Shared MLP → MaxPool
      └───────────┬───────────┘
                  │
                  ▼  (B, 512)  object_embedding
      ┌───────────────────────┐
      │  §5 DDPM 扩散生成     │  噪声预测网络
      │  g_T ~ N(0,I)         │  Diffusion Head =
      │  → ε_θ(g_t, t, emb)   │  timestep_enc + sample_enc
      │  → 逐步去噪 T=10 步   │  + prediction_head
      └───────────┬───────────┘
                  │
                  ▼  (B×K, 6)  r3_so3 = [tx, ty, tz, ωx, ωy, ωz]
      ┌───────────────────────┐
      │  §6 旋转 & 位姿解码   │  so(3) Exp Map
      │  → 6D 表示 → R(3×3)  │  Gram-Schmidt (r3_6d)
      │  → 4×4 齐次矩阵      │  [R t; 0 1]
      └───────────┬───────────┘
                  │
                  ▼  (B×K, 4, 4) 齐次位姿矩阵
      ┌───────────────────────┐
      │  §7 碰撞检测 & 后处理 │  夹爪网格 → 采样点
      │  → 物理合理性验证     │  变换到抓取坐标系
      │  → 抓取质量排序       │  → 最短距离判碰撞
      └───────────────────────┘
```

---

## 2. 从像素到点云：相机投影的数学原理

### 2.1 针孔相机模型

相机内参矩阵 \(K\) 建立了**像素坐标系**与**相机坐标系**之间的映射关系：

\[
K = \begin{bmatrix}
f_x & 0   & c_x \\
0   & f_y & c_y \\
0   & 0   & 1
\end{bmatrix}
\]

其中：
- \(f_x, f_y\)：以像素为单位的焦距，\(f_x = \frac{f}{pixel\_width}, f_y = \frac{f}{pixel\_height}\)
- \(c_x, c_y\)：光心在图像平面上的投影坐标（通常约为图像宽度/高度的一半）

### 2.2 FOV 与内参的转换关系（代码：`renderer.py:79`）

从视场角 FOV 计算焦距：

\[
f_x = \frac{W}{2 \cdot \tan(\text{FOV}/2)}, \quad f_y = \frac{H}{2 \cdot \tan(\text{FOV}/2)}
\]

### 2.3 反向投影：深度图 → 三维点云（代码：`renderer.py:93` `depth2points`）

对于深度图像中的每个像素 \((u, v)\)，其深度值为 \(d = D(u, v)\)：

\[
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} =
\begin{bmatrix}
\frac{u - c_x}{f_x} \cdot d \\[4pt]
\frac{v - c_y}{f_y} \cdot d \\[4pt]
d
\end{bmatrix}
\]

**推导**：由针孔模型投影公式

\[
d \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
\quad \Rightarrow \quad
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = Z_c \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
\]

因为 \(Z_c = d\)（深度值即相机坐标系下的 Z 坐标），代入得上述公式。

**代码位置**：`grasp_gen/utils/point_cloud_utils.py:141` 的 `depth_and_segmentation_to_point_clouds()` 调用了 `renderer.py:93` 的 `depth2points()`，完成整张深度图的反向投影。

### 2.4 实例分割掩膜

利用实例分割掩膜 \(S(u, v) \in \{0, 1, ..., N\}\) 提取目标物体的点云：

\[
\mathcal{P}_{\text{obj}} = \{(X_c, Y_c, Z_c) \mid S(u, v) = \text{target\_id}\}
\]

---

## 3. 点云预处理的数学方法

### 3.1 SOR (Statistical Outlier Removal) 离群点剔除（代码：`point_cloud_utils.py:53`）

对于点云中的每个点 \(p_i\)，计算其到 \(K\) 个最近邻的平均距离：

\[
\bar{d}_i = \frac{1}{K} \sum_{p_j \in \text{KNN}(p_i, K)} \|p_i - p_j\|_2
\]

若 \(\bar{d}_i > \tau_{\text{outlier}}\)，则将 \(p_i\) 标记为离群点并移除。

**实现细节**（`point_cloud_utils.py:26` `knn_points`）：
- 使用 `torch.cdist` 计算成对距离矩阵 \(D \in \mathbb{R}^{N \times N}\)
- 通过 `torch.topk` 取第 2 到第 K+1 小的距离（排除自身）
- 默认参数：\(K = 20\)，\(\tau_{\text{outlier}} = 0.014\)（14 mm）

### 3.2 FPS (Farthest Point Sampling) 均匀降采样

FPS 保证采样点在空间分布上的均匀性，避免点云密度不均匀的影响：

**算法**：
1. 随机选择初始点 \(s_0\)
2. 对于 \(i = 1, 2, ..., M\)：
   \[
   s_i = \arg\max_{p \notin \{s_0,...,s_{i-1}\}} \min_{j < i} \|p - s_j\|_2
   \]

FPS 的结果是 M 个均匀分布在物体表面的关键点。

**代码依赖**：`torch_cluster.fps`（在 `grasp_gen/dataset/dataset.py` 的 `collate` 函数中调用）

### 3.3 点云中心化与尺度缩放

将点云平移使均值为零，并乘以缩放因子 \(\kappa\)（GraspGen 中 \(\kappa = 3.27\)）：

\[
p_i' = \kappa \cdot (p_i - \bar{p}), \quad \bar{p} = \frac{1}{N} \sum_{i=1}^{N} p_i
\]

缩放因子 \(\kappa\) 的作用是将点云坐标映射到网络友好的数值范围（~[-1, 1]），有助于梯度传播的数值稳定性。

---

## 4. PointNet++ 特征编码的数学原理

### 4.1 Set Abstraction (SA) 模块

PointNet++ 的核心操作是 **Set Abstraction (SA)**，包含三个子步骤：

**步骤 1：采样 (Sampling)**

使用 FPS 从 \(N\) 个点中选取 \(N'\) 个关键点（称为 centroid）：
\[
\{c_1, ..., c_{N'}\} = \text{FPS}(\mathcal{P}, N')
\]

**步骤 2：分组 (Grouping)**

以每个 centroid \(c_j\) 为球心，在半径 \(r\) 内做 Ball Query，选取最多 \(K\) 个邻居：
\[
\mathcal{N}(c_j) = \{p_i \in \mathcal{P} \mid \|p_i - c_j\|_2 < r, |\mathcal{N}(c_j)| \leq K\}
\]

这定义了每个局部的**感受野**。GraspGen 使用多尺度分组（MSG），同时使用多个半径 \([r_1, r_2]\) 捕获不同尺度的局部几何：

\[
\mathcal{N}_m(c_j) = \{p_i \mid \|p_i - c_j\|_2 < r_m\}, \quad m = 1, 2
\]

**步骤 3：PointNet 局部特征提取**

对每个局域中的点，相对于 centroid 做坐标归一化：
\[
p_i' = p_i - c_j, \quad \forall p_i \in \mathcal{N}(c_j)
\]

然后通过 **Shared MLP** 对每个点独立提取特征：
\[
f_i^{(l)} = \text{ReLU}(\text{BN}(\text{Conv2d}(f_i^{(l-1)})))
\]

最后用 **Max Pooling** 聚合邻域内所有点的特征（对称函数，保证排列不变性）：
\[
F_j = \max_{p_i \in \mathcal{N}(c_j)} f_i^{(L)}
\]

**代码位置**：`grasp_gen/models/pointnet/pointnet2_modules.py`：
- `PointnetSAModuleMSG`：多尺度 SA 模块（line 181）
- `QueryAndGroup`：Ball Query 实现（line 209）
- `build_shared_mlp`：Shared MLP 构建（line 35）
- `max(dim=-1)[0]`：Max Pooling 操作（line 94）

### 4.2 三层层次化编码

GraspGen 的 PointNet++ 编码器（`model_utils.py` 中的 `PointNetPlusPlus`）使用三层 SA：

```
Layer 1: npoint=512, radius=0.1, MLP=[3→32→32→64]
  → 输出 (B, 64, 512)
  → 从 1024 点选 512 个关键点，小半径捕获细粒度几何

Layer 2: npoint=128, radius=0.2, MLP=[64→64→128]
  → 输出 (B, 128, 128)
  → 从 512 点选 128 个关键点，中半径捕获部件级几何

Layer 3: npoint=None, GroupAll, MLP=[128→256→512]
  → 输出 (B, 512)
  → 全局聚合，将 128 个半局部特征压缩为单一 512 维向量
```

### 4.3 排列不变性

PointNet++ 通过 **对称函数（Max Pooling）** 保证对输入点云排列的不变性：

\[
f(\{p_1, p_2, ..., p_N\}) = \max_{i} h(p_i)
\]

对任意排列 \(\pi\)，有 \(f(\{p_{\pi(1)}, ..., p_{\pi(N)}\}) = f(\{p_1, ..., p_N\})\)。

这表明无论输入点云以何种顺序排列，提取的全局特征都是相同的——这是处理无序点云的关键性质。

---

## 5. 扩散去噪过程的数学原理（DDPM）

### 5.1 前向扩散过程

将抓取位姿 \(g_0 \in \mathbb{R}^d\)（\(d = 6\) 或 \(d = 9\) 取决于位姿表示）逐步添加高斯噪声：

\[
q(g_t \mid g_{t-1}) = \mathcal{N}(g_t; \sqrt{1-\beta_t} \cdot g_{t-1}, \beta_t \mathbf{I})
\]

其中 \(\{\beta_t\}_{t=1}^{T}\) 是噪声调度（noise schedule），控制每步添加的噪声量。

重参数化技巧，可以直接从 \(g_0\) 采样任意 \(t\) 步的带噪版本：

\[
q(g_t \mid g_0) = \mathcal{N}(g_t; \sqrt{\bar{\alpha}_t} \cdot g_0, (1-\bar{\alpha}_t) \mathbf{I})
\]

其中 \(\alpha_t = 1 - \beta_t\)，\(\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s\)。

等价地：
\[
g_t = \sqrt{\bar{\alpha}_t} \cdot g_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \mathbf{I})
\]

### 5.2 逆向去噪过程（DDPM 反向过程）

从纯噪声 \(g_T \sim \mathcal{N}(0, \mathbf{I})\) 开始，逐步去噪恢复 \(g_0\)：

\[
p_\theta(g_{t-1} \mid g_t, pc) = \mathcal{N}(g_{t-1}; \mu_\theta(g_t, t, pc), \sigma_t^2 \mathbf{I})
\]

其中 \(pc\) 是物体点云（作为条件），\(\mu_\theta\) 由噪声预测网络参数化。

### 5.3 训练目标：噪声预测

训练一个神经网络 \(\varepsilon_\theta\) 在给定点云条件下预测添加的噪声：

\[
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, g_0, \varepsilon}\left[\|\varepsilon - \varepsilon_\theta(\sqrt{\bar{\alpha}_t} \cdot g_0 + \sqrt{1-\bar{\alpha}_t} \cdot \varepsilon, t, pc)\|^2\right]
\]

其中 \(t \sim \text{Uniform}(0, T-1)\)，\(\varepsilon \sim \mathcal{N}(0, \mathbf{I})\)。

这就是 GraspGen Generator 的核心训练损失。

**代码位置**：`grasp_gen/models/generator.py:268` `forward_train()`：
- `noise = torch.randn(...)` — 采样随机噪声（line 302）
- `timesteps = torch.randint(...)` — 随机采样时间步（line 304）
- `noisy_grasps = self.noise_scheduler.add_noise(grasps_gt, noise, timesteps)` — 前向加噪（line 362）
- `noise_pred = self.diffusion_head(object_embedding, timesteps, samples)` — 噪声预测（line 365）

### 5.4 推理时的去噪迭代（代码：`generator.py:407` `forward_inference`）

从 \(g_T \sim \mathcal{N}(0, \mathbf{I})\) 开始，对 \(t = T, T-1, ..., 1\)：

```
noise_pred = ε_θ(g_t, t, pc)           # 预测噪声
g_{t-1} = scheduler.step(noise_pred, t, g_t)  # DDPM 采样步
```

GraspGen 中 \(T = 10\)（训练和推理扩散步数相同）。

### 5.5 组合调度器（Compositional Scheduler）

GraspGen 对**平移**和**旋转**分量使用不同的噪声调度（`compositional_schedular=True`）：

- **平移** (`noise_scheduler_pos`)：`beta_schedule='scaled_linear'` — 线性噪声增长
- **旋转** (`noise_scheduler_rot`)：`beta_schedule='squaredcos_cap_v2'` — 余弦噪声调度

这源于平移空间（\(\mathbb{R}^3\)，欧几里得）与旋转空间（\(SO(3)\)，非欧几里得）的几何差异。

---

## 6. 旋转表示与 3D 几何数学

### 6.1 为什么旋转表示很重要

神经网络直接回归 \(SO(3)\) 元素面临**不连续性**问题：四元数存在 double-cover（\(q\) 与 \(-q\) 表示同一旋转），欧拉角存在万向节锁（Gimbal Lock），axis-angle 在 \(2\pi\) 处不连续。

### 6.2 6D 连续旋转表示（Zhou et al., CVPR 2019）

**编码（\(SO(3) \to \mathbb{R}^6\)）**：取旋转矩阵的前两列并展平：

\[
g_{\text{enc}}(R) = [R_{:,0}; R_{:,1}] \in \mathbb{R}^6
\]

**解码（\(\mathbb{R}^6 \to SO(3)\)）**：使用 Gram-Schmidt 正交化：

\[
\begin{aligned}
b_1 &= \frac{a_1}{\|a_1\|} \\
b_2 &= \frac{a_2 - (b_1 \cdot a_2) b_1}{\|a_2 - (b_1 \cdot a_2) b_1\|} \\
b_3 &= b_1 \times b_2
\end{aligned}
\]

其中 \(a_1 = g_{\text{enc}}[0:3]\)，\(a_2 = g_{\text{enc}}[3:6]\)。

**关键性质**：6D 表示在整个 \(SO(3)\) 上是**连续**的，任何两个接近的旋转在 6D 空间中也是接近的，这使得神经网络可以用标准的 L1/L2 损失进行回归。

**代码位置**：`grasp_gen/utils/math_utils.py:76` `rotation_6d_to_matrix()`

### 6.3 so(3) 李代数表示（GraspGen 的 `r3_so3` 模式）

**编码（\(SO(3) \to \mathfrak{so}(3) \simeq \mathbb{R}^3\)）**：

旋转矩阵通过**对数映射**转换为轴角向量：

\[
\omega = \text{Log}(R) = \frac{\theta}{2 \sin\theta}(R - R^T)^\vee
\]

其中 \(\theta = \arccos\left(\frac{\text{tr}(R) - 1}{2}\right)\) 是旋转角，\((\cdot)^\vee\) 是 hat 算子的逆。

GraspGen 将轴角缩放到 \([-1, 1]\)：
\[
g_{\text{enc}}(R) = \frac{\omega}{\pi}
\]

**解码（\(\mathfrak{so}(3) \to SO(3)\)）**：

通过**指数映射**（Rodrigues 公式）：

\[
R = \exp(\omega^\wedge) = I + \frac{\sin\theta}{\theta} \omega^\wedge + \frac{1-\cos\theta}{\theta^2} (\omega^\wedge)^2
\]

其中 \(\omega^\wedge\) 是轴角向量的反对称矩阵（hat 算子）：

\[
\omega^\wedge = \begin{bmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{bmatrix}
\]

**代码位置**：
- `grasp_gen/utils/so3.py:16` `hat()`
- `grasp_gen/utils/so3.py` `so3_log_map()` / `so3_exp_map()`
- `grasp_gen/utils/math_utils.py:22` `matrix_to_rt()` / `rt_to_matrix()`

### 6.4 抓取位姿的 4×4 齐次表示

最终从 \((t, R)\) 组装为齐次变换矩阵：

\[
T_{\text{grasp}} = \begin{bmatrix}
R_{3\times 3} & t_{3\times 1} \\
0_{1\times 3} & 1
\end{bmatrix} \in SE(3)
\]

该矩阵表示从**夹爪基座坐标系**到**物体坐标系**的刚体变换。

### 6.5 旋转测地线损失（Geodesic Loss）

训练时的旋转损失使用测地线距离（`grasp_gen/metrics.py:205` `GeodesicLoss`）：

\[
\mathcal{L}_{\text{rot}}(R_{\text{pred}}, R_{\text{gt}}) = \arccos\left(\frac{\text{tr}(R_{\text{pred}} R_{\text{gt}}^T) - 1}{2}\right)
\]

这是 \(SO(3)\) 上的内禀距离度量，范围在 \([0, \pi]\) 之间。

---

## 7. 扩散头（Diffusion Head）的内部结构

### 7.1 三路信息融合

```
 noisy_grasp    timestep t    object_embedding
      │              │              │
      ▼              ▼              │
 SampleEncoder  TimestepEncoder     │
 [Linear→ReLU   [SinusoidalPosEmb  │
  →Linear]      →MLP→(B,512)]     │
      │              │              │
      (B,512)       (B,512)       (B,512)
      │              │              │
      └──────────────┼──────────────┘
                     │
                     ▼
            concatenate → (B, 1536)
                     │
                     ▼
            Prediction Head
            [1536→768→384→6/9]
                     │
                     ▼
                 noise_pred
```

### 7.2 正弦位置编码（`model_utils.py:71` `SinusoidalPosEmb`）

将离散时间步 \(t\) 编码为连续向量，使得网络能感知扩散进度：

\[
\text{PE}(t, 2i) = \sin\left(t \cdot \exp\left(-\frac{2i \cdot \log 10000}{d}\right)\right)
\]
\[
\text{PE}(t, 2i+1) = \cos\left(t \cdot \exp\left(-\frac{2i \cdot \log 10000}{d}\right)\right)
\]

这与 Transformer 的位置编码原理相同，通过不同频率的正弦波将标量时间映射到高维空间。

---

## 8. 碰撞检测的数学原理

### 8.1 基于点云的碰撞检测（代码：`point_cloud_utils.py:237` `filter_colliding_grasps`）

**步骤 1**：在夹爪碰撞网格表面均匀采样 \(N_s\) 个点：
\[
\{s_1, ..., s_{N_s}\} \sim \text{Uniform}(\text{MeshSurface})
\]
默认 \(N_s = 2000\)。

**步骤 2**：将每个采样点变换到预测抓取位姿下的世界坐标：
\[
s_i' = T_{\text{grasp}} \cdot s_i
\]

**步骤 3**：计算每个夹爪采样点到场景点云的最近距离：
\[
d_i = \min_{p_j \in \mathcal{P}_{\text{scene}}} \|s_i' - p_j\|_2
\]

**步骤 4**：判定碰撞：
\[
\text{collision} = \begin{cases}
\text{True} & \text{if } \exists i: d_i < \tau_{\text{collision}} \\
\text{False} & \text{otherwise}
\end{cases}
\]
默认 \(\tau_{\text{collision}} = 0.002\)（2 mm）。

### 8.2 抓取闭塞区域（Grasp Closure Region）分析

平行夹爪的有效抓取需要夹爪闭合区域内存在物体的物理接触点。通过检查夹爪两个手指之间的区域是否包含物体点云来判断抓取是否有效。

---

## 9. 训练过程与损失函数总结

### 9.1 联合损失函数

GraspGen Generator 的完整训练目标（`generator.py:376-396`）：

\[
\mathcal{L}_{\text{total}} = w_1 \cdot \mathcal{L}_{\text{diffusion}} + w_2 \cdot \mathcal{L}_{\text{translation}} + w_3 \cdot \mathcal{L}_{\text{rotation}}
\]

| 损失项 | 公式 | 权重 | 代码变量 |
|--------|------|------|----------|
| 扩散噪声预测 | \(\mathbb{E}[\|\varepsilon - \hat{\varepsilon}\|^2]\) | 2.0 | `loss_pointmatching` / `noise_pred` |
| 平移 L1 | \(\|t_{\text{pred}} - t_{\text{gt}}\|_1\) | 1.0 | `loss_l1_pos` / `position_loss` |
| 旋转 L1 | \(\|r_{\text{pred}} - r_{\text{gt}}\|_1\) | 1.0 | `loss_l1_rot` / `rotation_loss` |

### 9.2 优化器配置

- AdamW 优化器，学习率 \(1 \times 10^{-5}\)
- 权重衰减（L2 正则化） 0.05
- 无梯度裁剪（`grad_clip = -1`）

---

## 10. 代码到数学的完整映射表

| 数学概念 | 代码文件 | 函数/类 |
|----------|----------|---------|
| 针孔相机投影 | `grasp_gen/dataset/renderer.py:93` | `depth2points()` |
| FOV → 内参 | `grasp_gen/dataset/renderer.py:79` | `fov_and_size_to_intrinsics()` |
| SOR 离群点剔除 | `grasp_gen/utils/point_cloud_utils.py:53` | `point_cloud_outlier_removal()` |
| KNN 距离计算 | `grasp_gen/utils/point_cloud_utils.py:26` | `knn_points()` |
| PointNet++ SA | `grasp_gen/models/pointnet/pointnet2_modules.py:181` | `PointnetSAModuleMSG` |
| Ball Query 分组 | `grasp_gen/models/pointnet/pointnet2_utils.py` | `QueryAndGroup` |
| Shared MLP | `grasp_gen/models/pointnet/pointnet2_modules.py:35` | `build_shared_mlp` |
| 前向扩散加噪 | `grasp_gen/models/generator.py:352` | `noise_scheduler.add_noise()` |
| 噪声预测 | `grasp_gen/models/generator.py:365` | `diffusion_head(...)` |
| DDPM 逆向采样 | `grasp_gen/models/generator.py:486` | `forward_inference()` 循环体 |
| 正弦时间编码 | `grasp_gen/models/model_utils.py:71` | `SinusoidalPosEmb` |
| 6D 旋转 → SO(3) | `grasp_gen/utils/math_utils.py:76` | `rotation_6d_to_matrix()` |
| SO(3) → 6D | `grasp_gen/utils/math_utils.py:102` | `matrix_to_rotation_6d()` |
| so(3) Log Map | `grasp_gen/utils/so3.py` | `so3_log_map()` |
| so(3) Exp Map | `grasp_gen/utils/so3.py` | `so3_exp_map()` |
| Hat 算子 \(\wedge\) | `grasp_gen/utils/so3.py:16` | `hat()` |
| 位姿 RT ↔ 向量 | `grasp_gen/utils/math_utils.py:22,49` | `matrix_to_rt()` / `rt_to_matrix()` |
| 测地线损失 | `grasp_gen/metrics.py:205` | `GeodesicLoss` |
| 碰撞检测 | `grasp_gen/utils/point_cloud_utils.py:237` | `filter_colliding_grasps()` |
| 平移+旋转评估 | `grasp_gen/metrics.py:40` | `compute_metrics_given_two_sets_of_poses()` |
