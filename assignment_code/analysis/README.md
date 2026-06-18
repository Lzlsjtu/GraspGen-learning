# analysis/ — 作业后处理与分析工具

> ⚠️ 此目录下的所有文件均为**作业新增**，非 GraspGen 原始项目文件。
> 这些脚本仅读取训练输出（TensorBoard events、checkpoint、日志），不依赖 `grasp_gen` 模块。

## 文件说明

| 文件 | 用途 | 依赖 |
|------|------|------|
| `monitor_training.py` | 解析 TensorBoard events，生成 loss/误差曲线图 | `tensorboard`, `matplotlib` |
| `plots/` | 自动生成的训练曲线图输出目录 | — |

## 用法

```bash
cd /home/lzl/Projects/6Dpose/GraspGen
source .venv/bin/activate

# 生成训练曲线图
python ../analysis/monitor_training.py \
    --log_dir ./runs/results_assignment/logs \
    --output_dir ../analysis/plots

# 持续监控（每 60 秒更新）
python ../analysis/monitor_training.py \
    --log_dir ./runs/results_assignment/logs \
    --output_dir ../analysis/plots \
    --continuous --interval 60
```
