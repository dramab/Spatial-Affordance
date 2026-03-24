# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

端到端 3D Visual Grounding 项目：输入 RGB 图像 + 深度图 + 文字 prompt，通过放置规划算法生成物体的合法放置位置，最终目标是训练一个输出 3D bounding box（cx, cy, cz, l, w, h, yaw）的模型。

---

## 常用命令

### 安装依赖
```bash
pip install -e .
pip install -r requirements.txt
```

### 放置规划（核心功能，目前在用）
```bash
# 单场景单帧
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --scene /path/to/scene_0001 \
    --frame 0000 \
    --output outputs/placement

# 串行批量处理
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --batch \
    --output outputs/placement

# 并行批量（8 workers）
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --batch --workers 8 \
    --output outputs/placement

# GPU 加速（CuPy FFT）
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --batch --gpu \
    --output outputs/placement

# 查看处理状态（显示 completed/processing/failed 统计）
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --status \
    --output outputs/placement

# 重试之前失败的帧（包括被 OOM kill 的）
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --batch --retry-failed \
    --output outputs/placement

# 强制重新处理所有帧（覆盖已有结果）
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --batch --force \
    --output outputs/placement

# 清除所有失败状态
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --clear-failed \
    --output outputs/placement
```

### 训练（Hydra 配置系统）
```bash
python scripts/train.py experiment=baseline
python scripts/train.py experiment=baseline batch_size=4 optimizer.lr=1e-4
python scripts/train.py experiment=baseline +resume=outputs/checkpoints/last.pth
torchrun --nproc_per_node=4 scripts/train.py experiment=baseline  # 多卡
```

### 自动标注
```bash
python scripts/run_annotation.py \
    --config configs/annotation/auto_label.yaml \
    --input_dir data/raw \
    --output_file data/annotations/train.json \
    --num_workers 4
```

### 测试
```bash
pytest tests/
pytest tests/test_dataset.py  # 单文件
pytest tests/test_dataset.py::test_specific_function  # 单函数
```

---

## 架构与代码结构

### 两大子系统

**子系统 A：放置规划（已完整实现，正在运行）**
入口：`tools/run_placement.py` → 配置：`configs/annotation/placement.yaml`
核心代码：`src/annotation/free_bbox/`

**子系统 B：3D Visual Grounding 模型（脚手架/规划中）**
入口：`scripts/train.py` → 配置：`configs/experiments/baseline.yaml`
核心代码：`src/models/`

---

### 放置规划流程（6步）

```
Adapter.load_scene() → SceneData
    ↓
PlacementPipeline.run(scene_data)
    1. depth_to_pointcloud()       深度图 → 世界坐标系彩色点云
    2. build_occupancy_grid()      点云 → 3D 体素占据格（FREE/OCCUPIED/UNKNOWN）
    3. prepare_grid_base()         所有物体 OBB 标记为 OCCUPIED
    4. Per-object loop:
        - detect_support_surfaces()  RANSAC 平面拟合 → 选取最近支撑面
        - find_table_placements()    FFT 2D卷积碰撞检测，遍历 24 个 yaw 角
        - filter_stable_placements() 足迹必须在支撑面上
        - filter_visible_placements() OBB 投影必须在图像内
        - filter_occluded_placements() Z-buffer 遮挡检查
        - cluster_placements()       DBSCAN 聚类，每簇选最优代表
    5/6. 保存 JSON/PLY/可视化
```

### 关键文件说明

| 文件 | 职责 |
|------|------|
| `src/annotation/free_bbox/datatypes.py` | 全局数据契约：`SceneData`, `CameraParams`, `ObjectInfo`, `PlacementResult` |
| `src/annotation/free_bbox/pipeline.py` | `PlacementPipeline`：编排完整 6 步流程 |
| `src/annotation/free_bbox/occupancy.py` | 深度图→点云+射线投射体素占据格 |
| `src/annotation/free_bbox/surface.py` | RANSAC 支撑面检测，选取与目标物体最近的候选面 |
| `src/annotation/free_bbox/collision.py` | FFT 碰撞检测，可选 CuPy GPU 加速 |
| `src/annotation/free_bbox/filters.py` | 可见性/稳定性/遮挡三类过滤器 |
| `src/annotation/free_bbox/cluster.py` | DBSCAN 聚类，按自由空间分值选代表 |
| `src/annotation/free_bbox/state_tracker.py` | OOM Kill 容错：状态跟踪文件管理 |
| `src/datasets/base_adapter.py` | `DatasetAdapter` 抽象基类 |
| `src/datasets/hope_adapter.py` | HOPE-Video 数据集适配器（深度单位：uint16 mm × 0.98042517 / 10 → cm） |
| `src/datasets/housecat6d_adapter.py` | HouseCat6D 数据集适配器（深度单位：uint16 mm × 0.1 → cm） |

### 数据集适配器扩展

新增数据集只需继承 `DatasetAdapter` 并实现 `load_scene()` 和 `list_scenes()`，所有单位转换和文件格式处理都封装在适配器内。

**切换数据集配置**：
- HOPE-Video: `configs/annotation/placement.yaml` (`dataset.type: hope`)
- HouseCat6D: `configs/annotation/placement_housecat6d.yaml` (`dataset.type: housecat6d`)

### 并行处理

`run_placement.py` 使用 `multiprocessing.ProcessPoolExecutor`（spawn 上下文）。每个 worker 独立重建 config、adapter 和 pipeline，无需共享状态。

### GPU 加速

FFT 碰撞检测（`collision.py`）：有 CuPy 时自动使用 `cupyx.scipy.signal.fftconvolve`，否则降级至 NumPy/SciPy，行为一致。

---

## 数据规格

- 深度图单位：存储为 uint16 (mm)，适配器统一转换为 cm
- 外参平移：存储为 m，适配器统一转换为 cm
- 体素大小默认：1.0 cm
- Yaw 搜索步数：24 步（15°/步）
- 安全边距：2 cm

---

## 输出结构

```
outputs/hope_placement/
├── placements/{prefix}.json       # 每物体放置标注
├── samples/{prefix}.json          # 训练用平铺样本
├── point_clouds/{prefix}.ply      # 带色彩点云
├── occupancy_grids/{prefix}.npy   # 占据格体素
├── grid_meta/{prefix}.json        # 体素格元数据
├── visualizations/{prefix}.png    # 双视图可视化（2D+3D）
└── frame_status.json              # 处理状态跟踪（processing/completed/failed）
```
## 代码运行环境
- 使用conda 激活spatial环境运行代码
