# 3D 物体放置规划 Pipeline 文档

> 本文档覆盖 `src/annotation/free_bbox/` 模块的处理原理、代码结构、数据接口、输出格式和使用教程，供后续数据集接入参考。

---

## 目录

1. [处理原理](#1-处理原理)
2. [代码结构](#2-代码结构)
3. [模块间数据流](#3-模块间数据流)
4. [接入新数据集](#4-接入新数据集)
5. [输出标注格式](#5-输出标注格式)
6. [使用教程](#6-使用教程)

---

## 1. 处理原理

### 1.1 总体目标

给定一帧 RGBD 图像和场景中物体的 3D 标注（位姿 + 尺寸），为每个物体找到若干**无碰撞、物理稳定、相机可见、无遮挡**的放置位置，输出放置候选的世界坐标和朝向。

### 1.2 流程概览

```
RGBD 图像 + 物体标注
    │
    ▼
[Step 1] 深度图 → 世界坐标点云
    │
    ▼
[Step 2] 点云 → 3D 占据栅格（ray-casting）
    │         FREE=0 / OCCUPIED=1 / UNKNOWN=2
    ▼
[Step 3] 所有物体 OBB 体素化，标记为 OCCUPIED
    │
    ▼
[Step 4] 逐物体处理：
    ├── 可见性预检查（8 角点是否在图像内）
    ├── 目标物体体素化（仅用于结果对齐与可视化）
    ├── 支撑面检测（优先在点云中用 RANSAC 检测水平支撑面，失败时回退到栅格搜索）
    ├── FFT 碰撞搜索（X, Y, θ 配置空间）
    ├── 稳定性过滤（支撑比 + 质心投影）
    ├── 可见性过滤（放置后 8 角点仍在图像内）
    ├── 遮挡过滤（Z-buffer 比较）
    └── DBSCAN 聚类（世界 XY 空间）
    │
    ▼
[Step 5] 输出 PlacementResult（每物体若干聚类代表）
```

### 1.3 占据栅格构建（Ray-Casting）

深度图每个像素反投影为世界坐标 3D 点，同时沿光线方向将相机到该点之间的体素标记为 FREE，该点所在体素标记为 OCCUPIED，未被任何光线穿过的体素为 UNKNOWN。

- 体素坐标系：以点云包围盒为范围，加 `grid_padding` 边界，体素边长 `voxel_size`（场景单位，如 cm）
- 采样步长 `pixel_stride`：每隔 N 个像素取一个深度点，平衡精度与速度

### 1.4 支撑面检测

当前 `surface.py` 的实现会优先使用 Step 1 生成的场景点云 `pts_world` 检测支撑面：

1. 在点云中随机采样 3 点，用 RANSAC 拟合平面
2. 仅保留法向量接近世界 `+Z` 方向的近似水平平面
3. 将平面内点投影回体素网格 XY 平面，做形态学闭运算与连通域筛选
4. 若提供当前物体体素，则在多个候选支撑面中选择与该物体 `3D` 欧氏距离最近的那个候选

若点云拟合失败，则自动退回到占据栅格逐层搜索方案，并沿用相同的“最近支撑面”选择规则。接口仍返回单个 `(table_z, surface_mask)`，但它不再固定表示全局面积最大的平面，而是表示当前物体对应的最近支撑面。

注意：当前支撑面检测默认世界坐标的竖直方向与 `+Z` 对齐。

### 1.5 FFT 碰撞检测

对每个 yaw 角度（共 `yaw_steps` 个离散角度），将旋转后的物体体素投影到 XY 平面得到 2D footprint，与支撑面上方的障碍物 2D 投影做**互相关（FFT 卷积）**，互相关为 0 的位置即无碰撞。

- 安全边距 `safety_margin`：在碰撞检测前对障碍物做 XY 平面膨胀
- 输出：所有 (x_voxel, y_voxel, yaw_idx) 三元组候选
- 当前实现保留目标物体原始占据，因此候选位置不会与物体当前所在位置重合

### 1.6 稳定性过滤

对每个候选位置，检查：
1. 物体底面与支撑面的重叠比例 ≥ `min_support_ratio`
2. 物体质心在 XY 平面的投影落在支撑区域内（防止悬空倾倒）

### 1.7 可见性过滤

将放置后物体的 8 个 OBB 角点投影到图像平面，要求所有角点均在图像范围内（含 `vis_margin_px` 像素边距）。

### 1.8 遮挡过滤

构建场景深度缓冲（Z-buffer）：对每个像素取所有 OCCUPIED 体素中最近的深度值。将放置后物体角点投影到图像，若投影深度远大于 Z-buffer 值（被遮挡比例 > `occlusion_threshold`），则过滤掉。

### 1.9 DBSCAN 聚类

将通过所有过滤的候选点在世界 XY 空间做 DBSCAN 聚类（半径 `dbscan_eps`），每个聚类选取**自由空间得分最高**的候选作为代表，减少输出冗余。

---

## 2. 代码结构

```
src/
├── utils/
│   └── coord_utils.py          # 通用坐标变换
│
├── annotation/
│   ├── free_bbox/              # 放置规划通用模块
│   │   ├── datatypes.py        # 数据类定义（SceneData, CameraParams, ObjectInfo, PlacementConfig, PlacementResult）
│   │   ├── occupancy.py        # 深度图 → 点云 → 占据栅格
│   │   ├── voxel_utils.py      # 体素坐标转换工具
│   │   ├── grid_ops.py         # 栅格操作（体素化、场景占据标记、膨胀）
│   │   ├── surface.py          # 支撑面检测
│   │   ├── collision.py        # FFT 碰撞检测
│   │   ├── filters.py          # 稳定性/可见性/遮挡过滤
│   │   ├── cluster.py          # DBSCAN 聚类
│   │   ├── pipeline.py         # PlacementPipeline 编排
│   │   ├── io_utils.py         # 文件读写（PLY, JSON）
│   │   └── visualize.py        # 可视化
│   │
│   └── bbox3d/
│       ├── bbox_utils.py       # 3D bbox 工具（角点、接触面）
│       └── visualize.py        # bbox 投影可视化
│
└── datasets/
    ├── base_adapter.py         # 抽象基类 DatasetAdapter
    └── hope_adapter.py         # HOPE-Video 数据集适配器

tools/
└── run_placement.py            # CLI 入口

configs/
└── annotation/
    └── placement.yaml          # 参数配置文件
```

### 各模块职责

| 文件 | 核心函数 | 职责 |
|---|---|---|
| `datatypes.py` | `SceneData`, `CameraParams`, `ObjectInfo`, `PlacementConfig` | 统一数据接口定义 |
| `coord_utils.py` | `transform_points`, `project_world`, `rotation_z_3x3` | 坐标变换、投影 |
| `occupancy.py` | `depth_to_pointcloud`, `build_occupancy_grid` | RGBD → 占据栅格 |
| `voxel_utils.py` | `make_voxel_params`, `world_to_voxel`, `voxel_to_world` | 体素坐标系管理 |
| `grid_ops.py` | `voxelize_obb`, `prepare_grid_base`, `dilate_obstacles_xy` | 栅格操作 |
| `surface.py` | `detect_support_surfaces` | 支撑面检测（点云 RANSAC 优先，栅格搜索回退） |
| `collision.py` | `find_table_placements` | FFT 碰撞搜索 |
| `filters.py` | `filter_stable/visible/occluded_placements` | 三级过滤 |
| `cluster.py` | `cluster_placements` | DBSCAN 聚类 |
| `pipeline.py` | `PlacementPipeline.run` | 流程编排 |
| `io_utils.py` | `save_ply`, `save_placement_result` | 结果持久化 |
| `visualize.py` | `save_placement_vis` | 双面板可视化 |

---

## 3. 模块间数据流

```
DatasetAdapter.load_scene()
    └─→ SceneData {rgb, depth, camera: CameraParams, objects: [ObjectInfo]}
            │
            ▼
    PlacementPipeline.run(scene)
            │
            ├─→ depth_to_pointcloud(depth, camera)
            │       └─→ pts_world (N,3), colors (N,3)
            │
            ├─→ build_occupancy_grid(depth, camera, voxel_size)
            │       └─→ grid (Gx,Gy,Gz) uint8, grid_min (3,), voxel_size
            │
            ├─→ make_voxel_params(grid_min, voxel_size)
            │       └─→ vp = {"origin": [...], "voxel_size": float}
            │
            ├─→ prepare_grid_base(grid, objects, vp)
            │       └─→ grid_base (Gx,Gy,Gz)  [所有物体 OBB 已标记]
            │
            └─→ 逐物体:
                    ├─→ voxelize_obb(bbox3d, pose_world, vp)
                    │       └─→ target_vox (M,3)
                    │
                    ├─→ detect_support_surfaces(grid_other, vp, points_world=pts_world,
                    │                          target_voxels=target_vox)
                    │       └─→ table_z (int), surface_mask (Gx,Gy) bool
                    │
                    ├─→ find_table_placements(grid_base, bbox3d, pose_world, vp, table_z, surface_mask)
                    │       └─→ candidates (N,3), meta dict, yaw_data dict
                    │
                    ├─→ filter_stable_placements(candidates, yaw_data, surface_mask)
                    │       └─→ candidates (K,3)
                    │
                    ├─→ filter_visible_placements(candidates, ..., K, E_w2c)
                    │       └─→ candidates (J,3)
                    │
                    ├─→ build_depth_buffer(grid_base, vp, K, E_w2c)
                    │   filter_occluded_placements(candidates, ..., depth_buf)
                    │       └─→ candidates (I,3)
                    │
                    └─→ cluster_placements(candidates, grid_base, yaw_data, table_z, vp)
                            └─→ reps (C,3), cluster_infos [list of dict]

    最终输出: {obj_id: PlacementResult}
```

### 关键数据结构

**`yaw_data` dict**（由 `find_table_placements` 返回）：
```python
{
    "yaw_angles": (S,) float,        # 每个 yaw 角度（弧度）
    "T_rotated":  (S, 4, 4) float,   # 每个 yaw 对应的旋转变换矩阵
    "vmin_rot_abs": (S, 3) float,    # 旋转后 bbox 的最小角（体素坐标）
    "footprints": [(Fx,Fy) bool],    # 每个 yaw 的 2D footprint
}
```

**`cluster_infos` list**（每个元素为一个聚类）：
```python
{
    "cluster_id": int,                 # 聚类编号
    "size": int,                       # 聚类内候选数量
    "anchor_voxel": [x, y, yaw_idx],   # 体素坐标 + yaw 索引
    "anchor_world": [wx, wy, wz],      # 世界坐标
    "yaw_index": int,                  # yaw 离散索引
    "yaw_degrees": float,              # yaw 角度（度）
    "free_score": float,               # 自由空间得分（越高越好）
}
```

---

## 4. 接入新数据集

### 4.1 需要实现的接口

继承 `src/datasets/base_adapter.py` 中的 `DatasetAdapter` 抽象基类：

```python
from src.datasets.base_adapter import DatasetAdapter
from src.annotation.free_bbox.datatypes import SceneData, CameraParams, ObjectInfo

class MyDatasetAdapter(DatasetAdapter):

    def load_scene(self, scene_path: str, frame_id: str) -> SceneData:
        """加载单帧场景数据，返回统一格式 SceneData。"""
        ...

    def list_scenes(self) -> list[tuple[str, list[str]]]:
        """返回所有场景路径和帧 ID 列表：[(scene_path, [frame_id, ...]), ...]"""
        ...
```

### 4.2 SceneData 各字段要求

| 字段 | 类型 | 要求 |
|---|---|---|
| `scene_id` | str | 场景唯一标识 |
| `frame_id` | str | 帧唯一标识 |
| `rgb` | (H,W,3) uint8 | 标准 RGB 图像 |
| `depth` | (H,W) float32 | **已转换为场景工作单位**（如 cm），与相机外参平移单位一致 |
| `camera.fx/fy/cx/cy` | float | 相机内参（像素单位） |
| `camera.E_c2w` | (4,4) float64 | camera→world 变换，**平移单位与 depth 一致** |
| `camera.img_w/img_h` | int | 图像分辨率 |
| `objects[i].bbox3d_canonical` | (6,) float | 物体规范坐标系 AABB：`[xmin,ymin,zmin, xmax,ymax,zmax]`，单位与场景一致 |
| `objects[i].pose_world` | (4,4) float64 | object→world 变换矩阵 |

### 4.3 单位一致性（最重要）

**所有长度单位必须统一**，通用代码不做任何单位转换。

以 HOPE 数据集为例（工作单位 = cm）：
```python
# 深度图：原始 uint16（mm 量级）→ cm
depth_cm = raw_depth.astype(np.float32) * DEPTH_SCALE  # DEPTH_SCALE = 0.98042517 / 10.0

# 相机外参：原始平移单位为 m → cm
E_w2c[:3, 3] *= 100.0

# mesh 顶点：原始单位 mm → cm
vertices_cm = vertices_mm * 0.1
bbox3d_canonical = np.concatenate([vertices_cm.min(0), vertices_cm.max(0)])

# 物体位姿：原始为 object→camera，需转为 object→world
pose_world = E_c2w @ pose_cam
```

### 4.4 bbox3d_canonical 的来源

`bbox3d_canonical` 是物体在**自身规范坐标系**下的 AABB，代表物体的尺度信息。可以从以下来源获得：

| 来源 | 方法 |
|---|---|
| CAD/mesh 文件 | 加载 mesh，取顶点包围盒 |
| 数据集标注文件 | 直接读取 size 字段，构造 `[-l/2,-w/2,-h/2, l/2,w/2,h/2]` |
| 手动测量 | 同上 |
| 点云拟合 | 对物体点云做 PCA，取主轴方向包围盒 |

### 4.5 配置文件适配

在 `configs/annotation/placement.yaml` 中修改 `dataset` 部分：

```yaml
dataset:
  type: my_dataset          # 对应 build_adapter() 中的分支
  root_dir: /path/to/data
  # 其他数据集特定参数...
```

在 `tools/run_placement.py` 的 `build_adapter()` 函数中添加对应分支：

```python
elif ds_type == "my_dataset":
    from src.datasets.my_adapter import MyDatasetAdapter
    return MyDatasetAdapter(root_dir=ds_cfg["root_dir"])
```

### 4.6 参数调整建议

不同数据集可能需要调整以下参数：

| 参数 | 说明 | 调整依据 |
|---|---|---|
| `voxel_size` | 体素边长（场景单位） | 场景尺度，通常为物体最小尺寸的 1/5~1/10 |
| `grid_padding` | 栅格边界扩展 | 场景单位，确保物体放置区域不超出栅格 |
| `safety_margin` | 碰撞安全边距 | 场景单位，通常为 0.5~2 个体素 |
| `min_surface_area` | 最小支撑面面积 | 场景单位²，内部会按 `voxel_size` 换算为体素面积阈值；过小会检测到噪声平面 |
| `pixel_stride` | 深度图采样步长 | 越小越精确但越慢，通常 2~8 |

---

## 5. 输出标注格式

### 5.1 目录结构

```
output_dir/
├── placement_result.json       # 主结果文件
├── grid_meta.json              # 占据栅格元信息
├── occupancy_grid.npy          # 占据栅格数组
├── occupancy_grid.ply          # 占据栅格点云（可视化用）
├── point_cloud.ply             # 场景点云
└── placement_{ClassName}/
    └── placement_vis.png       # 双面板可视化图
```

### 5.2 placement_result.json 格式

```json
{
  "config": {
    "voxel_size": 1.0,
    "safety_margin": 0.5,
    "yaw_steps": 24,
    "min_support_ratio": 1.0,
    "occlusion_threshold": 0.3,
    "dbscan_eps": 5.0,
    "gpu_used": false
  },
  "objects": {
    "obj_0": {
      "class_name": "CreamCheese",
      "num_raw": 13,
      "num_stable": 13,
      "num_visible": 8,
      "num_unoccluded": 5,
      "original_aabb_world": [x1, y1, z1, x2, y2, z2],
      "clusters": [
        {
          "cluster_id": 0,
          "size": 3,
          "anchor_voxel": [vx, vy, yaw_idx],
          "anchor_world": [wx, wy, wz],
          "yaw_index": 1,
          "yaw_degrees": 15.0,
          "free_score": 142.0
        }
      ]
    }
  }
}
```

### 5.3 字段说明

| 字段 | 说明 |
|---|---|
| `num_raw` | FFT 碰撞搜索后的原始候选数（所有 yaw 角度合计） |
| `num_stable` | 稳定性过滤后剩余数 |
| `num_visible` | 可见性过滤后剩余数 |
| `num_unoccluded` | 遮挡过滤后剩余数 |
| `clusters` | DBSCAN 聚类结果，每个元素为一个聚类的代表放置位置 |
| `anchor_world` | 放置锚点的世界坐标（对应候选 `anchor_voxel` 的体素中心） |
| `yaw_index` | 放置时使用的 yaw 离散索引 |
| `yaw_degrees` | 放置时物体绕 Z 轴的旋转角度（度） |
| `free_score` | 该位置周围自由空间体素数，越大表示周围越空旷 |
| `original_aabb_world` | 物体原始位置的世界坐标 AABB（用于对比） |

### 5.4 grid_meta.json 格式

```json
{
  "voxel_params": {
    "origin": [x, y, z],
    "voxel_size": 1.0
  },
  "grid_shape": [Gx, Gy, Gz],
  "voxel_counts": {
    "free": 123456,
    "occupied": 7890,
    "unknown": 11111
  }
}
```

### 5.5 如何使用放置结果

```python
import json
import numpy as np
from src.annotation.free_bbox.voxel_utils import make_voxel_params
from src.utils.coord_utils import transform_points

# 加载结果
with open("output_dir/placement_result.json") as f:
    result = json.load(f)

with open("output_dir/grid_meta.json") as f:
    meta = json.load(f)

vp = meta["voxel_params"]

# 读取某物体的放置位置
obj = result["objects"]["obj_0"]
for cluster in obj["clusters"]:
    anchor_world = np.array(cluster["anchor_world"])  # 世界坐标
    yaw_deg = cluster["yaw_degrees"]                 # 旋转角度（度）
    print(f"放置位置: {anchor_world}, yaw: {yaw_deg:.1f} deg")
```

---

## 6. 使用教程

### 6.1 环境准备

```bash
# 激活 conda 环境
conda activate spatial

# 确认依赖已安装
python -c "import numpy, scipy, sklearn, matplotlib, trimesh; print('OK')"
```

### 6.2 配置文件

编辑 `configs/annotation/placement.yaml`：

```yaml
dataset:
  type: hope
  root_dir: /path/to/hope_video
  mesh_dir: /path/to/hope_meshes
  frame_step: 60          # 每隔 60 帧取一帧

occupancy:
  voxel_size: 1.0         # 体素边长（cm）
  pixel_stride: 4         # 深度图采样步长
  grid_padding: 10.0      # 栅格边界扩展（cm）

placement:
  safety_margin: 0.5      # 碰撞安全边距（体素数）
  yaw_steps: 24           # yaw 离散步数
  min_surface_area: 50.0  # 最小支撑面面积（场景单位²）
  min_support_ratio: 1.0  # 最小支撑比
  world_up: [0, 0, 1]     # 世界坐标系上方向

clustering:
  dbscan_eps: 5.0         # DBSCAN 聚类半径（体素单位）
  dbscan_min_samples: 1

visualization:
  save_vis: true
  save_ply: true
  vis_margin_px: 30       # 可见性检查图像边距（像素）

compute:
  use_gpu: false
  stability_chunk_size: 2000

output:
  dir: outputs/placement
```

### 6.3 单帧处理

```bash
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --scene /path/to/hope_video/scene_0001 \
    --frame 0000 \
    --output outputs/placement/scene_0001/0000
```

### 6.4 批量处理

```bash
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --batch \
    --output outputs/placement
```

### 6.5 GPU 加速（需要 CuPy）

```bash
# 安装 CuPy（根据 CUDA 版本选择）
pip install cupy-cuda12x

# 启用 GPU
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --batch --gpu \
    --output outputs/placement
```

### 6.6 Python API 调用

```python
from src.annotation.free_bbox.datatypes import PlacementConfig
from src.annotation.free_bbox.pipeline import PlacementPipeline
from src.datasets.hope_adapter import HopeAdapter

# 加载数据
adapter = HopeAdapter(
    root_dir="/path/to/hope_video",
    mesh_dir="/path/to/hope_meshes",
    frame_step=60,
)
scene = adapter.load_scene("/path/to/hope_video/scene_0001", "0000")

# 配置参数
config = PlacementConfig(
    voxel_size=1.0,
    safety_margin=0.5,
    yaw_steps=24,
)

# 运行 pipeline
pipeline = PlacementPipeline(config=config, use_gpu=False)
results = pipeline.run(scene, output_dir="outputs/test", save_vis=True)

# 查看结果
for obj_id, result in results.items():
    print(f"{result.class_name}: {len(result.placements)} 个放置聚类")
    for p in result.placements:
        print(f"  位置: {p['anchor_world']}, yaw: {p['yaw_rad']:.3f}")
```

### 6.7 跳过可视化/PLY 导出（加速）

```bash
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --batch --no-vis --no-ply \
    --output outputs/placement
```

### 6.8 常见问题

**Q: 所有物体都被 SKIP（no support surface）**
- 检查 `min_surface_area` 是否过大
- 检查深度图单位是否与相机外参平移单位一致
- 检查 `world_up` 方向是否正确

**Q: 候选数量为 0（num_raw=0）**
- 检查 `safety_margin` 是否过大（超过支撑面尺寸）
- 检查物体 `bbox3d_canonical` 尺寸是否正确（单位一致性）
- 减小 `voxel_size` 提高分辨率

**Q: 运行速度慢**
- 增大 `pixel_stride`（4→8）
- 减小 `yaw_steps`（24→8）
- 增大 `voxel_size`（1.0→2.0）
- 启用 GPU（`--gpu`）

**Q: 接入新数据集后结果异常**
- 首先验证单位一致性：打印 `scene.depth.max()` 和 `scene.camera.E_c2w[:3, 3]`，两者量级应相同
- 验证 `pose_world`：将 `bbox3d_canonical` 角点用 `pose_world` 变换后，应与 RGB 图像中物体位置对应
- 验证 `bbox3d_canonical`：尺寸应与物体实际大小一致（场景单位）