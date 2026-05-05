# 数据处理与 Benchmark 流程说明

本文档描述从原始数据集到最终评测的完整数据处理流程。

---

## 一、整体架构

项目数据处理分为 **4 个阶段**，最终支撑 3D Visual Grounding 模型的训练与评测：

```
原始数据集 → 阶段0: 放置规划 → 阶段1: Benchmark构建 → 阶段2: 模型推理 → 阶段3: 评测
```

| 阶段 | 作用 | 核心工具 |
|------|------|----------|
| 0. 放置规划 | 生成 occupancy grid、场景物体信息 | `tools/run_placement.py` |
| 1. Benchmark 构建 | 打包标注+标签+occupancy 成自包含包 | `tools/build_benchmark_manifest.py` |
| 2. 模型推理 | 用训练好的模型预测 3D box | `scripts/infer_multimodal.py` |
| 3. 评测 | 计算 collision/direction/size 指标 | `tools/evaluate_benchmark_predictions.py` |

---

## 二、数据源

项目支持 4 个数据集，通过不同的 adapter 统一接口：

| 数据源 | Adapter | 配置 | 深度单位 |
|--------|---------|------|----------|
| HOPE-Video | `HopeAdapter` | `configs/annotation/placement.yaml` | uint16 mm × 0.98042517 / 10 → cm |
| HouseCat6D | `HouseCat6DAdapter` | `configs/annotation/placement_housecat6d.yaml` | uint16 mm × 0.1 → cm |
| DoPose | `DoPoseAdapter` | `configs/annotation/placement_dopose.yaml` | - |
| YCB-Video | `YCBVideoAdapter` | `configs/annotation/placement_ycbv_test.yaml` | - |

---

## 三、阶段详解

### 阶段 0：放置规划（生成中间数据）

**目标**：从原始 RGB + 深度图生成 occupancy grid 和场景物体信息。

**入口**：`tools/run_placement.py`

**输出结构**：
```
outputs/{source_name}/
├── scene_objects/{prefix}.json    # 场景中所有物体 [obj_id, class_name, pose_world, canonical_aabb]
├── grid_meta/{prefix}.json         # 体素参数 + 相机内外参
├── occupancy_grids/{prefix}.npy    # 体素占据格 (FREE=0, OCCUPIED=1, UNKNOWN=2)
├── samples/{prefix}.json           # 训练样本 (target_box, original_pose_world 等)
├── point_clouds/{prefix}.ply       # 带色彩点云 (可选)
└── frame_status.json               # 处理状态跟踪
```

**运行命令**：
```bash
# 单场景单帧
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --scene scene_0001 --frame 0000 \
    --output outputs/

# 并行批量处理 (8 workers)
python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --batch --workers 8 --output outputs/

# 查看处理状态
python tools/run_placement.py --config configs/annotation/placement.yaml --status --output outputs/

# 重试失败帧
python tools/run_placement.py --config configs/annotation/placement.yaml --batch --retry-failed --output outputs/
```

**核心处理流程**（6 步）：
1. `depth_to_pointcloud()` — 深度图 → 世界坐标系彩色点云
2. `build_occupancy_grid()` — 点云 → 3D 体素占据格
3. `prepare_grid_base()` — 所有物体 OBB 标记为 OCCUPIED
4. `detect_support_surfaces()` — RANSAC 平面拟合，选取最近支撑面
5. `find_table_placements()` — FFT 2D 卷积碰撞检测，遍历 24 个 yaw 角（15°/步）
6. `filter_*()` — 稳定性、可见性、遮挡过滤 → DBSCAN 聚类 → 保存

---

### 阶段 1：自动标注与 Benchmark 构建

#### 步骤 1a：自动标注 spatial_relation

**目标**：为每个 placement 样本生成自然语言空间关系描述。

**入口**：`tools/auto_label.py`

**输出**：`outputs/prompt_merged/all_labels.json`

**标注格式**：
```json
{
  "image_filename": "hope__scene_0000_0000_obj_1_p000.png",
  "sample_id": "scene_0000_0000_obj_1_p000",
  "source_name": "hope",
  "label": "Move Parmesan ... to the right of Mug.",
  "spatial_relation": {
    "original": { "relation": "the front left of", "reference_object_id": "obj_3", ... },
    "placement": { "relation": "the right of", "reference_object_id": "obj_5", ... }
  }
}
```

**空间关系类型**（10 种）：
- 垂直：`the top of`、`below`
- 水平：`the right of`、`the left of`、`in front of`、`behind`
- 对角：`the front right of`、`the front left of`、`the back right of`、`the back left of`
- 特殊：`near`（中心近似重合）

**运行命令**：
```bash
python tools/auto_label.py \
    --image-dir outputs/placement_rgb_bbox_vis \
    --outputs-base outputs \
    --output-dir outputs/prompt_merged \
    --mapping configs/annotation/mappingv2.json
```

#### 步骤 1b：构建 Benchmark 包

**目标**：将 annotation + auto_label + occupancy grid 打包成自包含 benchmark，评测时不依赖原始数据集 adapter。

**入口**：`tools/build_benchmark_manifest.py`

**输出结构**：
```
benchmark/placement_v1/
├── manifest.json      # 固化所有评测所需字段（target_box, camera, occupancy路径, direction信息）
├── summary.json        # 样本数/来源分布
└── occupancy_grids/    # 从 outputs/ 复制过来的占据格
    ├── hope/
    ├── dopose/
    ├── housecat6d/
    └── ycbv_test/
```

**manifest.json 每条样本包含**：
```json
{
  "sample_id": "scene_0000_0000_obj_1_p000",
  "source_name": "hope",
  "target_box_world": [cx, cy, cz, sx, sy, sz, yaw],
  "camera": { "fx", "fy", "cx", "cy", "E_c2w", ... },
  "occupancy": {
    "path": "occupancy_grids/hope/scene_0000_0000.npy",
    "voxel_params": { "origin": [...], "voxel_size": 1.0 },
    "grid_shape": [240, 240, 50]
  },
  "direction": {
    "expected_relation": "the right of",
    "reference_object_id": "obj_5",
    "reference_corners_world": [[x0,y0,z0], ...]  // 8个角点
  }
}
```

**运行命令**：
```bash
python tools/build_benchmark_manifest.py \
    --annotation-dir data/annotations/placement_multimodal \
    --label-json outputs/prompt_merged/all_labels.json \
    --outputs-base outputs \
    --output-dir benchmark/placement_v1 \
    --split test
```

---

### 阶段 2：模型推理

**目标**：用训练好的多模态模型预测 3D bounding box。

**入口**：`scripts/infer_multimodal.py`

**输入**：
- `--checkpoint`：模型权重路径
- `--split`：评测分割（test/valid/train）

**输出**：`outputs/{experiment_name}/predictions.json`

**predictions.json 格式**：
```json
{
  "predictions": [
    {
      "sample_id": "scene_0000_0000_obj_1_p000",
      "source_name": "hope",
      "pred_box_world": [cx, cy, cz, sx, sy, sz, yaw_degrees],
      "gt_box_world": [cx, cy, cz, sx, sy, sz, yaw_degrees]
    }
  ]
}
```

**运行命令**：
```bash
python scripts/infer_multimodal.py \
    --checkpoint outputs/model_ptv3/best.pth \
    --split test \
    --output-dir outputs/infer_ptv3
```

---

### 阶段 3：Benchmark 评测

**目标**：基于 manifest 和 predictions.json 计算三类评测指标。

**入口**：`tools/evaluate_benchmark_predictions.py`

**输出结构**：
```
outputs/{experiment_name}_benchmark_eval/
├── metrics_summary.json     # 全局汇总指标
├── per_sample_metrics.json  # 逐样本详细结果
└── per_sample_metrics.csv   # CSV 格式（可选）
```

#### 三类评测指标

| 指标 | 目标 | 计算方法 |
|------|------|----------|
| **Collision-Free** | 预测 box 是否与占据格中的 OCCUPIED 体素重叠 | 将 pred_box 体素化，统计落在 OCCUPIED 的比例，默认阈值 0.003 |
| **Direction-Correct** | 预测位置是否满足指令中的目标方位关系 | 计算 pred_box 与 reference 的空间关系，与 expected_relation 比较 |
| **Size-Consistent** | 预测 box 尺寸是否与目标物体一致 | 比较 pred_size 与 gt_size，默认最大单轴误差 ≤ 2cm |

**主指标定义**：
```
placement_success = collision_free AND direction_correct AND size_consistent
```

**碰撞检测特殊处理**：根据 GT 最低体素层向下忽略 N 层桌面支撑层（默认 2 层），避免桌面本身误判为碰撞。

#### 汇总指标

```json
{
  "summary": {
    "sample_count": 1718,
    "placement_success_rate": 0.654,
    "collision_free_rate": 0.744,
    "direction_correct_rate": 0.952,
    "size_consistent_rate": 0.895,
    "mean_occupied_collision_ratio": 0.020,
    "by_source": { ... }
  }
}
```

**运行命令**：
```bash
python tools/evaluate_benchmark_predictions.py \
    --benchmark-dir benchmark/placement_v1 \
    --predictions outputs/infer_ptv3/predictions.json \
    --output-dir outputs/infer_ptv3_benchmark_eval \
    --write-csv

# 可调阈值
python tools/evaluate_benchmark_predictions.py \
    --benchmark-dir benchmark/placement_v1 \
    --predictions outputs/infer_ptv3/predictions.json \
    --output-dir outputs/infer_ptv3_benchmark_eval \
    --collision-ratio-threshold 0.005 \
    --volume-error-ratio-threshold 0.1
```

---

## 四、可视化网站生成（可选）

将评测结果和推理可视化图片合成静态展示网站。

**入口**：`tools/build_benchmark_site.py`

**输出**：`--eval-dir/visualization/index.html` + assets/

**运行命令**：
```bash
python tools/build_benchmark_site.py \
    --eval-dir outputs/infer_ptv3_benchmark_eval \
    --infer-dir outputs/infer_ptv3
```

---

## 五、关键数据规格

| 项目 | 规格 |
|------|------|
| 深度图单位 | uint16 (mm)，适配器统一转换为 cm |
| 外参平移 | 存储为 m，适配器统一转换为 cm |
| 体素大小 | 默认 1.0 cm |
| Yaw 搜索步数 | 24 步（15°/步） |
| 安全边距 | 2 cm |
| 体素状态 | FREE=0, OCCUPIED=1, UNKNOWN=2 |

---

## 六、完整从头开始命令序列

```bash
# ========== 阶段 0：放置规划 ==========
conda run -n spatial python tools/run_placement.py \
    --config configs/annotation/placement.yaml \
    --batch --workers 8 \
    --output outputs/

# ========== 阶段 1a：自动标注 ==========
conda run -n spatial python tools/auto_label.py \
    --image-dir outputs/placement_rgb_bbox_vis \
    --outputs-base outputs \
    --output-dir outputs/prompt_merged \
    --mapping configs/annotation/mappingv2.json

# ========== 阶段 1b：构建 Benchmark ==========
conda run -n spatial python tools/build_benchmark_manifest.py \
    --annotation-dir data/annotations/placement_multimodal \
    --label-json outputs/prompt_merged/all_labels.json \
    --output-dir benchmark/placement_v1 \
    --split test

# ========== 阶段 2：模型推理 ==========
conda run -n spatial python scripts/infer_multimodal.py \
    --checkpoint outputs/model_ptv3/best.pth \
    --split test \
    --output-dir outputs/infer_ptv3

# ========== 阶段 3：评测 ==========
conda run -n spatial python tools/evaluate_benchmark_predictions.py \
    --benchmark-dir benchmark/placement_v1 \
    --predictions outputs/infer_ptv3/predictions.json \
    --output-dir outputs/infer_ptv3_benchmark_eval \
    --write-csv

# ========== 可选：生成可视化网站 ==========
conda run -n spatial python tools/build_benchmark_site.py \
    --eval-dir outputs/infer_ptv3_benchmark_eval \
    --infer-dir outputs/infer_ptv3
```

---

## 七、文件位置速查

| 文件 | 作用 |
|------|------|
| `tools/run_placement.py` | 场景级放置规划入口 |
| `tools/auto_label.py` | 生成 spatial_relation 标注 |
| `tools/build_benchmark_manifest.py` | 构建自包含 benchmark 包 |
| `scripts/infer_multimodal.py` | 模型推理入口 |
| `tools/evaluate_benchmark_predictions.py` | 计算评测指标 |
| `tools/build_benchmark_site.py` | 生成可视化网站（可选） |
| `src/metrics/placement_eval.py` | 三大指标实现（collision/direction/size） |
| `src/annotation/free_bbox/pipeline.py` | 放置规划 6 步流程 |
| `src/annotation/free_bbox/occupancy.py` | 深度图→点云→占据格 |
| `src/datasets/*_adapter.py` | 各数据集适配器 |