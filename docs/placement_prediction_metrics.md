# Placement Prediction Metrics

本文档描述多模态 placement test 预测结果的评测指标、依赖的上游文件、字段来源和运行方式。

## 评测目标

评测对象是 `scripts/infer_multimodal.py` 导出的 `predictions.json`，其中每条预测包含：

- `sample_id` / `source_name`
- `pred_box_world`: 预测 3D box，格式 `[cx, cy, cz, sx, sy, sz, yaw_degrees]`
- `pred_object_center_world`: 预测移动前目标物体中心，格式 `[x, y, z]`
- `gt_box_world`: GT 3D box，格式同上

指标定义为：

```text
placement_success = collision_free AND direction_correct AND size_consistent
overall_success = placement_success AND object_center_match
```

同时单独报告四类子指标，其中前三类用于放置成功率，`object_center` 用于评估模型是否定位到移动前目标物体。

## 上游文件依赖

必需文件：

- `--predictions`: 推理结果，例如 `outputs/infer_ptv3/predictions.json`
- `--benchmark-dir`: 自包含 benchmark 目录，例如 `benchmark/placement_v1`

评测脚本会读取：

- `{benchmark_dir}/manifest.json`
  - `target_box_world`
  - `target_object.corners_world`
  - `camera`
  - `occupancy`
  - `direction.reference_corners_world`
- `{benchmark_dir}/occupancy_grids/{source_name}/{scene_id}_{frame_id}.npy`
  - 碰撞 metric 使用的自包含 occupancy grid

方向 metric 的结构化上游字段来自 `tools/auto_label.py`：

```json
{
  "spatial_relation": {
    "placement": {
      "relation": "the right of",
      "reference_object_id": "obj_1",
      "reference_class_name": "mustard_bottle",
      "reference_name": "Mustard Bottle",
      "distance_cm": 12.3
    }
  }
}
```

`tools/build_multimodal_dataset.py` 会将该字段透传到 train/valid/test annotation。
`tools/build_benchmark_manifest.py` 会将 reference 物体角点和 target 物体原始 3D 框角点固化到 benchmark manifest。

## Metric 定义

### 1. Collision-Free

目标：判断预测放置框是否覆盖同帧 occupancy grid 中的 `OCCUPIED` 体素。

评测使用 `outputs/{source_name}/occupancy_grids/{scene_id}_{frame_id}.npy`，并读取同名前缀的 `grid_meta/*.json` 中的体素参数。`.ply` 文件只用于可视化，不参与 metric。

体素状态定义来自 `src/annotation/free_bbox/occupancy.py`：

```text
FREE     = 0
OCCUPIED = 1
UNKNOWN  = 2
```

benchmark 评测会根据 GT 放置框最低体素层向下忽略桌面支撑层。当前默认忽略 2 层，用于匹配 placement pipeline 中 `landing_z = table_z + 1`，以及桌面表面可能跨两层 `OCCUPIED` 的情况。支撑层之外的目标物体原位置或其他 `OCCUPIED` 仍计入碰撞。

流程：

1. 将 `pred_box_world = [cx, cy, cz, sx, sy, sz, yaw_degrees]` 转成世界坐标 yaw-only OBB。
2. 通过 `src.annotation.free_bbox.grid_ops.voxelize_obb` 将预测 OBB 体素化到 occupancy grid 索引空间。
3. 将 `gt_box_world` 体素化，取 GT 最低体素层 `gt_landing_z`，默认忽略 `gt_landing_z - 1` 和 `gt_landing_z - 2` 两层的 `OCCUPIED`。
4. 统计预测体素中状态为 `OCCUPIED` 和 `UNKNOWN` 的数量，其中 `OCCUPIED` 会排除上述支撑层。
5. 归一化 `OCCUPIED` 碰撞比例：

```text
occupied_collision_ratio = occupied_voxel_count / pred_voxel_count
```

默认通过条件：

```text
occupied_collision_ratio <= collision_ratio_threshold
```

`UNKNOWN` 体素代表未观测空间，只报告覆盖比例，不直接作为碰撞失败条件：

```text
unknown_overlap_ratio = unknown_voxel_count / pred_voxel_count
```

输出字段包括：

- `collision_free`
- `pred_voxel_count`
- `occupied_voxel_count`
- `occupied_collision_ratio`
- `ignored_support_layers`
- `ignored_support_occupied_count`
- `unknown_voxel_count`
- `unknown_overlap_ratio`
- `occupancy_grid_path`
- `grid_meta_path`

### 2. Direction-Correct

目标：判断预测放置位置是否满足指令中的目标方位。

评测不从 prompt 文本反解析方位，而是使用 `auto_label.py` 生成指令时保存的结构化目标关系：

- `expected_relation`
- `reference_object_id`

流程：

1. 根据 `reference_object_id` 在当前 scene 中找到参考物。
2. 若 `pred_box_world[:3]` 与 `gt_box_world[:3]` 的 L2 距离不超过直通阈值，则直接判定方向正确。
3. 否则将预测 box 与参考物 box 输入 `auto_label.py` 同源空间关系函数。
4. 得到 `pred_relation` 并比较：

```text
direction_correct = pred_relation == expected_relation
```

支持的关系包括：

- `the right of`
- `the front right of`
- `in front of`
- `the front left of`
- `the left of`
- `the back left of`
- `behind`
- `the back right of`
- `the top of`
- `below`

输出字段包括：

- `direction_correct`
- `expected_relation`
- `pred_relation`
- `center_match`
- `center_l2_error_cm`
- `center_l2_threshold_cm`
- `reference_object_id`
- `reference_class_name`
- `reference_name`

### 3. Size-Consistent

目标：判断预测 3D box 尺寸是否与原目标物体尺寸一致。

比较 `pred_box_world[3:6]` 与 `target_box_world[3:6]` 的体积：

```text
volume_error_ratio = abs(pred_volume - target_volume) / target_volume
```

默认通过条件：

```text
volume_error_ratio <= volume_error_ratio_threshold
```

输出字段包括：

- `size_consistent`
- `pred_volume_cm3`
- `target_volume_cm3`
- `volume_error_cm3`
- `volume_error_ratio`
- `volume_error_ratio_threshold`

### 4. Object-Center-In-Target

目标：判断模型预测的移动前物体中心是否落在目标物体原始 3D 框对应的图像区域内。

该指标不再使用 `object_center_world` 的 3D L2 距离阈值。新版 benchmark manifest 保存目标物体原始 3D 框角点：

```json
{
  "target_object": {
    "object_id": "obj_1",
    "class_name": "target_class",
    "corners_world": [[x0, y0, z0], ...]
  }
}
```

流程：

1. 将 `target_object.corners_world` 通过 `camera` 投影到图像二维坐标。
2. 对投影角点计算二维凸包，作为目标物体 3D 框在图像中的区域。
3. 将 `pred_object_center_world` 投影到图像二维点。
4. 判断该二维点是否位于目标物体投影凸包内：

```text
object_center_match = projected_center_in_target_box
```

输出字段包括：

- `center_match`
- `projected_center_in_target_box`
- `pred_center_projected`
- `pred_center_uv`
- `pred_center_depth`
- `target_projected_hull_uv`
- `target_projected_hull_area_px2`
- `target_visible_corner_count`

## 输出文件

运行后输出：

- `metrics_summary.json`
  - 全局 summary
  - 按 `source_name` 分组 summary
  - direction confusion matrix
  - 阈值和输入文件记录
- `per_sample_metrics.json`
  - 每个样本的四类 metric 详细结果
- `per_sample_metrics.csv`
  - 仅在使用 `--write-csv` 时生成，便于表格分析

summary 中的覆盖率含义：

- `collision_coverage`: 成功计算 collision 的样本比例
- `direction_coverage`: 成功计算 direction 的样本比例
- `size_coverage`: 成功计算 size 的样本比例
- `object_center_coverage`: 成功计算 object_center 的样本比例
- `full_metric_coverage`: collision、direction、size 三类 placement metric 都成功计算的样本比例
- `overall_metric_coverage`: 四类 metric 都成功计算的样本比例
- `mean_occupied_collision_ratio` / `median_occupied_collision_ratio`: 已评估样本的 OCCUPIED 体素碰撞比例统计
- `mean_unknown_overlap_ratio` / `median_unknown_overlap_ratio`: 已评估样本的 UNKNOWN 体素覆盖比例统计

`placement_success_rate` 只在三类 placement metric 都成功计算的样本上统计。`overall_success_rate` 会额外要求 `object_center_match` 通过。

## 运行示例

```bash
conda run -n spatial python tools/evaluate_benchmark_predictions.py \
    --benchmark-dir benchmark/placement_v1 \
    --predictions outputs/infer_ptv3/predictions.json \
    --output-dir outputs/infer_ptv3_benchmark_eval \
    --write-csv
```

可调整阈值：

```bash
conda run -n spatial python tools/evaluate_benchmark_predictions.py \
    --benchmark-dir benchmark/placement_v1 \
    --predictions outputs/infer_ptv3/predictions.json \
    --output-dir outputs/infer_ptv3_benchmark_eval_strict \
    --collision-ratio-threshold 0.005 \
    --volume-error-ratio-threshold 0.08
```

快速 smoke test 可只评测前几个样本：

```bash
conda run -n spatial python tools/evaluate_benchmark_predictions.py \
    --benchmark-dir benchmark/placement_v1 \
    --predictions outputs/infer_ptv3/predictions.json \
    --output-dir outputs/infer_ptv3_benchmark_eval_smoke \
    --limit 5
```

## 兼容策略

新数据推荐先重新运行：

1. `tools/auto_label.py` 生成带 `spatial_relation` 的 `all_labels.json`
2. `tools/build_multimodal_dataset.py` 生成带 scene/object/relation 字段的 annotation
3. `scripts/infer_multimodal.py` 生成 predictions
4. `tools/build_benchmark_manifest.py` 构建新版 benchmark
5. `tools/evaluate_benchmark_predictions.py` 评测

旧 benchmark 可通过 backfill 升级：

```bash
conda run -n spatial python tools/backfill_benchmark_manifest_target_object.py \
    --benchmark-dir benchmark/placement_v1 \
    --outputs-base outputs \
    --output-dir benchmark/placement_v2 \
    --overwrite
```
