# Placement Prediction Metrics

本文档描述多模态 placement test 预测结果的评测指标、依赖的上游文件、字段来源和运行方式。

## 评测目标

评测对象是 `scripts/infer_multimodal.py` 导出的 `predictions.json`，其中每条预测包含：

- `sample_id` / `source_name`
- `pred_box_world`: 预测 3D box，格式 `[cx, cy, cz, sx, sy, sz, yaw_degrees]`
- `gt_box_world`: GT 3D box，格式同上

主指标定义为：

```text
placement_success = collision_free AND direction_correct AND size_consistent
```

同时单独报告三类子指标，便于定位失败原因。

## 上游文件依赖

必需文件：

- `--predictions`: 推理结果，例如 `outputs/infer_ptv3/predictions.json`
- `--annotation-dir`: 多模态标注目录，例如 `data/annotations/placement_multimodal_simple`
- `--outputs-base`: placement 输出根目录，例如 `outputs`

评测脚本会读取：

- `{annotation_dir}/{split}.json`
  - 新数据应包含 `scene_id`、`frame_id`、`object_id`、`class_name`、`spatial_relation`
  - 旧数据至少需要 `placement.target_box` 和 `camera`
- `{outputs_base}/{source_name}/samples/*.json`
  - 用于旧数据 fallback，补齐 `scene_id`、`frame_id`、`object_id`、`canonical_aabb_object`、`transform_world`
- `{outputs_base}/{source_name}/occupancy_grids/{scene_id}_{frame_id}.npy`
  - 碰撞 metric 使用的上游 occupancy grid
- `{outputs_base}/{source_name}/grid_meta/{scene_id}_{frame_id}.json`
  - 提供 `voxel_params.origin`、`voxel_params.voxel_size` 和 `grid_shape`
- 数据集 adapter 可加载的原始 scene/object 标注
  - 用于 direction metric 获取参考物 3D box

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
ignored_support_layers 后的 occupied_voxel_count == 0
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
2. 若 `pred_box_world[:3]` 与 `gt_box_world[:3]` 三轴绝对误差都不超过 1cm，则直接判定方向正确。
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
- `center_abs_errors_cm`
- `reference_object_id`
- `reference_class_name`
- `reference_name`

如果旧 annotation 没有 `spatial_relation`，评测脚本会尝试从 `outputs/{source}/samples/*.json` 和原始 scene 临时重算。若重算失败，该样本的 direction metric 标记为未评估。

### 3. Size-Consistent

目标：判断预测 3D box 尺寸是否与原目标物体尺寸一致。

比较 `pred_box_world[3:6]` 与 `gt_box_world[3:6]`：

```text
axis_absolute_error_i = abs(pred_size_i - gt_size_i)
axis_relative_error_i = abs(pred_size_i - gt_size_i) / gt_size_i
mean_relative_size_error = mean(axis_relative_error)
max_axis_relative_size_error = max(axis_relative_error)
```

默认通过条件：

```text
max(axis_absolute_error) <= 2cm
```

输出字段包括：

- `size_consistent`
- `axis_absolute_size_errors_cm`
- `max_axis_absolute_size_error_cm`
- `relative_size_errors`
- `mean_relative_size_error`
- `max_axis_relative_size_error`
- `size_l2_cm`

## 输出文件

运行后输出：

- `metrics_summary.json`
  - 全局 summary
  - 按 `source_name` 分组 summary
  - direction confusion matrix
  - 阈值和输入文件记录
- `per_sample_metrics.json`
  - 每个样本的三类 metric 详细结果
- `per_sample_metrics.csv`
  - 仅在使用 `--write-csv` 时生成，便于表格分析

summary 中的覆盖率含义：

- `collision_coverage`: 成功计算 collision 的样本比例
- `direction_coverage`: 成功计算 direction 的样本比例
- `size_coverage`: 成功计算 size 的样本比例
- `full_metric_coverage`: 三类 metric 都成功计算的样本比例
- `mean_occupied_collision_ratio` / `median_occupied_collision_ratio`: 已评估样本的 OCCUPIED 体素碰撞比例统计
- `mean_unknown_overlap_ratio` / `median_unknown_overlap_ratio`: 已评估样本的 UNKNOWN 体素覆盖比例统计

`placement_success_rate` 只在三类 metric 都成功计算的样本上统计。

## 运行示例

```bash
conda run -n spatial python tools/evaluate_multimodal_predictions.py \
    --predictions outputs/infer_ptv3/predictions.json \
    --annotation-dir data/annotations/placement_multimodal_simple \
    --outputs-base outputs \
    --output-dir outputs/infer_ptv3_eval \
    --write-csv
```

可调整阈值：

```bash
conda run -n spatial python tools/evaluate_multimodal_predictions.py \
    --predictions outputs/infer_ptv3/predictions.json \
    --annotation-dir data/annotations/placement_multimodal_simple \
    --output-dir outputs/infer_ptv3_eval_strict \
    --collision-ratio-threshold 0.005 \
    --size-mean-rel-threshold 0.08 \
    --size-max-rel-threshold 0.12
```

快速 smoke test 可只评测前几个样本：

```bash
conda run -n spatial python tools/evaluate_multimodal_predictions.py \
    --predictions outputs/infer_ptv3/predictions.json \
    --annotation-dir data/annotations/placement_multimodal_simple \
    --output-dir outputs/infer_ptv3_eval_smoke \
    --limit 5
```

## 兼容策略

新数据推荐先重新运行：

1. `tools/auto_label.py` 生成带 `spatial_relation` 的 `all_labels.json`
2. `tools/build_multimodal_dataset.py` 生成带 scene/object/relation 字段的 annotation
3. `scripts/infer_multimodal.py` 生成 predictions
4. `tools/evaluate_multimodal_predictions.py` 评测

旧数据无需立刻重跑上游：

- size metric 可直接使用 `predictions.json` 中的 `gt_box_world`
- collision metric 需要 `outputs/{source}/occupancy_grids/{scene_id}_{frame_id}.npy` 和 `outputs/{source}/grid_meta/{scene_id}_{frame_id}.json`
- direction metric 会尽量用旧 placement sample 和 scene 重算结构化关系；若缺少原始 scene 或 reference object，则只跳过 direction
