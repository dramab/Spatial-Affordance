# Spatial-Affordance 项目目录结构

> 端到端 3D Visual Grounding 项目：输入 RGB 图像 + 点云 + 文字 prompt，输出 3D bounding box（cx, cy, cz, l, w, h, yaw）。

```
Spatial-Affordance/
├── .gitignore
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── base/
│   │   ├── dataset.yaml          # 数据路径、点云采样数、图像分辨率、增强策略
│   │   ├── model.yaml            # backbone 类型、融合层数、head 参数
│   │   └── train.yaml            # lr、batch_size、scheduler、loss 权重、amp
│   ├── experiments/
│   │   └── baseline.yaml         # 继承 base，具体实验配置覆盖
│   └── annotation/
│       └── auto_label.yaml       # 自动标注流程配置
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # 主 Dataset 类（SpatialAffordanceDataset）
│   │   ├── dataloader.py         # DataLoader 工厂函数
│   │   ├── augmentation.py       # 图像 + 点云联合增强
│   │   └── collate_fn.py         # 变长输入的 batch collate
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── grounding_model.py    # 顶层模型，串联所有子模块
│   │   ├── backbones/
│   │   │   ├── __init__.py
│   │   │   ├── image_backbone.py # ResNet/ViT 图像特征提取
│   │   │   └── pc_backbone.py    # PointNet++ 点云特征提取
│   │   ├── encoders/
│   │   │   ├── __init__.py
│   │   │   └── text_encoder.py   # BERT 文本编码
│   │   ├── fusion/
│   │   │   ├── __init__.py
│   │   │   └── multimodal_fusion.py  # 三模态 cross-attention 融合
│   │   └── heads/
│   │       ├── __init__.py
│   │       └── bbox3d_head.py    # 3D bbox 回归头
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── bbox3d_loss.py        # L1 + sin/cos yaw + 3D GIoU
│   │   └── grounding_loss.py     # 整体损失（含 Hungarian matching）
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── grounding_metrics.py  # Acc@0.25、Acc@0.5、mIoU3D
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── box_utils.py          # 3D bbox 操作（IoU、NMS、坐标转换）
│   │   ├── pc_utils.py           # 点云工具（采样、归一化、投影）
│   │   ├── vis_utils.py          # Open3D 可视化
│   │   └── checkpoint.py         # 模型保存/加载
│   │
│   └── annotation/
│       ├── __init__.py
│       ├── pipeline.py           # 标注主流程编排
│       ├── grounded_sam.py       # Grounded-DINO + SAM2 获取 2D mask
│       ├── depth_lifter.py       # 2D mask -> 3D 点云子集
│       ├── bbox_fitter.py        # 点云簇 -> 3D bbox 拟合（PCA）
│       └── quality_filter.py     # 标注质量过滤
│
├── scripts/
│   ├── train.py                  # 训练入口（Hydra 配置 + resume）
│   ├── evaluate.py               # 评估脚本
│   ├── inference.py              # 单样本推理
│   └── run_annotation.py         # 自动标注入口
│
├── tools/
│   ├── visualize_dataset.py      # 数据集可视化
│   └── visualize_predictions.py  # 预测结果可视化
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_losses.py
│
├── data/                         # 不入 git
└── outputs/                      # 不入 git
```

---

## 核心数据流

```
RGB Image (B,3,H,W)  ->  ImageBackbone  ->  img_feats (B,C,H',W')
Point Cloud (B,N,6)  ->  PCBackbone     ->  pc_feats (B,M,C), pc_xyz (B,M,3)
Text tokens (B,L)    ->  TextEncoder    ->  text_feats (B,L,C)

[img_feats, pc_feats, text_feats]
    -> MultimodalFusion (cross-attention，文字作 query)
    -> fused_feats (B,M,C)
    -> BBox3DHead
    -> pred_boxes (B,7): cx, cy, cz, l, w, h, yaw
```

---

## 标注数据格式（JSON）

```json
{
  "scene_id": "scene0001_00",
  "sample_id": "scene0001_00_0042",
  "image_path": "data/processed/images/scene0001_00/0042.jpg",
  "pointcloud_path": "data/processed/pointclouds/scene0001_00.pcd",
  "prompt": "the brown wooden chair near the window",
  "bbox3d": {
    "center": [1.23, 0.45, 0.67],
    "size": [0.52, 0.48, 0.91],
    "yaw": 0.314
  },
  "annotation_meta": {
    "source": "auto",
    "confidence": 0.87,
    "num_points": 342
  }
}
```

---

## 自动标注流程

```
原始数据（RGB + 深度图 + 点云 + 文字描述）
    -> Grounded-DINO + SAM2  ->  2D mask（文字对应区域）
    -> depth_lifter           ->  3D 点云子集（mask 反投影）
    -> bbox_fitter            ->  3D bbox（PCA 拟合）
    -> quality_filter         ->  过滤低质量标注
    -> 输出标注 JSON
```

---

## 配置系统

使用 **Hydra + OmegaConf**，支持命令行覆盖和实验继承。

| 配置文件 | 职责 |
|---|---|
| `configs/base/model.yaml` | backbone 类型、融合层数、head 参数 |
| `configs/base/train.yaml` | lr、batch_size、scheduler、loss 权重、amp |
| `configs/base/dataset.yaml` | 数据路径、点云采样数、图像分辨率、增强策略 |
| `configs/experiments/baseline.yaml` | 继承 base，覆盖具体实验参数 |
| `configs/annotation/auto_label.yaml` | 自动标注流程参数 |
| `configs/annotation/placement.yaml` | 放置规划流程参数 |

---

## 放置规划模块 (free_bbox)

位于 `src/annotation/free_bbox/`，为场景中每个物体生成合法放置位置（3D bbox）。

### 核心文件

| 文件 | 职责 |
|---|---|
| `datatypes.py` | 数据类型定义（SceneData, PlacementConfig 等） |
| `occupancy.py` | 深度图 → 占据栅格（ray-casting） |
| `surface.py` | RANSAC 支撑面检测 |
| `collision.py` | FFT 2D 碰撞检测 |
| `filters.py` | 稳定性/可见性/遮挡过滤 |
| `cluster.py` | DBSCAN 聚类 |
| `pipeline.py` | 完整流程编排 |
| `state_tracker.py` | **OOM Kill 容错状态管理** |

### OOM Kill 容错机制

批量处理时，如果某帧占用内存过大被 OOM killer 终止，下次运行会自动跳过该帧避免反复 kill。

```bash
# 查看处理状态
python tools/run_placement.py --config configs/annotation/placement.yaml --status --output outputs/placement

# 重试之前失败的帧
python tools/run_placement.py --config configs/annotation/placement.yaml --batch --retry-failed --output outputs/placement

# 强制重新处理所有帧
python tools/run_placement.py --config configs/annotation/placement.yaml --batch --force --output outputs/placement
```

详见 [docs/placement_pipeline.md](docs/placement_pipeline.md)。
