#!/usr/bin/env python3
"""
scripts/infer_multimodal.py
---------------------------
职责：加载多模态 3D box checkpoint，批量推理指定 split，并将预测框恢复到世界坐标系后投影到原图。

用法：
    conda run -n spatial python scripts/infer_multimodal.py \
        --checkpoint outputs/multimodal_train/best.pth \
        --split test \
        --device cuda \
        --output-dir outputs/multimodal_infer

作用：
    - 从 checkpoint 读取模型配置与训练时的数据集配置
    - 复用 PlacementMultimodalDataset/placement_multimodal_collate_fn 执行推理
    - 将 pred_boxes_norm 恢复为世界坐标系 7D box，尺寸通道按 log(size_norm) 处理
    - 将 pred_object_centers_norm 恢复为世界坐标系移动前物体中心
    - 将预测放置 3D box 与预测移动前物体中心点投影回 annotation.rgb_path 对应的原尺寸图片
    - 导出 predictions.json 与逐样本可视化图片

输入：
    --checkpoint: 训练得到的 checkpoint 路径
    --split: 数据划分，支持 train/valid/test
    --annotation-dir: 可选，覆盖 checkpoint 中的数据集标注目录
    --device: 推理设备，如 cpu / cuda / cuda:0
    --output-dir: 推理输出目录
    --sample-ids: 可选，仅推理指定 sample_id 列表
    --limit: 可选，仅推理筛选后的前 N 个样本
    --batch-size: 可选，覆盖 checkpoint 中的验证 batch size
    --num-workers: 可选，覆盖 checkpoint 中的 DataLoader worker 数

输出：
    output_dir/
        - predictions.json
        - vis/{source_name}__{sample_id}.png

使用示例：
    conda run -n spatial python scripts/infer_multimodal.py \
        --checkpoint outputs/multimodal_train/best.pth \
        --split test \
        --sample-ids scene_0000_0155_obj_7_p000 \
        --output-dir outputs/multimodal_infer_demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import PlacementMultimodalDataset, placement_multimodal_collate_fn
from src.models import MultimodalModel
from src.utils.coord_utils import box7d_to_corners_world, project_world, wrap_yaw_degrees


SCHEMA_VERSION = "multimodal_inference_predictions/v1"
COLOR_PREDICTION = (0, 191, 255)
COLOR_OBJECT_CENTER = (255, 0, 180)
COLOR_OBJECT_CENTER_OUTLINE = (255, 255, 255)
COLOR_OBJECT_CENTER_BORDER = (0, 0, 0)
LINE_WIDTH_PREDICTION = 4
OBJECT_CENTER_RADIUS = 12
BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser，配置完成的解析器
    """
    parser = argparse.ArgumentParser(description="多模态 3D box 批量推理与原图投影")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="训练得到的 checkpoint 路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="推理数据划分，支持 train/valid/test",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="可选，覆盖 checkpoint 中的数据集标注目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备，如 cpu / cuda / cuda:0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/multimodal_infer"),
        help="推理输出目录",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="可选，仅推理指定 sample_id 列表",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="可选，仅推理筛选后的前 N 个样本",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="可选，覆盖 checkpoint 中的验证 batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="可选，覆盖 checkpoint 中的 DataLoader worker 数",
    )
    return parser


def resolve_project_path(path_value: str | Path) -> Path:
    """
    用法: abs_path = resolve_project_path("outputs/demo/file.png")
    作用: 将相对仓库路径解析为绝对路径
    输入: path_value: str | Path，相对或绝对路径
    输出: Path，绝对路径
    """
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def path_to_record(path_value: Path) -> str:
    """
    用法: text = path_to_record(Path("outputs/demo/file.png"))
    作用: 将路径转换为适合写入结果 JSON 的字符串
    输入: path_value: Path，待序列化路径
    输出: str，相对仓库路径或绝对路径
    """
    resolved_path = path_value.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """
    用法: cfg = load_yaml_config(Path("configs/base/model.yaml"))
    作用: 读取 YAML 配置文件
    输入: config_path: Path，配置文件路径
    输出: dict，配置内容
    """
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def select_device(device_name: str) -> torch.device:
    """
    用法: device = select_device("cuda")
    作用: 选择推理设备，并在 CUDA 不可用时自动回退到 CPU
    输入: device_name: str，设备名
    输出: torch.device，最终使用的设备
    """
    normalized_name = str(device_name).strip().lower()
    if normalized_name.startswith("cuda") and not torch.cuda.is_available():
        print("检测到配置使用 CUDA，但当前环境不可用，自动回退到 CPU。")
        normalized_name = "cpu"
    return torch.device(normalized_name)


def flatten_pred_boxes(pred_boxes_norm: torch.Tensor) -> torch.Tensor:
    """
    用法: boxes = flatten_pred_boxes(outputs["pred_boxes_norm"])
    作用: 将模型预测框统一整理为 (B, 7)
    输入: pred_boxes_norm: Tensor(B, 7) 或 Tensor(B, 1, 7)
    输出: Tensor(B, 7)，压缩后的预测框
    """
    if pred_boxes_norm.ndim == 2 and pred_boxes_norm.shape[-1] == 7:
        return pred_boxes_norm
    if pred_boxes_norm.ndim == 3 and pred_boxes_norm.shape[1] == 1 and pred_boxes_norm.shape[2] == 7:
        return pred_boxes_norm[:, 0, :]
    raise ValueError(
        "infer_multimodal.py 当前仅支持扁平 box 输出或兼容旧单 query box 输出，"
        f"got pred_boxes_norm shape={tuple(pred_boxes_norm.shape)}"
    )


def flatten_pred_centers(pred_centers_norm: torch.Tensor) -> torch.Tensor:
    """
    用法: centers = flatten_pred_centers(outputs["pred_object_centers_norm"])
    作用: 将模型预测中心统一整理为 (B, 3)
    输入: pred_centers_norm: Tensor(B, 3) 或 Tensor(B, 1, 3)
    输出: Tensor(B, 3)，压缩后的预测中心
    """
    if pred_centers_norm.ndim == 2 and pred_centers_norm.shape[-1] == 3:
        return pred_centers_norm
    if pred_centers_norm.ndim == 3 and pred_centers_norm.shape[1] == 1 and pred_centers_norm.shape[2] == 3:
        return pred_centers_norm[:, 0, :]
    raise ValueError(
        "infer_multimodal.py 当前仅支持扁平 center 输出或兼容旧单 query center 输出，"
        f"got pred_object_centers_norm shape={tuple(pred_centers_norm.shape)}"
    )


def resolve_configs_from_checkpoint(
        checkpoint: dict[str, Any],
        annotation_dir_override: Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    用法: model_cfg, dataset_cfg, dataloader_cfg = resolve_configs_from_checkpoint(checkpoint, None)
    作用: 从 checkpoint 中恢复模型、数据集与 DataLoader 配置
    输入: checkpoint: dict；annotation_dir_override: Path | None
    输出: tuple，分别为模型配置、数据集配置、DataLoader 配置
    """
    train_cfg = dict(checkpoint.get("train_config") or {})
    model_cfg = dict(checkpoint.get("model_config") or {})
    if not model_cfg:
        model_config_path = resolve_project_path(train_cfg.get("model_config_path", "configs/base/model.yaml"))
        model_cfg = load_yaml_config(model_config_path)

    dataset_cfg = dict(train_cfg.get("dataset") or {})
    dataloader_cfg = dict(train_cfg.get("dataloader") or {})
    if annotation_dir_override is not None:
        dataset_cfg["annotation_dir"] = annotation_dir_override.as_posix()
    dataset_cfg.setdefault("annotation_dir", "data/annotations/placement_multimodal_v2")
    dataset_cfg.setdefault("prompt_key", "polished_prompt")
    dataset_cfg.setdefault("image_size", [480, 640])
    dataset_cfg.setdefault("scale_eps", 1.0e-6)
    dataset_cfg.setdefault("max_points", None)
    dataset_cfg.setdefault("point_sample_seed", 42)
    return model_cfg, dataset_cfg, dataloader_cfg


def build_dataset(dataset_cfg: dict[str, Any], split: str) -> PlacementMultimodalDataset:
    """
    用法: dataset = build_dataset(dataset_cfg, "test")
    作用: 根据配置构建指定 split 的多模态数据集
    输入: dataset_cfg: dict，数据集配置；split: str，train/valid/test
    输出: PlacementMultimodalDataset，数据集实例
    """
    image_size = tuple(int(v) for v in dataset_cfg.get("image_size", (480, 640)))
    return PlacementMultimodalDataset(
        annotation_dir=resolve_project_path(dataset_cfg.get("annotation_dir", "data/annotations/placement_multimodal_v2")),
        split=split,
        prompt_key=str(dataset_cfg.get("prompt_key", "polished_prompt")),
        image_size=image_size,
        scale_eps=float(dataset_cfg.get("scale_eps", 1.0e-6)),
        max_points=dataset_cfg.get("max_points"),
        point_sample_seed=int(dataset_cfg.get("point_sample_seed", 42)),
    )


def select_subset_indices(
        samples: Sequence[dict[str, Any]],
        sample_ids: Sequence[str] | None,
        limit: int | None) -> list[int]:
    """
    用法: indices = select_subset_indices(dataset.samples, ["sample_a"], 10)
    作用: 根据 sample_id 与 limit 选择待推理样本索引
    输入: samples: 样本列表；sample_ids: 指定样本 ID 列表；limit: 样本上限
    输出: list[int]，选中的样本索引
    """
    if sample_ids:
        sample_index = {str(sample["sample_id"]): idx for idx, sample in enumerate(samples)}
        missing_ids = [sample_id for sample_id in sample_ids if sample_id not in sample_index]
        if missing_ids:
            raise KeyError(f"sample_ids not found in dataset: {missing_ids}")
        indices = [sample_index[sample_id] for sample_id in sample_ids]
    else:
        indices = list(range(len(samples)))

    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("--limit must be positive when provided")
        indices = indices[:int(limit)]
    return indices


def build_dataloader(
        subset: Subset,
        batch_size: int,
        num_workers: int,
        device: torch.device,
        persistent_workers: bool) -> DataLoader:
    """
    用法: loader = build_dataloader(subset, 8, 2, torch.device("cuda"), True)
    作用: 构建推理 DataLoader
    输入: subset: Subset；batch_size: int；num_workers: int；device: torch.device；
         persistent_workers: bool
    输出: DataLoader，可直接用于推理
    """
    pin_memory = device.type != "cpu"
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers) and int(num_workers) > 0,
        collate_fn=placement_multimodal_collate_fn,
    )


def denormalize_world_box(
        box_norm: np.ndarray,
        scene_center: np.ndarray,
        scene_scale: float,
        yaw_scale: float) -> np.ndarray:
    """
    用法: box_world = denormalize_world_box(box_norm, scene_center, scene_scale, 180.0)
    作用: 将归一化 7D box 恢复为世界坐标系 box，尺寸项从 log(size_norm) 恢复
    输入: box_norm: (7,)；scene_center: (3,)；scene_scale: float；yaw_scale: float
    输出: (7,) float64，世界坐标系 7D box
    """
    box_norm = np.asarray(box_norm, dtype=np.float64)
    if box_norm.shape != (7,):
        raise ValueError(f"box_norm must have shape (7,), got {box_norm.shape}")

    box_world = np.array(box_norm, dtype=np.float64, copy=True)
    box_world[:3] = box_world[:3] * float(scene_scale) + np.asarray(scene_center, dtype=np.float64)
    box_world[3:6] = np.exp(box_world[3:6]) * float(scene_scale)
    box_world[6] = wrap_yaw_degrees(box_world[6] * float(yaw_scale))
    return box_world


def denormalize_world_center(
        center_norm: np.ndarray,
        scene_center: np.ndarray,
        scene_scale: float) -> np.ndarray:
    """
    用法: center_world = denormalize_world_center(center_norm, scene_center, scene_scale)
    作用: 将归一化 3D center 恢复为世界坐标
    输入: center_norm: (3,)；scene_center: (3,)；scene_scale: float
    输出: ndarray(3,)，世界坐标系中心
    """
    center_norm = np.asarray(center_norm, dtype=np.float64)
    if center_norm.shape != (3,):
        raise ValueError(f"center_norm must have shape (3,), got {center_norm.shape}")
    return center_norm * float(scene_scale) + np.asarray(scene_center, dtype=np.float64)


def make_camera_matrices(camera_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    用法: K, E_w2c = make_camera_matrices(camera_dict)
    作用: 从标注中的 camera 字段构造投影所需矩阵
    输入: camera_dict: dict，相机参数字典
    输出: tuple，分别为 3x3 内参与 world->camera 外参
    """
    e_c2w = np.asarray(camera_dict["E_c2w"], dtype=np.float64)
    K = np.array([
        [float(camera_dict["fx"]), 0.0, float(camera_dict["cx"])],
        [0.0, float(camera_dict["fy"]), float(camera_dict["cy"])],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K, np.linalg.inv(e_c2w)


def draw_projected_bbox(
        draw: ImageDraw.ImageDraw,
        corners_world: np.ndarray,
        K: np.ndarray,
        E_w2c: np.ndarray,
        color: tuple[int, int, int],
        width: int) -> None:
    """
    用法: draw_projected_bbox(draw, corners_world, K, E_w2c, (0, 191, 255), 4)
    作用: 将 3D bbox 投影到图像上并绘制线框
    输入: draw: ImageDraw.ImageDraw；corners_world: (8, 3)；K: (3, 3)；
         E_w2c: (4, 4)；color: RGB 三元组；width: int
    输出: None，直接在 draw 对象上绘制
    """
    uv, z_cam = project_world(corners_world, K, E_w2c)
    for start_idx, end_idx in BOX_EDGES:
        if z_cam[start_idx] <= 0 or z_cam[end_idx] <= 0:
            continue
        draw.line(
            [
                (float(uv[start_idx, 0]), float(uv[start_idx, 1])),
                (float(uv[end_idx, 0]), float(uv[end_idx, 1])),
            ],
            fill=color,
            width=width,
        )


def draw_projected_point(
        draw: ImageDraw.ImageDraw,
        point_world: np.ndarray,
        K: np.ndarray,
        E_w2c: np.ndarray,
        color: tuple[int, int, int],
        radius: int) -> None:
    """
    用法: draw_projected_point(draw, center_world, K, E_w2c, (255, 0, 180), 12)
    作用: 将 3D 世界坐标点投影到图像上并绘制高对比中心标记
    输入: draw: ImageDraw.ImageDraw；point_world: (3,)；K/E_w2c: 相机矩阵；color: RGB；radius: 圆点半径
    输出: None，点在相机后方时不绘制
    """
    point = np.asarray(point_world, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError(f"point_world must have shape (3,), got {point.shape}")
    uv, z_cam = project_world(point.reshape(1, 3), K, E_w2c)
    if float(z_cam[0]) <= 0.0:
        return

    center_x = float(uv[0, 0])
    center_y = float(uv[0, 1])
    radius = max(1, int(radius))
    draw.ellipse(
        [
            (center_x - radius, center_y - radius),
            (center_x + radius, center_y + radius),
        ],
        fill=COLOR_OBJECT_CENTER_OUTLINE,
        outline=COLOR_OBJECT_CENTER_BORDER,
        width=max(2, radius // 4),
    )
    inner_radius = max(2, radius // 2)
    draw.ellipse(
        [
            (center_x - inner_radius, center_y - inner_radius),
            (center_x + inner_radius, center_y + inner_radius),
        ],
        fill=color,
        outline=COLOR_OBJECT_CENTER_BORDER,
        width=max(1, radius // 6),
    )


def save_prediction_visualization(
        rgb_path: Path,
        pred_box_world: np.ndarray,
        pred_object_center_world: np.ndarray,
        camera_dict: dict[str, Any],
        output_path: Path) -> None:
    """
    用法: save_prediction_visualization(rgb_path, pred_box_world, pred_object_center_world, camera_dict, output_path)
    作用: 将预测放置框与预测移动前物体中心点投影到原图并保存
    输入: rgb_path: Path；pred_box_world: (7,)；pred_object_center_world: (3,)；camera_dict: dict；output_path: Path
    输出: None，结果图写入 output_path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(rgb_path) as image:
        rgb_image = image.convert("RGB")
        draw = ImageDraw.Draw(rgb_image)
        K, E_w2c = make_camera_matrices(camera_dict)
        corners_world = box7d_to_corners_world(pred_box_world)
        draw_projected_bbox(
            draw=draw,
            corners_world=corners_world,
            K=K,
            E_w2c=E_w2c,
            color=COLOR_PREDICTION,
            width=LINE_WIDTH_PREDICTION,
        )
        draw_projected_point(
            draw=draw,
            point_world=pred_object_center_world,
            K=K,
            E_w2c=E_w2c,
            color=COLOR_OBJECT_CENTER,
            radius=OBJECT_CENTER_RADIUS,
        )
        rgb_image.save(output_path)


def save_json(output_path: Path, payload: dict[str, Any]) -> None:
    """
    用法: save_json(Path("outputs/demo/predictions.json"), payload)
    作用: 将结果字典写入 JSON 文件
    输入: output_path: Path；payload: dict
    输出: None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    """
    用法: payload = run_inference(args)
    作用: 执行批量推理、世界坐标恢复与投影导出主流程
    输入: args: argparse.Namespace，命令行参数
    输出: dict，完整推理结果 JSON 载荷
    """
    checkpoint_path = resolve_project_path(args.checkpoint)
    output_dir = resolve_project_path(args.output_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg, dataset_cfg, dataloader_cfg = resolve_configs_from_checkpoint(
        checkpoint=checkpoint,
        annotation_dir_override=args.annotation_dir,
    )
    decoder_cfg = dict(model_cfg.get("decoder", {}))
    if int(decoder_cfg.get("num_queries", 2)) != 2:
        raise ValueError("infer_multimodal.py 当前仅支持 decoder.num_queries=2 的 object/placement 双 query 推理")

    device = select_device(args.device)
    dataset = build_dataset(dataset_cfg, split=args.split)
    if len(dataset) == 0:
        raise ValueError(f"{args.split} split is empty, unable to run inference")

    selected_indices = select_subset_indices(
        samples=dataset.samples,
        sample_ids=args.sample_ids,
        limit=args.limit,
    )
    if not selected_indices:
        raise ValueError("no samples selected for inference")

    persistent_workers = bool(dataloader_cfg.get("persistent_workers", False))
    batch_size = int(args.batch_size if args.batch_size is not None else dataloader_cfg.get(
        "val_batch_size",
        dataloader_cfg.get("batch_size", 1),
    ))
    num_workers = int(args.num_workers if args.num_workers is not None else dataloader_cfg.get("num_workers", 0))
    subset = Subset(dataset, selected_indices)
    dataloader = build_dataloader(
        subset=subset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        persistent_workers=persistent_workers,
    )

    model = MultimodalModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_lookup = {str(sample["sample_id"]): sample for sample in dataset.samples}
    vis_dir = output_dir / "vis"
    predictions: list[dict[str, Any]] = []

    print(
        f"开始推理：checkpoint={checkpoint_path}，split={args.split}，"
        f"selected_samples={len(selected_indices)}，device={device.type}"
    )
    with torch.inference_mode():
        for batch in dataloader:
            outputs = model(
                points_xyz=batch["points_xyz"].to(device, non_blocking=True),
                images=batch["images"].to(device, non_blocking=True),
                text_inputs=batch["text_inputs"],
            )
            pred_boxes_norm = flatten_pred_boxes(outputs["pred_boxes_norm"]).detach().cpu().numpy()
            pred_object_centers_norm = flatten_pred_centers(
                outputs["pred_object_centers_norm"]
            ).detach().cpu().numpy()
            scene_centers = batch["norm_meta"]["scene_center"].detach().cpu().numpy()
            scene_scales = batch["norm_meta"]["scene_scale"].detach().cpu().numpy()
            yaw_scales = batch["norm_meta"]["yaw_scale"].detach().cpu().numpy()

            for batch_idx, sample_id in enumerate(batch["sample_ids"]):
                sample_meta = sample_lookup[str(sample_id)]
                pred_box_world = denormalize_world_box(
                    box_norm=pred_boxes_norm[batch_idx],
                    scene_center=scene_centers[batch_idx],
                    scene_scale=float(scene_scales[batch_idx]),
                    yaw_scale=float(yaw_scales[batch_idx]),
                )
                pred_object_center_world = denormalize_world_center(
                    center_norm=pred_object_centers_norm[batch_idx],
                    scene_center=scene_centers[batch_idx],
                    scene_scale=float(scene_scales[batch_idx]),
                )
                gt_box_world = np.asarray(sample_meta["placement"]["target_box"], dtype=np.float64)
                gt_object_center_world = np.asarray(
                    sample_meta["placement"]["object_center"],
                    dtype=np.float64,
                )
                rgb_path = resolve_project_path(sample_meta["rgb_path"])
                vis_path = vis_dir / f"{sample_meta['source_name']}__{sample_id}.png"
                save_prediction_visualization(
                    rgb_path=rgb_path,
                    pred_box_world=pred_box_world,
                    pred_object_center_world=pred_object_center_world,
                    camera_dict=batch["cameras"][batch_idx],
                    output_path=vis_path,
                )
                predictions.append({
                    "sample_id": str(sample_id),
                    "source_name": str(sample_meta["source_name"]),
                    "rgb_path": str(sample_meta["rgb_path"]),
                    "pred_box_norm": np.asarray(pred_boxes_norm[batch_idx], dtype=np.float64).tolist(),
                    "pred_box_world": pred_box_world.tolist(),
                    "gt_box_world": gt_box_world.tolist(),
                    "pred_object_center_norm": np.asarray(
                        pred_object_centers_norm[batch_idx],
                        dtype=np.float64,
                    ).tolist(),
                    "pred_object_center_world": pred_object_center_world.tolist(),
                    "gt_object_center_world": gt_object_center_world.tolist(),
                    "vis_path": path_to_record(vis_path),
                })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_path": path_to_record(checkpoint_path),
        "split": str(args.split),
        "sample_count": len(predictions),
        "predictions": predictions,
    }
    save_json(output_dir / "predictions.json", payload)
    return payload


def main() -> None:
    """
    用法: main()
    作用: 命令行入口，执行批量推理与结果导出
    输入: 无，参数来自命令行
    输出: None，在终端打印摘要信息
    """
    args = build_parser().parse_args()
    payload = run_inference(args)
    output_dir = resolve_project_path(args.output_dir)
    print("推理完成")
    print(f"输出目录: {output_dir}")
    print(f"结果文件: {output_dir / 'predictions.json'}")
    print(f"样本数量: {payload['sample_count']}")


if __name__ == "__main__":
    main()
