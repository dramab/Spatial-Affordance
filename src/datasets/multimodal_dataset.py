"""
src/datasets/multimodal_dataset.py
---------------------------------
职责：加载 placement 多模态 train/valid/test 标注，并完成点云、3D box 与文本样本的预处理。

用法：
    from torch.utils.data import DataLoader
    from src.datasets.multimodal_dataset import (
        PlacementMultimodalDataset,
        placement_multimodal_collate_fn,
    )

    dataset = PlacementMultimodalDataset(
        annotation_dir="data/annotations/placement_multimodal",
        split="train",
        num_points=2048,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=placement_multimodal_collate_fn,
    )
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.coord_utils import normalize_box_from_aligned


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALID_SPLITS = {"train", "valid", "test"}
YAW_NORMALIZE_SCALE = 180.0
INVALID_POINT_VALUE = np.nan
DEFAULT_IMAGE_SIZE = (480, 640)


def _load_json(json_path: Path) -> dict[str, Any]:
    """
    用法: payload = _load_json(Path("data/annotations/placement_multimodal/train.json"))
    作用: 读取 JSON 标注文件
    输入: json_path: Path，JSON 文件路径
    输出: dict，解析后的 JSON 对象
    """
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_repo_path(path_str: str) -> Path:
    """
    用法: abs_path = _resolve_repo_path("outputs/demo/file.png")
    作用: 将仓库内相对路径解析为绝对路径
    输入: path_str: str，仓库根目录下的相对路径
    输出: Path，绝对路径
    """
    return PROJECT_ROOT / Path(path_str)


def _read_ascii_ply_points(ply_path: Path) -> np.ndarray:
    """
    用法: points = _read_ascii_ply_points(Path("outputs/hope/point_clouds/scene_0000_0000.ply"))
    作用: 读取 ASCII PLY 文件中的 xyz 点云
    输入: ply_path: Path，PLY 点云路径
    输出: ndarray(N, 3)，点云 xyz 坐标
    """
    with ply_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    end_header_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == "end_header":
            end_header_idx = idx
            break
    if end_header_idx is None:
        raise ValueError(f"PLY header missing end_header: {ply_path}")

    payload = "".join(lines[end_header_idx + 1 :]).strip()
    if not payload:
        raise ValueError(f"PLY contains no vertex payload: {ply_path}")

    points = np.loadtxt(io.StringIO(payload), usecols=(0, 1, 2), dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid PLY point shape: {ply_path} -> {points.shape}")
    return points


def _load_image_tensor(image_path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    """
    用法: image = _load_image_tensor(Path("outputs/placement_rgb_bbox_vis/demo.png"), (224, 224))
    作用: 读取 RGB 图像，统一 resize 后转换为 CHW uint8 Tensor
    输入: image_path: Path，图像路径；image_size: tuple[int, int]，目标图像尺寸 (H, W)
    输出: Tensor(3, H, W)，RGB 图像张量
    """
    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")
        target_height, target_width = int(image_size[0]), int(image_size[1])
        if target_height <= 0 or target_width <= 0:
            raise ValueError("image_size values must be positive")
        image_rgb = image_rgb.resize((target_width, target_height), resample=Image.BILINEAR)
        image_np = np.asarray(image_rgb, dtype=np.uint8)
    return torch.from_numpy(np.transpose(image_np, (2, 0, 1)).copy())


def _wrap_yaw_degrees(yaw_degrees: float) -> float:
    """
    用法: yaw_wrapped = _wrap_yaw_degrees(270.0)
    作用: 将角度约束到 [-180, 180) 区间
    输入: yaw_degrees: float，原始角度
    输出: float，包裹后的角度
    """
    return ((float(yaw_degrees) + 180.0) % 360.0) - 180.0


def _compute_scene_normalization(
        points_xyz: np.ndarray,
        scale_eps: float) -> tuple[np.ndarray, float]:
    """
    用法: scene_center, scene_scale = _compute_scene_normalization(points_xyz, 1e-6)
    作用: 依据点云均值中心与最大轴跨度计算样本归一化参数
    输入: points_xyz: ndarray(N, 3)，点云坐标；scale_eps: float，尺度下限
    输出: tuple，分别为场景中心与场景尺度
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3), got {points_xyz.shape}")
    if points_xyz.shape[0] == 0:
        raise ValueError("points_xyz must contain at least one point")

    scene_center = points_xyz.mean(axis=0, dtype=np.float64)
    axis_span = points_xyz.max(axis=0) - points_xyz.min(axis=0)
    scene_scale = max(float(axis_span.max()), float(scale_eps))
    return scene_center.astype(np.float64), scene_scale


def _normalize_points(points_xyz: np.ndarray, scene_center: np.ndarray, scene_scale: float) -> np.ndarray:
    """
    用法: points_norm = _normalize_points(points_xyz, scene_center, scene_scale)
    作用: 对点云做去中心化和缩放归一化
    输入: points_xyz: ndarray(N, 3)；scene_center: ndarray(3,)；scene_scale: float
    输出: ndarray(N, 3)，归一化后的点云
    """
    normalized = np.asarray(points_xyz, dtype=np.float64) - np.asarray(scene_center, dtype=np.float64)
    normalized /= float(scene_scale)
    return normalized.astype(np.float32)


def _normalize_box(
        target_box: np.ndarray,
        scene_center: np.ndarray,
        scene_scale: float) -> np.ndarray:
    """
    用法: box_norm = _normalize_box(target_box, scene_center, scene_scale)
    作用: 对 7D 3D box 做平移、尺度与 yaw 归一化
    输入: target_box: ndarray(7,)；scene_center: ndarray(3,)；scene_scale: float
    输出: ndarray(7,)，归一化后的 3D box
    """
    box_norm = normalize_box_from_aligned(
        box_aligned=np.asarray(target_box, dtype=np.float64),
        scene_center=np.asarray(scene_center, dtype=np.float64),
        scene_scale=float(scene_scale),
    )
    box_norm = np.asarray(box_norm, dtype=np.float64)
    box_norm[6] = _wrap_yaw_degrees(float(target_box[6])) / float(YAW_NORMALIZE_SCALE)
    return box_norm.astype(np.float32)


def _pad_points_for_batch(point_tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    用法: padded_points, point_counts = _pad_points_for_batch(point_tensors)
    作用: 将可变长点云补齐到 batch 内统一长度
    输入: point_tensors: list[Tensor(N_i, 3)]，样本点云列表
    输出: tuple，分别为补齐后的点云批量张量与每个样本原始点数
    """
    if not point_tensors:
        raise ValueError("point_tensors must not be empty")

    point_counts = torch.tensor([points.shape[0] for points in point_tensors], dtype=torch.long)
    max_points = int(point_counts.max().item())
    padded_points: list[torch.Tensor] = []
    for points in point_tensors:
        if points.shape[0] == max_points:
            padded_points.append(points)
            continue
        pad_count = max_points - points.shape[0]
        pad_points = torch.full(
            (pad_count, points.shape[1]),
            float("nan"),
            dtype=points.dtype,
            device=points.device,
        )
        padded_points.append(torch.cat([points, pad_points], dim=0))

    return torch.stack(padded_points, dim=0), point_counts


class PlacementMultimodalDataset(Dataset):
    """
    作用：加载 placement 多模态标注，并输出适配模型训练的单样本数据。

    输入：
        annotation_dir: 标注目录，包含 train/valid/test JSON
        split: 数据划分，支持 train/valid/test，允许 val 作为 valid 别名
        prompt_key: 文本字段名，支持 prompt 或 polished_prompt
        image_size: tuple[int, int]，统一后的图像尺寸 (H, W)
        scale_eps: float，场景尺度下限
    输出：
        Dataset，可按索引返回归一化后的多模态样本
    """

    def __init__(
            self,
            annotation_dir: str | Path,
            split: str = "train",
            prompt_key: str = "polished_prompt",
            image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
            scale_eps: float = 1e-6):
        self.annotation_dir = Path(annotation_dir)
        self.split = "valid" if str(split).lower() == "val" else str(split).lower()
        self.prompt_key = str(prompt_key)
        self.scale_eps = float(scale_eps)

        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split}")
        if self.prompt_key not in {"prompt", "polished_prompt"}:
            raise ValueError("prompt_key must be either 'prompt' or 'polished_prompt'")
        if len(image_size) != 2:
            raise ValueError("image_size must be a length-2 tuple of positive integers")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        if min(self.image_size) <= 0:
            raise ValueError("image_size must be a length-2 tuple of positive integers")

        split_path = self.annotation_dir / f"{self.split}.json"
        if not split_path.exists():
            if self.split == "valid" or self.split == "test":
                self.payload = {
                    "schema_version": None,
                    "split": self.split,
                    "sample_count": 0,
                    "samples": [],
                }
            else:
                raise FileNotFoundError(f"Split annotation not found: {split_path}")
        else:
            self.payload = _load_json(split_path)

        self.samples = list(self.payload.get("samples", []))

    def __len__(self) -> int:
        """
        用法: dataset_size = len(dataset)
        作用: 返回当前 split 的样本数量
        输入: 无
        输出: int，样本数
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        用法: sample = dataset[0]
        作用: 读取并预处理单个多模态样本
        输入: index: int，样本索引
        输出: dict，包含图像、归一化点云、文本、归一化 3D box 与归一化元信息
        """
        sample = self.samples[index]
        rgb_path = _resolve_repo_path(str(sample["rgb_path"]))
        point_cloud_path = _resolve_repo_path(str(sample["point_cloud_path"]))
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        if not point_cloud_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {point_cloud_path}")

        image = _load_image_tensor(rgb_path, self.image_size)
        points_xyz = _read_ascii_ply_points(point_cloud_path)
        scene_center, scene_scale = _compute_scene_normalization(
            points_xyz=points_xyz,
            scale_eps=self.scale_eps,
        )
        points_xyz_norm = _normalize_points(
            points_xyz=points_xyz,
            scene_center=scene_center,
            scene_scale=scene_scale,
        )

        target_box = np.asarray(sample["placement"]["target_box"], dtype=np.float64)
        if target_box.shape != (7,):
            raise ValueError(f"target_box must have shape (7,), got {target_box.shape}")
        target_box_norm = _normalize_box(
            target_box=target_box,
            scene_center=scene_center,
            scene_scale=scene_scale,
        )

        text_input = str(sample[self.prompt_key]).strip()
        if not text_input:
            raise ValueError(f"Empty text field {self.prompt_key} for sample: {sample['sample_id']}")

        return {
            "sample_id": str(sample["sample_id"]),
            "image": image,
            "points_xyz_norm": torch.from_numpy(points_xyz_norm).to(torch.float32),
            "text_input": text_input,
            "target_box_norm": torch.from_numpy(target_box_norm).to(torch.float32),
            "norm_meta": {
                "scene_center": torch.from_numpy(scene_center.astype(np.float32)),
                "scene_scale": torch.tensor(scene_scale, dtype=torch.float32),
                "yaw_scale": torch.tensor(YAW_NORMALIZE_SCALE, dtype=torch.float32),
            },
            "rgb_path": str(sample["rgb_path"]),
            "point_cloud_path": str(sample["point_cloud_path"]),
            "camera": dict(sample.get("camera", {})),
        }


def placement_multimodal_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    用法: batch = placement_multimodal_collate_fn(samples)
    作用: 将单样本列表整理为可直接输入多模态模型的 batch
    输入: samples: list[dict]，Dataset 返回的样本列表
    输出: dict，包含 images、points_xyz、text_inputs、target_boxes_norm 与归一化元信息
    """
    if not samples:
        raise ValueError("samples must not be empty")

    image_shapes = {tuple(sample["image"].shape) for sample in samples}
    if len(image_shapes) != 1:
        raise ValueError(f"All images must share the same shape, got {sorted(image_shapes)}")

    images = torch.stack([sample["image"] for sample in samples], dim=0)
    points_xyz, point_counts = _pad_points_for_batch([sample["points_xyz_norm"] for sample in samples])
    target_boxes_norm = torch.stack([sample["target_box_norm"] for sample in samples], dim=0)

    return {
        "sample_ids": [sample["sample_id"] for sample in samples],
        "images": images,
        "points_xyz": points_xyz,
        "point_counts": point_counts,
        "text_inputs": [sample["text_input"] for sample in samples],
        "target_boxes_norm": target_boxes_norm,
        "norm_meta": {
            "scene_center": torch.stack(
                [sample["norm_meta"]["scene_center"] for sample in samples], dim=0),
            "scene_scale": torch.stack(
                [sample["norm_meta"]["scene_scale"] for sample in samples], dim=0),
            "yaw_scale": torch.stack(
                [sample["norm_meta"]["yaw_scale"] for sample in samples], dim=0),
        },
        "rgb_paths": [sample["rgb_path"] for sample in samples],
        "point_cloud_paths": [sample["point_cloud_path"] for sample in samples],
        "cameras": [sample["camera"] for sample in samples],
    }
