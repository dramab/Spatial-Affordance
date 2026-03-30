#!/usr/bin/env python3
"""
tools/export_placement_rgb_bbox_vis.py
--------------------------------------
将 placement 输出目录中的每一个空位框样本导出为一张 RGB 投影可视化图片。

图片内容:
    - 原始物体 3D bbox 在 RGB 图上的投影
    - 空位框样本对应 3D bbox 在 RGB 图上的投影

用法:
    python tools/export_placement_rgb_bbox_vis.py \
        --inputs outputs/housecat6d_placement10 outputs/placement_hope5 \
        --output-dir outputs/placement_rgb_bbox_vis

作用:
    - 遍历一个或多个 placement 输出目录下的 samples/*.json
    - 将每个 JSON 文件中的 samples 数组展平
    - 为每个样本单独生成一张可视化图片
    - 所有图片统一输出到同一个目录，通过文件名区分来源与 sample_id

输入:
    --inputs: 一个或多个 placement 输出根目录
    --output-dir: 统一输出目录
    --limit: 可选，仅导出前 N 个样本用于快速验证
    --overwrite: 可选，覆盖已存在图片

输出:
    在输出目录下生成:
        - {source_dir}__{sample_id}.png

使用示例:
    python tools/export_placement_rgb_bbox_vis.py \
        --inputs outputs/housecat6d_placement10 outputs/placement_hope5 \
        --output-dir outputs/placement_rgb_bbox_vis \
        --limit 20
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.grid_ops import _get_bbox_corners
from src.utils.coord_utils import project_world, transform_points


BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

COLOR_ORIGINAL = (255, 109, 0)
COLOR_PLACEMENT = (0, 230, 118)
LINE_WIDTH_ORIGINAL = 4
LINE_WIDTH_PLACEMENT = 4


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser，配置完成的解析器
    """
    parser = argparse.ArgumentParser(
        description="导出 placement 空位框样本的 RGB 3D bbox 投影图"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        help="一个或多个 placement 输出根目录",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="所有样本图片统一输出目录",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅导出前 N 个样本，用于快速验证",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出图片",
    )
    return parser


def load_yaml_config(config_path: Path) -> dict:
    """
    用法: cfg = load_yaml_config(Path("configs/annotation/placement.yaml"))
    作用: 读取 YAML 配置文件
    输入: config_path: Path，配置文件路径
    输出: dict，配置内容
    """
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_adapter(cfg: dict):
    """
    用法: adapter = build_adapter(cfg)
    作用: 根据配置创建数据集适配器
    输入: cfg: dict，YAML 配置内容
    输出: DatasetAdapter，对应数据集适配器实例
    """
    ds_cfg = cfg.get("dataset", {})
    ds_type = ds_cfg.get("type", "hope")

    if ds_type == "hope":
        from src.datasets.hope_adapter import HopeAdapter
        return HopeAdapter(
            root_dir=ds_cfg["root_dir"],
            mesh_dir=ds_cfg.get("mesh_dir"),
            frame_step=ds_cfg.get("frame_step", 60),
        )

    if ds_type == "housecat6d":
        from src.datasets.housecat6d_adapter import HouseCat6DAdapter
        return HouseCat6DAdapter(
            root_dir=ds_cfg["root_dir"],
            frame_step=ds_cfg.get("frame_step", 60),
        )

    raise ValueError(f"Unsupported dataset type: {ds_type}")


def infer_config_path(input_dir: Path) -> Path:
    """
    用法: config_path = infer_config_path(Path("outputs/placement_hope5"))
    作用: 根据 placement 输出目录名推断对应数据集配置
    输入: input_dir: Path，placement 输出根目录
    输出: Path，对应 YAML 配置路径
    """
    name = input_dir.name.lower()
    if "housecat" in name:
        return PROJECT_ROOT / "configs/annotation/placement_housecat6d.yaml"
    return PROJECT_ROOT / "configs/annotation/placement.yaml"


def collect_sample_jsons(input_dirs: Iterable[Path]) -> List[Path]:
    """
    用法: files = collect_sample_jsons([Path("outputs/placement_hope5")])
    作用: 收集输入目录下 samples 子目录中的 JSON 文件
    输入: input_dirs: Iterable[Path]，placement 输出根目录列表
    输出: List[Path]，排序后的样本 JSON 路径列表
    """
    sample_files: List[Path] = []
    for input_dir in input_dirs:
        samples_dir = input_dir / "samples"
        if not samples_dir.exists():
            raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
        sample_files.extend(sorted(samples_dir.glob("*.json")))
    return sorted(sample_files)


def iter_sample_records(sample_files: Iterable[Path]) -> Iterator[Tuple[Path, dict]]:
    """
    用法: for sample_file, record in iter_sample_records(files): ...
    作用: 将样本 JSON 中的 samples 数组展平为逐条样本记录
    输入: sample_files: Iterable[Path]，样本 JSON 文件路径列表
    输出: Iterator[(Path, dict)]，样本文件路径与单条样本记录
    """
    for sample_file in sample_files:
        with sample_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for record in payload.get("samples", []):
            yield sample_file, record


def get_scene_path(adapter, scene_id: str) -> Path:
    """
    用法: scene_path = get_scene_path(adapter, "scene_0001")
    作用: 根据 scene_id 生成适配器对应的数据场景目录路径
    输入: adapter: DatasetAdapter；scene_id: str，场景标识
    输出: Path，场景目录路径
    """
    return Path(adapter.root_dir) / scene_id


def get_scene_cache_key(source_dir: Path, scene_id: str, frame_id: str) -> Tuple[str, str, str]:
    """
    用法: key = get_scene_cache_key(source_dir, scene_id, frame_id)
    作用: 构建场景缓存键，避免重复加载同一帧
    输入: source_dir: Path；scene_id: str；frame_id: str
    输出: tuple，可哈希缓存键
    """
    return source_dir.name, scene_id, frame_id


def load_scene_cached(scene_cache: Dict[Tuple[str, str, str], object],
                      adapter,
                      source_dir: Path,
                      scene_id: str,
                      frame_id: str):
    """
    用法: scene = load_scene_cached(cache, adapter, source_dir, scene_id, frame_id)
    作用: 加载场景帧并使用内存缓存减少重复 IO
    输入: scene_cache: dict，场景缓存；adapter: DatasetAdapter；
         source_dir: Path；scene_id: str；frame_id: str
    输出: SceneData，加载后的通用场景对象
    """
    key = get_scene_cache_key(source_dir, scene_id, frame_id)
    if key not in scene_cache:
        scene_path = get_scene_path(adapter, scene_id)
        scene_cache[key] = adapter.load_scene(str(scene_path), frame_id)
    return scene_cache[key]


def draw_projected_bbox(draw: ImageDraw.ImageDraw,
                        corners_world: np.ndarray,
                        K: np.ndarray,
                        E_w2c: np.ndarray,
                        color: Tuple[int, int, int],
                        width: int) -> None:
    """
    用法: draw_projected_bbox(draw, corners_world, K, E_w2c, color, width)
    作用: 将 3D bbox 投影到 RGB 图像上并绘制线框
    输入: draw: ImageDraw.ImageDraw 绘图对象；corners_world: (8, 3)；
         K: (3, 3)；E_w2c: (4, 4)；color: RGB 三元组；width: int 线宽
    输出: None，直接在 draw 对象上绘制
    """
    uv, z_cam = project_world(corners_world, K, E_w2c)
    for i, j in BOX_EDGES:
        if z_cam[i] <= 0 or z_cam[j] <= 0:
            continue
        draw.line(
            [
                (float(uv[i, 0]), float(uv[i, 1])),
                (float(uv[j, 0]), float(uv[j, 1])),
            ],
            fill=color,
            width=width,
        )


def render_sample_image(scene, sample_record: dict) -> Image.Image:
    """
    用法: image = render_sample_image(scene, sample_record)
    作用: 为单个空位框样本生成 RGB 投影可视化图片
    输入: scene: SceneData，原始场景帧；sample_record: dict，单个样本记录
    输出: PIL.Image.Image，绘制完成的图片
    """
    rgb_image = Image.fromarray(np.asarray(scene.rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(rgb_image)

    K = np.array(
        [
            [scene.camera.fx, 0.0, scene.camera.cx],
            [0.0, scene.camera.fy, scene.camera.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    E_w2c = np.linalg.inv(np.asarray(scene.camera.E_c2w, dtype=np.float64))
    corners_obj = _get_bbox_corners(np.asarray(sample_record["canonical_aabb_object"], dtype=np.float64))

    original_pose_world = np.asarray(sample_record["original_pose_world"], dtype=np.float64)
    placed_pose_world = np.asarray(sample_record["transform_world"], dtype=np.float64)

    draw_projected_bbox(
        draw=draw,
        corners_world=transform_points(corners_obj, original_pose_world),
        K=K,
        E_w2c=E_w2c,
        color=COLOR_ORIGINAL,
        width=LINE_WIDTH_ORIGINAL,
    )
    draw_projected_bbox(
        draw=draw,
        corners_world=transform_points(corners_obj, placed_pose_world),
        K=K,
        E_w2c=E_w2c,
        color=COLOR_PLACEMENT,
        width=LINE_WIDTH_PLACEMENT,
    )
    return rgb_image


def build_output_path(output_dir: Path, source_dir: Path, sample_id: str) -> Path:
    """
    用法: out_path = build_output_path(output_dir, source_dir, sample_id)
    作用: 生成统一输出目录下的图片路径
    输入: output_dir: Path；source_dir: Path；sample_id: str
    输出: Path，输出图片路径
    """
    safe_name = f"{source_dir.name}__{sample_id}.png"
    return output_dir / safe_name


def export_samples(input_dirs: List[Path],
                   output_dir: Path,
                   limit: int = None,
                   overwrite: bool = False) -> int:
    """
    用法: count = export_samples(input_dirs, output_dir, limit=10)
    作用: 批量导出所有空位框样本的 RGB 投影可视化图片
    输入: input_dirs: List[Path]，placement 输出目录；
         output_dir: Path，统一输出目录；
         limit: int | None，导出上限；
         overwrite: bool，是否覆盖已有图片
    输出: int，实际导出的图片数量
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_files = collect_sample_jsons(input_dirs)

    adapters = {}
    for input_dir in input_dirs:
        config_path = infer_config_path(input_dir)
        adapters[input_dir.resolve()] = build_adapter(load_yaml_config(config_path))

    scene_cache: Dict[Tuple[str, str, str], object] = {}
    exported = 0

    for sample_file, sample_record in iter_sample_records(sample_files):
        source_dir = sample_file.parents[1]
        output_path = build_output_path(output_dir, source_dir, sample_record["sample_id"])
        if output_path.exists() and not overwrite:
            continue

        adapter = adapters[source_dir.resolve()]
        scene = load_scene_cached(
            scene_cache=scene_cache,
            adapter=adapter,
            source_dir=source_dir,
            scene_id=str(sample_record["scene_id"]),
            frame_id=str(sample_record["frame_id"]),
        )
        image = render_sample_image(scene, sample_record)
        image.save(output_path)

        exported += 1
        if exported % 50 == 0:
            print(f"已导出 {exported} 张图片，最新输出: {output_path}")
        if limit is not None and exported >= limit:
            break

    return exported


def main() -> None:
    """
    用法: main()
    作用: 执行导出 placement RGB 3D bbox 投影图的 CLI 主流程
    输入: 无，参数来自命令行
    输出: None，在终端打印导出结果
    """
    args = build_parser().parse_args()
    input_dirs = [path.resolve() for path in args.inputs]
    output_dir = args.output_dir.resolve()

    exported = export_samples(
        input_dirs=input_dirs,
        output_dir=output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
    )

    print("导出完成")
    print(f"输入目录数: {len(input_dirs)}")
    print(f"统一输出目录: {output_dir}")
    print(f"实际导出图片数: {exported}")


if __name__ == "__main__":
    main()
