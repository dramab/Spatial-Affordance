#!/usr/bin/env python3
"""
tools/export_prediction_box_to_occupancy_ply.py
------------------------------------------------
将推理预测 3D box 叠加到对应样本的 occupancy PLY 点云中，便于可视化检查。

用法:
    conda run -n spatial python tools/export_prediction_box_to_occupancy_ply.py \
        --predictions outputs/infer_ptv3/predictions.json \
        --sample-id dopose__test_bin_000005_000000_obj_000002_2_p000 \
        --outputs-base outputs \
        --output-dir outputs/infer_ptv3_collision_ply

作用:
    - 从 predictions.json 中读取指定 sample_id 的 pred_box_world
    - 定位 outputs/{source}/occupancy_grids/{scene_id}_{frame_id}.ply
    - 保留原 occupancy PLY 点云，并追加预测 3D box 的彩色边线点
    - 输出叠加后的 PLY 和 summary JSON

输入:
    --predictions: 推理导出的 predictions.json
    --sample-id: 要导出的样本 ID，支持裸 sample_id 或 source__sample_id
    --outputs-base: placement 输出根目录，默认 outputs
    --output-dir: 导出目录

输出:
    output-dir/{sample_id}_occupied_with_pred_box.ply
    output-dir/{sample_id}_occupied_with_pred_box_summary.json

使用示例:
    conda run -n spatial python tools/export_prediction_box_to_occupancy_ply.py \
        --predictions outputs/infer_ptv3/predictions.json \
        --sample-id dopose__test_bin_000005_000000_obj_000002_2_p000 \
        --outputs-base outputs \
        --output-dir outputs/infer_ptv3_collision_ply \
        --edge-color 0 188 255
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.coord_utils import box7d_to_corners_world


DEFAULT_EDGE_COLOR = (0, 188, 255)
DEFAULT_EDGE_SPACING = 0.25
DEFAULT_EDGE_THICKNESS = 0.35
PLY_EDGE_INDICES = (
    (0, 1), (0, 2), (1, 3), (2, 3),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器。
    输入: 无。
    输出: argparse.ArgumentParser。
    """
    parser = argparse.ArgumentParser(
        description="将 predictions.json 中的预测 3D box 叠加到对应 occupancy PLY 点云"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="推理导出的 predictions.json",
    )
    parser.add_argument(
        "--sample-id",
        required=True,
        help="要导出的 sample_id，支持裸 sample_id 或 source__sample_id",
    )
    parser.add_argument(
        "--outputs-base",
        type=Path,
        default=Path("outputs"),
        help="placement 输出根目录，默认 outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/prediction_box_occupancy_ply"),
        help="导出目录",
    )
    parser.add_argument(
        "--box-field",
        default="pred_box_world",
        help="prediction 记录中的 7D box 字段名，默认 pred_box_world",
    )
    parser.add_argument(
        "--edge-color",
        type=int,
        nargs=3,
        default=DEFAULT_EDGE_COLOR,
        metavar=("R", "G", "B"),
        help="预测框边线 RGB 颜色，默认 0 188 255",
    )
    parser.add_argument(
        "--edge-spacing",
        type=float,
        default=DEFAULT_EDGE_SPACING,
        help="边线采样点间距，单位与场景一致，默认 0.25",
    )
    parser.add_argument(
        "--edge-thickness",
        type=float,
        default=DEFAULT_EDGE_THICKNESS,
        help="边线横向加粗采样偏移，单位与场景一致，默认 0.35",
    )
    return parser


def resolve_project_path(path_value: str | Path) -> Path:
    """
    用法: path = resolve_project_path("outputs/demo.json")
    作用: 将相对路径解析到仓库根目录。
    输入: path_value: str 或 Path。
    输出: 绝对 Path。
    """
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def path_to_record(path_value: Path) -> str:
    """
    用法: text = path_to_record(Path("outputs/demo.ply"))
    作用: 将路径转换为 summary JSON 中稳定记录的仓库相对路径。
    输入: path_value: Path。
    输出: str，相对仓库路径或绝对路径。
    """
    resolved_path = path_value.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def load_json(json_path: Path) -> Any:
    """
    用法: payload = load_json(Path("outputs/predictions.json"))
    作用: 读取 JSON 文件。
    输入: json_path: JSON 文件路径。
    输出: JSON 解析后的对象。
    """
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_path: Path, payload: Any) -> None:
    """
    用法: save_json(Path("outputs/summary.json"), payload)
    作用: 将对象保存为缩进 JSON。
    输入: json_path: 输出路径；payload: 可序列化对象。
    输出: None。
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_sample_selector(sample_selector: str) -> tuple[str | None, str]:
    """
    用法: source_name, sample_id = parse_sample_selector("hope__scene_0006_0135_obj_2_p000")
    作用: 解析用户输入的样本选择器，支持 source__sample_id 和裸 sample_id。
    输入: sample_selector: 命令行传入的样本选择字符串。
    输出: tuple(source_name 或 None, sample_id)。
    """
    selector = str(sample_selector)
    if "__" not in selector:
        return None, selector
    source_name, sample_id = selector.split("__", 1)
    if not source_name or not sample_id:
        raise ValueError(f"invalid sample selector: {sample_selector}")
    return source_name, sample_id


def make_output_stem(source_name: str, sample_id: str) -> str:
    """
    用法: stem = make_output_stem("hope", "scene_0006_0135_obj_2_p000")
    作用: 生成带 source 前缀的输出文件名前缀，避免跨 source 样本覆盖。
    输入: source_name: 数据源名称；sample_id: 样本 ID。
    输出: str，格式 source__sample_id。
    """
    return f"{source_name}__{sample_id}"


def find_prediction(predictions_payload: Mapping[str, Any], sample_selector: str) -> dict[str, Any]:
    """
    用法: prediction = find_prediction(payload, "hope__scene_0000_0000_obj_1_p000")
    作用: 从 predictions.json payload 中按 source/sample 查找预测记录。
    输入:
        predictions_payload: predictions.json 内容。
        sample_selector: 裸 sample_id 或 source__sample_id。
    输出: prediction dict。
    """
    target_source, target_sample_id = parse_sample_selector(sample_selector)
    matches = [
        dict(prediction)
        for prediction in predictions_payload.get("predictions", [])
        if str(prediction.get("sample_id")) == target_sample_id
        and (target_source is None or str(prediction.get("source_name")) == target_source)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        if target_source is None:
            raise KeyError(f"sample_id not found in predictions: {target_sample_id}")
        raise KeyError(
            f"sample not found in predictions: source={target_source}, sample_id={target_sample_id}"
        )

    candidates = [
        f"{prediction.get('source_name')}__{prediction.get('sample_id')}"
        for prediction in matches
    ]
    raise ValueError(
        "sample_id is ambiguous across sources, please pass source__sample_id. "
        f"candidates: {candidates}"
    )


def infer_scene_frame_from_sample_id(sample_id: str) -> tuple[str, str]:
    """
    用法: scene_id, frame_id = infer_scene_frame_from_sample_id(sample_id)
    作用: 从 placement sample_id 中解析 scene_id 和 frame_id。
    输入: sample_id，格式通常为 {scene_id}_{frame_id}_obj_{object_id}_pXXX。
    输出: tuple(scene_id, frame_id)。
    """
    prefix = str(sample_id).split("_obj_", 1)[0]
    if "_" not in prefix:
        raise ValueError(f"cannot infer scene/frame from sample_id: {sample_id}")
    scene_id, frame_id = prefix.rsplit("_", 1)
    if not scene_id or not frame_id:
        raise ValueError(f"invalid scene/frame inferred from sample_id: {sample_id}")
    return scene_id, frame_id


def get_scene_frame(prediction: Mapping[str, Any]) -> tuple[str, str]:
    """
    用法: scene_id, frame_id = get_scene_frame(prediction)
    作用: 优先读取 prediction 的 scene_id/frame_id，缺失时从 sample_id 推断。
    输入: prediction: 单条预测记录。
    输出: tuple(scene_id, frame_id)。
    """
    scene_id = prediction.get("scene_id")
    frame_id = prediction.get("frame_id")
    if scene_id is not None and frame_id is not None:
        return str(scene_id), str(frame_id)
    return infer_scene_frame_from_sample_id(str(prediction["sample_id"]))


def validate_box7d(box_value: Sequence[float], field_name: str) -> np.ndarray:
    """
    用法: box = validate_box7d(prediction["pred_box_world"], "pred_box_world")
    作用: 校验并转换 7D box。
    输入: box_value: 长度 7 的数值序列；field_name: 报错字段名。
    输出: ndarray(7,)，格式 [cx, cy, cz, sx, sy, sz, yaw_degrees]。
    """
    box = np.asarray(box_value, dtype=np.float64)
    if box.shape != (7,):
        raise ValueError(f"{field_name} must have shape (7,), got {box.shape}")
    if not np.isfinite(box).all():
        raise ValueError(f"{field_name} contains non-finite values")
    if np.any(box[3:6] <= 0.0):
        raise ValueError(f"{field_name} size values must be positive: {box[3:6].tolist()}")
    return box


def sample_box_edge_points(
    corners: np.ndarray,
    edge_indices: Sequence[tuple[int, int]] = PLY_EDGE_INDICES,
    spacing: float = DEFAULT_EDGE_SPACING,
    thickness: float = DEFAULT_EDGE_THICKNESS,
) -> np.ndarray:
    """
    用法: points = sample_box_edge_points(corners, spacing=0.25, thickness=0.35)
    作用: 将 3D box 的 12 条边采样为点，追加到 PLY 中显示边线。
    输入:
        corners: ndarray(8,3)，box 角点。
        edge_indices: 角点边连接关系。
        spacing: 沿边采样间距。
        thickness: 垂直边方向的加粗偏移，0 表示不加粗。
    输出: ndarray(N,3)，边线采样点。
    """
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError(f"corners must have shape (8,3), got {corners.shape}")
    if float(spacing) <= 0.0:
        raise ValueError("edge spacing must be positive")
    if float(thickness) < 0.0:
        raise ValueError("edge thickness must be non-negative")

    sampled_groups: list[np.ndarray] = []
    for start_idx, end_idx in edge_indices:
        start = corners[start_idx]
        end = corners[end_idx]
        edge_vec = end - start
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len <= 1.0e-9:
            continue

        num_steps = max(2, int(np.ceil(edge_len / float(spacing))) + 1)
        line_points = start[None, :] + np.linspace(0.0, 1.0, num_steps)[:, None] * edge_vec[None, :]
        sampled_groups.append(line_points)
        if thickness > 0.0:
            sampled_groups.extend(_make_thick_edge_offsets(line_points, edge_vec, float(thickness)))

    if not sampled_groups:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(sampled_groups).astype(np.float64)


def _make_thick_edge_offsets(line_points: np.ndarray, edge_vec: np.ndarray, thickness: float) -> list[np.ndarray]:
    """
    用法: groups = _make_thick_edge_offsets(line_points, edge_vec, 0.35)
    作用: 为一条边生成四组横向偏移点，使 PLY 中的边线更容易看见。
    输入: line_points: 边线基础点；edge_vec: 边方向向量；thickness: 偏移距离。
    输出: list[ndarray]，加粗后的边线点组。
    """
    direction = edge_vec / max(float(np.linalg.norm(edge_vec)), 1.0e-9)
    base_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(direction, base_axis))) > 0.9:
        base_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    normal_a = np.cross(direction, base_axis)
    normal_a = normal_a / max(float(np.linalg.norm(normal_a)), 1.0e-9)
    normal_b = np.cross(direction, normal_a)
    offsets = (thickness * normal_a, -thickness * normal_a, thickness * normal_b, -thickness * normal_b)
    return [line_points + offset[None, :] for offset in offsets]


def points_to_ply_rows(points: np.ndarray, color_rgb: Sequence[int]) -> list[str]:
    """
    用法: rows = points_to_ply_rows(points, (0, 188, 255))
    作用: 将彩色点转换为 ASCII PLY vertex 行。
    输入: points: ndarray(N,3)；color_rgb: RGB 三元组。
    输出: list[str]，每个元素为一行 vertex 文本。
    """
    color = np.asarray(color_rgb, dtype=int)
    if color.shape != (3,) or np.any(color < 0) or np.any(color > 255):
        raise ValueError(f"edge color must be three integers in [0,255], got {list(color_rgb)}")
    return [
        f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} "
        f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
        for point in np.asarray(points, dtype=np.float64)
    ]


def append_rows_to_ascii_ply(base_ply_path: Path, output_ply_path: Path, appended_rows: Sequence[str]) -> int:
    """
    用法: total = append_rows_to_ascii_ply(base_ply, output_ply, rows)
    作用: 复制 ASCII PLY 并追加 vertex 行，同时更新 header 中 element vertex 数。
    输入: base_ply_path: 原 PLY；output_ply_path: 输出 PLY；appended_rows: 追加点行。
    输出: int，输出 PLY 的总 vertex 数。
    """
    with base_ply_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    try:
        end_header_idx = lines.index("end_header\n")
    except ValueError as exc:
        raise ValueError(f"PLY header missing end_header: {base_ply_path}") from exc

    header = lines[:end_header_idx + 1]
    body = lines[end_header_idx + 1:]
    base_vertex_count = None
    for idx, line in enumerate(header):
        if line.startswith("element vertex "):
            base_vertex_count = int(line.split()[-1])
            header[idx] = f"element vertex {base_vertex_count + len(appended_rows)}\n"
            break
    if base_vertex_count is None:
        raise ValueError(f"PLY header missing element vertex: {base_ply_path}")
    if base_vertex_count != len(body):
        raise ValueError(
            f"PLY vertex count mismatch: header={base_vertex_count}, body={len(body)}, path={base_ply_path}"
        )

    output_ply_path.parent.mkdir(parents=True, exist_ok=True)
    with output_ply_path.open("w", encoding="utf-8") as f:
        f.writelines(header)
        f.writelines(body)
        f.writelines(appended_rows)
    return base_vertex_count + len(appended_rows)


def export_prediction_box_to_occupancy_ply(
    prediction: Mapping[str, Any],
    outputs_base: Path,
    output_dir: Path,
    box_field: str = "pred_box_world",
    edge_color: Sequence[int] = DEFAULT_EDGE_COLOR,
    edge_spacing: float = DEFAULT_EDGE_SPACING,
    edge_thickness: float = DEFAULT_EDGE_THICKNESS,
) -> dict[str, Any]:
    """
    用法: summary = export_prediction_box_to_occupancy_ply(prediction, outputs_base, output_dir)
    作用: 将单条 prediction 的 3D box 叠加到对应 occupancy PLY 并保存。
    输入:
        prediction: predictions.json 中的一条记录。
        outputs_base: placement 输出根目录。
        output_dir: 导出目录。
        box_field: 预测 box 字段名。
        edge_color: 预测框边线颜色。
        edge_spacing: 边线采样间距。
        edge_thickness: 边线加粗偏移。
    输出: dict，summary JSON 内容。
    """
    sample_id = str(prediction["sample_id"])
    source_name = str(prediction["source_name"])
    scene_id, frame_id = get_scene_frame(prediction)
    if box_field not in prediction:
        raise KeyError(f"missing box field {box_field!r} in prediction: {sample_id}")

    box = validate_box7d(prediction[box_field], box_field)
    corners = box7d_to_corners_world(box)
    edge_points = sample_box_edge_points(
        corners,
        spacing=edge_spacing,
        thickness=edge_thickness,
    )
    edge_rows = points_to_ply_rows(edge_points, edge_color)

    prefix = f"{scene_id}_{frame_id}"
    base_ply_path = outputs_base / source_name / "occupancy_grids" / f"{prefix}.ply"
    if not base_ply_path.exists():
        raise FileNotFoundError(f"occupancy PLY not found: {base_ply_path}")

    output_stem = make_output_stem(source_name, sample_id)
    output_ply_path = output_dir / f"{output_stem}_occupied_with_pred_box.ply"
    total_vertex_count = append_rows_to_ascii_ply(
        base_ply_path=base_ply_path,
        output_ply_path=output_ply_path,
        appended_rows=edge_rows,
    )
    base_vertex_count = total_vertex_count - len(edge_rows)
    return {
        "sample_id": sample_id,
        "source_name": source_name,
        "scene_id": scene_id,
        "frame_id": frame_id,
        "box_field": box_field,
        "output_stem": output_stem,
        "pred_box_world": box.tolist(),
        "base_occupancy_ply": path_to_record(base_ply_path),
        "output_ply": path_to_record(output_ply_path),
        "base_vertex_count": base_vertex_count,
        "pred_box_edge_vertex_count": len(edge_rows),
        "total_vertex_count": total_vertex_count,
        "pred_box_edge_color_rgb": [int(value) for value in edge_color],
        "edge_spacing": float(edge_spacing),
        "edge_thickness": float(edge_thickness),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """
    用法: summary = run(args)
    作用: 执行 CLI 主流程。
    输入: args: argparse.Namespace。
    输出: dict，导出 summary。
    """
    predictions_path = resolve_project_path(args.predictions)
    outputs_base = resolve_project_path(args.outputs_base)
    output_dir = resolve_project_path(args.output_dir)

    predictions_payload = load_json(predictions_path)
    prediction = find_prediction(predictions_payload, args.sample_id)
    summary = export_prediction_box_to_occupancy_ply(
        prediction=prediction,
        outputs_base=outputs_base,
        output_dir=output_dir,
        box_field=str(args.box_field),
        edge_color=args.edge_color,
        edge_spacing=float(args.edge_spacing),
        edge_thickness=float(args.edge_thickness),
    )
    summary["predictions"] = path_to_record(predictions_path)
    output_stem = make_output_stem(str(summary["source_name"]), str(summary["sample_id"]))
    summary_path = output_dir / f"{output_stem}_occupied_with_pred_box_summary.json"
    summary["summary_json"] = path_to_record(summary_path)
    save_json(summary_path, summary)
    return summary


def main() -> None:
    """
    用法: main()
    作用: CLI 入口，导出 PLY 并打印输出路径。
    输入: 无，参数来自命令行。
    输出: None。
    """
    args = build_parser().parse_args()
    summary = run(args)
    print("导出完成")
    print(f"样本: {summary['sample_id']}")
    print(f"输出 PLY: {summary['output_ply']}")
    print(f"summary: {summary['summary_json']}")
    print(f"预测框边线点数: {summary['pred_box_edge_vertex_count']}")


if __name__ == "__main__":
    main()

# 使用示例:
# conda run -n spatial python tools/export_prediction_box_to_occupancy_ply.py \
#     --predictions outputs/infer_ptv3/predictions.json \
#     --sample-id dopose__test_bin_000005_000000_obj_000002_2_p000 \
#     --outputs-base outputs \
#     --output-dir outputs/infer_ptv3_collision_ply
