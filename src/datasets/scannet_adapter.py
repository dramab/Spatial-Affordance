"""
src/datasets/scannet_adapter.py
--------------------------------
ScanNet extracted_scans 数据集适配器。

封装 ScanNet 已抽帧目录的读取与单位转换：
- 深度缩放: raw uint16 (mm) -> cm
- 相机位姿: pose/*.txt 为 camera->world，平移 m -> cm
- 物体尺寸: 从 *_vh_clean_2.ply + *_vh_clean_2.0.010000.segs.json + aggregation 计算实例 mesh AABB
- 物体过滤: 默认仅保留当前帧 2D instance 可见且非结构类的实例

用法:
    from src.datasets.scannet_adapter import ScanNetAdapter
    adapter = ScanNetAdapter(
        root_dir="/data/jiajun.xie/ScanNet/data/extracted_scans",
        frame_step=100,
    )
    scene_data = adapter.load_scene(
        "/data/jiajun.xie/ScanNet/data/extracted_scans/scene0081_02",
        "0",
    )
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from src.annotation.free_bbox.datatypes import CameraParams, ObjectInfo, SceneData
from src.datasets.base_adapter import DatasetAdapter


MM_TO_CM = 0.1
M_TO_CM = 100.0
DEFAULT_INSTANCE_DIR = "2d-instance"
DEFAULT_EXCLUDED_LABELS = {
    "wall",
    "floor",
    "ceiling",
    "window",
    "door",
    "curtain",
    "shower",
    "shower curtain",
    "blinds",
}


@dataclass(frozen=True)
class _InstanceMeshInfo:
    """
    单个 ScanNet 3D 实例的缓存信息。

    属性:
        object_id: int，ScanNet aggregation 中的 objectId
        label: str，类别名称
        center_cm: ndarray(3,)，实例 mesh AABB 中心，单位 cm
        size_cm: ndarray(3,)，实例 mesh AABB 尺寸，单位 cm
    """
    object_id: int
    label: str
    center_cm: np.ndarray
    size_cm: np.ndarray


def _load_json(path: Path):
    """
    用法: payload = _load_json(Path("scene.aggregation.json"))
    作用: 读取 JSON 文件
    输入: path: Path，JSON 路径
    输出: dict 或 list，JSON 内容
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sort_frame_ids(frame_ids: set[str]) -> List[str]:
    """
    用法: ordered = _sort_frame_ids({"10", "2"})
    作用: 按数值优先、字符串兜底的方式稳定排序帧 ID
    输入: frame_ids: set[str]，文件 stem 集合
    输出: list[str]，排序后的帧 ID
    """
    return sorted(
        frame_ids,
        key=lambda value: (
            not value.isdigit(),
            int(value) if value.isdigit() else value,
        ),
    )


def _read_matrix(path: Path) -> np.ndarray:
    """
    用法: mat = _read_matrix(Path("pose/0.txt"))
    作用: 读取 ScanNet 文本矩阵文件
    输入: path: Path，矩阵文本路径
    输出: ndarray(4,4)，float64 矩阵
    """
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix in {path}, got {matrix.shape}")
    return matrix


def _resolve_scene_file(scene_dir: Path, suffix: str) -> Path:
    """
    用法: path = _resolve_scene_file(scene_dir, "_vh_clean_2.ply")
    作用: 根据 scene_id 和固定后缀解析 ScanNet 场景文件
    输入: scene_dir: Path；suffix: str，文件后缀
    输出: Path，存在的文件路径
    """
    path = scene_dir / f"{scene_dir.name}{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Missing ScanNet scene file: {path}")
    return path


def _ply_scalar_dtype(type_name: str) -> str:
    """
    用法: dtype = _ply_scalar_dtype("float")
    作用: 将 PLY 标量类型映射为 numpy dtype 字符串
    输入: type_name: str，PLY 属性类型
    输出: str，numpy dtype 描述
    """
    mapping = {
        "char": "i1",
        "uchar": "u1",
        "int8": "i1",
        "uint8": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int16": "<i2",
        "uint16": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "int32": "<i4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    if type_name not in mapping:
        raise ValueError(f"Unsupported PLY scalar type: {type_name}")
    return mapping[type_name]


def _read_ply_xyz(path: Path) -> np.ndarray:
    """
    用法: xyz = _read_ply_xyz(Path("scene_vh_clean_2.ply"))
    作用: 读取二进制 little-endian PLY 顶点 xyz 坐标
    输入: path: Path，ScanNet PLY 文件路径
    输出: ndarray(N,3)，单位为 PLY 原始单位 m
    """
    with path.open("rb") as f:
        header = []
        vertex_count = None
        vertex_properties = []
        current_element = None

        while True:
            raw_line = f.readline()
            if raw_line == b"":
                raise ValueError(f"PLY header missing end_header: {path}")
            line = raw_line.decode("ascii", errors="replace").strip()
            header.append(line)

            if line.startswith("format ") and line != "format binary_little_endian 1.0":
                raise ValueError(f"Only binary_little_endian PLY is supported: {path}")
            if line.startswith("element "):
                parts = line.split()
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif line.startswith("property ") and current_element == "vertex":
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError(f"Unexpected list property in vertex element: {path}")
                vertex_properties.append((parts[2], _ply_scalar_dtype(parts[1])))
            elif line == "end_header":
                vertex_offset = f.tell()
                break

    if vertex_count is None:
        raise ValueError(f"PLY vertex count not found: {path}")
    if not {"x", "y", "z"}.issubset({name for name, _ in vertex_properties}):
        raise ValueError(f"PLY vertex xyz properties missing: {path}")

    dtype = np.dtype(vertex_properties)
    vertices = np.fromfile(path, dtype=dtype, count=vertex_count, offset=vertex_offset)
    return np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float64)


def _build_segment_to_vertex_indices(seg_indices: np.ndarray) -> Dict[int, np.ndarray]:
    """
    用法: lookup = _build_segment_to_vertex_indices(seg_indices)
    作用: 为每个 segment id 建立顶点索引列表，避免每个物体重复全量扫描
    输入: seg_indices: ndarray(N,)，每个顶点所属 segment id
    输出: dict[int, ndarray]，segment id 到顶点索引数组的映射
    """
    lookup: Dict[int, List[int]] = {}
    for vertex_idx, seg_id in enumerate(seg_indices.tolist()):
        lookup.setdefault(int(seg_id), []).append(vertex_idx)
    return {seg_id: np.asarray(indices, dtype=np.int64) for seg_id, indices in lookup.items()}


def _build_instance_mesh_cache(scene_dir: Path) -> Dict[int, _InstanceMeshInfo]:
    """
    用法: cache = _build_instance_mesh_cache(scene_dir)
    作用: 从 ScanNet mesh/segment/aggregation 计算所有实例的尺寸缓存
    输入: scene_dir: Path，单个 ScanNet scene 目录
    输出: dict[int, _InstanceMeshInfo]，objectId 到实例 mesh 信息
    """
    scene_id = scene_dir.name
    ply_path = _resolve_scene_file(scene_dir, "_vh_clean_2.ply")
    seg_path = _resolve_scene_file(scene_dir, "_vh_clean_2.0.010000.segs.json")
    aggregation_path = scene_dir / f"{scene_id}.aggregation.json"
    if not aggregation_path.exists():
        aggregation_path = _resolve_scene_file(scene_dir, "_vh_clean.aggregation.json")

    xyz_cm = _read_ply_xyz(ply_path) * M_TO_CM
    seg_indices = np.asarray(_load_json(seg_path)["segIndices"], dtype=np.int64)
    if len(seg_indices) != len(xyz_cm):
        raise ValueError(
            f"Segment count does not match vertex count in {scene_id}: "
            f"{len(seg_indices)} vs {len(xyz_cm)}"
        )

    segment_lookup = _build_segment_to_vertex_indices(seg_indices)
    aggregation = _load_json(aggregation_path)
    instance_infos: Dict[int, _InstanceMeshInfo] = {}

    for group in aggregation.get("segGroups", []):
        object_id = int(group["objectId"])
        vertex_indices = [
            segment_lookup[int(seg_id)]
            for seg_id in group.get("segments", [])
            if int(seg_id) in segment_lookup
        ]
        if not vertex_indices:
            continue

        points = xyz_cm[np.concatenate(vertex_indices)]
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        size = bbox_max - bbox_min
        if np.any(size <= 0.0):
            continue

        instance_infos[object_id] = _InstanceMeshInfo(
            object_id=object_id,
            label=str(group["label"]),
            center_cm=(bbox_min + bbox_max) * 0.5,
            size_cm=size,
        )

    return instance_infos


class ScanNetAdapter(DatasetAdapter):
    """
    ScanNet extracted_scans 数据集适配器。

    属性:
        root_dir: Path，包含 sceneXXXX_YY 子目录的数据根目录
        frame_step: int，批量处理帧采样步长
        instance_dir_name: str，2D instance 目录名
        min_visible_pixels: int，实例进入 SceneData 的最小可见像素数
        excluded_labels: set[str]，需要排除的结构类标签
    """

    def __init__(
        self,
        root_dir: str,
        frame_step: int = 100,
        instance_dir_name: str = DEFAULT_INSTANCE_DIR,
        min_visible_pixels: int = 1,
        excluded_labels: List[str] = None,
    ):
        """
        用法: adapter = ScanNetAdapter(root_dir, frame_step=100)
        作用: 初始化 ScanNet adapter 并配置可见实例过滤规则
        输入:
            root_dir: str，ScanNet extracted_scans 根目录
            frame_step: int，批量处理时的帧采样步长
            instance_dir_name: str，2D instance 目录名
            min_visible_pixels: int，实例最小可见像素数
            excluded_labels: list[str]，额外指定排除标签；None 使用默认结构类集合
        输出: None
        """
        self.root_dir = Path(root_dir)
        self.frame_step = int(frame_step)
        self.instance_dir_name = str(instance_dir_name)
        self.min_visible_pixels = int(min_visible_pixels)
        self.excluded_labels = {
            label.lower()
            for label in (excluded_labels if excluded_labels is not None else DEFAULT_EXCLUDED_LABELS)
        }
        self._instance_cache: Dict[Path, Dict[int, _InstanceMeshInfo]] = {}

    def load_scene(self, scene_path: str, frame_id: str) -> SceneData:
        """
        用法: scene = adapter.load_scene("/path/to/scene0081_02", "0")
        作用: 加载 ScanNet 单帧并转换为通用 SceneData
        输入:
            scene_path: str，scene 目录路径
            frame_id: str，帧 ID
        输出: SceneData，单位统一为 cm
        """
        scene_dir = Path(scene_path)
        scene_id = scene_dir.name

        depth_path = scene_dir / "depth" / f"{frame_id}.png"
        rgb_path = self._resolve_rgb_path(scene_dir, frame_id)
        pose_path = scene_dir / "pose" / f"{frame_id}.txt"
        instance_path = scene_dir / self.instance_dir_name / f"{frame_id}.png"
        intrinsic_path = scene_dir / "intrinsic" / "intrinsic_depth.txt"

        rgb = self._load_rgb_at_depth_resolution(rgb_path, depth_path)
        depth_cm = np.asarray(Image.open(depth_path), dtype=np.float32) * MM_TO_CM
        K_depth = _read_matrix(intrinsic_path)[:3, :3]
        E_c2w = _read_matrix(pose_path)
        E_c2w[:3, 3] *= M_TO_CM

        img_h, img_w = depth_cm.shape
        camera = CameraParams(
            fx=K_depth[0, 0],
            fy=K_depth[1, 1],
            cx=K_depth[0, 2],
            cy=K_depth[1, 2],
            E_c2w=E_c2w,
            img_w=img_w,
            img_h=img_h,
        )

        objects = self._build_visible_objects(scene_dir, instance_path)
        return SceneData(
            scene_id=scene_id,
            frame_id=frame_id,
            rgb=rgb,
            depth=depth_cm,
            camera=camera,
            objects=objects,
            unit="cm",
        )

    def list_scenes(self) -> List[Tuple[str, List[str]]]:
        """
        用法: scenes = adapter.list_scenes()
        作用: 列出 extracted_scans 根目录下所有可处理 scene 与帧 ID
        输入: 无
        输出: list[(scene_path, frame_ids)]，scene 路径与帧 ID 列表
        """
        results = []
        for scene_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            if not self._has_required_scene_files(scene_dir):
                continue

            color_ids = {path.stem for path in (scene_dir / "color").glob("*.jpg")}
            color_ids.update(path.stem for path in (scene_dir / "color").glob("*.png"))
            depth_ids = {path.stem for path in (scene_dir / "depth").glob("*.png")}
            pose_ids = {path.stem for path in (scene_dir / "pose").glob("*.txt")}
            instance_ids = {
                path.stem for path in (scene_dir / self.instance_dir_name).glob("*.png")
            }
            frame_ids = []
            for frame_id in _sort_frame_ids(color_ids & depth_ids & pose_ids & instance_ids):
                if self.frame_step > 1 and frame_id.isdigit() and int(frame_id) % self.frame_step != 0:
                    continue
                frame_ids.append(frame_id)

            if frame_ids:
                results.append((str(scene_dir), frame_ids))
        return results

    def _resolve_rgb_path(self, scene_dir: Path, frame_id: str) -> Path:
        """
        用法: path = self._resolve_rgb_path(scene_dir, "0")
        作用: 解析当前帧 RGB 路径，兼容 jpg/png
        输入: scene_dir: Path；frame_id: str
        输出: Path，RGB 文件路径
        """
        for suffix in (".jpg", ".png"):
            path = scene_dir / "color" / f"{frame_id}{suffix}"
            if path.exists():
                return path
        raise FileNotFoundError(f"RGB frame not found: {scene_dir}/color/{frame_id}.jpg")

    def _load_rgb_at_depth_resolution(self, rgb_path: Path, depth_path: Path) -> np.ndarray:
        """
        用法: rgb = self._load_rgb_at_depth_resolution(rgb_path, depth_path)
        作用: 读取 RGB 并缩放到 depth 分辨率，满足当前单相机 pipeline 假设
        输入: rgb_path: Path；depth_path: Path
        输出: ndarray(H,W,3)，uint8 RGB 图像
        """
        depth_image = Image.open(depth_path)
        target_size = depth_image.size
        rgb_image = Image.open(rgb_path).convert("RGB")
        if rgb_image.size != target_size:
            rgb_image = rgb_image.resize(target_size, Image.Resampling.BILINEAR)
        return np.asarray(rgb_image, dtype=np.uint8)

    def _get_instance_mesh_cache(self, scene_dir: Path) -> Dict[int, _InstanceMeshInfo]:
        """
        用法: cache = self._get_instance_mesh_cache(scene_dir)
        作用: 获取或构建单个 scene 的实例 mesh 尺寸缓存
        输入: scene_dir: Path，场景目录
        输出: dict[int, _InstanceMeshInfo]，objectId 到 mesh 信息
        """
        cache_key = scene_dir.resolve()
        if cache_key not in self._instance_cache:
            self._instance_cache[cache_key] = _build_instance_mesh_cache(scene_dir)
        return self._instance_cache[cache_key]

    def _build_visible_objects(self, scene_dir: Path, instance_path: Path) -> List[ObjectInfo]:
        """
        用法: objects = self._build_visible_objects(scene_dir, instance_path)
        作用: 根据当前帧 2D instance 可见性生成 ObjectInfo 列表
        输入: scene_dir: Path；instance_path: Path，当前帧 2D instance 路径
        输出: list[ObjectInfo]，过滤后的物体列表
        """
        instance_map = np.asarray(Image.open(instance_path))
        visible_values, visible_counts = np.unique(instance_map, return_counts=True)
        visible_counts_by_object_id = {
            int(value) - 1: int(count)
            for value, count in zip(visible_values, visible_counts)
            if int(value) > 0
        }
        instance_cache = self._get_instance_mesh_cache(scene_dir)
        objects = []

        for object_id in sorted(visible_counts_by_object_id):
            if visible_counts_by_object_id[object_id] < self.min_visible_pixels:
                continue
            if object_id not in instance_cache:
                continue

            info = instance_cache[object_id]
            if info.label.lower() in self.excluded_labels:
                continue

            half_size = info.size_cm * 0.5
            bbox3d_canonical = np.concatenate([-half_size, half_size])
            pose_world = np.eye(4, dtype=np.float64)
            pose_world[:3, 3] = info.center_cm

            objects.append(ObjectInfo(
                obj_id=f"obj_{info.object_id}",
                class_name=info.label,
                bbox3d_canonical=bbox3d_canonical,
                pose_world=pose_world,
            ))

        return objects

    def _has_required_scene_files(self, scene_dir: Path) -> bool:
        """
        用法: ok = self._has_required_scene_files(scene_dir)
        作用: 判断 scene 目录是否具备 adapter 需要的文件结构
        输入: scene_dir: Path，场景目录
        输出: bool，是否可处理
        """
        scene_id = scene_dir.name
        required_paths = [
            scene_dir / "color",
            scene_dir / "depth",
            scene_dir / "pose",
            scene_dir / self.instance_dir_name,
            scene_dir / "intrinsic" / "intrinsic_depth.txt",
            scene_dir / f"{scene_id}_vh_clean_2.ply",
            scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json",
        ]
        has_aggregation = (
            (scene_dir / f"{scene_id}.aggregation.json").exists()
            or (scene_dir / f"{scene_id}_vh_clean.aggregation.json").exists()
        )
        return all(path.exists() for path in required_paths) and has_aggregation
