"""
src/datasets/dopose_adapter.py
------------------------------
DoPose BOP 风格数据集适配器。

封装 DoPose 特定逻辑：
- 深度缩放: raw uint16 * depth_scale(mm) -> cm
- 相机位姿: scene_transformations.json 中 zivid_optical_frame -> scene_link，取逆后作为 camera->world，平移 m -> cm
- 物体位姿: scene_gt.json 中 object->camera，平移 mm -> cm
- 物体尺度: models_info.json 中 canonical AABB，mm -> cm
- 场景命名: test_bin/000001 合成为 test_bin_000001，避免后续输出重名

用法:
    from src.datasets.dopose_adapter import DoPoseAdapter
    adapter = DoPoseAdapter(
        root_dir="/data/jiajun.xie/Spatial-Affordance/data/dopose",
        models_info_path="/data/jiajun.xie/Spatial-Affordance/data/dopose/models/models_info.json",
        models_names_path="/data/jiajun.xie/Spatial-Affordance/data/dopose/models_names.json",
    )
    scene_data = adapter.load_scene(
        "/data/jiajun.xie/Spatial-Affordance/data/dopose/test_bin_000001",
        "000000",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from src.annotation.free_bbox.datatypes import CameraParams, ObjectInfo, SceneData
from src.datasets.base_adapter import DatasetAdapter


MM_TO_CM = 0.1
M_TO_CM = 100.0
DEFAULT_SUBSETS = ("test_bin", "test_table")
CAMERA_FRAME = "zivid_optical_frame"
WORLD_FRAME = "scene_link"


def _load_json(path: Path):
    """
    用法: payload = _load_json(Path("scene_gt.json"))
    作用: 读取 JSON 文件
    输入: path: Path，JSON 路径
    输出: dict 或 list，JSON 内容
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _frame_key(frame_id: str) -> str:
    """
    用法: key = _frame_key("000001")
    作用: 将零填充帧名转换为 BOP JSON 使用的数字字符串 key
    输入: frame_id: str，图片文件 stem
    输出: str，JSON key
    """
    return str(int(frame_id))


def _reshape_matrix(values, shape: Tuple[int, int]) -> np.ndarray:
    """
    用法: mat = _reshape_matrix(values, (3, 3))
    作用: 将列表或嵌套列表转换为 numpy 矩阵
    输入: values: list[float] 或 list[list[float]]；shape: 目标形状
    输出: ndarray，float64 矩阵
    """
    return np.asarray(values, dtype=np.float64).reshape(shape)


def _quat_to_rotation(quat: dict) -> np.ndarray:
    """
    用法: R = _quat_to_rotation({"x": 0, "y": 0, "z": 0, "w": 1})
    作用: 将 xyzw 四元数转换为 3x3 旋转矩阵
    输入: quat: dict，包含 x/y/z/w
    输出: ndarray(3,3)，旋转矩阵
    """
    x = float(quat["x"])
    y = float(quat["y"])
    z = float(quat["z"])
    w = float(quat["w"])
    norm = x * x + y * y + z * z + w * w
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float64,
    )


def _transform_from_record(record: dict) -> np.ndarray:
    """
    用法: T = _transform_from_record(transform_record)
    作用: 将 DoPose transformation 记录转为 4x4 齐次矩阵，平移保持原始 m 单位
    输入: record: dict，包含 translation 和 rotation_quaternion
    输出: ndarray(4,4)，齐次变换矩阵
    """
    translation = record["translation"]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_to_rotation(record["rotation_quaternion"])
    transform[:3, 3] = [
        float(translation["x"]),
        float(translation["y"]),
        float(translation["z"]),
    ]
    return transform


def _join_scene_id(subset: str, scene_name: str) -> str:
    """
    用法: scene_id = _join_scene_id("test_bin", "000001")
    作用: 生成后处理脚本兼容的无斜杠场景 ID
    输入: subset: str；scene_name: str
    输出: str，形如 test_bin_000001
    """
    return f"{subset}_{scene_name}"


class DoPoseAdapter(DatasetAdapter):
    """
    DoPose 数据集适配器。

    属性:
        root_dir: Path，DoPose 根目录
        models_info_path: Path，BOP models_info.json 路径
        models_names_path: Path，obj_id 到类别名映射路径
        subsets: tuple[str]，参与枚举的 split 名称
        frame_step: int，批量处理时的帧采样步长
        min_visib_fract: float，物体最小可见比例过滤阈值
    """

    def __init__(
        self,
        root_dir: str,
        models_info_path: str,
        models_names_path: str = None,
        subsets: List[str] = None,
        frame_step: int = 1,
        min_visib_fract: float = 0.0,
    ):
        """
        用法: adapter = DoPoseAdapter(root_dir, models_info_path)
        作用: 初始化 DoPose adapter 并缓存模型尺度与类别名
        输入:
            root_dir: str，DoPose 根目录
            models_info_path: str，models_info.json 路径
            models_names_path: str，可选 models_names.json 路径
            subsets: list[str]，如 ["test_bin", "test_table"]
            frame_step: int，批量处理时的帧采样步长
            min_visib_fract: float，过滤低可见物体的阈值
        输出: None
        """
        self.root_dir = Path(root_dir)
        self.models_info_path = Path(models_info_path)
        self.models_names_path = Path(models_names_path) if models_names_path else None
        self.subsets = tuple(subsets) if subsets else DEFAULT_SUBSETS
        self.frame_step = int(frame_step)
        self.min_visib_fract = float(min_visib_fract)
        self._models_info = _load_json(self.models_info_path)
        self._models_names = self._load_model_names()
        self._bbox_cache: Dict[int, np.ndarray] = {}

    def load_scene(self, scene_path: str, frame_id: str) -> SceneData:
        """
        用法: scene = adapter.load_scene("/path/to/dopose/test_bin_000001", "000000")
        作用: 加载 DoPose 单帧并转换为通用 SceneData
        输入:
            scene_path: str，真实 scene 目录或合成 scene 路径
            frame_id: str，零填充帧 ID
        输出: SceneData，单位统一为 cm
        """
        scene_dir, subset, scene_name = self._resolve_scene_path(scene_path)
        scene_id = _join_scene_id(subset, scene_name)
        key = _frame_key(frame_id)

        scene_camera = _load_json(scene_dir / "scene_camera.json")
        scene_gt = _load_json(scene_dir / "scene_gt.json")
        scene_gt_info = _load_json(scene_dir / "scene_gt_info.json")
        scene_transforms = _load_json(scene_dir / "scene_transformations.json")

        if key not in scene_camera or key not in scene_gt:
            raise KeyError(f"Frame {frame_id} not found in {scene_dir}")
        if key not in scene_transforms:
            raise KeyError(f"Frame {frame_id} transformations not found in {scene_dir}")

        rgb = np.asarray(Image.open(scene_dir / "rgb" / f"{frame_id}.png"), dtype=np.uint8)
        depth_raw = np.asarray(Image.open(scene_dir / "depth" / f"{frame_id}.png"), dtype=np.float32)

        camera_record = scene_camera[key]
        K = _reshape_matrix(camera_record["cam_K"], (3, 3))
        depth_cm = depth_raw * float(camera_record.get("depth_scale", 1.0)) * MM_TO_CM
        E_c2w = self._camera_to_world(scene_transforms[key])

        objects = self._build_objects(
            annotations=scene_gt[key],
            annotation_info=scene_gt_info.get(key, []),
            E_c2w=E_c2w,
        )

        img_h, img_w = rgb.shape[:2]
        camera = CameraParams(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            E_c2w=E_c2w,
            img_w=img_w,
            img_h=img_h,
        )

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
        作用: 列出 DoPose 中所有可处理 scene 与帧 ID
        输入: 无
        输出: list[(scene_path, frame_ids)]，scene_path 使用合成无斜杠路径
        """
        results = []
        for subset in self.subsets:
            subset_dir = self.root_dir / subset
            if not subset_dir.is_dir():
                continue
            for scene_dir in sorted(path for path in subset_dir.iterdir() if path.is_dir()):
                if not self._has_required_scene_files(scene_dir):
                    continue
                frame_ids = self._list_frame_ids(scene_dir)
                if frame_ids:
                    scene_path = self.root_dir / _join_scene_id(subset, scene_dir.name)
                    results.append((str(scene_path), frame_ids))
        return results

    def get_object_scale(self, class_name: str) -> np.ndarray:
        """
        用法: bbox = adapter.get_object_scale("choco_box")
        作用: 获取 DoPose 模型 canonical AABB
        输入: class_name: str，类别名或 obj_000001 格式名称
        输出: ndarray(6,)，[min_x, min_y, min_z, max_x, max_y, max_z]，单位 cm
        """
        obj_id = self._class_name_to_obj_id(class_name)
        return self._get_bbox_for_obj_id(obj_id)

    def _load_model_names(self) -> Dict[int, str]:
        """
        用法: names = self._load_model_names()
        作用: 读取 obj_id 到类别名的映射
        输入: 无
        输出: dict[int,str]，类别名映射
        """
        if self.models_names_path is None or not self.models_names_path.exists():
            return {}
        raw_names = _load_json(self.models_names_path)
        return {
            int(obj_id): str(record.get("name", f"obj_{int(obj_id):06d}"))
            for obj_id, record in raw_names.items()
        }

    def _resolve_scene_path(self, scene_path: str) -> Tuple[Path, str, str]:
        """
        用法: scene_dir, subset, scene_name = self._resolve_scene_path(scene_path)
        作用: 将真实路径或合成路径解析为真实 scene 目录
        输入: scene_path: str，可能为 root/test_bin/000001 或 root/test_bin_000001
        输出: (Path, subset, scene_name)
        """
        path = Path(scene_path)
        if path.is_dir() and path.parent.name in self.subsets:
            return path, path.parent.name, path.name

        scene_id = path.name
        for subset in self.subsets:
            prefix = f"{subset}_"
            if scene_id.startswith(prefix):
                scene_name = scene_id[len(prefix):]
                real_scene = self.root_dir / subset / scene_name
                if not real_scene.is_dir():
                    raise FileNotFoundError(f"DoPose scene not found: {real_scene}")
                return real_scene, subset, scene_name

        raise ValueError(
            f"Cannot resolve DoPose scene path: {scene_path}. "
            f"Expected real subset path or synthetic scene id like test_bin_000001."
        )

    def _camera_to_world(self, transform_records: List[dict]) -> np.ndarray:
        """
        用法: E_c2w = self._camera_to_world(records)
        作用: 从 DoPose frame transformations 构造 camera->world 矩阵
        输入: transform_records: list[dict]，当前帧变换记录
        输出: ndarray(4,4)，平移单位 cm
        """
        for record in transform_records:
            if (
                record.get("source_frame") == CAMERA_FRAME
                and record.get("target_frame") == WORLD_FRAME
            ):
                E_w2c = _transform_from_record(record)
                E_w2c[:3, 3] *= M_TO_CM
                return np.linalg.inv(E_w2c)
        raise ValueError(f"Missing {CAMERA_FRAME}->{WORLD_FRAME} transformation")

    def _build_objects(
        self,
        annotations: List[dict],
        annotation_info: List[dict],
        E_c2w: np.ndarray,
    ) -> List[ObjectInfo]:
        """
        用法: objects = self._build_objects(annotations, infos, E_c2w)
        作用: 将 DoPose BOP 物体标注转换为 ObjectInfo 列表
        输入:
            annotations: list[dict]，scene_gt 当前帧物体标注
            annotation_info: list[dict]，scene_gt_info 当前帧可见性信息
            E_c2w: ndarray(4,4)，camera->world 变换
        输出: list[ObjectInfo]，过滤后的物体列表
        """
        objects = []
        for idx, obj in enumerate(annotations):
            info = annotation_info[idx] if idx < len(annotation_info) else {}
            if not self._is_visible(info):
                continue

            obj_id = int(obj["obj_id"])
            pose_cam = np.eye(4, dtype=np.float64)
            pose_cam[:3, :3] = _reshape_matrix(obj["cam_R_m2c"], (3, 3))
            pose_cam[:3, 3] = np.asarray(obj["cam_t_m2c"], dtype=np.float64) * MM_TO_CM
            pose_world = E_c2w @ pose_cam

            objects.append(ObjectInfo(
                obj_id=f"obj_{obj_id:06d}_{idx}",
                class_name=self._obj_id_to_class_name(obj_id),
                bbox3d_canonical=self._get_bbox_for_obj_id(obj_id),
                pose_world=pose_world,
            ))
        return objects

    def _is_visible(self, info: dict) -> bool:
        """
        用法: ok = self._is_visible(info)
        作用: 根据 BOP 可见性字段判断是否保留物体
        输入: info: dict，scene_gt_info 单条记录
        输出: bool，是否可见
        """
        px_count_visib = int(info.get("px_count_visib", 1))
        visib_fract = float(info.get("visib_fract", 1.0))
        return px_count_visib > 0 and visib_fract >= self.min_visib_fract

    def _get_bbox_for_obj_id(self, obj_id: int) -> np.ndarray:
        """
        用法: bbox = self._get_bbox_for_obj_id(1)
        作用: 从 models_info.json 读取并缓存物体 canonical AABB
        输入: obj_id: int，DoPose 物体 ID
        输出: ndarray(6,)，单位 cm
        """
        if obj_id in self._bbox_cache:
            return self._bbox_cache[obj_id]

        model_info = self._models_info[str(obj_id)]
        min_xyz = np.array([
            model_info["min_x"],
            model_info["min_y"],
            model_info["min_z"],
        ], dtype=np.float64)
        size_xyz = np.array([
            model_info["size_x"],
            model_info["size_y"],
            model_info["size_z"],
        ], dtype=np.float64)
        bbox = np.concatenate([min_xyz, min_xyz + size_xyz]) * MM_TO_CM
        self._bbox_cache[obj_id] = bbox
        return bbox

    def _obj_id_to_class_name(self, obj_id: int) -> str:
        """
        用法: name = self._obj_id_to_class_name(1)
        作用: 将 DoPose obj_id 转为类别名
        输入: obj_id: int，物体 ID
        输出: str，类别名
        """
        return self._models_names.get(obj_id, f"obj_{obj_id:06d}")

    def _class_name_to_obj_id(self, class_name: str) -> int:
        """
        用法: obj_id = self._class_name_to_obj_id("choco_box")
        作用: 将类别名或 obj_000001 字符串还原为 DoPose obj_id
        输入: class_name: str，类别名
        输出: int，物体 ID
        """
        for obj_id, name in self._models_names.items():
            if class_name == name:
                return obj_id
        if class_name.startswith("obj_"):
            return int(class_name.split("_", 1)[1])
        raise KeyError(f"Unknown DoPose class name: {class_name}")

    def _has_required_scene_files(self, scene_dir: Path) -> bool:
        """
        用法: ok = self._has_required_scene_files(scene_dir)
        作用: 判断 DoPose scene 是否具备 adapter 必需文件
        输入: scene_dir: Path，真实 scene 目录
        输出: bool，是否可处理
        """
        required_paths = [
            scene_dir / "scene_camera.json",
            scene_dir / "scene_gt.json",
            scene_dir / "scene_gt_info.json",
            scene_dir / "scene_transformations.json",
            scene_dir / "rgb",
            scene_dir / "depth",
        ]
        return all(path.exists() for path in required_paths)

    def _list_frame_ids(self, scene_dir: Path) -> List[str]:
        """
        用法: frame_ids = self._list_frame_ids(scene_dir)
        作用: 列出当前 scene 中 RGB 和 depth 同时存在的帧
        输入: scene_dir: Path，真实 scene 目录
        输出: list[str]，零填充帧 ID
        """
        rgb_ids = {path.stem for path in (scene_dir / "rgb").glob("*.png")}
        depth_ids = {path.stem for path in (scene_dir / "depth").glob("*.png")}
        frame_ids = []
        for frame_id in sorted(rgb_ids & depth_ids, key=lambda value: int(value)):
            if self.frame_step > 1 and int(frame_id) % self.frame_step != 0:
                continue
            frame_ids.append(frame_id)
        return frame_ids
