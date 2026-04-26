"""
src/datasets/ycb_video_adapter.py
----------------------------------
YCB-Video BOP test split 数据集适配器。

封装 YCB-Video test 数据的读取与单位转换：
- 深度缩放: raw uint16 * depth_scale(mm) -> cm
- 相机外参: scene_camera.json 中 world->camera，平移 mm -> cm
- 物体位姿: scene_gt.json 中 object->camera，平移 mm -> cm
- 物体尺度: models_info.json 中 canonical AABB，mm -> cm

用法:
    from src.datasets.ycb_video_adapter import YCBVideoAdapter
    adapter = YCBVideoAdapter(
        root_dir="/data/wenhao.hai/ycb_video/ycbv_test_all/test",
        models_info_path="/data/wenhao.hai/ycb_video/ycbv_models/models/models_info.json",
    )
    scene_data = adapter.load_scene(
        "/data/wenhao.hai/ycb_video/ycbv_test_all/test/000048",
        "000001",
    )
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from src.annotation.free_bbox.datatypes import CameraParams, ObjectInfo, SceneData
from src.datasets.base_adapter import DatasetAdapter


MM_TO_CM = 0.1
YCB_OBJECT_NAMES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}


def _load_json(path: Path):
    """
    用法: payload = _load_json(Path("scene_gt.json"))
    作用: 读取 JSON 文件
    输入: path: Path，JSON 文件路径
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
    作用: 将 BOP 展平矩阵字段转换为 numpy 矩阵
    输入: values: list[float]；shape: 目标矩阵形状
    输出: ndarray，指定形状的 float64 矩阵
    """
    return np.asarray(values, dtype=np.float64).reshape(shape)


class YCBVideoAdapter(DatasetAdapter):
    """
    YCB-Video BOP test split 数据集适配器。

    属性:
        root_dir: Path，包含 test scene 目录的根目录
        models_info_path: Path，BOP models_info.json 路径
        frame_step: int，帧采样步长
        min_visib_fract: float，物体最小可见比例过滤阈值
    """

    def __init__(
        self,
        root_dir: str,
        models_info_path: str,
        frame_step: int = 5,
        min_visib_fract: float = 0.0,
    ):
        """
        用法: adapter = YCBVideoAdapter(root_dir, models_info_path)
        作用: 初始化 YCB-Video test adapter 并缓存模型尺度信息
        输入:
            root_dir: str，YCB-Video BOP test 根目录
            models_info_path: str，models_info.json 路径
            frame_step: int，批量处理时的帧采样步长
            min_visib_fract: float，过滤低可见物体的阈值
        输出: None
        """
        self.root_dir = Path(root_dir)
        self.models_info_path = Path(models_info_path)
        self.frame_step = int(frame_step)
        self.min_visib_fract = float(min_visib_fract)
        self._models_info = _load_json(self.models_info_path)
        self._bbox_cache: Dict[int, np.ndarray] = {}

    def load_scene(self, scene_path: str, frame_id: str) -> SceneData:
        """
        用法: scene = adapter.load_scene("/path/to/test/000048", "000001")
        作用: 加载 YCB-Video test 单帧并转换为通用 SceneData
        输入:
            scene_path: str，scene 目录路径
            frame_id: str，零填充帧 ID
        输出: SceneData，单位统一为 cm
        """
        scene_dir = Path(scene_path)
        scene_id = scene_dir.name
        key = _frame_key(frame_id)

        scene_camera = _load_json(scene_dir / "scene_camera.json")
        scene_gt = _load_json(scene_dir / "scene_gt.json")
        scene_gt_info = _load_json(scene_dir / "scene_gt_info.json")

        if key not in scene_camera or key not in scene_gt:
            raise KeyError(f"Frame {frame_id} not found in {scene_dir}")

        camera_record = scene_camera[key]
        if "cam_R_w2c" not in camera_record or "cam_t_w2c" not in camera_record:
            raise ValueError(
                f"YCBVideoAdapter only supports test frames with camera extrinsics: "
                f"{scene_id}/{frame_id}"
            )

        rgb = np.asarray(Image.open(self._resolve_rgb_path(scene_dir, frame_id)), dtype=np.uint8)
        depth_raw = np.asarray(Image.open(scene_dir / "depth" / f"{frame_id}.png"), dtype=np.float32)
        depth_cm = depth_raw * float(camera_record.get("depth_scale", 1.0)) * MM_TO_CM

        K = _reshape_matrix(camera_record["cam_K"], (3, 3))
        E_w2c = np.eye(4, dtype=np.float64)
        E_w2c[:3, :3] = _reshape_matrix(camera_record["cam_R_w2c"], (3, 3))
        E_w2c[:3, 3] = np.asarray(camera_record["cam_t_w2c"], dtype=np.float64) * MM_TO_CM
        E_c2w = np.linalg.inv(E_w2c)

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
        作用: 列出 test 根目录下所有可处理 scene 与帧 ID
        输入: 无
        输出: list[(scene_path, frame_ids)]，scene 路径与零填充帧 ID 列表
        """
        results = []
        for scene_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            required = [
                scene_dir / "scene_camera.json",
                scene_dir / "scene_gt.json",
                scene_dir / "scene_gt_info.json",
                scene_dir / "rgb",
                scene_dir / "depth",
            ]
            if not all(path.exists() for path in required):
                continue

            frame_ids = []
            for rgb_path in sorted((scene_dir / "rgb").glob("*.png")):
                frame_id = rgb_path.stem
                if self.frame_step > 1 and int(frame_id) % self.frame_step != 0:
                    continue
                if (scene_dir / "depth" / f"{frame_id}.png").exists():
                    frame_ids.append(frame_id)
            if frame_ids:
                results.append((str(scene_dir), frame_ids))
        return results

    def get_object_scale(self, class_name: str) -> np.ndarray:
        """
        用法: bbox = adapter.get_object_scale("002_master_chef_can")
        作用: 获取 YCB 模型 canonical AABB
        输入: class_name: str，YCB 类别名或 obj_000001 格式名称
        输出: ndarray(6,)，[min_x, min_y, min_z, max_x, max_y, max_z]，单位 cm
        """
        obj_id = self._class_name_to_obj_id(class_name)
        return self._get_bbox_for_obj_id(obj_id)

    def _build_objects(
        self,
        annotations: List[dict],
        annotation_info: List[dict],
        E_c2w: np.ndarray,
    ) -> List[ObjectInfo]:
        """
        用法: objects = self._build_objects(annotations, infos, E_c2w)
        作用: 将 BOP 物体标注转换为 ObjectInfo 列表
        输入:
            annotations: list[dict]，scene_gt 中当前帧的物体标注
            annotation_info: list[dict]，scene_gt_info 中当前帧的可见性信息
            E_c2w: ndarray(4,4)，camera->world 变换
        输出: list[ObjectInfo]，过滤后物体列表
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
        作用: 根据 BOP 可见性字段判断物体是否进入 SceneData
        输入: info: dict，scene_gt_info 中单个物体信息
        输出: bool，是否保留该物体
        """
        px_count_visib = int(info.get("px_count_visib", 1))
        visib_fract = float(info.get("visib_fract", 1.0))
        return px_count_visib > 0 and visib_fract >= self.min_visib_fract

    def _get_bbox_for_obj_id(self, obj_id: int) -> np.ndarray:
        """
        用法: bbox = self._get_bbox_for_obj_id(1)
        作用: 从 models_info.json 读取并缓存物体 canonical AABB
        输入: obj_id: int，BOP 物体 ID
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
        作用: 将 BOP 物体 ID 转换为 YCB 类别名
        输入: obj_id: int，BOP 物体 ID
        输出: str，类别名
        """
        return YCB_OBJECT_NAMES.get(obj_id, f"obj_{obj_id:06d}")

    def _class_name_to_obj_id(self, class_name: str) -> int:
        """
        用法: obj_id = self._class_name_to_obj_id("002_master_chef_can")
        作用: 将类别名或 obj_000001 字符串还原为 BOP 物体 ID
        输入: class_name: str，类别名
        输出: int，BOP 物体 ID
        """
        for obj_id, name in YCB_OBJECT_NAMES.items():
            if class_name == name:
                return obj_id
        if class_name.startswith("obj_"):
            return int(class_name.split("_", 1)[1])
        raise KeyError(f"Unknown YCB class name: {class_name}")

    def _resolve_rgb_path(self, scene_dir: Path, frame_id: str) -> Path:
        """
        用法: path = self._resolve_rgb_path(scene_dir, "000001")
        作用: 获取当前帧 RGB 图片路径
        输入: scene_dir: Path，scene 目录；frame_id: str，帧 ID
        输出: Path，RGB 文件路径
        """
        rgb_path = scene_dir / "rgb" / f"{frame_id}.png"
        if rgb_path.exists():
            return rgb_path
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
