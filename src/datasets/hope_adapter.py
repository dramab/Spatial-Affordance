"""
src/datasets/hope_adapter.py
------------------------------
HOPE-Video 数据集适配器。

封装所有 HOPE 特定的逻辑：
- 深度缩放: raw uint16 (mm) × 0.98042517 / 10 → cm
- 外参平移: m → cm (×100)
- 物体姿态: object→camera，需通过 E_c2w @ pose 转为 object→world
- Mesh 顶点: mm → cm (×0.1)
- 文件命名: {frame:04d}.json, {frame:04d}_rgb.jpg, {frame:04d}_depth.png

用法:
    from src.datasets.hope_adapter import HopeAdapter
    adapter = HopeAdapter(mesh_dir="/path/to/meshes/full")
    scene_data = adapter.load_scene("/path/to/scene_0001", frame_id="0000")

使用示例:
    python -c "
    from src.datasets.hope_adapter import HopeAdapter
    adapter = HopeAdapter(
        root_dir='/path/to/hope_video',
        mesh_dir='/path/to/meshes/full',
    )
    scenes = adapter.list_scenes()
    scene_data = adapter.load_scene(scenes[0][0], scenes[0][1][0])
    "
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh
from PIL import Image

from src.datasets.base_adapter import DatasetAdapter
from src.annotation.free_bbox.datatypes import (
    SceneData, CameraParams, ObjectInfo,
)

# HOPE 特定常量
DEPTH_SCALE = 0.98042517 / 10.0   # raw uint16 (mm) → cm
MM_TO_CM = 0.1                     # mesh 顶点 mm → cm


class HopeAdapter(DatasetAdapter):
    """
    HOPE-Video 数据集适配器。

    属性:
        root_dir: str HOPE-Video 数据根目录（包含 scene_XXXX 子目录）
        mesh_dir: str mesh 文件目录（包含 {class_name}.obj）
        frame_step: int 帧采样步长（默认 60）
    """

    def __init__(self, root_dir: str, mesh_dir: str = None,
                 frame_step: int = 60):
        self.root_dir = Path(root_dir)
        self.mesh_dir = Path(mesh_dir) if mesh_dir else None
        self.frame_step = frame_step
        self._bbox_cache = {}  # class_name → canonical AABB

    def load_scene(self, scene_path: str, frame_id: str) -> SceneData:
        """
        加载 HOPE 场景的单帧数据。

        输入:
            scene_path: str 场景目录路径（如 /path/to/scene_0001）
            frame_id: str 帧 ID（如 "0000"）
        输出:
            SceneData 通用场景数据（深度已转为 cm）
        """
        d = Path(scene_path)
        scene_id = d.name

        # 加载标注
        annot_path = d / f"{frame_id}.json"
        with open(annot_path, "r") as f:
            annots = json.load(f)

        # 加载图像
        rgb = np.asarray(
            Image.open(d / f"{frame_id}_rgb.jpg"), dtype=np.uint8)
        depth_raw = np.asarray(
            Image.open(d / f"{frame_id}_depth.png"), dtype=np.float32)

        # 深度转换: raw → cm
        depth_cm = depth_raw * DEPTH_SCALE

        # 相机参数
        K = np.array(annots["camera"]["intrinsics"], dtype=np.float64)
        E_w2c = np.array(annots["camera"]["extrinsics"], dtype=np.float64)
        E_w2c[:3, 3] *= 100.0  # HOPE 外参平移 m → cm
        E_c2w = np.linalg.inv(E_w2c)

        img_h, img_w = rgb.shape[:2]
        camera = CameraParams(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            E_c2w=E_c2w, img_w=img_w, img_h=img_h,
        )

        # 物体信息
        objects = []
        for i, obj in enumerate(annots.get("objects", [])):
            cls = obj["class"]
            pose_cam = np.array(obj["pose"], dtype=np.float64)  # object→camera

            # HOPE 姿态约定: object→camera → object→world
            pose_world = E_c2w @ pose_cam

            # 获取 canonical AABB
            bbox3d = obj.get("bbox3d")
            if bbox3d is not None:
                bbox3d_canonical = np.array(bbox3d, dtype=np.float64)
            else:
                bbox3d_canonical = self.get_object_scale(cls)

            objects.append(ObjectInfo(
                obj_id=f"obj_{i}",
                class_name=cls,
                bbox3d_canonical=bbox3d_canonical,
                pose_world=pose_world,
            ))

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
        列出 root_dir 下所有场景及其帧 ID。

        输出:
            list[(scene_path, [frame_ids])]
        """
        results = []
        scene_dirs = sorted(self.root_dir.glob("scene_*"))
        for sd in scene_dirs:
            frame_files = sorted(sd.glob("*_rgb.jpg"))
            frame_ids = [f.stem.replace("_rgb", "") for f in frame_files]
            # 按 frame_step 采样
            if self.frame_step > 1:
                frame_ids = [fid for fid in frame_ids
                             if int(fid) % self.frame_step == 0]
            if frame_ids:
                results.append((str(sd), frame_ids))
        return results

    def get_object_scale(self, class_name: str) -> np.ndarray:
        """
        从 mesh 文件计算物体 canonical AABB（带缓存）。

        输入:
            class_name: str 物体类别名
        输出:
            (6,) ndarray [min_x, min_y, min_z, max_x, max_y, max_z] (cm)
        """
        if class_name in self._bbox_cache:
            return self._bbox_cache[class_name]

        if self.mesh_dir is None:
            raise ValueError(
                f"mesh_dir not set, cannot compute bbox for {class_name}")

        mesh_path = self.mesh_dir / f"{class_name}.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        verts = np.array(mesh.vertices, dtype=np.float64) * MM_TO_CM

        bbox = np.concatenate([verts.min(axis=0), verts.max(axis=0)])
        self._bbox_cache[class_name] = bbox
        return bbox
