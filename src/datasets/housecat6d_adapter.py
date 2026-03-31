"""
src/datasets/housecat6d_adapter.py
-----------------------------------
HouseCat6D 数据集适配器。

封装 HouseCat6D 特定的逻辑：
- 深度缩放: raw uint16 (mm) -> cm
- 相机位姿: camera_pose/*.txt 为 camera->world，平移 m -> cm
- 物体位姿: labels/*_label.pkl 中 rotations/translations 为 object->camera
- 物体尺度: gt_scales 为 XYZ 尺寸（m），转换为 canonical AABB（cm）
"""

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from src.annotation.free_bbox.datatypes import (
    CameraParams,
    ObjectInfo,
    SceneData,
)
from src.datasets.base_adapter import DatasetAdapter

# HouseCat6D 特定常量
DEPTH_SCALE = 0.1   # raw uint16 (mm) -> cm
M_TO_CM = 100.0


class HouseCat6DAdapter(DatasetAdapter):
    """
    HouseCat6D 数据集适配器。

    属性:
        root_dir: 数据根目录（包含 scene01, scene02, ...）
        frame_step: 帧采样步长
    """

    def __init__(self, root_dir: str, frame_step: int = 60):
        self.root_dir = Path(root_dir)
        self.frame_step = frame_step

    def load_scene(self, scene_path: str, frame_id: str) -> SceneData:
        """
        加载 HouseCat6D 单帧数据并转换为通用 SceneData。
        """
        scene_dir = Path(scene_path)
        scene_id = scene_dir.name

        rgb_path = scene_dir / "rgb" / f"{frame_id}.png"
        depth_path = scene_dir / "depth" / f"{frame_id}.png"
        intrinsics_path = scene_dir / "intrinsics.txt"
        camera_pose_path = scene_dir / "camera_pose" / f"{frame_id}.txt"
        label_path = scene_dir / "labels" / f"{frame_id}_label.pkl"

        rgb = np.asarray(Image.open(rgb_path), dtype=np.uint8)
        depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
        depth_cm = depth_raw * DEPTH_SCALE

        K = np.loadtxt(intrinsics_path, dtype=np.float64).reshape(3, 3)
        E_c2w = np.loadtxt(camera_pose_path, dtype=np.float64).reshape(4, 4)
        E_c2w[:3, 3] *= M_TO_CM

        with open(label_path, "rb") as f:
            labels = pickle.load(f)

        model_list = list(labels.get("model_list", []))
        instance_ids = list(labels.get("instance_ids", []))
        rotations = np.asarray(labels.get("rotations", []), dtype=np.float64)
        translations = np.asarray(labels.get("translations", []), dtype=np.float64)
        gt_scales = np.asarray(labels.get("gt_scales", []), dtype=np.float64)

        num_objects = len(model_list)
        if len(rotations) != num_objects or len(translations) != num_objects:
            raise ValueError(
                f"Incomplete pose annotations for {scene_id}/{frame_id}: "
                f"{num_objects=} {len(rotations)=} {len(translations)=}"
            )
        if len(gt_scales) != num_objects:
            raise ValueError(
                f"Missing gt_scales for {scene_id}/{frame_id}: "
                f"{num_objects=} {len(gt_scales)=}"
            )

        objects = []
        for idx, class_name in enumerate(model_list):
            pose_cam = np.eye(4, dtype=np.float64)
            pose_cam[:3, :3] = rotations[idx]
            pose_cam[:3, 3] = translations[idx] * M_TO_CM
            pose_world = E_c2w @ pose_cam

            size_cm = gt_scales[idx] * M_TO_CM
            half_size = size_cm * 0.5
            bbox3d_canonical = np.concatenate([-half_size, half_size])

            if idx < len(instance_ids):
                obj_id = f"obj_{instance_ids[idx]}"
            else:
                obj_id = f"obj_{idx}"

            objects.append(ObjectInfo(
                obj_id=obj_id,
                class_name=str(class_name),
                bbox3d_canonical=bbox3d_canonical,
                pose_world=pose_world,
            ))

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
        列出 root_dir 下所有场景及其可用帧 ID。
        """
        results = []
        scene_dirs = sorted(self.root_dir.glob("scene*"))

        for scene_dir in scene_dirs:
            rgb_dir = scene_dir / "rgb"
            depth_dir = scene_dir / "depth"
            pose_dir = scene_dir / "camera_pose"
            labels_dir = scene_dir / "labels"
            intrinsics_path = scene_dir / "intrinsics.txt"

            if not all([
                rgb_dir.exists(),
                depth_dir.exists(),
                pose_dir.exists(),
                labels_dir.exists(),
                intrinsics_path.exists(),
            ]):
                continue

            frame_ids = []
            for rgb_path in sorted(rgb_dir.glob("*.png")):
                frame_id = rgb_path.stem
                if self.frame_step > 1 and int(frame_id) % self.frame_step != 0:
                    continue

                required_paths = [
                    depth_dir / f"{frame_id}.png",
                    pose_dir / f"{frame_id}.txt",
                    labels_dir / f"{frame_id}_label.pkl",
                ]
                if all(path.exists() for path in required_paths):
                    frame_ids.append(frame_id)

            if frame_ids:
                results.append((str(scene_dir), frame_ids))

        return results
