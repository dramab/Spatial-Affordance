"""
src/datasets/base_adapter.py
------------------------------
数据集适配器抽象基类。

定义从原始数据集加载场景数据的标准接口。
每个数据集实现自己的 adapter，负责：
- 文件路径解析和命名约定
- 深度图单位转换（原始值 → 场景工作单位）
- 相机外参单位转换
- 物体姿态格式转换（统一为 object→world）
- 物体尺度信息获取

用法:
    from src.datasets.base_adapter import DatasetAdapter
    class MyAdapter(DatasetAdapter):
        def load_scene(self, scene_path, frame_id) -> SceneData: ...
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from src.annotation.free_bbox.datatypes import SceneData


class DatasetAdapter(ABC):
    """数据集适配器抽象基类。"""

    @abstractmethod
    def load_scene(self, scene_path: str, frame_id: str) -> SceneData:
        """
        加载单帧场景数据，转换为通用 SceneData 格式。

        输入:
            scene_path: str 场景目录路径
            frame_id: str 帧标识
        输出:
            SceneData 通用场景数据
        """
        ...

    @abstractmethod
    def list_scenes(self) -> List[Tuple[str, List[str]]]:
        """
        列出数据集中所有场景及其帧 ID。

        输出:
            list[(scene_id, [frame_ids])] 场景列表
        """
        ...

    def get_object_scale(self, class_name: str):
        """
        获取物体 canonical AABB（可选）。

        默认未实现，子类可覆盖以支持从 mesh/CAD 模型获取尺度信息。

        输入:
            class_name: str 物体类别名
        输出:
            (6,) ndarray [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        raise NotImplementedError(
            f"get_object_scale not implemented for {class_name}. "
            "Override this method or provide bbox3d_canonical in annotations.")
