"""
src/annotation/bbox3d/bbox_utils.py
------------------------------------
3D bounding box 通用工具：角点生成、OBB 世界坐标、接触面检测。

用法:
    from src.annotation.bbox3d.bbox_utils import (
        get_bbox_corners, obb_corners_world, get_contact_face_indices,
    )
"""

import numpy as np

from src.utils.coord_utils import transform_points


def get_bbox_corners(bbox3d):
    """
    从 AABB 生成 8 个角点。

    角点编码: index = zi*4 + yi*2 + xi，取值 0/1 对应 min/max。

    输入:
        bbox3d: (6,) [min_x, min_y, min_z, max_x, max_y, max_z]
    输出:
        (8, 3) float64 角点坐标
    """
    mn, mx = np.array(bbox3d[:3]), np.array(bbox3d[3:])
    # 8 个角点的 min/max 选择掩码，顺序与原三重循环完全一致
    idx = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    return np.where(idx, mx, mn)


def obb_corners_world(bbox3d, T_obj2world):
    """
    获取 OBB 在世界坐标系下的 8 个角点。

    输入:
        bbox3d: (6,) 物体 canonical AABB
        T_obj2world: (4, 4) object→world 变换矩阵
    输出:
        (8, 3) float64 世界坐标角点
    """
    return transform_points(get_bbox_corners(bbox3d), T_obj2world)


# 接触面角点索引（按主轴和方向索引）
# key: (dominant_axis, sign)  value: 4 个角点索引（构成四边形）
CONTACT_FACE_CORNERS = {
    (0, +1): [1, 3, 7, 5],   # max_x 面
    (0, -1): [0, 2, 6, 4],   # min_x 面
    (1, +1): [2, 3, 7, 6],   # max_y 面
    (1, -1): [0, 1, 5, 4],   # min_y 面
    (2, +1): [4, 5, 7, 6],   # max_z 面
    (2, -1): [0, 1, 3, 2],   # min_z 面
}


def get_contact_face_indices(pose_world,
                              world_up=np.array([0.0, 0.0, 1.0])):
    """
    根据世界上方向和物体姿态，动态确定接触面（底面）的 4 个角点索引。

    原理:
        world_up 为世界坐标系上方向（默认 Z-up），世界下方向（重力）为 -world_up。
        将世界下方向变换到物体坐标系：down_obj = R_pose.T @ (-world_up)
        其中 R_pose = pose_world[:3, :3] 为 object→world 旋转矩阵，
        R_pose.T 即其逆（world→object 旋转）。
        |down_obj| 最大的分量对应"重力方向在物体坐标系中最接近的轴"，
        该分量的符号决定取该轴的 min 面还是 max 面作为接触面（底面）。

    输入:
        pose_world: (4, 4) object→world 变换矩阵
        world_up: (3,) 世界坐标系上方向
    输出:
        list[int] 4 个角点索引
    """
    R_pose = pose_world[:3, :3]
    # 世界下方向在物体坐标系中的表示
    down_world = -np.asarray(world_up, dtype=np.float64)
    down_obj = R_pose.T @ down_world

    axis = int(np.argmax(np.abs(down_obj)))
    sign = int(np.sign(down_obj[axis]))

    return CONTACT_FACE_CORNERS[(axis, sign)]
