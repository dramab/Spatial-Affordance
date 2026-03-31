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
    corners = []
    for zi in range(2):
        for yi in range(2):
            for xi in range(2):
                corners.append([[mn[0], mx[0]][xi],
                                [mn[1], mx[1]][yi],
                                [mn[2], mx[2]][zi]])
    return np.array(corners, dtype=np.float64)


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


def get_contact_face_indices(pose_world, E_w2c,
                              world_up=np.array([0.0, 0.0, 1.0])):
    """
    根据世界上方向、外参和物体姿态，动态确定接触面（底面）的 4 个角点索引。

    原理:
        down_cam = R_w2c @ (-world_up)
        down_obj = R_pose_world_inv @ down_cam_world = R_pose.T @ (-world_up)
        主导轴（|down_obj| 最大的分量）决定接触哪个 AABB 面。

    输入:
        pose_world: (4, 4) object→world 变换矩阵
        E_w2c: (4, 4) world→camera 变换矩阵（此处仅用于兼容，
               实际通过 world_up 直接在世界坐标系计算）
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
