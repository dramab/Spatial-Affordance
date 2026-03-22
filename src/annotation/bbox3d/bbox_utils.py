"""
src/annotation/bbox3d/bbox_utils.py
------------------------------------
3D bounding box 通用工具：角点生成、OBB 世界坐标、接触面检测。

用法:
    from src.annotation.bbox3d.bbox_utils import (
        get_bbox_corners, obb_corners_world, get_contact_face_indices,
        get_contact_face_info, align_pose_to_contact_face,
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

CONTACT_FACE_NORMALS = {
    (0, +1): np.array([1.0, 0.0, 0.0], dtype=np.float64),
    (0, -1): np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    (1, +1): np.array([0.0, 1.0, 0.0], dtype=np.float64),
    (1, -1): np.array([0.0, -1.0, 0.0], dtype=np.float64),
    (2, +1): np.array([0.0, 0.0, 1.0], dtype=np.float64),
    (2, -1): np.array([0.0, 0.0, -1.0], dtype=np.float64),
}

CONTACT_FACE_LABELS = {
    (0, +1): 'max_x',
    (0, -1): 'min_x',
    (1, +1): 'max_y',
    (1, -1): 'min_y',
    (2, +1): 'max_z',
    (2, -1): 'min_z',
}


def _normalize_vector(vec, eps=1e-8):
    """返回单位向量。"""
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        raise ValueError('Cannot normalize near-zero vector')
    return arr / norm


def _skew_matrix(vec):
    """构造向量的反对称矩阵。"""
    x, y, z = np.asarray(vec, dtype=np.float64)
    return np.array([
        [0.0, -z,  y],
        [z,   0.0, -x],
        [-y,  x,   0.0],
    ], dtype=np.float64)


def rotation_matrix_from_vectors(src_vec, dst_vec, eps=1e-8):
    """
    构造将 src_vec 旋转到 dst_vec 的最小旋转矩阵。
    """
    src = _normalize_vector(src_vec, eps=eps)
    dst = _normalize_vector(dst_vec, eps=eps)

    cross = np.cross(src, dst)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))

    if cross_norm < eps:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)

        # 180° 旋转时任选一个与 src 正交的轴。
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(src)))] = 1.0
        axis = basis - src * np.dot(src, basis)
        axis = _normalize_vector(axis, eps=eps)
        K = _skew_matrix(axis)
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)

    K = _skew_matrix(cross)
    factor = (1.0 - dot) / max(cross_norm * cross_norm, eps)
    return np.eye(3, dtype=np.float64) + K + factor * (K @ K)


def get_contact_face_info(pose_world, E_w2c=None,
                          world_up=np.array([0.0, 0.0, 1.0])):
    """
    根据世界上方向和物体姿态，返回当前最朝下的 canonical 面信息。

    这里不做真实接触检测，而是根据姿态 + 重力方向，
    选取与世界下方向最一致的一个 bbox 面，作为当前主支撑面。
    """
    del E_w2c

    pose_world = np.asarray(pose_world, dtype=np.float64)
    R_pose = pose_world[:3, :3]
    down_world = -_normalize_vector(world_up)
    down_obj = R_pose.T @ down_world

    axis = int(np.argmax(np.abs(down_obj)))
    sign = +1 if float(down_obj[axis]) >= 0.0 else -1
    key = (axis, sign)

    normal_obj = CONTACT_FACE_NORMALS[key].copy()
    normal_world = R_pose @ normal_obj

    return {
        'axis': axis,
        'sign': sign,
        'label': CONTACT_FACE_LABELS[key],
        'corner_indices': list(CONTACT_FACE_CORNERS[key]),
        'normal_object': normal_obj,
        'normal_world': normal_world,
        'down_object': down_obj,
        'alignment_score': float(np.dot(
            _normalize_vector(normal_world), down_world)),
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
    info = get_contact_face_info(
        pose_world, E_w2c=E_w2c, world_up=world_up)
    return info['corner_indices']


def align_pose_to_contact_face(pose_world, bbox3d, E_w2c=None,
                               world_up=np.array([0.0, 0.0, 1.0])):
    """
    将当前主支撑面旋到朝下，返回新的 object→world 姿态。

    旋转绕 bbox 中心进行，这样不会因对齐而引入额外平移偏差。
    """
    pose_world = np.asarray(pose_world, dtype=np.float64)
    bbox3d = np.asarray(bbox3d, dtype=np.float64)

    face_info = get_contact_face_info(
        pose_world, E_w2c=E_w2c, world_up=world_up)
    target_down = -_normalize_vector(world_up)

    R_pose = pose_world[:3, :3]
    R_align = rotation_matrix_from_vectors(
        face_info['normal_world'], target_down)
    R_aligned = R_align @ R_pose

    bbox_center_obj = (bbox3d[:3] + bbox3d[3:]) / 2.0
    bbox_center_world = (R_pose @ bbox_center_obj) + pose_world[:3, 3]

    aligned_pose = pose_world.copy()
    aligned_pose[:3, :3] = R_aligned
    aligned_pose[:3, 3] = bbox_center_world - R_aligned @ bbox_center_obj

    info = dict(face_info)
    info.update({
        'bbox_center_object': bbox_center_obj,
        'bbox_center_world': bbox_center_world,
        'alignment_rotation_world': R_align,
        'aligned_normal_world': R_aligned @ face_info['normal_object'],
    })
    return aligned_pose, info
