"""
src/annotation/free_bbox/grid_ops.py
-------------------------------------
占据栅格操作：物体体素化填充与障碍物膨胀。

用于放置规划中的栅格准备，将物体 OBB 标记为 OCCUPIED，
并为碰撞搜索构建安全边距膨胀掩码。

用法:
    from src.annotation.free_bbox.grid_ops import prepare_grid_base, voxelize_obb
"""

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from src.annotation.free_bbox.occupancy import OCCUPIED, UNKNOWN


def voxelize_obb(bbox3d, T_obj2world, vp, grid_shape):
    """
    将物体 OBB 转换为体素索引集合。

    通过在 OBB 的世界坐标包围盒内枚举体素，
    将体素中心逆变换回物体坐标系，检查是否在 canonical AABB 内。

    输入:
        bbox3d: (6,) [min_x, min_y, min_z, max_x, max_y, max_z] 物体规范 AABB
        T_obj2world: (4, 4) object→world 变换矩阵
        vp: dict 体素参数 {"voxel_size": float, "origin": [x,y,z]}
        grid_shape: (3,) 栅格尺寸 (Gx, Gy, Gz)
    输出:
        (M, 3) int 体素索引数组
    """
    from src.utils.coord_utils import transform_points

    origin = np.asarray(vp["origin"], dtype=np.float64)
    vs = float(vp["voxel_size"])

    corners_obj = _get_bbox_corners(bbox3d)
    cw = transform_points(corners_obj, T_obj2world)

    lo = np.maximum(np.floor((cw.min(0) - vs - origin) / vs).astype(int), 0)
    hi = np.minimum(np.ceil((cw.max(0) + vs - origin) / vs).astype(int),
                    np.array(grid_shape))

    ranges = [np.arange(lo[d], hi[d]) for d in range(3)]
    if any(len(r) == 0 for r in ranges):
        return np.empty((0, 3), dtype=int)

    gi, gj, gk = np.meshgrid(*ranges, indexing="ij")
    idx = np.stack([gi.ravel(), gj.ravel(), gk.ravel()], axis=1)

    centres = origin + (idx + 0.5) * vs
    co = transform_points(centres, np.linalg.inv(T_obj2world))
    bmin, bmax = np.array(bbox3d[:3]), np.array(bbox3d[3:])
    return idx[np.all((co >= bmin) & (co <= bmax), axis=1)]


def _get_bbox_corners(bbox3d):
    """从 AABB 生成 8 个角点 (8, 3)。"""
    mn, mx = np.array(bbox3d[:3]), np.array(bbox3d[3:])
    corners = []
    for zi in range(2):
        for yi in range(2):
            for xi in range(2):
                corners.append([[mn[0], mx[0]][xi],
                                [mn[1], mx[1]][yi],
                                [mn[2], mx[2]][zi]])
    return np.array(corners, dtype=np.float64)


def prepare_grid_base(grid, objects, vp):
    """
    一次性将所有物体 OBB 标记为 OCCUPIED，返回修改后的栅格副本。

    输入:
        grid: (Gx, Gy, Gz) uint8 原始占据栅格
        objects: list[ObjectInfo] 场景中所有物体
        vp: dict 体素参数
    输出:
        grid_base: (Gx, Gy, Gz) uint8 标记后的栅格副本
    """
    grid_base = grid.copy()
    gs = np.array(grid_base.shape)
    for obj in objects:
        voxels = voxelize_obb(obj.bbox3d_canonical, obj.pose_world, vp, gs)
        if len(voxels) > 0:
            grid_base[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = OCCUPIED
    return grid_base


def dilate_obstacles_xy(grid, margin_voxels):
    """
    在 XY 平面膨胀障碍物（OCCUPIED + UNKNOWN），用于安全边距。

    输入:
        grid: (Gx, Gy, Gz) uint8 占据栅格
        margin_voxels: int 膨胀迭代次数
    输出:
        (Gx, Gy, Gz) bool 膨胀后的障碍物掩码
    """
    occ = (grid == OCCUPIED) | (grid == UNKNOWN)
    if margin_voxels <= 0:
        return occ
    s2d = generate_binary_structure(2, 1)
    s3d = np.zeros((3, 3, 3), dtype=bool)
    s3d[:, :, 1] = s2d
    return binary_dilation(occ, structure=s3d, iterations=margin_voxels)
