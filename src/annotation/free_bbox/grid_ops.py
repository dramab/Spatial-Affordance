"""
src/annotation/free_bbox/grid_ops.py
-------------------------------------
占据栅格操作：物体体素化填充、移除/恢复、障碍物膨胀。

用于放置规划中的栅格准备，将物体 OBB 标记为 OCCUPIED，
模拟移除目标物体（设为 FREE），以及安全边距膨胀。

用法:
    from src.annotation.free_bbox.grid_ops import prepare_grid_base, prepare_grid
"""

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from src.annotation.free_bbox.occupancy import FREE, OCCUPIED, UNKNOWN


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


def prepare_grid(grid_base, target_voxels):
    """
    为目标物体准备工作栅格：将目标物体体素设为 FREE（模拟移除）。

    输入:
        grid_base: (Gx, Gy, Gz) uint8 所有物体已标记的栅格
        target_voxels: (M, 3) int 目标物体的体素索引
    输出:
        grid_work: (Gx, Gy, Gz) uint8 工作栅格副本
    """
    grid_work = grid_base.copy()
    if len(target_voxels) > 0:
        grid_work[target_voxels[:, 0],
                  target_voxels[:, 1],
                  target_voxels[:, 2]] = FREE
    return grid_work


def grid_remove_object(grid, target_voxels):
    """
    原地移除物体体素（设为 FREE），返回有效体素和保存的原始值用于恢复。

    会做边界检查，过滤掉超出栅格范围的体素。

    输入:
        grid: (Gx, Gy, Gz) uint8 栅格（原地修改）
        target_voxels: (M, 3) int 目标物体体素索引
    输出:
        valid_voxels: (K, 3) int 边界内的有效体素索引
        saved: (K,) uint8 被覆盖的原始体素值
    """
    if len(target_voxels) == 0:
        return target_voxels, np.array([], dtype=grid.dtype)
    gs = grid.shape
    m = ((target_voxels[:, 0] >= 0) & (target_voxels[:, 0] < gs[0]) &
         (target_voxels[:, 1] >= 0) & (target_voxels[:, 1] < gs[1]) &
         (target_voxels[:, 2] >= 0) & (target_voxels[:, 2] < gs[2]))
    v = target_voxels[m]
    if len(v) == 0:
        return v, np.array([], dtype=grid.dtype)
    saved = grid[v[:, 0], v[:, 1], v[:, 2]].copy()
    grid[v[:, 0], v[:, 1], v[:, 2]] = FREE
    return v, saved


def grid_restore_object(grid, target_voxels, saved):
    """
    原地恢复物体体素到保存的原始值。

    输入:
        grid: (Gx, Gy, Gz) uint8 栅格（原地修改）
        target_voxels: (M, 3) int 目标物体体素索引
        saved: (M,) uint8 保存的原始值
    """
    if len(target_voxels) == 0:
        return
    grid[target_voxels[:, 0],
         target_voxels[:, 1],
         target_voxels[:, 2]] = saved


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
