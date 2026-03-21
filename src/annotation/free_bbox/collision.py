"""
src/annotation/free_bbox/collision.py
--------------------------------------
FFT 碰撞检测：在 (X, Y, θ) 配置空间中搜索无碰撞放置位置。

通过逐层 2D FFT 卷积检测碰撞，支持可选的 CuPy GPU 加速。

用法:
    from src.annotation.free_bbox.collision import find_table_placements
"""

import math
import numpy as np
from scipy.signal import fftconvolve

from src.utils.coord_utils import rotation_z_3x3, transform_points
from src.annotation.free_bbox.occupancy import FREE, OCCUPIED, UNKNOWN
from src.annotation.free_bbox.grid_ops import voxelize_obb, dilate_obstacles_xy
from src.annotation.free_bbox.voxel_utils import voxel_to_world

# GPU 后端（可选）
try:
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve as gpu_fftconvolve
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def _compute_collision_slice(obstacle, obj_mask, landing_z, use_gpu=False):
    """
    在 landing_z 高度计算 2D 碰撞图。

    通过逐层 2D 卷积检测碰撞，仅对物体掩码实际占用的层做卷积。

    输入:
        obstacle: (Gx, Gy, Gz) bool 障碍物掩码
        obj_mask: (ox, oy, oz) bool 物体体素掩码
        landing_z: int 放置的 Z 层起始索引
        use_gpu: bool 是否使用 GPU 加速
    输出:
        (nx, ny) float32 碰撞图，值 > 0 表示碰撞；None 如果尺寸不合法
    """
    Gx, Gy, Gz = obstacle.shape
    ox, oy, oz = obj_mask.shape
    nx, ny = Gx - ox + 1, Gy - oy + 1

    if nx <= 0 or ny <= 0:
        return None

    if use_gpu and HAS_GPU:
        collision = cp.zeros((nx, ny), dtype=cp.float32)
        for dz in range(oz):
            z_idx = landing_z + dz
            if z_idx < 0 or z_idx >= Gz:
                continue
            obj_slice = obj_mask[:, :, dz]
            if obj_slice.sum() == 0:
                continue
            obs_gpu = cp.asarray(obstacle[:, :, z_idx].astype(np.float32))
            mask_gpu = cp.asarray(
                np.ascontiguousarray(obj_slice[::-1, ::-1]).astype(np.float32))
            collision += gpu_fftconvolve(obs_gpu, mask_gpu, mode="valid")
        return cp.asnumpy(collision)
    else:
        collision = np.zeros((nx, ny), dtype=np.float32)
        for dz in range(oz):
            z_idx = landing_z + dz
            if z_idx < 0 or z_idx >= Gz:
                continue
            obj_slice = obj_mask[:, :, dz]
            if obj_slice.sum() == 0:
                continue
            collision += fftconvolve(
                obstacle[:, :, z_idx].astype(np.float32),
                obj_slice[::-1, ::-1].astype(np.float32),
                mode="valid")
        return collision


def find_table_placements(grid_work, bbox3d, T_obj2world, vp,
                          table_z, surface_mask_2d,
                          safety_margin=0.5, yaw_steps=24,
                          use_gpu=False):
    """
    在 (X, Y, θ) 配置空间中搜索无碰撞放置位置。

    算法:
        1. 对每个 yaw 角度，旋转物体并体素化
        2. 膨胀障碍物（安全边距）
        3. FFT 卷积检测碰撞
        4. 收集所有无碰撞位置

    输入:
        grid_work: (Gx, Gy, Gz) uint8 工作栅格（目标物体已移除）
        bbox3d: (6,) 物体 canonical AABB
        T_obj2world: (4, 4) 物体原始 object→world 变换
        vp: dict 体素参数
        table_z: int 支撑面 Z 层索引
        surface_mask_2d: (Gx, Gy) bool 支撑面掩码
        safety_margin: float 安全边距（场景单位）
        yaw_steps: int yaw 旋转离散步数
        use_gpu: bool 是否使用 GPU
    输出:
        candidates: (N, 3) int 候选位置 [grid_x, grid_y, yaw_index]
        meta: dict 搜索统计信息
        yaw_data: dict 每个 yaw 角度的旋转数据（供后续过滤使用）
    """
    vs = float(vp["voxel_size"])
    grid_shape = np.array(grid_work.shape)
    Gx, Gy, Gz = grid_shape

    landing_z = table_z + 1
    margin_voxels = max(0, int(math.ceil(safety_margin / vs)))
    obstacle = dilate_obstacles_xy(grid_work, margin_voxels)

    # ROI 优化：基于支撑面掩码裁剪搜索区域
    if surface_mask_2d is not None:
        nz_x, nz_y = np.where(surface_mask_2d)
        if len(nz_x) > 0:
            pad = 5
            roi_x0 = max(int(nz_x.min()) - pad, 0)
            roi_y0 = max(int(nz_y.min()) - pad, 0)
            roi_x1 = min(int(nz_x.max()) + pad + 1, Gx)
            roi_y1 = min(int(nz_y.max()) + pad + 1, Gy)
        else:
            roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, Gx, Gy
    else:
        roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, Gx, Gy

    obstacle_roi = obstacle[roi_x0:roi_x1, roi_y0:roi_y1, :]

    bbox_center = (np.array(bbox3d[:3]) + np.array(bbox3d[3:])) / 2.0
    obj_center_world = (T_obj2world @ np.append(bbox_center, 1))[:3]

    yaw_angles = np.linspace(0, 2 * np.pi, yaw_steps, endpoint=False)

    all_candidates = []
    yaw_rel_voxels = []
    yaw_vmin_rot = []
    yaw_T_rotated = []
    yaw_footprints = []
    valid_yaw_count = 0
    total_raw = 0

    _empty_vox = np.empty((0, 3), dtype=int)
    _empty_foot = np.empty((0, 2), dtype=int)
    _zero3 = np.zeros(3, dtype=np.float64)

    for yaw_idx, angle in enumerate(yaw_angles):
        R_yaw = rotation_z_3x3(angle)

        T_rot = T_obj2world.copy()
        T_rot[:3, :3] = R_yaw @ T_obj2world[:3, :3]
        T_rot[:3, 3] = obj_center_world + R_yaw @ (
            T_obj2world[:3, 3] - obj_center_world)

        rot_voxels = voxelize_obb(bbox3d, T_rot, vp, grid_shape)

        if len(rot_voxels) == 0:
            yaw_rel_voxels.append(_empty_vox)
            yaw_vmin_rot.append(_zero3.copy())
            yaw_T_rotated.append(T_rot)
            yaw_footprints.append(_empty_foot)
            continue

        vmin_rot = rot_voxels.min(axis=0).astype(np.float64)
        rel_rot = rot_voxels - rot_voxels.min(axis=0)
        osize = rel_rot.max(axis=0) + 1

        obj_mask = np.zeros(tuple(osize), dtype=bool)
        obj_mask[rel_rot[:, 0], rel_rot[:, 1], rel_rot[:, 2]] = True

        # 底面足迹（Z 最低层）
        bottom = rel_rot[rel_rot[:, 2] == 0][:, :2]

        # FFT 碰撞检测
        coll = _compute_collision_slice(
            obstacle_roi, obj_mask, landing_z, use_gpu=use_gpu)

        if coll is None:
            yaw_rel_voxels.append(rel_rot)
            yaw_vmin_rot.append(vmin_rot)
            yaw_T_rotated.append(T_rot)
            yaw_footprints.append(bottom)
            continue

        free_mask = coll < 0.5
        ys, xs = np.where(free_mask)
        # 偏移回全局坐标
        ys += roi_x0
        xs += roi_y0
        n_free = len(ys)
        total_raw += n_free

        if n_free > 0:
            yaw_col = np.full(n_free, yaw_idx, dtype=int)
            all_candidates.append(np.stack([ys, xs, yaw_col], axis=1))
            valid_yaw_count += 1

        yaw_rel_voxels.append(rel_rot)
        yaw_vmin_rot.append(vmin_rot.astype(np.float64))
        yaw_T_rotated.append(T_rot)
        yaw_footprints.append(bottom)

    candidates = (np.vstack(all_candidates) if all_candidates
                  else np.empty((0, 3), dtype=int))

    yaw_data = {
        "yaw_angles":   yaw_angles,
        "rel_voxels":   yaw_rel_voxels,
        "vmin_rot_abs": yaw_vmin_rot,
        "T_rotated":    yaw_T_rotated,
        "footprints":   yaw_footprints,
        "original_yaw_index": 0,
    }

    meta = {
        "total_xy":         int(Gx * Gy),
        "valid_raw":        total_raw,
        "yaw_steps":        yaw_steps,
        "valid_yaw_angles": valid_yaw_count,
        "landing_z":        landing_z,
        "table_z":          table_z,
    }

    return candidates, meta, yaw_data
