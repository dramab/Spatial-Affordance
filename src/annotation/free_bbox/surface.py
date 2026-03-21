"""
src/annotation/free_bbox/surface.py
------------------------------------
支撑面检测：优先在点云中用 RANSAC 检测水平平面，
失败时退化到占据栅格的逐层连通域搜索。

用法:
    from src.annotation.free_bbox.surface import detect_support_surfaces
"""

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label

from src.annotation.free_bbox.occupancy import OCCUPIED


def _largest_component(mask_2d, min_voxels):
    """返回二维掩码中面积最大的连通域。"""
    opened = binary_opening(mask_2d, structure=np.ones((3, 3), dtype=bool))
    labeled, n_features = label(opened)

    best_area = 0
    best_mask = None
    for comp_id in range(1, n_features + 1):
        component = (labeled == comp_id)
        area = int(component.sum())
        if area >= min_voxels and area > best_area:
            best_area = area
            best_mask = component
    return best_area, best_mask


def _detect_support_surfaces_from_grid(grid, min_voxels):
    """在占据栅格中逐层搜索最大水平支撑面。"""
    occ = (grid == OCCUPIED)

    z_counts = occ.sum(axis=(0, 1))
    active_zs = np.where(z_counts >= min_voxels)[0]

    best_z, best_area, best_mask = None, 0, None
    for z in active_zs:
        slice_2d = occ[:, :, z]
        area, component = _largest_component(slice_2d, min_voxels)
        if component is not None and area > best_area:
            best_z = int(z)
            best_area = area
            best_mask = component

    return best_z, best_mask


def _sample_plane_triplets(points, num_iters, rng):
    """随机采样三点组，用于 RANSAC 平面拟合。"""
    n_points = len(points)
    if n_points < 3:
        return np.empty((0, 3), dtype=int)
    return rng.integers(0, n_points, size=(num_iters, 3))


def _fit_plane_from_triplet(points, triplet):
    """由三个点拟合平面，返回单位法向量和偏置。"""
    p0, p1, p2 = points[triplet]
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None, None
    normal = normal / norm
    d = -float(np.dot(normal, p0))
    return normal, d


def _plane_mask_to_surface(mask_2d, min_voxels):
    """平面投影后做形态学闭运算并提取最大连通域。"""
    if mask_2d is None or not mask_2d.any():
        return None

    closed = binary_closing(mask_2d, structure=np.ones((3, 3), dtype=bool))
    _, component = _largest_component(closed, min_voxels)
    return component


def _detect_support_surfaces_from_pointcloud(
        points_world, grid, vp, min_voxels,
        distance_thresh=None, num_iters=192,
        normal_z_min=0.9, random_seed=0):
    """
    在点云中用 RANSAC 检测近似水平平面，并投影回体素平面掩码。
    """
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        return None, None

    vs = float(vp["voxel_size"])
    origin = np.asarray(vp["origin"], dtype=np.float64)
    grid_shape = np.asarray(grid.shape, dtype=int)
    distance_thresh = float(distance_thresh or max(vs * 1.5, 1e-3))

    rng = np.random.default_rng(random_seed)
    triplets = _sample_plane_triplets(points, num_iters, rng)

    best_score = -1.0
    best_z = None
    best_mask = None

    for triplet in triplets:
        if len({int(triplet[0]), int(triplet[1]), int(triplet[2])}) < 3:
            continue

        normal, d = _fit_plane_from_triplet(points, triplet)
        if normal is None:
            continue

        if abs(normal[2]) < normal_z_min:
            continue
        if normal[2] < 0:
            normal = -normal
            d = -d

        distances = np.abs(points @ normal + d)
        inliers = distances <= distance_thresh
        if int(inliers.sum()) < min_voxels:
            continue

        plane_pts = points[inliers]
        voxel_idx = np.floor((plane_pts - origin) / vs).astype(int)
        valid = ((voxel_idx[:, 0] >= 0) & (voxel_idx[:, 0] < grid_shape[0]) &
                 (voxel_idx[:, 1] >= 0) & (voxel_idx[:, 1] < grid_shape[1]) &
                 (voxel_idx[:, 2] >= 0) & (voxel_idx[:, 2] < grid_shape[2]))
        voxel_idx = voxel_idx[valid]
        if len(voxel_idx) < min_voxels:
            continue

        table_z = int(np.median(voxel_idx[:, 2]))
        close_to_plane = np.abs(voxel_idx[:, 2] - table_z) <= 1
        voxel_idx = voxel_idx[close_to_plane]
        if len(voxel_idx) < min_voxels:
            continue

        mask_2d = np.zeros(grid_shape[:2], dtype=bool)
        mask_2d[voxel_idx[:, 0], voxel_idx[:, 1]] = True
        surface_mask = _plane_mask_to_surface(mask_2d, min_voxels)
        if surface_mask is None:
            continue

        area = int(surface_mask.sum())
        z_spread = float(np.std(plane_pts[:, 2])) if len(plane_pts) > 1 else 0.0
        score = area - z_spread / max(vs, 1e-6)
        if score > best_score:
            best_score = score
            best_z = table_z
            best_mask = surface_mask

    return best_z, best_mask


def detect_support_surfaces(grid, vp, min_area=50.0, points_world=None,
                            distance_thresh=None, num_iters=192,
                            normal_z_min=0.9, random_seed=0):
    """
    检测最大水平支撑面。

    算法:
        1. 若提供点云，优先用 RANSAC 拟合近似水平平面
        2. 将最佳平面投影到体素 XY 平面并提取最大连通域
        3. 若点云平面检测失败，则退化到占据栅格逐层搜索

    输入:
        grid: (Gx, Gy, Gz) uint8 占据栅格
        vp: dict 体素参数 {"voxel_size": float, "origin": [x,y,z]}
        min_area: float 最小支撑面面积（场景单位²）
        points_world: (N, 3) 世界坐标点云（可选，推荐）
        distance_thresh: float RANSAC 平面内点距离阈值（默认 1.5 voxel）
        num_iters: int RANSAC 迭代次数
        normal_z_min: float 法向量 z 分量阈值，越大越接近水平面
        random_seed: int RANSAC 随机种子
    输出:
        table_z: int 支撑面所在的 Z 层索引，未找到则为 None
        surface_mask_2d: (Gx, Gy) bool 支撑面的 2D 掩码，未找到则为 None
    """
    vs = float(vp["voxel_size"])
    min_voxels = max(1, int(min_area / (vs * vs)))

    if points_world is not None:
        best_z, best_mask = _detect_support_surfaces_from_pointcloud(
            points_world, grid, vp, min_voxels,
            distance_thresh=distance_thresh,
            num_iters=num_iters,
            normal_z_min=normal_z_min,
            random_seed=random_seed)
        if best_z is not None:
            return best_z, best_mask

    return _detect_support_surfaces_from_grid(grid, min_voxels)


def detect_table_z(grid):
    """
    简易版：找 Z 层中 OCCUPIED 体素最多的层作为桌面高度。

    输入:
        grid: (Gx, Gy, Gz) uint8 占据栅格
    输出:
        int Z 层索引，无 OCCUPIED 体素则为 None
    """
    occ = (grid == OCCUPIED)
    z_counts = occ.sum(axis=(0, 1))
    if z_counts.max() == 0:
        return None
    return int(np.argmax(z_counts))
