"""
src/annotation/free_bbox/occupancy.py
--------------------------------------
RGBD → 3D 点云 + 占据栅格构建（ray-casting）。

通用模块，不包含任何数据集特定的深度缩放或单位转换。
输入的 depth 已由 adapter 转换为场景工作单位。

体素状态:
    FREE (0)     — 射线穿过该体素后才到达表面
    OCCUPIED (1) — 深度测量命中，表面存在
    UNKNOWN (2)  — 从未被任何射线遍历（遮挡或在相机视锥外）

用法:
    from src.annotation.free_bbox.occupancy import depth_to_pointcloud, build_occupancy_grid
"""

import numpy as np

FREE, OCCUPIED, UNKNOWN = 0, 1, 2


def _sample_depth_pixels(depth, stride):
    """按 stride 采样有效深度像素。"""
    h, w = depth.shape
    u_arr = np.arange(0, w, stride)
    v_arr = np.arange(0, h, stride)
    uu, vv = np.meshgrid(u_arr, v_arr)
    uu, vv = uu.ravel(), vv.ravel()

    d = depth[vv, uu]
    mask = d > 0
    return uu[mask], vv[mask], d[mask]


def _depth_samples_to_world(uu, vv, d, fx, fy, cx, cy, R_c2w, cam_origin):
    """将采样后的深度像素反投影到世界坐标系。"""
    pts_cam = np.stack([
        (uu - cx) / fx * d,
        (vv - cy) / fy * d,
        d
    ], axis=1)
    return (R_c2w @ pts_cam.T).T + cam_origin


def depth_to_pointcloud(depth, rgb, fx, fy, cx, cy,
                        R_c2w, cam_origin, stride=2):
    """
    深度图反投影为世界坐标系 3D 彩色点云。

    输入:
        depth: (H, W) float 深度图，已转换为场景工作单位
        rgb: (H, W, 3) uint8 RGB 图像
        fx, fy, cx, cy: 相机内参
        R_c2w: (3, 3) camera→world 旋转矩阵
        cam_origin: (3,) 相机在世界坐标系中的位置
        stride: 采样步长（每 stride×stride 像素块取 1 个）
    输出:
        pts_world: (N, 3) 世界坐标点云
        colors: (N, 3) uint8 颜色
    """
    uu, vv, d = _sample_depth_pixels(depth, stride)
    pts_world = _depth_samples_to_world(
        uu, vv, d, fx, fy, cx, cy, R_c2w, cam_origin)
    colors = rgb[vv, uu].astype(np.uint8)            # (N, 3)
    return pts_world, colors


def build_occupancy_grid(depth, fx, fy, cx, cy,
                         R_c2w, cam_origin,
                         voxel_size=1.0, stride=4, padding=10.0,
                         surface_points=None):
    """
    通过 ray-casting 构建 3D 占据栅格。

    算法:
        对每个采样像素:
        1. 从深度值计算世界坐标表面点
        2. 从相机原点向表面点投射射线
        3. 射线经过的体素标记为 FREE
        4. 表面点所在体素标记为 OCCUPIED
        5. 未被任何射线到达的体素保持 UNKNOWN
        状态优先级: OCCUPIED > FREE > UNKNOWN

    输入:
        depth: (H, W) float 深度图，已转换为场景工作单位
        fx, fy, cx, cy: 相机内参
        R_c2w: (3, 3) camera→world 旋转矩阵
        cam_origin: (3,) 相机在世界坐标系中的位置
        voxel_size: 体素边长（场景单位）
        stride: 深度图采样步长
        padding: 栅格边界 padding（场景单位）
        surface_points: (N, 3) 预计算的世界坐标表面点（可选）
    输出:
        grid: (Gx, Gy, Gz) uint8 占据栅格
        grid_min: (3,) 栅格最小角世界坐标
        voxel_size: float 体素边长
    """
    step = voxel_size * 0.5  # 亚体素步长

    if surface_points is None:
        uu, vv, d = _sample_depth_pixels(depth, stride)
        surface = _depth_samples_to_world(
            uu, vv, d, fx, fy, cx, cy, R_c2w, cam_origin)
    else:
        surface = np.asarray(surface_points, dtype=np.float64)
        if surface.ndim != 2 or surface.shape[1] != 3:
            raise ValueError("surface_points must have shape (N, 3)")

    # 栅格边界
    all_pts = np.vstack([surface, cam_origin.reshape(1, 3)])
    grid_min = all_pts.min(axis=0) - padding
    grid_max = all_pts.max(axis=0) + padding
    grid_shape = np.ceil((grid_max - grid_min) / voxel_size).astype(int)

    grid = np.full(tuple(grid_shape), UNKNOWN, dtype=np.uint8)

    # 射线向量
    ray_vecs = surface - cam_origin
    ray_lens = np.linalg.norm(ray_vecs, axis=1)
    ray_dirs = ray_vecs / ray_lens[:, None]

    def to_idx(pts):
        return np.floor((pts - grid_min) / voxel_size).astype(int)

    def in_bounds(idx):
        return ((idx[:, 0] >= 0) & (idx[:, 0] < grid_shape[0]) &
                (idx[:, 1] >= 0) & (idx[:, 1] < grid_shape[1]) &
                (idx[:, 2] >= 0) & (idx[:, 2] < grid_shape[2]))

    # 沿射线步进 → 标记 FREE
    max_t = float(ray_lens.max())
    t_vals = np.arange(0.0, max_t, step)

    for t in t_vals:
        active = ray_lens > t
        if not active.any():
            break
        pts = cam_origin + ray_dirs[active] * t
        idx = to_idx(pts)
        valid = in_bounds(idx)
        vi = idx[valid]
        free_mask = grid[vi[:, 0], vi[:, 1], vi[:, 2]] != OCCUPIED
        fi = vi[free_mask]
        if len(fi) > 0:
            grid[fi[:, 0], fi[:, 1], fi[:, 2]] = FREE

    # 表面体素 → 标记 OCCUPIED
    surf_idx = to_idx(surface)
    valid = in_bounds(surf_idx)
    si = surf_idx[valid]
    grid[si[:, 0], si[:, 1], si[:, 2]] = OCCUPIED

    return grid, grid_min, voxel_size
