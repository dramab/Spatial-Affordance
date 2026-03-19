"""
src/annotation/free_bbox/filters.py
-------------------------------------
放置候选过滤器：可见性、物理稳定性、遮挡过滤。

所有过滤器接收 candidates (N, 3) [grid_x, grid_y, yaw_index] 格式，
返回过滤后的子集。

用法:
    from src.annotation.free_bbox.filters import (
        filter_visible_placements, filter_stable_placements,
        filter_occluded_placements, build_depth_buffer,
        is_fully_visible,
    )
"""

import numpy as np

from src.utils.coord_utils import transform_points, project_world
from src.annotation.free_bbox.grid_ops import _get_bbox_corners
from src.annotation.free_bbox.voxel_utils import voxel_to_world
from src.annotation.free_bbox.occupancy import FREE, OCCUPIED


def is_fully_visible(bbox3d, pose_cam, fx, fy, cx, cy, img_w, img_h):
    """
    检查物体 OBB 的 8 个角点是否全部投影在图像范围内。

    输入:
        bbox3d: (6,) 物体 canonical AABB
        pose_cam: (4, 4) object→camera 变换矩阵
        fx, fy, cx, cy: 相机内参
        img_w, img_h: 图像宽高
    输出:
        bool 是否完全可见
    """
    corners_cam = transform_points(
        _get_bbox_corners(bbox3d), np.asarray(pose_cam, dtype=np.float64))
    if np.any(corners_cam[:, 2] <= 0):
        return False
    uv = np.stack([
        fx * corners_cam[:, 0] / corners_cam[:, 2] + cx,
        fy * corners_cam[:, 1] / corners_cam[:, 2] + cy,
    ], axis=1)
    return bool(np.all(uv[:, 0] >= 0) and np.all(uv[:, 0] < img_w) and
                np.all(uv[:, 1] >= 0) and np.all(uv[:, 1] < img_h))


def filter_visible_placements(candidates, landing_z,
                               bbox3d, T_obj2world, E_w2c, K,
                               img_w, img_h, vp, yaw_data,
                               margin_px=30):
    """
    保留 OBB 投影在图像范围内的放置候选（向量化，按 yaw 角度批处理）。

    输入:
        candidates: (N, 3) int [grid_x, grid_y, yaw_index]
        landing_z: int 放置 Z 层
        bbox3d: (6,) 物体 canonical AABB
        T_obj2world: (4, 4) 物体原始变换
        E_w2c: (4, 4) world→camera
        K: (3, 3) 内参矩阵
        img_w, img_h: 图像宽高
        vp: dict 体素参数
        yaw_data: dict 来自 find_table_placements 的 yaw 旋转数据
        margin_px: int 像素边距容差
    输出:
        (M, 3) int 过滤后的候选
    """
    if len(candidates) == 0:
        return candidates

    vs = float(vp["voxel_size"])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    R_w2c = E_w2c[:3, :3]
    corners_canonical = _get_bbox_corners(bbox3d)

    keep = np.zeros(len(candidates), dtype=bool)
    yaw_angles = yaw_data["yaw_angles"]
    vmin_rots = yaw_data["vmin_rot_abs"]
    T_rot_list = yaw_data["T_rotated"]

    for yaw_idx in range(len(yaw_angles)):
        mask = candidates[:, 2] == yaw_idx
        if not mask.any():
            continue

        batch = candidates[mask]
        T_rot = T_rot_list[yaw_idx]
        vmin_rot = vmin_rots[yaw_idx]

        corners_cam_base = transform_points(corners_canonical, E_w2c @ T_rot)

        N = len(batch)
        anchors = np.column_stack([batch[:, :2].astype(np.float64),
                                   np.full(N, landing_z, dtype=np.float64)])
        delta_cam = (R_w2c @ ((anchors - vmin_rot) * vs).T).T

        all_cam = corners_cam_base[None, :, :] + delta_cam[:, None, :]

        Z = all_cam[:, :, 2]
        z_ok = np.all(Z > 0, axis=1)
        safe_Z = np.where(Z > 0, Z, 1.0)
        U = all_cam[:, :, 0] / safe_Z * fx + cx
        V = all_cam[:, :, 1] / safe_Z * fy + cy

        m = margin_px
        u_ok = np.all(U >= -m, axis=1) & np.all(U < img_w + m, axis=1)
        v_ok = np.all(V >= -m, axis=1) & np.all(V < img_h + m, axis=1)

        keep[mask] = z_ok & u_ok & v_ok

    return candidates[keep]


def filter_stable_placements(candidates, yaw_data, table_mask_2d,
                              min_support_ratio=1.0,
                              chunk_size=2000):
    """
    保留底面足迹被支撑面充分支撑且质心投影在支撑面上的候选。

    输入:
        candidates: (N, 3) int [grid_x, grid_y, yaw_index]
        yaw_data: dict 来自 find_table_placements 的 yaw 旋转数据
        table_mask_2d: (Gx, Gy) bool 支撑面掩码
        min_support_ratio: float 最小支撑比（0~1）
        chunk_size: int 每批处理的最大候选数
    输出:
        (M, 3) int 过滤后的候选
    """
    if len(candidates) == 0:
        return candidates

    Gx, Gy = table_mask_2d.shape
    footprints = yaw_data["footprints"]
    keep = np.zeros(len(candidates), dtype=bool)

    for yaw_idx in range(len(yaw_data["yaw_angles"])):
        mask = candidates[:, 2] == yaw_idx
        if not mask.any():
            continue

        batch = candidates[mask]
        foot = footprints[yaw_idx]
        if len(foot) == 0:
            continue

        n_foot = len(foot)
        batch_keep = np.zeros(len(batch), dtype=bool)

        for c0 in range(0, len(batch), chunk_size):
            c1 = min(c0 + chunk_size, len(batch))
            sub = batch[c0:c1]
            N = len(sub)

            # 足迹体素的全局坐标
            fi = sub[:, 0:1] + foot[:, 0:1].T  # (N, n_foot)
            fj = sub[:, 1:2] + foot[:, 1:2].T  # (N, n_foot)

            in_bounds = ((fi >= 0) & (fi < Gx) &
                         (fj >= 0) & (fj < Gy))

            fi_c = np.clip(fi, 0, Gx - 1)
            fj_c = np.clip(fj, 0, Gy - 1)
            on_table = table_mask_2d[fi_c, fj_c] & in_bounds

            ratio = on_table.sum(axis=1) / max(n_foot, 1)

            # 质心投影检查
            com_i = sub[:, 0] + foot[:, 0].mean()
            com_j = sub[:, 1] + foot[:, 1].mean()
            com_i_int = np.round(com_i).astype(int)
            com_j_int = np.round(com_j).astype(int)
            com_in = ((com_i_int >= 0) & (com_i_int < Gx) &
                      (com_j_int >= 0) & (com_j_int < Gy))
            com_ok = com_in & table_mask_2d[np.clip(com_i_int, 0, Gx - 1),
                                            np.clip(com_j_int, 0, Gy - 1)]

            batch_keep[c0:c1] = (ratio >= min_support_ratio) & com_ok

        keep[mask] = batch_keep

    return candidates[keep]


def build_depth_buffer(grid_work, vp, K, E_w2c, img_w, img_h):
    """
    从工作栅格的 OCCUPIED 体素构建每像素最小深度缓冲。

    输入:
        grid_work: (Gx, Gy, Gz) uint8 工作栅格（目标物体已移除）
        vp: dict 体素参数
        K: (3, 3) 内参矩阵
        E_w2c: (4, 4) world→camera
        img_w, img_h: 图像宽高
    输出:
        (img_h, img_w) float64 深度缓冲，未覆盖像素为 inf
    """
    occ_idx = np.argwhere(grid_work == OCCUPIED)
    depth_buf = np.full((img_h, img_w), np.inf, dtype=np.float64)

    if len(occ_idx) == 0:
        return depth_buf

    occ_world = voxel_to_world(occ_idx, vp)
    uv, z_cam = project_world(occ_world, K, E_w2c)

    valid = ((z_cam > 0) &
             np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) &
             (uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
             (uv[:, 1] >= 0) & (uv[:, 1] < img_h))

    u_int = np.clip(np.round(uv[valid, 0]).astype(int), 0, img_w - 1)
    v_int = np.clip(np.round(uv[valid, 1]).astype(int), 0, img_h - 1)
    z_valid = z_cam[valid]

    np.minimum.at(depth_buf, (v_int, u_int), z_valid)
    return depth_buf


def filter_occluded_placements(candidates, landing_z,
                                bbox3d, T_obj2world,
                                depth_buffer, K, E_w2c, vp,
                                yaw_data, img_w, img_h,
                                occlusion_threshold=0.3):
    """
    移除被现有场景几何遮挡的放置候选（Z-buffer 深度比较）。

    输入:
        candidates: (N, 3) int [grid_x, grid_y, yaw_index]
        landing_z: int 放置 Z 层
        bbox3d: (6,) 物体 canonical AABB
        T_obj2world: (4, 4) 物体原始变换
        depth_buffer: (img_h, img_w) float64 深度缓冲
        K: (3, 3) 内参矩阵
        E_w2c: (4, 4) world→camera
        vp: dict 体素参数
        yaw_data: dict yaw 旋转数据
        img_w, img_h: 图像宽高
        occlusion_threshold: float 遮挡比例阈值
    输出:
        (M, 3) int 过滤后的候选
    """
    if len(candidates) == 0:
        return candidates

    vs = float(vp["voxel_size"])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    R_w2c = E_w2c[:3, :3]
    corners_canonical = _get_bbox_corners(bbox3d)

    keep = np.zeros(len(candidates), dtype=bool)
    yaw_angles = yaw_data["yaw_angles"]
    vmin_rots = yaw_data["vmin_rot_abs"]
    T_rot_list = yaw_data["T_rotated"]

    for yaw_idx in range(len(yaw_angles)):
        mask = candidates[:, 2] == yaw_idx
        if not mask.any():
            continue

        batch = candidates[mask]
        T_rot = T_rot_list[yaw_idx]
        vmin_rot = vmin_rots[yaw_idx]

        corners_cam_base = transform_points(corners_canonical, E_w2c @ T_rot)

        N = len(batch)
        anchors = np.column_stack([batch[:, :2].astype(np.float64),
                                   np.full(N, landing_z, dtype=np.float64)])
        delta_cam = (R_w2c @ ((anchors - vmin_rot) * vs).T).T

        all_cam = corners_cam_base[None, :, :] + delta_cam[:, None, :]

        Z = all_cam[:, :, 2]
        safe_Z = np.where(Z > 0, Z, 1.0)
        U = all_cam[:, :, 0] / safe_Z * fx + cx
        V = all_cam[:, :, 1] / safe_Z * fy + cy

        U_int = np.clip(np.round(U).astype(int), 0, img_w - 1)
        V_int = np.clip(np.round(V).astype(int), 0, img_h - 1)

        buf_z = depth_buffer[V_int, U_int]  # (N, 8)
        behind = (Z > 0) & (Z > buf_z + vs)
        frac = behind.sum(axis=1) / 8.0

        keep[mask] = frac <= occlusion_threshold

    return candidates[keep]
