"""
src/annotation/free_bbox/cluster.py
------------------------------------
DBSCAN 聚类：对放置候选在世界坐标 XY 平面聚类，
每个簇选择周围自由空间最多的代表。

用法:
    from src.annotation.free_bbox.cluster import cluster_placements
"""

import numpy as np
from sklearn.cluster import DBSCAN

from src.annotation.free_bbox.occupancy import FREE
from src.annotation.free_bbox.voxel_utils import voxel_to_world


def _estimate_dbscan_eps(yaw_data, vp,
                         size_ratio=0.5,
                         min_eps_voxels=3.0,
                         max_eps_voxels=12.0):
    """
    根据物体水平尺度自适应估计 DBSCAN eps。

    以各 yaw 下 XY 包围盒较长边的中位数作为尺度，
    再乘一个比例系数，并限制在稳定范围内，
    避免不同大小物体共用固定半径时过聚合或过分裂。
    """
    vs = float(vp["voxel_size"])
    xy_sizes = []

    for rv in yaw_data.get("rel_voxels", []):
        if len(rv) == 0:
            continue
        osize = rv.max(axis=0) + 1
        xy_sizes.append(float(max(osize[0], osize[1])))

    if not xy_sizes:
        return 5.0 * vs

    obj_xy_scale = float(np.median(xy_sizes))
    return float(np.clip(size_ratio * obj_xy_scale * vs,
                         min_eps_voxels * vs,
                         max_eps_voxels * vs))


def cluster_placements(candidates, grid_work, yaw_data,
                       landing_z, vp, eps=None, min_samples=1):
    """
    在世界坐标 XY 平面对候选进行 DBSCAN 聚类，每个簇选最优代表。

    代表选择标准：若簇内存在原始朝向 yaw，则优先在原始朝向中选择；
    否则在所有成员中选择周围自由空间体素数最多的代表。

    输入:
        candidates: (N, 3) int [grid_x, grid_y, yaw_index]
        grid_work: (Gx, Gy, Gz) uint8 工作栅格
        yaw_data: dict yaw 旋转数据
        landing_z: int 放置 Z 层
        vp: dict 体素参数
        eps: float | None DBSCAN 聚类半径（场景单位）；None 表示自适应估计
        min_samples: int DBSCAN 最小样本数
    输出:
        reps: (K, 3) int 每个簇的代表候选
        infos: list[dict] 每个簇的详细信息
    """
    if len(candidates) == 0:
        return np.empty((0, 3), dtype=int), []

    effective_eps = (_estimate_dbscan_eps(yaw_data, vp)
                     if eps is None else float(eps))
    if effective_eps <= 0:
        raise ValueError("DBSCAN eps must be positive")

    anchors_3d = np.column_stack([candidates[:, :2],
                                  np.full(len(candidates), landing_z)])
    pts_w = voxel_to_world(anchors_3d, vp)[:, :2]

    labels = DBSCAN(eps=effective_eps,
                    min_samples=int(min_samples)).fit_predict(pts_w)
    unique = sorted(set(labels) - {-1})

    Gx, Gy, Gz = grid_work.shape
    rel_voxels_list = yaw_data["rel_voxels"]
    yaw_angles = yaw_data["yaw_angles"]
    original_yaw_index = int(yaw_data.get("original_yaw_index", 0))

    reps, infos = [], []
    for lbl in unique:
        members = candidates[labels == lbl]
        preferred = members[members[:, 2] == original_yaw_index]
        scored_members = preferred if len(preferred) > 0 else members
        used_original_yaw = len(preferred) > 0
        best_score, best_pos = -1, scored_members[0]

        for pos in scored_members:
            ix, iy, yi = int(pos[0]), int(pos[1]), int(pos[2])
            rv = rel_voxels_list[yi]
            osize = rv.max(axis=0) + 1 if len(rv) > 0 else np.array([1, 1, 1])
            # 自适应水平缓冲区：约取物体较长边的 25%，
            # 并限制在一个稳定范围内，避免对大小物体都用同一个固定 pad。
            pad = int(np.clip(np.ceil(0.25 * max(osize[0], osize[1])), 2, 6))
            region = grid_work[
                max(ix - pad, 0): min(ix + osize[0] + pad, Gx),
                max(iy - pad, 0): min(iy + osize[1] + pad, Gy),
                max(landing_z - 1, 0): min(landing_z + osize[2] + 1, Gz)]
            score = int((region == FREE).sum())
            if score > best_score:
                best_score, best_pos = score, pos

        anchor = np.array([int(best_pos[0]), int(best_pos[1]),
                           int(best_pos[2])], dtype=int)
        reps.append(anchor)

        anchor_3d = np.array([anchor[0], anchor[1], landing_z])
        infos.append({
            "cluster_id":          int(lbl),
            "size":                len(members),
            "anchor_voxel":        anchor_3d.tolist(),
            "anchor_world":        voxel_to_world(anchor_3d, vp).tolist(),
            "yaw_index":           int(anchor[2]),
            "yaw_degrees":         float(np.degrees(yaw_angles[anchor[2]])),
            "free_score":          best_score,
            "dbscan_eps":          effective_eps,
            "dbscan_min_samples":  int(min_samples),
            "used_original_yaw":   bool(used_original_yaw),
        })

    arr = np.array(reps, dtype=int) if reps else np.empty((0, 3), dtype=int)
    return arr, infos
