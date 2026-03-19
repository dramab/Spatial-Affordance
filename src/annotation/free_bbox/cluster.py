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


def cluster_placements(candidates, grid_work, yaw_data,
                       landing_z, vp, eps=5.0, min_samples=1):
    """
    在世界坐标 XY 平面对候选进行 DBSCAN 聚类，每个簇选最优代表。

    代表选择标准：周围自由空间体素数最多。

    输入:
        candidates: (N, 3) int [grid_x, grid_y, yaw_index]
        grid_work: (Gx, Gy, Gz) uint8 工作栅格
        yaw_data: dict yaw 旋转数据
        landing_z: int 放置 Z 层
        vp: dict 体素参数
        eps: float DBSCAN 聚类半径（场景单位）
        min_samples: int DBSCAN 最小样本数
    输出:
        reps: (K, 3) int 每个簇的代表候选
        infos: list[dict] 每个簇的详细信息
    """
    if len(candidates) == 0:
        return np.empty((0, 3), dtype=int), []

    vs = float(vp["voxel_size"])
    anchors_3d = np.column_stack([candidates[:, :2],
                                  np.full(len(candidates), landing_z)])
    pts_w = voxel_to_world(anchors_3d, vp)[:, :2]

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts_w)
    unique = sorted(set(labels) - {-1})

    Gx, Gy, Gz = grid_work.shape
    rel_voxels_list = yaw_data["rel_voxels"]
    yaw_angles = yaw_data["yaw_angles"]

    reps, infos = [], []
    for lbl in unique:
        members = candidates[labels == lbl]
        best_score, best_pos = -1, members[0]

        for pos in members:
            ix, iy, yi = int(pos[0]), int(pos[1]), int(pos[2])
            rv = rel_voxels_list[yi]
            osize = rv.max(axis=0) + 1 if len(rv) > 0 else np.array([1, 1, 1])
            pad = 3
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
            "cluster_id":      int(lbl),
            "size":            len(members),
            "anchor_voxel":    anchor_3d.tolist(),
            "anchor_world":    voxel_to_world(anchor_3d, vp).tolist(),
            "yaw_index":       int(anchor[2]),
            "yaw_degrees":     float(np.degrees(yaw_angles[anchor[2]])),
            "free_score":      best_score,
        })

    arr = np.array(reps, dtype=int) if reps else np.empty((0, 3), dtype=int)
    return arr, infos
