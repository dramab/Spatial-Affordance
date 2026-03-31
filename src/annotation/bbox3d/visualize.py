"""
src/annotation/bbox3d/visualize.py
-----------------------------------
3D bbox 投影到 RGB 图像的可视化，包含接触面高亮。

用法:
    from src.annotation.bbox3d.visualize import visualize_bbox3d_on_image
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon

from src.annotation.bbox3d.bbox_utils import (
    get_bbox_corners, get_contact_face_indices,
)
from src.utils.coord_utils import transform_points

# 12 条 bbox 边
BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

BOTTOM_COLOR = "orange"


def _project_points(points_3d, fx, fy, cx, cy):
    """将相机坐标系下的 3D 点投影到像素坐标。"""
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    return np.stack([fx * X / Z + cx, fy * Y / Z + cy], axis=1)


def visualize_bbox3d_on_image(rgb, objects, camera, E_w2c,
                               out_path, world_up=np.array([0., 0., 1.])):
    """
    将所有物体的 3D bbox 投影到 RGB 图像上，高亮接触面。

    输入:
        rgb: (H, W, 3) uint8 RGB 图像
        objects: list[ObjectInfo] 物体列表
        camera: CameraParams 相机参数
        E_w2c: (4, 4) world→camera 变换矩阵
        out_path: str 输出图像路径
        world_up: (3,) 世界坐标系上方向
    """
    fx, fy = camera.fx, camera.fy
    cx, cy = camera.cx, camera.cy

    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    ax.imshow(rgb)
    ax.axis("off")

    cmap = plt.get_cmap("tab10")
    legend_handles = []

    for idx, obj in enumerate(objects):
        color = cmap(idx % 10)
        corners_obj = get_bbox_corners(obj.bbox3d_canonical)

        # object→camera
        T_obj2cam = E_w2c @ obj.pose_world
        corners_cam = transform_points(corners_obj, T_obj2cam)

        # 跳过相机后方的物体
        if np.any(corners_cam[:, 2] <= 0):
            continue

        corners_2d = _project_points(corners_cam, fx, fy, cx, cy)

        # 绘制 12 条边
        for i, j in BOX_EDGES:
            ax.plot([corners_2d[i, 0], corners_2d[j, 0]],
                    [corners_2d[i, 1], corners_2d[j, 1]],
                    color=color, linewidth=1.5, alpha=0.9)

        # 接触面高亮
        contact_idx = get_contact_face_indices(obj.pose_world, E_w2c, world_up)
        contact_2d = corners_2d[contact_idx]
        poly = Polygon(contact_2d, closed=True,
                       facecolor=BOTTOM_COLOR, edgecolor=BOTTOM_COLOR,
                       alpha=0.35, linewidth=2.0, zorder=3)
        ax.add_patch(poly)

        # bbox 中心 → 接触面中心虚线
        center_obj = (obj.bbox3d_canonical[:3] + obj.bbox3d_canonical[3:]) / 2.0
        contact_face_obj = corners_obj[contact_idx].mean(axis=0)
        two_pts_cam = transform_points(
            np.stack([center_obj, contact_face_obj]), T_obj2cam)
        two_pts_2d = _project_points(two_pts_cam, fx, fy, cx, cy)
        ax.plot([two_pts_2d[0, 0], two_pts_2d[1, 0]],
                [two_pts_2d[0, 1], two_pts_2d[1, 1]],
                color=BOTTOM_COLOR, linewidth=1.5,
                linestyle="--", alpha=0.9, zorder=4)

        # 类别标注
        z_vals = corners_cam[:, 2]
        near_mask = z_vals <= np.partition(z_vals, 4)[4]
        near_2d = corners_2d[near_mask]
        ax.text(near_2d[:, 0].mean(), near_2d[:, 1].min() - 5,
                obj.class_name,
                color=color, fontsize=7, fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(fc="black", alpha=0.4, pad=1.5, edgecolor="none"))

        legend_handles.append(
            mpatches.Patch(color=color, label=obj.class_name))

    legend_handles.append(
        mpatches.Patch(color=BOTTOM_COLOR, alpha=0.6, label="contact face"))
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=7, framealpha=0.6)

    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
