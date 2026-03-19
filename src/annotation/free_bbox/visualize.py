"""
src/annotation/free_bbox/visualize.py
--------------------------------------
放置规划可视化：占据栅格切片/3D 视图、放置结果双面板可视化。

用法:
    from src.annotation.free_bbox.visualize import (
        visualize_slices, visualize_3d, visualize_overview, save_placement_vis,
    )
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.annotation.free_bbox.occupancy import FREE, OCCUPIED, UNKNOWN
from src.annotation.free_bbox.grid_ops import _get_bbox_corners
from src.annotation.free_bbox.voxel_utils import voxel_to_world
from src.utils.coord_utils import transform_points, project_world

# 颜色常量
PALETTE = ["#4CAF50", "#F44336", "#9E9E9E"]  # Free, Occupied, Unknown
CLR_BG = "#1A1A2E"
CLR_PANEL = "#0D0D1A"
CLR_ORIG = "#FF6D00"
CLR_PLACE = "#00E676"
CLR_CAM = "#FFEB3B"
CLR_ARROW = "#FFD54F"
CLR_OCC = "#78909C"
CLR_FREE = "#4CAF50"

BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def visualize_overview(rgb, pts_world, colors, cam_origin, out_path):
    """
    场景概览：RGB 图像 + 3D 点云俯视图。

    输入:
        rgb: (H, W, 3) uint8 RGB 图像
        pts_world: (N, 3) 世界坐标点云
        colors: (N, 3) uint8 点云颜色
        cam_origin: (3,) 相机位置
        out_path: str 输出路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    ax = axes[1]
    if len(pts_world) > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts_world), 10000, replace=False)
        pts_sub, clr_sub = pts_world[idx], colors[idx]
    else:
        pts_sub, clr_sub = pts_world, colors
    ax.scatter(pts_sub[:, 0], pts_sub[:, 1],
               c=clr_sub / 255.0, s=0.5, alpha=0.5)
    ax.plot(*cam_origin[:2], "r*", markersize=10, label="Camera")
    ax.set_title("Point Cloud (top view)")
    ax.set_aspect("equal")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def visualize_slices(grid, grid_min, voxel_size, out_path):
    """
    占据栅格三轴中间切片可视化。

    输入:
        grid: (Gx, Gy, Gz) uint8 占据栅格
        grid_min: (3,) 栅格最小角世界坐标
        voxel_size: float 体素边长
        out_path: str 输出路径
    """
    cmap = ListedColormap(PALETTE)
    Gx, Gy, Gz = grid.shape
    slices = [
        (grid[Gx // 2, :, :].T, f"X={Gx // 2}", "Y", "Z"),
        (grid[:, Gy // 2, :].T, f"Y={Gy // 2}", "X", "Z"),
        (grid[:, :, Gz // 2].T, f"Z={Gz // 2}", "X", "Y"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (data, title, xl, yl) in zip(axes, slices):
        im = ax.imshow(data, origin="lower", cmap=cmap,
                       vmin=0, vmax=2,
                       interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    plt.colorbar(im, ax=axes[-1], ticks=[0, 1, 2],
                 label="0=Free | 1=Occupied | 2=Unknown")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def visualize_3d(grid, grid_min, voxel_size, out_path, max_pts=6000):
    """
    3D 散点图：Occupied（红）+ 采样 Free（绿）。

    输入:
        grid: (Gx, Gy, Gz) uint8 占据栅格
        grid_min: (3,) 栅格最小角世界坐标
        voxel_size: float 体素边长
        out_path: str 输出路径
        max_pts: int 每类最大点数
    """
    occ = np.argwhere(grid == OCCUPIED)
    free = np.argwhere(grid == FREE)

    rng = np.random.default_rng(42)
    if len(free) > max_pts:
        free = free[rng.choice(len(free), max_pts, replace=False)]
    if len(occ) > max_pts:
        occ = occ[rng.choice(len(occ), max_pts, replace=False)]

    def to_world(idx):
        return grid_min + idx * voxel_size

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    if len(free):
        fw = to_world(free)
        ax.scatter(fw[:, 0], fw[:, 1], fw[:, 2],
                   c=PALETTE[0], s=1, alpha=0.15,
                   label=f"Free ({len(free):,})")
    if len(occ):
        ow = to_world(occ)
        ax.scatter(ow[:, 0], ow[:, 1], ow[:, 2],
                   c=PALETTE[1], s=6, alpha=0.9,
                   label=f"Occupied ({len(occ):,})")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _draw_bbox_2d(ax, corners_world, K, E_w2c,
                  color, lw=2.0, label=None, alpha=1.0):
    """在 RGB 图像上绘制 3D bbox 的 2D 投影线框。"""
    uv, z = project_world(corners_world, K, E_w2c)
    u, v = uv[:, 0], uv[:, 1]
    drawn = False
    for i, j in BOX_EDGES:
        if z[i] <= 0 or z[j] <= 0:
            continue
        lbl = label if (not drawn and label) else None
        ax.plot([u[i], u[j]], [v[i], v[j]],
                color=color, lw=lw, alpha=alpha, label=lbl)
        drawn = True


def _draw_bbox_3d(ax, corners_world, color, lw=1.5, label=None, alpha=1.0):
    """在 3D 坐标轴上绘制 bbox 线框。"""
    for idx, (i, j) in enumerate(BOX_EDGES):
        lbl = label if idx == 0 else None
        ax.plot([corners_world[i, 0], corners_world[j, 0]],
                [corners_world[i, 1], corners_world[j, 1]],
                [corners_world[i, 2], corners_world[j, 2]],
                color=color, lw=lw, alpha=alpha, label=lbl)


def _compute_placed_transform_vis(T_obj2world, bbox3d, anchor_xy,
                                   landing_z, yaw_data, yaw_idx, vp):
    """计算放置物体的 4×4 变换矩阵（可视化用）。"""
    vs = float(vp["voxel_size"])
    T_rot = yaw_data["T_rotated"][yaw_idx]
    vmin_rot = yaw_data["vmin_rot_abs"][yaw_idx]

    anchor_3d = np.array([anchor_xy[0], anchor_xy[1], landing_z],
                         dtype=np.float64)
    delta_world = (anchor_3d - vmin_rot) * vs

    T_placed = T_rot.copy()
    T_placed[:3, 3] += delta_world
    return T_placed


def save_placement_vis(rgb, obj_name, bbox3d, T_obj2world,
                       K, E_w2c, vp, cam_origin,
                       cluster_reps, cluster_infos,
                       obj_voxels, grid, out_path,
                       yaw_data, landing_z):
    """
    双面板暗色主题放置可视化。

    左面板: RGB + 原始 bbox（橙色）+ 放置 bbox（绿色）
    右面板: 3D 世界视图 + 点云 + bbox + 相机 + 箭头

    输入:
        rgb: (H, W, 3) uint8 RGB 图像
        obj_name: str 物体名称
        bbox3d: (6,) 物体 canonical AABB
        T_obj2world: (4, 4) 物体原始变换
        K: (3, 3) 内参矩阵
        E_w2c: (4, 4) world→camera
        vp: dict 体素参数
        cam_origin: (3,) 相机位置
        cluster_reps: (K, 3) int 聚类代表
        cluster_infos: list[dict] 聚类信息
        obj_voxels: (M, 3) int 物体体素索引
        grid: (Gx, Gy, Gz) uint8 栅格
        out_path: str 输出路径
        yaw_data: dict yaw 旋转数据
        landing_z: int 放置 Z 层
    """
    vs = float(vp["voxel_size"])
    corners_obj = _get_bbox_corners(bbox3d)
    orig_world = transform_points(corners_obj, T_obj2world)
    orig_ctr = orig_world.mean(axis=0)
    n_reps = len(cluster_reps)
    img_h, img_w = rgb.shape[:2]

    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor(CLR_BG)

    # ── 左面板: RGB 图像 ──────────────────────────────────────────────
    ax_rgb = fig.add_axes([0.02, 0.06, 0.47, 0.88])
    ax_rgb.imshow(rgb)

    _draw_bbox_2d(ax_rgb, orig_world, K, E_w2c,
                  color=CLR_ORIG, lw=2.5,
                  label=f"Original: {obj_name}")

    for k, rep in enumerate(cluster_reps):
        yaw_idx = int(rep[2])
        T_placed = _compute_placed_transform_vis(
            T_obj2world, bbox3d, rep[:2], landing_z, yaw_data, yaw_idx, vp)
        placed_world = transform_points(corners_obj, T_placed)
        lbl = f"Placement #{k} (of {n_reps})" if k == 0 else None
        _draw_bbox_2d(ax_rgb, placed_world, K, E_w2c,
                      color=CLR_PLACE, lw=2.0, label=lbl, alpha=0.85)

    ax_rgb.set_xlim(0, img_w)
    ax_rgb.set_ylim(img_h, 0)
    ax_rgb.axis("off")
    ax_rgb.legend(loc="upper left", fontsize=8,
                  facecolor=CLR_PANEL, labelcolor="white", framealpha=0.85)

    # ── 右面板: 3D 世界视图 ───────────────────────────────────────────
    ax3d = fig.add_axes([0.52, 0.06, 0.46, 0.88], projection="3d")
    ax3d.set_facecolor(CLR_PANEL)

    # 占据体素采样
    occ_idx = np.argwhere(grid == OCCUPIED)
    rng = np.random.default_rng(42)
    max_occ = 3000
    if len(occ_idx) > max_occ:
        occ_idx = occ_idx[rng.choice(len(occ_idx), max_occ, replace=False)]
    if len(occ_idx):
        occ_w = voxel_to_world(occ_idx, vp)
        ax3d.scatter(occ_w[:, 0], occ_w[:, 1], occ_w[:, 2],
                     c=CLR_OCC, s=1, alpha=0.15, label="Scene")

    # 原始 bbox
    _draw_bbox_3d(ax3d, orig_world, CLR_ORIG, lw=2.5,
                  label=f"Original: {obj_name}")

    # 放置 bbox + 箭头
    for k, rep in enumerate(cluster_reps):
        yaw_idx = int(rep[2])
        T_placed = _compute_placed_transform_vis(
            T_obj2world, bbox3d, rep[:2], landing_z, yaw_data, yaw_idx, vp)
        placed_world = transform_points(corners_obj, T_placed)
        lbl = f"Placement #{k}" if k == 0 else None
        _draw_bbox_3d(ax3d, placed_world, CLR_PLACE, lw=2.0,
                      label=lbl, alpha=0.85)

        p_ctr = placed_world.mean(axis=0)
        ax3d.plot([orig_ctr[0], p_ctr[0]],
                  [orig_ctr[1], p_ctr[1]],
                  [orig_ctr[2], p_ctr[2]],
                  color=CLR_ARROW, lw=1.2, ls="--", alpha=0.7)

    # 相机位置
    ax3d.scatter(*cam_origin, c=CLR_CAM, s=60, marker="^",
                 edgecolors="white", linewidths=0.5, label="Camera", zorder=5)

    ax3d.set_xlabel("X", color="white")
    ax3d.set_ylabel("Y", color="white")
    ax3d.set_zlabel("Z", color="white")
    ax3d.tick_params(colors="white")
    ax3d.legend(fontsize=8, loc="upper left",
                facecolor=CLR_PANEL, labelcolor="white", framealpha=0.85)

    # 标题
    fig.text(0.5, 0.97,
             f"Placement Planning — {obj_name}",
             ha="center", va="top", color="white",
             fontsize=13, fontweight="bold")

    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close()
