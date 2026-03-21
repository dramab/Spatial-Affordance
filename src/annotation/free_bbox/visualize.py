"""
src/annotation/free_bbox/visualize.py
--------------------------------------
放置规划可视化：放置结果双面板可视化。

用法:
    from src.annotation.free_bbox.visualize import save_placement_vis
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.annotation.free_bbox.occupancy import FREE, OCCUPIED
from src.annotation.free_bbox.grid_ops import _get_bbox_corners
from src.annotation.free_bbox.voxel_utils import voxel_to_world, world_to_voxel
from src.utils.coord_utils import transform_points, project_world

CLR_BG = "#1C1D31"
CLR_PANEL = "#101223"
CLR_TEXT = "#F3F5FF"
CLR_MUTED = "#CCD2E6"
CLR_GRID = "#FFFFFF"
CLR_ORIG = "#FF8A1C"
CLR_PLACE = "#2DE38A"
CLR_CAM = "#FFE23F"
CLR_ARROW = "#FFD65A"
CLR_OCC = "#9FB2CC"
CLR_SURFACE = "#D7E2EE"
CLR_FREE = "#2F9B63"

BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


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


def _style_3d_axes(ax):
    """统一 3D 右面板的暗色主题样式。"""
    ax.set_facecolor(CLR_PANEL)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(mcolors.to_rgba(CLR_PANEL, 1.0))
        axis.pane.set_edgecolor(mcolors.to_rgba("#56607F", 0.9))
        axis._axinfo["grid"]["color"] = mcolors.to_rgba(CLR_GRID, 0.55)
        axis._axinfo["grid"]["linewidth"] = 1.0
        axis._axinfo["grid"]["linestyle"] = "-"
        try:
            axis.line.set_color(mcolors.to_rgba(CLR_MUTED, 0.7))
        except Exception:
            pass
    ax.tick_params(colors=CLR_TEXT, labelsize=10, pad=2)
    ax.grid(True)


def _surface_mask_to_world(surface_mask, support_z, vp):
    """将支撑面 mask 转为世界坐标点。"""
    if surface_mask is None:
        return np.empty((0, 3), dtype=np.float64)

    support_idx = np.argwhere(surface_mask)
    if len(support_idx) == 0:
        return np.empty((0, 3), dtype=np.float64)

    support_idx_3d = np.column_stack([
        support_idx,
        np.full(len(support_idx), int(support_z), dtype=np.intp),
    ])
    return voxel_to_world(support_idx_3d, vp)


def _compute_view_limits(support_world, focus_points, voxel_size):
    """让支撑面位于底部，并由支撑面尺寸主导右面板尺度。"""
    if len(support_world):
        support_min = support_world.min(axis=0)
        support_max = support_world.max(axis=0)
        support_span_xy = np.maximum(
            support_max[:2] - support_min[:2],
            float(voxel_size) * 6.0,
        )
        xy_pad = np.maximum(support_span_xy * 0.18, float(voxel_size) * 2.0)
        z_bottom = float(support_min[2] - 0.75 * voxel_size)
        z_height = max(float(support_span_xy.max()) * 0.95, float(voxel_size) * 10.0)

        view_min = np.array([
            support_min[0] - xy_pad[0],
            support_min[1] - xy_pad[1],
            z_bottom,
        ], dtype=np.float64)
        view_max = np.array([
            support_max[0] + xy_pad[0],
            support_max[1] + xy_pad[1],
            z_bottom + z_height,
        ], dtype=np.float64)

        if focus_points:
            focus = np.concatenate(focus_points, axis=0)
            focus_min = focus.min(axis=0)
            focus_max = focus.max(axis=0)
            focus_pad_xy = np.maximum(support_span_xy * 0.06, float(voxel_size) * 1.5)
            view_min[:2] = np.minimum(view_min[:2], focus_min[:2] - focus_pad_xy)
            view_max[:2] = np.maximum(view_max[:2], focus_max[:2] + focus_pad_xy)
            view_max[2] = max(view_max[2], float(focus_max[2] + 2.5 * voxel_size))
        return view_min, view_max

    focus = np.concatenate(focus_points, axis=0)
    focus_min = focus.min(axis=0)
    focus_max = focus.max(axis=0)
    span = np.maximum(focus_max - focus_min, float(voxel_size) * 6.0)
    pad = np.maximum(span * np.array([0.14, 0.14, 0.18]), float(voxel_size) * 2.0)
    return focus_min - pad, focus_max + pad


def _filter_voxel_indices_to_view(indices, view_min, view_max, vp):
    """只保留落在核心可视化窗口内的体素。"""
    if len(indices) == 0:
        return indices

    idx_min = world_to_voxel(np.asarray(view_min, dtype=np.float64)[None, :], vp)[0] - 1
    idx_max = world_to_voxel(np.asarray(view_max, dtype=np.float64)[None, :], vp)[0] + 1
    mask = ((indices[:, 0] >= idx_min[0]) & (indices[:, 0] <= idx_max[0]) &
            (indices[:, 1] >= idx_min[1]) & (indices[:, 1] <= idx_max[1]) &
            (indices[:, 2] >= idx_min[2]) & (indices[:, 2] <= idx_max[2]))
    return indices[mask]


def _compute_placed_transform_vis(T_obj2world, bbox3d, anchor_xy,
                                  landing_z, yaw_data, yaw_idx, vp):
    """计算放置物体的 4x4 变换矩阵（可视化用）。"""
    del T_obj2world, bbox3d
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
                       cluster_reps, grid, out_path, yaw_data, landing_z,
                       surface_mask=None):
    """
    双面板暗色主题放置可视化。

    左面板: RGB + 原始 bbox（橙色）+ 放置 bbox（绿色）
    右面板: 支撑面位于底部、尺度由支撑面主导的核心 3D 视图

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
        grid: (Gx, Gy, Gz) uint8 栅格
        out_path: str 输出路径
        yaw_data: dict yaw 旋转数据
        landing_z: int 放置 Z 层
        surface_mask: (Gx, Gy) bool 支撑面掩码
    """
    vs = float(vp["voxel_size"])
    support_z = landing_z - 1
    corners_obj = _get_bbox_corners(bbox3d)
    orig_world = transform_points(corners_obj, T_obj2world)
    orig_ctr = orig_world.mean(axis=0)
    n_reps = len(cluster_reps)
    img_h, img_w = rgb.shape[:2]
    rng = np.random.default_rng(42)

    placed_world_list = []
    for rep in cluster_reps:
        yaw_idx = int(rep[2])
        T_placed = _compute_placed_transform_vis(
            T_obj2world, bbox3d, rep[:2], landing_z, yaw_data, yaw_idx, vp)
        placed_world_list.append(transform_points(corners_obj, T_placed))

    support_world = _surface_mask_to_world(surface_mask, support_z, vp)
    focus_points = [orig_world, *placed_world_list]
    view_min, view_max = _compute_view_limits(support_world, focus_points, vs)
    view_span = np.maximum(view_max - view_min, vs)

    occ_idx = _filter_voxel_indices_to_view(
        np.argwhere(grid == OCCUPIED), view_min, view_max, vp)
    max_occ = 2600
    if len(occ_idx) > max_occ:
        occ_idx = occ_idx[rng.choice(len(occ_idx), max_occ, replace=False)]
    occ_world_vis = voxel_to_world(occ_idx, vp) if len(occ_idx) else np.empty((0, 3), dtype=np.float64)

    free_idx = _filter_voxel_indices_to_view(
        np.argwhere(grid == FREE), view_min, view_max, vp)
    max_free = 2800
    if len(free_idx) > max_free:
        free_idx = free_idx[rng.choice(len(free_idx), max_free, replace=False)]
    free_world_vis = voxel_to_world(free_idx, vp) if len(free_idx) else np.empty((0, 3), dtype=np.float64)

    max_support = 3600
    if len(support_world) > max_support:
        support_world = support_world[rng.choice(len(support_world), max_support, replace=False)]

    fig = plt.figure(figsize=(18, 7.8))
    fig.patch.set_facecolor(CLR_BG)

    ax_rgb = fig.add_axes([0.02, 0.05, 0.45, 0.89])
    ax_rgb.imshow(rgb)

    _draw_bbox_2d(ax_rgb, orig_world, K, E_w2c,
                  color=CLR_ORIG, lw=2.8,
                  label=f"Original: {obj_name}")

    for k, placed_world in enumerate(placed_world_list):
        lbl = f"Placement #{k} (of {n_reps})" if k == 0 else None
        _draw_bbox_2d(ax_rgb, placed_world, K, E_w2c,
                      color=CLR_PLACE, lw=2.4, label=lbl, alpha=0.92)

    ax_rgb.set_xlim(0, img_w)
    ax_rgb.set_ylim(img_h, 0)
    ax_rgb.axis("off")
    ax_rgb.legend(loc="upper left", fontsize=8,
                  facecolor=CLR_PANEL, labelcolor=CLR_TEXT,
                  edgecolor=mcolors.to_rgba(CLR_GRID, 0.45), framealpha=0.92)

    ax3d = fig.add_axes([0.50, 0.05, 0.48, 0.89], projection="3d")
    _style_3d_axes(ax3d)

    if len(support_world):
        ax3d.scatter(support_world[:, 0], support_world[:, 1], support_world[:, 2],
                     c=CLR_SURFACE, s=13, alpha=0.42, marker="o", linewidths=0.0,
                     depthshade=False)

    if len(free_world_vis):
        ax3d.scatter(free_world_vis[:, 0], free_world_vis[:, 1], free_world_vis[:, 2],
                     c=CLR_FREE, s=7, alpha=0.18, marker="o", linewidths=0.0,
                     depthshade=False)

    support_world_z = voxel_to_world(np.array([[0, 0, support_z]], dtype=np.intp), vp)[0, 2]
    if len(occ_world_vis):
        upper_occ_mask = occ_world_vis[:, 2] > (support_world_z + 1.25 * vs)
        if np.any(upper_occ_mask):
            pts = occ_world_vis[upper_occ_mask]
            ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                         c=CLR_OCC, s=9, alpha=0.30, marker="o", linewidths=0.0,
                         depthshade=False)

    _draw_bbox_3d(ax3d, orig_world, CLR_ORIG, lw=3.0,
                  label=f"Original: {obj_name}")

    ax3d.plot([], [], [], color=CLR_ARROW, lw=2.2, label="displacement")
    for k, placed_world in enumerate(placed_world_list):
        lbl = f"Placement #{k}" if k == 0 else None
        _draw_bbox_3d(ax3d, placed_world, CLR_PLACE, lw=3.0,
                      label=lbl, alpha=0.96)

        p_ctr = placed_world.mean(axis=0)
        disp = p_ctr - orig_ctr
        ax3d.quiver(orig_ctr[0], orig_ctr[1], orig_ctr[2],
                    disp[0], disp[1], disp[2],
                    color=CLR_ARROW, linewidth=2.0,
                    arrow_length_ratio=0.16, alpha=0.95)

    cam_margin = np.array([vs * 2.0, vs * 2.0, vs * 2.0], dtype=np.float64)
    cam_in_core = np.all(cam_origin >= (view_min - cam_margin)) and np.all(cam_origin <= (view_max + cam_margin))
    if cam_in_core:
        ax3d.scatter(*cam_origin, c=CLR_CAM, s=260, marker="*",
                     edgecolors=mcolors.to_rgba("#FFF6A3", 0.95), linewidths=1.0,
                     label="Camera", zorder=6, depthshade=False)

    ax3d.set_xlim(view_min[0], view_max[0])
    ax3d.set_ylim(view_min[1], view_max[1])
    ax3d.set_zlim(view_min[2], view_max[2])
    ax3d.set_box_aspect(tuple(view_span.tolist()))
    ax3d.set_xlabel("X (cm)", color=CLR_TEXT, labelpad=10)
    ax3d.set_ylabel("Y (cm)", color=CLR_TEXT, labelpad=12)
    ax3d.set_zlabel("Z (cm)", color=CLR_TEXT, labelpad=8)
    ax3d.view_init(elev=26, azim=-60)

    handles, labels = ax3d.get_legend_handles_labels()
    desired = ["Camera", f"Original: {obj_name}", "Placement #0", "displacement"]
    ordered_handles = []
    ordered_labels = []
    for name in desired:
        if name in labels:
            idx = labels.index(name)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])
    ax3d.legend(ordered_handles, ordered_labels,
                fontsize=8, loc="upper left",
                facecolor=CLR_PANEL, labelcolor=CLR_TEXT,
                edgecolor=mcolors.to_rgba(CLR_GRID, 0.45), framealpha=0.94)

    fig.text(0.58, 0.965,
             f"Placement Planning ({obj_name})",
             ha="center", va="top", color=CLR_TEXT,
             fontsize=13, fontweight="bold")

    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close()
