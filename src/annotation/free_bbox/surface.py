"""
src/annotation/free_bbox/surface.py
------------------------------------
支撑面检测：通过形态学开运算 + 连通分量标记，
在占据栅格的每个 Z 层中找到最大水平支撑面。

用法:
    from src.annotation.free_bbox.surface import detect_support_surfaces
"""

import numpy as np
from scipy.ndimage import binary_opening, label

from src.annotation.free_bbox.occupancy import OCCUPIED


def detect_support_surfaces(grid, vp, min_area=50.0):
    """
    在占据栅格中检测最大水平支撑面。

    算法:
        1. 对每个 Z 层的 OCCUPIED 切片做形态学开运算（去噪）
        2. 连通分量标记
        3. 选择面积最大的连通分量作为支撑面

    输入:
        grid: (Gx, Gy, Gz) uint8 占据栅格
        vp: dict 体素参数 {"voxel_size": float, "origin": [x,y,z]}
        min_area: float 最小支撑面面积（场景单位²）
    输出:
        table_z: int 支撑面所在的 Z 层索引，未找到则为 None
        surface_mask_2d: (Gx, Gy) bool 支撑面的 2D 掩码，未找到则为 None
    """
    vs = float(vp["voxel_size"])
    min_voxels = max(1, int(min_area / (vs * vs)))

    occ = (grid == OCCUPIED)
    struct_2d = np.ones((3, 3), dtype=bool)

    # 仅扫描有足够 OCCUPIED 体素的 Z 层
    z_counts = occ.sum(axis=(0, 1))
    active_zs = np.where(z_counts >= min_voxels)[0]

    best_z, best_area, best_mask = None, 0, None

    for z in active_zs:
        slice_2d = occ[:, :, z]
        opened = binary_opening(slice_2d, structure=struct_2d)
        labeled, n_features = label(opened)
        for comp_id in range(1, n_features + 1):
            component = (labeled == comp_id)
            area = int(component.sum())
            if area >= min_voxels and area > best_area:
                best_z = int(z)
                best_area = area
                best_mask = component

    if best_z is None:
        return None, None
    return best_z, best_mask


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
