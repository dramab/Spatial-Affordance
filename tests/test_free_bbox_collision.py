"""
tests/test_free_bbox_collision.py
---------------------------------
职责：测试 free_bbox 放置搜索中的姿态保留逻辑。

测试内容：
- test_find_table_placements_preserves_axis_aligned_middle_axis_pose：
  验证当中间尺度轴竖直时不会回退成平放姿态

用法：
    pytest tests/test_free_bbox_collision.py -v
"""

import numpy as np

from src.annotation.free_bbox.collision import find_table_placements
from src.annotation.free_bbox.occupancy import FREE


def _make_voxel_params(voxel_size: float = 1.0) -> dict:
    """
    构造仅用于测试的体素参数。

    输入:
        voxel_size: float 体素大小
    输出:
        dict 包含 grid_min 与 voxel_size
    """
    return {
        "origin": np.array([0.0, 0.0, 0.0], dtype=np.float64).tolist(),
        "voxel_size": float(voxel_size),
    }


def test_find_table_placements_preserves_axis_aligned_middle_axis_pose():
    """
    验证当中间尺度轴竖直时，放置搜索会保留原始姿态而不是回退成平放。

    输入:
        无，内部使用 cup-green_actys 的原始姿态与 canonical AABB
    输出:
        无，通过断言验证 yaw_data 中的旋转矩阵
    """
    grid = np.full((80, 80, 20), FREE, dtype=np.uint8)
    surface_mask = np.ones((80, 80), dtype=bool)
    vp = _make_voxel_params(voxel_size=1.0)

    bbox = np.array([
        -5.507149919867516, -4.461599886417389, -4.048449918627739,
        5.507149919867516, 4.461599886417389, 4.048449918627739,
    ], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array([
        [-0.36858265550763186, -0.01955667608560278, 0.9293892416417485],
        [0.9291522909979096, 0.02310052084795846, 0.36897477701041403],
        [-0.028685295750162727, 0.9995418462260788, 0.00965667907149146],
    ], dtype=np.float64)
    pose[:3, 3] = np.array([18.266313950916015, -6.0077088740087055, 0.2267012762645777])

    _, _, yaw_data = find_table_placements(
        grid, bbox, pose, vp,
        table_z=0,
        surface_mask_2d=surface_mask,
        safety_margin=0.0,
        yaw_steps=1,
        preserve_orientation=True,
        orientation_threshold_deg=5.0,
    )

    pose_info = yaw_data["pose_info"]
    assert pose_info is not None
    assert bool(pose_info["is_axis_aligned"]) is True
    assert bool(pose_info["is_reasonable"]) is True

    preserved_rot = yaw_data["T_rotated"][0][:3, :3]
    assert np.allclose(preserved_rot, np.eye(3), atol=1e-3) is False
    assert abs(float(preserved_rot[2, 1])) > 0.99
