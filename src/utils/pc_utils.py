"""
src/utils/pc_utils.py
----------------------
职责：点云处理工具库。

功能：
- farthest_point_sample(xyz, n_points)：最远点采样（FPS）
- voxel_downsample(pcd, voxel_size)：体素下采样（调用 Open3D）
- normalize_pc(xyz)：将点云归一化到单位球内
- project_to_image(xyz, K, T)：点云投影到图像平面，返回像素坐标
- depth_to_pointcloud(depth, K)：深度图反投影为点云
- crop_pc_by_mask(xyz, mask_2d, K, T)：用 2D mask 裁剪对应的 3D 点

输入/输出：
    xyz: Tensor(N, 3) 或 ndarray(N, 3)

用法：
    from src.utils.pc_utils import farthest_point_sample
    sampled_xyz = farthest_point_sample(xyz, 1024)
"""
