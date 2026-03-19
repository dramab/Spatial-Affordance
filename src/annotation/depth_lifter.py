"""
src/annotation/depth_lifter.py
--------------------------------
职责：将 2D mask 反投影到 3D，获取对应的点云子集。

方法：
- 利用深度图和相机内参，将 mask 区域的像素反投影为 3D 点
- 或直接将点云投影到图像平面，筛选落在 mask 内的点

输入：
    depth:  ndarray(H, W)    # 深度图（单位：米）
    mask:   ndarray(H, W)    # 二值 mask
    K:      ndarray(3, 3)    # 相机内参矩阵
    xyz:    ndarray(N, 3)    # 原始点云（可选，优先使用）

输出：
    pc_subset: ndarray(M, 3)  # mask 对应的 3D 点云子集

用法：
    lifter = DepthLifter(cfg)
    pc_subset = lifter.lift(depth, mask, K)
"""
