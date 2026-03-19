"""
src/annotation/bbox_fitter.py
-------------------------------
职责：从点云簇拟合 3D bounding box。

方法：
- PCA 方法：对点云做 PCA，主方向作为 bbox 朝向，计算最小包围盒
- AABB 方法：轴对齐包围盒（忽略旋转，速度快）

输入：
    points: ndarray(N, 3)  # 点云子集（已过滤离群点）

输出：
    bbox3d: dict  # {"center": [cx,cy,cz], "size": [l,w,h], "yaw": float}

用法：
    fitter = BBoxFitter(method="pca")
    bbox3d = fitter.fit(points)
"""
