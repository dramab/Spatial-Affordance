"""
src/utils/vis_utils.py
-----------------------
职责：基于 Open3D 的 3D 可视化工具。

功能：
- visualize_pc_with_bbox(xyz, bbox3d, colors=None)：
    可视化点云 + 3D bbox（线框）
- visualize_predictions(xyz, pred_box, gt_box)：
    对比可视化预测框（红色）和 GT 框（绿色）
- draw_bbox_on_image(image, bbox2d, label)：
    在 RGB 图像上绘制 2D bbox 和文字标签
- save_visualization(vis, save_path)：
    将 Open3D 可视化结果保存为图片

用法：
    from src.utils.vis_utils import visualize_pc_with_bbox
    visualize_pc_with_bbox(xyz, bbox3d)
"""
