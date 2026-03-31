"""
src/utils/box_utils.py
-----------------------
职责：3D bounding box 几何操作工具库。

功能：
- iou3d(boxes_a, boxes_b)：计算两组 3D bbox 的 IoU（支持旋转框）
- giou3d(boxes_a, boxes_b)：计算 3D GIoU
- nms3d(boxes, scores, iou_thresh)：3D NMS
- box_center_to_corners(boxes)：(cx,cy,cz,l,w,h,yaw) -> 8个角点坐标
- corners_to_box_center(corners)：8角点 -> (cx,cy,cz,l,w,h,yaw)
- lidar_to_camera(boxes, calib)：LiDAR 坐标系 -> 相机坐标系转换

输入/输出：
    boxes: Tensor(N, 7)  # cx, cy, cz, l, w, h, yaw

用法：
    from src.utils.box_utils import iou3d
    ious = iou3d(pred_boxes, gt_boxes)  # Tensor(N,)
"""
