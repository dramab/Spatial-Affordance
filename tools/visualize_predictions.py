"""
tools/visualize_predictions.py
--------------------------------
职责：模型预测结果可视化工具，对比预测框与 GT 框。

功能：
- 加载模型 checkpoint 和测试集标注
- 对每个样本运行推理，获取预测 3D bbox
- 用 Open3D 可视化点云 + 预测框（红色）+ GT 框（绿色）
- 在图像上显示文字 prompt 和 3D IoU 值
- 支持按 IoU 阈值筛选（只显示失败案例）

用法：
    python tools/visualize_predictions.py \\
        --checkpoint outputs/checkpoints/best.pth \\
        --ann_file data/annotations/val.json \\
        --num_samples 20

    # 只显示 IoU < 0.25 的失败案例
    python tools/visualize_predictions.py \\
        --checkpoint outputs/checkpoints/best.pth \\
        --ann_file data/annotations/val.json \\
        --iou_threshold 0.25 \\
        --show_failures_only
"""
