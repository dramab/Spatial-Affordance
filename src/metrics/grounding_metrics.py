"""
src/metrics/grounding_metrics.py
----------------------------------
职责：3D Visual Grounding 评估指标计算。

指标：
- Acc@0.25：预测框与 GT 的 3D IoU >= 0.25 的样本比例
- Acc@0.5：预测框与 GT 的 3D IoU >= 0.5 的样本比例
- mIoU3D：所有样本 3D IoU 的均值

功能：
- 支持 batch 级别累积，最终调用 compute() 返回汇总结果
- 依赖 box_utils.iou3d 计算 3D IoU

用法：
    metrics = GroundingMetrics()
    metrics.update(pred_boxes, gt_boxes)   # 每个 batch 调用
    results = metrics.compute()            # epoch 结束后调用
    metrics.reset()                        # 重置状态
"""
