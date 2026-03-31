"""
src/annotation/quality_filter.py
-----------------------------------
职责：过滤低质量自动标注样本。

过滤规则：
- 置信度过低：Grounded-DINO 置信度 < min_confidence
- 点数过少：3D 点云子集点数 < min_points（目标太小或遮挡严重）
- 尺寸异常：bbox 长宽比 > max_aspect_ratio（拟合失败）
- 深度无效：bbox center 深度超出合理范围

输入：
    annotation: dict  # 单条标注（含 bbox3d 和 annotation_meta）

输出：
    is_valid: bool    # 是否通过过滤
    reason:   str     # 过滤原因（调试用）

用法：
    qf = QualityFilter(cfg)
    is_valid, reason = qf.check(annotation)
"""
