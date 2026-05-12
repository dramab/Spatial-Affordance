"""
src/metrics
-----------
评测指标模块入口。
"""

from src.metrics.placement_eval import (
    DEFAULT_COLLISION_RATIO_THRESHOLD,
    DEFAULT_DIRECTION_CENTER_L2_THRESHOLD_CM,
    DEFAULT_VOLUME_ERROR_RATIO_THRESHOLD,
    box7d_to_occupancy_voxels,
    evaluate_collision,
    evaluate_direction,
    evaluate_projected_object_center,
    evaluate_size_consistency,
    merge_sample_metric_status,
    object_info_to_corners_world,
    summarize_by_source,
    summarize_metric_records,
)

__all__ = [
    "DEFAULT_COLLISION_RATIO_THRESHOLD",
    "DEFAULT_DIRECTION_CENTER_L2_THRESHOLD_CM",
    "DEFAULT_VOLUME_ERROR_RATIO_THRESHOLD",
    "box7d_to_occupancy_voxels",
    "evaluate_collision",
    "evaluate_direction",
    "evaluate_projected_object_center",
    "evaluate_size_consistency",
    "merge_sample_metric_status",
    "object_info_to_corners_world",
    "summarize_by_source",
    "summarize_metric_records",
]
