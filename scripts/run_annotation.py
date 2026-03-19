"""
scripts/run_annotation.py
--------------------------
职责：自动标注流程入口脚本。

功能：
- 加载标注配置（configs/annotation/auto_label.yaml）
- 初始化 AnnotationPipeline
- 遍历原始数据目录，批量生成标注 JSON
- 输出处理统计报告（成功/失败/过滤数量）

用法：
    python scripts/run_annotation.py \\
        --config configs/annotation/auto_label.yaml \\
        --input_dir data/raw \\
        --output_file data/annotations/train.json \\
        --num_workers 4

    # 覆盖配置参数
    python scripts/run_annotation.py \\
        --config configs/annotation/auto_label.yaml \\
        grounded_dino.box_threshold=0.4 \\
        quality_filter.min_points=200
"""
