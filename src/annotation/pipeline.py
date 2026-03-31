"""
src/annotation/pipeline.py
----------------------------
职责：自动标注主流程编排器。

流程：
    原始数据（RGB + 深度图 + 点云 + 文字描述）
        -> GroundedSAM（2D mask）
        -> DepthLifter（3D 点云子集）
        -> BBoxFitter（3D bbox 拟合）
        -> QualityFilter（过滤低质量标注）
        -> 输出 JSON 标注文件

功能：
- 支持多进程并行处理（num_workers）
- 支持断点续标（跳过已处理样本）
- 记录处理统计信息（成功/失败/过滤数量）

用法：
    pipeline = AnnotationPipeline(cfg)
    pipeline.run(input_dir="data/raw", output_file="data/annotations/train.json")
"""
