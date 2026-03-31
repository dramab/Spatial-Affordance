"""
tools/visualize_dataset.py
---------------------------
职责：数据集样本可视化工具，用于检查标注质量。

功能：
- 随机抽取或指定 sample_id 加载样本
- 在 RGB 图像上叠加文字 prompt
- 用 Open3D 可视化点云 + GT 3D bbox
- 支持批量导出可视化图片（用于数据集审查）

用法：
    # 随机可视化 10 个样本
    python tools/visualize_dataset.py \\
        --ann_file data/annotations/train.json \\
        --num_samples 10

    # 可视化指定样本
    python tools/visualize_dataset.py \\
        --ann_file data/annotations/train.json \\
        --sample_id scene0001_00_0042

    # 批量导出
    python tools/visualize_dataset.py \\
        --ann_file data/annotations/train.json \\
        --export_dir outputs/vis_dataset
"""
