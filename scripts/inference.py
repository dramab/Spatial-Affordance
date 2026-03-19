"""
scripts/inference.py
---------------------
职责：单样本推理脚本，输入 RGB + 点云 + 文字，输出 3D bbox。

功能：
- 加载模型和 checkpoint
- 对单个样本进行预处理（图像归一化、点云采样）
- 运行模型推理，输出 7-DoF 3D bbox
- 可选：调用 vis_utils 可视化结果

用法：
    python scripts/inference.py \\
        --checkpoint outputs/checkpoints/best.pth \\
        --image data/sample/image.jpg \\
        --pointcloud data/sample/scene.pcd \\
        --text "the brown wooden chair near the window" \\
        --visualize
"""
