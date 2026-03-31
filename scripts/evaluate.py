"""
scripts/evaluate.py
--------------------
职责：在测试集上评估训练好的模型，输出完整评估报告。

功能：
- 加载指定 checkpoint 的模型权重
- 在 test split 上运行推理
- 计算 Acc@0.25、Acc@0.5、mIoU3D 等指标
- 支持按类别分组统计（如按物体类型）
- 将结果保存为 JSON 报告

用法：
    python scripts/evaluate.py \\
        --config configs/experiments/baseline.yaml \\
        --checkpoint outputs/checkpoints/best.pth \\
        --split test \\
        --output outputs/eval_results.json
"""
