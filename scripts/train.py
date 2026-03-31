"""
scripts/train.py
-----------------
职责：训练入口脚本，支持 Hydra 配置系统和断点续训。

功能：
- 使用 @hydra.main 装饰器加载配置（支持命令行覆盖）
- 初始化模型、数据集、优化器、scheduler、损失函数
- 训练循环：forward -> loss -> backward -> optimizer step
- 每 epoch 结束后在验证集上评估，记录指标
- 支持 AMP 混合精度训练
- 支持 DDP 多卡训练（通过 torchrun 启动）
- 支持 --resume 从 checkpoint 恢复训练

用法：
    # 单卡训练
    python scripts/train.py experiment=baseline

    # 覆盖参数
    python scripts/train.py experiment=baseline batch_size=4 optimizer.lr=1e-4

    # 断点续训
    python scripts/train.py experiment=baseline +resume=outputs/checkpoints/last.pth

    # 多卡训练
    torchrun --nproc_per_node=4 scripts/train.py experiment=baseline
"""
