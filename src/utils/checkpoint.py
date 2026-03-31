"""
src/utils/checkpoint.py
------------------------
职责：模型 checkpoint 保存与加载工具。

功能：
- save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_dir)：
    保存完整训练状态（支持 top-k 保留策略）
- load_checkpoint(model, ckpt_path, optimizer=None, scheduler=None)：
    加载 checkpoint，支持仅加载模型权重（推理模式）
- find_best_checkpoint(save_dir, monitor="val/acc@0.5")：
    从 save_dir 中找到最优 checkpoint 路径

用法：
    save_checkpoint(model, optimizer, scheduler, epoch=10,
                    metrics={"val/acc@0.5": 0.72}, save_dir="outputs/ckpts")
    epoch = load_checkpoint(model, "outputs/ckpts/best.pth", optimizer)
"""
