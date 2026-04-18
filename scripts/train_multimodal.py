#!/usr/bin/env python3
"""
scripts/train_multimodal.py
---------------------------
职责：训练多模态 3D BBox 回归模型，并在 valid split 上选择最优 checkpoint。

用法：
    conda run -n spatial python scripts/train_multimodal.py \
        --config configs/experiments/multimodal_train.yaml \
        --device cuda

作用：
    - 读取 YAML 训练配置与模型配置
    - 构建 train/valid DataLoader
    - 执行多轮训练、验证、进度显示、日志打印与 checkpoint 保存
    - 支持从已有 checkpoint 继续训练

输入：
    --config: 训练配置 YAML 路径
    --resume: 可选，覆盖配置中的 checkpoint 恢复路径
    --device: 可选，覆盖配置中的训练设备
    --output-dir: 可选，覆盖配置中的输出目录
    --disable-tqdm: 可选，关闭 tqdm 进度条，回退到普通日志输出

输出：
    output_dir/
        - train.log
        - train_config_resolved.yaml
        - model_config_snapshot.yaml
        - last.pth
        - best.pth

使用示例：
    conda run -n spatial python scripts/train_multimodal.py \
        --config configs/experiments/multimodal_train.yaml \
        --device cuda
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import PlacementMultimodalDataset, placement_multimodal_collate_fn
from src.losses import MultimodalBBoxLoss
from src.models import MultimodalModel


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser，配置完成的解析器
    """
    parser = argparse.ArgumentParser(description="训练多模态 3D BBox 回归模型")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/multimodal_train.yaml"),
        help="训练配置 YAML 路径",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="可选，覆盖配置中的 checkpoint 恢复路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="可选，覆盖配置中的训练设备，如 cpu 或 cuda",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="可选，覆盖配置中的输出目录",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="关闭 tqdm 进度条，强制使用普通日志输出",
    )
    return parser


def resolve_project_path(path_value: str | Path) -> Path:
    """
    用法: abs_path = resolve_project_path("configs/base/model.yaml")
    作用: 将相对仓库路径解析为绝对路径
    输入: path_value: str | Path，相对或绝对路径
    输出: Path，绝对路径
    """
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """
    用法: cfg = load_yaml_config(Path("configs/experiments/multimodal_train.yaml"))
    作用: 读取 YAML 配置文件
    输入: config_path: Path，配置文件路径
    输出: dict，配置内容
    """
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _to_yaml_serializable(value: Any) -> Any:
    """
    用法: payload = _to_yaml_serializable(config)
    作用: 将配置对象递归转换为可写入 YAML 的基础类型
    输入: value: Any，任意配置对象
    输出: Any，可被 YAML 安全序列化的对象
    """
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _to_yaml_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_yaml_serializable(item) for item in value]
    return value


def dump_yaml_config(payload: dict[str, Any], output_path: Path) -> None:
    """
    用法: dump_yaml_config(cfg, Path("outputs/run/train_config_resolved.yaml"))
    作用: 将配置字典写入 YAML 文件
    输入: payload: dict，配置内容；output_path: Path，输出路径
    输出: None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_payload = _to_yaml_serializable(payload)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            serializable_payload,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def initialize_log_file(log_path: Path) -> None:
    """
    用法: initialize_log_file(Path("outputs/run/train.log"))
    作用: 初始化训练日志文件，确保本次训练从空文件开始记录
    输入: log_path: Path，日志文件路径
    输出: None
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("")


def append_log_line(log_path: Path, message: str) -> None:
    """
    用法: append_log_line(Path("outputs/run/train.log"), "start training")
    作用: 将单行日志文本追加写入日志文件
    输入: log_path: Path，日志文件路径；message: str，日志内容
    输出: None
    """
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{message}\n")


def emit_message(message: str, log_path: Optional[Path] = None, console: bool = True) -> None:
    """
    用法: emit_message("开始训练", Path("outputs/run/train.log"), True)
    作用: 将日志同时输出到终端与可选的日志文件
    输入: message: str，日志内容；log_path: Path | None，日志文件路径；console: bool，是否输出到终端
    输出: None
    """
    if console:
        print(message)
    if log_path is not None:
        append_log_line(log_path, message)


def set_random_seed(seed: int) -> None:
    """
    用法: set_random_seed(42)
    作用: 设置 Python、NumPy 与 PyTorch 的随机种子
    输入: seed: int，随机种子
    输出: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str, log_path: Optional[Path] = None) -> torch.device:
    """
    用法: device = select_device("cuda", Path("outputs/run/train.log"))
    作用: 根据配置选择训练设备，并在 CUDA 不可用时自动回退到 CPU
    输入: device_name: str，期望设备名；log_path: Path | None，日志文件路径
    输出: torch.device，最终使用的设备
    """
    normalized_name = str(device_name).strip().lower()
    if normalized_name.startswith("cuda") and not torch.cuda.is_available():
        emit_message("检测到配置使用 CUDA，但当前环境不可用，自动回退到 CPU。", log_path=log_path)
        normalized_name = "cpu"
    return torch.device(normalized_name)


def build_dataset(dataset_cfg: dict[str, Any], split: str) -> PlacementMultimodalDataset:
    """
    用法: dataset = build_dataset(dataset_cfg, "train")
    作用: 根据配置构建指定 split 的多模态数据集
    输入: dataset_cfg: dict，数据集配置；split: str，train 或 valid
    输出: PlacementMultimodalDataset，数据集实例
    """
    image_size = tuple(int(v) for v in dataset_cfg.get("image_size", (480, 640)))
    return PlacementMultimodalDataset(
        annotation_dir=resolve_project_path(dataset_cfg.get("annotation_dir", "data/annotations/placement_multimodal")),
        split=split,
        prompt_key=str(dataset_cfg.get("prompt_key", "polished_prompt")),
        image_size=image_size,
        scale_eps=float(dataset_cfg.get("scale_eps", 1e-6)),
    )


def build_dataloader(
        dataset: PlacementMultimodalDataset,
        dataloader_cfg: dict[str, Any],
        split: str,
        device: torch.device) -> DataLoader:
    """
    用法: dataloader = build_dataloader(dataset, dataloader_cfg, "train")
    作用: 为指定 split 构建 DataLoader
    输入: dataset: PlacementMultimodalDataset；dataloader_cfg: dict；split: str；device: torch.device
    输出: DataLoader，可直接用于训练或验证
    """
    is_train = split == "train"
    batch_size_key = "train_batch_size" if is_train else "val_batch_size"
    batch_size = int(dataloader_cfg.get(batch_size_key, dataloader_cfg.get("batch_size", 1)))
    num_workers = int(dataloader_cfg.get("num_workers", 0))
    persistent_workers = bool(dataloader_cfg.get("persistent_workers", False)) and num_workers > 0
    pin_memory = bool(dataloader_cfg.get("pin_memory", True)) and device.type != "cpu"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(dataloader_cfg.get("train_shuffle", True)) if is_train else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=bool(dataloader_cfg.get("train_drop_last", False)) if is_train else False,
        persistent_workers=persistent_workers,
        collate_fn=placement_multimodal_collate_fn,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """
    用法: batch_on_device = move_batch_to_device(batch, device)
    作用: 将训练需要的 batch 张量搬运到目标设备
    输入: batch: dict，DataLoader 产出的批数据；device: torch.device，目标设备
    输出: dict，仅包含训练前向需要的字段
    """
    return {
        "images": batch["images"].to(device, non_blocking=True),
        "points_xyz": batch["points_xyz"].to(device, non_blocking=True),
        "text_inputs": batch["text_inputs"],
        "target_boxes_norm": batch["target_boxes_norm"].to(device, non_blocking=True),
    }


def create_optimizer(model: torch.nn.Module, optimization_cfg: dict[str, Any]) -> AdamW:
    """
    用法: optimizer = create_optimizer(model, optimization_cfg)
    作用: 根据配置创建 AdamW 优化器
    输入: model: nn.Module，训练模型；optimization_cfg: dict，优化器配置
    输出: AdamW，已初始化的优化器
    """
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("model has no trainable parameters")

    betas = optimization_cfg.get("betas", (0.9, 0.999))
    if len(betas) != 2:
        raise ValueError("optimization.betas must contain exactly 2 values")

    return AdamW(
        trainable_params,
        lr=float(optimization_cfg.get("lr", 1e-4)),
        betas=(float(betas[0]), float(betas[1])),
        weight_decay=float(optimization_cfg.get("weight_decay", 1e-4)),
    )


def create_scheduler(
        optimizer: AdamW,
        optimization_cfg: dict[str, Any],
        scheduler_cfg: dict[str, Any]) -> Optional[CosineAnnealingLR]:
    """
    用法: scheduler = create_scheduler(optimizer, optimization_cfg, scheduler_cfg)
    作用: 根据配置创建学习率调度器
    输入: optimizer: AdamW；optimization_cfg: dict；scheduler_cfg: dict
    输出: CosineAnnealingLR 或 None
    """
    scheduler_type = str(scheduler_cfg.get("type", "none")).strip().lower()
    if scheduler_type in {"", "none"}:
        return None
    if scheduler_type == "cosine":
        total_epochs = max(int(optimization_cfg.get("epochs", 1)), 1)
        return CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=float(scheduler_cfg.get("min_lr", 0.0)),
        )
    raise ValueError(f"unsupported scheduler type: {scheduler_type}")


def format_metrics(metrics: dict[str, float]) -> str:
    """
    用法: text = format_metrics({"loss": 0.1, "center_loss": 0.02})
    作用: 将标量指标字典格式化为便于日志打印的字符串
    输入: metrics: dict[str, float]，指标字典
    输出: str，格式化后的日志文本
    """
    ordered_keys = ("loss", "center_loss", "size_loss", "yaw_loss")
    chunks = []
    for key in ordered_keys:
        if key in metrics:
            chunks.append(f"{key}={metrics[key]:.6f}")
    for key, value in metrics.items():
        if key not in ordered_keys:
            chunks.append(f"{key}={value:.6f}")
    return ", ".join(chunks)


def format_duration(seconds: float) -> str:
    """
    用法: text = format_duration(125.3)
    作用: 将秒数格式化为便于日志展示的时长字符串
    输入: seconds: float，耗时秒数
    输出: str，格式化后的时长，例如 02:05.3
    """
    total_seconds = max(float(seconds), 0.0)
    minutes, remain_seconds = divmod(total_seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{remain_seconds:04.1f}"
    return f"{minutes:02d}:{remain_seconds:04.1f}"


def run_one_epoch(
        model: MultimodalModel,
        dataloader: DataLoader,
        criterion: MultimodalBBoxLoss,
        device: torch.device,
        epoch_idx: int,
        total_epochs: int,
        log_interval: int,
        enable_tqdm: bool,
        log_path: Optional[Path],
        optimizer: Optional[AdamW] = None,
        scaler: Optional[GradScaler] = None,
        grad_clip_norm: float = 0.0,
        use_amp: bool = False) -> dict[str, float]:
    """
    用法: metrics = run_one_epoch(model, dataloader, criterion, device, 1, 20, 10, True, Path("outputs/run/train.log"), optimizer)
    作用: 执行单个 epoch 的训练或验证循环，并返回平均指标
    输入: model: MultimodalModel；dataloader: DataLoader；criterion: MultimodalBBoxLoss；device: torch.device；
        epoch_idx: int；total_epochs: int；log_interval: int；enable_tqdm: bool；log_path: Path | None；
        optimizer: AdamW | None；scaler: GradScaler | None；
        grad_clip_norm: float；use_amp: bool
    输出: dict[str, float]，当前 epoch 的平均 loss 指标
    """
    is_train = optimizer is not None
    model.train(mode=is_train)
    phase_name = "Train" if is_train else "Valid"
    phase_name_cn = "训练" if is_train else "验证"

    metric_sums = {
        "loss": 0.0,
        "center_loss": 0.0,
        "size_loss": 0.0,
        "yaw_loss": 0.0,
    }
    sample_count = 0
    num_steps = len(dataloader)
    dataloader_batch_size = getattr(dataloader, "batch_size", None)
    emit_message(
        f"开始{phase_name_cn} epoch {epoch_idx}/{total_epochs}："
        f"steps={num_steps}，batch_size={dataloader_batch_size}",
        log_path=log_path,
    )

    progress_bar = None
    if enable_tqdm and tqdm is not None:
        progress_bar = tqdm(
            dataloader,
            total=num_steps,
            desc=f"{phase_name} {epoch_idx}/{total_epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        batch_iterator = enumerate(progress_bar, start=1)
    else:
        batch_iterator = enumerate(dataloader, start=1)

    for step_idx, batch in batch_iterator:
        batch_inputs = move_batch_to_device(batch, device)
        batch_size = int(batch_inputs["target_boxes_norm"].shape[0])

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = bool(use_amp) and device.type == "cuda"
        with torch.set_grad_enabled(is_train):
            with autocast(device_type=device.type, enabled=autocast_enabled):
                outputs = model(
                    points_xyz=batch_inputs["points_xyz"],
                    images=batch_inputs["images"],
                    text_inputs=batch_inputs["text_inputs"],
                )
                loss_dict = criterion(
                    pred_boxes_norm=outputs["pred_boxes_norm"],
                    target_boxes_norm=batch_inputs["target_boxes_norm"],
                )
                loss = loss_dict["loss"]

        if is_train:
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        for key in metric_sums:
            metric_sums[key] += float(loss_dict[key].detach().item()) * batch_size
        sample_count += batch_size

        if log_interval > 0 and (step_idx % log_interval == 0 or step_idx == num_steps):
            running_metrics = {
                key: value / max(sample_count, 1)
                for key, value in metric_sums.items()
            }
            if is_train:
                running_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
            step_message = (
                f"[{phase_name}] epoch {epoch_idx}/{total_epochs} "
                f"step {step_idx}/{num_steps} - {format_metrics(running_metrics)}"
            )
            if progress_bar is not None:
                progress_bar.set_postfix_str(format_metrics(running_metrics))
                emit_message(step_message, log_path=log_path, console=False)
            else:
                emit_message(step_message, log_path=log_path)

    if progress_bar is not None:
        progress_bar.close()

    if sample_count == 0:
        raise ValueError(f"{phase_name.lower()} dataloader produced zero samples")

    return {
        key: value / sample_count
        for key, value in metric_sums.items()
    }


def save_checkpoint(
        checkpoint_path: Path,
        model: MultimodalModel,
        optimizer: AdamW,
        scheduler: Optional[CosineAnnealingLR],
        scaler: Optional[GradScaler],
        epoch_idx: int,
        best_metric: Optional[float],
        best_metric_name: str,
        train_cfg: dict[str, Any],
        model_cfg: dict[str, Any]) -> None:
    """
    用法: save_checkpoint(Path("outputs/run/last.pth"), model, optimizer, scheduler, scaler, 1, 0.1, "val_loss", train_cfg, model_cfg)
    作用: 将当前训练状态保存为 checkpoint
    输入: checkpoint_path: Path；model: MultimodalModel；optimizer: AdamW；scheduler: CosineAnnealingLR | None；
        scaler: GradScaler | None；epoch_idx: int；best_metric: float | None；best_metric_name: str；
        train_cfg: dict；model_cfg: dict
    输出: None
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": int(epoch_idx),
        "best_metric": None if best_metric is None else float(best_metric),
        "best_metric_name": str(best_metric_name),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "train_config": copy.deepcopy(_to_yaml_serializable(train_cfg)),
        "model_config": copy.deepcopy(_to_yaml_serializable(model_cfg)),
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
        checkpoint_path: Path,
        model: MultimodalModel,
        optimizer: AdamW,
        scheduler: Optional[CosineAnnealingLR],
        scaler: Optional[GradScaler]) -> tuple[int, Optional[float], str]:
    """
    用法: start_epoch, best_metric, best_metric_name = load_checkpoint(path, model, optimizer, scheduler, scaler)
    作用: 从 checkpoint 恢复模型与优化器状态
    输入: checkpoint_path: Path；model: MultimodalModel；optimizer: AdamW；scheduler: CosineAnnealingLR | None；scaler: GradScaler | None
    输出: tuple，分别为下一轮 epoch 编号、当前 best metric 与 best metric 名称
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    last_epoch = int(checkpoint.get("epoch", 0))
    best_metric = checkpoint.get("best_metric")
    best_metric_name = str(checkpoint.get("best_metric_name", "val_loss"))
    return last_epoch + 1, best_metric, best_metric_name


def main() -> None:
    """
    用法: main()
    作用: 训练多模态模型并保存 checkpoint
    输入: 无，参数来自命令行
    输出: None
    """
    args = build_parser().parse_args()
    config_path = resolve_project_path(args.config)
    train_cfg = load_yaml_config(config_path)

    model_config_path = resolve_project_path(train_cfg.get("model_config_path", "configs/base/model.yaml"))
    model_cfg = load_yaml_config(model_config_path)
    decoder_cfg = dict(model_cfg.get("decoder", {}))
    if int(decoder_cfg.get("num_queries", 1)) != 1:
        raise ValueError("train_multimodal.py 当前仅支持 decoder.num_queries=1 的单 query 监督")

    runtime_cfg = dict(train_cfg.get("train", {}))
    if args.device is not None:
        runtime_cfg["device"] = args.device
    if args.output_dir is not None:
        runtime_cfg["output_dir"] = args.output_dir.as_posix()
    if args.disable_tqdm:
        runtime_cfg["disable_tqdm"] = True
    train_cfg["train"] = runtime_cfg

    checkpoint_cfg = dict(train_cfg.get("checkpoint", {}))
    if args.resume is not None:
        checkpoint_cfg["resume_path"] = args.resume.as_posix()
    train_cfg["checkpoint"] = checkpoint_cfg

    output_dir = resolve_project_path(runtime_cfg.get("output_dir", "outputs/multimodal_train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    initialize_log_file(log_path)
    dump_yaml_config(train_cfg, output_dir / "train_config_resolved.yaml")
    dump_yaml_config(model_cfg, output_dir / "model_config_snapshot.yaml")
    emit_message(f"训练日志将写入：{log_path}", log_path=log_path)

    seed = int(runtime_cfg.get("seed", 42))
    set_random_seed(seed)

    device = select_device(runtime_cfg.get("device", "cuda"), log_path=log_path)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))

    dataset_cfg = dict(train_cfg.get("dataset", {}))
    dataloader_cfg = dict(train_cfg.get("dataloader", {}))
    optimization_cfg = dict(train_cfg.get("optimization", {}))
    scheduler_cfg = dict(train_cfg.get("scheduler", {}))
    loss_cfg = dict(train_cfg.get("loss", {}))
    train_dataset = build_dataset(dataset_cfg, split="train")
    if len(train_dataset) == 0:
        raise ValueError("train split is empty, unable to start training")
    valid_dataset = build_dataset(dataset_cfg, split="valid")

    train_loader = build_dataloader(train_dataset, dataloader_cfg, split="train", device=device)
    valid_loader = None
    if len(valid_dataset) > 0:
        valid_loader = build_dataloader(valid_dataset, dataloader_cfg, split="valid", device=device)
    else:
        emit_message("valid split 为空，本次训练将使用 train loss 作为 best checkpoint 指标。", log_path=log_path)

    emit_message(
        f"数据集构建完成：train={len(train_dataset)}，"
        f"valid={len(valid_dataset)}，device={device.type}",
        log_path=log_path,
    )

    model = MultimodalModel(model_cfg).to(device)
    criterion = MultimodalBBoxLoss(
        center_weight=float(loss_cfg.get("center_weight", 1.0)),
        size_weight=float(loss_cfg.get("size_weight", 1.0)),
        yaw_weight=float(loss_cfg.get("yaw_weight", 0.5)),
        smooth_l1_beta=float(loss_cfg.get("smooth_l1_beta", 1.0)),
    )
    optimizer = create_optimizer(model, optimization_cfg)
    scheduler = create_scheduler(optimizer, optimization_cfg, scheduler_cfg)

    use_amp = bool(runtime_cfg.get("use_amp", False)) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    enable_tqdm = not bool(runtime_cfg.get("disable_tqdm", False))
    if enable_tqdm and tqdm is None:
        enable_tqdm = False
        emit_message("未检测到 tqdm，自动回退到普通日志输出。", log_path=log_path)
    if enable_tqdm and not sys.stderr.isatty():
        enable_tqdm = False
        emit_message("当前输出不是交互式终端，自动关闭 tqdm 进度条并使用普通日志输出。", log_path=log_path)

    start_epoch = 1
    best_metric = None
    best_metric_name = "val_loss" if valid_loader is not None else "train_loss"
    resume_path_value = checkpoint_cfg.get("resume_path")
    if resume_path_value:
        resume_path = resolve_project_path(resume_path_value)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        start_epoch, best_metric, best_metric_name = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        emit_message(
            f"已恢复训练状态：resume={resume_path}, start_epoch={start_epoch}, "
            f"best_{best_metric_name}={best_metric}",
            log_path=log_path,
        )

    total_epochs = int(optimization_cfg.get("epochs", 1))
    grad_clip_norm = float(optimization_cfg.get("grad_clip_norm", 0.0))
    log_interval = int(runtime_cfg.get("log_interval", 10))
    last_name = str(checkpoint_cfg.get("save_last_name", "last.pth"))
    best_name = str(checkpoint_cfg.get("save_best_name", "best.pth"))
    scheduler_type = str(scheduler_cfg.get("type", "none")).strip().lower() or "none"

    emit_message(
        f"训练配置：config={config_path}，model_config={model_config_path}，output_dir={output_dir}",
        log_path=log_path,
    )
    emit_message(
        "运行参数："
        f"device={device.type}，use_amp={use_amp}，seed={seed}，"
        f"log_interval={log_interval}，tqdm={'on' if enable_tqdm else 'off'}",
        log_path=log_path,
    )
    emit_message(
        "DataLoader："
        f"train_samples={len(train_dataset)}，valid_samples={len(valid_dataset)}，"
        f"train_batches={len(train_loader)}，valid_batches={0 if valid_loader is None else len(valid_loader)}，"
        f"train_batch_size={train_loader.batch_size}，"
        f"valid_batch_size={0 if valid_loader is None else valid_loader.batch_size}，"
        f"num_workers={int(dataloader_cfg.get('num_workers', 0))}",
        log_path=log_path,
    )
    emit_message(
        "优化配置："
        f"epochs={total_epochs}，lr={optimizer.param_groups[0]['lr']:.6g}，"
        f"weight_decay={float(optimization_cfg.get('weight_decay', 1e-4)):.6g}，"
        f"grad_clip_norm={grad_clip_norm:.6g}，scheduler={scheduler_type}",
        log_path=log_path,
    )

    for epoch_idx in range(start_epoch, total_epochs + 1):
        epoch_start_time = time.perf_counter()
        train_start_time = time.perf_counter()
        train_metrics = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            epoch_idx=epoch_idx,
            total_epochs=total_epochs,
            log_interval=log_interval,
            enable_tqdm=enable_tqdm,
            log_path=log_path,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
            use_amp=use_amp,
        )
        train_elapsed = time.perf_counter() - train_start_time

        valid_metrics = None
        valid_elapsed = 0.0
        if valid_loader is not None:
            valid_start_time = time.perf_counter()
            with torch.no_grad():
                valid_metrics = run_one_epoch(
                    model=model,
                    dataloader=valid_loader,
                    criterion=criterion,
                    device=device,
                    epoch_idx=epoch_idx,
                    total_epochs=total_epochs,
                    log_interval=log_interval,
                    enable_tqdm=enable_tqdm,
                    log_path=log_path,
                    optimizer=None,
                    scaler=None,
                    grad_clip_norm=0.0,
                    use_amp=use_amp,
                )
            valid_elapsed = time.perf_counter() - valid_start_time

        if scheduler is not None:
            scheduler.step()

        current_metric = train_metrics["loss"] if valid_metrics is None else valid_metrics["loss"]
        is_best = best_metric is None or current_metric < best_metric
        if is_best:
            best_metric = current_metric

        save_checkpoint(
            checkpoint_path=output_dir / last_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch_idx=epoch_idx,
            best_metric=best_metric,
            best_metric_name=best_metric_name,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
        )
        emit_message(f"已保存 last checkpoint：{output_dir / last_name}", log_path=log_path)
        if is_best:
            save_checkpoint(
                checkpoint_path=output_dir / best_name,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch_idx=epoch_idx,
                best_metric=best_metric,
                best_metric_name=best_metric_name,
                train_cfg=train_cfg,
                model_cfg=model_cfg,
            )
            emit_message(
                f"best checkpoint 已更新：{output_dir / best_name}，"
                f"{best_metric_name}={best_metric:.6f}",
                log_path=log_path,
            )

        epoch_elapsed = time.perf_counter() - epoch_start_time
        summary_chunks = [
            f"[Epoch {epoch_idx}/{total_epochs}]",
            f"train_time={format_duration(train_elapsed)}",
            f"train: {format_metrics(train_metrics)}",
        ]
        if valid_metrics is not None:
            summary_chunks.append(f"valid_time={format_duration(valid_elapsed)}")
            summary_chunks.append(f"valid: {format_metrics(valid_metrics)}")
        summary_chunks.append(f"epoch_time={format_duration(epoch_elapsed)}")
        summary_chunks.append(f"best_{best_metric_name}={best_metric:.6f}")
        emit_message(" | ".join(summary_chunks), log_path=log_path)


if __name__ == "__main__":
    main()

# 使用示例:
# conda run -n spatial python scripts/train_multimodal.py --config configs/experiments/multimodal_train.yaml
