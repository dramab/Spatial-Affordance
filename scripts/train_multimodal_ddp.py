#!/usr/bin/env python3
"""
scripts/train_multimodal_ddp.py
-------------------------------
职责：使用单机多卡 DDP 训练多模态 3D BBox 回归模型，并在 valid split 上选择最优 checkpoint。

用法：
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    conda run -n spatial torchrun --nproc_per_node=4 scripts/train_multimodal_ddp.py \
        --config configs/experiments/multimodal_train.yaml

作用：
    - 使用 torchrun 初始化单机多卡分布式训练环境
    - 构建 train/valid DistributedSampler 与 DataLoader
    - 执行多轮训练、验证、进度显示、日志打印与 checkpoint 保存
    - 仅在 rank 0 写入日志与保存 checkpoint

输入：
    --config: 训练配置 YAML 路径
    --resume: 可选，覆盖配置中的 checkpoint 恢复路径
    --device: 可选，仅支持 cuda，用于覆盖配置中的设备字段
    --output-dir: 可选，覆盖配置中的输出目录
    --disable-tqdm: 可选，关闭 tqdm 进度条，回退到普通日志输出

输出：
    output_dir/
        - train.log
        - train_config_resolved.yaml
        - model_config_snapshot.yaml
        - last.pth
        - best.pth

说明：
    - train_batch_size / val_batch_size 表示单卡 batch size
    - 全局 batch size = 单卡 batch size x world_size

使用示例：
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    conda run -n spatial torchrun --nproc_per_node=4 scripts/train_multimodal_ddp.py \
        --config configs/experiments/multimodal_train.yaml \
        --output-dir outputs/multimodal_train_ddp
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import train_multimodal as single_train

from src.losses import MultimodalBBoxLoss
from src.models import MultimodalModel


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建单机多卡 DDP 训练脚本的命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser，配置完成的解析器
    """
    parser = argparse.ArgumentParser(description="使用单机多卡 DDP 训练多模态 3D BBox 回归模型")
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
        help="可选，仅支持 cuda，用于覆盖配置中的设备字段",
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


def get_distributed_context() -> tuple[int, int, int]:
    """
    用法: local_rank, rank, world_size = get_distributed_context()
    作用: 从 torchrun 注入的环境变量中读取单机多卡 DDP 上下文
    输入: 无
    输出: tuple[int, int, int]，分别为 local_rank、rank 与 world_size
    """
    required_keys = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    missing_keys = [key for key in required_keys if key not in os.environ]
    if missing_keys:
        raise EnvironmentError(
            "train_multimodal_ddp.py 需要使用 torchrun 启动，"
            f"缺少环境变量: {', '.join(missing_keys)}"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return local_rank, rank, world_size


def initialize_distributed(local_rank: int, world_size: int) -> torch.device:
    """
    用法: device = initialize_distributed(local_rank=0, world_size=4)
    作用: 初始化 NCCL 进程组并将当前进程绑定到对应 GPU
    输入: local_rank: int，本机 GPU 序号；world_size: int，总进程数
    输出: torch.device，当前进程使用的 CUDA 设备
    """
    if not torch.cuda.is_available():
        raise EnvironmentError("DDP 训练需要 CUDA 环境，但当前未检测到可用 GPU。")

    gpu_count = torch.cuda.device_count()
    if local_rank >= gpu_count:
        raise ValueError(f"LOCAL_RANK={local_rank} 超出可用 GPU 数量 {gpu_count}。")
    if world_size > gpu_count:
        raise ValueError(f"WORLD_SIZE={world_size} 大于当前可用 GPU 数量 {gpu_count}。")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return torch.device(f"cuda:{local_rank}")


def cleanup_distributed() -> None:
    """
    用法: cleanup_distributed()
    作用: 释放当前进程的分布式进程组资源
    输入: 无
    输出: None
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """
    用法: flag = is_main_process(rank)
    作用: 判断当前进程是否为负责日志与 checkpoint 的主进程
    输入: rank: int，当前全局进程序号
    输出: bool，若为主进程则返回 True
    """
    return int(rank) == 0


def emit_rank0_message(message: str, rank: int, log_path: Optional[Path], console: bool = True) -> None:
    """
    用法: emit_rank0_message("start", rank=0, log_path=Path("outputs/run/train.log"))
    作用: 仅在 rank 0 输出终端日志并可选写入日志文件
    输入: message: str，日志内容；rank: int；log_path: Path | None；console: bool
    输出: None
    """
    if is_main_process(rank):
        single_train.emit_message(message, log_path=log_path, console=console)


def validate_runtime_device(device_name: str) -> None:
    """
    用法: validate_runtime_device("cuda")
    作用: 校验 DDP 训练脚本中的设备配置是否合法
    输入: device_name: str，配置或命令行中的设备字段
    输出: None，非法时抛出异常
    """
    normalized_name = str(device_name).strip().lower()
    if not normalized_name.startswith("cuda"):
        raise ValueError("train_multimodal_ddp.py 仅支持 CUDA 设备，请使用 torchrun 进行单机多卡训练。")


def build_distributed_dataloader(
    dataset,
    dataloader_cfg: dict[str, Any],
    split: str,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler]:
    """
    用法: dataloader, sampler = build_distributed_dataloader(dataset, cfg, "train", device, 0, 4)
    作用: 为指定 split 构建基于 DistributedSampler 的 DataLoader
    输入: dataset: 数据集实例；dataloader_cfg: dict；split: str；device: torch.device；rank: int；world_size: int
    输出: tuple[DataLoader, DistributedSampler]，分别为数据加载器与其分布式采样器
    """
    is_train = split == "train"
    batch_size_key = "train_batch_size" if is_train else "val_batch_size"
    batch_size = int(dataloader_cfg.get(batch_size_key, dataloader_cfg.get("batch_size", 1)))
    num_workers = int(dataloader_cfg.get("num_workers", 0))
    persistent_workers = bool(dataloader_cfg.get("persistent_workers", False)) and num_workers > 0
    pin_memory = bool(dataloader_cfg.get("pin_memory", True)) and device.type == "cuda"

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=bool(dataloader_cfg.get("train_shuffle", True)) if is_train else False,
        drop_last=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=bool(dataloader_cfg.get("train_drop_last", False)) if is_train else False,
        persistent_workers=persistent_workers,
        collate_fn=single_train.placement_multimodal_collate_fn,
    )
    return dataloader, sampler


def reduce_metric_sums(
    metric_sums: Mapping[str, float],
    sample_count: int,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    """
    用法: reduced_sums, reduced_count = reduce_metric_sums(metric_sums, 128, device)
    作用: 对多卡上的指标累加值与样本数执行 all_reduce 求和
    输入: metric_sums: Mapping[str, float]；sample_count: int；device: torch.device
    输出: tuple[dict[str, float], int]，聚合后的指标累加值与总样本数
    """
    ordered_keys = ("loss", "center_loss", "object_center_loss", "size_loss", "yaw_loss")
    payload = torch.tensor(
        [float(metric_sums[key]) for key in ordered_keys] + [float(sample_count)],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)

    reduced_sums = {
        key: float(payload[idx].item())
        for idx, key in enumerate(ordered_keys)
    }
    reduced_count = int(round(float(payload[-1].item())))
    return reduced_sums, reduced_count


def run_one_epoch_distributed(
    model: DDP,
    dataloader: DataLoader,
    criterion: MultimodalBBoxLoss,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    log_interval: int,
    enable_tqdm: bool,
    rank: int,
    log_path: Optional[Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    grad_clip_norm: float = 0.0,
    use_amp: bool = False,
) -> dict[str, float]:
    """
    用法: metrics = run_one_epoch_distributed(model, dataloader, criterion, device, 1, 20, 10, True, 0, log_path, optimizer)
    作用: 执行单个 epoch 的 DDP 训练或验证循环，并返回全局平均指标
    输入: model: DDP；dataloader: DataLoader；criterion: MultimodalBBoxLoss；device: torch.device；
        epoch_idx: int；total_epochs: int；log_interval: int；enable_tqdm: bool；rank: int；log_path: Path | None；
        optimizer: Optimizer | None；scaler: GradScaler | None；grad_clip_norm: float；use_amp: bool
    输出: dict[str, float]，当前 epoch 的全局平均 loss 指标
    """
    is_train = optimizer is not None
    model.train(mode=is_train)
    phase_name = "Train" if is_train else "Valid"
    phase_name_cn = "训练" if is_train else "验证"
    main_process = is_main_process(rank)

    metric_sums = {
        "loss": 0.0,
        "center_loss": 0.0,
        "object_center_loss": 0.0,
        "size_loss": 0.0,
        "yaw_loss": 0.0,
    }
    sample_count = 0
    num_steps = len(dataloader)
    dataloader_batch_size = getattr(dataloader, "batch_size", None)
    emit_rank0_message(
        f"开始{phase_name_cn} epoch {epoch_idx}/{total_epochs}："
        f"steps={num_steps}，per_gpu_batch_size={dataloader_batch_size}",
        rank=rank,
        log_path=log_path,
    )

    progress_bar = None
    if main_process and enable_tqdm and single_train.tqdm is not None:
        progress_bar = single_train.tqdm(
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
        batch_inputs = single_train.move_batch_to_device(batch, device)
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
                    pred_object_centers_norm=outputs["pred_object_centers_norm"],
                    target_object_centers_norm=batch_inputs["object_centers_norm"],
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
            reduced_sums, reduced_count = reduce_metric_sums(metric_sums, sample_count, device)
            running_metrics = {
                key: value / max(reduced_count, 1)
                for key, value in reduced_sums.items()
            }
            if is_train:
                running_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
            step_message = (
                f"[{phase_name}] epoch {epoch_idx}/{total_epochs} "
                f"step {step_idx}/{num_steps} - {single_train.format_metrics(running_metrics)}"
            )
            if progress_bar is not None:
                progress_bar.set_postfix_str(single_train.format_metrics(running_metrics))
                emit_rank0_message(step_message, rank=rank, log_path=log_path, console=False)
            else:
                emit_rank0_message(step_message, rank=rank, log_path=log_path)

    if progress_bar is not None:
        progress_bar.close()

    reduced_sums, reduced_count = reduce_metric_sums(metric_sums, sample_count, device)
    if reduced_count == 0:
        raise ValueError(f"{phase_name.lower()} dataloader produced zero samples")

    return {
        key: value / reduced_count
        for key, value in reduced_sums.items()
    }


def main() -> None:
    """
    用法: main()
    作用: 使用单机多卡 DDP 训练多模态模型并保存 checkpoint
    输入: 无，参数来自命令行与 torchrun 环境变量
    输出: None
    """
    args = build_parser().parse_args()
    local_rank, rank, world_size = get_distributed_context()
    device = initialize_distributed(local_rank=local_rank, world_size=world_size)

    try:
        config_path = single_train.resolve_project_path(args.config)
        train_cfg = single_train.load_yaml_config(config_path)

        model_config_path = single_train.resolve_project_path(
            train_cfg.get("model_config_path", "configs/base/model.yaml")
        )
        model_cfg = single_train.load_yaml_config(model_config_path)
        decoder_cfg = dict(model_cfg.get("decoder", {}))
        if int(decoder_cfg.get("num_queries", 1)) != 1:
            raise ValueError("train_multimodal_ddp.py 当前仅支持 decoder.num_queries=1 的单 query 监督")

        runtime_cfg = dict(train_cfg.get("train", {}))
        if args.device is not None:
            runtime_cfg["device"] = args.device
        if args.output_dir is not None:
            runtime_cfg["output_dir"] = args.output_dir.as_posix()
        if args.disable_tqdm:
            runtime_cfg["disable_tqdm"] = True
        train_cfg["train"] = runtime_cfg

        validate_runtime_device(runtime_cfg.get("device", "cuda"))

        checkpoint_cfg = dict(train_cfg.get("checkpoint", {}))
        if args.resume is not None:
            checkpoint_cfg["resume_path"] = args.resume.as_posix()
        train_cfg["checkpoint"] = checkpoint_cfg

        output_dir = single_train.resolve_project_path(
            runtime_cfg.get("output_dir", "outputs/multimodal_train_ddp")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train.log" if is_main_process(rank) else None
        if is_main_process(rank):
            single_train.initialize_log_file(log_path)
            single_train.dump_yaml_config(train_cfg, output_dir / "train_config_resolved.yaml")
            single_train.dump_yaml_config(model_cfg, output_dir / "model_config_snapshot.yaml")
            single_train.emit_message(f"训练日志将写入：{log_path}", log_path=log_path)

        seed = int(runtime_cfg.get("seed", 42))
        single_train.set_random_seed(seed)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))

        dataset_cfg = dict(train_cfg.get("dataset", {}))
        dataloader_cfg = dict(train_cfg.get("dataloader", {}))
        optimization_cfg = dict(train_cfg.get("optimization", {}))
        scheduler_cfg = dict(train_cfg.get("scheduler", {}))
        loss_cfg = dict(train_cfg.get("loss", {}))

        train_dataset = single_train.build_dataset(dataset_cfg, split="train")
        if len(train_dataset) == 0:
            raise ValueError("train split is empty, unable to start training")
        valid_dataset = single_train.build_dataset(dataset_cfg, split="valid")

        train_loader, train_sampler = build_distributed_dataloader(
            dataset=train_dataset,
            dataloader_cfg=dataloader_cfg,
            split="train",
            device=device,
            rank=rank,
            world_size=world_size,
        )
        valid_loader = None
        valid_sampler = None
        if len(valid_dataset) > 0:
            valid_loader, valid_sampler = build_distributed_dataloader(
                dataset=valid_dataset,
                dataloader_cfg=dataloader_cfg,
                split="valid",
                device=device,
                rank=rank,
                world_size=world_size,
            )
        else:
            emit_rank0_message(
                "valid split 为空，本次训练将使用 train loss 作为 best checkpoint 指标。",
                rank=rank,
                log_path=log_path,
            )

        emit_rank0_message(
            f"DDP 环境：world_size={world_size}，rank={rank}，local_rank={local_rank}，device={device}",
            rank=rank,
            log_path=log_path,
        )
        emit_rank0_message(
            f"数据集构建完成：train={len(train_dataset)}，valid={len(valid_dataset)}",
            rank=rank,
            log_path=log_path,
        )

        model = MultimodalModel(model_cfg).to(device)
        center_weight = float(loss_cfg.get("center_weight", 1.0))
        criterion = MultimodalBBoxLoss(
            center_weight=center_weight,
            size_weight=float(loss_cfg.get("size_weight", 1.0)),
            yaw_weight=float(loss_cfg.get("yaw_weight", 0.5)),
            object_center_weight=float(loss_cfg.get("object_center_weight", center_weight)),
            smooth_l1_beta=float(loss_cfg.get("smooth_l1_beta", 1.0)),
        )
        optimizer = single_train.create_optimizer(model, optimization_cfg)
        scheduler = single_train.create_scheduler(optimizer, optimization_cfg, scheduler_cfg)

        use_amp = bool(runtime_cfg.get("use_amp", False)) and device.type == "cuda"
        scaler = GradScaler("cuda", enabled=use_amp)
        enable_tqdm = not bool(runtime_cfg.get("disable_tqdm", False))
        if enable_tqdm and single_train.tqdm is None:
            enable_tqdm = False
            emit_rank0_message("未检测到 tqdm，自动回退到普通日志输出。", rank=rank, log_path=log_path)
        if enable_tqdm and not sys.stderr.isatty():
            enable_tqdm = False
            emit_rank0_message(
                "当前输出不是交互式终端，自动关闭 tqdm 进度条并使用普通日志输出。",
                rank=rank,
                log_path=log_path,
            )

        start_epoch = 1
        best_metric = None
        best_metric_name = "val_loss" if valid_loader is not None else "train_loss"
        resume_path_value = checkpoint_cfg.get("resume_path")
        if resume_path_value:
            resume_path = single_train.resolve_project_path(resume_path_value)
            if not resume_path.exists():
                raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
            start_epoch, best_metric, best_metric_name = single_train.load_checkpoint(
                checkpoint_path=resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            emit_rank0_message(
                f"已恢复训练状态：resume={resume_path}, start_epoch={start_epoch}, "
                f"best_{best_metric_name}={best_metric}",
                rank=rank,
                log_path=log_path,
            )

        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        total_epochs = int(optimization_cfg.get("epochs", 1))
        grad_clip_norm = float(optimization_cfg.get("grad_clip_norm", 0.0))
        log_interval = int(runtime_cfg.get("log_interval", 10))
        last_name = str(checkpoint_cfg.get("save_last_name", "last.pth"))
        best_name = str(checkpoint_cfg.get("save_best_name", "best.pth"))
        scheduler_type = str(scheduler_cfg.get("type", "none")).strip().lower() or "none"
        train_batch_size = int(getattr(train_loader, "batch_size", 0) or 0)
        valid_batch_size = 0 if valid_loader is None else int(getattr(valid_loader, "batch_size", 0) or 0)

        emit_rank0_message(
            f"训练配置：config={config_path}，model_config={model_config_path}，output_dir={output_dir}",
            rank=rank,
            log_path=log_path,
        )
        emit_rank0_message(
            "运行参数："
            f"world_size={world_size}，device={device.type}，use_amp={use_amp}，seed={seed}，"
            f"log_interval={log_interval}，tqdm={'on' if enable_tqdm else 'off'}",
            rank=rank,
            log_path=log_path,
        )
        emit_rank0_message(
            "DataLoader："
            f"train_samples={len(train_dataset)}，valid_samples={len(valid_dataset)}，"
            f"train_batches={len(train_loader)}，valid_batches={0 if valid_loader is None else len(valid_loader)}，"
            f"per_gpu_train_batch_size={train_batch_size}，"
            f"per_gpu_valid_batch_size={valid_batch_size}，"
            f"global_train_batch_size={train_batch_size * world_size}，"
            f"global_valid_batch_size={valid_batch_size * world_size if valid_loader is not None else 0}，"
            f"num_workers_per_process={int(dataloader_cfg.get('num_workers', 0))}",
            rank=rank,
            log_path=log_path,
        )
        emit_rank0_message(
            "优化配置："
            f"epochs={total_epochs}，lr={optimizer.param_groups[0]['lr']:.6g}，"
            f"weight_decay={float(optimization_cfg.get('weight_decay', 1e-4)):.6g}，"
            f"grad_clip_norm={grad_clip_norm:.6g}，scheduler={scheduler_type}",
            rank=rank,
            log_path=log_path,
        )

        for epoch_idx in range(start_epoch, total_epochs + 1):
            train_sampler.set_epoch(epoch_idx)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch_idx)

            epoch_start_time = time.perf_counter()
            train_start_time = time.perf_counter()
            train_metrics = run_one_epoch_distributed(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                device=device,
                epoch_idx=epoch_idx,
                total_epochs=total_epochs,
                log_interval=log_interval,
                enable_tqdm=enable_tqdm,
                rank=rank,
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
                    valid_metrics = run_one_epoch_distributed(
                        model=model,
                        dataloader=valid_loader,
                        criterion=criterion,
                        device=device,
                        epoch_idx=epoch_idx,
                        total_epochs=total_epochs,
                        log_interval=log_interval,
                        enable_tqdm=enable_tqdm,
                        rank=rank,
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

            if is_main_process(rank):
                single_train.save_checkpoint(
                    checkpoint_path=output_dir / last_name,
                    model=model.module,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch_idx=epoch_idx,
                    best_metric=best_metric,
                    best_metric_name=best_metric_name,
                    train_cfg=train_cfg,
                    model_cfg=model_cfg,
                )
                single_train.emit_message(f"已保存 last checkpoint：{output_dir / last_name}", log_path=log_path)
                if is_best:
                    single_train.save_checkpoint(
                        checkpoint_path=output_dir / best_name,
                        model=model.module,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch_idx=epoch_idx,
                        best_metric=best_metric,
                        best_metric_name=best_metric_name,
                        train_cfg=train_cfg,
                        model_cfg=model_cfg,
                    )
                    single_train.emit_message(
                        f"best checkpoint 已更新：{output_dir / best_name}，"
                        f"{best_metric_name}={best_metric:.6f}",
                        log_path=log_path,
                    )

            epoch_elapsed = time.perf_counter() - epoch_start_time
            summary_chunks = [
                f"[Epoch {epoch_idx}/{total_epochs}]",
                f"train_time={single_train.format_duration(train_elapsed)}",
                f"train: {single_train.format_metrics(train_metrics)}",
            ]
            if valid_metrics is not None:
                summary_chunks.append(f"valid_time={single_train.format_duration(valid_elapsed)}")
                summary_chunks.append(f"valid: {single_train.format_metrics(valid_metrics)}")
            summary_chunks.append(f"epoch_time={single_train.format_duration(epoch_elapsed)}")
            summary_chunks.append(f"best_{best_metric_name}={best_metric:.6f}")
            emit_rank0_message(" | ".join(summary_chunks), rank=rank, log_path=log_path)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

# 使用示例:
# CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n spatial torchrun --nproc_per_node=4 \
#     scripts/train_multimodal_ddp.py --config configs/experiments/multimodal_train.yaml
