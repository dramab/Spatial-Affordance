#!/usr/bin/env python3
"""
tools/export_placement_rgb_bbox_vis_parallel.py
-----------------------------------------------
并行导出 placement 输出目录中的空位框样本 RGB 投影可视化图片。

图片内容:
    - 原始物体 3D bbox 在 RGB 图上的投影
    - 空位框样本对应 3D bbox 在 RGB 图上的投影

用法:
    python tools/export_placement_rgb_bbox_vis_parallel.py \
        --inputs outputs/housecat6d_placement10 outputs/placement_hope5 \
        --output-dir outputs/placement_rgb_bbox_vis \
        --workers 4 \
        --batch-size 8

作用:
    - 复用串行脚本中已有的数据加载、投影绘制与命名逻辑
    - 按 scene_id + frame_id 对 sample 分组，避免同一帧重复加载
    - 使用多进程并发导出图片
    - 通过 batch-size 限制每轮提交的 frame 任务数，避免 future 过多导致内存上涨

输入:
    --inputs: 一个或多个 placement 输出根目录
    --output-dir: 统一输出目录
    --workers: 并发进程数
    --batch-size: 每轮最多提交的 frame 任务数
    --limit: 可选，仅导出前 N 个待导出样本用于快速验证
    --overwrite: 可选，覆盖已存在图片

输出:
    在输出目录下生成:
        - {source_dir}__{sample_id}.png

使用示例:
    python tools/export_placement_rgb_bbox_vis_parallel.py \
        --inputs outputs/housecat6d_placement10 outputs/placement_hope5 \
        --output-dir outputs/placement_rgb_bbox_vis_parallel \
        --workers 4 \
        --batch-size 8 \
        --limit 100
"""

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.export_placement_rgb_bbox_vis import (  # noqa: E402
    build_adapter,
    build_parser as build_serial_parser,
    collect_sample_jsons,
    infer_config_path,
    load_yaml_config,
    prepare_export_sample_entries,
    render_sample_image,
)


_WORKER_ADAPTER_CACHE: Dict[str, object] = {}


def build_parallel_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parallel_parser()
    作用: 构建并行导出脚本的命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser，配置完成的解析器
    """
    parser = build_serial_parser()
    parser.description = "并行导出 placement 空位框样本的 RGB 3D bbox 投影图"
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发进程数，默认 1",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每轮最多提交的 frame 任务数，默认与 workers 相同",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    用法: validate_args(args, parser)
    作用: 校验并行脚本的关键参数是否合法
    输入: args: argparse.Namespace，命令行解析结果；parser: argparse.ArgumentParser
    输出: None，参数非法时直接抛出 parser.error
    """
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")


def get_input_config_map(input_dirs: Iterable[Path]) -> Dict[str, str]:
    """
    用法: config_map = get_input_config_map(input_dirs)
    作用: 为每个输入目录生成对应的数据集配置文件映射
    输入: input_dirs: Iterable[Path]，placement 输出根目录列表
    输出: Dict[str, str]，key 为输入目录绝对路径，value 为配置文件绝对路径
    """
    config_map: Dict[str, str] = {}
    for input_dir in input_dirs:
        config_map[str(input_dir.resolve())] = str(infer_config_path(input_dir).resolve())
    return config_map


def collect_frame_tasks(input_dirs: List[Path],
                        output_dir: Path,
                        config_map: Dict[str, str],
                        limit: int = None,
                        overwrite: bool = False) -> Tuple[List[dict], int]:
    """
    用法: tasks, start_idx = collect_frame_tasks(input_dirs, output_dir, config_map, limit=100)
    作用: 收集待导出的样本并按帧分组为并行任务
    输入: input_dirs: List[Path]，placement 输出目录；
         output_dir: Path，统一输出目录；
         config_map: Dict[str, str]，输入目录到配置路径的映射；
         limit: int | None，最多保留多少个待导出样本；
         overwrite: bool，是否覆盖已有图片
    输出: Tuple[List[dict], int]，分别为按帧分组后的任务列表与起始样本下标
    """
    sample_files = collect_sample_jsons(input_dirs)
    sample_entries, start_index = prepare_export_sample_entries(
        sample_files=sample_files,
        output_dir=output_dir,
        limit=limit,
        overwrite=overwrite,
    )
    frame_tasks: Dict[Tuple[str, str, str], dict] = {}

    for entry in sample_entries:
        source_dir = entry["source_dir"]
        sample_record = entry["sample_record"]
        task_key = (
            str(source_dir),
            str(sample_record["scene_id"]),
            str(sample_record["frame_id"]),
        )
        if task_key not in frame_tasks:
            frame_tasks[task_key] = {
                "source_dir": str(source_dir),
                "scene_id": str(sample_record["scene_id"]),
                "frame_id": str(sample_record["frame_id"]),
                "config_path": config_map[str(source_dir)],
                "overwrite": overwrite,
                "samples": [],
            }

        frame_tasks[task_key]["samples"].append(
            {
                "sample_id": str(sample_record["sample_id"]),
                "sample_record": sample_record,
                "output_path": str(entry["output_path"]),
            }
        )

    return list(frame_tasks.values()), start_index


def iter_task_batches(tasks: List[dict], batch_size: int) -> Iterator[List[dict]]:
    """
    用法: for batch in iter_task_batches(tasks, batch_size): ...
    作用: 按固定批大小切分 frame 任务列表
    输入: tasks: List[dict]，待处理任务；batch_size: int，每批任务数
    输出: Iterator[List[dict]]，分批后的任务列表迭代器
    """
    for start in range(0, len(tasks), batch_size):
        yield tasks[start:start + batch_size]


def get_worker_adapter(config_path: str):
    """
    用法: adapter = get_worker_adapter(config_path)
    作用: 在 worker 进程内缓存并复用同一配置对应的 adapter
    输入: config_path: str，配置文件路径
    输出: DatasetAdapter，对应数据集适配器实例
    """
    if config_path not in _WORKER_ADAPTER_CACHE:
        _WORKER_ADAPTER_CACHE[config_path] = build_adapter(
            load_yaml_config(Path(config_path))
        )
    return _WORKER_ADAPTER_CACHE[config_path]


def process_frame_task(task: dict) -> dict:
    """
    用法: result = process_frame_task(task)
    作用: 处理单个 frame 任务，加载一次场景并导出该帧下所有样本图片
    输入: task: dict，包含 source_dir、scene_id、frame_id、config_path 与 samples
    输出: dict，包含成功状态、导出数量、失败信息与任务标识
    """
    source_dir = Path(task["source_dir"])
    scene_id = task["scene_id"]
    frame_id = task["frame_id"]

    try:
        adapter = get_worker_adapter(task["config_path"])
        scene_path = Path(adapter.root_dir) / scene_id
        scene = adapter.load_scene(str(scene_path), frame_id)

        exported = 0
        skipped = 0
        for sample in task["samples"]:
            output_path = Path(sample["output_path"])
            if output_path.exists() and not task["overwrite"]:
                skipped += 1
                continue

            image = render_sample_image(scene, sample["sample_record"])
            image.save(output_path)
            exported += 1

        return {
            "ok": True,
            "source_dir": source_dir.name,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "exported": exported,
            "skipped": skipped,
        }
    except Exception as exc:
        return {
            "ok": False,
            "source_dir": source_dir.name,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_batch_parallel(batch_tasks: List[dict], workers: int) -> Tuple[int, int]:
    """
    用法: exported, failed = run_batch_parallel(batch_tasks, workers=4)
    作用: 并行执行一批 frame 任务并统计结果
    输入: batch_tasks: List[dict]，当前批次任务；workers: int，并发进程数
    输出: Tuple[int, int]，分别为导出图片数与失败帧任务数
    """
    if not batch_tasks:
        return 0, 0

    exported_total = 0
    failed_total = 0
    max_workers = min(workers, len(batch_tasks))
    mp_ctx = mp.get_context("spawn")

    with cf.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
    ) as executor:
        future_to_task = {
            executor.submit(process_frame_task, task): task
            for task in batch_tasks
        }

        for idx, future in enumerate(cf.as_completed(future_to_task), start=1):
            task = future_to_task[future]
            frame_name = f"{Path(task['source_dir']).name}/{task['scene_id']}/{task['frame_id']}"
            try:
                result = future.result()
            except Exception as exc:
                failed_total += 1
                print(f"[Batch {idx}/{len(batch_tasks)}] [ERROR] {frame_name}: {exc}")
                print(traceback.format_exc())
                continue

            frame_name = (
                f"{result['source_dir']}/{result['scene_id']}/{result['frame_id']}"
            )
            if result["ok"]:
                exported_total += result["exported"]
                print(
                    f"[Batch {idx}/{len(batch_tasks)}] [OK] {frame_name} "
                    f"exported={result['exported']} skipped={result['skipped']}"
                )
            else:
                failed_total += 1
                print(f"[Batch {idx}/{len(batch_tasks)}] [ERROR] {frame_name}: {result['error']}")
                print(result["traceback"])

    return exported_total, failed_total


def export_samples_parallel(input_dirs: List[Path],
                            output_dir: Path,
                            workers: int,
                            batch_size: int = None,
                            limit: int = None,
                            overwrite: bool = False) -> Tuple[int, int, int]:
    """
    用法: exported, failed, task_count = export_samples_parallel(...)
    作用: 分批并行导出所有空位框样本的 RGB 投影可视化图片
    输入: input_dirs: List[Path]，placement 输出目录；
         output_dir: Path，统一输出目录；
         workers: int，并发进程数；
         batch_size: int | None，每轮最多提交的 frame 任务数；
         limit: int | None，导出上限；
         overwrite: bool，是否覆盖已有图片
    输出: Tuple[int, int, int]，分别为导出图片数、失败帧任务数、总 frame 任务数
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config_map = get_input_config_map(input_dirs)
    frame_tasks, resume_start_index = collect_frame_tasks(
        input_dirs=input_dirs,
        output_dir=output_dir,
        config_map=config_map,
        limit=limit,
        overwrite=overwrite,
    )
    if not frame_tasks:
        return 0, 0, 0

    if resume_start_index > 0 and not overwrite:
        print(f"检测到已有输出，已从第 {resume_start_index + 1} 条样本继续导出")

    effective_batch_size = batch_size or workers
    exported_total = 0
    failed_total = 0

    for batch_idx, batch_tasks in enumerate(
        iter_task_batches(frame_tasks, effective_batch_size),
        start=1,
    ):
        print(
            f"开始处理批次 {batch_idx}，frame 任务数: {len(batch_tasks)}，"
            f"workers: {min(workers, len(batch_tasks))}"
        )
        batch_exported, batch_failed = run_batch_parallel(batch_tasks, workers)
        exported_total += batch_exported
        failed_total += batch_failed
        print(
            f"批次 {batch_idx} 完成，累计导出图片: {exported_total}，"
            f"累计失败 frame 任务: {failed_total}"
        )

    return exported_total, failed_total, len(frame_tasks)


def main() -> None:
    """
    用法: main()
    作用: 执行并行导出 placement RGB 3D bbox 投影图的 CLI 主流程
    输入: 无，参数来自命令行
    输出: None，在终端打印导出结果
    """
    parser = build_parallel_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    input_dirs = [path.resolve() for path in args.inputs]
    output_dir = args.output_dir.resolve()
    exported, failed, task_count = export_samples_parallel(
        input_dirs=input_dirs,
        output_dir=output_dir,
        workers=args.workers,
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
    )

    print("导出完成")
    print(f"输入目录数: {len(input_dirs)}")
    print(f"统一输出目录: {output_dir}")
    print(f"总 frame 任务数: {task_count}")
    print(f"实际导出图片数: {exported}")
    print(f"失败 frame 任务数: {failed}")


if __name__ == "__main__":
    main()
