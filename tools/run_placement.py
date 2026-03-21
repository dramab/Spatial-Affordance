#!/usr/bin/env python3
"""
tools/run_placement.py
-----------------------
放置规划 CLI 入口。

从 YAML 配置文件加载参数，通过数据集适配器加载场景数据，
运行 PlacementPipeline 生成放置结果。

用法:
    # 处理单个场景的单帧
    python tools/run_placement.py \
        --config configs/annotation/placement.yaml \
        --scene /path/to/scene_0001 \
        --frame 0000 \
        --output outputs/placement

    # 串行批量处理所有场景
    python tools/run_placement.py \
        --config configs/annotation/placement.yaml \
        --batch \
        --output outputs/placement

    # 多进程并发批量处理
    python tools/run_placement.py \
        --config configs/annotation/placement.yaml \
        --batch --workers 8 \
        --output outputs/placement

    # 使用 GPU 加速
    python tools/run_placement.py \
        --config configs/annotation/placement.yaml \
        --batch --gpu \
        --output outputs/placement
"""

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import sys
import traceback
import yaml
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.datatypes import PlacementConfig
from src.annotation.free_bbox.pipeline import PlacementPipeline


def load_config(config_path):
    """加载 YAML 配置文件。"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_adapter(cfg):
    """根据配置构建数据集适配器。"""
    ds_cfg = cfg.get("dataset", {})
    ds_type = ds_cfg.get("type", "hope")

    if ds_type == "hope":
        from src.datasets.hope_adapter import HopeAdapter
        return HopeAdapter(
            root_dir=ds_cfg["root_dir"],
            mesh_dir=ds_cfg.get("mesh_dir"),
            frame_step=ds_cfg.get("frame_step", 60),
        )
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")


def build_placement_config(cfg):
    """从 YAML 配置构建 PlacementConfig。"""
    occ = cfg.get("occupancy", {})
    plc = cfg.get("placement", {})
    clu = cfg.get("clustering", {})
    vis = cfg.get("visualization", {})
    comp = cfg.get("compute", {})

    world_up = plc.get("world_up", [0, 0, 1])

    return PlacementConfig(
        voxel_size=occ.get("voxel_size", 1.0),
        pixel_stride=occ.get("pixel_stride", 4),
        grid_padding=occ.get("grid_padding", 10.0),
        safety_margin=plc.get("safety_margin", 0.5),
        yaw_steps=plc.get("yaw_steps", 24),
        min_surface_area=plc.get("min_surface_area", 50.0),
        min_support_ratio=plc.get("min_support_ratio", 1.0),
        occlusion_threshold=plc.get("occlusion_threshold", 0.3),
        dbscan_eps=clu.get("dbscan_eps"),
        dbscan_min_samples=clu.get("dbscan_min_samples", 1),
        world_up=tuple(world_up),
        vis_margin_px=vis.get("vis_margin_px", 30),
        stability_chunk_size=comp.get("stability_chunk_size", 2000),
    )


def process_single(adapter, pipeline, scene_path, frame_id,
                   output_root, save_vis):
    """处理单个场景的单帧。"""
    print(f"\n{'='*60}")
    print(f"Scene: {scene_path}  Frame: {frame_id}")
    print(f"{'='*60}")

    scene_data = adapter.load_scene(scene_path, frame_id)
    os.makedirs(output_root, exist_ok=True)

    results = pipeline.run(
        scene_data,
        output_dir=output_root,
        save_vis=save_vis,
    )

    n_objects = len(results)
    n_placed = sum(1 for r in results.values()
                   if len(r.placements) > 0)
    print(f"\nDone: {n_placed}/{n_objects} objects have valid placements.")
    return results


def expand_batch_tasks(scenes):
    """将 adapter.list_scenes() 结果展开成单帧任务列表。"""
    tasks = []
    for scene_path, frame_ids in scenes:
        for frame_id in frame_ids:
            tasks.append((scene_path, frame_id))
    return tasks


def process_task_worker(config_path, scene_path, frame_id,
                        output_root, save_vis, use_gpu):
    """子进程 worker：独立完成单个 scene/frame 任务。"""
    scene_name = Path(scene_path).name

    try:
        cfg = load_config(config_path)
        adapter = build_adapter(cfg)
        placement_cfg = build_placement_config(cfg)
        pipeline = PlacementPipeline(config=placement_cfg, use_gpu=use_gpu)

        process_single(
            adapter=adapter,
            pipeline=pipeline,
            scene_path=scene_path,
            frame_id=frame_id,
            output_root=output_root,
            save_vis=save_vis,
        )
        return {
            "ok": True,
            "scene": scene_name,
            "frame": frame_id,
        }
    except Exception as exc:
        return {
            "ok": False,
            "scene": scene_name,
            "frame": frame_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def process_batch_serial(adapter, pipeline, tasks, output_root, save_vis):
    """串行处理 batch 任务。"""
    success = 0
    failed = 0

    for idx, (scene_path, frame_id) in enumerate(tasks, start=1):
        scene_name = Path(scene_path).name
        print(f"\n[Task {idx}/{len(tasks)}] {scene_name}/{frame_id}")
        try:
            process_single(
                adapter,
                pipeline,
                scene_path,
                frame_id,
                output_root,
                save_vis,
            )
            success += 1
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {scene_name}/{frame_id}: {exc}")
            print(traceback.format_exc())

    return success, failed


def process_batch_parallel(config_path, tasks, output_root,
                           save_vis, use_gpu, workers):
    """多进程并发处理 batch 任务。"""
    success = 0
    failed = 0
    mp_ctx = mp.get_context("spawn")

    with cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_ctx,
    ) as executor:
        future_to_task = {
            executor.submit(
                process_task_worker,
                config_path,
                scene_path,
                frame_id,
                output_root,
                save_vis,
                use_gpu,
            ): (scene_path, frame_id)
            for scene_path, frame_id in tasks
        }

        for idx, future in enumerate(cf.as_completed(future_to_task), start=1):
            scene_path, frame_id = future_to_task[future]
            scene_name = Path(scene_path).name

            try:
                result = future.result()
            except Exception as exc:
                failed += 1
                print(f"[{idx}/{len(tasks)}] [ERROR] {scene_name}/{frame_id}: {exc}")
                print(traceback.format_exc())
                continue

            if result["ok"]:
                success += 1
                print(f"[{idx}/{len(tasks)}] [OK] {result['scene']}/{result['frame']}")
            else:
                failed += 1
                print(
                    f"[{idx}/{len(tasks)}] [ERROR] "
                    f"{result['scene']}/{result['frame']}: {result['error']}"
                )
                print(result["traceback"])

    return success, failed


def main():
    parser = argparse.ArgumentParser(
        description="Placement Planning Pipeline")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config file path")
    parser.add_argument("--scene", type=str, default=None,
                        help="Single scene directory path")
    parser.add_argument("--frame", type=str, default=None,
                        help="Frame ID (e.g. 0000)")
    parser.add_argument("--batch", action="store_true",
                        help="Batch process all scenes")
    parser.add_argument("--output", type=str, default=None,
                        help="Output root directory")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU acceleration")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes for --batch (default: 1)")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    cfg = load_config(args.config)
    adapter = build_adapter(cfg)
    placement_cfg = build_placement_config(cfg)

    use_gpu = args.gpu or cfg.get("compute", {}).get("use_gpu", False)
    save_vis = not args.no_vis and cfg.get("visualization", {}).get("save_vis", True)

    pipeline = PlacementPipeline(config=placement_cfg, use_gpu=use_gpu)

    output_root = args.output or cfg.get("output", {}).get("dir", "outputs/placement")

    if args.batch:
        scenes = adapter.list_scenes()
        tasks = expand_batch_tasks(scenes)
        workers = min(args.workers, max(len(tasks), 1))

        print(
            f"Found {len(scenes)} scenes and {len(tasks)} frames to process "
            f"with {workers} worker(s)."
        )
        if use_gpu and workers > 1:
            print(
                "[WARN] Running multiple GPU workers may cause GPU memory "
                "contention or reduced throughput."
            )

        if workers == 1:
            success, failed = process_batch_serial(
                adapter=adapter,
                pipeline=pipeline,
                tasks=tasks,
                output_root=output_root,
                save_vis=save_vis,
            )
        else:
            success, failed = process_batch_parallel(
                config_path=str(Path(args.config).resolve()),
                tasks=tasks,
                output_root=output_root,
                save_vis=save_vis,
                use_gpu=use_gpu,
                workers=workers,
            )

        print(
            f"\n[BATCH DONE] success={success} failed={failed} "
            f"total={len(tasks)}"
        )

    elif args.scene and args.frame:
        process_single(adapter, pipeline, args.scene, args.frame,
                       output_root, save_vis)
    else:
        parser.error("Specify --scene + --frame for single, or --batch for all.")


if __name__ == "__main__":
    main()
