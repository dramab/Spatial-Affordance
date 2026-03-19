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
        --output outputs/placement/scene_0001/0000

    # 批量处理所有场景
    python tools/run_placement.py \
        --config configs/annotation/placement.yaml \
        --batch \
        --output outputs/placement

    # 使用 GPU 加速
    python tools/run_placement.py \
        --config configs/annotation/placement.yaml \
        --batch --gpu
"""

import argparse
import os
import sys
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
        dbscan_eps=clu.get("dbscan_eps", 5.0),
        dbscan_min_samples=clu.get("dbscan_min_samples", 1),
        world_up=tuple(world_up),
        vis_margin_px=vis.get("vis_margin_px", 30),
        stability_chunk_size=comp.get("stability_chunk_size", 2000),
    )


def process_single(adapter, pipeline, scene_path, frame_id,
                    output_dir, save_vis, save_ply):
    """处理单个场景的单帧。"""
    print(f"\n{'='*60}")
    print(f"Scene: {scene_path}  Frame: {frame_id}")
    print(f"{'='*60}")

    scene_data = adapter.load_scene(scene_path, frame_id)
    os.makedirs(output_dir, exist_ok=True)

    results = pipeline.run(
        scene_data,
        output_dir=output_dir,
        save_vis=save_vis,
        save_ply_files=save_ply,
    )

    n_objects = len(results)
    n_placed = sum(1 for r in results.values()
                   if len(r.placements) > 0)
    print(f"\nDone: {n_placed}/{n_objects} objects have valid placements.")
    return results


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
                        help="Output directory")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU acceleration")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--no-ply", action="store_true",
                        help="Skip PLY export")
    args = parser.parse_args()

    cfg = load_config(args.config)
    adapter = build_adapter(cfg)
    placement_cfg = build_placement_config(cfg)

    use_gpu = args.gpu or cfg.get("compute", {}).get("use_gpu", False)
    save_vis = not args.no_vis and cfg.get("visualization", {}).get("save_vis", True)
    save_ply = not args.no_ply and cfg.get("visualization", {}).get("save_ply", True)

    pipeline = PlacementPipeline(config=placement_cfg, use_gpu=use_gpu)

    output_base = args.output or cfg.get("output", {}).get("dir", "outputs/placement")

    if args.batch:
        scenes = adapter.list_scenes()
        print(f"Found {len(scenes)} scenes to process.")
        for scene_path, frame_ids in scenes:
            scene_name = Path(scene_path).name
            for fid in frame_ids:
                out_dir = os.path.join(output_base, scene_name, fid)
                try:
                    process_single(adapter, pipeline, scene_path, fid,
                                   out_dir, save_vis, save_ply)
                except Exception as e:
                    print(f"[ERROR] {scene_name}/{fid}: {e}")
                    import traceback
                    traceback.print_exc()

    elif args.scene and args.frame:
        out_dir = output_base
        process_single(adapter, pipeline, args.scene, args.frame,
                       out_dir, save_vis, save_ply)
    else:
        parser.error("Specify --scene + --frame for single, or --batch for all.")


if __name__ == "__main__":
    main()
