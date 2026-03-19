"""
src/annotation/free_bbox/pipeline.py
--------------------------------------
放置规划 Pipeline：编排完整的放置规划流程。

流程:
    SceneData → 点云 → 占据栅格 → 物体体素化 → 逐物体处理:
        支撑面检测 → FFT 碰撞搜索 → 稳定性过滤 → 可见性过滤 →
        遮挡过滤 → DBSCAN 聚类 → 可视化

用法:
    from src.annotation.free_bbox.pipeline import PlacementPipeline
    from src.annotation.free_bbox.datatypes import PlacementConfig

    pipeline = PlacementPipeline(PlacementConfig())
    results = pipeline.run(scene_data, output_dir="/path/to/output")
"""

import os
import numpy as np

from src.annotation.free_bbox.datatypes import (
    SceneData, PlacementConfig, PlacementResult,
)
from src.annotation.free_bbox.occupancy import (
    depth_to_pointcloud, build_occupancy_grid, FREE, OCCUPIED, UNKNOWN,
)
from src.annotation.free_bbox.voxel_utils import make_voxel_params, voxel_to_world
from src.annotation.free_bbox.grid_ops import (
    voxelize_obb, prepare_grid_base, grid_remove_object, grid_restore_object,
)
from src.annotation.free_bbox.surface import detect_support_surfaces
from src.annotation.free_bbox.collision import find_table_placements
from src.annotation.free_bbox.filters import (
    is_fully_visible, filter_stable_placements,
    filter_visible_placements, filter_occluded_placements,
    build_depth_buffer,
)
from src.annotation.free_bbox.cluster import cluster_placements
from src.annotation.free_bbox.io_utils import (
    save_ply, save_occupancy_ply, save_placement_result, save_grid_meta,
)
from src.utils.coord_utils import transform_points


class PlacementPipeline:
    """
    放置规划 Pipeline。

    属性:
        config: PlacementConfig 配置参数
        use_gpu: bool 是否使用 GPU 加速
    """

    def __init__(self, config: PlacementConfig = None, use_gpu: bool = False):
        self.config = config or PlacementConfig()
        self.use_gpu = use_gpu

    def run(self, scene: SceneData, output_dir: str = None,
            save_vis: bool = True, save_ply_files: bool = True) -> dict:
        """
        对场景执行完整放置规划。

        输入:
            scene: SceneData 场景数据
            output_dir: str 输出目录（None 则不保存文件）
            save_vis: bool 是否保存可视化
            save_ply_files: bool 是否保存 PLY 点云
        输出:
            dict {obj_id: PlacementResult} 每个物体的放置结果
        """
        cfg = self.config
        camera = scene.camera
        R_c2w = camera.E_c2w[:3, :3]
        cam_origin = camera.E_c2w[:3, 3]
        K = camera.K
        E_w2c = camera.E_w2c

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # ── Step 1: 点云生成 ──────────────────────────────────────────────
        print("[1/6] Generating point cloud ...")
        pts_world, colors = depth_to_pointcloud(
            scene.depth, scene.rgb,
            camera.fx, camera.fy, camera.cx, camera.cy,
            R_c2w, cam_origin, stride=cfg.pixel_stride)

        if output_dir and save_ply_files:
            save_ply(os.path.join(output_dir, "point_cloud.ply"),
                     pts_world, colors)

        # ── Step 2: 占据栅格构建 ──────────────────────────────────────────
        print("[2/6] Building occupancy grid ...")
        grid, grid_min, vs = build_occupancy_grid(
            scene.depth,
            camera.fx, camera.fy, camera.cx, camera.cy,
            R_c2w, cam_origin,
            voxel_size=cfg.voxel_size, stride=cfg.pixel_stride,
            padding=cfg.grid_padding)

        vp = make_voxel_params(grid_min, vs)

        if output_dir:
            np.save(os.path.join(output_dir, "occupancy_grid.npy"), grid)
            counts = {
                "free": int((grid == FREE).sum()),
                "occupied": int((grid == OCCUPIED).sum()),
                "unknown": int((grid == UNKNOWN).sum()),
            }
            save_grid_meta(
                os.path.join(output_dir, "grid_meta.json"),
                vp, grid.shape, voxel_counts=counts)
            if save_ply_files:
                save_occupancy_ply(
                    os.path.join(output_dir, "occupancy_grid.ply"),
                    grid, grid_min, vs,
                    states=[FREE, OCCUPIED])

        # ── Step 3: 物体体素化 ────────────────────────────────────────────
        print("[3/6] Voxelizing objects ...")
        grid_base = prepare_grid_base(grid, scene.objects, vp)

        # ── Step 4 & 5: 逐物体放置规划 ──────────────────────────────────
        print("[4/6] Planning placements ...")
        all_results = {}

        for obj in scene.objects:
            name = obj.class_name
            print(f"\n  Processing: {name}")

            # 可见性预检查
            pose_cam = E_w2c @ obj.pose_world
            if not is_fully_visible(obj.bbox3d_canonical, pose_cam,
                                    camera.fx, camera.fy,
                                    camera.cx, camera.cy,
                                    camera.img_w, camera.img_h):
                print(f"    [SKIP] {name} not fully visible")
                all_results[obj.obj_id] = PlacementResult(
                    obj_id=obj.obj_id, class_name=name,
                    original_aabb_world=np.zeros(6),
                    placements=[], num_raw_candidates=0)
                continue

            # 目标物体体素化 + 移除
            target_vox = voxelize_obb(
                obj.bbox3d_canonical, obj.pose_world, vp,
                np.array(grid_base.shape))
            if len(target_vox) == 0:
                print(f"    [SKIP] {name} no voxels")
                all_results[obj.obj_id] = PlacementResult(
                    obj_id=obj.obj_id, class_name=name,
                    original_aabb_world=np.zeros(6),
                    placements=[], num_raw_candidates=0)
                continue

            target_vox, saved = grid_remove_object(grid_base, target_vox)

            try:
                # 每个物体移除后单独检测支撑面（与原始代码一致）
                table_z, surface_mask = detect_support_surfaces(
                    grid_base, vp, min_area=cfg.min_surface_area)
                if table_z is None:
                    print(f"    [SKIP] {name} no support surface")
                    all_results[obj.obj_id] = PlacementResult(
                        obj_id=obj.obj_id, class_name=name,
                        original_aabb_world=np.zeros(6),
                        placements=[], num_raw_candidates=0)
                    continue

                # FFT 碰撞搜索
                candidates, meta, yaw_data = find_table_placements(
                    grid_base, obj.bbox3d_canonical, obj.pose_world, vp,
                    table_z, surface_mask,
                    safety_margin=cfg.safety_margin,
                    yaw_steps=cfg.yaw_steps,
                    use_gpu=self.use_gpu)
                n_raw = meta["valid_raw"]
                print(f"    Raw candidates: {n_raw}")

                # 稳定性过滤
                candidates = filter_stable_placements(
                    candidates, yaw_data, surface_mask,
                    min_support_ratio=cfg.min_support_ratio,
                    chunk_size=cfg.stability_chunk_size)
                n_stable = len(candidates)
                print(f"    After stability: {n_stable}")

                # 可见性过滤
                candidates = filter_visible_placements(
                    candidates, table_z + 1,
                    obj.bbox3d_canonical, obj.pose_world,
                    E_w2c, K, camera.img_w, camera.img_h,
                    vp, yaw_data, margin_px=cfg.vis_margin_px)
                n_vis = len(candidates)
                print(f"    After visibility: {n_vis}")

                # 遮挡过滤
                depth_buf = build_depth_buffer(
                    grid_base, vp, K, E_w2c,
                    camera.img_w, camera.img_h)
                candidates = filter_occluded_placements(
                    candidates, table_z + 1,
                    obj.bbox3d_canonical, obj.pose_world,
                    depth_buf, K, E_w2c, vp, yaw_data,
                    camera.img_w, camera.img_h,
                    occlusion_threshold=cfg.occlusion_threshold)
                n_occ = len(candidates)
                print(f"    After occlusion: {n_occ}")

                # DBSCAN 聚类
                reps, c_infos = cluster_placements(
                    candidates, grid_base, yaw_data,
                    table_z + 1, vp,
                    eps=cfg.dbscan_eps,
                    min_samples=cfg.dbscan_min_samples)
                print(f"    Clusters: {len(reps)}")

                # 构建结果
                from src.annotation.free_bbox.grid_ops import _get_bbox_corners
                corners_obj = _get_bbox_corners(obj.bbox3d_canonical)
                orig_world = transform_points(corners_obj, obj.pose_world)
                orig_aabb = np.concatenate([orig_world.min(0), orig_world.max(0)])

                result = PlacementResult(
                    obj_id=obj.obj_id,
                    class_name=name,
                    original_aabb_world=orig_aabb,
                    placements=c_infos,
                    num_raw_candidates=n_raw,
                    num_after_stability=n_stable,
                    num_after_visibility=n_vis,
                    num_after_occlusion=n_occ,
                )
                all_results[obj.obj_id] = result

                # 可视化
                if output_dir and save_vis and len(reps) > 0:
                    from src.annotation.free_bbox.visualize import (
                        save_placement_vis)
                    vis_dir = os.path.join(output_dir, f"placement_{name}")
                    os.makedirs(vis_dir, exist_ok=True)
                    vis_path = os.path.join(vis_dir, "placement_vis.png")
                    save_placement_vis(
                        scene.rgb, name, obj.bbox3d_canonical,
                        obj.pose_world, K, E_w2c, vp, cam_origin,
                        reps, c_infos, target_vox, grid_base,
                        vis_path, yaw_data, table_z + 1)

            finally:
                grid_restore_object(grid_base, target_vox, saved)

        # ── Step 6: 保存结果 ──────────────────────────────────────────────
        if output_dir:
            print("\n[6/6] Saving results ...")
            config_dict = {
                "voxel_size": cfg.voxel_size,
                "safety_margin": cfg.safety_margin,
                "yaw_steps": cfg.yaw_steps,
                "min_support_ratio": cfg.min_support_ratio,
                "occlusion_threshold": cfg.occlusion_threshold,
                "dbscan_eps": cfg.dbscan_eps,
                "gpu_used": self.use_gpu,
            }
            obj_results = {}
            for oid, res in all_results.items():
                obj_results[oid] = {
                    "class_name": res.class_name,
                    "num_raw": res.num_raw_candidates,
                    "num_stable": res.num_after_stability,
                    "num_visible": res.num_after_visibility,
                    "num_unoccluded": res.num_after_occlusion,
                    "clusters": res.placements,
                    "original_aabb_world": res.original_aabb_world,
                }
            save_placement_result(
                os.path.join(output_dir, "placement_result.json"),
                config_dict, obj_results)

        print(f"\n[DONE] Processed {len(all_results)} objects.")
        return all_results
