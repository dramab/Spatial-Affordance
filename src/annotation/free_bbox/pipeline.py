"""
src/annotation/free_bbox/pipeline.py
--------------------------------------
放置规划 Pipeline：编排完整的放置规划流程。

流程:
    SceneData → 点云 → 占据栅格 → 物体体素化 → 逐物体处理:
        支撑面检测 → FFT 碰撞搜索 → 稳定性过滤 → 可见性过滤 →
        遮挡过滤 → DBSCAN 聚类 → 训练标注导出 / 可视化

用法:
    from src.annotation.free_bbox.pipeline import PlacementPipeline
    from src.annotation.free_bbox.datatypes import PlacementConfig

    pipeline = PlacementPipeline(PlacementConfig())
    results = pipeline.run(scene_data, output_dir="outputs/placement")
"""

import os
import re
import numpy as np

from src.annotation.free_bbox.datatypes import (
    SceneData, PlacementConfig, PlacementResult,
)
from src.annotation.free_bbox.occupancy import (
    depth_to_pointcloud, build_occupancy_grid, FREE, OCCUPIED, UNKNOWN,
)
from src.annotation.free_bbox.voxel_utils import make_voxel_params
from src.annotation.free_bbox.grid_ops import (
    voxelize_obb, prepare_grid_base, _get_bbox_corners,
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
    save_ply, save_occupancy_ply, save_placement_annotations,
    save_placement_samples, save_grid_meta,
)
from src.utils.coord_utils import transform_points


_REQUIRED_OUTPUT_DIRS = (
    "placements",
    "samples",
    "point_clouds",
    "occupancy_grids",
    "grid_meta",
)


def _sanitize_filename(value: str) -> str:
    """将标识转换为稳定的文件名片段。"""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return safe or "item"


def _make_scene_prefix(scene_id: str, frame_id: str) -> str:
    """生成同一帧所有输出共享的文件名前缀。"""
    return f"{_sanitize_filename(scene_id)}_{_sanitize_filename(frame_id)}"


def _build_output_paths(output_dir: str, scene_id: str, frame_id: str,
                        save_vis: bool) -> dict:
    """构建按文件类型分目录的输出路径。"""
    root = os.path.abspath(output_dir)
    prefix = _make_scene_prefix(scene_id, frame_id)

    paths = {
        "prefix": prefix,
        "placements": os.path.join(root, "placements", f"{prefix}.json"),
        "samples": os.path.join(root, "samples", f"{prefix}.json"),
        "point_cloud": os.path.join(root, "point_clouds", f"{prefix}.ply"),
        "occupancy_grid_npy": os.path.join(root, "occupancy_grids", f"{prefix}.npy"),
        "occupancy_grid_ply": os.path.join(root, "occupancy_grids", f"{prefix}.ply"),
        "grid_meta": os.path.join(root, "grid_meta", f"{prefix}.json"),
    }

    dirs = {os.path.join(root, name) for name in _REQUIRED_OUTPUT_DIRS}
    if save_vis:
        vis_dir = os.path.join(root, "visualizations")
        paths["visualizations"] = vis_dir
        dirs.add(vis_dir)

    for directory in sorted(dirs):
        os.makedirs(directory, exist_ok=True)

    return paths


def _compute_placed_transform(anchor_xy, landing_z, yaw_data, yaw_idx, vp):
    """根据聚类代表恢复该放置样本的 4x4 object→world 变换。"""
    vs = float(vp["voxel_size"])
    T_rot = yaw_data["T_rotated"][yaw_idx]
    vmin_rot = yaw_data["vmin_rot_abs"][yaw_idx]

    anchor_3d = np.array([anchor_xy[0], anchor_xy[1], landing_z],
                         dtype=np.float64)
    delta_world = (anchor_3d - vmin_rot) * vs

    T_placed = T_rot.copy()
    T_placed[:3, 3] += delta_world
    return T_placed


def _build_saved_placements(scene_prefix, obj_id, bbox3d_canonical,
                            reps, cluster_infos, landing_z, yaw_data, vp):
    """将聚类结果转换成训练友好的 placement 样本。"""
    corners_obj = _get_bbox_corners(bbox3d_canonical)
    placements = []

    for rank, (rep, info) in enumerate(zip(reps, cluster_infos)):
        yaw_idx = int(rep[2])
        T_placed = _compute_placed_transform(
            rep[:2], landing_z, yaw_data, yaw_idx, vp)
        placed_world = transform_points(corners_obj, T_placed)
        aabb_world = np.concatenate([
            placed_world.min(axis=0),
            placed_world.max(axis=0),
        ])
        center_world = placed_world.mean(axis=0)

        placements.append({
            "sample_id": f"{scene_prefix}_{obj_id}_p{rank:03d}",
            "rank": int(rank),
            "center_world": center_world.tolist(),
            "yaw_degrees": float(info["yaw_degrees"]),
            "transform_world": T_placed.tolist(),
            "aabb_world": aabb_world.tolist(),
            "free_space_score": int(info.get("free_score", info.get("centroid_distance", 0))),
        })

    return placements


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
            save_vis: bool = True) -> dict:
        """
        对场景执行完整放置规划。

        输入:
            scene: SceneData 场景数据
            output_dir: str 输出根目录（None 则不保存文件）
            save_vis: bool 是否保存可视化
        输出:
            dict {obj_id: PlacementResult} 每个物体的放置结果
        """
        cfg = self.config
        camera = scene.camera
        R_c2w = camera.E_c2w[:3, :3]
        cam_origin = camera.E_c2w[:3, 3]
        K = camera.K
        E_w2c = camera.E_w2c
        output_paths = None
        scene_prefix = _make_scene_prefix(scene.scene_id, scene.frame_id)

        if output_dir:
            output_paths = _build_output_paths(
                output_dir, scene.scene_id, scene.frame_id, save_vis)
            scene_prefix = output_paths["prefix"]

        # ── Step 1: 点云生成 ──────────────────────────────────────────────
        print("[1/6] Generating point cloud ...")
        pts_world, colors = depth_to_pointcloud(
            scene.depth, scene.rgb,
            camera.fx, camera.fy, camera.cx, camera.cy,
            R_c2w, cam_origin, stride=cfg.pixel_stride)

        if output_paths:
            save_ply(output_paths["point_cloud"], pts_world, colors)

        # ── Step 2: 占据栅格构建 ──────────────────────────────────────────
        print("[2/6] Building occupancy grid ...")
        grid, grid_min, vs = build_occupancy_grid(
            scene.depth,
            camera.fx, camera.fy, camera.cx, camera.cy,
            R_c2w, cam_origin,
            voxel_size=cfg.voxel_size, stride=cfg.pixel_stride,
            padding=cfg.grid_padding,
            surface_points=pts_world)

        vp = make_voxel_params(grid_min, vs)

        if output_paths:
            np.save(output_paths["occupancy_grid_npy"], grid)
            counts = {
                "free": int((grid == FREE).sum()),
                "occupied": int((grid == OCCUPIED).sum()),
                "unknown": int((grid == UNKNOWN).sum()),
            }
            save_grid_meta(
                output_paths["grid_meta"],
                vp,
                grid.shape,
                voxel_counts=counts,
                extra={
                    "scene_id": scene.scene_id,
                    "frame_id": scene.frame_id,
                    "unit": scene.unit,
                })
            save_occupancy_ply(
                output_paths["occupancy_grid_ply"],
                grid,
                grid_min,
                vs,
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

            corners_obj = _get_bbox_corners(obj.bbox3d_canonical)
            orig_world = transform_points(corners_obj, obj.pose_world)
            orig_aabb = np.concatenate([orig_world.min(0), orig_world.max(0)])

            # 目标物体体素化（用于预检查时移除自遮挡，并复用到结果对齐）
            target_vox = voxelize_obb(
                obj.bbox3d_canonical, obj.pose_world, vp,
                np.array(grid_base.shape))
            if len(target_vox) == 0:
                print(f"    [SKIP] {name} no voxels")
                all_results[obj.obj_id] = PlacementResult(
                    obj_id=obj.obj_id,
                    class_name=name,
                    original_aabb_world=orig_aabb,
                    placements=[],
                    num_raw_candidates=0,
                )
                continue

            grid_other = grid_base.copy()
            grid_other[target_vox[:, 0], target_vox[:, 1], target_vox[:, 2]] = FREE
            depth_buf = build_depth_buffer(
                grid_other, vp, K, E_w2c,
                camera.img_w, camera.img_h)

            # 可见性预检查：既要在图内，也不能被其他几何遮挡
            pose_cam = E_w2c @ obj.pose_world
            if not is_fully_visible(obj.bbox3d_canonical, pose_cam,
                                    camera.fx, camera.fy,
                                    camera.cx, camera.cy,
                                    camera.img_w, camera.img_h,
                                    depth_buffer=depth_buf,
                                    depth_margin=vs):
                print(f"    [SKIP] {name} not fully visible")
                all_results[obj.obj_id] = PlacementResult(
                    obj_id=obj.obj_id,
                    class_name=name,
                    original_aabb_world=orig_aabb,
                    placements=[],
                    num_raw_candidates=0,
                )
                continue

            # 支撑面检测移除当前物体，避免把物体自身表面误判为最近支撑面
            table_z, surface_mask = detect_support_surfaces(
                grid_other, vp,
                min_area=cfg.min_surface_area,
                points_world=pts_world,
                target_voxels=target_vox)
            if table_z is None:
                print(f"    [SKIP] {name} no support surface")
                all_results[obj.obj_id] = PlacementResult(
                    obj_id=obj.obj_id,
                    class_name=name,
                    original_aabb_world=orig_aabb,
                    placements=[],
                    num_raw_candidates=0,
                )
                continue

            # FFT 碰撞搜索
            candidates, meta, yaw_data = find_table_placements(
                grid_base, obj.bbox3d_canonical, obj.pose_world, vp,
                table_z, surface_mask,
                safety_margin=cfg.safety_margin,
                yaw_steps=cfg.yaw_steps,
                use_gpu=self.use_gpu,
                preserve_orientation=cfg.preserve_orientation,
                orientation_threshold_deg=cfg.orientation_threshold_deg)
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
                vp, yaw_data)
            n_vis = len(candidates)
            print(f"    After visibility: {n_vis}")

            # 遮挡过滤
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

            placements = _build_saved_placements(
                scene_prefix,
                obj.obj_id,
                obj.bbox3d_canonical,
                reps,
                c_infos,
                table_z + 1,
                yaw_data,
                vp,
            )

            result = PlacementResult(
                obj_id=obj.obj_id,
                class_name=name,
                original_aabb_world=orig_aabb,
                placements=placements,
                num_raw_candidates=n_raw,
                num_after_stability=n_stable,
                num_after_visibility=n_vis,
                num_after_occlusion=n_occ,
            )
            all_results[obj.obj_id] = result

            if output_paths and save_vis and len(reps) > 0:
                from src.annotation.free_bbox.visualize import save_placement_vis

                vis_name = (
                    f"{scene_prefix}_{_sanitize_filename(obj.obj_id)}_"
                    f"{_sanitize_filename(name)}.png"
                )
                vis_path = os.path.join(output_paths["visualizations"], vis_name)
                save_placement_vis(
                    scene.rgb, name, obj.bbox3d_canonical,
                    obj.pose_world, K, E_w2c, vp, cam_origin,
                    reps, grid_base,
                    vis_path, yaw_data, table_z + 1,
                    surface_mask=surface_mask)

        # ── Step 6: 保存结果 ──────────────────────────────────────────────
        if output_paths:
            print("\n[6/6] Saving results ...")
            placement_objects = []
            sample_records = []

            for obj in scene.objects:
                res = all_results.get(obj.obj_id)
                if res is None or len(res.placements) == 0:
                    continue

                placement_objects.append({
                    "object_id": obj.obj_id,
                    "class_name": obj.class_name,
                    "canonical_aabb_object": obj.bbox3d_canonical,
                    "original_pose_world": obj.pose_world,
                    "original_aabb_world": res.original_aabb_world,
                    "placements": res.placements,
                })

                for placement in res.placements:
                    sample_records.append({
                        "sample_id": placement["sample_id"],
                        "scene_id": scene.scene_id,
                        "frame_id": scene.frame_id,
                        "unit": scene.unit,
                        "object_id": obj.obj_id,
                        "class_name": obj.class_name,
                        "rank": placement["rank"],
                        "canonical_aabb_object": obj.bbox3d_canonical,
                        "original_pose_world": obj.pose_world,
                        "original_aabb_world": res.original_aabb_world,
                        "center_world": placement["center_world"],
                        "yaw_degrees": placement["yaw_degrees"],
                        "transform_world": placement["transform_world"],
                        "aabb_world": placement["aabb_world"],
                        "free_space_score": placement["free_space_score"],
                    })

            save_placement_annotations(
                output_paths["placements"],
                {
                    "schema_version": "placement_annotations/v1",
                    "scene_id": scene.scene_id,
                    "frame_id": scene.frame_id,
                    "unit": scene.unit,
                    "objects": placement_objects,
                })
            save_placement_samples(
                output_paths["samples"],
                {
                    "schema_version": "placement_samples/v1",
                    "scene_id": scene.scene_id,
                    "frame_id": scene.frame_id,
                    "unit": scene.unit,
                    "samples": sample_records,
                })

        print(f"\n[DONE] Processed {len(all_results)} objects.")
        return all_results
