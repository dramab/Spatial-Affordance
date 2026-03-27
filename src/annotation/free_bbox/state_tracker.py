"""
src/annotation/free_bbox/state_tracker.py
-----------------------------------------
状态跟踪模块：使用每样本状态文件管理处理中和失败状态，
并以核心结果文件是否齐全作为最终完成依据。

用法:
    from src.annotation.free_bbox.state_tracker import (
        mark_processing,
        mark_completed,
        mark_failed,
        recover_stale_processing,
        is_frame_failed,
        is_sample_complete,
        should_process_frame,
    )

    recover_stale_processing(output_root)

    if should_process_frame(output_root, scene_id, frame_id, retry_failed=False):
        mark_processing(output_root, scene_id, frame_id)
        try:
            # 处理帧...
            mark_completed(output_root, scene_id, frame_id)
        except Exception as exc:
            mark_failed(output_root, scene_id, frame_id, str(exc))
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional


STATUS_DIRNAME = "status"
RUNNING_DIRNAME = "running"
FAILED_DIRNAME = "failed"
STATUS_SUFFIX = ".json"
CORE_OUTPUT_SPECS = (
    ("placements", ".json"),
    ("samples", ".json"),
    ("point_clouds", ".ply"),
    ("occupancy_grids", ".npy"),
    ("occupancy_grids", ".ply"),
    ("grid_meta", ".json"),
)


def _sanitize_filename(value: str) -> str:
    """
    用法: _sanitize_filename(value)
    作用: 将场景 ID 或帧 ID 转为稳定文件名片段
    输入: value: str
    输出: str
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return safe or "item"



def _make_sample_prefix(scene_id: str, frame_id: str) -> str:
    """
    用法: _make_sample_prefix(scene_id, frame_id)
    作用: 生成单个样本共享的文件名前缀
    输入: scene_id/frame_id: str
    输出: str
    """
    return f"{_sanitize_filename(scene_id)}_{_sanitize_filename(frame_id)}"



def _get_status_root(output_root: str) -> Path:
    """
    用法: _get_status_root(output_root)
    作用: 返回状态根目录路径
    输入: output_root: str
    输出: Path
    """
    return Path(output_root) / STATUS_DIRNAME



def _get_running_dir(output_root: str) -> Path:
    """
    用法: _get_running_dir(output_root)
    作用: 返回 running 状态目录
    输入: output_root: str
    输出: Path
    """
    return _get_status_root(output_root) / RUNNING_DIRNAME



def _get_failed_dir(output_root: str) -> Path:
    """
    用法: _get_failed_dir(output_root)
    作用: 返回 failed 状态目录
    输入: output_root: str
    输出: Path
    """
    return _get_status_root(output_root) / FAILED_DIRNAME



def _ensure_status_dirs(output_root: str) -> None:
    """
    用法: _ensure_status_dirs(output_root)
    作用: 创建状态目录
    输入: output_root: str
    输出: None
    """
    _get_running_dir(output_root).mkdir(parents=True, exist_ok=True)
    _get_failed_dir(output_root).mkdir(parents=True, exist_ok=True)



def _get_running_path(output_root: str, scene_id: str, frame_id: str) -> Path:
    """
    用法: _get_running_path(output_root, scene_id, frame_id)
    作用: 返回样本 running 标记路径
    输入: output_root/scene_id/frame_id: str
    输出: Path
    """
    prefix = _make_sample_prefix(scene_id, frame_id)
    return _get_running_dir(output_root) / f"{prefix}{STATUS_SUFFIX}"



def _get_failed_path(output_root: str, scene_id: str, frame_id: str) -> Path:
    """
    用法: _get_failed_path(output_root, scene_id, frame_id)
    作用: 返回样本 failed 标记路径
    输入: output_root/scene_id/frame_id: str
    输出: Path
    """
    prefix = _make_sample_prefix(scene_id, frame_id)
    return _get_failed_dir(output_root) / f"{prefix}{STATUS_SUFFIX}"



def get_sample_output_paths(output_root: str, scene_id: str, frame_id: str) -> dict:
    """
    用法: get_sample_output_paths(output_root, scene_id, frame_id)
    作用: 返回单个样本的核心输出文件路径映射
    输入: output_root/scene_id/frame_id: str
    输出: dict
    """
    root = Path(output_root)
    prefix = _make_sample_prefix(scene_id, frame_id)
    return {
        f"{subdir}_{suffix.lstrip('.')}": root / subdir / f"{prefix}{suffix}"
        for subdir, suffix in CORE_OUTPUT_SPECS
    }



def _write_json_atomic(path: Path, payload: dict) -> None:
    """
    用法: _write_json_atomic(path, payload)
    作用: 原子写入 JSON 文件，避免中途写坏状态文件
    输入: path: Path；payload: dict
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



def _read_json(path: Path) -> dict:
    """
    用法: _read_json(path)
    作用: 读取状态 JSON，损坏时回退为空字典
    输入: path: Path
    输出: dict
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}



def _build_status_payload(scene_id: str, frame_id: str, state: str, reason: str = "") -> dict:
    """
    用法: _build_status_payload(scene_id, frame_id, state, reason="")
    作用: 构造状态文件内容
    输入: scene_id/frame_id/state/reason: str
    输出: dict
    """
    payload = {
        "scene_id": scene_id,
        "frame_id": frame_id,
        "state": state,
    }
    if reason:
        payload["reason"] = reason
    return payload



def _iter_status_files(directory: Path):
    """
    用法: _iter_status_files(directory)
    作用: 迭代状态目录中的 JSON 文件
    输入: directory: Path
    输出: iterator[Path]
    """
    if not directory.exists():
        return []
    return sorted(directory.glob(f"*{STATUS_SUFFIX}"))



def _status_to_sample(path: Path) -> tuple[str, str]:
    """
    用法: _status_to_sample(path)
    作用: 从状态文件还原样本标识，优先读取文件内容
    输入: path: Path
    输出: tuple[str, str]
    """
    payload = _read_json(path)
    scene_id = payload.get("scene_id")
    frame_id = payload.get("frame_id")
    if scene_id is not None and frame_id is not None:
        return str(scene_id), str(frame_id)

    stem = path.stem
    scene_id, _, frame_id = stem.rpartition("_")
    return scene_id or stem, frame_id or ""



def is_sample_complete(output_root: str, scene_id: str, frame_id: str) -> bool:
    """
    用法: is_sample_complete(output_root, scene_id, frame_id)
    作用: 检查单个样本的核心结果文件是否全部存在
    输入: output_root/scene_id/frame_id: str
    输出: bool
    """
    output_paths = get_sample_output_paths(output_root, scene_id, frame_id)
    return all(path.exists() for path in output_paths.values())



def _list_completed_samples(output_root: str) -> list[tuple[str, str]]:
    """
    用法: _list_completed_samples(output_root)
    作用: 扫描输出目录中所有完整样本
    输入: output_root: str
    输出: list[tuple[str, str]]
    """
    placements_dir = Path(output_root) / "placements"
    if not placements_dir.exists():
        return []

    completed = []
    for placement_path in sorted(placements_dir.glob("*.json")):
        stem = placement_path.stem
        scene_id, _, frame_id = stem.rpartition("_")
        if not scene_id or not frame_id:
            continue
        if is_sample_complete(output_root, scene_id, frame_id):
            completed.append((scene_id, frame_id))
    return completed



def mark_processing(output_root: str, scene_id: str, frame_id: str) -> None:
    """
    用法: mark_processing(output_root, scene_id, frame_id)
    作用: 创建样本 running 标记，并清理旧 failed 标记
    输入: output_root/scene_id/frame_id: str
    输出: None
    """
    _ensure_status_dirs(output_root)
    failed_path = _get_failed_path(output_root, scene_id, frame_id)
    if failed_path.exists():
        failed_path.unlink()

    running_path = _get_running_path(output_root, scene_id, frame_id)
    payload = _build_status_payload(scene_id, frame_id, "running")
    _write_json_atomic(running_path, payload)



def mark_completed(output_root: str, scene_id: str, frame_id: str) -> None:
    """
    用法: mark_completed(output_root, scene_id, frame_id)
    作用: 清理样本的 running 和 failed 标记
    输入: output_root/scene_id/frame_id: str
    输出: None
    """
    running_path = _get_running_path(output_root, scene_id, frame_id)
    failed_path = _get_failed_path(output_root, scene_id, frame_id)
    if running_path.exists():
        running_path.unlink()
    if failed_path.exists():
        failed_path.unlink()



def mark_failed(output_root: str, scene_id: str, frame_id: str, reason: str = "") -> None:
    """
    用法: mark_failed(output_root, scene_id, frame_id, reason="")
    作用: 将样本标记为失败，并记录失败原因
    输入: output_root/scene_id/frame_id/reason: str
    输出: None
    """
    _ensure_status_dirs(output_root)
    running_path = _get_running_path(output_root, scene_id, frame_id)
    if running_path.exists():
        running_path.unlink()

    failed_path = _get_failed_path(output_root, scene_id, frame_id)
    payload = _build_status_payload(scene_id, frame_id, "failed", reason=reason)
    _write_json_atomic(failed_path, payload)



def recover_stale_processing(output_root: str) -> list:
    """
    用法: recover_stale_processing(output_root)
    作用: 将残留 running 标记恢复为 failed，已完整落盘的样本只清理 running 标记
    输入: output_root: str
    输出: list[(scene_id, frame_id)]
    """
    stale_frames = []
    for running_path in _iter_status_files(_get_running_dir(output_root)):
        scene_id, frame_id = _status_to_sample(running_path)
        if is_sample_complete(output_root, scene_id, frame_id):
            running_path.unlink()
            continue

        stale_frames.append((scene_id, frame_id))
        payload = _build_status_payload(
            scene_id,
            frame_id,
            "failed",
            reason="Process killed (OOM or interrupted)",
        )
        _write_json_atomic(_get_failed_path(output_root, scene_id, frame_id), payload)
        running_path.unlink()

    return stale_frames



def is_frame_failed(output_root: str, scene_id: str, frame_id: str) -> bool:
    """
    用法: is_frame_failed(output_root, scene_id, frame_id)
    作用: 检查样本是否存在 failed 标记
    输入: output_root/scene_id/frame_id: str
    输出: bool
    """
    return _get_failed_path(output_root, scene_id, frame_id).exists()



def is_frame_completed(output_root: str, scene_id: str, frame_id: str) -> bool:
    """
    用法: is_frame_completed(output_root, scene_id, frame_id)
    作用: 检查样本核心结果是否完整
    输入: output_root/scene_id/frame_id: str
    输出: bool
    """
    return is_sample_complete(output_root, scene_id, frame_id)



def is_frame_processing(output_root: str, scene_id: str, frame_id: str) -> bool:
    """
    用法: is_frame_processing(output_root, scene_id, frame_id)
    作用: 检查样本是否存在 running 标记
    输入: output_root/scene_id/frame_id: str
    输出: bool
    """
    return _get_running_path(output_root, scene_id, frame_id).exists()



def should_process_frame(
    output_root: str,
    scene_id: str,
    frame_id: str,
    force: bool = False,
    retry_failed: bool = False,
) -> bool:
    """
    用法: should_process_frame(output_root, scene_id, frame_id, force=False, retry_failed=False)
    作用: 综合判断样本是否应加入本轮处理列表
    输入: output_root/scene_id/frame_id: str；force/retry_failed: bool
    输出: bool
    """
    if force:
        return True

    if is_sample_complete(output_root, scene_id, frame_id):
        return False

    if is_frame_processing(output_root, scene_id, frame_id):
        return False

    if is_frame_failed(output_root, scene_id, frame_id):
        return retry_failed

    return True



def get_failed_frames(output_root: str) -> list:
    """
    用法: get_failed_frames(output_root)
    作用: 返回所有 failed 样本列表
    输入: output_root: str
    输出: list[(scene_id, frame_id)]
    """
    failed_frames = []
    for failed_path in _iter_status_files(_get_failed_dir(output_root)):
        failed_frames.append(_status_to_sample(failed_path))
    return failed_frames



def get_failed_reason(output_root: str, scene_id: str, frame_id: str) -> str:
    """
    用法: get_failed_reason(output_root, scene_id, frame_id)
    作用: 读取指定样本的失败原因
    输入: output_root/scene_id/frame_id: str
    输出: str
    """
    payload = _read_json(_get_failed_path(output_root, scene_id, frame_id))
    return str(payload.get("reason", ""))



def get_frame_status_summary(output_root: str) -> dict:
    """
    用法: get_frame_status_summary(output_root)
    作用: 汇总 running/completed/failed 的样本数量
    输入: output_root: str
    输出: dict
    """
    return {
        "processing": len(list(_iter_status_files(_get_running_dir(output_root)))),
        "completed": len(_list_completed_samples(output_root)),
        "failed": len(list(_iter_status_files(_get_failed_dir(output_root)))),
    }



def clear_failed_status(output_root: str, scene_id: Optional[str] = None,
                        frame_id: Optional[str] = None) -> int:
    """
    用法: clear_failed_status(output_root, scene_id=None, frame_id=None)
    作用: 清除失败标记，支持按全部、场景或单帧过滤
    输入: output_root: str；scene_id/frame_id: Optional[str]
    输出: int
    """
    cleared = 0
    for failed_path in _iter_status_files(_get_failed_dir(output_root)):
        cur_scene_id, cur_frame_id = _status_to_sample(failed_path)
        if scene_id is not None and cur_scene_id != scene_id:
            continue
        if frame_id is not None and cur_frame_id != frame_id:
            continue
        failed_path.unlink()
        cleared += 1
    return cleared
