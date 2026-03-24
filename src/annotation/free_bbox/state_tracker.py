"""
src/annotation/free_bbox/state_tracker.py
-----------------------------------------
状态跟踪模块：管理帧处理状态，支持 OOM Kill 容错。

用法:
    from src.annotation.free_bbox.state_tracker import (
        mark_processing,
        mark_completed,
        mark_failed,
        recover_stale_processing,
        is_frame_failed,
        should_process_frame,
    )

    # 启动时恢复残留标记
    recover_stale_processing(output_root)

    # 检查是否应该处理某帧
    if should_process_frame(output_root, scene_id, frame_id, force=False):
        mark_processing(output_root, scene_id, frame_id)
        try:
            # 处理帧...
            mark_completed(output_root, scene_id, frame_id)
        except Exception as e:
            mark_failed(output_root, scene_id, frame_id, str(e))
"""

import json
import fcntl
import os
from pathlib import Path
from typing import Optional


# 状态文件名
STATE_FILENAME = "frame_status.json"


def _get_state_path(output_root: str) -> Path:
    """获取状态文件路径。"""
    return Path(output_root) / STATE_FILENAME


def _load_state(output_root: str) -> dict:
    """
    加载状态文件。

    输入:
        output_root: 输出根目录
    输出:
        dict: 状态字典，包含 processing/completed/failed 三个键
    """
    state_path = _get_state_path(output_root)
    if not state_path.exists():
        return {"processing": {}, "completed": {}, "failed": {}}

    try:
        with open(state_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # 文件损坏时返回空状态
        return {"processing": {}, "completed": {}, "failed": {}}


def _save_state(output_root: str, state: dict) -> None:
    """
    保存状态文件（带文件锁防止并发写入）。

    输入:
        output_root: 输出根目录
        state: 状态字典
    """
    state_path = _get_state_path(output_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用临时文件写入后重命名，保证原子性
    temp_path = state_path.with_suffix(".tmp")

    with open(temp_path, "w") as f:
        # 获取文件锁
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(state, f, indent=2, sort_keys=True)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # 原子重命名
    temp_path.rename(state_path)


def _get_scene_set(state: dict, status: str, scene_id: str) -> set:
    """
    获取某场景下某状态的帧集合。

    输入:
        state: 状态字典
        status: 状态类型 (processing/completed/failed)
        scene_id: 场景ID
    输出:
        set: 该场景下该状态的帧ID集合
    """
    return set(state.get(status, {}).get(scene_id, []))


def _set_scene_status(state: dict, status: str, scene_id: str, frame_id: str) -> None:
    """
    设置某帧的状态。

    输入:
        state: 状态字典
        status: 状态类型
        scene_id: 场景ID
        frame_id: 帧ID
    """
    if status not in state:
        state[status] = {}
    if scene_id not in state[status]:
        state[status][scene_id] = []
    if frame_id not in state[status][scene_id]:
        state[status][scene_id].append(frame_id)


def _remove_from_status(state: dict, status: str, scene_id: str, frame_id: str) -> None:
    """
    从某状态中移除某帧。

    输入:
        state: 状态字典
        status: 状态类型
        scene_id: 场景ID
        frame_id: 帧ID
    """
    if status in state and scene_id in state[status]:
        if frame_id in state[status][scene_id]:
            state[status][scene_id].remove(frame_id)
            # 清理空列表
            if not state[status][scene_id]:
                del state[status][scene_id]


def mark_processing(output_root: str, scene_id: str, frame_id: str) -> None:
    """
    标记帧开始处理。

    输入:
        output_root: 输出根目录
        scene_id: 场景ID
        frame_id: 帧ID
    """
    state = _load_state(output_root)
    # 从其他状态中移除
    _remove_from_status(state, "completed", scene_id, frame_id)
    _remove_from_status(state, "failed", scene_id, frame_id)
    # 添加到 processing
    _set_scene_status(state, "processing", scene_id, frame_id)
    _save_state(output_root, state)


def mark_completed(output_root: str, scene_id: str, frame_id: str) -> None:
    """
    标记帧处理完成。

    输入:
        output_root: 输出根目录
        scene_id: 场景ID
        frame_id: 帧ID
    """
    state = _load_state(output_root)
    # 从 processing 中移除
    _remove_from_status(state, "processing", scene_id, frame_id)
    # 添加到 completed
    _set_scene_status(state, "completed", scene_id, frame_id)
    _save_state(output_root, state)


def mark_failed(output_root: str, scene_id: str, frame_id: str, reason: str = "") -> None:
    """
    标记帧处理失败。

    输入:
        output_root: 输出根目录
        scene_id: 场景ID
        frame_id: 帧ID
        reason: 失败原因
    """
    state = _load_state(output_root)
    # 从 processing 中移除
    _remove_from_status(state, "processing", scene_id, frame_id)
    # 添加到 failed
    _set_scene_status(state, "failed", scene_id, frame_id)

    # 保存失败原因（单独存储）
    if "failed_reasons" not in state:
        state["failed_reasons"] = {}
    if scene_id not in state["failed_reasons"]:
        state["failed_reasons"][scene_id] = {}
    state["failed_reasons"][scene_id][frame_id] = reason

    _save_state(output_root, state)


def recover_stale_processing(output_root: str) -> list:
    """
    恢复残留的 processing 标记到 failed。
    用于程序启动时检测上次被 OOM kill 的帧。

    输入:
        output_root: 输出根目录
    输出:
        list: 被恢复的帧列表 [(scene_id, frame_id), ...]
    """
    state = _load_state(output_root)
    stale_frames = []

    processing = state.get("processing", {})
    for scene_id, frame_ids in list(processing.items()):
        for frame_id in frame_ids:
            stale_frames.append((scene_id, frame_id))
            # 移到 failed
            _set_scene_status(state, "failed", scene_id, frame_id)
            # 记录原因
            if "failed_reasons" not in state:
                state["failed_reasons"] = {}
            if scene_id not in state["failed_reasons"]:
                state["failed_reasons"][scene_id] = {}
            state["failed_reasons"][scene_id][frame_id] = "Process killed (OOM or interrupted)"

    # 清空 processing
    state["processing"] = {}

    if stale_frames:
        _save_state(output_root, state)

    return stale_frames


def is_frame_failed(output_root: str, scene_id: str, frame_id: str) -> bool:
    """
    检查帧是否在失败列表中。

    输入:
        output_root: 输出根目录
        scene_id: 场景ID
        frame_id: 帧ID
    输出:
        bool: 是否在失败列表中
    """
    state = _load_state(output_root)
    return frame_id in _get_scene_set(state, "failed", scene_id)


def is_frame_completed(output_root: str, scene_id: str, frame_id: str) -> bool:
    """
    检查帧是否已完成。

    输入:
        output_root: 输出根目录
        scene_id: 场景ID
        frame_id: 帧ID
    输出:
        bool: 是否已完成
    """
    state = _load_state(output_root)
    return frame_id in _get_scene_set(state, "completed", scene_id)


def is_frame_processing(output_root: str, scene_id: str, frame_id: str) -> bool:
    """
    检查帧是否正在处理中。

    输入:
        output_root: 输出根目录
        scene_id: 场景ID
        frame_id: 帧ID
    输出:
        bool: 是否正在处理中
    """
    state = _load_state(output_root)
    return frame_id in _get_scene_set(state, "processing", scene_id)


def should_process_frame(
    output_root: str,
    scene_id: str,
    frame_id: str,
    force: bool = False,
    retry_failed: bool = False,
) -> bool:
    """
    综合判断是否应该处理某帧。

    输入:
        output_root: 输出根目录
        scene_id: 场景ID
        frame_id: 帧ID
        force: 是否强制重新处理（包括已完成和失败的）
        retry_failed: 是否重试失败的帧
    输出:
        bool: 是否应该处理
    """
    if force:
        return True

    # 检查是否已完成（通过状态文件和结果文件）
    if is_frame_completed(output_root, scene_id, frame_id):
        return False

    # 检查是否正在处理中
    if is_frame_processing(output_root, scene_id, frame_id):
        return False

    # 检查是否在失败列表中
    if is_frame_failed(output_root, scene_id, frame_id):
        return retry_failed

    return True


def get_failed_frames(output_root: str) -> list:
    """
    获取所有失败的帧列表。

    输入:
        output_root: 输出根目录
    输出:
        list: 失败帧列表 [(scene_id, frame_id), ...]
    """
    state = _load_state(output_root)
    failed = []
    for scene_id, frame_ids in state.get("failed", {}).items():
        for frame_id in frame_ids:
            failed.append((scene_id, frame_id))
    return failed


def get_frame_status_summary(output_root: str) -> dict:
    """
    获取状态摘要统计。

    输入:
        output_root: 输出根目录
    输出:
        dict: 状态统计
    """
    state = _load_state(output_root)

    summary = {
        "processing": 0,
        "completed": 0,
        "failed": 0,
    }

    for status in ["processing", "completed", "failed"]:
        for frame_ids in state.get(status, {}).values():
            summary[status] += len(frame_ids)

    return summary


def clear_failed_status(output_root: str, scene_id: str = None, frame_id: str = None) -> int:
    """
    清除失败状态，用于手动重试前。

    输入:
        output_root: 输出根目录
        scene_id: 特定场景ID，为None则清除所有
        frame_id: 特定帧ID，为None则清除该场景所有
    输出:
        int: 清除的帧数
    """
    state = _load_state(output_root)
    cleared = 0

    if scene_id is None:
        # 清除所有失败状态
        cleared = sum(len(frames) for frames in state.get("failed", {}).values())
        state["failed"] = {}
        state["failed_reasons"] = {}
    elif frame_id is None:
        # 清除某场景的所有失败帧
        cleared = len(state.get("failed", {}).get(scene_id, []))
        if scene_id in state.get("failed", {}):
            del state["failed"][scene_id]
        if scene_id in state.get("failed_reasons", {}):
            del state["failed_reasons"][scene_id]
    else:
        # 清除特定帧
        _remove_from_status(state, "failed", scene_id, frame_id)
        if "failed_reasons" in state and scene_id in state["failed_reasons"]:
            if frame_id in state["failed_reasons"][scene_id]:
                del state["failed_reasons"][scene_id][frame_id]
        cleared = 1

    _save_state(output_root, state)
    return cleared
