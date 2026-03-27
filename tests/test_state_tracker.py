from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.state_tracker import (
    clear_failed_status,
    get_failed_frames,
    get_failed_reason,
    get_frame_status_summary,
    get_sample_output_paths,
    is_frame_failed,
    is_sample_complete,
    mark_failed,
    mark_processing,
    recover_stale_processing,
    should_process_frame,
)


def _write_complete_outputs(output_root: Path, scene_id: str, frame_id: str) -> None:
    """
    用法: _write_complete_outputs(output_root, scene_id, frame_id)
    作用: 为测试样本写出一整套核心结果文件
    输入: output_root: Path；scene_id/frame_id: str
    输出: None
    """
    for path in get_sample_output_paths(str(output_root), scene_id, frame_id).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npy":
            path.write_bytes(b"NUMPY")
        else:
            path.write_text("{}", encoding="utf-8")



def test_is_sample_complete_requires_all_core_outputs(tmp_path: Path) -> None:
    """
    用法: pytest tests/test_state_tracker.py
    作用: 验证只有核心结果文件全部存在时才视为样本完成
    输入: tmp_path: Path，pytest 临时目录
    输出: None
    """
    output_root = tmp_path / "placement_outputs"
    scene_id = "scene_0001"
    frame_id = "0000"
    output_paths = get_sample_output_paths(str(output_root), scene_id, frame_id)

    first_path = next(iter(output_paths.values()))
    first_path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_text("{}", encoding="utf-8")

    assert not is_sample_complete(str(output_root), scene_id, frame_id)

    _write_complete_outputs(output_root, scene_id, frame_id)

    assert is_sample_complete(str(output_root), scene_id, frame_id)



def test_recover_stale_processing_respects_complete_outputs(tmp_path: Path) -> None:
    """
    用法: pytest tests/test_state_tracker.py
    作用: 验证残留 running 恢复时，不会把结果已完整的样本误标为 failed
    输入: tmp_path: Path，pytest 临时目录
    输出: None
    """
    output_root = tmp_path / "placement_outputs"

    mark_processing(str(output_root), "scene_complete", "0001")
    _write_complete_outputs(output_root, "scene_complete", "0001")

    mark_processing(str(output_root), "scene_failed", "0002")

    stale_frames = recover_stale_processing(str(output_root))

    assert stale_frames == [("scene_failed", "0002")]
    assert not is_frame_failed(str(output_root), "scene_complete", "0001")
    assert is_frame_failed(str(output_root), "scene_failed", "0002")
    assert get_failed_reason(str(output_root), "scene_failed", "0002") == (
        "Process killed (OOM or interrupted)"
    )



def test_should_process_frame_prefers_complete_outputs_over_failed_markers(tmp_path: Path) -> None:
    """
    用法: pytest tests/test_state_tracker.py
    作用: 验证完整结果优先级高于 failed 标记，且失败样本支持 retry-failed 重试
    输入: tmp_path: Path，pytest 临时目录
    输出: None
    """
    output_root = tmp_path / "placement_outputs"

    _write_complete_outputs(output_root, "scene_done", "0001")
    mark_failed(str(output_root), "scene_done", "0001", "old failure")
    mark_failed(str(output_root), "scene_retry", "0002", "oom")

    assert not should_process_frame(str(output_root), "scene_done", "0001")
    assert not should_process_frame(str(output_root), "scene_retry", "0002")
    assert should_process_frame(
        str(output_root),
        "scene_retry",
        "0002",
        retry_failed=True,
    )

    summary = get_frame_status_summary(str(output_root))
    assert summary == {"processing": 0, "completed": 1, "failed": 2}
    assert get_failed_frames(str(output_root)) == [
        ("scene_done", "0001"),
        ("scene_retry", "0002"),
    ]

    cleared = clear_failed_status(str(output_root), "scene_done", "0001")
    assert cleared == 1
    assert get_failed_frames(str(output_root)) == [("scene_retry", "0002")]
