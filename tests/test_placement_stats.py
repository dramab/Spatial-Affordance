import json
from pathlib import Path

from src.utils.placement_stats import aggregate_statistics, export_statistics


def _write_sample_file(path: Path, scene_id: str, frame_id: str, samples: list) -> None:
    """
    用法: _write_sample_file(path, "scene01", "0000", samples)
    作用: 为测试生成 placement samples JSON 文件
    输入: path: Path；scene_id/frame_id: str；samples: list
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": "placement_samples/v1",
                "scene_id": scene_id,
                "frame_id": frame_id,
                "unit": "cm",
                "samples": samples,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def test_aggregate_statistics_counts_candidate_samples(tmp_path: Path) -> None:
    """
    用法: pytest tests/test_placement_stats.py
    作用: 验证候选框样本按 records 数量统计，而不是按物体数统计
    输入: tmp_path: Path，pytest 临时目录
    输出: None
    """
    dir_a = tmp_path / "out_a" / "samples"
    dir_b = tmp_path / "out_b" / "samples"

    _write_sample_file(
        dir_a / "scene01_0000.json",
        "scene01",
        "0000",
        [
            {"scene_id": "scene01", "frame_id": "0000", "object_id": "obj_1", "class_name": "cup"},
            {"scene_id": "scene01", "frame_id": "0000", "object_id": "obj_1", "class_name": "cup"},
            {"scene_id": "scene01", "frame_id": "0000", "object_id": "obj_2", "class_name": "bottle"},
        ],
    )
    _write_sample_file(
        dir_b / "scene02_0000.json",
        "scene02",
        "0000",
        [
            {"scene_id": "scene02", "frame_id": "0000", "object_id": "obj_7", "class_name": "cup"},
        ],
    )

    summary = aggregate_statistics([tmp_path / "out_a", tmp_path / "out_b"])

    assert summary["total_scenes"] == 2
    assert summary["total_categories"] == 2
    assert summary["total_samples"] == 4
    assert summary["source_file_count"] == 2
    assert summary["avg_samples_per_scene"] == 2.0

    cup_stats = next(row for row in summary["category_stats"] if row["category"] == "cup")
    assert cup_stats["sample_count"] == 3
    assert cup_stats["object_count"] == 2
    assert cup_stats["scene_count"] == 2


def test_export_statistics_writes_expected_files(tmp_path: Path) -> None:
    """
    用法: pytest tests/test_placement_stats.py
    作用: 验证统计结果能正常导出为表格、文本和图表
    输入: tmp_path: Path，pytest 临时目录
    输出: None
    """
    sample_dir = tmp_path / "out" / "samples"
    _write_sample_file(
        sample_dir / "scene03_0000.json",
        "scene03",
        "0000",
        [
            {"scene_id": "scene03", "frame_id": "0000", "object_id": "obj_1", "class_name": "apple"},
            {"scene_id": "scene03", "frame_id": "0000", "object_id": "obj_2", "class_name": "apple"},
            {"scene_id": "scene03", "frame_id": "0000", "object_id": "obj_3", "class_name": "pear"},
        ],
    )

    summary = aggregate_statistics([tmp_path / "out"])
    output_dir = tmp_path / "stats"
    outputs = export_statistics(summary, output_dir, top_k=5)

    expected_keys = {
        "summary_json",
        "category_csv",
        "scene_csv",
        "summary_txt",
        "category_bar",
        "category_pie",
        "scene_hist",
    }
    assert set(outputs.keys()) == expected_keys
    for path in outputs.values():
        assert path.exists()
