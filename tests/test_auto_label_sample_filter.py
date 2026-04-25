import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.auto_label import build_sample_id_filter, load_sample_ids_file


def test_load_sample_ids_file_ignores_blank_and_comment_lines(tmp_path):
    """
    用法: pytest tests/test_auto_label_sample_filter.py
    作用: 验证 sample_id 文件读取会忽略空行和注释行。
    输入: 临时 sample_id 文本文件。
    输出: 断言只返回有效 sample_id。
    """
    sample_ids_path = tmp_path / "sample_ids.txt"
    sample_ids_path.write_text(
        "\n# comment\nscene_0000_0000_obj_3_p000\n  scene_0000_0000_obj_8_p000  \n",
        encoding="utf-8",
    )

    sample_ids = load_sample_ids_file(sample_ids_path)

    assert sample_ids == [
        "scene_0000_0000_obj_3_p000",
        "scene_0000_0000_obj_8_p000",
    ]


def test_build_sample_id_filter_merges_cli_and_file_values(tmp_path):
    """
    用法: pytest tests/test_auto_label_sample_filter.py
    作用: 验证命令行和文件中的 sample_id 会合并去重。
    输入: CLI sample_id 列表和临时 sample_id 文件。
    输出: 断言返回合并后的过滤集合。
    """
    sample_ids_path = tmp_path / "sample_ids.txt"
    sample_ids_path.write_text(
        "scene_0000_0000_obj_8_p000\nscene_0000_0000_obj_1_p000\n",
        encoding="utf-8",
    )

    sample_id_filter = build_sample_id_filter(
        ["scene_0000_0000_obj_3_p000", "scene_0000_0000_obj_8_p000"],
        sample_ids_path,
    )

    assert sample_id_filter == {
        "scene_0000_0000_obj_1_p000",
        "scene_0000_0000_obj_3_p000",
        "scene_0000_0000_obj_8_p000",
    }


def test_build_sample_id_filter_returns_none_without_inputs():
    """
    用法: pytest tests/test_auto_label_sample_filter.py
    作用: 验证未指定 sample_id 时保持全量处理语义。
    输入: 无 CLI sample_id 且无文件。
    输出: 断言返回 None。
    """
    assert build_sample_id_filter(None, None) is None
