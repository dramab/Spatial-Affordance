#!/usr/bin/env python3
"""
tools/build_benchmark_site.py
-----------------------------
职责：将 placement benchmark 评测结果和推理可视化图片合并为静态展示网站。

用法：
    conda run -n spatial python tools/build_benchmark_site.py \
        --eval-dir outputs/infer_ptv3_benchmark_eval \
        --infer-dir outputs/infer_ptv3

作用：
    - 读取 eval_dir/per_sample_metrics.json 与 eval_dir/metrics_summary.json
    - 读取 infer_dir/predictions.json 中每个 sample_id 对应的 vis_path
    - 生成支持搜索、筛选、排序、图片放大和详情查看的静态网站
    - 网站直接引用 infer_dir/vis 下的图片，不复制图片文件

输入：
    --eval-dir: benchmark 评测输出目录，包含 per_sample_metrics.json 和 metrics_summary.json
    --infer-dir: 推理输出目录，包含 predictions.json 和 vis/
    --output-dir: 可选，网站输出目录；默认 eval_dir/visualization
    --title: 可选，网页标题

输出：
    output_dir/
        - index.html
        - assets/style.css
        - assets/app.js
        - assets/benchmark-data.js

使用示例：
    conda run -n spatial python tools/build_benchmark_site.py \
        --eval-dir outputs/infer_ptv3_benchmark_eval \
        --infer-dir outputs/infer_ptv3

预览示例：
    cd outputs/infer_ptv3_benchmark_eval
    python -m http.server 8000
    # 浏览器打开 http://localhost:8000/visualization/
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PER_SAMPLE_FILE = "per_sample_metrics.json"
METRICS_SUMMARY_FILE = "metrics_summary.json"
PREDICTIONS_FILE = "predictions.json"
SITE_SCHEMA_VERSION = "placement_benchmark_site/v1"


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建 benchmark 可视化网站生成脚本的命令行参数
    输入: 无
    输出: argparse.ArgumentParser，配置完成的解析器
    """
    parser = argparse.ArgumentParser(description="生成 placement benchmark 静态可视化网站")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("outputs/infer_ptv3_benchmark_eval"),
        help="benchmark 评测输出目录，默认 outputs/infer_ptv3_benchmark_eval",
    )
    parser.add_argument(
        "--infer-dir",
        type=Path,
        default=Path("outputs/infer_ptv3"),
        help="推理输出目录，默认 outputs/infer_ptv3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="网站输出目录；默认 eval-dir/visualization",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Placement Benchmark Visualization",
        help="网页标题",
    )
    return parser


def resolve_project_path(path_value: str | Path) -> Path:
    """
    用法: path = resolve_project_path("outputs/demo")
    作用: 将相对仓库路径转换为绝对路径
    输入: path_value: str | Path，相对或绝对路径
    输出: Path，绝对路径
    """
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json_object(json_path: Path) -> dict[str, Any]:
    """
    用法: payload = load_json_object(Path("metrics_summary.json"))
    作用: 读取 JSON 文件并校验顶层是对象
    输入: json_path: Path，JSON 文件路径
    输出: dict，JSON 顶层对象
    """
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {json_path}")
    return payload


def write_text_file(output_path: Path, content: str) -> None:
    """
    用法: write_text_file(Path("site/index.html"), html)
    作用: 写入文本文件，并自动创建父目录
    输入: output_path: Path，输出路径；content: str，文本内容
    输出: None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def path_to_record(path_value: Path) -> str:
    """
    用法: text = path_to_record(Path("outputs/demo/file.json"))
    作用: 将绝对路径压缩为仓库相对路径，方便前端展示
    输入: path_value: Path，待记录路径
    输出: str，相对仓库路径或绝对路径
    """
    resolved_path = path_value.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def path_from_site(path_value: Path, site_dir: Path) -> str:
    """
    用法: rel = path_from_site(image_path, site_dir)
    作用: 生成网站目录到目标文件的相对路径
    输入: path_value: Path，目标文件；site_dir: Path，网站目录
    输出: str，POSIX 相对路径
    """
    relative_path = os.path.relpath(path_value.resolve(), start=site_dir.resolve())
    return Path(relative_path).as_posix()


def load_eval_payloads(eval_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """
    用法: summary, per_sample, samples = load_eval_payloads(eval_dir)
    作用: 读取 benchmark 汇总指标和逐样本指标
    输入: eval_dir: Path，评测输出目录
    输出: tuple，汇总载荷、逐样本载荷、样本列表
    """
    summary_path = eval_dir / METRICS_SUMMARY_FILE
    per_sample_path = eval_dir / PER_SAMPLE_FILE
    if not summary_path.exists():
        raise FileNotFoundError(f"未找到汇总指标文件: {summary_path}")
    if not per_sample_path.exists():
        raise FileNotFoundError(f"未找到逐样本指标文件: {per_sample_path}")

    summary_payload = load_json_object(summary_path)
    per_sample_payload = load_json_object(per_sample_path)
    samples = per_sample_payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{per_sample_path} 缺少 list 类型的 samples 字段")
    return summary_payload, per_sample_payload, samples


def load_prediction_lookup(infer_dir: Path) -> dict[str, dict[str, Any]]:
    """
    用法: lookup = load_prediction_lookup(infer_dir)
    作用: 读取 predictions.json，并按 sample_id 构建快速查询表
    输入: infer_dir: Path，推理输出目录
    输出: dict，键为 sample_id，值为 prediction 记录
    """
    prediction_path = infer_dir / PREDICTIONS_FILE
    if not prediction_path.exists():
        raise FileNotFoundError(f"未找到推理结果文件: {prediction_path}")

    payload = load_json_object(prediction_path)
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"{prediction_path} 缺少 list 类型的 predictions 字段")

    lookup: dict[str, dict[str, Any]] = {}
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            continue
        sample_id = str(prediction.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"prediction 缺少 sample_id: index={index}")
        lookup[sample_id] = prediction
    return lookup


def resolve_visual_path(prediction: dict[str, Any], infer_dir: Path) -> Path:
    """
    用法: image_path = resolve_visual_path(prediction, infer_dir)
    作用: 根据 prediction.vis_path 或默认命名规则定位可视化图片
    输入: prediction: dict，单条推理结果；infer_dir: Path，推理目录
    输出: Path，可视化图片绝对路径
    """
    vis_path_value = prediction.get("vis_path")
    if vis_path_value:
        return resolve_project_path(str(vis_path_value))

    source_name = str(prediction.get("source_name", "")).strip()
    sample_id = str(prediction.get("sample_id", "")).strip()
    return infer_dir / "vis" / f"{source_name}__{sample_id}.png"


def get_nested_bool(payload: dict[str, Any], section: str, key: str) -> bool:
    """
    用法: ok = get_nested_bool(sample, "status", "placement_success")
    作用: 从嵌套字典中安全读取布尔值
    输入: payload: dict，样本对象；section/key: str，字段位置
    输出: bool，字段为 True 时返回 True，否则返回 False
    """
    section_value = payload.get(section)
    if not isinstance(section_value, dict):
        return False
    return bool(section_value.get(key))


def get_nested_number(payload: dict[str, Any], section: str, key: str) -> float | None:
    """
    用法: value = get_nested_number(sample, "collision", "occupied_collision_ratio")
    作用: 从嵌套字典中安全读取数值
    输入: payload: dict，样本对象；section/key: str，字段位置
    输出: float | None，读取失败时返回 None
    """
    section_value = payload.get(section)
    if not isinstance(section_value, dict):
        return None
    value = section_value.get(key)
    return float(value) if isinstance(value, int | float) else None


def summarize_sample(sample: dict[str, Any], prediction: dict[str, Any], image_path: Path, site_dir: Path) -> dict[str, Any]:
    """
    用法: item = summarize_sample(sample, prediction, image_path, site_dir)
    作用: 合并单个样本的评测指标、推理路径和前端筛选字段
    输入: sample: dict，评测样本；prediction: dict，推理记录；image_path/site_dir: Path，路径上下文
    输出: dict，前端展示条目
    """
    collision = sample.get("collision") if isinstance(sample.get("collision"), dict) else {}
    direction = sample.get("direction") if isinstance(sample.get("direction"), dict) else {}
    size = sample.get("size") if isinstance(sample.get("size"), dict) else {}
    object_center = sample.get("object_center") if isinstance(sample.get("object_center"), dict) else {}
    status = sample.get("status") if isinstance(sample.get("status"), dict) else {}
    errors = sample.get("errors") if isinstance(sample.get("errors"), list) else []

    return {
        "sample_id": str(sample.get("sample_id", "")),
        "source_name": str(sample.get("source_name", "")),
        "scene_id": str(sample.get("scene_id", "")),
        "frame_id": str(sample.get("frame_id", "")),
        "object_id": str(sample.get("object_id", "")),
        "reference_object_id": str(direction.get("reference_object_id", "")),
        "reference_name": str(direction.get("reference_name", "")),
        "expected_relation": str(direction.get("expected_relation", "")),
        "pred_relation": str(direction.get("pred_relation", "")),
        "image_path": path_from_site(image_path, site_dir),
        "image_file": image_path.name,
        "vis_path": path_to_record(image_path),
        "rgb_path": str(prediction.get("rgb_path", "")),
        "has_image": image_path.exists(),
        "full_metric_evaluated": bool(status.get("full_metric_evaluated")),
        "overall_metric_evaluated": bool(status.get("overall_metric_evaluated")),
        "placement_success": bool(status.get("placement_success")),
        "overall_success": bool(status.get("overall_success")),
        "collision_evaluated": bool(status.get("collision_evaluated")),
        "collision_free": bool(collision.get("collision_free")),
        "occupied_collision_ratio": collision.get("occupied_collision_ratio"),
        "unknown_overlap_ratio": collision.get("unknown_overlap_ratio"),
        "collision_ratio_threshold": collision.get("collision_ratio_threshold"),
        "direction_evaluated": bool(status.get("direction_evaluated")),
        "direction_correct": bool(direction.get("direction_correct")),
        "center_match": bool(direction.get("center_match")),
        "center_l2_error_cm": direction.get("center_l2_error_cm"),
        "center_l2_threshold_cm": direction.get("center_l2_threshold_cm"),
        "size_evaluated": bool(status.get("size_evaluated")),
        "size_consistent": bool(size.get("size_consistent")),
        "pred_volume_cm3": size.get("pred_volume_cm3"),
        "target_volume_cm3": size.get("target_volume_cm3"),
        "volume_error_cm3": size.get("volume_error_cm3"),
        "volume_error_ratio": size.get("volume_error_ratio"),
        "volume_error_ratio_threshold": size.get("volume_error_ratio_threshold"),
        "object_center_evaluated": bool(status.get("object_center_evaluated")),
        "object_center_match": bool(object_center.get("center_match")),
        "object_center_l2_error_cm": object_center.get("center_l2_error_cm"),
        "object_center_l2_threshold_cm": object_center.get("center_l2_threshold_cm"),
        "pred_box_world": prediction.get("pred_box_world", []),
        "gt_box_world": prediction.get("gt_box_world", []),
        "pred_object_center_world": prediction.get("pred_object_center_world", []),
        "gt_object_center_world": prediction.get("gt_object_center_world", []),
        "errors": [str(error) for error in errors],
    }


def build_site_items(
        samples: list[dict[str, Any]],
        prediction_lookup: dict[str, dict[str, Any]],
        infer_dir: Path,
        site_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    用法: items, counts = build_site_items(samples, lookup, infer_dir, site_dir)
    作用: 按 sample_id 合并评测样本与推理图片，并统计缺失情况
    输入: samples: list[dict]，评测样本；prediction_lookup: dict，推理查询表；infer_dir/site_dir: Path
    输出: tuple，展示条目列表和缺失计数字典
    """
    items: list[dict[str, Any]] = []
    missing_prediction_count = 0
    missing_image_count = 0

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id", "")).strip()
        if not sample_id:
            continue

        prediction = prediction_lookup.get(sample_id)
        if prediction is None:
            missing_prediction_count += 1
            prediction = {"sample_id": sample_id, "source_name": sample.get("source_name", "")}

        image_path = resolve_visual_path(prediction, infer_dir)
        if not image_path.exists():
            missing_image_count += 1

        item = summarize_sample(sample, prediction, image_path, site_dir)
        item["index"] = len(items) + 1
        items.append(item)

    return items, {
        "missing_prediction_count": missing_prediction_count,
        "missing_image_count": missing_image_count,
    }


def build_site_payload(
        summary_payload: dict[str, Any],
        per_sample_payload: dict[str, Any],
        items: list[dict[str, Any]],
        eval_dir: Path,
        infer_dir: Path,
        title: str,
        missing_counts: dict[str, int]) -> dict[str, Any]:
    """
    用法: payload = build_site_payload(summary, per_sample, items, eval_dir, infer_dir, title, counts)
    作用: 组织写入前端数据文件的完整载荷
    输入: summary_payload/per_sample_payload: dict，评测载荷；items: list[dict]，展示条目；
         eval_dir/infer_dir: Path，路径信息；title: str，标题；missing_counts: dict，缺失统计
    输出: dict，前端数据载荷
    """
    sources = sorted({item["source_name"] for item in items})
    return {
        "schema_version": SITE_SCHEMA_VERSION,
        "title": title,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": {
            "eval_dir": path_to_record(eval_dir),
            "infer_dir": path_to_record(infer_dir),
            "sample_schema_version": str(per_sample_payload.get("schema_version", "")),
            "summary_schema_version": str(summary_payload.get("schema_version", "")),
            "sample_count": len(items),
            "sources": sources,
            **missing_counts,
        },
        "summary": summary_payload.get("summary", {}),
        "by_source": summary_payload.get("by_source", {}),
        "direction_confusion": summary_payload.get("direction_confusion", {}),
        "thresholds": summary_payload.get("thresholds", {}),
        "items": items,
    }


def json_to_script(payload: dict[str, Any]) -> str:
    """
    用法: script = json_to_script(payload)
    作用: 将 Python 数据序列化为浏览器可加载的 JS 数据文件
    输入: payload: dict，前端数据载荷
    输出: str，JS 文本
    """
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"window.BENCHMARK_DATA = {json_text};\n"


def render_html(title: str) -> str:
    """
    用法: html = render_html("Benchmark")
    作用: 生成静态网站入口 HTML
    输入: title: str，网页标题
    输出: str，HTML 文本
    """
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
  <header class="page-header">
    <div>
      <p class="eyebrow">Spatial Affordance Benchmark</p>
      <h1 id="siteTitle">{safe_title}</h1>
      <p id="siteSubtitle" class="subtitle">加载 benchmark 样本中...</p>
    </div>
    <div id="summaryCards" class="summary-cards" aria-label="总体指标"></div>
  </header>

  <main class="layout">
    <aside class="sidebar" aria-label="筛选和统计">
      <section class="panel">
        <h2>筛选</h2>
        <label class="field">
          <span>搜索</span>
          <input id="searchInput" type="search" placeholder="sample / scene / object / relation">
        </label>
        <label class="field">
          <span>数据源</span>
          <select id="sourceFilter"></select>
        </label>
        <label class="field">
          <span>结果</span>
          <select id="statusFilter">
            <option value="all">全部样本</option>
            <option value="success">成功</option>
            <option value="failed">失败</option>
            <option value="collision_failed">碰撞失败</option>
            <option value="direction_wrong">方向错误</option>
            <option value="size_inconsistent">体积不一致</option>
            <option value="missing_image">缺失图片</option>
          </select>
        </label>
        <label class="field">
          <span>排序</span>
          <select id="sortMode">
            <option value="default">默认顺序</option>
            <option value="failed_first">失败优先</option>
            <option value="collision_desc">碰撞率从高到低</option>
            <option value="direction_desc">方向 L2 误差从高到低</option>
            <option value="size_error_desc">体积相对误差从高到低</option>
            <option value="source">数据源</option>
          </select>
        </label>
        <button id="resetButton" class="button" type="button">重置</button>
      </section>

      <section class="panel">
        <h2>当前视图</h2>
        <div id="viewStats" class="view-stats"></div>
      </section>

      <section class="panel">
        <h2>数据源</h2>
        <div id="sourceStats" class="source-stats"></div>
      </section>
    </aside>

    <section class="content">
      <div class="content-toolbar">
        <div>
          <strong id="shownCount">0</strong>
          <span id="totalCount">/ 0 samples</span>
        </div>
        <div id="activeFilters" class="active-filters"></div>
      </div>
      <section id="sampleGrid" class="sample-grid" aria-live="polite"></section>
      <p id="emptyState" class="empty" hidden>没有匹配的样本。</p>
    </section>
  </main>

  <dialog id="sampleDialog" class="sample-dialog">
    <button id="closeDialog" class="dialog-close" type="button" aria-label="关闭">×</button>
    <div class="dialog-body">
      <img id="dialogImage" alt="">
      <section class="dialog-info">
        <h2 id="dialogTitle"></h2>
        <div id="dialogBadges" class="badges"></div>
        <dl id="dialogMetrics" class="details"></dl>
        <details open>
          <summary>预测与 GT Box</summary>
          <pre id="dialogBoxes"></pre>
        </details>
      </section>
    </div>
  </dialog>

  <script src="assets/benchmark-data.js"></script>
  <script src="assets/app.js"></script>
</body>
</html>
"""


def render_css() -> str:
    """
    用法: css = render_css()
    作用: 生成 benchmark 可视化网站样式
    输入: 无
    输出: str，CSS 文本
    """
    return """:root {
  --bg: #f5f7fb;
  --paper: #ffffff;
  --paper-soft: #f8fafc;
  --ink: #172033;
  --muted: #667085;
  --line: #d8dee8;
  --accent: #2563eb;
  --accent-soft: #dbeafe;
  --good: #15803d;
  --good-soft: #dcfce7;
  --bad: #b42318;
  --bad-soft: #fee4e2;
  --warn: #b54708;
  --warn-soft: #fef0c7;
  --shadow: 0 16px 40px rgba(23, 32, 51, 0.08);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
}

.page-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(420px, 620px);
  gap: 28px;
  align-items: end;
  max-width: 1480px;
  margin: 0 auto;
  padding: 28px 28px 20px;
}

.eyebrow {
  margin: 0 0 8px;
  color: var(--accent);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  font-size: 34px;
  line-height: 1.15;
}

h2 {
  margin: 0 0 14px;
  font-size: 15px;
}

.subtitle {
  margin: 10px 0 0;
  color: var(--muted);
  font-size: 14px;
}

.summary-cards {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.metric-card,
.panel,
.sample-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--paper);
  box-shadow: var(--shadow);
}

.metric-card {
  padding: 14px;
}

.metric-card strong {
  display: block;
  font-size: 24px;
  line-height: 1;
}

.metric-card span {
  display: block;
  margin-top: 7px;
  color: var(--muted);
  font-size: 12px;
}

.layout {
  display: grid;
  grid-template-columns: 300px minmax(0, 1fr);
  gap: 20px;
  max-width: 1480px;
  margin: 0 auto;
  padding: 0 28px 32px;
}

.sidebar {
  position: sticky;
  top: 16px;
  align-self: start;
  display: grid;
  gap: 14px;
}

.panel {
  padding: 16px;
}

.field {
  display: grid;
  gap: 6px;
  margin-bottom: 12px;
}

.field span {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}

input,
select,
button {
  width: 100%;
  min-height: 38px;
  border: 1px solid var(--line);
  border-radius: 7px;
  color: var(--ink);
  font: inherit;
  background: var(--paper);
}

input,
select {
  padding: 0 10px;
}

.button {
  color: #ffffff;
  font-weight: 800;
  cursor: pointer;
  border-color: var(--accent);
  background: var(--accent);
}

.view-stats,
.source-stats {
  display: grid;
  gap: 8px;
}

.stat-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: var(--muted);
  font-size: 13px;
}

.stat-row strong {
  color: var(--ink);
}

.content {
  min-width: 0;
}

.content-toolbar {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  min-height: 56px;
  margin-bottom: 14px;
  padding: 10px 0;
  background: rgba(245, 247, 251, 0.92);
  backdrop-filter: blur(12px);
}

.active-filters {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
}

.chip,
.badge {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
}

.chip {
  color: var(--muted);
  background: var(--paper);
  border: 1px solid var(--line);
}

.sample-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
  gap: 14px;
}

.sample-card {
  overflow: hidden;
  cursor: pointer;
  transition: transform 0.16s ease, box-shadow 0.16s ease;
}

.sample-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 20px 46px rgba(23, 32, 51, 0.13);
}

.thumb {
  display: block;
  width: 100%;
  aspect-ratio: 4 / 3;
  object-fit: contain;
  background: #111827;
}

.missing-thumb {
  display: grid;
  place-items: center;
  width: 100%;
  aspect-ratio: 4 / 3;
  color: var(--muted);
  background: var(--paper-soft);
}

.card-body {
  padding: 12px;
}

.card-title {
  overflow-wrap: anywhere;
  margin: 0 0 8px;
  font-size: 13px;
  line-height: 1.35;
}

.badges {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.badge.good {
  color: var(--good);
  background: var(--good-soft);
}

.badge.bad {
  color: var(--bad);
  background: var(--bad-soft);
}

.badge.warn {
  color: var(--warn);
  background: var(--warn-soft);
}

.badge.neutral {
  color: var(--muted);
  background: var(--paper-soft);
}

.relation {
  margin-top: 10px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}

.empty {
  margin: 44px 0;
  color: var(--muted);
  text-align: center;
}

.sample-dialog {
  width: min(1320px, calc(100vw - 48px));
  max-height: calc(100vh - 48px);
  padding: 0;
  border: 0;
  border-radius: 8px;
  background: var(--paper);
}

.sample-dialog::backdrop {
  background: rgba(15, 23, 42, 0.72);
}

.dialog-close {
  position: absolute;
  top: 12px;
  right: 12px;
  z-index: 2;
  width: 36px;
  min-height: 36px;
  color: var(--ink);
  font-size: 22px;
  cursor: pointer;
  background: rgba(255, 255, 255, 0.92);
}

.dialog-body {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 380px;
  max-height: calc(100vh - 48px);
}

.dialog-body img {
  width: 100%;
  height: 100%;
  max-height: calc(100vh - 48px);
  object-fit: contain;
  background: #111827;
}

.dialog-info {
  overflow: auto;
  padding: 24px;
  border-left: 1px solid var(--line);
}

.dialog-info h2 {
  overflow-wrap: anywhere;
  margin-bottom: 12px;
  font-size: 18px;
  line-height: 1.35;
}

.details {
  display: grid;
  grid-template-columns: 150px minmax(0, 1fr);
  gap: 9px 12px;
  margin: 18px 0;
  font-size: 13px;
}

.details dt {
  color: var(--muted);
}

.details dd {
  min-width: 0;
  margin: 0;
  overflow-wrap: anywhere;
}

summary {
  cursor: pointer;
  font-weight: 800;
}

pre {
  overflow: auto;
  padding: 12px;
  border-radius: 8px;
  color: #d1d5db;
  background: #111827;
  font-size: 12px;
}

@media (max-width: 980px) {
  .page-header,
  .layout,
  .dialog-body {
    grid-template-columns: 1fr;
  }

  .summary-cards {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .sidebar {
    position: static;
  }

  .dialog-info {
    border-left: 0;
    border-top: 1px solid var(--line);
  }
}

@media (max-width: 640px) {
  .page-header,
  .layout {
    padding-left: 14px;
    padding-right: 14px;
  }

  .summary-cards {
    grid-template-columns: 1fr;
  }

  .sample-grid {
    grid-template-columns: 1fr;
  }
}
"""


def render_js() -> str:
    """
    用法: js = render_js()
    作用: 生成前端交互逻辑
    输入: 无
    输出: str，JavaScript 文本
    """
    return """const data = window.BENCHMARK_DATA || {};
const items = Array.isArray(data.items) ? data.items : [];

const els = {
  title: document.getElementById("siteTitle"),
  subtitle: document.getElementById("siteSubtitle"),
  summaryCards: document.getElementById("summaryCards"),
  search: document.getElementById("searchInput"),
  source: document.getElementById("sourceFilter"),
  status: document.getElementById("statusFilter"),
  sort: document.getElementById("sortMode"),
  reset: document.getElementById("resetButton"),
  viewStats: document.getElementById("viewStats"),
  sourceStats: document.getElementById("sourceStats"),
  shownCount: document.getElementById("shownCount"),
  totalCount: document.getElementById("totalCount"),
  activeFilters: document.getElementById("activeFilters"),
  grid: document.getElementById("sampleGrid"),
  empty: document.getElementById("emptyState"),
  dialog: document.getElementById("sampleDialog"),
  closeDialog: document.getElementById("closeDialog"),
  dialogImage: document.getElementById("dialogImage"),
  dialogTitle: document.getElementById("dialogTitle"),
  dialogBadges: document.getElementById("dialogBadges"),
  dialogMetrics: document.getElementById("dialogMetrics"),
  dialogBoxes: document.getElementById("dialogBoxes"),
};

const formatPercent = (value) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(1)}%`;
};

const formatNumber = (value, digits = 4) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
};

const formatCm = (value, digits = 2) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A";
  return `${value.toFixed(digits)} cm`;
};

const formatMetricPair = (value, threshold, formatter) => {
  const valueText = formatter(value);
  const thresholdText = formatter(threshold);
  if (valueText === "N/A" && thresholdText === "N/A") return "N/A";
  return `${valueText} / ${thresholdText}`;
};

const metricState = (evaluated, passed) => {
  if (!evaluated) return { text: "not eval", kind: "neutral" };
  return passed ? { text: "pass", kind: "good" } : { text: "fail", kind: "bad" };
};

const searchableText = (item) => [
  item.sample_id,
  item.source_name,
  item.scene_id,
  item.frame_id,
  item.object_id,
  item.reference_object_id,
  item.reference_name,
  item.expected_relation,
  item.pred_relation,
].join(" ").toLowerCase();

const passStatus = (item, status) => {
  if (status === "success") return item.placement_success;
  if (status === "failed") return !item.placement_success;
  if (status === "collision_failed") return item.collision_evaluated && !item.collision_free;
  if (status === "direction_wrong") return item.direction_evaluated && !item.direction_correct;
  if (status === "size_inconsistent") return item.size_evaluated && !item.size_consistent;
  if (status === "missing_image") return !item.has_image;
  return true;
};

const sortItems = (list, mode) => {
  const sorted = [...list];
  const num = (value) => (typeof value === "number" ? value : -Infinity);
  if (mode === "failed_first") {
    sorted.sort((a, b) => Number(a.placement_success) - Number(b.placement_success) || a.index - b.index);
  } else if (mode === "collision_desc") {
    sorted.sort((a, b) => num(b.occupied_collision_ratio) - num(a.occupied_collision_ratio));
  } else if (mode === "direction_desc") {
    sorted.sort((a, b) => num(b.center_l2_error_cm) - num(a.center_l2_error_cm));
  } else if (mode === "size_error_desc") {
    sorted.sort((a, b) => num(b.volume_error_ratio) - num(a.volume_error_ratio));
  } else if (mode === "source") {
    sorted.sort((a, b) => `${a.source_name} ${a.sample_id}`.localeCompare(`${b.source_name} ${b.sample_id}`));
  }
  return sorted;
};

const makeBadge = (text, kind) => `<span class="badge ${kind}">${text}</span>`;

const makeMetricBadge = (label, evaluated, passed, detailText) => {
  const state = metricState(evaluated, passed);
  const suffix = detailText && detailText !== "N/A" ? ` ${detailText}` : "";
  return makeBadge(`${label} ${state.text}${suffix}`, state.kind);
};

const renderBadges = (item) => {
  const badges = [];
  const placementState = metricState(item.full_metric_evaluated, item.placement_success);
  badges.push(makeBadge(`success ${placementState.text}`, placementState.kind));
  badges.push(makeMetricBadge(
    "collision",
    item.collision_evaluated,
    item.collision_free,
    formatMetricPair(item.occupied_collision_ratio, item.collision_ratio_threshold, (value) => formatPercent(value)),
  ));
  badges.push(makeMetricBadge(
    "direction",
    item.direction_evaluated,
    item.direction_correct,
    formatMetricPair(item.center_l2_error_cm, item.center_l2_threshold_cm, (value) => formatCm(value, 2)),
  ));
  badges.push(makeMetricBadge(
    "size",
    item.size_evaluated,
    item.size_consistent,
    formatMetricPair(item.volume_error_ratio, item.volume_error_ratio_threshold, (value) => formatPercent(value)),
  ));
  if (!item.has_image) badges.push(makeBadge("missing image", "bad"));
  return badges.join("");
};

const renderSummary = () => {
  const summary = data.summary || {};
  const cards = [
    ["样本数", summary.sample_count ?? data.metadata?.sample_count ?? items.length],
    ["成功率", formatPercent(summary.placement_success_rate)],
    ["无碰撞率", formatPercent(summary.collision_free_rate)],
    ["方向正确率", formatPercent(summary.direction_correct_rate)],
    ["体积一致率", formatPercent(summary.size_consistent_rate)],
    ["平均碰撞率", formatPercent(summary.mean_occupied_collision_ratio)],
  ];
  els.summaryCards.innerHTML = cards.map(([label, value]) => `
    <div class="metric-card">
      <strong>${value}</strong>
      <span>${label}</span>
    </div>
  `).join("");
};

const renderSourceFilter = () => {
  const sources = data.metadata?.sources || [...new Set(items.map((item) => item.source_name))].sort();
  els.source.innerHTML = `<option value="all">全部数据源</option>` +
    sources.map((source) => `<option value="${source}">${source}</option>`).join("");
};

const renderSourceStats = () => {
  const bySource = data.by_source || {};
  els.sourceStats.innerHTML = Object.entries(bySource).map(([source, stats]) => `
    <div class="stat-row">
      <span>${source}</span>
      <strong>${formatPercent(stats.placement_success_rate)} / ${stats.sample_count}</strong>
    </div>
  `).join("");
};

const currentFilteredItems = () => {
  const keyword = els.search.value.trim().toLowerCase();
  const source = els.source.value;
  const status = els.status.value;
  const filtered = items.filter((item) => {
    const matchesKeyword = !keyword || searchableText(item).includes(keyword);
    const matchesSource = source === "all" || item.source_name === source;
    return matchesKeyword && matchesSource && passStatus(item, status);
  });
  return sortItems(filtered, els.sort.value);
};

const renderViewStats = (list) => {
  const count = list.length || 1;
  const success = list.filter((item) => item.placement_success).length;
  const collisionFail = list.filter((item) => item.collision_evaluated && !item.collision_free).length;
  const directionWrong = list.filter((item) => item.direction_evaluated && !item.direction_correct).length;
  const sizeBad = list.filter((item) => item.size_evaluated && !item.size_consistent).length;
  els.viewStats.innerHTML = `
    <div class="stat-row"><span>成功</span><strong>${success} (${formatPercent(success / count)})</strong></div>
    <div class="stat-row"><span>碰撞失败</span><strong>${collisionFail}</strong></div>
    <div class="stat-row"><span>方向错误</span><strong>${directionWrong}</strong></div>
    <div class="stat-row"><span>体积不一致</span><strong>${sizeBad}</strong></div>
  `;
};

const renderActiveFilters = () => {
  const chips = [];
  if (els.search.value.trim()) chips.push(`搜索: ${els.search.value.trim()}`);
  if (els.source.value !== "all") chips.push(`数据源: ${els.source.value}`);
  if (els.status.value !== "all") chips.push(`结果: ${els.status.selectedOptions[0].textContent}`);
  if (els.sort.value !== "default") chips.push(`排序: ${els.sort.selectedOptions[0].textContent}`);
  els.activeFilters.innerHTML = chips.map((chip) => `<span class="chip">${chip}</span>`).join("");
};

const renderCard = (item) => {
  const thumb = item.has_image
    ? `<img class="thumb" loading="lazy" src="${item.image_path}" alt="${item.sample_id}">`
    : `<div class="missing-thumb">图片缺失</div>`;
  return `
    <article class="sample-card" data-index="${item.index}" tabindex="0">
      ${thumb}
      <div class="card-body">
        <h3 class="card-title">${item.sample_id}</h3>
        <div class="badges">${renderBadges(item)}</div>
        <div class="relation">${item.expected_relation || "N/A"} -> ${item.pred_relation || "N/A"}</div>
      </div>
    </article>
  `;
};

const openDialog = (item) => {
  els.dialogTitle.textContent = item.sample_id;
  els.dialogImage.src = item.has_image ? item.image_path : "";
  els.dialogImage.alt = item.sample_id;
  els.dialogBadges.innerHTML = renderBadges(item);
  const fields = [
    ["source", item.source_name],
    ["scene", item.scene_id],
    ["frame", item.frame_id],
    ["object", item.object_id],
    ["reference object", item.reference_object_id],
    ["reference name", item.reference_name],
    ["collision evaluated", item.collision_evaluated ? "yes" : "no"],
    ["collision free", item.collision_free ? "yes" : "no"],
    ["occupied collision ratio", formatPercent(item.occupied_collision_ratio)],
    ["collision ratio threshold", formatPercent(item.collision_ratio_threshold)],
    ["unknown overlap ratio", formatPercent(item.unknown_overlap_ratio)],
    ["direction evaluated", item.direction_evaluated ? "yes" : "no"],
    ["direction correct", item.direction_correct ? "yes" : "no"],
    ["center match", item.center_match ? "yes" : "no"],
    ["expected relation", item.expected_relation],
    ["pred relation", item.pred_relation],
    ["direction center L2 error cm", formatCm(item.center_l2_error_cm, 3)],
    ["direction center L2 threshold cm", formatCm(item.center_l2_threshold_cm, 3)],
    ["size evaluated", item.size_evaluated ? "yes" : "no"],
    ["size consistent", item.size_consistent ? "yes" : "no"],
    ["pred volume cm3", formatCm(item.pred_volume_cm3, 3)],
    ["target volume cm3", formatCm(item.target_volume_cm3, 3)],
    ["volume error cm3", formatCm(item.volume_error_cm3, 3)],
    ["volume error ratio", formatPercent(item.volume_error_ratio)],
    ["volume error ratio threshold", formatPercent(item.volume_error_ratio_threshold)],
    ["vis path", item.vis_path],
  ];
  els.dialogMetrics.innerHTML = fields.map(([key, value]) => `<dt>${key}</dt><dd>${value || "N/A"}</dd>`).join("");
  els.dialogBoxes.textContent = JSON.stringify({
    pred_box_world: item.pred_box_world,
    gt_box_world: item.gt_box_world,
    errors: item.errors,
  }, null, 2);
  els.dialog.showModal();
};

const renderGrid = () => {
  const list = currentFilteredItems();
  els.shownCount.textContent = list.length;
  els.totalCount.textContent = `/ ${items.length} samples`;
  els.empty.hidden = list.length > 0;
  els.grid.innerHTML = list.map(renderCard).join("");
  renderViewStats(list);
  renderActiveFilters();

  els.grid.querySelectorAll(".sample-card").forEach((card) => {
    const item = items.find((candidate) => candidate.index === Number(card.dataset.index));
    card.addEventListener("click", () => openDialog(item));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter") openDialog(item);
    });
  });
};

const resetFilters = () => {
  els.search.value = "";
  els.source.value = "all";
  els.status.value = "all";
  els.sort.value = "default";
  renderGrid();
};

els.title.textContent = data.title || "Placement Benchmark Visualization";
els.subtitle.textContent = `${data.generated_at || ""} · ${data.metadata?.eval_dir || ""} · ${data.metadata?.infer_dir || ""}`;
renderSummary();
renderSourceFilter();
renderSourceStats();
renderGrid();

[els.search, els.source, els.status, els.sort].forEach((el) => el.addEventListener("input", renderGrid));
els.reset.addEventListener("click", resetFilters);
els.closeDialog.addEventListener("click", () => els.dialog.close());
els.dialog.addEventListener("click", (event) => {
  if (event.target === els.dialog) els.dialog.close();
});
"""


def write_site(output_dir: Path, payload: dict[str, Any], title: str) -> None:
    """
    用法: write_site(output_dir, payload, title)
    作用: 写出静态网站所有文件
    输入: output_dir: Path，网站输出目录；payload: dict，前端数据；title: str，网页标题
    输出: None
    """
    write_text_file(output_dir / "index.html", render_html(title))
    write_text_file(output_dir / "assets" / "style.css", render_css())
    write_text_file(output_dir / "assets" / "app.js", render_js())
    write_text_file(output_dir / "assets" / "benchmark-data.js", json_to_script(payload))


def main() -> None:
    """
    用法: main()
    作用: 命令行入口，读取数据并生成 benchmark 静态网站
    输入: 命令行参数
    输出: None
    """
    args = build_parser().parse_args()
    eval_dir = resolve_project_path(args.eval_dir)
    infer_dir = resolve_project_path(args.infer_dir)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else eval_dir / "visualization"

    summary_payload, per_sample_payload, samples = load_eval_payloads(eval_dir)
    prediction_lookup = load_prediction_lookup(infer_dir)
    items, missing_counts = build_site_items(samples, prediction_lookup, infer_dir, output_dir)
    payload = build_site_payload(
        summary_payload=summary_payload,
        per_sample_payload=per_sample_payload,
        items=items,
        eval_dir=eval_dir,
        infer_dir=infer_dir,
        title=args.title,
        missing_counts=missing_counts,
    )
    write_site(output_dir, payload, args.title)

    print(f"已生成 benchmark 可视化网站: {path_to_record(output_dir / 'index.html')}")
    print(f"样本数: {len(items)}")
    print(f"缺失 prediction: {missing_counts['missing_prediction_count']}")
    print(f"缺失图片: {missing_counts['missing_image_count']}")
    print("使用示例:")
    print(f"  conda run -n spatial python tools/build_benchmark_site.py --eval-dir {path_to_record(eval_dir)} --infer-dir {path_to_record(infer_dir)}")
    print("预览示例:")
    print(f"  cd {path_to_record(eval_dir)} && python -m http.server 8000")
    print("  浏览器打开 http://localhost:8000/visualization/")


if __name__ == "__main__":
    main()
