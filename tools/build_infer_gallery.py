#!/usr/bin/env python3
"""
tools/build_infer_gallery.py
----------------------------
职责：根据多模态推理输出目录生成可复用的静态效果展示网站。

用法：
    conda run -n spatial python tools/build_infer_gallery.py \
        --infer-dir outputs/infer_model_textLen128_9_9_out512v2 \
        --annotation-dir data/annotations/placement_multimodal \
        --prompt-key polished_prompt

作用：
    - 读取 infer_dir/predictions.json 与 infer_dir/vis 图片
    - 从 annotation_dir/{split}.json 中匹配每张图对应的 prompt
    - 生成纯静态 gallery，支持单个或多个关键词搜索、source 过滤、prompt 切换和图片放大查看
    - 后续展示其它推理结果时，只需替换 --infer-dir 重新生成

输入：
    --infer-dir: 推理输出目录，目录下需要包含 predictions.json 和 vis/
    --annotation-dir: 多模态标注目录，目录下需要包含 train/valid/test JSON
    --split: 可选，标注 split；默认使用 predictions.json 中记录的 split
    --prompt-key: 默认展示的 prompt 字段，通常为 polished_prompt 或 prompt
    --output-dir: 可选，网站输出目录；默认写入 infer_dir/gallery
    --title: 可选，网站标题

输出：
    output_dir/
        - index.html
        - assets/style.css
        - assets/app.js
        - assets/gallery-data.js

使用示例：
    conda run -n spatial python tools/build_infer_gallery.py \
        --infer-dir outputs/infer_model_textLen128_9_9_out512v2 \
        --annotation-dir data/annotations/placement_multimodal \
        --prompt-key polished_prompt

预览示例：
    cd outputs/infer_model_textLen128_9_9_out512v2
    python -m http.server 8000
    # 浏览器打开 http://localhost:8000/gallery/
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = "predictions.json"
DEFAULT_PROMPT_KEY = "polished_prompt"
RAW_PROMPT_KEY = "prompt"
GALLERY_SCHEMA_VERSION = "infer_gallery/v1"


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建静态网站生成脚本的命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser，配置完成的解析器
    """
    parser = argparse.ArgumentParser(description="生成多模态推理结果静态展示网站")
    parser.add_argument(
        "--infer-dir",
        type=Path,
        required=True,
        help="推理输出目录，目录下需要包含 predictions.json 和 vis/",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("data/annotations/placement_multimodal"),
        help="多模态标注目录，默认 data/annotations/placement_multimodal",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="标注 split；默认读取 predictions.json 中的 split 字段",
    )
    parser.add_argument(
        "--prompt-key",
        type=str,
        default=DEFAULT_PROMPT_KEY,
        help="默认展示的 prompt 字段，通常为 polished_prompt 或 prompt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="网站输出目录；默认 infer_dir/gallery",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="网站标题；默认使用 infer-dir 目录名",
    )
    return parser


def resolve_project_path(path_value: str | Path) -> Path:
    """
    用法: path = resolve_project_path("outputs/demo")
    作用: 将仓库相对路径转换为绝对路径
    输入: path_value: str | Path，相对或绝对路径
    输出: Path，绝对路径
    """
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(json_path: Path) -> dict[str, Any]:
    """
    用法: payload = load_json(Path("outputs/demo/predictions.json"))
    作用: 读取 JSON 文件并返回字典
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
    用法: write_text_file(Path("gallery/index.html"), html)
    作用: 将文本内容写入文件，并自动创建父目录
    输入: output_path: Path，输出文件路径；content: str，文本内容
    输出: None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def path_to_record(path_value: Path) -> str:
    """
    用法: text = path_to_record(Path("outputs/demo/vis/a.png"))
    作用: 将绝对路径压缩为仓库相对路径，便于前端展示
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
    作用: 生成 site_dir 到 path_value 的相对路径，跨目录时保持浏览器可访问
    输入: path_value: Path，目标文件路径；site_dir: Path，网站目录
    输出: str，POSIX 相对路径
    """
    relative_path = os.path.relpath(Path(path_value).resolve(), start=site_dir.resolve())
    return Path(relative_path).as_posix()


def infer_split(prediction_payload: dict[str, Any], split_override: str | None) -> str:
    """
    用法: split = infer_split(payload, args.split)
    作用: 确定用于查找 prompt 的标注 split
    输入: prediction_payload: dict，推理结果；split_override: str | None，命令行覆盖值
    输出: str，最终 split 名称
    """
    split = split_override or prediction_payload.get("split")
    if not split:
        raise ValueError("无法确定 split，请通过 --split 指定 train/valid/test")
    split = str(split).strip().lower()
    return "valid" if split == "val" else split


def load_prediction_payload(infer_dir: Path) -> dict[str, Any]:
    """
    用法: payload = load_prediction_payload(Path("outputs/infer_demo"))
    作用: 读取推理目录中的 predictions.json 并检查 predictions 字段
    输入: infer_dir: Path，推理输出目录
    输出: dict，推理结果载荷
    """
    prediction_path = infer_dir / PREDICTIONS_FILE
    if not prediction_path.exists():
        raise FileNotFoundError(f"未找到推理结果文件: {prediction_path}")

    payload = load_json(prediction_path)
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"{prediction_path} 缺少 list 类型的 predictions 字段")
    return payload


def load_annotation_lookup(annotation_dir: Path, split: str) -> dict[tuple[str, str], dict[str, Any]]:
    """
    用法: lookup = load_annotation_lookup(annotation_dir, "test")
    作用: 读取标注 split，并按 (source_name, sample_id) 构建快速查询表
    输入: annotation_dir: Path，标注目录；split: str，数据划分
    输出: dict，键为 (source_name, sample_id)，值为样本标注
    """
    annotation_path = annotation_dir / f"{split}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"未找到标注文件: {annotation_path}")

    payload = load_json(annotation_path)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{annotation_path} 缺少 list 类型的 samples 字段")

    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        source_name = str(sample.get("source_name", "")).strip()
        sample_id = str(sample.get("sample_id", "")).strip()
        if source_name and sample_id:
            lookup[(source_name, sample_id)] = sample
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
        vis_path = resolve_project_path(str(vis_path_value))
    else:
        source_name = str(prediction.get("source_name", "")).strip()
        sample_id = str(prediction.get("sample_id", "")).strip()
        vis_path = infer_dir / "vis" / f"{source_name}__{sample_id}.png"
    if not vis_path.exists():
        raise FileNotFoundError(f"未找到可视化图片: {vis_path}")
    return vis_path


def get_prompt(sample: dict[str, Any] | None, prompt_key: str) -> tuple[str, str]:
    """
    用法: prompt, raw_prompt = get_prompt(sample, "polished_prompt")
    作用: 从样本标注中提取默认 prompt 与原始 prompt
    输入: sample: dict | None，样本标注；prompt_key: str，默认展示字段
    输出: tuple[str, str]，默认 prompt 与 raw prompt
    """
    if sample is None:
        return "", ""

    prompt = str(sample.get(prompt_key, "")).strip()
    raw_prompt = str(sample.get(RAW_PROMPT_KEY, "")).strip()
    if not prompt:
        prompt = raw_prompt
    return prompt, raw_prompt


def build_gallery_items(
        predictions: list[dict[str, Any]],
        annotation_lookup: dict[tuple[str, str], dict[str, Any]],
        prompt_key: str,
        infer_dir: Path,
        site_dir: Path) -> tuple[list[dict[str, Any]], int]:
    """
    用法: items, missing_count = build_gallery_items(predictions, lookup, "polished_prompt", infer_dir, site_dir)
    作用: 合并推理记录、图片路径和 prompt，生成前端可直接消费的数据
    输入: predictions: list[dict]，推理结果；annotation_lookup: dict，标注查询表；
         prompt_key: str，默认 prompt 字段；infer_dir/site_dir: Path，路径上下文
    输出: tuple，gallery item 列表与缺失 prompt 数量
    """
    items: list[dict[str, Any]] = []
    missing_prompt_count = 0

    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            continue

        source_name = str(prediction.get("source_name", "")).strip()
        sample_id = str(prediction.get("sample_id", "")).strip()
        if not source_name or not sample_id:
            raise ValueError(f"prediction 缺少 source_name 或 sample_id: index={index}")

        annotation = annotation_lookup.get((source_name, sample_id))
        prompt, raw_prompt = get_prompt(annotation, prompt_key)
        if not prompt:
            missing_prompt_count += 1

        image_path = resolve_visual_path(prediction, infer_dir)
        items.append({
            "index": len(items) + 1,
            "sample_id": sample_id,
            "source_name": source_name,
            "image_path": path_from_site(image_path, site_dir),
            "image_file": image_path.name,
            "vis_path": path_to_record(image_path),
            "rgb_path": str(prediction.get("rgb_path", "")),
            "prompt": prompt,
            "raw_prompt": raw_prompt,
        })
    return items, missing_prompt_count


def build_site_payload(
        prediction_payload: dict[str, Any],
        items: list[dict[str, Any]],
        infer_dir: Path,
        annotation_dir: Path,
        split: str,
        prompt_key: str,
        title: str,
        missing_prompt_count: int) -> dict[str, Any]:
    """
    用法: payload = build_site_payload(pred_payload, items, infer_dir, ann_dir, "test", "polished_prompt", title, 0)
    作用: 组织写入 gallery-data.js 的完整前端数据
    输入: prediction_payload: dict，推理结果；items: list[dict]，展示条目；
         infer_dir/annotation_dir: Path，路径信息；split/prompt_key/title: str，展示元信息；
         missing_prompt_count: int，缺失 prompt 数量
    输出: dict，前端数据载荷
    """
    sources = sorted({item["source_name"] for item in items})
    return {
        "schema_version": GALLERY_SCHEMA_VERSION,
        "title": title,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": {
            "infer_dir": path_to_record(infer_dir),
            "annotation_dir": path_to_record(annotation_dir),
            "split": split,
            "prompt_key": prompt_key,
            "checkpoint_path": str(prediction_payload.get("checkpoint_path", "")),
            "prediction_schema_version": str(prediction_payload.get("schema_version", "")),
            "prediction_count": int(prediction_payload.get("sample_count", len(items))),
            "item_count": len(items),
            "missing_prompt_count": missing_prompt_count,
            "sources": sources,
        },
        "items": items,
    }


def render_html(title: str) -> str:
    """
    用法: html = render_html("Infer Gallery")
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
  <header class="hero">
    <div class="hero__copy">
      <p class="eyebrow">Spatial Affordance Inference Gallery</p>
      <h1 id="galleryTitle">{safe_title}</h1>
      <p class="hero__subtitle" id="gallerySubtitle">加载推理结果中...</p>
    </div>
    <div class="hero__panel" aria-label="结果统计">
      <div>
        <span id="statShown">0</span>
        <small>当前展示</small>
      </div>
      <div>
        <span id="statTotal">0</span>
        <small>全部样本</small>
      </div>
      <div>
        <span id="statSources">0</span>
        <small>数据源</small>
      </div>
    </div>
  </header>

  <main>
    <section class="toolbar" aria-label="筛选工具">
      <label class="field field--search">
        <span>搜索</span>
        <textarea id="searchInput" rows="3" placeholder="sample_id / source / prompt；多个关键词可用换行、逗号或分号分隔"></textarea>
      </label>
      <label class="field">
        <span>数据源</span>
        <select id="sourceFilter"></select>
      </label>
      <label class="field">
        <span>Prompt</span>
        <select id="promptMode">
          <option value="prompt">默认 prompt</option>
          <option value="raw">原始 prompt</option>
        </select>
      </label>
      <button id="shuffleButton" class="secondary-button" type="button">随机打乱</button>
      <button id="resetButton" class="ghost-button" type="button">重置筛选</button>
    </section>

    <section class="meta-strip" id="metaStrip" aria-label="元信息"></section>
    <section class="grid" id="galleryGrid" aria-live="polite"></section>
    <p class="empty" id="emptyState" hidden>没有匹配的结果，请调整搜索或筛选条件。</p>
  </main>

  <dialog class="viewer" id="imageViewer">
    <button class="viewer__close" id="closeViewer" type="button" aria-label="关闭">×</button>
    <button class="viewer__nav viewer__nav--prev" id="prevImage" type="button" aria-label="上一张">‹</button>
    <figure>
      <img id="viewerImage" alt="">
      <figcaption>
        <strong id="viewerTitle"></strong>
        <span id="viewerPrompt"></span>
      </figcaption>
    </figure>
    <button class="viewer__nav viewer__nav--next" id="nextImage" type="button" aria-label="下一张">›</button>
  </dialog>

  <script src="assets/gallery-data.js"></script>
  <script src="assets/app.js"></script>
</body>
</html>
"""


def render_css() -> str:
    """
    用法: css = render_css()
    作用: 生成 gallery 的样式文件
    输入: 无
    输出: str，CSS 文本
    """
    return """* {
  box-sizing: border-box;
}

:root {
  --paper: #ffffff;
  --paper-soft: #f7f8fa;
  --ink: #1f2933;
  --muted: #687385;
  --line: #d9dee7;
  --line-soft: #edf0f4;
  --accent: #315f8c;
  --accent-deep: #244563;
  --card: #ffffff;
  --shadow: 0 10px 30px rgba(31, 41, 51, 0.08);
}

body {
  margin: 0;
  min-height: 100vh;
  color: var(--ink);
  font-family: Georgia, \"Times New Roman\", \"Noto Serif CJK SC\", \"Source Han Serif SC\", \"Songti SC\", SimSun, serif;
  background: var(--paper);
}

body::before {
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  content: \"\";
  background:
    linear-gradient(to bottom, rgba(49, 95, 140, 0.05), transparent 320px),
    repeating-linear-gradient(0deg, transparent 0, transparent 31px, rgba(31, 41, 51, 0.025) 32px);
}

.hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 2.5rem;
  align-items: center;
  width: min(1320px, calc(100% - 56px));
  margin: 0 auto;
  padding: 40px 0 24px;
  border-bottom: 1px solid var(--line);
}

.eyebrow {
  margin: 0 0 0.7rem;
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

h1 {
  max-width: 920px;
  margin: 0;
  font-size: clamp(1.95rem, 4vw, 3.6rem);
  font-weight: 700;
  line-height: 1.08;
  letter-spacing: -0.02em;
}

.hero__subtitle {
  max-width: 760px;
  margin: 1rem 0 0;
  color: var(--muted);
  font-size: 0.98rem;
  line-height: 1.6;
}

.hero__panel {
  display: grid;
  grid-template-columns: repeat(3, minmax(96px, 1fr));
  overflow: hidden;
  min-width: 360px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--card);
  box-shadow: none;
}

.hero__panel div {
  padding: 16px 18px;
  border-left: 1px solid var(--line);
  background: var(--paper-soft);
}

.hero__panel div:first-child {
  border-left: 0;
}

.hero__panel span {
  display: block;
  font-size: 2.1rem;
  font-weight: 700;
  line-height: 1;
}

.hero__panel small {
  color: var(--muted);
  font-size: 0.78rem;
}

main {
  width: min(1320px, calc(100% - 56px));
  margin: 0 auto;
  padding: 22px 0 56px;
}

.toolbar {
  position: sticky;
  top: 0;
  z-index: 5;
  display: grid;
  grid-template-columns: minmax(260px, 1fr) 170px 170px auto auto;
  gap: 12px;
  align-items: end;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.94);
  box-shadow: 0 8px 24px rgba(31, 41, 51, 0.05);
}

.field {
  display: grid;
  gap: 6px;
}

.field span {
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

input,
textarea,
select,
button {
  min-height: 40px;
  border: 1px solid var(--line);
  border-radius: 4px;
  color: var(--ink);
  font: inherit;
}

input,
textarea,
select {
  width: 100%;
  padding: 0 12px;
  background: #fff;
}

textarea {
  min-height: 88px;
  padding-top: 9px;
  padding-bottom: 9px;
  resize: vertical;
  line-height: 1.45;
}

.ghost-button,
.secondary-button {
  padding: 0 16px;
  font-weight: 700;
  cursor: pointer;
}

.ghost-button {
  background: var(--accent-deep);
  color: #fff;
}

.secondary-button {
  background: #fff;
  color: var(--accent-deep);
}

.meta-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 18px 0 22px;
}

.meta-chip {
  padding: 7px 11px;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--paper-soft);
  color: var(--muted);
  font-size: 0.82rem;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
  gap: 20px;
}

.card {
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--card);
  box-shadow: var(--shadow);
  animation: lift-in 0.42s ease both;
}

.card__image-button {
  display: block;
  width: 100%;
  min-height: 0;
  padding: 0;
  border: 0;
  border-radius: 0;
  background: var(--paper-soft);
  cursor: zoom-in;
}

.card img {
  display: block;
  width: 100%;
  aspect-ratio: 4 / 3;
  object-fit: cover;
  border-bottom: 1px solid var(--line);
}

.card__body {
  display: grid;
  gap: 12px;
  padding: 15px 16px 17px;
}

.card__topline {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
}

.badge {
  padding: 4px 8px;
  border: 1px solid rgba(49, 95, 140, 0.22);
  border-radius: 4px;
  background: rgba(49, 95, 140, 0.08);
  color: var(--accent-deep);
  font-size: 0.76rem;
  font-weight: 700;
}

.index {
  color: var(--muted);
  font-size: 0.82rem;
}

.sample-id {
  margin: 0;
  font-family: \"SFMono-Regular\", \"Cascadia Code\", Consolas, monospace;
  font-size: 0.88rem;
  word-break: break-all;
}

.prompt {
  margin: 0;
  color: #2d3742;
  font-size: 0.94rem;
  line-height: 1.56;
}

.prompt--missing {
  color: var(--accent-deep);
  font-style: italic;
}

.path {
  margin: 0;
  color: var(--muted);
  font-family: \"SFMono-Regular\", \"Cascadia Code\", Consolas, monospace;
  font-size: 0.72rem;
  word-break: break-all;
}

.empty {
  margin: 40px 0;
  padding: 28px;
  border: 1px dashed var(--line);
  border-radius: 6px;
  background: var(--paper-soft);
  color: var(--muted);
  text-align: center;
}

.viewer {
  width: min(1200px, calc(100vw - 40px));
  max-height: calc(100vh - 40px);
  padding: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--ink);
  box-shadow: 0 28px 90px rgba(31, 41, 51, 0.3);
}

.viewer::backdrop {
  background: rgba(31, 41, 51, 0.68);
}

.viewer figure {
  margin: 0;
}

.viewer img {
  display: block;
  width: 100%;
  max-height: calc(100vh - 190px);
  object-fit: contain;
  background: var(--paper-soft);
}

.viewer figcaption {
  display: grid;
  gap: 6px;
  padding: 18px 22px 22px;
  border-top: 1px solid var(--line);
  color: var(--ink);
}

.viewer figcaption span {
  color: var(--muted);
  line-height: 1.45;
}

.viewer__close,
.viewer__nav {
  position: absolute;
  z-index: 2;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.92);
  color: var(--ink);
  cursor: pointer;
}

.viewer__close {
  top: 12px;
  right: 12px;
  width: 44px;
  font-size: 1.6rem;
}

.viewer__nav {
  top: 45%;
  width: 48px;
  height: 64px;
  font-size: 2.8rem;
}

.viewer__nav--prev {
  left: 12px;
}

.viewer__nav--next {
  right: 12px;
}

@keyframes lift-in {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 900px) {
  .hero {
    grid-template-columns: 1fr;
  }

  .hero__panel {
    min-width: 0;
  }

  .toolbar {
    grid-template-columns: 1fr 1fr;
  }

  .field--search {
    grid-column: 1 / -1;
  }
}

@media (max-width: 620px) {
  .hero,
  main {
    width: min(100% - 24px, 1320px);
  }

  .hero__panel,
  .toolbar {
    grid-template-columns: 1fr;
  }

  .grid {
    grid-template-columns: 1fr;
  }
}
"""


def render_app_js() -> str:
    """
    用法: js = render_app_js()
    作用: 生成 gallery 交互逻辑脚本
    输入: 无
    输出: str，JavaScript 文本
    """
    return """"use strict";

const data = window.GALLERY_DATA || { metadata: {}, items: [] };
const state = {
  query: "",
  source: "all",
  promptMode: "prompt",
  shownItems: [],
  viewerIndex: 0,
};

const elements = {
  title: document.getElementById("galleryTitle"),
  subtitle: document.getElementById("gallerySubtitle"),
  statShown: document.getElementById("statShown"),
  statTotal: document.getElementById("statTotal"),
  statSources: document.getElementById("statSources"),
  searchInput: document.getElementById("searchInput"),
  sourceFilter: document.getElementById("sourceFilter"),
  promptMode: document.getElementById("promptMode"),
  shuffleButton: document.getElementById("shuffleButton"),
  resetButton: document.getElementById("resetButton"),
  metaStrip: document.getElementById("metaStrip"),
  grid: document.getElementById("galleryGrid"),
  emptyState: document.getElementById("emptyState"),
  viewer: document.getElementById("imageViewer"),
  viewerImage: document.getElementById("viewerImage"),
  viewerTitle: document.getElementById("viewerTitle"),
  viewerPrompt: document.getElementById("viewerPrompt"),
  closeViewer: document.getElementById("closeViewer"),
  prevImage: document.getElementById("prevImage"),
  nextImage: document.getElementById("nextImage"),
};

function getDisplayPrompt(item) {
  if (state.promptMode === "raw") {
    return item.raw_prompt || item.prompt || "";
  }
  return item.prompt || item.raw_prompt || "";
}

function normalizeText(value) {
  return String(value || "").toLowerCase();
}

// 用法: const terms = splitSearchTerms(searchText)
// 作用: 将搜索输入拆成多个 OR 匹配关键词，支持换行、逗号和分号分隔
// 输入: value: string，搜索框原始文本
// 输出: string[]，归一化后的关键词列表
function splitSearchTerms(value) {
  const rawQuery = String(value || "").trim();
  if (!rawQuery) {
    return [];
  }
  return rawQuery
    .split(/[\\r\\n,，;；]+/)
    .map((term) => normalizeText(term).trim())
    .filter(Boolean);
}

function buildSearchText(item) {
  return [
    item.sample_id,
    item.source_name,
    item.image_file,
    item.prompt,
    item.raw_prompt,
    item.vis_path,
  ].map(normalizeText).join(" ");
}

function getItemKey(item) {
  return `${item.source_name}::${item.sample_id}::${item.index}`;
}

function hasSameOrder(leftItems, rightItems) {
  if (leftItems.length !== rightItems.length) {
    return false;
  }
  return leftItems.every((item, index) => getItemKey(item) === getItemKey(rightItems[index]));
}

function shuffleItems(items) {
  const shuffled = [...items];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const randomIndex = Math.floor(Math.random() * (index + 1));
    [shuffled[index], shuffled[randomIndex]] = [shuffled[randomIndex], shuffled[index]];
  }
  if (shuffled.length > 1 && hasSameOrder(shuffled, state.shownItems)) {
    [shuffled[0], shuffled[1]] = [shuffled[1], shuffled[0]];
  }
  return shuffled;
}

function createOption(value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  return option;
}

function initHeader() {
  const metadata = data.metadata || {};
  elements.title.textContent = data.title || "Inference Gallery";
  elements.subtitle.textContent = [
    `infer_dir: ${metadata.infer_dir || "-"}`,
    `split: ${metadata.split || "-"}`,
    `prompt_key: ${metadata.prompt_key || "-"}`,
  ].join(" · ");
  elements.statTotal.textContent = String((data.items || []).length);
  elements.statSources.textContent = String((metadata.sources || []).length);
}

function initFilters() {
  const sources = (data.metadata && data.metadata.sources) || [];
  elements.sourceFilter.appendChild(createOption("all", "全部 source"));
  sources.forEach((source) => {
    elements.sourceFilter.appendChild(createOption(source, source));
  });
}

function renderMeta() {
  const metadata = data.metadata || {};
  const chips = [
    `checkpoint: ${metadata.checkpoint_path || "-"}`,
    `predictions: ${metadata.prediction_count ?? 0}`,
    `gallery_items: ${metadata.item_count ?? 0}`,
    `missing_prompt: ${metadata.missing_prompt_count ?? 0}`,
    `generated_at: ${data.generated_at || "-"}`,
  ];

  elements.metaStrip.replaceChildren(...chips.map((text) => {
    const chip = document.createElement("span");
    chip.className = "meta-chip";
    chip.textContent = text;
    return chip;
  }));
}

function getFilteredItems() {
  const searchTerms = splitSearchTerms(state.query);
  return (data.items || []).filter((item) => {
    const sourceMatched = state.source === "all" || item.source_name === state.source;
    const searchText = buildSearchText(item);
    const queryMatched = !searchTerms.length || searchTerms.some((term) => searchText.includes(term));
    return sourceMatched && queryMatched;
  });
}

function createCard(item, visibleIndex) {
  const card = document.createElement("article");
  card.className = "card";
  card.style.animationDelay = `${Math.min(visibleIndex, 12) * 24}ms`;

  const imageButton = document.createElement("button");
  imageButton.className = "card__image-button";
  imageButton.type = "button";
  imageButton.addEventListener("click", () => openViewer(visibleIndex));

  const image = document.createElement("img");
  image.loading = "lazy";
  image.decoding = "async";
  image.src = item.image_path;
  image.alt = `${item.source_name} ${item.sample_id}`;
  imageButton.appendChild(image);

  const body = document.createElement("div");
  body.className = "card__body";

  const topLine = document.createElement("div");
  topLine.className = "card__topline";

  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = item.source_name;

  const index = document.createElement("span");
  index.className = "index";
  index.textContent = `Fig. ${String(item.index).padStart(3, "0")}`;

  const sampleId = document.createElement("p");
  sampleId.className = "sample-id";
  sampleId.textContent = item.sample_id;

  const prompt = document.createElement("p");
  prompt.className = "prompt";
  const promptText = getDisplayPrompt(item);
  prompt.textContent = promptText || "Prompt 缺失";
  if (!promptText) {
    prompt.classList.add("prompt--missing");
  }

  const path = document.createElement("p");
  path.className = "path";
  path.textContent = item.vis_path;

  topLine.append(badge, index);
  body.append(topLine, sampleId, prompt, path);
  card.append(imageButton, body);
  return card;
}

function renderGrid(options = {}) {
  const filteredItems = getFilteredItems();
  state.shownItems = options.shuffle ? shuffleItems(filteredItems) : filteredItems;
  const cards = state.shownItems.map((item, visibleIndex) => createCard(item, visibleIndex));
  elements.grid.replaceChildren(...cards);
  elements.emptyState.hidden = state.shownItems.length > 0;
  elements.statShown.textContent = String(state.shownItems.length);
}

function openViewer(visibleIndex) {
  state.viewerIndex = visibleIndex;
  updateViewer();
  if (typeof elements.viewer.showModal === "function") {
    elements.viewer.showModal();
  } else {
    elements.viewer.setAttribute("open", "");
  }
}

function updateViewer() {
  const item = state.shownItems[state.viewerIndex];
  if (!item) {
    return;
  }
  elements.viewerImage.src = item.image_path;
  elements.viewerImage.alt = `${item.source_name} ${item.sample_id}`;
  elements.viewerTitle.textContent = `${item.source_name} · ${item.sample_id}`;
  elements.viewerPrompt.textContent = getDisplayPrompt(item) || "Prompt 缺失";
}

function shiftViewer(delta) {
  if (!state.shownItems.length) {
    return;
  }
  state.viewerIndex = (state.viewerIndex + delta + state.shownItems.length) % state.shownItems.length;
  updateViewer();
}

function bindEvents() {
  elements.searchInput.addEventListener("input", (event) => {
    state.query = event.target.value;
    renderGrid();
  });

  elements.sourceFilter.addEventListener("change", (event) => {
    state.source = event.target.value;
    renderGrid();
  });

  elements.promptMode.addEventListener("change", (event) => {
    state.promptMode = event.target.value;
    renderGrid();
  });

  elements.shuffleButton.addEventListener("click", () => {
    renderGrid({ shuffle: true });
  });

  elements.resetButton.addEventListener("click", () => {
    state.query = "";
    state.source = "all";
    state.promptMode = "prompt";
    elements.searchInput.value = "";
    elements.sourceFilter.value = "all";
    elements.promptMode.value = "prompt";
    renderGrid();
  });

  elements.closeViewer.addEventListener("click", () => elements.viewer.close());
  elements.prevImage.addEventListener("click", () => shiftViewer(-1));
  elements.nextImage.addEventListener("click", () => shiftViewer(1));

  document.addEventListener("keydown", (event) => {
    if (!elements.viewer.open) {
      return;
    }
    if (event.key === "ArrowLeft") {
      shiftViewer(-1);
    }
    if (event.key === "ArrowRight") {
      shiftViewer(1);
    }
  });
}

function main() {
  initHeader();
  initFilters();
  renderMeta();
  bindEvents();
  renderGrid();
}

main();
"""


def write_gallery_files(site_dir: Path, site_payload: dict[str, Any], title: str) -> None:
    """
    用法: write_gallery_files(site_dir, payload, "Demo Gallery")
    作用: 将 HTML、CSS、JS 与数据文件写入网站目录
    输入: site_dir: Path，网站目录；site_payload: dict，前端数据；title: str，网页标题
    输出: None
    """
    assets_dir = site_dir / "assets"
    data_js = "window.GALLERY_DATA = " + json.dumps(site_payload, ensure_ascii=False, indent=2) + ";\n"

    write_text_file(site_dir / "index.html", render_html(title))
    write_text_file(assets_dir / "style.css", render_css())
    write_text_file(assets_dir / "app.js", render_app_js())
    write_text_file(assets_dir / "gallery-data.js", data_js)


def get_preview_hint(site_dir: Path, infer_dir: Path) -> tuple[Path, str]:
    """
    用法: root, url_path = get_preview_hint(site_dir, infer_dir)
    作用: 计算可同时访问 gallery 与 vis 的本地 HTTP 服务根目录和 URL 路径
    输入: site_dir: Path，网站目录；infer_dir: Path，推理目录
    输出: tuple[Path, str]，服务根目录与浏览器访问路径
    """
    preview_root = Path(os.path.commonpath([site_dir.resolve(), infer_dir.resolve()]))
    url_path = site_dir.resolve().relative_to(preview_root).as_posix()
    return preview_root, f"{url_path}/" if url_path != "." else ""


def build_gallery(args: argparse.Namespace) -> dict[str, Any]:
    """
    用法: payload = build_gallery(args)
    作用: 执行静态 gallery 数据合并与文件生成
    输入: args: argparse.Namespace，命令行参数
    输出: dict，写入前端的数据载荷
    """
    infer_dir = resolve_project_path(args.infer_dir)
    annotation_dir = resolve_project_path(args.annotation_dir)
    site_dir = resolve_project_path(args.output_dir) if args.output_dir else infer_dir / "gallery"
    title = str(args.title).strip() if args.title else infer_dir.name
    prompt_key = str(args.prompt_key).strip() or DEFAULT_PROMPT_KEY

    prediction_payload = load_prediction_payload(infer_dir)
    split = infer_split(prediction_payload, args.split)
    annotation_lookup = load_annotation_lookup(annotation_dir, split)
    predictions = prediction_payload["predictions"]
    items, missing_prompt_count = build_gallery_items(
        predictions=predictions,
        annotation_lookup=annotation_lookup,
        prompt_key=prompt_key,
        infer_dir=infer_dir,
        site_dir=site_dir,
    )
    site_payload = build_site_payload(
        prediction_payload=prediction_payload,
        items=items,
        infer_dir=infer_dir,
        annotation_dir=annotation_dir,
        split=split,
        prompt_key=prompt_key,
        title=title,
        missing_prompt_count=missing_prompt_count,
    )
    write_gallery_files(site_dir, site_payload, title)
    return site_payload


def main() -> None:
    """
    用法: main()
    作用: 命令行入口，生成推理结果展示网站并打印摘要
    输入: 无，参数来自命令行
    输出: None
    """
    args = build_parser().parse_args()
    site_payload = build_gallery(args)
    infer_dir = resolve_project_path(args.infer_dir)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else infer_dir / "gallery"
    preview_root, preview_url_path = get_preview_hint(output_dir, infer_dir)
    metadata = site_payload["metadata"]
    print("Gallery 生成完成")
    print(f"输出目录: {output_dir}")
    print(f"入口文件: {output_dir / 'index.html'}")
    print(f"样本数量: {metadata['item_count']}")
    print(f"Prompt 缺失: {metadata['missing_prompt_count']}")
    print("本地预览示例:")
    print(f"  cd {preview_root}")
    print("  python -m http.server 8000")
    print(f"  打开 http://localhost:8000/{preview_url_path}")


if __name__ == "__main__":
    main()
