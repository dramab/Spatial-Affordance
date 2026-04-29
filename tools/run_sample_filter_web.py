#!/usr/bin/env python3
"""
tools/run_sample_filter_web.py
------------------------------
职责：启动一个本地网页，用键盘快速筛选图片样本，并将坏样本名称输出为 txt 文件。

用法：
    conda run -n spatial python tools/run_sample_filter_web.py \
        --image-dir outputs/placement_rgb_bbox_vis_dopose

作用：
    - 读取指定目录中的图片文件
    - 在浏览器中一次展示一张图片，方便人工快速筛选
    - 按空格跳到下一张，表示当前样本通过
    - 按回车将当前样本记为坏样本，并继续跳到下一张
    - 将坏样本文件名实时写入 txt，避免中途退出导致结果丢失

输入：
    --image-dir: 待筛选图片目录，支持 png/jpg/jpeg/webp/bmp
    --output-txt: 坏样本名称输出路径；默认写入 image_dir/bad_samples.txt
    --host: 本地服务监听地址，默认 127.0.0.1
    --port: 本地服务端口，默认 8765；传 0 表示自动分配可用端口

输出：
    - 坏样本 txt 文件，每行一个图片文件名
    - 本地网页服务地址，用于浏览器打开后进行筛选

使用示例：
    conda run -n spatial python tools/run_sample_filter_web.py \
        --image-dir outputs/placement_rgb_bbox_vis_dopose \
        --output-txt outputs/placement_rgb_bbox_vis_dopose/bad_samples.txt \
        --port 8765
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
HTML_CONTENT = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>样本筛选器</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255, 252, 246, 0.92);
      --line: #d8cdbd;
      --text: #2f241b;
      --muted: #6c5b4d;
      --accent: #8c4f2b;
      --accent-strong: #6d3a1e;
      --good: #e9f2e4;
      --bad: #f6dfd8;
      --shadow: 0 20px 60px rgba(72, 48, 31, 0.14);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Source Han Serif SC", "Noto Serif CJK SC", "Songti SC", serif;
      background:
        radial-gradient(circle at top left, rgba(140, 79, 43, 0.16), transparent 32%),
        radial-gradient(circle at top right, rgba(190, 150, 104, 0.18), transparent 28%),
        linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
    }

    body::before {
      position: fixed;
      inset: 0;
      z-index: -1;
      content: "";
      pointer-events: none;
      background:
        repeating-linear-gradient(
          0deg,
          transparent 0,
          transparent 29px,
          rgba(47, 36, 27, 0.04) 30px
        );
    }

    .page {
      width: min(1240px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 36px;
    }

    .hero {
      display: grid;
      gap: 18px;
      margin-bottom: 18px;
    }

    .hero h1 {
      margin: 0;
      font-size: clamp(1.8rem, 4vw, 3rem);
      line-height: 1.08;
      letter-spacing: -0.03em;
    }

    .hero p {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
    }

    .status-bar {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }

    .status-card {
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .status-card span {
      display: block;
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 0.86rem;
    }

    .status-card strong {
      font-size: 1.15rem;
    }

    .viewer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
      align-items: start;
    }

    .stage {
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .stage img {
      display: block;
      width: 100%;
      height: min(78vh, 920px);
      object-fit: contain;
      background:
        linear-gradient(135deg, rgba(140, 79, 43, 0.05), rgba(47, 36, 27, 0.03));
    }

    .sidebar {
      display: grid;
      gap: 16px;
    }

    .panel {
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .panel h2 {
      margin: 0 0 10px;
      font-size: 1rem;
    }

    .panel p,
    .panel li {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
    }

    .sample-name {
      margin-top: 8px;
      word-break: break-all;
      color: var(--text);
      font-weight: 700;
      line-height: 1.5;
    }

    .jump-controls {
      display: grid;
      gap: 10px;
    }

    .jump-controls input {
      width: 100%;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      color: var(--text);
      font: inherit;
      background: #fffdfa;
    }

    .jump-controls input:focus {
      outline: 2px solid rgba(140, 79, 43, 0.18);
      outline-offset: 1px;
      border-color: var(--accent);
    }

    .actions {
      display: grid;
      gap: 12px;
    }

    .actions button {
      padding: 14px 16px;
      border: 0;
      border-radius: 14px;
      color: #fffaf5;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }

    .actions button:hover {
      transform: translateY(-1px);
    }

    .actions button:active {
      transform: translateY(0);
      opacity: 0.92;
    }

    .pass-button {
      background: linear-gradient(135deg, #5d7d56, #3f5e3c);
    }

    .bad-button {
      background: linear-gradient(135deg, var(--accent), var(--accent-strong));
    }

    .hint {
      padding: 12px 14px;
      border-radius: 14px;
      font-size: 0.92rem;
    }

    .hint--good {
      background: var(--good);
    }

    .hint--bad {
      background: var(--bad);
    }

    .finished {
      padding: 48px 24px;
      text-align: center;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .finished h2 {
      margin: 0 0 12px;
      font-size: clamp(1.5rem, 3vw, 2.2rem);
    }

    .finished p {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
    }

    .message {
      min-height: 24px;
      color: var(--accent-strong);
      font-size: 0.95rem;
    }

    @media (max-width: 980px) {
      .status-bar,
      .viewer {
        grid-template-columns: 1fr;
      }

      .stage img {
        height: 58vh;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>样本筛选器</h1>
      <p>空格表示当前样本通过并跳转下一张；回车表示当前样本不合格，并继续跳转下一张。</p>
    </section>

    <section class="status-bar">
      <div class="status-card">
        <span>当前进度</span>
        <strong id="progressText">-</strong>
      </div>
      <div class="status-card">
        <span>坏样本数量</span>
        <strong id="badCountText">0</strong>
      </div>
      <div class="status-card">
        <span>图片目录</span>
        <strong id="imageDirText">-</strong>
      </div>
      <div class="status-card">
        <span>输出文件</span>
        <strong id="outputTxtText">-</strong>
      </div>
    </section>

    <div id="mainContainer"></div>
  </div>

  <script>
    const PRELOAD_AHEAD_COUNT = 1;

    const state = {
      items: [],
      badNames: new Set(),
      currentIndex: 0,
      imageDir: "",
      outputTxt: "",
      busy: false,
    };
    const preloadedImages = new Set();

    const elements = {
      progressText: document.getElementById("progressText"),
      badCountText: document.getElementById("badCountText"),
      imageDirText: document.getElementById("imageDirText"),
      outputTxtText: document.getElementById("outputTxtText"),
      mainContainer: document.getElementById("mainContainer"),
    };

    function escapeHtml(text) {
      return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function getCurrentItem() {
      return state.items[state.currentIndex] || null;
    }

    function updateHeader() {
      const total = state.items.length;
      const current = total ? Math.min(state.currentIndex + 1, total) : 0;
      elements.progressText.textContent = total ? `${current} / ${total}` : "0 / 0";
      elements.badCountText.textContent = String(state.badNames.size);
      elements.imageDirText.textContent = state.imageDir || "-";
      elements.outputTxtText.textContent = state.outputTxt || "-";
    }

    function renderFinished() {
      elements.mainContainer.innerHTML = `
        <section class="finished">
          <h2>筛选完成</h2>
          <p>坏样本共 ${state.badNames.size} 个，结果已写入：</p>
          <p><strong>${escapeHtml(state.outputTxt)}</strong></p>
        </section>
      `;
    }

    function renderViewer() {
      const item = getCurrentItem();
      if (!item) {
        renderFinished();
        return;
      }

      const isBad = state.badNames.has(item.name);
      elements.mainContainer.innerHTML = `
        <section class="viewer">
          <div class="stage">
            <img src="/image/${encodeURIComponent(item.name)}" alt="${escapeHtml(item.name)}">
          </div>
          <aside class="sidebar">
            <section class="panel">
              <h2>当前样本</h2>
              <p class="sample-name">${escapeHtml(item.name)}</p>
            </section>
            <section class="panel">
              <h2>跳转</h2>
              <div class="jump-controls">
                <input id="jumpInput" type="text" placeholder="输入序号或完整文件名">
                <button class="bad-button" id="jumpButton" type="button">跳转</button>
              </div>
            </section>
            <section class="panel actions">
              <button class="pass-button" id="passButton" type="button">空格：通过并下一张</button>
              <button class="bad-button" id="badButton" type="button">回车：标记为坏样本并下一张</button>
              <div class="message" id="messageText">${isBad ? "当前样本已在坏样本列表中。" : ""}</div>
            </section>
            <section class="panel">
              <h2>快捷键说明</h2>
              <p class="hint hint--good">Space：当前样本正常，直接跳到下一张。</p>
              <p class="hint hint--bad">Enter：当前样本不合格，写入 txt 后跳到下一张。</p>
            </section>
          </aside>
        </section>
      `;

      document.getElementById("passButton").addEventListener("click", () => goNext());
      document.getElementById("badButton").addEventListener("click", () => markBadAndNext());
      document.getElementById("jumpButton").addEventListener("click", () => jumpToTarget());
      document.getElementById("jumpInput").addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          jumpToTarget();
        }
      });
    }

    function render() {
      updateHeader();
      renderViewer();
      preloadUpcomingImages();
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, options);
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `Request failed: ${response.status}`);
      }
      return response.json();
    }

    function goNext() {
      if (state.busy) {
        return;
      }
      if (state.currentIndex < state.items.length) {
        state.currentIndex += 1;
        render();
      }
    }

    function preloadUpcomingImages() {
      const startIndex = state.currentIndex + 1;
      const endIndex = Math.min(state.items.length, startIndex + PRELOAD_AHEAD_COUNT);
      for (let index = startIndex; index < endIndex; index += 1) {
        const item = state.items[index];
        if (!item) {
          continue;
        }
        const imageUrl = `/image/${encodeURIComponent(item.name)}`;
        if (preloadedImages.has(imageUrl)) {
          continue;
        }
        preloadedImages.add(imageUrl);
        const image = new Image();
        image.decoding = "async";
        image.src = imageUrl;
      }
    }

    function jumpToTarget() {
      if (state.busy) {
        return;
      }

      const jumpInput = document.getElementById("jumpInput");
      const messageText = document.getElementById("messageText");
      if (!jumpInput || !messageText) {
        return;
      }

      const rawValue = jumpInput.value.trim();
      if (!rawValue) {
        messageText.textContent = "请输入序号或完整文件名。";
        return;
      }

      let targetIndex = -1;
      if (/^\d+$/.test(rawValue)) {
        const displayIndex = Number.parseInt(rawValue, 10);
        targetIndex = displayIndex - 1;
        if (targetIndex < 0 || targetIndex >= state.items.length) {
          messageText.textContent = `序号超出范围，当前共有 ${state.items.length} 张。`;
          return;
        }
      } else {
        targetIndex = state.items.findIndex((item) => item.name === rawValue);
        if (targetIndex < 0) {
          messageText.textContent = "未找到对应文件名，请输入完整样本名。";
          return;
        }
      }

      state.currentIndex = targetIndex;
      render();
      const nextMessageText = document.getElementById("messageText");
      if (nextMessageText) {
        nextMessageText.textContent = `已跳转到第 ${targetIndex + 1} 张。`;
      }
    }

    async function markBadAndNext() {
      const item = getCurrentItem();
      if (!item || state.busy) {
        return;
      }
      state.busy = true;
      try {
        const payload = await fetchJson("/api/mark_bad", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({name: item.name}),
        });
        state.badNames = new Set(payload.bad_names || []);
        state.currentIndex += 1;
        render();
      } catch (error) {
        const messageText = document.getElementById("messageText");
        if (messageText) {
          messageText.textContent = `写入失败：${error.message}`;
        }
      } finally {
        state.busy = false;
        updateHeader();
      }
    }

    async function loadState() {
      const payload = await fetchJson("/api/state");
      state.items = payload.items || [];
      state.badNames = new Set(payload.bad_names || []);
      state.imageDir = payload.image_dir || "";
      state.outputTxt = payload.output_txt || "";
      render();
    }

    document.addEventListener("keydown", (event) => {
      if (event.repeat || state.busy || state.currentIndex >= state.items.length) {
        return;
      }
      if (event.key === " ") {
        event.preventDefault();
        goNext();
      } else if (event.key === "Enter") {
        event.preventDefault();
        markBadAndNext();
      }
    });

    loadState().catch((error) => {
      elements.mainContainer.innerHTML = `
        <section class="finished">
          <h2>加载失败</h2>
          <p>${escapeHtml(error.message)}</p>
        </section>
      `;
    });
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser，配置完成的解析器
    """
    parser = argparse.ArgumentParser(description="启动本地网页筛选图片样本")
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="待筛选图片目录",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="坏样本输出 txt 路径，默认写入 image_dir/bad_samples.txt",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="本地服务监听地址，默认 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="本地服务端口，默认 8765；传 0 表示自动分配可用端口",
    )
    return parser


def list_image_files(image_dir: Path) -> list[Path]:
    """
    用法: image_files = list_image_files(Path("outputs/demo"))
    作用: 收集并排序目录中的图片文件
    输入: image_dir: Path，待筛选图片目录
    输出: list[Path]，按文件名排序后的图片路径列表
    """
    image_files = [
        path for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_files:
        raise FileNotFoundError(f"目录下未找到可筛选图片: {image_dir}")
    return image_files


def load_existing_bad_names(output_txt: Path) -> list[str]:
    """
    用法: bad_names = load_existing_bad_names(Path("outputs/demo/bad_samples.txt"))
    作用: 读取已有坏样本 txt，便于断点续筛时去重
    输入: output_txt: Path，坏样本输出文件
    输出: list[str]，按原顺序读取到的坏样本文件名列表
    """
    if not output_txt.exists():
        return []

    bad_names: list[str] = []
    seen_names: set[str] = set()
    for line in output_txt.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        bad_names.append(name)
    return bad_names


def write_bad_names(output_txt: Path, bad_names: list[str]) -> None:
    """
    用法: write_bad_names(Path("bad_samples.txt"), ["a.png", "b.png"])
    作用: 将坏样本文件名完整写回 txt 文件
    输入: output_txt: Path，输出路径；bad_names: list[str]，坏样本文件名列表
    输出: None
    """
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(bad_names)
    if content:
        content += "\n"
    output_txt.write_text(content, encoding="utf-8")


class FilterApp:
    """
    用法: app = FilterApp(image_dir, output_txt)
    作用: 管理筛选图片列表与坏样本落盘逻辑
    输入: image_dir: Path，图片目录；output_txt: Path，坏样本输出文件
    输出: FilterApp，封装后的应用状态对象
    """

    def __init__(self, image_dir: Path, output_txt: Path) -> None:
        """
        用法: app = FilterApp(image_dir, output_txt)
        作用: 初始化筛选应用，加载图片列表与已有坏样本
        输入: image_dir: Path，图片目录；output_txt: Path，坏样本输出文件
        输出: None
        """
        self.image_dir = image_dir.resolve()
        self.output_txt = output_txt.resolve()
        self.image_files = list_image_files(self.image_dir)
        self.image_lookup = {path.name: path for path in self.image_files}
        self.bad_names = load_existing_bad_names(self.output_txt)
        self.bad_name_set = set(self.bad_names)

    def build_state_payload(self) -> dict[str, Any]:
        """
        用法: payload = app.build_state_payload()
        作用: 生成前端初始化所需的筛选状态
        输入: 无
        输出: dict[str, Any]，包含图片列表与坏样本列表的状态载荷
        """
        return {
            "items": [{"name": path.name} for path in self.image_files],
            "bad_names": list(self.bad_names),
            "image_dir": str(self.image_dir),
            "output_txt": str(self.output_txt),
        }

    def get_image_path(self, image_name: str) -> Path:
        """
        用法: image_path = app.get_image_path("demo.png")
        作用: 根据图片文件名返回对应路径，并校验名称是否合法
        输入: image_name: str，图片文件名
        输出: Path，对应图片绝对路径
        """
        image_path = self.image_lookup.get(image_name)
        if image_path is None:
            raise FileNotFoundError(f"未找到图片: {image_name}")
        return image_path

    def mark_bad(self, image_name: str) -> dict[str, Any]:
        """
        用法: payload = app.mark_bad("demo.png")
        作用: 将指定图片记为坏样本，并实时写回 txt
        输入: image_name: str，图片文件名
        输出: dict[str, Any]，最新坏样本列表
        """
        self.get_image_path(image_name)
        if image_name not in self.bad_name_set:
            self.bad_name_set.add(image_name)
            self.bad_names.append(image_name)
            write_bad_names(self.output_txt, self.bad_names)
        return {"bad_names": list(self.bad_names)}


class FilterRequestHandler(BaseHTTPRequestHandler):
    """
    用法: handler = FilterRequestHandler(*args, app=app, **kwargs)
    作用: 处理网页、图片和筛选接口请求
    输入: 由 HTTPServer 注入的请求参数，以及 app: FilterApp
    输出: FilterRequestHandler，请求处理器实例
    """

    server_version = "SampleFilterHTTP/1.0"

    def __init__(self, *args: Any, app: FilterApp, **kwargs: Any) -> None:
        """
        用法: handler = FilterRequestHandler(*args, app=app, **kwargs)
        作用: 在请求处理器中注入筛选应用状态
        输入: *args/**kwargs 为 HTTPServer 参数；app: FilterApp，筛选应用
        输出: None
        """
        self.app = app
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        """
        用法: 由 HTTPServer 在收到 GET 请求时自动调用
        作用: 返回主页、状态数据或图片内容
        输入: HTTP GET 请求
        输出: None
        """
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_CONTENT)
            return
        if parsed.path == "/api/state":
            self._send_json(self.app.build_state_payload())
            return
        if parsed.path.startswith("/image/"):
            image_name = unquote(parsed.path.removeprefix("/image/"))
            self._send_image(image_name)
            return
        self._send_error_response(HTTPStatus.NOT_FOUND, "未找到请求资源")

    def do_POST(self) -> None:
        """
        用法: 由 HTTPServer 在收到 POST 请求时自动调用
        作用: 处理坏样本标记请求，并写回 txt
        输入: HTTP POST 请求
        输出: None
        """
        parsed = urlparse(self.path)
        if parsed.path != "/api/mark_bad":
            self._send_error_response(HTTPStatus.NOT_FOUND, "未找到请求资源")
            return

        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_error_response(HTTPStatus.BAD_REQUEST, str(exc))
            return
        image_name = str(payload.get("name", "")).strip()
        if not image_name:
            self._send_error_response(HTTPStatus.BAD_REQUEST, "缺少图片名称")
            return

        try:
            response_payload = self.app.mark_bad(image_name)
        except FileNotFoundError as exc:
            self._send_error_response(HTTPStatus.NOT_FOUND, str(exc))
            return
        self._send_json(response_payload)

    def log_message(self, format: str, *args: Any) -> None:
        """
        用法: handler.log_message(fmt, *args)
        作用: 精简默认日志输出，避免刷屏
        输入: format: str，日志模板；*args: Any，模板参数
        输出: None
        """
        return

    def _read_json_body(self) -> dict[str, Any]:
        """
        用法: payload = self._read_json_body()
        作用: 读取并解析 POST 请求体中的 JSON 数据
        输入: 当前 HTTP 请求体
        输出: dict[str, Any]，解析后的 JSON 对象
        """
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""
        if not raw_body:
            return {}
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("请求体不是合法 JSON") from exc

    def _send_html(self, html_content: str) -> None:
        """
        用法: self._send_html("<html>...</html>")
        作用: 向浏览器返回 HTML 页面
        输入: html_content: str，HTML 文本
        输出: None
        """
        data = html_content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any]) -> None:
        """
        用法: self._send_json({"ok": True})
        作用: 向前端返回 JSON 数据
        输入: payload: dict[str, Any]，待返回 JSON 对象
        输出: None
        """
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error_response(self, status: HTTPStatus, message: str) -> None:
        """
        用法: self._send_error_response(HTTPStatus.NOT_FOUND, "未找到图片")
        作用: 返回 UTF-8 编码的错误响应，避免 send_error 对中文状态消息编码失败
        输入: status: HTTPStatus，HTTP 状态码；message: str，错误信息
        输出: None
        """
        data = message.encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_image(self, image_name: str) -> None:
        """
        用法: self._send_image("demo.png")
        作用: 读取并返回指定图片文件
        输入: image_name: str，图片文件名
        输出: None
        """
        try:
            image_path = self.app.get_image_path(image_name)
        except FileNotFoundError as exc:
            self._send_error_response(HTTPStatus.NOT_FOUND, str(exc))
            return

        image_bytes = image_path.read_bytes()
        mime_type = guess_mime_type(image_path)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(image_bytes)))
        self.end_headers()
        self.wfile.write(image_bytes)


def guess_mime_type(image_path: Path) -> str:
    """
    用法: mime_type = guess_mime_type(Path("demo.png"))
    作用: 根据图片后缀返回合适的 MIME 类型
    输入: image_path: Path，图片路径
    输出: str，MIME 类型
    """
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    """
    用法: image_dir, output_txt = validate_args(args)
    作用: 校验输入目录与输出路径，并返回规范化结果
    输入: args: argparse.Namespace，命令行解析结果
    输出: tuple[Path, Path]，规范化后的图片目录与输出 txt 路径
    """
    image_dir = args.image_dir.resolve()
    if not image_dir.exists() or not image_dir.is_dir():
        raise NotADirectoryError(f"图片目录不存在或不是目录: {image_dir}")

    output_txt = args.output_txt.resolve() if args.output_txt else image_dir / "bad_samples.txt"
    return image_dir, output_txt


def run_server(app: FilterApp, host: str, port: int) -> None:
    """
    用法: run_server(app, "127.0.0.1", 8765)
    作用: 启动本地 HTTP 服务，供浏览器进行样本筛选
    输入: app: FilterApp，筛选应用；host: str，监听地址；port: int，监听端口
    输出: None
    """
    handler_class = partial(FilterRequestHandler, app=app)
    with ThreadingHTTPServer((host, port), handler_class) as server:
        actual_host, actual_port = server.server_address[:2]
        print("筛选网页已启动")
        print(f"图片目录: {app.image_dir}")
        print(f"输出文件: {app.output_txt}")
        print(f"访问地址: http://{actual_host}:{actual_port}")
        print("键盘操作: 空格下一张，回车标记坏样本并下一张")
        print("按 Ctrl+C 可停止服务")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n服务已停止")


def main() -> None:
    """
    用法: main()
    作用: 解析命令行参数，初始化筛选应用并启动网页服务
    输入: 命令行参数
    输出: None
    """
    parser = build_parser()
    args = parser.parse_args()
    image_dir, output_txt = validate_args(args)
    app = FilterApp(image_dir=image_dir, output_txt=output_txt)
    run_server(app=app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
