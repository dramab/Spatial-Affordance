#!/usr/bin/env bash
# 用法:
#   bash tools/download_gdrive.sh <google_drive_url> [output_dir]
#
# 作用:
#   使用 gdown 下载 Google Drive 文件或文件夹，并在网络中断或下载失败时自动重试。
#
# 输入:
#   google_drive_url: Google Drive 文件或文件夹链接，必填。
#   output_dir: 下载保存目录，可选，默认 outputs/gdrive_downloads。
#
# 输出:
#   将下载内容保存到 output_dir，并在终端输出下载状态。
#
# 可选环境变量:
#   MAX_RETRIES: 最大重试次数，默认 5。
#   RETRY_INTERVAL: 每次失败后的等待秒数，默认 10。
#
# 示例:
#   bash tools/download_gdrive.sh "https://drive.google.com/file/d/xxx/view" outputs/downloads
#   MAX_RETRIES=10 RETRY_INTERVAL=20 bash tools/download_gdrive.sh "https://drive.google.com/drive/folders/xxx" outputs/data

set -u

DRIVE_URL="${1:-}"
OUTPUT_DIR="${2:-outputs/gdrive_downloads}"
MAX_RETRIES="${MAX_RETRIES:-20}"
RETRY_INTERVAL="${RETRY_INTERVAL:-10}"

print_usage() {
  # 用法: print_usage
  # 作用: 输出脚本参数说明和运行示例。
  # 输入: 无。
  # 输出: 参数说明文本。
  cat <<'EOF'
Usage:
  bash tools/download_gdrive.sh <google_drive_url> [output_dir]

Examples:
  bash tools/download_gdrive.sh "https://drive.google.com/file/d/xxx/view" outputs/downloads
  MAX_RETRIES=10 RETRY_INTERVAL=20 bash tools/download_gdrive.sh "https://drive.google.com/drive/folders/xxx" outputs/data
EOF
}

log_message() {
  # 用法: log_message <message>
  # 作用: 统一输出带时间戳的日志。
  # 输入: message 为待输出文本。
  # 输出: 带时间戳的日志行。
  local message="$1"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${message}"
}

validate_positive_integer() {
  # 用法: validate_positive_integer <name> <value>
  # 作用: 校验指定变量是否为正整数。
  # 输入: name 为变量名，value 为变量值。
  # 输出: 校验失败时输出错误信息。
  local name="$1"
  local value="$2"

  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ${name} must be a positive integer, got '${value}'." >&2
    return 1
  fi
}

is_gdrive_folder_url() {
  # 用法: is_gdrive_folder_url <google_drive_url>
  # 作用: 判断链接是否为 Google Drive 文件夹链接。
  # 输入: google_drive_url 为待检查链接。
  # 输出: 是文件夹链接返回 0，否则返回 1。
  local url="$1"

  [[ "$url" == *"drive.google.com/drive/folders/"* || "$url" == *"folders/"* ]]
}

build_gdown_command() {
  # 用法: build_gdown_command <google_drive_url> <output_dir>
  # 作用: 根据链接类型生成 gdown 下载命令。
  # 输入: google_drive_url 为下载链接，output_dir 为保存目录。
  # 输出: 全局数组 GDOWN_CMD。
  local url="$1"
  local output_dir="$2"

  if is_gdrive_folder_url "$url"; then
    GDOWN_CMD=(gdown --folder "$url" --output "$output_dir" --continue)
  else
    GDOWN_CMD=(gdown "$url" --output "$output_dir" --continue)
  fi
}

run_download_with_retries() {
  # 用法: run_download_with_retries
  # 作用: 执行 gdown 下载命令，失败后按配置自动重试。
  # 输入: 使用全局数组 GDOWN_CMD、MAX_RETRIES、RETRY_INTERVAL。
  # 输出: 下载成功返回 0，超过重试次数仍失败返回最后一次退出码。
  local attempt=1
  local exit_code=0

  while [ "$attempt" -le "$MAX_RETRIES" ]; do
    log_message "Starting download attempt ${attempt}/${MAX_RETRIES}..."
    "${GDOWN_CMD[@]}"
    exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
      log_message "Download completed successfully. Output directory: ${OUTPUT_DIR}"
      return 0
    fi

    log_message "Download failed with exit code ${exit_code}."
    if [ "$attempt" -lt "$MAX_RETRIES" ]; then
      log_message "Retrying in ${RETRY_INTERVAL}s..."
      sleep "$RETRY_INTERVAL"
    fi

    attempt=$((attempt + 1))
  done

  log_message "Download failed after ${MAX_RETRIES} attempts."
  return "$exit_code"
}

if [ -z "$DRIVE_URL" ] || [ "$DRIVE_URL" = "-h" ] || [ "$DRIVE_URL" = "--help" ]; then
  print_usage
  if [ -z "$DRIVE_URL" ]; then
    exit 1
  fi
  exit 0
fi

if ! command -v gdown >/dev/null 2>&1; then
  echo "Error: gdown is not installed or not in PATH." >&2
  echo "Please install it manually first, for example:" >&2
  echo "  pip install gdown" >&2
  exit 1
fi

validate_positive_integer "MAX_RETRIES" "$MAX_RETRIES" || exit 1
validate_positive_integer "RETRY_INTERVAL" "$RETRY_INTERVAL" || exit 1

mkdir -p "$OUTPUT_DIR"
build_gdown_command "$DRIVE_URL" "$OUTPUT_DIR"

log_message "Google Drive URL: ${DRIVE_URL}"
log_message "Output directory: ${OUTPUT_DIR}"
if is_gdrive_folder_url "$DRIVE_URL"; then
  log_message "Detected Google Drive folder URL."
else
  log_message "Detected Google Drive file URL."
fi

run_download_with_retries
