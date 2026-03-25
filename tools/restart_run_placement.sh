#!/usr/bin/env bash

set -u

CMD=(
python tools/run_placement.py \
        --config configs/annotation/placement_housecat6d.yaml \
        --batch --workers 4 \
        --output outputs/housecat6d_placement10
)

RESTART_DELAY="${RESTART_DELAY:-5}"
MEMORY_MAX="${MEMORY_MAX:-}"
MEMORY_CHECK_INTERVAL="${MEMORY_CHECK_INTERVAL:-2}"
ATTEMPT=1

parse_memory_to_kb() {
  local raw="$1"
  local upper number unit

  upper="$(printf '%s' "$raw" | tr '[:lower:]' '[:upper:]')"

  if [[ "$upper" =~ ^([0-9]+)([KMGTP]?)B?$ ]]; then
    number="${BASH_REMATCH[1]}"
    unit="${BASH_REMATCH[2]}"
  else
    echo "Invalid MEMORY_MAX value: $raw" >&2
    return 1
  fi

  case "$unit" in
    "") echo "$number" ;;
    K) echo "$number" ;;
    M) echo $((number * 1024)) ;;
    G) echo $((number * 1024 * 1024)) ;;
    T) echo $((number * 1024 * 1024 * 1024)) ;;
    P) echo $((number * 1024 * 1024 * 1024 * 1024)) ;;
    *)
      echo "Unsupported MEMORY_MAX unit: $raw" >&2
      return 1
      ;;
  esac
}

get_group_rss_kb() {
  local pgid="$1"
  local pids

  pids="$(pgrep -g "$pgid" | tr '\n' ',' | sed 's/,$//')"
  if [ -z "$pids" ]; then
    echo 0
    return 0
  fi

  ps -o rss= -p "$pids" | awk '{sum += $1} END {print sum + 0}'
}

monitor_memory_limit() {
  local leader_pid="$1"
  local limit_kb="$2"
  local rss_kb

  while kill -0 "$leader_pid" 2>/dev/null; do
    rss_kb="$(get_group_rss_kb "$leader_pid")"

    if [ "$rss_kb" -gt "$limit_kb" ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Memory limit exceeded: ${rss_kb} KB > ${limit_kb} KB. Killing process group ${leader_pid}."
      kill -TERM -- -"$leader_pid" 2>/dev/null || true
      sleep 3
      kill -KILL -- -"$leader_pid" 2>/dev/null || true
      return 0
    fi

    sleep "$MEMORY_CHECK_INTERVAL"
  done
}

run_cmd() {
  local leader_pid monitor_pid exit_code limit_kb

  if [ -n "$MEMORY_MAX" ]; then
    limit_kb="$(parse_memory_to_kb "$MEMORY_MAX")" || return 2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Applying user-space memory monitor: ${MEMORY_MAX}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Check interval: ${MEMORY_CHECK_INTERVAL}s"
  fi

  setsid "${CMD[@]}" &
  leader_pid=$!

  if [ -n "$MEMORY_MAX" ]; then
    monitor_memory_limit "$leader_pid" "$limit_kb" &
    monitor_pid=$!
  else
    monitor_pid=""
  fi

  wait "$leader_pid"
  exit_code=$?

  if [ -n "$monitor_pid" ]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi

  return "$exit_code"
}

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting attempt ${ATTEMPT}..."
  run_cmd
  EXIT_CODE=$?

  if [ "$EXIT_CODE" -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Command exited normally with code 0. Stopping."
    exit 0
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Command exited abnormally with code ${EXIT_CODE}."
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting in ${RESTART_DELAY}s..."
  sleep "$RESTART_DELAY"
  ATTEMPT=$((ATTEMPT + 1))
done
