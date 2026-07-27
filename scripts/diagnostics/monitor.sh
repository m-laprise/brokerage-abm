#!/bin/bash
# Lightweight progress monitor for long-running background diagnostics.
# Runs in a tmux session and appends a status snapshot at a configurable interval.
# Its output is intentionally kept out of the assistant's context.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO" || exit 1

OUT="${BROKERAGE_ABM_DIAGNOSTIC_OUT:-$SCRIPT_DIR/out}"
INTERVAL="${BROKERAGE_ABM_MONITOR_INTERVAL:-60}"
mkdir -p "$OUT"

while true; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  # Any running julia .jl jobs (pid / elapsed seconds / %cpu / script)
  procs=$(ps -eo pid,etimes,pcpu,args | awk '/[j]ulia.*\.jl/ {printf "pid=%s t=%ss cpu=%s%% %s\n",$1,$2,$3,$NF}')
  [ -z "$procs" ] && procs="(no julia .jl jobs running)"

  # Done-markers for all diagnostic logs except this monitor's own log.
  status=""
  for log in "$OUT"/*.log; do
    [ -e "$log" ] || continue
    [ "$(basename "$log")" = "monitor.log" ] && continue
    lines=$(grep -cE '^CELL ' "$log" 2>/dev/null || true)
    status="${status}$(basename "$log") CELL lines=$lines
"
  done
  [ -z "$status" ] && status="(no diagnostic logs found)
"

  printf '[%s]\n%s\n%s\n\n' "$ts" "$procs" "$status" >> "$OUT/monitor.log"
  sleep "$INTERVAL"
done
