#!/bin/bash
# Lightweight progress monitor for long-running background diagnostics.
# Runs in a tmux session, appends one status line per minute to monitor.log.
# Its output is intentionally kept out of the assistant's context.
cd /home/kamesh/workspace/transientbrokerage || exit 1
OUT=scripts/diagnostics/out
mkdir -p "$OUT"
while true; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  # Any running julia .jl jobs (pid / elapsed seconds / %cpu / script)
  procs=$(ps -eo pid,etimes,pcpu,args | awk '/[j]ulia.*\.jl/ {printf "pid=%s t=%ss cpu=%s%% %s\n",$1,$2,$3,$NF}')
  [ -z "$procs" ] && procs="(no julia .jl jobs running)"
  # Done-markers for the active comparison logs (if present)
  status=""
  lines=$(grep -cE '^CELL ' "$OUT/sweep3.log" 2>/dev/null)
  status="sweep3 CELL lines=$lines/24"
  printf '[%s]\n%s\n%s\n\n' "$ts" "$procs" "$status" >> "$OUT/monitor.log"
  sleep 60
done
