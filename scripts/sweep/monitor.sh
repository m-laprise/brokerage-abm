#!/bin/bash
# Sweep progress monitor — run detached in tmux (see submit.sh notes). Appends one
# status line every INTERVAL seconds to <sweep_dir>/logs/monitor.log AND the pane.
# Ground-truth progress = completed shards on disk (each (cell,seed) task writes
# one seed_*.jld2); queue state is informational. Runs until killed.
#   bash monitor.sh <sweep_dir> [interval_seconds=300]
set -u
SWEEP_DIR="${1:?usage: monitor.sh <sweep_dir> [interval_s]}"
INTERVAL="${2:-300}"
LOG="$SWEEP_DIR/logs/monitor.log"
mkdir -p "$SWEEP_DIR/logs"

prev=0
echo "[$(date '+%F %T')] monitor started: $SWEEP_DIR (every ${INTERVAL}s)" | tee -a "$LOG"
while true; do
    ts=$(date '+%F %T')
    NRUNS=0; NPLOT=0
    # shellcheck disable=SC1090
    [ -f "$SWEEP_DIR/counts.env" ] && source "$SWEEP_DIR/counts.env"

    shards=$(find "$SWEEP_DIR/oat" "$SWEEP_DIR/phase" -name 'seed_*.jld2' 2>/dev/null | wc -l)
    data=$(find "$SWEEP_DIR" -name 'data.jld2' 2>/dev/null | wc -l)
    png=$(find "$SWEEP_DIR" -name '*.png' 2>/dev/null | wc -l)

    run_R=$(squeue -u "$USER" -h -r -n tb_run  -t RUNNING 2>/dev/null | wc -l)
    run_P=$(squeue -u "$USER" -h -r -n tb_run  -t PENDING 2>/dev/null | wc -l)
    plt_R=$(squeue -u "$USER" -h -r -n tb_plot -t RUNNING 2>/dev/null | wc -l)
    plt_P=$(squeue -u "$USER" -h -r -n tb_plot -t PENDING 2>/dev/null | wc -l)

    d=$(( shards - prev )); prev=$shards
    pct=0;  [ "${NRUNS:-0}" -gt 0 ] && pct=$(( shards * 100 / NRUNS ))
    eta="--"
    if [ "$d" -gt 0 ] && [ "${NRUNS:-0}" -gt 0 ]; then
        rem=$(( NRUNS - shards ))
        eta="$(( rem * INTERVAL / d / 60 ))m"   # ~remaining at current rate
    fi

    printf '[%s] runs %d/%d (%d%%)  +%d/%ss  ETA~%s | queue run R%d/P%d  plot R%d/P%d | data %d  png %d\n' \
        "$ts" "$shards" "${NRUNS:-0}" "$pct" "$d" "$INTERVAL" "$eta" \
        "$run_R" "$run_P" "$plt_R" "$plt_P" "$data" "$png" | tee -a "$LOG"

    sleep "$INTERVAL"
done
