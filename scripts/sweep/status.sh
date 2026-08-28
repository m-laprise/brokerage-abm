#!/bin/bash
# One-shot status snapshot for the sweep (used by the agent's heartbeat checks).
#   bash status.sh <sweep_dir>
set -u
SWEEP_DIR="${1:?usage: status.sh <sweep_dir>}"
NCONDITIONS=0; NGRIDCELLS=0; NRUNS=0; NPLOT=0; COMPUTE_JOBID=""; PLOT_JOBID=""
# shellcheck disable=SC1090
[ -f "$SWEEP_DIR/counts.env" ] && source "$SWEEP_DIR/counts.env"
# shellcheck disable=SC1090
[ -f "$SWEEP_DIR/sweep.env" ] && source "$SWEEP_DIR/sweep.env"

echo "time: $(date '+%F %T')"
shards=$(find "$SWEEP_DIR/oat" "$SWEEP_DIR/phase" -name 'seed_*.jld2' 2>/dev/null | wc -l)
data=$(find "$SWEEP_DIR" -name 'data.jld2' 2>/dev/null | wc -l)
png=$(find "$SWEEP_DIR" -name '*.png' 2>/dev/null | wc -l)
echo "design: ${NCONDITIONS} effective realizations, ${NGRIDCELLS} grid coordinates"
echo "progress: ${shards}/${NRUNS} realization-seed shards | ${data} data.jld2 | ${png} png"

echo "queue (this user, name/state):"
squeue -u "$USER" -h -r -o "%j|%T" 2>/dev/null | sort | uniq -c | sed 's/^/  /'

# longest-pending compute task wait (mins since submit) + reason
pend=$(squeue -u "$USER" -h -r -n brokerage_abm_run -t PENDING -o "%V|%r" 2>/dev/null | head -1)
[ -n "$pend" ] && echo "oldest pending brokerage_abm_run: submit=${pend%%|*} reason=${pend##*|}"

if [ -n "${COMPUTE_JOBID:-}" ]; then
    echo "compute job $COMPUTE_JOBID terminal states (sacct):"
    sacct -j "$COMPUTE_JOBID" -n -X -o State%-14 2>/dev/null | awk '{print $1}' | sort | uniq -c | sed 's/^/  /'
    bad=$(sacct -j "$COMPUTE_JOBID" -n -X -o JobID,State 2>/dev/null | grep -E 'FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|CANCELLED' | head -8)
    [ -n "$bad" ] && { echo "NON-SUCCESS TASKS:"; echo "$bad" | sed 's/^/  /'; }
fi

# scan stderr of THIS run's tasks only (avoid stale logs from cancelled jobs)
scan_glob="$SWEEP_DIR/logs/${COMPUTE_JOBID:-NONE}_*.err"
errs=$(grep -liE 'ERROR:|Stacktrace|signal \(|Out of memory|Killed' $scan_glob 2>/dev/null | head -6)
[ -n "$errs" ] && { echo "STDERR with error signatures (current run):"; echo "$errs" | sed 's/^/  /'; }
if [ -n "${COMPUTE_JOBID:-}" ]; then
    g="$SWEEP_DIR/logs/${COMPUTE_JOBID}_*.out"
    echo "task detail (current run):"
    echo "  real sim completions (done in): $(grep -l 'done in' $g 2>/dev/null | wc -l)"
    echo "  idempotent skips:               $(grep -l '^SKIP'  $g 2>/dev/null | wc -l)"
    maxsec=$(squeue -u "$USER" -h -r -n brokerage_abm_run -t RUNNING -o '%M' 2>/dev/null | awk -F: 'NF==2{print $1*60+$2} NF==3{print $1*3600+$2*60+$3}' | sort -n | tail -1)
    echo "  longest-running task:           $(( ${maxsec:-0} / 60 ))m"
fi
if [ -n "${PLOT_JOBID:-}" ]; then
    echo "plot job $PLOT_JOBID states (sacct):"
    sacct -j "$PLOT_JOBID" -n -X -o State 2>/dev/null | awk '{print $1}' | sort | uniq -c | sed 's/^/  /'
    pbad=$(sacct -j "$PLOT_JOBID" -n -X -o JobID,State 2>/dev/null | grep -E 'FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL' | head -8)
    [ -n "$pbad" ] && { echo "PLOT FAILURES:"; echo "$pbad" | sed 's/^/  /'; }
    perr=$(grep -liE 'ERROR:|Stacktrace|MethodError|BoundsError|UndefVar' "$SWEEP_DIR"/logs/plot_*.err 2>/dev/null | head -6)
    [ -n "$perr" ] && { echo "PLOT STDERR errors:"; echo "$perr" | sed 's/^/  /'; }
    echo "  summary.jld2: $(find "$SWEEP_DIR" -name 'summary.jld2' 2>/dev/null | wc -l)/7 phase grids"
fi
exit 0   # always succeed: a clean scan (no errors) must not look like a failed heartbeat
