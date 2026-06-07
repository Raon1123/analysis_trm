#!/usr/bin/env bash
# File-based GPU job queue runner — see CONCURRENCY.md.
#
#   - One worker per GPU (structural ownership: no shared pool, no locks).
#   - jobs/ is the shared FIFO (lexicographic filename order = queue order).
#   - Claim = atomic rename into processing/ ("rename before process").
#   - Stale processing/ entries are recovered back into jobs/ on startup.
#   - The queue is editable WHILE RUNNING: add / delete / rename *.job files
#     in jobs/ at any time. Workers poll every POLL_S seconds.
#
# Usage:
#   scripts/queue_run.sh                 # workers on GPUs 4 5 6 7
#   GPUS="0 1" scripts/queue_run.sh      # override GPU set
#   scripts/queue_run.sh status          # show queued / running / done / failed
#   touch scripts/queue/stop             # drain: exit when jobs/ is empty
#
# Job file = plain bash script. The runner executes it with
# CUDA_VISIBLE_DEVICES pinned to the worker's GPU and logs to logs/queue/.

set -u -o pipefail

QUEUE_DIR="${QUEUE_DIR:-scripts/queue}"
GPUS="${GPUS:-4 5 6 7}"
POLL_S="${POLL_S:-10}"

JOBS_DIR="$QUEUE_DIR/jobs"
PROC_DIR="$QUEUE_DIR/processing"
DONE_DIR="$QUEUE_DIR/done"
FAIL_DIR="$QUEUE_DIR/failed"
LOG_DIR="logs/queue"
STOP_FILE="$QUEUE_DIR/stop"

mkdir -p "$JOBS_DIR" "$PROC_DIR" "$DONE_DIR" "$FAIL_DIR" "$LOG_DIR"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

if [[ "${1:-}" == "status" ]]; then
    echo "== queued (FIFO order) =="
    ls -1 "$JOBS_DIR" 2>/dev/null || true
    echo "== running (suffix = gpu) =="
    ls -1 "$PROC_DIR" 2>/dev/null || true
    echo "== done: $(ls -1 "$DONE_DIR" 2>/dev/null | wc -l)  failed: $(ls -1 "$FAIL_DIR" 2>/dev/null | wc -l) =="
    ls -1 "$FAIL_DIR" 2>/dev/null || true
    exit 0
fi

# Crash recovery: a previous runner may have died mid-job. Anything left in
# processing/ goes back to the head of the queue (CONCURRENCY.md: "rename
# before process, recover on startup").
for stale in "$PROC_DIR"/*.job.gpu*; do
    [[ -e "$stale" ]] || continue
    orig="$JOBS_DIR/$(basename "${stale%.gpu*}")"
    log "recovering stale job: $(basename "$stale") -> jobs/"
    mv "$stale" "$orig"
done

# A leftover stop file would make a fresh runner drain immediately.
rm -f "$STOP_FILE"

worker_loop() {
    local gpu="$1"
    while true; do
        # Oldest job first: glob expansion is lexicographically sorted, and
        # enqueued filenames carry a zero-padded sequence number.
        local job=""
        local f
        for f in "$JOBS_DIR"/*.job; do
            [[ -e "$f" ]] && { job="$f"; break; }
        done

        if [[ -z "$job" ]]; then
            if [[ -e "$STOP_FILE" ]]; then
                log "gpu$gpu: queue empty and stop file present, worker exiting"
                return 0
            fi
            sleep "$POLL_S"
            continue
        fi

        # Atomic claim: if another worker grabbed it first, mv fails -> retry.
        local name claimed
        name="$(basename "$job")"
        claimed="$PROC_DIR/$name.gpu$gpu"
        mv "$job" "$claimed" 2>/dev/null || continue

        local log_file="$LOG_DIR/${name%.job}.log"
        log "gpu$gpu: start $name (log: $log_file)"
        if CUDA_VISIBLE_DEVICES="$gpu" bash "$claimed" >> "$log_file" 2>&1; then
            mv "$claimed" "$DONE_DIR/$name"
            log "gpu$gpu: done  $name"
        else
            mv "$claimed" "$FAIL_DIR/$name"
            log "gpu$gpu: FAIL  $name (see $log_file)"
        fi
    done
}

# Ctrl-C / TERM: kill the whole process group (workers AND their training
# processes), not just the worker shells.
trap 'trap - INT TERM; log "stopping: killing all running jobs"; kill 0 2>/dev/null' INT TERM

n=0
for gpu in $GPUS; do
    worker_loop "$gpu" &
    n=$((n + 1))
done

log "started $n workers on GPUs: $GPUS"
log "queue dir:   $JOBS_DIR (drop/edit/delete *.job files anytime)"
log "watch:       $0 status"
log "drain+exit:  touch $STOP_FILE"

wait
log "all workers exited"
