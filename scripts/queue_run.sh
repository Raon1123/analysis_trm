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
#   GPUS="5 6" scripts/queue_run.sh      # subset of the allowed GPUs
#   scripts/queue_run.sh dry-run         # preview launch (GPUs, FIFO order), run nothing
#   scripts/queue_run.sh status          # show queued / running / done / failed
#   touch scripts/queue/stop             # drain: exit when jobs/ is empty
#
# GPU convention: this box is shared — queue workers may only claim GPUs 4-7.
# GPUS outside that set is refused at startup; FORCE_GPUS=1 bypasses (do not
# use casually: GPUs 0-3 belong to other users' jobs).
#
# Job file = plain bash script. The runner executes it with
# CUDA_VISIBLE_DEVICES pinned to the worker's GPU and logs to logs/queue/.

set -u -o pipefail

QUEUE_DIR="${QUEUE_DIR:-scripts/queue}"
GPUS="${GPUS:-4 5 6 7}"
POLL_S="${POLL_S:-10}"

# GPU convention guard (see header): refuse GPUs outside the allowed set.
ALLOWED_GPUS="4 5 6 7"
if [[ "${FORCE_GPUS:-0}" != "1" ]]; then
    for g in $GPUS; do
        case " $ALLOWED_GPUS " in
            *" $g "*) ;;
            *) echo "ERROR: GPU $g is outside the allowed set {$ALLOWED_GPUS}." >&2
               echo "       GPUs 0-3 belong to other users. FORCE_GPUS=1 to override." >&2
               exit 1 ;;
        esac
    done
fi

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

# Preview what a real launch would do — GPU set (already validated by the
# guard above), startup side effects, and the FIFO claim order. Runs nothing.
if [[ "${1:-}" == "dry-run" ]]; then
    n_workers=$(wc -w <<< "$GPUS")
    echo "== dry run: nothing will be executed =="
    echo "workers: $n_workers (GPUs: $GPUS)   poll: ${POLL_S}s"
    n_stale=$(ls -1 "$PROC_DIR"/*.job.gpu* 2>/dev/null | wc -l)
    (( n_stale > 0 )) && echo "startup would recover $n_stale stale processing/ job(s) into jobs/"
    [[ -e "$STOP_FILE" ]] && echo "startup would remove leftover stop file"
    echo "== FIFO order (first $n_workers claim a GPU immediately) =="
    i=0
    for f in "$JOBS_DIR"/*.job; do
        [[ -e "$f" ]] || continue
        i=$((i + 1))
        printf '%4d. %s\n' "$i" "$(basename "$f" .job)"
    done
    echo "== total queued: $i =="
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
