#!/usr/bin/env bash
# One command for the whole ReDAG pipeline.  Safe to run again at any time: finished jobs are skipped, interrupted
# ones resume from their last checkpoint, and a scheduler that is already running is left alone.
#
#   bash scripts/launch/go.sh            start / resume in the background (no-GPU mode: only the CPU data phase runs)
#   bash scripts/launch/go.sh status     what runs where, progress, ETAs, projected finish   (--json for monitors)
#   bash scripts/launch/go.sh logs JOB [-f]
#   bash scripts/launch/go.sh stop       stop now; jobs resume from their checkpoints next time
#   bash scripts/launch/go.sh drain      let running jobs finish, start nothing new (e.g. before a planned shutdown)
#   bash scripts/launch/go.sh retry      start again and also retry jobs that failed with the current code
#   bash scripts/launch/go.sh plan [--gpus 1 2 4 8]    projected timeline for other GPU counts
#   bash scripts/launch/go.sh reload     restart only the scheduler (after git pull); running jobs continue, are adopted
#   bash scripts/launch/go.sh fg         run the scheduler in the foreground
#   bash scripts/launch/go.sh list | reset JOB [--dependents] | bundle export data|results | bundle import MANIFEST
#   bash scripts/launch/go.sh notify-test
# Settings: scripts/launch/env.sh; machine-local overrides (NOTIFY_URL, AUTO_SHUTDOWN=done|idle, ...) in ./local.env.
source "$(dirname "$0")/env.sh"
cd "$REDAG_ROOT" || exit 1
P=("$PY" scripts/launch/pipeline.py)
command=${1:-start}
[ $# -gt 0 ] && shift

start() {
    if "${P[@]}" is-running; then
        echo "scheduler already running"
        "${P[@]}" status --brief
        return 0
    fi
    "${P[@]}" run --check-only "$@" || return 1
    if command -v setsid >/dev/null; then
        setsid nohup "${P[@]}" run "$@" >> "$LOGS/scheduler.out" 2>&1 < /dev/null &
    else
        nohup "${P[@]}" run "$@" >> "$LOGS/scheduler.out" 2>&1 < /dev/null &
        disown
    fi
    sleep 4
    "${P[@]}" status --brief
    echo "scheduler log: tail -f $LOGS/scheduler.out    status: bash scripts/launch/go.sh status"
}

case "$command" in
    start|run) start "$@" ;;
    retry) start --retry-failed "$@" ;;
    reload) "${P[@]}" stop --scheduler-only && start "$@" ;;   # pick up new code / job definitions; jobs keep running
    preflight) MIN_GPUS=1 exec bash scripts/launch/00_check_env.sh ;;   # re-validate by hand (uses a GPU for ~2 min)
    fg) exec "${P[@]}" run "$@" ;;
    drain) exec "${P[@]}" stop --drain ;;
    *) exec "${P[@]}" "$command" "$@" ;;
esac
