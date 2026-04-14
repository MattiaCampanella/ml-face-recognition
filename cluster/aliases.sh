#!/usr/bin/env bash
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

alias proj='cd "$PROJ_DIR"'
alias myjobs='squeue --me --format="%.10i %.20j %.8T %.10M %.6D %.20R"'
alias killjob='scancel'
alias killalljobs='scancel --me'

jobinfo() {
	if [ -z "${1:-}" ]; then
		echo "Usage: jobinfo <JOB_ID>"
		return 1
	fi
	scontrol show job "$1"
}

latest_run() {
	local run_dir
	run_dir=$(find "$PROJ_DIR/experiments/runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)
	if [ -z "$run_dir" ]; then
		echo "No runs found in experiments/runs/"
		return 1
	fi
	echo "$run_dir"
}

trainlog() {
	if [ -z "${1:-}" ]; then
		echo "Usage: trainlog <JOB_ID>"
		return 1
	fi
	tail -f "$PROJ_DIR/logs/slurm-train-${1}.log"
}

evallog() {
	if [ -z "${1:-}" ]; then
		echo "Usage: evallog <JOB_ID>"
		return 1
	fi
	tail -f "$PROJ_DIR/logs/slurm-eval-${1}.log"
}

lastlog() {
	local logfile
	logfile=$(ls -t "$PROJ_DIR"/logs/*.log 2>/dev/null | head -1)
	if [ -z "$logfile" ]; then
		echo "No log files found in $PROJ_DIR/logs/"
		return 1
	fi
	if [ -n "${1:-}" ]; then
		tail -n "$1" "$logfile"
	else
		tail -f "$logfile"
	fi
}

train() {
	local config="${1:-${CONFIG:-experiments/configs/base.yaml}}"
	shift || true
	cd "$PROJ_DIR" && CONFIG="$config" sbatch cluster/train.sh "$@"
}

run-eval() {
	local config="${1:-${CONFIG:-experiments/configs/base.yaml}}"
	shift || true
	cd "$PROJ_DIR" && CONFIG="$config" sbatch cluster/eval.sh "$@"
}

run-all() {
	cd "$PROJ_DIR" && bash cluster/run_all.sh "$@"
}

cleanruns() {
	cd "$PROJ_DIR" && bash cluster/clean.sh --force
}
