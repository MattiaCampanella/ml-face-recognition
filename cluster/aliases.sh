#!/usr/bin/env bash
# Sourced by the user's shell — do NOT use set -e here.

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
	tail -f "$PROJ_DIR/experiments/logs/slurm-train-${1}.log"
}

evallog() {
	if [ -z "${1:-}" ]; then
		echo "Usage: evallog <JOB_ID>"
		return 1
	fi
	tail -f "$PROJ_DIR/experiments/logs/slurm-eval-${1}.log"
}

taillast() {
	local logfile
	logfile=$(ls -t "$PROJ_DIR"/experiments/logs/*.log 2>/dev/null | head -1)
	if [ -z "$logfile" ]; then
		echo "No log files found in $PROJ_DIR/experiments/logs/"
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
	mkdir -p "$PROJ_DIR/experiments/logs" && cd "$PROJ_DIR" && CONFIG="$config" sbatch cluster/train.sh "$@"
}

run-eval() {
	local config="${1:-${CONFIG:-experiments/configs/base.yaml}}"
	shift || true
	mkdir -p "$PROJ_DIR/experiments/logs" && cd "$PROJ_DIR" && CONFIG="$config" sbatch cluster/eval.sh "$@"
}

run-all() {
	cd "$PROJ_DIR" && bash cluster/run_all.sh "$@"
}

cleanruns() {
	cd "$PROJ_DIR" && bash cluster/clean.sh --force
}

_monitor_snapshot() {
	# ── find the latest train log ──────────────────────────────────────
	local logfile
	logfile=$(ls -t "$PROJ_DIR"/experiments/logs/slurm-train-*.log 2>/dev/null | head -1)
	if [ -z "$logfile" ]; then
		echo "No train log files found in $PROJ_DIR/experiments/logs/"
		return 1
	fi

	# ── extract job id from filename ───────────────────────────────────
	local job_id
	job_id=$(basename "$logfile" | sed 's/slurm-train-\(.*\)\.log/\1/')

	# ── extract config path from the log header ────────────────────────
	local config_path
	config_path=$(grep -m1 'Config:' "$logfile" | sed 's/.*Config:[[:space:]]*//' | tr -d '\r' | sed 's/[[:space:]]*$//')

	# ── extract total epochs from config YAML ──────────────────────────
	local total_epochs=0
	if [ -n "$config_path" ] && [ -f "$PROJ_DIR/$config_path" ]; then
		total_epochs=$(grep -E '^[[:space:]]+epochs:' "$PROJ_DIR/$config_path" | head -1 | sed 's/.*epochs:[[:space:]]*//' | tr -d '\r' | sed 's/[[:space:]]*$//')
	fi
	if [ "$total_epochs" -le 0 ] 2>/dev/null; then
		total_epochs="?"
	fi

	# ── extract header info ────────────────────────────────────────────
	local node date_str
	node=$(grep -m1 'Node:' "$logfile" | sed 's/.*Node:[[:space:]]*//')
	date_str=$(grep -m1 'Date:' "$logfile" | sed 's/.*Date:[[:space:]]*//')

	# ── check if job is still running via squeue ───────────────────────
	local job_state="COMPLETED"
	if command -v squeue &>/dev/null; then
		local sq
		sq=$(squeue --me --noheader -j "$job_id" -o "%T" 2>/dev/null)
		if [ -n "$sq" ]; then
			job_state="$sq"
		else
			# check if training completed successfully
			if grep -q "Training completed." "$logfile" 2>/dev/null; then
				job_state="COMPLETED"
			else
				job_state="FINISHED"
			fi
		fi
	fi

	# ── parse latest epoch summary line ────────────────────────────────
	local last_epoch_line current_epoch
	# handle both "[epoch 001]" and "[triplet epoch 001]" formats
	last_epoch_line=$(grep -E '^\[(triplet )?epoch [0-9]+\]' "$logfile" | tail -1)
	current_epoch=0
	local train_loss_ep="" train_acc_ep=""
	local val_map1="" val_map5="" val_map10=""
	local val_loss="" val_acc=""
	local hard_pos="" hard_neg="" valid_anchors=""
	local best_val=""
	local is_triplet=false

	if [ -n "$last_epoch_line" ]; then
		current_epoch=$(echo "$last_epoch_line" | grep -oP 'epoch \K[0-9]+' | head -1)
		# remove leading zeros
		current_epoch=$((10#$current_epoch))

		# supervised format
		train_loss_ep=$(echo "$last_epoch_line" | grep -oP 'train_loss=\K[0-9.]+')
		train_acc_ep=$(echo "$last_epoch_line" | grep -oP 'train_acc=\K[0-9.]+')
		val_map1=$(echo "$last_epoch_line" | grep -oP 'val_map@1=\K[0-9.]+')
		val_map5=$(echo "$last_epoch_line" | grep -oP 'val_map@5=\K[0-9.]+')
		val_map10=$(echo "$last_epoch_line" | grep -oP 'val_map@10=\K[0-9.]+')
		val_loss=$(echo "$last_epoch_line" | grep -oP 'val_loss=\K[0-9.]+')
		val_acc=$(echo "$last_epoch_line" | grep -oP 'val_acc=\K[0-9.]+')
		best_val=$(echo "$last_epoch_line" | grep -oP 'best=\K[0-9.]+')

		# triplet format
		if echo "$last_epoch_line" | grep -q 'hard_pos='; then
			is_triplet=true
			hard_pos=$(echo "$last_epoch_line" | grep -oP 'hard_pos=\K[0-9.]+')
			hard_neg=$(echo "$last_epoch_line" | grep -oP 'hard_neg=\K[0-9.]+')
			valid_anchors=$(echo "$last_epoch_line" | grep -oP 'valid_anchors=\K[0-9/]+')
			# triplet uses "loss=" not "train_loss="
			if [ -z "$train_loss_ep" ]; then
				train_loss_ep=$(echo "$last_epoch_line" | grep -oP '(?<=\] )loss=\K[0-9.]+')
			fi
		fi
	fi

	# ── parse latest step-level line ───────────────────────────────────
	local last_step_line step_num step_samples step_loss step_acc step_valid
	# handle [train] and [triplet] step log lines
	last_step_line=$(grep -E '^\[(train|triplet)\] step=' "$logfile" | tail -1)
	step_num="" step_samples="" step_loss="" step_acc="" step_valid=""
	if [ -n "$last_step_line" ]; then
		step_num=$(echo "$last_step_line" | grep -oP 'step=\K[0-9]+')
		step_samples=$(echo "$last_step_line" | grep -oP 'samples=\K[0-9]+')
		step_loss=$(echo "$last_step_line" | grep -oP 'loss=\K[0-9.]+')
		step_acc=$(echo "$last_step_line" | grep -oP 'acc=\K[0-9.]+')
		step_valid=$(echo "$last_step_line" | grep -oP 'valid_anchors=\K[0-9/]+')
	fi

	# ── colors ─────────────────────────────────────────────────────────
	local RST='\033[0m'
	local BOLD='\033[1m'
	local DIM='\033[2m'
	local CYAN='\033[36m'
	local GREEN='\033[32m'
	local YELLOW='\033[33m'
	local MAGENTA='\033[35m'
	local WHITE='\033[97m'
	local BLUE='\033[34m'
	local BG_DIM='\033[48;5;236m'

	# ── progress bar ──────────────────────────────────────────────────
	local bar_width=40
	local bar="" pct_text="" filled=0

	if [ "$total_epochs" != "?" ] && [ "$total_epochs" -gt 0 ] 2>/dev/null; then
		filled=$(( current_epoch * bar_width / total_epochs ))
		local remaining=$(( bar_width - filled ))
		local pct=$(( current_epoch * 100 / total_epochs ))
		bar=$(printf '█%.0s' $(seq 1 $filled 2>/dev/null))
		bar="${bar}$(printf '░%.0s' $(seq 1 $remaining 2>/dev/null))"
		pct_text="${current_epoch}/${total_epochs} (${pct}%)"
	else
		bar=$(printf '░%.0s' $(seq 1 $bar_width))
		pct_text="${current_epoch}/?"
	fi

	# ── pick state color ──────────────────────────────────────────────
	local state_color="$GREEN"
	case "$job_state" in
		RUNNING|PENDING) state_color="$YELLOW" ;;
		FAILED|CANCELLED*) state_color='\033[31m' ;;
	esac

	# ── render ─────────────────────────────────────────────────────────
	echo ""
	echo -e "  ${BOLD}${CYAN}TRAINING MONITOR${RST}"
	echo -e "  ${DIM}----------------------------------------------${RST}"
	printf "  ${DIM}Job${RST}  %-10s  ${DIM}Node${RST}  %-10s  ${state_color}%s${RST}\n" "$job_id" "${node:-?}" "$job_state"
	[ -n "$config_path" ] && printf "  ${DIM}Config${RST}  %s\n" "$config_path"
	[ -n "$date_str" ]    && printf "  ${DIM}Started${RST} %s\n" "$date_str"
	echo ""

	# epoch progress
	echo -e "  ${BOLD}${WHITE}Epoch Progress${RST}"
	echo -e "  ${GREEN}${bar}${RST}  ${BOLD}${pct_text}${RST}"
	echo ""

	# current step metrics (if training in progress within an epoch)
	if [ -n "$step_num" ]; then
		echo -e "  ${BOLD}${WHITE}Current Step${RST} ${DIM}(within next epoch)${RST}"
		printf "  ${DIM}Step${RST} %-6s  ${DIM}Samples${RST} %-8s  ${DIM}Loss${RST} ${YELLOW}%s${RST}\n" "$step_num" "$step_samples" "$step_loss"
		[ -n "$step_acc" ]   && printf "  ${DIM}Accuracy${RST}      ${MAGENTA}%s${RST}\n" "$step_acc"
		[ -n "$step_valid" ] && printf "  ${DIM}Valid Anchors${RST} ${MAGENTA}%s${RST}\n" "$step_valid"
		echo ""
	fi

	# last completed epoch metrics
	if [ -n "$last_epoch_line" ]; then
		echo -e "  ${BOLD}${WHITE}Last Completed Epoch ${current_epoch}${RST}"
		echo -e "  ${DIM}----------------------------------------------${RST}"

		[ -n "$train_loss_ep" ] && printf "  ${DIM}Train Loss${RST}          ${YELLOW}%s${RST}\n" "$train_loss_ep"
		[ -n "$train_acc_ep" ]  && printf "  ${DIM}Train Accuracy${RST}      ${MAGENTA}%s${RST}\n" "$train_acc_ep"
		if $is_triplet; then
			[ -n "$hard_pos" ]       && printf "  ${DIM}Hard Positive Dist${RST}  ${BLUE}%s${RST}\n" "$hard_pos"
			[ -n "$hard_neg" ]       && printf "  ${DIM}Hard Negative Dist${RST}  ${BLUE}%s${RST}\n" "$hard_neg"
			[ -n "$valid_anchors" ]  && printf "  ${DIM}Valid Anchors${RST}       ${WHITE}%s${RST}\n" "$valid_anchors"
		fi
		[ -n "$val_loss" ]  && printf "  ${DIM}Val Loss${RST}            ${YELLOW}%s${RST}\n" "$val_loss"
		[ -n "$val_acc" ]   && printf "  ${DIM}Val Accuracy${RST}        ${MAGENTA}%s${RST}\n" "$val_acc"
		[ -n "$val_map1" ]  && printf "  ${DIM}Val MAP@1${RST}           ${GREEN}%s${RST}\n" "$val_map1"
		[ -n "$val_map5" ]  && printf "  ${DIM}Val MAP@5${RST}           ${GREEN}%s${RST}\n" "$val_map5"
		[ -n "$val_map10" ] && printf "  ${DIM}Val MAP@10${RST}          ${GREEN}%s${RST}\n" "$val_map10"
		[ -n "$best_val" ]  && printf "  ${DIM}Best (monitored)${RST}    ${BOLD}${GREEN}%s${RST}\n" "$best_val"
	else
		echo -e "  ${DIM}No completed epochs yet.${RST}"
	fi

	echo -e "  ${DIM}----------------------------------------------${RST}"
	local now
	now=$(date '+%H:%M:%S')
	echo -e "  ${DIM}Updated ${now}  ·  Ctrl+C to exit${RST}"
	echo ""
}

monitor() {
	local interval="${1:-5}"
	trap 'printf "\n"; return 0' INT
	while true; do
		clear
		_monitor_snapshot
		sleep "$interval"
	done
}
