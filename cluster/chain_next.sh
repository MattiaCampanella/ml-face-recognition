#!/usr/bin/env bash
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

if [ $# -lt 1 ]; then
	echo "Usage: bash cluster/chain_next.sh <TRAIN_JOB_ID> [--config PATH] [--checkpoint PATH] [--split NAME] [--output-dir PATH]"
	exit 1
fi

UPSTREAM_JOB_ID="$1"
shift

CONFIG="experiments/configs/base.yaml"
CHECKPOINT=""
SPLIT="test"
OUTPUT_DIR=""

for arg in "$@"; do
	case "$arg" in
		--config=*) CONFIG="${arg#--config=}" ;;
		--checkpoint=*) CHECKPOINT="${arg#--checkpoint=}" ;;
		--split=*) SPLIT="${arg#--split=}" ;;
		--output-dir=*) OUTPUT_DIR="${arg#--output-dir=}" ;;
		*) echo "Unknown argument: $arg"; exit 1 ;;
	esac
done

mkdir -p logs

EVAL_EXPORT="ALL,CONFIG=$CONFIG,SPLIT=$SPLIT"
if [ -n "$CHECKPOINT" ]; then
	EVAL_EXPORT="$EVAL_EXPORT,CHECKPOINT=$CHECKPOINT"
fi
if [ -n "$OUTPUT_DIR" ]; then
	EVAL_EXPORT="$EVAL_EXPORT,OUTPUT_DIR=$OUTPUT_DIR"
fi

sbatch --parsable --dependency="afterok:${UPSTREAM_JOB_ID}" --export="$EVAL_EXPORT" cluster/eval.sh
