#!/usr/bin/env bash
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

CONFIG="experiments/configs/triplet_hardmining.yaml"
CHECKPOINT=""
SPLIT="test"
OUTPUT_DIR=""
TRAIN_ONLY=0
EVAL_ONLY=0

usage() {
	echo "Usage: bash cluster/run_all.sh [--config PATH] [--checkpoint PATH] [--split NAME] [--output-dir PATH] [--train-only] [--eval-only]"
}

for arg in "$@"; do
	case "$arg" in
		--config=*) CONFIG="${arg#--config=}" ;;
		--checkpoint=*) CHECKPOINT="${arg#--checkpoint=}" ;;
		--split=*) SPLIT="${arg#--split=}" ;;
		--output-dir=*) OUTPUT_DIR="${arg#--output-dir=}" ;;
		--train-only) TRAIN_ONLY=1 ;;
		--eval-only) EVAL_ONLY=1 ;;
		--help|-h) usage; exit 0 ;;
		*) echo "Unknown argument: $arg"; usage; exit 1 ;;
	esac
done

if [ "$TRAIN_ONLY" -eq 1 ] && [ "$EVAL_ONLY" -eq 1 ]; then
	echo "--train-only and --eval-only are mutually exclusive."
	exit 1
fi

mkdir -p logs

if [ "$EVAL_ONLY" -eq 0 ]; then
	TRAIN_JOB_ID=$(sbatch --parsable --export=ALL,CONFIG="$CONFIG" cluster/train.sh)
	echo "Submitted training job: $TRAIN_JOB_ID"

	if [ "$TRAIN_ONLY" -eq 1 ]; then
		exit 0
	fi

	DEPENDENCY="afterok:${TRAIN_JOB_ID}"
else
	DEPENDENCY=""
fi

EVAL_EXPORT="ALL,CONFIG=$CONFIG,SPLIT=$SPLIT"
if [ -n "$CHECKPOINT" ]; then
	EVAL_EXPORT="$EVAL_EXPORT,CHECKPOINT=$CHECKPOINT"
fi
if [ -n "$OUTPUT_DIR" ]; then
	EVAL_EXPORT="$EVAL_EXPORT,OUTPUT_DIR=$OUTPUT_DIR"
fi

if [ -n "$DEPENDENCY" ]; then
	EVAL_JOB_ID=$(sbatch --parsable --dependency="$DEPENDENCY" --export="$EVAL_EXPORT" cluster/eval.sh)
else
	EVAL_JOB_ID=$(sbatch --parsable --export="$EVAL_EXPORT" cluster/eval.sh)
fi

echo "Submitted evaluation job: $EVAL_JOB_ID"
