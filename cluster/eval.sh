#!/usr/bin/env bash
#SBATCH --job-name=face-eval
#SBATCH --partition=default
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-eval-%j.log
#SBATCH --error=logs/slurm-eval-%j.log

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

if command -v python >/dev/null 2>&1; then
	PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
	PYTHON=python3
else
	echo "Python interpreter not found."
	exit 1
fi

mkdir -p logs

CONFIG="${CONFIG:-experiments/configs/triplet_hardmining.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PCA_COMPONENTS="${PCA_COMPONENTS:-2}"
TSNE_PERPLEXITY="${TSNE_PERPLEXITY:-30.0}"
TSNE_ITER="${TSNE_ITER:-1000}"
SEED="${SEED:-42}"

if [ ! -f "$CONFIG" ]; then
	echo "Config not found: $CONFIG"
	exit 1
fi

echo "============================================"
echo "  Evaluation"
echo "  Job ID:    ${SLURM_JOB_ID:-local}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Config:    ${CONFIG}"
echo "  Checkpoint:${CHECKPOINT:-auto}"
echo "  Split:     ${SPLIT}"
echo "============================================"

ARGS=(
	--config "$CONFIG"
	--split "$SPLIT"
	--device "$DEVICE"
	--batch-size "$BATCH_SIZE"
	--num-workers "$NUM_WORKERS"
	--pca-components "$PCA_COMPONENTS"
	--tsne-perplexity "$TSNE_PERPLEXITY"
	--tsne-iter "$TSNE_ITER"
	--seed "$SEED"
)

if [ -n "$CHECKPOINT" ]; then
	ARGS+=(--checkpoint "$CHECKPOINT")
fi

if [ -n "$OUTPUT_DIR" ]; then
	ARGS+=(--output-dir "$OUTPUT_DIR")
fi

$PYTHON -m src.evaluation.evaluate "${ARGS[@]}"

echo "Evaluation completed."
