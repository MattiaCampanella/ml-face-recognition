#!/usr/bin/env bash
#SBATCH --job-name=face-train
#SBATCH --partition=default
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-train-%j.log
#SBATCH --error=logs/slurm-train-%j.log

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
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -f "$CONFIG" ]; then
	echo "Config not found: $CONFIG"
	exit 1
fi

echo "============================================"
echo "  Training"
echo "  Job ID:    ${SLURM_JOB_ID:-local}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Config:    ${CONFIG}"
echo "============================================"

$PYTHON -m src.training.train --config "$CONFIG" ${EXTRA_ARGS}

echo "Training completed."
