#!/usr/bin/env bash
#SBATCH --job-name=face-train
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matticamp02@gmail.com
#SBATCH --output=experiments/logs/slurm-train-%j.log
#SBATCH --error=experiments/logs/slurm-train-%j.log

set -euo pipefail

# In SLURM, the script is copied to /var/spool before execution, so BASH_SOURCE[0] is wrong.
# SLURM_SUBMIT_DIR is the directory from which sbatch was called.
PROJ_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export WANDB_MODE=offline

mkdir -p experiments/logs

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

echo "Avvio training dentro Apptainer..."
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env TORCH_HOME="$PROJ_DIR/pretrained_weights" \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    /shared/sifs/latest.sif \
    python -m src.training.train --config "$CONFIG" ${EXTRA_ARGS}

echo "Training completed."
