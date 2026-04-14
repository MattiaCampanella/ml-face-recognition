#!/usr/bin/env bash
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

mkdir -p logs experiments/runs

# Optional sync toggle for uploading local assets to a remote cluster login node.
SYNC_TO_CLUSTER="${SYNC_TO_CLUSTER:-0}"
CLUSTER_HOST="${CLUSTER_HOST:-}"
CLUSTER_PROJECT_DIR="${CLUSTER_PROJECT_DIR:-}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-data/casia-webface}"
LOCAL_WEIGHTS_DIR="${LOCAL_WEIGHTS_DIR:-experiments/runs}"
FORCE_SYNC_DATASET="${FORCE_SYNC_DATASET:-0}"
FORCE_SYNC_WEIGHTS="${FORCE_SYNC_WEIGHTS:-0}"

echo "=== Face Recognition Cluster Setup ==="
echo "Project root: $PROJ_DIR"
if command -v python >/dev/null 2>&1; then
    PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    echo "Python interpreter not found."
    exit 1
fi
echo "Python: $($PYTHON --version 2>&1)"
echo ""

$PYTHON - <<'PY'
import importlib

modules = ["torch", "torchvision", "sklearn", "yaml", "PIL"]
for module_name in modules:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "unknown")
    print(f"{module_name}: {version}")
PY

echo ""
if [ -f "data/casia-webface/train.lst" ]; then
    echo "Dataset root: data/casia-webface (found)"
else
    echo "Dataset root: data/casia-webface (missing)"
fi

if [ -f "data/splits/casia_identity_split_v1.json" ]; then
    echo "Split file: data/splits/casia_identity_split_v1.json (found)"
else
    echo "Split file: data/splits/casia_identity_split_v1.json (missing)"
    echo "Generate it with:"
    echo "  python -m src.datasets.make_split --data-root data/casia-webface --output data/splits/casia_identity_split_v1.json --version v1 --overwrite"
fi

echo ""
echo "Training command:"
echo "  CONFIG=experiments/configs/base.yaml sbatch cluster/train.sh"
echo ""
echo "Evaluation command:"
echo "  CONFIG=experiments/configs/base.yaml sbatch cluster/eval.sh"
echo ""
echo "Combined pipeline:"
echo "  bash cluster/run_all.sh"

remote_has_content() {
    local remote_dir="$1"
    ssh "$CLUSTER_HOST" "if [ -d '$remote_dir' ] && [ \"\$(find '$remote_dir' -mindepth 1 -print -quit 2>/dev/null)\" ]; then exit 0; else exit 1; fi"
}

sync_if_needed() {
    local local_dir="$1"
    local remote_dir="$2"
    local label="$3"
    local force_flag="$4"

    if [ ! -d "$local_dir" ]; then
        echo "[sync] $label: local directory missing, skip ($local_dir)"
        return
    fi

    if [ "$force_flag" -eq 0 ] && remote_has_content "$remote_dir"; then
        echo "[sync] $label: already present on cluster, skip"
        echo "       remote: $remote_dir"
        return
    fi

    echo "[sync] $label: uploading to cluster..."
    rsync -az --info=progress2 "$local_dir/" "$CLUSTER_HOST:$remote_dir/"
    echo "[sync] $label: done"
}

if [ "$SYNC_TO_CLUSTER" -eq 1 ]; then
    echo ""
    echo "=== Optional sync to cluster ==="

    if [ -z "$CLUSTER_HOST" ] || [ -z "$CLUSTER_PROJECT_DIR" ]; then
        echo "SYNC_TO_CLUSTER=1 requires CLUSTER_HOST and CLUSTER_PROJECT_DIR"
        echo "Example:"
        echo "  SYNC_TO_CLUSTER=1 CLUSTER_HOST=user@gcluster.dmi.unict.it CLUSTER_PROJECT_DIR=/home/user/ml-face-recognition bash cluster/setup.sh"
        exit 1
    fi

    REMOTE_DATA_DIR="$CLUSTER_PROJECT_DIR/data/casia-webface"
    REMOTE_WEIGHTS_DIR="$CLUSTER_PROJECT_DIR/experiments/runs"

    ssh "$CLUSTER_HOST" "mkdir -p '$REMOTE_DATA_DIR' '$REMOTE_WEIGHTS_DIR'"

    sync_if_needed "$LOCAL_DATA_DIR" "$REMOTE_DATA_DIR" "dataset" "$FORCE_SYNC_DATASET"
    sync_if_needed "$LOCAL_WEIGHTS_DIR" "$REMOTE_WEIGHTS_DIR" "model weights" "$FORCE_SYNC_WEIGHTS"
fi
