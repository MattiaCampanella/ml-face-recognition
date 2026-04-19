#!/usr/bin/env bash
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

FORCE=0
if [ "${1:-}" = "--force" ]; then
	FORCE=1
fi

if [ "$FORCE" -eq 0 ]; then
	echo "=== DRY RUN - add --force to delete files ==="
	RM="echo [DRY] rm -rf"
else
	RM="rm -rf"
fi

echo "Workspace: $PROJ_DIR"
echo ""

echo "[1] experiments/runs/"
if [ -d "experiments/runs" ]; then
	$RM experiments/runs/* 2>/dev/null || true
fi

echo "[2] experiments/logs/*.log"
if [ -d "experiments/logs" ]; then
	$RM experiments/logs/*.log 2>/dev/null || true
fi

echo "[3] cluster state files"
for file in .chain_pid .chain_failed .job_chain .monitor_cache; do
	[ -e "$file" ] && $RM "$file"
done

echo "[4] Python caches"
find . -type d -name "__pycache__" -prune -exec $RM {} + 2>/dev/null || true

echo ""
if [ "$FORCE" -eq 0 ]; then
	echo "Dry run complete. Use: bash cluster/clean.sh --force"
else
	echo "Cleanup completed."
fi
