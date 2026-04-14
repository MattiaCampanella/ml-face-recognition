#!/usr/bin/env bash
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

TARGET=""
FORCE=0

for arg in "$@"; do
	case "$arg" in
		--force) FORCE=1 ;;
		--help|-h)
			echo "Usage: bash cluster/clean_model.sh [RUN_NAME_OR_GLOB] [--force]"
			echo ""
			echo "Examples:"
			echo "  bash cluster/clean_model.sh"
			echo "  bash cluster/clean_model.sh base"
			echo "  bash cluster/clean_model.sh experiments/runs/base-20260412-120000 --force"
			exit 0
			;;
		-*) echo "Unknown argument: $arg"; exit 1 ;;
		*)
			if [ -z "$TARGET" ]; then
				TARGET="$arg"
			else
				echo "Too many positional arguments: $arg"
				exit 1
			fi
			;;
	esac
done

if [ "$FORCE" -eq 0 ]; then
	echo "=== DRY RUN ==="
fi

if [ ! -d experiments/runs ]; then
	echo "No experiment runs found."
	exit 0
fi

if [ -z "$TARGET" ]; then
	echo "Available runs:"
	find experiments/runs -mindepth 1 -maxdepth 1 -type d | sort | sed 's/^/  /'
	echo ""
	echo "Pass a run name or glob to remove it."
	exit 0
fi

if [[ "$TARGET" != *"/"* && "$TARGET" != *"*"* ]]; then
	TARGET="experiments/runs/${TARGET}*"
fi

MATCHES=$(compgen -G "$TARGET" || true)
if [ -z "$MATCHES" ]; then
	echo "No matching runs found for: $TARGET"
	exit 0
fi

echo "$MATCHES" | sed 's/^/  /'

if [ "$FORCE" -eq 0 ]; then
	echo "Run again with --force to delete these directories."
	exit 0
fi

for run_dir in $MATCHES; do
	rm -rf "$run_dir"
done

echo "Removed matching run directories."
