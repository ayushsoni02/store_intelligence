#!/bin/bash
set -e

echo "========================================"
echo "Store Intelligence — Detection Pipeline"
echo "========================================"

# Activate virtualenv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Default: process all cameras, no force
FORCE=""
MAX_FRAMES=""

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --force)      FORCE="--force" ;;
        --max-frames) MAX_FRAMES="--max-frames $2"; shift ;;
        --cameras)    shift; CAMERAS="--cameras $@"; break ;;
    esac
    shift
done

echo "Starting pipeline at $(date)"
python -m pipeline.run_all $FORCE $MAX_FRAMES ${CAMERAS:-}
echo "Pipeline complete at $(date)"
echo "Events written to data/processed/all_events.jsonl"
