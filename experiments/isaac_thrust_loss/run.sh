#!/usr/bin/env bash
# Launcher for Git Bash (Windows) or Linux. Usage, from anywhere:
#     ./run.sh <script> [script options]
#     ./run.sh 03_collect_data.py --name run1 --hold 10
#     ./run.sh 01_inspect_env.py --headless
#
# It moves to your IsaacLab folder and runs the script with Isaac Lab's own Python (via `uv`).

# ======================= EDIT THIS =======================
# Folder where YOU installed Isaac Lab (the folder that contains pyproject.toml).
# Git Bash writes D:\isaac\IsaacLab as /d/isaac/IsaacLab. You can also set the ISAACLAB_DIR environment variable instead.
ISAACLAB_DIR="${ISAACLAB_DIR:-/d/isaac/IsaacLab}"
# =========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: ./run.sh <script> [options]   e.g. ./run.sh 03_collect_data.py --name run1"
    exit 1
fi
if [ ! -f "$ISAACLAB_DIR/pyproject.toml" ]; then
    echo "ERROR: Isaac Lab not found at: $ISAACLAB_DIR"
    echo "Edit ISAACLAB_DIR at the top of run.sh (or set the ISAACLAB_DIR environment variable)."
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' is not installed on your machine. Install it from https://docs.astral.sh/uv/ or run pip install uv."
    exit 1
fi

script="$1"
shift
cd "$ISAACLAB_DIR" || exit 1
uv run --extra isaacsim python "$SCRIPT_DIR/$script" "$@"
