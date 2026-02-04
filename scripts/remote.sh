#!/bin/bash
# Remote execution helper
# Usage: ./remote.sh <command>
# Requires: REMOTE_HOST and REMOTE_USER env vars (or SSH config)

set -e

REMOTE="${REMOTE_USER:-ubuntu}@${REMOTE_HOST}"
REMOTE_DIR="${REMOTE_DIR:-~/kaggle-comp}"

if [ -z "$REMOTE_HOST" ]; then
    echo "Error: Set REMOTE_HOST environment variable"
    echo "Example: export REMOTE_HOST=123.45.67.89"
    exit 1
fi

case "$1" in
    setup)
        echo "Setting up remote environment..."
        ssh "$REMOTE" "git clone https://github.com/YOUR_USER/kaggle-comp.git $REMOTE_DIR 2>/dev/null || (cd $REMOTE_DIR && git pull)"
        ssh "$REMOTE" "cd $REMOTE_DIR && bash scripts/setup_gpu.sh"
        ;;
    sync)
        echo "Syncing code to remote..."
        rsync -avz --exclude '.venv' --exclude 'data' --exclude 'checkpoints' --exclude '.git' \
            ./ "$REMOTE:$REMOTE_DIR/"
        ;;
    run)
        shift
        echo "Running on remote: $@"
        ssh "$REMOTE" "cd $REMOTE_DIR && source .venv/bin/activate && $@"
        ;;
    logs)
        echo "Fetching experiment logs..."
        ssh "$REMOTE" "cd $REMOTE_DIR && cat experiments/log.jsonl 2>/dev/null || echo 'No experiments yet'"
        ;;
    status)
        echo "Checking remote status..."
        ssh "$REMOTE" "cd $REMOTE_DIR && source .venv/bin/activate && python experiments/status.py"
        ;;
    tail)
        echo "Tailing training output..."
        ssh "$REMOTE" "cd $REMOTE_DIR && tail -f checkpoints/*/history.json 2>/dev/null || echo 'No active training'"
        ;;
    fetch)
        echo "Fetching results..."
        mkdir -p checkpoints experiments
        rsync -avz "$REMOTE:$REMOTE_DIR/checkpoints/" ./checkpoints/
        rsync -avz "$REMOTE:$REMOTE_DIR/experiments/" ./experiments/
        ;;
    gpu)
        echo "Checking GPU status..."
        ssh "$REMOTE" "nvidia-smi"
        ;;
    *)
        echo "Usage: ./remote.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup   - Clone repo and install dependencies"
        echo "  sync    - Sync local code to remote"
        echo "  run     - Run a command on remote (e.g., ./scripts/remote.sh run ./scripts/run_experiment.sh baseline)"
        echo "  logs    - Fetch experiment logs"
        echo "  status  - Show experiment status"
        echo "  tail    - Tail training output"
        echo "  fetch   - Download checkpoints and results"
        echo "  gpu     - Check GPU status"
        ;;
esac
