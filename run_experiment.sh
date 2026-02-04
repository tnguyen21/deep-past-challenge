#!/bin/bash
# Run a training experiment and log results
set -e

EXPERIMENT_NAME="${1:-$(date +%Y%m%d_%H%M%S)}"
MODEL="${2:-google/byt5-small}"
EPOCHS="${3:-10}"
BATCH_SIZE="${4:-8}"
LR="${5:-5e-5}"
SEED="${6:-42}"
NOTES="${7:-}"

# Track start time
START_TIME=$(date +%s)

# Get current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Branch: $BRANCH"
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Seed: $SEED"
echo "=========================================="

# Ensure data exists
if [ ! -f "data/train.csv" ]; then
    echo "Error: data/train.csv not found"
    echo "Run: kaggle competitions download -c deep-past-initiative-machine-translation -p data/ && unzip data/*.zip -d data/"
    exit 1
fi

# Run training
python train.py \
    --train-data data/train.csv \
    --model "$MODEL" \
    --output-dir checkpoints \
    --experiment-name "$EXPERIMENT_NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --grad-accum 4 \
    --num-beams 4

# Calculate runtime
END_TIME=$(date +%s)
RUNTIME_MINS=$(( (END_TIME - START_TIME) / 60 ))

# Log experiment
python -c "
import json
from pathlib import Path
from datetime import datetime

# Load history
history_path = Path('checkpoints/$EXPERIMENT_NAME/history.json')
if history_path.exists():
    with open(history_path) as f:
        history = json.load(f)

    # Get best metrics
    val_metrics = history.get('val_metrics', [{}])
    best_metrics = max(val_metrics, key=lambda x: x.get('geom_mean', 0), default={})

    # Clean metrics (remove epoch key if present)
    metrics = {k: v for k, v in best_metrics.items() if k != 'epoch'}

    # Load baseline for comparison
    log_path = Path('experiments/log.jsonl')
    baseline_comparison = None
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get('name') == 'baseline':
                        baseline_geom = entry.get('metrics', {}).get('geom_mean', 0)
                        current_geom = metrics.get('geom_mean', 0)
                        diff = current_geom - baseline_geom
                        baseline_comparison = f'{diff:+.2f} geom_mean vs baseline ({baseline_geom:.2f})'
                        break

    # Create log entry (matches CLAUDE.md schema)
    entry = {
        'name': '$EXPERIMENT_NAME',
        'branch': '$BRANCH',
        'config': {
            'model': '$MODEL',
            'epochs': $EPOCHS,
            'batch_size': $BATCH_SIZE,
            'learning_rate': $LR,
            'seed': $SEED
        },
        'seed': $SEED,
        'metrics': metrics,
        'baseline_comparison': baseline_comparison,
        'runtime_mins': $RUNTIME_MINS,
        'notes': '$NOTES' if '$NOTES' else None,
        'timestamp': datetime.now().isoformat(),
        'model_path': 'checkpoints/$EXPERIMENT_NAME/best'
    }

    # Remove None values
    entry = {k: v for k, v in entry.items() if v is not None}

    # Append to log
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')

    print(f'Logged experiment: {entry[\"name\"]}')
    print(f'Best GeomMean: {metrics.get(\"geom_mean\", \"N/A\")}')
    if baseline_comparison:
        print(f'Baseline comparison: {baseline_comparison}')
    print(f'Runtime: $RUNTIME_MINS minutes')
"

echo "=========================================="
echo "Experiment complete: $EXPERIMENT_NAME"
echo "Runtime: $RUNTIME_MINS minutes"
echo "=========================================="
