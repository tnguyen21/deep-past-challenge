# Experiment Tracking

This directory stores experiment logs and results.

## Files

- `log.jsonl` - Append-only experiment log (one JSON object per line)

## Log Schema

See `CLAUDE.md` for the authoritative schema. Each entry contains:

```json
{
  "name": "experiment_name",
  "branch": "exp/experiment_name",
  "config": {
    "model": "google/byt5-small",
    "epochs": 10,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "seed": 42
  },
  "seed": 42,
  "metrics": {
    "bleu": 12.5,
    "chrf": 35.2,
    "geom_mean": 21.0
  },
  "baseline_comparison": "+2.3 geom_mean vs baseline (18.7)",
  "runtime_mins": 45,
  "notes": "Description of what was tried",
  "timestamp": "2024-02-04T12:00:00Z",
  "model_path": "checkpoints/experiment_name/best"
}
```

## Usage

```bash
# View all experiments
cat experiments/log.jsonl | jq .

# Get best experiment
cat experiments/log.jsonl | jq -s 'max_by(.metrics.geom_mean)'

# List experiments sorted by score
cat experiments/log.jsonl | jq -s 'sort_by(.metrics.geom_mean) | reverse | .[].name'
```
