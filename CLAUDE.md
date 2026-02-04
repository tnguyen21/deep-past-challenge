# Claude Code Conventions for This Project

## Session Start Protocol

When starting a new session or after context reset:
1. Read `experiments/log.jsonl` to understand what's been tried
2. Summarize last 3-5 experiments (metrics, what worked/didn't)
3. Check current branch state: `git status`, `git branch`
4. Check VM status if connected: `nvidia-smi`, `ps aux | grep python`
5. Then proceed with next experiment

## Git Workflow

- **Work locally, run remotely**: Make code changes and commits here, push to branches, then pull on the VM to run experiments
- **Branch per experiment**: `git checkout -b exp/<experiment-name>`. Don't work directly on main
- **Only commit working code**: Every commit should be runnable. If something is broken, fix it before committing
- **Failed experiments stay on branches**: If a change doesn't improve val score, document the results in the PR/commit message but do NOT merge to main
- **PR before merge**: Open a PR for review before merging any experiment into main. Include metrics comparison vs baseline

## Experiment Discipline

- **Baseline first**: Must establish and document a baseline experiment before any other experimentation. All subsequent experiments compare against baseline metrics
- **One variable at a time**: Change only one variable per experiment unless explicitly bundling related changes. This enables clear attribution of what helped
- **Name experiments clearly**: `baseline`, `byt5_base_lr1e4`, `longer_20ep`, not `test1`, `final_v2`
- **Document negative results**: Failed experiments are valuable - note what was tried and why it didn't work
- **Reproducibility**: Always set and log random seeds. Commit `uv.lock` for dependency pinning

## Experiment Log Schema

All experiments logged to `experiments/log.jsonl` (append-only, one JSON object per line):

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
  "baseline_comparison": "+2.3 geom_mean vs baseline",
  "runtime_mins": 45,
  "notes": "Tried larger batch size, slight improvement",
  "timestamp": "2024-02-04T12:00:00Z"
}
```

## VM/Remote Execution

- **One experiment at a time**: Run sequentially, don't parallelize training runs
- **Don't run destructive commands without confirmation**: No `rm -rf`, `git reset --hard`, etc. without asking
- **Max runtime**: Default 2-hour timeout per experiment. Abort if validation loss hasn't improved for 3 consecutive epochs

## Escalation Protocol

- **Debugging limit**: If debugging an issue exceeds 30 minutes without progress, stop, document the issue clearly, and escalate to user
- **Unknown errors**: If an error is unclear or outside normal training failures, ask before attempting fixes
- **Git conflicts**: Ask before any force operations or conflict resolution

## Timeouts and Guardrails

- **Experiment timeout**: 2 hours max per training run
- **Early stopping**: Abort if validation loss hasn't improved in 3 epochs
- **Debugging timeout**: 30 minutes max, then escalate

## Code Quality

- **Format before commit**: `uvx ruff format . && uvx ruff check .`
- **Test before commit**: Run a quick sanity check (e.g., `--dry-run` or small data subset)
- **No large files in git**: Models, data, checkpoints stay in `.gitignore`
- **Pin dependencies**: Commit `uv.lock` for reproducibility

## Communication

- **Summarize after each iteration**: What was tried, what the metrics were, what to try next
- **Be explicit about uncertainty**: If I'm not sure something will help, say so
- **Compare to baseline**: Always report metrics relative to baseline

## File Structure

```
kaggle-comp/
├── train.py              # Training script
├── byt_ensemble.py       # Inference script
├── status.py             # Experiment status/suggestions
├── run_experiment.sh     # Experiment runner
├── setup_gpu.sh          # VM setup
├── remote.sh             # Remote execution helper
├── uv.lock               # Dependency lockfile (committed)
├── data/                 # Competition data (gitignored)
├── checkpoints/          # Model checkpoints (gitignored)
├── experiments/
│   └── log.jsonl         # Experiment log (append-only)
└── output/               # Inference outputs (gitignored)
```

## Iteration Protocol

1. **Check status**: `python status.py` - review experiments, get suggestions
2. **Plan**: Decide what single variable to change next
3. **Branch**: `git checkout -b exp/<experiment-name>`
4. **Implement**: Make changes, commit locally
5. **Deploy**: Push, pull on VM
6. **Run**: `./run_experiment.sh <name> ...` (one at a time)
7. **Analyze**: Compare metrics to baseline
8. **Document**: If improved → PR to main. If not → document in commit, leave branch unmerged
9. **Repeat**

## Failure Handling

- **Training crash**: Check logs, if fix is obvious apply it, otherwise escalate after 30 min
- **VM disconnect**: Experiment logs persist in `experiments/log.jsonl` - can resume
- **OOM**: Reduce batch size, try gradient accumulation
- **Hung process**: Kill after timeout, log the failure, try with different config
