# Session Notes

Notes for continuity across Claude instances.

---

## 2026-02-04: Initial Setup

### Current State
- VM running at `ubuntu@161.118.191.241` (A100 40GB)
- SSH key: `~/.ssh/lambda-labs.pem`
- Kaggle API token set via env var: `KAGGLE_API_TOKEN`
- Repo cloned to `~/deep-past-challenge` on VM
- Dependencies installed via `uv sync`
- Competition data downloaded and extracted to `data/`

### Baseline Experiment
- **Status**: Running (task ID: b2cf14e)
- **Config**: byt5-small, 10 epochs, batch_size=8, lr=5e-5, seed=42
- **Data**: 1405 train, 156 val samples
- **Progress**: Epoch 1 complete, avg loss 5.49, currently evaluating
- **Note**: Running in background on VM, check with `tail` on output file

### Bug Fix Applied
- Initial run had NaN loss due to FP16 numerical instability with ByT5
- Fixed by switching to BF16 (bfloat16) which has larger dynamic range
- Commit: 624d432

### Data Stats
- train.csv: 1561 rows (transliteration -> translation pairs)
- test.csv: 4 rows (need to predict translations)
- Additional resources: OA_Lexicon_eBL.csv, eBL_Dictionary.csv, publications.csv

### On Resume
```bash
# Check if baseline still running on VM
ssh -i ~/.ssh/lambda-labs.pem ubuntu@161.118.191.241 "ps aux | grep python"

# Check experiment log
ssh -i ~/.ssh/lambda-labs.pem ubuntu@161.118.191.241 "cat ~/deep-past-challenge/experiments/log.jsonl"

# If baseline complete, results will be in log.jsonl
```

### Next Steps (after baseline completes)
1. Review baseline metrics (BLEU, chrF++, GeomMean)
2. Log results to experiments/log.jsonl
3. Consider experiments:
   - Learning rate sweep (1e-4, 3e-5, 1e-5)
   - Longer training (20 epochs)
   - byt5-base model
   - Data augmentation with auxiliary files

### Potential Issues to Watch
- Byte-level tokenization means long sequences - may need to adjust max_length
- Train/test mismatch: training data is document-level, test is sentence-level
- Low-resource setting (~1500 examples) - overfitting risk
