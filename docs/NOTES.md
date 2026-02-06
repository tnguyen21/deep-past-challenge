# Session Notes

Notes for continuity across Claude instances.

---

## 2026-02-05: Phase 2 Progress - NEW BEST RESULT!

### byt5_large_long - NEW BEST! ✓
- **GeomMean: 15.26** (BLEU 9.85, chrF++ 23.66)
- **+20.4% improvement** over previous best (12.67)
- Config: byt5-large, 50 epochs, FP32, Adafactor, sentence-level data
- Final loss: 2.13
- Runtime: ~7 hours

### Current Running Experiment
- **Name**: combined_winners
- **Branch**: exp/combined-winners
- **Config**: byt5-base, Adafactor, lr=1e-4, label_smoothing=0.3, sentence data
- **Status**: Running (~2 min/epoch, 30 epochs)
- **Log**: `train_combined_winners.log` on VM

### Code Changes Made
1. Added `--weight-decay` CLI arg and passed to Adafactor
2. Added `--patience` and `--min-delta` for early stopping
3. Created `scripts/prepare_hybrid_data.py` for sentence+dict data

### Experiments Remaining (from plan)
1. ✓ byt5_large_long - DONE (15.26 GeomMean)
2. → combined_winners - RUNNING
3. ◯ sentence_dict_hybrid - Need to run prepare_hybrid_data.py first
4. ◯ larger_batch_adafactor
5. ◯ weight_decay_adafactor
6. ◯ longer_training_base

### Key Findings
- byt5-large with FP32 + Adafactor + 50 epochs = 15.26 GeomMean
- Longer training helps (50 epochs better than 20)
- Early stopping didn't trigger - loss kept improving throughout

---

## Previous Results Summary

| Experiment | GeomMean | Notes |
|------------|----------|-------|
| **byt5_large_long** | **15.26** | **NEW BEST** |
| adafactor (byt5-base) | 12.67 | Previous best |
| sentence_level_data | 12.12 | |
| byt5_large_fp32 | 5.91 | 20 epochs wasn't enough |
| byt5_base_30ep | 5.23 | Baseline for byt5-base |
| baseline (byt5-small) | 1.57 | |

---

## VM Status
- A100 40GB at `ubuntu@161.118.191.241`
- SSH key: `~/.ssh/lambda-labs.pem`
- combined_winners running in background

### On Resume
```bash
# Check current experiment
ssh -i ~/.ssh/lambda-labs.pem ubuntu@161.118.191.241 "tail -20 ~/deep-past-challenge/train_combined_winners.log"

# Check GPU status
ssh -i ~/.ssh/lambda-labs.pem ubuntu@161.118.191.241 "nvidia-smi"

# Check process
ssh -i ~/.ssh/lambda-labs.pem ubuntu@161.118.191.241 "ps aux | grep train.py"
```

---

## Earlier Sessions (Archived)

### 2026-02-04: Initial Setup
- VM setup, baseline experiment, discovered FP16 instability
- Analyzed competitor model (byt5-large, Adafactor, label_smoothing=0.2)

### 2026-02-04: Overnight Experiments
- Ran 10+ experiments on byt5-base
- Best results: Adafactor (12.67), sentence_level_data (12.12)
- Key insight: Optimizer matters most
