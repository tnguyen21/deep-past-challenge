# Session Notes

Notes for continuity across Claude instances.

---

## 2026-02-06: byt5_large_longer - NEW BEST GeomMean 19.00!

### Results
- **GeomMean: 19.00** (BLEU 13.01, chrF++ 27.73) - **NEW BEST!**
- **+24.0% improvement** over previous best (combined_winners 15.32)
- **+24.5% improvement** over byt5_large_long (15.26)
- Early stopped at epoch 97/100 (patience=10, min_delta=0.005)
- Final training loss: 2.46 (down from 5.69)
- Runtime: ~17 hours

### Training Trajectory
- Loss: 5.69 → 3.57 (ep5) → 2.94 (ep10) → 2.68 (ep25) → 2.54 (ep50) → 2.48 (ep75) → 2.46 (ep97)
- Greedy evals: ep25=14.18, ep50=13.58, ep75=14.52
- Final beam search eval (num_beams=4): **19.00** (much higher than greedy!)

### Key Insight
Applying base-model winning hyperparams to byt5-large was hugely effective:
- label_smoothing 0.3 (vs 0.2): better regularization
- weight_decay 0.01 (fixed): was silently ignored before
- eff_batch 32 (vs 128): more optimizer steps = better for this dataset size
- 97 epochs (vs 50): model kept improving with more training

### Results Summary

| Experiment | GeomMean | Notes |
|------------|----------|-------|
| **byt5_large_longer** | **19.00** | **NEW BEST** - large model + winning hyperparams |
| combined_winners (byt5-base) | 15.32 | Previous best |
| byt5_large_long | 15.26 | 50ep, eff_batch=128, label_smooth=0.2 |
| adafactor (byt5-base) | 12.67 | |
| sentence_level_data | 12.12 | |
| baseline (byt5-small) | 1.57 | |

---

## 2026-02-07: Experiment Suite on byt5-base (12 experiments planned)

### Goal
Run systematic experiments on byt5-base to find additional improvements before scaling to larger models.

### Current Status
- **Baseline**: combined_winners (byt5-base) GeomMean 15.32
- **Best overall**: byt5_large_longer GeomMean 19.00

### Experiments In Progress
1. **context_768** - ✅ COMPLETED
   - GeomMean: 15.41 (BLEU 10.08, chrF++ 23.56)
   - **vs baseline (15.32): +0.09 (+0.6%)**
   - Small positive improvement with longer context
   - Runtime: 2.7 hours

2. **context_1024** - RUNNING (started 12:45 UTC, PID 72519)
   - Branch: exp/context-length-sweep
   - Testing max_source/target_length=1024 (vs 512)
   - Expected: ~3-3.5 hours (longer sequences = slower)

### Code Changes Complete ✓
All experiment branches created and pushed:
1. ✓ exp/context-length-sweep (in progress)
2. ✓ exp/dropout-tuning (code ready)
3. ✓ exp/gap-augmentation (code ready)
4. ✓ exp/enhanced-lexicon (script ready)
5. ✓ exp/discriminative-lr (code ready)
6. ✓ exp/reverse-translation (code ready)

### Planned Experiments (Priority Order)
1. ✓ Context length sweep: 768, 1024 (Priority 1) - IN PROGRESS
2. ✓ Dropout tuning: 0.1, 0.2, 0.3 (Priority 2) - CODE READY
3. ✓ Gap augmentation: random swap <gap>/<big_gap> (Priority 3) - CODE READY
4. ✓ Enhanced lexicon: 30%, 50% mix ratios (Priority 4) - CODE READY
5. ✓ Discriminative LR: encoder/decoder different rates (Priority 5) - CODE READY
6. ✓ Reverse translation: 20% en→akk examples (Priority 6) - CODE READY

### Next Steps After context_768 Completes
1. Log results to experiments/log.jsonl ON THIS BRANCH
2. Start context_1024 experiment
3. After both complete, create PR with results
4. Move to Priority 2 (dropout tuning)

---

## 2026-02-04: Initial Setup

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
