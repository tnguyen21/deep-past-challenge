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

---

## 2026-02-04: Beam Search Optimizations & Training Config Updates

### Baseline Results
- **GeomMean**: 1.57 (BLEU 0.28, chrF++ 8.64)
- **Target**: ~35.1 (from pre-trained model in byt_ensemble.py)
- **Gap**: ~22x improvement needed
- Logged to `experiments/log.jsonl`

### Changes Made (Branch: exp/perf)

#### Beam Search Optimizations (Priority 0)
1. **use_cache=True** added to model.generate() - enables KV-cache reuse (20-40% speedup)
2. **Adaptive beam sizing** - fewer beams for short sequences (<100 tokens)
3. **BetterTransformer support** - optional, requires `pip install optimum`
4. **max_new_tokens reduced to 256** for validation (was 512)

#### Training Config Updates (Tier 1 Quick Wins)
1. **epochs: 10 → 30** - loss was still decreasing, model undertrained
2. **num_beams: 1 → 4** - match inference-time decoding for better metrics

### Commits
- `460197c` Add beam search optimizations for faster validation
- `3dda8ad` Update training defaults for better convergence

### Next Steps
1. Push branch to remote, pull on VM
2. Run experiment with new config: `--epochs 30 --num-beams 4`
3. If improved, try learning rate sweep: `--lr 1e-4`, `--lr 3e-5`, `--lr 1e-5`
4. Consider byt5-base model next

---

## 2026-02-04: Competitor Model Analysis

Downloaded and analyzed top competitor's model from Kaggle: `llkh0a/byt5-akkadian-model`

### Model Architecture
- **Model**: byt5-large (d_model=1536, d_ff=3968, 18 encoder layers, 6 decoder layers)
- **Size**: 2.01GB
- Much larger than our byt5-base (d_model=1472, 12 layers)

### Key Training Choices Comparison

| Parameter | Competitor | Our Best | Difference |
|-----------|------------|----------|------------|
| **Model** | byt5-large | byt5-base | Larger model |
| **Batch size** | 20 | 4 | 5x larger |
| **Grad accum** | 8 | 8 | Same |
| **Effective batch** | 160 | 32 | 5x larger |
| **Learning rate** | 1e-4 | 5e-5 | 2x higher |
| **Optimizer** | Adafactor | AdamW | Different |
| **Weight decay** | 0.01 | 0.0 | Regularization |
| **Label smoothing** | 0.2 | 0.0 | Regularization |
| **Epochs** | 20 | 30 | Fewer |
| **Precision** | FP32 | BF16 | Higher precision |
| **Best metric** | geo_mean | loss | Metric-based selection |

### Key Insights

1. **Much larger effective batch size (160 vs 32)** - Major difference. Larger batches + higher LR is a known scaling pattern.
2. **Adafactor optimizer** - Memory-efficient, doesn't store momentum like Adam. Good for large models.
3. **Label smoothing (0.2)** - Regularization that helps with overconfident predictions.
4. **load_best_model_at_end with geo_mean metric** - Select checkpoints by actual competition metric, not loss.
5. **Weight decay (0.01)** - Additional regularization.

### Recommendations for Next Experiments

1. **Try byt5-large** if memory allows (may need smaller batch + more grad accum)
2. **Increase effective batch size** - larger grad_accum (e.g., 16 or 32)
3. **Switch to Adafactor** optimizer
4. **Add label smoothing (0.2)**
5. **Add weight decay (0.01)**
6. **Increase learning rate to 1e-4**
7. **Select best checkpoint by geo_mean**, not loss

### Competitor Model Location on VM
```
/home/ubuntu/.cache/kagglehub/datasets/llkh0a/byt5-akkadian-model/versions/1/
```

---

## 2026-02-04: Overnight Experiment Setup

### byt5-large-regularized FAILED
- **GeomMean**: 0.55 (BLEU 0.11, chrF++ 2.80) - WORSE than baseline!
- Loss plateaued at ~5.6 (never decreased from initial)
- Likely cause: Label smoothing + higher LR caused training instability
- **DO NOT MERGE** this branch

### Overnight Experiment Plan (10 experiments, all using byt5-base for speed)

All experiment branches pushed and ready:
1. `exp/sentence-level-data` - Split docs into sentences (RUNNING NOW)
2. `exp/cosine-lr` - Cosine annealing LR schedule
3. `exp/adafactor` - Adafactor optimizer
4. `exp/larger-batch` - Effective batch 128 with higher LR
5. `exp/layer-freeze` - Freeze N encoder layers
6. `exp/muon` - Muon optimizer
7. `exp/lexicon-augment` - Add dictionary entries to training data
8. `exp/early-stopping` - Early stopping with patience
9. `exp/warmup-sweep` - Try warmup_ratio 0.05, 0.2
10. `exp/label-smoothing-sweep` - Try label_smoothing 0.1, 0.3

### Current Running Experiment
- **Name**: sentence_level_data
- **Branch**: exp/sentence-level-data
- **Data**: 6,220 sentence pairs (4x expansion from 1,561 docs)
- **Model**: byt5-base, 30 epochs
- **Progress**: Epoch 4 complete, loss decreasing (5.35 → 3.97 → 3.31 → 2.85)
- **Log**: `train_sentence_level.log` on VM

### Scripts Created
- `scripts/prepare_sentence_data.py` - Split docs into sentence pairs
- `scripts/prepare_lexicon_data.py` - Mix dictionary entries with training data

### Execution Order
After each experiment completes:
1. Log results to `experiments/log.jsonl`
2. Create PR with metrics
3. Start next experiment on VM

### VM Status
- Connected to A100 40GB at `ubuntu@161.118.191.241`
- Sentence-level training running: `train_sentence_level.log`
