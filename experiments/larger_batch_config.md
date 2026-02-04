# Experiment: Larger Effective Batch Size

Hypothesis: Competitor used effective batch 160 vs our 32. Larger batches with higher LR often work better.

Run with:
```
python train.py \
    --experiment-name larger_batch \
    --model google/byt5-base \
    --epochs 30 \
    --batch-size 4 \
    --grad-accum 32 \
    --lr 1e-4
```

Effective batch size: 4 * 32 = 128 (vs baseline 32)

