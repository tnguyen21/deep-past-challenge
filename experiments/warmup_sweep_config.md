# Experiment: Warmup Ratio Sweep

Hypothesis: 10% warmup may not be optimal; try 5% and 20%.

## Run 1: 5% warmup
```
python train.py \
    --experiment-name warmup_5pct \
    --model google/byt5-base \
    --epochs 30 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 5e-5 \
    --warmup-ratio 0.05
```

## Run 2: 20% warmup
```
python train.py \
    --experiment-name warmup_20pct \
    --model google/byt5-base \
    --epochs 30 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 5e-5 \
    --warmup-ratio 0.2
```

Compare to baseline which uses warmup_ratio=0.1

