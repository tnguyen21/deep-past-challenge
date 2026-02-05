# Experiment: Label Smoothing Sweep

Hypothesis: Competitor used label_smoothing=0.2; test 0.1 and 0.3 to find optimal.

## Run 1: 0.1 label smoothing
```
python train.py \
    --experiment-name label_smoothing_01 \
    --model google/byt5-base \
    --epochs 30 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 5e-5 \
    --label-smoothing 0.1
```

## Run 2: 0.3 label smoothing
```
python train.py \
    --experiment-name label_smoothing_03 \
    --model google/byt5-base \
    --epochs 30 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 5e-5 \
    --label-smoothing 0.3
```

Baseline uses no label smoothing (0.0). Competitor used 0.2.

