#!/usr/bin/env python3
"""
ByT5 Fine-tuning Script for Akkadian-to-English Translation
"""

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Suppress tqdm progress bars from HuggingFace downloads
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import pandas as pd
import torch

# Evaluation metrics
from evaluate import load as load_metric
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Performance optimizations
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.set_float32_matmul_precision("medium")  # Enable TF32 for matmuls


@dataclass
class TrainConfig:
    # Paths
    train_data: str = "data/train.csv"
    val_data: Optional[str] = None
    model_name: str = "google/byt5-small"
    output_dir: str = "checkpoints"
    experiment_name: str = ""

    # Training
    epochs: int = 30  # 30 epochs for better convergence (loss still decreasing at 10)
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    optimizer: str = "adamw"  # "adamw" or "adafactor"
    weight_decay: float = 0.01
    label_smoothing: float = 0.0  # Label smoothing factor (0 = disabled)
    max_grad_norm: float = 1.0

    # Data
    max_source_length: int = 512
    max_target_length: int = 512
    val_split: float = 0.1

    # Generation (for validation)
    num_beams: int = 1  # Use greedy decoding for intermediate evals (fast iteration)
    final_num_beams: int = 4  # Use beam search only on final epoch for accurate metrics
    max_new_tokens: int = 256  # Max tokens for validation generation (256 is faster, 512 for final eval)
    use_adaptive_beams: bool = True  # Use fewer beams for short sequences

    # Misc
    seed: int = 42
    save_every: int = 1
    eval_every: int = 0  # Evaluate every N epochs (0 = only final epoch)
    num_workers: int = 4  # Parallel data loading
    use_amp: bool = True  # Use automatic mixed precision (BF16 on CUDA, disabled elsewhere)
    use_better_transformer: bool = False  # Use optimum BetterTransformer for faster inference

    # Device
    device: torch.device = field(init=False)
    device_type: str = field(init=False)
    amp_dtype: torch.dtype = field(init=False)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_type = "cuda"
            # Use BF16 for better numerical stability with transformers
            self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_type = "mps"
            self.use_amp = False
            self.amp_dtype = torch.float32
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"
            self.use_amp = False
            self.amp_dtype = torch.float32

        if not self.experiment_name:
            self.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class AkkadianDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_source_length: int, max_target_length: int):
        # Preprocess sources
        sources = [self._preprocess_source(t) for t in df["transliteration"].tolist()]
        targets = df["translation"].tolist()

        # Pre-tokenize everything once (much faster than tokenizing per __getitem__)
        logger.info(f"Pre-tokenizing {len(sources)} samples...")
        source_enc = tokenizer(
            sources,
            max_length=max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = tokenizer(
            targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        self.input_ids = source_enc.input_ids
        self.attention_mask = source_enc.attention_mask
        self.labels = target_enc.input_ids.clone()
        self.labels[self.labels == tokenizer.pad_token_id] = -100

    def _preprocess_source(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        # Normalize gaps
        text = re.sub(r"(\.{3,}|…+|……)", "<big_gap>", text)
        text = re.sub(r"(xx+|\s+x\s+)", "<gap>", text)
        return "translate Akkadian to English: " + text

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def label_smoothed_nll_loss(logits: torch.Tensor, labels: torch.Tensor, smoothing: float = 0.0, ignore_index: int = -100) -> torch.Tensor:
    """Compute cross-entropy loss with label smoothing."""
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    mask = labels_flat != ignore_index

    if smoothing > 0:
        log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_probs)
            smooth_labels.fill_(smoothing / (vocab_size - 1))
            valid_labels = labels_flat.clone()
            valid_labels[~mask] = 0
            smooth_labels.scatter_(1, valid_labels.unsqueeze(1), 1.0 - smoothing)
        loss = (-smooth_labels * log_probs).sum(dim=-1)
        loss = (loss * mask.float()).sum() / mask.float().sum()
    else:
        loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index)

    return loss


def get_adaptive_beam_size(attention_mask: torch.Tensor, base_beams: int, use_adaptive: bool = True) -> int:
    """Use fewer beams for shorter sequences to speed up generation."""
    if not use_adaptive or base_beams <= 2:
        return base_beams

    # Compute max length in batch
    lengths = attention_mask.sum(dim=1)
    max_len = int(lengths.max().item())

    # Short sequences (<100 tokens): use fewer beams
    if max_len < 100:
        return max(2, base_beams // 2)
    return base_beams


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute competition metric: geometric mean of BLEU and chrF++."""
    bleu = load_metric("sacrebleu")
    chrf = load_metric("chrf")

    # BLEU expects list of references for each prediction
    refs_for_bleu = [[r] for r in references]

    bleu_result = bleu.compute(predictions=predictions, references=refs_for_bleu)
    chrf_result = chrf.compute(predictions=predictions, references=references, word_order=2)  # chrF++

    bleu_score = bleu_result["score"]
    chrf_score = chrf_result["score"]

    # Geometric mean
    if bleu_score > 0 and chrf_score > 0:
        geom_mean = (bleu_score * chrf_score) ** 0.5
    else:
        geom_mean = 0.0

    return {
        "bleu": bleu_score,
        "chrf++": chrf_score,
        "geom_mean": geom_mean,
    }


def evaluate(model, tokenizer, val_loader, config: TrainConfig, num_beams: int | None = None) -> dict:
    """Run evaluation and return metrics.

    Args:
        num_beams: Override beam count. If None, uses config.num_beams.
    """
    model.eval()
    predictions = []
    references = []

    base_beams = num_beams if num_beams is not None else config.num_beams

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)

            # Use adaptive beam sizing for speed
            num_beams = get_adaptive_beam_size(attention_mask, base_beams, config.use_adaptive_beams)

            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": config.max_new_tokens,
                "num_beams": num_beams,
                "use_cache": True,
            }
            # early_stopping only valid for beam search
            if num_beams > 1:
                generate_kwargs["early_stopping"] = True

            outputs = model.generate(**generate_kwargs)

            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(preds)

            # Decode references (need to handle -100)
            labels = batch["labels"].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            references.extend(refs)

    metrics = compute_metrics(predictions, references)
    model.train()
    return metrics


def train(config: TrainConfig):
    """Main training loop."""
    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Device: {config.device} ({config.device_type})")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"AMP: {config.use_amp}, dtype: {config.amp_dtype}")

    # Set seed
    torch.manual_seed(config.seed)

    # Load data
    logger.info(f"Loading data from {config.train_data}")
    df = pd.read_csv(config.train_data)
    logger.info(f"Loaded {len(df)} samples")

    # Train/val split
    if config.val_data:
        train_df = df
        val_df = pd.read_csv(config.val_data)
    else:
        val_size = int(len(df) * config.val_split)
        val_df = df.sample(n=val_size, random_state=config.seed)
        train_df = df.drop(val_df.index)

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Load tokenizer and model
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    model = model.to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Apply BetterTransformer for faster inference (optional)
    if config.use_better_transformer and config.device_type == "cuda":
        try:
            from optimum.bettertransformer import BetterTransformer

            logger.info("Applying BetterTransformer...")
            model = BetterTransformer.transform(model)
            logger.info("BetterTransformer applied (20-50% inference speedup)")
        except ImportError:
            logger.warning("'optimum' not installed, skipping BetterTransformer. Install with: pip install optimum")
        except Exception as exc:
            logger.warning(f"BetterTransformer failed: {exc}")

    # Compile model for faster training (PyTorch 2.0+)
    if hasattr(torch, "compile") and config.device_type == "cuda":
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Create datasets
    train_dataset = AkkadianDataset(train_df, tokenizer, config.max_source_length, config.max_target_length)
    val_dataset = AkkadianDataset(val_df, tokenizer, config.max_source_length, config.max_target_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device_type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device_type == "cuda",
        persistent_workers=config.num_workers > 0,
    )

    # Optimizer and scheduler
    if config.optimizer == "adafactor":
        # Adafactor with relative step size (no external LR scheduler needed)
        optimizer = Adafactor(
            model.parameters(),
            lr=config.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
        logger.info(f"Using Adafactor optimizer (lr={config.learning_rate})")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        logger.info(f"Using AdamW optimizer (lr={config.learning_rate}, wd={config.weight_decay})")

    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Mixed precision (BF16 doesn't need GradScaler)
    use_scaler = config.use_amp and config.amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    # Training history
    history = {
        "config": {
            "model_name": config.model_name,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
        },
        "train_loss": [],
        "val_metrics": [],
    }

    best_geom_mean = 0.0
    global_step = 0

    # Early stopping tracking
    best_train_loss = float("inf")
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)

            with torch.amp.autocast(device_type=config.device_type, dtype=config.amp_dtype, enabled=config.use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                if config.label_smoothing > 0:
                    loss = label_smoothed_nll_loss(outputs.logits, labels, config.label_smoothing)
                else:
                    loss = outputs.loss
                loss = loss / config.gradient_accumulation_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            epoch_loss += loss.item() * config.gradient_accumulation_steps

            if (step + 1) % config.gradient_accumulation_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}")

        # Early stopping check (based on training loss)
        if config.patience > 0:
            if best_train_loss - avg_loss > config.min_delta:
                best_train_loss = avg_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement (delta={best_train_loss - avg_loss:.4f} < {config.min_delta}). Patience: {epochs_without_improvement}/{config.patience}")
                if epochs_without_improvement >= config.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    # Force final evaluation before stopping
                    eval_beams = config.final_num_beams
                    logger.info(f"Running final evaluation with beam search (num_beams={eval_beams})...")
                    metrics = evaluate(model, tokenizer, val_loader, config, num_beams=eval_beams)
                    history["val_metrics"].append({"epoch": epoch + 1, "num_beams": eval_beams, **metrics})
                    logger.info(f"Final - Val BLEU: {metrics['bleu']:.2f}, chrF++: {metrics['chrf++']:.2f}, GeomMean: {metrics['geom_mean']:.2f}")
                    if metrics["geom_mean"] > best_geom_mean:
                        best_geom_mean = metrics["geom_mean"]
                        best_path = Path(config.output_dir) / config.experiment_name / "best"
                        best_path.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        logger.info(f"New best model saved! GeomMean: {best_geom_mean:.2f}")
                    break

        # Evaluate every N epochs or on final epoch
        is_final_epoch = epoch + 1 == config.epochs
        should_eval = is_final_epoch or (config.eval_every > 0 and (epoch + 1) % config.eval_every == 0)

        if should_eval:
            # Use beam search on final epoch for accurate metrics, greedy otherwise for speed
            eval_beams = config.final_num_beams if is_final_epoch else config.num_beams
            decode_method = "beam search" if eval_beams > 1 else "greedy"
            logger.info(f"Evaluating with {decode_method} (num_beams={eval_beams})...")

            metrics = evaluate(model, tokenizer, val_loader, config, num_beams=eval_beams)
            history["val_metrics"].append({"epoch": epoch + 1, "num_beams": eval_beams, **metrics})
            logger.info(
                f"Epoch {epoch + 1} - Val BLEU: {metrics['bleu']:.2f}, chrF++: {metrics['chrf++']:.2f}, GeomMean: {metrics['geom_mean']:.2f}"
            )

            # Save best model
            if metrics["geom_mean"] > best_geom_mean:
                best_geom_mean = metrics["geom_mean"]
                best_path = Path(config.output_dir) / config.experiment_name / "best"
                best_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                logger.info(f"New best model saved! GeomMean: {best_geom_mean:.2f}")

    # Save history
    history_path = Path(config.output_dir) / config.experiment_name / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete! Best GeomMean: {best_geom_mean:.2f}")
    logger.info(f"Results saved to {config.output_dir}/{config.experiment_name}")

    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train ByT5 for Akkadian translation")

    parser.add_argument("--train-data", type=str, default="data/train.csv")
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--model", type=str, default="google/byt5-small")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--experiment-name", type=str, default="")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adafactor"], help="Optimizer type")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0=disabled). Stop if loss doesn't improve by min-delta for this many epochs")
    parser.add_argument("--min-delta", type=float, default=0.01, help="Minimum loss improvement to reset patience")

    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument("--val-split", type=float, default=0.1)

    parser.add_argument("--num-beams", type=int, default=1, help="Beam count for intermediate evals (1=greedy)")
    parser.add_argument("--final-num-beams", type=int, default=4, help="Beam count for final epoch eval")
    parser.add_argument("--eval-every", type=int, default=0, help="Evaluate every N epochs (0 = only final)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")

    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainConfig(
        train_data=args.train_data,
        val_data=args.val_data,
        model_name=args.model,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        label_smoothing=args.label_smoothing,
        optimizer=args.optimizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        val_split=args.val_split,
        num_beams=args.num_beams,
        final_num_beams=args.final_num_beams,
        eval_every=args.eval_every,
        seed=args.seed,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )

    train(config)


if __name__ == "__main__":
    main()
