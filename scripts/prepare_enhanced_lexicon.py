#!/usr/bin/env python3
"""Enhanced lexicon augmentation with higher mix ratios."""

import argparse
import random
from pathlib import Path

import pandas as pd


def load_dictionary(path: str) -> list[dict]:
    """Load eBL Dictionary entries."""
    df = pd.read_csv(path)
    pairs = []

    for _, row in df.iterrows():
        lemma = row.get("lemma", row.get("Lemma", None))
        english = row.get("english", row.get("English", None))

        if pd.isna(lemma) or pd.isna(english):
            continue

        lemma = str(lemma).strip()
        english = str(english).strip()

        # Filter out very short or invalid entries
        if len(lemma) >= 2 and len(english) >= 5:
            pairs.append(
                {
                    "oare_id": f"dict_{hash(lemma) % 1000000:06d}",
                    "transliteration": lemma,
                    "translation": english,
                }
            )

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/train_sentences.csv")
    parser.add_argument("--dict", type=str, default="data/eBL_Dictionary.csv")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dict-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load training data
    train_df = pd.read_csv(args.train)
    print(f"Loaded {len(train_df)} training samples")

    # Load dictionary
    dict_pairs = load_dictionary(args.dict)
    print(f"Loaded {len(dict_pairs)} dictionary entries")

    # Sample dictionary entries
    num_dict_samples = int(len(train_df) * args.dict_ratio)
    dict_sample = random.sample(dict_pairs, min(num_dict_samples, len(dict_pairs)))
    print(f"Sampled {len(dict_sample)} dictionary entries ({args.dict_ratio * 100:.0f}% of training size)")

    # Combine and shuffle
    combined = pd.concat([train_df, pd.DataFrame(dict_sample)], ignore_index=True)
    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} samples to {output_path}")
    print(f"Training: {len(train_df)}, Dictionary: {len(dict_sample)}, Total: {len(combined)}")


if __name__ == "__main__":
    main()
