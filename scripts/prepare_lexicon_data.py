#!/usr/bin/env python3
"""
Prepare lexicon-augmented training data.

Extracts word→definition pairs from eBL_Dictionary.csv and mixes with
original training data to improve vocabulary coverage.
"""

import argparse
import random
from pathlib import Path

import pandas as pd


def load_dictionary(path: str) -> list[dict]:
    """Load and clean dictionary entries."""
    df = pd.read_csv(path)

    pairs = []
    for _, row in df.iterrows():
        word = row["word"]
        definition = row["definition"]

        # Skip if missing data
        if pd.isna(word) or pd.isna(definition):
            continue

        word = str(word).strip()
        definition = str(definition).strip()

        # Skip very short entries
        if len(word) < 2 or len(definition) < 5:
            continue

        # Clean up definition (remove quotes if wrapped)
        if definition.startswith('"') and definition.endswith('"'):
            definition = definition[1:-1]

        pairs.append(
            {
                "oare_id": f"dict_{hash(word) % 1000000:06d}",
                "transliteration": word,
                "translation": definition,
            }
        )

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare lexicon-augmented training data")
    parser.add_argument("--train", type=str, default="data/train.csv", help="Original training data")
    parser.add_argument("--dict", type=str, default="data/eBL_Dictionary.csv", help="Dictionary file")
    parser.add_argument("--output", type=str, default="data/train_lexicon.csv", help="Output file")
    parser.add_argument("--dict-ratio", type=float, default=0.2, help="Ratio of dictionary entries to include (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)

    # Load original training data
    print(f"Loading training data from {args.train}")
    train_df = pd.read_csv(args.train)
    print(f"Loaded {len(train_df)} training samples")

    # Load dictionary
    print(f"Loading dictionary from {args.dict}")
    dict_pairs = load_dictionary(args.dict)
    print(f"Loaded {len(dict_pairs)} dictionary entries")

    # Sample dictionary entries
    num_dict_samples = int(len(train_df) * args.dict_ratio)
    dict_sample = random.sample(dict_pairs, min(num_dict_samples, len(dict_pairs)))
    print(f"Sampled {len(dict_sample)} dictionary entries ({args.dict_ratio * 100:.0f}% of training size)")

    # Combine
    combined = pd.concat([train_df, pd.DataFrame(dict_sample)], ignore_index=True)

    # Shuffle
    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} samples to {output_path}")

    # Stats
    print("\nStatistics:")
    print(f"  Original training: {len(train_df)}")
    print(f"  Dictionary entries added: {len(dict_sample)}")
    print(f"  Total: {len(combined)}")

    # Sample output
    print("\nSample dictionary pairs added:")
    for pair in dict_sample[:3]:
        print(f"  {pair['transliteration']}: {pair['translation'][:60]}...")


if __name__ == "__main__":
    main()
