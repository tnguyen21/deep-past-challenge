#!/usr/bin/env python3
"""
Prepare sentence + dictionary hybrid training data.

Combines sentence-level data with high-frequency dictionary entries
to provide both contextual sentence pairs and lexical coverage.
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
    parser = argparse.ArgumentParser(description="Prepare sentence + dictionary hybrid data")
    parser.add_argument("--sentences", type=str, default="data/train_sentences.csv", help="Sentence-level data")
    parser.add_argument("--dict", type=str, default="data/eBL_Dictionary.csv", help="Dictionary file")
    parser.add_argument("--output", type=str, default="data/train_sentence_dict.csv", help="Output file")
    parser.add_argument("--dict-count", type=int, default=600, help="Number of dictionary entries to include")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)

    # Load sentence-level data
    print(f"Loading sentence data from {args.sentences}")
    sentence_df = pd.read_csv(args.sentences)
    print(f"Loaded {len(sentence_df)} sentence pairs")

    # Load dictionary
    print(f"Loading dictionary from {args.dict}")
    dict_pairs = load_dictionary(args.dict)
    print(f"Loaded {len(dict_pairs)} dictionary entries")

    # Sample dictionary entries
    dict_sample = random.sample(dict_pairs, min(args.dict_count, len(dict_pairs)))
    print(f"Sampled {len(dict_sample)} dictionary entries")

    # Combine
    combined = pd.concat([sentence_df, pd.DataFrame(dict_sample)], ignore_index=True)

    # Shuffle
    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} samples to {output_path}")

    # Stats
    print("\nStatistics:")
    print(f"  Sentence pairs: {len(sentence_df)}")
    print(f"  Dictionary entries added: {len(dict_sample)}")
    print(f"  Total: {len(combined)}")
    print(f"  Dictionary ratio: {len(dict_sample) / len(combined) * 100:.1f}%")

    # Sample output
    print("\nSample dictionary pairs added:")
    for pair in dict_sample[:3]:
        print(f"  {pair['transliteration']}: {pair['translation'][:60]}...")


if __name__ == "__main__":
    main()
