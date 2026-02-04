#!/usr/bin/env python3
"""
Prepare sentence-level training data from document-level pairs.

Splits document-level training pairs into sentence-level pairs to better match
the test distribution (which appears to be sentence-level).

Strategy:
1. Split English translations by sentence boundaries
2. Heuristically split source transliterations to match sentence count
3. Create sentence pairs for training
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def split_translation(text: str) -> list[str]:
    """Split English translation into sentences."""
    if pd.isna(text) or not text.strip():
        return []

    # Split on sentence-ending punctuation followed by space or end
    # Handle abbreviations and special cases
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())

    # Filter empty and clean up
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def split_transliteration(text: str, num_sentences: int) -> list[str]:
    """Heuristically split transliteration into N parts.

    This is an approximation since Akkadian doesn't have clear sentence markers.
    We try to split on likely boundaries (spaces after certain patterns).
    """
    if pd.isna(text) or not text.strip():
        return [""] * num_sentences

    text = text.strip()

    if num_sentences <= 1:
        return [text]

    # Try to split evenly by word count
    words = text.split()
    if len(words) < num_sentences:
        # Not enough words, just return the whole text for first sentence
        return [text] + [""] * (num_sentences - 1)

    words_per_sent = len(words) // num_sentences
    sentences = []

    for i in range(num_sentences):
        start = i * words_per_sent
        if i == num_sentences - 1:
            # Last sentence gets remaining words
            sentences.append(" ".join(words[start:]))
        else:
            sentences.append(" ".join(words[start : start + words_per_sent]))

    return sentences


def process_document(row: pd.Series) -> list[dict]:
    """Process a single document into sentence pairs."""
    oare_id = row["oare_id"]
    source = row["transliteration"]
    target = row["translation"]

    # Split target into sentences
    target_sentences = split_translation(target)

    if not target_sentences:
        # No valid sentences, return original
        return [
            {
                "oare_id": f"{oare_id}_0",
                "transliteration": source if pd.notna(source) else "",
                "translation": target if pd.notna(target) else "",
            }
        ]

    # Split source to match
    source_sentences = split_transliteration(source, len(target_sentences))

    # Create sentence pairs
    pairs = []
    for i, (src, tgt) in enumerate(zip(source_sentences, target_sentences)):
        if tgt.strip():  # Only include if target is non-empty
            pairs.append(
                {
                    "oare_id": f"{oare_id}_{i}",
                    "transliteration": src,
                    "translation": tgt,
                }
            )

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare sentence-level training data")
    parser.add_argument("--input", type=str, default="data/train.csv", help="Input CSV path")
    parser.add_argument("--output", type=str, default="data/train_sentences.csv", help="Output CSV path")
    parser.add_argument("--min-source-words", type=int, default=3, help="Minimum words in source")
    parser.add_argument("--min-target-words", type=int, default=3, help="Minimum words in target")

    args = parser.parse_args()

    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} documents")

    # Process all documents
    all_pairs = []
    for _, row in df.iterrows():
        pairs = process_document(row)
        all_pairs.extend(pairs)

    print(f"Generated {len(all_pairs)} sentence pairs")

    # Filter by minimum length
    filtered_pairs = []
    for pair in all_pairs:
        src_words = len(pair["transliteration"].split()) if pair["transliteration"] else 0
        tgt_words = len(pair["translation"].split()) if pair["translation"] else 0

        if src_words >= args.min_source_words and tgt_words >= args.min_target_words:
            filtered_pairs.append(pair)

    print(f"After filtering (min {args.min_source_words}/{args.min_target_words} words): {len(filtered_pairs)} pairs")

    # Create output dataframe
    output_df = pd.DataFrame(filtered_pairs)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # Stats
    print("\nStatistics:")
    print(f"  Original documents: {len(df)}")
    print(f"  Sentence pairs: {len(filtered_pairs)}")
    print(f"  Expansion ratio: {len(filtered_pairs) / len(df):.2f}x")

    # Sample output
    print("\nSample sentence pairs:")
    for i in range(min(3, len(filtered_pairs))):
        pair = filtered_pairs[i]
        print(f"\n  ID: {pair['oare_id']}")
        print(f"  Source: {pair['transliteration'][:80]}...")
        print(f"  Target: {pair['translation'][:80]}...")


if __name__ == "__main__":
    main()
