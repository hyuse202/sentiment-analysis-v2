#!/usr/bin/env python3
"""
merge_expansion.py
==================
Merge all labeled batches with existing data into final 5000-sample dataset.

Usage:
    python scripts/merge_expansion.py --batches 4
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"


def merge_all(batches: int):
    """Merge all labeled files."""

    print(f"\n{'='*60}")
    print("  MERGING ALL LABELED DATA")
    print(f"{'='*60}\n")

    # Load existing samples
    existing_file = LABELED_DIR / "vific_labeled_1000_research.csv"
    if not existing_file.exists():
        print(f"ERROR: Existing file not found: {existing_file}")
        return None

    all_dfs = [pd.read_csv(existing_file)]
    print(f"Existing: {len(all_dfs[0])} samples")

    # Load batch outputs
    batch_count = 0
    for i in range(1, batches + 1):
        batch_file = LABELED_DIR / f"expansion_batch{i}_labeled.csv"
        if batch_file.exists():
            df = pd.read_csv(batch_file)
            all_dfs.append(df)
            batch_count += 1
            print(f"Batch {i}: {len(df)} samples")
        else:
            print(f"Batch {i}: NOT FOUND - {batch_file.name}")

    if batch_count == 0:
        print("\nERROR: No batch files found. Run labeling first.")
        return None

    # Concatenate and deduplicate
    merged = pd.concat(all_dfs, ignore_index=True)
    original_count = len(merged)
    merged = merged.drop_duplicates(subset=["id"], keep="last")
    duplicates_removed = original_count - len(merged)

    print(f"\nTotal before dedup: {original_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Final count: {len(merged)}")

    # Ensure label_numeric column
    if "label_numeric" not in merged.columns:
        merged["label_numeric"] = merged["sentiment"].map(
            {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        )

    # Reorder columns
    cols = ["id", "title", "content", "sentiment", "confidence",
            "method", "raw_response", "timestamp", "label_numeric"]
    merged = merged[[c for c in cols if c in merged.columns]]

    # Save
    output_file = LABELED_DIR / "vific_labeled_5000_research.csv"
    merged.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("  FINAL DATASET")
    print(f"{'='*60}")
    print(f"  Output: {output_file}")
    print(f"  Total samples: {len(merged)}")

    print(f"\n  Sentiment Distribution:")
    dist = merged["sentiment"].value_counts()
    for label in ["POSITIVE", "NEGATIVE", "NEUTRAL", "ERROR"]:
        if label in dist.index:
            count = dist[label]
            pct = count / len(merged) * 100
            print(f"    {label:<10}: {count:>5} ({pct:>5.1f}%)")

    # Quality check
    errors = (merged["sentiment"] == "ERROR").sum() if "ERROR" in dist.index else 0
    low_conf = (merged["confidence"] < 50).sum()

    print(f"\n  Quality Metrics:")
    print(f"    Errors: {errors} ({errors/len(merged):.1%})")
    print(f"    Low confidence (<50): {low_conf} ({low_conf/len(merged):.1%})")

    # Validation
    print(f"\n  Validation:")
    unique_ids = merged["id"].nunique()
    print(f"    Unique IDs: {unique_ids} (expected: {len(merged)})")
    if unique_ids != len(merged):
        print("    WARNING: Duplicate IDs detected!")

    # Expected distribution check
    print(f"\n  Distribution Analysis:")
    neutral_pct = dist.get("NEUTRAL", 0) / len(merged) * 100
    positive_pct = dist.get("POSITIVE", 0) / len(merged) * 100
    negative_pct = dist.get("NEGATIVE", 0) / len(merged) * 100

    print(f"    Literature expects: NEUTRAL ~43%, POSITIVE ~33%, NEGATIVE ~24%")
    if neutral_pct < 15:
        print(f"    WARNING: NEUTRAL is low ({neutral_pct:.1f}%)")
    elif neutral_pct > 60:
        print(f"    WARNING: NEUTRAL is high ({neutral_pct:.1f}%)")
    else:
        print(f"    NEUTRAL looks reasonable ({neutral_pct:.1f}%)")

    print(f"\n{'='*60}")
    print("  DONE!")
    print(f"{'='*60}\n")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge all labeled batches into final dataset"
    )
    parser.add_argument("--batches", type=int, default=4,
                        help="Number of batch files to merge (default: 4)")
    args = parser.parse_args()

    merge_all(args.batches)


if __name__ == "__main__":
    main()
