#!/usr/bin/env python3
"""Merge all labeled files into one final dataset"""
import pandas as pd
from pathlib import Path

LABELED_DIR = Path("data/labeled")

# Files to merge
files_to_merge = [
    "vific_labeled_100_research.csv",
    "vific_labeled_100_batch2.csv",
    "vific_labeled_100_batch3.csv",
    "vific_labeled_100_batch4.csv",
]

output_file = LABELED_DIR / "vific_labeled_1000_research.csv"

all_dfs = []
for f in files_to_merge:
    file_path = LABELED_DIR / f
    if file_path.exists():
        df = pd.read_csv(file_path, on_bad_lines='skip')
        all_dfs.append(df)
        print(f"Loaded {f}: {len(df)} samples")

if all_dfs:
    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["id"], keep="last")

    # Add label_numeric if missing
    if "label_numeric" not in merged.columns:
        merged["label_numeric"] = merged["sentiment"].map(
            {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        )

    # Reorder columns
    cols = ["id", "title", "content", "sentiment", "confidence", "method", "raw_response", "timestamp", "label_numeric"]
    merged = merged[[c for c in cols if c in merged.columns]]

    merged.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nTotal merged: {len(merged)} unique samples")

    print("\nSentiment distribution:")
    print(merged["sentiment"].value_counts())
