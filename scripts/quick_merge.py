#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

LABELED_DIR = Path("data/labeled")

files = [
    LABELED_DIR / "vific_labeled_100_research.csv",
    LABELED_DIR / "vific_labeled_100_batch2.csv",
    LABELED_DIR / "vific_labeled_100_batch3.csv",
    LABELED_DIR / "vific_labeled_100_batch4.csv",
]

all_dfs = []
for f in files:
    if f.exists():
        try:
            df = pd.read_csv(f, on_bad_lines='skip')
            all_dfs.append(df)
            print(f"{f.name}: {len(df)} samples")
        except Exception as e:
            print(f"{f.name}: ERROR - {e}")

merged = pd.concat(all_dfs, ignore_index=True)
merged = merged.drop_duplicates(subset=["id"], keep="last")

# Add label_numeric if missing
if "label_numeric" not in merged.columns:
    merged["label_numeric"] = merged["sentiment"].map(
        {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
    )

# Reorder columns
cols = ["id", "title", "content", "sentiment", "confidence", "method", "raw_response", "timestamp", "label_numeric"]
merged = merged[cols]

merged.to_csv(LABELED_DIR / "vific_labeled_1000_research.csv", index=False, encoding="utf-8-sig")
print(f"\nTotal: {len(merged)} unique samples")
print("\nSentiment distribution:")
print(merged["sentiment"].value_counts())
