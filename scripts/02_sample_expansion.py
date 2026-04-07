#!/usr/bin/env python3
"""
02_sample_expansion.py
======================
Sample new articles from ViFiC for dataset expansion.
Excludes existing labeled IDs and outputs multiple batch files for parallel processing.

Usage:
    python scripts/02_sample_expansion.py \
        --existing data/labeled/vific_labeled_1000_research.csv \
        --vific data/vific/ViFiC-120M/train.txt \
        --output data/processed/ \
        --total-samples 4500 \
        --batches 4

Requirements:
    pip install pandas tqdm
"""

import argparse
import random
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw): return x


def load_existing_ids(filepath: str) -> set:
    """Load IDs from existing labeled file."""
    df = pd.read_csv(filepath)
    ids = set(df['id'].astype(int).tolist())
    print(f"Loaded {len(ids)} existing IDs to exclude")
    return ids


def sample_vific_lines(
    vific_path: str,
    existing_ids: set,
    total_samples: int,
    batches: int,
    output_dir: str,
    seed: int = 42
):
    """
    Sample lines from ViFiC, excluding existing IDs.
    Output multiple batch files for parallel processing.
    """
    random.seed(seed)

    # Count total lines and identify valid candidates
    print(f"\nScanning ViFiC for valid lines (excluding {len(existing_ids)} existing IDs)...")

    valid_line_nums = []

    with open(vific_path, 'r', encoding='utf-8') as f:
        iterator = enumerate(f)
        if HAS_TQDM:
            # Get total lines first for progress bar
            print("Counting total lines...")
            total_lines = sum(1 for _ in open(vific_path, 'r', encoding='utf-8'))
            f.seek(0)
            iterator = tqdm(enumerate(f), total=total_lines, desc="Scanning ViFiC")

        for line_num, line in iterator:
            if line_num not in existing_ids:
                text = line.strip()
                if len(text) > 50:  # Filter very short lines
                    valid_line_nums.append(line_num)

    print(f"\nFound {len(valid_line_nums):,} valid candidate lines")

    # Sample randomly
    if len(valid_line_nums) < total_samples:
        print(f"Warning: Only {len(valid_line_nums)} valid lines available")
        sampled_nums = valid_line_nums
    else:
        sampled_nums = random.sample(valid_line_nums, total_samples)

    print(f"Sampled {len(sampled_nums)} articles")

    # Divide into batches
    batch_size = total_samples // batches
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nCreating {batches} batch files (~{batch_size} samples each)...")

    for batch_idx in range(batches):
        start = batch_idx * batch_size
        end = start + batch_size if batch_idx < batches - 1 else len(sampled_nums)
        batch_nums = set(sampled_nums[start:end])

        # Read actual content for this batch
        articles = []
        with open(vific_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num in batch_nums:
                    text = line.strip()
                    articles.append({
                        'id': line_num,
                        'title': text[:100] + '...' if len(text) > 100 else text,
                        'content': text
                    })

        df = pd.DataFrame(articles)
        output_path = Path(output_dir) / f"expansion_batch{batch_idx+1}.csv"
        df.to_csv(output_path, index=False)
        print(f"  Batch {batch_idx+1}: {len(df)} samples -> {output_path.name}")

    print(f"\nDone! Created {batches} batch files in {output_dir}")
    return sampled_nums


def main():
    parser = argparse.ArgumentParser(
        description="Sample new articles from ViFiC for dataset expansion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/02_sample_expansion.py \\
      --existing data/labeled/vific_labeled_1000_research.csv \\
      --vific data/vific/ViFiC-120M/train.txt \\
      --output data/processed/ \\
      --total-samples 4500 \\
      --batches 4
        """
    )
    parser.add_argument("--existing", required=True,
                        help="Existing labeled CSV file with 'id' column")
    parser.add_argument("--vific", required=True,
                        help="ViFiC train.txt path")
    parser.add_argument("--output", required=True,
                        help="Output directory for batch files")
    parser.add_argument("--total-samples", type=int, default=4500,
                        help="Total samples to extract (default: 4500)")
    parser.add_argument("--batches", type=int, default=4,
                        help="Number of batch files to create (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.existing).exists():
        raise FileNotFoundError(f"Existing file not found: {args.existing}")
    if not Path(args.vific).exists():
        raise FileNotFoundError(f"ViFiC file not found: {args.vific}")

    # Load existing IDs
    existing_ids = load_existing_ids(args.existing)

    # Sample new articles
    sample_vific_lines(
        args.vific,
        existing_ids,
        args.total_samples,
        args.batches,
        args.output,
        args.seed
    )


if __name__ == "__main__":
    main()
