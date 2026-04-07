#!/usr/bin/env python3
"""
batch_label_parallel.py
=======================
Parallel batch labeling script.
Each instance processes one batch independently with separate checkpointing.

Usage (run in separate terminals):
    # Terminal 1
    python scripts/batch_label_parallel.py --batch 1 --input data/processed/expansion_batch1.csv

    # Terminal 2
    python scripts/batch_label_parallel.py --batch 2 --input data/processed/expansion_batch2.csv

    # Terminal 3
    python scripts/batch_label_parallel.py --batch 3 --input data/processed/expansion_batch3.csv

    # Terminal 4
    python scripts/batch_label_parallel.py --batch 4 --input data/processed/expansion_batch4.csv
"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from auto_label_fixed import label_article, print_calibration_report

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw): return x


def run_batch_labeling(
    batch_id: int,
    input_file: str,
    model: str = "glm-5",
    temperature: float = 0.15,
    delay: float = 0.8,
    checkpoint_every: int = 50,
    debug: bool = False
):
    """Label a single batch with independent checkpointing."""

    # Unique output files per batch
    input_path = Path(input_file)
    output_file = LABELED_DIR / f"expansion_batch{batch_id}_labeled.csv"
    checkpoint_file = LABELED_DIR / f".expansion_batch{batch_id}_checkpoint.csv"

    # Ensure output directory exists
    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    # Load input
    df = pd.read_csv(input_path)
    total = len(df)
    print(f"\n{'='*60}")
    print(f"  BATCH {batch_id} - {total} samples")
    print(f"{'='*60}")
    print(f"  Input: {input_path.name}")
    print(f"  Output: {output_file.name}")
    print(f"  Model: {model} | Temperature: {temperature}")
    print(f"  Delay: {delay}s | Checkpoint every: {checkpoint_every}")
    print(f"{'='*60}\n")

    # Resume from checkpoint if exists
    results = []
    start_idx = 0

    if checkpoint_file.exists():
        ckpt = pd.read_csv(checkpoint_file)
        results = ckpt.to_dict("records")
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx}/{total} already done\n")

    if start_idx >= total:
        print(f"Batch {batch_id} already complete!")
        return

    # Calculate estimated time
    remaining = total - start_idx
    est_time = remaining * delay / 60
    print(f"Remaining: {remaining} samples (~{est_time:.1f} minutes)\n")

    # Label remaining
    iterator = df.iloc[start_idx:].iterrows()
    if HAS_TQDM:
        iterator = tqdm(iterator, total=remaining, desc=f"Batch {batch_id}")

    for idx, row in iterator:
        try:
            r = label_article(
                title=str(row.get("title", "")),
                content=str(row.get("content", "")),
                model=model,
                temperature=temperature,
                debug=debug,
            )

            record = {
                "id": row.get("id", idx),
                "title": str(row.get("title", ""))[:120],
                "content": str(row.get("content", ""))[:300],
                "sentiment": r["label"],
                "confidence": r["confidence"],
                "method": r["method"],
                "raw_response": r.get("raw", ""),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Log errors
            if r["label"] == "ERROR" and not HAS_TQDM:
                print(f"\n  [ERROR] id={row.get('id')}: {r.get('error', 'Unknown')}")

        except Exception as e:
            record = {
                "id": row.get("id", idx),
                "title": str(row.get("title", ""))[:120],
                "content": str(row.get("content", ""))[:300],
                "sentiment": "ERROR",
                "confidence": 0,
                "method": "exception",
                "raw_response": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            if not HAS_TQDM:
                print(f"\n  [EXCEPTION] id={row.get('id')}: {e}")

        results.append(record)

        # Checkpoint periodically
        if len(results) % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(checkpoint_file, index=False)

        time.sleep(delay)

    # Save final output
    results_df = pd.DataFrame(results)
    results_df["label_numeric"] = results_df["sentiment"].map(
        {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
    )
    results_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print(f"\n{'='*60}")
    print(f"  BATCH {batch_id} COMPLETE!")
    print(f"{'='*60}")
    print(f"  Output: {output_file}")
    print(f"  Total samples: {len(results_df)}")
    print_calibration_report(results_df)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel batch labeling for dataset expansion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run in separate terminals for parallel processing
    python scripts/batch_label_parallel.py --batch 1 --input data/processed/expansion_batch1.csv
    python scripts/batch_label_parallel.py --batch 2 --input data/processed/expansion_batch2.csv
    python scripts/batch_label_parallel.py --batch 3 --input data/processed/expansion_batch3.csv
    python scripts/batch_label_parallel.py --batch 4 --input data/processed/expansion_batch4.csv

    # Resume interrupted labeling
    python scripts/batch_label_parallel.py --batch 1 --input data/processed/expansion_batch1.csv

    # Debug mode (show raw responses)
    python scripts/batch_label_parallel.py --batch 1 --input data/processed/expansion_batch1.csv --debug
        """
    )
    parser.add_argument("--batch", type=int, required=True,
                        help="Batch ID (1, 2, 3, 4, ...)")
    parser.add_argument("--input", required=True,
                        help="Input CSV for this batch")
    parser.add_argument("--model", default="glm-5",
                        choices=["glm-4-flash", "glm-4-air", "glm-4", "glm-4-plus", "glm-5"],
                        help="GLM model to use (default: glm-5)")
    parser.add_argument("--temperature", type=float, default=0.15,
                        help="Temperature 0.0-1.0 (default: 0.15)")
    parser.add_argument("--delay", type=float, default=0.8,
                        help="Delay between API calls in seconds (default: 0.8)")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Checkpoint every N samples (default: 50)")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug info (raw GLM responses)")

    args = parser.parse_args()

    run_batch_labeling(
        batch_id=args.batch,
        input_file=args.input,
        model=args.model,
        temperature=args.temperature,
        delay=args.delay,
        checkpoint_every=args.checkpoint_every,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
