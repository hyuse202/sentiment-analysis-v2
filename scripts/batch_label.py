#!/usr/bin/env python3
"""
batch_label.py
==============
Batch labeling script with resume capability.
Labels data in batches of 100 samples, automatically skipping already labeled samples.

Usage:
    python batch_label.py                          # Label all remaining samples
    python batch_label.py --batch-size 50          # Use smaller batch size
    python batch_label.py --dry-run                # Show what would be labeled
    python batch_label.py --status                 # Show current progress
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import from auto_label_fixed
from auto_label_fixed import (
    label_article,
    print_calibration_report,
)

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "vific_sample_1000.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "labeled" / "vific_labeled_1000_research.csv"
PROGRESS_FILE = PROJECT_ROOT / "data" / "labeled" / ".batch_progress.json"

# Existing labeled file (100 samples already done)
EXISTING_LABELED_FILE = PROJECT_ROOT / "data" / "labeled" / "vific_labeled_100_research.csv"


def load_input_data():
    """Load the source data file."""
    df = pd.read_csv(INPUT_FILE)
    df.columns = [c.lower().strip() for c in df.columns]
    if "title" not in df.columns:
        df["title"] = ""
    print(f"Loaded {len(df)} samples from {INPUT_FILE.name}")
    return df


def load_labeled_data():
    """Load already labeled data if exists."""
    # First check existing labeled file (100 samples already done)
    if EXISTING_LABELED_FILE.exists():
        df = pd.read_csv(EXISTING_LABELED_FILE)
        print(f"Found {len(df)} already labeled samples in {EXISTING_LABELED_FILE.name}")
        return df
    # Then check output file (if resuming from previous batch run)
    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE)
        print(f"Found {len(df)} already labeled samples in {OUTPUT_FILE.name}")
        return df
    return None


def get_labeled_ids():
    """Get set of IDs that have already been labeled."""
    labeled_df = load_labeled_data()
    if labeled_df is not None and "id" in labeled_df.columns:
        return set(labeled_df["id"].astype(str))
    return set()


def get_pending_samples(input_df, labeled_ids):
    """Get samples that haven't been labeled yet."""
    pending = input_df[~input_df["id"].astype(str).isin(labeled_ids)]
    return pending


def save_results(results_df, mode="append"):
    """Save results to output file."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # If output file doesn't exist but existing labeled file does, start from there
    if not OUTPUT_FILE.exists() and EXISTING_LABELED_FILE.exists():
        existing = pd.read_csv(EXISTING_LABELED_FILE)
        results_df = pd.concat([existing, results_df], ignore_index=True)
        print(f"Appended to existing {len(existing)} labeled samples")
    elif mode == "append" and OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        results_df = pd.concat([existing, results_df], ignore_index=True)

    # Remove duplicates by id, keep last
    results_df = results_df.drop_duplicates(subset=["id"], keep="last")

    results_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved {len(results_df)} total labeled samples to {OUTPUT_FILE.name}")


def label_batch(
    samples_df,
    batch_num,
    model="glm-5",
    temperature=0.15,
    delay=0.8,
    debug=False,
):
    """Label a batch of samples."""
    results = []
    total = len(samples_df)

    print(f"\n{'='*60}")
    print(f"  BATCH {batch_num} — {total} samples")
    print(f"{'='*60}")

    try:
        from tqdm import tqdm
        iterator = tqdm(samples_df.iterrows(), total=total, desc=f"Batch {batch_num}")
    except ImportError:
        iterator = samples_df.iterrows()

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
                "raw_response": r.get("raw", "") or r.get("error", ""),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Log errors
            if r["label"] == "ERROR":
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
            print(f"\n  [EXCEPTION] id={row.get('id')}: {e}")

        results.append(record)

        import time
        time.sleep(delay)

    # Add numeric label
    results_df = pd.DataFrame(results)
    results_df["label_numeric"] = results_df["sentiment"].map(
        {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
    )

    return results_df


def show_status():
    """Show current labeling progress."""
    input_df = load_input_data()
    labeled_ids = get_labeled_ids()
    pending = get_pending_samples(input_df, labeled_ids)

    total = len(input_df)
    done = len(labeled_ids)
    remaining = len(pending)

    print(f"\n{'='*50}")
    print(f"  LABELING STATUS")
    print(f"{'='*50}")
    print(f"  Total samples:     {total}")
    print(f"  Already labeled:   {done} ({done/total*100:.1f}%)")
    print(f"  Remaining:         {remaining} ({remaining/total*100:.1f}%)")
    print(f"{'='*50}")

    if remaining > 0:
        batches = (remaining + 99) // 100
        print(f"\n  Estimated batches remaining: {batches}")

    return done, remaining


def run_batch_labeling(
    batch_size=100,
    model="glm-5",
    temperature=0.15,
    delay=0.8,
    max_batches=None,
    debug=False,
    dry_run=False,
):
    """Main batch labeling function."""
    input_df = load_input_data()
    labeled_ids = get_labeled_ids()
    pending = get_pending_samples(input_df, labeled_ids)

    if len(pending) == 0:
        print("\n✓ All samples have been labeled!")
        show_status()
        return

    total_batches = (len(pending) + batch_size - 1) // batch_size
    print(f"\n{len(pending)} samples pending, divided into {total_batches} batch(es)")

    if max_batches:
        print(f"Will process at most {max_batches} batch(es)")

    if dry_run:
        print("\n[DRY RUN] Would label:")
        for i in range(min(3, total_batches)):
            start = i * batch_size
            end = min(start + batch_size, len(pending))
            batch_samples = pending.iloc[start:end]
            print(f"  Batch {i+1}: {len(batch_samples)} samples (IDs: {batch_samples['id'].iloc[0]} - {batch_samples['id'].iloc[-1]})")
        return

    # Process batches
    batches_to_process = min(max_batches, total_batches) if max_batches else total_batches

    for batch_idx in range(batches_to_process):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(pending))
        batch_samples = pending.iloc[start:end]

        batch_num = (len(labeled_ids) // batch_size) + batch_idx + 1

        try:
            results_df = label_batch(
                batch_samples,
                batch_num,
                model=model,
                temperature=temperature,
                delay=delay,
                debug=debug,
            )

            # Save after each batch
            save_results(results_df)

            # Print batch summary
            print(f"\n  Batch {batch_num} complete!")
            print_calibration_report(results_df)

        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted at batch {batch_num}")
            print("Progress saved. Run again to resume.")
            sys.exit(0)
        except Exception as e:
            print(f"\n✗ Error in batch {batch_num}: {e}")
            print("Progress saved. Run again to resume.")
            raise

    # Final status
    print(f"\n{'='*60}")
    print("  FINAL STATUS")
    show_status()

    # Print full calibration report
    final_df = pd.read_csv(OUTPUT_FILE)
    print_calibration_report(final_df)


def main():
    parser = argparse.ArgumentParser(
        description="Batch label Vietnamese financial news sentiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Samples per batch (default: 100)")
    parser.add_argument("--model", type=str, default="glm-5",
                        choices=["glm-4-flash", "glm-4-air", "glm-4", "glm-4-plus", "glm-5"],
                        help="GLM model to use")
    parser.add_argument("--temperature", type=float, default=0.15,
                        help="Temperature 0.0-1.0")
    parser.add_argument("--delay", type=float, default=0.8,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Maximum batches to process (for testing)")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug info")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without labeling")
    parser.add_argument("--status", action="store_true",
                        help="Show current progress only")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    run_batch_labeling(
        batch_size=args.batch_size,
        model=args.model,
        temperature=args.temperature,
        delay=args.delay,
        max_batches=args.max_batches,
        debug=args.debug,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
