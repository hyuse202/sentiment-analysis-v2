#!/usr/bin/env python3
"""
Preprocess ViFiC (Vietnamese Financial Corpus) for sentiment analysis
Usage: python scripts/01_preprocess_vific.py --input data/vific/ --output data/processed/ --sample 1000
"""

import os
import re
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Clean text by removing HTML, URLs, and special characters"""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep Vietnamese characters
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ.,!?%\-]', '', text)

    return text.strip()


def load_vific_data(input_path: str) -> pd.DataFrame:
    """Load ViFiC data from CSV or TXT files"""
    input_dir = Path(input_path)

    # Find all CSV and TXT files
    csv_files = list(input_dir.glob("*.csv"))
    txt_files = list(input_dir.glob("**/*.txt"))  # Recursive search for txt files

    all_articles = []

    # Load TXT files (ViFiC format: each line is an article)
    if txt_files:
        print(f"Found {len(txt_files)} TXT files")
        for txt_file in tqdm(txt_files, desc="Loading TXT files"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        text = line.strip()
                        if text and len(text) > 50:  # Filter very short lines
                            all_articles.append({
                                'title': text[:100] + '...' if len(text) > 100 else text,
                                'content': text
                            })
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")

    # Load CSV files
    if csv_files:
        print(f"Found {len(csv_files)} CSV files")
        for csv_file in tqdm(csv_files, desc="Loading CSV files"):
            try:
                df = pd.read_csv(csv_file)
                all_articles.extend(df.to_dict('records'))
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

    if not all_articles:
        raise ValueError("No valid files loaded")

    combined_df = pd.DataFrame(all_articles)
    print(f"Loaded {len(combined_df)} total articles")

    return combined_df


def filter_relevant_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Filter articles relevant to financial sentiment"""
    # Define relevant keywords for Vietnamese financial news
    relevant_keywords = [
        'chứng khoán', 'stock', 'đầu tư', 'invest',
        'tài chính', 'financial', 'ngân hàng', 'bank',
        'kinh tế', 'economic', 'vĩ mô', 'macro',
        'doanh nghiệp', 'enterprise', 'công ty', 'company',
        'lãi suất', 'interest rate', 'tăng trưởng', 'growth',
        'thị trường', 'market', 'bất động sản', 'real estate',
        'vn-index', 'vnindex', 'hose', 'hnx'
    ]

    # Check if category column exists
    if 'category' in df.columns:
        # Filter by category
        df['category_lower'] = df['category'].astype(str).str.lower()
        mask = df['category_lower'].apply(
            lambda x: any(kw in x for kw in relevant_keywords)
        )
        filtered_df = df[mask].copy()
        df = filtered_df.drop(columns=['category_lower'])

    # Also filter by title/content keywords
    if 'title' in df.columns:
        df['text_combined'] = df['title'].astype(str) + ' ' + df.get('content', df.get('summary', '')).astype(str)
        df['text_lower'] = df['text_combined'].str.lower()

        mask = df['text_lower'].apply(
            lambda x: any(kw in x for kw in relevant_keywords)
        )
        df = df[mask].copy()
        df = df.drop(columns=['text_combined', 'text_lower'])

    print(f"Filtered to {len(df)} relevant articles")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate articles based on title"""
    if 'title' not in df.columns:
        return df

    before = len(df)
    df = df.drop_duplicates(subset=['title'], keep='first')
    after = len(df)

    print(f"Removed {before - after} duplicates, {after} unique articles remaining")
    return df


def sample_articles(df: pd.DataFrame, n_samples: int, random_state: int = 42) -> pd.DataFrame:
    """Sample n articles from dataset"""
    if len(df) <= n_samples:
        print(f"Dataset has {len(df)} articles, returning all")
        return df

    sampled_df = df.sample(n=n_samples, random_state=random_state)
    print(f"Sampled {n_samples} articles")
    return sampled_df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for sentiment labeling"""
    # Map common column names
    column_mapping = {
        'title': 'title',
        'headline': 'title',
        'tiêu đề': 'title',
        'content': 'content',
        'body': 'content',
        'text': 'content',
        'nội dung': 'content',
        'summary': 'content',
        'description': 'content'
    }

    # Rename columns
    new_columns = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in column_mapping:
            new_columns[col] = column_mapping[col_lower]

    if new_columns:
        df = df.rename(columns=new_columns)

    # Ensure required columns exist
    if 'title' not in df.columns:
        # Create title from first 100 chars of content
        if 'content' in df.columns:
            df['title'] = df['content'].str[:100] + '...'
        else:
            raise ValueError("No title or content column found")

    if 'content' not in df.columns:
        df['content'] = df['title']

    # Add id column
    df['id'] = range(len(df))

    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess ViFiC data")
    parser.add_argument("--input", type=str, required=True, help="Input directory with ViFiC CSV files")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--sample", type=int, default=1000, help="Number of articles to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-filter", action="store_true", help="Skip category filtering")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=" * 50)
    print("ViFiC Preprocessing Pipeline")
    print("=" * 50)

    # Step 1: Load data
    print("\n[1/6] Loading ViFiC data...")
    df = load_vific_data(args.input)

    # Step 2: Filter relevant categories
    if not args.skip_filter:
        print("\n[2/6] Filtering relevant categories...")
        df = filter_relevant_categories(df)
    else:
        print("\n[2/6] Skipping category filter...")

    # Step 3: Remove duplicates
    print("\n[3/6] Removing duplicates...")
    df = remove_duplicates(df)

    # Step 4: Clean text
    print("\n[4/6] Cleaning text...")
    if 'title' in df.columns:
        df['title'] = df['title'].apply(clean_text)
    if 'content' in df.columns:
        df['content'] = df['content'].apply(clean_text)

    # Step 5: Standardize columns
    print("\n[5/6] Standardizing columns...")
    df = standardize_columns(df)

    # Step 6: Sample
    print("\n[6/6] Sampling articles...")
    df = sample_articles(df, args.sample, args.seed)

    # Save output
    output_path = os.path.join(args.output, f"vific_sample_{args.sample}.csv")
    df[['id', 'title', 'content']].to_csv(output_path, index=False)

    print("\n" + "=" * 50)
    print("Preprocessing Complete!")
    print("=" * 50)
    print(f"Output: {output_path}")
    print(f"Total articles: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Preview
    print("\nPreview (first 3 articles):")
    for i, row in df.head(3).iterrows():
        print(f"\n[{row['id']}] {row['title'][:80]}...")
        print(f"    Content: {str(row['content'])[:100]}...")


if __name__ == "__main__":
    main()
