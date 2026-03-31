#!/usr/bin/env python3
"""
MVP: Auto-label Vietnamese financial news with GLM-5
Usage: python scripts/auto_label_mvp.py --sample 100
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment
load_dotenv()

# Initialize GLM client
from zai import ZaiClient

client = ZaiClient(api_key=os.getenv("ZAI_API_KEY"))

# Prompt template - Optimized for financial impact analysis
SENTIMENT_PROMPT = """Bạn là một Chuyên gia Phân tích Tài chính và Chứng khoán cao cấp tại Việt Nam.
Nhiệm vụ của bạn là đánh giá tác động kinh tế/tài chính của tin tức đối với doanh nghiệp hoặc thị trường được nhắc tới.

**HƯỚNG DẪN PHÂN LOẠI:**
- POSITIVE: Tin tức mang tính hỗ trợ, tăng trưởng tài sản, tăng lợi nhuận, nâng cao uy tín, mở rộng quy mô hoặc thị trường khởi sắc (ví dụ: lãi tăng, tài sản tăng, trúng thầu, được khen thưởng, phục hồi).
- NEGATIVE: Tin tức mang tính rủi ro, suy giảm giá trị, nợ xấu, vi phạm pháp luật, áp lực tài chính hoặc thị trường đi xuống (ví dụ: bán giải chấp, nợ xấu, bị thanh tra, lỗ, giảm điểm, mất uy tín).
- NEUTRAL: Chỉ dành cho các thông báo mang tính thủ tục thuần túy, lịch trình họp hành, hoặc các thay đổi không có tác động rõ ràng đến giá trị doanh nghiệp/thị trường.

**TIN TỨC CẦN PHÂN LOẠI:**
Tiêu đề: {title}
Nội dung: {content}

**Lưu ý:** Đánh giá dựa trên tác động tài chính thực tế, không chỉ ngữ pháp bề mặt.
Chỉ trả lời MỘT nhãn: POSITIVE, NEGATIVE, hoặc NEUTRAL."""


def label_article(title: str, content: str, max_retries: int = 3) -> str:
    """Label a single article with GLM-5"""
    # Truncate content to save tokens (increased for more context)
    content_truncated = content[:800] if len(content) > 800 else content

    prompt = SENTIMENT_PROMPT.format(title=title, content=content_truncated)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="glm-5",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,  # GLM-5 needs more tokens for reasoning
                temperature=0.1
            )

            label = response.choices[0].message.content.strip().upper()

            # Normalize label
            if "POSITIVE" in label:
                return "POSITIVE"
            elif "NEGATIVE" in label:
                return "NEGATIVE"
            elif "NEUTRAL" in label:
                return "NEUTRAL"
            else:
                return "NEUTRAL"  # Default fallback

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error after {max_retries} retries: {e}")
                return "ERROR"
            time.sleep(2 ** attempt)  # Exponential backoff

    return "ERROR"


def create_sample_data(output_path: str, n_samples: int = 100):
    """Create sample Vietnamese financial news data for testing"""
    sample_articles = [
        {
            "title": "VN-Index tăng mạnh lên mức 1,300 điểm",
            "content": "Thị trường chứng khoán Việt Nam hôm nay ghi nhận đà tăng mạnh của VN-Index, vượt mốc 1,300 điểm với thanh khoản cao. Các cổ phiếu ngân hàng và bất động sản dẫn dắt thị trường."
        },
        {
            "title": "Lãi suất ngân hàng tiếp tục giảm",
            "content": "Nhiều ngân hàng thương mại tiếp tục điều chỉnh giảm lãi suất huy động xuống mức 5-6% mỗi năm. Đây là tín hiệu tích cực cho doanh nghiệp và người vay vốn."
        },
        {
            "title": "Đồng USD tăng giá so với VND",
            "content": "Tỷ giá USD/VND đã vượt mức 25,000, tạo áp lực lên các doanh nghiệp nhập khẩu và có nợ bằng USD."
        },
        {
            "title": "Kinh tế Việt Nam tăng trưởng 6.5% trong quý III",
            "content": "Tổng cục Thống kê công bố kinh tế Việt Nam tăng trưởng 6.5% trong quý III, cao hơn kỳ vọng của các chuyên gia. Các ngành dịch vụ và công nghiệp chế biến chế tạo đóng góp chính vào đà tăng trưởng này."
        },
        {
            "title": "Xuất khẩu giảm sút do suy thoái kinh tế toàn cầu",
            "content": "Kim ngạch xuất khẩu tháng 10 giảm 15% so với cùng kỳ năm trước. Các thị trường lớn như Mỹ và EU đều giảm cầu nhập khẩu mạnh."
        },
        {
            "title": "FED giữ nguyên lãi suất, thị trường phản ứng tích cực",
            "content": "Cục dự trữ liên bang Mỹ quyết định giữ nguyên lãi suất ở mức 5.25-5.5%. Thị trường chứng khoán Việt Nam phản ứng tăng nhẹ sau tin tức này."
        },
        {
            "title": "Bất động sản tiếp tục suy yếu",
            "content": "Giao dịch bất động sản trong tháng qua giảm 40% so với cùng kỳ. Nhiều dự án trì hoãn do khó khăn về vốn và pháp lý."
        },
        {
            "title": "VinFast thông báo sản xuất xe điện mới",
            "content": "VinFast công bố mẫu xe điện VF3 với mức giá cạnh tranh, dự kiến mở bán trong quý tới. Cổ phiếu VFS tăng 5% sau thông báo."
        },
        {
            "title": "Chỉ số PMI sản xuất đạt 52.5 điểm",
            "content": "Chỉ số PMI sản xuất của Việt Nam đạt 52.5 điểm trong tháng, cho thấy hoạt động sản xuất tiếp tục mở rộng với tốc độ vững chắc."
        },
        {
            "title": "Thanh khoản thị trường thấp, nhà đầu tư thận trọng",
            "content": "Thanh khoản thị trường chứng khoán giảm mạnh trong các phiên gần đây. Nhà đầu tư thận trọng chờ đợi các tín hiệu từ thị trường quốc tế."
        }
    ]

    # Duplicate and shuffle to create more samples
    import random
    random.seed(42)

    all_articles = []
    for i in range(n_samples):
        article = sample_articles[i % len(sample_articles)].copy()
        article["id"] = i
        all_articles.append(article)

    df = pd.DataFrame(all_articles)
    df.to_csv(output_path, index=False)
    print(f"Created sample data: {output_path} ({n_samples} articles)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Auto-label Vietnamese financial news")
    parser.add_argument("--sample", type=int, default=100, help="Number of samples to label")
    parser.add_argument("--input", type=str, help="Input CSV file (optional, will create sample if not provided)")
    parser.add_argument("--output", type=str, default="data/labeled/labeled_mvp.csv", help="Output file")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save checkpoint every N articles")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")

    args = parser.parse_args()

    # Checkpoint path
    checkpoint_path = args.output.replace(".csv", "_checkpoint.csv")

    # Load or create data
    if args.input:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} articles from {args.input}")
    else:
        # Create sample data
        raw_path = f"data/raw/sample_{args.sample}.csv"
        os.makedirs("data/raw", exist_ok=True)
        df = create_sample_data(raw_path, args.sample)

    # Limit samples if --sample is specified
    if args.sample and len(df) > args.sample:
        df = df.head(args.sample)
        print(f"Limiting to {args.sample} samples")

    # Resume from checkpoint
    results = []
    start_idx = 0

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)
        results = checkpoint_df.to_dict('records')
        start_idx = len(results)
        print(f"\nResuming from checkpoint: {start_idx} articles already labeled")

    print(f"\nLabeling {len(df) - start_idx} articles with GLM-5...")

    # Label articles
    for idx, row in tqdm(df.iterrows(), total=len(df), initial=start_idx):
        # Skip already labeled
        if idx < start_idx:
            continue

        label = label_article(title=row["title"], content=row["content"])

        results.append({
            "id": row.get("id", idx),
            "title": row["title"],
            "content": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
            "sentiment": label,
            "timestamp": datetime.now().isoformat()
        })

        # Rate limit protection
        time.sleep(args.delay)

        # Save checkpoint
        if len(results) % args.checkpoint_every == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            print(f"\nCheckpoint saved: {len(results)} articles")

    # Save final results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output, index=False)

    # Remove checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint removed: {checkpoint_path}")

    # Print summary
    print(f"\n=== Results ===")
    print(f"Output: {args.output}")
    print(f"\nSentiment Distribution:")
    print(result_df["sentiment"].value_counts())
    print(f"\nDone! Labeled {len(results)} articles.")


if __name__ == "__main__":
    main()
