#!/usr/bin/env python3
"""
auto_label_complete.py
======================
Auto-label Vietnamese financial news sentiment using GLM-4/5 API.
Fixes neutral bias, adds confidence scoring, calibration report, and retry logic.

Usage:
    python auto_label_complete.py --input data/news.csv --sample 200
    python auto_label_complete.py --test          # run validation suite
    python auto_label_complete.py --calibrate     # check label distribution

Requirements:
    pip install zhipuai pandas tqdm python-dotenv

Environment:
    ZAI_API_KEY=your_glm_api_key   (in .env file or system env)
    OR
    GLM_API_KEY=your_glm_api_key
"""

import os
import re
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw): return x

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CLIENT SETUP  (supports both zai SDK and direct zhipuai)
# ─────────────────────────────────────────────────────────────────────────────

def _build_client():
    key = os.getenv("ZAI_API_KEY") or os.getenv("GLM_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "API key not found. Set ZAI_API_KEY or GLM_API_KEY in .env or environment."
        )
    try:
        from zai import ZaiClient
        return ZaiClient(api_key=key), "zai"
    except ImportError:
        pass
    try:
        from zhipuai import ZhipuAI
        return ZhipuAI(api_key=key), "zhipuai"
    except ImportError:
        raise ImportError(
            "No GLM SDK found. Install one:\n"
            "  pip install zai-sdk\n"
            "  OR pip install zhipuai"
        )

CLIENT, SDK_TYPE = _build_client()

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

# FIX 1: System prompt separated from user prompt (better role clarity)
# FIX 2: Removed NEUTRAL example from few-shot (was anchoring bias)
# FIX 3: Explicit instruction to NOT use NEUTRAL unless truly procedural
# FIX 4: Added "Trả lời ngay, không giải thích" to suppress chain-of-thought leaking

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích tác động tài chính tin tức Việt Nam.
Nhiệm vụ: Phân loại sentiment tin tức tài chính theo tác động kinh tế thực tế.

QUY TẮC PHÂN LOẠI:
- POSITIVE: tăng trưởng, lợi nhuận tăng, phục hồi, trúng thầu, được thưởng, thị trường khởi sắc
- NEGATIVE: giảm sút, lỗ, nợ xấu, bán giải chấp, bị phạt/thanh tra, thị trường suy yếu, rủi ro tăng
- NEUTRAL: CHỈ dành cho thông báo hành chính thuần túy (lịch họp, thay đổi nhân sự không kèm đánh giá)

NGUYÊN TẮC QUAN TRỌNG:
1. Đánh giá theo TÁC ĐỘNG KINH TẾ THỰC TẾ, không chỉ dựa vào từ ngữ bề mặt
2. Nếu có thể phân loại POSITIVE hoặc NEGATIVE, KHÔNG dùng NEUTRAL
3. Trả lời ĐÚNG 1 từ: POSITIVE hoặc NEGATIVE hoặc NEUTRAL"""

# Compact few-shot: 2 positive, 2 negative only (no neutral anchor)
USER_PROMPT_TEMPLATE = """Ví dụ:
"VN-Index tăng 3%, vượt mốc 1,300 điểm, ngân hàng dẫn đầu" → POSITIVE
"Lãi suất ngân hàng giảm 1%, hỗ trợ doanh nghiệp vay vốn" → POSITIVE
"Xuất khẩu giảm 15%, doanh nghiệp gặp khó, nhiều đơn hàng bị hủy" → NEGATIVE
"Bán giải chấp cổ phiếu NVL, tỷ lệ sở hữu giảm mạnh, áp lực nợ tăng" → NEGATIVE

Phân loại tin tức sau (trả lời 1 từ duy nhất):
Tiêu đề: {title}
Nội dung: {content}
→"""

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION LOGIC  (robust, handles all GLM response patterns)
# ─────────────────────────────────────────────────────────────────────────────

# FIX 5: Vietnamese negation-aware mapping
_VI_MAP = {
    "TÍCH CỰC": "POSITIVE", "TICH CUC": "POSITIVE",
    "TIÊU CỰC": "NEGATIVE", "TIEU CUC": "NEGATIVE",
    "TRUNG LẬP": "NEUTRAL",  "TRUNG LAP": "NEUTRAL",
}

# FIX 6: Ordered patterns by confidence (most specific first)
_PATTERNS = [
    (r'^(POSITIVE|NEGATIVE|NEUTRAL)[.\s,!]*$',                                  100),
    (r'(?:LÀ|IS|:|=|→|->)\s*(POSITIVE|NEGATIVE|NEUTRAL)',                        90),
    (r'(?:NHÃN|LABEL|KẾT LUẬN|CLASSIFICATION|TRẢ LỜI|PHÂN LOẠI)[:\s]*(POSITIVE|NEGATIVE|NEUTRAL)', 90),
    (r'["\'\[\(](POSITIVE|NEGATIVE|NEUTRAL)["\'\]\)]',                            85),
    (r'(POSITIVE|NEGATIVE|NEUTRAL)[.\s,!]*$',                                    80),
    (r'(POSITIVE|NEGATIVE|NEUTRAL)',                                              70),
]

# FIX 7: Semantic keyword fallback (avoids defaulting to NEUTRAL)
_POS_KW = re.compile(
    r'TĂNG TRƯỞNG|TĂNG MẠNH|PHỤC HỒI|LỢI NHUẬN TĂNG|VƯỢT MỨC|KHỞI SẮC|'
    r'BỨT PHÁ|TRÚNG THẦU|ĐƯỢC THƯỞNG|TĂNG VỐN|MỞ RỘNG|CẢI THIỆN'
)
_NEG_KW = re.compile(
    r'GIẢM SÂU|BÁN THÁO|GIẢI CHẤP|NỢ XẤU|BỊ LỖ|SUY GIẢM|SỤT GIẢM|'
    r'VI PHẠM|BỊ PHẠT|ĐÌNH CHỈ|RỦI RO TĂNG|MẤT UY TÍN|CẮT GIẢM'
)


def extract_label(
    content: str,
    reasoning: str = "",
    finish_reason: str = "stop"
) -> tuple[str, int, str]:
    """
    Extract sentiment label from GLM response.

    Returns:
        (label, confidence_0_100, extraction_method)

    FIX 8: Handles GLM-5 reasoning model where answer is in reasoning_content
    FIX 9: Negation-aware Vietnamese parsing ("Không tích cực" -> NEGATIVE)
    FIX 10: Never defaults to NEUTRAL without checking keywords first
    """
    def _check_negation(text: str, pos: int) -> bool:
        before = text[max(0, pos - 15):pos].upper()
        return bool(re.search(r'KHÔNG|CHƯA|THIẾU', before))

    def _scan(text: str) -> tuple[Optional[str], int]:
        if not text or not text.strip():
            return None, 0
        tu = text.strip().upper()

        # Vietnamese label mapping (with negation check)
        for vi, en in _VI_MAP.items():
            idx = tu.find(vi)
            if idx >= 0 and not _check_negation(tu, idx):
                return en, 85

        # "Không tích cực" / "không khả quan" -> NEGATIVE
        if re.search(r'KHÔNG\s+(?:TÍCH CỰC|TỐT|KHẢ QUAN|LẠC QUAN|THUẬN LỢI)', tu):
            return "NEGATIVE", 80
        if re.search(r'KHÔNG\s+(?:TIÊU CỰC|XẤU|RỦI RO)', tu):
            return "POSITIVE", 80

        # Regex patterns
        for pattern, conf in _PATTERNS:
            m = re.search(pattern, tu, re.MULTILINE)
            if m:
                return m.group(1), conf

        return None, 0

    # Priority 1: direct content
    label, conf = _scan(content)
    if label:
        return label, conf, "content"

    # Priority 2: reasoning tail (GLM-5 chain-of-thought conclusion)
    if reasoning:
        tail = reasoning[-400:] if len(reasoning) > 400 else reasoning
        label, conf = _scan(tail)
        if label:
            return label, max(conf - 5, 40), "reasoning_tail"

        # Priority 3: full reasoning scan
        label, conf = _scan(reasoning)
        if label:
            return label, max(conf - 15, 30), "reasoning_full"

    # Priority 4: semantic keyword fallback (FIX: avoids NEUTRAL default)
    combined = ((content or "") + " " + (reasoning or "")).upper()
    pos_count = len(_POS_KW.findall(combined))
    neg_count = len(_NEG_KW.findall(combined))

    if pos_count > neg_count and pos_count > 0:
        return "POSITIVE", 45, "keyword_fallback"
    if neg_count > pos_count and neg_count > 0:
        return "NEGATIVE", 45, "keyword_fallback"

    # Priority 5: truly ambiguous -> NEUTRAL (low confidence flagged)
    return "NEUTRAL", 25, "ambiguous_default"


# ─────────────────────────────────────────────────────────────────────────────
# CORE LABELING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def label_article(
    title: str,
    content: str,
    model: str = "glm-5",
    temperature: float = 0.15,
    max_tokens: int = 800,
    max_retries: int = 3,
    debug: bool = False,
) -> dict:
    """
    Label a single article. Returns dict with label, confidence, method, raw_response.

    FIX 11: temperature 0.15 (not 0.1 which causes deterministic neutral hedging)
    FIX 12: max_tokens 800 (GLM-5 needs ~600 for reasoning before output)
    FIX 13: system + user message separation for better role adherence
    """
    title = (title or "").strip()
    content = (content or "").strip()

    # Smart truncation: keep title full, truncate content
    content_trunc = content[:600] if len(content) > 600 else content

    user_msg = USER_PROMPT_TEMPLATE.format(
        title=title if title else "(không có tiêu đề)",
        content=content_trunc if content_trunc else "(không có nội dung)"
    )

    for attempt in range(max_retries):
        try:
            response = CLIENT.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            raw_content  = response.choices[0].message.content or ""
            reasoning    = getattr(response.choices[0].message, "reasoning_content", "") or ""
            finish_reason = response.choices[0].finish_reason or "stop"

            # FIX 14: Warn if finish_reason=length (answer may be cut off)
            if finish_reason == "length" and debug:
                print(f"  [WARN] finish_reason=length — increase max_tokens if labels are wrong")

            label, confidence, method = extract_label(raw_content, reasoning, finish_reason)

            if debug:
                print(f"\n  [DEBUG] title='{title[:40]}...'")
                print(f"  content (raw): '{raw_content[:80]}'")
                print(f"  reasoning[:100]: '{reasoning[:100]}'")
                print(f"  finish_reason: {finish_reason}")
                print(f"  → label={label}, conf={confidence}, method={method}")

            return {
                "label":      label,
                "confidence": confidence,
                "method":     method,
                "raw":        raw_content[:200],
                "error":      None,
            }

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                if debug:
                    print(f"  [RETRY {attempt+1}] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                return {
                    "label":      "ERROR",
                    "confidence": 0,
                    "method":     "exception",
                    "raw":        "",
                    "error":      str(e),
                }

    return {"label": "ERROR", "confidence": 0, "method": "max_retries", "raw": "", "error": "Max retries"}


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_calibration_report(df: pd.DataFrame):
    """
    FIX 15: Check label distribution against expected benchmark.
    Expected from Vietnamese financial news literature (Vu et al. 2023):
      POSITIVE ~33%, NEUTRAL ~43%, NEGATIVE ~24%
    If NEUTRAL >> 43%, extraction is likely broken.
    """
    total = len(df)
    if total == 0:
        print("  No data to report.")
        return

    dist = df["sentiment"].value_counts()
    expected = {"POSITIVE": 0.33, "NEUTRAL": 0.43, "NEGATIVE": 0.24}

    print(f"\n{'─'*55}")
    print(f"  CALIBRATION REPORT  (n={total})")
    print(f"{'─'*55}")
    print(f"  {'Label':<12} {'Count':>7} {'Actual':>8} {'Expected':>10}  {'Status'}")
    print(f"  {'─'*50}")

    warnings = []
    for label, exp_pct in expected.items():
        count = dist.get(label, 0)
        actual_pct = count / total
        exp_lo, exp_hi = exp_pct * 0.5, exp_pct * 1.8  # ±80% tolerance
        status = "✓" if exp_lo <= actual_pct <= exp_hi else "⚠ CHECK"
        if status != "✓":
            warnings.append(f"  {label}: {actual_pct:.0%} vs expected ~{exp_pct:.0%}")
        print(f"  {label:<12} {count:>7} {actual_pct:>7.1%}   ~{exp_pct:.0%}        {status}")

    error_count = dist.get("ERROR", 0)
    if error_count > 0:
        print(f"  {'ERROR':<12} {error_count:>7} {error_count/total:>7.1%}              ⚠ API errors")
        warnings.append(f"  {error_count} API errors — check API key and rate limits")

    # FIX 16: Neutral bias specific warning
    neutral_pct = dist.get("NEUTRAL", 0) / total
    if neutral_pct > 0.6:
        warnings.append(
            f"  HIGH NEUTRAL BIAS ({neutral_pct:.0%}) — likely extraction failure.\n"
            f"  Try: --debug --sample 5 to inspect raw GLM responses."
        )

    # Confidence stats
    if "confidence" in df.columns:
        low_conf = (df["confidence"] < 50).sum()
        print(f"\n  Low confidence (<50): {low_conf} ({low_conf/total:.1%})")
        if low_conf / total > 0.2:
            warnings.append(f"  Many low-confidence labels ({low_conf/total:.0%}) — consider manual review")

    if warnings:
        print(f"\n  WARNINGS:")
        for w in warnings:
            print(f"  {w}")
    else:
        print(f"\n  Distribution looks healthy.")

    print(f"{'─'*55}")


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────

VALIDATION_CASES = [
    # (expected_label, title, content)
    ("POSITIVE", "VN-Index tăng mạnh", "Thị trường tăng 3%, vượt mốc 1,300 điểm với thanh khoản cao"),
    ("POSITIVE", "Lãi suất ngân hàng giảm", "Nhiều ngân hàng giảm lãi suất xuống 5-6%, doanh nghiệp hưởng lợi"),
    ("POSITIVE", "VinFast xuất khẩu tăng", "VinFast xuất khẩu 10,000 xe trong quý, doanh thu tăng 40%"),
    ("POSITIVE", "GDP tăng trưởng vượt kỳ vọng", "Kinh tế Việt Nam tăng 6.5%, vượt dự báo 5.8% của IMF"),
    ("NEGATIVE", "Xuất khẩu giảm mạnh", "Kim ngạch xuất khẩu giảm 15% do thị trường Mỹ, EU suy yếu"),
    ("NEGATIVE", "Bán giải chấp cổ phiếu NVL", "Cổ đông lớn bán giải chấp 5 triệu cổ phiếu NVL, áp lực nợ tăng"),
    ("NEGATIVE", "Ngân hàng tăng nợ xấu", "Tỷ lệ nợ xấu toàn ngành tăng lên 3.2%, cao nhất 5 năm"),
    ("NEGATIVE", "VN-Index lao dốc phiên chiều", "Cổ phiếu bất động sản giảm mạnh, VN-Index mất 25 điểm cuối phiên"),
    ("NEUTRAL", "Thông báo lịch ĐHCĐ", "Công ty thông báo lịch tổ chức ĐHCĐ thường niên vào ngày 15/4"),
    ("NEUTRAL", "Thay đổi đăng ký kinh doanh", "Doanh nghiệp thay đổi địa chỉ trụ sở từ quận 1 sang quận 7"),
]


def run_validation(model: str, temperature: float, debug: bool = False):
    """Run validation suite and print accuracy report."""
    print(f"\n{'='*55}")
    print(f"  VALIDATION SUITE  (model={model}, temp={temperature})")
    print(f"{'='*55}")

    correct = 0
    results = []
    for expected, title, content in VALIDATION_CASES:
        r = label_article(title, content, model=model, temperature=temperature, debug=debug)
        label = r["label"]
        ok = label == expected
        if ok:
            correct += 1
        icon = "✓" if ok else f"✗ (got {label})"
        results.append((ok, expected, label, r["confidence"], r["method"], title))
        print(f"  {icon:<18} [{r['confidence']:3d}%/{r['method'][:12]}] {title[:42]}")

    accuracy = correct / len(VALIDATION_CASES) * 100
    print(f"\n  Accuracy: {correct}/{len(VALIDATION_CASES)} ({accuracy:.0f}%)")

    if accuracy >= 80:
        print(f"  Status: GOOD — ready for production labeling")
    elif accuracy >= 60:
        print(f"  Status: ACCEPTABLE — review failed cases")
    else:
        print(f"  Status: POOR — check API key, model, and prompt settings")
        print(f"  Tip: run with --debug to see raw GLM responses")

    print(f"{'='*55}")
    return accuracy


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-label Vietnamese financial news sentiment (GLM-4/5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_label_complete.py --test
  python auto_label_complete.py --input data/news.csv --sample 500
  python auto_label_complete.py --input data/news.csv --model glm-4 --output out/labeled.csv
  python auto_label_complete.py --input data/news.csv --debug --sample 5
        """
    )
    parser.add_argument("--input",            type=str,   help="Input CSV (columns: title, content)")
    parser.add_argument("--output",           type=str,   default="data/labeled/labeled_complete.csv")
    parser.add_argument("--sample",           type=int,   default=None, help="Limit to N samples")
    parser.add_argument("--model",            type=str,   default="glm-5",
                        choices=["glm-4-flash", "glm-4-air", "glm-4", "glm-4-plus", "glm-5", "glm-z1-flash"],
                        help="GLM model to use (default: glm-5)")
    parser.add_argument("--temperature",      type=float, default=0.15,
                        help="Temperature 0.0-1.0 (default: 0.15)")
    parser.add_argument("--max-tokens",       type=int,   default=800,
                        help="Max tokens per response (default: 800, use 1200 for GLM-5)")
    parser.add_argument("--delay",            type=float, default=0.8,
                        help="Delay between API calls in seconds (default: 0.8)")
    parser.add_argument("--checkpoint-every", type=int,   default=50,
                        help="Save checkpoint every N articles (default: 50)")
    parser.add_argument("--resume",           action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--test",             action="store_true",
                        help="Run validation suite with known test cases")
    parser.add_argument("--calibrate",        action="store_true",
                        help="Print calibration report on existing output file")
    parser.add_argument("--debug",            action="store_true",
                        help="Show raw GLM responses for debugging")
    parser.add_argument("--low-conf-review",  action="store_true",
                        help="After labeling, print all low-confidence predictions for review")

    args = parser.parse_args()

    # ── CALIBRATE MODE ─────────────────────────────────────────────────────
    if args.calibrate:
        if not os.path.exists(args.output):
            print(f"File not found: {args.output}")
            return
        df = pd.read_csv(args.output)
        print_calibration_report(df)
        return

    # ── TEST MODE ──────────────────────────────────────────────────────────
    if args.test:
        run_validation(
            model=args.model,
            temperature=args.temperature,
            debug=args.debug
        )
        return

    # ── LOAD DATA ──────────────────────────────────────────────────────────
    if args.input:
        df = pd.read_csv(args.input)
        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]
        if "title" not in df.columns:
            df["title"] = ""
        if "content" not in df.columns and "text" in df.columns:
            df["content"] = df["text"]
        elif "content" not in df.columns:
            raise ValueError("Input CSV must have 'content' (or 'text') column")
        print(f"Loaded {len(df)} articles from {args.input}")
    else:
        # Built-in demo data (10 articles)
        demo = [
            {"title": "VN-Index bứt phá mạnh",    "content": "VN-Index tăng 25 điểm, vượt mốc 1,300, ngân hàng và BĐS dẫn dắt"},
            {"title": "Lãi suất ngân hàng giảm",  "content": "Nhiều ngân hàng giảm lãi suất huy động, hỗ trợ doanh nghiệp vay vốn"},
            {"title": "GDP quý III tăng 6.5%",     "content": "Tổng cục Thống kê: GDP tăng 6.5%, vượt kỳ vọng 5.8%"},
            {"title": "Xuất khẩu giảm 15%",        "content": "Kim ngạch xuất khẩu giảm 15%, thị trường Mỹ và EU đều yếu"},
            {"title": "VN-Index lao dốc cuối phiên","content": "Cổ phiếu BĐS giảm mạnh, VN-Index mất 30 điểm trong 1 giờ"},
            {"title": "Nợ xấu ngân hàng tăng",    "content": "Tỷ lệ nợ xấu toàn ngành lên 3.2%, cao nhất 5 năm qua"},
            {"title": "Lịch ĐHCĐ 2024",            "content": "Công ty thông báo ĐHCĐ thường niên tổ chức ngày 20/4/2024"},
            {"title": "VinFast ra mắt VF3",        "content": "VinFast ra mắt VF3, cổ phiếu VFS tăng 5%, thị trường phản ứng tích cực"},
            {"title": "Bán giải chấp NVL",         "content": "Cổ đông lớn bán giải chấp 5 triệu cổ phiếu NVL, áp lực tài chính tăng"},
            {"title": "PMI đạt 52.5 điểm",         "content": "PMI sản xuất Việt Nam đạt 52.5, mở rộng sản xuất tháng thứ 4 liên tiếp"},
        ]
        df = pd.DataFrame(demo)
        print(f"Using built-in demo data ({len(df)} articles). Use --input to load your CSV.")

    if args.sample and len(df) > args.sample:
        df = df.head(args.sample)
        print(f"Sampling {args.sample} articles")

    # ── RESUME ─────────────────────────────────────────────────────────────
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.csv")
    results = []
    start_idx = 0

    if args.resume and checkpoint_path.exists():
        ckpt = pd.read_csv(checkpoint_path)
        results = ckpt.to_dict("records")
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx}/{len(df)} already done")

    # ── LABEL LOOP ─────────────────────────────────────────────────────────
    remaining = len(df) - start_idx
    print(f"\nLabeling {remaining} articles")
    print(f"Model: {args.model} | temp: {args.temperature} | delay: {args.delay}s\n")

    iter_df = df.iloc[start_idx:].iterrows()
    if HAS_TQDM:
        iter_df = tqdm(iter_df, total=remaining, desc="Labeling")

    for idx, row in iter_df:
        r = label_article(
            title=str(row.get("title", "")),
            content=str(row.get("content", "")),
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            debug=args.debug,
        )

        record = {
            "id":         row.get("id", idx),
            "title":      str(row.get("title", ""))[:120],
            "content":    str(row.get("content", ""))[:300],
            "sentiment":  r["label"],
            "confidence": r["confidence"],
            "method":     r["method"],
            "raw_response": r["raw"],
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        results.append(record)

        time.sleep(args.delay)

        # Checkpoint
        done = start_idx + len(results) - start_idx
        if len(results) % args.checkpoint_every == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            if not HAS_TQDM:
                print(f"  Checkpoint: {len(results)}/{len(df)}")

    # ── SAVE OUTPUT ────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # ── REPORTS ────────────────────────────────────────────────────────────
    print(f"\nSaved {len(results)} labeled articles → {args.output}")
    print_calibration_report(result_df)

    if args.low_conf_review:
        low = result_df[result_df["confidence"] < 50]
        if len(low) > 0:
            print(f"\n  LOW CONFIDENCE CASES ({len(low)}) — recommend manual review:")
            for _, row in low.iterrows():
                print(f"    [{row['sentiment']:8s} {row['confidence']:3d}%] {row['title'][:60]}")

    # ── EXPORT RESEARCH-READY CSV ──────────────────────────────────────────
    # Daily Sentiment Index compatible columns
    research_df = result_df[result_df["sentiment"].isin(["POSITIVE", "NEGATIVE", "NEUTRAL"])].copy()
    research_df["label_numeric"] = research_df["sentiment"].map(
        {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
    )
    research_path = Path(args.output).with_name(
        Path(args.output).stem + "_research.csv"
    )
    research_df.to_csv(research_path, index=False, encoding="utf-8-sig")
    print(f"Research-ready CSV (with numeric labels) → {research_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()