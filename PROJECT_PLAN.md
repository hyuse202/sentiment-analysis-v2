# Vietnamese Stock Sentiment Analysis Project Plan

**Created:** March 30, 2026
**Author:** Hyuse (Tấn Huy)
**Status:** Draft - Pending Approval

---

## 1. Overview

### 1.1 Objective
Xây dựng hệ thống phân tích sentiment cho tin tức tài chính Việt Nam, sử dụng GLM-5 API để auto-label dữ liệu ViFiC, từ đó fine-tune PhoBERT cho bài toán sentiment classification.

### 1.2 Key Innovation
- **Auto-labeling với LLM:** Sử dụng GLM-5 API thay vì manual labeling (tiết kiệm thời gian)
- **Vietnamese-specific:** PhoBERT model chuyên cho tiếng Việt
- **Large-scale data:** ViFiC dataset với 160,490 articles

### 1.3 Timeline
- **Estimated:** 3-5 days
- **Deadline:** TBD

---

## 2. Data Sources

### 2.1 Primary Dataset: ViFiC (Vietnamese Financial Corpus)

| Attribute | Value |
|-----------|-------|
| **Source** | Kaggle |
| **Link** | `kaggle.com/datasets/daddychillonkaggle/vietnamese-financial-corpus` |
| **Size** | 160,490 articles |
| **Period** | 2010-2025 (15 năm) |
| **Format** | Title + Content |
| **Labels** | Chưa có (cần auto-label) |

### 2.2 Reference Datasets

| Dataset | Size | Labels | Purpose |
|---------|------|--------|---------|
| Financial PhraseBank | 4,846 sentences | Human-annotated | Benchmark |
| ViFiC-93M (local) | ~60,000 articles | None | Backup |

### 2.3 Stock Price Data

| Source | Library | Use |
|--------|---------|-----|
| vnstock | `pip install vnstock` | Historical prices |
| Yahoo Finance | `yfinance` | VN-Index |

---

## 3. Methodology

### 3.1 Phase 1: Data Preparation (Day 1)

**Step 1.1: Download ViFiC**
```bash
kaggle datasets download -d daddychillonkaggle/vietnamese-financial-corpus
unzip vietnamese-financial-corpus.zip -d data/vific/
```

**Step 1.2: Filter Relevant Articles**
- Categories: stock market, finance, business, macroeconomics
- Remove: specific stock news (không ảnh hưởng cả market)
- Target: ~50,000-100,000 articles

**Step 1.3: Preprocessing**
```python
# Text cleaning
- Lowercase
- Remove HTML tags
- Remove URLs
- Remove special characters
- Tokenize with VnCoreNLP
```

---

### 3.2 Phase 2: Auto-Labeling with GLM-5 (Day 1-2)

**Step 2.1: Design Prompt Template**

```
Bạn là chuyên gia phân tích tài chính. Hãy phân loại sentiment của tin tức sau:

Tiêu đề: {title}
Nội dung: {content}

Chỉ trả lời một trong ba nhãn: POSITIVE, NEGATIVE, hoặc NEUTRAL.
Sentiment:
```

**Step 2.2: Batch Processing Strategy**

| Batch Size | 100 articles/batch |
|------------|-------------------|
| Rate Limit | Check API docs |
| Estimated Cost | ~$5-20 for 100k articles |
| Estimated Time | 3-6 hours |

**Step 2.3: Implementation**

```python
import requests
import json

def label_with_glm5(title, content, api_key):
    prompt = f"""Bạn là chuyên gia phân tích tài chính. Hãy phân loại sentiment của tin tức sau:

Tiêu đề: {title}
Nội dung: {content[:500]}  # Truncate to save tokens

Chỉ trả lời một trong ba nhãn: POSITIVE, NEGATIVE, hoặc NEUTRAL.
Sentiment:"""
    
    response = requests.post(
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "glm-4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.1
        }
    )
    return response.json()["choices"][0]["message"]["content"].strip()
```

**Step 2.4: Quality Control**
- Manual validate 100-200 samples
- Calculate inter-annotator agreement với sample
- Confidence threshold > 0.8

---

### 3.3 Phase 3: Model Training (Day 2-3)

**Step 3.1: Data Split (Time-based)**

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2010-2022 | Training |
| Validation | 2023 | Hyperparameter tuning |
| Test | 2024-2025 | Final evaluation |

**Step 3.2: Fine-tune PhoBERT**

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "vinai/phobert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3  # positive, negative, neutral
)

# Training config
training_args = {
    "num_train_epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01
}
```

**Step 3.3: Baseline Models**

| Model | Purpose |
|-------|---------|
| Random Baseline | Lower bound |
| Majority Class | Baseline |
| TF-IDF + SVM | Traditional ML |
| PhoBERT (fine-tuned) | Main model |
| Zero-shot GLM-5 | Comparison |

---

### 3.4 Phase 4: Evaluation (Day 3-4)

**Step 4.1: Metrics**

| Metric | Formula | Use |
|--------|---------|-----|
| Accuracy | (TP+TN)/Total | Overall |
| Precision | TP/(TP+FP) | Per class |
| Recall | TP/(TP+FN) | Per class |
| F1-Score | 2*P*R/(P+R) | Balance |
| MCC | - | Imbalanced data |

**Step 4.2: Expected Results**

Based on papers:
- Paper 3: PhoBERT + CNN achieved **81%+ accuracy**
- Paper 4: Llama 3 achieved **89.3%** on Financial PhraseBank
- Our target: **75-85% accuracy**

**Step 4.3: Error Analysis**
- Confusion matrix
- Misclassified examples
- Sentiment distribution

---

### 3.5 Phase 5: Application & Report (Day 4-5)

**Step 5.1: Granger Causality Test**
- Sentiment → Stock returns
- Returns → Sentiment
- Lag analysis

**Step 5.2: Event Study**
- CAR (Cumulative Abnormal Returns)
- Window: [-5, +5] days around news
- FF5 factor adjustment

**Step 5.3: Report Writing**
- Introduction
- Literature Review
- Methodology
- Results
- Discussion
- Conclusion

---

## 4. Technical Architecture

```
sentiment_analysis_v2/
├── data/
│   ├── raw/                    # Original ViFiC data
│   ├── processed/              # Cleaned data
│   ├── labeled/                # Auto-labeled data
│   └── splits/                 # Train/val/test
├── scripts/
│   ├── 01_download_data.py     # Kaggle download
│   ├── 02_preprocess.py        # Text cleaning
│   ├── 03_auto_label.py        # GLM-5 labeling
│   ├── 04_train_model.py       # PhoBERT training
│   ├── 05_evaluate.py          # Evaluation
│   └── 06_analysis.py          # Statistical analysis
├── models/
│   └── phobert_sentiment/      # Trained model
├── results/
│   ├── metrics.json            # Evaluation metrics
│   ├── confusion_matrix.png    # Visualization
│   └── analysis/               # Statistical tests
├── papers/                     # Reference papers + MD
│   ├── *.pdf
│   └── *.md
├── docs/
│   └── API_NOTES.md            # GLM-5 API notes
├── PROJECT_PLAN.md             # This file
└── README.md                   # Project overview
```

---

## 5. Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API rate limit | Medium | High | Batch processing, retry logic |
| Poor label quality | Medium | High | Manual validation, confidence threshold |
| Model overfitting | Low | Medium | Time-based split, regularization |
| Disk space | Low | High | Already addressed |

---

## 6. Resources Required

### 6.1 API Keys
- [x] GLM-5 API key (Zhipu AI)
- [x] Kaggle API key

### 6.2 Compute
- [x] VPS Azure (current) - for labeling
- [x] Local machine (hyuse-msi) - for training (GPU available)

### 6.3 Libraries
```
transformers
torch
pandas
numpy
scikit-learn
vncorenlp
tqdm
requests
```

---

## 7. Comparison with Previous Project

| Aspect | Previous Project | This Project |
|--------|------------------|--------------|
| Data | Synthetic (11,227 articles) | Real ViFiC (160,490 articles) |
| Labels | Keyword-based (data leakage) | LLM auto-labeled |
| Model | TF-IDF + SVM | PhoBERT fine-tuned |
| Quality | Low (synthetic) | High (real data) |
| Time | 1 week | 3-5 days |

---

## 8. Next Steps (Pending Approval)

1. [ ] Download ViFiC dataset from Kaggle
2. [ ] Setup GLM-5 API credentials
3. [ ] Run auto-labeling pipeline
4. [ ] Train and evaluate PhoBERT
5. [ ] Write final report

---

## 9. References

1. Nguyen & Pham (2018). Search-based Sentiment and Stock Market Reactions: Vietnam Evidence
2. Ya et al. (2023). Forecasting ACB Stock Prices using ML and Vietnamese News Sentiment
3. Vu et al. (2023). Sentiments Extracted from News and Stock Market Reactions in Vietnam
4. Chen & Kawashima (2024). Stock Price Prediction Using LLM-Based Sentiment Analysis

---

*Plan created by Hermes Assistant*
*Last updated: March 30, 2026*
