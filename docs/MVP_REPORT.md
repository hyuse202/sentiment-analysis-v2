# Vietnamese Financial Sentiment Analysis - MVP Report

**Date:** March 30, 2026
**Author:** Hyuse (Tấn Huy)

---

## 1. Project Overview

### 1.1 Objective
Xây dựng hệ thống phân tích sentiment cho tin tức tài chính Việt Nam sử dụng:
- **Auto-labeling:** GLM-5 API (Zhipu AI)
- **Training:** Traditional ML (SVM, RF, LR, NB) + Deep Learning (PhoBERT)

### 1.2 Key Findings from Paper (Chen & Kawashima, 2024)
| Model | Financial Phrasebank Accuracy |
|-------|------------------------------|
| GPT-4 | 96.6% |
| Llama 3 | 89.3% |
| FinBERT | 92.0% |
| VADER | 58.0% |

**Insight:** LLMs outperform traditional models on financial sentiment, especially for domain-specific terminology.

---

## 2. Technical Implementation

### 2.1 GLM-5 API Integration

**Discovery:** GLM-5 is a **reasoning model** (similar to DeepSeek):
- `content`: Final answer
- `reasoning_content`: Chain-of-thought reasoning
- Requires `max_tokens >= 500` for complete output

```python
from zai import ZaiClient

client = ZaiClient(api_key=os.getenv("ZAI_API_KEY"))
response = client.chat.completions.create(
    model="glm-5",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=500,  # Critical: needs enough tokens for reasoning
    temperature=0.1
)
label = response.choices[0].message.content  # Final answer
```

### 2.2 Prompt Template

```
Bạn là chuyên gia phân tích tài chính Việt Nam. Hãy phân loại sentiment của tin tức sau:

Tiêu đề: {title}
Nội dung: {content}

Chỉ trả lời một trong ba nhãn: POSITIVE, NEGATIVE, hoặc NEUTRAL.
Sentiment:
```

### 2.3 Labeling Performance

| Metric | Value |
|--------|-------|
| Articles labeled | 100 |
| Time per article | ~15 seconds |
| Total time | ~25 minutes |
| Cost estimate | ~$0.10 (500 tokens × 100 articles) |

**Sentiment Distribution:**
```
POSITIVE: 50 (50%)
NEUTRAL:  34 (34%)
NEGATIVE: 16 (16%)
```

---

## 3. Model Training Results

### 3.1 Traditional ML Models (TF-IDF + Classifier)

| Model | Accuracy | F1 | CV F1 (mean ± std) |
|-------|----------|-----|-------------------|
| Naive Bayes | 86.67% | 85.75% | 74.80% ± 6.73% |
| Logistic Regression | 86.67% | 85.75% | 76.60% ± 6.94% |
| SVM (Linear) | 86.67% | 85.75% | 76.60% ± 6.94% |
| SVM (RBF) | 86.67% | 85.75% | 76.60% ± 6.94% |
| Random Forest | 86.67% | 85.75% | 76.60% ± 6.94% |

### 3.2 Deep Learning Model

| Model | Accuracy | F1 | Epochs |
|-------|----------|-----|--------|
| PhoBERT (fine-tuned) | 66.67% | 59.39% | 5 |

### 3.3 Classification Report (PhoBERT)

```
              precision    recall  f1-score   support

    NEGATIVE       0.00      0.00      0.00         3
     NEUTRAL       1.00      0.60      0.75         5
    POSITIVE       0.58      1.00      0.74         7

    accuracy                           0.67        15
   macro avg       0.53      0.53      0.50        15
weighted avg       0.61      0.67      0.59        15
```

---

## 4. Overfitting Analysis

### 4.1 Root Cause

**Data Duplication Issue:**
```python
# In create_sample_data():
for i in range(n_samples):
    article = sample_articles[i % len(sample_articles)]  # i % 10 → duplicates!
```

- 10 unique articles × 10 = 100 "samples"
- Train and test contain nearly identical data
- Model "memorizes" instead of learning

### 4.2 Evidence of Overfitting

| Indicator | Value | Interpretation |
|-----------|-------|----------------|
| CV F1 std | 6.73% | High variance |
| NEGATIVE recall | 0% | Model fails on minority class |
| Train vs Test | Similar | Data leakage |
| All models same accuracy | 86.67% | Suspicious uniformity |

### 4.3 Solutions

| Solution | Implementation |
|----------|---------------|
| **Use real data** | Download ViFiC from Kaggle |
| **Stratified split** | `train_test_split(..., stratify=labels)` |
| **Regularization** | SVM(C=0.1), LR(penalty='l2') |
| **Class balance** | SMOTE, class_weight='balanced' |
| **Cross-validation** | StratifiedKFold with unique samples |
| **More data** | At least 1000+ unique samples |

---

## 5. Files Created

```
sentiment_analysis_v2/
├── .env                          # API keys
├── requirements.txt              # Dependencies
├── scripts/
│   ├── auto_label_mvp.py         # GLM-5 labeling
│   ├── train_phobert.py          # PhoBERT training
│   └── train_baselines.py        # Traditional ML
├── data/
│   ├── raw/sample_100.csv        # Raw samples
│   └── labeled/
│       ├── labeled_10.csv        # 10 samples test
│       └── labeled_100.csv       # 100 samples
├── models/
│   └── phobert_sentiment/        # Trained model
├── results/
│   └── baselines/
│       ├── baseline_results.json
│       └── cm_*.png              # Confusion matrices
└── docs/
    ├── GLM5_API_NOTES.md
    └── MVP_REPORT.md             # This file
```

---

## 6. Commands Reference

```bash
# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Label data with GLM-5
python scripts/auto_label_mvp.py --sample 100 --delay 0.5

# Train traditional ML
python scripts/train_baselines.py --input data/labeled/labeled_100.csv

# Train PhoBERT
python scripts/train_phobert.py --input data/labeled/labeled_100.csv --epochs 5
```

---

## 7. Key Learnings

1. **GLM-5 is a reasoning model** - needs `max_tokens >= 500`
2. **Data quality > quantity** - duplicated data causes overfitting
3. **Small datasets favor traditional ML** - PhoBERT needs 1000+ samples
4. **Class imbalance matters** - NEGATIVE class severely underrepresented
5. **Cross-validation std reveals overfitting** - high std = unstable model

---

## 8. Next Steps

See: `PLAN_REAL_DATA.md` for implementation with ViFiC dataset

