# Plan: Vietnamese Financial Sentiment Analysis with Real Data (ViFiC)

## 1. Context

### Current Issues
- **Overfitting:** Sample data duplicated (10 unique × 10 = 100)
- **Small dataset:** Only 100 samples
- **Class imbalance:** NEGATIVE only 16%

### Solution
Use **ViFiC (Vietnamese Financial Corpus)** - 160,490 real articles from Kaggle

---

## 2. Data Pipeline

### Phase 1: Download ViFiC (5 minutes)

```bash
# Download from Kaggle
kaggle datasets download -d daddychillonkaggle/vietnamese-financial-corpus

# Unzip
unzip vietnamese-financial-corpus.zip -d data/vific/

# Check structure
ls -la data/vific/
```

### Phase 2: Data Preprocessing (10 minutes)

**Script:** `scripts/01_preprocess_vific.py`

```python
# Tasks:
# 1. Load ViFiC CSV
# 2. Filter relevant categories (stock, finance, macro)
# 3. Remove duplicates
# 4. Clean text (HTML, URLs, special chars)
# 5. Sample 1000-5000 articles for MVP
# 6. Save to data/processed/
```

### Phase 3: Auto-Labeling with GLM-5 (2-4 hours)

**Script:** `scripts/02_label_vific.py`

| Batch Size | Articles | Est. Time |
|------------|----------|-----------|
| Small MVP | 500 | ~2 hours |
| Medium | 1,000 | ~4 hours |
| Large | 5,000 | ~20 hours |

**Optimizations:**
- Batch processing with rate limiting
- Resume from checkpoint (save every 100 articles)
- Parallel requests (if API allows)

### Phase 4: Quality Validation (30-60 minutes) ⚠️ BẮT BUỘC

**Justification for LLM Labeling:**
> Theo paper "LLM Sentiment Stock Prediction" (Chen & Kawashima, 2024), GPT-4 đạt 96.6% accuracy trên Financial Phrasebank, cao hơn FinBERT (92%). Điều này cho thấy LLM đủ tin cậy để label financial sentiment. Tuy nhiên, cần validate vì:
> 1. GLM-5 có thể khác GPT-4
> 2. Vietnamese financial text khác English
> 3. Domain-specific terminology cần kiểm tra

**Script:** `scripts/03_validate_labels.py`

**Step 1: Random Sampling (5 min)**
```python
# Random sample 100 articles from labeled data
# Export to CSV for manual labeling
# Format: id, title, content, glm_label, human_label (empty)
```

**Step 2: Manual Labeling (30-45 min)**
- Label 100 samples manually (hoặc nhờ bạn bè)
- Chỉ cần label: POSITIVE, NEGATIVE, NEUTRAL
- Save to `data/validation/human_labels.csv`

**Step 3: Agreement Analysis (5 min)**
```python
# Calculate metrics:
# - Overall Agreement Rate: % matching labels
# - Cohen's Kappa: Inter-annotator agreement
# - Per-class accuracy: POSITIVE, NEGATIVE, NEUTRAL

# Thresholds:
# - Agreement >= 80%: Acceptable
# - Agreement < 70%: Review prompt, may need re-labeling
# - NEGATIVE class often has lowest agreement (imbalanced)
```

**Step 4: Confidence Filtering (5 min)**
```python
# If GLM-5 returns confidence/probability:
# - Remove labels with confidence < 0.7
# - Or keep all if confidence not available
```

**Output:**
- `data/validation/validation_report.json`
- Agreement rate, Kappa score, per-class metrics
- List of disputed labels for review

### Phase 5: Model Training (30 minutes)

```bash
# Traditional ML (fast)
python scripts/train_baselines.py --input data/labeled/vific_1000.csv

# PhoBERT (slower, needs GPU recommended)
python scripts/train_phobert.py --input data/labeled/vific_1000.csv --epochs 3
```

### Phase 6: Create Vietnamese Gold Test Set (30 min) ⚠️ QUAN TRỌNG

**Why not Financial Phrasebank?**
- Financial Phrasebank = English → Không phù hợp với Vietnamese project
- PhoBERT = Vietnamese model → Cần Vietnamese test data
- GLM-5 vs GPT-4 comparison không phải mục tiêu của môn

**Approach:**
```python
# 1. Reserve 200 articles from labeled data (never used in training)
# 2. OR manually label 200 NEW articles (best for evaluation)
# 3. This becomes the "gold standard" test set
# 4. All models evaluated on this same test set
```

**Output:** `data/gold_test/vietnamese_gold_200.csv`

### Phase 7: Model Training (30 minutes)

**Updated scripts with anti-overfitting measures:**

```bash
# Traditional ML (fast, with class_weight='balanced')
python scripts/train_baselines.py --input data/labeled/vific_1000.csv --balanced

# PhoBERT (with early stopping)
python scripts/train_phobert.py --input data/labeled/vific_1000.csv --epochs 5
```

### Phase 8: Final Evaluation (15 minutes)

**Evaluate on Gold Test Set:**
- All models tested on same Vietnamese gold test set
- Compare: Random, Majority, TF-IDF+ML, PhoBERT
- Metrics: Accuracy, F1, Precision, Recall, Confusion Matrix
- Per-class analysis (especially NEGATIVE class)

---

## 3. Files to Create

| File | Purpose |
|------|---------|
| `scripts/01_preprocess_vific.py` | Load & clean ViFiC |
| `scripts/02_label_vific.py` | Batch labeling with checkpoint |
| `scripts/03_validate_labels.py` | Quality validation (agreement analysis) |
| `scripts/04_create_gold_test.py` | Create Vietnamese gold test set |
| `scripts/05_compare_models.py` | Full model comparison on gold test |
| `data/validation/human_labels.csv` | Manual labels for validation |
| `data/gold_test/vietnamese_gold_200.csv` | Gold standard test set |

---

## 4. Anti-Overfitting Measures

```python
# 1. Stratified split
train_test_split(..., stratify=labels, test_size=0.2)

# 2. Cross-validation
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Class weights
RandomForestClassifier(class_weight='balanced')
SVC(class_weight='balanced')

# 4. Regularization
LogisticRegression(C=0.1, penalty='l2')
SVC(C=0.1)

# 5. Early stopping (PhoBERT)
EarlyStoppingCallback(early_stopping_patience=2)
```

---

## 5. Expected Results (Based on Paper)

**Note on ViFiC Data Distribution:**
ViFiC là tin tức tài chính thực tế từ báo chí Việt Nam. Khác với sample data nhân tạo:
- Phần lớn là **factual reporting** → NEUTRAL
- POSITIVE/NEGATIVE ít hơn nhưng có giá trị quan trọng
- Distribution thực tế có thể: 60% NEUTRAL, 25% POSITIVE, 15% NEGATIVE

| Model | Expected Accuracy |
|-------|-------------------|
| Random Baseline | ~33% |
| Majority Class | ~50-60% (NEUTRAL dominant) |
| TF-IDF + SVM | ~70-75% |
| PhoBERT (fine-tuned) | **75-80%** |

**Prompt Engineering Notes:**
- Prompt v1: Quá an toàn, toàn NEUTRAL
- Prompt v2: Focus vào "Tác động kinh tế" thay vì ngữ pháp bề mặt
- Cần test với sample lớn hơn để xác nhận distribution

---

## 6. Timeline Summary

| Phase | Time | Note |
|-------|------|------|
| Download ViFiC | 5 min | |
| Preprocessing | 10 min | |
| Labeling 1000 samples | 4 hours | Can run overnight |
| **Validation** | **30-60 min** | ⚠️ Manual work needed |
| **Create Gold Test** | **30 min** | Reserve 200 samples |
| Training | 30 min | |
| Evaluation | 15 min | |
| **Total** | **~6 hours** | |

**Note:** Manual labeling for validation (100 samples) can be done while API labeling runs.

---

## 7. Commands Quick Reference

```bash
# 1. Download data
kaggle datasets download -d daddychillonkaggle/vietnamese-financial-corpus
unzip vietnamese-financial-corpus.zip -d data/vific/

# 2. Preprocess
python scripts/01_preprocess_vific.py --input data/vific/ --output data/processed/ --sample 1000

# 3. Label (takes ~4 hours)
python scripts/02_label_vific.py --input data/processed/vific_sample.csv --output data/labeled/vific_1000.csv

# 4. Validate labels (requires manual labeling first)
python scripts/03_validate_labels.py --glm data/labeled/vific_1000.csv --human data/validation/human_labels.csv

# 5. Create gold test set
python scripts/04_create_gold_test.py --input data/labeled/vific_1000.csv --output data/gold_test/ --size 200

# 6. Train
python scripts/train_baselines.py --input data/labeled/vific_1000.csv --balanced
python scripts/train_phobert.py --input data/labeled/vific_1000.csv --epochs 5

# 7. Evaluate on gold test
python scripts/05_compare_models.py --gold data/gold_test/vietnamese_gold_200.csv
```

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| API rate limits | Batch processing, delay between requests |
| Poor label quality | Manual validation, confidence threshold |
| Class imbalance | SMOTE, class weights, stratified sampling |
| Long labeling time | Resume from checkpoint, parallel processing |

---

## 9. Justification for LLM Labeling (Để trả lời thầy)

**Q: Tại sao dùng LLM để label thay vì human annotation?**

| Lý do | Giải thích |
|-------|------------|
| **Scale** | ViFiC có 160K articles → Human label tốn tháng, LLM tốn giờ |
| **Cost** | Human: ~$0.10/article → $16K cho 160K. LLM: ~$0.001/article → $160 |
| **Research backing** | Paper Chen & Kawashima (2024) chứng minh GPT-4 đạt 96.6% accuracy, cao hơn FinBERT (92%) |
| **Industry standard** | Many companies use LLM for data labeling (Scale AI, Labelbox) |

**Q: Làm sao đảm bảo label chất lượng?**

1. **Validation:** Manual check 100 samples, tính Agreement Rate
2. **Threshold:** Chỉ giữ labels có confidence >= 70%
3. **Domain expertise:** Prompt được thiết kế cho financial sentiment
4. **Gold test set:** Tạo test set riêng để đánh giá model cuối cùng

**Q: GLM-5 có tin cậy không?**

- GLM-5 là model của Zhipu AI, comparable với GPT-4
- Paper reference: GLM-4 achieves competitive results on various benchmarks
- Validation step sẽ verify chất lượng thực tế

---

## 10. What to Present to Teacher

**Slide structure:**
1. **Problem:** Vietnamese financial sentiment analysis
2. **Data:** ViFiC (160K articles), sample 1000 for training
3. **Method:** GLM-5 auto-labeling + Validation
4. **Models:** Traditional ML (SVM, RF, LR, NB) + PhoBERT
5. **Evaluation:** Vietnamese gold test set (200 samples)
6. **Results:** Compare models, show validation metrics
7. **Limitations:** LLM labeling, small test set
8. **Future work:** Human annotation, larger dataset

