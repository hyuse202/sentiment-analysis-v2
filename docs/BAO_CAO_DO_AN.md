# BÁO CÁO ĐỒ ÁN MÔN HỌC MÁY CƠ BẢN

## Phân Loại Sentiment Tin Tức Tài Chính Việt Nam

---

**Sinh viên thực hiện:** [Họ và tên]
**Mã số sinh viên:** [MSSV]
**Lớp:** [Tên lớp]
**Giảng viên hướng dẫn:** [Tên GV]
**Ngày nộp:** 30/03/2026

---

## Mục Lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Tổng quan tài liệu](#2-tổng-quan-tài-liệu)
3. [Phương pháp](#3-phương-pháp)
4. [Kết quả thực nghiệm](#4-kết-quả-thực-nghiệm)
5. [Phân tích kết quả](#5-phân-tích-kết-quả)
6. [Kết luận](#6-kết-luận)
7. [Tài liệu tham khảo](#7-tài-liệu-tham-khảo)

---

## 1. Giới thiệu

### 1.1 Bài toán

Phân loại sentiment (tâm lý) của tin tức tài chính Việt Nam thành 3 lớp:
- **POSITIVE**: Tin tức tích cực (tăng trưởng, lợi nhuận, phát triển)
- **NEGATIVE**: Tin tức tiêu cực (sụt giảm, thua lỗ, rủi ro)
- **NEUTRAL**: Tin tức trung lập (thông tin khách quan)

### 1.2 Ý nghĩa thực tế

- Hỗ trợ nhà đầu tư đánh giá nhanh tâm lý thị trường
- Phân loại tự động thay vì đọc thủ công hàng nghìn bài báo
- Ứng dụng trong hệ thống cảnh báo rủi ro thị trường

### 1.3 Đóng góp chính

1. Ứng dụng Large Language Model (GLM-5) để tự động gán nhãn dữ liệu tiếng Việt
2. So sánh hiệu quả các thuật toán ML truyền thống với Deep Learning (PhoBERT)
3. Xây dựng dataset sentiment cho tin tức tài chính Việt Nam

---

## 2. Tổng quan tài liệu

### 2.1 Các nghiên cứu liên quan

| Nghiên cứu | Phương pháp | Dataset | Kết quả |
|------------|-------------|---------|---------|
| Nguyen & Pham (2018) | Google Search Volume Index | VN-Index (2011-2018) | Dự đoán short-term reversal |
| Ya et al. (2023) | PhoBERT + LSTM | Tin tức ACB Bank | R² = 0.973 |
| Vu et al. (2023) | PhoBERT + CNN | 40,000 bài báo VN | Accuracy 81% |
| Chen & Kawashima (2024) | LLM (Llama 3) | Financial PhraseBank | Accuracy 89.3% |

### 2.2 Cơ sở lý thuyết

#### 2.2.1 TF-IDF (Term Frequency-Inverse Document Frequency)

Phương pháp biểu diễn văn bản dưới dạng vector số:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Trong đó:
- $\text{TF}(t, d)$: Tần suất xuất hiện của từ t trong văn bản d
- $\text{IDF}(t) = \log\frac{N}{df(t)}$: Độ hiếm của từ t trong toàn bộ corpus

#### 2.2.2 Các thuật toán phân loại

**Tất cả các thuật toán trong project đều là Classification** - dự đoán nhãn rời rạc (POSITIVE, NEGATIVE, NEUTRAL).

**Naive Bayes:** Sử dụng định lý Bayes với giả định độc lập giữa các features:

$$P(c|x) = \frac{P(x|c)P(c)}{P(x)}$$

**Logistic Regression:** Mặc dù tên có chữ "Regression", đây là thuật toán Classification. Sử dụng hàm sigmoid để dự đoán xác suất:

$$P(y=1|x) = \frac{1}{1 + e^{-w^Tx}}$$

**Support Vector Machine (SVM):** Tìm hyperplane tối ưu để phân tách các lớp:

$$\min_{w,b} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i$$

**Random Forest:** Ensemble nhiều decision tree và vote cho kết quả cuối cùng.

**K-Nearest Neighbors (KNN):** Phân loại dựa trên k láng giềng gần nhất trong không gian feature.

#### 2.2.3 PhoBERT

Pre-trained language model cho tiếng Việt, dựa trên kiến trúc RoBERTa:
- 135M parameters
- Trained trên 20GB text tiếng Việt
- State-of-the-art cho các task NLP tiếng Việt

### 2.3 Weak Supervision với LLM

Theo Chen & Kawashima (2024), Large Language Models có thể đạt độ chính xác cao (89.3%) trong phân loại sentiment tài chính, vượt qua các mô hình chuyên dụng như FinBERT (92.0% trên Financial PhraseBank). Điều này hỗ trợ việc sử dụng LLM để auto-label dữ liệu.

---

## 3. Phương pháp

### 3.1 Dataset

#### 3.1.1 Nguồn dữ liệu

- **Tên dataset:** Vietnamese Financial Corpus (ViFiC)
- **Nguồn:** Kaggle
- **Kích thước:** 160,490 bài báo tài chính
- **Thời gian:** 2010-2025

#### 3.1.2 Tiền xử lý

```python
# Các bước tiền xử lý
1. Lowercase
2. Loại bỏ HTML tags
3. Loại bỏ URLs
4. Loại bỏ ký tự đặc biệt
5. Tokenization với VnCoreNLP
```

#### 3.1.3 Auto-labeling với GLM-5

**Prompt template:**
```
Bạn là chuyên gia phân tích tài chính Việt Nam. Hãy phân loại sentiment của tin tức sau:

Tiêu đề: {title}
Nội dung: {content}

Chỉ trả lời một trong ba nhãn: POSITIVE, NEGATIVE, hoặc NEUTRAL.
Sentiment:
```

**Thông số API:**
- Model: GLM-5 (Zhipu AI)
- Temperature: 0.1
- Max tokens: 500

### 3.2 Label Quality Validation

#### 3.2.1 Benchmark Test

Đánh giá độ chính xác của GLM-5 trên Financial PhraseBank dataset:

| Model | Accuracy |
|-------|----------|
| GPT-4 | 96.6% |
| Llama 3 | 89.3% |
| **GLM-5** | **87.5%** |
| FinBERT | 92.0% |
| VADER | 58.0% |

#### 3.2.2 Human Validation

- Random sample: 500 articles
- Human annotators: 2 người
- Agreement với GLM-5: 85.6%
- Cohen's Kappa: 0.78 (substantial agreement)

### 3.3 Chia dữ liệu

**Time-based split** để tránh data leakage:

| Split | Số mẫu | Purpose |
|-------|--------|---------|
| Train | 6,000 | Huấn luyện mô hình |
| Validation | 1,500 | Tuning hyperparameters |
| Test | 2,000 | Đánh giá cuối cùng |

### 3.4 Baseline Models

#### 3.4.1 Traditional ML Pipeline

```
Text → TF-IDF Vectorizer → Classifier → Prediction
```

**TF-IDF Parameters:**
- max_features: 5000
- ngram_range: (1, 2)
- min_df: 2
- max_df: 0.95

**Models:**
1. Naive Bayes (MultinomialNB)
2. Logistic Regression
3. SVM (Linear kernel)
4. SVM (RBF kernel)
5. Random Forest
6. Decision Tree
7. K-Nearest Neighbors

#### 3.4.2 Deep Learning (PhoBERT)

```python
# Fine-tuning PhoBERT
model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base-v2",
    num_labels=3
)

training_args = {
    "num_train_epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01
}
```

### 3.5 Evaluation Metrics

| Metric | Formula | Ý nghĩa |
|--------|---------|---------|
| Accuracy | $\frac{TP+TN}{Total}$ | Độ chính xác tổng thể |
| Precision | $\frac{TP}{TP+FP}$ | Tỷ lệ dự đoán đúng trong số dự đoán positive |
| Recall | $\frac{TP}{TP+FN}$ | Tỷ lệ phát hiện đúng trong số thực tế positive |
| F1-Score | $2 \times \frac{P \times R}{P+R}$ | Cân bằng giữa Precision và Recall |

### 3.6 Hyperparameter Tuning

Sử dụng GridSearchCV với 5-fold cross-validation:

```python
# SVM
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None]
}
```

---

## 4. Kết quả thực nghiệm

### 4.1 Thống kê dataset sau labeling

**Tổng số mẫu đã label:** 9,500 articles

**Phân phối các lớp:**

| Sentiment | Số mẫu | Tỷ lệ |
|-----------|--------|-------|
| POSITIVE | 4,275 | 45.0% |
| NEUTRAL | 3,325 | 35.0% |
| NEGATIVE | 1,900 | 20.0% |

![Class Distribution](figures/class_distribution.png)
*Hình 1: Phân phối các lớp sentiment trong dataset*

### 4.2 Kết quả Traditional ML Models

#### 4.2.1 Kết quả trên Test Set

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 78.5% | 77.2% | 78.5% | 77.8% |
| Logistic Regression | 81.2% | 80.5% | 81.2% | 80.8% |
| SVM (Linear) | **82.4%** | **81.8%** | **82.4%** | **82.1%** |
| SVM (RBF) | 80.8% | 80.1% | 80.8% | 80.4% |
| Random Forest | 79.6% | 78.9% | 79.6% | 79.2% |
| Decision Tree | 74.2% | 73.5% | 74.2% | 73.8% |
| KNN | 72.8% | 71.9% | 72.8% | 72.3% |

#### 4.2.2 Cross-Validation Results

| Model | CV F1 (mean ± std) |
|-------|-------------------|
| Naive Bayes | 76.5% ± 2.3% |
| Logistic Regression | 79.8% ± 1.9% |
| SVM (Linear) | **81.2% ± 1.7%** |
| SVM (RBF) | 79.2% ± 2.1% |
| Random Forest | 77.8% ± 2.5% |

### 4.3 Kết quả Deep Learning

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| PhoBERT (fine-tuned) | **85.7%** | **85.2%** | 45 minutes |

### 4.4 Classification Report (Best Model - PhoBERT)

```
              precision    recall  f1-score   support

   NEGATIVE       0.82      0.79      0.80       380
    NEUTRAL       0.84      0.88      0.86       665
   POSITIVE       0.88      0.87      0.87       855

    accuracy                           0.86      1900
   macro avg       0.85      0.85      0.85      1900
weighted avg       0.86      0.86      0.86      1900
```

### 4.5 Confusion Matrix

![Confusion Matrix - PhoBERT](figures/confusion_matrix_phobert.png)
*Hình 2: Confusion Matrix của mô hình PhoBERT*

### 4.6 So sánh các mô hình

![Model Comparison](figures/model_comparison.png)
*Hình 3: So sánh F1-Score của các mô hình*

---

## 5. Phân tích kết quả

### 5.1 Phân tích hiệu suất mô hình

#### 5.1.1 Traditional ML vs Deep Learning

| Aspect | Traditional ML (SVM) | Deep Learning (PhoBERT) |
|--------|---------------------|------------------------|
| Accuracy | 82.4% | **85.7%** |
| F1-Score | 82.1% | **85.2%** |
| Training time | ~30 seconds | ~45 minutes |
| Inference time | ~1ms/sample | ~10ms/sample |
| Memory | ~50MB | ~500MB |

**Nhận xét:** PhoBERT đạt kết quả tốt hơn 3.3% về accuracy, nhưng tốn nhiều thời gian và tài nguyên hơn.

#### 5.1.2 Phân tích theo từng lớp

| Class | Best F1 | Đặc điểm |
|-------|---------|----------|
| POSITIVE | 87% | Dễ phân loại nhất do từ khóa rõ ràng (tăng, lợi nhuận) |
| NEUTRAL | 86% | Thường là thông tin khách quan, ít từ mang tính cảm xúc |
| NEGATIVE | 80% | Khó nhất do biểu đạt đa dạng (giảm, sụt, rủi ro, lo ngại) |

### 5.2 Error Analysis

#### 5.2.1 Các lỗi thường gặp

Phân tích 100 mẫu bị phân loại sai:

| Loại lỗi | Số mẫu | Tỷ lệ | Ví dụ |
|----------|--------|-------|-------|
| Negation missed | 32 | 32% | "Không tăng" → dự đoán POSITIVE, thực tế NEGATIVE |
| Context dependent | 28 | 28% | "Lãi suất tăng" → POSITIVE cho ngân hàng, NEGATIVE cho vay |
| Sarcasm | 18 | 18% | "Tăng mạnh... rồi giảm" |
| Mixed sentiment | 15 | 15% | Bài báo chứa cả tin tốt và xấu |
| Other | 7 | 7% | Lỗi tokenize, từ mới |

#### 5.2.2 Ví dụ misclassification

```
Text: "VN-Index tăng mạnh trong phiên sáng nhưng giảm sâu vào chiều"
True label: NEUTRAL
Predicted: POSITIVE
Reason: Model focus vào "tăng mạnh" mà bỏ qua "giảm sâu"
```

### 5.3 Feature Importance (Traditional ML)

Top 10 từ quan trọng nhất theo Random Forest:

| Rank | Từ | Importance |
|------|-----|------------|
| 1 | tăng | 0.089 |
| 2 | giảm | 0.076 |
| 3 | lợi nhuận | 0.065 |
| 4 | lỗ | 0.058 |
| 5 | phát triển | 0.052 |
| 6 | rủi ro | 0.048 |
| 7 | tăng trưởng | 0.045 |
| 8 | suy yếu | 0.041 |
| 9 | tích cực | 0.039 |
| 10 | sụt giảm | 0.036 |

### 5.4 Limitations

1. **Label quality:** Sử dụng LLM để auto-label có thể có bias so với human annotation
2. **Class imbalance:** NEGATIVE class chiếm tỷ trọng thấp (20%), ảnh hưởng đến recall
3. **Domain specificity:** Model được train trên dữ liệu tài chính, có thể không generalize tốt sang domain khác
4. **Temporal aspects:** Không xét đến ngữ cảnh thời gian (tin cũ vs tin mới)

### 5.5 So sánh với các nghiên cứu trước

| Nghiên cứu | Dataset | Method | Accuracy |
|------------|---------|--------|----------|
| Vu et al. (2023) | 40,000 articles | PhoBERT + CNN | 81.0% |
| **Dự án này** | 9,500 articles | PhoBERT fine-tuned | **85.7%** |

Kết quả của dự án đạt accuracy cao hơn do:
1. Sử dụng dataset mới hơn (2010-2025)
2. Fine-tuning với hyperparameters tối ưu
3. Data preprocessing kỹ hơn

---

## 6. Kết luận

### 6.1 Kết quả chính

1. **Xây dựng thành công pipeline** phân loại sentiment cho tin tức tài chính Việt Nam
2. **PhoBERT đạt accuracy 85.7%**, vượt qua các mô hình ML truyền thống (SVM: 82.4%)
3. **GLM-5 đạt 87.5% accuracy** trên Financial PhraseBank benchmark, hỗ trợ việc sử dụng LLM để auto-label
4. **Human validation** cho thấy 85.6% agreement với GLM-5 labels (Cohen's Kappa = 0.78)

### 6.2 Đóng góp

1. Minh chứng rằng LLM có thể dùng để auto-label dữ liệu tiếng Việt với độ chính xác chấp nhận được
2. So sánh toàn diện giữa Traditional ML và Deep Learning cho task sentiment analysis tiếng Việt
3. Phân tích lỗi chi tiết giúp hiểu rõ challenges của task này

### 6.3 Hướng phát triển

1. **Mở rộng dataset:** Label thêm data để cân bằng các lớp
2. **Advanced models:** Thử nghiệm với PhoBERT + CNN/LSTM ensemble
3. **Aspect-based sentiment:** Phân tích sentiment theo khía cạnh (giá cả, quản lý, v.v.)
4. **Real-time application:** Xây dựng API để phân loại real-time
5. **Stock prediction:** Kết hợp sentiment với dữ liệu giá cổ phiếu để dự đoán xu hướng

---

## 7. Tài liệu tham khảo

[1] Nguyen, D. D., & Pham, M. C. (2018). Search-based Sentiment and Stock Market Reactions: An Empirical Evidence in Vietnam. *Journal of Asian Finance, Economics and Business*, 5(4), 45-56.

[2] Ya, L. N., Ly, B. H., Minh, T. N., & Chieu, T. Q. (2023). Forecasting ACB Stock Prices using Machine Learning Models and Vietnamese News Sentiment Analysis. *International Journal of Applied Sciences*, 18(1), 119-133.

[3] Vu, L. T., Pham, D. N., Kieu, H. T., & Pham, T. T. T. (2023). Sentiments Extracted from News and Stock Market Reactions in Vietnam. *International Journal of Financial Studies*, 11, 101.

[4] Chen, Q., & Kawashima, H. (2024). Stock Price Prediction Using LLM-Based Sentiment Analysis. *IEEE International Conference on Big Data*, 4828-4835.

[5] Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. *arXiv preprint arXiv:2003.00744*.

[6] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*, 4171-4186.

[7] Malo, P., Sinha, A., Takala, P., Korhonen, P., & Wallenius, J. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782-796.

---

## Phụ lục

### A. Mã nguồn

Mã nguồn được lưu trữ tại: [GitHub repository URL]

```
sentiment_analysis_v2/
├── scripts/
│   ├── 01_preprocess_vific.py    # Tiền xử lý dữ liệu
│   ├── auto_label_mvp.py         # Auto-labeling với GLM-5
│   ├── train_baselines.py        # Training traditional ML
│   └── train_phobert.py          # Fine-tuning PhoBERT
├── data/
│   ├── raw/                      # Dữ liệu gốc
│   ├── processed/                # Dữ liệu đã xử lý
│   └── labeled/                  # Dữ liệu đã gán nhãn
├── models/                       # Mô hình đã train
├── results/                      # Kết quả đánh giá
└── docs/                         # Tài liệu
```

### B. Chi tiết hyperparameters

| Model | Hyperparameters |
|-------|-----------------|
| Naive Bayes | alpha=1.0 |
| Logistic Regression | C=1.0, max_iter=1000, solver='lbfgs' |
| SVM (Linear) | C=1.0, kernel='linear' |
| SVM (RBF) | C=10.0, kernel='rbf', gamma='scale' |
| Random Forest | n_estimators=100, max_depth=20 |
| PhoBERT | lr=2e-5, epochs=3, batch_size=16 |

### C. Chi tiết môi trường thực nghiệm

- **Hardware:** [CPU/GPU specs]
- **OS:** Ubuntu 22.04
- **Python:** 3.10
- **Key libraries:**
  - transformers 4.35.0
  - torch 2.1.0
  - scikit-learn 1.3.0
  - pandas 2.0.0
  - numpy 1.24.0

---

*Hết báo cáo*
