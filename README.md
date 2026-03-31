# Vietnamese Stock Sentiment Analysis V2

## Overview

Project phân loại sentiment cho tin tức tài chính Việt Nam, sử dụng **GLM-5 API** để auto-label dữ liệu ViFiC, từ đó fine-tune **PhoBERT** cho bài toán sentiment classification.

## Key Features

- **Auto-labeling với LLM**: Tiết kiệm thời gian manual labeling
- **Large-scale data**: 160,490 articles từ ViFiC
- **Vietnamese-specific**: PhoBERT model chuyên cho tiếng Việt
- **Quality**: Overcome data leakage issue from previous project

## Quick Start

```bash
# 1. Download ViFiC
kaggle datasets download -d daddychillonkaggle/vietnamese-financial-corpus

# 2. Setup environment
pip install -r requirements.txt

# 3. Run auto-labeling
python scripts/03_auto_label.py

# 4. Train model
python scripts/04_train_model.py

# 5. Evaluate
python scripts/05_evaluate.py
```

## Project Structure

```
sentiment_analysis_v2/
├── papers/          # Reference papers (PDF + MD)
├── data/            # Dataset
├── scripts/         # Pipeline scripts
├── models/          # Trained models
├── results/         # Evaluation results
├── docs/            # Documentation
├── PROJECT_PLAN.md  # Detailed plan
└── README.md        # This file
```

## Timeline

- Day 1: Data preparation + Auto-labeling
- Day 2: Continue labeling + Start training
- Day 3: Evaluation + Analysis
- Day 4-5: Report writing

## Target Metrics

- Accuracy: 75-85%
- F1-Score: 0.70+

## Contact

- Owner: Hyuse (Tấn Huy)
- Assistant: Hermes
