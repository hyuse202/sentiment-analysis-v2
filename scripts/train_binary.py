#!/usr/bin/env python3
"""
Binary Sentiment Classification: POSITIVE vs NON-POSITIVE (NEGATIVE+NEUTRAL)
This approach merges NEUTRAL and NEGATIVE into one class to handle class imbalance.

Usage: python scripts/train_binary.py --input data/labeled/vific_labeled_1000_research.csv
"""

import os
import json
import argparse
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


# Binary label mapping: POSITIVE=1, NON-POSITIVE=0
# Rationale: In financial context, neutral news often has negative implications
LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 0, "POSITIVE": 1}
ID2LABEL = {0: "NON-POSITIVE", 1: "POSITIVE"}

# Vietnamese stopwords
VIETNAMESE_STOPWORDS = [
    'của', 'và', 'các', 'có', 'được', 'trong', 'với', 'cho', 'này', 'để',
    'tại', 'trên', 'từ', 'về', 'là', 'đến', 'như', 'khi', 'cũng', 'nhưng',
    'đã', 'đang', 'sẽ', 'mà', 'thì', 'nên', 'vẫn', 'rất', 'nhiều', 'hơn',
    'khác', 'phải', 'nếu', 'hay', 'hoặc', 'nhất', 'mỗi', 'ngay', 'ra', 'vào',
    'lại', 'đây', 'kia', 'đó', 'những', 'còn', 'không', 'trong', 'ngoài',
]

# Financial sentiment keywords
POSITIVE_KEYWORDS = [
    'tăng', 'tăng_trưởng', 'mạnh', 'khởi_sắc', 'tích_cực', 'đạt', 'vượt',
    'thắng_lợi', 'thành_công', 'lợi_nhuận', 'lời', 'tốt', 'hưởng_ứng',
    'hồi_phục', 'phát_triển', 'mở_rộng', 'đột_phá', 'tăng_trưởng'
]

NEGATIVE_KEYWORDS = [
    'giảm', 'sụt', 'tiêu_cực', 'rớt', 'thất_bại', 'lỗ', 'rủi_ro',
    'khó_khăn', 'suy_giảm', 'đình_trệ', 'ảnh_hưởng', 'chưa_đạt'
]


def load_data(input_path, remove_stopwords=True, use_lexicon_features=False):
    """Load labeled data from CSV and convert to binary labels"""
    df = pd.read_csv(input_path)
    df["text"] = df["title"] + ". " + df["content"]

    if remove_stopwords:
        def remove_sw(text):
            words = text.lower().split()
            return ' '.join([w for w in words if w not in VIETNAMESE_STOPWORDS])
        df["text"] = df["text"].apply(remove_sw)

    # Binary labels
    df["label_id"] = df["sentiment"].map(LABEL2ID)
    df = df[df["label_id"].notna()]

    # Add lexicon features if requested
    if use_lexicon_features:
        def count_keywords(text, keywords):
            count = 0
            for kw in keywords:
                count += text.lower().count(kw.replace('_', ' '))
            return count

        df["pos_count"] = df["text"].apply(lambda x: count_keywords(x, POSITIVE_KEYWORDS))
        df["neg_count"] = df["text"].apply(lambda x: count_keywords(x, NEGATIVE_KEYWORDS))
        df["sentiment_ratio"] = (df["pos_count"] - df["neg_count"]) / (df["pos_count"] + df["neg_count"] + 1)

    return df


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    if y_prob is not None:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NON-POSITIVE", "POSITIVE"],
        yticklabels=["NON-POSITIVE", "POSITIVE"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cm_{model_name.lower().replace(' ', '_')}.png")
    plt.close()


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, output_dir, use_smote=False):
    """Train and evaluate a single model"""
    print(f"\n{'=' * 50}")
    print(f"Training {model_name}...")

    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"  After SMOTE: {len(y_train_res)} samples (was {len(y_train)})")
    else:
        X_train_res, y_train_res = X_train, y_train

    # Train
    model.fit(X_train_res, y_train_res)

    # Predict
    y_pred = model.predict(X_test)

    # Get probabilities if available
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    if 'auc' in metrics:
        print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  CV F1 (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["NON-POSITIVE", "POSITIVE"]))

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name, output_dir)

    # Save model
    model_path = f"{output_dir}/{model_name.lower().replace(' ', '_')}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {
        "model_name": model_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics.get("auc", 0),
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std()
    }


def main():
    parser = argparse.ArgumentParser(description="Train binary sentiment classifier")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output_dir", type=str, default="results/binary", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_smote", action="store_true", help="Use SMOTE oversampling")

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    df = load_data(args.input, remove_stopwords=True, use_lexicon_features=False)
    texts = df["text"].tolist()
    labels = df["label_id"].astype(int).tolist()
    print(f"Total samples: {len(texts)}")

    # Show class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nBinary class distribution:")
    for c, count in zip(unique, counts):
        print(f"  {ID2LABEL[c]}: {count} ({100*count/len(labels):.1f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=args.seed, stratify=labels
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # TF-IDF Vectorization
    print("\nVectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Feature dimension: {X_train_tfidf.shape[1]}")

    # Save vectorizer
    with open(f"{args.output_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    all_results = []

    # Define models optimized for binary classification
    models = [
        (MultinomialNB(), "Naive Bayes"),
        (LogisticRegression(max_iter=1000, random_state=args.seed, class_weight='balanced', C=0.5), "Logistic Regression"),
        (SVC(kernel="linear", random_state=args.seed, class_weight='balanced', probability=True, C=0.5), "SVM (Linear)"),
        (SVC(kernel="rbf", random_state=args.seed, class_weight='balanced', probability=True, C=1.0, gamma='scale'), "SVM (RBF)"),
        (RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=args.seed,
            n_jobs=-1
        ), "Random Forest"),
        (xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=1.0,  # Will be adjusted
            random_state=42
        ), "XGBoost"),
        (GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=args.seed
        ), "Gradient Boosting"),
    ]

    # Adjust XGBoost scale_pos_weight for class imbalance
    pos_count = sum(y_train)
    neg_count = len(y_train) - pos_count
    for i, (model, name) in enumerate(models):
        if name == "XGBoost":
            models[i] = (xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective='binary:logistic',
                eval_metric='logloss',
                scale_pos_weight=neg_count / pos_count,
                random_state=42
            ), name)

    # Train and evaluate each model
    for model, name in models:
        result = train_and_evaluate_model(
            model, name, X_train_tfidf, X_test_tfidf, y_train, y_test,
            args.output_dir, use_smote=args.use_smote
        )
        all_results.append(result)

    # Create ensemble
    print("\n" + "=" * 50)
    print("Creating Ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=5,
                class_weight='balanced', random_state=args.seed, n_jobs=-1
            )),
            ('xgb', xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                objective='binary:logistic', eval_metric='logloss',
                scale_pos_weight=neg_count / pos_count, random_state=42
            )),
            ('lr', LogisticRegression(max_iter=1000, random_state=args.seed, class_weight='balanced', C=0.5))
        ],
        voting='soft'
    )
    ensemble_result = train_and_evaluate_model(
        ensemble, "Ensemble (RF+XGB+LR)", X_train_tfidf, X_test_tfidf,
        y_train, y_test, args.output_dir, use_smote=args.use_smote
    )
    all_results.append(ensemble_result)

    # Summary comparison
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<30} {'Accuracy':<10} {'F1':<10} {'AUC':<10} {'CV F1':<15}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model_name']:<30} {r['accuracy']:<10.4f} {r['f1']:<10.4f} {r['auc']:<10.4f} {r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}")

    # Find best model
    best_result = max(all_results, key=lambda x: x['cv_f1_mean'])
    print(f"\nBest model: {best_result['model_name']} with CV F1: {best_result['cv_f1_mean']:.4f}")
    print(f"Test Accuracy: {best_result['accuracy']:.2%}")

    # Check if target achieved
    if best_result['accuracy'] >= 0.80:
        print("\n✅ TARGET ACHIEVED: Accuracy >= 80%")
    else:
        print(f"\n❌ Target not yet achieved. Need {0.80 - best_result['accuracy']:.2%} more improvement")

    # Save results
    results_path = f"{args.output_dir}/binary_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "results": all_results,
            "timestamp": datetime.now().isoformat(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_dimension": X_train_tfidf.shape[1],
            "best_model": best_result['model_name'],
            "target_achieved": best_result['accuracy'] >= 0.80
        }, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
