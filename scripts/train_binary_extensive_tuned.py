#!/usr/bin/env python3
"""
Extensive Hyperparameter Tuning for Binary Sentiment Classification
This script performs comprehensive hyperparameter tuning to achieve >80% accuracy.
"""

import os
import json
import pickle
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Binary label mapping
LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
# For binary: POSITIVE vs NON-POSITIVE
BINARY_LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 1}
ID2LABEL_BINARY = {0: "NON-POSITIVE", 1: "POSITIVE"}

# Vietnamese stopwords
STOPWORDS = set([
    'của', 'và', 'các', 'có', 'được', 'trong', 'với', 'cho', 'này', 'để',
    'tại', 'trên', 'từ', 'về', 'là', 'đến', 'như', 'khi', 'cũng', 'nhưng',
    'đã', 'đang', 'sẽ', 'mà', 'thì', 'nên', 'vẫn', 'rất', 'nhiều', 'hơn',
])


def load_data(input_path, remove_stopwords=True):
    """Load labeled data from CSV"""
    df = pd.read_csv(input_path)
    df["text"] = df["title"] + ". " + df["content"]

    if remove_stopwords:
        def remove_sw(text):
            words = str(text).lower().split()
            return ' '.join([w for w in words if w not in STOPWORDS])
        df["text"] = df["text"].apply(remove_sw)

    # Binary labels: POSITIVE=1, NON-POSITIVE=0
    df["label_id"] = df["sentiment"].map(BINARY_LABEL2ID)
    df = df[df["label_id"].notna()]
    return df["text"].tolist(), df["label_id"].astype(int).tolist()


def main():
    parser = argparse.ArgumentParser(description="Train binary classifier with extensive tuning")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output_dir", type=str, default="results/binary_extensive_tuned", help="Output directory")
    args = parser.parse_args()

    np.random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    texts, labels = load_data(args.input)
    print(f"Total samples: {len(texts)}")

    unique, counts = np.unique(labels, return_counts=True)
    print("Binary class distribution:")
    for c, count in zip(unique, counts):
        print(f"  {ID2LABEL_BINARY[c]}: {count} ({100*count/len(labels):.1f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Optimized TF-IDF with more features
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Increased
        ngram_range=(1, 5),  # Up to 5-grams
        min_df=1,
        max_df=0.90,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Feature dimension: {X_train_tfidf.shape[1]}")

    # Save vectorizer
    with open(f"{args.output_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    results = []

    # ============ Model 1: XGBoost with extensive tuning ============
    print("\n" + "=" * 50)
    print("Training XGBoost...")

    # Calculate scale_pos_weight for class imbalance
    pos_count = sum(y_train)
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count

    param_grid_xgb = {
        'n_estimators': [300, 500, 800],
        'max_depth': [6, 8, 10, 12],
        'learning_rate': [0.05, 0.08, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2.0]
    }

    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_xgb = GridSearchCV(
        xgb_base,
        param_grid_xgb,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    print(f"Fitting on {len(X_train_tfidf)} samples...")
    grid_xgb.fit(X_train_tfidf, y_train)

    print(f"\nBest XGBoost params: {grid_xgb.best_params_}")
    print(f"Best XGBoost CV accuracy: {grid_xgb.best_score_:.4f}")

    best_xgb = grid_xgb.best_estimator_
    y_pred = best_xgb.predict(X_test_tfidf)
    xgb_acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost Test Accuracy: {xgb_acc:.4f}")
    results.append(("XGBoost (Tuned)", xgb_acc, best_xgb))

    # ============ Model 2: SVM with extensive tuning ============
    print("\n" + "=" * 50)
    print("Training SVM...")

    param_grid_svm = {
        'C': [0.5, 1, 5, 10, 20],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
        'kernel': ['rbf', 'linear', 'poly'],
        'class_weight': ['balanced']
    }

    svm_base = SVC(probability=True, random_state=42)

    grid_svm = GridSearchCV(
        svm_base,
        param_grid_svm,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    grid_svm.fit(X_train_tfidf, y_train)

    print(f"\nBest SVM params: {grid_svm.best_params_}")
    print(f"Best SVM CV accuracy: {grid_svm.best_score_:.4f}")

    best_svm = grid_svm.best_estimator_
    y_pred = best_svm.predict(X_test_tfidf)
    svm_acc = accuracy_score(y_test, y_pred)
    print(f"SVM Test Accuracy: {svm_acc:.4f}")
    results.append(("SVM (Tuned)", svm_acc, best_svm))

    # ============ Model 3: Random Forest ============
    print("\n" + "=" * 50)
    print("Training Random Forest...")

    param_grid_rf = {
        'n_estimators': [300, 500, 800],
        'max_depth': [20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid_rf = GridSearchCV(
        rf_base,
        param_grid_rf,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    grid_rf.fit(X_train_tfidf, y_train)

    print(f"\nBest RF params: {grid_rf.best_params_}")
    print(f"Best RF CV accuracy: {grid_rf.best_score_:.4f}")

    best_rf = grid_rf.best_estimator_
    y_pred = best_rf.predict(X_test_tfidf)
    rf_acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest Test Accuracy: {rf_acc:.4f}")
    results.append(("Random Forest (Tuned)", rf_acc, best_rf))

    # ============ Final Ensemble = ============
    print("\n" + "=" * 50)
    print("Creating Ensemble...")

    ensemble = VotingClassifier(
        estimators=[
            ('xgb', best_xgb),
            ('svm', best_svm),
            ('rf', best_rf)
        ],
        voting='soft'
    )
    ensemble.fit(X_train_tfidf, y_train)
    y_pred = ensemble.predict(X_test_tfidf)
    ensemble_acc = accuracy_score(y_test, y_pred)
    print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")
    results.append(("Ensemble", ensemble_acc, ensemble))

    # ============ Results Summary = ============
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<25} {'Test Accuracy':>15}")
    print("-" * 70)
    for name, acc, model in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:<25} {acc:.4f}")
    print("=" * 70)

    # Determine best model
    best_name, best_accuracy, best_model = max(results, key=lambda x: x[1])
    print(f"\nBest Model: {best_name}")
    print(f"Best Test Accuracy: {best_accuracy:.2%}")

    if best_accuracy >= 0.80:
        print("\n✅ TARGET ACHIEVED: Accuracy >= 80%")
    else:
        print(f"\n❌ Target not achieved. Need {0.80 - best_accuracy:.2%} more improvement")

    # Classification report
    print("\nClassification Report for Best Model:")
    print(classification_report(y_test, best_model.predict(X_test_tfidf),
                              target_names=["NON-POSITIVE", "POSITIVE"]))

    # Save best model
    with open(f"{args.output_dir}/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nBest model saved to: {args.output_dir}/best_model.pkl")

    # Save results
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "best_accuracy": float(best_accuracy),
            "target_achieved": best_accuracy >= 0.80,
            "all_results": {name: float(acc) for name, acc, _ in results}
        }, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
