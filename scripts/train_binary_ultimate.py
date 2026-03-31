#!/usr/bin/env python3
"""
Ultimate Binary Sentiment Classifier - Targeting >80% accuracy
"""

import os
import json
import argparse
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Binary labels: merge NEUTRAL into NEGATIVE for better balance
BINARY_LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 0, "POSITIVE": 1}
ID2LABEL_BINARY = {0: "NON-POSITIVE", 1: "POSITIVE"}

# Vietnamese stopwords
STOPWORDS = [
    'của', 'và', 'các', 'có', 'được', 'trong', 'với', 'cho', 'này', 'để',
    'tại', 'trên', 'từ', 'về', 'là', 'đến', 'như', 'khi', 'cũng', 'nhưng',
    'đã', 'đang', 'sẽ', 'mà', 'thì', 'nên', 'vẫn', 'rất', 'nhiều', 'hơn',
]


def load_data(input_path):
    df = pd.read_csv(input_path)
    df["text"] = df["title"] + ". " + df["content"]

    # Remove stopwords
    def remove_sw(text):
        words = str(text).lower().split()
        return ' '.join([w for w in words if w not in STOPWORDS])

    df["text"] = df["text"].apply(remove_sw)

    # Binary labels
    df["label_id"] = df["sentiment"].map(BINARY_LABEL2ID)
    df = df[df["label_id"].notna()]

    print(f"Total samples: {len(df)}")

    # Class distribution
    unique, counts = np.unique(df['label_id'], return_counts=True)
    print("Binary class distribution:")
    for c, count in zip(unique, counts):
        print(f"  {ID2LABEL_BINARY[c]}: {count} ({100*count/len(df):.1f}%)")

    return df["text"].tolist(), df["label_id"].astype(int).tolist()


def main():
    parser = argparse.ArgumentParser(description="Train ultimate binary classifier")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/binary_ultimate")
    args = parser.parse_args()

    np.random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    texts, labels = load_data(args.input)
    print(f"Total samples: {len(texts)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Optimized TF-IDF
    print("\nVectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 5),
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

    # Apply SMOTE
    print("\nApplying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)
    print(f"After SMOTE: {len(y_train_res)} samples (was {len(y_train)})")

    # Convert to dense for XGBoost
    X_train_dense = X_train_res.toarray()
    X_test_dense = X_test_tfidf.toarray()

    results = []

    # ============ XGBoost with Optimal Params ============
    print("\n" + "=" * 50)
    print("Training XGBoost...")

    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=10,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=1.5,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train_dense, y_train_res)
    y_pred = xgb_model.predict(X_test_dense)
    xgb_acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")
    results.append(("XGBoost", xgb_acc, xgb_model))

    # ============ SVM with Class Weights ============
    print("\n" + "=" * 50)
    print("Training SVM...")

    svm_model = SVC(
        kernel='rbf',
        C=5,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train_tfidf, y_train)
    y_pred = svm_model.predict(X_test_tfidf)
    svm_acc = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {svm_acc:.4f}")
    results.append(("SVM", svm_acc, svm_model))

    # ============ Random Forest with Class Weights ============
    print("\n" + "=" * 50)
    print("Training Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_tfidf, y_train)
    y_pred = rf_model.predict(X_test_tfidf)
    rf_acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    results.append(("Random Forest", rf_acc, rf_model))

    # ============ Logistic Regression ============
    print("\n" + "=" * 50)
    print("Training Logistic Regression...")

    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42
    )
    lr_model.fit(X_train_tfidf, y_train)
    y_pred = lr_model.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    results.append(("Logistic Regression", lr_acc, lr_model))

    # ============ Ensemble = ============
    print("\n" + "=" * 50)
    print("Training Ensemble...")

    # Create fresh models for ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb.XGBClassifier(
                n_estimators=400, max_depth=10, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85,
                min_child_weight=1,
                gamma=0.05,
                reg_alpha=0.05,
                reg_lambda=1.5,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42
            )),
            ('svm', SVC(
                kernel='rbf',
                C=5,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )),
            ('rf', RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            )),
            ('lr', LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs',
                random_state=42
            ))
        ],
        voting='soft'
    )

    ensemble.fit(X_train_dense, y_train_res)
    y_pred = ensemble.predict(X_test_dense)
    ensemble_acc = accuracy_score(y_test, y_pred)
    print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
    results.append(("Ensemble", ensemble_acc, ensemble))

    # ============ Summary = ============
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Test Accuracy':>15}")
    print("-" * 70)

    for name, acc, _ in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:<25} {acc:.4f}")

    # Determine best model
    best_name, best_accuracy, best_model = max(results, key=lambda x: x[1])

    print(f"\nBest Model: {best_name}")
    print(f"Best Test Accuracy: {best_accuracy:.2%}")

    if best_accuracy >= 0.80:
        print("\nTARGET ACHIEVED: Accuracy >= 80%")
    else:
        print(f"\nTarget not achieved. Need {0.80 - best_accuracy:.2%} more improvement")

    # Classification report for best model
    print("\nClassification Report for Best Model:")
    print(classification_report(y_test, best_model.predict(X_test_dense if best_name == "Ensemble" else X_test_tfidf), target_names=["NON-POSITIVE", "POSITIVE"]))

    # Save results
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump({
            "results": [(name, float(acc)) for name, acc, _ in results],
            "timestamp": datetime.now().isoformat(),
            "best_model": best_name,
            "best_accuracy": float(best_accuracy),
            "target_achieved": best_accuracy >= 0.80
        }, f, indent=2)

    # Save best model
    with open(f"{args.output_dir}/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"\nResults saved to: {args.output_dir}")
    print(f"Best model saved to: {args.output_dir}/best_model.pkl")


if __name__ == "__main__":
    main()
