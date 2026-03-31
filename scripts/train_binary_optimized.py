#!/usr/bin/env python3
"""
Optimized Binary Sentiment Classification - Target 80%+ accuracy
Strategy: Convert sparse to dense for XGBoost,"""

import os
import json
import argparse
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Binary labels: POSITIVE vs NON-POSITIVE
BINARY_LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 0, "POSITIVE": 1}
ID2LABEL_BINARY = {0: "NON-POSITIVE", 1: "POSITIVE"}

# Stopwords
STOPWORDS = set(['của', 'và', 'các', 'có', 'được', 'trong', 'với', 'cho', 'này', 'để',
                 'tại', 'trên', 'từ', 'về', 'là', 'đến', 'như', 'khi', 'cũng', 'nhưng'])


def load_data(input_path):
    df = pd.read_csv(input_path)
    df["text"] = df["title"] + ". " + df["content"]
    df["text"] = df["text"].apply(lambda x: ' '.join([w for w in str(x).lower().split() if w not in STOPWORDS]))
    df["label_id"] = df["sentiment"].map(BINARY_LABEL2ID)
    df = df[df["label_id"].notna()]
    return df["text"].tolist(), df["label_id"].astype(int).tolist()


def main():
    parser = argparse.ArgumentParser(description="Optimized binary classifier")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/binary_optimized")
    args = parser.parse_args()

    np.random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    texts, labels = load_data(args.input)
    print(f"Total samples: {len(texts)}")

    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution:")
    for c, count in zip(unique, counts):
        print(f"  {ID2LABEL_BINARY[c]}: {count} ({100*count/len(labels):.1f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # TF-IDF
    print("\nVectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 5),
        min_df=1,
        max_df=0.90,
        sublinear_tf=True
    )
    X_train_sparse = vectorizer.fit_transform(X_train)
    X_test_sparse = vectorizer.transform(X_test)
    print(f"Feature dimension: {X_train_sparse.shape[1]}")

    # Convert to dense for XGBoost and Ensemble
    X_train_dense = X_train_sparse.toarray()
    X_test_dense = X_test_sparse.toarray()

    results = []

    # ============ XGBoost ============
    print("\n" + "=" * 50)
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_dense, y_train)
    y_pred = xgb_model.predict(X_test_dense)
    xgb_acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")
    results.append(("XGBoost", xgb_acc, xgb_model))

    # ============ SVM ============
    print("\n" + "=" * 50)
    print("Training SVM...")
    svm_model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train_sparse, y_train)
    y_pred = svm_model.predict(X_test_sparse)
    svm_acc = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {svm_acc:.4f}")
    results.append(("SVM", svm_acc, svm_model))

    # ============ Random Forest ============
    print("\n" + "=" * 50)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_sparse, y_train)
    y_pred = rf_model.predict(X_test_sparse)
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
    lr_model.fit(X_train_sparse, y_train)
    y_pred = lr_model.predict(X_test_sparse)
    lr_acc = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    results.append(("Logistic Regression", lr_acc, lr_model))

    # ============ Ensemble ============
    print("\n" + "=" * 50)
    print("Training Ensemble (XGB+SVM+RF+LR)...")

    # Create fresh models for ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb.XGBClassifier(
                n_estimators=300, max_depth=10, learning_rate=0.1,
                subsample=0.85, colsample_bytree=0.85,
                objective='binary:logistic', random_state=42
            )),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=500, max_depth=20, class_weight='balanced', random_state=42)),
            ('lr', LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42))
        ],
        voting='soft'
    )
    ensemble.fit(X_train_dense, y_train)
    y_pred = ensemble.predict(X_test_dense)
    ensemble_acc = accuracy_score(y_test, y_pred)
    print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
    results.append(("Ensemble", ensemble_acc, ensemble))

    # ============ Summary ============
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Test Accuracy':>15}")
    print("-" * 70)
    for name, acc, model in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:<25} {acc:.4f} ({acc:.1%})")

    best_name, best_acc, best_model = max(results, key=lambda x: x[1])
    print(f"\nBest Model: {best_name}")
    print(f"Best Test Accuracy: {best_acc:.2%}")

    if best_acc >= 0.80:
        print("\n✅ TARGET ACHIEVED: Accuracy >= 80%")
    else:
        print(f"\n❌ Target not achieved. Need {0.80 - best_acc:.2%} more improvement")

    # Classification report
    print("\nClassification Report for Best Model:")
    if best_name == "Ensemble":
        y_pred = best_model.predict(X_test_dense)
    else:
        y_pred = best_model.predict(X_test_sparse if "XGB" not in best_name else X_test_dense)
    print(classification_report(y_test, y_pred, target_names=["NON-POSITIVE", "POSITIVE"]))

    # Save results
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "best_accuracy": float(best_acc),
            "target_achieved": best_acc >= 0.80,
            "all_results": {name: float(acc) for name, acc, _ in results}
        }, f, indent=2)

    # Save vectorizer and best model
    with open(f"{args.output_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"{args.output_dir}/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
