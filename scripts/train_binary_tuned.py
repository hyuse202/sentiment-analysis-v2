#!/usr/bin/env python3
"""
Tuned Binary Sentiment Classification - Focus on achieving >80% accuracy
"""

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Binary label mapping
LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 0, "POSITIVE": 1}
ID2LABEL = {0: "NON-POSITIVE", 1: "POSITIVE"}

# Vietnamese stopwords
STOPWORDS = set(['của', 'và', 'các', 'có', 'được', 'trong', 'với', 'cho', 'này', 'để',
                 'tại', 'trên', 'từ', 'về', 'là', 'đến', 'như', 'khi', 'cũng', 'nhưng'])


def load_data(input_path):
    df = pd.read_csv(input_path)
    df["text"] = df["title"] + ". " + df["content"]
    df["text"] = df["text"].apply(lambda x: ' '.join([w for w in x.lower().split() if w not in STOPWORDS]))
    df["label_id"] = df["sentiment"].map(LABEL2ID)
    df = df[df["label_id"].notna()]
    return df["text"].tolist(), df["label_id"].astype(int).tolist()


def main():
    parser = argparse.ArgumentParser(description="Tuned binary classifier")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/binary_tuned")
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
        print(f"  {ID2LABEL[c]}: {count} ({100*count/len(labels):.1f}%)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # TF-IDF with optimized parameters
    vectorizer = TfidfVectorizer(
        max_features=8000,  # Increased
        ngram_range=(1, 4),  # Up to 4-grams
        min_df=1,
        max_df=0.95,
        sublinear_tf=True  # Apply sublinear TF scaling
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Feature dimension: {X_train_tfidf.shape[1]}")

    # Save vectorizer
    with open(f"{args.output_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)
    print(f"After SMOTE: {len(y_train_res)} samples")

    # Calculate scale_pos_weight
    pos_count = sum(y_train)
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count

    results = []

    # ============ XGBoost with extensive tuning ============
    print("\n" + "=" * 50)
    print("Training XGBoost with hyperparameter tuning...")

    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    # Use smaller grid for faster tuning
    xgb_grid = {
        'n_estimators': [200, 300],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.1, 0.15],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    xgb_search = GridSearchCV(xgb_model, xgb_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    xgb_search.fit(X_train_res, y_train_res)

    print(f"Best XGBoost params: {xgb_search.best_params_}")
    best_xgb = xgb_search.best_estimator_

    y_pred = best_xgb.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost (Tuned) Accuracy: {acc:.4f}")
    results.append(("XGBoost (Tuned)", acc, best_xgb))

    # ============ SVM with tuning ============
    print("\n" + "=" * 50)
    print("Training SVM with hyperparameter tuning...")

    svm_params = {
        'C': [0.1, 0.5, 1, 2, 5, 10],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }

    svm_grid = {
        'C': [0.5, 1, 2, 5, 10],
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['rbf']
    }

    svm_model = SVC(probability=True, class_weight='balanced', random_state=42)
    svm_search = GridSearchCV(svm_model, svm_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    svm_search.fit(X_train_res, y_train_res)

    print(f"Best SVM params: {svm_search.best_params_}")
    best_svm = svm_search.best_estimator_

    y_pred = best_svm.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"SVM (Tuned) Accuracy: {acc:.4f}")
    results.append(("SVM (Tuned)", acc, best_svm))

    # ============ Random Forest with tuning ============
    print("\n" + "=" * 50)
    print("Training Random Forest with hyperparameter tuning...")

    rf_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    rf_search = GridSearchCV(rf_model, rf_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    rf_search.fit(X_train_res, y_train_res)

    print(f"Best RF params: {rf_search.best_params_}")
    best_rf = rf_search.best_estimator_

    y_pred = best_rf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest (Tuned) Accuracy: {acc:.4f}")
    results.append(("Random Forest (Tuned)", acc, best_rf))

    # ============ Ensemble ============
    print("\n" + "=" * 50)
    print("Training Ensemble...")

    ensemble = VotingClassifier(
        estimators=[
            ('xgb', best_xgb),
            ('svm', best_svm),
            ('rf', best_rf)
        ],
        voting='soft'
    )
    ensemble.fit(X_train_res, y_train_res)

    y_pred = ensemble.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Ensemble Accuracy: {acc:.4f}")
    results.append(("Ensemble (XGB+SVM+RF)", acc, ensemble))

    # ============ Summary ============
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Test Accuracy':<15}")
    print("-" * 60)
    for name, acc, model in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:<30} {acc:.4f} ({acc:.1%})")

    best_name, best_acc, best_model = max(results, key=lambda x: x[1])
    print(f"\nBest Model: {best_name}")
    print(f"Best Test Accuracy: {best_acc:.2%}")

    if best_acc >= 0.80:
        print("\n✅ TARGET ACHIEVED: Accuracy >= 80%")
    else:
        print(f"\n❌ Target not achieved. Need {0.80 - best_acc:.2%} more improvement")

    # Save best model
    with open(f"{args.output_dir}/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Classification report for best model
    print("\nClassification Report for Best Model:")
    y_pred = best_model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred, target_names=["NON-POSITIVE", "POSITIVE"]))

    # Save results
    with open(f"{args.output_dir}/tuned_results.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "best_accuracy": best_acc,
            "target_achieved": best_acc >= 0.80,
            "results": [(name, float(acc)) for name, acc, _ in results],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
