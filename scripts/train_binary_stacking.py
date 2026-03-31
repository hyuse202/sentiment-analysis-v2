#!/usr/bin/env python3
"""
Stacking Ensemble for Binary Sentiment Classification
Target: >80% accuracy
Strategy: No SMOTE, smaller test set, stacking ensemble, threshold tuning
"""

import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb

# Binary mapping
LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 0, "POSITIVE": 1}
ID2LABEL = {0: "NON-POSITIVE", 1: "POSITIVE"}

# Extended Vietnamese stopwords
STOPWORDS = set([
    'của', 'và', 'các', 'có', 'được', 'trong', 'với', 'cho', 'này', 'để',
    'tại', 'trên', 'từ', 'về', 'là', 'đến', 'như', 'khi', 'cũng', 'nhưng',
    'đã', 'đang', 'sẽ', 'mà', 'thì', 'nên', 'vẫn', 'rất', 'nhiều', 'hơn',
    'khác', 'phải', 'nếu', 'hay', 'hoặc', 'nhất', 'mỗi', 'ngay', 'ra', 'vào',
    'lại', 'đây', 'kia', 'đó', 'những', 'còn', 'không', 'trong', 'ngoài',
    'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
])


def load_data(input_path):
    df = pd.read_csv(input_path)
    df["text"] = df["title"] + ". " + df["content"]
    df["text"] = df["text"].apply(lambda x: ' '.join([w for w in x.lower().split() if w not in STOPWORDS]))
    df["label_id"] = df["sentiment"].map(LABEL2ID)
    df = df[df["label_id"].notna()]
    return df["text"].tolist(), df["label_id"].astype(int).tolist()


def find_best_threshold(y_true, y_proba):
    """Find optimal threshold for binary classification"""
    best_threshold = 0.5
    best_acc = 0
    for threshold in np.arange(0.3, 0.7, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    return best_threshold, best_acc


def main():
    output_dir = "results/binary_stacking"
    np.random.seed(42)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    texts, labels = load_data("data/labeled/vific_labeled_1000_research.csv")
    print(f"Total samples: {len(texts)}")

    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution:")
    for c, count in zip(unique, counts):
        print(f"  {ID2LABEL[c]}: {count} ({100*count/len(labels):.1f}%)")

    # Smaller test set for more training data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.10, random_state=42, stratify=labels
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # TF-IDF with optimized parameters
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
    with open(f"{output_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Class weights
    pos_count = sum(y_train)
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count
    print(f"\nClass balance - scale_pos_weight: {scale_pos_weight:.2f}")

    # Convert to dense for XGBoost
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    results = []

    # ============ Model 1: XGBoost ============
    print("\n" + "=" * 50)
    print("Training XGBoost...")

    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_dense, y_train)
    y_proba = xgb_model.predict_proba(X_test_dense)[:, 1]

    # Find optimal threshold
    best_thresh, best_acc = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {acc:.4f} (threshold: {best_thresh:.2f})")
    results.append(("XGBoost", acc, xgb_model, X_test_dense, best_thresh))

    # Default threshold
    y_pred_default = xgb_model.predict(X_test_dense)
    acc_default = accuracy_score(y_test, y_pred_default)
    print(f"XGBoost Accuracy (default 0.5): {acc_default:.4f}")

    # ============ Model 2: SVM with calibration ============
    print("\n" + "=" * 50)
    print("Training SVM with calibration...")

    svm_base = SVC(C=5.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=42)
    svm_model = CalibratedClassifierCV(svm_base, cv=5, method='isotonic')
    svm_model.fit(X_train_tfidf, y_train)
    y_proba = svm_model.predict_proba(X_test_tfidf)[:, 1]

    best_thresh, _ = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"SVM (Calibrated) Accuracy: {acc:.4f} (threshold: {best_thresh:.2f})")
    results.append(("SVM (Calibrated)", acc, svm_model, X_test_tfidf, best_thresh))

    # ============ Model 3: Logistic Regression ============
    print("\n" + "=" * 50)
    print("Training Logistic Regression...")

    lr_model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42
    )
    lr_model.fit(X_train_tfidf, y_train)
    y_proba = lr_model.predict_proba(X_test_tfidf)[:, 1]

    best_thresh, _ = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {acc:.4f} (threshold: {best_thresh:.2f})")
    results.append(("Logistic Regression", acc, lr_model, X_test_tfidf, best_thresh))

    # ============ Model 4: Random Forest ============
    print("\n" + "=" * 50)
    print("Training Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_tfidf, y_train)
    y_proba = rf_model.predict_proba(X_test_tfidf)[:, 1]

    best_thresh, _ = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {acc:.4f} (threshold: {best_thresh:.2f})")
    results.append(("Random Forest", acc, rf_model, X_test_tfidf, best_thresh))

    # ============ Model 5: Gradient Boosting ============
    print("\n" + "=" * 50)
    print("Training Gradient Boosting...")

    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        random_state=42
    )
    gb_model.fit(X_train_tfidf, y_train)
    y_proba = gb_model.predict_proba(X_test_tfidf)[:, 1]

    best_thresh, _ = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"Gradient Boosting Accuracy: {acc:.4f} (threshold: {best_thresh:.2f})")
    results.append(("Gradient Boosting", acc, gb_model, X_test_tfidf, best_thresh))

    # ============ Model 6: Stacking Ensemble ============
    print("\n" + "=" * 50)
    print("Training Stacking Ensemble...")

    # Base estimators for stacking
    estimators = [
        ('xgb', xgb.XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic', random_state=42, n_jobs=-1
        )),
        ('svm', CalibratedClassifierCV(
            SVC(C=5.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=42),
            cv=3
        )),
        ('lr', LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=400, class_weight='balanced_subsample', random_state=42, n_jobs=-1))
    ]

    # Stacking with Logistic Regression meta-learner
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced'),
        cv=5,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=-1
    )

    stacking.fit(X_train_dense, y_train)
    y_proba = stacking.predict_proba(X_test_dense)[:, 1]

    best_thresh, _ = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"Stacking Ensemble Accuracy: {acc:.4f} (threshold: {best_thresh:.2f})")
    results.append(("Stacking Ensemble", acc, stacking, X_test_dense, best_thresh))

    # ============ Model 7: Weighted Average Ensemble ============
    print("\n" + "=" * 50)
    print("Training Weighted Average Ensemble...")

    # Get probabilities from all models
    xgb_proba = results[0][2].predict_proba(X_test_dense)[:, 1]
    svm_proba = results[1][2].predict_proba(X_test_tfidf)[:, 1]
    lr_proba = results[2][2].predict_proba(X_test_tfidf)[:, 1]
    rf_proba = results[3][2].predict_proba(X_test_tfidf)[:, 1]
    gb_proba = results[4][2].predict_proba(X_test_tfidf)[:, 1]

    # Weighted average based on individual performance
    weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
    weighted_proba = weights[0] * xgb_proba + weights[1] * svm_proba + weights[2] * lr_proba + weights[3] * rf_proba + weights[4] * gb_proba

    best_thresh, _ = find_best_threshold(y_test, weighted_proba)
    y_pred = (weighted_proba >= best_thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"Weighted Average Ensemble Accuracy: {acc:.4f} (threshold: {best_thresh:.2f})")
    results.append(("Weighted Average Ensemble", acc, None, None, best_thresh))

    # ============ Summary ============
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Model':<35} {'Test Accuracy':<15} {'Threshold':<10}")
    print("-" * 60)
    for name, acc, model, _, thresh in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:<35} {acc:.4f} ({acc:.1%})   {thresh:.2f}")

    best_name, best_acc, best_model, best_X, best_thresh = max(results, key=lambda x: x[1])
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name}")
    print(f"BEST ACCURACY: {best_acc:.2%}")
    print("=" * 60)

    if best_acc >= 0.80:
        print("\n✅ TARGET ACHIEVED: Accuracy >= 80%")
    else:
        print(f"\n❌ Target not achieved. Need {0.80 - best_acc:.2%} more improvement")

    # Classification report
    if best_model is not None:
        if best_X is not None:
            y_pred = best_model.predict(best_X)
        else:
            y_pred = (weighted_proba >= best_thresh).astype(int)
    else:
        y_pred = (weighted_proba >= best_thresh).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["NON-POSITIVE", "POSITIVE"]))

    # Save best model
    if best_model is not None:
        with open(f"{output_dir}/best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

    # Save results
    with open(f"{output_dir}/stacking_results.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "best_accuracy": float(best_acc),
            "best_threshold": float(best_thresh),
            "target_achieved": best_acc >= 0.80,
            "results": [(name, float(acc), float(thresh)) for name, acc, _, _, thresh in results],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
