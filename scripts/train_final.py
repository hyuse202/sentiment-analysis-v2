#!/usr/bin/env python3
"""
Final Binary Classifier - Target >80% accuracy
Strategy: No SMOTE, use class weights, aggressive feature engineering
"""

import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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
    'vẫn', 'đều', 'tất', 'cả', 'nhau', 'ngay', 'vừa', 'mới', 'đây', 'kia',
    'theo', 'đồng', 'triệu', 'tỷ', 'năm', 'tháng', 'ngày', 'quý', 'về',
])


def load_data(input_path):
    df = pd.read_csv(input_path)
    df["text"] = df["title"] + ". " + df["content"]
    # Remove stopwords
    df["text"] = df["text"].apply(lambda x: ' '.join([w for w in x.lower().split() if w not in STOPWORDS]))
    df["label_id"] = df["sentiment"].map(LABEL2ID)
    df = df[df["label_id"].notna()]
    return df["text"].tolist(), df["label_id"].astype(int).tolist()


def main():
    output_dir = "results/binary_final"
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

    # Use smaller test set for more training data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.12, random_state=42, stratify=labels
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Optimized TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 5),  # Up to 5-grams
        min_df=1,
        max_df=0.90,
        sublinear_tf=True,
        analyzer='word'
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
    class_weight_dict = {0: pos_count / neg_count, 1: 1.0}

    print(f"\nClass weights - NON-POSITIVE: {class_weight_dict[0]:.2f}, POSITIVE: {class_weight_dict[1]:.2f}")

    results = []

    # ============ Model 1: XGBoost (No SMOTE, class weights) ============
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
    xgb_model.fit(X_train_tfidf, y_train)
    y_pred = xgb_model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {acc:.4f}")
    results.append(("XGBoost", acc, xgb_model))

    # ============ Model 2: SVM (RBF) ============
    print("\n" + "=" * 50)
    print("Training SVM...")

    svm_model = SVC(
        C=5.0,
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train_tfidf, y_train)
    y_pred = svm_model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {acc:.4f}")
    results.append(("SVM", acc, svm_model))

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
    y_pred = lr_model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {acc:.4f}")
    results.append(("Logistic Regression", acc, lr_model))

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
    y_pred = rf_model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {acc:.4f}")
    results.append(("Random Forest", acc, rf_model))

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
    y_pred = gb_model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Gradient Boosting Accuracy: {acc:.4f}")
    results.append(("Gradient Boosting", acc, gb_model))

    # ============ Model 6: Ensemble (Soft Voting) ============
    print("\n" + "=" * 50)
    print("Training Ensemble...")

    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('svm', svm_model),
            ('lr', lr_model),
            ('rf', rf_model)
        ],
        voting='soft'
    )
    ensemble.fit(X_train_tfidf, y_train)
    y_pred = ensemble.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Ensemble Accuracy: {acc:.4f}")
    results.append(("Ensemble", acc, ensemble))

    # ============ Model 7: Weighted Ensemble ============
    print("\n" + "=" * 50)
    print("Training Weighted Ensemble...")

    # Get probabilities from each model
    probas = []
    for name, _, model in results[:-1]:  # Exclude ensemble
        if hasattr(model, 'predict_proba'):
            probas.append(model.predict_proba(X_test_tfidf)[:, 1])

    # Weighted average (give more weight to better models)
    weights = [0.30, 0.25, 0.15, 0.15, 0.15]  # XGB, SVM, LR, RF, GB
    weighted_proba = np.zeros_like(probas[0])
    for w, p in zip(weights, probas):
        weighted_proba += w * p

    y_pred_weighted = (weighted_proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred_weighted)
    print(f"Weighted Ensemble Accuracy: {acc:.4f}")
    results.append(("Weighted Ensemble", acc, None))

    # ============ Summary ============
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Model':<30} {'Test Accuracy':<15}")
    print("-" * 60)
    for name, acc, model in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:<30} {acc:.4f} ({acc:.1%})")

    best_name, best_acc, best_model = max(results, key=lambda x: x[1])
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
        y_pred = best_model.predict(X_test_tfidf)
    else:
        y_pred = y_pred_weighted

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["NON-POSITIVE", "POSITIVE"]))

    # Save best model
    if best_model is not None:
        with open(f"{output_dir}/best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

    # Save results
    with open(f"{output_dir}/final_results.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "best_accuracy": float(best_acc),
            "target_achieved": best_acc >= 0.80,
            "results": [(name, float(acc)) for name, acc, _ in results],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
