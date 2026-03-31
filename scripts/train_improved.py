#!/usr/bin/env python3
"""
Train Improved ML Models for Sentiment Classification
Enhancements:
- Class balancing with SMOTE and class_weight
- XGBoost with tuned hyperparameters
- Ensemble voting classifier
- Vietnamese stopwords removal
- Hyperparameter tuning with GridSearchCV

Usage: python scripts/train_improved.py --input data/labeled/vific_labeled_1000_research.csv --tune
"""

import os
import json
import argparse
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


# Label mapping
LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

# Vietnamese stopwords for financial domain
VIETNAMESE_STOPWORDS = [
    'của', 'và', 'các', 'có', 'được', 'trong', 'với', 'cho', 'này', 'để',
    'tại', 'trên', 'từ', 'về', 'là', 'đến', 'như', 'khi', 'cũng', 'nhưng',
    'đã', 'đang', 'sẽ', 'mà', 'thì', 'nên', 'vẫn', 'rất', 'nhiều', 'hơn',
    'khác', 'phải', 'nếu', 'hay', 'hoặc', 'nhất', 'mỗi', 'ngay', 'ra', 'vào',
    'lại', 'đây', 'kia', 'đó', 'những', 'còn', 'không', 'trong', 'ngoài',
    'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
    'vẫn', 'đều', 'tất', 'cả', 'nhau', 'ngay', 'vừa', 'mới', 'đây', 'kia'
]


def load_data(input_path, remove_stopwords=True):
    """Load labeled data from CSV"""
    df = pd.read_csv(input_path)
    df["text"] = df["title"] + ". " + df["content"]

    if remove_stopwords:
        # Simple stopwords removal
        def remove_sw(text):
            words = text.lower().split()
            return ' '.join([w for w in words if w not in VIETNAMESE_STOPWORDS])
        df["text"] = df["text"].apply(remove_sw)

    df["label_id"] = df["sentiment"].map(LABEL2ID)
    df = df[df["label_id"].notna()]
    return df["text"].tolist(), df["label_id"].astype(int).tolist()


def compute_metrics(y_true, y_pred):
    """Compute classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NEGATIVE", "NEUTRAL", "POSITIVE"],
        yticklabels=["NEGATIVE", "NEUTRAL", "POSITIVE"]
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
        print("  Using SMOTE for oversampling...")

    # Apply SMOTE if requested
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

    # Metrics
    metrics = compute_metrics(y_test, y_pred)

    # Cross-validation (on original data, not SMOTE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")

    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  CV F1 (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["NEGATIVE", "NEUTRAL", "POSITIVE"]))

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
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std()
    }


def tune_xgboost(X_train, y_train):
    """Tune XGBoost hyperparameters"""
    print("\n" + "=" * 50)
    print("Tuning XGBoost hyperparameters...")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
    }

    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )

    # Apply SMOTE for tuning
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    grid_search.fit(X_res, y_res)
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV F1: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def tune_random_forest(X_train, y_train):
    """Tune Random Forest hyperparameters"""
    print("\n" + "=" * 50)
    print("Tuning Random Forest hyperparameters...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV F1: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def main():
    parser = argparse.ArgumentParser(description="Train improved ML models")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output_dir", type=str, default="results/improved", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_smote", action="store_true", help="Use SMOTE oversampling")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    texts, labels = load_data(args.input, remove_stopwords=True)
    print(f"Total samples: {len(texts)}")

    # Show class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for c, count in zip(unique, counts):
        print(f"  {ID2LABEL[c]}: {count} ({100*count/len(labels):.1f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=args.seed, stratify=labels
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # TF-IDF Vectorization with expanded n-grams
    print("\nVectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),  # Expanded to trigrams
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

    # If tuning requested, run hyperparameter search
    if args.tune:
        best_rf = tune_random_forest(X_train_tfidf, y_train)
        best_xgb = tune_xgboost(X_train_tfidf, y_train)
    else:
        # Use balanced class weights by default
        best_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight='balanced',
            random_state=args.seed,
            n_jobs=-1
        )
        best_xgb = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

    # Define models with balanced weights
    models = [
        (MultinomialNB(), "Naive Bayes"),
        (LogisticRegression(max_iter=1000, random_state=args.seed, class_weight='balanced'), "Logistic Regression"),
        (SVC(kernel="linear", random_state=args.seed, class_weight='balanced', probability=True), "SVM (Linear)"),
        (SVC(kernel="rbf", random_state=args.seed, class_weight='balanced', probability=True), "SVM (RBF)"),
        (best_rf, "Random Forest (Tuned)" if args.tune else "Random Forest"),
        (best_xgb, "XGBoost"),
    ]

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
            ('rf', best_rf),
            ('xgb', best_xgb),
            ('lr', LogisticRegression(max_iter=1000, random_state=args.seed, class_weight='balanced'))
        ],
        voting='soft'
    )
    ensemble_result = train_and_evaluate_model(
        ensemble, "Ensemble (RF+XGB+LR)", X_train_tfidf, X_test_tfidf,
        y_train, y_test, args.output_dir, use_smote=args.use_smote
    )
    all_results.append(ensemble_result)

    # Summary comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Accuracy':<10} {'F1':<10} {'CV F1':<15}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['model_name']:<30} {r['accuracy']:<10.4f} {r['f1']:<10.4f} {r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}")

    # Find best model
    best_result = max(all_results, key=lambda x: x['cv_f1_mean'])
    print(f"\nBest model: {best_result['model_name']} with CV F1: {best_result['cv_f1_mean']:.4f}")

    # Save results
    results_path = f"{args.output_dir}/improved_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "results": all_results,
            "timestamp": datetime.now().isoformat(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_dimension": X_train_tfidf.shape[1],
            "best_model": best_result['model_name']
        }, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")
    print("Models saved as .pkl files")


if __name__ == "__main__":
    main()
