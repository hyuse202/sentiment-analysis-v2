#!/usr/bin/env python3
"""
Train Traditional ML Baselines for Sentiment Classification
Includes: TF-IDF + SVM, Random Forest, Logistic Regression, Naive Bayes

Usage: python scripts/train_baselines.py --input data/labeled/labeled_100.csv
"""

import os
import json
import argparse
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


# Label mapping
LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}


def load_data(input_path):
    """Load labeled data from CSV"""
    df = pd.read_csv(input_path)
    df["text"] = df["title"] + ". " + df["content"]
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


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, output_dir):
    """Train and evaluate a single model"""
    print(f"\n{'=' * 50}")
    print(f"Training {model_name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    metrics = compute_metrics(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")

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


def main():
    parser = argparse.ArgumentParser(description="Train traditional ML baselines")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output_dir", type=str, default="results/baselines", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    texts, labels = load_data(args.input)
    print(f"Total samples: {len(texts)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=args.seed, stratify=labels
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # TF-IDF Vectorization
    print("\nVectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Feature dimension: {X_train_tfidf.shape[1]}")

    # Save vectorizer
    with open(f"{args.output_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Define models with class_weight='balanced' to handle imbalance
    models = [
        (MultinomialNB(), "Naive Bayes"),  # No class_weight support
        (LogisticRegression(max_iter=1000, random_state=args.seed, class_weight='balanced'), "Logistic Regression"),
        (SVC(kernel="linear", random_state=args.seed, class_weight='balanced'), "SVM (Linear)"),
        (SVC(kernel="rbf", random_state=args.seed, class_weight='balanced'), "SVM (RBF)"),
        (RandomForestClassifier(n_estimators=100, random_state=args.seed, class_weight='balanced'), "Random Forest"),
    ]

    # Train and evaluate each model
    all_results = []

    for model, name in models:
        result = train_and_evaluate_model(
            model, name, X_train_tfidf, X_test_tfidf, y_train, y_test, args.output_dir
        )
        all_results.append(result)

    # Summary comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':<10} {'F1':<10} {'CV F1':<15}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['model_name']:<25} {r['accuracy']:<10.4f} {r['f1']:<10.4f} {r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}")

    # Save results
    results_path = f"{args.output_dir}/baseline_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "results": all_results,
            "timestamp": datetime.now().isoformat(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_dimension": X_train_tfidf.shape[1]
        }, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")
    print("Models saved as .pkl files")


if __name__ == "__main__":
    main()
