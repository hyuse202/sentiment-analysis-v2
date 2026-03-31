#!/usr/bin/env python3
"""
Evaluate sentiment models and compare baselines
Usage: python scripts/evaluate.py --input data/labeled/labeled_100.csv
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Load environment for GLM-5
from dotenv import load_dotenv
load_dotenv()


# Label mapping
LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}


def compute_metrics(y_true, y_pred, model_name="Model"):
    """Compute and print metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    print(f"\n{'=' * 40}")
    print(f"{model_name} Results:")
    print(f"{'=' * 40}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["NEGATIVE", "NEUTRAL", "POSITIVE"]))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def random_baseline(y_true, seed=42):
    """Random prediction baseline"""
    np.random.seed(seed)
    y_pred = np.random.randint(0, 3, size=len(y_true))
    return compute_metrics(y_true, y_pred, "Random Baseline")


def majority_baseline(y_true):
    """Majority class baseline"""
    majority_class = max(set(y_true), key=y_true.count)
    y_pred = [majority_class] * len(y_true)
    return compute_metrics(y_true, y_pred, "Majority Class Baseline")


def evaluate_phobert(texts, labels, model_path, test_size=0.15, seed=42):
    """Evaluate fine-tuned PhoBERT model"""
    # Split data
    _, test_texts, _, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")

    # Predict
    predictions = []
    for text in tqdm(test_texts, desc="Predicting"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(pred)

    return compute_metrics(test_labels, predictions, "PhoBERT (Fine-tuned)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment models")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file with labeled data")
    parser.add_argument("--model", type=str, default="models/phobert_sentiment", help="Trained model path")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    df = pd.read_csv(args.input)
    df["text"] = df["title"] + ". " + df["content"]
    df["label_id"] = df["sentiment"].map(LABEL2ID)
    df = df[df["label_id"].notna()]

    texts = df["text"].tolist()
    labels = df["label_id"].astype(int).tolist()

    print(f"Total samples: {len(texts)}")
    print(f"Label distribution: {df['sentiment'].value_counts().to_dict()}")

    # Get test split for baselines
    _, _, _, test_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=args.seed, stratify=labels
    )

    results = {}

    # 1. Random Baseline
    print("\n" + "=" * 50)
    print("Evaluating Random Baseline...")
    results["random_baseline"] = random_baseline(test_labels, args.seed)

    # 2. Majority Class Baseline
    print("\n" + "=" * 50)
    print("Evaluating Majority Class Baseline...")
    results["majority_baseline"] = majority_baseline(test_labels)

    # 3. PhoBERT (if model exists)
    if os.path.exists(args.model):
        print("\n" + "=" * 50)
        print("Evaluating Fine-tuned PhoBERT...")
        results["phobert_finetuned"] = evaluate_phobert(texts, labels, args.model, seed=args.seed)
    else:
        print(f"\n[Warning] Model not found at {args.model}")
        print("Train the model first with: python scripts/train_phobert.py")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    results_path = f"{args.output}/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
