#!/usr/bin/env python3
"""
Train PhoBERT for Vietnamese Financial Sentiment Classification
Usage: python scripts/train_phobert.py --input data/labeled/labeled_100.csv
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from tqdm import tqdm


# Label mapping
LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment classification"""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def load_data(input_path):
    """Load labeled data from CSV"""
    df = pd.read_csv(input_path)

    # Combine title and content
    df["text"] = df["title"] + ". " + df["content"]

    # Convert labels to integers
    df["label_id"] = df["sentiment"].map(LABEL2ID)

    # Filter out invalid labels
    df = df[df["label_id"].notna()]

    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['sentiment'].value_counts()}")

    return df["text"].tolist(), df["label_id"].tolist()


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def main():
    parser = argparse.ArgumentParser(description="Train PhoBERT for sentiment classification")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file with labeled data")
    parser.add_argument("--output_dir", type=str, default="models/phobert_sentiment", help="Output directory")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base-v2", help="Pretrained model name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print("=" * 50)
    print("Loading data...")
    texts, labels = load_data(args.input)

    # Split data: 70% train, 15% val, 15% test
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=args.seed, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=args.seed, stratify=temp_labels
    )

    print(f"\nData splits:")
    print(f"  Train: {len(train_texts)}")
    print(f"  Val: {len(val_texts)}")
    print(f"  Test: {len(test_texts)}")

    # Load tokenizer and model
    print("\n" + "=" * 50)
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, args.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        seed=args.seed,
        use_cpu=True,  # Force CPU
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    print("\n" + "=" * 50)
    print("Training...")
    trainer.train()

    # Evaluate on test set
    print("\n" + "=" * 50)
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    # Detailed classification report
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)

    print("\nClassification Report:")
    print(classification_report(
        test_labels,
        pred_labels,
        target_names=["NEGATIVE", "NEUTRAL", "POSITIVE"]
    ))

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save metrics
    results = {
        "test_accuracy": test_results["eval_accuracy"],
        "test_f1": test_results["eval_f1"],
        "test_precision": test_results["eval_precision"],
        "test_recall": test_results["eval_recall"],
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "test_size": len(test_texts),
        "timestamp": datetime.now().isoformat()
    }

    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test F1: {test_results['eval_f1']:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
