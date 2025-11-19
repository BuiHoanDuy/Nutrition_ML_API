"""
Train a Vietnamese question classifier to route user queries to the correct module.

Sample usage:
    python scripts/train_question_classifier.py \
        --data-path data/dataset_clean_text_v7.csv \
        --output-dir models/question_classifier
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


@dataclass
class Config:
    data_path: Path
    model_name: str
    output_dir: Path
    test_size: float
    num_train_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train question classifier")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data") / "dataset_clean_text_v7.csv",
        help="CSV file with columns `user_query` and `category`.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vinai/phobert-base",
        help="Pretrained model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models") / "question_classifier",
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of samples for evaluation.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()
    return Config(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        test_size=args.test_size,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


def load_dataframe(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    expected_cols = {"user_query", "category"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    df = df[list(expected_cols)].dropna()
    df = df.rename(columns={"user_query": "text", "category": "label"})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    if df.empty:
        raise ValueError("Dataset has no usable rows after cleaning.")
    return df


def encode_labels(df: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    labels = sorted(df["label"].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label_id"] = df["label"].map(label2id)
    return {"label2id": label2id, "id2label": id2label}


def tokenize_dataset(
    df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 128
) -> Dataset:
    dataset = Dataset.from_pandas(df[["text", "label_id"]])

    def preprocess(batch):
        tokens = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tokens["labels"] = batch["label_id"]
        return tokens

    return dataset.map(preprocess, batched=True, remove_columns=["text", "label_id"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def train_question_classifier(config: Config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"[INFO] Loading dataset from {config.data_path}")
    df = load_dataframe(config.data_path)
    label_maps = encode_labels(df)
    print(f"[INFO] Found {len(label_maps['label2id'])} labels: {label_maps['label2id']}")

    train_df, eval_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=df["label_id"],
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label_maps["label2id"]),
        id2label=label_maps["id2label"],
        label2id=label_maps["label2id"],
    )

    train_dataset = tokenize_dataset(train_df, tokenizer)
    eval_dataset = tokenize_dataset(eval_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir / "checkpoints"),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        logging_steps=50,
        seed=config.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Starting training...")
    trainer.train()
    metrics = trainer.evaluate(eval_dataset)
    print(f"[SUCCESS] Evaluation metrics: {metrics}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving model to {config.output_dir}")
    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    with open(config.output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label_maps["label2id"],
                "id2label": label_maps["id2label"],
                "metrics": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[DONE] Question classifier training complete.")


def main():
    config = parse_args()
    train_question_classifier(config)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

