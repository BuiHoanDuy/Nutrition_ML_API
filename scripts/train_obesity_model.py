"""
Train obesity prediction model.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


NUMERIC_COLUMNS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]
CATEGORICAL_FIELDS = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]


def _compute_numeric_stats(raw_df: pd.DataFrame) -> dict:
    """Compute mean/std for numeric columns from original dataset."""
    numeric_stats = {}
    df = raw_df.copy()
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    for col in NUMERIC_COLUMNS:
        mean = float(df[col].mean())
        std = float(df[col].std(ddof=0)) or 1.0
        numeric_stats[col] = {"mean": mean, "std": std}
    return numeric_stats


def _infer_one_hot_mapping(feature_cols: list[str]) -> dict:
    """Infer one-hot columns mapping from preprocessed feature names."""
    mapping = {}
    for field in CATEGORICAL_FIELDS:
        prefix = f"{field}_"
        mapped_cols = [col for col in feature_cols if col.startswith(prefix)]
        mapping[field] = mapped_cols
    return mapping


def _save_classification_report(report: dict, output_path: Path) -> None:
    df = pd.DataFrame(report).transpose()
    df.to_csv(output_path, index=True, encoding="utf-8-sig")


def _plot_confusion_matrix(cm: np.ndarray, class_labels: list[str], output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        linewidths=0.5,
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("Obesity Model - Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_feature_importance(feature_names: list[str], importances: np.ndarray, output_path: Path) -> None:
    top_k = min(20, len(feature_names))
    indices = np.argsort(importances)[::-1][:top_k]
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        legend=False,
    )
    plt.title("Top Feature Importances")
    plt.xlabel("Importance score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_obesity_model():
    """Train obesity prediction model using the preprocessed dataset."""
    print("[INFO] Loading obesity datasets...")

    preprocessed_path = Path(__file__).parent.parent / "data" / "obesity_preprocessed.csv"
    raw_path = Path(__file__).parent.parent / "data" / "ObesityDataSet.csv"

    df = pd.read_csv(preprocessed_path)
    raw_df = pd.read_csv(raw_path)

    print(f"[INFO] Preprocessed dataset shape: {df.shape}")
    print(f"[INFO] Raw dataset shape: {raw_df.shape}")

    target_col = "NObeyesdad"
    drop_cols = [target_col]
    if "NObeyesdad_Label" in df.columns:
        drop_cols.append("NObeyesdad_Label")

    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target_col]

    # Ensure boolean columns are integers
    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    print(f"[INFO] Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"[INFO] Target classes: {sorted(y.unique().tolist())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set: {X_test.shape[0]} samples")

    target_le = LabelEncoder()
    y_train_encoded = target_le.fit_transform(y_train)
    y_test_encoded = target_le.transform(y_test)

    print(
        f"[INFO] Target encoding: "
        f"{dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))}"
    )

    print("[INFO] Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_encoded)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)

    print(f"[SUCCESS] Accuracy: {accuracy:.4f}")
    report_dict = classification_report(
        y_test_encoded,
        y_pred,
        target_names=target_le.classes_,
        output_dict=True,
    )
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=target_le.classes_))

    numeric_stats = _compute_numeric_stats(raw_df)
    categorical_mapping = _infer_one_hot_mapping(feature_cols)

    model_dict = {
        "model": model,
        "feature_columns": feature_cols,
        "target_le": target_le,
        "numeric_columns": NUMERIC_COLUMNS,
        "numeric_stats": numeric_stats,
        "categorical_mapping": categorical_mapping,
    }

    model_dir = Path(__file__).parent.parent / "models" / "obesity"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "final_model.pkl"
    joblib.dump(model_dict, model_path)

    # === Visualization artifacts ===
    cm = confusion_matrix(y_test_encoded, y_pred)
    confusion_path = model_dir / "confusion_matrix.png"
    _plot_confusion_matrix(cm, list(target_le.classes_), confusion_path)

    report_path = model_dir / "classification_report.csv"
    _save_classification_report(report_dict, report_path)

    importance_csv_path = model_dir / "feature_importances.csv"
    feature_importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feature_importance_df.to_csv(importance_csv_path, index=False, encoding="utf-8-sig")

    importance_plot_path = model_dir / "feature_importances_top20.png"
    _plot_feature_importance(feature_cols, model.feature_importances_, importance_plot_path)

    metrics_path = model_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")

    print(f"[SUCCESS] Model saved to {model_path}")
    print(f"[INFO] Confusion matrix saved to {confusion_path}")
    print(f"[INFO] Classification report saved to {report_path}")
    print(f"[INFO] Feature importance CSV saved to {importance_csv_path}")
    print(f"[INFO] Feature importance plot saved to {importance_plot_path}")
    print(f"[INFO] Metrics summary saved to {metrics_path}")

    return model, accuracy

if __name__ == "__main__":
    train_obesity_model()
