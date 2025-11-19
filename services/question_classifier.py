"""
Question classifier inference helper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LOGGER = logging.getLogger(__name__)

HERE = Path(__file__).parent.parent
MODEL_DIR = HERE / "models" / "question_classifier"
CLASSIFIER_THRESHOLD = 0.45

_tokenizer = None
_model = None


def _load_classifier():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return

    if not MODEL_DIR.exists():
        LOGGER.warning(
            "Question classifier directory %s not found. Guardrail will fallback.",
            MODEL_DIR,
        )
        return

    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.eval()
        LOGGER.info(
            "Loaded question classifier with labels: %s",
            _model.config.id2label,
        )
    except Exception as exc:
        LOGGER.error("Failed to load question classifier: %s", exc)
        _tokenizer = None
        _model = None


def is_ready() -> bool:
    if _tokenizer is None or _model is None:
        _load_classifier()
    return _tokenizer is not None and _model is not None


def classify_question(text: str) -> Dict[str, float | str] | None:
    if not text or not isinstance(text, str):
        return None

    if not is_ready():
        return None

    inputs = _tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        score, pred_idx = torch.max(probs, dim=-1)

    label = _model.config.id2label.get(pred_idx.item(), str(pred_idx.item()))
    return {
        "label": label,
        "confidence": float(score.item()),
        "threshold": CLASSIFIER_THRESHOLD,
    }


def is_meal_plan_intent(text: str) -> bool:
    result = classify_question(text)
    if not result:
        return True  # fallback to legacy behavior
    return result["label"] == "meal_plan" and result["confidence"] >= CLASSIFIER_THRESHOLD

