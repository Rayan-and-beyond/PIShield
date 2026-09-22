"""Evaluation metrics for prompt-injection detection."""
from __future__ import annotations
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)


def evaluate_model(name: str, model, X_test, y_test, train_time: float = None,
                   feature_name: str = "TFIDF") -> dict:
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    metrics = {
        "model": name,
        "features": feature_name,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "macro_f1":  f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "train_time_sec":     train_time,
        "inference_time_sec": inference_time,
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except Exception:
            metrics["roc_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, target_names=["benign", "injection"], output_dict=True, zero_division=0
    )
    return metrics
