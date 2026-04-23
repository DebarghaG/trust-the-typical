"""
Common evaluation utilities for postprocessors.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def evaluate_binary_classifier(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    """
    Common evaluation function for binary classification of ID vs OOD samples.

    This function computes standard metrics for evaluating OOD detection:
    - AUROC: Area Under the ROC Curve
    - FPR@95TPR: False Positive Rate at 95% True Positive Rate
    - AUPRC: Area Under the Precision-Recall Curve
    - F1: Best F1 score across all thresholds

    Args:
        id_scores: Detection scores for in-distribution samples (higher = more likely ID)
        ood_scores: Detection scores for out-of-distribution samples

    Returns:
        Dictionary with evaluation metrics
    """
    # Create labels: 1 for ID, 0 for OOD
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores_all = np.concatenate([id_scores, ood_scores])

    # AUROC
    auroc = float(roc_auc_score(labels, scores_all))

    # FPR at 95% TPR
    fpr, tpr, _ = roc_curve(labels, scores_all)
    idx = int(np.argmin(np.abs(tpr - 0.95)))
    fpr95 = float(fpr[idx]) if idx < len(fpr) else 1.0

    # AUPRC and best F1
    precision_vals, recall_vals, _ = precision_recall_curve(labels, scores_all)
    auprc = float(average_precision_score(labels, scores_all))
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    f1_score = float(np.max(f1_scores))

    return {
        "AUROC": auroc,
        "FPR@95TPR": fpr95,
        "AUPRC": auprc,
        "F1": f1_score,
    }


def print_score_statistics(id_scores: np.ndarray, ood_scores: np.ndarray) -> None:
    """
    Print statistics about ID and OOD scores.

    Args:
        id_scores: Detection scores for in-distribution samples
        ood_scores: Detection scores for out-of-distribution samples
    """
    print("\nScore Statistics:")
    print(
        f"ID  - Mean: {np.mean(id_scores):.4f}, "
        f"Std: {np.std(id_scores):.4f}, "
        f"Min: {np.min(id_scores):.4f}, "
        f"Max: {np.max(id_scores):.4f}"
    )
    print(
        f"OOD - Mean: {np.mean(ood_scores):.4f}, "
        f"Std: {np.std(ood_scores):.4f}, "
        f"Min: {np.min(ood_scores):.4f}, "
        f"Max: {np.max(ood_scores):.4f}"
    )
