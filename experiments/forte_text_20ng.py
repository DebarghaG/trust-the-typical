#!/usr/bin/env python3
"""
Filename: forte_text_20ng.py
Forte OOD Detection Demo for 20 Newsgroups - Category-based Distribution Shift
Uses 2 categories as in-distribution and 2 different categories as out-of-distribution
"""

import argparse
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import auc, roc_curve

from ..api.forte_text_api import ForteTextOODDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Forte20NGDemo")


def load_20ng_data(
    id_categories: list, ood_categories: list, max_samples_per_category: int = 500
) -> tuple[list[str], list[str], list[str]]:
    """
    Load 20 Newsgroups data with specified ID and OOD categories.

    Args:
        id_categories: List of category names for in-distribution
        ood_categories: List of category names for out-of-distribution
        max_samples_per_category: Maximum samples per category to use

    Returns:
        Tuple of (id_train_texts, id_test_texts, ood_test_texts)
    """
    logger.info("Loading 20 Newsgroups data...")
    logger.info(f"ID categories: {id_categories}")
    logger.info(f"OOD categories: {ood_categories}")

    # Load ID training data
    id_train = fetch_20newsgroups(
        subset="train",
        categories=id_categories,
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )

    # Load ID test data
    id_test = fetch_20newsgroups(
        subset="test",
        categories=id_categories,
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )

    # Load OOD test data
    ood_test = fetch_20newsgroups(
        subset="test",
        categories=ood_categories,
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )

    # Limit samples if specified
    id_train_texts = id_train.data[: max_samples_per_category * len(id_categories)]
    id_test_texts = id_test.data[: max_samples_per_category * len(id_categories) // 2]
    ood_test_texts = ood_test.data[: max_samples_per_category * len(ood_categories) // 2]

    # Clean texts (remove empty or very short)
    id_train_texts = [t for t in id_train_texts if len(t.strip()) > 50]
    id_test_texts = [t for t in id_test_texts if len(t.strip()) > 50]
    ood_test_texts = [t for t in ood_test_texts if len(t.strip()) > 50]

    logger.info(f"Loaded {len(id_train_texts)} ID training texts")
    logger.info(f"Loaded {len(id_test_texts)} ID test texts")
    logger.info(f"Loaded {len(ood_test_texts)} OOD test texts")

    return id_train_texts, id_test_texts, ood_test_texts


def print_sample_texts(id_texts: list[str], ood_texts: list[str], n_samples: int = 2) -> None:
    """Print sample texts from ID and OOD datasets."""
    logger.info("\n=== Sample Texts ===")
    logger.info("\nIn-Distribution Samples:")
    for i, text in enumerate(id_texts[:n_samples]):
        logger.info(f"ID Sample {i + 1}: {text[:200]}...")

    logger.info("\nOut-of-Distribution Samples:")
    for i, text in enumerate(ood_texts[:n_samples]):
        logger.info(f"OOD Sample {i + 1}: {text[:200]}...")


def visualize_results(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    results: dict,
    save_path: str = "forte_20ng_results.png",
) -> None:
    """Create visualization of OOD detection results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Score distributions
    bins = np.linspace(
        min(np.min(id_scores), np.min(ood_scores)), max(np.max(id_scores), np.max(ood_scores)), 30
    )

    axes[0].hist(
        id_scores, bins=bins, alpha=0.7, label="In-Distribution", density=True, color="blue"
    )
    axes[0].hist(
        ood_scores, bins=bins, alpha=0.7, label="Out-of-Distribution", density=True, color="red"
    )

    # Add threshold line
    threshold = np.percentile(id_scores, 5)
    axes[0].axvline(
        x=threshold, color="green", linestyle="--", alpha=0.7, label=f"Threshold ({threshold:.4f})"
    )

    axes[0].legend()
    axes[0].set_title("Score Distributions")
    axes[0].set_xlabel("OOD Score (higher = more in-distribution like)")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.3)

    # ROC curve
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores_combined = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores_combined)
    roc_auc = auc(fpr, tpr)

    axes[1].plot(fpr, tpr, lw=2, label=f"ROC curve (AUROC = {roc_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

    # Mark FPR@95TPR
    idx_95tpr = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95tpr = fpr[idx_95tpr]
    axes[1].scatter(
        fpr_at_95tpr, 0.95, color="red", label=f"FPR@95TPR = {fpr_at_95tpr:.3f}", zorder=5, s=100
    )

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Visualization saved to {save_path}")
    plt.show()


def main(args: argparse.Namespace) -> dict:
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    logger.info(f"Running with configuration: {args}")

    # Define categories
    # Computer topics as in-distribution
    id_categories = ["comp.graphics", "comp.windows.x"]
    # Sports topics as out-of-distribution
    ood_categories = ["rec.sport.baseball", "rec.sport.hockey"]

    # Load data
    logger.info("\n=== Phase 1: Data Loading ===")
    id_train_texts, id_test_texts, ood_test_texts = load_20ng_data(
        id_categories, ood_categories, max_samples_per_category=args.max_samples
    )

    # Print sample texts
    if args.show_samples:
        print_sample_texts(id_test_texts, ood_test_texts)

    # Create and fit detector
    logger.info("\n=== Phase 2: Model Initialization ===")
    # Select text embedding backbones
    model_catalog = {
        "qwen3": "Qwen/Qwen3-Embedding-0.6B",
        "bge-m3": "BAAI/bge-m3",
        "e5": "intfloat/e5-large-v2",
    }
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    model_names = []
    for key in selected_models:
        if key in model_catalog:
            model_names.append((key, model_catalog[key]))
        else:
            logger.warning(f"Unknown model key '{key}'. Skipping.")
    if not model_names:
        model_names = [
            ("qwen3", model_catalog["qwen3"]),
            ("bge-m3", model_catalog["bge-m3"]),
            ("e5", model_catalog["e5"]),
        ]
        logger.info("No valid models specified; using default set: qwen3,bge-m3,e5")

    detector = ForteTextOODDetector(
        batch_size=args.batch_size,
        device=args.device,
        embedding_dir=args.embedding_dir,
        nearest_k=args.nearest_k,
        method=args.method,
        model_names=model_names,
        use_flash_attention=args.use_flash_attention,
    )

    # Fit the detector
    logger.info("\n=== Phase 3: Training ===")
    start_time = time.time()
    detector.fit(id_train_texts, val_split=0.2, random_state=args.seed)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Evaluation
    logger.info("\n=== Phase 4: Evaluation ===")

    # Get scores
    logger.info("Computing scores for ID test data...")
    id_scores = detector._get_ood_scores(id_test_texts, cache_name="id_test")

    logger.info("Computing scores for OOD test data...")
    ood_scores = detector._get_ood_scores(ood_test_texts, cache_name="ood_test")

    # Calculate threshold
    threshold = np.percentile(id_scores, 5)
    logger.info(f"Threshold (5th percentile of ID scores): {threshold:.4f}")

    # Calculate detection accuracy
    id_correct = (id_scores > threshold).mean()
    ood_correct = (ood_scores <= threshold).mean()
    overall_acc = (id_correct * len(id_scores) + ood_correct * len(ood_scores)) / (
        len(id_scores) + len(ood_scores)
    )

    logger.info(f"ID Detection Rate: {id_correct:.4f}")
    logger.info(f"OOD Detection Rate: {ood_correct:.4f}")
    logger.info(f"Overall Accuracy: {overall_acc:.4f}")

    # Full evaluation
    evaluation_start = time.time()
    results = detector.evaluate(id_test_texts, ood_test_texts)
    evaluation_time = time.time() - evaluation_start

    # Print results
    logger.info("\n=== OOD Detection Performance ===")
    logger.info(f"Categories - ID: {id_categories}, OOD: {ood_categories}")
    logger.info(f"Method: {args.method}, Nearest_k: {args.nearest_k}")
    logger.info(f"AUROC: {results['AUROC']:.4f}")
    logger.info(f"FPR@95TPR: {results['FPR@95TPR']:.4f}")
    logger.info(f"AUPRC: {results['AUPRC']:.4f}")
    logger.info(f"F1 Score: {results['F1']:.4f}")
    logger.info(f"Evaluation time: {evaluation_time:.2f} seconds")

    # Visualize results
    if args.visualize:
        visualize_results(id_scores, ood_scores, results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forte OOD Detection - 20 Newsgroups")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gmm",
        choices=["gmm", "kde", "ocsvm"],
        help="OOD detection method",
    )
    parser.add_argument(
        "--nearest_k", type=int, default=10, help="Number of nearest neighbors for PRDC"
    )
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--show_samples", action="store_true", help="Show sample texts")
    parser.add_argument(
        "--embedding_dir", type=str, default="embeddings_20ng", help="Directory to store embeddings"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen3,bge-m3,e5",
        help="Comma-separated list of text embedding models to use: qwen3,bge-m3,e5",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        default=True,
        help="Use Flash Attention 2 for acceleration",
    )

    args = parser.parse_args()
    main(args)
