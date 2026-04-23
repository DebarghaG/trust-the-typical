#!/usr/bin/env python3
"""
Filename: unified_baselines_math_domain.py
Unified Baseline Script for Mathematical Reasoning Domain Adherence

Supports multiple postprocessor methods:
- Simple RMD: Relative Mahalanobis Distance (simplified)
- VIM: Virtual-logit Matching
- CIDER: K-nearest neighbor distance (without FAISS)
- fDBD: feature Distance-Based Detection
- NNGuide: Nearest Neighbor Guided OOD detection
- ReAct: Rectified Activation thresholding
- GMM: Gaussian Mixture Model
- AdaScale: Adaptive Scaling with gradient perturbations
- OpenMax: Extreme Value Theory with Weibull distributions

All methods use the same experimental setup for fair comparison.
"""

import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset

from baselines.adascale_text_postprocessor import AdaScaleTextPostprocessor
from baselines.cider_text_postprocessor import CIDERTextPostprocessor
from baselines.fdbd_text_postprocessor import fDBDTextPostprocessor
from baselines.gmm_text_postprocessor import GMMTextPostprocessor
from baselines.nnguide_text_postprocessor import NNGuideTextPostprocessor
from baselines.openmax_text_postprocessor import OpenMaxTextPostprocessor
from baselines.react_text_postprocessor import ReActTextPostprocessor
from baselines.rmd_simple_postprocessor import SimpleRMDTextPostprocessor
from baselines.vim_text_postprocessor import VIMTextPostprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MathDomainBaselines")


def load_math_data(
    dataset_name: str = "math", max_samples: int = 500
) -> tuple[list[str], list[str]]:
    """Load mathematical reasoning data (MATH or GSM8K)."""
    logger.info(f"Loading mathematical data from {dataset_name}...")

    train_texts = []
    test_texts = []

    if dataset_name.lower() == "math":
        try:
            # Load MATH dataset
            dataset = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")

            # Format as question-answer pairs for training
            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break
                problem = example.get("problem", "")
                solution = example.get("solution", "")
                if len(problem.strip()) > 20:
                    formatted_text = f"Problem: {problem}\nSolution: {solution}"
                    train_texts.append(formatted_text)

            # Load test split
            test_dataset = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
            for i, example in enumerate(test_dataset):
                if i >= max_samples // 2:
                    break
                problem = example.get("problem", "")
                solution = example.get("solution", "")
                if len(problem.strip()) > 20:
                    formatted_text = f"Problem: {problem}\nSolution: {solution}"
                    test_texts.append(formatted_text)

        except Exception as e:
            logger.warning(f"Could not load MATH dataset: {e}")
            return [], []

    elif dataset_name.lower() == "gsm8k":
        try:
            # Load GSM8K dataset
            train_dataset = load_dataset("openai/gsm8k", "main", split="train")

            for i, example in enumerate(train_dataset):
                if i >= max_samples:
                    break
                question = example.get("question", "")
                answer = example.get("answer", "")
                if len(question.strip()) > 20:
                    formatted_text = f"Problem: {question}\nSolution: {answer}"
                    train_texts.append(formatted_text)

            # Load test split
            test_dataset = load_dataset("openai/gsm8k", "main", split="test")
            for i, example in enumerate(test_dataset):
                if i >= max_samples // 2:
                    break
                question = example.get("question", "")
                answer = example.get("answer", "")
                if len(question.strip()) > 20:
                    formatted_text = f"Problem: {question}\nSolution: {answer}"
                    test_texts.append(formatted_text)

        except Exception as e:
            logger.warning(f"Could not load GSM8K dataset: {e}")
            return [], []

    logger.info(f"Loaded {len(train_texts)} math training texts")
    logger.info(f"Loaded {len(test_texts)} math test texts")

    return train_texts, test_texts


def load_ood_data(ood_type: str, max_samples: int = 300) -> list[str]:
    """Load out-of-distribution data for domain adherence testing."""
    logger.info(f"Loading OOD data: {ood_type}...")

    ood_texts = []

    if ood_type.lower() == "philosophy":
        # Stanford Encyclopedia of Philosophy-style content
        philosophy_prompts = [
            "What is the nature of consciousness and how does it relate to physical reality?",
            "Explain the difference between deontological and consequentialist ethics.",
            "What is the problem of induction as described by David Hume?",
            "How does Kant's categorical imperative work in moral philosophy?",
            "What is the mind-body problem and what are the main proposed solutions?",
            "Explain Plato's theory of Forms and its implications for knowledge.",
            "What is existentialism and how does it differ from essentialism?",
            "Discuss the problem of free will versus determinism.",
            "What is phenomenology and how did Husserl develop this approach?",
            "Explain the philosophical implications of artificial intelligence.",
        ]
        ood_texts.extend(philosophy_prompts * (max_samples // len(philosophy_prompts) + 1))

    elif ood_type.lower() == "literature":
        # Poetry and literature analysis
        literature_prompts = [
            "Analyze the symbolism in Robert Frost's 'The Road Not Taken'.",
            "Write a sonnet about the changing seasons in the style of Shakespeare.",
            "Compare the narrative techniques used in Joyce's Ulysses and Woolf's Mrs. Dalloway.",
            "Explain the themes of alienation in Kafka's 'The Metamorphosis'.",
            "Create a haiku about autumn leaves falling from trees.",
            "Discuss the use of stream of consciousness in modernist literature.",
            "Write a short story in the style of Edgar Allan Poe.",
            "Analyze the character development of Elizabeth Bennet in Pride and Prejudice.",
            "Compare the tragic heroes in Hamlet and Macbeth.",
            "Explain the significance of the green light in The Great Gatsby.",
        ]
        ood_texts.extend(literature_prompts * (max_samples // len(literature_prompts) + 1))

    elif ood_type.lower() == "history":
        # Historical questions and analysis
        history_prompts = [
            "What were the main causes of World War I and how did they interconnect?",
            "Explain the impact of the Industrial Revolution on social structures in Britain.",
            "How did the fall of Constantinople affect European trade routes?",
            "What role did the Silk Road play in cultural exchange between East and West?",
            "Analyze the factors that led to the decline of the Roman Empire.",
            "What were the consequences of the French Revolution for European monarchies?",
            "How did the discovery of the New World change global economics?",
            "Explain the significance of the Renaissance in European intellectual history.",
            "What factors contributed to the rise and fall of the Maya civilization?",
            "How did the Cold War shape international relations in the 20th century?",
        ]
        ood_texts.extend(history_prompts * (max_samples // len(history_prompts) + 1))

    elif ood_type.lower() == "cooking":
        # Recipe instructions and cooking advice
        cooking_prompts = [
            "How do you make a perfect risotto with mushrooms and parmesan cheese?",
            "What's the best way to roast a chicken with herbs and vegetables?",
            "Explain the technique for making homemade pasta from scratch.",
            "How do you properly season and grill a ribeye steak?",
            "What are the steps to make a classic French onion soup?",
            "How do you bake a chocolate cake that's moist and fluffy?",
            "What's the secret to making crispy fried chicken?",
            "How do you prepare a traditional Thanksgiving turkey with stuffing?",
            "What's the best method for making homemade bread from scratch?",
            "How do you create a balanced stir-fry with vegetables and protein?",
        ]
        ood_texts.extend(cooking_prompts * (max_samples // len(cooking_prompts) + 1))

    elif ood_type.lower() == "medical":
        # Clinical guidelines and medical information
        medical_prompts = [
            "What are the symptoms and treatment options for Type 2 diabetes?",
            "Explain the differences between viral and bacterial infections.",
            "What are the risk factors for cardiovascular disease?",
            "How is hypertension diagnosed and managed in clinical practice?",
            "What are the stages of wound healing and how can it be optimized?",
            "Explain the mechanism of action of common antibiotics.",
            "What are the signs and symptoms of clinical depression?",
            "How do vaccines work to prevent infectious diseases?",
            "What are the treatment protocols for acute myocardial infarction?",
            "Explain the pathophysiology of asthma and its management.",
        ]
        ood_texts.extend(medical_prompts * (max_samples // len(medical_prompts) + 1))

    elif ood_type.lower() == "creative_writing":
        # Creative writing prompts
        creative_prompts = [
            "Write a short story about a time traveler who gets stuck in medieval times.",
            "Create a dialogue between two characters meeting for the first time at a coffee shop.",
            "Write a descriptive paragraph about a mysterious abandoned house.",
            "Compose a letter from a character to their future self.",
            "Write a scene where someone discovers they have a hidden talent.",
            "Create a story that begins with 'The last person on Earth sat alone in a room.'",
            "Write a character sketch of someone waiting for a bus in the rain.",
            "Compose a story told entirely through text messages.",
            "Write about a world where colors have different meanings than they do now.",
            "Create a narrative about someone finding an old diary in their attic.",
        ]
        ood_texts.extend(creative_prompts * (max_samples // len(creative_prompts) + 1))

    elif ood_type.lower() == "legal":
        # Legal advice and law questions
        legal_prompts = [
            "What are the key elements required to establish a contract under common law?",
            "Explain the difference between civil and criminal liability.",
            "What constitutes intellectual property and how is it protected?",
            "What are the tenant's rights in a residential lease agreement?",
            "How does the statute of limitations work in personal injury cases?",
            "What is the process for filing a trademark application?",
            "Explain the concept of corporate liability and piercing the corporate veil.",
            "What are the requirements for establishing a valid will?",
            "How do employment laws protect workers from discrimination?",
            "What is the difference between felonies and misdemeanors?",
        ]
        ood_texts.extend(legal_prompts * (max_samples // len(legal_prompts) + 1))

    elif ood_type.lower() == "programming":
        # Programming and coding challenges
        programming_prompts = [
            "Write a Python function to find the longest palindromic substring.",
            "Implement a binary search tree with insert and delete operations.",
            "Create a function to reverse a linked list iteratively and recursively.",
            "Write a program to solve the Two Sum problem efficiently.",
            "Implement a stack data structure using arrays in Java.",
            "Create a function to check if a string has all unique characters.",
            "Write a program to find the kth largest element in an array.",
            "Implement a hash table with collision resolution using chaining.",
            "Create a function to detect cycles in a directed graph.",
            "Write a program to merge two sorted arrays in-place.",
        ]
        ood_texts.extend(programming_prompts * (max_samples // len(programming_prompts) + 1))

    # Limit to max_samples
    ood_texts = ood_texts[:max_samples]
    logger.info(f"Loaded {len(ood_texts)} {ood_type} OOD texts")
    return ood_texts


def create_postprocessor(method: str, args: argparse.Namespace) -> Any:
    """Create postprocessor based on method."""
    embedding_dir = f"embeddings_{method.lower()}_math"

    if method.lower() == "rmd":
        return SimpleRMDTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
        )
    elif method.lower() == "vim":
        return VIMTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            dim=args.vim_dim,
        )
    elif method.lower() == "cider":
        return CIDERTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            K=args.cider_k,
        )
    elif method.lower() == "fdbd":
        return fDBDTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            distance_as_normalizer=args.fdbd_distance_normalizer,
        )
    elif method.lower() == "nnguide":
        return NNGuideTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            K=args.nnguide_k,
            alpha=args.nnguide_alpha,
            min_score=args.nnguide_min_score,
        )
    elif method.lower() == "react":
        return ReActTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            percentile=args.react_percentile,
        )
    elif method.lower() == "gmm":
        return GMMTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            num_clusters=args.gmm_num_clusters,
            feature_type=args.gmm_feature_type,
            reduce_dim_method=args.gmm_reduce_method,
            target_dim=args.gmm_target_dim,
            covariance_type=args.gmm_covariance_type,
            use_sklearn_gmm=args.gmm_use_sklearn,
        )
    elif method.lower() == "adascale":
        # Parse percentile range from string
        percentile_parts = [float(x.strip()) for x in args.adascale_percentile.split(",")]
        if len(percentile_parts) == 2:
            percentile_tuple = (percentile_parts[0], percentile_parts[1])
        else:
            percentile_tuple = (90.0, 99.0)

        return AdaScaleTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            percentile=percentile_tuple,
            k1=args.adascale_k1,
            k2=args.adascale_k2,
            lmbda=args.adascale_lambda,
            o=args.adascale_o,
            num_samples=args.adascale_num_samples,
        )
    elif method.lower() == "openmax":
        return OpenMaxTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            weibull_alpha=args.openmax_alpha,
            weibull_threshold=args.openmax_threshold,
            weibull_tail=args.openmax_tail,
            distance_type=args.openmax_distance_type,
            eu_weight=args.openmax_eu_weight,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")


def visualize_results(
    results_dict: dict[str, dict],
    math_scores_dict: dict[str, np.ndarray],
    ood_scores_dict: dict[str, np.ndarray],
    model_name: str,
    math_dataset: str,
    ood_type: str,
    save_path: str | None = None,
) -> None:
    """Create visualization comparing all methods."""
    if save_path is None:
        save_path = f"math_domain_{math_dataset}_{ood_type}_results.png"

    methods = list(results_dict.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(2, n_methods, figsize=(6 * n_methods, 10))
    if n_methods == 1:
        axes = axes.reshape(2, 1)

    colors = ["purple", "orange", "green", "red", "blue"]

    for i, method in enumerate(methods):
        math_scores = math_scores_dict[method]
        ood_scores = ood_scores_dict[method]

        # Score distributions
        bins = np.linspace(
            min(np.min(math_scores), np.min(ood_scores)),
            max(np.max(math_scores), np.max(ood_scores)),
            30,
        )

        axes[0, i].hist(
            math_scores, bins=bins, alpha=0.7, label="Math (ID)", density=True, color="blue"
        )
        axes[0, i].hist(
            ood_scores, bins=bins, alpha=0.7, label=f"{ood_type} (OOD)", density=True, color="red"
        )

        threshold = np.percentile(math_scores, 5)
        axes[0, i].axvline(
            x=threshold,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Threshold ({threshold:.3f})",
        )

        axes[0, i].legend()
        axes[0, i].set_title(f"{method} - Score Distributions")
        axes[0, i].set_xlabel(f"{method} Score")
        axes[0, i].set_ylabel("Density")
        axes[0, i].grid(True, alpha=0.3)

        # ROC curve
        labels = np.concatenate([np.ones(len(math_scores)), np.zeros(len(ood_scores))])
        scores_combined = np.concatenate([math_scores, ood_scores])
        from sklearn.metrics import auc, roc_curve

        fpr, tpr, _ = roc_curve(labels, scores_combined)
        roc_auc = auc(fpr, tpr)

        axes[1, i].plot(
            fpr, tpr, lw=2, label=f"{method} (AUROC = {roc_auc:.3f})", color=colors[i % len(colors)]
        )
        axes[1, i].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", alpha=0.5)

        idx_95tpr = np.argmin(np.abs(tpr - 0.95))
        fpr_at_95tpr = fpr[idx_95tpr]
        axes[1, i].scatter(
            fpr_at_95tpr, 0.95, color="red", label=f"FPR@95TPR = {fpr_at_95tpr:.3f}", zorder=5, s=80
        )

        axes[1, i].set_xlim([0.0, 1.0])
        axes[1, i].set_ylim([0.0, 1.05])
        axes[1, i].set_xlabel("False Positive Rate")
        axes[1, i].set_ylabel("True Positive Rate")
        axes[1, i].set_title(f"{method} - ROC Curve")
        axes[1, i].legend(loc="lower right")
        axes[1, i].grid(alpha=0.3)

    plt.suptitle(f"Math Domain Adherence: {math_dataset} vs {ood_type}\n{model_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Visualization saved to {save_path}")
    plt.show()


def main(args: argparse.Namespace) -> dict:
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        # Clear GPU cache at start
        torch.cuda.empty_cache()
        # Print GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU total memory: {total_memory:.1f} GB")

    logger.info(f"Running math domain adherence with configuration: {args}")
    logger.info(f"Methods to run: {args.methods}")

    # Load data
    logger.info("\n=== Phase 1: Data Loading ===")

    math_train_texts, math_test_texts = load_math_data(
        dataset_name=args.math_dataset, max_samples=args.max_math_samples
    )

    if not math_train_texts or not math_test_texts:
        logger.error("Failed to load mathematical data. Exiting.")
        return {}

    ood_texts = load_ood_data(ood_type=args.ood_type, max_samples=args.max_ood_samples)

    if not ood_texts:
        logger.error(f"Failed to load {args.ood_type} OOD data. Exiting.")
        return {}

    # Parse methods to run
    methods_to_run = [m.strip().upper() for m in args.methods.split(",") if m.strip()]

    # Run each method
    results_dict = {}
    math_scores_dict = {}
    ood_scores_dict = {}

    for method in methods_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {method} Method")
        logger.info(f"{'='*60}")

        # Create postprocessor
        logger.info(f"\n=== {method}: Model Initialization ===")
        postprocessor = create_postprocessor(method, args)

        # Setup postprocessor
        logger.info(f"\n=== {method}: Setup on Math Data ===")
        start_time = time.time()
        postprocessor.setup(math_train_texts, random_state=args.seed)
        setup_time = time.time() - start_time
        logger.info(f"{method} setup completed in {setup_time:.2f} seconds")

        # Evaluation
        logger.info(f"\n=== {method}: Evaluation ===")

        # Get scores
        logger.info(f"Computing {method} scores for math test data...")
        math_scores = postprocessor.postprocess(
            math_test_texts, cache_name=f"{method.lower()}_math_test_{args.math_dataset}"
        )

        logger.info(f"Computing {method} scores for {args.ood_type} data...")
        ood_scores = postprocessor.postprocess(
            ood_texts, cache_name=f"{method.lower()}_ood_{args.ood_type}"
        )

        # Calculate threshold and accuracy
        threshold = np.percentile(math_scores, 5)
        math_correct = (math_scores > threshold).mean()
        ood_correct = (ood_scores <= threshold).mean()
        overall_acc = (math_correct * len(math_scores) + ood_correct * len(ood_scores)) / (
            len(math_scores) + len(ood_scores)
        )

        logger.info(f"{method} - Threshold (5th percentile): {threshold:.4f}")
        logger.info(f"{method} - Math Detection Rate: {math_correct:.4f}")
        logger.info(f"{method} - OOD Detection Rate: {ood_correct:.4f}")
        logger.info(f"{method} - Overall Accuracy: {overall_acc:.4f}")

        # Full evaluation
        evaluation_start = time.time()
        results = postprocessor.evaluate(math_test_texts, ood_texts)
        evaluation_time = time.time() - evaluation_start

        # Store results
        results_dict[method] = results
        math_scores_dict[method] = math_scores
        ood_scores_dict[method] = ood_scores

        # Print method results
        logger.info(f"\n=== {method} Performance ===")
        logger.info(f"AUROC: {results['AUROC']:.4f}")
        logger.info(f"FPR@95TPR: {results['FPR@95TPR']:.4f}")
        logger.info(f"AUPRC: {results['AUPRC']:.4f}")
        logger.info(f"F1 Score: {results['F1']:.4f}")
        logger.info(f"Evaluation time: {evaluation_time:.2f} seconds")

        # Distribution analysis
        logger.info(f"\n=== {method} Distribution Analysis ===")
        logger.info(
            f"Math scores - Mean: {np.mean(math_scores):.4f}, Std: {np.std(math_scores):.4f}"
        )
        logger.info(f"OOD scores - Mean: {np.mean(ood_scores):.4f}, Std: {np.std(ood_scores):.4f}")
        logger.info(
            f"Distribution separation: {abs(np.mean(math_scores) - np.mean(ood_scores)):.4f}"
        )

    # Summary comparison
    logger.info(f"\n{'='*60}")
    logger.info("MATH DOMAIN ADHERENCE RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Embedding Model: {args.embedding_model}")
    logger.info(f"Math dataset: {args.math_dataset}")
    logger.info(f"OOD type: {args.ood_type}")
    logger.info("")
    logger.info(f"{'Method':<10} {'AUROC':<8} {'FPR@95':<8} {'AUPRC':<8} {'F1':<8}")
    logger.info("-" * 50)

    for method in methods_to_run:
        results = results_dict[method]
        logger.info(
            f"{method:<10} {results['AUROC']:<8.4f} {results['FPR@95TPR']:<8.4f} {results['AUPRC']:<8.4f} {results['F1']:<8.4f}"
        )

    # Visualize results
    if args.visualize and len(methods_to_run) > 0:
        visualize_results(
            results_dict,
            math_scores_dict,
            ood_scores_dict,
            args.embedding_model,
            args.math_dataset,
            args.ood_type,
        )

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Baselines - Math Domain Adherence")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace embedding model name",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax",
        help="Comma-separated list of methods to run",
    )
    parser.add_argument(
        "--math_dataset",
        type=str,
        default="math",
        choices=["math", "gsm8k"],
        help="Mathematical reasoning dataset to use as safe distribution",
    )
    parser.add_argument(
        "--ood_type",
        type=str,
        default="philosophy",
        choices=[
            "philosophy",
            "literature",
            "history",
            "cooking",
            "medical",
            "creative_writing",
            "legal",
            "programming",
        ],
        help="Type of out-of-distribution data",
    )
    # VIM parameters
    parser.add_argument(
        "--vim_dim", type=int, default=512, help="Dimension of principal subspace for VIM"
    )
    # CIDER parameters
    parser.add_argument(
        "--cider_k", type=int, default=5, help="K value for CIDER nearest neighbors"
    )
    # fDBD parameters
    parser.add_argument(
        "--fdbd_distance_normalizer",
        action="store_true",
        default=True,
        help="Use distance to mean for fDBD normalization",
    )
    # NNGuide parameters
    parser.add_argument(
        "--nnguide_k", type=int, default=100, help="K value for NNGuide nearest neighbors"
    )
    parser.add_argument(
        "--nnguide_alpha",
        type=float,
        default=1.0,
        help="Fraction of training data to use for NNGuide bank",
    )
    parser.add_argument(
        "--nnguide_min_score",
        action="store_true",
        help="Use min (vs mean) similarity score for NNGuide",
    )
    # ReAct parameters
    parser.add_argument(
        "--react_percentile",
        type=float,
        default=90.0,
        help="Percentile for ReAct activation threshold",
    )
    # GMM parameters
    parser.add_argument("--gmm_num_clusters", type=int, default=8, help="Number of GMM components")
    parser.add_argument(
        "--gmm_feature_type",
        type=str,
        default="penultimate",
        choices=["penultimate", "raw", "embedding", "norm", "normalized"],
        help="Feature processing type for GMM",
    )
    parser.add_argument(
        "--gmm_reduce_method",
        type=str,
        default="none",
        choices=["none", "pca", "lda"],
        help="Dimensionality reduction method for GMM",
    )
    parser.add_argument(
        "--gmm_target_dim", type=int, default=50, help="Target dimension after reduction for GMM"
    )
    parser.add_argument(
        "--gmm_covariance_type",
        type=str,
        default="tied",
        choices=["tied", "full", "diag", "spherical"],
        help="GMM covariance type",
    )
    parser.add_argument(
        "--gmm_use_sklearn",
        action="store_true",
        default=True,
        help="Use sklearn GMM implementation",
    )
    # AdaScale parameters
    parser.add_argument(
        "--adascale_percentile",
        type=str,
        default="90.0,99.0",
        help="AdaScale percentile range (min,max)",
    )
    parser.add_argument(
        "--adascale_k1", type=float, default=50.0, help="AdaScale k1 percentage for correction term"
    )
    parser.add_argument(
        "--adascale_k2", type=float, default=50.0, help="AdaScale k2 percentage for shift term"
    )
    parser.add_argument(
        "--adascale_lambda",
        type=float,
        default=1.0,
        help="AdaScale lambda weight for shift term combination",
    )
    parser.add_argument(
        "--adascale_o", type=float, default=0.1, help="AdaScale perturbation strength"
    )
    parser.add_argument(
        "--adascale_num_samples",
        type=int,
        default=None,
        help="AdaScale number of samples for setup",
    )
    # OpenMax parameters
    parser.add_argument(
        "--openmax_alpha", type=int, default=3, help="OpenMax number of top classes to modify"
    )
    parser.add_argument(
        "--openmax_threshold",
        type=float,
        default=0.9,
        help="OpenMax threshold for known vs unknown classification",
    )
    parser.add_argument(
        "--openmax_tail", type=int, default=20, help="OpenMax tail size for Weibull fitting"
    )
    parser.add_argument(
        "--openmax_distance_type",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine", "eucos"],
        help="OpenMax distance metric",
    )
    parser.add_argument(
        "--openmax_eu_weight",
        type=float,
        default=0.5,
        help="OpenMax euclidean weight for eucos distance",
    )
    # Data parameters
    parser.add_argument("--max_math_samples", type=int, default=500, help="Max math samples")
    parser.add_argument("--max_ood_samples", type=int, default=300, help="Max OOD samples")
    # General parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")

    args = parser.parse_args()
    main(args)
