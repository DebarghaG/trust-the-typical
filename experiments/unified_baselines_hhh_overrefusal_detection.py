#!/usr/bin/env python3
"""
Filename: unified_baselines_hhh_overrefusal_detection.py
Unified Baseline Script for HHH Overrefusal Detection

Evaluates on the HHH alignment dataset from HuggingFaceH4/hhh_alignment:
- Harmless: Responses that appropriately balance safety vs helpfulness
- Helpful: Responses that provide assistance when appropriate
- Honest: Responses that are truthful and accurate
- Other: Additional alignment examples

Uses appropriate/helpful responses as the ID distribution and overrefusal responses as OOD.
The goal is to detect when models overrefuse legitimate requests.
"""

import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import time
from typing import Any, Literal

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Import Forte Text API for comparison
from api.forte_text_api import ForteTextOODDetector
from baselines.adascale_text_postprocessor import AdaScaleTextPostprocessor
from baselines.cider_text_postprocessor import CIDERTextPostprocessor
from baselines.fdbd_text_postprocessor import fDBDTextPostprocessor
from baselines.gmm_text_postprocessor import GMMTextPostprocessor
from baselines.llama_guard_logit_text_postprocessor import LlamaGuardLogitTextPostprocessor
from baselines.llama_guard_text_postprocessor import LlamaGuardTextPostprocessor
from baselines.nnguide_text_postprocessor import NNGuideTextPostprocessor
from baselines.openmax_text_postprocessor import OpenMaxTextPostprocessor
from baselines.react_text_postprocessor import ReActTextPostprocessor
from baselines.rmd_simple_postprocessor import SimpleRMDTextPostprocessor
from baselines.vim_text_postprocessor import VIMTextPostprocessor
from baselines.wildguard_text_postprocessor import WildGuardTextPostprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HHHOverrefusalDetectionBaselines")


def load_hhh_data(subset: str = "all", max_samples: int = 1000) -> tuple[list[str], list[str]]:
    """Load HHH alignment data and separate appropriate vs overrefusal responses."""

    import requests

    logger.info(f"Loading HHH alignment data for subset: {subset}...")

    appropriate_texts: list[str] = []  # ID distribution (score=1)
    overrefusal_texts: list[str] = []  # OOD distribution (score=0)

    # Define subsets to load
    subsets_to_load = ["harmless", "helpful", "honest", "other"] if subset == "all" else [subset]

    for subset_name in subsets_to_load:
        try:
            # Load JSON data directly from HuggingFace repository
            url = f"https://huggingface.co/datasets/HuggingFaceH4/hhh_alignment/raw/main/data/{subset_name}/task.json"
            response = requests.get(url)

            if response.status_code != 200:
                logger.warning(f"Could not fetch {subset_name} data: HTTP {response.status_code}")
                continue

            data = response.json()
            examples = data.get("examples", [])
            logger.info(f"Loaded {len(examples)} examples from {subset_name} subset")

            for example in examples:
                input_text = example.get("input", "")
                target_scores = example.get("target_scores", {})

                if isinstance(target_scores, dict) and len(target_scores) >= 2:
                    # Process each response-score pair
                    for response_text, score in target_scores.items():
                        if isinstance(response_text, str) and len(response_text.strip()) > 20:
                            # Create full context: input + response
                            full_text = (
                                f"Input: {input_text.strip()}\nResponse: {response_text.strip()}"
                            )

                            if score == 1:
                                # Appropriate/helpful responses (ID)
                                if len(appropriate_texts) < max_samples:
                                    appropriate_texts.append(full_text)
                            elif score == 0:
                                # Overrefusal/problematic responses (OOD)
                                if len(overrefusal_texts) < max_samples:
                                    overrefusal_texts.append(full_text)

                            # Stop if we have enough samples
                            if (
                                len(appropriate_texts) >= max_samples
                                and len(overrefusal_texts) >= max_samples
                            ):
                                break

                    if (
                        len(appropriate_texts) >= max_samples
                        and len(overrefusal_texts) >= max_samples
                    ):
                        break

        except Exception as e:
            logger.warning(f"Could not load {subset_name} subset: {e}")
            continue

    logger.info(f"Loaded {len(appropriate_texts)} appropriate responses (ID)")
    logger.info(f"Loaded {len(overrefusal_texts)} overrefusal responses (OOD)")

    return appropriate_texts, overrefusal_texts


def split_data_for_training(
    appropriate_texts: list[str], overrefusal_texts: list[str], train_ratio: float = 0.7
) -> tuple[list[str], list[str], list[str]]:
    """Split data into training (appropriate only) and test sets."""

    # Split appropriate texts for training and testing
    train_size = int(len(appropriate_texts) * train_ratio)
    appropriate_train = appropriate_texts[:train_size]
    appropriate_test = appropriate_texts[train_size:]

    # Use all overrefusal texts for testing
    overrefusal_test = overrefusal_texts

    logger.info(f"Training set: {len(appropriate_train)} appropriate responses")
    logger.info(
        f"Test set: {len(appropriate_test)} appropriate + {len(overrefusal_test)} overrefusal responses"
    )

    return appropriate_train, appropriate_test, overrefusal_test


def create_postprocessor(method: str, args: argparse.Namespace) -> Any:
    """Create postprocessor based on method."""
    embedding_dir = f"embeddings_{method.lower()}_hhh_overrefusal"

    # Handle Forte methods
    if method.lower().startswith("forte_"):
        detector_type_str = method.lower().replace("forte_", "")
        detector_type: Literal["gmm", "kde", "ocsvm"]
        if detector_type_str == "gmm":
            detector_type = "gmm"
        elif detector_type_str == "kde":
            detector_type = "kde"
        elif detector_type_str == "ocsvm":
            detector_type = "ocsvm"
        else:
            raise ValueError(f"Unsupported Forte method: {detector_type_str}")

        # Convert forte_models string to model_names format
        model_names = None
        if args.forte_models:
            model_mapping = {
                "qwen3": ("qwen3", "Qwen/Qwen3-Embedding-0.6B"),
                "bge-m3": ("bge-m3", "BAAI/bge-m3"),
                "e5": ("e5", "intfloat/e5-large-v2"),
            }
            model_names = [
                model_mapping[m.strip()]
                for m in args.forte_models.split(",")
                if m.strip() in model_mapping
            ]

        return ForteTextOODDetector(
            method=detector_type,
            device=args.device,
            model_names=model_names,
            embedding_dir=embedding_dir,
            use_flash_attention=False,  # Disable FP16 to avoid torch.cdist issues
        )
    elif method.lower() == "rmd":
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
    elif method.lower() == "llamaguard":
        return LlamaGuardTextPostprocessor(
            model_id=args.llama_guard_model_id,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            max_new_tokens=args.llama_guard_max_tokens,
            temperature=args.llama_guard_temperature,
        )
    elif method.lower() == "llamaguard_logit":
        return LlamaGuardLogitTextPostprocessor(
            model_id=args.llama_guard_model_id,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir.replace("hhh_", "hhh_logit_"),
            max_new_tokens=args.llama_guard_max_tokens,
            temperature=args.llama_guard_temperature,
        )
    elif method.lower() == "wildguard":
        return WildGuardTextPostprocessor(
            model_id=args.wildguard_model_id,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir.replace("hhh_", "wildguard_hhh_"),
            max_new_tokens=args.wildguard_max_tokens,
            temperature=args.wildguard_temperature,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")


def run_hhh_evaluation(
    detector: Any,
    appropriate_test_texts: list[str],
    overrefusal_test_texts: list[str],
    method: str,
    subset: str,
    args: argparse.Namespace,
) -> dict:
    """Run evaluation on HHH overrefusal detection."""
    logger.info(f"Evaluating {method} on HHH subset: {subset}...")

    if method.lower().startswith("forte_"):
        # Use Forte API
        results = detector.evaluate(appropriate_test_texts, overrefusal_test_texts)
        return results
    else:
        # Use postprocessor-based methods
        # Get scores for appropriate and overrefusal data
        appropriate_scores = detector.postprocess(
            appropriate_test_texts, cache_name=f"{method.lower()}_hhh_appropriate_test"
        )

        overrefusal_scores = detector.postprocess(
            overrefusal_test_texts, cache_name=f"{method.lower()}_hhh_overrefusal_test"
        )

        # Compute metrics directly from scores to avoid cache reuse
        labels = np.concatenate(
            [np.ones(len(appropriate_scores)), np.zeros(len(overrefusal_scores))]
        )
        scores_all = np.concatenate([appropriate_scores, overrefusal_scores])

        auroc = float(roc_auc_score(labels, scores_all))
        fpr, tpr, _ = roc_curve(labels, scores_all)

        # FPR at 95% TPR
        idx_95tpr = np.argmin(np.abs(tpr - 0.95))
        fpr_at_95tpr = float(fpr[idx_95tpr])

        # AUPRC
        auprc = float(average_precision_score(labels, scores_all))

        # F1 Score (at threshold that maximizes F1)
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores_all)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_f1 = float(np.max(f1_scores))

        results = {"AUROC": auroc, "FPR@95TPR": fpr_at_95tpr, "AUPRC": auprc, "F1": best_f1}
        return results


def main(args: argparse.Namespace) -> dict:
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU total memory: {total_memory:.1f} GB")

    logger.info(f"Running HHH overrefusal detection baselines with configuration: {args}")

    # Load HHH data
    logger.info("\n=== Phase 1: Loading HHH Alignment Data ===")

    # Results storage for different subsets
    all_results = {}
    subsets_to_evaluate = (
        ["harmless", "helpful", "honest", "other"] if args.subset == "all" else [args.subset]
    )

    for subset in subsets_to_evaluate:
        logger.info(f"\n--- Processing {subset.upper()} subset ---")

        # Load data for this subset
        appropriate_texts, overrefusal_texts = load_hhh_data(
            subset=subset, max_samples=args.max_samples_per_subset
        )

        if not appropriate_texts or not overrefusal_texts:
            logger.warning(f"Insufficient data for {subset} subset, skipping...")
            continue

        # Split data for training/testing
        logger.info(f"\n=== Phase 2: Splitting {subset.upper()} Data ===")
        appropriate_train, appropriate_test, overrefusal_test = split_data_for_training(
            appropriate_texts, overrefusal_texts, train_ratio=args.train_ratio
        )

        # Parse methods to run
        methods_to_run = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
        subset_results = {}

        # Run each method on this subset
        for method in methods_to_run:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {method} Method on {subset.upper()}")
            logger.info(f"{'='*60}")

            # Create and setup detector
            logger.info(f"\n=== {method}: Model Initialization ===")
            detector = create_postprocessor(method, args)

            logger.info(f"\n=== {method}: Setup on Appropriate Responses ===")
            start_time = time.time()
            if method.lower().startswith("forte_"):
                # Use Forte API
                detector.fit(appropriate_train)
            else:
                # Use postprocessor-based methods
                detector.setup(appropriate_train, random_state=args.seed)
            setup_time = time.time() - start_time
            logger.info(f"{method} setup completed in {setup_time:.2f} seconds")

            # Evaluate on test set
            logger.info(f"\n=== {method}: HHH Overrefusal Evaluation ===")
            results = run_hhh_evaluation(
                detector, appropriate_test, overrefusal_test, method, subset, args
            )
            subset_results[method] = results

            logger.info(
                f"{method} on {subset}: AUROC={results['AUROC']:.4f}, FPR@95={results['FPR@95TPR']:.4f}"
            )

        all_results[subset] = subset_results

    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info("HHH OVERREFUSAL DETECTION RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Embedding Model: {args.embedding_model}")
    logger.info("")

    # Parse methods for header
    methods_to_run = [m.strip().upper() for m in args.methods.split(",") if m.strip()]

    # Header
    header = f"{'Method':<12}"
    for subset in subsets_to_evaluate:
        if subset in all_results:
            header += f"{subset.upper():<18}"
    logger.info(header)
    logger.info("-" * len(header))

    # Results for each method
    for method in methods_to_run:
        row = f"{method:<12}"
        for subset in subsets_to_evaluate:
            if subset in all_results and method in all_results[subset]:
                auroc = all_results[subset][method]["AUROC"]
                fpr95 = all_results[subset][method]["FPR@95TPR"]
                row += f"{auroc:.3f}/{fpr95:.3f}   "
            else:
                row += f"{'N/A':<18}"
        logger.info(row)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Baselines - HHH Overrefusal Detection")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
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
        default="RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax,Forte_OCSVM,Forte_GMM,LlamaGuard,LlamaGuard_Logit,WildGuard",
        help="Comma-separated list of methods to run (include Forte_GMM, Forte_KDE, Forte_OCSVM for Forte API, LlamaGuard for Llama Guard 3-1B with hardcoded scores, LlamaGuard_Logit for Llama Guard 3-1B with logit-based scoring, WildGuard for AllenAI WildGuard 7B)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=["all", "harmless", "helpful", "honest", "other"],
        help="HHH subset to evaluate on",
    )
    # Forte API parameters
    parser.add_argument(
        "--forte_models",
        type=str,
        default="qwen3,bge-m3,e5",
        help="Comma-separated list of Forte embedding models (qwen3, bge-m3, e5)",
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
    # Llama Guard parameters
    parser.add_argument(
        "--llama_guard_model_id",
        type=str,
        default="meta-llama/Llama-Guard-3-1B",
        help="Llama Guard model ID to use",
    )
    parser.add_argument(
        "--llama_guard_max_tokens",
        type=int,
        default=20,
        help="Maximum new tokens to generate for Llama Guard",
    )
    parser.add_argument(
        "--llama_guard_temperature",
        type=float,
        default=0.1,
        help="Temperature for Llama Guard generation",
    )

    # WildGuard parameters
    parser.add_argument(
        "--wildguard_model_id",
        type=str,
        default="allenai/wildguard",
        help="WildGuard model ID to use",
    )
    parser.add_argument(
        "--wildguard_max_tokens",
        type=int,
        default=32,
        help="Maximum new tokens to generate for WildGuard",
    )
    parser.add_argument(
        "--wildguard_temperature",
        type=float,
        default=0.1,
        help="Temperature for WildGuard generation",
    )

    # Data parameters
    parser.add_argument(
        "--max_samples_per_subset", type=int, default=10000, help="Max samples per HHH subset"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.5,
        help="Ratio of appropriate responses to use for training",
    )
    # General parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
