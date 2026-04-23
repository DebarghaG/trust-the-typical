#!/usr/bin/env python3
"""
Filename: unified_baselines_orbench_overrefusal_detection.py
Unified Baseline Script for OR-Bench Overrefusal Detection

Evaluates on the OR-Bench dataset from bench-llm/or-bench:
- or-bench-80k: 80,000 safe prompts that may trigger overrefusal (main benchmark)
- or-bench-hard-1k: 1,000 challenging prompts that are hard even for SOTA models
- or-bench-toxic: 600 toxic prompts to prevent indiscriminate responses

Uses safe prompts as the ID distribution and toxic prompts as OOD.
The goal is to detect when models overrefuse legitimate requests vs properly refuse toxic ones.
"""

import contextlib
import gc
import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
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
from baselines.duoguard_text_postprocessor import DuoGuardTextPostprocessor
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

# Try to import VLLM versions, fall back gracefully if not available
MDJudgeVLLMTextPostprocessor: Any = None
try:
    from baselines.mdjudge_vllm_text_postprocessor import (
        MDJudgeVLLMTextPostprocessor as MDJudgeVLLMTextPostprocessor_class,
    )

    MDJudgeVLLMTextPostprocessor = MDJudgeVLLMTextPostprocessor_class
    MDJUDGE_VLLM_AVAILABLE = True
except Exception as e:
    MDJUDGE_VLLM_AVAILABLE = False
    print(f"Warning: MD-Judge VLLM not available: {e}")

PolyGuardVLLMTextPostprocessor: Any = None
try:
    from baselines.polyguard_vllm_text_postprocessor import (
        PolyGuardVLLMTextPostprocessor as PolyGuardVLLMTextPostprocessor_class,
    )

    PolyGuardVLLMTextPostprocessor = PolyGuardVLLMTextPostprocessor_class
    POLYGUARD_VLLM_AVAILABLE = True
except Exception as e:
    POLYGUARD_VLLM_AVAILABLE = False
    print(f"Warning: PolyGuard VLLM not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ORBenchOverrefusalDetectionBaselines")


def cleanup_vllm_memory(model_instance: Any = None, method_name: str = "") -> None:
    """
    Comprehensive cleanup for vLLM models to free GPU memory.
    Based on latest working solution from vLLM issue #1908.
    """
    try:
        logger.info(f"Starting vLLM memory cleanup for {method_name}...")

        # Set tokenizers parallelism to false to avoid deadlock warnings
        import os

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Set PyTorch CUDA memory allocation to avoid fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Try to import necessary cleanup functions
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        # Cleanup model-specific components
        if model_instance is not None:
            try:
                # For vLLM v0.6+ compatibility - try different paths to the model executor
                if hasattr(model_instance, "llm_engine"):
                    # Method 1: Try the standard model_executor path
                    if hasattr(model_instance.llm_engine, "model_executor"):
                        logger.info("Cleaning up model_executor...")
                        del model_instance.llm_engine.model_executor

                    # Method 2: Try the driver_worker path for single GPU setups
                    elif hasattr(model_instance.llm_engine, "driver_worker"):
                        logger.info("Cleaning up driver_worker...")
                        del model_instance.llm_engine.driver_worker

                    # Method 3: For newer versions, try engine_core
                    elif hasattr(model_instance.llm_engine, "engine_core"):
                        logger.info("Shutting down engine_core...")
                        model_instance.llm_engine.engine_core.shutdown()

                # Clean up the main model instance
                logger.info("Deleting model instance...")
                del model_instance

            except Exception as e:
                logger.warning(f"Error during model-specific cleanup: {e}")

        # Destroy parallel state
        logger.info("Destroying model parallel state...")
        destroy_model_parallel()

        # Destroy distributed environment
        logger.info("Destroying distributed environment...")
        destroy_distributed_environment()

        # Clean up torch distributed if it exists
        with contextlib.suppress(Exception):
            if torch.distributed.is_initialized():
                logger.info("Destroying torch distributed process group...")
                torch.distributed.destroy_process_group()

        # Python garbage collection
        logger.info("Running garbage collection...")
        gc.collect()

        # Clear CUDA cache more aggressively
        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Try to force release of reserved memory
            try:
                logger.info("Attempting to reset CUDA context...")
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

                # Try multiple rounds of cleanup
                for _ in range(3):
                    torch.cuda.empty_cache()
                    gc.collect()

                # Force memory pool to release cached memory
                if hasattr(torch.cuda, "memory_pool"):
                    torch.cuda.memory_pool().empty_cache()

                # Try to reset CUDA context (aggressive but effective)
                try:
                    import ctypes
                    import ctypes.util

                    cuda = ctypes.CDLL(ctypes.util.find_library("cuda"))
                    result = cuda.cuDeviceReset(0)
                    if result == 0:
                        logger.info("Successfully reset CUDA device")
                except Exception as reset_e:
                    logger.debug(f"CUDA device reset failed: {reset_e}")

            except Exception as cleanup_e:
                logger.warning(f"Aggressive cleanup failed: {cleanup_e}")

        # Shutdown Ray if it's running
        try:
            import ray

            if ray.is_initialized():
                logger.info("Shutting down Ray...")
                ray.shutdown()
        except Exception as e:
            logger.debug(f"Ray cleanup not needed or failed: {e}")

        logger.info(f"vLLM memory cleanup completed for {method_name}")

        # Log memory status if available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.info(
                f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB"
            )

    except Exception as e:
        logger.error(f"Error during vLLM cleanup for {method_name}: {e}")
        # Fallback: basic cleanup
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as fallback_e:
            logger.error(f"Even fallback cleanup failed: {fallback_e}")


def load_orbench_data(
    subset: str = "or-bench-80k", max_samples: int = 10000
) -> tuple[list[str], list[str], dict]:
    """Load OR-Bench data and separate safe vs toxic prompts."""
    logger.info(f"Loading OR-Bench data for subset: {subset}...")

    safe_prompts: list[str] = []  # ID distribution (safe prompts that may trigger overrefusal)
    toxic_prompts: list[str] = []  # OOD distribution (toxic prompts that should be refused)

    dataset_name = "bench-llm/or-bench"

    try:
        # Load the specified subset
        data = load_dataset(dataset_name, subset, split="train")
        logger.info(f"Loaded {len(data)} examples from {subset}")

        # Track category distribution
        category_counts: dict[str, int] = {}

        for example in data:
            prompt = example.get("prompt", "")
            category = example.get("category", "unknown")

            if len(prompt.strip()) < 20:  # Skip very short prompts
                continue

            # Count categories
            category_counts[category] = category_counts.get(category, 0) + 1

            # For OR-Bench, the main dataset contains safe prompts that may trigger overrefusal
            if subset in ["or-bench-80k", "or-bench-hard-1k"]:
                # These are safe prompts (ID distribution)
                if len(safe_prompts) < max_samples:
                    safe_prompts.append(prompt.strip())
            elif subset == "or-bench-toxic":
                # These are toxic prompts (OOD distribution)
                if len(toxic_prompts) < max_samples:
                    toxic_prompts.append(prompt.strip())

    except Exception as e:
        logger.error(f"Could not load OR-Bench subset {subset}: {e}")
        return [], [], {}

    logger.info(f"Loaded {len(safe_prompts)} safe prompts (ID)")
    logger.info(f"Loaded {len(toxic_prompts)} toxic prompts (OOD)")
    logger.info(f"Category distribution: {category_counts}")

    return safe_prompts, toxic_prompts, category_counts


def load_combined_orbench_data(
    max_safe_samples: int = 5000, max_toxic_samples: int = 600
) -> tuple[list[str], list[str], list[str]]:
    """Load combined OR-Bench data: safe prompts for training, safe + toxic for testing."""
    logger.info("Loading combined OR-Bench data...")

    # Load safe prompts from 80k dataset (main benchmark)
    safe_prompts_80k, _, _ = load_orbench_data("or-bench-80k", max_safe_samples)

    # Load hard prompts if we have room
    remaining_safe = max(0, max_safe_samples - len(safe_prompts_80k))
    if remaining_safe > 0:
        safe_prompts_hard, _, _ = load_orbench_data("or-bench-hard-1k", remaining_safe)
        safe_prompts_80k.extend(safe_prompts_hard)

    # Load toxic prompts for OOD evaluation
    _, toxic_prompts, _ = load_orbench_data("or-bench-toxic", max_toxic_samples)

    # Split safe prompts into train/test
    train_size = int(len(safe_prompts_80k) * 0.7)  # Use 70% for training
    safe_train = safe_prompts_80k[:train_size]
    safe_test = safe_prompts_80k[train_size:]

    logger.info(f"Training set: {len(safe_train)} safe prompts")
    logger.info(f"Test set: {len(safe_test)} safe + {len(toxic_prompts)} toxic prompts")

    return safe_train, safe_test, toxic_prompts


def create_postprocessor(method: str, args: argparse.Namespace) -> Any:
    """Create postprocessor based on method."""
    embedding_dir = f"embeddings_{method.lower()}_orbench_overrefusal"

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
            embedding_dir=embedding_dir.replace("orbench_", "orbench_logit_"),
            max_new_tokens=args.llama_guard_max_tokens,
            temperature=args.llama_guard_temperature,
        )
    elif method.lower() == "duoguard":
        return DuoGuardTextPostprocessor(
            model_id=args.duoguard_model_id,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            threshold=args.duoguard_threshold,
            max_length=args.duoguard_max_length,
        )
    elif method.lower() == "wildguard":
        return WildGuardTextPostprocessor(
            model_id=args.wildguard_model_id,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir.replace("orbench_", "wildguard_orbench_"),
            max_new_tokens=args.wildguard_max_tokens,
            temperature=args.wildguard_temperature,
        )
    elif method.lower() == "mdjudge_vllm":
        if not MDJUDGE_VLLM_AVAILABLE:
            raise RuntimeError(
                "MD-Judge VLLM is not available. This may be due to:\n"
                "1. VLLM not installed: pip install vllm\n"
                "2. CUDA/GPU compatibility issues\n"
                "3. Library loading issues\n"
                "Consider using 'MDJudge' (transformers version) instead."
            )
        return MDJudgeVLLMTextPostprocessor(
            model_id=args.mdjudge_vllm_model_id,
            batch_size=args.mdjudge_vllm_batch_size,
            embedding_dir=embedding_dir,
            max_tokens=args.mdjudge_vllm_max_tokens,
            temperature=args.mdjudge_vllm_temperature,
            tensor_parallel_size=args.mdjudge_vllm_tensor_parallel_size,
            gpu_memory_utilization=args.mdjudge_vllm_gpu_memory_utilization,
            max_model_len=args.mdjudge_vllm_max_model_len,
            quantization=args.mdjudge_vllm_quantization,
            enforce_eager=args.mdjudge_vllm_enforce_eager,
        )
    elif method.lower() == "polyguard_vllm":
        if not POLYGUARD_VLLM_AVAILABLE:
            raise RuntimeError(
                "PolyGuard VLLM is not available. This may be due to:\n"
                "1. VLLM not installed: pip install vllm\n"
                "2. CUDA/GPU compatibility issues\n"
                "3. Library loading issues\n"
                "Please ensure VLLM is properly installed."
            )
        return PolyGuardVLLMTextPostprocessor(
            model_id=args.polyguard_vllm_model_id,
            batch_size=args.polyguard_vllm_batch_size,
            embedding_dir=embedding_dir,
            max_tokens=args.polyguard_vllm_max_tokens,
            temperature=args.polyguard_vllm_temperature,
            tensor_parallel_size=args.polyguard_vllm_tensor_parallel_size,
            gpu_memory_utilization=args.polyguard_vllm_gpu_memory_utilization,
            max_model_len=args.polyguard_vllm_max_model_len,
            quantization=args.polyguard_vllm_quantization,
            enforce_eager=args.polyguard_vllm_enforce_eager,
            use_dummy_response=args.polyguard_vllm_use_dummy_response,
            dummy_response=args.polyguard_vllm_dummy_response,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")


def calculate_rejection_rates(
    scores: np.ndarray, labels: np.ndarray, thresholds: list[float] | None = None
) -> dict:
    """Calculate rejection rates at different thresholds for OR-Bench style table.

    Args:
        scores: OOD scores (higher = more likely OOD/toxic)
        labels: 1 for safe (ID), 0 for toxic (OOD)
        thresholds: List of thresholds to evaluate

    Returns:
        Dictionary with rejection rates at different thresholds
    """
    if thresholds is None:
        # Use percentiles of the score distribution as thresholds
        thresholds = [np.percentile(scores, p) for p in [10, 25, 50, 75, 90, 95, 99]]

    rejection_results = {}

    for i, threshold in enumerate(thresholds):
        # Predictions: reject if score > threshold
        predictions = scores > threshold

        # Calculate rejection rates
        safe_mask = labels == 1
        toxic_mask = labels == 0

        safe_rejection_rate = np.mean(predictions[safe_mask]) if np.sum(safe_mask) > 0 else 0.0
        toxic_rejection_rate = np.mean(predictions[toxic_mask]) if np.sum(toxic_mask) > 0 else 0.0

        # Overall accuracy (safe accepted + toxic rejected)
        accuracy = np.mean((predictions == 0) & safe_mask) + np.mean(predictions & toxic_mask)

        rejection_results[f"threshold_{i+1}"] = {
            "threshold": float(threshold),
            "safe_rejection_rate": float(safe_rejection_rate),
            "toxic_rejection_rate": float(toxic_rejection_rate),
            "accuracy": float(accuracy),
        }

    # Also calculate at optimal F1 threshold
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    optimal_threshold = float(pr_thresholds[best_f1_idx])

    predictions_optimal = scores > optimal_threshold
    safe_rejection_optimal = (
        np.mean(predictions_optimal[safe_mask]) if np.sum(safe_mask) > 0 else 0.0
    )
    toxic_rejection_optimal = (
        np.mean(predictions_optimal[toxic_mask]) if np.sum(toxic_mask) > 0 else 0.0
    )
    accuracy_optimal = np.mean((predictions_optimal == 0) & safe_mask) + np.mean(
        predictions_optimal & toxic_mask
    )

    rejection_results["optimal_f1"] = {
        "threshold": optimal_threshold,
        "safe_rejection_rate": float(safe_rejection_optimal),
        "toxic_rejection_rate": float(toxic_rejection_optimal),
        "accuracy": float(accuracy_optimal),
    }

    return rejection_results


def run_orbench_evaluation(
    detector: Any,
    safe_test_prompts: list[str],
    toxic_test_prompts: list[str],
    method: str,
    args: argparse.Namespace,
) -> dict:
    """Run evaluation on OR-Bench overrefusal detection."""
    logger.info(f"Evaluating {method} on OR-Bench...")

    if method.lower().startswith("forte_"):
        # Use Forte API
        results = detector.evaluate(safe_test_prompts, toxic_test_prompts)

        # Add rejection rates for Forte methods
        if "scores" in results or "safe_scores" in results:
            # Get scores from Forte results
            if "safe_scores" in results and "toxic_scores" in results:
                safe_scores = np.array(results["safe_scores"])
                toxic_scores = np.array(results["toxic_scores"])
                labels = np.concatenate([np.ones(len(safe_scores)), np.zeros(len(toxic_scores))])
                scores_all = np.concatenate([safe_scores, toxic_scores])
            elif "scores" in results and "labels" in results:
                scores_all = np.array(results["scores"])
                labels = np.array(results["labels"])
            else:
                # Fallback: try to extract from other fields
                scores_all = None
                labels = None

            if scores_all is not None and labels is not None:
                rejection_rates = calculate_rejection_rates(scores_all, labels)
                results["rejection_rates"] = rejection_rates

        return results
    else:
        # Use postprocessor-based methods

        # Clean up memory before VLLM-based model inference
        if method.lower() in ["mdjudge_vllm", "polyguard_vllm"]:
            logger.info(f"Cleaning up memory before {method} inference...")
            cleanup_vllm_memory(method_name=f"{method}_pre_inference")

        # Get scores for safe and toxic data
        safe_scores = detector.postprocess(
            safe_test_prompts, cache_name=f"{method.lower()}_orbench_safe_test"
        )

        toxic_scores = detector.postprocess(
            toxic_test_prompts, cache_name=f"{method.lower()}_orbench_toxic_test"
        )

        # Compute metrics directly from scores to avoid cache reuse
        labels = np.concatenate([np.ones(len(safe_scores)), np.zeros(len(toxic_scores))])
        scores_all = np.concatenate([safe_scores, toxic_scores])

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

        # Calculate rejection rates for OR-Bench style table
        rejection_rates = calculate_rejection_rates(scores_all, labels)

        results = {
            "AUROC": auroc,
            "FPR@95TPR": fpr_at_95tpr,
            "AUPRC": auprc,
            "F1": best_f1,
            "rejection_rates": rejection_rates,
        }
        return results


def save_orbench_results_to_csv(
    all_results: dict,
    embedding_model: str,
    num_safe_train: int,
    num_safe_test: int,
    num_toxic_test: int,
    args: argparse.Namespace | None = None,
) -> str:
    """Save OR-Bench results to CSV file in results directory."""
    from datetime import datetime

    # Create a list to store all rows
    rows = []

    for method, results in all_results.items():
        # Basic metrics row
        base_row = {
            "Method": method,
            "Embedding_Model": embedding_model,
            "Safe_Train_Samples": num_safe_train,
            "Safe_Test_Samples": num_safe_test,
            "Toxic_Test_Samples": num_toxic_test,
            "AUROC": results["AUROC"],
            "FPR@95TPR": results["FPR@95TPR"],
            "AUPRC": results["AUPRC"],
            "F1": results["F1"],
        }

        # Add rejection rate data if available
        if "rejection_rates" in results:
            # Add optimal F1 threshold data
            optimal_data = results["rejection_rates"].get("optimal_f1", {})
            base_row.update(
                {
                    "Optimal_Threshold": optimal_data.get("threshold", None),
                    "Optimal_Safe_Rejection_Rate": optimal_data.get("safe_rejection_rate", None),
                    "Optimal_Toxic_Rejection_Rate": optimal_data.get("toxic_rejection_rate", None),
                    "Optimal_Accuracy": optimal_data.get("accuracy", None),
                }
            )

            # Add threshold-based rejection rates
            rejection_rates = results["rejection_rates"]
            threshold_keys = [k for k in rejection_rates.keys() if k.startswith("threshold_")]

            # Add data for each threshold (we'll include first few thresholds)
            for i, key in enumerate(sorted(threshold_keys)[:3]):  # Include first 3 thresholds
                threshold_data = rejection_rates[key]
                base_row.update(
                    {
                        f"Threshold_{i+1}": threshold_data.get("threshold", None),
                        f"Threshold_{i+1}_Safe_Rejection": threshold_data.get(
                            "safe_rejection_rate", None
                        ),
                        f"Threshold_{i+1}_Toxic_Rejection": threshold_data.get(
                            "toxic_rejection_rate", None
                        ),
                        f"Threshold_{i+1}_Accuracy": threshold_data.get("accuracy", None),
                    }
                )

        rows.append(base_row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"orbench_overrefusal_detection_results_{timestamp}.csv"
    filepath = f"results/{filename}"

    # Save main results to CSV
    df.to_csv(filepath, index=False)
    logger.info(f"Results saved to {filepath}")

    # Save experiment parameters to a separate metadata file if args provided
    if args is not None:
        metadata_filepath = filepath.replace(".csv", "_metadata.json")
        import json

        # Convert args namespace to dictionary
        args_dict = vars(args)

        # Add additional metadata
        metadata = {
            "timestamp": timestamp,
            "results_file": filename,
            "dataset": "bench-llm/or-bench",
            "embedding_model": embedding_model,
            "num_safe_train": num_safe_train,
            "num_safe_test": num_safe_test,
            "num_toxic_test": num_toxic_test,
            "experiment_parameters": args_dict,
        }

        # Save metadata to JSON file
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Experiment parameters saved to {metadata_filepath}")

        # Also add key parameters directly to each row in the DataFrame
        df["Seed"] = args.seed
        df["Batch_Size"] = args.batch_size
        df["Max_Safe_Samples"] = args.max_safe_samples
        df["Max_Toxic_Samples"] = args.max_toxic_samples
        df["Num_Epochs"] = args.num_epochs

        # Save again with the additional parameter columns
        df.to_csv(filepath, index=False)

    return filepath


def main(args: argparse.Namespace) -> dict:
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU total memory: {total_memory:.1f} GB")

    logger.info(f"Running OR-Bench overrefusal detection baselines with configuration: {args}")

    # Load OR-Bench data
    logger.info("\n=== Phase 1: Loading OR-Bench Data ===")
    safe_train, safe_test, toxic_test = load_combined_orbench_data(
        max_safe_samples=args.max_safe_samples, max_toxic_samples=args.max_toxic_samples
    )

    if not safe_train or not safe_test or not toxic_test:
        logger.error("Insufficient data loaded. Exiting.")
        return {}

    # Parse methods to run
    methods_to_run = [m.strip().upper() for m in args.methods.split(",") if m.strip()]

    # Results storage
    all_results = {}

    # Run each method
    for method in methods_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {method} Method on OR-Bench")
        logger.info(f"{'='*60}")

        # Create and setup detector
        logger.info(f"\n=== {method}: Model Initialization ===")

        # Clean up memory before VLLM-based models
        if method.lower() in ["mdjudge_vllm", "polyguard_vllm"]:
            logger.info(f"Cleaning up memory before initializing {method}...")
            cleanup_vllm_memory(method_name=f"{method}_initialization")

        detector = create_postprocessor(method, args)

        logger.info(f"\n=== {method}: Setup on Safe Prompts ===")
        start_time = time.time()
        if method.lower().startswith("forte_"):
            # Use Forte API
            detector.fit(safe_train)
        else:
            # Use postprocessor-based methods
            detector.setup(safe_train, random_state=args.seed)
        setup_time = time.time() - start_time
        logger.info(f"{method} setup completed in {setup_time:.2f} seconds")

        # Evaluate on test set
        logger.info(f"\n=== {method}: OR-Bench Evaluation ===")
        results = run_orbench_evaluation(detector, safe_test, toxic_test, method, args)
        all_results[method] = results

        logger.info(
            f"{method} on OR-Bench: AUROC={results['AUROC']:.4f}, FPR@95={results['FPR@95TPR']:.4f}"
        )

        # Critical: Clean up after each method to free GPU memory for the next one
        logger.info(f"Performing cleanup after {method}...")

        # Special aggressive cleanup for vLLM models
        if method.lower() in ["mdjudge_vllm", "polyguard_vllm"]:
            logger.info(f"Performing comprehensive cleanup after {method}...")
            # Try to get the vLLM model instance for proper cleanup
            model_instance = None
            if hasattr(detector, "model") and detector.model is not None:
                model_instance = detector.model
            elif hasattr(detector, "llm") and detector.llm is not None:
                model_instance = detector.llm
            elif hasattr(detector, "llm_engine") and detector.llm_engine is not None:
                model_instance = detector

            cleanup_vllm_memory(model_instance=model_instance, method_name=f"{method}_completion")

            # Additional cleanup: delete the detector itself
            try:
                if hasattr(detector, "model"):
                    detector.model = None
                if hasattr(detector, "llm"):
                    detector.llm = None
                del detector
            except Exception as e:
                logger.warning(f"Error deleting {method} detector: {e}")

        # General cleanup for all methods
        try:
            # Delete the detector to free memory
            if hasattr(detector, "embedding_model") and detector.embedding_model is not None:
                del detector.embedding_model
            if hasattr(detector, "model") and detector.model is not None:
                del detector.model
            del detector

            # Force garbage collection and CUDA cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            memory_after = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
            logger.info(f"GPU memory after {method} cleanup: {memory_after:.2f} GB")

        except Exception as cleanup_error:
            logger.warning(f"General cleanup failed for {method}: {cleanup_error}")

    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info("OR-BENCH OVERREFUSAL DETECTION RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Embedding Model: {args.embedding_model}")
    logger.info(f"Safe Training Samples: {len(safe_train)}")
    logger.info(f"Safe Test Samples: {len(safe_test)}")
    logger.info(f"Toxic Test Samples: {len(toxic_test)}")
    logger.info("")

    # Header
    header = f"{'Method':<12}{'AUROC':<8}{'FPR@95':<8}{'AUPRC':<8}{'F1':<8}"
    logger.info(header)
    logger.info("-" * len(header))

    # Results for each method
    for method in methods_to_run:
        if method in all_results:
            results = all_results[method]
            row = f"{method:<12}{results['AUROC']:<8.3f}{results['FPR@95TPR']:<8.3f}{results['AUPRC']:<8.3f}{results['F1']:<8.3f}"
            logger.info(row)

    # Print OR-Bench style rejection rate table (similar to paper Table 2)
    logger.info(f"\n{'='*80}")
    logger.info("OR-BENCH STYLE REJECTION RATE TABLE (Similar to Paper Table 2)")
    logger.info(f"{'='*80}")
    logger.info(
        "Shows rejection rates for Safe prompts (should be low) vs Toxic prompts (should be high)"
    )
    logger.info("")

    # Check if we have rejection rate data
    has_rejection_data = any(
        "rejection_rates" in all_results.get(method, {}) for method in methods_to_run
    )

    if has_rejection_data:
        # Print table at optimal F1 threshold
        logger.info("=== At Optimal F1 Threshold ===")
        header_rr = f"{'Method':<12}{'Threshold':<12}{'Safe Rejection %':<16}{'Toxic Rejection %':<17}{'Accuracy':<10}"
        logger.info(header_rr)
        logger.info("-" * len(header_rr))

        for method in methods_to_run:
            if method in all_results and "rejection_rates" in all_results[method]:
                rr_data = all_results[method]["rejection_rates"]["optimal_f1"]
                safe_reject_pct = rr_data["safe_rejection_rate"] * 100
                toxic_reject_pct = rr_data["toxic_rejection_rate"] * 100
                accuracy = rr_data["accuracy"]
                threshold = rr_data["threshold"]

                row_rr = f"{method:<12}{threshold:<12.3f}{safe_reject_pct:<16.1f}{toxic_reject_pct:<17.1f}{accuracy:<10.3f}"
                logger.info(row_rr)

        logger.info("")

        # Print table at different threshold percentiles
        logger.info("=== Rejection Rates at Different Threshold Percentiles ===")
        for method in methods_to_run:
            if method in all_results and "rejection_rates" in all_results[method]:
                logger.info(f"\n--- {method} Method ---")
                rr_header = (
                    f"{'Threshold':<12}{'Safe Rej %':<12}{'Toxic Rej %':<13}{'Accuracy':<10}"
                )
                logger.info(rr_header)
                logger.info("-" * len(rr_header))

                rejection_data = all_results[method]["rejection_rates"]

                # Show threshold-based results
                for key in sorted(rejection_data.keys()):
                    if key.startswith("threshold_"):
                        rr_info = rejection_data[key]
                        safe_reject_pct = rr_info["safe_rejection_rate"] * 100
                        toxic_reject_pct = rr_info["toxic_rejection_rate"] * 100
                        accuracy = rr_info["accuracy"]
                        threshold = rr_info["threshold"]

                        row_detail = f"{threshold:<12.3f}{safe_reject_pct:<12.1f}{toxic_reject_pct:<13.1f}{accuracy:<10.3f}"
                        logger.info(row_detail)
    else:
        logger.info(
            "No rejection rate data available. This may occur if methods don't return score information."
        )

    # Save results to CSV
    csv_filepath = save_orbench_results_to_csv(
        all_results, args.embedding_model, len(safe_train), len(safe_test), len(toxic_test), args
    )
    logger.info(f"\nResults have been saved to: {csv_filepath}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Baselines - OR-Bench Overrefusal Detection"
    )
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
        default="MDJudge_VLLM,RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax,Forte_GMM,Forte_OCSVM,LlamaGuard,LlamaGuard_Logit,DuoGuard,WildGuard,PolyGuard_VLLM",
        help="Comma-separated list of methods to run (LlamaGuard uses hardcoded scores, LlamaGuard_Logit uses logit-based scoring, WildGuard for AllenAI WildGuard 7B)",
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
    # DuoGuard parameters
    parser.add_argument(
        "--duoguard_model_id",
        type=str,
        default="DuoGuard/DuoGuard-0.5B",
        help="DuoGuard model ID to use",
    )
    parser.add_argument(
        "--duoguard_threshold",
        type=float,
        default=0.5,
        help="DuoGuard threshold for binary safe/unsafe classification",
    )
    parser.add_argument(
        "--duoguard_max_length",
        type=int,
        default=512,
        help="Maximum input sequence length for DuoGuard",
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

    # MD-Judge VLLM parameters
    parser.add_argument(
        "--mdjudge_vllm_model_id",
        type=str,
        default="OpenSafetyLab/MD-Judge-v0_2-internlm2_7b",
        help="MD-Judge VLLM model ID",
    )
    parser.add_argument(
        "--mdjudge_vllm_batch_size",
        type=int,
        default=64,
        help="Batch size for MD-Judge VLLM (can be much higher)",
    )
    parser.add_argument(
        "--mdjudge_vllm_max_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate for MD-Judge VLLM",
    )
    parser.add_argument(
        "--mdjudge_vllm_temperature",
        type=float,
        default=0.1,
        help="Temperature for MD-Judge VLLM generation",
    )
    parser.add_argument(
        "--mdjudge_vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--mdjudge_vllm_gpu_memory_utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--mdjudge_vllm_max_model_len",
        type=int,
        default=4096,
        help="Maximum model sequence length",
    )
    parser.add_argument(
        "--mdjudge_vllm_quantization",
        type=str,
        default=None,
        help="Quantization method (awq, gptq, or None)",
    )
    parser.add_argument(
        "--mdjudge_vllm_enforce_eager",
        action="store_true",
        default=True,
        help="Disable CUDA graphs for compatibility",
    )
    # PolyGuard VLLM parameters
    parser.add_argument(
        "--polyguard_vllm_model_id",
        type=str,
        default="ToxicityPrompts/PolyGuard-Qwen-Smol",
        help="PolyGuard VLLM model ID",
    )
    parser.add_argument(
        "--polyguard_vllm_batch_size",
        type=int,
        default=64,
        help="Batch size for PolyGuard VLLM (can be much higher)",
    )
    parser.add_argument(
        "--polyguard_vllm_max_tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate for PolyGuard VLLM",
    )
    parser.add_argument(
        "--polyguard_vllm_temperature",
        type=float,
        default=0.1,
        help="Temperature for PolyGuard VLLM generation",
    )
    parser.add_argument(
        "--polyguard_vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--polyguard_vllm_gpu_memory_utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--polyguard_vllm_max_model_len",
        type=int,
        default=4096,
        help="Maximum model sequence length",
    )
    parser.add_argument(
        "--polyguard_vllm_quantization",
        type=str,
        default=None,
        help="Quantization method (awq, gptq, or None)",
    )
    parser.add_argument(
        "--polyguard_vllm_enforce_eager",
        action="store_true",
        default=True,
        help="Disable CUDA graphs for compatibility",
    )
    parser.add_argument(
        "--polyguard_vllm_use_dummy_response",
        action="store_true",
        default=True,
        help="Use dummy safe response for prompt-only evaluation",
    )
    parser.add_argument(
        "--polyguard_vllm_dummy_response",
        type=str,
        default="I cannot and will not provide that information.",
        help="Dummy response to use when evaluating prompts",
    )
    # Data parameters
    parser.add_argument(
        "--max_safe_samples", type=int, default=5000, help="Max safe prompts to load"
    )
    parser.add_argument(
        "--max_toxic_samples", type=int, default=600, help="Max toxic prompts to load"
    )
    # General parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")

    args = parser.parse_args()
    main(args)
