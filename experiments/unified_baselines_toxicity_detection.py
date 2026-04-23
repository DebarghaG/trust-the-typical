#!/usr/bin/env python3
"""
Filename: unified_baselines_toxicity_detection.py
Unified Baseline Script for Toxicity and Harmful Content Detection

Evaluates on multiple toxicity detection benchmarks:
- RealToxicity: Real-world toxic prompts
- HatEval: Hate speech detection
- Davidson et al.: Hate speech vs offensive language
- HASOC: Hate speech and offensive content
- OffensEval: Offensive language identification


Uses safe/benign text as the ID distribution and toxic content as OOD.
"""

import contextlib
import gc
import os
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
from baselines.energy_text_postprocessor import EnergyBasedTextPostprocessor
from baselines.fdbd_text_postprocessor import fDBDTextPostprocessor
from baselines.gmm_text_postprocessor import GMMTextPostprocessor
from baselines.knn_text_postprocessor import KNNTextPostprocessor
from baselines.llama_guard_logit_text_postprocessor import LlamaGuardLogitTextPostprocessor
from baselines.llama_guard_logit_vllm_text_postprocessor import LlamaGuardLogitVLLMTextPostprocessor
from baselines.llama_guard_text_postprocessor import LlamaGuardTextPostprocessor
from baselines.llm_guard_text_postprocessor import (
    create_llm_guard_toxicity_detector,
)
from baselines.mahalanobis_text_postprocessor import MahalanobisTextPostprocessor
from baselines.nnguide_text_postprocessor import NNGuideTextPostprocessor
from baselines.openai_omni_moderation_postprocessor import OpenaiOmniModerationPostprocessor
from baselines.openmax_text_postprocessor import OpenMaxTextPostprocessor
from baselines.perspective_text_postprocessor import PerspectiveTextPostprocessor
from baselines.react_text_postprocessor import ReActTextPostprocessor
from baselines.rmd_simple_postprocessor import SimpleRMDTextPostprocessor
from baselines.vim_text_postprocessor import VIMTextPostprocessor
from baselines.wildguard_text_postprocessor import WildGuardTextPostprocessor
from baselines.wildguard_vllm_text_postprocessor import WildGuardVLLMTextPostprocessor

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
logger = logging.getLogger("ToxicityDetectionBaselines")


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


def load_safe_instructions(
    max_samples: int = 1000, dataset_name: str = "alpaca"
) -> tuple[list[str], list[str]]:
    """Load safe/helpful instructions as ID distribution."""
    logger.info(f"Loading safe instruction data from {dataset_name}...")

    train_texts: list[str] = []
    test_texts: list[str] = []

    if dataset_name.lower() == "alpaca":
        try:
            # Load Alpaca instruction dataset
            dataset = load_dataset("tatsu-lab/alpaca", split="train")

            for example in dataset:
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")

                # Combine instruction and input if available
                if input_text:
                    full_instruction = f"{instruction}\n{input_text}"
                else:
                    full_instruction = instruction

                if len(full_instruction.strip()) > 20:
                    if len(train_texts) < max_samples:
                        train_texts.append(full_instruction.strip())
                    elif len(test_texts) < max_samples // 2:
                        test_texts.append(full_instruction.strip())
                    else:
                        break

        except Exception as e:
            logger.error(f"Could not load Alpaca dataset: {e}")
            raise RuntimeError(f"Failed to load Alpaca dataset for safe instructions: {e}") from e

    elif dataset_name.lower() == "dolly":
        try:
            # Load Dolly instruction dataset
            dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

            for example in dataset:
                instruction = example.get("instruction", "")
                context = example.get("context", "")

                # Combine instruction and context if available
                if context:
                    full_instruction = f"{instruction}\nContext: {context}"
                else:
                    full_instruction = instruction

                if len(full_instruction.strip()) > 20:
                    if len(train_texts) < max_samples:
                        train_texts.append(full_instruction.strip())
                    elif len(test_texts) < max_samples // 2:
                        test_texts.append(full_instruction.strip())
                    else:
                        break

        except Exception as e:
            logger.error(f"Could not load Dolly dataset: {e}")
            raise RuntimeError(f"Failed to load Dolly dataset for safe instructions: {e}") from e

    elif dataset_name.lower() == "helpful_base":
        try:
            # Load helpful/harmless base dataset
            dataset = load_dataset("Anthropic/hh-rlhf", split="train")

            for example in dataset:
                chosen = example.get("chosen", "")
                # Extract the human question/prompt from the conversation
                if chosen and "Human:" in chosen:
                    human_part = chosen.split("Human:")[1].split("Assistant:")[0].strip()
                    if len(human_part.strip()) > 20:
                        if len(train_texts) < max_samples:
                            train_texts.append(human_part.strip())
                        elif len(test_texts) < max_samples // 2:
                            test_texts.append(human_part.strip())
                        else:
                            break

        except Exception as e:
            logger.error(f"Could not load helpful_base dataset: {e}")
            raise RuntimeError(
                f"Failed to load helpful_base dataset for safe instructions: {e}"
            ) from e

    elif dataset_name.lower() == "openassistant":
        try:
            # Load OpenAssistant conversations dataset
            dataset = load_dataset("OpenAssistant/oasst2", split="train")

            for example in dataset:
                text = example.get("text", "")
                role = example.get("role", "")

                # Only use prompter (human) messages as instructions
                if role == "prompter" and len(text.strip()) > 20:
                    if len(train_texts) < max_samples:
                        train_texts.append(text.strip())
                    elif len(test_texts) < max_samples // 2:
                        test_texts.append(text.strip())
                    else:
                        break

        except Exception as e:
            logger.error(f"Could not load OpenAssistant dataset: {e}")
            raise RuntimeError(
                f"Failed to load OpenAssistant dataset for safe instructions: {e}"
            ) from e

    elif dataset_name.lower() == "id_mix":
        try:
            # Load and combine samples from all ID datasets
            logger.info("Loading ID mix dataset - combining all safe instruction datasets...")

            all_datasets = ["alpaca", "dolly", "helpful_base", "openassistant"]
            samples_per_dataset = max_samples // len(all_datasets)  # Distribute samples evenly
            test_samples_per_dataset = (max_samples // 2) // len(all_datasets)

            for _idx, dataset in enumerate(all_datasets):
                try:
                    logger.info(f"Loading {samples_per_dataset} samples from {dataset}...")

                    # Recursively call this function for each individual dataset
                    dataset_train, dataset_test = load_safe_instructions(
                        max_samples=samples_per_dataset + test_samples_per_dataset,
                        dataset_name=dataset,
                    )

                    # Add samples to our combined lists
                    train_texts.extend(dataset_train[:samples_per_dataset])
                    test_texts.extend(dataset_test[:test_samples_per_dataset])

                    logger.info(
                        f"Added {len(dataset_train[:samples_per_dataset])} train and {len(dataset_test[:test_samples_per_dataset])} test samples from {dataset}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to load {dataset} for ID mix: {e}")
                    logger.info("Continuing with other datasets...")
                    continue

            # Shuffle the combined data to mix datasets properly
            import random

            random.seed(42)  # For reproducibility
            random.shuffle(train_texts)
            random.shuffle(test_texts)

            logger.info(
                f"ID mix: Combined {len(train_texts)} train samples and {len(test_texts)} test samples from {len(all_datasets)} datasets"
            )

        except Exception as e:
            logger.error(f"Could not create ID mix dataset: {e}")
            raise RuntimeError(f"Failed to create ID mix dataset: {e}") from e

    else:
        available_datasets = ["alpaca", "dolly", "helpful_base", "openassistant", "id_mix"]
        logger.error(f"Unknown safe dataset: {dataset_name}")
        logger.error(f"Available datasets: {', '.join(available_datasets)}")
        raise ValueError(
            f"Unknown safe dataset: {dataset_name}. Available: {', '.join(available_datasets)}"
        )

    logger.info(f"Loaded {len(train_texts)} safe training instructions from {dataset_name}")
    logger.info(f"Loaded {len(test_texts)} safe test instructions from {dataset_name}")

    return train_texts, test_texts


def load_toxic_data(dataset_name: str, max_samples: int = 500) -> list[str]:
    """Load toxic/harmful content from various benchmarks."""
    logger.info(f"Loading toxic data from {dataset_name}...")

    toxic_texts: list[str] = []

    if dataset_name.lower().startswith("rtp_alx"):
        # Handle RTP_ALX datasets (multilingual toxicity prompts)
        try:
            import json
            from pathlib import Path

            # Extract language code if specified (e.g., rtp_alx_en, rtp_alx_es)
            parts = dataset_name.lower().split("_")
            if len(parts) > 2:
                lang_code = parts[2].upper()
            else:
                lang_code = "EN"  # Default to English

            # Map common language codes to file suffixes
            lang_mapping = {
                "EN": "EN",
                "ES": "ES",
                "FR": "FR",
                "DE": "DE",
                "IT": "IT",
                "PT": "PT",
                "RU": "RU",
                "JA": "JA",
                "KO": "KO",
                "ZH": "ZH-Hans",
                "NL": "NL",
                "PL": "PL",
                "TR": "TR",
                "HI": "HI",
                "ID": "ID",
                "TH": "TH",
                "VI": "VI",
                "SV": "SV",
                "NO": "NO-NB",
                "DA": "DA",
                "FI": "FI",
                "CS": "CS",
                "HU": "HU",
                "UK": "UK",
                "HE": "HE",
                "EL": "EL",
                "BG": "BG",
                "RO": "RO",
                "SK": "SK",
                "SL": "SL",
                "LT": "LT",
                "LV": "LV",
                "ET": "ET",
                "CA": "CA",
                "CY": "CY",
                "SW": "SW",
                "DE-CH": "DE-CH",
            }

            file_suffix = lang_mapping.get(lang_code, lang_code)
            file_path = Path(f"data/RTP_LX_{file_suffix}.json")

            if not file_path.exists():
                logger.warning(f"RTP_ALX file not found: {file_path}")
                logger.info(
                    f"Available RTP_ALX languages: {', '.join(sorted(lang_mapping.keys()))}"
                )
                return toxic_texts

            logger.info(f"Loading RTP_ALX data from {file_path}")

            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    if len(toxic_texts) >= max_samples:
                        break

                    try:
                        item = json.loads(line.strip())

                        # Get prompt and completion
                        prompt = item.get("Prompt", "")
                        completion = item.get("Completion", "")

                        # Get toxicity annotations
                        prompt_toxicity = item.get("PromptAnnotations", {}).get("Toxicity", 0)
                        completion_toxicity = item.get("CompletionAnnotations", {}).get(
                            "Toxicity", 0
                        )

                        # Add toxic prompts (toxicity score >= 3)
                        if prompt and prompt_toxicity >= 3 and len(prompt.strip()) > 20:
                            toxic_texts.append(prompt.strip())

                        # Also add toxic completions if we need more samples
                        if (
                            len(toxic_texts) < max_samples
                            and completion
                            and completion_toxicity >= 3
                        ):
                            # Take first 500 chars of completion to avoid very long texts
                            completion_text = completion.strip()[:500]
                            if len(completion_text) > 20:
                                toxic_texts.append(completion_text)

                    except json.JSONDecodeError:
                        continue

            logger.info(f"Loaded {len(toxic_texts)} toxic samples from RTP_ALX {lang_code}")

        except Exception as e:
            logger.warning(f"Could not load RTP_ALX data: {e}")

    elif dataset_name.lower() == "realtoxicity":
        try:
            # Load RealToxicityPrompts dataset
            dataset = load_dataset("allenai/real-toxicity-prompts", split="train")

            for example in dataset:
                if len(toxic_texts) >= max_samples:
                    break

                # Check both prompt and continuation for high toxicity
                prompt = example.get("prompt", {})
                continuation = example.get("continuation", {})

                # Extract toxic prompts
                if isinstance(prompt, dict):
                    text = prompt.get("text", "")
                    toxicity = prompt.get("toxicity", 0)
                    if toxicity and toxicity > 0.6 and len(text.strip()) > 20:
                        toxic_texts.append(text.strip())
                        continue

                # Extract toxic continuations
                if isinstance(continuation, dict) and len(toxic_texts) < max_samples:
                    text = continuation.get("text", "")
                    toxicity = continuation.get("toxicity", 0)
                    if toxicity and toxicity > 0.6 and len(text.strip()) > 20:
                        toxic_texts.append(text.strip())

        except Exception as e:
            logger.warning(f"Could not load RealToxicity: {e}")

    elif dataset_name.lower() == "civil_comments":
        try:
            # Load Civil Comments dataset with toxicity scores
            dataset = load_dataset("google/civil_comments", split="train")

            for example in dataset:
                if len(toxic_texts) >= max_samples:
                    break

                # Get toxicity score and text
                toxicity = example.get("toxicity", 0)
                text = example.get("text", "")

                if toxicity and toxicity > 0.7 and len(text.strip()) > 20:
                    toxic_texts.append(text.strip())

        except Exception as e:
            logger.warning(f"Could not load Civil Comments: {e}")

    elif dataset_name.lower() == "hateval":
        try:
            # Load actual HatEval dataset from SemEval-2019 Task 5
            dataset = load_dataset("valeriobasile/HatEval", split="train")

            for example in dataset:
                if len(toxic_texts) >= max_samples:
                    break

                text = example.get("text", "")
                hs_label = example.get("HS", 0)  # Hate Speech binary label

                # Extract texts with hate speech (HS = 1)
                if hs_label == 1 and len(text.strip()) > 20:
                    toxic_texts.append(text.strip())

        except Exception as e:
            logger.warning(f"Could not load HatEval: {e}")

    elif dataset_name.lower() == "davidson":
        try:
            # Load Davidson et al. hate speech and offensive language dataset
            dataset = load_dataset("tdavidson/hate_speech_offensive", split="train")

            for example in dataset:
                if len(toxic_texts) >= max_samples:
                    break

                text = example.get("tweet", "")
                class_label = example.get("class", 2)

                # Extract hate speech (class=0) and offensive language (class=1)
                # Skip "neither" (class=2)
                if class_label in [0, 1] and len(text.strip()) > 20:
                    toxic_texts.append(text.strip())

        except Exception as e:
            logger.warning(f"Could not load Davidson dataset: {e}")

    elif dataset_name.lower() == "hasoc":
        try:
            # Use another toxicity dataset
            dataset = load_dataset("SetFit/toxic_conversations_50k", split="train")

            for example in dataset:
                if len(toxic_texts) >= max_samples:
                    break

                text = example.get("text", "")
                label = example.get("label", 0)

                # Assume label 1 is toxic
                if label == 1 and len(text.strip()) > 20:
                    toxic_texts.append(text.strip())

        except Exception as e:
            logger.warning(f"Could not load HASOC: {e}")

    elif dataset_name.lower() == "offenseval":
        try:
            # Use offensive language detection dataset
            dataset = load_dataset("cardiffnlp/tweet_eval", "offensive", split="train")

            for example in dataset:
                if len(toxic_texts) >= max_samples:
                    break

                text = example.get("text", "")
                label = example.get("label", 0)

                # Assume label 1 is offensive
                if label == 1 and len(text.strip()) > 20:
                    toxic_texts.append(text.strip())

        except Exception as e:
            logger.warning(f"Could not load OffensEval: {e}")

    elif dataset_name.lower().startswith("xsafety"):
        try:
            # Handle XSafety dataset with language-specific filtering
            # Extract language code if specified (e.g., xsafety_zh, xsafety_ar)
            if dataset_name.lower() == "xsafety":
                # Load all languages
                target_language = None
                logger.info("Loading XSafety dataset from ToxicityPrompts/XSafety (all languages)")
            else:
                # Extract language from dataset name (e.g., xsafety_zh -> zh)
                parts = dataset_name.lower().split("_")
                if len(parts) > 1:
                    target_language = parts[1]
                    logger.info(
                        f"Loading XSafety dataset from ToxicityPrompts/XSafety (language: {target_language})"
                    )
                else:
                    target_language = None
                    logger.info(
                        "Loading XSafety dataset from ToxicityPrompts/XSafety (all languages)"
                    )

            # Available languages in XSafety
            available_languages = ["zh", "ar", "sp", "fr", "de", "bn", "en", "ja", "hi", "ru"]

            if target_language and target_language not in available_languages:
                logger.warning(f"Unknown XSafety language: {target_language}")
                logger.info(f"Available languages: {', '.join(available_languages)}")
                target_language = None  # Fallback to all languages

            dataset = load_dataset("ToxicityPrompts/XSafety", split="test")

            loaded_count = 0
            language_counts: dict[str, int] = {}

            for example in dataset:
                if len(toxic_texts) >= max_samples:
                    break

                # XSafety has 'text', 'language', and 'category' fields
                text = example.get("text", "")
                language = example.get("language", "")

                # Filter by language if specified
                if target_language and language != target_language:
                    continue

                # All examples in XSafety are considered toxic/unsafe prompts
                if text and len(text.strip()) > 20:
                    toxic_texts.append(text.strip())
                    loaded_count += 1

                    # Track language distribution
                    if language in language_counts:
                        language_counts[language] += 1
                    else:
                        language_counts[language] = 1

            if target_language:
                logger.info(
                    f"Loaded {loaded_count} toxic samples from XSafety dataset (language: {target_language})"
                )
            else:
                logger.info(
                    f"Loaded {loaded_count} toxic samples from XSafety dataset (all languages)"
                )
                logger.info(f"Language distribution: {dict(sorted(language_counts.items()))}")

        except Exception as e:
            logger.warning(f"Could not load XSafety: {e}")

    elif dataset_name.lower().startswith("polyguard"):
        # This is the polyguard trai set
        try:
            from datasets import get_dataset_split_names

            # Handle PolyGuard dataset with different subsets
            # Extract subset if specified (e.g., polyguard_code, polyguard_cyber)
            # Handle special case for social_media which has underscore
            if dataset_name.lower() == "polyguard_social_media":
                subset = "social_media"
            elif dataset_name.lower().startswith("polyguard_"):
                # Remove "polyguard_" prefix and join remaining parts
                subset = dataset_name.lower().replace("polyguard_", "")
                # Handle multi-word subsets with underscores
                if subset in [
                    "finance_input",
                    "finance_output",
                    "law_input",
                    "law_output",
                    "regulation_input",
                    "regulation_output",
                ]:
                    pass  # Keep as is
                else:
                    # For single word subsets
                    subset = subset.split("_")[0] if "_" in subset else subset
            else:
                logger.warning("Using default subset of social_media")
                subset = "social_media"  # Default subset

            # Available subsets from PolyGuard (updated based on actual dataset)
            available_subsets = [
                "social_media",
                "education",
                "hr",
                "finance_input",
                "finance_output",
                "law_input",
                "law_output",
                "regulation_input",
                "regulation_output",
                "code",
                "cyber",
            ]

            if subset not in available_subsets:
                logger.warning(f"Unknown PolyGuard subset: {subset}")
                logger.info(f"Available subsets: {', '.join(available_subsets)}")
                subset = "social_media"  # Fallback to social_media subset

            logger.info(f"Loading PolyGuard dataset with subset: {subset}")

            # Get authentication token
            import os

            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

            # Try to get available splits for this subset
            try:
                from datasets import get_dataset_split_names

                available_splits = get_dataset_split_names(
                    "AI-Secure/PolyGuard", subset, token=token
                )
                logger.info(f"Available splits for {subset}: {available_splits}")
            except Exception as e:
                logger.warning(f"Could not get splits for {subset}: {e}")
                available_splits = []

            # Load all unsafe splits or splits containing unsafe data
            combined_dataset = []

            if available_splits:
                # Load all splits that contain "unsafe" in the name or all splits if no unsafe-specific
                unsafe_splits = [s for s in available_splits if "unsafe" in s.lower()]

                if not unsafe_splits:
                    # If no unsafe-specific splits, load all splits and filter by label
                    splits_to_load = available_splits
                else:
                    splits_to_load = unsafe_splits

                for split_name in splits_to_load:
                    try:
                        logger.info(f"Loading PolyGuard {subset} - {split_name}")
                        split_data = load_dataset(
                            "AI-Secure/PolyGuard", subset, split=split_name, token=token
                        )

                        # If this is not an unsafe-specific split, we'll filter by label later
                        if "unsafe" in split_name.lower():
                            # All examples in unsafe splits are toxic
                            combined_dataset.extend(split_data)
                            logger.info(
                                f"Loaded {len(split_data)} examples from {split_name} (all unsafe)"
                            )
                        else:
                            # Need to filter by label
                            unsafe_count = 0
                            for ex in split_data:
                                if ex.get("label") == "unsafe":
                                    combined_dataset.append(ex)
                                    unsafe_count += 1
                            logger.info(
                                f"Loaded {unsafe_count} unsafe examples from {split_name} (out of {len(split_data)} total)"
                            )

                    except Exception as e:
                        logger.warning(f"Could not load split {split_name}: {e}")
                        continue

                dataset = combined_dataset
                logger.info(f"Total combined dataset size: {len(dataset)} unsafe examples")
            else:
                # Fallback: try standard splits
                for split in ["train", "test", "validation"]:
                    try:
                        dataset = load_dataset(
                            "AI-Secure/PolyGuard", subset, split=split, token=token
                        )
                        logger.info(f"Successfully loaded PolyGuard {subset} with split: {split}")
                        break
                    except Exception:
                        continue
                else:
                    logger.warning(f"Could not find any valid split for PolyGuard {subset}")
                    return toxic_texts

            total_examples = 0
            unsafe_examples = 0

            for idx, example in enumerate(dataset):
                total_examples += 1
                if len(toxic_texts) >= max_samples:
                    break

                # Debug first few examples to see what fields are available
                if idx < 3:
                    logger.info(
                        f"Example {idx+1} keys: {list(example.keys()) if hasattr(example, 'keys') else 'N/A'}"
                    )
                    logger.info(
                        f"Example {idx+1} instance length: {len(str(example.get('instance', '')))}"
                    )
                    logger.info(
                        f"Example {idx+1} prompt length: {len(str(example.get('prompt', '')))}"
                    )
                    logger.info(
                        f"Example {idx+1} response length: {len(str(example.get('response', '')))}"
                    )
                    if idx == 0:
                        # Show first 100 chars of instance, prompt and response for debugging
                        instance_sample = str(example.get("instance", ""))[:100]
                        prompt_sample = str(example.get("prompt", ""))[:100]
                        response_sample = str(example.get("response", ""))[:100]
                        logger.info(f"Sample instance: {instance_sample}...")
                        logger.info(f"Sample prompt: {prompt_sample}...")
                        logger.info(f"Sample response: {response_sample}...")

                # PolyGuard actually uses: instance, category, rule (not prompt/response)
                instance = example.get("instance", "")  # This contains the actual text content
                prompt = example.get("prompt", "")  # Fallback for other subsets
                response = example.get("response", "")  # Fallback for other subsets
                label = example.get("label", "")

                # Use instance if available, otherwise fall back to prompt
                text_content = instance or prompt

                # Determine if this example is unsafe
                # Since we loaded all unsafe splits (containing _unsafe in name),
                # all examples in our combined dataset are unsafe by definition
                is_unsafe = True  # We only loaded unsafe splits or filtered for unsafe examples

                # Only take unsafe examples
                if is_unsafe:
                    unsafe_examples += 1

                    # Add text content (instance or prompt) if available and long enough
                    if text_content and len(str(text_content).strip()) > 10:
                        toxic_texts.append(str(text_content).strip()[:1000])  # Limit to 1000 chars

                    # Also add response if we need more samples (for other subsets that might have responses)
                    if (
                        len(toxic_texts) < max_samples
                        and response
                        and len(str(response).strip()) > 10
                    ):
                        # Take first 500 chars of response to avoid very long texts
                        response_text = str(response).strip()[:500]
                        toxic_texts.append(response_text)

            logger.info(
                f"Loaded {len(toxic_texts)} toxic samples from PolyGuard {subset} (examined {total_examples} examples, found {unsafe_examples} unsafe)"
            )

        except Exception as e:
            logger.warning(f"Could not load PolyGuard: {e}")

    elif dataset_name.lower().startswith("polyguardprompts"):
        try:
            # Handle ToxicityPrompts/PolyGuardPrompts dataset with language-specific filtering
            # Extract language code if specified (e.g., polyguardprompts_en, polyguardprompts_zh)
            if dataset_name.lower() == "polyguardprompts":
                # Load all languages
                target_language = None
                logger.info("Loading PolyGuardPrompts dataset (all languages)")
            else:
                # Extract language from dataset name (e.g., polyguardprompts_zh -> zh)
                parts = dataset_name.lower().split("_")
                if len(parts) > 1:
                    target_language = parts[1]
                    logger.info(f"Loading PolyGuardPrompts dataset (language: {target_language})")
                else:
                    target_language = None
                    logger.info("Loading PolyGuardPrompts dataset (all languages)")

            # Map language codes to full language names as they appear in the dataset
            # The dataset uses capitalized full language names
            language_mapping = {
                "en": "English",
                "zh": "Chinese",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ru": "Russian",
                "ja": "Japanese",
                "ko": "Korean",
                "ar": "Arabic",
                "hi": "Hindi",
                "nl": "Dutch",
                "pl": "Polish",
                "sv": "Swedish",
                "cs": "Czech",
                "th": "Thai",
            }

            # If target language code provided, map to full name
            if target_language:
                # First try direct mapping from code
                target_language_full = language_mapping.get(target_language.lower())

                # If not found, check if it's already a full language name (case-insensitive)
                if not target_language_full:
                    # Check if user provided a full language name
                    available_language_names = set(language_mapping.values())
                    for lang in available_language_names:
                        if lang.lower() == target_language.lower():
                            target_language_full = lang
                            break

                if not target_language_full:
                    logger.warning(f"Unknown PolyGuardPrompts language: {target_language}")
                    logger.info(
                        f"Available language codes: {', '.join(sorted(language_mapping.keys()))}"
                    )
                    logger.info(
                        f"Available language names: {', '.join(sorted(language_mapping.values()))}"
                    )
                    target_language = None  # Fallback to all languages
                else:
                    target_language = target_language_full
                    logger.info(f"Using language filter: {target_language}")

            # Load the dataset
            dataset = load_dataset("ToxicityPrompts/PolyGuardPrompts", split="test")

            loaded_count = 0
            polyguard_language_counts: dict[str, int] = {}
            subcategory_counts: dict[str, int] = {}

            for example in dataset:
                if len(toxic_texts) >= max_samples:
                    break

                # Extract fields
                prompt = example.get("prompt", "")
                language = example.get("language", "")
                prompt_harm_label = example.get("prompt_harm_label", "")
                prompt_label = example.get("prompt_label", "")
                subcategory = example.get("subcategory", "")

                # Filter by language if specified
                if target_language and language != target_language:
                    continue

                # Filter for harmful/unsafe prompts
                # Check both prompt_harm_label and prompt_label fields
                # Consider a prompt toxic if:
                # - prompt_harm_label != "unharmful" OR
                # - prompt_label != "safe" OR
                # - subcategory != "benign"
                is_toxic = (
                    (prompt_harm_label and prompt_harm_label != "unharmful")
                    or (prompt_label and prompt_label != "safe")
                    or (subcategory and subcategory != "benign")
                )

                # Add toxic prompts
                if is_toxic and prompt and len(prompt.strip()) > 20:
                    toxic_texts.append(prompt.strip())
                    loaded_count += 1

                    # Track language distribution
                    if language:
                        polyguard_language_counts[language] = (
                            polyguard_language_counts.get(language, 0) + 1
                        )

                    # Track subcategory distribution
                    if subcategory and subcategory != "benign":
                        subcategory_counts[subcategory] = subcategory_counts.get(subcategory, 0) + 1

            if target_language:
                logger.info(
                    f"Loaded {loaded_count} toxic samples from PolyGuardPrompts (language: {target_language})"
                )
            else:
                logger.info(
                    f"Loaded {loaded_count} toxic samples from PolyGuardPrompts (all languages)"
                )
                if polyguard_language_counts:
                    logger.info(
                        f"Language distribution: {dict(sorted(polyguard_language_counts.items()))}"
                    )
                if subcategory_counts:
                    # Log top 5 subcategories
                    top_subcats = sorted(
                        subcategory_counts.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    logger.info(f"Top 5 harmful subcategories: {dict(top_subcats)}")

        except Exception as e:
            logger.warning(f"Could not load PolyGuardPrompts: {e}")

    # # If we still don't have enough samples, add some fallback from streaming
    # if len(toxic_texts) < max_samples // 2:
    #     try:
    #         if dataset_name.lower() not in ["civil_comments", "realtoxicity"]:
    #             # Try civil comments as universal fallback
    #             dataset = load_dataset("google/civil_comments", split="train", streaming=True)
    #             for example in dataset:
    #                 if len(toxic_texts) >= max_samples:
    #                     break
    #                 if example.get("toxicity", 0) > 0.7:
    #                     text = example.get("text", "")
    #                     if len(text.strip()) > 20:
    #                         toxic_texts.append(text.strip())
    #     except Exception as e:
    #         logger.warning(f"Could not load fallback dataset: {e}")

    logger.info(f"Loaded {len(toxic_texts)} {dataset_name} toxic texts")
    return toxic_texts[:max_samples]


def create_postprocessor(method: str, args: argparse.Namespace) -> Any:
    """Create postprocessor based on method."""
    safe_dataset = getattr(args, "safe_dataset", "alpaca")
    embedding_dir = f"embeddings_{method.lower()}_toxicity_{safe_dataset}"

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
    elif method.lower() == "energy":
        return EnergyBasedTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            temperature=args.energy_temperature,
        )
    elif method.lower() == "knn":
        return KNNTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            n_neighbors=args.knn_neighbors,
            metric=args.knn_metric,
        )
    elif method.lower() == "mahalanobis":
        return MahalanobisTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
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
            embedding_dir=embedding_dir.replace("toxicity_", "toxicity_logit_"),
            max_new_tokens=args.llama_guard_max_tokens,
            temperature=args.llama_guard_temperature,
        )
    elif method.lower() == "llamaguard_logit_vllm":
        return LlamaGuardLogitVLLMTextPostprocessor(
            model_id=args.llama_guard_model_id,
            batch_size=args.batch_size,
            embedding_dir=embedding_dir.replace("toxicity_", "toxicity_logit_vllm_"),
            max_tokens=args.llama_guard_max_tokens,
            temperature=args.llama_guard_temperature,
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.75),
        )
    # openai moderation
    elif method.lower() == "openai_omni_moderation":
        return OpenaiOmniModerationPostprocessor(
            api_key=args.openai_api_key,
            model_id=args.openai_moderation_model_id,
            batch_size=args.batch_size,
            embedding_dir=embedding_dir,
            threshold=args.openai_moderation_threshold,
        )

    # perspective api
    elif method.lower() == "perspective":
        return PerspectiveTextPostprocessor(
            api_key=args.perspective_api_key,
            embedding_dir=embedding_dir,
            threshold=args.perspective_threshold,
            rate_limit_delay=args.perspective_rate_limit_delay,
            # Pass other args from the script that this class will ignore
            batch_size=args.batch_size,
            device=args.device,
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
            embedding_dir=embedding_dir.replace("toxicity_", "wildguard_toxicity_"),
            max_new_tokens=args.wildguard_max_tokens,
            temperature=args.wildguard_temperature,
        )
    elif method.lower() == "wildguard_vllm":
        return WildGuardVLLMTextPostprocessor(
            model_id=args.wildguard_model_id,
            batch_size=args.batch_size,
            embedding_dir=embedding_dir.replace("toxicity_", "wildguard_vllm_toxicity_"),
            max_tokens=args.wildguard_max_tokens,
            temperature=args.wildguard_temperature,
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.75),
        )
    elif method.lower() == "llmguard_tox":
        return create_llm_guard_toxicity_detector(
            batch_size=args.batch_size,
            embedding_dir=embedding_dir.replace("jailbreak_", "llmguard_tox_"),
            fail_fast=args.llmg_fail_fast,
            toxicity_threshold=args.llmg_toxicity_threshold,
            lang=args.llmg_lang,
            token_limit=args.llmg_token_limit,
            agg_mode=args.llmg_agg,
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


def run_benchmark_evaluation(
    detector: Any,
    safe_test_texts: list[str],
    toxic_datasets: dict,
    method: str,
    args: argparse.Namespace,
) -> dict:
    """Run evaluation across multiple toxicity benchmarks."""
    benchmark_results = {}

    for dataset_name, toxic_texts in toxic_datasets.items():
        logger.info(f"Evaluating {method} on {dataset_name}...")

        if method.lower().startswith("forte_"):
            # Use Forte API
            results = detector.evaluate(safe_test_texts, toxic_texts)
            benchmark_results[dataset_name] = results
        else:
            # Use postprocessor-based methods

            # Clean up memory before VLLM-based model inference
            if method.lower() in ["mdjudge_vllm", "polyguard_vllm"]:
                logger.info(f"Cleaning up memory before {method} inference...")
                cleanup_vllm_memory(method_name=f"{method}_pre_inference")

            # Get scores for safe and toxic data
            safe_scores = detector.postprocess(
                safe_test_texts, cache_name=f"{method.lower()}_safe_test"
            )

            toxic_scores = detector.postprocess(
                toxic_texts, cache_name=f"{method.lower()}_{dataset_name}_toxic"
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

            results = {"AUROC": auroc, "FPR@95TPR": fpr_at_95tpr, "AUPRC": auprc, "F1": best_f1}
            benchmark_results[dataset_name] = results

        logger.info(
            f"{method} on {dataset_name}: AUROC={results['AUROC']:.4f}, FPR@95={results['FPR@95TPR']:.4f}"
        )

        # Clean up memory after each dataset evaluation for VLLM-based models
        if method.lower() in ["mdjudge_vllm", "polyguard_vllm"]:
            logger.info(f"Cleaning up memory after {method} evaluation on {dataset_name}...")
            cleanup_vllm_memory(method_name=f"{method}_{dataset_name}_post")

    return benchmark_results


def save_results_to_csv(
    all_results: dict,
    embedding_model: str,
    safe_dataset: str,
    datasets_loaded: list[str],
    args: argparse.Namespace | None = None,
) -> str:
    """Save results to a wide-format CSV file with methods as rows and datasets/metrics as columns."""

    # Get all possible metrics from the first result to create the columns
    try:
        first_method = list(all_results.keys())[0]
        first_benchmark = list(all_results[first_method].keys())[0]
        metrics_names = sorted(list(all_results[first_method][first_benchmark].keys()))
    except IndexError:
        logger.warning("No results to save.")
        return ""

    # Create a multi-level header for the DataFrame
    header = pd.MultiIndex.from_product(
        [sorted(datasets_loaded), metrics_names], names=["Dataset", "Metric"]
    )

    # Create the DataFrame with methods as the index
    df = pd.DataFrame(index=sorted(all_results.keys()), columns=header)
    df.index.name = "Method"

    # Populate the DataFrame
    for method, benchmark_results in all_results.items():
        for dataset_name, metrics in benchmark_results.items():
            # Ensure the dataset is in the columns before trying to access it
            if dataset_name in df.columns.get_level_values("Dataset"):
                for metric_name, value in metrics.items():
                    if (dataset_name, metric_name) in df.columns:
                        # Format float values to 4 decimal places
                        if isinstance(value, float):
                            df.loc[method, (dataset_name, metric_name)] = f"{value:.4f}"
                        else:
                            df.loc[method, (dataset_name, metric_name)] = value

    # Add metadata as additional columns.
    df["Embedding_Model"] = embedding_model
    df["Safe_Dataset"] = safe_dataset

    # Generate filename with timestamp
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"toxicity_detection_results_{timestamp}.csv"
    filepath = f"results/{filename}"

    # Save main results to CSV
    df.to_csv(filepath)

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
            "embedding_model": embedding_model,
            "safe_dataset": safe_dataset,
            "datasets_loaded": datasets_loaded,
            "experiment_parameters": args_dict,
        }

        # Save metadata to JSON file
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Experiment parameters saved to {metadata_filepath}")

        # Also add key parameters as columns to the CSV for quick reference
        df["Seed"] = args.seed
        df["Batch_Size"] = args.batch_size
        df["Max_Safe_Samples"] = args.max_safe_samples
        df["Max_Toxic_Samples"] = args.max_toxic_samples
        df["Num_Epochs"] = args.num_epochs

        # Save again with the additional parameter columns
        df.to_csv(filepath)
    logger.info(f"Results saved to {filepath}")

    return filepath


def main(args: argparse.Namespace) -> dict:
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.manual_seed(args.seed)
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU total memory: {total_memory:.1f} GB")

    logger.info(f"Running toxicity detection baselines with configuration: {args}")

    # Load safe instructions (ID distribution)
    logger.info("\n=== Phase 1: Loading Safe Instructions ===")
    safe_dataset = getattr(args, "safe_dataset", "alpaca")
    safe_train_texts, safe_test_texts = load_safe_instructions(
        max_samples=args.max_safe_samples, dataset_name=safe_dataset
    )

    # Load toxic data from multiple benchmarks (OOD)
    logger.info("\n=== Phase 2: Loading Toxic Benchmark Data ===")
    toxic_datasets = {}

    # Available real datasets - you can configure which ones to use
    available_datasets = [
        "realtoxicity",
        "civil_comments",
        "hateval",
        "davidson",
        "hasoc",
        "offenseval",
        "xsafety",  # XSafety multilingual safety dataset (all languages)
        "xsafety_zh",  # XSafety Chinese
        "xsafety_ar",  # XSafety Arabic
        "xsafety_sp",  # XSafety Spanish
        "xsafety_fr",  # XSafety French
        "xsafety_de",  # XSafety German
        "xsafety_bn",  # XSafety Bengali
        "xsafety_en",  # XSafety English
        "xsafety_ja",  # XSafety Japanese
        "xsafety_hi",  # XSafety Hindi
        "xsafety_ru",  # XSafety Russian
        "polyguard",  # Default to social_media subset
        "polyguard_social_media",  # Social media safety
        "polyguard_education",  # Education safety
        "polyguard_hr",  # HR/workplace safety
        "polyguard_finance_input",  # Financial input safety
        "polyguard_finance_output",  # Financial output safety
        "polyguard_law_input",  # Legal input safety
        "polyguard_law_output",  # Legal output safety
        "polyguard_regulation_input",  # Regulation input safety
        "polyguard_regulation_output",  # Regulation output safety
        "polyguard_code",  # Code safety
        "polyguard_cyber",  # Cybersecurity
        "rtp_alx",  # Default to English RTP_ALX
        "rtp_alx_en",  # English
        "rtp_alx_es",  # Spanish
        "rtp_alx_fr",  # French
        "rtp_alx_de",  # German
        "rtp_alx_it",  # Italian
        "rtp_alx_pt",  # Portuguese
        "rtp_alx_ru",  # Russian
        "rtp_alx_ja",  # Japanese
        "rtp_alx_ko",  # Korean
        "rtp_alx_zh",  # Chinese (Simplified)
        "rtp_alx_nl",  # Dutch
        "rtp_alx_pl",  # Polish
        "rtp_alx_tr",  # Turkish
        "rtp_alx_hi",  # Hindi
        "rtp_alx_id",  # Indonesian
        "rtp_alx_th",  # Thai
        "rtp_alx_vi",  # Vietnamese
        "rtp_alx_sv",  # Swedish
        "rtp_alx_no",  # Norwegian
        "rtp_alx_da",  # Danish
        "rtp_alx_fi",  # Finnish
        "rtp_alx_cs",  # Czech
        "rtp_alx_hu",  # Hungarian
        "rtp_alx_uk",  # Ukrainian
        "rtp_alx_he",  # Hebrew
        "rtp_alx_el",  # Greek
        "rtp_alx_bg",  # Bulgarian
        "rtp_alx_ro",  # Romanian
        "rtp_alx_sk",  # Slovak
        "rtp_alx_sl",  # Slovenian
        "rtp_alx_lt",  # Lithuanian
        "rtp_alx_lv",  # Latvian
        "rtp_alx_et",  # Estonian
        "rtp_alx_ca",  # Catalan
        "rtp_alx_cy",  # Welsh
        "rtp_alx_sw",  # Swahili
        "polyguardprompts",
    ]

    # Configure which datasets to load (you can modify this list)
    datasets_to_load = getattr(args, "datasets", "realtoxicity,civil_comments").split(",")
    datasets_to_load = [d.strip() for d in datasets_to_load if d.strip()]

    if not datasets_to_load:
        datasets_to_load = ["realtoxicity"]  # Default fallback

    logger.info(f"Loading datasets: {', '.join(datasets_to_load)}")

    for benchmark in datasets_to_load:
        if benchmark.lower() in [d.lower() for d in available_datasets]:
            try:
                toxic_texts = load_toxic_data(benchmark, max_samples=args.max_toxic_samples)
                if toxic_texts:
                    toxic_datasets[benchmark] = toxic_texts
                    logger.info(f"Successfully loaded {len(toxic_texts)} samples from {benchmark}")
                else:
                    logger.warning(f"No samples loaded from {benchmark}")
            except Exception as e:
                logger.error(f"Failed to load {benchmark}: {e}")
                logger.info(f"Skipping {benchmark} due to loading error")
        else:
            logger.warning(f"Unknown dataset '{benchmark}' - skipping")
            logger.info(f"Available datasets: {', '.join(available_datasets)}")

    if not toxic_datasets:
        logger.error("No toxic datasets were successfully loaded!")
        raise RuntimeError("Failed to load any toxic datasets")

    # Slice datasets to a smaller subset if specified for testing
    if args.test_subset_size is not None and args.test_subset_size > 0:
        logger.warning(
            f"--- RUNNING ON A SUBSET OF {args.test_subset_size} SAMPLES FOR TESTING ---"
        )
        safe_train_texts = safe_train_texts[: args.test_subset_size]
        safe_test_texts = safe_test_texts[: args.test_subset_size]

        # Loop through the toxic datasets dictionary
        for dataset_name in list(toxic_datasets.keys()):
            toxic_datasets[dataset_name] = toxic_datasets[dataset_name][: args.test_subset_size]

        logger.info("--- DATASET SUBSETTING COMPLETE ---")
    # Parse methods to run
    methods_to_run = [m.strip().upper() for m in args.methods.split(",") if m.strip()]

    # Results storage
    all_results = {}

    # Run each method
    for method in methods_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {method} Method")
        logger.info(f"{'='*60}")

        # Create and setup detector
        logger.info(f"\n=== {method}: Model Initialization ===")

        # Clean up memory before VLLM-based models
        if method.lower() in ["mdjudge_vllm", "polyguard_vllm"]:
            logger.info(f"Cleaning up memory before initializing {method}...")
            cleanup_vllm_memory(method_name=f"{method}_initialization")

        detector = create_postprocessor(method, args)

        logger.info(f"\n=== {method}: Setup on Safe Instructions ===")
        start_time = time.time()
        if method.lower().startswith("forte_"):
            # Use Forte API
            detector.fit(safe_train_texts)
        else:
            # Use postprocessor-based methods
            detector.setup(safe_train_texts, random_state=args.seed)
        setup_time = time.time() - start_time
        logger.info(f"{method} setup completed in {setup_time:.2f} seconds")

        # Evaluate across all benchmarks
        logger.info(f"\n=== {method}: Multi-Benchmark Evaluation ===")
        benchmark_results = run_benchmark_evaluation(
            detector, safe_test_texts, toxic_datasets, method, args
        )
        all_results[method] = benchmark_results

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

            # Final memory status check
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
    logger.info("TOXICITY DETECTION RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Embedding Model: {args.embedding_model}")
    logger.info("")

    # Get actual loaded datasets for table header
    loaded_datasets = list(toxic_datasets.keys())

    # Header
    header = f"{'Method':<12}"
    for benchmark in loaded_datasets:
        header += f"{benchmark.upper():<20}"
    logger.info(header)
    logger.info("-" * len(header))

    # Results for each method
    for method in methods_to_run:
        if method in all_results:
            row = f"{method:<12}"
            for benchmark in loaded_datasets:
                if benchmark in all_results[method]:
                    auroc = all_results[method][benchmark]["AUROC"]
                    fpr95 = all_results[method][benchmark]["FPR@95TPR"]
                    row += f"{auroc:.3f}/{fpr95:.3f}     "
                else:
                    row += f"{'N/A':<20}"
            logger.info(row)

    # Save results to CSV
    csv_filepath = save_results_to_csv(
        all_results, args.embedding_model, safe_dataset, loaded_datasets, args
    )
    logger.info(f"\nResults have been saved to: {csv_filepath}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Baselines - Toxicity Detection")
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
        default="MDJudge_VLLM,RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax,Energy,KNN,Mahalanobis,Forte_GMM,Forte_OCSVM,LlamaGuard,LlamaGuard_Logit,LlamaGuard_Logit_VLLM,DuoGuard,WildGuard,WildGuard_VLLM,PolyGuard_VLLM",
        help="Comma-separated list of methods to run (include Forte_GMM, Forte_KDE, Forte_OCSVM for Forte API, Energy for Energy-based OOD, KNN for k-NN distance, Mahalanobis for Mahalanobis distance, LlamaGuard for Llama Guard 3-1B with hardcoded scores, LlamaGuard_Logit for Llama Guard 3-1B with logit-based scoring (transformers), LlamaGuard_Logit_VLLM for Llama Guard 3-1B with logit-based scoring (vLLM - faster), DuoGuard for DuoGuard-0.5B, WildGuard for AllenAI WildGuard 7B (transformers), WildGuard_VLLM for AllenAI WildGuard 7B (vLLM - faster), MDJudge_VLLM for VLLM optimized version, PolyGuard_VLLM for multilingual safety)",
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
    # Energy-based parameters
    parser.add_argument(
        "--energy_temperature",
        type=float,
        default=1.0,
        help="Temperature parameter for Energy-based OOD detection",
    )
    # KNN parameters
    parser.add_argument(
        "--knn_neighbors",
        type=int,
        default=1,
        help="Number of nearest neighbors for KNN OOD detection",
    )
    parser.add_argument(
        "--knn_metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine", "manhattan"],
        help="Distance metric for KNN",
    )
    # Mahalanobis parameters (no hyperparameters needed)

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
    parser.add_argument(
        "--llmg_prompt_injection_threshold",
        type=float,
        default=0.5,
        help="Threshold for PromptInjection scanner",
    )

    parser.add_argument(
        "--llmg_toxicity_threshold", type=float, default=0.5, help="Threshold for Toxicity scanner"
    )

    parser.add_argument("--llmg_token_limit", type=int, default=2048, help="TokenLimit for prompts")

    parser.add_argument(
        "--llmg_agg",
        type=str,
        default="max",
        choices=["max", "mean", "weighted"],
        help="Aggregation mode over scanner risks",
    )

    parser.add_argument(
        "--llmg_fail_fast", action="store_true", help="Stop scanning at first violation"
    )

    parser.add_argument("--llmg_lang", type=str, default="en", help="Language for Toxicity scanner")

    parser.add_argument(
        "--llmg_weights",
        type=str,
        default=None,
        help="Weights for 'weighted' aggregation, e.g. 'PromptInjection:0.7,Toxicity:0.3'",
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

    # OpenAI Omni Moderation parameters
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key. If not provided, the OPENAI_API_KEY environment variable will be used.",
    )
    parser.add_argument(
        "--openai_moderation_model_id",
        type=str,
        default="omni-moderation-latest",
        help="OpenAI Moderation model ID to use.",
    )
    parser.add_argument(
        "--openai_moderation_threshold",
        type=float,
        default=0.5,
        help="OpenAI Moderation threshold for binary classification.",
    )

    # Perspective API parameters
    parser.add_argument(
        "--perspective_api_key",
        type=str,
        default=os.getenv("PERSPECTIVE_API_KEY", ""),
        help="Google Perspective API key. Set the PERSPECTIVE_API_KEY environment variable or pass --perspective_api_key.",
    )
    parser.add_argument(
        "--perspective_threshold",
        type=float,
        default=0.5,
        help="Perspective API threshold for binary classification.",
    )
    parser.add_argument(
        "--perspective_rate_limit_delay",
        type=float,
        default=1.05,
        help="Delay between Perspective API calls (seconds).",
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
        "--max_safe_samples", type=int, default=5000, help="Max safe instruction samples"
    )
    parser.add_argument(
        "--max_toxic_samples", type=int, default=1000, help="Max toxic samples per benchmark"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="realtoxicity,civil_comments,hateval,davidson,hasoc,offenseval,rtp_alx_en,polyguard_social_media,xsafety_hi,polyguardprompts",
        help="Comma-separated list of toxic datasets to load (realtoxicity,civil_comments,hateval,davidson,hasoc,offenseval,xsafety,polyguardprompts,xsafety_[lang] where [lang] can be zh,ar,sp,fr,de,bn,en,ja,hi,ru,polyguard,polyguard_[subset] where subset can be social_media,education,hr,finance_input,finance_output,law_input,law_output,regulation_input,regulation_output,code,cyber,rtp_alx,rtp_alx_[lang] where [lang] can be en,es,fr,de,it,pt,ru,ja,ko,zh,nl,pl,tr,hi,id,th,vi,sv,no,da,fi,cs,hu,uk,he,el,bg,ro,sk,sl,lt,lv,et,ca,cy,sw)",
    )
    parser.add_argument(
        "--safe_dataset",
        type=str,
        default="alpaca",
        help="Safe instruction dataset to use as ID distribution (alpaca,dolly,helpful_base,openassistant,id_mix)",
    )
    # General parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--test_subset_size",
        type=int,
        default=None,
        help="Run on a small subset of N samples for quick testing. If None, runs on all data.",
    )

    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")

    args = parser.parse_args()
    main(args)
