#!/usr/bin/env python3
"""
Filename: unified_baselines_jailbreaking_detection.py
Unified Baseline Script for Jailbreaking and Adversarial Prompt Detection

Evaluates on multiple jailbreaking/adversarial benchmarks:
- AdvBench: Adversarial prompts designed to bypass safety measures
- HarmBench: Harmful instruction benchmarks
- JailbreakBench: Comprehensive jailbreaking attempts
- RedTeam: Red team attack prompts
- MaliciousInstruct: Malicious instruction dataset

Uses safe/helpful instructions as ID and adversarial prompts as OOD.
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
    create_llm_guard_jailbreak_detector,
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
logger = logging.getLogger("JailbreakingDetectionBaselines")


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


def load_adversarial_data(dataset_name: str, max_samples: int = 500) -> list[str]:
    """Load adversarial/jailbreaking prompts from various benchmarks."""
    logger.info(f"Loading adversarial data from {dataset_name}...")

    adversarial_texts: list[str] = []

    if dataset_name.lower() == "advbench":
        try:
            # Load actual AdvBench dataset
            dataset = load_dataset("AlignmentResearch/AdvBench", split="train")
            logger.info(f"AdvBench dataset loaded with {len(dataset)} examples")

            # Extract harmful instructions from the dataset
            # The actual harmful instructions are in the 'content' field (which is a list)
            for example in dataset:
                if len(adversarial_texts) >= max_samples:
                    break

                # Extract the harmful instruction from 'content' field (it's a list)
                content = example.get("content", [])
                if content and len(content) > 0:
                    instruction = content[0]  # Take the first (and usually only) item from the list
                    if instruction and len(instruction.strip()) > 20:
                        adversarial_texts.append(instruction.strip())

        except Exception as e:
            logger.error(f"Could not load AdvBench dataset: {e}")
            raise RuntimeError(f"Failed to load AdvBench dataset: {e}") from e

    elif dataset_name.lower() == "harmbench":
        try:
            # Load HarmBench dataset - try different configs
            # Available configs: 'contextual', 'copyright', 'standard'
            configs_to_try = ["standard", "contextual", "copyright"]
            dataset_loaded = False

            for config in configs_to_try:
                try:
                    dataset = load_dataset("walledai/HarmBench", config, split="train")
                    logger.info(f"HarmBench {config} dataset loaded with {len(dataset)} examples")
                    dataset_loaded = True

                    for example in dataset:
                        if len(adversarial_texts) >= max_samples:
                            break

                        # HarmBench has 'prompt' field containing the harmful instruction
                        prompt = example.get("prompt", "")
                        if prompt and len(prompt.strip()) > 20:
                            adversarial_texts.append(prompt.strip())

                    break  # Successfully loaded one config, stop trying others

                except Exception as config_e:
                    logger.debug(f"Could not load HarmBench config '{config}': {config_e}")
                    continue

            if not dataset_loaded:
                raise RuntimeError("Failed to load any HarmBench configuration")

        except Exception as e:
            logger.error(f"Could not load any HarmBench dataset: {e}")
            raise RuntimeError(f"Failed to load HarmBench dataset: {e}") from e

    elif dataset_name.lower() == "harmbench_standard":
        try:
            # Load specific HarmBench standard subset
            dataset = load_dataset("walledai/HarmBench", "standard", split="train")
            logger.info(f"HarmBench standard dataset loaded with {len(dataset)} examples")

            for example in dataset:
                if len(adversarial_texts) >= max_samples:
                    break

                prompt = example.get("prompt", "")
                if prompt and len(prompt.strip()) > 20:
                    adversarial_texts.append(prompt.strip())

        except Exception as e:
            logger.error(f"Could not load HarmBench standard dataset: {e}")
            raise RuntimeError(f"Failed to load HarmBench standard dataset: {e}") from e

    elif dataset_name.lower() == "harmbench_contextual":
        try:
            # Load specific HarmBench contextual subset
            dataset = load_dataset("walledai/HarmBench", "contextual", split="train")
            logger.info(f"HarmBench contextual dataset loaded with {len(dataset)} examples")

            for example in dataset:
                if len(adversarial_texts) >= max_samples:
                    break

                prompt = example.get("prompt", "")
                if prompt and len(prompt.strip()) > 20:
                    adversarial_texts.append(prompt.strip())

        except Exception as e:
            logger.error(f"Could not load HarmBench contextual dataset: {e}")
            raise RuntimeError(f"Failed to load HarmBench contextual dataset: {e}") from e

    elif dataset_name.lower() == "harmbench_copyright":
        try:
            # Load specific HarmBench copyright subset as mentioned in your example
            dataset = load_dataset("walledai/HarmBench", "copyright", split="train")
            logger.info(f"HarmBench copyright dataset loaded with {len(dataset)} examples")

            for example in dataset:
                if len(adversarial_texts) >= max_samples:
                    break

                prompt = example.get("prompt", "")
                if prompt and len(prompt.strip()) > 20:
                    adversarial_texts.append(prompt.strip())

        except Exception as e:
            logger.error(f"Could not load HarmBench copyright dataset: {e}")
            raise RuntimeError(f"Failed to load HarmBench copyright dataset: {e}") from e

    elif dataset_name.lower() == "jailbreakbench":
        try:
            # Load JailbreakBench dataset - try available configs
            configs_to_try = ["behaviors", "judge_comparison"]
            dataset_loaded = False

            for config in configs_to_try:
                try:
                    dataset = load_dataset("JailbreakBench/JBB-Behaviors", config, split="test")
                    logger.info(
                        f"JailbreakBench {config} dataset loaded with {len(dataset)} examples"
                    )
                    dataset_loaded = True

                    for example in dataset:
                        if len(adversarial_texts) >= max_samples:
                            break

                        # JailbreakBench typically has 'behavior' or 'prompt' field
                        prompt = (
                            example.get("behavior", "")
                            or example.get("prompt", "")
                            or example.get("goal", "")
                        )
                        if prompt and len(prompt.strip()) > 20:
                            adversarial_texts.append(prompt.strip())

                    break  # Successfully loaded one config, stop trying others

                except Exception as config_e:
                    logger.debug(f"Could not load JailbreakBench config '{config}': {config_e}")
                    continue

            if not dataset_loaded:
                raise RuntimeError("Failed to load any JailbreakBench configuration")

        except Exception as e:
            logger.error(f"Could not load JailbreakBench dataset: {e}")
            raise RuntimeError(f"Failed to load JailbreakBench dataset: {e}") from e

    elif dataset_name.lower() == "maliciousinstruct":
        try:
            # Load MaliciousInstruct dataset
            dataset = load_dataset("walledai/MaliciousInstruct", split="train")
            logger.info(f"MaliciousInstruct dataset loaded with {len(dataset)} examples")

            for example in dataset:
                if len(adversarial_texts) >= max_samples:
                    break

                instruction = example.get("instruction", "") or example.get("prompt", "")
                if instruction and len(instruction.strip()) > 20:
                    adversarial_texts.append(instruction.strip())

        except Exception as e:
            logger.error(f"Could not load MaliciousInstruct dataset: {e}")
            raise RuntimeError(f"Failed to load MaliciousInstruct dataset: {e}") from e

    elif dataset_name.lower() == "xstest":
        try:
            # Load XSTest dataset for safety evaluation
            dataset = load_dataset("natolambert/xstest-v2-copy", split="gpt4")
            logger.info(f"XSTest dataset loaded with {len(dataset)} examples")

            for example in dataset:
                if len(adversarial_texts) >= max_samples:
                    break

                prompt = example.get("prompt", "")
                final_label = example.get("final_label", "")

                # XSTest uses specific labels:
                # "1_full_compliance" = safe prompts (should comply)
                # "2_full_refusal" = unsafe prompts (should refuse)
                # "3_partial_refusal" = borderline prompts
                # We want the unsafe prompts for adversarial evaluation
                is_unsafe = final_label == "2_full_refusal"

                if is_unsafe and prompt and len(prompt.strip()) > 20:
                    adversarial_texts.append(prompt.strip())

        except Exception as e:
            logger.error(f"Could not load XSTest dataset: {e}")
            raise RuntimeError(f"Failed to load XSTest dataset: {e}") from e

    elif dataset_name.lower() == "beavertails":
        try:
            # Load BeaverTails dataset (harmful QA pairs)
            dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
            logger.info(f"BeaverTails dataset loaded with {len(dataset)} examples")

            for example in dataset:
                if len(adversarial_texts) >= max_samples:
                    break

                prompt = example.get("prompt", "")
                is_safe = example.get("is_safe", True)
                # Only use unsafe prompts
                if not is_safe and prompt and len(prompt.strip()) > 20:
                    adversarial_texts.append(prompt.strip())

        except Exception as e:
            logger.error(f"Could not load BeaverTails dataset: {e}")
            raise RuntimeError(f"Failed to load BeaverTails dataset: {e}") from e

    elif dataset_name.lower() == "sorrybench":
        try:
            dataset = load_dataset("sorry-bench/sorry-bench-202406", split="train")

            for example in dataset:
                if len(adversarial_texts) >= max_samples:
                    break
                turns_list = example.get("turns", [])
                if turns_list and isinstance(turns_list, list):
                    prompt = turns_list[0]
                    if prompt and len(prompt.strip()) > 20:
                        adversarial_texts.append(prompt.strip())

        except Exception as e:
            logger.error(f"Could not load Sorry-Bench dataset: {e}")
            raise RuntimeError(f"Failed to load Sorry-Bench dataset: {e}") from e

    elif dataset_name.lower() == "wildguardmix_train":
        try:
            # Load WildGuardMix training split (GPT-4 labeled, larger dataset)
            dataset = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
            logger.info(f"WildGuardMix train dataset loaded with {len(dataset)} examples")

            for example in dataset:
                if len(adversarial_texts) >= max_samples:
                    break

                prompt = example.get("prompt", "")
                prompt_harm_label = example.get("prompt_harm_label", "")
                is_adversarial = example.get("adversarial", False)

                # Include if harmful OR adversarial
                is_target = (prompt_harm_label == "harmful") or (is_adversarial is True)

                if is_target and prompt and len(prompt.strip()) > 20:
                    adversarial_texts.append(prompt.strip())

        except Exception as e:
            logger.error(f"Could not load WildGuardMix train dataset: {e}")
            raise RuntimeError(f"Failed to load WildGuardMix train dataset: {e}") from e

    elif dataset_name.lower() == "wildguardmix_test":
        try:
            # Load WildGuardMix test split (human annotated, high quality)
            dataset = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
            logger.info(f"WildGuardMix test dataset loaded with {len(dataset)} examples")

            for example in dataset:
                if len(adversarial_texts) >= max_samples:
                    break

                prompt = example.get("prompt", "")
                prompt_harm_label = example.get("prompt_harm_label", "")
                is_adversarial = example.get("adversarial", False)

                # Include if harmful OR adversarial
                is_target = (prompt_harm_label == "harmful") or (is_adversarial is True)

                if is_target and prompt and len(prompt.strip()) > 20:
                    adversarial_texts.append(prompt.strip())

        except Exception as e:
            logger.error(f"Could not load WildGuardMix test dataset: {e}")
            raise RuntimeError(f"Failed to load WildGuardMix test dataset: {e}") from e

    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        available_datasets = [
            "advbench",
            "harmbench",
            "harmbench_standard",
            "harmbench_contextual",
            "harmbench_copyright",
            "jailbreakbench",
            "maliciousinstruct",
            "xstest",
            "beavertails",
            "sorrybench",
            "wildguardmix_train",
            "wildguardmix_test",
        ]
        logger.error(f"Available datasets: {', '.join(available_datasets)}")
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available datasets: {', '.join(available_datasets)}"
        )

    logger.info(f"Loaded {len(adversarial_texts)} {dataset_name} adversarial texts")
    return adversarial_texts[:max_samples]


def create_postprocessor(method: str, args: argparse.Namespace) -> Any:
    """Create postprocessor based on method."""
    safe_dataset = getattr(args, "safe_dataset", "alpaca")
    embedding_dir = f"embeddings_{method.lower()}_jailbreak_{safe_dataset}"

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
            max_tokens=20,
            temperature=args.llama_guard_temperature,
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.75),
        )
    elif method.lower() == "llmguard_jb":
        return create_llm_guard_jailbreak_detector(
            batch_size=args.batch_size,
            embedding_dir=embedding_dir.replace("jailbreak_", "llmguard_jb_"),
            fail_fast=args.llmg_fail_fast,
            prompt_injection_threshold=args.llmg_prompt_injection_threshold,
            token_limit=args.llmg_token_limit,
            agg_mode=args.llmg_agg,
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
            max_tokens=20000,
            temperature=args.wildguard_temperature,
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.75),
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
    adversarial_datasets: dict,
    method: str,
    args: argparse.Namespace,
) -> dict:
    """Run evaluation across multiple jailbreaking benchmarks."""
    benchmark_results = {}

    for dataset_name, adversarial_texts in adversarial_datasets.items():
        logger.info(f"Evaluating {method} on {dataset_name}...")

        if method.lower().startswith("forte_"):
            # Use Forte API
            results = detector.evaluate(safe_test_texts, adversarial_texts)
            benchmark_results[dataset_name] = results
        else:
            # Use postprocessor-based methods

            # Clean up memory before VLLM-based model inference
            if method.lower() in ["mdjudge_vllm", "polyguard_vllm"]:
                logger.info(f"Cleaning up memory before {method} inference...")
                cleanup_vllm_memory(method_name=f"{method}_pre_inference")

            # Get scores for safe and adversarial data
            safe_scores = detector.postprocess(
                safe_test_texts, cache_name=f"{method.lower()}_safe_test"
            )

            adversarial_scores = detector.postprocess(
                adversarial_texts, cache_name=f"{method.lower()}_{dataset_name}_adversarial"
            )

            # Compute metrics directly from scores to avoid cache reuse
            labels = np.concatenate([np.ones(len(safe_scores)), np.zeros(len(adversarial_scores))])
            scores_all = np.concatenate([safe_scores, adversarial_scores])

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

    return benchmark_results


def save_results_to_csv(
    all_results: dict,
    embedding_model: str,
    safe_dataset: str,
    datasets_loaded: list[str],
    num_safe_train: int,
    num_safe_test: int,
    num_adversarial_per_dataset: dict[str, int],
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
    df["Safe_Train_Samples"] = num_safe_train
    df["Safe_Test_Samples"] = num_safe_test

    # Generate filename with timestamp
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"jailbreaking_detection_results_{timestamp}.csv"
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
            "dataset": "jailbreaking_detection",
            "embedding_model": embedding_model,
            "safe_dataset": safe_dataset,
            "datasets_loaded": datasets_loaded,
            "num_safe_train": num_safe_train,
            "num_safe_test": num_safe_test,
            "num_adversarial_per_dataset": num_adversarial_per_dataset,
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
        df["Max_Adversarial_Samples"] = args.max_adversarial_samples
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

    logger.info(f"Running jailbreaking detection baselines with configuration: {args}")

    # Load safe instructions (ID distribution)
    logger.info("\n=== Phase 1: Loading Safe Instructions ===")
    safe_dataset = getattr(args, "safe_dataset", "alpaca")
    safe_train_texts, safe_test_texts = load_safe_instructions(
        max_samples=args.max_safe_samples, dataset_name=safe_dataset
    )

    # Load adversarial data from multiple benchmarks (OOD)
    logger.info("\n=== Phase 2: Loading Adversarial Benchmark Data ===")
    adversarial_datasets = {}

    # Available real datasets - you can configure which ones to use
    available_datasets = [
        "advbench",
        "harmbench",
        "harmbench_standard",
        "harmbench_contextual",
        "harmbench_copyright",
        "jailbreakbench",
        "maliciousinstruct",
        "xstest",
        "beavertails",
        "sorrybench",
        "wildguardmix_train",
        "wildguardmix_test",
    ]

    # Configure which datasets to load (you can modify this list)
    datasets_to_load = getattr(args, "datasets", "advbench,harmbench").split(",")
    datasets_to_load = [d.strip() for d in datasets_to_load if d.strip()]

    if not datasets_to_load:
        datasets_to_load = ["advbench"]  # Default fallback

    logger.info(f"Loading datasets: {', '.join(datasets_to_load)}")

    for benchmark in datasets_to_load:
        if benchmark.lower() in [d.lower() for d in available_datasets]:
            try:
                adversarial_texts = load_adversarial_data(
                    benchmark, max_samples=args.max_adversarial_samples
                )
                if adversarial_texts:
                    adversarial_datasets[benchmark] = adversarial_texts
                    logger.info(
                        f"Successfully loaded {len(adversarial_texts)} samples from {benchmark}"
                    )
                else:
                    logger.warning(f"No samples loaded from {benchmark}")
            except Exception as e:
                logger.error(f"Failed to load {benchmark}: {e}")
                logger.info(f"Skipping {benchmark} due to loading error")
        else:
            logger.warning(f"Unknown dataset '{benchmark}' - skipping")
            logger.info(f"Available datasets: {', '.join(available_datasets)}")

    if not adversarial_datasets:
        logger.error("No adversarial datasets were successfully loaded!")
        raise RuntimeError("Failed to load any adversarial datasets")

    # Slice datasets to a smaller subset if specified for testing
    if args.test_subset_size is not None and args.test_subset_size > 0:
        logger.warning(
            f"--- RUNNING ON A SUBSET OF {args.test_subset_size} SAMPLES FOR TESTING ---"
        )
        safe_train_texts = safe_train_texts[: args.test_subset_size]
        safe_test_texts = safe_test_texts[: args.test_subset_size]

        # Loop through the adversarial datasets dictionary
        for dataset_name in list(adversarial_datasets.keys()):
            adversarial_datasets[dataset_name] = adversarial_datasets[dataset_name][
                : args.test_subset_size
            ]

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
            detector, safe_test_texts, adversarial_datasets, method, args
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
    logger.info("JAILBREAKING DETECTION RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Embedding Model: {args.embedding_model}")
    logger.info(f"Safe Dataset: {safe_dataset}")
    logger.info(f"Safe Training Samples: {len(safe_train_texts)}")
    logger.info(f"Safe Test Samples: {len(safe_test_texts)}")

    # Calculate sample counts for logging
    adversarial_counts = {dataset: len(texts) for dataset, texts in adversarial_datasets.items()}
    for dataset, count in adversarial_counts.items():
        logger.info(f"{dataset.capitalize()} Adversarial Samples: {count}")
    logger.info("")

    # Get actual loaded datasets for table header
    loaded_datasets = list(adversarial_datasets.keys())

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
        all_results,
        args.embedding_model,
        safe_dataset,
        loaded_datasets,
        len(safe_train_texts),
        len(safe_test_texts),
        adversarial_counts,
        args,
    )
    logger.info(f"\nResults have been saved to: {csv_filepath}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Baselines - Jailbreaking Detection")
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
        default="RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax,Energy,KNN,Mahalanobis,Forte_GMM,Forte_OCSVM,LlamaGuard,LlamaGuard_Logit,DuoGuard,WildGuard,MDJudge_VLLM,PolyGuard_VLLM",
        help="Comma-separated list of methods to run (include Forte_GMM, Forte_KDE, Forte_OCSVM for Forte API, Energy for Energy-based OOD, KNN for k-NN distance, Mahalanobis for Mahalanobis distance, LlamaGuard for Llama Guard 3-1B with hardcoded scores, LlamaGuard_Logit for Llama Guard 3-1B with logit-based scoring, DuoGuard for DuoGuard-0.5B, WildGuard for AllenAI WildGuard 7B, MDJudge_VLLM for VLLM optimized version, PolyGuard_VLLM for multilingual safety)",
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
    # LLM-Guard parameters (input scanners)
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

    # Data parameters
    parser.add_argument(
        "--max_safe_samples", type=int, default=1500, help="Max safe instruction samples"
    )
    parser.add_argument(
        "--max_adversarial_samples",
        type=int,
        default=1000,
        help="Max adversarial samples per benchmark",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="advbench,harmbench,harmbench_standard,harmbench_contextual,harmbench_copyright,jailbreakbench,maliciousinstruct,xstest,beavertails,sorrybench,wildguardmix_train,wildguardmix_test",
        help="Comma-separated list of adversarial datasets to load (advbench,harmbench,harmbench_standard,harmbench_contextual,harmbench_copyright,jailbreakbench,maliciousinstruct,xstest,beavertails,sorrybench,wildguardmix_train,wildguardmix_test)",
    )
    parser.add_argument(
        "--safe_dataset",
        type=str,
        default="alpaca",
        help="Safe instruction dataset to use as ID distribution (alpaca,dolly,helpful_base,openassistant,id_mix)",
    )
    # General parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument(
        "--test_subset_size",
        type=int,
        default=None,
        help="Run on a small subset of N samples for quick testing. If None, runs on all data.",
    )
    args = parser.parse_args()
    main(args)
