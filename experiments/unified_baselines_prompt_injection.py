#!/usr/bin/env python3
"""
Unified Baselines - Prompt Injection Detection (Purple Llama)

- OOD/adversarial set: Purple Llama `prompt_injection.json` (or .jsonl)
  * By default we score ONLY the "user_input" field
  * Optional: prepend "test_case_prompt" as context via --cs_use_context

- ID/safe set: safe instruction datasets (alpaca/dolly/helpful_base/openassistant/id_mix)

- Methods supported (pick any subset with --methods):
  * LLMGuard_JB      (your LLM-Guard Prompt-Injection scanner)
  * LLMGuard_TOX     (optional: for comparison)
  * LlamaGuard       (Meta Llama Guard 3-1B)
  * DuoGuard         (DuoGuard-0.5B)
  * MDJudge_VLLM     (if available)
  * PolyGuard_VLLM   (if available)
  * Forte_*          (text OOD via Forte API: Forte_GMM, Forte_OCSVM, Forte_KDE)
  * Classic OOD: RMD, VIM, CIDER, fDBD, NNGuide, ReAct, GMM, AdaScale, OpenMax

The script trains/sets up on safe ID text and evaluates on Purple Llama prompts.
It saves a wide CSV with AUROC / FPR@95TPR / AUPRC / F1 per method.
"""

from __future__ import annotations

import contextlib
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

# Make repo imports work when run from experiments/
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Optional / repo-provided baselines
from api.forte_text_api import ForteTextOODDetector
from baselines.adascale_text_postprocessor import AdaScaleTextPostprocessor
from baselines.cider_text_postprocessor import CIDERTextPostprocessor
from baselines.duoguard_text_postprocessor import DuoGuardTextPostprocessor
from baselines.fdbd_text_postprocessor import fDBDTextPostprocessor
from baselines.gmm_text_postprocessor import GMMTextPostprocessor
from baselines.llama_guard_logit_text_postprocessor import LlamaGuardLogitTextPostprocessor
from baselines.llama_guard_text_postprocessor import LlamaGuardTextPostprocessor
from baselines.llm_guard_text_postprocessor import (
    create_llm_guard_jailbreak_detector,
    create_llm_guard_toxicity_detector,
)
from baselines.nnguide_text_postprocessor import NNGuideTextPostprocessor
from baselines.openmax_text_postprocessor import OpenMaxTextPostprocessor
from baselines.react_text_postprocessor import ReActTextPostprocessor
from baselines.rmd_simple_postprocessor import SimpleRMDTextPostprocessor
from baselines.vim_text_postprocessor import VIMTextPostprocessor
from baselines.wildguard_text_postprocessor import WildGuardTextPostprocessor

# Try to import VLLM versions (ok if unavailable)
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

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PromptInjectionBaselines")


# ---------------------------
# Helpers
# ---------------------------
def cleanup_vllm_memory(model_instance: Any = None, method_name: str = "") -> None:
    """Aggressive cleanup to free GPU mem after vLLM runs (safe if no vLLM)."""
    try:
        logger.info(f"[{method_name}] vLLM cleanup starting...")
        import os

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )
        except Exception:
            destroy_distributed_environment = None
            destroy_model_parallel = None

        if model_instance is not None:
            try:
                if hasattr(model_instance, "llm_engine"):
                    le = model_instance.llm_engine
                    if hasattr(le, "model_executor"):
                        del le.model_executor
                    elif hasattr(le, "driver_worker"):
                        del le.driver_worker
                    elif hasattr(le, "engine_core"):
                        le.engine_core.shutdown()
                del model_instance
            except Exception as e:
                logger.warning(f"[{method_name}] model-specific cleanup error: {e}")

        if destroy_model_parallel:
            destroy_model_parallel()
        if destroy_distributed_environment:
            destroy_distributed_environment()

        with contextlib.suppress(Exception):
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info(f"[{method_name}] vLLM cleanup done.")
    except Exception as e:
        logger.warning(f"[{method_name}] vLLM cleanup failed softly: {e}")


def _parse_llmg_weights(s: str | None) -> dict | None:
    if not s:
        return None
    out: dict[str, float] = {}
    for kv in s.split(","):
        k, v = kv.split(":")
        out[k.strip()] = float(v)
    return out


# ---------------------------
# Data loading
# ---------------------------
def load_safe_instructions(
    max_samples: int = 1000, dataset_name: str = "alpaca"
) -> tuple[list[str], list[str]]:
    """Load safe/helpful instructions as ID distribution."""
    logger.info(f"Loading safe instruction data from {dataset_name}...")

    import random

    from datasets import load_dataset

    train_texts: list[str] = []
    test_texts: list[str] = []

    def _split_push(s: str) -> None:
        if len(train_texts) < max_samples:
            train_texts.append(s)
        elif len(test_texts) < max_samples // 2:
            test_texts.append(s)

    name = dataset_name.lower()
    if name == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        for ex in ds:
            instr = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            txt = f"{instr}\n{inp}" if inp else instr
            if len(txt) > 20:
                _split_push(txt)

    elif name == "dolly":
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        for ex in ds:
            instr = (ex.get("instruction") or "").strip()
            ctx = (ex.get("context") or "").strip()
            txt = f"{instr}\nContext: {ctx}" if ctx else instr
            if len(txt) > 20:
                _split_push(txt)

    elif name == "helpful_base":
        ds = load_dataset("Anthropic/hh-rlhf", split="train")
        for ex in ds:
            chosen = ex.get("chosen") or ""
            if "Human:" in chosen:
                human = chosen.split("Human:", 1)[1].split("Assistant:", 1)[0].strip()
                if len(human) > 20:
                    _split_push(human)

    elif name == "openassistant":
        ds = load_dataset("OpenAssistant/oasst2", split="train")
        for ex in ds:
            text = (ex.get("text") or "").strip()
            role = ex.get("role") or ""
            if role == "prompter" and len(text) > 20:
                _split_push(text)

    elif name == "id_mix":
        logger.info("Building id_mix from alpaca/dolly/helpful_base/openassistant...")
        all_names = ["alpaca", "dolly", "helpful_base", "openassistant"]
        per = max_samples // len(all_names)
        per_test = (max_samples // 2) // len(all_names)
        for nm in all_names:
            sub_train, sub_test = load_safe_instructions(
                max_samples=per + per_test, dataset_name=nm
            )
            train_texts.extend(sub_train[:per])
            test_texts.extend(sub_test[:per_test])
        random.seed(42)
        random.shuffle(train_texts)
        random.shuffle(test_texts)

    else:
        raise ValueError(
            f"Unknown safe dataset: {dataset_name}. "
            f"Available: alpaca,dolly,helpful_base,openassistant,id_mix"
        )

    logger.info(f"Loaded safe: train={len(train_texts)}, test={len(test_texts)}")
    return train_texts, test_texts


def load_cybersec_prompt_injection(
    json_path: str, use_context: bool = False, max_samples: int | None = None
) -> list[str]:
    """
    Load Purple Llama `prompt_injection.json` or `.jsonl`.
    Returns ONLY user_input by default; if use_context=True, prepends test_case_prompt.
    """
    # Try array JSON; fallback to JSONL
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Top-level JSON is not a list.")
    except Exception:
        data = []
        with open(json_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

    out: list[str] = []
    for ex in data:
        user = (ex.get("user_input") or "").strip()
        if not user:
            continue
        if use_context:
            sys_prompt = (ex.get("test_case_prompt") or "").strip()
            text = f"[SYSTEM] {sys_prompt}\n[USER] {user}" if sys_prompt else user
        else:
            text = user
        out.append(text)
        if max_samples and len(out) >= max_samples:
            break
    return out


# ---------------------------
# Model factory
# ---------------------------
def create_postprocessor(method: str, args: argparse.Namespace) -> Any:
    """Create postprocessor based on method name."""
    safe_dataset = getattr(args, "safe_dataset", "alpaca")
    embedding_dir = f"embeddings_{method.lower()}_pi_{safe_dataset}"

    m = method.lower()

    # Forte_* methods
    if m.startswith("forte_"):
        det = m.replace("forte_", "")
        det_type: Literal["gmm", "kde", "ocsvm"]
        if det == "gmm":
            det_type = "gmm"
        elif det == "kde":
            det_type = "kde"
        elif det == "ocsvm":
            det_type = "ocsvm"
        else:
            raise ValueError(f"Unsupported Forte method: {det}")

        model_names = None
        if args.forte_models:
            model_mapping = {
                "qwen3": ("qwen3", "Qwen/Qwen3-Embedding-0.6B"),
                "bge-m3": ("bge-m3", "BAAI/bge-m3"),
                "e5": ("e5", "intfloat/e5-large-v2"),
            }
            model_names = [
                model_mapping[s.strip()]
                for s in args.forte_models.split(",")
                if s.strip() in model_mapping
            ]
        return ForteTextOODDetector(
            method=det_type,
            device=args.device,
            model_names=model_names,
            embedding_dir=embedding_dir,
        )

    # Classic OOD family
    if m == "rmd":
        return SimpleRMDTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
        )
    if m == "vim":
        return VIMTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            dim=args.vim_dim,
        )
    if m == "cider":
        return CIDERTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            K=args.cider_k,
        )
    if m == "fdbd":
        return fDBDTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            distance_as_normalizer=args.fdbd_distance_normalizer,
        )
    if m == "nnguide":
        return NNGuideTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            K=args.nnguide_k,
            alpha=args.nnguide_alpha,
            min_score=args.nnguide_min_score,
        )
    if m == "react":
        return ReActTextPostprocessor(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            percentile=args.react_percentile,
        )
    if m == "gmm":
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
    if m == "adascale":
        parts = [float(x.strip()) for x in args.adascale_percentile.split(",")]
        percentile_tuple = (parts[0], parts[1]) if len(parts) == 2 else (90.0, 99.0)
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
    if m == "openmax":
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

    # Safety classifiers
    if m == "llamaguard":
        return LlamaGuardTextPostprocessor(
            model_id=args.llama_guard_model_id,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            max_new_tokens=args.llama_guard_max_tokens,
            temperature=args.llama_guard_temperature,
        )
    if m == "llamaguard_logit":
        return LlamaGuardLogitTextPostprocessor(
            model_id=args.llama_guard_model_id,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir.replace("promptinjection_", "promptinjection_logit_"),
            max_new_tokens=args.llama_guard_max_tokens,
            temperature=args.llama_guard_temperature,
        )
    if m == "duoguard":
        return DuoGuardTextPostprocessor(
            model_id=args.duoguard_model_id,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir,
            threshold=args.duoguard_threshold,
            max_length=args.duoguard_max_length,
        )
    if m == "wildguard":
        return WildGuardTextPostprocessor(
            model_id=args.wildguard_model_id,
            batch_size=args.batch_size,
            device=args.device,
            embedding_dir=embedding_dir.replace("promptinjection_", "wildguard_promptinjection_"),
            max_new_tokens=args.wildguard_max_tokens,
            temperature=args.wildguard_temperature,
        )
    if m in ("llmguard_jb", "llmguard_pi", "llmguard_promptinjection"):
        return create_llm_guard_jailbreak_detector(
            batch_size=args.batch_size,
            embedding_dir=embedding_dir,
            fail_fast=args.llmg_fail_fast,
            prompt_injection_threshold=args.llmg_prompt_injection_threshold,
            token_limit=args.llmg_token_limit,
            agg_mode=args.llmg_agg,
            agg_weights=_parse_llmg_weights(args.llmg_weights),
        )
    if m in ("llmguard_tox", "llmguard_toxicity"):
        return create_llm_guard_toxicity_detector(
            batch_size=args.batch_size,
            embedding_dir=embedding_dir,
            fail_fast=args.llmg_fail_fast,
            toxicity_threshold=args.llmg_toxicity_threshold,
            lang=args.llmg_lang,
            token_limit=args.llmg_token_limit,
            agg_mode=args.llmg_agg,
            agg_weights=_parse_llmg_weights(args.llmg_weights),
        )
    if m == "mdjudge_vllm":
        if not MDJUDGE_VLLM_AVAILABLE:
            raise RuntimeError("MD-Judge VLLM not available; install vllm / ensure GPU.")
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
    if m == "polyguard_vllm":
        if not POLYGUARD_VLLM_AVAILABLE:
            raise RuntimeError("PolyGuard VLLM not available; install vllm / ensure GPU.")
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

    raise ValueError(f"Unsupported method: {method}")


# ---------------------------
# Evaluation
# ---------------------------
def run_evaluation(
    detector: Any,
    id_test_texts: list[str],
    adv_texts: list[str],
    method: str,
) -> dict[str, float]:
    """Evaluate a single method on the prompt-injection set."""
    logger.info(f"Evaluating {method} on Purple Llama prompt-injection set...")
    if method.lower().startswith("forte_"):
        # Forte API provides its own evaluate on ID/OOD
        return detector.evaluate(id_test_texts, adv_texts)

    # Most postprocessors output a "safety" score in [0,1] where higher=safer
    id_scores = detector.postprocess(id_test_texts, cache_name=f"{method.lower()}_safe_test")
    adv_scores = detector.postprocess(adv_texts, cache_name=f"{method.lower()}_purplellama_pi")

    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(adv_scores))])
    scores = np.concatenate([id_scores, adv_scores])

    auroc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = int(np.argmin(np.abs(tpr - 0.95)))
    fpr95 = float(fpr[idx]) if idx < len(fpr) else 1.0

    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = float(average_precision_score(labels, scores))
    f1 = float(np.max(2 * (precision * recall) / (precision + recall + 1e-10)))
    return {"AUROC": auroc, "FPR@95TPR": fpr95, "AUPRC": auprc, "F1": f1}


def save_results_to_csv(all_results: dict[str, dict[str, float]], args: argparse.Namespace) -> str:
    """Save a single-row CSV (per method) for the Purple Llama PI benchmark."""
    # Build a tidy DF: one row per method, columns = metrics
    df = pd.DataFrame.from_dict(all_results, orient="index")
    df.index.name = "Method"

    # Add metadata
    df["Embedding_Model"] = args.embedding_model
    df["Safe_Dataset"] = args.safe_dataset

    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(parents=True, exist_ok=True)
    out_path = f"results/prompt_injection_results_{ts}.csv"
    df.to_csv(out_path)

    # Also dump args metadata
    meta_path = f"results/prompt_injection_results_{ts}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "timestamp": ts,
                "results_file": out_path.split("/")[-1],
                "args": vars(args),
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {out_path}")
    logger.info(f"Metadata saved to {meta_path}")
    return out_path


# ---------------------------
# Main
# ---------------------------
def main(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    # Seeds / device info
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.manual_seed(args.seed)
        torch.cuda.empty_cache()
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU total memory: {total_mem:.1f} GB")

    logger.info("=== Loading ID (safe) data ===")
    id_train, id_test = load_safe_instructions(
        max_samples=args.max_safe_samples, dataset_name=args.safe_dataset
    )

    logger.info("=== Loading Purple Llama prompt-injection data ===")
    adv_texts = load_cybersec_prompt_injection(
        json_path=args.cs_json,
        use_context=args.cs_use_context,
        max_samples=args.max_adversarial_samples,
    )
    if not adv_texts:
        raise RuntimeError(f"No adversarial samples loaded from {args.cs_json}")

    logger.info(f"ID test: {len(id_test)} | Adv (PI): {len(adv_texts)}")

    # Prepare methods
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    all_results: dict[str, dict[str, float]] = {}

    for method in methods:
        logger.info("\n" + "=" * 70)
        logger.info(f"Method: {method}")
        logger.info("=" * 70)

        # vLLM pre-cleanup if needed
        if method.lower() in ("mdjudge_vllm", "polyguard_vllm"):
            cleanup_vllm_memory(method_name=f"{method}_pre_init")

        # Create & setup
        det = create_postprocessor(method, args)
        logger.info("Setting up detector on safe ID train set...")
        start = time.time()
        if method.lower().startswith("forte_"):
            det.fit(id_train)
        else:
            det.setup(id_train, random_state=args.seed)
        logger.info(f"Setup completed in {time.time() - start:.2f}s")

        # Evaluate
        res = run_evaluation(det, id_test, adv_texts, method)
        all_results[method] = res
        logger.info(
            f"{method} → AUROC={res['AUROC']:.4f}  FPR@95={res['FPR@95TPR']:.4f}  "
            f"AUPRC={res['AUPRC']:.4f}  F1={res['F1']:.4f}"
        )

        # Cleanup after each method
        if method.lower() in ("mdjudge_vllm", "polyguard_vllm"):
            # Try to release internal engines
            try:
                if hasattr(det, "model"):
                    det.model = None
                if hasattr(det, "llm"):
                    det.llm = None
            except Exception:
                pass
            cleanup_vllm_memory(method_name=f"{method}_post_eval")

        try:
            if hasattr(det, "embedding_model"):
                del det.embedding_model
            if hasattr(det, "model"):
                del det.model
            del det
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Pretty summary
    logger.info("\n" + "=" * 80)
    logger.info("PROMPT-INJECTION DETECTION SUMMARY (Purple Llama)")
    logger.info("=" * 80)
    for m in methods:
        if m in all_results:
            r = all_results[m]
            logger.info(
                f"{m:<20} AUROC={r['AUROC']:.4f} | FPR@95={r['FPR@95TPR']:.4f} | "
                f"AUPRC={r['AUPRC']:.4f} | F1={r['F1']:.4f}"
            )

    save_results_to_csv(all_results, args)
    return all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unified Baselines - Prompt Injection (Purple Llama)")
    # General
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    # Embedding model for classic OOD methods
    p.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")

    # Methods
    p.add_argument(
        "--methods",
        type=str,
        default="LLMGuard_JB,LlamaGuard,LlamaGuard_Logit,DuoGuard,WildGuard,MDJudge_VLLM,PolyGuard_VLLM,"
        "RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax,Forte_GMM,Forte_OCSVM",
        help="Comma-separated list of methods to run (LlamaGuard uses hardcoded scores, LlamaGuard_Logit uses logit-based scoring, WildGuard for AllenAI WildGuard 7B)",
    )

    # Safe ID datasets
    p.add_argument(
        "--safe_dataset",
        type=str,
        default="alpaca",
        help="alpaca|dolly|helpful_base|openassistant|id_mix",
    )
    p.add_argument("--max_safe_samples", type=int, default=5000)
    default_cs = str((Path(__file__).parent / "prompt_injection.json").resolve())
    # Purple Llama JSON
    p.add_argument(
        "--cs_json",
        type=str,
        default=default_cs,
        help="Path to Purple Llama prompt_injection.json or .jsonl",
    )
    p.add_argument(
        "--cs_use_context",
        action="store_true",
        help="Prepend test_case_prompt as [SYSTEM] context before user_input",
    )
    p.add_argument("--max_adversarial_samples", type=int, default=2000)

    # Classic OOD hyperparams
    p.add_argument("--vim_dim", type=int, default=512)
    p.add_argument("--cider_k", type=int, default=5)
    p.add_argument("--fdbd_distance_normalizer", action="store_true", default=True)
    p.add_argument("--nnguide_k", type=int, default=100)
    p.add_argument("--nnguide_alpha", type=float, default=1.0)
    p.add_argument("--nnguide_min_score", action="store_true")
    p.add_argument("--react_percentile", type=float, default=90.0)
    p.add_argument("--gmm_num_clusters", type=int, default=8)
    p.add_argument(
        "--gmm_feature_type",
        type=str,
        default="penultimate",
        choices=["penultimate", "raw", "embedding", "norm", "normalized"],
    )
    p.add_argument("--gmm_reduce_method", type=str, default="none", choices=["none", "pca", "lda"])
    p.add_argument("--gmm_target_dim", type=int, default=50)
    p.add_argument(
        "--gmm_covariance_type",
        type=str,
        default="tied",
        choices=["tied", "full", "diag", "spherical"],
    )
    p.add_argument("--gmm_use_sklearn", action="store_true", default=True)

    p.add_argument("--adascale_percentile", type=str, default="90.0,99.0")
    p.add_argument("--adascale_k1", type=float, default=50.0)
    p.add_argument("--adascale_k2", type=float, default=50.0)
    p.add_argument("--adascale_lambda", type=float, default=1.0)
    p.add_argument("--adascale_o", type=float, default=0.1)
    p.add_argument("--adascale_num_samples", type=int, default=None)

    p.add_argument("--openmax_alpha", type=int, default=3)
    p.add_argument("--openmax_threshold", type=float, default=0.9)
    p.add_argument("--openmax_tail", type=int, default=20)
    p.add_argument(
        "--openmax_distance_type",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine", "eucos"],
    )
    p.add_argument("--openmax_eu_weight", type=float, default=0.5)

    # Forte text API
    p.add_argument(
        "--forte_models",
        type=str,
        default="qwen3,bge-m3,e5",
        help="Comma-separated: qwen3,bge-m3,e5",
    )

    # Llama Guard
    p.add_argument("--llama_guard_model_id", type=str, default="meta-llama/Llama-Guard-3-1B")
    p.add_argument("--llama_guard_max_tokens", type=int, default=20)
    p.add_argument("--llama_guard_temperature", type=float, default=0.1)

    # WildGuard
    p.add_argument("--wildguard_model_id", type=str, default="allenai/wildguard")
    p.add_argument("--wildguard_max_tokens", type=int, default=32)
    p.add_argument("--wildguard_temperature", type=float, default=0.1)

    # LLM-Guard
    p.add_argument("--llmg_prompt_injection_threshold", type=float, default=0.5)
    p.add_argument("--llmg_toxicity_threshold", type=float, default=0.5)
    p.add_argument("--llmg_token_limit", type=int, default=2048)
    p.add_argument("--llmg_agg", type=str, default="max", choices=["max", "mean", "weighted"])
    p.add_argument("--llmg_fail_fast", action="store_true")
    p.add_argument("--llmg_lang", type=str, default="en")
    p.add_argument(
        "--llmg_weights",
        type=str,
        default=None,
        help="e.g. 'PromptInjection:0.7,Toxicity:0.3' when agg=weighted",
    )

    # MD-Judge VLLM
    p.add_argument(
        "--mdjudge_vllm_model_id", type=str, default="OpenSafetyLab/MD-Judge-v0_2-internlm2_7b"
    )
    p.add_argument("--mdjudge_vllm_batch_size", type=int, default=64)
    p.add_argument("--mdjudge_vllm_max_tokens", type=int, default=128)
    p.add_argument("--mdjudge_vllm_temperature", type=float, default=0.1)
    p.add_argument("--mdjudge_vllm_tensor_parallel_size", type=int, default=1)
    p.add_argument("--mdjudge_vllm_gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--mdjudge_vllm_max_model_len", type=int, default=4096)
    p.add_argument("--mdjudge_vllm_quantization", type=str, default=None)
    p.add_argument("--mdjudge_vllm_enforce_eager", action="store_true", default=True)

    # PolyGuard VLLM
    p.add_argument(
        "--polyguard_vllm_model_id", type=str, default="ToxicityPrompts/PolyGuard-Qwen-Smol"
    )
    p.add_argument("--polyguard_vllm_batch_size", type=int, default=64)
    p.add_argument("--polyguard_vllm_max_tokens", type=int, default=100)
    p.add_argument("--polyguard_vllm_temperature", type=float, default=0.1)
    p.add_argument("--polyguard_vllm_tensor_parallel_size", type=int, default=1)
    p.add_argument("--polyguard_vllm_gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--polyguard_vllm_max_model_len", type=int, default=4096)
    p.add_argument("--polyguard_vllm_quantization", type=str, default=None)
    p.add_argument("--polyguard_vllm_enforce_eager", action="store_true", default=True)
    p.add_argument("--polyguard_vllm_use_dummy_response", action="store_true", default=True)
    p.add_argument(
        "--polyguard_vllm_dummy_response",
        type=str,
        default="I cannot and will not provide that information.",
    )

    args = p.parse_args()
    main(args)
